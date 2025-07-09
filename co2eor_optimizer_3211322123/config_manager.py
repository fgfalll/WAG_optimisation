import json
import logging
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from copy import deepcopy # For returning copies of config sections

# Initialize module-level logger
logger = logging.getLogger(__name__)
# Basic configuration for the logger if no handlers are configured by the application's entry point
if not logger.hasHandlers():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ConfigNotLoadedError(Exception):
    """Custom exception raised when an operation requires config but it hasn't been loaded."""
    pass

class ConfigManager:
    """
    Manages loading and accessing configuration data from multiple JSON files
    within a specified directory.
    This class is designed as a singleton in terms of its data store, meaning
    all instances will share the same loaded configuration data once loaded.
    """
    _config_data: Dict[str, Any] = {}
    _configs_loaded: bool = False
    _loaded_config_dir: Optional[Path] = None
    _loaded_files_order: List[Path] = [] # To track the order files were processed for merging

    def __init__(self, config_dir_path: Optional[str] = None, require_config: bool = True, autoload: bool = True):
        """
        Initializes the ConfigManager.

        Args:
            config_dir_path (str, optional): Path to the directory containing JSON configuration files.
                                             Defaults to package-relative "config" directory.
            require_config (bool): If True, an error will be raised during first access
                                   if no configurations have been successfully loaded.
            autoload (bool): If True, attempts to load configs from config_dir_path immediately
                             upon instantiation, if not already loaded from that directory.
        """
        # Set default config path relative to package directory
        if config_dir_path is None:
            config_dir_path = str(Path(__file__).parent / "config")
            
        self._default_config_dir = Path(config_dir_path)
        self._require_config = require_config

        # Class-level attributes ensure only one load attempt from a given directory path,
        # preventing redundant loads if multiple ConfigManager instances are (incorrectly) created
        # or if the global instance's __init__ is hit multiple times due to import patterns.
        if ConfigManager._configs_loaded:
            if ConfigManager._loaded_config_dir == self._default_config_dir and autoload:
                logger.debug(f"ConfigManager: Configurations already loaded from {ConfigManager._loaded_config_dir}. Skipping redundant autoload.")
                return
            elif ConfigManager._loaded_config_dir != self._default_config_dir and autoload:
                logger.warning(f"ConfigManager: Re-initializing for new config directory '{self._default_config_dir}'. "
                               f"Previously loaded from '{ConfigManager._loaded_config_dir}'. Old config will be cleared on load.")
                # Reset class state to allow loading from new directory.
                # The actual clearing and loading happens in load_configs_from_directory.
        
        if autoload:
            try:
                self.load_configs_from_directory()
            except FileNotFoundError as e:
                if self._require_config:
                    # If autoload fails and config is required, this makes it an immediate startup error.
                    logger.critical(f"CRITICAL: Autoload failed for required configuration directory '{self._default_config_dir}': {e}")
                    raise ConfigNotLoadedError(f"Autoload failed for required configuration: {e}") from e
                else:
                    logger.warning(f"Autoload failed for configuration directory '{self._default_config_dir}': {e}. Proceeding without loaded config.")
            except Exception as e_auto: # Catch any other exception during autoload
                logger.error(f"Unexpected error during autoload from '{self._default_config_dir}': {e_auto}", exc_info=True)
                if self._require_config:
                    raise ConfigNotLoadedError(f"Unexpected error during autoload: {e_auto}") from e_auto


    def load_configs_from_directory(self, dir_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Loads and merges all JSON configuration files from the specified directory.
        Files are loaded in alphabetical order. Top-level keys from files loaded
        later will override earlier ones. A simple one-level deep merge is performed
        for dictionary values under the same top-level key.

        Args:
            dir_path (Optional[Union[str, Path]]): The directory to load from.
                                                  If None, uses the default_config_dir
                                                  set during initialization.

        Returns:
            bool: True if at least one configuration file was successfully loaded and processed,
                  False otherwise.

        Raises:
            FileNotFoundError: If `require_config` is True and the directory is not found
                               or contains no JSON files.
        """
        target_dir = Path(dir_path) if dir_path else self._default_config_dir

        # Prevent reloading if already loaded from the exact same directory path.
        if ConfigManager._configs_loaded and ConfigManager._loaded_config_dir == target_dir:
            logger.info(f"Configurations from {target_dir} are already loaded. Skipping reload.")
            return True # Considered successful as configs are already loaded

        # If attempting to load from a new directory, reset previous config state.
        if ConfigManager._configs_loaded and ConfigManager._loaded_config_dir != target_dir:
             logger.warning(f"Loading configurations from new directory '{target_dir}'. "
                            f"Previously loaded from '{ConfigManager._loaded_config_dir}'. Clearing old configuration data.")
        
        # Reset class-level data before attempting a new load.
        ConfigManager._config_data = {}
        ConfigManager._configs_loaded = False
        ConfigManager._loaded_config_dir = None # Will be set to target_dir on successful load attempt
        ConfigManager._loaded_files_order = []

        if not target_dir.is_dir():
            logger.error(f"Configuration directory not found: {target_dir}.")
            if self._require_config:
                raise FileNotFoundError(f"Required configuration directory not found: {target_dir}")
            return False

        config_files = sorted(list(target_dir.glob("*.json")))

        if not config_files:
            logger.warning(f"No JSON configuration files found in directory: {target_dir}")
            ConfigManager._loaded_config_dir = target_dir # Mark that we attempted to load from here
            if self._require_config:
                raise FileNotFoundError(f"Required configuration files not found in {target_dir}.")
            return False

        loaded_at_least_one_file = False
        temp_merged_config: Dict[str, Any] = {}

        for file_path in config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Merge logic:
                for key, value in data.items():
                    if key in temp_merged_config and \
                       isinstance(temp_merged_config[key], dict) and \
                       isinstance(value, dict):
                        # Perform a one-level deep merge for dictionaries
                        temp_merged_config[key].update(value)
                    else:
                        # For non-dicts or if key is new, overwrite/set
                        temp_merged_config[key] = value
                
                logger.info(f"Successfully loaded and merged configuration from {file_path.name}")
                ConfigManager._loaded_files_order.append(file_path)
                loaded_at_least_one_file = True
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path.name}: {e}")
                # Decide if one bad file should halt all config loading or just be skipped.
                # For robustness, we'll skip the bad file and continue.
            except Exception as e:
                logger.error(f"Unexpected error loading config from {file_path.name}: {e}", exc_info=True)

        if loaded_at_least_one_file:
            ConfigManager._config_data = temp_merged_config
            ConfigManager._configs_loaded = True
            ConfigManager._loaded_config_dir = target_dir
            logger.info(f"All valid configurations successfully loaded and merged from directory: {target_dir}")
            return True
        else: # No files were successfully processed
            logger.error(f"No valid configuration files were processed from {target_dir}.")
            ConfigManager._loaded_config_dir = target_dir # Still mark attempt
            if self._require_config:
                raise ConfigNotLoadedError(f"No valid configurations processed from {target_dir}, and config is required.")
            return False

    def _ensure_config_loaded(self) -> None:
        """Checks if config is loaded, raising ConfigNotLoadedError if required and not loaded."""
        if self._require_config and not ConfigManager._configs_loaded:
            raise ConfigNotLoadedError(
                "Configuration has not been successfully loaded. "
                "Ensure a valid config directory is provided and load_configs_from_directory() was successful."
            )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key path.
        Example: get('Logging.level', 'INFO')

        Args:
            key_path (str): Dot-separated path to the desired configuration value.
            default (Any): Default value to return if the key_path is not found.

        Returns:
            Any: The configuration value or the default.
        """
        self._ensure_config_loaded()
        keys = key_path.split('.')
        current_level_data = ConfigManager._config_data
        try:
            for key_part in keys:
                if isinstance(current_level_data, list) and key_part.isdigit():
                    idx = int(key_part)
                    if 0 <= idx < len(current_level_data):
                        current_level_data = current_level_data[idx]
                    else:
                        # Index out of bounds for list
                        return default
                elif isinstance(current_level_data, dict):
                    current_level_data = current_level_data[key_part]
                else:
                    # Cannot traverse further (e.g., trying to get a sub-key from a string value)
                    return default
            return current_level_data
        except (KeyError, TypeError, IndexError):
            # Key not found at some level, or type mismatch (e.g. list index on dict)
            return default

    def get_section(self, section_key: str) -> Dict[str, Any]:
        """
        Retrieves an entire configuration section as a dictionary.
        Returns a deepcopy to prevent modification of the internal config state.

        Args:
            section_key (str): The top-level key for the desired section.

        Returns:
            Dict[str, Any]: A deepcopy of the section dictionary, or an empty dictionary
                            if the section is not found or not a dictionary.
        """
        self._ensure_config_loaded()
        section_data = self.get(section_key, default={})
        # Ensure what's returned is actually a dictionary and is a copy.
        return deepcopy(section_data) if isinstance(section_data, dict) else {}

    @property
    def is_loaded(self) -> bool:
        """Returns True if configuration files have been successfully loaded and processed."""
        return ConfigManager._configs_loaded

    @property
    def loaded_config_directory(self) -> Optional[Path]:
        """Returns the Path object of the directory from which configs were loaded."""
        return ConfigManager._loaded_config_dir

    @property
    def loaded_files_in_order(self) -> List[Path]:
        """Returns a list of Path objects for files loaded, in their processing order."""
        return ConfigManager._loaded_files_order

    @property
    def raw_config(self) -> Dict[str, Any]:
        """
        Returns a deepcopy of the entire raw (merged) configuration data dictionary.
        """
        self._ensure_config_loaded()
        return deepcopy(ConfigManager._config_data)

    def reload_configs(self, dir_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Forces a reload of configurations. If dir_path is provided, it loads from
        that new directory. Otherwise, it reloads from the last successfully used directory.

        Args:
            dir_path (Optional[Union[str, Path]]): New directory to load from.
                                                  If None, reloads from last used directory.

        Returns:
            bool: True if reload was successful, False otherwise.
        """
        if dir_path:
            logger.info(f"Reloading configurations from specified directory: {dir_path}")
            # Reset class state to allow loading from a potentially new directory by load_configs_from_directory
            ConfigManager._configs_loaded = False 
            return self.load_configs_from_directory(dir_path)
        elif ConfigManager._loaded_config_dir:
            logger.info(f"Re-triggering load from last used directory: {ConfigManager._loaded_config_dir}")
            # Reset class state to allow load_configs_from_directory to actually perform the load
            ConfigManager._configs_loaded = False 
            return self.load_configs_from_directory(ConfigManager._loaded_config_dir)
        else:
            logger.warning("Cannot reload configs: No configuration directory was previously loaded.")
            return False

# --- Global Instance ---
# This instance will be shared across the application upon first import of this module.
# It attempts to autoload from the "config" directory by default.
# If this default directory isn't found and require_config is True,
# the __init__ will raise an error during this global instantiation if autoload fails.
# Applications should handle potential ConfigNotLoadedError at their entry point if
# they depend on this global instance being successfully auto-loaded.
try:
    # Use absolute path to config directory
    config_dir_abs = str(Path(__file__).parent / "config")
    config_manager = ConfigManager(config_dir_path=config_dir_abs, require_config=True, autoload=True)
except ConfigNotLoadedError as e_global_init:
    logger.critical(f"Failed to initialize global ConfigManager: {e_global_init}. "
                    "Application's configuration-dependent features may not work.")
    # Fallback to a non-requiring, non-autoloading instance if critical init fails
    # This allows the module to be imported, but `is_loaded` will be False.
    config_manager = ConfigManager(config_dir_path=config_dir_abs, require_config=False, autoload=False)