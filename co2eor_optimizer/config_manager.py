import json
import logging
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ConfigNotLoadedError(Exception):
    """Custom exception raised when an operation requires config but it hasn't been loaded."""
    pass

class ConfigManager:
    """
    Manages loading and accessing configuration data from multiple JSON files
    within a specified directory.
    This class is designed as a simple utility for loading and merging configurations,
    without maintaining a global state.
    """
    def __init__(self, config_dir_path: Optional[str] = None, require_config: bool = True, autoload: bool = True):
        """
        Initializes the ConfigManager.

        Args:
            config_dir_path (str, optional): Path to the directory containing JSON configuration files.
                                             If None, a path must be provided to `load_configs_from_directory`.
            require_config (bool): If True, an error will be raised if configurations cannot be loaded.
            autoload (bool): If True, attempts to load configs from config_dir_path immediately.
        """
        self._config_data: Dict[str, Any] = {}
        self._configs_loaded: bool = False
        self._loaded_config_dir: Optional[Path] = None
        self._loaded_files_order: List[Path] = []
        
        self._default_config_dir = Path(config_dir_path) if config_dir_path else None
        self._require_config = require_config

        if autoload and self._default_config_dir:
            try:
                self.load_configs_from_directory()
            except FileNotFoundError as e:
                if self._require_config:
                    logger.critical(f"CRITICAL: Autoload failed for required configuration directory '{self._default_config_dir}': {e}")
                    raise ConfigNotLoadedError(f"Autoload failed for required configuration: {e}") from e
                else:
                    logger.warning(f"Autoload failed for configuration directory '{self._default_config_dir}': {e}. Proceeding without loaded config.")
            except Exception as e_auto:
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
            ValueError: If no directory path is provided either at initialization or to this method.
        """
        target_dir = Path(dir_path) if dir_path else self._default_config_dir
        if not target_dir:
            raise ValueError("No configuration directory path provided to load from.")

        # Reset instance state for a fresh load
        self._config_data = {}
        self._configs_loaded = False
        self._loaded_config_dir = None
        self._loaded_files_order = []

        if not target_dir.is_dir():
            logger.error(f"Configuration directory not found: {target_dir}.")
            if self._require_config:
                raise FileNotFoundError(f"Required configuration directory not found: {target_dir}")
            return False

        config_files = sorted(list(target_dir.glob("*.json")))

        if not config_files:
            logger.warning(f"No JSON configuration files found in directory: {target_dir}")
            self._loaded_config_dir = target_dir
            if self._require_config:
                raise FileNotFoundError(f"Required configuration files not found in {target_dir}.")
            return False

        loaded_at_least_one_file = False
        temp_merged_config: Dict[str, Any] = {}

        for file_path in config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for key, value in data.items():
                    if key in temp_merged_config and \
                       isinstance(temp_merged_config[key], dict) and \
                       isinstance(value, dict):
                        temp_merged_config[key].update(value)
                    else:
                        temp_merged_config[key] = value
                
                logger.info(f"Successfully loaded and merged configuration from {file_path.name}")
                self._loaded_files_order.append(file_path)
                loaded_at_least_one_file = True
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {file_path.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading config from {file_path.name}: {e}", exc_info=True)

        if loaded_at_least_one_file:
            self._config_data = temp_merged_config
            self._configs_loaded = True
            self._loaded_config_dir = target_dir
            logger.info(f"All valid configurations successfully loaded and merged from directory: {target_dir}")
            return True
        else:
            logger.error(f"No valid configuration files were processed from {target_dir}.")
            self._loaded_config_dir = target_dir
            if self._require_config:
                raise ConfigNotLoadedError(f"No valid configurations processed from {target_dir}, and config is required.")
            return False

    def _ensure_config_loaded(self) -> None:
        """Checks if config is loaded, raising ConfigNotLoadedError if required and not loaded."""
        if self._require_config and not self._configs_loaded:
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
        current_level_data = self._config_data
        try:
            for key_part in keys:
                if isinstance(current_level_data, list) and key_part.isdigit():
                    idx = int(key_part)
                    if 0 <= idx < len(current_level_data):
                        current_level_data = current_level_data[idx]
                    else:
                        return default
                elif isinstance(current_level_data, dict):
                    current_level_data = current_level_data[key_part]
                else:
                    return default
            return current_level_data
        except (KeyError, TypeError, IndexError):
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
        return deepcopy(section_data) if isinstance(section_data, dict) else {}

    @property
    def is_loaded(self) -> bool:
        """Returns True if configuration files have been successfully loaded and processed."""
        return self._configs_loaded

    @property
    def loaded_config_directory(self) -> Optional[Path]:
        """Returns the Path object of the directory from which configs were loaded."""
        return self._loaded_config_dir

    @property
    def loaded_files_in_order(self) -> List[Path]:
        """Returns a list of Path objects for files loaded, in their processing order."""
        return self._loaded_files_order

    @property
    def raw_config(self) -> Dict[str, Any]:
        """
        Returns a deepcopy of the entire raw (merged) configuration data dictionary.
        """
        self._ensure_config_loaded()
        return deepcopy(self._config_data)

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
        target_dir = dir_path or self._loaded_config_dir
        if target_dir:
            logger.info(f"Reloading configurations from specified directory: {target_dir}")
            return self.load_configs_from_directory(target_dir)
        else:
            logger.warning("Cannot reload configs: No configuration directory was previously loaded or provided.")
            return False