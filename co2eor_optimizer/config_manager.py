import json
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class ConfigNotLoadedError(Exception):
    """Custom exception raised when an operation requires config but it hasn't been loaded."""
    pass

class ConfigManager:
    _config_data: Dict[str, Any] = {}
    _is_loaded_from_file: bool = False # More descriptive name
    _loaded_config_path: Optional[str] = None # Store the path of the loaded config

    def __init__(self, config_file_path: Optional[str] = None, require_config: bool = True):
        """
        Initializes the ConfigManager.
        If a config_file_path is provided, it attempts to load it immediately.

        Args:
            config_file_path (Optional[str]): Path to the configuration file.
            require_config (bool): If True, an error will be raised during initialization
                                   or first access if a config file hasn't been successfully loaded.
        """
        self._require_config = require_config # Store the requirement

        if ConfigManager._is_loaded_from_file:
            logger.info(f"ConfigManager: Configuration already loaded from {ConfigManager._loaded_config_path}.")
            return

        if config_file_path:
            self.load_config(config_file_path)
        elif self._require_config:
            # If config is required and no path is given at init,
            # it won't be loaded yet. Error will be raised on first access if not loaded by then.
            logger.warning(
                "ConfigManager initialized without an initial config file path, but config is required. "
                "Ensure load_config() is called before accessing configuration values."
            )
            # We don't raise ConfigNotLoadedError here yet, to allow load_config to be called later.

    def load_config(self, file_path: str) -> None:
        """
        Loads the configuration from the specified JSON file.
        If successful, subsequent calls to load_config with a different path will log a warning
        and not reload, unless explicitly reset (future feature).

        Args:
            file_path (str): The path to the JSON configuration file.
        
        Raises:
            FileNotFoundError: If the config file is not found.
            json.JSONDecodeError: If the config file is not valid JSON.
            Exception: For other unexpected errors during loading.
        """
        if ConfigManager._is_loaded_from_file and ConfigManager._loaded_config_path != file_path:
            logger.warning(
                f"ConfigManager: Configuration already loaded from {ConfigManager._loaded_config_path}. "
                f"Ignoring request to load from {file_path}. "
                "Re-initialization or a reset mechanism would be needed to change config source."
            )
            return
        if ConfigManager._is_loaded_from_file and ConfigManager._loaded_config_path == file_path:
            logger.info(f"ConfigManager: Configuration from {file_path} is already loaded. Skipping reload.")
            return

        try:
            with open(file_path, 'r') as f:
                ConfigManager._config_data = json.load(f)
            ConfigManager._is_loaded_from_file = True
            ConfigManager._loaded_config_path = file_path
            logger.info(f"Configuration loaded successfully from {file_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}.")
            ConfigManager._config_data = {} # Clear any partial/stale data
            ConfigManager._is_loaded_from_file = False
            raise # Re-raise the error to be handled by the caller
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            ConfigManager._config_data = {}
            ConfigManager._is_loaded_from_file = False
            raise # Re-raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config from {file_path}: {e}")
            ConfigManager._config_data = {}
            ConfigManager._is_loaded_from_file = False
            raise # Re-raise

    def _ensure_config_loaded(self) -> None:
        """Checks if config is loaded, raising an error if required and not loaded."""
        if self._require_config and not ConfigManager._is_loaded_from_file:
            raise ConfigNotLoadedError(
                "Configuration has not been successfully loaded. "
                "Please ensure a valid config file is provided and load_config() is called."
            )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key path.
        Example: get('optimizer.ga.population_size', 50)

        Raises:
            ConfigNotLoadedError: If config is required and not loaded.
        """
        self._ensure_config_loaded() # Check before trying to access

        keys = key_path.split('.')
        value = ConfigManager._config_data
        try:
            for key in keys:
                if isinstance(value, list) and key.isdigit():
                    idx = int(key)
                    if 0 <= idx < len(value):
                        value = value[idx]
                    else:
                        return default
                elif isinstance(value, dict):
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError, IndexError):
            return default

    def get_section(self, section_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an entire section dictionary.

        Returns None if the section is not found and config is loaded.
        Returns an empty dict as a default if the key isn't found, but this might be
        misleading if the intent is to know if the section truly exists.

        Raises:
            ConfigNotLoadedError: If config is required and not loaded.
        """
        self._ensure_config_loaded()
        # Using get with a specific default (like an empty dict) might hide
        # whether the section truly exists or if it's just returning the default.
        # It might be better to return None if not found, and let the caller handle it.
        # For now, let's keep it consistent with current usage in core.py where empty dict is expected.
        return self.get(section_key, default={})


    @property
    def is_loaded(self) -> bool: # Renamed for clarity
        """Returns True if a configuration file has been successfully loaded."""
        return ConfigManager._is_loaded_from_file

    @property
    def loaded_config_file_path(self) -> Optional[str]:
        """Returns the path of the successfully loaded configuration file."""
        return ConfigManager._loaded_config_path

    @property
    def raw_config(self) -> Dict[str, Any]:
        """
        Returns the raw configuration data dictionary.

        Raises:
            ConfigNotLoadedError: If config is required and not loaded.
        """
        self._ensure_config_loaded()
        return ConfigManager._config_data

# Global instance initialization
# The application's entry point should handle the initial load_config call.
# Initialize with require_config=True by default.
# The initial config_file_path can be None here, relying on explicit load_config calls.
config_manager = ConfigManager(config_file_path=None, require_config=True)