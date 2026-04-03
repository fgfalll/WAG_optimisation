import json
import logging
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)




class ConfigNotLoadedError(Exception):
    pass

class ConfigManager:
    def __init__(self, config_dir_path: Optional[str] = None, require_config: bool = True, autoload: bool = True):
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
        target_dir = Path(dir_path) if dir_path else self._default_config_dir
        if not target_dir:
            raise ValueError("No configuration directory path provided to load from.")

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

                if not isinstance(data, dict):
                    logger.debug(f"Skipping non-dictionary file {file_path.name} (type: {type(data).__name__})")
                    continue
                
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
        if self._require_config and not self._configs_loaded:
            raise ConfigNotLoadedError(
                "Configuration has not been successfully loaded. "
                "Ensure a valid config directory is provided and load_configs_from_directory() was successful."
            )

    def get(self, key_path: str, default: Any = None) -> Any:
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
        self._ensure_config_loaded()
        section_data = self.get(section_key, default={})
        return deepcopy(section_data) if isinstance(section_data, dict) else {}

    @property
    def is_loaded(self) -> bool:
        return self._configs_loaded

    @property
    def loaded_config_directory(self) -> Optional[Path]:
        return self._loaded_config_dir

    @property
    def loaded_files_in_order(self) -> List[Path]:
        return self._loaded_files_order

    @property
    def raw_config(self) -> Dict[str, Any]:
        self._ensure_config_loaded()
        return deepcopy(self._config_data)

    def reload_configs(self, dir_path: Optional[Union[str, Path]] = None) -> bool:
        target_dir = dir_path or self._loaded_config_dir
        if target_dir:
            logger.info(f"Reloading configurations from specified directory: {target_dir}")
            return self.load_configs_from_directory(target_dir)
        else:
            logger.warning("Cannot reload configs: No configuration directory was previously loaded or provided.")
            return False