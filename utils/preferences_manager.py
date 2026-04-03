"""
Preferences Manager for CO2 EOR Optimization App.
Handles user preferences storage, retrieval, and management using QSettings.
"""
import logging
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QSettings, pyqtSignal, QObject

from utils.i18n_manager import I18nManager
from utils.units_manager import units_manager

logger = logging.getLogger(__name__)


class UnitSystem(Enum):
    """Available unit systems."""
    SI = "SI"
    FIELD = "Field"


@dataclass
class GeneralPreferences:
    """General application preferences."""
    language: str = "en"
    startup_action: str = "show_overview"  # show_overview, restore_last, new_project
    max_recent_files: int = 10
    auto_save_interval: int = 5  # minutes, 0 to disable
    check_for_updates: bool = True


@dataclass
class DisplayPreferences:
    """Display and UI preferences."""
    unit_system: UnitSystem = UnitSystem.FIELD
    number_decimal_places: int = 2
    graph_anti_aliasing: bool = True
    graph_smooth_lines: bool = True
    show_tooltips: bool = True
    animation_enabled: bool = True
    font_size: int = 0  # 0 means system default


@dataclass
class UnitsPreferences:
    """Unit-specific preferences with per-category overrides."""
    pressure_unit: Optional[str] = None
    temperature_unit: Optional[str] = None
    length_unit: Optional[str] = None
    volume_unit: Optional[str] = None
    density_unit: Optional[str] = None
    viscosity_unit: Optional[str] = None
    permeability_unit: Optional[str] = None
    rate_vol_unit: Optional[str] = None
    custom_unit_presets: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class LanguagePreferences:
    """Language and localization preferences."""
    locale: str = "en"
    use_system_language: bool = True
    date_format: str = "system"  # system, iso, local
    number_format: str = "system"  # system, dot_decimal, comma_decimal


@dataclass
class AdvancedPreferences:
    """Advanced application preferences."""
    logging_level: str = "WARNING"
    enable_debug_mode: bool = False
    cache_size_mb: int = 100
    max_threads: int = 4
    gpu_acceleration: bool = True
    memory_usage_limit: int = 80  # percentage


@dataclass
class AIPreferences:
    """AI service preferences."""
    active_service: str = "OpenAI"
    services: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "OpenAI": {"api_key": "", "base_url": "https://api.openai.com/v1"},
        "Gemini": {"api_key": "", "base_url": ""},
        "OpenRouter": {"api_key": "", "base_url": "https://openrouter.ai/api/v1"},
    })


class PreferencesManager(QObject):
    """
    Manages application preferences with multiple categories and QSettings persistence.
    Emits signals when preferences change for real-time UI updates.
    """
    
    # Signals for preference changes
    general_preferences_changed = pyqtSignal(GeneralPreferences)
    display_preferences_changed = pyqtSignal(DisplayPreferences)
    units_preferences_changed = pyqtSignal(UnitsPreferences)
    language_preferences_changed = pyqtSignal(LanguagePreferences)
    advanced_preferences_changed = pyqtSignal(AdvancedPreferences)
    ai_preferences_changed = pyqtSignal(AIPreferences)
    preferences_reset = pyqtSignal()
    
    def __init__(self, app_name: str, organization_name: str, i18n_manager: Optional[I18nManager] = None):
        super().__init__()
        self.settings = QSettings(organization_name, app_name)
        self.i18n_manager = i18n_manager
        
        # Initialize default preferences
        self._general = GeneralPreferences()
        self._display = DisplayPreferences()
        self._units = UnitsPreferences()
        self._language = LanguagePreferences()
        self._advanced = AdvancedPreferences()
        self._ai = AIPreferences()
        
        self.load_preferences()
    
    def load_preferences(self) -> None:
        """Load preferences from QSettings or use defaults."""
        logger.info("Loading application preferences from QSettings")
        
        # Load general preferences
        self._general.language = self.settings.value("General/language", self._general.language)
        self._general.startup_action = self.settings.value("General/startup_action", self._general.startup_action)
        self._general.max_recent_files = int(self.settings.value("General/max_recent_files", self._general.max_recent_files))
        self._general.auto_save_interval = int(self.settings.value("General/auto_save_interval", self._general.auto_save_interval))
        self._general.check_for_updates = self.settings.value("General/check_for_updates", self._general.check_for_updates, type=bool)
        
        # Load display preferences
        self._display.unit_system = UnitSystem(self.settings.value("Display/unit_system", self._display.unit_system.value))
        self._display.number_decimal_places = int(self.settings.value("Display/number_decimal_places", self._display.number_decimal_places))
        self._display.graph_anti_aliasing = self.settings.value("Display/graph_anti_aliasing", self._display.graph_anti_aliasing, type=bool)
        self._display.graph_smooth_lines = self.settings.value("Display/graph_smooth_lines", self._display.graph_smooth_lines, type=bool)
        self._display.show_tooltips = self.settings.value("Display/show_tooltips", self._display.show_tooltips, type=bool)
        self._display.animation_enabled = self.settings.value("Display/animation_enabled", self._display.animation_enabled, type=bool)
        self._display.font_size = int(self.settings.value("Display/font_size", self._display.font_size))
        
        # Load units preferences
        self._units.pressure_unit = self.settings.value("Units/pressure_unit", self._units.pressure_unit)
        self._units.temperature_unit = self.settings.value("Units/temperature_unit", self._units.temperature_unit)
        self._units.length_unit = self.settings.value("Units/length_unit", self._units.length_unit)
        self._units.volume_unit = self.settings.value("Units/volume_unit", self._units.volume_unit)
        self._units.density_unit = self.settings.value("Units/density_unit", self._units.density_unit)
        self._units.viscosity_unit = self.settings.value("Units/viscosity_unit", self._units.viscosity_unit)
        self._units.permeability_unit = self.settings.value("Units/permeability_unit", self._units.permeability_unit)
        self._units.rate_vol_unit = self.settings.value("Units/rate_vol_unit", self._units.rate_vol_unit)
        self._units.custom_unit_presets = self.settings.value("Units/custom_unit_presets", self._units.custom_unit_presets, type=dict) or {}
        
        # Load language preferences
        self._language.locale = self.settings.value("Language/locale", self._language.locale)
        self._language.use_system_language = self.settings.value("Language/use_system_language", self._language.use_system_language, type=bool)
        self._language.date_format = self.settings.value("Language/date_format", self._language.date_format)
        self._language.number_format = self.settings.value("Language/number_format", self._language.number_format)
        
        # Load advanced preferences
        self._advanced.logging_level = self.settings.value("Advanced/logging_level", self._advanced.logging_level)
        self._advanced.enable_debug_mode = self.settings.value("Advanced/enable_debug_mode", self._advanced.enable_debug_mode, type=bool)
        self._advanced.cache_size_mb = int(self.settings.value("Advanced/cache_size_mb", self._advanced.cache_size_mb))
        self._advanced.max_threads = int(self.settings.value("Advanced/max_threads", self._advanced.max_threads))
        self._advanced.gpu_acceleration = self.settings.value("Advanced/gpu_acceleration", self._advanced.gpu_acceleration, type=bool)
        self._advanced.memory_usage_limit = int(self.settings.value("Advanced/memory_usage_limit", self._advanced.memory_usage_limit))

        # Load AI preferences
        self._ai.active_service = self.settings.value("AI/active_service", self._ai.active_service)
        self.settings.beginGroup("AI/services")
        for service_name in self._ai.services:
            self.settings.beginGroup(service_name)
            self._ai.services[service_name]["api_key"] = self.settings.value("api_key", "")
            self._ai.services[service_name]["base_url"] = self.settings.value("base_url", self._ai.services[service_name].get("base_url", ""))
            self.settings.endGroup()
        self.settings.endGroup()

    def save_preferences(self) -> None:
        """Save all preferences to QSettings."""
        logger.info("Saving application preferences to QSettings")
        
        # Save general preferences
        self.settings.setValue("General/language", self._general.language)
        self.settings.setValue("General/startup_action", self._general.startup_action)
        self.settings.setValue("General/max_recent_files", self._general.max_recent_files)
        self.settings.setValue("General/auto_save_interval", self._general.auto_save_interval)
        self.settings.setValue("General/check_for_updates", self._general.check_for_updates)
        
        # Save display preferences
        self.settings.setValue("Display/unit_system", self._display.unit_system.value)
        self.settings.setValue("Display/number_decimal_places", self._display.number_decimal_places)
        self.settings.setValue("Display/graph_anti_aliasing", self._display.graph_anti_aliasing)
        self.settings.setValue("Display/graph_smooth_lines", self._display.graph_smooth_lines)
        self.settings.setValue("Display/show_tooltips", self._display.show_tooltips)
        self.settings.setValue("Display/animation_enabled", self._display.animation_enabled)
        self.settings.setValue("Display/font_size", self._display.font_size)
        
        # Save units preferences
        if self._units.pressure_unit:
            self.settings.setValue("Units/pressure_unit", self._units.pressure_unit)
        if self._units.temperature_unit:
            self.settings.setValue("Units/temperature_unit", self._units.temperature_unit)
        if self._units.length_unit:
            self.settings.setValue("Units/length_unit", self._units.length_unit)
        if self._units.volume_unit:
            self.settings.setValue("Units/volume_unit", self._units.volume_unit)
        if self._units.density_unit:
            self.settings.setValue("Units/density_unit", self._units.density_unit)
        if self._units.viscosity_unit:
            self.settings.setValue("Units/viscosity_unit", self._units.viscosity_unit)
        if self._units.permeability_unit:
            self.settings.setValue("Units/permeability_unit", self._units.permeability_unit)
        if self._units.rate_vol_unit:
            self.settings.setValue("Units/rate_vol_unit", self._units.rate_vol_unit)
        if self._units.custom_unit_presets:
            self.settings.setValue("Units/custom_unit_presets", self._units.custom_unit_presets)
        
        # Save language preferences
        self.settings.setValue("Language/locale", self._language.locale)
        self.settings.setValue("Language/use_system_language", self._language.use_system_language)
        self.settings.setValue("Language/date_format", self._language.date_format)
        self.settings.setValue("Language/number_format", self._language.number_format)
        
        # Save advanced preferences
        self.settings.setValue("Advanced/logging_level", self._advanced.logging_level)
        self.settings.setValue("Advanced/enable_debug_mode", self._advanced.enable_debug_mode)
        self.settings.setValue("Advanced/cache_size_mb", self._advanced.cache_size_mb)
        self.settings.setValue("Advanced/max_threads", self._advanced.max_threads)
        self.settings.setValue("Advanced/gpu_acceleration", self._advanced.gpu_acceleration)
        self.settings.setValue("Advanced/memory_usage_limit", self._advanced.memory_usage_limit)
        
        # Save AI preferences
        self.settings.setValue("AI/active_service", self._ai.active_service)
        self.settings.beginGroup("AI/services")
        for service_name, service_data in self._ai.services.items():
            self.settings.beginGroup(service_name)
            self.settings.setValue("api_key", service_data.get("api_key", ""))
            self.settings.setValue("base_url", service_data.get("base_url", ""))
            self.settings.endGroup()
        self.settings.endGroup()

        self.settings.sync()
        logger.info("Preferences saved successfully")
    
    def reset_to_defaults(self) -> None:
        """Reset all preferences to default values."""
        logger.info("Resetting preferences to defaults")
        
        self._general = GeneralPreferences()
        self._display = DisplayPreferences()
        self._units = UnitsPreferences()
        self._language = LanguagePreferences()
        self._advanced = AdvancedPreferences()
        self._ai = AIPreferences()
        
        # Clear all settings
        self.settings.clear()
        self.settings.sync()
        
        self.preferences_reset.emit()
        logger.info("Preferences reset to defaults")
    
    def apply_language_preferences(self) -> None:
        """Apply language preferences using the i18n manager."""
        if self.i18n_manager:
            locale_to_use = self._language.locale
            if self._language.use_system_language:
                # Use system language if enabled
                import locale
                system_locale = locale.getdefaultlocale()[0]
                if system_locale:
                    locale_to_use = system_locale.split('_')[0] if '_' in system_locale else system_locale
            
            self.i18n_manager.load_and_install_translator(locale_to_use)
    
    def get_display_unit(self, category: str) -> str:
        """
        Get the display unit for a given category, considering unit system and overrides.
        
        Args:
            category: The unit category (e.g., 'pressure', 'temperature')
            
        Returns:
            The appropriate display unit string
        """
        # Check for unit override first
        unit_override = getattr(self._units, f"{category}_unit", None)
        if unit_override:
            return unit_override
        
        # Use the unit system default
        unit_system = self._display.unit_system.value
        return units_manager.get_display_unit(category, unit_system)
    
    # Property accessors for preferences
    @property
    def general(self) -> GeneralPreferences:
        return self._general
    
    @general.setter
    def general(self, value: GeneralPreferences) -> None:
        self._general = value
        self.general_preferences_changed.emit(value)
    
    @property
    def display(self) -> DisplayPreferences:
        return self._display
    
    @display.setter
    def display(self, value: DisplayPreferences) -> None:
        self._display = value
        self.display_preferences_changed.emit(value)
    
    @property
    def units(self) -> UnitsPreferences:
        return self._units
    
    @units.setter
    def units(self, value: UnitsPreferences) -> None:
        self._units = value
        self.units_preferences_changed.emit(value)
    
    @property
    def language(self) -> LanguagePreferences:
        return self._language
    
    @language.setter
    def language(self, value: LanguagePreferences) -> None:
        self._language = value
        self.language_preferences_changed.emit(value)
    
    @property
    def advanced(self) -> AdvancedPreferences:
        return self._advanced
    
    @advanced.setter
    def advanced(self, value: AdvancedPreferences) -> None:
        self._advanced = value
        self.advanced_preferences_changed.emit(value)

    @property
    def ai(self) -> AIPreferences:
        return self._ai

    @ai.setter
    def ai(self, value: AIPreferences) -> None:
        self._ai = value
        self.ai_preferences_changed.emit(value)


    def _migrate_preferences(self, old_version: int) -> None:
        """Migrate preferences from older versions to current version."""
        logger.info(f"Migrating preferences from version {old_version} to {self.CURRENT_PREFERENCES_VERSION}")
        
        # Migration steps for each version
        if old_version == 0:
            # Version 0 to 1: Initial version, reset to defaults
            logger.warning("Migrating from version 0, resetting preferences to defaults")
            self.reset_to_defaults()
        # Add more migration steps for future versions here
        # elif old_version == 1:
        #     # Migration from version 1 to 2
        #     pass
        
        # After migration, update the version
        self._preferences_version = self.CURRENT_PREFERENCES_VERSION
        self.settings.setValue("Preferences/version", self._preferences_version)
        self.settings.sync()
    
    def _validate_preferences(self) -> None:
        """Validate loaded preferences and correct invalid values."""
        logger.info("Validating loaded preferences")
        
        # Validate general preferences
        if self._general.max_recent_files < 0 or self._general.max_recent_files > 50:
            logger.warning(f"Invalid max_recent_files value: {self._general.max_recent_files}. Resetting to default.")
            self._general.max_recent_files = GeneralPreferences().max_recent_files
        
        if self._general.auto_save_interval < 0 or self._general.auto_save_interval > 60:
            logger.warning(f"Invalid auto_save_interval value: {self._general.auto_save_interval}. Resetting to default.")
            self._general.auto_save_interval = GeneralPreferences().auto_save_interval
        
        # Validate display preferences
        if self._display.number_decimal_places < 0 or self._display.number_decimal_places > 8:
            logger.warning(f"Invalid number_decimal_places value: {self._display.number_decimal_places}. Resetting to default.")
            self._display.number_decimal_places = DisplayPreferences().number_decimal_places
        
        if self._display.font_size < -5 or self._display.font_size > 20:
            logger.warning(f"Invalid font_size value: {self._display.font_size}. Resetting to default.")
            self._display.font_size = DisplayPreferences().font_size
        
        # Validate units preferences
        # Check if unit overrides are valid for their categories
        for category in ['pressure', 'temperature', 'length', 'volume', 'density', 'viscosity', 'permeability', 'rate_vol']:
            unit_value = getattr(self._units, f"{category}_unit")
            if unit_value and unit_value not in units_manager.get_available_units_for_category(category):
                logger.warning(f"Invalid unit for {category}: {unit_value}. Resetting to default.")
                setattr(self._units, f"{category}_unit", None)
        
        # Validate advanced preferences
        if self._advanced.cache_size_mb < 10 or self._advanced.cache_size_mb > 1000:
            logger.warning(f"Invalid cache_size_mb value: {self._advanced.cache_size_mb}. Resetting to default.")
            self._advanced.cache_size_mb = AdvancedPreferences().cache_size_mb
        
        if self._advanced.max_threads < 1 or self._advanced.max_threads > 16:
            logger.warning(f"Invalid max_threads value: {self._advanced.max_threads}. Resetting to default.")
            self._advanced.max_threads = AdvancedPreferences().max_threads
        
        if self._advanced.memory_usage_limit < 50 or self._advanced.memory_usage_limit > 95:
            logger.warning(f"Invalid memory_usage_limit value: {self._advanced.memory_usage_limit}. Resetting to default.")
            self._advanced.memory_usage_limit = AdvancedPreferences().memory_usage_limit
        
        # Validate logging level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._advanced.logging_level not in valid_log_levels:
            logger.warning(f"Invalid logging_level: {self._advanced.logging_level}. Resetting to default.")
            self._advanced.logging_level = AdvancedPreferences().logging_level

# Global instance for easy access
_preferences_manager: Optional[PreferencesManager] = None

def get_preferences_manager() -> PreferencesManager:
    """Get the global preferences manager instance."""
    global _preferences_manager
    if _preferences_manager is None:
        raise RuntimeError("PreferencesManager has not been initialized. Call initialize_preferences_manager first.")
    return _preferences_manager

def initialize_preferences_manager(app_name: str, organization_name: str, i18n_manager: Optional[I18nManager] = None) -> PreferencesManager:
    """Initialize the global preferences manager."""
    global _preferences_manager
    _preferences_manager = PreferencesManager(app_name, organization_name, i18n_manager)
    return _preferences_manager
