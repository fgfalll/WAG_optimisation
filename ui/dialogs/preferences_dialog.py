import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QDialogButtonBox, QLabel, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QStackedWidget,
    QScrollArea, QWidget, QSizePolicy, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QEvent
from PyQt6.QtGui import QIcon

from utils.preferences_manager import (
    PreferencesManager, UnitSystem, get_preferences_manager, AIPreferences
)
from utils.i18n_manager import I18nManager
from utils.units_manager import units_manager

logger = logging.getLogger(__name__)


class PreferencesDialog(QDialog):
    """Main preferences dialog with multiple configuration tabs."""
    
    preferences_applied = pyqtSignal()
    
    def __init__(self, parent=None, i18n_manager: Optional[I18nManager] = None, preferences_manager: Optional[PreferencesManager] = None):
        super().__init__(parent)
        self.i18n_manager = i18n_manager
        # Use the provided preferences manager or get the global one
        if preferences_manager is not None:
            self.pref_manager = preferences_manager
        else:
            self.pref_manager = get_preferences_manager()
        self.original_preferences = self._capture_current_preferences()
        self.setMinimumSize(700, 500)
        self._setup_ui()
        self._load_current_preferences()
        self.retranslateUi()

        self._is_dirty = False
        self.apply_button.setEnabled(False)

        self.button_box.accepted.connect(self._on_ok_clicked)
        self.button_box.rejected.connect(self.reject)
        self.apply_button.clicked.connect(self._apply_preferences)

        self._connect_dirty_signals()
        
    def _setup_ui(self):
        """Set up the dialog UI with tabs and buttons."""
        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.general_tab = GeneralPreferencesTab(self.pref_manager, self.i18n_manager)
        self.display_tab = DisplayPreferencesTab(self.pref_manager)
        self.units_tab = UnitsPreferencesTab(self.pref_manager)
        self.language_tab = LanguagePreferencesTab(self.pref_manager, self.i18n_manager)
        self.ai_tab = AIPreferencesTab(self.pref_manager)
        self.advanced_tab = AdvancedPreferencesTab(self.pref_manager)
        
        self.tab_widget.addTab(self.general_tab, "")
        self.tab_widget.addTab(self.display_tab, "")
        self.tab_widget.addTab(self.units_tab, "")
        self.tab_widget.addTab(self.language_tab, "")
        self.tab_widget.addTab(self.ai_tab, "")
        self.tab_widget.addTab(self.advanced_tab, "")

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.RestoreDefaults
        )
        self.apply_button = self.button_box.button(QDialogButtonBox.StandardButton.Apply)
        self.restore_button = self.button_box.button(QDialogButtonBox.StandardButton.RestoreDefaults)
        self.restore_button.clicked.connect(self._restore_defaults)
        
        main_layout.addWidget(self.button_box)

    def _connect_dirty_signals(self):
        """Connect signals from all tabs to mark the dialog as dirty."""
        self.general_tab.startup_combo.currentIndexChanged.connect(self._set_dirty)
        self.display_tab.unit_system_combo.currentIndexChanged.connect(self._set_dirty)
        self.language_tab.language_combo.currentIndexChanged.connect(self._set_dirty)
        self.advanced_tab.log_level_combo.currentIndexChanged.connect(self._set_dirty)
        self.ai_tab.active_service_combo.currentIndexChanged.connect(self._set_dirty)
        # self.ai_tab.service_name_edit.textChanged.connect(self._set_dirty)
        # self.ai_tab.api_key_edit.textChanged.connect(self._set_dirty)
        # self.ai_tab.base_url_edit.textChanged.connect(self._set_dirty)

    def _set_dirty(self):
        """Mark the dialog as dirty and enable the Apply button."""
        if not self._is_dirty:
            self._is_dirty = True
            self.apply_button.setEnabled(True)
    
    def retranslateUi(self):
        """Retranslate all UI elements."""
        self.setWindowTitle(self.tr("Preferences"))
        self.tab_widget.setTabText(0, self.tr("General"))
        self.tab_widget.setTabText(1, self.tr("Display"))
        self.tab_widget.setTabText(2, self.tr("Units"))
        self.tab_widget.setTabText(3, self.tr("Language"))
        self.tab_widget.setTabText(4, self.tr("AI Services"))
        self.tab_widget.setTabText(5, self.tr("Advanced"))

    def changeEvent(self, event: QEvent):
        """Handle language change event."""
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _capture_current_preferences(self) -> Dict[str, Any]:
        """Capture current preferences for cancel operation."""
        return {
            'general': self.pref_manager.general,
            'display': self.pref_manager.display,
            'units': self.pref_manager.units,
            'language': self.pref_manager.language,
            'ai': self.pref_manager.ai,
            'advanced': self.pref_manager.advanced
        }
    
    def _load_current_preferences(self):
        """Load current preferences into UI controls."""
        self.general_tab.load_preferences()
        self.display_tab.load_preferences()
        self.units_tab.load_preferences()
        self.language_tab.load_preferences()
        self.ai_tab.load_preferences()
        self.advanced_tab.load_preferences()
    
    def _apply_preferences(self):
        """Apply preferences from all tabs."""
        if not self._is_dirty:
            return
            
        try:
            self.general_tab.apply_preferences()
            self.display_tab.apply_preferences()
            self.units_tab.apply_preferences()
            self.language_tab.apply_preferences()
            self.ai_tab.apply_preferences()
            self.advanced_tab.apply_preferences()
            
            self.pref_manager.save_preferences()
            self.preferences_applied.emit()
            
            self._is_dirty = False
            self.apply_button.setEnabled(False)
            self.original_preferences = self._capture_current_preferences()
            
            logger.info("Preferences applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to apply preferences: {e}")
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to apply preferences: {}").format(e))
    
    def _on_ok_clicked(self):
        """Handle OK button click - apply preferences and close dialog."""
        if self._is_dirty:
            self._apply_preferences()
        self.accept()
    
    def _restore_defaults(self):
        """Restore all preferences to default values."""
        reply = QMessageBox.question(
            self,
            self.tr("Restore Defaults"),
            self.tr("Are you sure you want to restore all preferences to their default values?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.pref_manager.reset_to_defaults()
            self._load_current_preferences()
            logger.info("Preferences restored to defaults")
    
    def reject(self):
        """Handle cancel operation by restoring original preferences."""
        self.pref_manager.general = self.original_preferences['general']
        self.pref_manager.display = self.original_preferences['display']
        self.pref_manager.units = self.original_preferences['units']
        self.pref_manager.language = self.original_preferences['language']
        self.pref_manager.ai = self.original_preferences['ai']
        self.pref_manager.advanced = self.original_preferences['advanced']
        super().reject()


class GeneralPreferencesTab(QWidget):
    """General application preferences tab."""
    def __init__(self, pref_manager: PreferencesManager, i18n_manager: Optional[I18nManager] = None):
        super().__init__()
        self.pref_manager = pref_manager
        self.i18n_manager = i18n_manager
        self._setup_ui()
        self.retranslateUi()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.startup_group = QGroupBox()
        startup_layout = QFormLayout(self.startup_group)
        
        self.startup_combo = QComboBox()
        self.startup_label = QLabel()
        startup_layout.addRow(self.startup_label, self.startup_combo)
        
        self.auto_save_spin = QSpinBox()
        self.auto_save_label = QLabel()
        startup_layout.addRow(self.auto_save_label, self.auto_save_spin)
        
        self.max_recent_spin = QSpinBox()
        self.max_recent_label = QLabel()
        startup_layout.addRow(self.max_recent_label, self.max_recent_spin)
        
        self.check_updates_check = QCheckBox()
        startup_layout.addRow(self.check_updates_check)
        
        layout.addWidget(self.startup_group)
        layout.addStretch()

    def retranslateUi(self):
        self.startup_group.setTitle(self.tr("Startup Behavior"))
        self.startup_label.setText(self.tr("On Startup:"))
        
        self.startup_combo.clear()
        self.startup_combo.addItem(self.tr("Show Overview Page"), "show_overview")
        self.startup_combo.addItem(self.tr("Restore Last Session"), "restore_last")
        self.startup_combo.addItem(self.tr("Start New Project"), "new_project")

        self.auto_save_label.setText(self.tr("Auto-save Interval:"))
        self.auto_save_spin.setSuffix(self.tr(" minutes"))
        self.auto_save_spin.setSpecialValueText(self.tr("Disabled"))
        
        self.max_recent_label.setText(self.tr("Max Recent Files:"))
        self.check_updates_check.setText(self.tr("Check for updates automatically"))
        self.load_preferences()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)
    
    def load_preferences(self):
        """Load current preferences into UI controls."""
        general = self.pref_manager.general
        self.auto_save_spin.setRange(0, 60)
        self.max_recent_spin.setRange(0, 50)
        
        index = self.startup_combo.findData(general.startup_action)
        if index >= 0: self.startup_combo.setCurrentIndex(index)
        
        self.auto_save_spin.setValue(general.auto_save_interval)
        self.max_recent_spin.setValue(general.max_recent_files)
        self.check_updates_check.setChecked(general.check_for_updates)
    
    def apply_preferences(self):
        """Apply preferences from UI controls."""
        general = self.pref_manager.general
        general.startup_action = self.startup_combo.currentData()
        general.auto_save_interval = self.auto_save_spin.value()
        general.max_recent_files = self.max_recent_spin.value()
        general.check_for_updates = self.check_updates_check.isChecked()
        self.pref_manager.general = general


class DisplayPreferencesTab(QWidget):
    """Display and UI preferences tab."""
    
    def __init__(self, pref_manager: PreferencesManager):
        super().__init__()
        self.pref_manager = pref_manager
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.units_group = QGroupBox()
        units_layout = QFormLayout(self.units_group)
        self.unit_system_combo = QComboBox()
        self.unit_system_label = QLabel()
        units_layout.addRow(self.unit_system_label, self.unit_system_combo)
        layout.addWidget(self.units_group)
        self.format_group = QGroupBox()
        format_layout = QFormLayout(self.format_group)
        self.decimal_places_spin = QSpinBox()
        self.decimal_places_label = QLabel()
        format_layout.addRow(self.decimal_places_label, self.decimal_places_spin)
        layout.addWidget(self.format_group)
        self.graphics_group = QGroupBox()
        graphics_layout = QFormLayout(self.graphics_group)
        self.anti_aliasing_check = QCheckBox()
        self.smooth_lines_check = QCheckBox()
        self.animation_check = QCheckBox()
        graphics_layout.addRow(self.anti_aliasing_check)
        graphics_layout.addRow(self.smooth_lines_check)
        graphics_layout.addRow(self.animation_check)
        layout.addWidget(self.graphics_group)
        self.ui_group = QGroupBox()
        ui_layout = QFormLayout(self.ui_group)
        self.tooltips_check = QCheckBox()
        self.font_size_spin = QSpinBox()
        self.font_size_label = QLabel()
        ui_layout.addRow(self.tooltips_check)
        ui_layout.addRow(self.font_size_label, self.font_size_spin)
        layout.addWidget(self.ui_group)
        layout.addStretch()

    def retranslateUi(self):
        self.units_group.setTitle(self.tr("Unit System"))
        self.unit_system_label.setText(self.tr("Default Unit System:"))
        self.unit_system_combo.clear()
        self.unit_system_combo.addItem(self.tr("SI Units"), UnitSystem.SI)
        self.unit_system_combo.addItem(self.tr("Field Units"), UnitSystem.FIELD)
        
        self.format_group.setTitle(self.tr("Number Formatting"))
        self.decimal_places_label.setText(self.tr("Decimal Places:"))
        
        self.graphics_group.setTitle(self.tr("Graphics"))
        self.anti_aliasing_check.setText(self.tr("Enable anti-aliasing"))
        self.smooth_lines_check.setText(self.tr("Smooth lines"))
        self.animation_check.setText(self.tr("Enable animations"))
        
        self.ui_group.setTitle(self.tr("User Interface"))
        self.tooltips_check.setText(self.tr("Show tooltips"))
        self.font_size_label.setText(self.tr("Font Size Adjustment:"))
        self.font_size_spin.setSpecialValueText(self.tr("System Default"))
        
        self.load_preferences()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)
    
    def load_preferences(self):
        display = self.pref_manager.display
        self.decimal_places_spin.setRange(0, 8)
        self.font_size_spin.setRange(0, 20)
        
        index = self.unit_system_combo.findData(display.unit_system)
        if index >= 0: self.unit_system_combo.setCurrentIndex(index)
        
        self.decimal_places_spin.setValue(display.number_decimal_places)
        self.anti_aliasing_check.setChecked(display.graph_anti_aliasing)
        self.smooth_lines_check.setChecked(display.graph_smooth_lines)
        self.animation_check.setChecked(display.animation_enabled)
        self.tooltips_check.setChecked(display.show_tooltips)
        self.font_size_spin.setValue(display.font_size)
    
    def apply_preferences(self):
        display = self.pref_manager.display
        display.unit_system = self.unit_system_combo.currentData()
        display.number_decimal_places = self.decimal_places_spin.value()
        display.graph_anti_aliasing = self.anti_aliasing_check.isChecked()
        display.graph_smooth_lines = self.smooth_lines_check.isChecked()
        display.animation_enabled = self.animation_check.isChecked()
        display.show_tooltips = self.tooltips_check.isChecked()
        display.font_size = self.font_size_spin.value()
        self.pref_manager.display = display


class UnitsPreferencesTab(QWidget):
    """Unit-specific preferences tab."""
    
    def __init__(self, pref_manager: PreferencesManager):
        super().__init__()
        self.pref_manager = pref_manager
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)
        
        self.overrides_group = QGroupBox()
        self.overrides_layout = QFormLayout(self.overrides_group)
        
        categories = units_manager.get_all_categories()
        self.unit_combos = {}
        self.unit_labels = {}
        
        for category in categories:
            combo = QComboBox()
            available_units = units_manager.get_available_units_for_category(category)
            for unit in available_units:
                combo.addItem(unit, unit)
            
            label = QLabel()
            self.overrides_layout.addRow(label, combo)
            self.unit_combos[category] = combo
            self.unit_labels[category] = label
        
        self.scroll_layout.addWidget(self.overrides_group)
        self.scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

    def retranslateUi(self):
        self.overrides_group.setTitle(self.tr("Unit Overrides"))
        for category, combo in self.unit_combos.items():
            self.unit_labels[category].setText(self.tr(category.capitalize()) + ":")
            current_data = combo.currentData()
            combo.insertItem(0, self.tr("Use Default"), None)
            index = combo.findData(current_data)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setCurrentIndex(0)
        
        self.load_preferences()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)
    
    def load_preferences(self):
        units = self.pref_manager.units
        for category, combo in self.unit_combos.items():
            unit_override = getattr(units, f"{category}_unit", None)
            if unit_override:
                index = combo.findData(unit_override)
                if index >= 0: combo.setCurrentIndex(index)
            else:
                combo.setCurrentIndex(0)
    
    def apply_preferences(self):
        units = self.pref_manager.units
        for category, combo in self.unit_combos.items():
            unit_value = combo.currentData()
            setattr(units, f"{category}_unit", unit_value)
        self.pref_manager.units = units


class LanguagePreferencesTab(QWidget):
    """Language and localization preferences tab."""
    
    def __init__(self, pref_manager: PreferencesManager, i18n_manager: Optional[I18nManager] = None):
        super().__init__()
        self.pref_manager = pref_manager
        self.i18n_manager = i18n_manager
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.language_group = QGroupBox()
        language_layout = QFormLayout(self.language_group)
        self.language_combo = QComboBox()
        self.language_label = QLabel()
        language_layout.addRow(self.language_label, self.language_combo)
        self.use_system_lang_check = QCheckBox()
        language_layout.addRow(self.use_system_lang_check)
        layout.addWidget(self.language_group)
        
        self.format_group = QGroupBox()
        format_layout = QFormLayout(self.format_group)
        self.date_format_combo = QComboBox()
        self.date_format_label = QLabel()
        format_layout.addRow(self.date_format_label, self.date_format_combo)
        self.number_format_combo = QComboBox()
        self.number_format_label = QLabel()
        format_layout.addRow(self.number_format_label, self.number_format_combo)
        layout.addWidget(self.format_group)
        
        layout.addStretch()

    def retranslateUi(self):
        self.language_group.setTitle(self.tr("Language"))
        self.language_label.setText(self.tr("Language:"))
        self.use_system_lang_check.setText(self.tr("Use system language"))
        
        self.format_group.setTitle(self.tr("Formatting"))
        self.date_format_label.setText(self.tr("Date Format:"))
        self.date_format_combo.clear()
        self.date_format_combo.addItem(self.tr("System Default"), "system")
        self.date_format_combo.addItem(self.tr("ISO (YYYY-MM-DD)"), "iso")
        self.date_format_combo.addItem(self.tr("Local Format"), "local")

        self.number_format_label.setText(self.tr("Number Format:"))
        self.number_format_combo.clear()
        self.number_format_combo.addItem(self.tr("System Default"), "system")
        self.number_format_combo.addItem(self.tr("Dot Decimal (1,000.00)"), "dot_decimal")
        self.number_format_combo.addItem(self.tr("Comma Decimal (1.000,00)"), "comma_decimal")

        self.language_combo.clear()
        if self.i18n_manager:
            available_locales = self.i18n_manager.get_available_locales()
            for locale_code, language_name in available_locales:
                self.language_combo.addItem(language_name, locale_code)
        else:
            self.language_combo.addItem("English", "en")
        
        self.load_preferences()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)
    
    def load_preferences(self):
        language = self.pref_manager.language
        index = self.language_combo.findData(language.locale)
        if index >= 0: self.language_combo.setCurrentIndex(index)
        
        self.use_system_lang_check.setChecked(language.use_system_language)
        
        index = self.date_format_combo.findData(language.date_format)
        if index >= 0: self.date_format_combo.setCurrentIndex(index)
        
        index = self.number_format_combo.findData(language.number_format)
        if index >= 0: self.number_format_combo.setCurrentIndex(index)
    
    def apply_preferences(self):
        language = self.pref_manager.language
        language.locale = self.language_combo.currentData()
        language.use_system_language = self.use_system_lang_check.isChecked()
        language.date_format = self.date_format_combo.currentData()
        language.number_format = self.number_format_combo.currentData()
        self.pref_manager.language = language
        self.pref_manager.apply_language_preferences()


class AIPreferencesTab(QWidget):
    """AI services preferences tab with custom provider support."""
    def __init__(self, pref_manager: PreferencesManager):
        super().__init__()
        self.pref_manager = pref_manager
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        active_service_group = QGroupBox("Active Service")
        active_service_layout = QFormLayout(active_service_group)
        self.active_service_combo = QComboBox()
        active_service_layout.addRow("Active AI Service:", self.active_service_combo)
        layout.addWidget(active_service_group)

        config_group = QGroupBox("Service Configuration")
        config_layout = QHBoxLayout(config_group)
        
        list_layout = QVBoxLayout()
        self.service_list = QListWidget()
        self.service_list.itemSelectionChanged.connect(self._on_service_selected)
        list_layout.addWidget(self.service_list)
        
        button_layout = QHBoxLayout()
        self.add_btn = QPushButton("+")
        self.remove_btn = QPushButton("-")
        self.add_btn.clicked.connect(self._add_service)
        self.remove_btn.clicked.connect(self._remove_service)
        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.remove_btn)
        button_layout.addStretch()
        list_layout.addLayout(button_layout)
        config_layout.addLayout(list_layout, 1)

        editor_layout = QFormLayout()
        self.service_name_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.base_url_edit = QLineEdit()
        editor_layout.addRow("Service Name:", self.service_name_edit)
        editor_layout.addRow("API Key:", self.api_key_edit)
        editor_layout.addRow("Base URL:", self.base_url_edit)
        config_layout.addLayout(editor_layout, 2)

        layout.addWidget(config_group)
        layout.addStretch()

    def retranslateUi(self):
        pass

    def _load_services(self):
        self.service_list.clear()
        self.active_service_combo.clear()
        ai_prefs = self.pref_manager.ai
        for service_name in ai_prefs.services.keys():
            item = QListWidgetItem(service_name)
            is_default = service_name in AIPreferences().services
            item.setData(Qt.ItemDataRole.UserRole, is_default)
            self.service_list.addItem(item)
            self.active_service_combo.addItem(service_name)

        self.active_service_combo.setCurrentText(ai_prefs.active_service)
        if self.service_list.count() > 0:
            self.service_list.setCurrentRow(0)

    def _on_service_selected(self):
        selected_items = self.service_list.selectedItems()
        if not selected_items:
            self._clear_editor()
            return

        item = selected_items[0]
        service_name = item.text()
        is_default = item.data(Qt.ItemDataRole.UserRole)

        self.service_name_edit.setText(service_name)
        self.service_name_edit.setReadOnly(is_default)
        self.remove_btn.setEnabled(not is_default)

        ai_prefs = self.pref_manager.ai
        service_data = ai_prefs.services.get(service_name, {})
        self.api_key_edit.setText(service_data.get("api_key", ""))
        self.base_url_edit.setText(service_data.get("base_url", ""))

    def _clear_editor(self):
        self.service_name_edit.clear()
        self.api_key_edit.clear()
        self.base_url_edit.clear()

    def _add_service(self):
        new_name, ok = QInputDialog.getText(self, "New Service", "Enter name for the new AI service:")
        if ok and new_name:
            ai_prefs = self.pref_manager.ai
            if new_name in ai_prefs.services:
                QMessageBox.warning(self, "Duplicate Name", "A service with this name already exists.")
                return
            
            ai_prefs.services[new_name] = {"api_key": "", "base_url": ""}
            self.pref_manager.ai = ai_prefs
            self._load_services()
            for i in range(self.service_list.count()):
                if self.service_list.item(i).text() == new_name:
                    self.service_list.setCurrentRow(i)
                    break

    def _remove_service(self):
        selected_items = self.service_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        service_name = item.text()
        is_default = item.data(Qt.ItemDataRole.UserRole)

        if is_default:
            QMessageBox.information(self, "Cannot Delete", "Default services cannot be deleted.")
            return

        reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete '{service_name}'?")
        if reply == QMessageBox.StandardButton.Yes:
            ai_prefs = self.pref_manager.ai
            if service_name in ai_prefs.services:
                del ai_prefs.services[service_name]
                self.pref_manager.ai = ai_prefs
                self._load_services()

    def load_preferences(self):
        self._load_services()

    def apply_preferences(self):
        selected_items = self.service_list.selectedItems()
        if selected_items:
            service_name = selected_items[0].text()
            original_name = self.service_name_edit.text()
            ai_prefs = self.pref_manager.ai

            if service_name != original_name:
                if original_name in ai_prefs.services:
                    QMessageBox.warning(self, "Duplicate Name", "A service with this name already exists.")
                    self.service_name_edit.setText(service_name)
                else:
                    ai_prefs.services[original_name] = ai_prefs.services.pop(service_name)
                    service_name = original_name

            ai_prefs.services[service_name] = {
                "api_key": self.api_key_edit.text(),
                "base_url": self.base_url_edit.text()
            }
            self.pref_manager.ai = ai_prefs

        ai_prefs = self.pref_manager.ai
        ai_prefs.active_service = self.active_service_combo.currentText()
        self.pref_manager.ai = ai_prefs


class AdvancedPreferencesTab(QWidget):
    """Advanced application preferences tab."""
    
    def __init__(self, pref_manager: PreferencesManager):
        super().__init__()
        self.pref_manager = pref_manager
        self._setup_ui()
        self.retranslateUi()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.logging_group = QGroupBox()
        logging_layout = QFormLayout(self.logging_group)
        self.log_level_combo = QComboBox()
        self.log_level_label = QLabel()
        logging_layout.addRow(self.log_level_label, self.log_level_combo)
        self.debug_mode_check = QCheckBox()
        logging_layout.addRow(self.debug_mode_check)
        layout.addWidget(self.logging_group)
        
        self.perf_group = QGroupBox()
        perf_layout = QFormLayout(self.perf_group)
        self.cache_size_spin = QSpinBox()
        self.cache_size_label = QLabel()
        perf_layout.addRow(self.cache_size_label, self.cache_size_spin)
        self.max_threads_spin = QSpinBox()
        self.max_threads_label = QLabel()
        perf_layout.addRow(self.max_threads_label, self.max_threads_spin)
        self.gpu_accel_check = QCheckBox()
        perf_layout.addRow(self.gpu_accel_check)
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_label = QLabel()
        perf_layout.addRow(self.memory_limit_label, self.memory_limit_spin)
        layout.addWidget(self.perf_group)
        
        layout.addStretch()

    def retranslateUi(self):
        self.logging_group.setTitle(self.tr("Logging"))
        self.log_level_label.setText(self.tr("Log Level:"))
        self.log_level_combo.clear()
        self.log_level_combo.addItem("DEBUG", "DEBUG")
        self.log_level_combo.addItem("INFO", "INFO")
        self.log_level_combo.addItem("WARNING", "WARNING")
        self.log_level_combo.addItem("ERROR", "ERROR")
        self.log_level_combo.addItem("CRITICAL", "CRITICAL")
        self.debug_mode_check.setText(self.tr("Enable debug mode"))
        
        self.perf_group.setTitle(self.tr("Performance"))
        self.cache_size_label.setText(self.tr("Cache Size:"))
        self.cache_size_spin.setSuffix(self.tr(" MB"))
        self.max_threads_label.setText(self.tr("Max Threads:"))
        self.gpu_accel_check.setText(self.tr("GPU acceleration"))
        self.memory_limit_label.setText(self.tr("Memory Usage Limit:"))
        self.memory_limit_spin.setSuffix(self.tr(" %"))
        self.load_preferences()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)
    
    def load_preferences(self):
        advanced = self.pref_manager.advanced
        self.cache_size_spin.setRange(10, 1000)
        self.max_threads_spin.setRange(1, 16)
        self.memory_limit_spin.setRange(50, 95)
        
        index = self.log_level_combo.findData(advanced.logging_level)
        if index >= 0: self.log_level_combo.setCurrentIndex(index)
        
        self.debug_mode_check.setChecked(advanced.enable_debug_mode)
        self.cache_size_spin.setValue(advanced.cache_size_mb)
        self.max_threads_spin.setValue(advanced.max_threads)
        self.gpu_accel_check.setChecked(advanced.gpu_acceleration)
        self.memory_limit_spin.setValue(advanced.memory_usage_limit)
    
    def apply_preferences(self):
        advanced = self.pref_manager.advanced
        advanced.logging_level = self.log_level_combo.currentData()
        advanced.enable_debug_mode = self.debug_mode_check.isChecked()
        advanced.cache_size_mb = self.cache_size_spin.value()
        advanced.max_threads = self.max_threads_spin.value()
        advanced.gpu_acceleration = self.gpu_accel_check.isChecked()
        advanced.memory_usage_limit = self.memory_limit_spin.value()
        self.pref_manager.advanced = advanced
