import logging
from typing import Optional, Any, Dict, Type, get_origin, get_args, Union, List
from copy import deepcopy
from dataclasses import fields, is_dataclass, asdict, Field
import json
from pathlib import Path
from types import UnionType
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QScrollArea, QLabel,
    QPushButton, QMessageBox, QFileDialog, QHBoxLayout, QFrame, QStackedWidget,
    QGroupBox, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QSplitter, QListView
)
from PyQt6.QtGui import QIcon, QStandardItemModel, QStandardItem
from PyQt6.QtCore import pyqtSignal, Qt, QEvent

from ui.widgets.parameter_input_group import ParameterInputGroup
try:
    from core.data_models import (
        EconomicParameters, EORParameters, OperationalParameters, ProfileParameters,
        GeneticAlgorithmParams, BayesianOptimizationParams, CO2StorageParameters,
        ParticleSwarmParams, DifferentialEvolutionParams, TuningParams, AdvancedEngineParams,
        GeomechanicsParameters
    )
    from ui.dialogs.injection_scheme_dialog import InjectionSchemeDialog
    CONFIGURABLE_DATACLASSES: Dict[str, Type] = {
        "Economic": EconomicParameters,
        "CO2 Storage": CO2StorageParameters,
        "EOR": EORParameters,
        "Operational": OperationalParameters,
        "Profile": ProfileParameters,
        "Geomechanics": GeomechanicsParameters,
        "Advanced Engine": AdvancedEngineParams,
        "Genetic Algorithm": GeneticAlgorithmParams,
        "Bayesian Optimizer": BayesianOptimizationParams,
        "Particle Swarm": ParticleSwarmParams,
        "Differential Evolution": DifferentialEvolutionParams,
        "Tuning": TuningParams
    }
    ALGORITHM_CLASSES = {
        GeneticAlgorithmParams, BayesianOptimizationParams,
        ParticleSwarmParams, DifferentialEvolutionParams
    }
except ImportError as e:
    logging.critical(f"ConfigWidget: Core configuration dataclasses not found. {e}")
    CONFIGURABLE_DATACLASSES = {}
    ALGORITHM_CLASSES = set()

logger = logging.getLogger(__name__)


class ConfigWidget(QWidget):
    """A visually organized widget for editing application configurations defined by dataclasses."""
    configuration_changed = pyqtSignal(str, str, object)
    configurations_updated = pyqtSignal(dict)
    save_configuration_to_file_requested = pyqtSignal(dict)
    help_requested = pyqtSignal(str)
    engine_selection_changed = pyqtSignal(str)  # Signal emitted when engine type changes

    def __init__(self, config_manager: "ConfigManager", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config_manager = config_manager
        
        ui_config = self.config_manager.get_section("ui_config").get("config_widget", {})
        self.OIL_PROFILE_DISPLAY_MAP: Dict[str, str] = ui_config.get("oil_profile_display_map", {})
        self.INJECTION_PROFILE_DISPLAY_MAP: Dict[str, str] = ui_config.get("injection_profile_display_map", {})
        self.OBJECTIVES_DISPLAY_MAP: Dict[str, str] = ui_config.get("objectives_display_map", {})
        self.REVERSE_OIL_PROFILE_MAP: Dict[str, str] = {v: k for k, v in self.OIL_PROFILE_DISPLAY_MAP.items()}
        self.REVERSE_INJECTION_PROFILE_MAP: Dict[str, str] = {v: k for k, v in self.INJECTION_PROFILE_DISPLAY_MAP.items()}
        self.REVERSE_OBJECTIVES_MAP: Dict[str, str] = {v: k for k, v in self.OBJECTIVES_DISPLAY_MAP.items()}
        self.SPECIAL_DROPDOWNS: Dict[str, List[str]] = ui_config.get("special_dropdowns", {})
        
        self.recovery_model_params: Dict[str, Dict[str, Any]] = {}

        self.default_instances: Dict[str, Any] = {
            dc_type.__name__: dc_type() for dc_type in CONFIGURABLE_DATACLASSES.values()
        } if CONFIGURABLE_DATACLASSES else {}
        
        self.config_instances: Dict[str, Any] = deepcopy(self.default_instances)
        self.input_groups: Dict[str, ParameterInputGroup] = {}
        self.operational_widgets: Dict[str, QWidget] = {}
        self._is_dirty = False

        self._setup_ui()
        self.setStyleSheet(self._get_stylesheet())
        self.retranslateUi()
        self._set_dirty(False)

    def _get_stylesheet(self) -> str:
        return '''
            QListView {
                border: 1px solid #C2C7CB;
                background-color: #f0f0f0;
            }
            QListView::item {
                padding: 8px;
            }
            QListView::item:selected {
                background-color: #c7d8f3;
                border: 1px solid #a3bde3;
                color: #000;
            }
            QScrollArea { border: none; background-color: white; }
            ParameterInputGroup[isModified="true"] {
                background-color: #e8f4e8; border: 1px solid #a3d8a3;
                border-radius: 6px; margin: 2px 0;
            }
            ParameterInputGroup[isModified="true"] > QLabel { font-weight: bold; }
            #ConfigSourceLabel { font-style: italic; color: #555; padding: 4px; background-color: #f0f0f0; border-radius: 4px; }
            #ApplyDiscardFrame { background-color: #fffac1; border-radius: 5px; border: 1px solid #f0e68c; }
            QGroupBox { font-weight: bold; margin-top: 10px; }
            QLineEdit#SearchBar {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        '''

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(10)

        top_button_layout = QHBoxLayout()
        self.load_btn = QPushButton(QIcon.fromTheme("document-open"), "", self)
        self.save_btn = QPushButton(QIcon.fromTheme("document-save"), "", self)
        self.reset_btn = QPushButton(QIcon.fromTheme("view-refresh"), "", self)
        top_button_layout.addWidget(self.load_btn)
        top_button_layout.addWidget(self.save_btn)
        top_button_layout.addStretch()
        top_button_layout.addWidget(self.reset_btn)

        status_layout = QHBoxLayout()
        self.status_label = QLabel()
        self.config_source_label = QLabel()
        self.config_source_label.setObjectName("ConfigSourceLabel")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.config_source_label)
        status_layout.addStretch()

        self.apply_discard_frame = QFrame()
        self.apply_discard_frame.setObjectName("ApplyDiscardFrame")
        apply_layout = QHBoxLayout(self.apply_discard_frame)
        self.apply_btn = QPushButton(QIcon.fromTheme("dialog-ok-apply"), "", self)
        self.discard_btn = QPushButton(QIcon.fromTheme("dialog-cancel"), "", self)
        self.unsaved_changes_label = QLabel()
        apply_layout.addWidget(self.unsaved_changes_label)
        apply_layout.addStretch()
        apply_layout.addWidget(self.discard_btn)
        apply_layout.addWidget(self.apply_btn)
        
        main_layout.addLayout(top_button_layout)
        main_layout.addLayout(status_layout)
        main_layout.addWidget(self.apply_discard_frame)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(self.splitter, 1)

        # Left side: Search and Category List
        left_pane = QWidget()
        left_pane.setMinimumWidth(180)
        left_pane.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.search_bar = QLineEdit(self)
        self.search_bar.setObjectName("SearchBar")
        self.search_bar.setPlaceholderText(self.tr("Search settings..."))
        self.search_bar.textChanged.connect(self._filter_categories)
        left_layout.addWidget(self.search_bar)

        self.category_list = QListView(self)
        self.category_model = QStandardItemModel(self)
        self.category_list.setModel(self.category_model)
        left_layout.addWidget(self.category_list)
        self.splitter.addWidget(left_pane)

        # Right side: Settings Stack
        self.settings_stack = QStackedWidget(self)
        self.settings_stack.setMinimumWidth(400)
        self.splitter.addWidget(self.settings_stack)

        # Use stretch factors instead of fixed sizes for better resizing
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([250, 750])  # Initial sizes only

        self.load_btn.clicked.connect(self._load_from_file)
        self.save_btn.clicked.connect(self._save_to_file)
        self.reset_btn.clicked.connect(self._confirm_reset_all)
        self.apply_btn.clicked.connect(self._apply_changes)
        self.discard_btn.clicked.connect(self._discard_changes)
        self.category_list.selectionModel().currentChanged.connect(self._on_category_selected)

    def _filter_categories(self, text: str):
        for i in range(self.category_model.rowCount()):
            item = self.category_model.item(i)
            self.category_list.setRowHidden(i, text.lower() not in item.text().lower())

    def _on_category_selected(self, current, previous):
        if current.isValid():
            self.settings_stack.setCurrentIndex(current.row())
        
    def retranslateUi(self):
        # Translate static UI elements
        self.load_btn.setText(self.tr(" Load from File..."))
        self.save_btn.setText(self.tr(" Save to File..."))
        self.reset_btn.setText(self.tr(" Reset All to Defaults"))
        self.status_label.setText(self.tr("<b>Current Configuration Source:</b>"))
        self.apply_btn.setText(self.tr(" Apply Changes"))
        self.discard_btn.setText(self.tr(" Discard Changes"))
        self.unsaved_changes_label.setText(self.tr("You have unsaved changes:"))
        
        # Translate maps and dropdown content

        # Repopulate dynamic content with new translations
        self._populate_all_forms()
        # Set initial text
        if not self.config_source_label.text():
            self.config_source_label.setText(self.tr("Application Defaults"))

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def get_current_config_data_instances(self) -> Dict[str, Any]:
        """Returns a deep copy of the currently applied configuration instances."""
        data = deepcopy(self.config_instances)
        data["RecoveryModelKwargsDefaults"] = deepcopy(self.recovery_model_params)
        return data

    def update_configs_from_project(self, config_instances: Dict[str, Any]):
        """Updates the widget with configuration data from a loaded project."""
        logger.info("ConfigWidget updating with new configuration data from project.")
        self.recovery_model_params = config_instances.pop("RecoveryModelKwargsDefaults", {})
        self.update_configurations(config_instances)
        self.config_source_label.setText(self.tr("Loaded from Project"))

    def _set_dirty(self, is_dirty: bool):
        self._is_dirty = is_dirty
        self.apply_discard_frame.setVisible(is_dirty)

    def _mark_as_dirty(self, _=None):
        if not self._is_dirty: self._set_dirty(True)
        sender = self.sender()
        if isinstance(sender, ParameterInputGroup):
            if sender.param_name.endswith("oil_profile_type"):
                self._update_profile_param_visibility()

    def _discard_changes(self):
        logger.debug("Discarding pending configuration changes.")
        self.update_configurations(self.config_instances)

    def _apply_changes(self):
        logger.info("Attempting to apply configuration changes.")

        # DEBUG: Log current EOR parameters before applying changes
        eor_instance = self.config_instances.get('EORParameters')
        if eor_instance:
            logger.info(f"ConfigWidget - Current EOR Parameters - Injection Scheme: '{eor_instance.injection_scheme}', "
                       f"WAG Ratio: {eor_instance.WAG_ratio}")

        pending_data = {dc_name: asdict(instance) for dc_name, instance in self.config_instances.items()}
        pending_recovery_params = deepcopy(self.recovery_model_params)
        all_valid = True

        # Handle engine_type from combo box (not tracked in input_groups)
        if hasattr(self, 'engine_type_combo'):
            engine_type = self.engine_type_combo.currentData()
            if engine_type and 'AdvancedEngineParams' in pending_data:
                pending_data['AdvancedEngineParams']['engine_type'] = engine_type
                pending_data['AdvancedEngineParams']['use_simple_physics'] = (engine_type == 'simple')
                logger.info(f"ConfigWidget: Including engine_type '{engine_type}' in pending changes")
        
        for key, widget in self.input_groups.items():
            if not widget.isVisible() or '.' not in key: continue
            
            scope, param_name = key.split('.', 1)
            raw_value = widget.get_value()

            if scope in pending_data:
                if widget.is_checkable() and not widget.is_checked():
                    pending_data[scope][param_name] = None
                    continue

                if param_name == 'oil_profile_type': internal_value = self.REVERSE_OIL_PROFILE_MAP.get(raw_value, raw_value)
                elif param_name == 'injection_profile_type': internal_value = self.REVERSE_INJECTION_PROFILE_MAP.get(raw_value, raw_value)
                else: internal_value = raw_value

                try:
                    field_type = self.config_instances[scope].__class__.__annotations__[param_name]
                    coerced_value = self._coerce_value(internal_value, field_type)
                    pending_data[scope][param_name] = coerced_value
                    widget.clear_error()
                except (ValueError, TypeError) as e:
                    widget.show_error(str(e)); all_valid = False
            
            elif scope in pending_recovery_params:
                try:
                    coerced_value = float(raw_value) if '.' in str(raw_value) else int(raw_value)
                    pending_recovery_params[scope][param_name] = coerced_value
                    widget.clear_error()
                except (ValueError, TypeError) as e:
                    widget.show_error(str(e)); all_valid = False
        
        if self.operational_widgets:
            op_instance_name = OperationalParameters.__name__
            try:
                enable_checkbox = self.operational_widgets['enable_target_seeking_checkbox']
                if enable_checkbox.isChecked():
                    display_name = self.operational_widgets['target_objective_combo'].currentText()
                    pending_data[op_instance_name]['target_objective_name'] = self.REVERSE_OBJECTIVES_MAP.get(display_name, 'npv')
                    pending_data[op_instance_name]['target_objective_value'] = float(self.operational_widgets['target_value_input'].value())
                else:
                    pending_data[op_instance_name]['target_objective_name'] = None
                    pending_data[op_instance_name]['target_objective_value'] = None
            except (ValueError, TypeError) as e:
                QMessageBox.critical(self, self.tr("Validation Error"), self.tr("Invalid input for Target Seeking: {}").format(e)); all_valid = False

        if not all_valid:
            QMessageBox.critical(self, self.tr("Validation Error"), self.tr("One or more fields have invalid values. Please correct the highlighted fields."))
            return

        try:
            new_instances = {}
            for name, data in pending_data.items():
                if name in self.default_instances:
                    dc_type = self.default_instances[name].__class__
                    # Filter out None values for non-optional fields before creating the instance
                    filtered_data = {k: v for k, v in data.items() if v is not None or (get_origin(dc_type.__annotations__.get(k)) in (Union, UnionType) and type(None) in get_args(dc_type.__annotations__.get(k)))}
                    new_instances[name] = dc_type(**filtered_data)
            self.config_instances = new_instances
            self.recovery_model_params = pending_recovery_params
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, self.tr("Validation Error"), self.tr("Could not apply changes due to inconsistent data:\n\n{}").format(e))
            return
            
        logger.info("Configuration changes applied successfully.")
        
        self.config_source_label.setText(self.tr("Custom Applied"))
        self.configurations_updated.emit(self.get_current_config_data_instances())
        self._set_dirty(False)

        for group in self.input_groups.values():
            group.setProperty("isModified", False)
            group.style().unpolish(group)
            group.style().polish(group)

        self._update_profile_param_visibility()
        self._update_recovery_model_widgets()

    def update_configurations(self, config_data: Dict[str, Any]):
        """Updates the internal state and repopulates all UI forms from the given data."""
        self.config_instances = deepcopy(config_data)
        self._populate_all_forms()
        self._set_dirty(False)

    def _populate_all_forms(self):
        """Clears and rebuilds all configuration pages based on current instances."""
        current_index = self.settings_stack.currentIndex()
        self.category_model.clear()
        while self.settings_stack.count() > 0:
            widget = self.settings_stack.widget(0)
            self.settings_stack.removeWidget(widget)
            if widget:
                widget.deleteLater()
        self.input_groups.clear()
        self.operational_widgets.clear()

        if not CONFIGURABLE_DATACLASSES:
            self.settings_stack.addWidget(QLabel(self.tr("Configuration models not loaded.")))
            self.category_model.appendRow(QStandardItem(self.tr("Error")))
            return

        pages = {}
        
        # Create algorithm settings page first
        algo_page = QWidget()
        algo_layout = QVBoxLayout(algo_page)
        algo_layout.setSpacing(15)
        algo_layout.setContentsMargins(10, 15, 10, 15)
        pages[self.tr("Algorithm Settings")] = algo_page

        for name, dc_type in CONFIGURABLE_DATACLASSES.items():
            instance = self.config_instances.get(dc_type.__name__)
            if not instance: continue

            if dc_type in ALGORITHM_CLASSES:
                self._create_algorithm_groupbox(name, instance, algo_page)
            else:
                if name == "EOR":
                    # Create EOR Parameters page
                    eor_params_page = QWidget()
                    eor_params_layout = QVBoxLayout(eor_params_page)
                    eor_params_layout.setSpacing(15)
                    eor_params_layout.setContentsMargins(10, 15, 10, 15)
                    self._create_eor_parameters_page(instance, eor_params_layout)
                    pages[self.tr("EOR Parameters")] = eor_params_page

                    # Create Engine constrains page
                    engine_constrains_page = QWidget()
                    engine_constrains_layout = QVBoxLayout(engine_constrains_page)
                    engine_constrains_layout.setSpacing(15)
                    engine_constrains_layout.setContentsMargins(10, 15, 10, 15)
                    self._create_engine_constrains_page(instance, engine_constrains_layout)
                    pages[self.tr("Engine constrains")] = engine_constrains_page
                else:
                    page_widget = QWidget()
                    page_layout = QVBoxLayout(page_widget)
                    page_layout.setSpacing(15)
                    page_layout.setContentsMargins(10, 15, 10, 15)
                    if name == "Operational":
                        self._create_operational_page(instance, page_layout)
                    else:
                        self._create_standard_page(instance, page_layout)
                    pages[self.tr(name)] = page_widget
        
        # Add pages and categories in a specific order
        category_order = ["Operational", "EOR Parameters", "Engine constrains", "Profile", "Economic", "CO2 Storage", "Advanced Engine", "Algorithm Settings"]
        for name in category_order:
            translated_name = self.tr(name)
            if translated_name in pages:
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(pages[translated_name])
                self.settings_stack.addWidget(scroll_area)
                self.category_model.appendRow(QStandardItem(translated_name))

        self._update_profile_param_visibility()
        if current_index != -1 and current_index < self.settings_stack.count():
            self.category_list.setCurrentIndex(self.category_model.index(current_index, 0))
        elif self.category_model.rowCount() > 0:
            self.category_list.setCurrentIndex(self.category_model.index(0, 0))

    def _create_eor_parameters_page(self, instance: Any, layout: QVBoxLayout):
        # Main EOR parameters (not in nested injection scheme objects)
        main_injection_fields = [
            'injection_scheme', 'injection_rate', 'target_pressure_psi', 'max_pressure_psi',
            'min_injection_rate_mscfd', 'max_injection_rate_mscfd', 'mobility_ratio',
            'density_contrast', 'dip_angle', 'interfacial_tension',
            'sor', 'gas_oil_ratio_at_breakthrough', 'water_cut_bwow',
            'co2_density_tonne_per_mscf', 'injection_gor', 'initial_gor',
            'default_mmp_fallback', 'kv_kh_ratio'
        ]

        # Relative permeability parameters
        rel_perm_fields = [
            's_gc', 'n_o', 'n_g', 's_wc', 's_orw', 'n_w', 'n_ow'
        ]

        # Well control parameters
        well_control_fields = [
            'productivity_index', 'wellbore_pressure', 'well_shut_in_threshold_bpd',
            'max_injector_bhp_psi', 'timestep_days'
        ]
        # Create main parameters group
        main_group = QGroupBox(self.tr("Main Injection Parameters"))
        main_layout = QFormLayout(main_group)

        for field_name in main_injection_fields:
            field = next((f for f in fields(instance) if f.name == field_name), None)
            if field:
                self._create_input_group_for_field(instance, field, main_layout, "EORParameters")

        layout.addWidget(main_group)

        # Create injection scheme configuration area
        scheme_group = QGroupBox(self.tr("Injection Scheme Configuration"))
        scheme_layout = QVBoxLayout(scheme_group)

        # Current scheme display
        current_scheme_layout = QHBoxLayout()
        current_scheme_layout.addWidget(QLabel(self.tr("Current Scheme:")))
        self.current_eor_scheme_display = QLabel()
        self.current_eor_scheme_display.setStyleSheet("font-weight: bold; color: #0066cc;")
        current_scheme_layout.addWidget(self.current_eor_scheme_display)
        current_scheme_layout.addStretch()
        scheme_layout.addLayout(current_scheme_layout)

        # Configure scheme button
        config_btn = QPushButton(self.tr("Configure Injection Scheme..."))
        config_btn.clicked.connect(self._open_eor_injection_scheme_dialog)
        scheme_layout.addWidget(config_btn)

        layout.addWidget(scheme_group)

        # Create relative permeability group
        rel_perm_group = QGroupBox(self.tr("Relative Permeability Parameters"))
        rel_perm_layout = QFormLayout(rel_perm_group)

        for field_name in rel_perm_fields:
            field = next((f for f in fields(instance) if f.name == field_name), None)
            if field:
                self._create_input_group_for_field(instance, field, rel_perm_layout, "EORParameters")

        layout.addWidget(rel_perm_group)

        # Create well control group
        well_control_group = QGroupBox(self.tr("Well Control Parameters"))
        well_control_layout = QFormLayout(well_control_group)

        for field_name in well_control_fields:
            field = next((f for f in fields(instance) if f.name == field_name), None)
            if field:
                self._create_input_group_for_field(instance, field, well_control_layout, "EORParameters")

        layout.addWidget(well_control_group)

        # Update current scheme display
        self._update_current_eor_scheme_display()

    def _create_engine_constrains_page(self, instance: Any, layout: QVBoxLayout):
        # Engine selection dropdown (FIRST item - single source of truth for engine selection)
        engine_group = QGroupBox(self.tr("Engine Selection"))
        engine_group.setToolTip(self.tr("Select the simulation engine to use. This is the primary location for engine selection."))
        engine_layout = QFormLayout(engine_group)

        self.engine_type_combo = QComboBox()
        self.engine_type_combo.addItem(self.tr("Simple Engine (Fast, Approximate)"), "simple")
        self.engine_type_combo.addItem(self.tr("Detailed Engine (Accurate, Physics-based)"), "detailed")
        self.engine_type_combo.addItem(self.tr("Surrogate Engine (ML-based, Very Fast)"), "surrogate")

        # Set current engine from config instances
        current_engine = getattr(instance, 'engine_type', 'simple')
        for i in range(self.engine_type_combo.count()):
            if self.engine_type_combo.itemData(i) == current_engine:
                self.engine_type_combo.setCurrentIndex(i)
                break

        # Connect signal to handle changes
        self.engine_type_combo.currentTextChanged.connect(self._on_engine_type_combo_changed)

        engine_layout.addRow(self.tr("Simulation Engine:"), self.engine_type_combo)
        layout.addWidget(engine_group)

        # Existing constraint fields
        form_layout = QFormLayout()

        include_fields = [
            'min_gravity_factor', 'max_gravity_factor', 'min_sor', 'max_sor',
            'min_transition_alpha', 'max_transition_alpha', 'min_transition_beta', 'max_transition_beta',
            'min_WAG_ratio', 'max_WAG_ratio',
            'min_cycle_length_days', 'max_cycle_length_days', 'min_water_fraction', 'max_water_fraction',
            'min_productivity_index', 'max_productivity_index',
            'min_wellbore_pressure', 'max_wellbore_pressure'
        ]

        instance_name = type(instance).__name__
        for field in fields(instance):
            if field.name in include_fields:
                self._create_input_group_for_field(instance, field, form_layout, instance_name)

        layout.addLayout(form_layout)
        layout.addStretch()

    def _create_standard_page(self, instance: Any, layout: QVBoxLayout, exclude_fields: List[str] = None):
        form_layout = QFormLayout()
        bounds_group = QGroupBox(self.tr("Optimizer Search Bounds (Optional)"))
        bounds_group.setToolTip(self.tr("Override automatic search bounds. Leave fields blank to use dynamic, physically-based estimates."))
        bounds_layout = QFormLayout(bounds_group)
        bounds_group.setVisible(False)

        params_with_bounds = {}
        
        handled_fields = set()
        instance_name = type(instance).__name__

        for field in fields(instance):
            if field.type is np.ndarray:  # Skip numpy arrays
                continue

            if (exclude_fields and field.name in exclude_fields) or (field.name in handled_fields):
                continue

            if isinstance(field.type, type) and is_dataclass(field.type):
                nested_instance = getattr(instance, field.name)
                nested_group = QGroupBox(field.name.replace("_", " ").title())
                nested_layout = QFormLayout(nested_group)
                for nested_field in fields(nested_instance):
                    self._create_input_group_for_field(nested_instance, nested_field, nested_layout, instance_name)
                layout.addWidget(nested_group)
                handled_fields.add(field.name)

            elif field.type is dict:
                dict_instance = getattr(instance, field.name)
                dict_group = QGroupBox(field.name.replace("_", " ").title())
                dict_layout = QFormLayout(dict_group)
                for key, value in dict_instance.items():
                    fake_field = Field(name=key, type=type(value), default=value, init=True, repr=True, hash=None, compare=True, kw_only=False)
                    self._create_input_group_for_field(dict_instance, fake_field, dict_layout, instance_name)
                layout.addWidget(dict_group)
                handled_fields.add(field.name)

            elif field.name in params_with_bounds:
                main_param_field = field
                min_bound_field_name, max_bound_field_name = params_with_bounds[field.name]
                min_field = next((f for f in fields(instance) if f.name == min_bound_field_name), None)
                max_field = next((f for f in fields(instance) if f.name == max_bound_field_name), None)
                self._create_input_group_for_field(instance, main_param_field, form_layout, instance_name)
                if min_field: self._create_input_group_for_field(instance, min_field, bounds_layout, instance_name); handled_fields.add(min_field.name)
                if max_field: self._create_input_group_for_field(instance, max_field, bounds_layout, instance_name); handled_fields.add(max_field.name)
                bounds_group.setVisible(True); handled_fields.add(main_param_field.name)
            elif field.name.startswith(('min_', 'max_')):
                continue
            else:
                self._create_input_group_for_field(instance, field, form_layout, instance_name)

        layout.addLayout(form_layout)
        if bounds_group.isVisible():
            layout.addWidget(bounds_group)
        layout.addStretch()

    def _create_algorithm_groupbox(self, name: str, instance: Any, parent_widget: QWidget):
        layout = parent_widget.layout()
        group_box = QGroupBox(self.tr(name))
        form_layout = QFormLayout(group_box) 
        
        for field in fields(instance):
            self._create_input_group_for_field(instance, field, form_layout, type(instance).__name__)
            
        layout.addWidget(group_box)

    def _create_operational_page(self, op_instance: OperationalParameters, layout: QVBoxLayout):
        general_group = QGroupBox(self.tr("General"))
        general_layout = QFormLayout(general_group)
        exclude_from_general = {"target_objective_name", "target_objective_value", "target_recovery_factor",
                                "recovery_model_selection", "injection_scheme"}
        for field in fields(op_instance):
            if field.name not in exclude_from_general:
                self._create_input_group_for_field(op_instance, field, general_layout, "Operational")
        layout.addWidget(general_group)
        
        # Injection Settings Section
        injection_group = QGroupBox(self.tr("Injection Settings"))
        injection_layout = QVBoxLayout(injection_group)
        
        # Current injection scheme display
        current_scheme_layout = QHBoxLayout()
        current_scheme_label = QLabel(self.tr("Current Injection Scheme:"))
        self.current_scheme_display = QLabel()
        self.current_scheme_display.setStyleSheet("font-weight: bold; color: #0066cc;")
        current_scheme_layout.addWidget(current_scheme_label)
        current_scheme_layout.addWidget(self.current_scheme_display)
        current_scheme_layout.addStretch()
        injection_layout.addLayout(current_scheme_layout)
        
        # Injection scheme configuration button
        scheme_button_layout = QHBoxLayout()
        self.configure_scheme_btn = QPushButton(self.tr("Configure Injection Scheme..."))
        self.configure_scheme_btn.clicked.connect(self._open_injection_scheme_dialog)
        scheme_button_layout.addWidget(self.configure_scheme_btn)
        scheme_button_layout.addStretch()
        injection_layout.addLayout(scheme_button_layout)
        
        # Basic injection parameters
        basic_injection_group = QGroupBox(self.tr("Basic Injection Parameters"))
        basic_injection_layout = QFormLayout(basic_injection_group)
        
        # Get EOR instance for injection parameters
        eor_instance = self.config_instances.get('EORParameters')
        if eor_instance:
            # Add basic injection parameters
            injection_rate_field = next((f for f in fields(eor_instance) if f.name == "injection_rate"), None)
            if injection_rate_field:
                self._create_input_group_for_field(eor_instance, injection_rate_field, basic_injection_layout, "EORParameters")
            
            target_pressure_field = next((f for f in fields(eor_instance) if f.name == "target_pressure_psi"), None)
            if target_pressure_field:
                self._create_input_group_for_field(eor_instance, target_pressure_field, basic_injection_layout, "EORParameters")
        
        injection_layout.addWidget(basic_injection_group)
        layout.addWidget(injection_group)
        
        # Update current scheme display
        self._update_current_scheme_display()
        
        recovery_group = QGroupBox(self.tr("Recovery Model"))
        recovery_layout = QVBoxLayout(recovery_group)
        recovery_form_layout = QFormLayout()
        self._create_input_group_for_field(
            op_instance, next(f for f in fields(op_instance) if f.name == "recovery_model_selection"), recovery_form_layout, "Operational"
        )
        recovery_layout.addLayout(recovery_form_layout)
        
        self.recovery_model_container = QWidget(recovery_group)
        self.recovery_model_layout = QVBoxLayout(self.recovery_model_container)
        self.recovery_model_layout.setContentsMargins(20, 10, 0, 0)
        recovery_layout.addWidget(self.recovery_model_container)
        
        layout.addWidget(recovery_group)
        
        target_group = QGroupBox(self.tr("Optimizer Target Seeking")); target_group_layout = QVBoxLayout(target_group)
        enable_checkbox = QCheckBox(self.tr("Enable Target Seeking")); enable_checkbox.setToolTip(self.tr("Check this to make the optimizer try to achieve a specific target value for an objective."))
        target_group_layout.addWidget(enable_checkbox)
        self.operational_widgets['enable_target_seeking_checkbox'] = enable_checkbox
        target_controls_group = QGroupBox(); target_controls_group.setFlat(True)
        target_controls_layout = QFormLayout(target_controls_group); target_controls_layout.setContentsMargins(0, 5, 0, 0)
        objective_combo = QComboBox(); objective_combo.addItems(self.OBJECTIVES_DISPLAY_MAP.values())
        self.operational_widgets['target_objective_combo'] = objective_combo
        value_input = QDoubleSpinBox(); value_input.setRange(-1e9, 1e9); value_input.setDecimals(4); value_input.setSingleStep(0.01)
        self.operational_widgets['target_value_input'] = value_input
        target_controls_layout.addRow(self.tr("Target Objective:"), objective_combo); target_controls_layout.addRow(self.tr("Target Value:"), value_input)
        target_group_layout.addWidget(target_controls_group); layout.addWidget(target_group)
        
        enable_checkbox.toggled.connect(target_controls_group.setVisible); enable_checkbox.toggled.connect(self._mark_as_dirty)
        objective_combo.currentTextChanged.connect(self._mark_as_dirty); value_input.valueChanged.connect(self._mark_as_dirty)

        is_target_enabled = op_instance.target_objective_name is not None and op_instance.target_objective_value is not None
        enable_checkbox.setChecked(is_target_enabled); target_controls_group.setVisible(is_target_enabled)
        if is_target_enabled:
            display_name = self.OBJECTIVES_DISPLAY_MAP.get(op_instance.target_objective_name)
            if display_name: objective_combo.setCurrentText(display_name)
            value_input.setValue(op_instance.target_objective_value or 0.0)

        layout.addStretch()
        
        op_instance_name = op_instance.__class__.__name__
        recovery_model_widget = self.input_groups.get(f"{op_instance_name}.recovery_model_selection")
        if recovery_model_widget:
            recovery_model_widget.finalValueChanged.connect(self._update_recovery_model_widgets)
        self._update_recovery_model_widgets()


    def _update_recovery_model_widgets(self):
        """Dynamically create and show widgets for the selected recovery model."""
        if not hasattr(self, 'recovery_model_container'): return

        while self.recovery_model_layout.count():
            item = self.recovery_model_layout.takeAt(0)
            widget = item.widget()
            if widget:
                keys_to_remove = [k for k, v in self.input_groups.items() if v is widget]
                for k in keys_to_remove: del self.input_groups[k]
                widget.deleteLater()
        
        op_instance_name = OperationalParameters.__name__
        model_selector_widget = self.input_groups.get(f"{op_instance_name}.recovery_model_selection")
        if not model_selector_widget: return

        selected_model = model_selector_widget.get_value()
        model_params = self.recovery_model_params.get(selected_model, {})

        for param_name, param_value in model_params.items():
            key = f"{selected_model}.{param_name}"
            kwargs = {'input_type': 'doublespinbox', 'decimals': 4} if isinstance(param_value, float) else {'input_type': 'spinbox'}
            kwargs.update({
                "param_name": key,
                "label_text": param_name.replace("_", " ").title(),
                "default_value": param_value,
                "help_text": self.tr("Parameter for {} model.").format(selected_model)
            })
            input_group = ParameterInputGroup(**kwargs)
            input_group.finalValueChanged.connect(self._mark_as_dirty)
            self.recovery_model_layout.addWidget(input_group)
            self.input_groups[key] = input_group

    def _get_input_type_and_options(self, field: Field) -> dict:
        """Determines the widget type and options from a dataclass field."""
        kwargs = {}
        base_type = field.type; origin = get_origin(base_type)
        if origin in (Union, UnionType): base_type = next((t for t in get_args(base_type) if t is not type(None)), str)

        if field.name in self.SPECIAL_DROPDOWNS: kwargs['input_type'] = 'combobox'; kwargs['items'] = self.SPECIAL_DROPDOWNS[field.name]
        elif base_type is bool: kwargs['input_type'] = 'checkbox'
        elif base_type is int: kwargs['input_type'] = 'spinbox'
        elif base_type is float: kwargs['input_type'] = 'doublespinbox'
        elif base_type is tuple: kwargs['input_type'] = 'lineedit'
        else: kwargs['input_type'] = 'lineedit'
            
        if 'min' in field.metadata: kwargs['min_val'] = field.metadata['min']
        if 'max' in field.metadata: kwargs['max_val'] = field.metadata['max']
        if 'step' in field.metadata: kwargs['step'] = field.metadata['step']
        if 'decimals' in field.metadata: kwargs['decimals'] = field.metadata['decimals']
        return kwargs

    def _create_input_group_for_field(self, instance: Any, field: Field, layout: QFormLayout, scope_name: str):
        """Creates and configures a ParameterInputGroup for a given dataclass field."""
        key = f"{scope_name}.{field.name}"; kwargs = self._get_input_type_and_options(field)
        
        current_value = getattr(instance, field.name); default_value = getattr(self.default_instances.get(type(instance).__name__), field.name, None)
        is_modified = current_value != default_value
        if isinstance(current_value, float) and isinstance(default_value, (float, int)): is_modified = not np.isclose(current_value, default_value)
        
        if field.name == 'oil_profile_type': display_value = self.OIL_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif field.name == 'injection_profile_type': display_value = self.INJECTION_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif isinstance(current_value, list): display_value = ', '.join(map(str, current_value))
        elif isinstance(current_value, tuple): display_value = str(current_value)
        else: display_value = current_value

        help_text = field.metadata.get("help", self.tr("No description available."))
        kwargs.update({"param_name": key, "label_text": field.name.replace("_", " ").title(), "default_value": display_value if current_value is not None else "", "help_text": help_text})
        
        input_group = ParameterInputGroup(**kwargs)
        input_group.finalValueChanged.connect(self._mark_as_dirty)
        input_group.help_requested.connect(self.help_requested)
        
        if get_origin(field.type) in (Union, UnionType) and type(None) in get_args(field.type):
            input_group.set_checkable(True)
            input_group.set_checked(current_value is not None)

        layout.addRow(input_group)
        self.input_groups[key] = input_group
        

    def _confirm_reset_all(self):
        """Confirms and resets all configurations to their default state."""
        reply = QMessageBox.question(self, self.tr("Confirm Reset"), 
                                     self.tr("Reset all parameters to application defaults? This will discard any applied or pending changes."), 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("ConfigWidget resetting all parameters to application defaults.")
            self.update_configurations(deepcopy(self.default_instances))
            self.recovery_model_params.clear()
            self.config_source_label.setText(self.tr("Application Defaults"))
            full_config_data = self.get_current_config_data_instances()
            self.configurations_updated.emit(full_config_data)

    def _load_from_file(self):
        """Loads a configuration from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(self, self.tr("Load Configuration File"), "", self.tr("JSON Files (*.json)"))
        if not filepath: return
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            loaded_instances = {}
            for dc_type in CONFIGURABLE_DATACLASSES.values():
                instance_name = dc_type.__name__
                if instance_name in data and isinstance(data[instance_name], dict):
                    valid_data = {k: v for k, v in data[instance_name].items() if k in {f.name for f in fields(dc_type)}}
                    loaded_instances[instance_name] = dc_type(**valid_data)
                else: loaded_instances[instance_name] = deepcopy(self.default_instances[instance_name])
            
            self.recovery_model_params = data.get("RecoveryModelKwargsDefaults", {})

            if loaded_instances:
                self.update_configurations(loaded_instances)
                self.config_source_label.setText(self.tr("Loaded from File: {}").format(Path(filepath).name))

                full_config_data = self.get_current_config_data_instances()
                self.configurations_updated.emit(full_config_data)

        except Exception as e: QMessageBox.critical(self, self.tr("Load Error"), self.tr("Could not load file:\n\n{}").format(e))

    def _save_to_file(self):
        """Saves the current configuration to a JSON file."""
        if self._is_dirty:
            QMessageBox.warning(self, self.tr("Unapplied Changes"), self.tr("You have unapplied changes. Please apply or discard them before saving."))
            return
        filepath, _ = QFileDialog.getSaveFileName(self, self.tr("Save Configuration As"), "config.json", self.tr("JSON Files (*.json)"))
        if not filepath: return
        try:
            data_to_save = {name: asdict(inst) for name, inst in self.config_instances.items()}
            data_to_save["RecoveryModelKwargsDefaults"] = self.recovery_model_params
            
            with open(filepath, 'w') as f: json.dump(data_to_save, f, indent=2, sort_keys=True)
            self.save_configuration_to_file_requested.emit(data_to_save)
            QMessageBox.information(self, self.tr("Save Successful"), self.tr("Configuration saved to:\n{}").format(filepath))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Save Error"), self.tr("Could not save file:\n\n{}").format(e))
            
    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        """Coerces a value to the specified type, handling optionality, lists, and tuples."""
        if isinstance(value, str): value = value.strip()
        origin = get_origin(type_hint); args = get_args(type_hint)
        is_optional = origin in (Union, UnionType) and type(None) in args
        base_type = next((t for t in args if t is not type(None)), type_hint) if is_optional else type_hint
        if value is None or (isinstance(value, str) and not value):
            if is_optional: return None
            else: raise ValueError(self.tr("This field cannot be empty."))
        
        base_origin = get_origin(base_type)
        if base_origin is list or base_origin is List:
            if isinstance(value, list): return value
            if not str(value): return []
            try: return [float(x.strip()) for x in str(value).split(',') if x.strip()]
            except ValueError: raise ValueError(self.tr("Must be a comma-separated list of numbers."))
        if base_origin is tuple or base_type is tuple:
            if isinstance(value, tuple): return value
            try: return eval(str(value)) # Safely evaluate string " (0.5, 1.0) " into a tuple
            except Exception: raise ValueError(self.tr("Must be a valid tuple, e.g., (0.5, 1.0)"))
        if base_type is bool: return bool(value)
        if base_type is float: return float(value)
        if base_type is int: return int(value)
        return value

    def _update_current_scheme_display(self):
        """Update the current injection scheme display label (for OperationalParameters)."""
        eor_instance = self.config_instances.get('EORParameters')
        if eor_instance and hasattr(self, 'current_scheme_display'):
            scheme = eor_instance.injection_scheme
            # Convert scheme name to display format
            display_name = scheme.replace('_', ' ').title()
            if scheme == 'wag':
                display_name = 'WAG'
            elif scheme == 'huff_n_puff':
                display_name = 'Huff-n-Puff'
            elif scheme == 'swag':
                display_name = 'SWAG'
            self.current_scheme_display.setText(display_name)

    def _update_current_eor_scheme_display(self):
        """Update the current injection scheme display label (for EORParameters page)."""
        eor_instance = self.config_instances.get('EORParameters')
        if eor_instance and hasattr(self, 'current_eor_scheme_display'):
            scheme = eor_instance.injection_scheme
            # Convert scheme name to display format
            display_name = scheme.replace('_', ' ').title()
            if scheme == 'wag':
                display_name = 'WAG'
            elif scheme == 'huff_n_puff':
                display_name = 'Huff-n-Puff'
            elif scheme == 'swag':
                display_name = 'SWAG'
            elif scheme == 'tapered':
                display_name = 'Tapered Injection'
            elif scheme == 'pulsed':
                display_name = 'Pulsed Injection'
            elif scheme == 'continuous':
                display_name = 'Continuous Injection'
            self.current_eor_scheme_display.setText(display_name)

    def _open_injection_scheme_dialog(self):
        """Open the injection scheme configuration dialog."""
        eor_instance = self.config_instances.get('EORParameters')
        if not eor_instance:
            logger.error("EORParameters instance not found")
            return

        from ui.dialogs.injection_scheme_dialog import InjectionSchemeDialog
        dialog = InjectionSchemeDialog(eor_instance, self)
        dialog.scheme_updated.connect(self._on_eor_scheme_updated)
        dialog.exec()

    def _open_eor_injection_scheme_dialog(self):
        """Open the EOR injection scheme configuration dialog."""
        eor_instance = self.config_instances.get('EORParameters')
        if not eor_instance:
            logger.error("EORParameters instance not found")
            return

        from ui.dialogs.injection_scheme_dialog import InjectionSchemeDialog
        dialog = InjectionSchemeDialog(eor_instance, self)
        dialog.scheme_updated.connect(self._on_eor_scheme_updated)
        dialog.exec()

    def _on_scheme_updated(self, updated_params: dict):
        """Handle injection scheme updates from the dialog."""
        try:
            # Update EOR parameters with the new scheme configuration
            eor_instance = self.config_instances.get('EORParameters')
            if eor_instance:
                # Create new EOR instance with updated parameters
                updated_eor = EORParameters(**updated_params)
                self.config_instances['EORParameters'] = updated_eor

                # Mark as dirty and update display
                self._set_dirty(True)
                self._update_current_scheme_display()
                self._update_current_eor_scheme_display()

                # Update any visible injection parameter widgets
                self._update_injection_parameter_widgets(updated_eor)

                logger.info(f"Injection scheme updated to: {updated_eor.injection_scheme}")

        except Exception as e:
            logger.error(f"Error updating injection scheme: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to update injection scheme: {e}")

    def _on_eor_scheme_updated(self, updated_params: dict):
        """Handle EOR injection scheme updates from the dialog."""
        try:
            # Update EOR parameters with the new scheme configuration
            eor_instance = self.config_instances.get('EORParameters')
            if eor_instance:
                # Create new EOR instance with updated parameters
                updated_eor = EORParameters(**updated_params)
                self.config_instances['EORParameters'] = updated_eor

                # Mark as dirty and update display
                self._set_dirty(True)
                self._update_current_scheme_display()
                self._update_current_eor_scheme_display()

                # Update any visible injection parameter widgets
                self._update_injection_parameter_widgets(updated_eor)

                logger.info(f"EOR injection scheme updated to: {updated_eor.injection_scheme}")

        except Exception as e:
            logger.error(f"Error updating EOR injection scheme: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to update EOR injection scheme: {e}")

    def _update_injection_parameter_widgets(self, eor_instance: EORParameters):
        """Update injection parameter widgets when scheme changes."""
        # Update basic injection parameters if they exist in the UI
        injection_rate_widget = self.input_groups.get("EORParameters.injection_rate")
        if injection_rate_widget:
            injection_rate_widget.set_value(eor_instance.injection_rate)
            
        target_pressure_widget = self.input_groups.get("EORParameters.target_pressure_psi")
        if target_pressure_widget:
            target_pressure_widget.set_value(eor_instance.target_pressure_psi)

    def _update_profile_param_visibility(self):
        """Shows/hides profile parameter inputs based on the selected profile type."""
        profile_params_instance_name = ProfileParameters.__name__
        oil_profile_type_widget = self.input_groups.get(f"{profile_params_instance_name}.oil_profile_type")
        if not oil_profile_type_widget: return

        display_name = oil_profile_type_widget.get_value()
        profile_type = self.REVERSE_OIL_PROFILE_MAP.get(display_name, display_name)

        visibility_map = {
            'oil_annual_fraction_of_total': (profile_type == 'custom_fractions'),
            'plateau_duration_fraction_of_life': 'plateau' in profile_type,
            'initial_decline_rate_annual_fraction': 'decline' in profile_type,
            'hyperbolic_b_factor': 'hyperbolic' in profile_type,
            'min_economic_rate_fraction_of_peak': 'decline' in profile_type,
            'co2_breakthrough_year_fraction': True, 'co2_production_ratio_after_breakthrough': True,
            'co2_recycling_efficiency_fraction': True, 'warn_if_defaults_used': True
        }
        for param_name, is_visible in visibility_map.items():
            widget_key = f"{profile_params_instance_name}.{param_name}"
            if widget_key in self.input_groups:
                self.input_groups[widget_key].setVisible(is_visible)

    def _on_engine_type_combo_changed(self, engine_display_name: str):
        """Handle engine selection combo box change."""
        # Get actual engine type from combo box data
        engine_type = self.engine_type_combo.currentData()
        if not engine_type:
            return

        logger.info(f"ConfigWidget: Engine type selection changed to '{engine_type}'")

        # Mark as dirty
        self._set_dirty(True)

        # Emit signal to notify other widgets
        self.engine_selection_changed.emit(engine_type)

        # Update the config_instances to reflect the change (will be applied when user clicks Apply)
        advanced_params_instance = self.config_instances.get('AdvancedEngineParams')
        if advanced_params_instance:
            # Update the engine_type field
            advanced_params_instance.engine_type = engine_type
            # Also update use_simple_physics for backward compatibility
            advanced_params_instance.use_simple_physics = (engine_type == 'simple')
            logger.info(f"ConfigWidget: Updated AdvancedEngineParams.engine_type to '{engine_type}'")
        
        visibility_map = {
            'oil_annual_fraction_of_total': (profile_type == 'custom_fractions'),
            'plateau_duration_fraction_of_life': 'plateau' in profile_type,
            'initial_decline_rate_annual_fraction': 'decline' in profile_type,
            'hyperbolic_b_factor': 'hyperbolic' in profile_type,
            'min_economic_rate_fraction_of_peak': 'decline' in profile_type,
            'co2_breakthrough_year_fraction': True, 'co2_production_ratio_after_breakthrough': True,
            'co2_recycling_efficiency_fraction': True, 'warn_if_defaults_used': True
        }
        for param_name, is_visible in visibility_map.items():
            widget_key = f"{profile_params_instance_name}.{param_name}"
            if widget_key in self.input_groups:
                self.input_groups[widget_key].setVisible(is_visible)