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
    QPushButton, QMessageBox, QFileDialog, QHBoxLayout, QFrame, QTabWidget,
    QGroupBox, QCheckBox, QComboBox, QDoubleSpinBox
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, Qt

from .widgets.parameter_input_group import ParameterInputGroup
try:
    from co2eor_optimizer.core.data_models import (
        EconomicParameters, EORParameters, OperationalParameters, ProfileParameters, 
        GeneticAlgorithmParams, BayesianOptimizationParams
    )
    # Added BayesianOptimizationParams
    CONFIGURABLE_DATACLASSES: Dict[str, Type] = {
        "Economic": EconomicParameters, "EOR": EORParameters,
        "Operational": OperationalParameters, "Profile": ProfileParameters,
        "Genetic Algorithm": GeneticAlgorithmParams,
        "Bayesian Optimization": BayesianOptimizationParams
    }
except ImportError:
    logging.critical("ConfigWidget: Core configuration dataclasses not found.")
    CONFIGURABLE_DATACLASSES = {}
    OPTIMIZATION_OBJECTIVES = {}

logger = logging.getLogger(__name__)

class ConfigWidget(QWidget):
    """A visually organized widget for editing application configurations defined by dataclasses."""
    configuration_changed = pyqtSignal(str, str, object)
    # ENHANCED: A more robust signal that emits the full configuration dictionary at once.
    configurations_updated = pyqtSignal(dict)
    save_configuration_to_file_requested = pyqtSignal(dict)

    OIL_PROFILE_DISPLAY_MAP = {
        "linear_distribution": "Linear Distribution",
        "plateau_linear_decline": "Plateau with Linear Decline",
        "plateau_exponential_decline": "Plateau with Exponential Decline",
        "plateau_hyperbolic_decline": "Plateau with Hyperbolic Decline",
        "custom_fractions": "Custom Annual Fractions"
    }
    INJECTION_PROFILE_DISPLAY_MAP = {"constant_during_phase": "Constant During Each Phase"}
    
    # Invert the objectives map for display -> internal key mapping
    REVERSE_OBJECTIVES_MAP = {v: k for k, v in {
        "npv": "Net Present Value (NPV)", 
        "recovery_factor": "Recovery Factor (RF)",
        "co2_utilization": "CO2 Utilization"}.items()}


    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.REVERSE_OIL_PROFILE_MAP = {v: k for k, v in self.OIL_PROFILE_DISPLAY_MAP.items()}
        self.REVERSE_INJECTION_PROFILE_MAP = {v: k for k, v in self.INJECTION_PROFILE_DISPLAY_MAP.items()}
        self.SPECIAL_DROPDOWNS = {
            "injection_scheme": ['continuous', 'wag'],
            "oil_profile_type": list(self.OIL_PROFILE_DISPLAY_MAP.values()),
            "injection_profile_type": list(self.INJECTION_PROFILE_DISPLAY_MAP.values())
        }
        
        # Store recovery model parameters separately
        self.recovery_model_params: Dict[str, Dict[str, Any]] = {}

        self.default_instances: Dict[str, Any] = {
            dc_type.__name__: dc_type() for dc_type in CONFIGURABLE_DATACLASSES.values()
        } if CONFIGURABLE_DATACLASSES else {}
        
        self.config_instances: Dict[str, Any] = deepcopy(self.default_instances)
        self.input_groups: Dict[str, ParameterInputGroup] = {}
        # Widgets for the custom Operational tab
        self.operational_widgets: Dict[str, QWidget] = {}
        self._is_dirty = False

        self._setup_ui()
        self.setStyleSheet(self._get_stylesheet())
        self._set_dirty(False) # Initially hide apply/discard buttons

    def _get_stylesheet(self) -> str:
        return """
            QTabWidget::pane { border-top: 1px solid #C2C7CB; background: white; }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E1E1E1, stop:1 #D3D3D3);
                border: 1px solid #C4C4C3; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                padding: 6px 12px; font-weight: bold; color: #333;
            }
            QTabBar::tab:selected { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f3f9ff, stop:1 #c7d8f3); }
            QScrollArea { border: none; background-color: transparent; }
            ParameterInputGroup[isModified="true"] {
                background-color: #e8f4e8; border: 1px solid #a3d8a3;
                border-radius: 6px; margin: 2px 0;
            }
            ParameterInputGroup[isModified="true"] > QLabel { font-weight: bold; }
            #ConfigSourceLabel { font-style: italic; color: #555; padding: 4px; background-color: #f0f0f0; border-radius: 4px; }
            #ApplyDiscardFrame { background-color: #fffac1; border-radius: 5px; border: 1px solid #f0e68c; }
            QGroupBox { font-weight: bold; }
        """

    def _setup_ui(self):
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(15, 10, 15, 10); main_layout.setSpacing(10)

        top_button_layout = QHBoxLayout()
        load_btn = QPushButton(QIcon.fromTheme("document-open"), " Load from File..."); save_btn = QPushButton(QIcon.fromTheme("document-save"), " Save to File...")
        reset_btn = QPushButton(QIcon.fromTheme("view-refresh"), " Reset All to Defaults")
        top_button_layout.addWidget(load_btn); top_button_layout.addWidget(save_btn); top_button_layout.addStretch(); top_button_layout.addWidget(reset_btn)

        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("<b>Current Configuration Source:</b>"))
        self.config_source_label = QLabel(); self.config_source_label.setObjectName("ConfigSourceLabel")
        status_layout.addWidget(self.config_source_label); status_layout.addStretch()

        self.apply_discard_frame = QFrame(); self.apply_discard_frame.setObjectName("ApplyDiscardFrame")
        apply_layout = QHBoxLayout(self.apply_discard_frame)
        apply_btn = QPushButton(QIcon.fromTheme("dialog-ok-apply"), " Apply Changes"); discard_btn = QPushButton(QIcon.fromTheme("dialog-cancel"), " Discard Changes")
        apply_layout.addWidget(QLabel("You have unsaved changes:"))
        apply_layout.addStretch(); apply_layout.addWidget(discard_btn); apply_layout.addWidget(apply_btn)
        
        main_layout.addLayout(top_button_layout); main_layout.addLayout(status_layout)
        main_layout.addWidget(self.apply_discard_frame)

        self.tabs = QTabWidget(self)
        main_layout.addWidget(self.tabs, 1)

        load_btn.clicked.connect(self._load_from_file); save_btn.clicked.connect(self._save_to_file)
        reset_btn.clicked.connect(self._confirm_reset_all); apply_btn.clicked.connect(self._apply_changes); discard_btn.clicked.connect(self._discard_changes)
        
        self.config_source_label.setText("Application Defaults")

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
        self.config_source_label.setText("Loaded from Project")

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
        
        pending_data = {dc_name: asdict(instance) for dc_name, instance in self.config_instances.items()}
        all_valid = True
        
        # First, handle the special case of the Operational tab
        op_instance_name = OperationalParameters.__name__
        if self.operational_widgets:
            try:
                lifetime_widget = self.input_groups[f"{op_instance_name}.project_lifetime_years"]
                pending_data[op_instance_name]['project_lifetime_years'] = int(lifetime_widget.get_value())
                
                sharpness_widget = self.input_groups[f"{op_instance_name}.target_seeking_sharpness"]
                pending_data[op_instance_name]['target_seeking_sharpness'] = float(sharpness_widget.get_value())

                enable_checkbox = self.operational_widgets['enable_target_seeking_checkbox']
                if enable_checkbox.isChecked():
                    objective_combo = self.operational_widgets['target_objective_combo']
                    value_input = self.operational_widgets['target_value_input']
                    
                    display_name = objective_combo.currentText()
                    internal_name = self.REVERSE_OBJECTIVES_MAP.get(display_name, 'npv')
                    
                    pending_data[op_instance_name]['target_objective_name'] = internal_name
                    pending_data[op_instance_name]['target_objective_value'] = float(value_input.value())
                else:
                    pending_data[op_instance_name]['target_objective_name'] = None
                    pending_data[op_instance_name]['target_objective_value'] = None
            except (ValueError, TypeError) as e:
                QMessageBox.critical(self, "Validation Error", f"Invalid input for Operational Parameters: {e}")
                all_valid = False

        # Handle all other standard ParameterInputGroup widgets
        for key, widget in self.input_groups.items():
            if not widget.isVisible() or '.' not in key: continue
            
            scope, param_name = key.split('.', 1)
            raw_value = widget.get_value()

            if scope in pending_data:
                if scope == op_instance_name: continue # Already handled

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
                    widget.show_error(str(e))
                    all_valid = False
            
            elif scope in self.recovery_model_params:
                try:
                    coerced_value = float(raw_value) if '.' in str(raw_value) else int(raw_value)
                    self.recovery_model_params[scope][param_name] = coerced_value
                    widget.clear_error()
                except (ValueError, TypeError) as e:
                    widget.show_error(str(e))
                    all_valid = False

        if not all_valid:
            QMessageBox.critical(self, "Validation Error", "One or more fields have invalid values. Please correct the highlighted fields.")
            return

        try:
            new_instances = {}
            for name, data in pending_data.items():
                dc_type = self.default_instances[name].__class__
                new_instances[name] = dc_type(**data)
            self.config_instances = new_instances
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Validation Error", f"Could not apply changes due to inconsistent data:\n\n{e}")
            return
            
        logger.info("Configuration changes applied successfully.")
        self.config_source_label.setText("Custom Applied")

        full_config_data = self.get_current_config_data_instances()
        self.configurations_updated.emit(full_config_data)
        
        self.update_configurations(self.config_instances)

    def update_configurations(self, config_data: Dict[str, Any]):
        """Updates the internal state and repopulates all UI forms from the given data."""
        self.config_instances = deepcopy(config_data)
        self._populate_all_forms()
        self._set_dirty(False)

    def _populate_all_forms(self):
        """Clears and rebuilds all configuration tabs based on current instances."""
        while self.tabs.count() > 0: self.tabs.removeTab(0)
        self.input_groups.clear()
        self.operational_widgets.clear()
        if not CONFIGURABLE_DATACLASSES:
            self.tabs.addTab(QLabel("Configuration models not loaded."), "Error")
            return
            
        for name, dc_type in CONFIGURABLE_DATACLASSES.items():
            instance = self.config_instances.get(dc_type.__name__)
            if instance:
                if name == "Operational":
                    self._create_operational_tab(instance)
                else:
                    self._create_standard_tab(name, instance)
        
        self._create_recovery_models_tab()
        self._update_profile_param_visibility()

    def _create_standard_tab(self, tab_name: str, instance: Any):
        """Creates a tab with a standard form layout for a dataclass."""
        page_widget = QWidget()
        form_layout = QFormLayout(page_widget)
        form_layout.setSpacing(10); form_layout.setContentsMargins(10, 15, 10, 15)
        for field in fields(instance): 
            self._create_input_group_for_field(instance, field, form_layout)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(page_widget)
        self.tabs.addTab(scroll_area, tab_name)

    def _create_operational_tab(self, instance: OperationalParameters):
        """Creates the custom, more intuitive tab for OperationalParameters."""
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        layout.setSpacing(15); layout.setContentsMargins(10, 15, 10, 15)
        
        # --- General Settings ---
        general_group = QGroupBox("General")
        general_layout = QFormLayout(general_group)
        for field in fields(instance):
            if field.name not in ["target_objective_name", "target_objective_value", "target_recovery_factor"]:
                self._create_input_group_for_field(instance, field, general_layout)
        layout.addWidget(general_group)

        # --- Target Seeking Settings ---
        target_group = QGroupBox("Optimizer Target Seeking")
        target_group_layout = QVBoxLayout(target_group)
        
        enable_checkbox = QCheckBox("Enable Target Seeking")
        enable_checkbox.setToolTip("Check this to make the optimizer try to achieve a specific target value for an objective.")
        target_group_layout.addWidget(enable_checkbox)

        self.operational_widgets['enable_target_seeking_checkbox'] = enable_checkbox

        target_controls_group = QGroupBox()
        target_controls_group.setFlat(True)
        target_controls_layout = QFormLayout(target_controls_group)
        target_controls_layout.setContentsMargins(0, 5, 0, 0)
        
        objective_combo = QComboBox()
        objective_combo.addItems(self.REVERSE_OBJECTIVES_MAP.keys())
        self.operational_widgets['target_objective_combo'] = objective_combo

        value_input = QDoubleSpinBox()
        value_input.setRange(-1e9, 1e9)
        value_input.setDecimals(4)
        value_input.setSingleStep(0.01)
        self.operational_widgets['target_value_input'] = value_input
        
        target_controls_layout.addRow("Target Objective:", objective_combo)
        target_controls_layout.addRow("Target Value:", value_input)
        
        target_group_layout.addWidget(target_controls_group)
        layout.addWidget(target_group)
        
        # --- Connect Signals ---
        enable_checkbox.toggled.connect(target_controls_group.setVisible)
        enable_checkbox.toggled.connect(self._mark_as_dirty)
        objective_combo.currentTextChanged.connect(self._mark_as_dirty)
        value_input.valueChanged.connect(self._mark_as_dirty)

        # --- Set Initial State ---
        is_target_enabled = instance.target_objective_name is not None and instance.target_objective_value is not None
        enable_checkbox.setChecked(is_target_enabled)
        target_controls_group.setVisible(is_target_enabled)
        
        if is_target_enabled:
            display_name = {v:k for k, v in self.REVERSE_OBJECTIVES_MAP.items()}.get(instance.target_objective_name)
            if display_name:
                objective_combo.setCurrentText(display_name)
            value_input.setValue(instance.target_objective_value or 0.0)

        layout.addStretch()
        
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(page_widget)
        self.tabs.addTab(scroll_area, "Operational")

    def _create_recovery_models_tab(self):
        """Creates the tab for editing recovery model parameters."""
        page_widget = QWidget(); layout = QVBoxLayout(page_widget)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(15)

        model_defs = {
            "Koval": {"v_dp_coefficient": 0.5, "mobility_ratio": 2.0},
            "Miscible": {"kv_factor": 0.5, "gravity_factor": 0.1},
            "Immiscible": {"sor": 0.25, "krw_max": 0.4}
        }
        
        if not self.recovery_model_params:
            self.recovery_model_params = deepcopy(model_defs)

        for model_name, params in self.recovery_model_params.items():
            group_box = QGroupBox(f"{model_name} Model Parameters")
            form_layout = QFormLayout(group_box)
            form_layout.setSpacing(8); form_layout.setContentsMargins(10, 10, 10, 10)

            for param_name, default_val in params.items():
                key = f"{model_name}.{param_name}"
                input_group = ParameterInputGroup(param_name=key, label_text=param_name.replace("_", " ").title(), default_value=default_val, input_type='doublespinbox')
                input_group.finalValueChanged.connect(self._mark_as_dirty)
                form_layout.addRow(input_group)
                self.input_groups[key] = input_group
            
            layout.addWidget(group_box)

        layout.addStretch(1); scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(page_widget)
        self.tabs.addTab(scroll_area, "Recovery Models")

    def _get_input_type_and_options(self, field: Field) -> dict:
        """Determines the widget type and options from a dataclass field."""
        kwargs = {}
        base_type = field.type; origin = get_origin(base_type)
        if origin in (Union, UnionType): base_type = next((t for t in get_args(base_type) if t is not type(None)), str)

        if field.name in self.SPECIAL_DROPDOWNS: kwargs['input_type'] = 'combobox'; kwargs['items'] = self.SPECIAL_DROPDOWNS[field.name]
        elif base_type is bool: kwargs['input_type'] = 'checkbox'
        elif base_type is int: kwargs['input_type'] = 'spinbox'
        elif base_type is float: kwargs['input_type'] = 'doublespinbox'
        else: kwargs['input_type'] = 'lineedit'
            
        if 'min' in field.metadata: kwargs['min_val'] = field.metadata['min']
        if 'max' in field.metadata: kwargs['max_val'] = field.metadata['max']
        if 'step' in field.metadata: kwargs['step'] = field.metadata['step']
        if 'decimals' in field.metadata: kwargs['decimals'] = field.metadata['decimals']
        return kwargs

    def _create_input_group_for_field(self, instance: Any, field: Field, layout: QFormLayout):
        """Creates and configures a ParameterInputGroup for a given dataclass field."""
        instance_name = type(instance).__name__; key = f"{instance_name}.{field.name}"; kwargs = self._get_input_type_and_options(field)
        
        current_value = getattr(instance, field.name); default_value = getattr(self.default_instances.get(instance_name), field.name, None)
        is_modified = current_value != default_value
        if isinstance(current_value, float) and isinstance(default_value, (float, int)): is_modified = not np.isclose(current_value, default_value)
        
        if field.name == 'oil_profile_type': display_value = self.OIL_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif field.name == 'injection_profile_type': display_value = self.INJECTION_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif isinstance(current_value, list): display_value = ', '.join(map(str, current_value))
        else: display_value = current_value

        kwargs.update({"param_name": key, "label_text": field.name.replace("_", " ").title(), "default_value": display_value if current_value is not None else "", "help_text": field.metadata.get("help", "No description available.")})
        
        input_group = ParameterInputGroup(**kwargs)
        input_group.setProperty("isModified", is_modified)
        input_group.finalValueChanged.connect(self._mark_as_dirty)
        
        if get_origin(field.type) in (Union, UnionType): input_group.set_checkable(True); input_group.set_checked(current_value is not None)

        layout.addRow(input_group)
        self.input_groups[key] = input_group

    def _confirm_reset_all(self):
        """Confirms and resets all configurations to their default state."""
        reply = QMessageBox.question(self, "Confirm Reset", "Reset all parameters to application defaults? This will discard any applied or pending changes.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("ConfigWidget resetting all parameters to application defaults.")
            self.update_configurations(deepcopy(self.default_instances))
            self.recovery_model_params.clear()
            self.config_source_label.setText("Application Defaults")
            full_config_data = self.get_current_config_data_instances()
            self.configurations_updated.emit(full_config_data)

    def _load_from_file(self):
        """Loads a configuration from a JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration File", "", "JSON Files (*.json)")
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
                self.config_source_label.setText(f"Loaded from File: {Path(filepath).name}")

                full_config_data = self.get_current_config_data_instances()
                self.configurations_updated.emit(full_config_data)

        except Exception as e: QMessageBox.critical(self, "Load Error", f"Could not load file:\n\n{e}")

    def _save_to_file(self):
        """Saves the current configuration to a JSON file."""
        if self._is_dirty:
            QMessageBox.warning(self, "Unapplied Changes", "You have unapplied changes. Please apply or discard them before saving.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", "config.json", "JSON Files (*.json)")
        if not filepath: return
        try:
            data_to_save = {name: asdict(inst) for name, inst in self.config_instances.items()}
            data_to_save["RecoveryModelKwargsDefaults"] = self.recovery_model_params
            
            with open(filepath, 'w') as f: json.dump(data_to_save, f, indent=2, sort_keys=True)
            self.save_configuration_to_file_requested.emit(data_to_save)
            QMessageBox.information(self, "Save Successful", f"Configuration saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n\n{e}")
            
    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        """Coerces a value to the specified type, handling optionality and lists."""
        if isinstance(value, str): value = value.strip()
        origin = get_origin(type_hint); args = get_args(type_hint)
        is_optional = origin in (Union, UnionType) and type(None) in args
        base_type = next((t for t in args if t is not type(None)), type_hint) if is_optional else type_hint
        if value is None or (isinstance(value, str) and not value):
            if is_optional: return None
            else: raise ValueError("This field cannot be empty.")
        base_origin = get_origin(base_type)
        if base_origin is list or base_origin is List:
            if isinstance(value, list): return value
            if not str(value): return []
            try: return [float(x.strip()) for x in str(value).split(',') if x.strip()]
            except ValueError: raise ValueError("Must be a comma-separated list of numbers.")
        if base_type is bool: return bool(value)
        if base_type is float: return float(value)
        if base_type is int: return int(value)
        return value

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
            if widget_key in self.input_groups: self.input_groups[widget_key].setVisible(is_visible)