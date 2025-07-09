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
    QPushButton, QMessageBox, QFileDialog, QHBoxLayout, QFrame, QTabWidget
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import pyqtSignal, Qt

from .widgets.parameter_input_group import ParameterInputGroup
try:
    from co2eor_optimizer.core.data_models import (
        EconomicParameters, EORParameters, OperationalParameters, ProfileParameters, GeneticAlgorithmParams
    )
    CONFIGURABLE_DATACLASSES: Dict[str, Type] = {
        "Economic": EconomicParameters, "EOR": EORParameters,
        "Operational": OperationalParameters, "Profile": ProfileParameters,
        "Genetic Algorithm": GeneticAlgorithmParams,
    }
except ImportError:
    logging.critical("ConfigWidget: Core configuration dataclasses not found.")
    CONFIGURABLE_DATACLASSES = {}

logger = logging.getLogger(__name__)

class ConfigWidget(QWidget):
    """A visually organized widget for editing application configurations defined by dataclasses."""
    configuration_changed = pyqtSignal(str, str, object)
    save_configuration_to_file_requested = pyqtSignal(dict)

    OIL_PROFILE_DISPLAY_MAP = {
        "linear_distribution": "Linear Distribution",
        "plateau_linear_decline": "Plateau with Linear Decline",
        "plateau_exponential_decline": "Plateau with Exponential Decline",
        "plateau_hyperbolic_decline": "Plateau with Hyperbolic Decline",
        "custom_fractions": "Custom Annual Fractions"
    }
    INJECTION_PROFILE_DISPLAY_MAP = {"constant_during_phase": "Constant During Each Phase"}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.REVERSE_OIL_PROFILE_MAP = {v: k for k, v in self.OIL_PROFILE_DISPLAY_MAP.items()}
        self.REVERSE_INJECTION_PROFILE_MAP = {v: k for k, v in self.INJECTION_PROFILE_DISPLAY_MAP.items()}
        self.SPECIAL_DROPDOWNS = {
            "injection_scheme": ['continuous', 'wag'],
            "oil_profile_type": list(self.OIL_PROFILE_DISPLAY_MAP.values()),
            "injection_profile_type": list(self.INJECTION_PROFILE_DISPLAY_MAP.values())
        }

        self.default_instances: Dict[str, Any] = {
            dc_type.__name__: dc_type() for dc_type in CONFIGURABLE_DATACLASSES.values()
        } if CONFIGURABLE_DATACLASSES else {}
        
        self.config_instances: Dict[str, Any] = deepcopy(self.default_instances)
        self.input_groups: Dict[str, ParameterInputGroup] = {}
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

    # --- Public API Methods ---
    
    def get_current_config_data_instances(self) -> Dict[str, Any]:
        """Returns a deep copy of the currently applied configuration instances."""
        return deepcopy(self.config_instances)

    def update_configs_from_project(self, config_instances: Dict[str, Any]):
        """Updates the widget with configuration data from a loaded project."""
        logger.info("ConfigWidget updating with new configuration data from project.")
        self.update_configurations(config_instances)
        self.config_source_label.setText("Loaded from Project")

    # --- Internal State Management and UI Logic ---

    def _set_dirty(self, is_dirty: bool):
        self._is_dirty = is_dirty
        self.apply_discard_frame.setVisible(is_dirty)

    def _mark_as_dirty(self, _=None):
        if not self._is_dirty: self._set_dirty(True)
        # When a dropdown changes, we need to update visibility of related fields
        if isinstance(self.sender(), ParameterInputGroup) and self.sender().param_name.endswith("oil_profile_type"):
            self._update_profile_param_visibility()

    def _discard_changes(self):
        logger.debug("Discarding pending configuration changes.")
        self.update_configurations(self.config_instances)

    def _apply_changes(self):
        logger.info("Attempting to apply configuration changes.")
        pending_data = {dc_name: asdict(instance) for dc_name, instance in self.config_instances.items()}
        all_valid = True
        
        for key, widget in self.input_groups.items():
            if not widget.isVisible(): continue
            instance_name, param_name = key.split('.', 1)
            raw_value = widget.get_value()
            
            # Handle optional fields that are not checked
            if widget.is_checkable() and not widget.is_checked():
                pending_data[instance_name][param_name] = None
                continue

            if param_name == 'oil_profile_type': internal_value = self.REVERSE_OIL_PROFILE_MAP.get(raw_value, raw_value)
            elif param_name == 'injection_profile_type': internal_value = self.REVERSE_INJECTION_PROFILE_MAP.get(raw_value, raw_value)
            else: internal_value = raw_value

            field_type = self.config_instances[instance_name].__class__.__annotations__[param_name]
            try:
                coerced_value = self._coerce_value(internal_value, field_type)
                pending_data[instance_name][param_name] = coerced_value
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
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Validation Error", f"Could not apply changes due to inconsistent data:\n\n{e}")
            return

        self.config_instances = new_instances
        logger.info("Configuration changes applied successfully.")
        self.config_source_label.setText("Custom Applied")

        for name, instance in self.config_instances.items():
            for f in fields(instance):
                self.configuration_changed.emit(name, f.name, getattr(instance, f.name))
        
        self.update_configurations(self.config_instances)

    def update_configurations(self, config_data: Dict[str, Any]):
        """Internal method to populate the form from a given data dictionary."""
        self.config_instances = deepcopy(config_data)
        self._populate_all_forms()
        self._set_dirty(False)

    def _populate_all_forms(self):
        while self.tabs.count() > 0: self.tabs.removeTab(0)
        self.input_groups.clear()
        if not CONFIGURABLE_DATACLASSES:
            self.tabs.addTab(QLabel("Configuration models not loaded."), "Error")
            return
            
        for name, dc_type in CONFIGURABLE_DATACLASSES.items():
            instance = self.config_instances.get(dc_type.__name__)
            if instance:
                page_widget = QWidget()
                form_layout = QFormLayout(page_widget)
                form_layout.setSpacing(10); form_layout.setContentsMargins(10, 15, 10, 15)
                for field in fields(instance): self._create_input_group_for_field(instance, field, form_layout)
                scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
                scroll_area.setWidget(page_widget)
                self.tabs.addTab(scroll_area, name)
        self._update_profile_param_visibility()

    def _get_input_type_and_options(self, field: Field) -> dict:
        """Inspects a dataclass field and returns the appropriate input_type and widget options."""
        kwargs = {}
        base_type = field.type
        origin = get_origin(base_type)
        if origin in (Union, UnionType):
            args = get_args(base_type)
            base_type = next((t for t in args if t is not type(None)), str)

        if field.name in self.SPECIAL_DROPDOWNS:
            kwargs['input_type'] = 'combobox'; kwargs['items'] = self.SPECIAL_DROPDOWNS[field.name]
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
        instance_name = type(instance).__name__
        key = f"{instance_name}.{field.name}"
        kwargs = self._get_input_type_and_options(field)
        
        current_value = getattr(instance, field.name)
        default_value = getattr(self.default_instances.get(instance_name), field.name, None)
        is_modified = current_value != default_value
        if isinstance(current_value, float) and isinstance(default_value, (float, int)):
            is_modified = not np.isclose(current_value, default_value)
        
        if field.name == 'oil_profile_type': display_value = self.OIL_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif field.name == 'injection_profile_type': display_value = self.INJECTION_PROFILE_DISPLAY_MAP.get(current_value, current_value)
        elif isinstance(current_value, list): display_value = ', '.join(map(str, current_value))
        else: display_value = current_value

        kwargs.update({
            "param_name": key, "label_text": field.name.replace("_", " ").title(),
            "default_value": display_value, "help_text": field.metadata.get("help", "No description available.")
        })
        
        input_group = ParameterInputGroup(**kwargs)
        input_group.setProperty("isModified", is_modified)
        input_group.finalValueChanged.connect(self._mark_as_dirty)
        
        if get_origin(field.type) in (Union, UnionType):
            input_group.set_checkable(True)
            if current_value is None:
                input_group.set_checked(False)

        layout.addRow(input_group)
        self.input_groups[key] = input_group

    def _confirm_reset_all(self):
        reply = QMessageBox.question(self, "Confirm Reset", "Reset all parameters to application defaults? This will discard any applied or pending changes.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("ConfigWidget resetting all parameters to application defaults.")
            self.update_configurations(deepcopy(self.default_instances))
            self.config_source_label.setText("Application Defaults")
            for instance_name, instance in self.config_instances.items():
                for field in fields(instance): self.configuration_changed.emit(instance_name, field.name, getattr(instance, field.name))

    def _load_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Configuration File", "", "JSON Files (*.json)")
        if not filepath: return
        try:
            with open(filepath, 'r') as f: data = json.load(f)
            loaded_instances = {}
            for dc_name, dc_type in CONFIGURABLE_DATACLASSES.items():
                instance_name = dc_type.__name__
                if instance_name in data and isinstance(data[instance_name], dict):
                    valid_data = {k: v for k, v in data[instance_name].items() if k in {f.name for f in fields(dc_type)}}
                    loaded_instances[instance_name] = dc_type(**valid_data)
                else: loaded_instances[instance_name] = self.default_instances[instance_name]
            if loaded_instances:
                self.update_configurations(loaded_instances)
                self.config_source_label.setText(f"Loaded from File: {Path(filepath).name}")
                for name, inst in loaded_instances.items():
                    for f in fields(inst): self.configuration_changed.emit(name, f.name, getattr(inst, f.name))
        except Exception as e: QMessageBox.critical(self, "Load Error", f"Could not load file:\n\n{e}")

    def _save_to_file(self):
        if self._is_dirty:
            QMessageBox.warning(self, "Unapplied Changes", "You have unapplied changes. Please apply or discard them before saving.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Configuration As", "config.json", "JSON Files (*.json)")
        if not filepath: return
        try:
            data_to_save = {name: asdict(inst) for name, inst in self.config_instances.items()}
            with open(filepath, 'w') as f: json.dump(data_to_save, f, indent=2, sort_keys=True)
            self.save_configuration_to_file_requested.emit(data_to_save)
            QMessageBox.information(self, "Save Successful", f"Configuration saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n\n{e}")
            
    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
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
        oil_profile_type_widget = self.input_groups.get(f"{ProfileParameters.__name__}.oil_profile_type")
        if not oil_profile_type_widget: return
        
        display_name = oil_profile_type_widget.get_value()
        profile_type = self.REVERSE_OIL_PROFILE_MAP.get(display_name, display_name)
        
        visibility_map = {
            'oil_annual_fraction_of_total': (profile_type == 'custom_fractions'),
            'plateau_duration_fraction_of_life': 'plateau' in profile_type,
            'initial_decline_rate_annual_fraction': 'decline' in profile_type,
            'hyperbolic_b_factor': 'hyperbolic' in profile_type,
            'min_economic_rate_fraction_of_peak': 'decline' in profile_type,
            'warn_if_defaults_used': True
        }
        for param_name, is_visible in visibility_map.items():
            widget_key = f"ProfileParameters.{param_name}"
            if widget_key in self.input_groups: self.input_groups[widget_key].setVisible(is_visible)