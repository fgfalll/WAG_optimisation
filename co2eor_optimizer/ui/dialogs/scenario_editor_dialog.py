import logging
from typing import Optional, Any, Dict, Type, get_origin, get_args, Union
from copy import deepcopy
from dataclasses import fields, is_dataclass
from types import UnionType
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox,
    QScrollArea, QWidget, QLabel, QTabWidget, QMessageBox
)
from PyQt6.QtCore import pyqtSignal

# Decoupled imports for resilience
try:
    from co2eor_optimizer.ui.widgets.parameter_input_group import ParameterInputGroup
except ImportError:
    ParameterInputGroup = None
try:
    from co2eor_optimizer.core.data_models import EconomicParameters, EORParameters, OperationalParameters
    # A map from the internal key to the dataclass type
    DATACLASS_MAP: Dict[str, Type] = {
        "EconomicParameters": EconomicParameters, 
        "EORParameters": EORParameters, 
        "OperationalParameters": OperationalParameters
    }
except ImportError:
    DATACLASS_MAP, ParameterInputGroup = {}, None
    logging.warning("ScenarioEditorDialog: Could not import core data models or ParameterInputGroup. Functionality will be limited.")

logger = logging.getLogger(__name__)

class ScenarioEditorDialog(QDialog):
    """A dialog for creating or editing a comparative scenario by overriding base parameters."""
    scenario_saved = pyqtSignal(dict)

    # Re-define special dropdowns to stay decoupled from ConfigWidget
    SPECIAL_DROPDOWNS = { "injection_scheme": ['continuous', 'wag'] }

    def __init__(self, scenario_data: Dict, base_config: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.base_config = base_config
        self.base_values_map: Dict[str, Any] = {} # "ClassName.param_name" -> base_value
        
        self.current_data = deepcopy(scenario_data) or {"name": "New Scenario", "params": {}}
        self.current_overrides = self.current_data.get("params", {})

        self._setup_ui()
        self._populate_all_tabs()
        
    def _setup_ui(self):
        self.setWindowTitle(f"Scenario Editor")
        self.setMinimumSize(650, 550)
        
        self.setStyleSheet("""
            QTabWidget::pane { border-top: 1px solid #C2C7CB; }
            QScrollArea { border: none; background-color: transparent; }
            
            /* Style for overridden parameters */
            ParameterInputGroup[isOverride="true"] {
                background-color: #e8f0fe; /* Light blue highlight */
                border: 1px solid #a3c5f8;
                border-radius: 6px;
                padding: 2px;
            }
            ParameterInputGroup[isOverride="true"] > QLabel {
                font-weight: bold; /* Make label bold to draw attention */
            }
        """)

        main_layout = QVBoxLayout(self)
        
        name_layout = QFormLayout()
        self.name_edit = QLineEdit(self.current_data.get("name", "New Scenario"))
        self.name_edit.setPlaceholderText("Enter a unique name for this scenario")
        self.name_edit.textChanged.connect(self._update_window_title)
        name_layout.addRow("<b>Scenario Name:</b>", self.name_edit)
        main_layout.addLayout(name_layout)

        self.tab_widget = QTabWidget()
        if not DATACLASS_MAP or not ParameterInputGroup:
            self.tab_widget.addTab(QLabel("Parameter editing disabled: core application modules not loaded."), "Error")
        main_layout.addWidget(self.tab_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        self._update_window_title(self.name_edit.text())

    def _update_window_title(self, name: str):
        title = "Scenario Editor"
        if name: title = f"{title}: {name}"
        self.setWindowTitle(title)

    def _populate_all_tabs(self):
        if not DATACLASS_MAP: return
        for name, dc_type in DATACLASS_MAP.items():
            tab_title = name.replace("Parameters", "")
            self._create_parameter_tab(tab_title, name, dc_type)

    def _create_parameter_tab(self, title: str, dc_name: str, dc_type: Type):
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        container = QWidget(); layout = QFormLayout(container)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        
        base_instance = self.base_config.get(dc_name, dc_type())
        scenario_overrides = self.current_overrides.get(dc_name, {})

        for field in fields(dc_type):
            param_key = f"{dc_name}.{field.name}"
            base_value = getattr(base_instance, field.name)
            self.base_values_map[param_key] = base_value

            is_overridden = field.name in scenario_overrides
            current_value = scenario_overrides.get(field.name, base_value)
            display_value = ', '.join(map(str, current_value)) if isinstance(current_value, list) else current_value
            
            kwargs = {
                "param_name": param_key, "label_text": field.name.replace("_", " ").title(),
                "default_value": display_value, "placeholder_text": f"Base: {base_value}",
                "help_text": field.metadata.get("help", f"Base value for this parameter is {base_value}.")
            }

            if field.name in self.SPECIAL_DROPDOWNS:
                kwargs.update({'input_type': 'combobox', 'items': self.SPECIAL_DROPDOWNS[field.name]})
            else:
                kwargs['input_type'] = 'lineedit'

            input_group = ParameterInputGroup(**kwargs)
            input_group.setProperty("isOverride", is_overridden)
            input_group.finalValueChanged.connect(self._on_parameter_updated)
            layout.addRow(input_group)
        
        scroll_area.setWidget(container)
        self.tab_widget.addTab(scroll_area, title)

    def _on_parameter_updated(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return

        param_key = sender.param_name
        dc_name, param_name = param_key.split('.', 1)
        base_value = self.base_values_map.get(param_key)
        
        dc_type = DATACLASS_MAP.get(dc_name)
        if not dc_type: return
        field_type_hint = dc_type.__annotations__.get(param_name)

        try:
            coerced_value = self._coerce_value(value, field_type_hint)
            
            is_override = coerced_value != base_value
            if isinstance(coerced_value, float) and isinstance(base_value, (float, int)):
                is_override = not np.isclose(coerced_value, base_value)

            sender.setProperty("isOverride", is_override)
            sender.style().unpolish(sender); sender.style().polish(sender)

            if dc_name not in self.current_overrides: self.current_overrides[dc_name] = {}
            
            if is_override:
                self.current_overrides[dc_name][param_name] = coerced_value
            elif param_name in self.current_overrides.get(dc_name, {}):
                del self.current_overrides[dc_name][param_name]
                if not self.current_overrides[dc_name]:
                    del self.current_overrides[dc_name]
            
            sender.clear_error()
            logger.debug(f"Scenario overrides updated: {self.current_overrides}")
        except (ValueError, TypeError) as e:
            sender.show_error(str(e))

    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        origin = get_origin(type_hint); args = get_args(type_hint)
        is_optional = origin in (Union, UnionType) and type(None) in args
        base_type = next((t for t in args if t is not type(None)), type_hint) if is_optional else type_hint
        
        if value is None or (isinstance(value, str) and not value.strip()):
            if is_optional: return None
            else: raise ValueError("This field cannot be empty.")

        if base_type is bool: return str(value).lower() in ('true', '1', 'yes') or value is True
        if base_type is float: return float(str(value).replace(',', '.'))
        if base_type is int: return int(float(str(value).replace(',', '.')))
        return value # For strings and other types

    def save_and_accept(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Scenario name cannot be empty.")
            return
        self.current_data["name"] = name
        self.current_data["params"] = self.current_overrides
        self.scenario_saved.emit(self.current_data)
        self.accept()

    @staticmethod
    def edit_scenario(data: Dict, base_config: Dict, parent: QWidget) -> Optional[Dict]:
        """Convenience static method to create, show, and get data from the dialog."""
        if not DATACLASS_MAP or not ParameterInputGroup:
            QMessageBox.critical(parent, "Error", "Cannot open Scenario Editor because core application components are missing.")
            return None
        dialog = ScenarioEditorDialog(data, base_config, parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.current_data
        return None