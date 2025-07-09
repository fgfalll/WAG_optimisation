import logging
from typing import Optional, Dict, Any, Type, get_origin, get_args, Union
from types import UnionType
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTabWidget, QMessageBox, QLineEdit,
    QPlainTextEdit, QDialogButtonBox, QWidget, QFormLayout
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

try:
    from .parameter_input_group import ParameterInputGroup
except ImportError:
    class ParameterInputGroup(QWidget): pass
    logging.critical("ManualWellDialog: Failed to import ParameterInputGroup widget.")

try:
    from co2eor_optimizer.core.data_models import WellData
except ImportError:
    class WellData: pass
    logging.critical("ManualWellDialog: Could not import WellData model.")

logger = logging.getLogger(__name__)

class ManualWellDialog(QDialog):
    """A dialog to manually input a well with key analysis parameters and optional log curves."""
    
    KEY_PARAM_DEFS = {
        'API': ("Oil Gravity (°API)", "lineedit", float, {'default_value': 32.0}),
        'Temperature': ("Reservoir Temperature (°F)", "lineedit", float, {'default_value': 212.0}),
        'ReferenceDepth': ("Reference Depth (ft)", "lineedit", float, {'default_value': 10000.0})
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Well Manually")
        self.setMinimumSize(600, 550)

        self.key_param_widgets: Dict[str, ParameterInputGroup] = {}
        self.key_param_values: Dict[str, Any] = {}
        
        main_layout = QVBoxLayout(self)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Well Name:"))
        self.well_name_edit = QLineEdit("Well-Manual-1")
        name_layout.addWidget(self.well_name_edit)
        main_layout.addLayout(name_layout)

        params_group = QGroupBox("Key Parameters for Analysis")
        params_form_layout = QFormLayout(params_group)
        params_form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        
        for name, (label, w_type, p_type, kwargs) in self.KEY_PARAM_DEFS.items():
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.key_param_widgets[name] = input_group
            params_form_layout.addRow(input_group)

        main_layout.addWidget(params_group)

        self.log_curves_group = QGroupBox("Detailed Log Curves (Optional)")
        self.log_curves_group.setCheckable(True); self.log_curves_group.setChecked(False)
        log_group_layout = QVBoxLayout(self.log_curves_group)
        depth_group = QGroupBox("Shared Depth Curve (MD)"); depth_layout = QVBoxLayout(depth_group)
        self.depth_edit = QPlainTextEdit(); self.depth_edit.setPlaceholderText("Paste comma-separated depth values here...")
        depth_layout.addWidget(self.depth_edit); log_group_layout.addWidget(depth_group)
        self.log_tabs = QTabWidget(); self.log_tabs.setTabsClosable(True)
        self.add_log_curve_tab("GR")
        tab_button_layout = QHBoxLayout()
        self.add_tab_btn = QPushButton(QIcon.fromTheme("list-add"), "Add Another Curve")
        tab_button_layout.addStretch(); tab_button_layout.addWidget(self.add_tab_btn)
        log_group_layout.addWidget(self.log_tabs); log_group_layout.addLayout(tab_button_layout)
        main_layout.addWidget(self.log_curves_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

        self.add_tab_btn.clicked.connect(lambda: self.add_log_curve_tab())
        self.log_tabs.tabCloseRequested.connect(self.remove_log_curve_tab)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self._initialize_defaults()
        
        self.setStyleSheet("""
            ParameterInputGroup #InputWidgetFrame[feedbackLevel="error"] {
                border: 1px solid #E74C3C; border-radius: 4px;
            }
            ParameterInputGroup #FeedbackLabel[feedbackLevel="error"] { color: #E74C3C; }
        """)

    def _initialize_defaults(self):
        for name, widget in self.key_param_widgets.items():
            param_def = self.KEY_PARAM_DEFS[name]
            default_value = param_def[3].get('default_value')
            param_type = param_def[2]
            self.key_param_values[name] = self._coerce_value(default_value, param_type)
            widget.set_value(default_value)

    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        if isinstance(value, str): value = value.replace(',', '.')
        origin = get_origin(type_hint); args = get_args(type_hint)
        is_optional = origin in (Union, UnionType) and type(None) in args
        base_type = next((t for t in args if t is not type(None)), type_hint) if is_optional else type_hint
        if value is None or (isinstance(value, str) and not value.strip()):
            if is_optional: return None
            else: raise ValueError("This field cannot be empty.")
        try:
            if base_type is float: return float(value)
            if base_type is int: return int(float(value))
        except (ValueError, TypeError): raise ValueError(f"Must be a valid number.")
        return value

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return
        param_name = sender.param_name; param_type = sender.property("param_type")
        try:
            # First, clear any previous error state
            sender.clear_error()
            corrected_value = self._coerce_value(value, param_type)
            
            # --- ENHANCEMENT: Add specific validation logic ---
            if param_name == 'API' and not (10 <= corrected_value <= 60):
                raise ValueError("API Gravity must be between 10 and 60.")
            if param_name == 'Temperature' and not (50 <= corrected_value <= 400):
                raise ValueError("Temperature must be between 50°F and 400°F.")
            if param_name == 'ReferenceDepth' and corrected_value < 0:
                raise ValueError("Depth cannot be negative.")
            
            # If all checks pass, update the internal value
            self.key_param_values[param_name] = corrected_value
            logger.debug(f"Dialog param '{param_name}' updated to: {corrected_value}")

        except (ValueError, TypeError) as e:
            logger.warning(f"Validation failed for {param_name} with value '{value}': {e}")
            sender.show_error(str(e))
            # Mark the value as invalid by removing it from our valid dictionary
            if param_name in self.key_param_values:
                del self.key_param_values[param_name]

    def add_log_curve_tab(self, name=""):
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        name_edit = QLineEdit(name); name_edit.setPlaceholderText("Curve Name (e.g., GR, NPHI)")
        data_edit = QPlainTextEdit(); data_edit.setPlaceholderText("Paste comma-separated log values...")
        layout.addWidget(QLabel("Curve Name:"), 0); layout.addWidget(name_edit, 0)
        layout.addWidget(QLabel("Curve Data:"), 0); layout.addWidget(data_edit, 1)
        tab_name = name if name else f"Curve {self.log_tabs.count() + 1}"
        index = self.log_tabs.addTab(tab_widget, tab_name)
        self.log_tabs.setCurrentIndex(index)
        tab_widget.setProperty("name_widget", name_edit); tab_widget.setProperty("data_widget", data_edit)
        name_edit.textChanged.connect(lambda text, idx=index: self.log_tabs.setTabText(idx, text if text else f"Curve {idx + 1}"))

    def remove_log_curve_tab(self, index: int):
        if self.log_tabs.count() > 1: self.log_tabs.removeTab(index)
        else: QMessageBox.warning(self, "Cannot Remove", "At least one log curve is required if this section is enabled.")

    def get_well_data(self) -> Optional[WellData]:
        well_name = self.well_name_edit.text().strip()
        if not well_name:
            QMessageBox.warning(self, "Input Error", "A well name is required.")
            return None
        
        # Check if all required parameters have valid, stored values
        if len(self.key_param_values) < len(self.KEY_PARAM_DEFS):
            QMessageBox.warning(self, "Input Error", "One or more key parameters have invalid values. Please correct them.")
            return None
            
        try:
            metadata = self.key_param_values.copy()
            depths = np.array([metadata['ReferenceDepth']], dtype=float)
            properties: Dict[str, np.ndarray] = {}; units: Dict[str, str] = {}

            if self.log_curves_group.isChecked():
                depth_text = self.depth_edit.toPlainText().strip()
                if not depth_text: raise ValueError("If providing detailed logs, the Depth curve cannot be empty.")
                depths = np.array([float(v.strip()) for v in depth_text.split(',') if v.strip()], dtype=float)
                properties['DEPT'] = depths; units['DEPT'] = 'ft'

                for i in range(self.log_tabs.count()):
                    tab = self.log_tabs.widget(i)
                    name_widget = tab.property("name_widget"); data_widget = tab.property("data_widget")
                    log_name = name_widget.text().strip().upper(); log_data_text = data_widget.toPlainText().strip()
                    if not log_name or not log_data_text: raise ValueError(f"Curve name and data are required for Tab {i+1}.")
                    log_values = np.array([float(v.strip()) for v in log_data_text.split(',') if v.strip()], dtype=float)
                    if log_values.size != depths.size: raise ValueError(f"Data size mismatch for curve '{log_name}'. Expected {depths.size} values, but got {log_values.size}.")
                    properties[log_name] = log_values; units[log_name] = 'N/A'
            
            return WellData(name=well_name, depths=depths, properties=properties, units=units, metadata=metadata)
        except Exception as e:
            QMessageBox.critical(self, "Data Error", f"Could not create well data: {e}")
            return None