import logging
from typing import Optional, Dict, Any, Type, List
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QMessageBox, QLineEdit, QDialogButtonBox, QWidget, QFormLayout,
    QTableWidget, QTableWidgetItem, QAbstractItemView
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QPointF, pyqtSignal

try:
    from .parameter_input_group import ParameterInputGroup
except ImportError:
    class ParameterInputGroup(QWidget):
        finalValueChanged = pyqtSignal(object)
        param_name: str = ""
        def __init__(self, param_name="", label_text="", input_type="", **kwargs): super().__init__()
        def get_value(self): return ""
        def set_value(self, v): pass
        def setEnabled(self, b): pass
        def setProperty(self, n, v): pass
        def property(self, n): pass
    logging.critical("ManualWellDialog: Failed to import ParameterInputGroup.")
try:
    from .depth_profile_dialog import DepthProfileDialog
except ImportError:
    class DepthProfileDialog(QDialog): pass
    logging.critical("ManualWellDialog: Failed to import DepthProfileDialog.")
try:
    from co2eor_optimizer.core.data_models import WellData
except ImportError:
    class WellData: pass
    logging.critical("ManualWellDialog: Could not import WellData model.")

logger = logging.getLogger(__name__)

class ManualWellDialog(QDialog):
    """A dialog to manually input a well with detailed perforations and an optional path editor."""
    
    KEY_PARAM_DEFS = {
        'API': ("Oil Gravity (°API)", "lineedit", float, {'default_value': 32.0}),
        'Temperature': ("Reservoir Temperature (°F)", "lineedit", float, {'default_value': 212.0}),
        'TopDepth': ("Well Top Depth / MD (ft)", "lineedit", float, {'default_value': 9900.0}),
        'BottomDepth': ("Well Bottom Depth / MD (ft)", "lineedit", float, {'default_value': 10100.0}),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Add Well Manually")
        self.setMinimumSize(600, 500)

        self.key_param_widgets: Dict[str, ParameterInputGroup] = {}
        self.key_param_values: Dict[str, Any] = {}
        self.well_path: List[QPointF] = []
        
        main_layout = QVBoxLayout(self)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Well Name:"))
        self.well_name_edit = QLineEdit("Well-Manual-1")
        name_layout.addWidget(self.well_name_edit)
        main_layout.addLayout(name_layout)

        params_group = QGroupBox("Key Well Parameters")
        params_form_layout = QFormLayout(params_group)
        for name, (label, w_type, p_type, kwargs) in self.KEY_PARAM_DEFS.items():
            widget = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            widget.setProperty("param_type", p_type)
            widget.finalValueChanged.connect(self._on_parameter_changed)
            self.key_param_widgets[name] = widget
            params_form_layout.addRow(widget)
        
        self.edit_path_btn = QPushButton(QIcon.fromTheme("document-edit"), " Edit Depth Profile... (Optional)")
        params_form_layout.addRow(self.edit_path_btn)
        main_layout.addWidget(params_group)

        perf_group = QGroupBox("Perforations")
        perf_layout = QVBoxLayout(perf_group)
        self.no_perf_warning_label = QLabel(
            "<b>Warning:</b> No perforations defined. The entire well path will be treated as connected to the reservoir."
        )
        self.no_perf_warning_label.setStyleSheet("color: #D32F2F;")
        perf_button_layout = QHBoxLayout()
        self.add_perf_btn = QPushButton(QIcon.fromTheme("list-add"), "Add Perforation")
        self.remove_perf_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove Selected")
        perf_button_layout.addStretch()
        perf_button_layout.addWidget(self.add_perf_btn)
        perf_button_layout.addWidget(self.remove_perf_btn)
        
        self.perf_table = QTableWidget()
        self.perf_table.setColumnCount(4)
        self.perf_table.setHorizontalHeaderLabels(["Top MD (ft)", "Bottom MD (ft)", "Oil Gravity (°API)", "Temperature (°F)"])
        self.perf_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        perf_layout.addWidget(self.no_perf_warning_label)
        perf_layout.addWidget(self.perf_table)
        perf_layout.addLayout(perf_button_layout)
        main_layout.addWidget(perf_group, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

        self.edit_path_btn.clicked.connect(self._open_depth_profile_editor)
        self.add_perf_btn.clicked.connect(self._add_perforation_row)
        self.remove_perf_btn.clicked.connect(self._remove_perforation_row)
        self.perf_table.model().rowsInserted.connect(self._update_ui_state)
        self.perf_table.model().rowsRemoved.connect(self._update_ui_state)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initializes widgets with default values and populates the internal data dictionary."""
        for name, widget in self.key_param_widgets.items():
            param_def = self.KEY_PARAM_DEFS[name]
            default_value = param_def[3].get('default_value')
            p_type = param_def[2]
            
            widget.set_value(default_value)
            
            try:
                self.key_param_values[name] = self._coerce_value(str(default_value), p_type)
            except (ValueError, TypeError) as e:
                logger.error(f"Could not coerce default value for {name}: {e}")

        self._update_ui_state()

    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        try:
            if isinstance(value, str): value = value.replace(',', '.')
            if value is None or (isinstance(value, str) and not value.strip()): return None
            if type_hint is float: return float(value)
            if type_hint is int: return int(float(value))
            return value
        except (ValueError, TypeError): raise ValueError("Must be a valid number.")

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return
        param_name = sender.param_name
        try:
            p_type = self.KEY_PARAM_DEFS[param_name][2]
            self.key_param_values[param_name] = self._coerce_value(value, p_type)
            if param_name in ('TopDepth', 'BottomDepth') and self.well_path:
                top = self.key_param_values.get('TopDepth', 0)
                bottom = self.key_param_values.get('BottomDepth', 0)
                if top is not None and bottom is not None and bottom > top:
                    self.well_path[0].setY(top)
                    self.well_path[-1].setY(bottom)
        except (ValueError, TypeError):
            if param_name in self.key_param_values: del self.key_param_values[param_name]

    def _open_depth_profile_editor(self):
        if not self.well_path:
            top = self.key_param_values.get('TopDepth')
            bottom = self.key_param_values.get('BottomDepth')
            if top is None or bottom is None or bottom <= top:
                QMessageBox.warning(self, "Invalid Depths", "Please set valid Top and Bottom depths before editing the profile.")
                return
            self.well_path = [QPointF(0, top), QPointF(0, bottom)]
            
        dialog = DepthProfileDialog(self.well_path, self)
        if dialog.exec():
            self.well_path = dialog.get_path()
            logger.info("Well path updated from the editor dialog.")

    def _add_perforation_row(self):
        row_pos = self.perf_table.rowCount()
        self.perf_table.insertRow(row_pos)
        self.perf_table.setItem(row_pos, 2, QTableWidgetItem(str(self.key_param_values.get('API', '32.0'))))
        self.perf_table.setItem(row_pos, 3, QTableWidgetItem(str(self.key_param_values.get('Temperature', '212.0'))))

    def _remove_perforation_row(self):
        selected_rows = sorted({index.row() for index in self.perf_table.selectedIndexes()}, reverse=True)
        for row in selected_rows:
            self.perf_table.removeRow(row)

    def _update_ui_state(self):
        has_perforations = self.perf_table.rowCount() > 0
        self.key_param_widgets['API'].setEnabled(not has_perforations)
        self.key_param_widgets['Temperature'].setEnabled(not has_perforations)
        self.no_perf_warning_label.setVisible(not has_perforations)

    def get_well_data(self) -> Optional[WellData]:
        well_name = self.well_name_edit.text().strip()
        if not well_name:
            QMessageBox.warning(self, "Input Error", "A well name is required."); return None
        
        try:
            path_to_use = self.well_path

            if not path_to_use:
                top = self.key_param_values.get('TopDepth')
                bottom = self.key_param_values.get('BottomDepth')
                if top is None or bottom is None or bottom <= top:
                    raise ValueError("Could not create default well path. Please ensure Top and Bottom depths are valid and Top < Bottom.")
                logger.info("No explicit path set. Creating default straight vertical well path with 10ft sampling.")
                depths_np = np.arange(top, bottom, 10.0)
                if depths_np.size == 0 or depths_np[-1] < bottom:
                    depths_np = np.append(depths_np, bottom)
                if depths_np.size == 0:
                     depths_np = np.array([top, bottom])
                well_path_np = np.column_stack((np.zeros_like(depths_np), depths_np))
            else:
                well_path_np = np.array([[p.x(), p.y()] for p in path_to_use])
            
            # Ensure depths are always sorted and unique, which is good practice for analysis
            depths_np = np.sort(np.unique(well_path_np[:, 1]))

            perfs = []
            for row in range(self.perf_table.rowCount()):
                try:
                    top = float(self.perf_table.item(row, 0).text())
                    bot = float(self.perf_table.item(row, 1).text())
                    api = float(self.perf_table.item(row, 2).text())
                    temp = float(self.perf_table.item(row, 3).text())
                    if top >= bot: raise ValueError(f"Row {row+1}: Top depth must be less than bottom depth.")
                    perfs.append({'top': top, 'bottom': bot, 'api': api, 'temp': temp})
                except (AttributeError, ValueError) as e:
                    raise ValueError(f"Invalid data in perforations table at row {row + 1}: {e}")

            final_metadata = self.key_param_values.copy()
            if not perfs:
                final_metadata['perforations_treatment'] = 'entire_wellbore'
            
            return WellData(
                name=well_name, depths=depths_np,
                well_path=well_path_np, perforation_properties=perfs,
                metadata=final_metadata, properties={}, units={}
            )
        except Exception as e:
            QMessageBox.critical(self, "Data Creation Error", f"Could not create well data: {e}"); return None