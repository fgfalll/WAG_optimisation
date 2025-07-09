import logging
from typing import Optional, Any, Dict, List, Type, get_origin, get_args, Union, Tuple
from types import UnionType
import numpy as np
from pathlib import Path
from copy import deepcopy
import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QSplitter, QTabWidget, QFileDialog, QMessageBox, QLineEdit,
    QToolBox, QDialog, QListWidget, QListWidgetItem, QPlainTextEdit, QDialogButtonBox,
    QSizePolicy, QTextBrowser, QCheckBox
)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import pyqtSignal, Qt

try:
    from .widgets.parameter_input_group import ParameterInputGroup
    from .widgets.pvt_editor_dialog import PVTEditorDialog
    from .widgets.log_viewer_dialog import LogViewerDialog
    from .widgets.manual_well_dialog import ManualWellDialog
except ImportError as e:
    class ParameterInputGroup(QWidget): pass
    class PVTEditorDialog(QDialog): pass
    class LogViewerDialog(QDialog): pass
    class ManualWellDialog(QDialog): pass
    logging.critical(f"DataManagementWidget: Failed to import critical UI components: {e}")

try:
    from co2eor_optimizer.core.data_models import WellData, ReservoirData, EOSModelParameters, PVTProperties
    from co2eor_optimizer.parsers.las_parser import parse_las, MissingWellNameError
except ImportError:
    class WellData: pass
    class ReservoirData: pass
    class EOSModelParameters: pass
    class PVTProperties: pass
    def parse_las(*args, **kwargs): return None
    class MissingWellNameError(Exception): pass
    logging.critical("DataManagementWidget: Core data models or LAS parser not found.")

logger = logging.getLogger(__name__)


class DataManagementWidget(QWidget):
    project_data_updated = pyqtSignal(dict)
    status_message_updated = pyqtSignal(str, int)

    MANUAL_RES_DEFS = {
        'nx': ("NX", "lineedit", int, {'default_value': 50}),
        'ny': ("NY", "lineedit", int, {'default_value': 50}),
        'nz': ("NZ", "lineedit", int, {'default_value': 10}),
        'poro': ("Porosity", "lineedit", float, {'default_value': 0.20, 'decimals': 3}),
        'perm': ("Perm (mD)", "lineedit", float, {'default_value': 100.0})
    }
    
    MANUAL_PVT_DEFS = {
        'temperature': ("Reservoir Temp (°F)", "lineedit", float, {'default_value': 212.0}),
        'gas_specific_gravity': ("Gas Gravity (air=1)", "lineedit", float, {'default_value': 0.7}),
        'ref_pres': ("Ref. Pressure (psia)", "lineedit", float, {'default_value': 3000.0}),
        'oil_fvf': ("Oil FVF (Bo, RB/STB)", "lineedit", float, {'default_value': 1.25}),
        'oil_visc': ("Oil Viscosity (cP)", "lineedit", float, {'default_value': 0.8}),
        'sol_gor': ("Solution GOR (Rs, scf/STB)", "lineedit", float, {'default_value': 500.0}),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.well_data_list: List[WellData] = []
        self.reservoir_data: Optional[ReservoirData] = None
        self.pvt_properties: Optional[PVTProperties] = None 
        self.detailed_pvt_data: Optional[Dict[str, Any]] = None
        
        self.manual_inputs_widgets: Dict[str, ParameterInputGroup] = {}
        self.manual_inputs_values: Dict[str, Any] = {}

        self._setup_ui()
        self._connect_signals()
        self.clear_all_project_data()
        
        # --- ENHANCEMENT: Added styles for the new report format ---
        self.setStyleSheet("""
            QTextBrowser {
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
                background-color: #fdfdfd;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            details {
                border: 1px solid #d0d7de;
                border-radius: 6px;
                padding: .5em .5em 0;
                margin-bottom: 1em;
                background-color: #ffffff;
            }
            summary {
                font-weight: bold;
                font-size: 11pt;
                margin: -.5em -.5em 0;
                padding: .5em;
                cursor: pointer;
            }
            details[open] {
                padding: .5em;
            }
            details[open] summary {
                border-bottom: 1px solid #d0d7de;
                margin-bottom: .5em;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 5px;
            }
            th, td {
                padding: 6px 8px;
                text-align: left;
                border-bottom: 1px solid #e1e4e8;
            }
            th {
                background-color: #f6f8fa;
                font-weight: 600;
            }
            tr:last-child > td {
                border-bottom: none;
            }
            ParameterInputGroup #InputWidgetFrame[feedbackLevel="error"]   { border: 1px solid #E74C3C; border-radius: 4px; }
            ParameterInputGroup #InputWidgetFrame[feedbackLevel="warning"] { border: 1px solid #F39C12; border-radius: 4px; }
            ParameterInputGroup #InputWidgetFrame[feedbackLevel="info"]    { border: 1px solid #3498DB; border-radius: 4px; }
            #FeedbackLabel[feedbackLevel="error"]   { color: #E74C3C; }
            #FeedbackLabel[feedbackLevel="warning"] { color: #F39C12; }
            #FeedbackLabel[feedbackLevel="info"]    { color: #3498DB; }
        """)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        self.left_toolbox = QToolBox()
        self.left_toolbox.setMinimumWidth(450)
        self.left_toolbox.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        manual_entry_panel = self._create_manual_entry_panel()
        self.left_toolbox.addItem(manual_entry_panel, QIcon.fromTheme("document-edit"), "Manual Data Entry")
        file_import_panel = self._create_file_import_panel()
        self.left_toolbox.addItem(file_import_panel, QIcon.fromTheme("folder-open"), "File Import (Advanced)")
        main_layout.addWidget(self.left_toolbox)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def _create_manual_entry_panel(self) -> QWidget:
        panel = QWidget(); layout = QVBoxLayout(panel)

        res_group = QGroupBox("1. Uniform Reservoir Definition")
        res_layout = QGridLayout(res_group)
        for name, (label, w_type, p_type, kwargs) in self.MANUAL_RES_DEFS.items():
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
        res_layout.addWidget(self.manual_inputs_widgets['nx'], 0, 0); res_layout.addWidget(self.manual_inputs_widgets['ny'], 0, 1)
        res_layout.addWidget(self.manual_inputs_widgets['nz'], 0, 2); res_layout.addWidget(self.manual_inputs_widgets['poro'], 1, 0, 1, 2)
        res_layout.addWidget(self.manual_inputs_widgets['perm'], 1, 2)
        layout.addWidget(res_group)

        well_group = QGroupBox("2. Well Data"); well_layout = QVBoxLayout(well_group)
        self.well_list_widget = QListWidget(); self.well_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        well_button_layout = QHBoxLayout(); self.add_well_btn = QPushButton(QIcon.fromTheme("list-add"), "Add from LAS...")
        self.add_manual_well_btn = QPushButton(QIcon.fromTheme("edit-add"), "Add Manually...")
        self.view_well_btn = QPushButton(QIcon.fromTheme("document-preview"), "View Well...")
        well_button_layout.addWidget(self.add_well_btn); well_button_layout.addWidget(self.add_manual_well_btn); well_button_layout.addWidget(self.view_well_btn)
        well_layout.addWidget(self.well_list_widget); well_layout.addLayout(well_button_layout); layout.addWidget(well_group)
        
        pvt_main_group = QGroupBox("3. Fluid Properties (PVT)")
        pvt_main_layout = QVBoxLayout(pvt_main_group)
        
        self.simple_pvt_group = QGroupBox("Simplified Fluid Properties (for Black Oil Analysis)")
        simple_pvt_layout = QGridLayout(self.simple_pvt_group)
        simple_pvt_layout.setColumnStretch(0, 1); simple_pvt_layout.setColumnStretch(1, 1)
        pvt_defs_list = list(self.MANUAL_PVT_DEFS.items())
        for i, (name, (label, w_type, p_type, kwargs)) in enumerate(pvt_defs_list):
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            row, col = i // 2, i % 2
            simple_pvt_layout.addWidget(input_group, row, col)
        pvt_main_layout.addWidget(self.simple_pvt_group)

        detailed_pvt_layout = QHBoxLayout()
        self.use_detailed_pvt_checkbox = QCheckBox("Use Detailed PVT / EOS Model")
        self.edit_pvt_btn = QPushButton(QIcon.fromTheme("document-edit"), "Edit Detailed PVT...")
        self.edit_pvt_btn.setEnabled(False)
        
        detailed_pvt_layout.addWidget(self.use_detailed_pvt_checkbox)
        detailed_pvt_layout.addWidget(self.edit_pvt_btn)
        detailed_pvt_layout.addStretch()
        pvt_main_layout.addLayout(detailed_pvt_layout)
        
        layout.addWidget(pvt_main_group)
        
        self.generate_data_btn = QPushButton(QIcon.fromTheme("go-jump"), "Generate Project Data")
        layout.addWidget(self.generate_data_btn); layout.addStretch(); return panel

    def _create_file_import_panel(self) -> QWidget:
        panel = QWidget(); layout = QVBoxLayout(panel)
        info_label = QLabel("Use this panel for advanced workflows involving ECLIPSE data files."); info_label.setWordWrap(True)
        res_group = QGroupBox("Reservoir Model (DATA/INC File)"); res_layout = QHBoxLayout(res_group)
        self.res_path_edit = QLineEdit(); self.res_path_edit.setReadOnly(True); self.browse_res_btn = QPushButton("Browse...")
        res_layout.addWidget(self.res_path_edit); res_layout.addWidget(self.browse_res_btn)
        self.process_files_btn = QPushButton(QIcon.fromTheme("system-run"), "Process Reservoir File")
        layout.addWidget(info_label); layout.addWidget(res_group); layout.addWidget(self.process_files_btn)
        layout.addStretch(); return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget(); layout = QVBoxLayout(panel)
        log_group = QGroupBox("Log"); log_layout = QVBoxLayout(log_group)
        self.log_browser = QTextBrowser(); log_layout.addWidget(self.log_browser)
        review_group = QGroupBox("Data Review"); review_layout = QVBoxLayout(review_group)
        self.res_details = QTextBrowser(); review_layout.addWidget(self.res_details)
        self.confirm_btn = QPushButton(QIcon.fromTheme("dialog-ok-apply"), "Confirm & Finalize Data")
        layout.addWidget(log_group, 1); layout.addWidget(review_group, 2)
        layout.addWidget(self.confirm_btn, 0, Qt.AlignmentFlag.AlignRight); return panel

    def _connect_signals(self):
        self.add_well_btn.clicked.connect(self._add_well_from_file)
        self.add_manual_well_btn.clicked.connect(self._add_well_manually)
        self.view_well_btn.clicked.connect(self._view_well_data)
        self.generate_data_btn.clicked.connect(self._process_manual_data)
        self.browse_res_btn.clicked.connect(self._load_reservoir_file)
        self.process_files_btn.clicked.connect(lambda: QMessageBox.information(self, "Not Implemented", "File processing is currently disabled."))
        self.confirm_btn.clicked.connect(self._confirm_and_emit_data)
        remove_action = QAction("Remove Selected Well", self.well_list_widget)
        remove_action.triggered.connect(self._remove_selected_well)
        self.edit_pvt_btn.clicked.connect(self._open_pvt_editor)
        self.use_detailed_pvt_checkbox.toggled.connect(self._toggle_pvt_mode)

    def _toggle_pvt_mode(self, checked: bool):
        self.simple_pvt_group.setEnabled(not checked)
        self.edit_pvt_btn.setEnabled(checked)
        
    def clear_all_project_data(self):
        self.well_data_list.clear(); self.well_list_widget.clear()
        self.reservoir_data = None
        self.pvt_properties = None 
        self.detailed_pvt_data = None
        if hasattr(self, 'use_detailed_pvt_checkbox'):
            self.use_detailed_pvt_checkbox.setChecked(False)
        self.log_browser.clear(); self.res_details.clear(); self.confirm_btn.setEnabled(False)
        self.log_browser.setHtml("<i>Define reservoir properties and add well data to begin.</i>")
        self.manual_inputs_values.clear()
        all_defs = {**self.MANUAL_RES_DEFS, **self.MANUAL_PVT_DEFS}
        for name, widget in self.manual_inputs_widgets.items():
            param_def = all_defs[name]
            default_value = param_def[3].get('default_value')
            param_type = param_def[2]
            widget.clear_error(); widget.set_value(default_value)
            if default_value is not None:
                self.manual_inputs_values[name] = self._coerce_value(default_value, param_type)
            
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

    def _validate_perm_value(self, perm: float) -> Tuple[Optional[str], Optional[str]]:
        if perm <= 0: return "Permeability must be positive.", "error"
        if perm < 1: return "Warning: Permeability is very low (tight formation).", "warning"
        if perm > 500: return "Warning: Permeability may be unrealistically high.", "warning"
        return None, None

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return
        param_name = sender.param_name; param_type = sender.property("param_type")
        try:
            sender.clear_error()
            corrected_value = self._coerce_value(value, param_type)
            if param_name == 'poro' and not (0 <= corrected_value <= 1):
                raise ValueError("Porosity must be between 0 and 1.")
            if param_name in ('nx', 'ny') and not (1 <= corrected_value <= 1000):
                raise ValueError("Grid dimensions NX and NY must be between 1 and 1000.")
            if param_name == 'nz' and not (1 <= corrected_value <= 200):
                raise ValueError("Grid dimension NZ must be between 1 and 200.")
            if param_name == 'oil_fvf' and not (0.9 <= corrected_value <= 4):
                raise ValueError("Oil FVF (Bo) is out of typical range (0.9-4).")
            if param_name == 'oil_visc' and not (0.1 <= corrected_value <= 100):
                 raise ValueError("Oil Viscosity is out of typical range (0.1-100 cP).")
            if param_name == 'temperature' and not (50 <= corrected_value <= 400):
                raise ValueError("Temperature must be between 50-400°F.")
            if param_name == 'gas_specific_gravity' and not (0.5 <= corrected_value <= 1.2):
                raise ValueError("Gas Gravity must be between 0.5-1.2.")
            if param_name == 'perm':
                message, level = self._validate_perm_value(corrected_value)
                if level == 'error': raise ValueError(message)
                elif level == 'warning': sender.show_warning(message)
            self.manual_inputs_values[param_name] = corrected_value
            logger.debug(f"Manual input '{param_name}' updated to: {corrected_value}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Validation failed for {param_name} with value '{value}': {e}")
            sender.show_error(str(e))
            if param_name in self.manual_inputs_values:
                del self.manual_inputs_values[param_name]

    def _process_manual_data(self):
        try:
            res_params = {k: self.manual_inputs_values[k] for k in self.MANUAL_RES_DEFS if k in self.manual_inputs_values}
            if len(res_params) < len(self.MANUAL_RES_DEFS):
                 QMessageBox.warning(self, "Incomplete Data", "Please ensure all reservoir fields are valid."); return
            nx, ny, nz = res_params['nx'], res_params['ny'], res_params['nz']
            grid = {'PORO': np.full(nx*ny*nz, res_params['poro']), 'PERMX': np.full(nx*ny*nz, res_params['perm']),
                    'PERMY': np.full(nx*ny*nz, res_params['perm']), 'PERMZ': np.full(nx*ny*nz, res_params['perm'] * 0.1)}
            
            pvt_tables_dict = {}
            eos_model_data: Optional[EOSModelParameters] = None

            if self.use_detailed_pvt_checkbox.isChecked():
                if not self.detailed_pvt_data:
                    raise ValueError("Detailed PVT / EOS mode is selected, but no data has been entered.")
                
                pvt_tables_dict = {k: v for k, v in self.detailed_pvt_data.items() if k in ['PVTO', 'PVTG', 'PVTW'] and isinstance(v, np.ndarray) and v.size > 0}
                if 'eos_model' in self.detailed_pvt_data and self.detailed_pvt_data['eos_model']:
                    raw_eos = self.detailed_pvt_data['eos_model']
                    eos_model_data = EOSModelParameters(**raw_eos) if isinstance(raw_eos, dict) else raw_eos
                
                if 'PVTO' in pvt_tables_dict and pvt_tables_dict['PVTO'].shape[0] > 0:
                    pvto = pvt_tables_dict['PVTO']
                    self.pvt_properties = PVTProperties(
                        oil_fvf=pvto[:, 2], oil_viscosity=pvto[:, 3], rs=pvto[:, 1],
                        gas_fvf=pvt_tables_dict.get('PVTG', np.array([[0, 0.005, 0.02]]))[:,1], 
                        gas_viscosity=pvt_tables_dict.get('PVTG', np.array([[0, 0.005, 0.02]]))[:,2], 
                        pvt_type='compositional' if eos_model_data else 'black_oil',
                        gas_specific_gravity=self.detailed_pvt_data.get('gas_specific_gravity', 0.7),
                        temperature=self.detailed_pvt_data.get('ref_temperature', 212.0)
                    )
            else: 
                pvt_params = {k: self.manual_inputs_values[k] for k in self.MANUAL_PVT_DEFS if k in self.manual_inputs_values}
                if len(pvt_params) < len(self.MANUAL_PVT_DEFS):
                     QMessageBox.warning(self, "Incomplete Data", "Please ensure all simplified fluid fields are valid."); return
                
                self.pvt_properties = PVTProperties(
                    oil_fvf=np.array([pvt_params['oil_fvf']]), oil_viscosity=np.array([pvt_params['oil_visc']]),
                    gas_fvf=np.array([0.005]), gas_viscosity=np.array([0.02]), 
                    rs=np.array([pvt_params['sol_gor']]), pvt_type='black_oil',
                    gas_specific_gravity=pvt_params['gas_specific_gravity'], temperature=pvt_params['temperature']
                )
                pvt_tables_dict = {'PVTO': np.array([[pvt_params['ref_pres'], pvt_params['sol_gor'], pvt_params['oil_fvf'], pvt_params['oil_visc']]])}

            self.reservoir_data = ReservoirData(grid=grid, pvt_tables=pvt_tables_dict, runspec={'DIMENSIONS': [nx, ny, nz]}, eos_model=eos_model_data)
            
            self.log_browser.setHtml("<b>Generated Project Data:</b>"); self._populate_review_ui(); self.confirm_btn.setEnabled(True)
        except Exception as e: QMessageBox.critical(self, "Generation Error", f"An error occurred: {e}")

    def _open_pvt_editor(self):
        if PVTEditorDialog.__name__ != "PVTEditorDialog":
            QMessageBox.critical(self, "Component Missing", "The PVT Editor Dialog could not be loaded.")
            return
        dialog = PVTEditorDialog(self.detailed_pvt_data, self)
        if dialog.exec():
            self.detailed_pvt_data = dialog.get_data()
            self.status_message_updated.emit("Detailed PVT data updated.", 3000)

    def _add_well_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select LAS File", "", "LAS Files (*.las)");
        if not filepath: return
        try:
            well_data = parse_las(filepath)
            if well_data: self._add_well_to_ui(well_data); self.log_browser.append(f"Loaded well '{well_data.name}' from {Path(filepath).name}")
            else: raise ValueError("LAS parser returned no data.")
        except Exception as e: QMessageBox.warning(self, "LAS Parse Error", f"Could not load file: {Path(filepath).name}\n\nError: {e}")

    def _add_well_manually(self):
        if ManualWellDialog.__name__ == "ManualWellDialog":
            dialog = ManualWellDialog(self)
            if dialog.exec():
                well_data = dialog.get_well_data()
                if well_data: self._add_well_to_ui(well_data); self.log_browser.append(f"Manually added well '{well_data.name}'.")
        else: QMessageBox.critical(self, "Component Missing", "The Manual Well Dialog could not be loaded.")

    def _add_well_to_ui(self, well_data: WellData):
        self.well_data_list.append(well_data); display_text = f"✅ {well_data.name}"; metadata_parts = []
        if 'API' in well_data.metadata: metadata_parts.append(f"API: {well_data.metadata['API']:.1f}")
        if 'Temperature' in well_data.metadata: metadata_parts.append(f"T: {well_data.metadata['Temperature']:.0f}°F")
        if metadata_parts: display_text += f" ({', '.join(metadata_parts)})"
        item = QListWidgetItem(display_text); item.setData(Qt.ItemDataRole.UserRole, well_data)
        tooltip_text = f"Well: {well_data.name}\nDepth Points: {len(well_data.depths)}\n"
        log_keys = [k for k in well_data.properties.keys() if k != 'DEPT']
        tooltip_text += f"Log Curves: {', '.join(log_keys) if log_keys else 'None'}\nMetadata:\n"
        if well_data.metadata:
            for k, v in well_data.metadata.items(): tooltip_text += f"  - {k}: {v}\n"
        else: tooltip_text += "  None\n"
        item.setToolTip(tooltip_text); self.well_list_widget.addItem(item)
    
    def _remove_selected_well(self):
        for item in self.well_list_widget.selectedItems():
            well_to_remove = item.data(Qt.ItemDataRole.UserRole)
            self.well_data_list = [w for w in self.well_data_list if w is not well_to_remove]
            self.well_list_widget.takeItem(self.well_list_widget.row(item))

    def _view_well_data(self):
        selected_items = self.well_list_widget.selectedItems()
        if not selected_items: QMessageBox.information(self, "No Well Selected", "Please select a well from the list to view."); return
        selected_well = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not selected_well.properties: QMessageBox.information(self, "No Log Data", f"Well '{selected_well.name}' has no detailed log curves to display."); return
        if LogViewerDialog.__name__ == "LogViewerDialog": dialog = LogViewerDialog([selected_well], self); dialog.exec()
        else: QMessageBox.critical(self, "Component Missing", "The Log Viewer dialog could not be loaded.")

    def _load_reservoir_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Reservoir File", "", "DATA/INC Files (*.data *.inc)");
        if filepath: self.res_path_edit.setText(filepath)

    def _populate_review_ui(self):
        """Generates a comprehensive HTML report of the loaded project data."""
        if not self.reservoir_data or not self.pvt_properties:
            self.res_details.setText("<i>Generate data to see a detailed review.</i>")
            return

        res = self.reservoir_data
        pvt = self.pvt_properties
        
        def get_stats(arr: Optional[np.ndarray], precision: int = 3, is_int: bool = False) -> str:
            if arr is None or arr.size == 0: return "N/A"
            fmt = f"{{:.{precision}f}}" if not is_int else "{:d}"
            try:
                valid_arr = arr[~np.isnan(arr)]
                if valid_arr.size == 0: return "N/A (all empty)"
                return (f"Min: {fmt.format(np.min(valid_arr))}, "
                        f"Max: {fmt.format(np.max(valid_arr))}, "
                        f"Avg: {fmt.format(np.mean(valid_arr))}")
            except Exception: return "Calculation Error"

        # --- I. General Overview ---
        html = "<details open><summary><b>General Overview</b></summary>"
        html += "<table>"
        html += f"<tr><th>Timestamp</th><td>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>"
        html += f"<tr><th>Data Source</th><td>Manual Entry / In-App Generation</td></tr>"
        dims = res.runspec.get('DIMENSIONS', ['N/A'])
        total_cells = np.prod(dims) if 'N/A' not in dims else 'N/A'
        html += f"<tr><th>Data Points (Grid Cells)</th><td>{total_cells}</td></tr>"
        html += "</table></details>"

        # --- II. Reservoir Information ---
        dims_str = f"{dims[0]} x {dims[1]} x {dims[2]}" if len(dims) == 3 else "N/A"
        html += "<details open><summary><b>Reservoir Information</b></summary>"
        html += "<table>"
        html += f"<tr><th>Grid Dimensions (NX, NY, NZ)</th><td>{dims_str}</td></tr>"
        html += f"<tr><th>Reservoir Type</th><td>Sandstone (Assumed)</td></tr>"
        html += f"<tr><th>Initial Water Saturation</th><td>0.2 (Assumed)</td></tr>"
        html += f"<tr><th>Porosity Range</th><td>{get_stats(res.grid.get('PORO'), 3)}</td></tr>"
        html += f"<tr><th>Permeability X (mD)</th><td>{get_stats(res.grid.get('PERMX'), 1)}</td></tr>"
        html += f"<tr><th>Permeability Y (mD)</th><td>{get_stats(res.grid.get('PERMY'), 1)}</td></tr>"
        html += f"<tr><th>Permeability Z (mD)</th><td>{get_stats(res.grid.get('PERMZ'), 2)}</td></tr>"
        html += "</table></details>"

        # --- III. Well Information ---
        html += f"<details><summary><b>Well Information ({len(self.well_data_list)} loaded)</b></summary>"
        if not self.well_data_list:
            html += "<p>No wells have been loaded.</p>"
        else:
            html += "<table><tr><th>Well Name</th><th>Log Curves</th><th>Metadata Keys</th></tr>"
            for well in self.well_data_list:
                log_keys = ", ".join(k for k in well.properties.keys() if k != 'DEPT') or "<i>None</i>"
                meta_keys = ", ".join(well.metadata.keys()) or "<i>None</i>"
                html += f"<tr><td>{well.name}</td><td>{log_keys}</td><td>{meta_keys}</td></tr>"
            html += "</table>"
        html += "</details>"

        # --- IV. PVT Model Details ---
        html += "<details open><summary><b>PVT Model Details</b></summary>"
        html += "<table>"
        html += f"<tr><th>Model Type</th><td><b>{pvt.pvt_type.replace('_', ' ').title()}</b></td></tr>"
        html += f"<tr><th>Reservoir Temperature</th><td>{pvt.temperature:.1f} °F</td></tr>"
        html += f"<tr><th>Gas Specific Gravity</th><td>{pvt.gas_specific_gravity:.3f}</td></tr>"
        html += f"<tr><th>Number of PVT Table Points</th><td>{pvt.rs.size}</td></tr>"
        html += f"<tr><th>Solution GOR (Rs) Range</th><td>{get_stats(pvt.rs, 1)} scf/STB</td></tr>"
        html += f"<tr><th>Oil FVF (Bo) Range</th><td>{get_stats(pvt.oil_fvf, 3)} RB/STB</td></tr>"
        html += f"<tr><th>Oil Viscosity Range</th><td>{get_stats(pvt.oil_viscosity, 3)} cP</td></tr>"
        html += "</table>"
        
        if res.eos_model:
            eos = res.eos_model
            html += "<h4 style='margin-top:10px; margin-bottom: 2px;'>Equation of State (EOS) Model</h4>"
            html += "<table>"
            html += f"<tr><th>EOS Type</th><td>{eos.eos_type}</td></tr>"
            html += f"<tr><th>Component Count</th><td>{len(eos.component_properties)}</td></tr>"
            html += "<tr><th>Components (zi)</th><td>"
            comp_list = [f"{comp[0]} ({float(comp[1]):.3f})" for comp in eos.component_properties]
            html += ", ".join(comp_list)
            html += "</td></tr>"
            html += "</table>"
        html += "</details>"

        # --- V. Quality Control ---
        missing_poro = np.sum(np.isnan(res.grid.get('PORO', np.array([]))))
        missing_perm = np.sum(np.isnan(res.grid.get('PERMX', np.array([]))))
        wells_with_no_logs = sum(1 for w in self.well_data_list if not w.properties)
        
        score = 0
        if missing_poro == 0 and missing_perm == 0: score += 1
        if len(self.well_data_list) > 0: score += 1
        if wells_with_no_logs == 0 and len(self.well_data_list) > 0: score += 1
        if pvt.pvt_type != 'black_oil' or pvt.rs.size > 1: score += 1

        if score >= 4: confidence, color = "Excellent", "green"
        elif score == 3: confidence, color = "Good", "blue"
        elif score >= 1: confidence, color = "Fair", "orange"
        else: confidence, color = "Poor", "red"

        html += "<details><summary><b>Data Quality Metrics</b></summary>"
        html += "<table>"
        html += f"<tr><th>Missing Porosity Values</th><td>{missing_poro}</td></tr>"
        html += f"<tr><th>Missing Permeability Values</th><td>{missing_perm}</td></tr>"
        html += f"<tr><th>Wells Without Log Curves</th><td>{wells_with_no_logs}</td></tr>"
        html += f"<tr><th>Data Confidence Score</th><td style='color:{color}; font-weight:bold;'>{confidence}</td></tr>"
        html += "</table></details>"

        self.res_details.setHtml(html)
        
    def _confirm_and_emit_data(self):
        if not self.reservoir_data or not self.pvt_properties:
            QMessageBox.warning(self, "No Data", "Reservoir and PVT properties must be generated before confirming."); return
        
        payload = {
            "well_data_list": deepcopy(self.well_data_list), 
            "reservoir_data": deepcopy(self.reservoir_data),
            "pvt_properties": deepcopy(self.pvt_properties)
        }
        self.project_data_updated.emit(payload); self.status_message_updated.emit("Project data confirmed.", 5000)
        logger.info("Project data confirmed and emitted.")