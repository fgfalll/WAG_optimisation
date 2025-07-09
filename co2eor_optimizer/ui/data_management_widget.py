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
        
        self.setStyleSheet("""
            QTextBrowser {
                font-family: 'Segoe UI', sans-serif; font-size: 10pt;
                background-color: #fdfdfd; border: 1px solid #dee2e6; border-radius: 4px;
            }
            details {
                border: 1px solid #d0d7de; border-radius: 6px; padding: .5em .5em 0;
                margin-bottom: 1em; background-color: #ffffff;
            }
            summary {
                font-weight: bold; font-size: 11pt; margin: -.5em -.5em 0;
                padding: .5em; cursor: pointer;
            }
            details[open] { padding: .5em; }
            details[open] summary { border-bottom: 1px solid #d0d7de; margin-bottom: .5em; }
            table { width: 100%; border-collapse: collapse; margin-top: 5px; }
            th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #e1e4e8; }
            th { background-color: #f6f8fa; font-weight: 600; }
            tr:last-child > td { border-bottom: none; }
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
        for i, (name, (label, w_type, p_type, kwargs)) in enumerate(self.MANUAL_PVT_DEFS.items()):
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            simple_pvt_layout.addWidget(input_group, i // 2, i % 2)
        pvt_main_layout.addWidget(self.simple_pvt_group)
        detailed_pvt_layout = QHBoxLayout()
        self.use_detailed_pvt_checkbox = QCheckBox("Use Detailed PVT / EOS Model")
        self.edit_pvt_btn = QPushButton(QIcon.fromTheme("document-edit"), "Edit Detailed PVT...")
        self.edit_pvt_btn.setEnabled(False)
        detailed_pvt_layout.addWidget(self.use_detailed_pvt_checkbox); detailed_pvt_layout.addWidget(self.edit_pvt_btn); detailed_pvt_layout.addStretch()
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
        self.well_list_widget.addAction(remove_action)
        self.edit_pvt_btn.clicked.connect(self._open_pvt_editor)
        self.use_detailed_pvt_checkbox.toggled.connect(self._toggle_pvt_mode)

    def _toggle_pvt_mode(self, checked: bool):
        self.simple_pvt_group.setEnabled(not checked)
        self.edit_pvt_btn.setEnabled(checked)
        
    def clear_all_project_data(self):
        self.well_data_list.clear(); self.well_list_widget.clear()
        self.reservoir_data = None; self.pvt_properties = None; self.detailed_pvt_data = None
        if hasattr(self, 'use_detailed_pvt_checkbox'): self.use_detailed_pvt_checkbox.setChecked(False)
        self.log_browser.clear(); self.res_details.clear(); self.confirm_btn.setEnabled(False)
        self.log_browser.setHtml("<i>Define reservoir properties and add well data to begin.</i>")
        self.manual_inputs_values.clear()
        all_defs = {**self.MANUAL_RES_DEFS, **self.MANUAL_PVT_DEFS}
        for name, widget in self.manual_inputs_widgets.items():
            param_def = all_defs[name]
            default_value = param_def[3].get('default_value')
            widget.clear_error(); widget.set_value(default_value)
            if default_value is not None:
                self.manual_inputs_values[name] = self._coerce_value(default_value, param_def[2])
            
    def _coerce_value(self, value: Any, type_hint: Type) -> Any:
        try:
            if isinstance(value, str): value = value.replace(',', '.')
            origin = get_origin(type_hint)
            is_optional = origin in (Union, UnionType) and type(None) in get_args(type_hint)
            base_type = next((t for t in get_args(type_hint) if t is not type(None)), type_hint) if is_optional else type_hint
            
            if value is None or (isinstance(value, str) and not value.strip()):
                if is_optional: return None
                raise ValueError("This field cannot be empty.")
            
            if base_type is float: return float(value)
            if base_type is int: return int(float(value))
        except (ValueError, TypeError): raise ValueError(f"Must be a valid number.")
        return value

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return
        param_name = sender.param_name; param_type = sender.property("param_type")
        try:
            sender.clear_error()
            corrected_value = self._coerce_value(value, param_type)
            self.manual_inputs_values[param_name] = corrected_value
        except (ValueError, TypeError) as e:
            sender.show_error(str(e))
            if param_name in self.manual_inputs_values: del self.manual_inputs_values[param_name]

    def _process_manual_data(self):
        try:
            # --- [ENHANCED] Validation check for required fields based on PVT mode ---
            required_res_keys = set(self.MANUAL_RES_DEFS.keys())
            required_pvt_keys = set() if self.use_detailed_pvt_checkbox.isChecked() else set(self.MANUAL_PVT_DEFS.keys())
            if not required_res_keys.issubset(self.manual_inputs_values.keys()) or not required_pvt_keys.issubset(self.manual_inputs_values.keys()):
                QMessageBox.warning(self, "Incomplete Data", "Please ensure all required fields are filled and valid before generating data."); return
            
            res_params = {k: self.manual_inputs_values[k] for k in self.MANUAL_RES_DEFS}
            nx, ny, nz = res_params['nx'], res_params['ny'], res_params['nz']
            grid = {'PORO': np.full(nx*ny*nz, res_params['poro']), 'PERMX': np.full(nx*ny*nz, res_params['perm'])}
            
            eos_model_data = None
            # --- [REFACTORED] Logic to handle both black-oil and compositional PVT data ---
            if self.use_detailed_pvt_checkbox.isChecked():
                if not self.detailed_pvt_data: 
                    raise ValueError("Detailed PVT/EOS is selected, but no model data has been defined. Please use the 'Edit' button.")
                # The detailed_pvt_data from PVTEditorDialog should be an EOSModelParameters instance
                eos_model_data = self.detailed_pvt_data
                # Create a placeholder PVTProperties for compatibility, but mark as compositional
                self.pvt_properties = PVTProperties(oil_fvf=np.array([1.2]), oil_viscosity=np.array([1.0]), gas_fvf=np.array([0.005]), gas_viscosity=np.array([0.02]), rs=np.array([500]), pvt_type='compositional', gas_specific_gravity=0.7, temperature=212.0)
                pvt_tables_dict = {} # No PVTO table for compositional
                logger.info("Assembling compositional model data.")
            else:
                pvt_params = {k: self.manual_inputs_values[k] for k in self.MANUAL_PVT_DEFS}
                self.pvt_properties = PVTProperties(oil_fvf=np.array([pvt_params['oil_fvf']]), oil_viscosity=np.array([pvt_params['oil_visc']]), gas_fvf=np.array([0.005]), gas_viscosity=np.array([0.02]), rs=np.array([pvt_params['sol_gor']]), pvt_type='black_oil', gas_specific_gravity=pvt_params['gas_specific_gravity'], temperature=pvt_params['temperature'])
                pvt_tables_dict = {'PVTO': np.array([[pvt_params['ref_pres'], pvt_params['sol_gor'], pvt_params['oil_fvf'], pvt_params['oil_visc']]])}
                logger.info("Assembling black-oil model data.")

            self.reservoir_data = ReservoirData(grid=grid, pvt_tables=pvt_tables_dict, runspec={'DIMENSIONS': [nx, ny, nz]}, eos_model=eos_model_data)
            
            self.log_browser.setHtml("<b>Generated Project Data:</b>"); self._populate_review_ui(); self.confirm_btn.setEnabled(True)
        except Exception as e:
             QMessageBox.critical(self, "Generation Error", f"An error occurred while processing manual data:\n\n{e}")
             logger.error(f"Error processing manual data: {e}", exc_info=True)


    def _open_pvt_editor(self):
        # --- [ENHANCED] This dialog now returns a fully formed EOSModelParameters object
        dialog = PVTEditorDialog(self.detailed_pvt_data, self)
        if dialog.exec(): 
            self.detailed_pvt_data = dialog.get_data()
            if isinstance(self.detailed_pvt_data, EOSModelParameters):
                self.log_browser.append(f"Successfully configured '{self.detailed_pvt_data.eos_type}' model.")
            else:
                 self.log_browser.append("<i style='color:orange;'>Detailed PVT data saved, but it's not a complete EOS model.</i>")


    def _add_well_from_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select LAS File", "", "LAS Files (*.las)");
        if not filepath: return
        try:
            well_data = parse_las(filepath)
            if well_data: self._add_well_to_ui(well_data)
        except Exception as e: QMessageBox.warning(self, "LAS Parse Error", f"Could not load file: {e}")

    def _add_well_manually(self):
        dialog = ManualWellDialog(self)
        if dialog.exec():
            well_data = dialog.get_well_data()
            if well_data: self._add_well_to_ui(well_data)

    def _add_well_to_ui(self, well_data: WellData):
        self.well_data_list.append(well_data)
        display_text = f"✅ {well_data.name}"
        item = QListWidgetItem(display_text); item.setData(Qt.ItemDataRole.UserRole, well_data)
        
        # --- [ENHANCED] Richer tooltip for better data overview ---
        tooltip_text = f"Well: {well_data.name}\nDepth Range: {well_data.depths[0]:.1f} - {well_data.depths[-1]:.1f} ft\n"
        log_keys = [k for k in well_data.properties.keys() if k != 'DEPT']
        tooltip_text += f"Log Curves: {', '.join(log_keys) if log_keys else 'None'}\n"
        
        if well_data.perforation_properties:
            perf_str = "\n".join([f"  - {p['top']:.1f} to {p['bottom']:.1f} ft" for p in well_data.perforation_properties])
            tooltip_text += f"Perforations:\n{perf_str}\n"
        else:
            tooltip_text += "Perforations: None specified (entire wellbore)\n"
        
        if well_data.metadata:
            tooltip_text += "Global Properties:\n"
            if 'API' in well_data.metadata:
                tooltip_text += f"  - API: {well_data.metadata['API']:.1f}°\n"
            if 'Temperature' in well_data.metadata:
                 tooltip_text += f"  - Temp: {well_data.metadata['Temperature']:.1f}°F\n"

        item.setToolTip(tooltip_text)
        self.well_list_widget.addItem(item)
    
    def _remove_selected_well(self):
        for item in self.well_list_widget.selectedItems():
            self.well_data_list.remove(item.data(Qt.ItemDataRole.UserRole))
            self.well_list_widget.takeItem(self.well_list_widget.row(item))

    def _view_well_data(self):
        selected_items = self.well_list_widget.selectedItems()
        if not selected_items: return
        dialog = LogViewerDialog([selected_items[0].data(Qt.ItemDataRole.UserRole)], self); dialog.exec()

    def _load_reservoir_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Reservoir File", "", "DATA/INC Files (*.data *.inc)");
        if filepath: self.res_path_edit.setText(filepath)

    def _populate_review_ui(self):
        if not self.reservoir_data or not self.pvt_properties:
            self.res_details.setText("<i>Generate data to see a detailed review.</i>"); return
        res, pvt = self.reservoir_data, self.pvt_properties
        
        # --- [REFACTORED] Significantly improved HTML review panel ---
        html = f"""
        <details open><summary><b>General & Reservoir Info</b></summary>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Grid Dimensions (NX, NY, NZ)</td><td>{res.runspec['DIMENSIONS'][0]} x {res.runspec['DIMENSIONS'][1]} x {res.runspec['DIMENSIONS'][2]}</td></tr>
            <tr><td>Average Porosity</td><td>{np.mean(res.grid['PORO']):.3f}</td></tr>
            <tr><td>Average Permeability (mD)</td><td>{np.mean(res.grid['PERMX']):.1f}</td></tr>
        </table>
        </details>
        """

        html += f"<details open><summary><b>Well Information ({len(self.well_data_list)} loaded)</b></summary>"
        if not self.well_data_list:
            html += "<p>No wells have been loaded.</p>"
        else:
            html += "<table><tr><th>Well Name</th><th>Depth Range (ft)</th><th>Perforated Intervals (ft)</th><th>Interval Properties (API/Temp)</th><th>Log Curves</th></tr>"
            for well in self.well_data_list:
                log_keys = ", ".join(k for k in well.properties.keys() if k != 'DEPT') or "<i>None</i>"
                depth_range = f"{well.depths[0]:.1f} - {well.depths[-1]:.1f}"
                
                if well.perforation_properties:
                    perf_str = "<br>".join([f"{p['top']:.1f} - {p['bottom']:.1f}" for p in well.perforation_properties])
                    prop_str = "<br>".join([f"{p['api']:.1f}° / {p['temp']:.1f}°F" for p in well.perforation_properties])
                else:
                    perf_str = "<i>Entire wellbore</i>"
                    global_api = well.metadata.get('API', 'N/A')
                    global_temp = well.metadata.get('Temperature', 'N/A')
                    prop_str = f"<i>{global_api}° / {global_temp}°F (Global)</i>"
                    
                html += f"<tr><td>{well.name}</td><td>{depth_range}</td><td>{perf_str}</td><td>{prop_str}</td><td>{log_keys}</td></tr>"
            html += "</table>"
        html += "</details>"
        
        html += f"""
        <details open><summary><b>PVT & Fluid Info ({pvt.pvt_type.replace('_',' ').title()})</b></summary>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Reservoir Temperature</td><td>{pvt.temperature:.1f} °F</td></tr>
            <tr><td>Gas Specific Gravity</td><td>{pvt.gas_specific_gravity:.2f} (air=1)</td></tr>
        """
        if pvt.pvt_type == 'black_oil':
            pvt_table = res.pvt_tables.get('PVTO', [[0,0,0,0]])[0]
            html += f"""
            <tr><td>Reference Pressure</td><td>{pvt_table[0]:.0f} psia</td></tr>
            <tr><td>Solution GOR (Rs)</td><td>{pvt_table[1]:.1f} scf/STB</td></tr>
            <tr><td>Oil FVF (Bo)</td><td>{pvt_table[2]:.3f} RB/STB</td></tr>
            <tr><td>Oil Viscosity</td><td>{pvt_table[3]:.3f} cP</td></tr>
            """
        else: # Compositional
             html += "<tr><td colspan='2'><i>Detailed EOS model is in use. See below.</i></td></tr>"
        html += "</table>"
        if res.eos_model:
            eos = res.eos_model
            html += f"<h6>EOS Model: {eos.eos_type} ({len(eos.component_properties)} components)</h6>"
        html += "</details>"

        self.res_details.setHtml(html)
        
    def _confirm_and_emit_data(self):
        if not self.reservoir_data or not self.pvt_properties: return
        payload = {
            "well_data_list": deepcopy(self.well_data_list), 
            "reservoir_data": deepcopy(self.reservoir_data),
            "pvt_properties": deepcopy(self.pvt_properties)
        }
        self.project_data_updated.emit(payload); self.status_message_updated.emit("Project data confirmed.", 5000)