import logging
from typing import Optional, Any, Dict, List, Type, get_origin, get_args, Union
from types import UnionType
import numpy as np
from copy import deepcopy

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QTabWidget, QFileDialog, QMessageBox, QLineEdit,
    QDialog, QListWidget, QListWidgetItem,
    QSizePolicy, QCheckBox, QTableWidget, QTableWidgetItem,
    QRadioButton, QHeaderView, QApplication, QSplitter, QComboBox, QFormLayout
)
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import pyqtSignal, Qt, QLocale, QEvent

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
from PyQt6.QtWebEngineWidgets import QWebEngineView

try:
    from utils.preferences_manager import PreferencesManager
    from config_manager import ConfigManager
except ImportError:
    PreferencesManager = None
    ConfigManager = None
    logging.critical("DataManagementWidget: PreferencesManager or ConfigManager not found. Unit system preferences will not work.")

try:
    from .widgets.parameter_input_group import ParameterInputGroup
    from .widgets.pvt_editor_dialog import PVTEditorDialog
    from .widgets.log_viewer_dialog import WellViewerDialog
    from .widgets.manual_well_dialog import ManualWellDialog
except ImportError as e:
    class ParameterInputGroup(QWidget):
        def set_label_text(self, text: str): pass
        def get_value(self): return None
        def set_value(self, value): pass
        def clear_error(self): pass
        def show_error(self, msg: str): pass
    class PVTEditorDialog(QDialog): pass
    class LogViewerDialog(QDialog): pass
    class ManualWellDialog(QDialog): pass
    logging.critical(f"DataManagementWidget: Failed to import critical UI components: {e}")

from core.data_models import WellData, ReservoirData, EOSModelParameters, PVTProperties, GeostatisticalParams, LayerDefinition
from parsers.las_parser import parse_las, MissingWellNameError

# Data integration and engine factory for engine compatibility
try:
    from core.data_integration_engine import DataIntegrationEngine
    from core.engine_factory import EngineFactory
    DATA_INTEGRATION_AVAILABLE = True
    ENGINE_FACTORY_AVAILABLE = True
except ImportError:
    DATA_INTEGRATION_AVAILABLE = False
    ENGINE_FACTORY_AVAILABLE = False
    logging.warning("DataIntegrationEngine or EngineFactory not available. Engine integration will be limited.")


logger = logging.getLogger(__name__)


class DataManagementWidget(QWidget):
    project_data_updated = pyqtSignal(dict)
    status_message_updated = pyqtSignal(str, int)

    MANUAL_RES_DEFS = {
        'nx': ("NX", "lineedit", int, {'default_value': 50}),
        'ny': ("NY", "lineedit", int, {'default_value': 50}),
        'nz': ("NZ", "lineedit", int, {'default_value': 10}),
        'poro': ("Porosity (φ)", "lineedit", float, {'default_value': 0.20, 'decimals': 3}),
        'perm': ("Perm ({unit})", "lineedit", float, {'default_value': 100.0}),
        'area': ("Area ({unit})", "lineedit", float, {'default_value': 1000.0}),
        'thickness': ("Net Pay ({unit})", "lineedit", float, {'default_value': 50.0}),
        'swi': ("Initial Water Sat. (Swi)", "lineedit", float, {'default_value': 0.25, 'decimals': 3}),
        'boi': ("Initial Oil FVF (Boi)", "lineedit", float, {'default_value': 1.2, 'decimals': 3}),
        'ooip_stb': ("Direct OOIP ({unit})", "lineedit", float, {'default_value': 10000000.0}),
        'length': ("Reservoir Length ({unit})", "lineedit", float, {'default_value': 2000.0}),
        'dip_angle': ("Dip Angle (°)", "lineedit", float, {'default_value': 0.0}),
        'density_contrast': ("Density Contrast (g/cm³)", "lineedit", float, {'default_value': 0.3, 'decimals': 3}),
        'interfacial_tension': ("Interfacial Tension (dynes/cm)", "lineedit", float, {'default_value': 5.0}),
        'rock_compressibility': ("Rock Compressibility (1/psi)", "lineedit", float, {'default_value': 3e-6, 'decimals': 8}),
        'kv_kh_ratio': ("Kv/Kh Ratio", "lineedit", float, {'default_value': 0.1, 'decimals': 3}),
        's_gc': ("Critical Gas Saturation (Sgc)", "lineedit", float, {'default_value': 0.05, 'decimals': 3}),
        'n_o': ("Corey Exponent - Oil (No)", "lineedit", float, {'default_value': 2.0, 'decimals': 2}),
        'n_g': ("Corey Exponent - Gas (Ng)", "lineedit", float, {'default_value': 2.0, 'decimals': 2}),
        's_wc': ("Connate Water Saturation (Swc)", "lineedit", float, {'default_value': 0.2, 'decimals': 3}),
        's_orw': ("Residual Oil Sat. - Water (Sorw)", "lineedit", float, {'default_value': 0.2, 'decimals': 3}),
        'n_w': ("Corey Exponent - Water (Nw)", "lineedit", float, {'default_value': 2.0, 'decimals': 2}),
        'n_ow': ("Corey Exponent - Oil in Water (Now)", "lineedit", float, {'default_value': 2.0, 'decimals': 2}),
        
        'rock_type': ("Rock Type", "combobox", str, {
            'default_value': 'sandstone',
            'items': ['sandstone', 'carbonate', 'shale', 'dolomite', 'limestone', 'other']
        }),
        'depositional_environment': ("Depositional Environment", "combobox", str, {
            'default_value': 'fluvial',
            'items': ['fluvial', 'deltaic', 'aeolian', 'shallow_marine', 'deep_marine', 'lacustrine', 'other']
        }),
        'structural_complexity': ("Structural Complexity", "combobox", str, {
            'default_value': 'simple',
            'items': ['simple', 'moderate', 'complex', 'very_complex']
        }),
    }

    MANUAL_PVT_DEFS = {
        'temperature': ("Reservoir Temp ({unit})", "lineedit", float, {'default_value': 212.0}),
        'initial_pressure': ("Initial Pressure ({unit})", "lineedit", float, {'default_value': 4000.0}),
        'api_gravity': ("Oil API Gravity", "lineedit", float, {'default_value': 35.0}),
        'gas_specific_gravity': ("Gas Gravity (air=1)", "lineedit", float, {'default_value': 0.7}),
        'ref_pres': ("Ref. Pressure ({unit})", "lineedit", float, {'default_value': 3000.0}),
        'sol_gor': ("Solution GOR (Rs, {unit})", "lineedit", float, {'default_value': 500.0}),
        'oil_fvf_simple': ("Oil FVF (Bo)", "lineedit", float, {'default_value': 1.2, 'decimals': 3}),
        'oil_viscosity_cp': ("Oil Viscosity (cp)", "lineedit", float, {'default_value': 1.0, 'decimals': 3}),
        'water_viscosity_cp': ("Water Viscosity (cp)", "lineedit", float, {'default_value': 0.5, 'decimals': 3}),
        'gas_viscosity_cp': ("Gas Viscosity (cp)", "lineedit", float, {'default_value': 0.02, 'decimals': 3}),
        'water_fvf': ("Water FVF (Bw)", "lineedit", float, {'default_value': 1.0, 'decimals': 3}),
        'gas_fvf_simple': ("Gas FVF (Bg)", "lineedit", float, {'default_value': 0.01, 'decimals': 4}),
        'c7_plus_fraction': ("C7+ Fraction", "lineedit", float, {'default_value': 0.35, 'decimals': 3}),
        'co2_solubility_scm_per_bbl': ("CO2 Solubility (scm/bbl)", "lineedit", float, {'default_value': 200.0}),
    }


    SURROGATE_TUNING_DEFS = {
        # Miscibility transition parameters
        'alpha_base': ("Alpha Base (P/MMP midpoint)", "lineedit", float, {'default_value': 1.0, 'min': 0.5, 'max': 1.5, 'decimals': 3}),
        'miscibility_window': ("Miscibility Window (ΔP/MMP)", "lineedit", float, {'default_value': 0.10, 'min': 0.01, 'max': 0.5, 'decimals': 3}),

        # Production dynamics
        'breakthrough_time_years': ("CO2 Breakthrough Time (years)", "lineedit", float, {'default_value': 1.5, 'min': 0.1, 'max': 10.0}),
        'trapping_efficiency': ("CO2 Trapping Efficiency", "lineedit", float, {'default_value': 0.4, 'min': 0.0, 'max': 1.0, 'decimals': 2}),

        # Initial conditions
        'initial_gor_scf_per_stb': ("Initial GOR (scf/STB)", "lineedit", float, {'default_value': 500.0, 'min': 0.0}),

        # Mobility and mixing
        'transverse_mixing_calibration': ("Transverse Mixing Calibration", "lineedit", float, {'default_value': 0.5, 'min': 0.0, 'max': 1.0, 'decimals': 2}),
        'omega_tl': ("Todd-Longstaff Omega", "lineedit", float, {'default_value': 0.6, 'min': 0.0, 'max': 1.0, 'decimals': 2}),

        # Relative permeability endpoints (Corey parameters)
        'k_ro_0': ("Oil Rel Perm Endpoint (k_ro0)", "lineedit", float, {'default_value': 0.8, 'min': 0.1, 'max': 1.0, 'decimals': 2}),
        'k_rg_0': ("Gas Rel Perm Endpoint (k_rg0)", "lineedit", float, {'default_value': 0.3, 'min': 0.1, 'max': 1.0, 'decimals': 2}),
        'n_o': ("Corey Exponent Oil (n_o)", "lineedit", float, {'default_value': 2.0, 'min': 1.0, 'max': 5.0, 'decimals': 2}),
        'n_g': ("Corey Exponent Gas (n_g)", "lineedit", float, {'default_value': 2.0, 'min': 1.0, 'max': 5.0, 'decimals': 2}),
    }

    UNIT_CATEGORY_MAP = {
        'perm': 'permeability',
        'area': 'area',
        'thickness': 'length',
        'ooip_stb': 'volume',
        'temperature': 'temperature',
        'ref_pres': 'pressure',
        'initial_pressure': 'pressure',
        'sol_gor': 'gor',
        'oil_fvf': 'fvf',
        'oil_visc': 'viscosity',
        'gas_fvf': 'fvf',
        'co2_visc': 'viscosity'
    }

    def __init__(self, parent: Optional[QWidget] = None, preferences_manager: Optional[PreferencesManager] = None, config_manager: Optional[ConfigManager] = None):
        super().__init__(parent)
        self.well_data_list: List[WellData] = []
        self.reservoir_data: Optional[ReservoirData] = None
        self.pvt_properties: Optional[PVTProperties] = None
        self.detailed_pvt_data: Optional[Dict[str, Any]] = None

        self.preferences_manager = preferences_manager
        self.config_manager = config_manager
        if self.preferences_manager is None and parent is not None:
            self.preferences_manager = getattr(parent, 'preferences_manager', None)

        # Initialize data integration engine for engine compatibility
        self.data_integration_engine = DataIntegrationEngine() if DATA_INTEGRATION_AVAILABLE else None

        self.manual_inputs_widgets: Dict[str, ParameterInputGroup] = {}
        self.manual_inputs_values: Dict[str, Any] = {}
        self._is_3d_view = False


        self._setup_ui()
        self._connect_signals()
        self.retranslateUi()
        self.clear_all_project_data()

        self.ooip_calc_radio.setChecked(True)
        self._toggle_ooip_mode()
        self._calculate_and_display_ooip()
        self._update_calculated_eor_params()

        self.use_detailed_pvt_checkbox.setChecked(False)
        self._toggle_detailed_pvt_button(False)

        if self.preferences_manager:
            self.preferences_manager.display_preferences_changed.connect(self._on_preferences_changed)
            self.preferences_manager.units_preferences_changed.connect(self._on_preferences_changed)

        self.setStyleSheet("...")

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def _connect_signals(self):
        self.generate_data_btn.clicked.connect(self._process_manual_data)
        self.ooip_calc_radio.toggled.connect(self._toggle_ooip_mode)
        self.ooip_direct_radio.toggled.connect(self._toggle_ooip_mode)

        self.use_layered_model_checkbox.toggled.connect(self._toggle_layered_model)
        self.use_detailed_pvt_checkbox.toggled.connect(self._toggle_detailed_pvt_button)

        self.add_layer_btn.clicked.connect(self._add_layer_row)
        self.remove_layer_btn.clicked.connect(self._remove_selected_layer)
        self.plot_rel_perm_btn.clicked.connect(self._plot_rel_perm_curves)

        self.detailed_pvt_btn.clicked.connect(self._open_pvt_editor)

        self.add_well_btn.clicked.connect(self._add_well_manually)
        self.remove_well_btn.clicked.connect(self._remove_selected_well)
        self.view_well_btn.clicked.connect(self._view_selected_well)



    def _toggle_detailed_pvt_button(self, checked):
        self.detailed_pvt_btn.setEnabled(checked)

    def retranslateUi(self):
        self.main_tab_widget.setTabText(0, self.tr("Reservoir"))
        self.main_tab_widget.setTabText(1, self.tr("PVT"))
        self.main_tab_widget.setTabText(2, self.tr("Wells"))
        self.main_tab_widget.setTabText(3, self.tr("Surrogate Tuning"))

        self.generate_data_btn.setText(self.tr("Generate Project Data"))

        self.dims_group.setTitle(self.tr("Grid Dimensions"))
        self.ooip_group.setTitle(self.tr("OOIP Determination"))
        self.ooip_calc_radio.setText(self.tr("Calculate from Parameters"))
        self.ooip_direct_radio.setText(self.tr("Direct Input"))
        self.ooip_calc_params_group.setTitle(self.tr("Volumetric Parameters"))
        self.ooip_direct_input_group.setTitle(self.tr("Direct OOIP Value"))
        self.use_layered_model_checkbox.setText(self.tr("Use Layered Reservoir Model"))
        
        self.uniform_props_group.setTitle(self.tr("Uniform Properties"))
        self.layered_props_group.setTitle(self.tr("Layered Properties"))
        self.rel_perm_group.setTitle(self.tr("Relative Permeability"))
        self.plot_rel_perm_btn.setText(self.tr("Plot Curves"))
        self.calculated_params_group.setTitle(self.tr("Calculated Parameters"))

        self.use_detailed_pvt_checkbox.setText(self.tr("Use Detailed PVT Model"))
        self.detailed_pvt_btn.setText(self.tr("Edit Detailed PVT Data (PVTO, PVTG, etc.)"))

        self.wells_group.setTitle(self.tr("Well Data"))
        self.add_well_btn.setText(self.tr("Add Well"))
        self.remove_well_btn.setText(self.tr("Remove Well"))
        self.view_well_btn.setText(self.tr("View Well"))



        # Update parameter labels
        self._on_preferences_changed()


    def _toggle_ooip_mode(self):
        is_calc_mode = self.ooip_calc_radio.isChecked()
        self.ooip_calc_params_group.setVisible(is_calc_mode)
        self.ooip_direct_input_group.setVisible(not is_calc_mode)

    def _on_preferences_changed(self):
        if not self.preferences_manager:
            return

        for param_name, widget in self.manual_inputs_widgets.items():
            if param_name in self.UNIT_CATEGORY_MAP:
                category = self.UNIT_CATEGORY_MAP[param_name]
                unit = self.preferences_manager.get_display_unit(category)
                
                label_template, _, _, _ = {**self.MANUAL_RES_DEFS, **self.MANUAL_PVT_DEFS}[param_name]
                
                if "{unit}" in label_template:
                    widget.set_label_text(label_template.format(unit=unit))

        self._calculate_and_display_ooip()
        self._update_calculated_eor_params()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([480, 520])
        
        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.main_tab_widget = QTabWidget()
        
        self.reservoir_tab = self._create_reservoir_tab()
        self.pvt_tab = self._create_pvt_tab()
        self.wells_tab = self._create_wells_tab()
        self.surrogate_tuning_tab = self._create_surrogate_tuning_tab()

        self.main_tab_widget.addTab(self.reservoir_tab, QIcon.fromTheme("drive-harddisk"), "Reservoir")
        self.main_tab_widget.addTab(self.pvt_tab, QIcon.fromTheme("applications-science"), "PVT")
        self.main_tab_widget.addTab(self.wells_tab, QIcon.fromTheme("view-list-tree"), "Wells")
        self.main_tab_widget.addTab(self.surrogate_tuning_tab, QIcon.fromTheme("sliders"), "Surrogate Tuning")

        layout.addWidget(self.main_tab_widget)
        
        self.generate_data_btn = QPushButton(QIcon.fromTheme("go-jump"), "Generate Project Data")
        layout.addWidget(self.generate_data_btn)
        
        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.right_tab_widget = QTabWidget()
        
        # 2D Plot View
        self.plot_view_widget = QWidget()
        plot_view_layout = QVBoxLayout(self.plot_view_widget)
        self.plot_view = QWebEngineView()
        plot_view_layout.addWidget(self.plot_view)
        self.right_tab_widget.addTab(self.plot_view_widget, "2D Plot")
        
        # 3D View
        self.view_3d_widget = QWidget()
        view_3d_layout = QVBoxLayout(self.view_3d_widget)
        self.canvas_3d = FigureCanvas(plt.figure())
        self.ax_3d = self.canvas_3d.figure.add_subplot(111, projection='3d')
        view_3d_layout.addWidget(self.canvas_3d)
        self.right_tab_widget.addTab(self.view_3d_widget, "3D View")
        
        layout.addWidget(self.right_tab_widget)
        
        return panel

    def _create_reservoir_tab(self) -> QWidget:
        tab_panel = QWidget()
        res_main_layout = QVBoxLayout(tab_panel)
        
        self.dims_group = QGroupBox("Grid Dimensions")
        dims_layout = QHBoxLayout(self.dims_group)
        for name in ['nx', 'ny', 'nz']:
            label, w_type, p_type, kwargs = self.MANUAL_RES_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            dims_layout.addWidget(input_group)
        res_main_layout.addWidget(self.dims_group)
        
        self.ooip_group = QGroupBox("OOIP Determination")
        ooip_layout = QVBoxLayout(self.ooip_group)
        ooip_choice_layout = QHBoxLayout()
        self.ooip_calc_radio = QRadioButton("Calculate from Parameters")
        self.ooip_direct_radio = QRadioButton("Direct Input")
        ooip_choice_layout.addWidget(self.ooip_calc_radio)
        ooip_choice_layout.addWidget(self.ooip_direct_radio)
        ooip_layout.addLayout(ooip_choice_layout)
        
        self.ooip_calc_params_group = QGroupBox("Volumetric Parameters")
        calc_params_layout = QGridLayout(self.ooip_calc_params_group)
        ooip_calc_keys = ['area', 'thickness', 'length', 'swi', 'boi']
        for i, name in enumerate(ooip_calc_keys):
            label, w_type, p_type, kwargs = self.MANUAL_RES_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            input_group.finalValueChanged.connect(self._calculate_and_display_ooip)
            self.manual_inputs_widgets[name] = input_group
            calc_params_layout.addWidget(input_group, i // 2, i % 2)
            
        if 'boi' in self.manual_inputs_widgets:
            self.manual_inputs_widgets['boi'].finalValueChanged.connect(self._calculate_and_display_ooip)
        
        ooip_layout.addWidget(self.ooip_calc_params_group)
        
        self.ooip_direct_input_group = QGroupBox("Direct OOIP Value")
        direct_layout = QVBoxLayout(self.ooip_direct_input_group)
        name = 'ooip_stb'
        label, w_type, p_type, kwargs = self.MANUAL_RES_DEFS[name]
        input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
        input_group.setProperty("param_type", p_type)
        input_group.finalValueChanged.connect(self._on_parameter_changed)
        self.manual_inputs_widgets[name] = input_group
        direct_layout.addWidget(input_group)
        ooip_layout.addWidget(self.ooip_direct_input_group)
        res_main_layout.addWidget(self.ooip_group)
        
        self.use_layered_model_checkbox = QCheckBox("Use Layered Reservoir Model")
        res_main_layout.addWidget(self.use_layered_model_checkbox)

        

        
        self.uniform_props_group = QGroupBox("Uniform Properties")
        uniform_props_layout = QGridLayout(self.uniform_props_group)
        for i, name in enumerate(['poro', 'perm', 'rock_compressibility', 'kv_kh_ratio']):
            label, w_type, p_type, kwargs = self.MANUAL_RES_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            uniform_props_layout.addWidget(input_group, i // 2, i % 2)
        self.manual_inputs_widgets['poro'].finalValueChanged.connect(self._calculate_and_display_ooip)
        res_main_layout.addWidget(self.uniform_props_group)
        
        self.layered_props_group = QGroupBox("Layered Properties")
        layered_props_layout = QVBoxLayout(self.layered_props_group)
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(4)
        self.layers_table.horizontalHeader().setStretchLastSection(True)
        self._add_layer_row(pv_frac=0.4, perm_factor=2.5, poro=0.22, thickness=10.0)
        self._add_layer_row(pv_frac=0.6, perm_factor=0.5, poro=0.18, thickness=15.0)
        layer_button_layout = QHBoxLayout()
        self.add_layer_btn = QPushButton(QIcon.fromTheme("list-add"), "Add Layer")
        self.remove_layer_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove Selected Layer")
        layer_button_layout.addWidget(self.add_layer_btn)
        layer_button_layout.addWidget(self.remove_layer_btn)
        layer_button_layout.addStretch()
        layered_props_layout.addWidget(self.layers_table)
        layered_props_layout.addLayout(layer_button_layout)
        res_main_layout.addWidget(self.layered_props_group)
        
        self.rel_perm_group = QGroupBox("Relative Permeability")
        rel_perm_layout = QGridLayout(self.rel_perm_group)
        rel_perm_keys = ['s_gc', 'n_o', 'n_g', 's_wc', 's_orw', 'n_w', 'n_ow']
        for i, name in enumerate(rel_perm_keys):
            label, w_type, p_type, kwargs = self.MANUAL_RES_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            rel_perm_layout.addWidget(input_group, i // 3, i % 3)
        self.plot_rel_perm_btn = QPushButton(QIcon.fromTheme("view-statistics"), "Plot Curves")
        rel_perm_layout.addWidget(self.plot_rel_perm_btn, (len(rel_perm_keys)) // 3, (len(rel_perm_keys)) % 3)
        res_main_layout.addWidget(self.rel_perm_group)

        self.calculated_params_group = QGroupBox("Calculated Parameters")
        calculated_params_layout = QFormLayout(self.calculated_params_group)
        self.calculated_ooip_label = QLabel("N/A")
        self.calculated_pv_label = QLabel("N/A")
        self.mobility_ratio_label = QLabel("N/A")
        self.v_dp_coefficient_label = QLabel("N/A")
        calculated_params_layout.addRow("Calculated OOIP:", self.calculated_ooip_label)
        calculated_params_layout.addRow("Calculated Pore Volume:", self.calculated_pv_label)
        calculated_params_layout.addRow("Mobility Ratio:", self.mobility_ratio_label)
        calculated_params_layout.addRow("V_DP Coefficient:", self.v_dp_coefficient_label)
        res_main_layout.addWidget(self.calculated_params_group)
        
        self.layered_props_group.setVisible(False)
        self.uniform_props_group.setVisible(True)
        
        
        res_main_layout.addStretch()
        return tab_panel

    def _create_pvt_tab(self) -> QWidget:
        tab_panel = QWidget()
        pvt_main_layout = QVBoxLayout(tab_panel)

        self.use_detailed_pvt_checkbox = QCheckBox("Use Detailed PVT Model")
        pvt_main_layout.addWidget(self.use_detailed_pvt_checkbox)
        
        pvt_props_group = QGroupBox("PVT Properties")
        pvt_props_layout = QGridLayout(pvt_props_group)
        
        for i, name in enumerate(self.MANUAL_PVT_DEFS.keys()):
            label, w_type, p_type, kwargs = self.MANUAL_PVT_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            pvt_props_layout.addWidget(input_group, i // 2, i % 2)
            
        pvt_main_layout.addWidget(pvt_props_group)
        
        self.detailed_pvt_btn = QPushButton(QIcon.fromTheme("document-edit"), "Edit Detailed PVT Data (PVTO, PVTG, etc.)")
        pvt_main_layout.addWidget(self.detailed_pvt_btn)
        
        pvt_main_layout.addStretch()
        return tab_panel

    def _create_wells_tab(self) -> QWidget:
        tab_panel = QWidget()
        wells_main_layout = QVBoxLayout(tab_panel)
        
        self.wells_group = QGroupBox("Well Data")
        wells_layout = QVBoxLayout(self.wells_group)
        
        self.well_list_widget = QListWidget()
        wells_layout.addWidget(self.well_list_widget)
        
        buttons_layout = QHBoxLayout()
        self.add_well_btn = QPushButton(QIcon.fromTheme("list-add"), "Add Well")
        self.remove_well_btn = QPushButton(QIcon.fromTheme("list-remove"), "Remove Well")
        self.view_well_btn = QPushButton(QIcon.fromTheme("document-open"), "View Well")
        buttons_layout.addWidget(self.add_well_btn)
        buttons_layout.addWidget(self.remove_well_btn)
        buttons_layout.addWidget(self.view_well_btn)
        buttons_layout.addStretch()
        
        wells_layout.addLayout(buttons_layout)
        
        # Info label for one-well support
        self.well_info_label = QLabel()
        self.well_info_label.setWordWrap(True)
        self.well_info_label.setStyleSheet("color: #555; font-style: italic; margin-top: 5px;")
        wells_layout.addWidget(self.well_info_label)
        
        wells_main_layout.addWidget(self.wells_group)
        
        return tab_panel

    def _create_surrogate_tuning_tab(self) -> QWidget:
        """Create surrogate engine tuning parameters tab"""
        tab_panel = QWidget()
        layout = QVBoxLayout(tab_panel)
        
        # Add Calculate Button
        btn_layout = QHBoxLayout()
        self.calc_tuning_btn = QPushButton(QIcon.fromTheme("calculator"), "Estimate from Reservoir/PVT Data")
        self.calc_tuning_btn.clicked.connect(self._calculate_tuning_parameters)
        btn_layout.addWidget(self.calc_tuning_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Use grid layout for groups to be more compact
        groups_layout = QGridLayout()

        # Miscibility transition
        miscibility_group = QGroupBox("Miscibility Transition")
        miscibility_layout = QGridLayout(miscibility_group)
        param_names = ['alpha_base', 'miscibility_window']
        for i, name in enumerate(param_names):
            label, w_type, p_type, kwargs = self.SURROGATE_TUNING_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            miscibility_layout.addWidget(input_group, i // 2, i % 2)
        groups_layout.addWidget(miscibility_group, 0, 0)

        # Production dynamics
        dynamics_group = QGroupBox("Production Dynamics")
        dynamics_layout = QGridLayout(dynamics_group)
        param_names = ['breakthrough_time_years', 'trapping_efficiency']
        for i, name in enumerate(param_names):
            label, w_type, p_type, kwargs = self.SURROGATE_TUNING_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            dynamics_layout.addWidget(input_group, i // 2, i % 2)
        groups_layout.addWidget(dynamics_group, 0, 1)

        # Mobility and mixing
        mobility_group = QGroupBox("Mobility & Mixing")
        mobility_layout = QGridLayout(mobility_group)
        param_names = ['transverse_mixing_calibration', 'omega_tl']
        for i, name in enumerate(param_names):
            label, w_type, p_type, kwargs = self.SURROGATE_TUNING_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            mobility_layout.addWidget(input_group, i // 2, i % 2)
        groups_layout.addWidget(mobility_group, 1, 0)

        # Relative permeability endpoints
        relperm_group = QGroupBox("Relative Permeability Endpoints")
        relperm_layout = QGridLayout(relperm_group)
        param_names = ['k_ro_0', 'k_rg_0']
        for i, name in enumerate(param_names):
            label, w_type, p_type, kwargs = self.SURROGATE_TUNING_DEFS[name]
            input_group = ParameterInputGroup(param_name=name, label_text=label, input_type=w_type, **kwargs)
            input_group.setProperty("param_type", p_type)
            input_group.finalValueChanged.connect(self._on_parameter_changed)
            self.manual_inputs_widgets[name] = input_group
            relperm_layout.addWidget(input_group, i // 2, i % 2)
        groups_layout.addWidget(relperm_group, 1, 1)

        layout.addLayout(groups_layout)
        layout.addStretch()

        return tab_panel

    def _calculate_tuning_parameters(self):
        try:
            # 1. Endpoints
            s_wc = self.manual_inputs_values.get('s_wc')
            if s_wc is not None and 'k_ro_0' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['k_ro_0'].set_value(round(max(0.1, 1.0 - s_wc), 2))
                    
            sorw = self.manual_inputs_values.get('s_orw')
            if sorw is not None and 'k_rg_0' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['k_rg_0'].set_value(round(max(0.1, 1.0 - sorw), 2))
                
            # 2. C7+ Fraction Estimation
            c7_est = None
            api = self.manual_inputs_values.get('api_gravity', 35.0)
            gor = self.manual_inputs_values.get('sol_gor', 500.0)

            # A. Attempt to get explicit mole fractions from EOS compositional setup
            if self.config_manager and self.config_manager.is_loaded:
                eos_data = self.config_manager.get_section("eos_composition")
                if eos_data and "component_names" in eos_data and "component_properties" in eos_data:
                    names = eos_data["component_names"]
                    props = eos_data["component_properties"]
                    try:
                        # Find indices for all components C7 and heavier
                        c7_idx = [i for i, n in enumerate(names) if 'C7' in n.upper() or 'C10' in n.upper() or 'C12' in n.upper() or 'HEAVY' in n.upper() or 'PLUS' in n.upper() or '+' in n]
                        if c7_idx:
                            c7_est = sum(props[i][0] for i in c7_idx) # Sum of mole fractions
                            logger.info(f"Extracted C7+ fraction ({c7_est}) directly from EOS Composition.")
                    except Exception as e:
                        logger.warning(f"Could not extract C7+ from EOS: {e}")

            # B. Check if we have Detailed Black Oil PVT data to extract max Rs as GOR
            if c7_est is None and self.use_detailed_pvt_checkbox.isChecked() and self.detailed_pvt_data:
                pvto = self.detailed_pvt_data.get('PVTO', np.array([]))
                if pvto.size > 0 and pvto.shape[1] > 1:
                    max_rs = np.max(pvto[:, 1])
                    if max_rs > 0:
                        gor = max_rs # Override manual GOR with max Rs from table
                        logger.info(f"Using max Rs ({gor} scf/bbl) from detailed PVTO table for C7+ estimation.")

            # C. Fallback to empirical correlations (Katz-Firoozabadi or Ovalle)
            if c7_est is None:
                if api < 45.0:
                    c7_est = max(0.01, min(0.99, 1.0 - 0.015 * api)) # Katz-Firoozabadi for Black Oil
                else:
                    gor_mscf = max(gor / 1000.0, 0.01)
                    c7_est = max(0.01, min(0.99, 0.3157 * (gor_mscf ** -0.9205))) # Ovalle for Volatile Oil

            if 'c7_plus_fraction' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['c7_plus_fraction'].set_value(round(c7_est, 3))
                
            # 3. Todd-Longstaff Omega (Web Research: 1/3 for heterogeneous field, 2/3 for homogeneous/lab)
            v_dp = self.manual_inputs_values.get('v_dp_coefficient', 0.7)
            omega = 0.33 if v_dp > 0.6 else 0.66
            if 'omega_tl' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['omega_tl'].set_value(omega)

            # 4. Breakthrough Time (Influenced by Mobility, Heterogeneity, and Injection Scheme)
            M = self.manual_inputs_values.get('mobility_ratio', 5.0)
            if M is None or M <= 0: M = 5.0
            
            # Retrieve scheme from configuration if available
            inj_scheme = "continuous"
            wag_ratio = 1.0
            if self.config_manager and self.config_manager.is_loaded:
                eor_params = self.config_manager.get('EORParameters')
                if not eor_params:
                    # check if nested
                    eor_params = self.config_manager.get('optimization.EORParameters')
                if eor_params and isinstance(eor_params, dict):
                    inj_scheme = eor_params.get('injection_scheme', 'continuous')
                    wag_ratio = eor_params.get('WAG_ratio', 1.0)
                    if inj_scheme == 'swag' and 'swag' in eor_params and isinstance(eor_params['swag'], dict):
                        wag_ratio = eor_params['swag'].get('water_gas_ratio', wag_ratio)
            
            W_factor = 0.0
            if inj_scheme in ['swag', 'wag', 'pulsed']:
                W_factor = wag_ratio
                
            # Dampen M based on 1 / (1 + W) due to water mobility
            M_eff = M ** (1.0 / (1.0 + max(0.0, W_factor)))
            # Heuristic base time
            base_time = 5.0 / (M_eff * (1.0 + v_dp**2))
            # Actual breakthrough shifted by volumetric WAG displacement factor
            bt_years = max(0.2, min(25.0, base_time * (1.0 + W_factor)))
            
            if 'breakthrough_time_years' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['breakthrough_time_years'].set_value(round(bt_years, 2))

            # 5. Transverse Mixing (Scales with Kv/Kh ratio)
            kv_kh = self.manual_inputs_values.get('kv_kh_ratio', 0.1)
            # Baseline 0.5, adjusting up for higher vertical permeability
            mixing = min(1.0, max(0.1, 0.5 + (max(0.0, kv_kh - 0.1) * 2.0)))
            if 'transverse_mixing_calibration' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['transverse_mixing_calibration'].set_value(round(mixing, 2))
                
            # 6. Trapping Efficiency (Land's type heuristic based on porosity)
            poro = self.manual_inputs_values.get('poro', 0.2)
            trapping = min(0.4, max(0.1, poro * 1.5))
            if 'trapping_efficiency' in self.manual_inputs_widgets:
                self.manual_inputs_widgets['trapping_efficiency'].set_value(round(trapping, 2))

            # 7. Miscibility Window and Alpha Base (Physics-based defaults)
            if 'miscibility_window' in self.manual_inputs_widgets:
                misc_win = 0.05 if api > 40.0 else 0.10
                self.manual_inputs_widgets['miscibility_window'].set_value(misc_win)
            
            if 'alpha_base' in self.manual_inputs_widgets:
                # Alpha base defines exact P/MMP ratio where miscibility transition is 50%
                alpha_base_est = 1.0 - 0.1 * (c7_est - 0.3)
                self.manual_inputs_widgets['alpha_base'].set_value(round(alpha_base_est, 3))

            QMessageBox.information(self, self.tr("Estimation Complete"), self.tr("Advanced tuning parameters have been estimated using industry heuristic correlations (Todd-Longstaff, Dykstra-Parsons) from Reservoir & PVT inputs."))
        except Exception as e:
            logger.error(f"Tuning calculation error: {e}", exc_info=True)
            QMessageBox.warning(self, self.tr("Calculation Error"), self.tr(f"Failed to estimate tuning parameters: {e}"))

    def _coerce_value(self, value: Any, param_type: Type) -> Any:
        if value is None or value == '':
            return None
        try:
            if param_type is bool:
                return bool(value)
            if param_type is int:
                return int(value)
            if param_type is float:
                return float(value)
            return str(value)
        except (ValueError, TypeError):
            raise ValueError(f"Could not convert '{value}' to {param_type.__name__}")

    def _on_parameter_changed(self, value: Any):
        sender = self.sender()
        if not isinstance(sender, ParameterInputGroup): return
        param_name = sender.param_name
        all_defs = {**self.MANUAL_RES_DEFS, **self.MANUAL_PVT_DEFS, **self.SURROGATE_TUNING_DEFS}
        param_type = all_defs.get(param_name, (None, None, str))[2]
        try:
            sender.clear_error()
            corrected_value = self._coerce_value(value, param_type)
            self.manual_inputs_values[param_name] = corrected_value
        except (ValueError, TypeError) as e:
            sender.show_error(str(e))
            if param_name in self.manual_inputs_values: del self.manual_inputs_values[param_name]
        self._update_calculated_eor_params()


    def _create_reservoir_data_from_ui(self) -> ReservoirData:
        # Basic grid properties
        nx = self.manual_inputs_values.get('nx', 50)
        ny = self.manual_inputs_values.get('ny', 50)
        nz = self.manual_inputs_values.get('nz', 10)
        
        grid = {
            'NX': np.array([nx]),
            'NY': np.array([ny]),
            'NZ': np.array([nz]),
        }

        layer_definitions = None
        geostat_params = None
        geostatistical_grid = None

        # Handle different property models
        if self.use_layered_model_checkbox.isChecked():
            # Layered model
            layer_definitions = []
            for row in range(self.layers_table.rowCount()):
                try:
                    pv_frac = float(self.layers_table.item(row, 0).text())
                    perm_factor = float(self.layers_table.item(row, 1).text())
                    poro = float(self.layers_table.item(row, 2).text())
                    thickness = float(self.layers_table.item(row, 3).text())
                    layer_definitions.append(LayerDefinition(thickness=thickness, porosity=poro, permeability_multiplier=perm_factor))
                except (ValueError, AttributeError):
                    pass
            
            if layer_definitions:
                grid['PORO'] = np.zeros((nx, ny, nz))
                grid['PERMX'] = np.zeros((nx, ny, nz))
                grid['PERMY'] = np.zeros((nx, ny, nz))
                grid['PERMZ'] = np.zeros((nx, ny, nz))
                
                current_z = 0
                total_thickness = sum(layer.thickness for layer in layer_definitions)
                for layer in layer_definitions:
                    layer_thickness_cells = int(layer.thickness / total_thickness * nz)
                    if current_z + layer_thickness_cells > nz:
                        layer_thickness_cells = nz - current_z

                    grid['PORO'][:, :, current_z:current_z+layer_thickness_cells] = layer.porosity
                    perm = self.manual_inputs_values.get('perm', 100.0) * layer.permeability_multiplier
                    grid['PERMX'][:, :, current_z:current_z+layer_thickness_cells] = perm
                    grid['PERMY'][:, :, current_z:current_z+layer_thickness_cells] = perm
                    grid['PERMZ'][:, :, current_z:current_z+layer_thickness_cells] = perm * self.manual_inputs_values.get('kv_kh_ratio', 0.1)
                    current_z += layer_thickness_cells
        else:
            # Uniform model
            poro = self.manual_inputs_values.get('poro', 0.2)
            perm = self.manual_inputs_values.get('perm', 100.0)
            grid['PORO'] = np.full((nx, ny, nz), poro)
            grid['PERMX'] = np.full((nx, ny, nz), perm)
            grid['PERMY'] = np.full((nx, ny, nz), perm)
            grid['PERMZ'] = np.full((nx, ny, nz), perm * self.manual_inputs_values.get('kv_kh_ratio', 0.1))

        ooip_stb = self.manual_inputs_values.get('ooip_stb', 1000000.0)
        if self.ooip_calc_radio.isChecked():
            # Recalculate to be sure
            try:
                area = self.manual_inputs_values.get('area')
                thickness = self.manual_inputs_values.get('thickness')
                poro = self.manual_inputs_values.get('poro')
                swi = self.manual_inputs_values.get('swi')
                boi = self.manual_inputs_values.get('boi')
                if all(v is not None for v in [area, thickness, poro, swi, boi]):
                    pv = 7758 * float(area) * float(thickness) * float(poro)
                    ooip_stb = pv * (1 - float(swi)) / float(boi)
            except (TypeError, ValueError, KeyError):
                pass # Keep default if calculation fails

        reservoir_data = ReservoirData(
            grid=grid,
            pvt_tables=self.detailed_pvt_data if self.detailed_pvt_data else {},
            ooip_stb=ooip_stb,
            initial_pressure=self.manual_inputs_values.get('initial_pressure', 4000.0),
            rock_compressibility=self.manual_inputs_values.get('rock_compressibility', 3e-6),
            temperature=self.manual_inputs_values.get('temperature', 150.0),
            length_ft=self.manual_inputs_values.get('length', 2000.0),
            area_acres=self.manual_inputs_values.get('area'),
            thickness_ft=self.manual_inputs_values.get('thickness'),
            average_porosity=self.manual_inputs_values.get('poro'),
            initial_water_saturation=self.manual_inputs_values.get('swi'),
            oil_fvf=self.manual_inputs_values.get('boi'),
            rock_type=self.manual_inputs_values.get('rock_type'),
            depositional_environment=self.manual_inputs_values.get('depositional_environment'),
            structural_complexity=self.manual_inputs_values.get('structural_complexity'),
            layer_definitions=layer_definitions,
            geostatistical_params=geostat_params,
        )

        eos_composition_data = self.config_manager.get_section("eos_composition")
        if eos_composition_data:
            try:
                # Extract component names from the config data
                component_properties = eos_composition_data.get("component_properties")
                binary_interaction_coeffs = eos_composition_data.get("binary_interaction_coeffs")
                
                # Smart default for component names
                default_names = ["CO2", "C1", "C2", "C3", "C4+", "C5+", "C6+", "C7+"]
                if component_properties is not None:
                    try:
                        props_array = np.array(component_properties)
                        if props_array.ndim == 2:
                            num_components = props_array.shape[0]
                            # If we have properties but no names, generate generic names or use default if counts match
                            if "component_names" not in eos_composition_data:
                                if num_components == len(default_names):
                                    component_names = default_names
                                else:
                                    component_names = [f"Comp_{i+1}" for i in range(num_components)]
                                    logger.info(f"Inferring component names from data: {component_names}")
                            else:
                                component_names = eos_composition_data.get("component_names")
                        else:
                            component_names = eos_composition_data.get("component_names", default_names)
                    except Exception:
                         component_names = eos_composition_data.get("component_names", default_names)
                else:
                    component_names = eos_composition_data.get("component_names", default_names)

                # Validate component properties shape
                if component_properties is not None:
                    component_properties = np.array(component_properties)
                    expected_shape = (len(component_names), 5)

                    if component_properties.shape != expected_shape:
                        from error_handler import report_caught_error, ErrorSeverity, ErrorCategory
                        error_msg = f"EOS component_properties shape {component_properties.shape} != expected {expected_shape}"
                        logger.error(error_msg)

                        # Report to centralized error manager
                        report_caught_error(
                            operation="validate EOS component properties shape",
                            exception=ValueError(error_msg),
                            context={
                                "component_names": component_names,
                                "expected_shape": expected_shape,
                                "actual_shape": component_properties.shape,
                                "config_source": "EORParameters" if hasattr(self, 'current_eor_params') else "unknown"
                            },
                            user_action_suggested="Check EOS configuration file for correct component properties matrix dimensions",
                            show_dialog=True,
                            severity=ErrorSeverity.WARNING,
                            category=ErrorCategory.CONFIGURATION
                        )

                        QMessageBox.warning(self, "EOS Configuration Error",
                                          f"Invalid EOS component properties shape.\n"
                                          f"Expected: {expected_shape}, Got: {component_properties.shape}\n"
                                          f"Using default EOS component properties instead.")
                        component_properties = self._create_default_eos_properties(component_names)
                        logger.warning(f"USING DEFAULT EOS PROPERTIES: {component_properties.shape}")
                else:
                    from error_handler import report_caught_error, ErrorSeverity, ErrorCategory
                    error_msg = "EOS component_properties not found in configuration"
                    logger.error(error_msg)

                    # Report to centralized error manager
                    report_caught_error(
                        operation="load EOS component properties from configuration",
                        exception=KeyError(error_msg),
                        context={
                            "component_names": component_names,
                            "config_keys": list(self.eor_params.keys()) if hasattr(self, 'eor_params') else [],
                            "config_source": "EORParameters" if hasattr(self, 'current_eor_params') else "unknown"
                        },
                        user_action_suggested="Add EOS component properties to configuration file or use configuration manager to generate proper EOS data",
                        show_dialog=True,
                        severity=ErrorSeverity.WARNING,
                        category=ErrorCategory.CONFIGURATION
                    )

                    QMessageBox.warning(self, "EOS Configuration Error",
                                      f"{error_msg}.\n"
                                      f"Using default EOS component properties instead.")
                    component_properties = self._create_default_eos_properties(component_names)
                    logger.warning(f"USING DEFAULT EOS PROPERTIES: {component_properties.shape}")

                # Validate binary interaction coefficients
                if binary_interaction_coeffs is not None:
                    binary_interaction_coeffs = np.array(binary_interaction_coeffs)
                    expected_shape = (len(component_names), len(component_names))

                    if binary_interaction_coeffs.shape != expected_shape:
                        from error_handler import report_caught_error, ErrorSeverity, ErrorCategory
                        error_msg = f"EOS binary_interaction_coeffs shape {binary_interaction_coeffs.shape} != expected {expected_shape}"
                        logger.error(error_msg)

                        # Report to centralized error manager
                        report_caught_error(
                            operation="validate EOS binary interaction coefficients shape",
                            exception=ValueError(error_msg),
                            context={
                                "component_names": component_names,
                                "expected_shape": expected_shape,
                                "actual_shape": binary_interaction_coeffs.shape,
                                "config_source": "EORParameters" if hasattr(self, 'current_eor_params') else "unknown"
                            },
                            user_action_suggested="Check EOS configuration file for correct binary interaction coefficients matrix dimensions (should be square matrix NxN where N is number of components)",
                            show_dialog=True,
                            severity=ErrorSeverity.WARNING,
                            category=ErrorCategory.CONFIGURATION
                        )

                        QMessageBox.warning(self, "EOS Configuration Error",
                                          f"Invalid EOS binary interaction coefficients shape.\n"
                                          f"Expected: {expected_shape}, Got: {binary_interaction_coeffs.shape}\n"
                                          f"Using default identity matrix instead.")
                        binary_interaction_coeffs = np.eye(len(component_names))
                        logger.warning(f"USING DEFAULT EOS BINARY COEFFICIENTS: {binary_interaction_coeffs.shape}")
                else:
                    from error_handler import report_caught_error, ErrorSeverity, ErrorCategory
                    error_msg = "EOS binary_interaction_coeffs not found in configuration"
                    logger.error(error_msg)

                    # Report to centralized error manager
                    report_caught_error(
                        operation="load EOS binary interaction coefficients from configuration",
                        exception=KeyError(error_msg),
                        context={
                            "component_names": component_names,
                            "config_keys": list(self.eor_params.keys()) if hasattr(self, 'eor_params') else [],
                            "config_source": "EORParameters" if hasattr(self, 'current_eor_params') else "unknown"
                        },
                        user_action_suggested="Add EOS binary interaction coefficients to configuration file or use configuration manager to generate proper EOS data",
                        show_dialog=True,
                        severity=ErrorSeverity.WARNING,
                        category=ErrorCategory.CONFIGURATION
                    )

                    QMessageBox.warning(self, "EOS Configuration Error",
                                      f"{error_msg}.\n"
                                      f"Using default identity matrix instead.")
                    binary_interaction_coeffs = np.eye(len(component_names))
                    logger.warning(f"USING DEFAULT EOS BINARY COEFFICIENTS: {binary_interaction_coeffs.shape}")

                eos_params = EOSModelParameters(
                    eos_type=eos_composition_data.get("eos_type", "PR"),
                    component_names=component_names,
                    component_properties=component_properties,
                    binary_interaction_coeffs=binary_interaction_coeffs
                )
                reservoir_data.eos_model = eos_params
                logger.info("Loaded EOS composition from config.")
            except Exception as e:
                error_msg = f"Failed to load EOS composition from config: {e}"
                logger.error(error_msg)
                QMessageBox.critical(self, "EOS Configuration Error",
                                    f"{error_msg}\n"
                                    f"Creating default CO2-EOR EOS model instead.")
                # EOS is REQUIRED for CO2-EOR - never set to None
                component_names = ["CO2", "C1", "C4-C6", "C7+", "C10+"]
                component_properties = _create_default_co2_eor_properties()
                binary_interaction_coeffs = _create_default_co2_eor_binary_coeffs(len(component_names))

                eos_params = EOSModelParameters(
                    eos_type="PR",  # Peng-Robinson is standard for petroleum
                    component_names=component_names,
                    component_properties=component_properties,
                    binary_interaction_coeffs=binary_interaction_coeffs
                )
                reservoir_data.eos_model = eos_params
                logger.warning("Created default CO2-EOR EOS model due to configuration error.")
        else:
            # No EOS composition found in config - EOS is REQUIRED for CO2-EOR
            logger.warning("No EOS composition found in configuration. Creating default CO2-EOR EOS model.")
            QMessageBox.information(self, "EOS Configuration",
                                   "No EOS model configuration found.\n"
                                   "Creating a default CO2-EOR EOS model for optimization.\n"
                                   "You can configure custom EOS parameters in settings.")

            component_names = ["CO2", "C1", "C4-C6", "C7+", "C10+"]
            component_properties = _create_default_co2_eor_properties()
            binary_interaction_coeffs = _create_default_co2_eor_binary_coeffs(len(component_names))

            eos_params = EOSModelParameters(
                eos_type="PR",  # Peng-Robinson is standard for petroleum
                component_names=component_names,
                component_properties=component_properties,
                binary_interaction_coeffs=binary_interaction_coeffs
            )
            reservoir_data.eos_model = eos_params
            logger.info("Created default CO2-EOR EOS model.")

        # Ensure EOS model was successfully created
        if reservoir_data.eos_model is None:
            logger.critical("EOS model is None after all attempts - this should never happen in CO2-EOR!")
            raise ValueError("EOS model is required for CO2-EOR optimization but could not be created.")

        return reservoir_data

    def _create_pvt_properties_from_ui(self) -> PVTProperties:
        if self.use_detailed_pvt_checkbox.isChecked() and self.detailed_pvt_data:
            pvt_type = 'compositional' if 'eos_model' in self.detailed_pvt_data else 'black_oil'

            pvto_data = self.detailed_pvt_data.get('PVTO', np.array([]))
            pvtg_data = self.detailed_pvt_data.get('PVTG', np.array([]))

            pressure_points = pvto_data[:, 0] if pvto_data.size > 0 else np.array([])
            rs = pvto_data[:, 1] if pvto_data.size > 0 and pvto_data.shape[1] > 1 else np.array([])
            oil_fvf = pvto_data[:, 2] if pvto_data.size > 0 and pvto_data.shape[1] > 2 else np.array([])
            oil_viscosity = pvto_data[:, 3] if pvto_data.size > 0 and pvto_data.shape[1] > 3 else np.array([])

            gas_fvf = pvtg_data[:, 1] if pvtg_data.size > 0 and pvtg_data.shape[1] > 1 else np.array([])
            co2_viscosity = np.array([self.manual_inputs_values.get('co2_visc', 0.02)] * len(pressure_points)) if pressure_points.size > 0 else np.array([])

            return PVTProperties(
                pressure_points=pressure_points,
                oil_fvf=oil_fvf,
                oil_viscosity=oil_viscosity,
                gas_fvf=gas_fvf,
                co2_viscosity=co2_viscosity,
                rs=rs,
                pvt_type=pvt_type,
                gas_specific_gravity=self.manual_inputs_values.get('gas_specific_gravity', 0.7),
                temperature=self.manual_inputs_values.get('temperature', 212.0),
                api_gravity=self.manual_inputs_values.get('api_gravity', 35.0),
                c7_plus_fraction=self.manual_inputs_values.get('c7_plus_fraction', 0.35),
                co2_solubility_scm_per_bbl=self.manual_inputs_values.get('co2_solubility_scm_per_bbl', 200.0),
                oil_viscosity_cp=self.manual_inputs_values.get('oil_visc', 0.8),
            )
        else:
            return PVTProperties(
                pressure_points=np.array([]),
                oil_fvf=np.array([]),
                oil_viscosity=np.array([]),
                gas_fvf=np.array([]),
                co2_viscosity=np.array([]),
                rs=np.array([]),
                pvt_type='black_oil',
                gas_specific_gravity=self.manual_inputs_values.get('gas_specific_gravity', 0.7),
                temperature=self.manual_inputs_values.get('temperature', 212.0),
                api_gravity=self.manual_inputs_values.get('api_gravity', 35.0),
                c7_plus_fraction=self.manual_inputs_values.get('c7_plus_fraction', 0.35),
                co2_solubility_scm_per_bbl=self.manual_inputs_values.get('co2_solubility_scm_per_bbl', 200.0),
                oil_viscosity_cp=self.manual_inputs_values.get('oil_viscosity_cp', 1.0),
                water_viscosity_cp=self.manual_inputs_values.get('water_viscosity_cp', 0.5),
                gas_viscosity_cp=self.manual_inputs_values.get('gas_viscosity_cp', 0.02),
                oil_fvf_simple=self.manual_inputs_values.get('oil_fvf_simple', 1.2),
                water_fvf=self.manual_inputs_values.get('water_fvf', 1.0),
                gas_fvf_simple=self.manual_inputs_values.get('gas_fvf_simple', 0.01),
            )

    def _add_pvt_row(self):
        row_position = self.pvt_table.rowCount()
        self.pvt_table.insertRow(row_position)

    def _remove_pvt_row(self):
        current_row = self.pvt_table.currentRow()
        if current_row >= 0:
            self.pvt_table.removeRow(current_row)

    def _plot_pvt_data(self):
        try:
            pressure_points = []
            oil_fvf = []
            oil_viscosity = []
            gas_fvf = []
            co2_viscosity = []
            rs = []

            for row in range(self.pvt_table.rowCount()):
                try:
                    pressure_points.append(float(self.pvt_table.item(row, 0).text()))
                    oil_fvf.append(float(self.pvt_table.item(row, 1).text()))
                    oil_viscosity.append(float(self.pvt_table.item(row, 2).text()))
                    gas_fvf.append(float(self.pvt_table.item(row, 3).text()))
                    co2_viscosity.append(float(self.pvt_table.item(row, 4).text()))
                    rs.append(float(self.pvt_table.item(row, 5).text()))
                except (ValueError, AttributeError):
                    # Skip rows with invalid data
                    pass

            if not pressure_points:
                QMessageBox.warning(self, "No Data", "No valid PVT data to plot.")
                return

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pressure_points, y=oil_fvf, mode='lines+markers', name='Oil FVF'))
            fig.add_trace(go.Scatter(x=pressure_points, y=oil_viscosity, mode='lines+markers', name='Oil Viscosity'))
            fig.add_trace(go.Scatter(x=pressure_points, y=gas_fvf, mode='lines+markers', name='Gas FVF'))
            fig.add_trace(go.Scatter(x=pressure_points, y=co2_viscosity, mode='lines+markers', name='CO2 Viscosity'))
            fig.add_trace(go.Scatter(x=pressure_points, y=rs, mode='lines+markers', name='Rs'))

            fig.update_layout(
                title="PVT Properties",
                xaxis_title="Pressure",
                yaxis_title="Value",
            )

            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
            self.right_tab_widget.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Could not plot PVT data:\n{e}")
            logger.error(f"Error plotting PVT data: {e}", exc_info=True)

    def _process_manual_data(self):
        try:
            # Comprehensive validation for CO2-EOR system - all data must be user-provided
            required_reservoir_params = [
                ('poro', 'Porosity'),
                ('perm', 'Permeability'),
                ('thickness', 'Reservoir Thickness'),
                ('area', 'Reservoir Area'),
                ('temperature', 'Reservoir Temperature'),
                ('initial_pressure', 'Initial Reservoir Pressure'),
                ('swi', 'Initial Water Saturation')
            ]

            missing_params = []
            for param_key, param_name in required_reservoir_params:
                value = self.manual_inputs_values.get(param_key)
                if value is None or (isinstance(value, (int, float)) and value <= 0):
                    missing_params.append(f"• {param_name} (required, positive value)")

            # Check PVT properties
            required_pvt_params = [
                ('api_gravity', 'API Gravity'),
                ('gas_specific_gravity', 'Gas Specific Gravity'),
                ('oil_viscosity_cp', 'Oil Viscosity at reservoir conditions')
            ]

            for param_key, param_name in required_pvt_params:
                value = self.manual_inputs_values.get(param_key)
                if value is None or (isinstance(value, (int, float)) and value <= 0):
                    missing_params.append(f"• {param_name} (required for accurate PVT calculations)")

            # Check PVT data - support both black oil and detailed approaches for CO2-EOR
            has_detailed_pvt = bool(self.detailed_pvt_data)
            has_black_oil_pvt = (
                self.manual_inputs_values.get('api_gravity') is not None and
                self.manual_inputs_values.get('gas_specific_gravity') is not None and
                self.manual_inputs_values.get('oil_viscosity_cp') is not None
            )

            if not has_detailed_pvt and not has_black_oil_pvt:
                missing_params.append("• PVT Data (either Detailed PVT with PVTO/PVTG tables OR Black Oil with API gravity, gas gravity, and oil viscosity)")

            # For CO2-EOR, validate PVT parameters based on approach
            if has_black_oil_pvt:
                # Validate black oil parameters for CO2-EOR suitability
                api_gravity = self.manual_inputs_values.get('api_gravity')
                gas_gravity = self.manual_inputs_values.get('gas_specific_gravity')
                oil_viscosity = self.manual_inputs_values.get('oil_viscosity_cp')

                if not (10 <= api_gravity <= 60):  # Reasonable API range
                    missing_params.append("• API Gravity (must be between 10-60 for CO2-EOR)")
                if not (0.5 <= gas_gravity <= 2.0):  # Reasonable gas gravity
                    missing_params.append("• Gas Specific Gravity (must be between 0.5-2.0 for CO2-EOR)")
                if not (0.1 <= oil_viscosity <= 100):  # Reasonable oil viscosity range
                    missing_params.append("• Oil Viscosity (must be between 0.1-100 cp for CO2-EOR)")

            # Check well data - essential for CO2-EOR
            if not self.well_data_list:
                # Huff-n-Puff can theoretically work with simulation defaults, but we still prefer a well.
                # Relax for specific scheme if needed, but here we still keep it as a requirement for optimization
                missing_params.append("• Well Data (at least one well required for CO2-EOR optimization)")
            else:
                # Check for injector/producer requirements based on scheme
                injection_scheme = self.config_manager.get_section("eor_parameters").get("injection_scheme", "continuous").lower()
                has_injector = any(w.metadata.get('type') == 'injector' or w.metadata.get('status', '').lower() == 'injector' for w in self.well_data_list)
                
                if injection_scheme != "huff_n_puff" and not has_injector:
                    # Continuous/WAG usually need an explicit injector.
                    # We'll allow it but warn the user.
                    logger.warning("No injection wells defined for continuous/WAG scheme. Using field-wide injection rates.")

            # Check EOS model - absolutely required for CO2-EOR
            eos_composition_data = self.config_manager.get_section("eos_composition")
            if not eos_composition_data:
                missing_params.append("• EOS Model Configuration (required for accurate CO2-EOR thermodynamics)")

            if missing_params:
                missing_text = "\n".join(missing_params)
                error_title = "Missing Required Data for CO2-EOR"
                error_message = self.tr(
                    "The following required parameters are missing or invalid for CO2-EOR optimization:\n\n"
                    f"{missing_text}\n\n"
                    "CO2-EOR optimization requires comprehensive data to produce accurate results. "
                    "Please provide all required parameters in user interface."
                )
                QMessageBox.critical(self, self.tr(error_title), self.tr(error_message))
                return

            # Additional validation for well data completeness
            incomplete_wells = []
            for i, well in enumerate(self.well_data_list):
                if not hasattr(well, 'well_path') or well.well_path is None or len(well.well_path) == 0:
                    incomplete_wells.append(f"• Well {i+1} ({well.name}): Missing well trajectory")
                elif not hasattr(well, 'metadata') or not well.metadata:
                    incomplete_wells.append(f"• Well {i+1} ({well.name}): Missing well metadata (completion, type, etc.)")

            if incomplete_wells:
                incomplete_text = "\n".join(incomplete_wells)
                warning_title = "Incomplete Well Data"
                warning_message = self.tr(
                    "The following wells have incomplete data:\n\n"
                    f"{incomplete_text}\n\n"
                    "For optimal CO2-EOR results, ensure all wells have complete trajectory and metadata."
                )
                reply = QMessageBox.question(
                    self,
                    self.tr(warning_title),
                    self.tr(warning_message + "\n\n" + "Continue with incomplete well data?"),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Log successful validation
            logger.info("CO2-EOR data validation passed - all required parameters provided by user")
            logger.info(f"Reservoir: {self.manual_inputs_values.get('area')} acres, {self.manual_inputs_values.get('thickness')} ft thick")

            # Log PVT approach being used
            if has_detailed_pvt:
                logger.info(f"PVT: Detailed PVT tables with {len(self.detailed_pvt_data.get('PVTO', []))} pressure points")
            elif has_black_oil_pvt:
                logger.info(f"PVT: Black Oil - API {self.manual_inputs_values.get('api_gravity')}, Gas grav {self.manual_inputs_values.get('gas_specific_gravity')}, Oil visc {self.manual_inputs_values.get('oil_viscosity_cp')} cp")
            else:
                logger.warning("PVT: No valid PVT data - this should not happen after validation")

            logger.info(f"Wells: {len(self.well_data_list)} wells defined")
            logger.info(f"EOS: {'User-configured' if eos_composition_data else 'Default CO2-EOR'}")

            if not self.well_data_list:
                reply = QMessageBox.question(self, self.tr("No Wells Defined"), 
                                             self.tr("You have not defined any wells. Do you want to proceed with a simulation-only setup?"),
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return

            reservoir_data = self._create_reservoir_data_from_ui()
            pvt_properties = self._create_pvt_properties_from_ui()

            project_data = {
                "reservoir_data": reservoir_data,
                "pvt_properties": pvt_properties,
                "well_data_list": self.well_data_list,
                "detailed_pvt_data": self.detailed_pvt_data,
            }

            # Process data through integration engine if available for engine compatibility
            if self.data_integration_engine:
                try:
                    # Calculate block sizes
                    nx = self.manual_inputs_values.get('nx', 50)
                    ny = self.manual_inputs_values.get('ny', 50)
                    nz = self.manual_inputs_values.get('nz', 10)
                    length_ft = self.manual_inputs_values.get('length', 2000.0)
                    area_acres = self.manual_inputs_values.get('area', 100.0)
                    thickness_ft = self.manual_inputs_values.get('thickness', 50.0)
                    
                    dx = length_ft / nx if nx > 0 else 0
                    # Area (acres) * 43560 = Area (sq ft). Width = Area / Length
                    width_ft = (area_acres * 43560) / length_ft if length_ft > 0 else 0
                    dy = width_ft / ny if ny > 0 else 0
                    dz = thickness_ft / nz if nz > 0 else 0

                    # Get calculated EOR parameters
                    mobility_ratio = self.manual_inputs_values.get('mobility_ratio', 5.0)
                    v_dp_coefficient = self.manual_inputs_values.get('v_dp_coefficient', 0.5)

                    # Create integration-compatible data structure with all surrogate engine parameters
                    integration_data = {
                        "reservoir_parameters": {
                            "grid_dimensions": {"nx": nx, "ny": ny, "nz": nz},
                            "block_sizes": {"dx": dx, "dy": dy, "dz": dz},
                            "initial_pressure": self.manual_inputs_values.get('initial_pressure', 4000.0),
                            "temperature": self.manual_inputs_values.get('temperature', 150.0),
                            "rock_compressibility": self.manual_inputs_values.get('rock_compressibility', 3e-6),
                            "initial_water_saturation": self.manual_inputs_values.get('swi', 0.25),
                            "ooip_stb": self.manual_inputs_values.get('ooip_stb', 1000000.0),
                        },
                        "pvt_parameters": {
                            "api_gravity": self.manual_inputs_values.get('api_gravity', 35.0),
                            "gas_specific_gravity": self.manual_inputs_values.get('gas_specific_gravity', 0.7),
                            "oil_viscosity_cp": self.manual_inputs_values.get('oil_viscosity_cp', 1.0),
                            "gas_viscosity_cp": self.manual_inputs_values.get('gas_viscosity_cp', 0.02),
                            "water_viscosity_cp": self.manual_inputs_values.get('water_viscosity_cp', 0.5),
                            "gas_fvf_simple": self.manual_inputs_values.get('gas_fvf_simple', 0.005),
                        },
                        "eor_parameters": {
                            "injection_rate": 5000.0,  # Default injection rate - will be set by optimization widget
                            "target_pressure_psi": 3000.0,  # Default target pressure
                            "mobility_ratio": mobility_ratio,  # Use calculated value
                            "WAG_ratio": 1.0,  # Default WAG ratio
                            "default_mmp_fallback": 2500.0,  # Default MMP fallback
                            "default_oil_viscosity_cp": self.manual_inputs_values.get('oil_viscosity_cp', 1.0),
                            "default_co2_viscosity_cp": self.manual_inputs_values.get('gas_viscosity_cp', 0.02),
                        },
                        "operational_parameters": {
                            "project_lifetime_years": 15,  # Default project lifetime - will be set by optimization widget
                            "recovery_model_selection": "phd_hybrid",
                        },
                        "economic_parameters": {
                            "oil_price_usd_per_bbl": 70.0,  # Default from config
                            "co2_purchase_cost_usd_per_tonne": 50.0,  # Default from config
                            "co2_recycle_cost_usd_per_tonne": 15.0,  # Default
                            "co2_storage_credit_usd_per_tonne": 25.0,  # Default
                            "water_injection_cost_usd_per_bbl": 1.0,  # Default
                            "water_disposal_cost_usd_per_bbl": 2.0,  # Default
                            "discount_rate_fraction": 0.10,  # Default
                            "capex_usd": 5_000_000.0,  # Default
                            "fixed_opex_usd_per_year": 200_000.0,  # Default
                            "variable_opex_usd_per_bbl": 5.0,  # Default
                        },
                        "fitting_parameters": {
                            "c7_plus_fraction": self.manual_inputs_values.get('c7_plus_fraction', 0.35),
                            "alpha_base": self.manual_inputs_values.get('alpha_base', 1.0),
                            "miscibility_window": self.manual_inputs_values.get('miscibility_window', 0.011),
                            "breakthrough_time_years": self.manual_inputs_values.get('breakthrough_time_years', 1.5),
                            "trapping_efficiency": self.manual_inputs_values.get('trapping_efficiency', 0.4),
                            "initial_gor_scf_per_stb": self.manual_inputs_values.get('initial_gor_scf_per_stb', 500.0),
                            "transverse_mixing_calibration": self.manual_inputs_values.get('transverse_mixing_calibration', 0.5),
                            "omega_tl": self.manual_inputs_values.get('omega_tl', 0.6),
                            "k_ro_0": self.manual_inputs_values.get('k_ro_0', 0.8),
                            "k_rg_0": self.manual_inputs_values.get('k_rg_0', 1.0),
                            "n_o": self.manual_inputs_values.get('n_o', 2.0),
                            "n_g": self.manual_inputs_values.get('n_g', 2.0),
                        },
                        "well_data": [
                            {
                                "name": w.name,
                                "type": w.metadata.get('type', 'producer'),
                                "x": w.well_path[0][0] if w.well_path is not None and len(w.well_path) > 0 else 0,
                                "y": w.well_path[0][1] if w.well_path is not None and len(w.well_path) > 0 else 0,
                                "z": w.well_path[0][2] if w.well_path is not None and len(w.well_path) > 0 and len(w.well_path[0]) > 2 else 0,
                                "status": w.metadata.get('status', 'active')
                            } for w in self.well_data_list
                        ],
                        "geostatistical_enabled": hasattr(self, 'use_geostatistical_model_checkbox') and self.use_geostatistical_model_checkbox.isChecked(),
                        "geostatistical_params": getattr(self, 'geostatistical_params', None).__dict__ if getattr(self, 'geostatistical_params', None) else None
                    }

                    # Validate data first
                    validation_results = self.data_integration_engine.process_and_validate_dataset(integration_data)

                    if not validation_results.get("is_valid", True):
                        error_msg = "Data validation failed:\n" + "\n".join(validation_results.get("errors", []))
                        QMessageBox.warning(self, self.tr("Validation Error"), self.tr(error_msg))
                        return

                    # Create complete engine data structures for surrogate engine
                    engine_data = self.data_integration_engine.create_engine_data_structures(integration_data)

                    # Update project data with engine-ready structures
                    if engine_data:
                        project_data["reservoir_data"] = engine_data.get("reservoir_data")
                        project_data["pvt_properties"] = engine_data.get("pvt_properties")
                        project_data["eor_parameters"] = engine_data.get("eor_parameters")
                        project_data["operational_parameters"] = engine_data.get("operational_parameters")
                        project_data["economic_parameters"] = engine_data.get("economic_parameters")
                        project_data["well_data_list"] = engine_data.get("well_data", [])
                        logger.info("Data successfully processed for surrogate engine")

                except Exception as e:
                    logger.warning(f"Data integration processing failed: {e}. Using raw data.")

            self.project_data_updated.emit(project_data)
            self.status_message_updated.emit(self.tr("Project data generated successfully."), 5000)

        except Exception as e:
             QMessageBox.critical(self, self.tr("Generation Error"), self.tr("An error occurred while processing manual data:\n\n{e}").format(e=e))
             logger.error(f"Error processing manual data: {e}", exc_info=True)

    def _update_well_table_tooltips(self):
        for i in range(self.well_list_widget.count()):
            item = self.well_list_widget.item(i)
            well_name = item.text()
            well_data = next((w for w in self.well_data_list if w.name == well_name), None)
            if well_data:
                tooltip = f"""<b>Well:</b> {well_data.name}<br>
                           <b>Location:</b> ({well_data.well_path[0][0]}, {well_data.well_path[0][1]})<br>
                           <b>Type:</b> {well_data.metadata.get('status', 'N/A')}<br>
                           <b>Curves:</b> {', '.join(well_data.properties.keys())}
                        """
                item.setToolTip(tooltip)

    def _add_well_to_ui(self, well_data: WellData):
        item = QListWidgetItem(QIcon.fromTheme("document"), well_data.name)
        self.well_list_widget.addItem(item)
        self._update_well_table_tooltips()
        self._update_calculated_eor_params()
        self._update_well_info_label()


    def _remove_selected_well(self):
        selected_items = self.well_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, self.tr("No Well Selected"), self.tr("Please select a well to remove."))
            return

        for item in selected_items:
            well_name = item.text()
            # Remove from internal list
            self.well_data_list = [w for w in self.well_data_list if w.name != well_name]
            # Remove from UI
            self.well_list_widget.takeItem(self.well_list_widget.row(item))
        
        self.status_message_updated.emit(self.tr("Removed selected well(s)."), 3000)
        self._update_calculated_eor_params()
        self._update_well_info_label()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_well_info_label()

    def _update_well_info_label(self):
        if not hasattr(self, 'well_info_label'):
            return
            
        scheme = "continuous"
        if self.config_manager:
            scheme = self.config_manager.get_section("eor_parameters").get("injection_scheme", "continuous").lower()
            
        well_count = len(self.well_data_list)
        
        if scheme == "huff_n_puff":
            if well_count == 1:
                self.well_info_label.setText(self.tr("Single-well Huff-n-Puff: This well will alternate between injection and production phases."))
            elif well_count > 1:
                self.well_info_label.setText(self.tr(f"Multi-well Huff-n-Puff: {well_count} wells will operate in cyclic mode independently."))
            else:
                self.well_info_label.setText(self.tr("Huff-n-Puff: Please add at least one well."))
        else:
            if well_count == 1:
                self.well_info_label.setText(self.tr("Single well detected: Surrogate engine will use field-wide injection rates. For detailed simulation, both injector and producer are recommended."))
            elif well_count == 0:
                self.well_info_label.setText(self.tr("Please add at least one well for CO2-EOR optimization."))
            else:
                self.well_info_label.setText("")




    def clear_all_project_data(self):
        # Block signals during bulk reset to avoid redundant calculations and potential crashes
        for widget in self.manual_inputs_widgets.values():
            widget.blockSignals(True)
            
        try:
            # Clear data stores
            self.well_data_list.clear()
            self.reservoir_data = None
            self.pvt_properties = None
            self.detailed_pvt_data = None
            self.manual_inputs_values.clear()

            # Clear UI elements
            self.well_list_widget.clear()
            self.layers_table.setRowCount(0)

            # Reset all manual input widgets to their default values
            all_defs = {**self.MANUAL_RES_DEFS, **self.MANUAL_PVT_DEFS, **self.SURROGATE_TUNING_DEFS}
            for name, widget in self.manual_inputs_widgets.items():
                if name in all_defs:
                    _, _, _, kwargs = all_defs[name]
                    default_value = kwargs.get('default_value')
                    widget.set_value(default_value)
                    self.manual_inputs_values[name] = default_value # Also update internal values

            # Reset checkboxes and radio buttons
            self.use_layered_model_checkbox.setChecked(False)
            self.use_detailed_pvt_checkbox.setChecked(False)
            self.ooip_calc_radio.setChecked(True)

            # Reset plot views
            self.plot_view.setHtml("")
            self.ax_3d.clear()
            self.canvas_3d.draw()
            self.canvas_3d.setVisible(False)

            # Re-add default layers
            self._add_layer_row(pv_frac=0.4, perm_factor=2.5, poro=0.22, thickness=10.0)
            self._add_layer_row(pv_frac=0.6, perm_factor=0.5, poro=0.18, thickness=15.0)

        finally:
            for widget in self.manual_inputs_widgets.values():
                widget.blockSignals(False)

        # Update calculated fields once after all resets
        self._calculate_and_display_ooip()
        self._update_calculated_eor_params()

        self.status_message_updated.emit(self.tr("Project data cleared."), 3000)

    def _calculate_mobility_ratio(self) -> Optional[float]:
        try:
            krw_end = 1.0
            kro_end = 1.0
            
            visc_gas = self.manual_inputs_values.get('gas_viscosity_cp', 0.02)
            visc_oil = self.manual_inputs_values.get('oil_viscosity_cp', 1.0)

            if visc_oil > 0 and visc_gas > 0:
                mobility_inj = krw_end / visc_gas
                mobility_oil = kro_end / visc_oil
                if mobility_oil > 0:
                    return mobility_inj / mobility_oil
        except (TypeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"Could not calculate mobility ratio: {e}")
        return None

    def _calculate_v_dp_coefficient(self) -> Optional[float]:
        permeabilities = []
        if self.well_data_list:
            for well in self.well_data_list:
                if 'PERM' in well.properties:
                    permeabilities.extend(well.properties['PERM'])
                elif 'PERMX' in well.properties:
                    permeabilities.extend(well.properties['PERMX'])

        if not permeabilities and self.use_layered_model_checkbox.isChecked():
            base_perm = self.manual_inputs_values.get('perm', 100.0)
            for row in range(self.layers_table.rowCount()):
                try:
                    perm_factor = float(self.layers_table.item(row, 1).text())
                    pv_frac = float(self.layers_table.item(row, 0).text())
                    num_cells = int(pv_frac * 1000)
                    permeabilities.extend([base_perm * perm_factor] * num_cells)
                except (ValueError, AttributeError):
                    pass

        if permeabilities:
            try:
                k = np.array(permeabilities)
                if len(k) > 1:
                    k_sorted = np.sort(k)
                    k_50 = np.percentile(k_sorted, 50)
                    k_84_1 = np.percentile(k_sorted, 84.1)
                    if k_50 > 0:
                        return (k_50 - k_84_1) / k_50
            except Exception as e:
                logger.warning(f"Could not calculate V_DP coefficient from data: {e}")
        
        return 0.7

    def _update_calculated_eor_params(self):
        if not hasattr(self, 'mobility_ratio_label'): return
        mobility_ratio = self._calculate_mobility_ratio()
        if mobility_ratio is not None:
            self.mobility_ratio_label.setText(f"{mobility_ratio:.3f}")
            self.manual_inputs_values['mobility_ratio'] = mobility_ratio
        else:
            self.mobility_ratio_label.setText("N/A")
            if 'mobility_ratio' in self.manual_inputs_values:
                del self.manual_inputs_values['mobility_ratio']

        v_dp_coefficient = self._calculate_v_dp_coefficient()
        if v_dp_coefficient is not None:
            self.v_dp_coefficient_label.setText(f"{v_dp_coefficient:.3f}")
            self.manual_inputs_values['v_dp_coefficient'] = v_dp_coefficient
        else:
            self.v_dp_coefficient_label.setText("N/A")
            if 'v_dp_coefficient' in self.manual_inputs_values:
                del self.manual_inputs_values['v_dp_coefficient']

    def _calculate_and_display_ooip(self):
        try:
            area_val = self.manual_inputs_widgets['area'].get_value()
            thickness_val = self.manual_inputs_widgets['thickness'].get_value()
            poro_val = self.manual_inputs_widgets['poro'].get_value()
            swi_val = self.manual_inputs_widgets['swi'].get_value()
            boi_val = self.manual_inputs_widgets['boi'].get_value()

            all_values = [area_val, thickness_val, poro_val, swi_val, boi_val]

            if all(v is not None and str(v).strip() != '' for v in all_values):
                area = float(area_val)
                thickness = float(thickness_val)
                poro = float(poro_val)
                swi = float(swi_val)
                boi = float(boi_val)

                pv = 7758 * area * thickness * poro
                ooip = pv * (1 - swi) / boi
                self.calculated_pv_label.setText(f"{pv:,.0f} bbl")
                self.calculated_ooip_label.setText(f"{ooip:,.0f} STB")
                self.manual_inputs_values['ooip_stb'] = ooip
            else:
                self.calculated_pv_label.setText("N/A")
                self.calculated_ooip_label.setText("N/A")
        except (KeyError, ZeroDivisionError, TypeError, ValueError) as e:
            self.calculated_pv_label.setText("Error")
            self.calculated_ooip_label.setText("Error")
            logger.warning(f"Could not calculate OOIP: {e}")

    def _add_layer_row(self, pv_frac: float = 0.0, perm_factor: float = 1.0, poro: float = 0.2, thickness: float = 10.0):
        row_position = self.layers_table.rowCount()
        self.layers_table.insertRow(row_position)
        self.layers_table.setItem(row_position, 0, QTableWidgetItem(str(pv_frac)))
        self.layers_table.setItem(row_position, 1, QTableWidgetItem(str(perm_factor)))
        self.layers_table.setItem(row_position, 2, QTableWidgetItem(str(poro)))
        self.layers_table.setItem(row_position, 3, QTableWidgetItem(str(thickness)))

    def _toggle_layered_model(self, checked):
        self.layered_props_group.setVisible(checked)
        self.uniform_props_group.setVisible(not checked)

    def _toggle_geostatistical_model(self, checked):
        self.geostatistical_props_group.setVisible(checked)
        # Hide other property groups if geostat is selected
        self.uniform_props_group.setVisible(not checked)
        self.layered_props_group.setVisible(not checked)
        self.use_layered_model_checkbox.setEnabled(not checked)

    def _remove_selected_layer(self):
        current_row = self.layers_table.currentRow()
        if current_row >= 0:
            self.layers_table.removeRow(current_row)

    def _plot_rel_perm_curves(self):
        try:
            s_wc = self.manual_inputs_values.get('s_wc', 0.2)
            s_orw = self.manual_inputs_values.get('s_orw', 0.2)
            n_o = self.manual_inputs_values.get('n_o', 2.0)
            n_w = self.manual_inputs_values.get('n_w', 2.0)
            
            if not all(isinstance(v, (int, float)) for v in [s_wc, s_orw, n_o, n_w]):
                raise TypeError("All parameters must be numbers.")

            s_w = np.linspace(s_wc, 1 - s_orw, 100)
            
            # Water relative permeability
            k_rw = ((s_w - s_wc) / (1 - s_wc - s_orw))**n_w
            
            # Oil relative permeability
            s_o = 1 - s_w
            k_ro = ((s_o - s_orw) / (1 - s_wc - s_orw))**n_o
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_w, y=k_rw, mode='lines', name='krw'))
            fig.add_trace(go.Scatter(x=s_w, y=k_ro, mode='lines', name='kro'))
            
            fig.update_layout(
                title="Relative Permeability Curves (Water-Oil)",
                xaxis_title="Water Saturation (Sw)",
                yaxis_title="Relative Permeability",
                yaxis_range=[0, 1]
            )
            
            self.plot_view.setHtml(fig.to_html(include_plotlyjs='cdn'))
            self.right_tab_widget.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Could not plot relative permeability curves:\n{e}")
            logger.error(f"Error plotting rel perm curves: {e}", exc_info=True)

    def _open_pvt_editor(self):
        if not self.detailed_pvt_data:
            self.detailed_pvt_data = {}
        dialog = PVTEditorDialog(self.detailed_pvt_data, self)
        if dialog.exec():
            self.detailed_pvt_data = dialog.get_data()
            logger.info("Detailed PVT data updated from editor.")



    def _add_well_manually(self):
        existing_names = [w.name for w in self.well_data_list]
        dialog = ManualWellDialog(existing_names, self)
        if dialog.exec():
            well_data = dialog.get_well_data()
            if well_data:
                self.well_data_list.append(well_data)
                self._add_well_to_ui(well_data)

    def _get_or_create_well(self, well_name: str) -> WellData:
        well_data = next((w for w in self.well_data_list if w.name == well_name), None)
        if not well_data:
            # Create a default WellData matching the dataclass in core/data_models.py
            well_data = WellData(
                name=well_name,
                depths=np.array([0.0, 1000.0]),
                properties={},
                units={},
                metadata={"status": "Producer", "type": "producer"},
                well_path=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1000.0]])
            )
            self.well_data_list.append(well_data)
            self._add_well_to_ui(well_data)
        return well_data

    def _view_selected_well(self):
        selected_items = self.well_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Well Selected", "Please select a well to view.")
            return
        
        well_name = selected_items[0].text()
        well_data = next((w for w in self.well_data_list if w.name == well_name), None)

        if well_data:
            dialog = WellViewerDialog(self.well_data_list, selected_well=well_data, parent=self)
            dialog.exec()

    def _create_default_eos_properties(self, component_names: List[str]) -> np.ndarray:
        """Create default EOS component properties with correct shape (n_components, 5).

        The 5 columns are: [mole_fraction, critical_pressure, critical_temperature,
                           acentric_factor, critical_volume]
        """
        n_components = len(component_names)
        properties = np.zeros((n_components, 5))

        # Default values for common components
        default_props = {
            "CO2": [0.15, 1070.6, 547.9, 0.225, 94.07],
            "C1": [0.60, 667.8, 343.0, 0.011, 99.2],
            "C2": [0.10, 707.8, 549.9, 0.099, 147.0],
            "C3": [0.05, 616.3, 666.0, 0.152, 203.0],
            "C4+": [0.04, 550.7, 756.0, 0.200, 250.0],
            "C5+": [0.03, 490.0, 845.0, 0.250, 300.0],
            "C6+": [0.02, 440.0, 920.0, 0.300, 350.0],
            "C7+": [0.01, 400.0, 1000.0, 0.350, 400.0]
        }

        # Fill properties with defaults or reasonable estimates
        for i, component in enumerate(component_names):
            if component in default_props:
                properties[i] = default_props[component]
            else:
                # Default generic hydrocarbon properties
                properties[i] = [0.125, 500.0, 600.0, 0.200, 200.0]

        # Normalize mole fractions to sum to 1.0
        total_mole_frac = np.sum(properties[:, 0])
        if total_mole_frac > 0:
            properties[:, 0] = properties[:, 0] / total_mole_frac
        else:
            properties[:, 0] = 1.0 / n_components  # Equal fractions if all zero

        return properties

    def load_project_data(self, project_data: Dict[str, Any]):
        try:
            self.clear_all_project_data()

            self.reservoir_data = project_data.get('reservoir_data')
            self.pvt_properties = project_data.get('pvt_properties')
            self.well_data_list = project_data.get('well_data_list', [])
            self.detailed_pvt_data = project_data.get('detailed_pvt_data')

            if self.reservoir_data:
                # Load reservoir data into UI
                if self.reservoir_data.grid:
                    self.manual_inputs_widgets['nx'].set_value(self.reservoir_data.grid.get('NX', [50])[0])
                    self.manual_inputs_widgets['ny'].set_value(self.reservoir_data.grid.get('NY', [50])[0])
                    self.manual_inputs_widgets['nz'].set_value(self.reservoir_data.grid.get('NZ', [10])[0])
                
                self.manual_inputs_widgets['poro'].set_value(self.reservoir_data.average_porosity)
                # Assuming uniform permeability for now
                if self.reservoir_data.grid and 'PERMX' in self.reservoir_data.grid:
                    self.manual_inputs_widgets['perm'].set_value(self.reservoir_data.grid['PERMX'][0,0,0])

                self.manual_inputs_widgets['rock_compressibility'].set_value(self.reservoir_data.rock_compressibility)
                self.manual_inputs_widgets['swi'].set_value(self.reservoir_data.initial_water_saturation)
                self.manual_inputs_widgets['boi'].set_value(self.reservoir_data.oil_fvf)
                self.manual_inputs_widgets['ooip_stb'].set_value(self.reservoir_data.ooip_stb)

                if self.reservoir_data.layer_definitions:
                    self.use_layered_model_checkbox.setChecked(True)
                    self.layers_table.setRowCount(0)
                    for layer in self.reservoir_data.layer_definitions:
                        self._add_layer_row(pv_frac=0, perm_factor=layer.permeability_multiplier, poro=layer.porosity, thickness=layer.thickness)
                
                if self.reservoir_data.geostatistical_params:
                    self.use_geostatistical_model_checkbox.setChecked(True)

            if self.pvt_properties:
                # Load PVT data into UI
                self.manual_inputs_widgets['temperature'].set_value(self.pvt_properties.temperature)
                self.manual_inputs_widgets['initial_pressure'].set_value(self.reservoir_data.initial_pressure if self.reservoir_data else 4000.0)
                self.manual_inputs_widgets['api_gravity'].set_value(self.pvt_properties.api_gravity)
                self.manual_inputs_widgets['gas_specific_gravity'].set_value(self.pvt_properties.gas_specific_gravity)
                self.manual_inputs_widgets['oil_viscosity_cp'].set_value(self.pvt_properties.oil_viscosity_cp)

            if self.detailed_pvt_data:
                self.use_detailed_pvt_checkbox.setChecked(True)

            # Load wells
            for well_data in self.well_data_list:
                self._add_well_to_ui(well_data)

            self._update_calculated_eor_params()
            self._calculate_and_display_ooip()
            self.status_message_updated.emit(self.tr("Project data loaded successfully."), 5000)

        except Exception as e:
            QMessageBox.critical(self, self.tr("Loading Error"), self.tr("An error occurred while loading project data:\n{e}").format(e=e))
            logger.error(f"Error loading project data: {e}", exc_info=True)


def _create_default_co2_eor_properties() -> np.ndarray:
    """Create default component properties for CO2-EOR applications."""
    return np.array([
        # CO2 - primary injection component
        [0.30, 44.01, 304.13, 7.376e6, 0.225],    # CO2
        # C1 - light hydrocarbons
        [0.25, 16.04, 190.6, 4.604e6, 0.011],      # C1
        # C4-C6 - intermediate hydrocarbons
        [0.20, 86.18, 450.0, 3.0e6, 0.200],        # C4-C6 average
        # C7+ - medium hydrocarbons
        [0.15, 120.0, 540.2, 2.736e6, 0.350],     # C7+
        # C10+ - heavy hydrocarbons
        [0.10, 180.0, 650.0, 1.8e6, 0.480],       # C10+ (estimated)
    ])


def _create_default_co2_eor_binary_coeffs(n_components: int) -> np.ndarray:
    """Create default binary interaction coefficients for CO2-EOR."""
    coeffs = np.zeros((n_components, n_components))

    # CO2 interactions (first component)
    default_co2_interactions = [0.0, 0.10, 0.12, 0.15, 0.18]
    coeffs[0, :] = default_co2_interactions[:n_components]
    coeffs[:, 0] = default_co2_interactions[:n_components]

    # Other typical interactions
    if n_components >= 3:
        coeffs[1, 2] = 0.03  # C1-C4-C6
        coeffs[2, 1] = 0.03
    if n_components >= 4:
        coeffs[1, 3] = 0.04  # C1-C7+
        coeffs[3, 1] = 0.04
    if n_components >= 5:
        coeffs[1, 4] = 0.05  # C1-C10+
        coeffs[4, 1] = 0.05

    return coeffs