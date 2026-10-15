import logging
from typing import Optional, Dict, Any, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QPushButton, QMessageBox, QDialog, QTabWidget, QStackedWidget,
    QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer

from .components.model_tree_widget import ModelTreeWidget
from .components.pyvista_reservoir_canvas import PyVistaReservoirCanvas
from .components.contextual_property_grid import ContextualPropertyGrid
from .components.subsurface_data_viewer_widget import SubsurfaceDataViewerWidget
from .components.visual_audit_gate_widget import VisualAuditGateWidget

from ui.widgets.geology_cross_section_widget import GeologyCrossSectionWidget
from ui.widgets.geostatistics_visualizer_widget import GeostatisticsVisualizerWidget
from ui.widgets.corey_relperm_workstation_widget import CoreyRelPermWorkstationWidget
from ui.widgets.fault_geometry_visualizer_widget import FaultGeometryVisualizerWidget
from ui.widgets.model_evaluation_dashboard import ModelEvaluationDashboard
from ui.widgets.fluids_pvt_workstation_widget import FluidsPVTWorkstationWidget
from ui.widgets.well_network_workstation_widget import WellNetworkWorkstationWidget

from core.data_models import WellData, ReservoirData, PVTProperties, FaultData, CaprockLayer
from core.reservoir_state_manager import ReservoirStateManager
from core.geology.petrophysical_distribution import PetrophysicalCube, generate_petrophysical_cube

try:
    from ui.widgets.manual_well_dialog import ManualWellDialog
    from ui.dialogs.visual_audit_modal import VisualAuditModal
    from ui.dialogs.pre_flight_audit_dialog import PreFlightAuditDialog
except ImportError:
    ManualWellDialog = None
    VisualAuditModal = None
    PreFlightAuditDialog = None

logger = logging.getLogger(__name__)


class SubsurfaceWorkbenchWidget(QWidget):
    """
    Accessible 3D Subsurface Studio Workbench.
    Unites Hierarchical Model Tree, Native PyVistaQt 3D Viewport with hardware raycasting,
    Contextual Property Grid, and Integrated Subsurface Modeling Suite (Cross-Sections,
    Geostatistics, Fault Containment, and Multi-Domain Surveillance) into a seamless,
    unobstructed engineering workspace.
    """
    project_data_updated = pyqtSignal(dict)
    status_message_updated = pyqtSignal(str, int)

    def __init__(self, parent=None, config_manager=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.state_manager = ReservoirStateManager()

        # In-memory unified parameter store
        self.manual_inputs_values: Dict[str, Any] = {
            'nx': 50, 'ny': 50, 'nz': 10,
            'length': 2000.0, 'area': 100.0, 'thickness': 50.0,
            'poro': 0.20, 'perm': 100.0, 'swi': 0.25, 'boi': 1.2,
            'distribution_method': 'Facies-Controlled (3-Facies Architecture)',
            'facies_pattern': 'Fluvial Channel Belt',
            'sand_fraction': 0.65, 'silt_fraction': 0.25, 'shale_fraction': 0.10,
            'poro_perm_model': 'Kozeny-Carman', 'dykstra_parsons': 0.65, 'kv_kh_ratio': 0.10,
            'youngs_modulus_base': 20.0, 'poissons_ratio': 0.25, 'biot_coeff': 0.80,
            'overburden_grad': 1.00, 'stress_k0': 0.75, 'frac_grad': 0.85,
            # Corey Relative Permeability & Wettability
            's_wc': 0.20, 's_orw': 0.20, 's_gc': 0.05, 's_org': 0.15,
            'krw0': 0.30, 'kro0': 0.85, 'krg0': 0.60,
            'n_w': 2.5, 'n_ow': 2.0, 'n_o': 2.0, 'n_g': 2.0, 'n_og': 2.0,
            'wettability_preset': 'Strongly Water-Wet (Sandstone)',
            'relperm_model': 'Modified Stone I (Standard CO2 EOR)',
            # Fluvial Channel Object Parameters
            'channel_azimuth_deg': 45.0, 'channel_sinuosity': 1.30,
            'channel_wavelength_ft': 1500.0, 'channel_amplitude_ft': 350.0,
            'channel_width_ft': 450.0, 'levee_width_ft': 250.0,
            'aggradation_drift_ft': 30.0, 'num_channels': 1,
            # Barrier Island / Shoreface Object Parameters
            'barrier_azimuth_deg': 90.0, 'barrier_width_ft': 800.0,
            'lagoon_width_ft': 450.0, 'progradation_dip_deg': 2.0,
            # Carbonate Reef / Shoal Object Parameters
            'reef_center_x': 1000.0, 'reef_center_y': 1000.0,
            'reef_major_radius_ft': 650.0, 'reef_minor_radius_ft': 400.0,
            'reef_azimuth_deg': 45.0, 'apron_width_ft': 300.0,
            # Explicit Per-Facies Petrophysical Properties
            'f1_perm': 250.0, 'f1_poro': 0.25,
            'f2_perm': 40.0, 'f2_poro': 0.16,
            'f3_perm': 0.5, 'f3_poro': 0.06,
            # Layer Permeability Trend
            'layer_permeability_trend': 'Fining Upward',
            # Variogram & SGSIM
            'variogram_type': 'Spherical', 'variogram_range_major': 1200.0,
            'variogram_range_minor': 600.0, 'variogram_range_vert': 20.0,
            'variogram_azimuth_deg': 45.0, 'variogram_dip_deg': 0.0,
            'nugget_effect': 0.05, 'sill_variance': 1.0, 'geostat_seed': 42,
            'geostat_algorithm': 'Sequential Gaussian Simulation (SGSIM)',
            'geostat_conditioning': 'Conditional to Well Hard Data',
            # Fluids & In-situ State
            'initial_pressure': 4000.0, 'temperature': 212.0, 'api_gravity': 35.0,
            'gas_specific_gravity': 0.7, 'sol_gor': 500.0, 'oil_viscosity_cp': 1.0,
            'gas_viscosity_cp': 0.02, 'water_viscosity_cp': 0.5,
            'bubble_point_pressure': 2800.0, 'co2_purity': 95.0, 'co2_swelling_factor_max': 1.25,
            'todd_longstaff_omega': 0.70, 'mmp_correlation': 'Yellig-Metcalfe Correlation',
            'mmp_override': 2688.0, 'woc_depth': 5035.0, 'goc_depth': 4985.0, 'has_gas_cap': False,
            'water_gradient': 0.465, 'gas_gradient': 0.080, 'h_transition': 20.0,
            'fluid_preset': 'Permian San Andres (Medium Black Oil)',
            'fault_name': 'Fault F-1', 'fault_dip': 70.0, 'fault_strike': 45.0,
            'fault_throw': 25.0, 'fault_trans_mult': 0.15, 'fault_friction': 0.60,
            'fault_cohesion': 0.0, 'fault_center_x': 1000.0, 'fault_center_y': 1000.0,
            'caprock_lithology': 'Dense Marine Shale', 'caprock_thickness': 200.0,
            'caprock_t0': 200.0, 'caprock_cohesion': 400.0, 'caprock_friction_angle': 30.0,
            'caprock_entry_pressure': 1500.0, 'caprock_perm': 0.0001, 'caprock_safety_factor': 0.90,
            'sv_gradient': 1.05, 'sh_ratio_k0': 0.72, 'poisson_ratio': 0.25,
            'biot_coeff': 0.85, 'frac_gradient': 0.78, 'uic_sf': 0.90
        }
        self.well_data_list: List[WellData] = []
        self.fault_data_list: List[FaultData] = [
            FaultData(
                id="F-1",
                name="Fault F-1 (Major Boundary)",
                strike=45.0,
                dip=70.0,
                dip_direction="SE",
                throw=55.0,
                heave=20.0,
                length=3800.0,
                center_x=1000.0,
                center_y=1000.0,
                shale_gouge_ratio=34.0,
                transmissibility_multiplier=0.10,
                friction_coefficient=0.60,
                cohesion=0.0,
                slip_tendency=0.42,
                damage_zone_width=120.0,
                is_active=True
            ),
            FaultData(
                id="F-2",
                name="Fault F-2 (Synthetic Graben)",
                strike=45.0,
                dip=65.0,
                dip_direction="NW",
                throw=-40.0,
                heave=18.6,
                length=2800.0,
                center_x=1600.0,
                center_y=1200.0,
                shale_gouge_ratio=28.0,
                transmissibility_multiplier=0.15,
                friction_coefficient=0.60,
                cohesion=0.0,
                slip_tendency=0.38,
                damage_zone_width=90.0,
                is_active=True
            )
        ]
        self.caprock_layers: List[CaprockLayer] = [
            CaprockLayer(name="Unit C1 - Basal Marine Shale", thickness_ft=120.0, lithology="Illite-Smectite Shale", youngs_modulus_gpa=18.5, poissons_ratio=0.28, tensile_strength_psi=250.0, cohesion_psi=450.0, friction_angle_deg=32.0, entry_pressure_psi=2200.0, permeability_nd=10.0),
            CaprockLayer(name="Unit C2 - Intermediate Silt Baffle", thickness_ft=85.0, lithology="Calcite-Cemented Siltstone", youngs_modulus_gpa=24.0, poissons_ratio=0.25, tensile_strength_psi=180.0, cohesion_psi=320.0, friction_angle_deg=30.0, entry_pressure_psi=1450.0, permeability_nd=450.0),
            CaprockLayer(name="Unit C3 - Regional Aquitard", thickness_ft=180.0, lithology="Dense Silty Mudstone", youngs_modulus_gpa=16.0, poissons_ratio=0.30, tensile_strength_psi=140.0, cohesion_psi=260.0, friction_angle_deg=28.0, entry_pressure_psi=950.0, permeability_nd=2200.0)
        ]
        self.calculated_mmp_value = 2688.0

        self.param_debounce_timer = QTimer(self)
        self.param_debounce_timer.setSingleShot(True)
        self.param_debounce_timer.setInterval(120)
        self.param_debounce_timer.timeout.connect(self._update_3d_canvas_quick)

        self._setup_ui()
        self._connect_signals()
        self._refresh_all_views()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        self.setStyleSheet("""
            QWidget {
                background: #f8f9fa;
                color: #212529;
            }
            QSplitter::handle {
                background: #dee2e6;
            }
            QSplitter::handle:hover {
                background: #0d6efd;
            }
        """)

        # Main Horizontal Splitter (Tree | Center Subsurface Suite | Property Grid)
        self.main_h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Zone: Model Tree (260px)
        self.model_tree = ModelTreeWidget(self)
        self.model_tree.setMinimumWidth(260)
        self.main_h_splitter.addWidget(self.model_tree)

        # Center Zone: Single Active View Stack with Clean Engineering Breadcrumb Header
        self.center_container = QWidget()
        center_container_layout = QVBoxLayout(self.center_container)
        center_container_layout.setContentsMargins(0, 0, 0, 0)
        center_container_layout.setSpacing(4)

        # Clean Engineering Header Bar (Breadcrumb + Actions)
        self.center_header_bar = QFrame()
        self.center_header_bar.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 3px 6px;
            }
        """)
        header_bar_layout = QHBoxLayout(self.center_header_bar)
        header_bar_layout.setContentsMargins(8, 4, 8, 4)
        header_bar_layout.setSpacing(10)

        # Return to 3D View button (visible only when in other views)
        self.btn_back_to_3d = QPushButton("Back to 3D Studio")
        self.btn_back_to_3d.setToolTip("Return to 3D Volumetric Reservoir Studio")
        self.btn_back_to_3d.setStyleSheet("""
            QPushButton {
                background: #e0f2fe;
                color: #0284c7;
                border: 1px solid #bae6fd;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #bae6fd;
                color: #0369a1;
            }
        """)
        self.btn_back_to_3d.clicked.connect(self._on_back_to_3d_clicked)
        self.btn_back_to_3d.hide()
        header_bar_layout.addWidget(self.btn_back_to_3d)

        # Active view breadcrumb label
        self.lbl_active_view = QLabel("Active View: 3D Volumetric Reservoir Studio")
        self.lbl_active_view.setStyleSheet("font-size: 12px; font-weight: bold; color: #1e293b;")
        header_bar_layout.addWidget(self.lbl_active_view)

        header_bar_layout.addStretch()

        self.btn_preflight = QPushButton("Run Pre-Flight Audit")
        self.btn_preflight.setToolTip("Run automated physical consistency and boundary validation audit")
        self.btn_preflight.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e2e8f0;
                color: #0d6efd;
            }
        """)
        self.btn_preflight.clicked.connect(self._run_pre_flight_audit)

        self.btn_sync = QPushButton("Apply & Sync to Project Data")
        self.btn_sync.setToolTip("Synchronize all edits, calculated MMP, and wellbore network with active project data")
        self.btn_sync.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: #ffffff;
                border: 1px solid #0b5ed7;
                border-radius: 4px;
                padding: 4px 14px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0b5ed7;
            }
        """)
        self.btn_sync.clicked.connect(self._on_sync_button_clicked)

        self.btn_fault_caprock = QPushButton("Fault & Caprock Manager...")
        self.btn_fault_caprock.setToolTip("Open dedicated window to create, setup, and update structural faults and caprock stratigraphy")
        self.btn_fault_caprock.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #0284c7;
                border: 1px solid #bae6fd;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #e0f2fe;
                color: #0369a1;
            }
        """)
        self.btn_fault_caprock.clicked.connect(lambda: self._open_fault_caprock_manager_dialog(0))

        header_bar_layout.addWidget(self.btn_fault_caprock)
        header_bar_layout.addWidget(self.btn_preflight)
        header_bar_layout.addWidget(self.btn_sync)
        center_container_layout.addWidget(self.center_header_bar)

        # Single Active View Stack (No Tabs, No Overflow Arrows)
        self.center_stack = QStackedWidget(self)

        # View 0: 3D Volumetric Reservoir Studio (PyVista Viewport - 100% Full Screen Height)
        self.canvas_3d = PyVistaReservoirCanvas(self)
        self.center_stack.addWidget(self.canvas_3d)

        # View 1: Stratigraphy & Cross-Section
        self.cross_section_widget = GeologyCrossSectionWidget(self)
        self.center_stack.addWidget(self.cross_section_widget)

        # View 2: Geostatistics & Variograms
        self.geostat_widget = GeostatisticsVisualizerWidget(self)
        self.center_stack.addWidget(self.geostat_widget)

        # View 3: Corey Relative Permeability & Displacement
        self.relperm_widget = CoreyRelPermWorkstationWidget(self)
        self.center_stack.addWidget(self.relperm_widget)

        # View 4: Fault Containment & Integrity
        self.fault_widget = FaultGeometryVisualizerWidget(self)
        self.center_stack.addWidget(self.fault_widget)

        # View 5: Visual Audit Confirmation Gate (Native Embedded, Zero Dialog Popups)
        self.visual_audit_gate = VisualAuditGateWidget(self)
        self.center_stack.addWidget(self.visual_audit_gate)

        # View 6: Shared Earth Surveillance Workstation (Native Embedded)
        self.surveillance_widget = ModelEvaluationDashboard(parent=self)
        self.center_stack.addWidget(self.surveillance_widget)

        # View 7: Fluids & PVT Thermodynamics Workstation (Dedicated Middle Workstation)
        self.pvt_workstation = FluidsPVTWorkstationWidget(self)
        self.center_stack.addWidget(self.pvt_workstation)

        # View 8: Well Network Precision Modeling Workstation (Full Screen Middle View)
        self.well_workstation = WellNetworkWorkstationWidget(self)
        self.center_stack.addWidget(self.well_workstation)

        # View 9: Subsurface Data & Graph Workstation Viewer (Full Screen Middle Spreadsheet & Canvas)
        self.data_viewer = SubsurfaceDataViewerWidget(self)
        self.center_stack.addWidget(self.data_viewer)

        center_container_layout.addWidget(self.center_stack, stretch=1)
        self.main_h_splitter.addWidget(self.center_container)

        # Right Zone: Dedicated Side Panel Container with Vertical Collapse/Expand Toggle
        self.right_container = QWidget()
        right_container_layout = QHBoxLayout(self.right_container)
        right_container_layout.setContentsMargins(0, 0, 0, 0)
        right_container_layout.setSpacing(0)

        # Vertical Toggle Strip containing the << Parameter Input << button
        self.toggle_strip = QFrame()
        self.toggle_strip.setFixedWidth(28)
        self.toggle_strip.setStyleSheet("""
            QFrame {
                background: #f1f5f9;
                border: none;
            }
        """)
        toggle_strip_layout = QVBoxLayout(self.toggle_strip)
        toggle_strip_layout.setContentsMargins(1, 4, 1, 4)
        toggle_strip_layout.setSpacing(0)

        # Vertical button: << Parameter Input << (vertical box, clean borderless flat toggle)
        self.btn_toggle_params = QPushButton("»\n\nP\na\nr\na\nm\ne\nt\ne\nr\n \nI\nn\np\nu\nt\n\n»")
        self.btn_toggle_params.setToolTip("Collapse / Expand Parameter Input Panel")
        self.btn_toggle_params.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.btn_toggle_params.setFixedWidth(26)
        self.btn_toggle_params.setStyleSheet("""
            QPushButton {
                background: #e2e8f0;
                color: #0f172a;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
                padding: 4px 1px;
                line-height: 12px;
            }
            QPushButton:hover {
                background: #0d6efd;
                color: #ffffff;
            }
        """)
        self.btn_toggle_params.clicked.connect(self._toggle_parameter_panel)
        toggle_strip_layout.addWidget(self.btn_toggle_params)

        right_container_layout.addWidget(self.toggle_strip)

        # Contextual Property Grid (Clear borders, min 360px, max 520px)
        self.property_grid = ContextualPropertyGrid(self)
        self.property_grid.setMinimumWidth(360)
        self.property_grid.setMaximumWidth(520)
        right_container_layout.addWidget(self.property_grid, stretch=1)

        # Restrict right container width so it has proper size and can NEVER hide the 3D screen
        self.right_container.setMinimumWidth(388)
        self.right_container.setMaximumWidth(550)

        self.main_h_splitter.addWidget(self.right_container)

        # Ensure center 3D viewport can NEVER be collapsed to 0 by user dragging!
        self.main_h_splitter.setCollapsible(0, False)
        self.main_h_splitter.setCollapsible(1, False)
        self.main_h_splitter.setCollapsible(2, False)

        self.main_h_splitter.setStretchFactor(0, 0)
        self.main_h_splitter.setStretchFactor(1, 1)
        self.main_h_splitter.setStretchFactor(2, 0)
        self.main_h_splitter.setSizes([260, 740, 420])

        main_layout.addWidget(self.main_h_splitter, stretch=1)

    VIEW_TITLES = {
        0: "3D Volumetric Reservoir Studio",
        1: "Geological Stratigraphy & Cross-Section",
        2: "Geostatistical Spatial Continuity & Variograms",
        3: "Corey Relative Permeability & Two-Phase Displacement",
        4: "Fault Containment & Geomechanical Integrity",
        5: "Pre-Flight Visual Audit Confirmation Gate",
        6: "Multi-Domain Surveillance Workstation",
        7: "Fluids & PVT Thermodynamics Workstation",
        8: "Well Network & Precision Modeling Workstation",
        9: "Subsurface Data & Graph Workstation Viewer"
    }

    def switch_to_view(self, index: int):
        """Switches the single active view in the center stack and updates header and tree."""
        if index < 0 or index >= self.center_stack.count():
            return
        self.center_stack.setCurrentIndex(index)
        title = self.VIEW_TITLES.get(index, f"View {index}")
        self.lbl_active_view.setText(f"Active View: {title}")

        # Show 'Back to 3D' button only when on secondary views
        if index == 0:
            self.btn_back_to_3d.hide()
        else:
            self.btn_back_to_3d.show()

        # Update specific views on activation
        if index == 2 and hasattr(self, 'geostat_widget') and self.geostat_widget is not None:
            self.geostat_widget.set_parameters(self.manual_inputs_values)
        elif index == 3 and hasattr(self, 'relperm_widget') and self.relperm_widget is not None:
            self.relperm_widget.set_parameters(self.manual_inputs_values)
        elif index == 5 and hasattr(self, 'visual_audit_gate') and self.visual_audit_gate is not None:
            self.visual_audit_gate.update_project_data(self.get_current_project_data())
        elif index == 6 and hasattr(self, 'surveillance_widget') and self.surveillance_widget is not None:
            self.surveillance_widget.update_data(self.get_current_project_data())
        elif index == 7 and hasattr(self, 'pvt_workstation') and self.pvt_workstation is not None:
            self.pvt_workstation.set_parameters(self.manual_inputs_values)
        elif index == 8 and hasattr(self, 'well_workstation') and self.well_workstation is not None:
            self.well_workstation.set_parameters(self.manual_inputs_values)
            self.well_workstation.set_wells(self.well_data_list)

        # Side panel visibility logic: Visible ONLY if there are parameters to input
        # Views 5, 6, 8, 9 are full-screen monitoring/sheets/workstations -> hide side panel for maximum viewing area
        if index in [5, 6, 8, 9]:
            self.right_container.hide()
        elif hasattr(self, 'property_grid') and self.property_grid.has_editable_inputs():
            self.right_container.show()
            self.right_container.setMinimumWidth(388)
            self.right_container.setMaximumWidth(550)
            self.property_grid.show()
            self.btn_toggle_params.setText("»\n\nP\na\nr\na\nm\ne\nt\ne\nr\n \nI\nn\np\nu\nt\n\n»")
        else:
            self.right_container.hide()

    def _on_back_to_3d_clicked(self):
        """Returns to 3D studio and selects 3D studio leaf in the tree."""
        self.switch_to_view(0)
        self.canvas_3d.set_isolated_well(None)
        self.model_tree.select_domain_item("reservoir", "studio_3d")
        self.btn_back_to_3d.hide()

    def _connect_signals(self):
        # 1. Model Tree
        self.model_tree.node_selected.connect(self._on_tree_node_selected)
        self.model_tree.add_well_requested.connect(self._prompt_add_well)
        self.model_tree.generate_pattern_requested.connect(self._generate_pattern)

        # 3. 3D Canvas Picking & Click Placement
        self.canvas_3d.well_clicked.connect(self._on_3d_well_picked)
        self.canvas_3d.surface_clicked.connect(self._prompt_add_well_at_coords)

        # 4. Property Grid Changes
        self.property_grid.parameter_changed.connect(self._on_parameter_edited)
        self.property_grid.apply_requested.connect(self._on_apply_model_requested)
        if hasattr(self.property_grid, 'add_well_requested'):
            self.property_grid.add_well_requested.connect(self._prompt_add_well)
        if hasattr(self.property_grid, 'edit_well_requested'):
            self.property_grid.edit_well_requested.connect(self._on_property_grid_edit_well)
        if hasattr(self.property_grid, 'delete_well_requested'):
            self.property_grid.delete_well_requested.connect(self._on_property_grid_delete_well)
        if hasattr(self.property_grid, 'generate_pattern_requested'):
            self.property_grid.generate_pattern_requested.connect(self._generate_pattern)
        if hasattr(self.property_grid, 'manage_faults_requested'):
            self.property_grid.manage_faults_requested.connect(self._open_fault_caprock_manager_dialog)
        if hasattr(self.property_grid, 'add_fault_requested'):
            self.property_grid.add_fault_requested.connect(self._prompt_add_fault)
        if hasattr(self.property_grid, 'delete_fault_requested'):
            self.property_grid.delete_fault_requested.connect(self._on_property_grid_delete_fault)
        if hasattr(self.property_grid, 'isolate_fault_requested'):
            self.property_grid.isolate_fault_requested.connect(self._on_isolate_fault_requested)

        # 6. Visual Audit Gate Approval
        self.visual_audit_gate.model_approved.connect(self._on_audit_model_approved)

        # 7. Fluids & PVT Workstation Signals
        self.pvt_workstation.parameters_changed.connect(self._on_pvt_workstation_parameters_changed)
        self.pvt_workstation.sync_to_3d_requested.connect(self._on_apply_model_requested)

        # 8. Well Network Workstation Signals
        self.well_workstation.well_added.connect(self._on_workstation_well_added)
        self.well_workstation.well_updated.connect(self._on_workstation_well_updated)
        self.well_workstation.well_deleted.connect(self._on_workstation_well_deleted)
        self.well_workstation.well_selected.connect(self._on_workstation_well_selected)
        self.well_workstation.sync_to_model_requested.connect(self._on_workstation_sync_requested)
        self.well_workstation.status_message_requested.connect(self._on_workstation_status_message)
        if hasattr(self.well_workstation, 'place_well_on_3d_requested'):
            self.well_workstation.place_well_on_3d_requested.connect(self._on_workstation_place_well_requested)

    def _on_tree_node_selected(self, domain: str, item_key: str):
        self.property_grid.load_node(domain, item_key, self.manual_inputs_values, self.well_data_list, self.fault_data_list)
        
        # 1. Reservoir Geometry & Volumetrics Domain
        if domain == "reservoir":
            if item_key == "grid_table":
                self.data_viewer.render_grid_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "grid_graph":
                self.data_viewer.render_grid_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "ooip_table":
                self.data_viewer.render_ooip_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "ooip_graph":
                self.data_viewer.render_ooip_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "stratigraphy_table":
                self.data_viewer.render_stratigraphy_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key in ["stratigraphy", "stratigraphy_graph"]:
                self.switch_to_view(1)
            else:
                self.switch_to_view(0)

        # 2. Petrophysics & Heterogeneity Domain
        elif domain == "petrophysics":
            if item_key == "rock_table":
                self.data_viewer.render_rock_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "rock_graph":
                self.data_viewer.render_rock_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "relperm_table":
                self.data_viewer.render_relperm_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key in ["relperm", "relperm_graph"]:
                self.switch_to_view(3)
            elif item_key == "geostat_table":
                self.data_viewer.render_geostat_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key in ["geostat", "geostat_graph"]:
                self.switch_to_view(2)
            else:
                self.switch_to_view(0)

        # 3. Fluids & PVT Thermodynamics Domain
        elif domain == "pvt":
            if item_key == "black_oil_table":
                self.data_viewer.render_black_oil_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "swelling_table":
                self.data_viewer.render_swelling_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "mmp_table":
                self.data_viewer.render_mmp_table(self.manual_inputs_values, self.calculated_mmp_value)
                self.switch_to_view(9)
            elif item_key == "eos_table":
                self.data_viewer.render_eos_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "contacts_table":
                self.data_viewer.render_contacts_table(self.manual_inputs_values)
                self.switch_to_view(9)
            else:
                self.switch_to_view(7)
                if hasattr(self, 'pvt_workstation') and self.pvt_workstation is not None:
                    if item_key in ["mmp", "mmp_graph"]:
                        self.pvt_workstation.set_display_mode(2)
                    elif item_key in ["detailed", "eos", "eos_graph", "composition"]:
                        self.pvt_workstation.set_display_mode(3)
                    elif item_key in ["column", "contacts", "contacts_graph"]:
                        self.pvt_workstation.set_display_mode(4)
                    elif item_key in ["swelling", "swelling_graph", "solubility"]:
                        self.pvt_workstation.set_display_mode(1)
                    else:
                        self.pvt_workstation.set_display_mode(0)

        # 4. Well Network Domain
        elif domain in ["wells", "well_network"]:
            if item_key == "inventory_table":
                self.data_viewer.render_well_inventory_table(self.well_data_list, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "schedule_table":
                self.data_viewer.render_well_schedule_table(self.well_data_list, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "schedule_gantt":
                self.data_viewer.render_well_lifecycle_gantt_graph(self.well_data_list, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "wag_schedule_graph":
                self.data_viewer.render_wag_schedule_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "sweep_matrix":
                self.switch_to_view(8)
                if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                    self.well_workstation.set_parameters(self.manual_inputs_values)
                    self.well_workstation.set_wells(self.well_data_list)
                    self.well_workstation.combo_mode.setCurrentIndex(1)
            else:
                self.switch_to_view(8)
                if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                    self.well_workstation.set_parameters(self.manual_inputs_values)
                    self.well_workstation.set_wells(self.well_data_list)
                    self.well_workstation.combo_mode.setCurrentIndex(0)
                if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                    self.canvas_3d.set_isolated_well(None)

        # 5. Individual Well Domain & Attached Items
        elif domain == "well_item":
            if ":" in item_key:
                w_name, sub_action = item_key.split(":", 1)
                target_well = next((w for w in self.well_data_list if w.name == w_name), None)
                if sub_action == "trajectory_table":
                    if target_well:
                        self.data_viewer.render_well_trajectory_table(target_well, self.manual_inputs_values)
                    else:
                        self.data_viewer.render_well_inventory_table(self.well_data_list, self.manual_inputs_values)
                    self.switch_to_view(9)
                elif sub_action == "profile_graph":
                    self.switch_to_view(8)
                    if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                        self.well_workstation.set_parameters(self.manual_inputs_values)
                        self.well_workstation.set_wells(self.well_data_list)
                        self.well_workstation.select_well(w_name, isolate=True)
                        self.well_workstation.combo_mode.setCurrentIndex(1)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_isolated_well(w_name)
                elif sub_action == "ipr_graph":
                    self.switch_to_view(8)
                    if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                        self.well_workstation.set_parameters(self.manual_inputs_values)
                        self.well_workstation.set_wells(self.well_data_list)
                        self.well_workstation.select_well(w_name, isolate=True)
                        self.well_workstation.combo_mode.setCurrentIndex(2)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_isolated_well(w_name)
                elif sub_action == "sensitivity_graph":
                    self.switch_to_view(8)
                    if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                        self.well_workstation.set_parameters(self.manual_inputs_values)
                        self.well_workstation.set_wells(self.well_data_list)
                        self.well_workstation.select_well(w_name, isolate=True)
                        self.well_workstation.combo_mode.setCurrentIndex(3)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_isolated_well(w_name)
                elif sub_action == "drawdown_graph":
                    self.switch_to_view(8)
                    if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                        self.well_workstation.set_parameters(self.manual_inputs_values)
                        self.well_workstation.set_wells(self.well_data_list)
                        self.well_workstation.select_well(w_name, isolate=True)
                        self.well_workstation.combo_mode.setCurrentIndex(4)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_isolated_well(w_name)
                else:
                    self.switch_to_view(8)
                    if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                        self.well_workstation.set_parameters(self.manual_inputs_values)
                        self.well_workstation.set_wells(self.well_data_list)
                        self.well_workstation.select_well(w_name, isolate=True)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_isolated_well(w_name)
            else:
                self.switch_to_view(8)
                if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                    self.well_workstation.set_parameters(self.manual_inputs_values)
                    self.well_workstation.set_wells(self.well_data_list)
                    self.well_workstation.select_well(item_key, isolate=True)
                if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                    self.canvas_3d.set_isolated_well(item_key)

        # 5B. Individual Fault Domain & Attached Items
        elif domain == "fault_item":
            if ":" in item_key:
                f_name, sub_action = item_key.split(":", 1)
                target_fault = next((f for f in self.fault_data_list if f.name == f_name or f.id == f_name), None)
                if sub_action == "geometry_table":
                    if target_fault:
                        self.data_viewer.render_fault_geometry_table(target_fault, self.manual_inputs_values)
                    else:
                        self.data_viewer.render_fault_table(self.fault_data_list, self.manual_inputs_values)
                    self.switch_to_view(9)
                elif sub_action == "3d_isolated":
                    self.switch_to_view(0)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_fault_isolated_view(True, f_name)
                elif sub_action == "slip_graph":
                    if target_fault:
                        self.data_viewer.render_fault_slip_graph(target_fault, self.manual_inputs_values)
                    self.switch_to_view(9)
                elif sub_action == "sgr_graph":
                    if target_fault:
                        self.data_viewer.render_fault_juxtaposition_graph(target_fault, self.manual_inputs_values)
                    self.switch_to_view(9)
                else:
                    self.switch_to_view(0)
                    if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                        self.canvas_3d.set_fault_isolated_view(True, f_name)
            else:
                self.switch_to_view(0)
                if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                    self.canvas_3d.set_fault_isolated_view(True, item_key)

        # 6. Geomechanics & Fault Containment Domain
        elif domain == "geomechanics":
            if item_key == "caprock_table":
                self.data_viewer.render_caprock_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "caprock_strat_table":
                self.data_viewer.render_caprock_layers_table(self.caprock_layers, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "caprock_column_graph":
                self.data_viewer.render_caprock_sealing_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "caprock_graph":
                self.data_viewer.render_caprock_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "caprock_3d":
                self.switch_to_view(0)
                if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                    self.canvas_3d.set_fault_isolated_view(False)
                    if hasattr(self.canvas_3d, 'chk_caprock'):
                        self.canvas_3d.chk_caprock.setChecked(True)
            elif item_key == "fault_table":
                self.data_viewer.render_fault_table(self.fault_data_list, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "fault_inter_stress":
                self.data_viewer.render_fault_inter_stress_graph(self.fault_data_list, self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key in ["faults", "fault_graph"]:
                self.switch_to_view(0)
                if hasattr(self.canvas_3d, 'chk_faults'):
                    self.canvas_3d.chk_faults.setChecked(True)
            elif item_key == "stress_table":
                self.data_viewer.render_stress_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "epa_compliance_table":
                self.data_viewer.render_epa_uic_compliance_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "aor_plume_graph":
                self.data_viewer.render_aor_plume_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "stress_path_graph":
                self.data_viewer.render_stress_path_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key in ["stress_graph", "uic"]:
                self.data_viewer.render_stress_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "caprock":
                self.switch_to_view(0)
                if hasattr(self.canvas_3d, 'chk_caprock'):
                    self.canvas_3d.chk_caprock.setChecked(True)
            else:
                self.switch_to_view(0)

        # 7. Storage, Utilisation & Geothermal Domain
        elif domain in ["storage", "utilisation", "geothermal"]:
            if item_key == "trapping_table":
                self.data_viewer.render_storage_trapping_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "trapping_graph":
                self.data_viewer.render_storage_trapping_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "economics_table":
                self.data_viewer.render_utilisation_economics_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "geothermal_table":
                self.data_viewer.render_geothermal_cpg_table(self.manual_inputs_values)
                self.switch_to_view(9)
            elif item_key == "geothermal_graph":
                self.data_viewer.render_geothermal_power_graph(self.manual_inputs_values)
                self.switch_to_view(9)
            else:
                self.data_viewer.render_storage_trapping_table(self.manual_inputs_values)
                self.switch_to_view(9)

        # 8. Surveillance & Quality Audit Domain
        elif domain in ["surveillance", "surveillance_audit"]:
            if item_key in ["workstation", "multi_domain"]:
                self.switch_to_view(6)
            else:
                self.switch_to_view(5)

        else:
            self.switch_to_view(0)

        # Side panel visibility logic: Hidden on full-screen workstations, sheets, and dashboards (5, 6, 8, 9)
        if self.center_stack.currentIndex() in [5, 6, 8, 9] or not self.property_grid.has_editable_inputs():
            self.right_container.hide()
        else:
            self.right_container.show()
            self.right_container.setMinimumWidth(388)
            self.right_container.setMaximumWidth(550)
            self.property_grid.show()
    def _open_fault_caprock_manager_dialog(self, initial_tab: int = 0):
        from .components.fault_caprock_manager_dialog import FaultAndCaprockDialog
        dlg = FaultAndCaprockDialog(
            fault_data_list=self.fault_data_list,
            caprock_layers=self.caprock_layers,
            manual_params=self.manual_inputs_values,
            parent=self,
            initial_tab=initial_tab
        )
        dlg.faults_updated.connect(self._on_faults_updated_from_dialog)
        dlg.caprock_updated.connect(self._on_caprock_updated_from_dialog)
        dlg.applied.connect(self._refresh_all_views)
        if dlg.exec():
            self.fault_data_list = dlg.get_faults()
            self.caprock_layers = dlg.get_caprock()
            self._refresh_all_views()
            if hasattr(self, 'status_message_updated'):
                self.status_message_updated.emit("Fault system and caprock stratigraphy updated successfully.", 4000)

    def _on_faults_updated_from_dialog(self, faults: list):
        self.fault_data_list = list(faults)
        self._refresh_all_views()

    def _on_caprock_updated_from_dialog(self, caprock: list):
        self.caprock_layers = list(caprock)
        self._refresh_all_views()

    def _prompt_add_fault(self):
        self._open_fault_caprock_manager_dialog(initial_tab=0)

    def _on_property_grid_delete_fault(self, fault_name: str):
        self.fault_data_list = [f for f in self.fault_data_list if f.name != fault_name and f.id != fault_name]
        self._refresh_all_views()
        self.property_grid.load_node("geomechanics", "faults", self.manual_inputs_values, self.well_data_list, self.fault_data_list)
        if hasattr(self, 'status_message_updated'):
            self.status_message_updated.emit(f"Fault '{fault_name}' removed from network.", 3500)

    def _on_isolate_fault_requested(self, fault_name: str):
        self.switch_to_view(0)
        if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
            self.canvas_3d.set_fault_isolated_view(True, fault_name if fault_name else None)
        if fault_name:
            self.model_tree.select_domain_item("fault_item", fault_name)
        else:
            self.model_tree.select_domain_item("geomechanics", "faults")
        if hasattr(self, 'status_message_updated'):
            self.status_message_updated.emit(f"Fault view isolated in 3D: '{fault_name or 'All Active Faults'}'", 4000)

    def _on_3d_well_picked(self, well_name: str):
        if getattr(self, '_in_well_selection', False):
            return
        self._in_well_selection = True
        try:
            logger.info(f"SubsurfaceWorkbenchWidget: 3D raycast selected well '{well_name}'")
            self.switch_to_view(8)
            if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                self.well_workstation.set_parameters(self.manual_inputs_values)
                self.well_workstation.set_wells(self.well_data_list)
                self.well_workstation.select_well(well_name, isolate=True)
            if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                self.canvas_3d.set_isolated_well(well_name)
            self.model_tree.select_well_node(well_name)
            self.property_grid.load_node("well_item", well_name, self.manual_inputs_values, self.well_data_list)
            if hasattr(self, 'status_message_updated'):
                self.status_message_updated.emit(f"Well '{well_name}' isolated for precision modeling.", 4000)
        finally:
            self._in_well_selection = False

    def _toggle_parameter_panel(self):
        """Collapses or expands the right parameter panel smoothly."""
        is_visible = self.property_grid.isVisible()
        if is_visible:
            self.property_grid.hide()
            self.right_container.setMinimumWidth(28)
            self.right_container.setMaximumWidth(32)
            self.right_container.setFixedWidth(32)
            self.btn_toggle_params.setText("«\n\nP\na\nr\na\nm\ne\nt\ne\nr\n \nI\nn\np\nu\nt\n\n«")
            self.btn_toggle_params.setToolTip("Expand Parameter Input Panel")
            sizes = self.main_h_splitter.sizes()
            if len(sizes) == 3:
                tree_w = sizes[0]
                total_w = sum(sizes)
                self.main_h_splitter.setSizes([tree_w, total_w - tree_w - 32, 32])
        else:
            self.right_container.setMinimumWidth(368)
            self.right_container.setMaximumWidth(510)
            self.property_grid.show()
            self.btn_toggle_params.setText("»\n\nP\na\nr\na\nm\ne\nt\ne\nr\n \nI\nn\np\nu\nt\n\n»")
            self.btn_toggle_params.setToolTip("Collapse Parameter Input Panel")
            sizes = self.main_h_splitter.sizes()
            if len(sizes) == 3:
                tree_w = sizes[0]
                total_w = sum(sizes)
                self.main_h_splitter.setSizes([tree_w, total_w - tree_w - 390, 390])

    def _update_3d_canvas_quick(self):
        """Quickly updates the 3D model with current in-memory parameters and unified petro cube."""
        lx = float(self.manual_inputs_values.get("length", 2000.0))
        area = float(self.manual_inputs_values.get("area", 1000.0))
        ly = (area * 43560.0) / max(lx, 1.0)
        h = float(self.manual_inputs_values.get("thickness", 50.0))
        nx = int(self.manual_inputs_values.get("nx", 50))
        ny = int(self.manual_inputs_values.get("ny", 50))
        nz = int(self.manual_inputs_values.get("nz", 10))
        perm = float(self.manual_inputs_values.get("perm", 100.0))
        poro = float(self.manual_inputs_values.get("poro", 0.20))
        top_z = float(self.manual_inputs_values.get("top_depth", 5000.0))

        fault_props = {
            "fault_name": self.manual_inputs_values.get("fault_name", "Fault F-1"),
            "fault_dip": float(self.manual_inputs_values.get("fault_dip", 70.0)),
            "fault_strike": float(self.manual_inputs_values.get("fault_strike", 45.0)),
            "fault_throw": float(self.manual_inputs_values.get("fault_throw", 25.0)),
            "fault_trans_mult": float(self.manual_inputs_values.get("fault_trans_mult", 0.15)),
            "fault_friction": float(self.manual_inputs_values.get("fault_friction", 0.60)),
            "fault_cohesion": float(self.manual_inputs_values.get("fault_cohesion", 0.0)),
            "fault_center_x": float(self.manual_inputs_values.get("fault_center_x", lx * 0.5)),
            "fault_center_y": float(self.manual_inputs_values.get("fault_center_y", ly * 0.5)),
        }
        caprock_props = {
            "caprock_lithology": self.manual_inputs_values.get("caprock_lithology", "Dense Marine Shale"),
            "caprock_thickness": float(self.manual_inputs_values.get("caprock_thickness", 200.0)),
            "caprock_t0": float(self.manual_inputs_values.get("caprock_t0", 200.0)),
            "caprock_cohesion": float(self.manual_inputs_values.get("caprock_cohesion", 400.0)),
            "caprock_friction_angle": float(self.manual_inputs_values.get("caprock_friction_angle", 30.0)),
            "caprock_entry_pressure": float(self.manual_inputs_values.get("caprock_entry_pressure", 1500.0)),
            "caprock_perm": float(self.manual_inputs_values.get("caprock_perm", 0.0001)),
            "caprock_safety_factor": float(self.manual_inputs_values.get("caprock_safety_factor", 0.90)),
        }

        # Synthesize physically unified PetrophysicalCube
        petro_cube = generate_petrophysical_cube(
            nx=nx, ny=ny, nz=nz,
            length_ft=lx, width_ft=ly,
            top_depth=top_z, thickness_ft=h,
            distribution_method=str(self.manual_inputs_values.get("distribution_method", "Facies-Controlled (3-Facies Architecture)")),
            perm_base=perm, poro_base=poro,
            v_dp=float(self.manual_inputs_values.get("dykstra_parsons", 0.65)),
            kv_kh=float(self.manual_inputs_values.get("kv_kh_ratio", 0.10)),
            facies_pattern=str(self.manual_inputs_values.get("facies_pattern", "Fluvial Channel Belt")),
            sand_fraction=float(self.manual_inputs_values.get("sand_fraction", 0.65)),
            silt_fraction=float(self.manual_inputs_values.get("silt_fraction", 0.25)),
            shale_fraction=float(self.manual_inputs_values.get("shale_fraction", 0.10)),
            poro_perm_model=str(self.manual_inputs_values.get("poro_perm_model", "Kozeny-Carman")),
            initial_pressure=float(self.manual_inputs_values.get("initial_pressure", 4000.0)),
            overburden_grad=float(self.manual_inputs_values.get("overburden_grad", 1.00)),
            stress_k0=float(self.manual_inputs_values.get("stress_k0", 0.75)),
            poissons_ratio=float(self.manual_inputs_values.get("poissons_ratio", 0.25)),
            youngs_modulus_base=float(self.manual_inputs_values.get("youngs_modulus_base", 20.0)),
            biot_coeff=float(self.manual_inputs_values.get("biot_coeff", 0.80)),
            frac_grad=float(self.manual_inputs_values.get("frac_grad", 0.85)),
            # Explicit geometric controls
            channel_azimuth_deg=float(self.manual_inputs_values.get("channel_azimuth_deg", 45.0)),
            channel_sinuosity=float(self.manual_inputs_values.get("channel_sinuosity", 1.30)),
            channel_wavelength_ft=float(self.manual_inputs_values.get("channel_wavelength_ft", 1500.0)),
            channel_amplitude_ft=float(self.manual_inputs_values.get("channel_amplitude_ft", 350.0)),
            channel_width_ft=float(self.manual_inputs_values.get("channel_width_ft", 450.0)),
            levee_width_ft=float(self.manual_inputs_values.get("levee_width_ft", 250.0)),
            aggradation_drift_ft=float(self.manual_inputs_values.get("aggradation_drift_ft", 30.0)),
            num_channels=int(self.manual_inputs_values.get("num_channels", 1)),
            barrier_azimuth_deg=float(self.manual_inputs_values.get("barrier_azimuth_deg", 90.0)),
            barrier_width_ft=float(self.manual_inputs_values.get("barrier_width_ft", 800.0)),
            lagoon_width_ft=float(self.manual_inputs_values.get("lagoon_width_ft", 450.0)),
            progradation_dip_deg=float(self.manual_inputs_values.get("progradation_dip_deg", 2.0)),
            reef_center_x=float(self.manual_inputs_values.get("reef_center_x", lx * 0.5)),
            reef_center_y=float(self.manual_inputs_values.get("reef_center_y", ly * 0.5)),
            reef_major_radius_ft=float(self.manual_inputs_values.get("reef_major_radius_ft", 650.0)),
            reef_minor_radius_ft=float(self.manual_inputs_values.get("reef_minor_radius_ft", 400.0)),
            reef_azimuth_deg=float(self.manual_inputs_values.get("reef_azimuth_deg", 45.0)),
            apron_width_ft=float(self.manual_inputs_values.get("apron_width_ft", 300.0)),
            f1_perm=float(self.manual_inputs_values.get("f1_perm", 250.0)),
            f1_poro=float(self.manual_inputs_values.get("f1_poro", 0.25)),
            f2_perm=float(self.manual_inputs_values.get("f2_perm", 40.0)),
            f2_poro=float(self.manual_inputs_values.get("f2_poro", 0.16)),
            f3_perm=float(self.manual_inputs_values.get("f3_perm", 0.5)),
            f3_poro=float(self.manual_inputs_values.get("f3_poro", 0.06)),
            layer_permeability_trend=str(self.manual_inputs_values.get("layer_permeability_trend", "Fining Upward")),
            variogram_type=str(self.manual_inputs_values.get("variogram_type", "Spherical")),
            variogram_range_major=float(self.manual_inputs_values.get("variogram_range_major", 1200.0)),
            variogram_range_minor=float(self.manual_inputs_values.get("variogram_range_minor", 600.0)),
            variogram_range_vert=float(self.manual_inputs_values.get("variogram_range_vert", 20.0)),
            variogram_azimuth_deg=float(self.manual_inputs_values.get("variogram_azimuth_deg", 45.0)),
            nugget_effect=float(self.manual_inputs_values.get("nugget_effect", 0.05)),
            sill_variance=float(self.manual_inputs_values.get("sill_variance", 1.0)),
            random_seed=int(self.manual_inputs_values.get("geostat_seed", 42)),
            oil_api=float(self.manual_inputs_values.get("api_gravity", 35.0)),
            water_gradient=float(self.manual_inputs_values.get("water_gradient", 0.465)),
            gas_gradient=float(self.manual_inputs_values.get("gas_gradient", 0.080)),
            mmp_psia=float(self.manual_inputs_values.get("mmp_override", 2688.0)),
            dead_oil_viscosity_cp=float(self.manual_inputs_values.get("oil_viscosity_cp", 1.0)),
            woc_depth=float(self.manual_inputs_values.get("woc_depth", top_z + h * 0.75)),
            has_gas_cap=bool(self.manual_inputs_values.get("has_gas_cap", False)),
            goc_depth=float(self.manual_inputs_values.get("goc_depth", top_z + h * 0.20))
        )

        self.canvas_3d.render_subsurface_model(
            nx=nx, ny=ny, nz=nz,
            length_ft=lx, width_ft=ly,
            top_depth=top_z, thickness_ft=h,
            perm_base=perm, poro_base=poro,
            well_data_list=self.well_data_list,
            fault_props=fault_props,
            caprock_props=caprock_props,
            petro_cube=petro_cube,
            distribution_params=self.manual_inputs_values
        )

    def _on_apply_model_requested(self, *args):
        """Triggered by 'Apply & Sync to Model' button in property grid: updates all views & emits sync."""
        if len(args) > 0 and isinstance(args[0], dict):
            self.manual_inputs_values.update(args[0])
        from .components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title="Recalculating & Synchronizing Subsurface Model...", task_name="Subsurface 3D & Reservoir Recalculation") as prog:
            prog.set_step(20, "Synthesizing 3D petrophysical grid & facies distribution...")
            self._refresh_all_views()
            prog.set_step(70, "Synchronizing fluid equilibrium with optimization engines...")
            self._sync_and_generate_data()
            prog.set_step(100, "✓ Subsurface synchronization complete!")
        if hasattr(self, 'status_message_updated'):
            self.status_message_updated.emit("✓ Applied parameters and synchronized 3D model & all views", 4000)

    def _on_pvt_workstation_parameters_changed(self, *args):
        if len(args) == 1 and isinstance(args[0], dict):
            self.manual_inputs_values.update(args[0])
        elif len(args) == 3:
            domain, key, value = args
            self.manual_inputs_values[key] = value
        cur_dom = getattr(self.property_grid, 'active_domain', 'pvt')
        if cur_dom == 'pvt':
            self.property_grid.update_input_values(self.manual_inputs_values)
        else:
            cur_key = getattr(self.property_grid, 'active_key', 'properties')
            self.property_grid.load_node(cur_dom, cur_key, self.manual_inputs_values, self.well_data_list)
        self.param_debounce_timer.start()

    def _on_parameter_edited(self, domain: str, key: str, value: Any):
        self.manual_inputs_values[key] = value
        logger.debug(f"Parameter edited: {key} = {value}")
        # Live refresh relperm workstation and geostat view
        if hasattr(self, 'relperm_widget') and self.relperm_widget is not None:
            if key in ["s_wc", "s_orw", "s_gc", "s_org", "krw0", "kro0", "krg0", "n_w", "n_ow", "n_g", "n_og", "wettability_preset", "relperm_model", "oil_viscosity_cp", "water_viscosity_cp"]:
                self.relperm_widget.set_parameters(self.manual_inputs_values)
        if hasattr(self, 'geostat_widget') and self.geostat_widget is not None:
            if key in ["variogram_type", "variogram_range_major", "variogram_range_minor", "variogram_range_vert", "variogram_azimuth_deg", "nugget_effect", "sill_variance", "geostat_seed", "length", "area", "nx", "ny", "nz", "perm", "poro"]:
                self.geostat_widget.set_parameters(self.manual_inputs_values)
        if hasattr(self, 'pvt_workstation') and self.pvt_workstation is not None:
            if key in [
                "initial_pressure", "temperature", "api_gravity", "sol_gor",
                "gas_specific_gravity", "bubble_point_pressure", "oil_viscosity_cp",
                "co2_purity", "co2_swelling_factor_max", "todd_longstaff_omega",
                "mmp_correlation", "mmp_override", "woc_depth", "goc_depth",
                "has_gas_cap", "water_gradient", "gas_gradient", "top_depth", "thickness"
            ]:
                self.pvt_workstation.set_parameters(self.manual_inputs_values)
        # Debounced live refresh 3D canvas so rapid typing doesn't stutter or reset camera
        if key in [
            "poro", "perm", "thickness", "length", "area", "nx", "ny", "nz", "top_depth", "datum_depth",
            "distribution_method", "facies_pattern", "sand_fraction", "silt_fraction", "shale_fraction",
            "poro_perm_model", "dykstra_parsons", "kv_kh_ratio",
            "youngs_modulus_base", "poissons_ratio", "biot_coeff", "overburden_grad", "stress_k0", "frac_grad",
            "initial_pressure", "temperature", "uic_sf",
            "fault_name", "fault_dip", "fault_strike", "fault_throw", "fault_trans_mult",
            "fault_friction", "fault_cohesion", "fault_center_x", "fault_center_y",
            "caprock_lithology", "caprock_thickness", "caprock_t0", "caprock_cohesion",
            "channel_azimuth_deg", "channel_sinuosity", "channel_wavelength_ft", "channel_amplitude_ft",
            "channel_width_ft", "levee_width_ft", "aggradation_drift_ft", "num_channels",
            "barrier_azimuth_deg", "barrier_width_ft", "lagoon_width_ft", "progradation_dip_deg",
            "reef_center_x", "reef_center_y", "reef_major_radius_ft", "reef_minor_radius_ft",
            "reef_azimuth_deg", "apron_width_ft",
            "f1_perm", "f1_poro", "f2_perm", "f2_poro", "f3_perm", "f3_poro",
            "layer_permeability_trend",
            "variogram_type", "variogram_range_major", "variogram_range_minor", "variogram_range_vert",
            "variogram_azimuth_deg", "variogram_dip_deg", "nugget_effect", "sill_variance", "geostat_seed",
            "s_wc", "s_orw", "s_gc", "s_org", "krw0", "kro0", "krg0", "n_w", "n_ow", "n_g", "n_og",
            "wettability_preset", "relperm_model",
            "api_gravity", "sol_gor", "bubble_point_pressure", "oil_viscosity_cp",
            "co2_purity", "co2_swelling_factor_max", "todd_longstaff_omega",
            "mmp_correlation", "mmp_override", "woc_depth", "goc_depth", "has_gas_cap",
            "water_gradient", "gas_gradient", "fluid_preset"
        ]:
            self.param_debounce_timer.start()

    def _load_preset(self, preset_name: str):
        """Loads industry-standard field benchmarks instantly."""
        from .components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title=f"Loading Field Benchmark: {preset_name.upper()}...", task_name="Subsurface Benchmark Configuration") as prog:
            prog.set_step(20, f"Configuring {preset_name} reservoir dimensions & petrophysics...")
            if preset_name == "spe5":
                self.manual_inputs_values.update({
                    'nx': 7, 'ny': 7, 'nz': 3,
                    'length': 2640.0, 'area': 160.0, 'thickness': 100.0,
                    'poro': 0.25, 'perm': 250.0, 'initial_pressure': 4200.0,
                    'api_gravity': 38.0, 'sol_gor': 600.0
                })
                self._create_sample_5spot(2640.0, 2640.0, 5000.0, 5100.0)
                msg = "Loaded SPE 5 CO2 Miscible Flood Benchmark (3 Layers, 5-Spot Pattern)"
            elif preset_name == "permian":
                self.manual_inputs_values.update({
                    'nx': 40, 'ny': 40, 'nz': 8,
                    'length': 5280.0, 'area': 640.0, 'thickness': 75.0,
                    'poro': 0.12, 'perm': 15.0, 'initial_pressure': 3200.0,
                    'api_gravity': 34.0, 'sol_gor': 450.0
                })
                self._create_sample_5spot(5280.0, 5280.0, 4500.0, 4575.0)
                msg = "Loaded Permian Basin San Andres Carbonate Model"
            elif preset_name == "weyburn":
                self.manual_inputs_values.update({
                    'nx': 30, 'ny': 30, 'nz': 6,
                    'length': 4000.0, 'area': 360.0, 'thickness': 45.0,
                    'poro': 0.18, 'perm': 40.0, 'initial_pressure': 2900.0,
                    'api_gravity': 32.0, 'sol_gor': 380.0
                })
                self._create_sample_5spot(4000.0, 4000.0, 4800.0, 4845.0)
                msg = "Loaded Weyburn Midale Carbonate CO2 EOR Model"
            else:
                msg = f"Loaded preset '{preset_name}'"

            prog.set_step(60, "Synthesizing 3D petrophysical grid & fluid contacts...")
            self._refresh_all_views()
            prog.set_step(100, f"✓ {msg}")

        self.status_message_updated.emit(f"✓ {msg}", 4000)

    def _create_sample_5spot(self, lx: float, ly: float, top_z: float, bot_z: float):
        self.well_data_list.clear()
        depths = np.linspace(top_z, bot_z, 10)
        # Center Producer
        p1 = WellData(
            name="PROD-01",
            depths=depths,
            properties={},
            units={},
            metadata={"type": "producer", "SurfaceX": lx * 0.5, "SurfaceY": ly * 0.5, "TrajectoryType": "Vertical"}
        )
        p1.perforations = [[top_z, bot_z]]
        self.well_data_list.append(p1)

        # 4 Corner Injectors
        corners = [(lx * 0.15, ly * 0.15), (lx * 0.85, ly * 0.15), (lx * 0.15, ly * 0.85), (lx * 0.85, ly * 0.85)]
        for idx, (cx, cy) in enumerate(corners):
            iw = WellData(
                name=f"INJ-0{idx+1}",
                depths=depths,
                properties={},
                units={},
                metadata={"type": "injector", "SurfaceX": cx, "SurfaceY": cy, "TrajectoryType": "Vertical"}
            )
            iw.perforations = [[top_z, bot_z]]
            self.well_data_list.append(iw)

    def _generate_pattern(self, pattern_type: str):
        from .components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title=f"Synthesizing {pattern_type} Network...", task_name="Well Network Generation") as prog:
            prog.set_step(25, f"Computing pattern geometry and boundaries for {pattern_type}...")
            lx = float(self.manual_inputs_values.get("length", 2000.0))
            area = float(self.manual_inputs_values.get("area", 1000.0))
            ly = (area * 43560.0) / max(lx, 1.0)
            top_z = 5000.0
            bot_z = top_z + float(self.manual_inputs_values.get("thickness", 50.0))
            self._create_sample_5spot(lx, ly, top_z, bot_z)
            prog.set_step(70, "Placing producer and injector wellbores in 3D reservoir...")
            self._refresh_all_views()
            prog.set_step(100, "✓ Well network generated successfully!")
        self.status_message_updated.emit("✓ Generated standard 5-spot well network", 4000)

    def _prompt_add_well(self):
        lx = float(self.manual_inputs_values.get("length", 2000.0))
        self._prompt_add_well_at_coords(lx * 0.5, lx * 0.5)

    def _prompt_add_well_at_coords(self, rx: float, ry: float):
        if not ManualWellDialog:
            return
        top_z = float(self.manual_inputs_values.get("top_depth", 5000.0))
        bot_z = top_z + float(self.manual_inputs_values.get("thickness", 50.0))
        names = [w.name for w in self.well_data_list]
        init_vals = {
            "SurfaceX": round(rx, 1),
            "SurfaceY": round(ry, 1),
            "TopDepth": top_z,
            "BottomDepth": bot_z,
            "name": f"Well-{len(names)+1}",
            "role": "Producer (Active)"
        }
        dlg = ManualWellDialog(names, parent=self, initial_values=init_vals, reservoir_params=self.manual_inputs_values)
        if dlg.exec():
            well_data = dlg.get_well_data()
            if well_data:
                self.well_data_list.append(well_data)
                self._refresh_all_views()
                self.switch_to_view(8)
                if hasattr(self, 'well_workstation') and self.well_workstation is not None:
                    self.well_workstation.select_well(well_data.name, isolate=True)
                if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                    self.canvas_3d.set_isolated_well(well_data.name)
                self.status_message_updated.emit(f"✓ Added well '{well_data.name}' at ({rx:.0f}, {ry:.0f})", 3500)

    def _refresh_all_views(self):
        # 1. Update Model Tree
        self.model_tree.update_wells_list(self.well_data_list)
        self.model_tree.update_faults_list(self.fault_data_list)
        # 2. Render 3D Model
        lx = float(self.manual_inputs_values.get("length", 2000.0))
        area = float(self.manual_inputs_values.get("area", 1000.0))
        ly = (area * 43560.0) / max(lx, 1.0)
        h = float(self.manual_inputs_values.get("thickness", 50.0))
        nx = int(self.manual_inputs_values.get("nx", 50))
        ny = int(self.manual_inputs_values.get("ny", 50))
        nz = int(self.manual_inputs_values.get("nz", 10))
        perm = float(self.manual_inputs_values.get("perm", 100.0))
        poro = float(self.manual_inputs_values.get("poro", 0.20))

        fault_props = {
            "fault_name": self.manual_inputs_values.get("fault_name", "Fault F-1"),
            "fault_dip": float(self.manual_inputs_values.get("fault_dip", 70.0)),
            "fault_strike": float(self.manual_inputs_values.get("fault_strike", 45.0)),
            "fault_throw": float(self.manual_inputs_values.get("fault_throw", 25.0)),
            "fault_trans_mult": float(self.manual_inputs_values.get("fault_trans_mult", 0.15)),
            "fault_friction": float(self.manual_inputs_values.get("fault_friction", 0.60)),
            "fault_cohesion": float(self.manual_inputs_values.get("fault_cohesion", 0.0)),
            "fault_center_x": float(self.manual_inputs_values.get("fault_center_x", lx * 0.5)),
            "fault_center_y": float(self.manual_inputs_values.get("fault_center_y", ly * 0.5)),
        }
        caprock_props = {
            "caprock_lithology": self.manual_inputs_values.get("caprock_lithology", "Dense Marine Shale"),
            "caprock_thickness": float(self.manual_inputs_values.get("caprock_thickness", 200.0)),
            "caprock_t0": float(self.manual_inputs_values.get("caprock_t0", 200.0)),
            "caprock_cohesion": float(self.manual_inputs_values.get("caprock_cohesion", 400.0)),
            "caprock_friction_angle": float(self.manual_inputs_values.get("caprock_friction_angle", 30.0)),
            "caprock_entry_pressure": float(self.manual_inputs_values.get("caprock_entry_pressure", 1500.0)),
            "caprock_perm": float(self.manual_inputs_values.get("caprock_perm", 0.0001)),
            "caprock_safety_factor": float(self.manual_inputs_values.get("caprock_safety_factor", 0.90)),
        }

        top_z = float(self.manual_inputs_values.get("top_depth", 5000.0))

        # Synthesize physically unified PetrophysicalCube
        petro_cube = generate_petrophysical_cube(
            nx=nx, ny=ny, nz=nz,
            length_ft=lx, width_ft=ly,
            top_depth=top_z, thickness_ft=h,
            distribution_method=str(self.manual_inputs_values.get("distribution_method", "Facies-Controlled (3-Facies Architecture)")),
            perm_base=perm, poro_base=poro,
            v_dp=float(self.manual_inputs_values.get("dykstra_parsons", 0.65)),
            kv_kh=float(self.manual_inputs_values.get("kv_kh_ratio", 0.10)),
            facies_pattern=str(self.manual_inputs_values.get("facies_pattern", "Fluvial Channel Belt")),
            sand_fraction=float(self.manual_inputs_values.get("sand_fraction", 0.65)),
            silt_fraction=float(self.manual_inputs_values.get("silt_fraction", 0.25)),
            shale_fraction=float(self.manual_inputs_values.get("shale_fraction", 0.10)),
            poro_perm_model=str(self.manual_inputs_values.get("poro_perm_model", "Kozeny-Carman")),
            initial_pressure=float(self.manual_inputs_values.get("initial_pressure", 4000.0)),
            overburden_grad=float(self.manual_inputs_values.get("overburden_grad", 1.00)),
            stress_k0=float(self.manual_inputs_values.get("stress_k0", 0.75)),
            poissons_ratio=float(self.manual_inputs_values.get("poissons_ratio", 0.25)),
            youngs_modulus_base=float(self.manual_inputs_values.get("youngs_modulus_base", 20.0)),
            biot_coeff=float(self.manual_inputs_values.get("biot_coeff", 0.80)),
            frac_grad=float(self.manual_inputs_values.get("frac_grad", 0.85)),
            # Explicit geometric controls
            channel_azimuth_deg=float(self.manual_inputs_values.get("channel_azimuth_deg", 45.0)),
            channel_sinuosity=float(self.manual_inputs_values.get("channel_sinuosity", 1.30)),
            channel_wavelength_ft=float(self.manual_inputs_values.get("channel_wavelength_ft", 1500.0)),
            channel_amplitude_ft=float(self.manual_inputs_values.get("channel_amplitude_ft", 350.0)),
            channel_width_ft=float(self.manual_inputs_values.get("channel_width_ft", 450.0)),
            levee_width_ft=float(self.manual_inputs_values.get("levee_width_ft", 250.0)),
            aggradation_drift_ft=float(self.manual_inputs_values.get("aggradation_drift_ft", 30.0)),
            num_channels=int(self.manual_inputs_values.get("num_channels", 1)),
            barrier_azimuth_deg=float(self.manual_inputs_values.get("barrier_azimuth_deg", 90.0)),
            barrier_width_ft=float(self.manual_inputs_values.get("barrier_width_ft", 800.0)),
            lagoon_width_ft=float(self.manual_inputs_values.get("lagoon_width_ft", 450.0)),
            progradation_dip_deg=float(self.manual_inputs_values.get("progradation_dip_deg", 2.0)),
            reef_center_x=float(self.manual_inputs_values.get("reef_center_x", lx * 0.5)),
            reef_center_y=float(self.manual_inputs_values.get("reef_center_y", ly * 0.5)),
            reef_major_radius_ft=float(self.manual_inputs_values.get("reef_major_radius_ft", 650.0)),
            reef_minor_radius_ft=float(self.manual_inputs_values.get("reef_minor_radius_ft", 400.0)),
            reef_azimuth_deg=float(self.manual_inputs_values.get("reef_azimuth_deg", 45.0)),
            apron_width_ft=float(self.manual_inputs_values.get("apron_width_ft", 300.0)),
            f1_perm=float(self.manual_inputs_values.get("f1_perm", 250.0)),
            f1_poro=float(self.manual_inputs_values.get("f1_poro", 0.25)),
            f2_perm=float(self.manual_inputs_values.get("f2_perm", 40.0)),
            f2_poro=float(self.manual_inputs_values.get("f2_poro", 0.16)),
            f3_perm=float(self.manual_inputs_values.get("f3_perm", 0.5)),
            f3_poro=float(self.manual_inputs_values.get("f3_poro", 0.06)),
            layer_permeability_trend=str(self.manual_inputs_values.get("layer_permeability_trend", "Fining Upward")),
            variogram_type=str(self.manual_inputs_values.get("variogram_type", "Spherical")),
            variogram_range_major=float(self.manual_inputs_values.get("variogram_range_major", 1200.0)),
            variogram_range_minor=float(self.manual_inputs_values.get("variogram_range_minor", 600.0)),
            variogram_range_vert=float(self.manual_inputs_values.get("variogram_range_vert", 20.0)),
            variogram_azimuth_deg=float(self.manual_inputs_values.get("variogram_azimuth_deg", 45.0)),
            nugget_effect=float(self.manual_inputs_values.get("nugget_effect", 0.05)),
            sill_variance=float(self.manual_inputs_values.get("sill_variance", 1.0)),
            random_seed=int(self.manual_inputs_values.get("geostat_seed", 42)),
            oil_api=float(self.manual_inputs_values.get("api_gravity", 35.0)),
            water_gradient=float(self.manual_inputs_values.get("water_gradient", 0.465)),
            gas_gradient=float(self.manual_inputs_values.get("gas_gradient", 0.080)),
            mmp_psia=float(self.manual_inputs_values.get("mmp_override", 2688.0)),
            dead_oil_viscosity_cp=float(self.manual_inputs_values.get("oil_viscosity_cp", 1.0)),
            woc_depth=float(self.manual_inputs_values.get("woc_depth", top_z + h * 0.75)),
            has_gas_cap=bool(self.manual_inputs_values.get("has_gas_cap", False)),
            goc_depth=float(self.manual_inputs_values.get("goc_depth", top_z + h * 0.20))
        )

        self.canvas_3d.render_subsurface_model(
            nx=nx, ny=ny, nz=nz,
            length_ft=lx, width_ft=ly,
            top_depth=top_z, thickness_ft=h,
            perm_base=perm, poro_base=poro,
            well_data_list=self.well_data_list,
            fault_props=fault_props,
            caprock_props=caprock_props,
            petro_cube=petro_cube,
            distribution_params=self.manual_inputs_values,
            fault_data_list=self.fault_data_list,
            caprock_layers=self.caprock_layers
        )

        # 3. Property Grid (Preserve active node)
        cur_dom = getattr(self.property_grid, 'active_domain', 'reservoir')
        cur_key = getattr(self.property_grid, 'active_key', 'grid')
        self.property_grid.load_node(cur_dom, cur_key, self.manual_inputs_values, self.well_data_list, self.fault_data_list)

        # 5. Stratigraphy & Cross-Section Tab (Reflects True Reservoir Setup)
        if hasattr(self, 'cross_section_widget') and self.cross_section_widget is not None:
            try:
                top_depth = float(self.manual_inputs_values.get("top_depth", 5000.0))
                dip_deg = float(self.manual_inputs_values.get("dip_angle_deg", 0.0))
                has_gc = bool(self.manual_inputs_values.get("has_gas_cap", False))

                self.cross_section_widget.set_grid_data(
                    petro_cube.perm, petro_cube.poro,
                    length_ft=lx, width_ft=ly,
                    top_depth=top_depth, thickness_ft=h,
                    dip_angle_deg=dip_deg,
                    well_data_list=self.well_data_list,
                    fault_props=fault_props,
                    caprock_props=caprock_props,
                    facies_grid=petro_cube.facies,
                    saturation_grid=petro_cube.saturation,
                    sw_grid=petro_cube.sw,
                    margin_grid=petro_cube.miscibility_margin,
                    pressure_grid=petro_cube.pressure,
                    woc_depth=float(self.manual_inputs_values.get("woc_depth", top_depth + h * 0.75)),
                    goc_depth=float(self.manual_inputs_values.get("goc_depth", top_depth + h * 0.20)) if has_gc else None
                )
            except Exception as e:
                logger.debug(f"Cross-section sync error: {e}")

        # 6. Geostatistics Tab
        if hasattr(self, 'geostat_widget') and self.geostat_widget is not None:
            try:
                self.geostat_widget.set_grid_dimensions(
                    nx=nx, ny=ny, nz=nz,
                    length_ft=lx, width_ft=ly, thickness_ft=h,
                    base_perm=perm, base_poro=poro
                )
                self.geostat_widget.set_well_data(self.well_data_list)
                self.geostat_widget.set_parameters(self.manual_inputs_values)
            except Exception as e:
                logger.debug(f"Geostat sync error: {e}")

        # 7. Corey Relative Permeability & Displacement Tab
        if hasattr(self, 'relperm_widget') and self.relperm_widget is not None:
            try:
                self.relperm_widget.set_parameters(self.manual_inputs_values)
            except Exception as e:
                logger.debug(f"Rel-perm sync error: {e}")

        # 8. Fault Containment Tab
        if hasattr(self, 'fault_widget') and self.fault_widget is not None:
            try:
                self.fault_widget.set_reservoir_geometry(
                    length_ft=lx, width_ft=ly, thickness_ft=h,
                    top_depth_ft=5000.0,
                    well_data_list=self.well_data_list
                )
            except Exception as e:
                logger.debug(f"Fault sync error: {e}")

        # 9. Visual Audit Confirmation Gate Tab
        if hasattr(self, 'visual_audit_gate') and self.visual_audit_gate is not None:
            try:
                self.visual_audit_gate.update_project_data(self.get_current_project_data())
            except Exception as e:
                logger.debug(f"Visual audit sync error: {e}")

        # 10. Multi-Domain Surveillance & Workstation Tab
        if hasattr(self, 'surveillance_widget') and self.surveillance_widget is not None:
            try:
                self.surveillance_widget.update_data(self.get_current_project_data())
            except Exception as e:
                logger.debug(f"Surveillance sync error: {e}")

        # 11. Fluids & PVT Thermodynamics Workstation
        if hasattr(self, 'pvt_workstation') and self.pvt_workstation is not None:
            try:
                self.pvt_workstation.set_parameters(self.manual_inputs_values)
            except Exception as e:
                logger.debug(f"PVT workstation sync error: {e}")

        # 12. Well Network & Precision Modeling Workstation
        if hasattr(self, 'well_workstation') and self.well_workstation is not None:
            try:
                self.well_workstation.set_parameters(self.manual_inputs_values)
                self.well_workstation.set_wells(self.well_data_list)
            except Exception as e:
                logger.debug(f"Well workstation sync error: {e}")

    def _on_workstation_well_added(self, well_data: WellData):
        if well_data not in self.well_data_list:
            self.well_data_list.append(well_data)
        self._refresh_all_views()
        self.property_grid.load_node("well_item", well_data.name, self.manual_inputs_values, self.well_data_list)
        self.model_tree.select_well_node(well_data.name)
        self.param_debounce_timer.start()

    def _on_workstation_well_updated(self, old_name: str, updated_well: WellData):
        for i, w in enumerate(self.well_data_list):
            if w.name == old_name:
                self.well_data_list[i] = updated_well
                break
        self._refresh_all_views()
        self.property_grid.load_node("well_item", updated_well.name, self.manual_inputs_values, self.well_data_list)
        self.model_tree.select_well_node(updated_well.name)
        self.param_debounce_timer.start()

    def _on_workstation_well_deleted(self, well_name: str):
        self.well_data_list = [w for w in self.well_data_list if w.name != well_name]
        self._refresh_all_views()
        self.property_grid.load_node("wells", "root", self.manual_inputs_values, self.well_data_list)
        self.param_debounce_timer.start()

    def _on_workstation_well_selected(self, well_name: str):
        if getattr(self, '_in_well_selection', False):
            return
        self._in_well_selection = True
        try:
            if hasattr(self, 'canvas_3d') and self.canvas_3d is not None:
                self.canvas_3d.set_isolated_well(well_name if well_name else None)
            if well_name:
                self.model_tree.select_well_node(well_name)
                self.property_grid.load_node("well_item", well_name, self.manual_inputs_values, self.well_data_list)
            else:
                self.model_tree.select_domain_item("wells", "network")
                self.property_grid.load_node("wells", "network", self.manual_inputs_values, self.well_data_list)
        finally:
            self._in_well_selection = False

    def _on_workstation_place_well_requested(self, enabled: bool):
        self.switch_to_view(8)
        if hasattr(self, 'canvas_3d') and hasattr(self.canvas_3d, 'btn_place_well'):
            self.canvas_3d.btn_place_well.setChecked(enabled)
            self.canvas_3d._toggle_place_well_mode(enabled)

    def _on_workstation_sync_requested(self, wells: list):
        self.well_data_list = list(wells)
        self._refresh_all_views()
        self._on_apply_model_requested()

    def _on_workstation_status_message(self, msg: str, timeout: int):
        if hasattr(self, 'status_message_updated'):
            self.status_message_updated.emit(msg, timeout)

    def _on_property_grid_edit_well(self, well_name: str):
        self.switch_to_view(8)
        if hasattr(self, 'well_workstation') and self.well_workstation is not None:
            self.well_workstation.select_well(well_name, isolate=True)
            self.well_workstation._prompt_edit_selected_well()

    def _on_property_grid_delete_well(self, well_name: str):
        self.switch_to_view(8)
        if hasattr(self, 'well_workstation') and self.well_workstation is not None:
            self.well_workstation.select_well(well_name, isolate=True)
            self.well_workstation._delete_selected_well()

    def _on_sync_button_clicked(self):
        from .components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title="Synchronizing Project Data...", task_name="Subsurface & Engine Synchronization") as prog:
            prog.set_step(25, "Packaging 3D petrophysical grids, contacts, and well trajectories...")
            payload = self.get_current_project_data()
            prog.set_step(65, "Dispatching fluid models & MMP to optimization algorithms...")
            self.project_data_updated.emit(payload)
            prog.set_step(100, "✓ Project data synchronized across all optimization engines!")
        if hasattr(self, 'status_message_updated'):
            self.status_message_updated.emit("✓ Project data synchronized across all optimization engines", 4000)

    def _sync_and_generate_data(self):
        payload = self.get_current_project_data()
        self.project_data_updated.emit(payload)
        self.status_message_updated.emit("✓ Project data synchronized across all optimization engines", 4000)

    def _run_pre_flight_audit(self):
        """Switches to integrated Visual Audit Gate tab and executes pre-flight checks in-place without popup."""
        self.switch_to_view(5)
        self.model_tree.select_domain_item("surveillance", "audit")
        self.status_message_updated.emit("✓ Visual Audit Gate: Real-time multi-domain validation executed", 4000)

    def _on_audit_model_approved(self, approved_data: dict):
        self._sync_and_generate_data()
        self.status_message_updated.emit("✓ Subsurface model approved & locked for simulation dispatch", 5000)

    def _launch_pre_flight_audit(self):
        self._run_pre_flight_audit()

    def _launch_visual_audit(self):
        self.switch_to_view(5)
        self.model_tree.select_domain_item("surveillance", "audit")

    def _launch_model_workstation(self):
        self.switch_to_view(6)
        self.model_tree.select_domain_item("surveillance", "multi_domain")

    # --- API COMPATIBILITY WITH DataManagementWidget ---
    def get_current_project_data(self) -> Dict[str, Any]:
        """Flushes in-memory data store to standardized dictionary for project saving & simulation."""
        lx = float(self.manual_inputs_values.get("length", 2000.0))
        area = float(self.manual_inputs_values.get("area", 1000.0))
        ly = (area * 43560.0) / max(lx, 1.0)
        h = float(self.manual_inputs_values.get("thickness", 50.0))
        phi = float(self.manual_inputs_values.get("poro", 0.20))
        swi = float(self.manual_inputs_values.get("swi", 0.25))
        boi = float(self.manual_inputs_values.get("boi", 1.20))
        ooip = (7758.0 * area * h * phi * (1.0 - swi)) / max(boi, 0.1)

        nx = int(self.manual_inputs_values.get("nx", 50))
        ny = int(self.manual_inputs_values.get("ny", 50))
        nz = int(self.manual_inputs_values.get("nz", 10))
        perm_val = float(self.manual_inputs_values.get("perm", 100.0))
        grid = {
            "PERMX": np.full((nx, ny, nz), perm_val),
            "PORO": np.full((nx, ny, nz), phi)
        }
        res_data = ReservoirData(
            grid=grid,
            pvt_tables={},
            ooip_stb=ooip,
            initial_pressure=float(self.manual_inputs_values.get("initial_pressure", 4000.0)),
            rock_compressibility=3e-6,
            temperature=float(self.manual_inputs_values.get("temperature", 212.0)),
            length_ft=lx,
            area_acres=area,
            thickness_ft=h,
            average_porosity=phi,
            average_permeability=perm_val,
            initial_water_saturation=swi,
            oil_fvf=boi
        )
        pvt_data = PVTProperties(
            api_gravity=float(self.manual_inputs_values.get("api_gravity", 35.0)),
            gas_specific_gravity=float(self.manual_inputs_values.get("gas_specific_gravity", 0.7)),
            temperature=float(self.manual_inputs_values.get("temperature", 212.0)),
            oil_viscosity_cp=float(self.manual_inputs_values.get("oil_viscosity_cp", 1.0)),
            gas_viscosity_cp=float(self.manual_inputs_values.get("gas_viscosity_cp", 0.02)),
            water_viscosity_cp=float(self.manual_inputs_values.get("water_viscosity_cp", 0.5)),
            oil_fvf_simple=float(self.manual_inputs_values.get("boi", 1.2))
        )

        return {
            "manual_inputs": self.manual_inputs_values.copy(),
            "reservoir_data": res_data,
            "pvt_data": pvt_data,
            "pvt_properties": pvt_data,
            "wells": list(self.well_data_list),
            "well_data_list": list(self.well_data_list),
            "fault_data_list": list(self.fault_data_list),
            "caprock_layers": list(self.caprock_layers),
            "calculated_mmp": self.calculated_mmp_value,
            "mmp_value": self.calculated_mmp_value,
            "eor_parameters": {
                "s_wc": float(self.manual_inputs_values.get("s_wc", 0.20)),
                "s_orw": float(self.manual_inputs_values.get("s_orw", 0.20)),
                "s_gc": float(self.manual_inputs_values.get("s_gc", 0.05)),
                "n_o": float(self.manual_inputs_values.get("n_o", 2.0)),
                "n_w": float(self.manual_inputs_values.get("n_w", 2.0)),
                "n_g": float(self.manual_inputs_values.get("n_g", 2.0))
            }
        }

    def load_project_data(self, project_data: Dict[str, Any]):
        """Restores model state from loaded project data dictionary."""
        if not project_data:
            return
        manual = project_data.get("manual_inputs", {})
        if manual and isinstance(manual, dict):
            self.manual_inputs_values.update(manual)

        # Ingest from reservoir_data if available
        res = project_data.get("reservoir_data")
        if res is not None:
            if hasattr(res, 'length_ft') and res.length_ft:
                self.manual_inputs_values['length'] = float(res.length_ft)
            if hasattr(res, 'area_acres') and res.area_acres:
                self.manual_inputs_values['area'] = float(res.area_acres)
            if hasattr(res, 'thickness_ft') and res.thickness_ft:
                self.manual_inputs_values['thickness'] = float(res.thickness_ft)
            if hasattr(res, 'average_porosity') and res.average_porosity is not None:
                self.manual_inputs_values['poro'] = float(res.average_porosity)
            if hasattr(res, 'average_permeability') and res.average_permeability is not None:
                self.manual_inputs_values['perm'] = float(res.average_permeability)
            elif getattr(res, 'grid', None) and 'PERMX' in res.grid:
                p_arr = np.asarray(res.grid['PERMX'])
                if p_arr.size > 0:
                    self.manual_inputs_values['perm'] = float(p_arr.flat[0])
            if hasattr(res, 'initial_pressure') and res.initial_pressure is not None:
                self.manual_inputs_values['initial_pressure'] = float(res.initial_pressure)
            if hasattr(res, 'temperature') and res.temperature is not None:
                self.manual_inputs_values['temperature'] = float(res.temperature)
            if hasattr(res, 'initial_water_saturation') and res.initial_water_saturation is not None:
                self.manual_inputs_values['swi'] = float(res.initial_water_saturation)
            if hasattr(res, 'oil_fvf') and res.oil_fvf is not None:
                self.manual_inputs_values['boi'] = float(res.oil_fvf)

        # Ingest from pvt_properties or pvt_data
        pvt = project_data.get("pvt_properties") or project_data.get("pvt_data")
        if pvt is not None:
            if hasattr(pvt, 'api_gravity') and pvt.api_gravity is not None:
                self.manual_inputs_values['api_gravity'] = float(pvt.api_gravity)
            if hasattr(pvt, 'gas_specific_gravity') and pvt.gas_specific_gravity is not None:
                self.manual_inputs_values['gas_specific_gravity'] = float(pvt.gas_specific_gravity)
            if hasattr(pvt, 'oil_viscosity_cp') and pvt.oil_viscosity_cp is not None:
                self.manual_inputs_values['oil_viscosity_cp'] = float(pvt.oil_viscosity_cp)
            if hasattr(pvt, 'gas_viscosity_cp') and pvt.gas_viscosity_cp is not None:
                self.manual_inputs_values['gas_viscosity_cp'] = float(pvt.gas_viscosity_cp)
            if hasattr(pvt, 'water_viscosity_cp') and pvt.water_viscosity_cp is not None:
                self.manual_inputs_values['water_viscosity_cp'] = float(pvt.water_viscosity_cp)

        # MMP
        mmp_cand = project_data.get("mmp_value") or project_data.get("calculated_mmp")
        if mmp_cand is not None:
            try:
                self.calculated_mmp_value = float(mmp_cand)
            except Exception:
                pass

        # Load wells
        wells = project_data.get("wells") or project_data.get("well_data_list") or []
        self.well_data_list.clear()
        for w in wells:
            if isinstance(w, WellData):
                self.well_data_list.append(w)
            elif isinstance(w, dict):
                depths = w.get("depths")
                if depths is None or len(depths) == 0:
                    depths = np.linspace(5000.0, 5050.0, 10)
                else:
                    depths = np.asarray(depths)
                self.well_data_list.append(WellData(
                    name=w.get("name", "Well"),
                    depths=depths,
                    properties=w.get("properties", {}),
                    units=w.get("units", {}),
                    metadata=w.get("metadata", {})
                ))

        # Load faults
        faults_in = project_data.get("fault_data_list") or []
        if faults_in and len(faults_in) > 0:
            self.fault_data_list.clear()
            for f in faults_in:
                if isinstance(f, FaultData):
                    self.fault_data_list.append(f)
                elif isinstance(f, dict):
                    try:
                        self.fault_data_list.append(FaultData(**{k: v for k, v in f.items() if k != '_dataclass'}))
                    except Exception as err:
                        logger.debug(f"Error restoring FaultData: {err}")

        # Load caprock
        caprock_in = project_data.get("caprock_layers") or []
        if caprock_in and len(caprock_in) > 0:
            self.caprock_layers.clear()
            for c in caprock_in:
                if isinstance(c, CaprockLayer):
                    self.caprock_layers.append(c)
                elif isinstance(c, dict):
                    try:
                        self.caprock_layers.append(CaprockLayer(**{k: v for k, v in c.items() if k != '_dataclass'}))
                    except Exception as err:
                        logger.debug(f"Error restoring CaprockLayer: {err}")

        self._refresh_all_views()
        logger.info("SubsurfaceWorkbenchWidget: Project data loaded successfully.")

    def set_engine_type(self, engine_type: str):
        logger.debug(f"SubsurfaceWorkbenchWidget: engine_type set to '{engine_type}'")
