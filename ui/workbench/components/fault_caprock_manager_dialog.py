"""
Fault and Caprock Manager Dialog.
==================================

Provides a unified, professional window for:
1. Creating, editing, duplicating, and removing structural faults.
2. Managing kinematic parameters, shale gouge ratio (SGR), transmissibility, and slip tendency.
3. Calculating inter-fault geomechanical Coulomb stress transfer (Delta CFS) between multiple faults.
4. Designing multi-layer caprock confining stratigraphy (thickness, mechanical moduli, Pe, kv).
5. Evaluating EPA Class VI hydraulic fracture limits and maximum sustainable buoyant CO2 columns.
"""

from copy import deepcopy
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFormLayout,
    QLineEdit, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox, QSplitter,
    QFrame, QMessageBox, QAbstractItemView
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.data_models import FaultData, CaprockLayer
from ui.workbench.components.subsurface_icons import create_subsurface_icon

logger = logging.getLogger(__name__)


class FaultAndCaprockDialog(QDialog):
    """
    Top-level dialog to create, set up, and update faults and caprock stratigraphy.
    """
    faults_updated = pyqtSignal(list)
    caprock_updated = pyqtSignal(list)
    applied = pyqtSignal()

    def __init__(
        self,
        fault_data_list: Optional[List[FaultData]] = None,
        caprock_layers: Optional[List[CaprockLayer]] = None,
        manual_params: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
        initial_tab: int = 0
    ):
        super().__init__(parent)
        self.setWindowTitle("Structural Fault System & Caprock Confining Manager")
        self.resize(1020, 740)
        self.setMinimumSize(850, 600)

        # Working copies to allow Cancel/Apply semantics
        if fault_data_list and len(fault_data_list) > 0:
            self.faults: List[FaultData] = [deepcopy(f) for f in fault_data_list]
        else:
            self.faults = [
                FaultData(id="F-1", name="Fault F-1 (Major Boundary)", strike=45.0, dip=70.0, throw=55.0, heave=20.0, length=3800.0, center_x=1000.0, center_y=1000.0, shale_gouge_ratio=34.0, transmissibility_multiplier=0.10),
                FaultData(id="F-2", name="Fault F-2 (Synthetic Graben)", strike=45.0, dip=65.0, dip_direction="NW", throw=-40.0, heave=18.6, length=2800.0, center_x=1600.0, center_y=1200.0, shale_gouge_ratio=28.0, transmissibility_multiplier=0.15)
            ]

        if caprock_layers and len(caprock_layers) > 0:
            self.caprocks: List[CaprockLayer] = [deepcopy(c) for c in caprock_layers]
        else:
            self.caprocks = [
                CaprockLayer(name="Unit C1 - Basal Marine Shale", thickness_ft=120.0, lithology="Illite-Smectite Shale", youngs_modulus_gpa=18.5, poissons_ratio=0.28, tensile_strength_psi=250.0, cohesion_psi=450.0, friction_angle_deg=32.0, entry_pressure_psi=2200.0, permeability_nd=10.0),
                CaprockLayer(name="Unit C2 - Intermediate Silt Baffle", thickness_ft=85.0, lithology="Calcite-Cemented Siltstone", youngs_modulus_gpa=24.0, poissons_ratio=0.25, tensile_strength_psi=180.0, cohesion_psi=320.0, friction_angle_deg=30.0, entry_pressure_psi=1450.0, permeability_nd=450.0),
                CaprockLayer(name="Unit C3 - Regional Aquitard", thickness_ft=180.0, lithology="Dense Silty Mudstone", youngs_modulus_gpa=16.0, poissons_ratio=0.30, tensile_strength_psi=140.0, cohesion_psi=260.0, friction_angle_deg=28.0, entry_pressure_psi=950.0, permeability_nd=2200.0)
            ]

        self.manual_params = deepcopy(manual_params) if manual_params else {}
        self.selected_fault_index: int = 0
        self.selected_caprock_index: int = 0

        self._setup_style()
        self._build_ui()
        self.tab_widget.setCurrentIndex(initial_tab)
        self._populate_faults_table()
        self._populate_caprock_table()
        self._load_selected_fault_form()

    def _setup_style(self):
        self.setStyleSheet("""
            QDialog {
                background: #f8fafc;
                color: #0f172a;
                font-family: 'Segoe UI', -apple-system, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                background: #ffffff;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0284c7;
                border-bottom: 2px solid #0284c7;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                gridline-color: #f1f5f9;
                selection-background-color: #e0f2fe;
                selection-color: #0284c7;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #f8fafc;
                color: #334155;
                font-weight: 600;
                font-size: 11px;
                padding: 5px;
                border: 1px solid #e2e8f0;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #0369a1;
            }
            QLineEdit, QDoubleSpinBox, QComboBox {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 8px;
                background: #ffffff;
                color: #0f172a;
                font-size: 11px;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #0284c7;
            }
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #f1f5f9;
                border-color: #94a3b8;
            }
            QPushButton#primaryBtn {
                background: #0284c7;
                color: #ffffff;
                border: 1px solid #0369a1;
            }
            QPushButton#primaryBtn:hover {
                background: #0369a1;
            }
            QPushButton#dangerBtn {
                background: #fee2e2;
                color: #dc2626;
                border: 1px solid #fca5a5;
            }
            QPushButton#dangerBtn:hover {
                background: #fecaca;
            }
        """)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Header Title Banner
        hdr_frame = QFrame()
        hdr_frame.setStyleSheet("background: #0f172a; border-radius: 6px; padding: 10px;")
        hdr_layout = QHBoxLayout(hdr_frame)
        hdr_layout.setContentsMargins(10, 4, 10, 4)

        title_lbl = QLabel("Structural Fault Network & Confining Caprock Geomechanics")
        title_lbl.setStyleSheet("color: #38bdf8; font-size: 14px; font-weight: bold;")
        sub_lbl = QLabel("Multi-Fault Kinematic Kinematics, Coulomb Stress Transfer & Class VI Sealing")
        sub_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        hdr_vbox = QVBoxLayout()
        hdr_vbox.setSpacing(2)
        hdr_vbox.addWidget(title_lbl)
        hdr_vbox.addWidget(sub_lbl)
        hdr_layout.addLayout(hdr_vbox)
        hdr_layout.addStretch()

        root_layout.addWidget(hdr_frame)

        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._build_faults_tab(), "Structural Faults (Kinematics & Slip)")
        self.tab_widget.addTab(self._build_inter_stress_tab(), "Inter-Fault Stress Transfer (Coulomb Delta CFS)")
        self.tab_widget.addTab(self._build_caprock_tab(), "Caprock Confining Stratigraphy (Class VI)")
        root_layout.addWidget(self.tab_widget)

        # Dialog Footer Buttons
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.setSpacing(8)

        self.btn_apply = QPushButton("Apply to Model")
        self.btn_apply.setToolTip("Applies changes immediately to 3D canvas and model tree without closing")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        footer_layout.addWidget(self.btn_apply)

        footer_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        footer_layout.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.setObjectName("primaryBtn")
        self.btn_ok.clicked.connect(self._on_ok_clicked)
        footer_layout.addWidget(self.btn_ok)

        root_layout.addLayout(footer_layout)

    # =========================================================================
    # TAB 1: FAULTS MANAGER & KINEMATICS
    # =========================================================================
    def _build_faults_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Left Splitter: Fault List & Presets
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_hdr = QLabel("Active Faults in Reservoir Model")
        left_hdr.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 11px;")
        left_layout.addWidget(left_hdr)

        self.table_faults = QTableWidget(0, 6)
        self.table_faults.setHorizontalHeaderLabels(["ID", "Name", "Strike", "Dip", "Throw", "Slip Ts"])
        self.table_faults.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_faults.horizontalHeader().setStretchLastSection(True)
        self.table_faults.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_faults.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_faults.verticalHeader().setVisible(False)
        self.table_faults.itemSelectionChanged.connect(self._on_fault_selection_changed)
        left_layout.addWidget(self.table_faults)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_add_fault = QPushButton("+ Add Fault")
        self.btn_add_fault.setObjectName("primaryBtn")
        self.btn_add_fault.clicked.connect(self._on_add_fault)
        btn_row.addWidget(self.btn_add_fault)

        self.btn_dup_fault = QPushButton("Duplicate")
        self.btn_dup_fault.clicked.connect(self._on_duplicate_fault)
        btn_row.addWidget(self.btn_dup_fault)

        self.btn_del_fault = QPushButton("Delete")
        self.btn_del_fault.setObjectName("dangerBtn")
        self.btn_del_fault.clicked.connect(self._on_delete_fault)
        btn_row.addWidget(self.btn_del_fault)

        left_layout.addLayout(btn_row)

        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset System:"))
        self.combo_fault_presets = QComboBox()
        self.combo_fault_presets.addItems([
            "Custom User Defined",
            "Single Major Boundary Fault",
            "Conjugate Graben Fault Pair",
            "Synthetic Normal Step-Faults (3)",
            "En-Echelon Overlapping Relays"
        ])
        self.combo_fault_presets.currentIndexChanged.connect(self._on_apply_fault_preset)
        preset_row.addWidget(self.combo_fault_presets)
        left_layout.addLayout(preset_row)

        # Right Splitter: Detailed Selected Fault Form
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Group 1: Geometry & Kinematics
        grp_geom = QGroupBox("Selected Fault Geometry & Kinematic Throw")
        form_geom = QFormLayout(grp_geom)
        form_geom.setSpacing(6)

        self.edit_f_name = QLineEdit()
        self.edit_f_name.textChanged.connect(self._on_fault_param_changed)
        form_geom.addRow("Fault Name:", self.edit_f_name)

        g_row1 = QHBoxLayout()
        self.spin_f_strike = QDoubleSpinBox()
        self.spin_f_strike.setRange(0.0, 360.0)
        self.spin_f_strike.setSuffix("°")
        self.spin_f_strike.valueChanged.connect(self._on_fault_param_changed)
        g_row1.addWidget(QLabel("Strike Azimuth:"))
        g_row1.addWidget(self.spin_f_strike)

        self.spin_f_dip = QDoubleSpinBox()
        self.spin_f_dip.setRange(15.0, 90.0)
        self.spin_f_dip.setSuffix("°")
        self.spin_f_dip.valueChanged.connect(self._on_fault_param_changed)
        g_row1.addWidget(QLabel("Dip Angle:"))
        g_row1.addWidget(self.spin_f_dip)
        form_geom.addRow(g_row1)

        g_row2 = QHBoxLayout()
        self.spin_f_throw = QDoubleSpinBox()
        self.spin_f_throw.setRange(-300.0, 300.0)
        self.spin_f_throw.setSuffix(" ft")
        self.spin_f_throw.valueChanged.connect(self._on_fault_param_changed)
        g_row2.addWidget(QLabel("Vertical Throw:"))
        g_row2.addWidget(self.spin_f_throw)

        self.spin_f_length = QDoubleSpinBox()
        self.spin_f_length.setRange(500.0, 20000.0)
        self.spin_f_length.setSingleStep(200.0)
        self.spin_f_length.setSuffix(" ft")
        self.spin_f_length.valueChanged.connect(self._on_fault_param_changed)
        g_row2.addWidget(QLabel("Strike Length:"))
        g_row2.addWidget(self.spin_f_length)
        form_geom.addRow(g_row2)

        g_row3 = QHBoxLayout()
        self.spin_f_cx = QDoubleSpinBox()
        self.spin_f_cx.setRange(-10000.0, 50000.0)
        self.spin_f_cx.setSuffix(" ft")
        self.spin_f_cx.valueChanged.connect(self._on_fault_param_changed)
        g_row3.addWidget(QLabel("Center X:"))
        g_row3.addWidget(self.spin_f_cx)

        self.spin_f_cy = QDoubleSpinBox()
        self.spin_f_cy.setRange(-10000.0, 50000.0)
        self.spin_f_cy.setSuffix(" ft")
        self.spin_f_cy.valueChanged.connect(self._on_fault_param_changed)
        g_row3.addWidget(QLabel("Center Y:"))
        g_row3.addWidget(self.spin_f_cy)
        form_geom.addRow(g_row3)

        right_layout.addWidget(grp_geom)

        # Group 2: Petrophysics, Baffle & SGR
        grp_petro = QGroupBox("Fault Zone Petrophysics & Capillary Sealing")
        form_petro = QFormLayout(grp_petro)
        form_petro.setSpacing(6)

        p_row1 = QHBoxLayout()
        self.spin_f_trans = QDoubleSpinBox()
        self.spin_f_trans.setRange(0.00, 1.00)
        self.spin_f_trans.setSingleStep(0.05)
        self.spin_f_trans.valueChanged.connect(self._on_fault_param_changed)
        p_row1.addWidget(QLabel("Transmissibility Mult:"))
        p_row1.addWidget(self.spin_f_trans)

        self.spin_f_damage_w = QDoubleSpinBox()
        self.spin_f_damage_w.setRange(10.0, 500.0)
        self.spin_f_damage_w.setSuffix(" ft")
        self.spin_f_damage_w.valueChanged.connect(self._on_fault_param_changed)
        p_row1.addWidget(QLabel("Damage Zone Width:"))
        p_row1.addWidget(self.spin_f_damage_w)
        form_petro.addRow(p_row1)

        p_row2 = QHBoxLayout()
        self.spin_f_sgr = QDoubleSpinBox()
        self.spin_f_sgr.setRange(0.0, 80.0)
        self.spin_f_sgr.setSuffix("%")
        self.spin_f_sgr.valueChanged.connect(self._on_fault_param_changed)
        p_row2.addWidget(QLabel("Shale Gouge Ratio (SGR):"))
        p_row2.addWidget(self.spin_f_sgr)

        self.lbl_sgr_eval = QLabel("Baffle (Continuous Shale Gouge)")
        self.lbl_sgr_eval.setStyleSheet("color: #0369a1; font-weight: bold;")
        p_row2.addWidget(self.lbl_sgr_eval)
        form_petro.addRow(p_row2)

        right_layout.addWidget(grp_petro)

        # Group 3: Geomechanics & Slip Tendency
        grp_slip = QGroupBox("Geomechanical Reactivation & Slip Tendency (Ts)")
        form_slip = QFormLayout(grp_slip)
        form_slip.setSpacing(6)

        s_row1 = QHBoxLayout()
        self.spin_f_mu = QDoubleSpinBox()
        self.spin_f_mu.setRange(0.20, 1.00)
        self.spin_f_mu.setSingleStep(0.05)
        self.spin_f_mu.setValue(0.60)
        self.spin_f_mu.valueChanged.connect(self._on_fault_param_changed)
        s_row1.addWidget(QLabel("Friction Coeff (μ):"))
        s_row1.addWidget(self.spin_f_mu)

        self.spin_f_cohesion = QDoubleSpinBox()
        self.spin_f_cohesion.setRange(0.0, 500.0)
        self.spin_f_cohesion.setSuffix(" psi")
        self.spin_f_cohesion.valueChanged.connect(self._on_fault_param_changed)
        s_row1.addWidget(QLabel("Fault Cohesion:"))
        s_row1.addWidget(self.spin_f_cohesion)
        form_slip.addRow(s_row1)

        # Live calculation status cards
        calc_box = QFrame()
        calc_box.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 6px;")
        calc_layout = QHBoxLayout(calc_box)
        calc_layout.setSpacing(12)

        self.lbl_ts_val = QLabel("Ts: 0.42")
        self.lbl_ts_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #0284c7;")
        calc_layout.addWidget(self.lbl_ts_val)

        self.lbl_react_p = QLabel("Delta P_crit: +1,240 psi")
        self.lbl_react_p.setStyleSheet("font-size: 12px; font-weight: bold; color: #16a34a;")
        calc_layout.addWidget(self.lbl_react_p)

        self.lbl_slip_badge = QLabel("STABLE (Unreactivated)")
        self.lbl_slip_badge.setStyleSheet("background: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 3px; font-weight: bold;")
        calc_layout.addWidget(self.lbl_slip_badge)

        form_slip.addRow("Reactivation Margin:", calc_box)
        right_layout.addWidget(grp_slip)

        right_layout.addStretch()

        # Splitter to allow user resize
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

        return tab

    # =========================================================================
    # TAB 2: INTER-FAULT STRESS TRANSFER (COULOMB DELTA CFS)
    # =========================================================================
    def _build_inter_stress_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top Information Banner
        info_banner = QFrame()
        info_banner.setStyleSheet("background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 4px; padding: 6px;")
        ib_layout = QHBoxLayout(info_banner)
        ib_layout.setContentsMargins(8, 2, 8, 2)
        ib_lbl = QLabel(
            "Inter-Fault Coulomb Stress Transfer: Delta CFS = Delta tau - mu * (Delta sigma_n - Delta P). "
            "Slip or pressure on Fault A triggers stress re-distribution, unclamping or stabilizing adjacent faults."
        )
        ib_lbl.setStyleSheet("color: #0369a1; font-size: 11px; font-weight: 500;")
        ib_layout.addWidget(ib_lbl)
        layout.addWidget(info_banner)

        # Middle Matrix Table
        matrix_lbl = QLabel("Multi-Fault Geomechanical Interaction Matrix")
        matrix_lbl.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 11px;")
        layout.addWidget(matrix_lbl)

        self.table_cfs_matrix = QTableWidget(0, 6)
        self.table_cfs_matrix.setHorizontalHeaderLabels([
            "Source Fault", "Target Fault", "Separation (ft)", "Delta tau (psi)", "Delta CFS (psi)", "Geomechanical Impact"
        ])
        self.table_cfs_matrix.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_cfs_matrix.verticalHeader().setVisible(False)
        self.table_cfs_matrix.setFixedHeight(140)
        layout.addWidget(self.table_cfs_matrix)

        # Bottom Matplotlib Chart (Inter-fault stress decay & orientation)
        self.fig_cfs = Figure(figsize=(7, 3.5), facecolor="#ffffff")
        self.canvas_cfs = FigureCanvas(self.fig_cfs)
        layout.addWidget(self.canvas_cfs)

        return tab

    # =========================================================================
    # TAB 3: CAPROCK CONFINING STRATIGRAPHY
    # =========================================================================
    def _build_caprock_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Left Column: Layer Table & Layer Modifiers
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_hdr = QLabel("Overlying Confining Stratigraphic Units (EPA Class VI)")
        left_hdr.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 11px;")
        left_layout.addWidget(left_hdr)

        self.table_caprock = QTableWidget(0, 6)
        self.table_caprock.setHorizontalHeaderLabels(["Unit Name", "Lithology", "Thick (ft)", "E (GPa)", "Pe (psi)", "kv (nD)"])
        self.table_caprock.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_caprock.horizontalHeader().setStretchLastSection(True)
        self.table_caprock.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_caprock.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_caprock.verticalHeader().setVisible(False)
        self.table_caprock.itemSelectionChanged.connect(self._on_caprock_selection_changed)
        left_layout.addWidget(self.table_caprock)

        # Action buttons
        c_btn_row = QHBoxLayout()
        self.btn_add_cap = QPushButton("+ Add Unit")
        self.btn_add_cap.setObjectName("primaryBtn")
        self.btn_add_cap.clicked.connect(self._on_add_caprock)
        c_btn_row.addWidget(self.btn_add_cap)

        self.btn_del_cap = QPushButton("Delete")
        self.btn_del_cap.setObjectName("dangerBtn")
        self.btn_del_cap.clicked.connect(self._on_delete_caprock)
        c_btn_row.addWidget(self.btn_del_cap)

        self.btn_up_cap = QPushButton("Move Up")
        self.btn_up_cap.clicked.connect(self._on_move_up_caprock)
        c_btn_row.addWidget(self.btn_up_cap)

        self.btn_down_cap = QPushButton("Move Down")
        self.btn_down_cap.clicked.connect(self._on_move_down_caprock)
        c_btn_row.addWidget(self.btn_down_cap)
        left_layout.addLayout(c_btn_row)

        # Right Column: Selected Layer Parameters & Sealing Integrity Summary
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        grp_cap_edit = QGroupBox("Selected Confining Unit Petrophysics & Mechanical Moduli")
        form_c = QFormLayout(grp_cap_edit)
        form_c.setSpacing(6)

        self.edit_c_name = QLineEdit()
        self.edit_c_name.textChanged.connect(self._on_caprock_param_changed)
        form_c.addRow("Unit Name:", self.edit_c_name)

        self.edit_c_litho = QLineEdit()
        self.edit_c_litho.textChanged.connect(self._on_caprock_param_changed)
        form_c.addRow("Lithology / Mineralogy:", self.edit_c_litho)

        c_row1 = QHBoxLayout()
        self.spin_c_thick = QDoubleSpinBox()
        self.spin_c_thick.setRange(10.0, 2000.0)
        self.spin_c_thick.setSuffix(" ft")
        self.spin_c_thick.valueChanged.connect(self._on_caprock_param_changed)
        c_row1.addWidget(QLabel("Thickness:"))
        c_row1.addWidget(self.spin_c_thick)

        self.spin_c_young = QDoubleSpinBox()
        self.spin_c_young.setRange(1.0, 100.0)
        self.spin_c_young.setSuffix(" GPa")
        self.spin_c_young.valueChanged.connect(self._on_caprock_param_changed)
        c_row1.addWidget(QLabel("Young's Modulus E:"))
        c_row1.addWidget(self.spin_c_young)
        form_c.addRow(c_row1)

        c_row2 = QHBoxLayout()
        self.spin_c_pe = QDoubleSpinBox()
        self.spin_c_pe.setRange(100.0, 8000.0)
        self.spin_c_pe.setSuffix(" psi")
        self.spin_c_pe.valueChanged.connect(self._on_caprock_param_changed)
        c_row2.addWidget(QLabel("Entry Pressure Pe:"))
        c_row2.addWidget(self.spin_c_pe)

        self.spin_c_perm = QDoubleSpinBox()
        self.spin_c_perm.setRange(0.01, 100000.0)
        self.spin_c_perm.setSuffix(" nD")
        self.spin_c_perm.valueChanged.connect(self._on_caprock_param_changed)
        c_row2.addWidget(QLabel("Permeability kv:"))
        c_row2.addWidget(self.spin_c_perm)
        form_c.addRow(c_row2)

        c_row3 = QHBoxLayout()
        self.spin_c_t0 = QDoubleSpinBox()
        self.spin_c_t0.setRange(0.0, 2000.0)
        self.spin_c_t0.setSuffix(" psi")
        self.spin_c_t0.valueChanged.connect(self._on_caprock_param_changed)
        c_row3.addWidget(QLabel("Tensile Strength T0:"))
        c_row3.addWidget(self.spin_c_t0)

        self.spin_c_cohesion = QDoubleSpinBox()
        self.spin_c_cohesion.setRange(0.0, 3000.0)
        self.spin_c_cohesion.setSuffix(" psi")
        self.spin_c_cohesion.valueChanged.connect(self._on_caprock_param_changed)
        c_row3.addWidget(QLabel("Cohesion C0:"))
        c_row3.addWidget(self.spin_c_cohesion)
        form_c.addRow(c_row3)

        right_layout.addWidget(grp_cap_edit)

        # Integrity Summary Cards
        grp_summary = QGroupBox("Class VI Confining System Sealing Capacity")
        sum_layout = QVBoxLayout(grp_summary)
        sum_layout.setSpacing(6)

        self.lbl_cap_total_thick = QLabel("Total Confining Thickness: 385 ft")
        self.lbl_cap_total_thick.setStyleSheet("font-weight: 600; color: #1e293b;")
        sum_layout.addWidget(self.lbl_cap_total_thick)

        self.lbl_cap_max_col = QLabel("Max Sustainable CO2 Column: 3,800 ft (Primary Seal: 2,200 psi)")
        self.lbl_cap_max_col.setStyleSheet("font-weight: 600; color: #0284c7;")
        sum_layout.addWidget(self.lbl_cap_max_col)

        self.lbl_cap_maip = QLabel("Sandface MAIP Limit (0.90 Pfrac): 4,250 psia")
        self.lbl_cap_maip.setStyleSheet("font-weight: 600; color: #15803d;")
        sum_layout.addWidget(self.lbl_cap_maip)

        self.lbl_cap_strain = QLabel("Estimated Caprock Bending Shear Strain: 0.042% (Within Elastic Limit)")
        self.lbl_cap_strain.setStyleSheet("font-weight: 500; color: #64748b;")
        sum_layout.addWidget(self.lbl_cap_strain)

        right_layout.addWidget(grp_summary)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

        return tab

    # =========================================================================
    # POPULATION & EVENT HANDLERS
    # =========================================================================
    def _populate_faults_table(self):
        self.table_faults.blockSignals(True)
        self.table_faults.setRowCount(len(self.faults))
        for row, f in enumerate(self.faults):
            self.table_faults.setItem(row, 0, QTableWidgetItem(str(f.id)))
            self.table_faults.setItem(row, 1, QTableWidgetItem(str(f.name)))
            self.table_faults.setItem(row, 2, QTableWidgetItem(f"{f.strike:.0f}°"))
            self.table_faults.setItem(row, 3, QTableWidgetItem(f"{f.dip:.0f}°"))
            self.table_faults.setItem(row, 4, QTableWidgetItem(f"{f.throw:.0f} ft"))
            
            ts_item = QTableWidgetItem(f"{f.slip_tendency:.2f}")
            if f.slip_tendency >= f.friction_coefficient:
                ts_item.setBackground(QColor("#fee2e2"))
                ts_item.setForeground(QColor("#dc2626"))
            else:
                ts_item.setBackground(QColor("#dcfce7"))
                ts_item.setForeground(QColor("#15803d"))
            self.table_faults.setItem(row, 5, ts_item)

        self.table_faults.blockSignals(False)
        if len(self.faults) > 0:
            idx = min(self.selected_fault_index, len(self.faults) - 1)
            self.table_faults.selectRow(idx)

    def _load_selected_fault_form(self):
        if not (0 <= self.selected_fault_index < len(self.faults)):
            return
        f = self.faults[self.selected_fault_index]

        self.edit_f_name.blockSignals(True)
        self.spin_f_strike.blockSignals(True)
        self.spin_f_dip.blockSignals(True)
        self.spin_f_throw.blockSignals(True)
        self.spin_f_length.blockSignals(True)
        self.spin_f_cx.blockSignals(True)
        self.spin_f_cy.blockSignals(True)
        self.spin_f_trans.blockSignals(True)
        self.spin_f_damage_w.blockSignals(True)
        self.spin_f_sgr.blockSignals(True)
        self.spin_f_mu.blockSignals(True)
        self.spin_f_cohesion.blockSignals(True)

        self.edit_f_name.setText(f.name)
        self.spin_f_strike.setValue(f.strike)
        self.spin_f_dip.setValue(f.dip)
        self.spin_f_throw.setValue(f.throw)
        self.spin_f_length.setValue(f.length)
        self.spin_f_cx.setValue(f.center_x)
        self.spin_f_cy.setValue(f.center_y)
        self.spin_f_trans.setValue(f.transmissibility_multiplier)
        self.spin_f_damage_w.setValue(f.damage_zone_width)
        self.spin_f_sgr.setValue(f.shale_gouge_ratio)
        self.spin_f_mu.setValue(f.friction_coefficient)
        self.spin_f_cohesion.setValue(f.cohesion)

        self.edit_f_name.blockSignals(False)
        self.spin_f_strike.blockSignals(False)
        self.spin_f_dip.blockSignals(False)
        self.spin_f_throw.blockSignals(False)
        self.spin_f_length.blockSignals(False)
        self.spin_f_cx.blockSignals(False)
        self.spin_f_cy.blockSignals(False)
        self.spin_f_trans.blockSignals(False)
        self.spin_f_damage_w.blockSignals(False)
        self.spin_f_sgr.blockSignals(False)
        self.spin_f_mu.blockSignals(False)
        self.spin_f_cohesion.blockSignals(False)

        # SGR evaluation label
        if f.shale_gouge_ratio >= 30.0:
            self.lbl_sgr_eval.setText("Baffle (Continuous Clay Smear)")
            self.lbl_sgr_eval.setStyleSheet("color: #0369a1; font-weight: bold;")
        elif f.shale_gouge_ratio >= 20.0:
            self.lbl_sgr_eval.setText("Partial Baffle (Discontinuous Smear)")
            self.lbl_sgr_eval.setStyleSheet("color: #d97706; font-weight: bold;")
        else:
            self.lbl_sgr_eval.setText("Open Conduit (Low SGR, High Leak Risk)")
            self.lbl_sgr_eval.setStyleSheet("color: #dc2626; font-weight: bold;")

        # Evaluate slip geomechanics
        res_depth = float(self.manual_params.get("depth", 5000.0))
        pore_p = float(self.manual_params.get("initial_pressure", 3200.0))
        sv = res_depth * float(self.manual_params.get("sv_gradient", 1.05))
        sh = sv * float(self.manual_params.get("sh_ratio_k0", 0.72))

        stresses = f.calculate_resolved_stresses(sv, sh, pore_p)
        ts = stresses["slip_tendency"]
        dp_crit = stresses["delta_p_crit_psi"]
        is_crit = stresses["is_critically_stressed"]

        self.lbl_ts_val.setText(f"Ts: {ts:.2f}")
        self.lbl_react_p.setText(f"Delta P_crit: {dp_crit:+.0f} psi")

        if is_crit or ts >= f.friction_coefficient:
            self.lbl_slip_badge.setText("CRITICALLY STRESSED")
            self.lbl_slip_badge.setStyleSheet("background: #fee2e2; color: #dc2626; padding: 3px 8px; border-radius: 3px; font-weight: bold;")
        else:
            self.lbl_slip_badge.setText("STABLE (Unreactivated)")
            self.lbl_slip_badge.setStyleSheet("background: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 3px; font-weight: bold;")

        self._refresh_inter_stress_matrix()

    def _on_fault_selection_changed(self):
        rows = self.table_faults.selectedIndexes()
        if rows:
            self.selected_fault_index = rows[0].row()
            self._load_selected_fault_form()

    def _on_fault_param_changed(self):
        if not (0 <= self.selected_fault_index < len(self.faults)):
            return
        f = self.faults[self.selected_fault_index]
        f.name = self.edit_f_name.text()
        f.strike = self.spin_f_strike.value()
        f.dip = self.spin_f_dip.value()
        f.throw = self.spin_f_throw.value()
        f.length = self.spin_f_length.value()
        f.center_x = self.spin_f_cx.value()
        f.center_y = self.spin_f_cy.value()
        f.transmissibility_multiplier = self.spin_f_trans.value()
        f.damage_zone_width = self.spin_f_damage_w.value()
        f.shale_gouge_ratio = self.spin_f_sgr.value()
        f.friction_coefficient = self.spin_f_mu.value()
        f.cohesion = self.spin_f_cohesion.value()

        # Update table row
        row = self.selected_fault_index
        self.table_faults.item(row, 1).setText(f.name)
        self.table_faults.item(row, 2).setText(f"{f.strike:.0f}°")
        self.table_faults.item(row, 3).setText(f"{f.dip:.0f}°")
        self.table_faults.item(row, 4).setText(f"{f.throw:.0f} ft")

        # Re-evaluate slip
        res_depth = float(self.manual_params.get("depth", 5000.0))
        pore_p = float(self.manual_params.get("initial_pressure", 3200.0))
        sv = res_depth * float(self.manual_params.get("sv_gradient", 1.05))
        sh = sv * float(self.manual_params.get("sh_ratio_k0", 0.72))
        stresses = f.calculate_resolved_stresses(sv, sh, pore_p)
        self.table_faults.item(row, 5).setText(f"{f.slip_tendency:.2f}")

        self._refresh_inter_stress_matrix()

    def _on_add_fault(self):
        num = len(self.faults) + 1
        new_f = FaultData(
            id=f"F-{num}",
            name=f"Fault F-{num}",
            strike=45.0,
            dip=65.0,
            throw=35.0,
            heave=16.3,
            length=3000.0,
            center_x=1200.0 + num * 300.0,
            center_y=1100.0 + num * 200.0,
            transmissibility_multiplier=0.12,
            damage_zone_width=75.0,
            shale_gouge_ratio=30.0
        )
        self.faults.append(new_f)
        self.selected_fault_index = len(self.faults) - 1
        self._populate_faults_table()

    def _on_duplicate_fault(self):
        if not (0 <= self.selected_fault_index < len(self.faults)):
            return
        orig = self.faults[self.selected_fault_index]
        clone = deepcopy(orig)
        clone.id = f"F-{len(self.faults) + 1}"
        clone.name = f"{orig.name} (Copy)"
        clone.center_x += 400.0
        clone.center_y += 300.0
        self.faults.append(clone)
        self.selected_fault_index = len(self.faults) - 1
        self._populate_faults_table()

    def _on_delete_fault(self):
        if len(self.faults) <= 1:
            QMessageBox.information(self, "Fault Deletion", "The reservoir model requires at least one fault definition.")
            return
        if not (0 <= self.selected_fault_index < len(self.faults)):
            return
        del self.faults[self.selected_fault_index]
        self.selected_fault_index = max(0, self.selected_fault_index - 1)
        self._populate_faults_table()

    def _on_apply_fault_preset(self, index: int):
        if index == 1:  # Single Major Boundary
            self.faults = [
                FaultData(id="F-1", name="Major Boundary Fault F-1", strike=45.0, dip=72.0, throw=60.0, heave=19.5, length=4200.0, center_x=1000.0, center_y=1000.0, shale_gouge_ratio=35.0, transmissibility_multiplier=0.08)
            ]
        elif index == 2:  # Conjugate Graben
            self.faults = [
                FaultData(id="F-1", name="Master Border Fault F-1", strike=45.0, dip=70.0, dip_direction="SE", throw=50.0, heave=18.2, length=3800.0, center_x=800.0, center_y=900.0, shale_gouge_ratio=32.0, transmissibility_multiplier=0.10),
                FaultData(id="F-2", name="Antithetic Graben Fault F-2", strike=45.0, dip=68.0, dip_direction="NW", throw=-45.0, heave=18.1, length=3200.0, center_x=1800.0, center_y=1300.0, shale_gouge_ratio=28.0, transmissibility_multiplier=0.15)
            ]
        elif index == 3:  # Synthetic Normal Step-Faults (3)
            self.faults = [
                FaultData(id="F-1", name="Step Fault F-1", strike=30.0, dip=70.0, throw=40.0, heave=14.5, length=3500.0, center_x=600.0, center_y=800.0, shale_gouge_ratio=30.0, transmissibility_multiplier=0.12),
                FaultData(id="F-2", name="Step Fault F-2", strike=32.0, dip=68.0, throw=35.0, heave=14.1, length=3200.0, center_x=1300.0, center_y=1100.0, shale_gouge_ratio=33.0, transmissibility_multiplier=0.10),
                FaultData(id="F-3", name="Step Fault F-3", strike=28.0, dip=65.0, throw=30.0, heave=14.0, length=2900.0, center_x=2000.0, center_y=1400.0, shale_gouge_ratio=26.0, transmissibility_multiplier=0.18)
            ]
        elif index == 4:  # En-Echelon Relays
            self.faults = [
                FaultData(id="F-1", name="En-Echelon Segment F-1", strike=55.0, dip=70.0, throw=45.0, heave=16.3, length=2400.0, center_x=800.0, center_y=800.0, shale_gouge_ratio=31.0, transmissibility_multiplier=0.14),
                FaultData(id="F-2", name="En-Echelon Relay Segment F-2", strike=52.0, dip=68.0, throw=42.0, heave=17.0, length=2600.0, center_x=1600.0, center_y=1500.0, shale_gouge_ratio=29.0, transmissibility_multiplier=0.16)
            ]
        else:
            return

        self.selected_fault_index = 0
        self._populate_faults_table()

    # =========================================================================
    # REFRESH COULOMB INTER-STRESS MATRIX & PLOT
    # =========================================================================
    def _refresh_inter_stress_matrix(self):
        n = len(self.faults)
        self.table_cfs_matrix.setRowCount(0)
        if n < 2:
            self.table_cfs_matrix.setRowCount(1)
            self.table_cfs_matrix.setItem(0, 0, QTableWidgetItem(self.faults[0].name if n > 0 else "N/A"))
            self.table_cfs_matrix.setItem(0, 1, QTableWidgetItem("None (Single Fault)"))
            self.table_cfs_matrix.setItem(0, 2, QTableWidgetItem("-"))
            self.table_cfs_matrix.setItem(0, 3, QTableWidgetItem("-"))
            self.table_cfs_matrix.setItem(0, 4, QTableWidgetItem("-"))
            self.table_cfs_matrix.setItem(0, 5, QTableWidgetItem("No Inter-Fault Geomechanical Impact"))
            self._render_cfs_plot()
            return

        row_idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f_src = self.faults[i]
                f_tgt = self.faults[j]

                # Distance between fault centers
                dist = np.sqrt((f_tgt.center_x - f_src.center_x)**2 + (f_tgt.center_y - f_src.center_y)**2)
                mu = f_tgt.friction_coefficient
                # Analytical approximation for 2D dislocation stress field
                d_ft = max(dist, 100.0)
                delta_tau = 350.0 * (1500.0 / d_ft)**1.8 * np.cos(np.radians(f_tgt.strike - f_src.strike))
                delta_sigman = 180.0 * (1500.0 / d_ft)**1.8 * np.sin(np.radians(f_tgt.dip - f_src.dip))
                delta_cfs = delta_tau - mu * delta_sigman

                self.table_cfs_matrix.insertRow(row_idx)
                self.table_cfs_matrix.setItem(row_idx, 0, QTableWidgetItem(f_src.id))
                self.table_cfs_matrix.setItem(row_idx, 1, QTableWidgetItem(f_tgt.id))
                self.table_cfs_matrix.setItem(row_idx, 2, QTableWidgetItem(f"{dist:.0f} ft"))
                self.table_cfs_matrix.setItem(row_idx, 3, QTableWidgetItem(f"{delta_tau:+.1f}"))

                cfs_item = QTableWidgetItem(f"{delta_cfs:+.1f}")
                if delta_cfs > 15.0:
                    cfs_item.setBackground(QColor("#fee2e2"))
                    cfs_item.setForeground(QColor("#dc2626"))
                    impact_str = "Destabilizing (Unclamping slip risk)"
                elif delta_cfs < -15.0:
                    cfs_item.setBackground(QColor("#dcfce7"))
                    cfs_item.setForeground(QColor("#15803d"))
                    impact_str = "Stabilizing (Normal clamping)"
                else:
                    impact_str = "Neutral Stress Coupling"
                self.table_cfs_matrix.setItem(row_idx, 4, cfs_item)
                self.table_cfs_matrix.setItem(row_idx, 5, QTableWidgetItem(impact_str))
                row_idx += 1

        self._render_cfs_plot()

    def _render_cfs_plot(self):
        self.fig_cfs.clear()
        ax1 = self.fig_cfs.add_subplot(1, 2, 1)
        ax2 = self.fig_cfs.add_subplot(1, 2, 2)

        # Plot 1: Coulomb Stress Decay vs Perpendicular Distance
        distances = np.linspace(100.0, 5000.0, 100)
        cfs_curve_pos = 420.0 * (1000.0 / distances)**1.6
        cfs_curve_neg = -210.0 * (1000.0 / distances)**1.6

        ax1.plot(distances, cfs_curve_pos, color="#dc2626", lw=2.0, label="Extensional Lobe (+Delta CFS)")
        ax1.plot(distances, cfs_curve_neg, color="#16a34a", lw=2.0, ls="--", label="Compressional Lobe (-Delta CFS)")
        ax1.axhline(0, color="#94a3b8", lw=0.8, ls=":")
        ax1.axhline(15.0, color="#ef4444", lw=1.0, ls="-.", label="Trigger Threshold (15 psi)")
        ax1.set_xlabel("Separation Distance (ft)", fontsize=9, fontweight="bold", color="#1e293b")
        ax1.set_ylabel("Coulomb Stress Transfer Delta CFS (psi)", fontsize=9, fontweight="bold", color="#1e293b")
        ax1.set_title("Stress Decay vs Distance from Fault Plane", fontsize=10, fontweight="bold", color="#0f172a")
        ax1.grid(True, ls=":", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=8)

        # Plot 2: Strike Sensitivity of Slip Tendency
        azimuths = np.linspace(0, 360, 180)
        sh_az = float(self.manual_params.get("sh_azimuth", 90.0))
        rel_ang = np.radians(azimuths - sh_az)
        ts_polar = 0.25 + 0.35 * (np.sin(2.0 * rel_ang)**2)

        ax2.plot(azimuths, ts_polar, color="#0284c7", lw=2.0, label="Slip Tendency Ts(alpha)")
        ax2.axhline(0.60, color="#dc2626", lw=1.5, ls="--", label="Reactivation Friction (mu=0.60)")

        # Mark each fault's strike on the curve
        for f in self.faults:
            ax2.plot([f.strike], [f.slip_tendency], marker="o", markersize=7, label=f"{f.id} ({f.strike:.0f}°)")

        ax2.set_xlabel("Fault Strike Azimuth (degrees from North)", fontsize=9, fontweight="bold", color="#1e293b")
        ax2.set_ylabel("Resolved Slip Tendency Ts", fontsize=9, fontweight="bold", color="#1e293b")
        ax2.set_title("Slip Tendency vs Strike Orientation", fontsize=10, fontweight="bold", color="#0f172a")
        ax2.set_xlim(0, 360)
        ax2.grid(True, ls=":", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=8)

        self.fig_cfs.tight_layout()
        self.canvas_cfs.draw()

    # =========================================================================
    # TAB 3: CAPROCK EVENT HANDLERS
    # =========================================================================
    def _populate_caprock_table(self):
        self.table_caprock.blockSignals(True)
        self.table_caprock.setRowCount(len(self.caprocks))
        for row, c in enumerate(self.caprocks):
            self.table_caprock.setItem(row, 0, QTableWidgetItem(c.name))
            self.table_caprock.setItem(row, 1, QTableWidgetItem(c.lithology))
            self.table_caprock.setItem(row, 2, QTableWidgetItem(f"{c.thickness_ft:.0f}"))
            self.table_caprock.setItem(row, 3, QTableWidgetItem(f"{c.youngs_modulus_gpa:.1f}"))
            self.table_caprock.setItem(row, 4, QTableWidgetItem(f"{c.entry_pressure_psi:.0f}"))
            self.table_caprock.setItem(row, 5, QTableWidgetItem(f"{c.permeability_nd:.0f}"))

        self.table_caprock.blockSignals(False)
        if len(self.caprocks) > 0:
            idx = min(self.selected_caprock_index, len(self.caprocks) - 1)
            self.table_caprock.selectRow(idx)
            self._load_selected_caprock_form()

    def _load_selected_caprock_form(self):
        if not (0 <= self.selected_caprock_index < len(self.caprocks)):
            return
        c = self.caprocks[self.selected_caprock_index]

        self.edit_c_name.blockSignals(True)
        self.edit_c_litho.blockSignals(True)
        self.spin_c_thick.blockSignals(True)
        self.spin_c_young.blockSignals(True)
        self.spin_c_pe.blockSignals(True)
        self.spin_c_perm.blockSignals(True)
        self.spin_c_t0.blockSignals(True)
        self.spin_c_cohesion.blockSignals(True)

        self.edit_c_name.setText(c.name)
        self.edit_c_litho.setText(c.lithology)
        self.spin_c_thick.setValue(c.thickness_ft)
        self.spin_c_young.setValue(c.youngs_modulus_gpa)
        self.spin_c_pe.setValue(c.entry_pressure_psi)
        self.spin_c_perm.setValue(c.permeability_nd)
        self.spin_c_t0.setValue(c.tensile_strength_psi)
        self.spin_c_cohesion.setValue(c.cohesion_psi)

        self.edit_c_name.blockSignals(False)
        self.edit_c_litho.blockSignals(False)
        self.spin_c_thick.blockSignals(False)
        self.spin_c_young.blockSignals(False)
        self.spin_c_pe.blockSignals(False)
        self.spin_c_perm.blockSignals(False)
        self.spin_c_t0.blockSignals(False)
        self.spin_c_cohesion.blockSignals(False)

        self._refresh_caprock_summary()

    def _on_caprock_selection_changed(self):
        rows = self.table_caprock.selectedIndexes()
        if rows:
            self.selected_caprock_index = rows[0].row()
            self._load_selected_caprock_form()

    def _on_caprock_param_changed(self):
        if not (0 <= self.selected_caprock_index < len(self.caprocks)):
            return
        c = self.caprocks[self.selected_caprock_index]
        c.name = self.edit_c_name.text()
        c.lithology = self.edit_c_litho.text()
        c.thickness_ft = self.spin_c_thick.value()
        c.youngs_modulus_gpa = self.spin_c_young.value()
        c.entry_pressure_psi = self.spin_c_pe.value()
        c.permeability_nd = self.spin_c_perm.value()
        c.tensile_strength_psi = self.spin_c_t0.value()
        c.cohesion_psi = self.spin_c_cohesion.value()

        # Update table
        row = self.selected_caprock_index
        self.table_caprock.item(row, 0).setText(c.name)
        self.table_caprock.item(row, 1).setText(c.lithology)
        self.table_caprock.item(row, 2).setText(f"{c.thickness_ft:.0f}")
        self.table_caprock.item(row, 3).setText(f"{c.youngs_modulus_gpa:.1f}")
        self.table_caprock.item(row, 4).setText(f"{c.entry_pressure_psi:.0f}")
        self.table_caprock.item(row, 5).setText(f"{c.permeability_nd:.0f}")

        self._refresh_caprock_summary()

    def _on_add_caprock(self):
        num = len(self.caprocks) + 1
        new_c = CaprockLayer(
            name=f"Unit C{num} - Confining Member",
            thickness_ft=90.0,
            lithology="Silty Mudstone Barrier",
            youngs_modulus_gpa=20.0,
            poissons_ratio=0.26,
            tensile_strength_psi=190.0,
            cohesion_psi=350.0,
            friction_angle_deg=30.0,
            entry_pressure_psi=1600.0,
            permeability_nd=350.0
        )
        self.caprocks.append(new_c)
        self.selected_caprock_index = len(self.caprocks) - 1
        self._populate_caprock_table()

    def _on_delete_caprock(self):
        if len(self.caprocks) <= 1:
            QMessageBox.information(self, "Caprock Deletion", "The confining model requires at least one caprock seal layer.")
            return
        if not (0 <= self.selected_caprock_index < len(self.caprocks)):
            return
        del self.caprocks[self.selected_caprock_index]
        self.selected_caprock_index = max(0, self.selected_caprock_index - 1)
        self._populate_caprock_table()

    def _on_move_up_caprock(self):
        idx = self.selected_caprock_index
        if idx > 0:
            self.caprocks[idx], self.caprocks[idx - 1] = self.caprocks[idx - 1], self.caprocks[idx]
            self.selected_caprock_index = idx - 1
            self._populate_caprock_table()

    def _on_move_down_caprock(self):
        idx = self.selected_caprock_index
        if idx < len(self.caprocks) - 1:
            self.caprocks[idx], self.caprocks[idx + 1] = self.caprocks[idx + 1], self.caprocks[idx]
            self.selected_caprock_index = idx + 1
            self._populate_caprock_table()

    def _refresh_caprock_summary(self):
        total_thk = sum(c.thickness_ft for c in self.caprocks)
        min_pe = min((c.entry_pressure_psi for c in self.caprocks), default=1500.0)
        max_col = min((c.max_sustainable_column_ft() for c in self.caprocks), default=3000.0)

        res_depth = float(self.manual_params.get("depth", 5000.0))
        frac_grad = float(self.manual_params.get("frac_gradient", 0.78))
        maip = res_depth * frac_grad * 0.90

        self.lbl_cap_total_thick.setText(f"Total Confining Thickness: {total_thk:.0f} ft ({len(self.caprocks)} Units)")
        self.lbl_cap_max_col.setText(f"Max Sustainable CO2 Column: {max_col:.0f} ft (Min Breakthrough Pe: {min_pe:.0f} psi)")
        self.lbl_cap_maip.setText(f"Sandface MAIP Limit (40 CFR § 146.86): {maip:.0f} psia")

    # =========================================================================
    # DIALOG ACTIONS
    # =========================================================================
    def _on_apply_clicked(self):
        self.faults_updated.emit(self.faults)
        self.caprock_updated.emit(self.caprocks)
        self.applied.emit()

    def _on_ok_clicked(self):
        self.faults_updated.emit(self.faults)
        self.caprock_updated.emit(self.caprocks)
        self.applied.emit()
        self.accept()

    def get_faults(self) -> List[FaultData]:
        return self.faults

    def get_caprock(self) -> List[CaprockLayer]:
        return self.caprocks

