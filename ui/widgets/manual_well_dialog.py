"""
Comprehensive 3D Wellbore Architecture, Perforation, and Inflow Modeling Dialog
================================================================================

State-of-the-art interactive dialog for designing and modeling wells with
maximum physical precision (Petrel / Prosper / CMG standard):
1. Identification, Role, and Reservoir Surface Positioning (with Pattern Grid Snapping).
2. 3D Trajectory Architecture (Vertical, Anisotropic Horizontal with KOP and Lateral, Deviated S-Curve).
3. Multi-interval Perforation Log & Casing/Tubing Completion Geometry.
4. Live Inflow Performance Relationship (IPR Composite Vogel-Darcy) & Peaceman Productivity Index.
5. Operating Controls & EPA Class VI Geomechanical Safe Sandface Pressure Enforcement.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QGroupBox, QFrame,
    QMessageBox, QDialogButtonBox, QWidget, QSplitter
)
from PyQt6.QtGui import QIcon, QFont, QColor
from PyQt6.QtCore import Qt, QPointF, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.data_models import WellData
from core.engine_surrogate.well_mechanics import (
    calculate_peaceman_index_vertical,
    calculate_peaceman_index_horizontal,
    generate_synthetic_well_trajectory,
    DARCY_FIELD_CONSTANT,
)

logger = logging.getLogger(__name__)


class ComprehensiveWellEditorDialog(QDialog):
    """
    Precision engineering dialog for designing, editing, and modeling reservoir wellbores
    with live IPR deliverability, Peaceman Well Index, and EPA Class VI geomechanical checks.
    """

    def __init__(
        self,
        existing_names: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
        initial_values: Optional[Dict[str, Any]] = None,
        reservoir_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Well Architecture & Precision Inflow Modeling")
        self.setMinimumSize(880, 680)
        self.resize(960, 720)

        self.existing_names = list(existing_names) if existing_names is not None else []
        self.initial_values = dict(initial_values) if initial_values is not None else {}
        self.res_params = dict(reservoir_params) if reservoir_params is not None else {}

        # Default reservoir background parameters
        self.res_len = float(self.res_params.get("length", 2000.0) or 2000.0)
        self.res_area = float(self.res_params.get("area", 100.0) or 100.0)
        self.res_width = (self.res_area * 43560.0) / max(self.res_len, 1.0)
        self.res_top = float(self.res_params.get("top_depth", 5000.0) or 5000.0)
        self.res_thick = float(self.res_params.get("thickness", 50.0) or 50.0)
        self.res_bot = self.res_top + self.res_thick
        self.res_perm = float(self.res_params.get("perm", 100.0) or 100.0)
        self.res_kv_kh = float(self.res_params.get("kv_kh_ratio", 0.10) or 0.10)
        self.res_p_ini = float(self.res_params.get("initial_pressure", 4000.0) or 4000.0)
        self.res_pb = float(self.res_params.get("bubble_point_pressure", 2250.0) or 2250.0)
        self.res_frac_grad = float(self.res_params.get("frac_grad", 0.85) or 0.85)
        self.res_uic_sf = float(self.res_params.get("uic_sf", 0.90) or 0.90)

        self.well_path: Optional[np.ndarray] = None

        self._setup_ui()
        self._populate_initial_values()
        self._update_all_calculations()

    @property
    def well_name_edit(self):
        return self.edit_name

    @property
    def key_param_values(self) -> Dict[str, Any]:
        t_raw = self.combo_traj_type.currentText()
        if "horiz" in t_raw.lower():
            traj_str = "Horizontal"
        elif "dev" in t_raw.lower():
            traj_str = "Deviated"
        else:
            traj_str = "Vertical"
        return {
            "SurfaceX": self.spin_x.value(),
            "SurfaceY": self.spin_y.value(),
            "TopDepth": self.spin_top_depth.value(),
            "BottomDepth": self.spin_bot_depth.value(),
            "TrajectoryType": traj_str,
            "LateralLength": self.spin_lat_len.value(),
            "Azimuth": self.spin_azimuth.value(),
            "KickoffDepth": self.spin_kop.value(),
        }

    @property
    def peaceman_wi_label(self):
        class ProxyLabel:
            def __init__(self, dlg):
                self.dlg = dlg
            def text(self):
                t = "Horizontal" if "horiz" in self.dlg.combo_traj_type.currentText().lower() else "Vertical"
                return f"Peaceman Well Index ({t}): {getattr(self.dlg, 'lbl_ipr_wi', QLabel()).text()}"
        return ProxyLabel(self)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 1. Header Banner
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 4, 4, 4)

        icon_label = QLabel("🎯")
        icon_label.setStyleSheet("font-size: 20px;")
        h_layout.addWidget(icon_label)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("Well Architecture & Flow Deliverability Studio")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #0f172a;")
        lbl_sub = QLabel("Integral 3D trajectory synthesis, multi-zone completions, Peaceman Well Index, and Vogel IPR")
        lbl_sub.setStyleSheet("font-size: 10.5px; color: #64748b;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        h_layout.addLayout(title_vbox, stretch=1)

        self.lbl_header_wi = QLabel("Peaceman WI: Calculating...")
        self.lbl_header_wi.setStyleSheet("""
            background: #e0f2fe;
            color: #0369a1;
            font-size: 11px;
            font-weight: bold;
            padding: 5px 10px;
            border: 1px solid #bae6fd;
            border-radius: 4px;
        """)
        h_layout.addWidget(self.lbl_header_wi)
        main_layout.addWidget(header_frame)

        # 2. Main Tabbed Workstation
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f1f5f9;
                color: #475569;
                padding: 7px 16px;
                font-weight: 600;
                font-size: 11px;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0d6efd;
                border-bottom: 2px solid #0d6efd;
            }
        """)

        # Tab 1: Identification & Surface Location
        self.tab_general = QWidget()
        self._setup_tab_general()
        self.tabs.addTab(self.tab_general, "1. Surface & Identity")

        # Tab 2: 3D Trajectory Architecture & Preview
        self.tab_trajectory = QWidget()
        self._setup_tab_trajectory()
        self.tabs.addTab(self.tab_trajectory, "2. Trajectory Architecture")

        # Tab 3: Perforations & Completions
        self.tab_perfs = QWidget()
        self._setup_tab_perforations()
        self.tabs.addTab(self.tab_perfs, "3. Completions & Perforations")

        # Tab 4: Inflow & Peaceman Deliverability (Live IPR)
        self.tab_ipr = QWidget()
        self._setup_tab_ipr()
        self.tabs.addTab(self.tab_ipr, "4. Inflow & Vogel IPR")

        # Tab 5: Operating Controls & EPA Class VI Safety
        self.tab_operations = QWidget()
        self._setup_tab_operations()
        self.tabs.addTab(self.tab_operations, "5. Operating Controls & EPA Class VI")

        main_layout.addWidget(self.tabs, stretch=1)

        # 3. Bottom Dialog Actions
        bottom_frame = QFrame()
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #64748b; font-size: 11px;")
        bottom_layout.addWidget(self.lbl_status)
        bottom_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover { background: #f1f5f9; }
        """)
        self.btn_cancel.clicked.connect(self.reject)
        bottom_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Save & Insert Well into Network")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: #ffffff;
                border: 1px solid #0b5ed7;
                border-radius: 4px;
                padding: 6px 20px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background: #0b5ed7; }
        """)
        self.btn_save.clicked.connect(self._on_save_clicked)
        bottom_layout.addWidget(self.btn_save)

        main_layout.addWidget(bottom_frame)

    # --------------------------------------------------------------------------
    # Tab 1: General & Location
    # --------------------------------------------------------------------------
    def _setup_tab_general(self):
        layout = QVBoxLayout(self.tab_general)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Form section
        form_group = QGroupBox("Well Identification & Role")
        form_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        fg_layout = QVBoxLayout(form_group)
        fg_layout.setSpacing(10)

        row1 = QHBoxLayout()
        lbl_name = QLabel("Well Name:")
        lbl_name.setFixedWidth(130)
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. PROD-01 or INJ-01")
        self.edit_name.textChanged.connect(self._on_name_changed)
        row1.addWidget(lbl_name)
        row1.addWidget(self.edit_name)
        fg_layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl_role = QLabel("Well Role / Function:")
        lbl_role.setFixedWidth(130)
        self.combo_role = QComboBox()
        self.combo_role.addItems([
            "Producer (Active Oil/Gas)",
            "Producer (Shut-in / Observation)",
            "CO2 Injector (Continuous Gas)",
            "WAG Injector (Water-Alternating-Gas)",
            "Water Injector (Pattern Disposal / Pressure Maintenance)"
        ])
        self.combo_role.currentIndexChanged.connect(self._on_role_changed)
        row2.addWidget(lbl_role)
        row2.addWidget(self.combo_role)
        fg_layout.addLayout(row2)

        layout.addWidget(form_group)

        # Surface coordinates section
        coord_group = QGroupBox("Surface Coordinates & Field Boundaries")
        coord_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        cg_layout = QVBoxLayout(coord_group)
        cg_layout.setSpacing(10)

        row_c1 = QHBoxLayout()
        lbl_x = QLabel("Surface X Coordinate:")
        lbl_x.setFixedWidth(130)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(0.0, max(self.res_len * 2.0, 50000.0))
        self.spin_x.setSingleStep(50.0)
        self.spin_x.setValue(self.res_len * 0.5)
        self.spin_x.valueChanged.connect(self._update_all_calculations)

        lbl_y = QLabel("Surface Y Coordinate:")
        lbl_y.setFixedWidth(130)
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(0.0, max(self.res_width * 2.0, 50000.0))
        self.spin_y.setSingleStep(50.0)
        self.spin_y.setValue(self.res_width * 0.5)
        self.spin_y.valueChanged.connect(self._update_all_calculations)

        row_c1.addWidget(lbl_x)
        row_c1.addWidget(self.spin_x)
        row_c1.addWidget(QLabel("ft"))
        row_c1.addSpacing(20)
        row_c1.addWidget(lbl_y)
        row_c1.addWidget(self.spin_y)
        row_c1.addWidget(QLabel("ft"))
        cg_layout.addLayout(row_c1)

        # Bounds check badge & Snap buttons
        row_snap = QHBoxLayout()
        self.lbl_boundary_check = QLabel("Boundary: Validated within active reservoir grid ✓")
        self.lbl_boundary_check.setStyleSheet("color: #16a34a; font-size: 11px; font-weight: bold;")
        row_snap.addWidget(self.lbl_boundary_check)
        row_snap.addStretch()

        btn_snap_center = QPushButton("Center of Pattern")
        btn_snap_center.setToolTip("Snap coordinates to reservoir centroid")
        btn_snap_center.clicked.connect(lambda: self._snap_coordinates(self.res_len * 0.5, self.res_width * 0.5))
        row_snap.addWidget(btn_snap_center)

        btn_snap_nw = QPushButton("NW Corner (Inj)")
        btn_snap_nw.clicked.connect(lambda: self._snap_coordinates(self.res_len * 0.15, self.res_width * 0.85))
        row_snap.addWidget(btn_snap_nw)

        btn_snap_se = QPushButton("SE Corner (Inj)")
        btn_snap_se.clicked.connect(lambda: self._snap_coordinates(self.res_len * 0.85, self.res_width * 0.15))
        row_snap.addWidget(btn_snap_se)

        cg_layout.addLayout(row_snap)
        layout.addWidget(coord_group)

        # Well elevation / KB reference
        kb_group = QGroupBox("Wellhead Reference Elevation")
        kb_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        kb_layout = QHBoxLayout(kb_group)
        lbl_kb = QLabel("Kelly Bushing (KB) Elevation:")
        lbl_kb.setFixedWidth(160)
        self.spin_kb = QDoubleSpinBox()
        self.spin_kb.setRange(-500.0, 10000.0)
        self.spin_kb.setValue(25.0)
        kb_layout.addWidget(lbl_kb)
        kb_layout.addWidget(self.spin_kb)
        kb_layout.addWidget(QLabel("ft above Sea Level"))
        kb_layout.addStretch()
        layout.addWidget(kb_group)

        layout.addStretch()

    # --------------------------------------------------------------------------
    # Tab 2: 3D Trajectory & Profile Preview
    # --------------------------------------------------------------------------
    def _setup_tab_trajectory(self):
        layout = QHBoxLayout(self.tab_trajectory)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left control column
        left_ctrl = QWidget()
        left_layout = QVBoxLayout(left_ctrl)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        traj_type_group = QGroupBox("Trajectory Architecture Type")
        traj_type_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        tt_layout = QVBoxLayout(traj_type_group)
        self.combo_traj_type = QComboBox()
        self.combo_traj_type.addItems([
            "Vertical Well (Standard Plumb-line)",
            "Horizontal Well (with KOP, Build Curve & Pay Lateral)",
            "Deviated S-Curve (Directional Kick & Hold)"
        ])
        self.combo_traj_type.currentIndexChanged.connect(self._on_traj_type_changed)
        tt_layout.addWidget(self.combo_traj_type)
        left_layout.addWidget(traj_type_group)

        # Depth parameters
        geom_group = QGroupBox("Trajectory Depth Milestones (TVD & MD)")
        geom_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        gm_layout = QVBoxLayout(geom_group)
        gm_layout.setSpacing(8)

        r_top = QHBoxLayout()
        lbl_top = QLabel("Top Depth (MD):")
        lbl_top.setFixedWidth(140)
        self.spin_top_depth = QDoubleSpinBox()
        self.spin_top_depth.setRange(0.0, 30000.0)
        self.spin_top_depth.setSingleStep(50.0)
        self.spin_top_depth.setValue(self.res_top)
        self.spin_top_depth.valueChanged.connect(self._update_all_calculations)
        r_top.addWidget(lbl_top)
        r_top.addWidget(self.spin_top_depth)
        r_top.addWidget(QLabel("ft"))
        gm_layout.addLayout(r_top)

        r_bot = QHBoxLayout()
        lbl_bot = QLabel("Bottom / Landing TVD:")
        lbl_bot.setFixedWidth(140)
        self.spin_bot_depth = QDoubleSpinBox()
        self.spin_bot_depth.setRange(50.0, 30000.0)
        self.spin_bot_depth.setSingleStep(50.0)
        self.spin_bot_depth.setValue(self.res_bot)
        self.spin_bot_depth.valueChanged.connect(self._update_all_calculations)
        r_bot.addWidget(lbl_bot)
        r_bot.addWidget(self.spin_bot_depth)
        r_bot.addWidget(QLabel("ft"))
        gm_layout.addLayout(r_bot)

        # Horizontal parameters (Kick-off, Lateral length, Azimuth)
        self.frame_horiz = QFrame()
        fh_layout = QVBoxLayout(self.frame_horiz)
        fh_layout.setContentsMargins(0, 0, 0, 0)
        fh_layout.setSpacing(8)

        r_kop = QHBoxLayout()
        lbl_kop = QLabel("Kick-Off Point (KOP):")
        lbl_kop.setFixedWidth(140)
        self.spin_kop = QDoubleSpinBox()
        self.spin_kop.setRange(0.0, 30000.0)
        self.spin_kop.setSingleStep(25.0)
        self.spin_kop.setValue(self.res_top - 50.0)
        self.spin_kop.valueChanged.connect(self._update_all_calculations)
        r_kop.addWidget(lbl_kop)
        r_kop.addWidget(self.spin_kop)
        r_kop.addWidget(QLabel("ft"))
        fh_layout.addLayout(r_kop)

        r_lat = QHBoxLayout()
        lbl_lat = QLabel("Completed Lateral Length:")
        lbl_lat.setFixedWidth(140)
        self.spin_lat_len = QDoubleSpinBox()
        self.spin_lat_len.setRange(100.0, 15000.0)
        self.spin_lat_len.setSingleStep(100.0)
        self.spin_lat_len.setValue(1500.0)
        self.spin_lat_len.valueChanged.connect(self._update_all_calculations)
        r_lat.addWidget(lbl_lat)
        r_lat.addWidget(self.spin_lat_len)
        r_lat.addWidget(QLabel("ft"))
        fh_layout.addLayout(r_lat)

        r_az = QHBoxLayout()
        lbl_az = QLabel("Lateral Azimuth (θ):")
        lbl_az.setFixedWidth(140)
        self.spin_azimuth = QDoubleSpinBox()
        self.spin_azimuth.setRange(0.0, 360.0)
        self.spin_azimuth.setSingleStep(15.0)
        self.spin_azimuth.setValue(90.0)
        self.spin_azimuth.valueChanged.connect(self._update_all_calculations)
        r_az.addWidget(lbl_az)
        r_az.addWidget(self.spin_azimuth)
        r_az.addWidget(QLabel("deg"))
        fh_layout.addLayout(r_az)

        gm_layout.addWidget(self.frame_horiz)
        left_layout.addWidget(geom_group)
        left_layout.addStretch()

        left_ctrl.setFixedWidth(360)
        layout.addWidget(left_ctrl)

        # Right: Live 2D/3D Trajectory Cross-Section Preview Plot
        right_panel = QFrame()
        right_panel.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px;")
        rp_layout = QVBoxLayout(right_panel)
        rp_layout.setContentsMargins(6, 6, 6, 6)

        rp_title = QLabel("Live Wellbore Trajectory & Formation Intersection Preview")
        rp_title.setStyleSheet("font-weight: bold; color: #1e293b; font-size: 11px;")
        rp_layout.addWidget(rp_title)

        self.fig_traj = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_traj = FigureCanvas(self.fig_traj)
        self.ax_traj = self.fig_traj.add_subplot(111)
        rp_layout.addWidget(self.canvas_traj, stretch=1)

        layout.addWidget(right_panel, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 3: Perforations & Completions Architecture
    # --------------------------------------------------------------------------
    def _setup_tab_perforations(self):
        layout = QVBoxLayout(self.tab_perfs)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Wellbore casing & skin parameters
        comp_group = QGroupBox("Casing, Wellbore Radius & Mechanical Skin")
        comp_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        cg_layout = QHBoxLayout(comp_group)
        cg_layout.setSpacing(16)

        lbl_rw = QLabel("Wellbore Radius (r_w):")
        self.spin_rw = QDoubleSpinBox()
        self.spin_rw.setRange(0.05, 2.0)
        self.spin_rw.setSingleStep(0.05)
        self.spin_rw.setDecimals(3)
        self.spin_rw.setValue(0.354)  # standard 8.5" hole
        self.spin_rw.valueChanged.connect(self._update_all_calculations)

        combo_hole_size = QComboBox()
        combo_hole_size.addItems(["Custom", "4.5\" Liner (rw=0.188 ft)", "5.5\" Casing (rw=0.229 ft)", "7.0\" Casing (rw=0.292 ft)", "8.5\" Standard (rw=0.354 ft)", "9.875\" Casing (rw=0.411 ft)"])
        combo_hole_size.setCurrentIndex(4)
        def _on_hole_preset(idx):
            presets = [None, 0.188, 0.229, 0.292, 0.354, 0.411]
            if idx > 0 and idx < len(presets) and presets[idx]:
                self.spin_rw.setValue(presets[idx])
        combo_hole_size.currentIndexChanged.connect(_on_hole_preset)

        lbl_skin = QLabel("Total Skin Factor (S):")
        self.spin_skin = QDoubleSpinBox()
        self.spin_skin.setRange(-5.0, 50.0)
        self.spin_skin.setSingleStep(0.5)
        self.spin_skin.setValue(0.0)
        self.spin_skin.valueChanged.connect(self._update_all_calculations)

        cg_layout.addWidget(lbl_rw)
        cg_layout.addWidget(self.spin_rw)
        cg_layout.addWidget(QLabel("ft"))
        cg_layout.addWidget(combo_hole_size)
        cg_layout.addSpacing(20)
        cg_layout.addWidget(lbl_skin)
        cg_layout.addWidget(self.spin_skin)
        cg_layout.addStretch()
        layout.addWidget(comp_group)

        # Perforation Interval Table
        perf_group = QGroupBox("Completed Perforation Intervals (Pay Zone Communication)")
        perf_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        pg_layout = QVBoxLayout(perf_group)
        pg_layout.setSpacing(8)

        # Presets bar
        preset_bar = QHBoxLayout()
        btn_perf_all = QPushButton("Perforate Entire Pay Zone")
        btn_perf_all.clicked.connect(self._preset_perf_all)
        preset_bar.addWidget(btn_perf_all)

        btn_perf_upper = QPushButton("Perforate Upper 1/3 (Water Buffer)")
        btn_perf_upper.clicked.connect(self._preset_perf_upper)
        preset_bar.addWidget(btn_perf_upper)

        btn_perf_lower = QPushButton("Perforate Lower 1/2 (Gas Override Buffer)")
        btn_perf_lower.clicked.connect(self._preset_perf_lower)
        preset_bar.addWidget(btn_perf_lower)
        preset_bar.addStretch()

        btn_add = QPushButton("+ Add Interval")
        btn_add.setStyleSheet("background: #e0f2fe; color: #0284c7; font-weight: bold;")
        btn_add.clicked.connect(self._add_perforation_row)
        preset_bar.addWidget(btn_add)

        btn_remove = QPushButton("- Remove Interval")
        btn_remove.clicked.connect(self._remove_perforation_row)
        preset_bar.addWidget(btn_remove)

        pg_layout.addLayout(preset_bar)

        self.table_perfs = QTableWidget()
        self.table_perfs.setColumnCount(5)
        self.table_perfs.setHorizontalHeaderLabels([
            "Top MD (ft)", "Bottom MD (ft)", "Interval Length (ft)", "Shot Density (SPF)", "Phase Angle (°)"
        ])
        self.table_perfs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_perfs.itemChanged.connect(self._on_perf_table_changed)
        pg_layout.addWidget(self.table_perfs)

        self.lbl_perf_summary = QLabel("Total Perforated Length: 0.0 ft")
        self.lbl_perf_summary.setStyleSheet("color: #0284c7; font-weight: bold; font-size: 11px;")
        pg_layout.addWidget(self.lbl_perf_summary)

        layout.addWidget(perf_group, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 4: Inflow Performance & Vogel IPR
    # --------------------------------------------------------------------------
    def _setup_tab_ipr(self):
        layout = QHBoxLayout(self.tab_ipr)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left: Deliverability metrics card
        left_card = QFrame()
        left_card.setStyleSheet("background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px;")
        left_card.setFixedWidth(320)
        lc_layout = QVBoxLayout(left_card)
        lc_layout.setSpacing(10)

        lc_title = QLabel("Peaceman Inflow Performance")
        lc_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #1e293b;")
        lc_layout.addWidget(lc_title)

        self.lbl_ipr_res_p = QLabel(f"Reservoir Pressure (P_res): {self.res_p_ini:.0f} psia")
        self.lbl_ipr_pb = QLabel(f"Bubble Point (P_b): {self.res_pb:.0f} psia")
        self.lbl_ipr_wi = QLabel("Peaceman Well Index: 0.00 STB/d/psi")
        self.lbl_ipr_wi.setStyleSheet("font-weight: bold; color: #0284c7;")
        self.lbl_ipr_aof = QLabel("Absolute Open Flow (AOF): 0.0 STB/d")
        self.lbl_ipr_aof.setStyleSheet("font-weight: bold; color: #16a34a;")

        lc_layout.addWidget(self.lbl_ipr_res_p)
        lc_layout.addWidget(self.lbl_ipr_pb)
        lc_layout.addWidget(self.lbl_ipr_wi)
        lc_layout.addWidget(self.lbl_ipr_aof)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e2e8f0;")
        lc_layout.addWidget(sep)

        lbl_desc = QLabel(
            "<b>Inflow Formulation:</b><br>"
            "• Darcy linear deliverability for P_wf ≥ P_b<br>"
            "• Vogel parabolic expansion below P_b (SCI-VOGEL):<br>"
            "&nbsp;&nbsp;q/q_max = 1 - 0.2(P_wf/P_res) - 0.8(P_wf/P_res)²<br>"
            "• Anisotropic 3D Peaceman formulation accounting for lateral length and kv/kh."
        )
        lbl_desc.setStyleSheet("color: #475569; font-size: 10px; line-height: 1.3;")
        lc_layout.addWidget(lbl_desc)
        lc_layout.addStretch()

        layout.addWidget(left_card)

        # Right: Matplotlib Live Composite IPR Plot
        right_plot_frame = QFrame()
        right_plot_frame.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px;")
        rpf_layout = QVBoxLayout(right_plot_frame)
        rpf_layout.setContentsMargins(6, 6, 6, 6)

        self.fig_ipr = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_ipr = FigureCanvas(self.fig_ipr)
        self.ax_ipr = self.fig_ipr.add_subplot(111)
        rpf_layout.addWidget(self.canvas_ipr, stretch=1)

        layout.addWidget(right_plot_frame, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 5: Operating Limits & EPA Class VI Safety
    # --------------------------------------------------------------------------
    def _setup_tab_operations(self):
        layout = QVBoxLayout(self.tab_operations)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Control mode
        ctrl_group = QGroupBox("Operating Control Mode & Setpoints")
        ctrl_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        cg_layout = QVBoxLayout(ctrl_group)
        cg_layout.setSpacing(10)

        r_mode = QHBoxLayout()
        lbl_m = QLabel("Well Control Constraint:")
        lbl_m.setFixedWidth(160)
        self.combo_control_mode = QComboBox()
        self.combo_control_mode.addItems([
            "Rate-Controlled (Target Surface Volume)",
            "BHP-Controlled (Sandface Flowing Pressure Limit)"
        ])
        r_mode.addWidget(lbl_m)
        r_mode.addWidget(self.combo_control_mode)
        cg_layout.addLayout(r_mode)

        r_rate = QHBoxLayout()
        self.lbl_target_rate = QLabel("Target Surface Rate:")
        self.lbl_target_rate.setFixedWidth(160)
        self.spin_target_rate = QDoubleSpinBox()
        self.spin_target_rate.setRange(0.0, 50000.0)
        self.spin_target_rate.setSingleStep(50.0)
        self.spin_target_rate.setValue(500.0)
        self.lbl_rate_unit = QLabel("STB/d (Liquid)")
        r_rate.addWidget(self.lbl_target_rate)
        r_rate.addWidget(self.spin_target_rate)
        r_rate.addWidget(self.lbl_rate_unit)
        cg_layout.addLayout(r_rate)

        r_bhp = QHBoxLayout()
        self.lbl_bhp = QLabel("Min Flowing BHP (P_wf,min):")
        self.lbl_bhp.setFixedWidth(160)
        self.spin_bhp = QDoubleSpinBox()
        self.spin_bhp.setRange(14.7, 15000.0)
        self.spin_bhp.setSingleStep(50.0)
        self.spin_bhp.setValue(1000.0)
        self.spin_bhp.valueChanged.connect(self._update_all_calculations)
        r_bhp.addWidget(self.lbl_bhp)
        r_bhp.addWidget(self.spin_bhp)
        r_bhp.addWidget(QLabel("psia"))
        cg_layout.addLayout(r_bhp)

        layout.addWidget(ctrl_group)

        # Geomechanical Safety (EPA Class VI Mandate)
        uic_group = QGroupBox("EPA Class VI Underground Injection Control (UIC) Geomechanical Safeguards")
        uic_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        ug_layout = QVBoxLayout(uic_group)
        ug_layout.setSpacing(10)

        # Depth and Frac limits calculation
        p_frac = self.res_frac_grad * self.res_top
        p_safe_ceiling = self.res_uic_sf * p_frac

        self.lbl_frac_info = QLabel(
            f"Formation Fracture Pressure (P_frac): <b>{p_frac:.0f} psia</b> (Gradient: {self.res_frac_grad:.2f} psi/ft at {self.res_top:.0f} ft)"
        )
        self.lbl_ceiling_info = QLabel(
            f"Mandatory EPA Class VI Ceiling (0.90 × P_frac): <b>{p_safe_ceiling:.0f} psia</b>"
        )
        self.lbl_ceiling_info.setStyleSheet("color: #0369a1; font-weight: bold;")

        self.lbl_safety_verdict = QLabel("Geomechanical Status: Safe Operating Margin ✓")
        self.lbl_safety_verdict.setStyleSheet("""
            background: #dcfce7;
            color: #15803d;
            font-size: 11px;
            font-weight: bold;
            padding: 8px 12px;
            border: 1px solid #86efac;
            border-radius: 4px;
        """)

        ug_layout.addWidget(self.lbl_frac_info)
        ug_layout.addWidget(self.lbl_ceiling_info)
        ug_layout.addWidget(self.lbl_safety_verdict)

        layout.addWidget(uic_group)
        layout.addStretch()

    # --------------------------------------------------------------------------
    # Populating initial values & Event Handlers
    # --------------------------------------------------------------------------
    def _populate_initial_values(self):
        # Name
        w_name = self.initial_values.get("name") or self.initial_values.get("well_name")
        if not w_name:
            count = len(self.existing_names) + 1
            w_name = f"Well-{count}"
        self.edit_name.setText(str(w_name))

        # Role
        role_str = str(self.initial_values.get("role") or self.initial_values.get("type") or "").lower()
        if "inj" in role_str or "inj" in w_name.lower():
            if "wag" in role_str:
                self.combo_role.setCurrentIndex(3)
            elif "water" in role_str:
                self.combo_role.setCurrentIndex(4)
            else:
                self.combo_role.setCurrentIndex(2)
        else:
            self.combo_role.setCurrentIndex(0)

        # Coordinates
        sx = float(self.initial_values.get("SurfaceX", self.initial_values.get("surface_x", self.res_len * 0.5)))
        sy = float(self.initial_values.get("SurfaceY", self.initial_values.get("surface_y", self.res_width * 0.5)))
        self.spin_x.setValue(sx)
        self.spin_y.setValue(sy)

        # Depths
        top_d = float(self.initial_values.get("TopDepth", self.res_top))
        bot_d = float(self.initial_values.get("BottomDepth", self.res_bot))
        self.spin_top_depth.setValue(top_d)
        self.spin_bot_depth.setValue(bot_d)

        # Trajectory
        traj = str(self.initial_values.get("TrajectoryType", self.initial_values.get("trajectory_type", "Vertical")))
        if "horiz" in traj.lower():
            self.combo_traj_type.setCurrentIndex(1)
        elif "dev" in traj.lower():
            self.combo_traj_type.setCurrentIndex(2)
        else:
            self.combo_traj_type.setCurrentIndex(0)

        lat_len = float(self.initial_values.get("LateralLength", self.initial_values.get("lateral_length", 1500.0)))
        self.spin_lat_len.setValue(lat_len)

        # Completions
        rw = float(self.initial_values.get("WellboreRadius", self.initial_values.get("wellbore_radius_ft", 0.354)))
        self.spin_rw.setValue(rw)

        skin = float(self.initial_values.get("SkinFactor", self.initial_values.get("skin_factor", 0.0)))
        self.spin_skin.setValue(skin)

        # Perforations
        perfs = self.initial_values.get("perforations") or self.initial_values.get("perforation_properties")
        if perfs:
            self.table_perfs.setRowCount(0)
            for p in perfs:
                if isinstance(p, dict):
                    pt, pb = float(p.get("top", top_d)), float(p.get("bottom", bot_d))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pt, pb = float(p[0]), float(p[1])
                else:
                    continue
                self._insert_perforation_row(pt, pb)
        else:
            # Default to full pay perforation
            self._preset_perf_all()

    def _on_name_changed(self, text: str):
        text_lower = text.strip().lower()
        if text_lower.startswith("inj") or "injector" in text_lower:
            if "inj" not in self.combo_role.currentText().lower():
                self.combo_role.blockSignals(True)
                self.combo_role.setCurrentIndex(2)
                self.combo_role.blockSignals(False)
                self._on_role_changed(2)

    def _on_role_changed(self, index: int):
        is_inj = (index >= 2)
        if is_inj:
            self.lbl_target_rate.setText("Target Injection Rate:")
            self.lbl_rate_unit.setText("MSCF/d (CO2) or BWPD")
            self.lbl_bhp.setText("Max Sandface Pressure (P_inj):")
            self.spin_bhp.setValue(min(self.res_frac_grad * self.res_top * self.res_uic_sf, 3500.0))
        else:
            self.lbl_target_rate.setText("Target Production Rate:")
            self.lbl_rate_unit.setText("STB/d (Liquid)")
            self.lbl_bhp.setText("Min Flowing BHP (P_wf,min):")
            self.spin_bhp.setValue(1000.0)
        self._update_all_calculations()

    def _on_traj_type_changed(self, index: int):
        is_horiz = (index == 1)
        self.frame_horiz.setVisible(is_horiz)
        self._update_all_calculations()

    def _snap_coordinates(self, x: float, y: float):
        self.spin_x.setValue(x)
        self.spin_y.setValue(y)

    # --------------------------------------------------------------------------
    # Perforation management
    # --------------------------------------------------------------------------
    def _add_perforation_row(self):
        pt = self.spin_top_depth.value()
        pb = self.spin_bot_depth.value()
        self._insert_perforation_row(pt, pb)

    def _insert_perforation_row(self, top_md: float, bot_md: float):
        row = self.table_perfs.rowCount()
        self.table_perfs.insertRow(row)

        self.table_perfs.setItem(row, 0, QTableWidgetItem(f"{top_md:.1f}"))
        self.table_perfs.setItem(row, 1, QTableWidgetItem(f"{bot_md:.1f}"))
        self.table_perfs.setItem(row, 2, QTableWidgetItem(f"{abs(bot_md - top_md):.1f}"))
        self.table_perfs.setItem(row, 3, QTableWidgetItem("6"))   # 6 SPF
        self.table_perfs.setItem(row, 4, QTableWidgetItem("60"))  # 60 deg phasing
        self._update_perforation_summary()

    def _remove_perforation_row(self):
        cur_row = self.table_perfs.currentRow()
        if cur_row >= 0:
            self.table_perfs.removeRow(cur_row)
            self._update_perforation_summary()

    def _preset_perf_all(self):
        self.table_perfs.setRowCount(0)
        self._insert_perforation_row(self.spin_top_depth.value(), self.spin_bot_depth.value())

    def _preset_perf_upper(self):
        self.table_perfs.setRowCount(0)
        top = self.spin_top_depth.value()
        bot = top + (self.spin_bot_depth.value() - top) * 0.35
        self._insert_perforation_row(top, bot)

    def _preset_perf_lower(self):
        self.table_perfs.setRowCount(0)
        top = self.spin_top_depth.value() + (self.spin_bot_depth.value() - self.spin_top_depth.value()) * 0.50
        bot = self.spin_bot_depth.value()
        self._insert_perforation_row(top, bot)

    def _on_perf_table_changed(self, item):
        self._update_perforation_summary()

    def _update_perforation_summary(self):
        tot_len = 0.0
        for r in range(self.table_perfs.rowCount()):
            try:
                t = float(self.table_perfs.item(r, 0).text())
                b = float(self.table_perfs.item(r, 1).text())
                tot_len += max(0.0, b - t)
            except Exception:
                pass
        self.lbl_perf_summary.setText(f"Total Completed Perforation Length: {tot_len:.1f} ft")
        self._update_all_calculations()

    # --------------------------------------------------------------------------
    # Master Physics & Graphic Calculations
    # --------------------------------------------------------------------------
    def _update_all_calculations(self):
        # 1. Boundary checks
        x_val = self.spin_x.value()
        y_val = self.spin_y.value()
        if 0 <= x_val <= self.res_len and 0 <= y_val <= self.res_width:
            self.lbl_boundary_check.setText("Boundary: Validated within active reservoir grid ✓")
            self.lbl_boundary_check.setStyleSheet("color: #16a34a; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_boundary_check.setText("Boundary Warning: Well coordinates outside reservoir grid!")
            self.lbl_boundary_check.setStyleSheet("color: #dc2626; font-size: 11px; font-weight: bold;")

        # 2. Peaceman Well Index calculation
        top_d = self.spin_top_depth.value()
        bot_d = self.spin_bot_depth.value()
        traj_idx = self.combo_traj_type.currentIndex()
        lat_len = self.spin_lat_len.value() if traj_idx == 1 else 0.0
        rw = self.spin_rw.value()
        skin = self.spin_skin.value()

        # Extract perfs
        perfs = []
        tot_perf_h = 0.0
        for r in range(self.table_perfs.rowCount()):
            try:
                t = float(self.table_perfs.item(r, 0).text())
                b = float(self.table_perfs.item(r, 1).text())
                perfs.append((t, b))
                tot_perf_h += max(0.0, b - t)
            except Exception:
                pass
        if tot_perf_h <= 0.0:
            tot_perf_h = max(bot_d - top_d, 10.0)

        dx, dy, dz = 100.0, 100.0, 20.0
        if traj_idx == 1:
            wi = calculate_peaceman_index_horizontal(
                ky_md=self.res_perm,
                kz_md=self.res_perm * self.res_kv_kh,
                length_lateral_ft=lat_len,
                dy_ft=dy,
                dz_ft=dz,
                r_w_ft=rw,
                skin=skin,
                mu_cp=1.0
            )
        else:
            wi = calculate_peaceman_index_vertical(
                kx_md=self.res_perm,
                ky_md=self.res_perm,
                h_perf_ft=tot_perf_h,
                dx_ft=dx,
                dy_ft=dy,
                r_w_ft=rw,
                skin=skin,
                mu_cp=1.0
            )

        wi = round(float(wi), 2)
        traj_tag = "Horizontal" if traj_idx == 1 else ("Deviated" if traj_idx == 2 else "Vertical")
        self.lbl_header_wi.setText(f"Peaceman WI ({traj_tag}): {wi:.2f} STB/d/psi")
        self.lbl_ipr_wi.setText(f"Peaceman Well Index: {wi:.2f} STB/d/psi")

        # 3. IPR Curve calculation
        p_res = self.res_p_ini
        pb = self.res_pb
        is_inj = (self.combo_role.currentIndex() >= 2)

        if not is_inj:
            # Composite Vogel-Darcy
            if p_res > pb:
                # Undersaturated Darcy Section
                q_b = wi * (p_res - pb)
                q_max = q_b + (wi * pb) / 1.8
            else:
                q_b = 0.0
                q_max = (wi * p_res) / 1.8
            self.lbl_ipr_aof.setText(f"Absolute Open Flow (AOF): {q_max:,.0f} STB/d")
        else:
            q_max = wi * (p_res * 1.5)
            self.lbl_ipr_aof.setText(f"Max Injectivity Potential: {q_max:,.0f} MSCF/d")

        # 4. EPA Class VI Safety Check
        p_frac = self.res_frac_grad * self.res_top
        p_safe_ceiling = self.res_uic_sf * p_frac
        target_bhp = self.spin_bhp.value()

        if is_inj:
            if target_bhp > p_safe_ceiling:
                self.lbl_safety_verdict.setText(f"⚠ EPA CLASS VI VIOLATION: Injection BHP ({target_bhp:.0f} psia) exceeds safe ceiling ({p_safe_ceiling:.0f} psia)!")
                self.lbl_safety_verdict.setStyleSheet("""
                    background: #fee2e2;
                    color: #b91c1c;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 8px 12px;
                    border: 1px solid #fca5a5;
                    border-radius: 4px;
                """)
            else:
                margin = p_safe_ceiling - target_bhp
                self.lbl_safety_verdict.setText(f"✓ Geomechanical Status: Safe Operating Margin (+{margin:.0f} psi below EPA Class VI Ceiling)")
                self.lbl_safety_verdict.setStyleSheet("""
                    background: #dcfce7;
                    color: #15803d;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 8px 12px;
                    border: 1px solid #86efac;
                    border-radius: 4px;
                """)

        # 5. Redraw Trajectory and IPR plots
        self._render_trajectory_preview()
        self._render_ipr_preview(wi, p_res, pb, q_max, is_inj)

    def _render_trajectory_preview(self):
        self.ax_traj.clear()
        traj_idx = self.combo_traj_type.currentIndex()
        traj_name = "Horizontal" if traj_idx == 1 else ("Deviated" if traj_idx == 2 else "Vertical")
        top_d = self.spin_top_depth.value()
        bot_d = self.spin_bot_depth.value()
        lat_len = self.spin_lat_len.value() if traj_idx == 1 else 0.0

        pts = generate_synthetic_well_trajectory(
            surface_x=self.spin_x.value(),
            surface_y=self.spin_y.value(),
            top_tvd=top_d,
            bottom_tvd=bot_d,
            trajectory_type=traj_name,
            lateral_length_ft=lat_len,
            azimuth_deg=self.spin_azimuth.value()
        )
        self.well_path = pts

        # Plot 2D cross section: Horizontal displacement vs Depth
        disp = np.sqrt((pts[:, 0] - pts[0, 0])**2 + (pts[:, 1] - pts[0, 1])**2)
        depths = pts[:, 2]

        # Draw formation top & base
        min_x = 0
        max_x = max(np.max(disp) * 1.25, 500.0)
        self.ax_traj.axhspan(self.res_top, self.res_bot, color="#fef3c7", alpha=0.55, label="Pay Zone Formation")
        self.ax_traj.axhline(self.res_top, color="#d97706", linestyle="--", linewidth=1.1, label=f"Formation Top ({self.res_top:.0f} ft)")
        self.ax_traj.axhline(self.res_bot, color="#b45309", linestyle="--", linewidth=1.1, label=f"Formation Base ({self.res_bot:.0f} ft)")

        # Main wellbore path
        is_inj = (self.combo_role.currentIndex() >= 2)
        well_col = "#dc2626" if is_inj else "#0284c7"
        self.ax_traj.plot(disp, depths, color=well_col, linewidth=2.8, label=f"Wellbore Path ({traj_name})")
        self.ax_traj.scatter([disp[0]], [depths[0]], color=well_col, s=80, marker="v", edgecolors="#000000", zorder=5, label="Wellhead (Surface)")

        # Highlight Perforations in Amber
        for r in range(self.table_perfs.rowCount()):
            try:
                t = float(self.table_perfs.item(r, 0).text())
                b = float(self.table_perfs.item(r, 1).text())
                mask = (depths >= t) & (depths <= b)
                if np.any(mask):
                    self.ax_traj.plot(disp[mask], depths[mask], color="#f59e0b", linewidth=6.0, alpha=0.85, zorder=4)
            except Exception:
                pass

        self.ax_traj.set_xlim(min_x, max_x)
        self.ax_traj.set_ylim(max(np.max(depths) * 1.08, self.res_bot + 100.0), max(0.0, self.res_top - 300.0))
        self.ax_traj.set_xlabel("Horizontal Displacement from Wellhead (ft)", fontsize=9, color="#1e293b")
        self.ax_traj.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9, color="#1e293b")
        self.ax_traj.grid(True, linestyle=":", alpha=0.55)
        self.ax_traj.legend(loc="lower right", fontsize=8, framealpha=0.85)

        self.canvas_traj.draw_idle()

    def _render_ipr_preview(self, wi: float, p_res: float, pb: float, q_max: float, is_inj: bool):
        self.ax_ipr.clear()
        target_bhp = self.spin_bhp.value()

        if not is_inj:
            # Producer Composite Vogel-Darcy curve
            pwf_arr = np.linspace(0.0, p_res, 60)
            q_arr = np.zeros_like(pwf_arr)
            for i, pwf in enumerate(pwf_arr):
                if p_res > pb:
                    if pwf >= pb:
                        q_arr[i] = wi * (p_res - pwf)
                    else:
                        q_b = wi * (p_res - pb)
                        vogel_term = 1.0 - 0.2 * (pwf / pb) - 0.8 * ((pwf / pb)**2)
                        q_arr[i] = q_b + ((wi * pb) / 1.8) * vogel_term
                else:
                    vogel_term = 1.0 - 0.2 * (pwf / p_res) - 0.8 * ((pwf / p_res)**2)
                    q_arr[i] = ((wi * p_res) / 1.8) * vogel_term

            self.ax_ipr.plot(q_arr, pwf_arr, color="#0284c7", linewidth=2.4, label="Composite Vogel-Darcy IPR")
            self.ax_ipr.axhline(p_res, color="gray", linestyle="--", linewidth=1.1, label=f"P_res ({p_res:.0f} psia)")
            self.ax_ipr.axhline(pb, color="#ea580c", linestyle=":", linewidth=1.2, label=f"Bubble Point P_b ({pb:.0f} psia)")

            # Operating point
            op_pwf = np.clip(target_bhp, 14.7, p_res)
            op_q = float(np.interp(op_pwf, pwf_arr[::-1], q_arr[::-1]))
            self.ax_ipr.scatter([op_q], [op_pwf], color="#16a34a", s=90, zorder=6, label=f"Operating Point ({op_q:.0f} STB/d @ {op_pwf:.0f} psi)")
            self.ax_ipr.set_xlabel("Liquid Production Rate q_o (STB/day)", fontsize=9, color="#1e293b")
            self.ax_ipr.set_ylabel("Flowing Bottomhole Pressure P_wf (psia)", fontsize=9, color="#1e293b")
        else:
            # Injector Sandface Deliverability curve
            q_inj_arr = np.linspace(0.0, max(q_max, 1000.0), 60)
            pinj_arr = p_res + (q_inj_arr / max(wi, 0.01))
            self.ax_ipr.plot(q_inj_arr, pinj_arr, color="#dc2626", linewidth=2.4, label="Injection Deliverability Line")
            self.ax_ipr.axhline(p_res, color="gray", linestyle="--", linewidth=1.1, label=f"Reservoir P_res ({p_res:.0f} psia)")

            p_safe_ceiling = self.res_uic_sf * self.res_frac_grad * self.res_top
            self.ax_ipr.axhline(p_safe_ceiling, color="#dc2626", linestyle="--", linewidth=1.3, label=f"EPA Class VI Ceiling ({p_safe_ceiling:.0f} psia)")

            op_q = self.spin_target_rate.value()
            op_pinj = p_res + (op_q / max(wi, 0.01))
            self.ax_ipr.scatter([op_q], [op_pinj], color="#dc2626", s=90, zorder=6, label=f"Operating Point ({op_q:.0f} MSCF/d @ {op_pinj:.0f} psi)")
            self.ax_ipr.set_xlabel("Gas Injection Rate q_inj (MSCF/day)", fontsize=9, color="#1e293b")
            self.ax_ipr.set_ylabel("Sandface Injection Pressure P_sandface (psia)", fontsize=9, color="#1e293b")

        self.ax_ipr.grid(True, linestyle=":", alpha=0.55)
        self.ax_ipr.legend(loc="upper right", fontsize=8, framealpha=0.85)
        self.canvas_ipr.draw_idle()

    # --------------------------------------------------------------------------
    # Save & Export to WellData
    # --------------------------------------------------------------------------
    def _on_save_clicked(self):
        w_name = self.edit_name.text().strip()
        if not w_name:
            QMessageBox.warning(self, "Validation Error", "A well name is required.")
            return

        # Check existing names for collision (except if editing itself)
        initial_name = self.initial_values.get("name") or self.initial_values.get("well_name")
        if w_name in self.existing_names and w_name != initial_name:
            QMessageBox.warning(self, "Name Collision", f"A well named '{w_name}' already exists in the project.")
            return

        self.accept()

    def get_well_data(self) -> Optional[WellData]:
        """Synthesize high-fidelity WellData model compatible with serialization and simulator."""
        try:
            w_name = self.edit_name.text().strip()
            role_idx = self.combo_role.currentIndex()
            is_inj = (role_idx >= 2)
            role_str = "injector" if is_inj else "producer"
            status_str = self.combo_role.currentText()

            sx = float(self.spin_x.value())
            sy = float(self.spin_y.value())
            top_d = float(self.spin_top_depth.value())
            bot_d = float(self.spin_bot_depth.value())
            traj_idx = self.combo_traj_type.currentIndex()
            traj_name = "Horizontal" if traj_idx == 1 else ("Deviated" if traj_idx == 2 else "Vertical")
            lat_len = float(self.spin_lat_len.value()) if traj_idx == 1 else 0.0
            azimuth = float(self.spin_azimuth.value())

            rw = float(self.spin_rw.value())
            skin = float(self.spin_skin.value())

            # Trajectory points
            if self.well_path is not None and len(self.well_path) >= 2:
                pts = self.well_path
            else:
                pts = generate_synthetic_well_trajectory(
                    surface_x=sx,
                    surface_y=sy,
                    top_tvd=top_d,
                    bottom_tvd=bot_d,
                    trajectory_type=traj_name,
                    lateral_length_ft=lat_len,
                    azimuth_deg=azimuth
                )

            depths_arr = np.sort(np.unique(pts[:, 2]))

            # Perforations
            perfs_list = []
            perfs_props = []
            for r in range(self.table_perfs.rowCount()):
                try:
                    t = float(self.table_perfs.item(r, 0).text())
                    b = float(self.table_perfs.item(r, 1).text())
                    perfs_list.append([t, b])
                    perfs_props.append({"top": t, "bottom": b})
                except Exception:
                    pass

            if not perfs_list:
                perfs_list = [[top_d, bot_d]]
                perfs_props = [{"top": top_d, "bottom": bot_d}]

            # Peaceman WI
            h_perf = sum(abs(p[1] - p[0]) for p in perfs_list)
            dx, dy, dz = 100.0, 100.0, 20.0
            if traj_idx == 1:
                wi = calculate_peaceman_index_horizontal(
                    ky_md=self.res_perm,
                    kz_md=self.res_perm * self.res_kv_kh,
                    length_lateral_ft=max(lat_len, 100.0),
                    dy_ft=dy,
                    dz_ft=dz,
                    r_w_ft=rw,
                    skin=skin
                )
            else:
                wi = calculate_peaceman_index_vertical(
                    kx_md=self.res_perm,
                    ky_md=self.res_perm,
                    h_perf_ft=max(h_perf, 10.0),
                    dx_ft=dx,
                    dy_ft=dy,
                    r_w_ft=rw,
                    skin=skin
                )

            metadata = {
                "type": role_str,
                "status": status_str,
                "SurfaceX": sx,
                "SurfaceY": sy,
                "surface_x": sx,
                "surface_y": sy,
                "TopDepth": top_d,
                "BottomDepth": bot_d,
                "TrajectoryType": traj_name,
                "trajectory_type": traj_name,
                "lateral_length": lat_len,
                "lateral_length_ft": lat_len,
                "azimuth_deg": azimuth,
                "peaceman_well_index": round(float(wi), 2),
                "control_mode": "rate" if self.combo_control_mode.currentIndex() == 0 else "bhp",
                "target_rate": float(self.spin_target_rate.value()),
                "target_bhp": float(self.spin_bhp.value()),
                "wellbore_radius_ft": rw,
                "skin_factor": skin,
            }

            properties = {
                "WellboreRadius": np.array([rw]),
                "SkinFactor": np.array([skin]),
                "PeacemanWellIndex": np.array([float(wi)]),
            }

            return WellData(
                name=w_name,
                depths=depths_arr,
                properties=properties,
                units={"WellboreRadius": "ft", "SkinFactor": "dimensionless", "PeacemanWellIndex": "STB/d/psi"},
                metadata=metadata,
                perforation_properties=perfs_props,
                well_path=pts,
                skin_factor=skin,
                wellbore_radius_ft=rw,
                perforations=perfs_list,
                well_index=float(wi),
            )
        except Exception as e:
            logger.error(f"Failed to build WellData: {e}", exc_info=True)
            QMessageBox.critical(self, "Data Error", f"Could not construct well data: {e}")
            return None


# Backwards compatibility alias
ManualWellDialog = ComprehensiveWellEditorDialog
