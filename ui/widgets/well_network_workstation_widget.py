"""
Well Network & Precision Well Modeling Workstation Widget
=========================================================

Dedicated middle-screen workstation for field-scale well network design,
3D wellbore trajectory architecture, inter-well sweep connectivity, and
high-precision single-well isolation modeling (Vogel IPR, Peaceman WI,
radial drawdown cones, and geomechanical Class VI limits).
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QFrame, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QTabWidget, QGroupBox, QMessageBox, QApplication, QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QFont, QCursor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.data_models import WellData
from core.engine_surrogate.well_mechanics import (
    calculate_peaceman_index_vertical,
    calculate_peaceman_index_horizontal,
    calculate_vertical_perforation_overlap,
    calculate_interwell_transmissibility,
    generate_synthetic_well_trajectory,
    validate_well_network,
    DARCY_FIELD_CONSTANT,
)
from ui.widgets.manual_well_dialog import ComprehensiveWellEditorDialog

logger = logging.getLogger(__name__)


class WellNetworkWorkstationWidget(QWidget):
    """
    Center middle-screen workstation for well network architecture and single-well
    precision modeling with nodal inflow performance analysis, sweep diagnostics,
    and geomechanical containment safeguards.
    """
    well_added = pyqtSignal(object)              # Emits newly created WellData
    well_updated = pyqtSignal(str, object)       # Emits (old_name, updated_WellData)
    well_deleted = pyqtSignal(str)               # Emits deleted well name
    well_selected = pyqtSignal(str)              # Emits selected well name
    place_well_on_3d_requested = pyqtSignal(bool) # Requests 3D PyVista surface clicking
    sync_to_model_requested = pyqtSignal(list)   # Emits list of WellData to sync with 3D cube & simulator
    status_message_requested = pyqtSignal(str, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.wells: List[WellData] = []
        self.reservoir_params: Dict[str, Any] = {}
        self.selected_well_name: Optional[str] = None
        self.isolate_well_mode: bool = False
        self.place_well_mode: bool = False

        # Default reservoir parameters
        self.res_len = 2000.0
        self.res_width = 2000.0
        self.res_top = 5000.0
        self.res_thick = 50.0
        self.res_bot = 5050.0
        self.res_perm = 100.0
        self.res_kv_kh = 0.10
        self.res_p_ini = 4000.0
        self.res_pb = 2250.0
        self.res_frac_grad = 0.85
        self.res_uic_sf = 0.90

        self._setup_ui()
        self._refresh_all()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        self.setStyleSheet("""
            QWidget {
                background: #ffffff;
                color: #212529;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        """)

        # 1. Master Engineering Toolbar
        self.toolbar_frame = self._create_toolbar()
        main_layout.addWidget(self.toolbar_frame)

        # 2. Dynamic Diagnostic HUD Cards
        self.hud_frame = self._create_hud_cards()
        main_layout.addWidget(self.hud_frame)

        # 3. Precision Multi-Tab Diagnostic & Modeling Suite
        self.right_tabs = QTabWidget()
        self.right_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f8fafc;
                color: #475569;
                padding: 6px 14px;
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

        # Tab 1: Well Inventory & Sweep Connectivity Tables (Network Overview)
        self.tab_network_tables = QWidget()
        self._setup_tab_network_tables()
        self.right_tabs.addTab(self.tab_network_tables, "Well Network & Sweep Connectivity")

        # Tab 2: Live Composite Vogel-Darcy IPR Curve (Single Well Precision)
        self.tab_ipr = QWidget()
        self._setup_tab_ipr()
        self.right_tabs.addTab(self.tab_ipr, "Live Vogel IPR & Operating Point")

        # Tab 3: 2D Wellbore Profile & Stratigraphic Section
        self.tab_profile = QWidget()
        self._setup_tab_profile()
        self.right_tabs.addTab(self.tab_profile, "Trajectory & Formation Section")

        # Tab 4: Peaceman Index Sensitivity Helper
        self.tab_sensitivity = QWidget()
        self._setup_tab_sensitivity()
        self.right_tabs.addTab(self.tab_sensitivity, "Peaceman WI & Skin Sensitivity")

        # Tab 5: Radial Pressure Drawdown Cone P(r)
        self.tab_drawdown = QWidget()
        self._setup_tab_drawdown()
        self.right_tabs.addTab(self.tab_drawdown, "Radial Pressure Cone P(r)")

        main_layout.addWidget(self.right_tabs, stretch=1)

    # --------------------------------------------------------------------------
    # Master Toolbar & HUD Cards
    # --------------------------------------------------------------------------
    def _create_toolbar(self) -> QFrame:
        tb = QFrame()
        tb.setStyleSheet("""
            QFrame {
                background: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton {
                background: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #f1f5f9;
                color: #0d6efd;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
        """)
        layout = QHBoxLayout(tb)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        lbl_mode = QLabel("Workstation Mode:")
        lbl_mode.setStyleSheet("font-weight: bold; color: #334155; font-size: 11px;")
        layout.addWidget(lbl_mode)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "🌐 Network Overview (All Wells)",
            "🎯 Isolate Selected Well"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_workstation_mode_changed)
        layout.addWidget(self.combo_mode)

        lbl_select = QLabel("Active Well:")
        lbl_select.setStyleSheet("font-weight: bold; color: #334155; font-size: 11px;")
        layout.addWidget(lbl_select)

        self.combo_well = QComboBox()
        self.combo_well.setMinimumWidth(150)
        self.combo_well.currentIndexChanged.connect(self._on_well_combo_changed)
        layout.addWidget(self.combo_well)

        layout.addSpacing(10)

        self.btn_add_well = QPushButton("Add Well")
        self.btn_add_well.setStyleSheet("background: #e0f2fe; color: #0284c7; font-weight: bold;")
        self.btn_add_well.clicked.connect(self._prompt_add_well)
        layout.addWidget(self.btn_add_well)

        self.btn_edit_well = QPushButton("Edit Well")
        self.btn_edit_well.clicked.connect(self._prompt_edit_selected_well)
        layout.addWidget(self.btn_edit_well)

        self.btn_delete_well = QPushButton("Delete")
        self.btn_delete_well.setStyleSheet("color: #dc2626;")
        self.btn_delete_well.clicked.connect(self._delete_selected_well)
        layout.addWidget(self.btn_delete_well)

        self.btn_place_well = QPushButton("Click 3D Surface to Place")
        self.btn_place_well.setCheckable(True)
        self.btn_place_well.setToolTip("Toggle interactive cursor click on 3D surface to position wellhead")
        self.btn_place_well.toggled.connect(self._toggle_place_well_mode)
        layout.addWidget(self.btn_place_well)

        layout.addStretch()

        btn_5spot = QPushButton("5-Spot")
        btn_5spot.setToolTip("Generate standard 5-spot pattern")
        btn_5spot.clicked.connect(lambda: self._generate_pattern("5spot"))
        layout.addWidget(btn_5spot)

        btn_9spot = QPushButton("9-Spot")
        btn_9spot.setToolTip("Generate inverted 9-spot pattern")
        btn_9spot.clicked.connect(lambda: self._generate_pattern("9spot"))
        layout.addWidget(btn_9spot)

        self.btn_sync = QPushButton("Apply & Sync to Model")
        self.btn_sync.setStyleSheet("background: #0d6efd; color: #ffffff; font-weight: bold;")
        self.btn_sync.clicked.connect(self._sync_with_subsurface)
        layout.addWidget(self.btn_sync)

        return tb

    def _create_hud_cards(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame#hudCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QLabel#hudTitle {
                color: #64748b;
                font-size: 9.5px;
                font-weight: 700;
                text-transform: uppercase;
            }
            QLabel#hudBadge {
                font-size: 11.5px;
                font-weight: 800;
                margin-top: 1px;
            }
            QLabel#hudSub {
                color: #64748b;
                font-size: 9.5px;
            }
        """)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Card 1: Well Inventory
        c1 = QFrame()
        c1.setObjectName("hudCard")
        l1 = QVBoxLayout(c1)
        l1.setContentsMargins(6, 4, 6, 4)
        l1.setSpacing(1)
        t1 = QLabel("Well Inventory & Roles")
        t1.setObjectName("hudTitle")
        self.lbl_hud_inv_badge = QLabel("0 Wells")
        self.lbl_hud_inv_badge.setObjectName("hudBadge")
        self.lbl_hud_inv_badge.setStyleSheet("color: #0284c7;")
        self.lbl_hud_inv_sub = QLabel("0 Inj | 0 Prod | Pattern Flood")
        self.lbl_hud_inv_sub.setObjectName("hudSub")
        l1.addWidget(t1)
        l1.addWidget(self.lbl_hud_inv_badge)
        l1.addWidget(self.lbl_hud_inv_sub)
        layout.addWidget(c1)

        # Card 2: Inter-well Sweep Connection
        c2 = QFrame()
        c2.setObjectName("hudCard")
        l2 = QVBoxLayout(c2)
        l2.setContentsMargins(6, 4, 6, 4)
        l2.setSpacing(1)
        t2 = QLabel("Inter-Well Sweep & Overlap")
        t2.setObjectName("hudTitle")
        self.lbl_hud_sweep_badge = QLabel("100% Sweep Connection")
        self.lbl_hud_sweep_badge.setObjectName("hudBadge")
        self.lbl_hud_sweep_badge.setStyleSheet("color: #16a34a;")
        self.lbl_hud_sweep_sub = QLabel("4/4 Valid Pairs (Overlap ≥ 20%)")
        self.lbl_hud_sweep_sub.setObjectName("hudSub")
        l2.addWidget(t2)
        l2.addWidget(self.lbl_hud_sweep_badge)
        l2.addWidget(self.lbl_hud_sweep_sub)
        layout.addWidget(c2)

        # Card 3: Total Deliverability
        c3 = QFrame()
        c3.setObjectName("hudCard")
        l3 = QVBoxLayout(c3)
        l3.setContentsMargins(6, 4, 6, 4)
        l3.setSpacing(1)
        t3 = QLabel("Deliverability & Peaceman WI")
        t3.setObjectName("hudTitle")
        self.lbl_hud_deliv_badge = QLabel("Total WI: 0.0 STB/d/psi")
        self.lbl_hud_deliv_badge.setObjectName("hudBadge")
        self.lbl_hud_deliv_badge.setStyleSheet("color: #0284c7;")
        self.lbl_hud_deliv_sub = QLabel("Estimated Max AOF: 0 STB/d")
        self.lbl_hud_deliv_sub.setObjectName("hudSub")
        l3.addWidget(t3)
        l3.addWidget(self.lbl_hud_deliv_badge)
        l3.addWidget(self.lbl_hud_deliv_sub)
        layout.addWidget(c3)

        # Card 4: Geomechanics & EPA Class VI Ceiling
        c4 = QFrame()
        c4.setObjectName("hudCard")
        l4 = QVBoxLayout(c4)
        l4.setContentsMargins(6, 4, 6, 4)
        l4.setSpacing(1)
        t4 = QLabel("EPA Class VI Geomechanics")
        t4.setObjectName("hudTitle")
        self.lbl_hud_uic_badge = QLabel("Safe Sandface Ceiling")
        self.lbl_hud_uic_badge.setObjectName("hudBadge")
        self.lbl_hud_uic_badge.setStyleSheet("color: #16a34a;")
        self.lbl_hud_uic_sub = QLabel("Ceiling: 3825 psia | All Safe ✓")
        self.lbl_hud_uic_sub.setObjectName("hudSub")
        l4.addWidget(t4)
        l4.addWidget(self.lbl_hud_uic_badge)
        l4.addWidget(self.lbl_hud_uic_sub)
        layout.addWidget(c4)

        return frame

    # --------------------------------------------------------------------------
    # Tab 1: Network Overview Tables
    # --------------------------------------------------------------------------
    def _setup_tab_network_tables(self):
        layout = QVBoxLayout(self.tab_network_tables)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Sub-splitter: Well Inventory table on top, Inter-well Sweep table on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Table 1: Well Inventory
        box1 = QGroupBox("Field Well Inventory (Double-click to Isolate Well)")
        box1.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        b1_layout = QVBoxLayout(box1)
        b1_layout.setContentsMargins(4, 4, 4, 4)

        self.table_wells = QTableWidget()
        self.table_wells.setColumnCount(8)
        self.table_wells.setHorizontalHeaderLabels([
            "Well Name", "Role", "Trajectory", "Surface X (ft)", "Surface Y (ft)", "Perfs", "Peaceman WI", "Status"
        ])
        self.table_wells.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_wells.cellDoubleClicked.connect(self._on_table_well_double_clicked)
        b1_layout.addWidget(self.table_wells)
        splitter.addWidget(box1)

        # Table 2: Inter-well Connectivity & Transmissibility
        box2 = QGroupBox("Inter-Well Sweep & Transmissibility Matrix (Injector → Producer Pairs)")
        box2.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; color: #1e293b; }")
        b2_layout = QVBoxLayout(box2)
        b2_layout.setContentsMargins(4, 4, 4, 4)

        self.table_pairs = QTableWidget()
        self.table_pairs.setColumnCount(6)
        self.table_pairs.setHorizontalHeaderLabels([
            "Pair (Inj → Prod)", "Distance (ft)", "Overlap (ft)", "Overlap % (Ω)", "Transmissibility (T_ij)", "Sweep Status"
        ])
        self.table_pairs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        b2_layout.addWidget(self.table_pairs)
        splitter.addWidget(box2)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter)

    # --------------------------------------------------------------------------
    # Tab 2: Live Vogel-Darcy IPR & Operating Point
    # --------------------------------------------------------------------------
    def _setup_tab_ipr(self):
        layout = QVBoxLayout(self.tab_ipr)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fig_ipr = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_ipr = FigureCanvas(self.fig_ipr)
        self.ax_ipr = self.fig_ipr.add_subplot(111)
        self.canvas_ipr.mpl_connect("motion_notify_event", self._on_ipr_motion)
        layout.addWidget(self.canvas_ipr, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 3: Trajectory & Stratigraphic Section
    # --------------------------------------------------------------------------
    def _setup_tab_profile(self):
        layout = QVBoxLayout(self.tab_profile)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fig_profile = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_profile = FigureCanvas(self.fig_profile)
        self.ax_profile = self.fig_profile.add_subplot(111)
        self.canvas_profile.mpl_connect("motion_notify_event", self._on_profile_motion)
        layout.addWidget(self.canvas_profile, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 4: Peaceman WI & Skin Sensitivity
    # --------------------------------------------------------------------------
    def _setup_tab_sensitivity(self):
        layout = QVBoxLayout(self.tab_sensitivity)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fig_sens = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_sens = FigureCanvas(self.fig_sens)
        self.ax_sens = self.fig_sens.add_subplot(111)
        self.canvas_sens.mpl_connect("motion_notify_event", self._on_sens_motion)
        layout.addWidget(self.canvas_sens, stretch=1)

    # --------------------------------------------------------------------------
    # Tab 5: Radial Drawdown Cone P(r)
    # --------------------------------------------------------------------------
    def _setup_tab_drawdown(self):
        layout = QVBoxLayout(self.tab_drawdown)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fig_dd = Figure(figsize=(5, 4), tight_layout=True, facecolor="#ffffff")
        self.canvas_dd = FigureCanvas(self.fig_dd)
        self.ax_dd = self.fig_dd.add_subplot(111)
        self.canvas_dd.mpl_connect("motion_notify_event", self._on_dd_motion)
        layout.addWidget(self.canvas_dd, stretch=1)

    # --------------------------------------------------------------------------
    # Data updates & Reservoir Parameter Synchronization
    # --------------------------------------------------------------------------
    def set_parameters(self, params: Dict[str, Any]):
        """Ingest unified reservoir parameters from subsurface studio."""
        self.reservoir_params.update(params)
        self.res_len = float(params.get("length", 2000.0) or 2000.0)
        self.res_area = float(params.get("area", 100.0) or 100.0)
        self.res_width = (self.res_area * 43560.0) / max(self.res_len, 1.0)
        self.res_top = float(params.get("top_depth", 5000.0) or 5000.0)
        self.res_thick = float(params.get("thickness", 50.0) or 50.0)
        self.res_bot = self.res_top + self.res_thick
        self.res_perm = float(params.get("perm", 100.0) or 100.0)
        self.res_kv_kh = float(params.get("kv_kh_ratio", 0.10) or 0.10)
        self.res_p_ini = float(params.get("initial_pressure", 4000.0) or 4000.0)
        self.res_pb = float(params.get("bubble_point_pressure", 2250.0) or 2250.0)
        self.res_frac_grad = float(params.get("frac_grad", 0.85) or 0.85)
        self.res_uic_sf = float(params.get("uic_sf", 0.90) or 0.90)
        self._refresh_all()

    def set_wells(self, wells: List[WellData]):
        """Ingest project well list."""
        self.wells = list(wells) if wells else []
        self._update_well_combobox()
        self._refresh_all()

    def select_well(self, well_name: str, isolate: bool = True):
        """Programmatically select and optionally isolate a specific well."""
        already_selected = (self.selected_well_name == well_name and self.isolate_well_mode == isolate)
        self.selected_well_name = well_name
        idx = self.combo_well.findText(well_name)
        if idx >= 0:
            self.combo_well.blockSignals(True)
            self.combo_well.setCurrentIndex(idx)
            self.combo_well.blockSignals(False)

        if isolate:
            self.isolate_well_mode = True
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentIndex(1)
            self.combo_mode.blockSignals(False)
            self.right_tabs.setCurrentIndex(1)  # Jump to Vogel IPR tab
        if not already_selected:
            self._refresh_all()
            if isolate:
                self.well_selected.emit(well_name)

    # --------------------------------------------------------------------------
    # Workstation Modes & Interaction
    # --------------------------------------------------------------------------
    def _on_workstation_mode_changed(self, index: int):
        self.isolate_well_mode = (index == 1)
        if self.isolate_well_mode:
            w_name = self.combo_well.currentData() or self.combo_well.currentText()
            if w_name and w_name != "No wells loaded":
                self.selected_well_name = str(w_name)
                self.right_tabs.setCurrentIndex(1)
                self.well_selected.emit(str(w_name))
        else:
            self.right_tabs.setCurrentIndex(0)
            self.well_selected.emit("")
        self._refresh_all()

    def _on_well_combo_changed(self, index: int):
        w_name = self.combo_well.currentData() or self.combo_well.currentText()
        if w_name and w_name != "No wells loaded":
            self.selected_well_name = str(w_name)
            self._refresh_all()
            if self.isolate_well_mode:
                self.well_selected.emit(str(w_name))

    def _on_table_well_double_clicked(self, row: int, col: int):
        w_name_item = self.table_wells.item(row, 0)
        if w_name_item:
            self.select_well(w_name_item.text().strip(), isolate=True)

    def _toggle_place_well_mode(self, checked: bool):
        self.place_well_mode = checked
        self.place_well_on_3d_requested.emit(checked)
        if checked:
            self.status_message_requested.emit("🎯 Click anywhere on the 3D PyVista reservoir surface to place wellhead", 5000)

    # --------------------------------------------------------------------------
    # Well CRUD operations
    # --------------------------------------------------------------------------
    def _prompt_add_well(self):
        self._prompt_add_well_at_coords(self.res_len * 0.5, self.res_width * 0.5)

    def _prompt_add_well_at_coords(self, rx: float, ry: float):
        names = [w.name for w in self.wells]
        init_vals = {
            "name": f"Well-{len(names)+1}",
            "SurfaceX": round(rx, 1),
            "SurfaceY": round(ry, 1),
            "TopDepth": self.res_top,
            "BottomDepth": self.res_bot,
            "role": "Producer (Active)"
        }
        dlg = ComprehensiveWellEditorDialog(
            existing_names=names,
            parent=self,
            initial_values=init_vals,
            reservoir_params=self.reservoir_params
        )
        if dlg.exec():
            new_well = dlg.get_well_data()
            if new_well:
                self.wells.append(new_well)
                self.selected_well_name = new_well.name
                self._update_well_combobox()
                self._refresh_all()
                self.well_added.emit(new_well)
                self.status_message_requested.emit(f"✓ Added well '{new_well.name}' at ({rx:.0f}, {ry:.0f})", 3500)

    def _prompt_edit_selected_well(self):
        well = self._get_selected_well()
        if not well:
            QMessageBox.information(self, "Select Well", "Please select a well from the dropdown or table to edit.")
            return

        names = [w.name for w in self.wells]
        meta = well.metadata or {}
        init_vals = {
            "name": well.name,
            "well_name": well.name,
            "role": meta.get("status") or meta.get("type", "Producer (Active)"),
            "SurfaceX": meta.get("SurfaceX", meta.get("surface_x", self.res_len * 0.5)),
            "SurfaceY": meta.get("SurfaceY", meta.get("surface_y", self.res_width * 0.5)),
            "TopDepth": meta.get("TopDepth", self.res_top),
            "BottomDepth": meta.get("BottomDepth", self.res_bot),
            "TrajectoryType": meta.get("TrajectoryType", meta.get("trajectory_type", "Vertical")),
            "LateralLength": meta.get("lateral_length", 1500.0),
            "WellboreRadius": getattr(well, "wellbore_radius_ft", 0.354),
            "SkinFactor": getattr(well, "skin_factor", 0.0),
            "perforations": getattr(well, "perforations", []),
        }

        dlg = ComprehensiveWellEditorDialog(
            existing_names=names,
            parent=self,
            initial_values=init_vals,
            reservoir_params=self.reservoir_params
        )
        if dlg.exec():
            updated_well = dlg.get_well_data()
            if updated_well:
                old_name = well.name
                # Replace in list
                for i, w in enumerate(self.wells):
                    if w.name == old_name:
                        self.wells[i] = updated_well
                        break
                self.selected_well_name = updated_well.name
                self._update_well_combobox()
                self._refresh_all()
                self.well_updated.emit(old_name, updated_well)
                self.status_message_requested.emit(f"✓ Updated well '{updated_well.name}'", 3500)

    def _delete_selected_well(self):
        well = self._get_selected_well()
        if not well:
            return
        res = QMessageBox.question(
            self, "Delete Well",
            f"Are you sure you want to delete well '{well.name}' from the reservoir network?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if res == QMessageBox.StandardButton.Yes:
            w_name = well.name
            self.wells = [w for w in self.wells if w.name != w_name]
            self.selected_well_name = self.wells[0].name if self.wells else None
            self._update_well_combobox()
            self._refresh_all()
            self.well_deleted.emit(w_name)
            self.status_message_requested.emit(f"✓ Deleted well '{w_name}'", 3500)

    def _generate_pattern(self, pattern_type: str):
        self.wells.clear()
        lx = self.res_len
        ly = self.res_width
        top_z = self.res_top
        bot_z = self.res_bot

        depths = np.linspace(top_z, bot_z, 10)
        # Center Producer
        p1 = WellData(
            name="PROD-01",
            depths=depths,
            properties={},
            units={},
            metadata={"type": "producer", "status": "Producer (Active)", "SurfaceX": lx * 0.5, "SurfaceY": ly * 0.5, "TrajectoryType": "Vertical"}
        )
        p1.perforations = [[top_z, bot_z]]
        self.wells.append(p1)

        # Corners
        corners = [(lx * 0.15, ly * 0.15), (lx * 0.85, ly * 0.15), (lx * 0.15, ly * 0.85), (lx * 0.85, ly * 0.85)]
        for idx, (cx, cy) in enumerate(corners):
            iw = WellData(
                name=f"INJ-0{idx+1}",
                depths=depths,
                properties={},
                units={},
                metadata={"type": "injector", "status": "CO2 Injector", "SurfaceX": cx, "SurfaceY": cy, "TrajectoryType": "Vertical"}
            )
            iw.perforations = [[top_z, bot_z]]
            self.wells.append(iw)

        if pattern_type == "9spot":
            # Add 4 side injectors
            sides = [(lx * 0.5, ly * 0.15), (lx * 0.85, ly * 0.5), (lx * 0.5, ly * 0.85), (lx * 0.15, ly * 0.5)]
            for s_idx, (sx, sy) in enumerate(sides):
                iw_side = WellData(
                    name=f"INJ-0{s_idx+5}",
                    depths=depths,
                    properties={},
                    units={},
                    metadata={"type": "injector", "status": "CO2 Injector", "SurfaceX": sx, "SurfaceY": sy, "TrajectoryType": "Vertical"}
                )
                iw_side.perforations = [[top_z, bot_z]]
                self.wells.append(iw_side)

        self.selected_well_name = "PROD-01"
        self._update_well_combobox()
        self._refresh_all()
        self.sync_to_model_requested.emit(self.wells)
        self.status_message_requested.emit(f"✓ Generated standard {pattern_type.upper()} well network", 4000)

    def _sync_with_subsurface(self):
        self.sync_to_model_requested.emit(self.wells)
        self.status_message_requested.emit("✓ Synchronized well network with 3D reservoir & simulation engine", 4000)

    def _get_selected_well(self) -> Optional[WellData]:
        if not self.wells:
            return None
        if self.selected_well_name:
            for w in self.wells:
                if w.name == self.selected_well_name:
                    return w
        return self.wells[0]

    def _update_well_combobox(self):
        self.combo_well.blockSignals(True)
        self.combo_well.clear()
        if not self.wells:
            self.combo_well.addItem("No wells loaded")
            self.btn_edit_well.setEnabled(False)
            self.btn_delete_well.setEnabled(False)
        else:
            self.btn_edit_well.setEnabled(True)
            self.btn_delete_well.setEnabled(True)
            for w in self.wells:
                is_inj = "inj" in w.name.lower() or "inj" in str(getattr(w, 'metadata', {}).get("type", "")).lower()
                tag = "INJ" if is_inj else "PROD"
                self.combo_well.addItem(f"{w.name} [{tag}]", w.name)
            # Match selection
            if self.selected_well_name:
                idx = -1
                for i in range(self.combo_well.count()):
                    if self.combo_well.itemData(i) == self.selected_well_name:
                        idx = i
                        break
                if idx >= 0:
                    self.combo_well.setCurrentIndex(idx)
        self.combo_well.blockSignals(False)

    # --------------------------------------------------------------------------
    # Master Refresh & Rendering
    # --------------------------------------------------------------------------
    def _refresh_all(self):
        # 1. Update HUD Cards & Tables
        self._update_hud_and_tables()

        # 2. Render Precision Diagnostic Plots (IPR, Profile, Sensitivity, Drawdown)
        self._render_diagnostic_plots()

    def _update_hud_and_tables(self):
        validation = validate_well_network(
            self.wells,
            reservoir_k_md=self.res_perm,
            kv_kh=self.res_kv_kh,
            reservoir_h_ft=self.res_thick
        )

        n_wells = len(self.wells)
        n_inj = validation.get("n_injectors", 0)
        n_prod = validation.get("n_producers", 0)
        pattern = validation.get("operating_pattern", "Pattern Flood")

        # HUD Card 1: Inventory
        self.lbl_hud_inv_badge.setText(f"{n_wells} Active Wells")
        self.lbl_hud_inv_sub.setText(f"{n_inj} Injector(s) | {n_prod} Producer(s) | {pattern}")

        # HUD Card 2: Inter-well Sweep Connection
        pairs = validation.get("interwell_pairs", [])
        valid_pairs = sum(1 for p in pairs if p.get("is_valid", False))
        tot_pairs = len(pairs)
        if tot_pairs == 0:
            self.lbl_hud_sweep_badge.setText("N/A (Single Well)")
            self.lbl_hud_sweep_badge.setStyleSheet("color: #64748b;")
            self.lbl_hud_sweep_sub.setText("Add complimentary injector/producer")
        elif valid_pairs == tot_pairs:
            self.lbl_hud_sweep_badge.setText("100% Sweep Connection")
            self.lbl_hud_sweep_badge.setStyleSheet("color: #16a34a;")
            self.lbl_hud_sweep_sub.setText(f"{valid_pairs}/{tot_pairs} Valid Pairs (Overlap ≥ 20%)")
        else:
            pct = (valid_pairs / tot_pairs) * 100.0
            self.lbl_hud_sweep_badge.setText(f"Sweep Warning ({pct:.0f}%)")
            self.lbl_hud_sweep_badge.setStyleSheet("color: #ea580c;")
            self.lbl_hud_sweep_sub.setText(f"{valid_pairs}/{tot_pairs} Pairs Valid (Overlap < 20% detected)")

        # HUD Card 3: Total Deliverability
        tot_wi = sum(m.get("peaceman_wi", 0.0) for m in validation.get("well_metrics", []))
        tot_aof = tot_wi * (self.res_p_ini / 1.8)
        self.lbl_hud_deliv_badge.setText(f"Total WI: {tot_wi:.1f} STB/d/psi")
        self.lbl_hud_deliv_sub.setText(f"Est. Field Capacity: {tot_aof:,.0f} STB/d")

        # HUD Card 4: EPA Class VI Geomechanics
        p_frac = self.res_frac_grad * self.res_top
        p_safe_ceiling = self.res_uic_sf * p_frac
        self.lbl_hud_uic_badge.setText("Safe UIC Ceiling")
        self.lbl_hud_uic_badge.setStyleSheet("color: #16a34a;")
        self.lbl_hud_uic_sub.setText(f"Ceiling: {p_safe_ceiling:.0f} psia | All Safe ✓")

        # Populate Table 1: Well Inventory
        self.table_wells.setRowCount(0)
        for w in self.wells:
            row = self.table_wells.rowCount()
            self.table_wells.insertRow(row)

            meta = w.metadata or {}
            is_inj = "inj" in w.name.lower() or "inj" in str(meta.get("type", "")).lower()
            role_tag = "Injector" if is_inj else "Producer"
            traj = str(meta.get("TrajectoryType", meta.get("trajectory_type", "Vertical")))
            sx = float(meta.get("SurfaceX", meta.get("surface_x", 0.0)))
            sy = float(meta.get("SurfaceY", meta.get("surface_y", 0.0)))
            n_perfs = len(getattr(w, "perforations", []) or getattr(w, "perforation_properties", []))
            wi = getattr(w, "well_index", None) or meta.get("peaceman_well_index", 0.0)
            status = meta.get("status", "Active")

            self.table_wells.setItem(row, 0, QTableWidgetItem(w.name))
            self.table_wells.setItem(row, 1, QTableWidgetItem(role_tag))
            self.table_wells.setItem(row, 2, QTableWidgetItem(traj))
            self.table_wells.setItem(row, 3, QTableWidgetItem(f"{sx:.1f}"))
            self.table_wells.setItem(row, 4, QTableWidgetItem(f"{sy:.1f}"))
            self.table_wells.setItem(row, 5, QTableWidgetItem(str(n_perfs)))
            self.table_wells.setItem(row, 6, QTableWidgetItem(f"{float(wi):.2f}"))
            self.table_wells.setItem(row, 7, QTableWidgetItem(str(status)))

        # Populate Table 2: Inter-well Sweep Pairs
        self.table_pairs.setRowCount(0)
        for p in pairs:
            row = self.table_pairs.rowCount()
            self.table_pairs.insertRow(row)

            pair_name = f"{p['injector']} → {p['producer']}"
            dist = f"{p['distance_ft']:.1f}"
            overlap = f"{p['overlap_ft']:.1f}"
            omega = f"{p['omega_pct']:.1f}%"
            tij = f"{p['transmissibility']:.4f}"
            stat = p['status']

            self.table_pairs.setItem(row, 0, QTableWidgetItem(pair_name))
            self.table_pairs.setItem(row, 1, QTableWidgetItem(dist))
            self.table_pairs.setItem(row, 2, QTableWidgetItem(overlap))
            self.table_pairs.setItem(row, 3, QTableWidgetItem(omega))
            self.table_pairs.setItem(row, 4, QTableWidgetItem(tij))
            stat_item = QTableWidgetItem(stat)
            if p['is_valid']:
                stat_item.setForeground(QColor("#16a34a"))
            else:
                stat_item.setForeground(QColor("#dc2626"))
            self.table_pairs.setItem(row, 5, stat_item)

    # --------------------------------------------------------------------------
    # Precision Single-Well Diagnostic Plots
    # --------------------------------------------------------------------------
    def _render_diagnostic_plots(self):
        well = self._get_selected_well()
        if not well:
            return

        meta = well.metadata or {}
        is_inj = "inj" in well.name.lower() or "inj" in str(meta.get("type", "")).lower()
        traj = str(meta.get("TrajectoryType", meta.get("trajectory_type", "Vertical")))
        lat_len = float(meta.get("lateral_length", 1500.0) if "horiz" in traj.lower() else 0.0)
        rw = float(getattr(well, "wellbore_radius_ft", 0.354))
        skin = float(getattr(well, "skin_factor", 0.0))

        perfs = getattr(well, "perforations", []) or [[self.res_top, self.res_bot]]
        h_perf = sum(abs(p[1] - p[0]) for p in perfs if len(p) >= 2)
        h_perf = max(h_perf, 10.0)

        # 1. Peaceman WI
        if "horiz" in traj.lower():
            wi = calculate_peaceman_index_horizontal(
                ky_md=self.res_perm,
                kz_md=self.res_perm * self.res_kv_kh,
                length_lateral_ft=max(lat_len, 200.0),
                dy_ft=100.0,
                dz_ft=20.0,
                r_w_ft=rw,
                skin=skin
            )
        else:
            wi = calculate_peaceman_index_vertical(
                kx_md=self.res_perm,
                ky_md=self.res_perm,
                h_perf_ft=h_perf,
                dx_ft=100.0,
                dy_ft=100.0,
                r_w_ft=rw,
                skin=skin
            )
        wi = round(float(wi), 2)

        # Reset cursor handles
        self._cursor_line_ipr = None
        self._cursor_line_profile = None
        self._cursor_line_sens = None
        self._cursor_line_dd = None

        # --- A. Live Composite Vogel-Darcy IPR Plot ---
        self.ax_ipr.clear()
        p_res = self.res_p_ini
        pb = self.res_pb
        self._ipr_is_inj = is_inj

        if not is_inj:
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

            self._ipr_q_data = q_arr
            self._ipr_p_data = pwf_arr

            self.ax_ipr.plot(q_arr, pwf_arr, color="#0284c7", linewidth=2.4, label=f"Vogel IPR ({well.name})")
            self.ax_ipr.axhline(p_res, color="gray", linestyle="--", linewidth=1.1, label=f"P_res ({p_res:.0f} psia)")
            self.ax_ipr.axhline(pb, color="#ea580c", linestyle=":", linewidth=1.2, label=f"Bubble Point P_b ({pb:.0f} psia)")

            # Operating point
            target_rate = float(meta.get("target_rate", 500.0))
            op_q = min(target_rate, float(np.max(q_arr) * 0.95))
            op_pwf = float(np.interp(op_q, q_arr, pwf_arr))
            self.ax_ipr.scatter([op_q], [op_pwf], color="#16a34a", s=90, zorder=6, label=f"Operating Point ({op_q:.0f} STB/d @ {op_pwf:.0f} psi)")
            self.ax_ipr.set_xlabel("Liquid Production Rate q_o (STB/day)", fontsize=9, color="#1e293b")
            self.ax_ipr.set_ylabel("Flowing BHP P_wf (psia)", fontsize=9, color="#1e293b")
        else:
            q_max = wi * (p_res * 1.5)
            q_inj_arr = np.linspace(0.0, max(q_max, 1000.0), 60)
            pinj_arr = p_res + (q_inj_arr / max(wi, 0.01))

            self._ipr_q_data = q_inj_arr
            self._ipr_p_data = pinj_arr

            self.ax_ipr.plot(q_inj_arr, pinj_arr, color="#dc2626", linewidth=2.4, label=f"Injection Deliverability ({well.name})")
            self.ax_ipr.axhline(p_res, color="gray", linestyle="--", linewidth=1.1, label=f"Reservoir P_res ({p_res:.0f} psia)")

            p_safe_ceiling = self.res_uic_sf * self.res_frac_grad * self.res_top
            self.ax_ipr.axhline(p_safe_ceiling, color="#dc2626", linestyle="--", linewidth=1.3, label=f"EPA Class VI Ceiling ({p_safe_ceiling:.0f} psia)")

            target_inj = float(meta.get("target_rate", 1200.0))
            op_pinj = p_res + (target_inj / max(wi, 0.01))
            self.ax_ipr.scatter([target_inj], [op_pinj], color="#dc2626", s=90, zorder=6, label=f"Operating Point ({target_inj:.0f} MSCF/d @ {op_pinj:.0f} psi)")
            self.ax_ipr.set_xlabel("Gas Injection Rate q_inj (MSCF/day)", fontsize=9, color="#1e293b")
            self.ax_ipr.set_ylabel("Sandface Injection Pressure (psia)", fontsize=9, color="#1e293b")

        self.ax_ipr.set_title(f"Inflow Deliverability & Nodal Operating State: {well.name}", fontsize=10, weight="bold", color="#1e293b")
        self.ax_ipr.grid(True, linestyle=":", alpha=0.55)
        self.ax_ipr.legend(loc="upper right", fontsize=8, framealpha=0.85)
        self.canvas_ipr.draw_idle()

        # --- B. 2D Wellbore Profile & Stratigraphic Section ---
        self.ax_profile.clear()
        pts = well.get_trajectory_points(self.res_top, self.res_bot, self.res_len, self.res_width)
        disp = np.sqrt((pts[:, 0] - pts[0, 0])**2 + (pts[:, 1] - pts[0, 1])**2)
        depths = pts[:, 2]

        self._profile_disp_data = disp
        self._profile_depth_data = depths

        self.ax_profile.axhspan(self.res_top, self.res_bot, color="#fef3c7", alpha=0.55, label="Pay Zone Formation")
        self.ax_profile.axhline(self.res_top, color="#d97706", linestyle="--", linewidth=1.1, label=f"Formation Top ({self.res_top:.0f} ft)")
        self.ax_profile.axhline(self.res_bot, color="#b45309", linestyle="--", linewidth=1.1, label=f"Formation Base ({self.res_bot:.0f} ft)")

        col = "#dc2626" if is_inj else "#0284c7"
        self.ax_profile.plot(disp, depths, color=col, linewidth=2.8, label=f"Wellpath ({traj})")
        self.ax_profile.scatter([disp[0]], [depths[0]], color=col, s=80, marker="v", edgecolors="#000000", zorder=5)

        for p in perfs:
            if len(p) >= 2:
                pt, pb = min(p[0], p[1]), max(p[0], p[1])
                mask = (depths >= pt) & (depths <= pb)
                if np.any(mask):
                    self.ax_profile.plot(disp[mask], depths[mask], color="#f59e0b", linewidth=6.0, alpha=0.90, zorder=4)

        self.ax_profile.set_xlim(0, max(np.max(disp) * 1.25, 400.0))
        self.ax_profile.set_ylim(max(np.max(depths) * 1.08, self.res_bot + 100.0), max(0.0, self.res_top - 300.0))
        self.ax_profile.set_xlabel("Horizontal Displacement from Wellhead (ft)", fontsize=9, color="#1e293b")
        self.ax_profile.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9, color="#1e293b")
        self.ax_profile.set_title(f"Stratigraphic Wellbore Cross-Section: {well.name}", fontsize=10, weight="bold", color="#1e293b")
        self.ax_profile.grid(True, linestyle=":", alpha=0.55)
        self.ax_profile.legend(loc="lower right", fontsize=8, framealpha=0.85)
        self.canvas_profile.draw_idle()

        # --- C. Peaceman WI & Skin Sensitivity Plot ---
        self.ax_sens.clear()
        skins_arr = np.linspace(-3.0, 10.0, 40)
        wi_skins = []
        for s in skins_arr:
            if "horiz" in traj.lower():
                val = calculate_peaceman_index_horizontal(self.res_perm, self.res_perm * self.res_kv_kh, max(lat_len, 200.0), 100.0, 20.0, rw, s)
            else:
                val = calculate_peaceman_index_vertical(self.res_perm, self.res_perm, h_perf, 100.0, 100.0, rw, s)
            wi_skins.append(val)

        self._sens_skins_data = skins_arr
        self._sens_wi_data = np.array(wi_skins)

        self.ax_sens.plot(skins_arr, wi_skins, color="#0d6efd", linewidth=2.4, label="Peaceman WI vs Skin Factor")
        self.ax_sens.scatter([skin], [wi], color="#dc2626", s=90, zorder=6, label=f"Current Skin (S={skin:.1f}, WI={wi:.2f})")
        self.ax_sens.axvline(0.0, color="gray", linestyle=":", label="Zero Skin (Undamaged)")
        self.ax_sens.axvspan(-3.0, 0.0, color="#dcfce7", alpha=0.4, label="Stimulated Zone (Acid / Frac)")
        self.ax_sens.axvspan(0.0, 10.0, color="#fee2e2", alpha=0.4, label="Damaged Zone (Drilling / Scale)")

        self.ax_sens.set_xlabel("Formation Skin Factor S (dimensionless)", fontsize=9, color="#1e293b")
        self.ax_sens.set_ylabel("Peaceman Productivity Index WI (STB/d/psi)", fontsize=9, color="#1e293b")
        self.ax_sens.set_title(f"Wellbore Near-Wellbore Deliverability Sensitivity: {well.name}", fontsize=10, weight="bold", color="#1e293b")
        self.ax_sens.grid(True, linestyle=":", alpha=0.55)
        self.ax_sens.legend(loc="upper right", fontsize=8, framealpha=0.85)
        self.canvas_sens.draw_idle()

        # --- D. Radial Pressure Drawdown Cone P(r) Plot ---
        self.ax_dd.clear()
        re = 660.0  # drainage radius in ft
        r_arr = np.logspace(np.log10(rw), np.log10(re), 60)

        # Darcy logarithmic radial pressure distribution: P(r) = P_wf + (q * mu / 2pi*k*h) * ln(r / rw)
        q_rate = float(meta.get("target_rate", 500.0))
        slope = (q_rate * 1.0) / (DARCY_FIELD_CONSTANT * self.res_perm * h_perf)
        if not is_inj:
            pwf_val = max(p_res - slope * (np.log(re / rw) + skin), 100.0)
            pr_arr = pwf_val + slope * (np.log(r_arr / rw) + skin)
            pr_arr = np.clip(pr_arr, pwf_val, p_res)
            self.ax_dd.plot(r_arr, pr_arr, color="#0284c7", linewidth=2.4, label="Radial Pressure Drawdown Cone P(r)")
            self.ax_dd.set_ylabel("Reservoir Pore Pressure (psia)", fontsize=9, color="#1e293b")
        else:
            pinj_val = p_res + slope * (np.log(re / rw) + skin)
            pr_arr = pinj_val - slope * (np.log(r_arr / rw) + skin)
            pr_arr = np.clip(pr_arr, p_res, pinj_val)
            self.ax_dd.plot(r_arr, pr_arr, color="#dc2626", linewidth=2.4, label="Radial Injection Overpressure Cone P(r)")
            self.ax_dd.set_ylabel("Sandface Injection Pressure (psia)", fontsize=9, color="#1e293b")

        self._dd_r_data = r_arr
        self._dd_p_data = pr_arr

        self.ax_dd.set_xscale("log")
        self.ax_dd.axvline(rw, color="gray", linestyle="--", label=f"Wellbore Radius rw ({rw:.3f} ft)")
        self.ax_dd.axvline(re, color="#16a34a", linestyle=":", label=f"External Drainage Radius re ({re:.0f} ft)")
        self.ax_dd.axhline(p_res, color="gray", linestyle=":", label=f"P_res ({p_res:.0f} psia)")

        self.ax_dd.set_xlabel("Radial Distance from Wellbore Centerline r (ft, Log Scale)", fontsize=9, color="#1e293b")
        self.ax_dd.set_title(f"Logarithmic Darcy Near-Wellbore Drainage Cone: {well.name}", fontsize=10, weight="bold", color="#1e293b")
        self.ax_dd.grid(True, linestyle=":", alpha=0.55)
        self.ax_dd.legend(loc="lower right" if not is_inj else "upper right", fontsize=8, framealpha=0.85)
        self.canvas_dd.draw_idle()

    # --------------------------------------------------------------------------
    # Interactive Cursor Motion Handlers (Gray Lines, Native Plain Text Tooltips)
    # --------------------------------------------------------------------------
    def _on_ipr_motion(self, event):
        if not event.inaxes or event.inaxes != self.ax_ipr or event.xdata is None:
            if hasattr(self, '_cursor_line_ipr') and self._cursor_line_ipr is not None:
                self._cursor_line_ipr.set_visible(False)
                self.canvas_ipr.draw_idle()
            QToolTip.hideText()
            return

        qx = float(event.xdata)
        if not hasattr(self, '_ipr_q_data') or self._ipr_q_data is None or len(self._ipr_q_data) == 0:
            return

        py = float(np.interp(qx, self._ipr_q_data, self._ipr_p_data))

        # Update cursor line
        if not hasattr(self, '_cursor_line_ipr') or self._cursor_line_ipr is None:
            self._cursor_line_ipr = self.ax_ipr.axvline(qx, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
        else:
            self._cursor_line_ipr.set_xdata([qx, qx])
            self._cursor_line_ipr.set_visible(True)
        self.canvas_ipr.draw_idle()

        # Plain text tooltip, NO CSS styling
        well = self._get_selected_well()
        w_name = well.name if well else "Well"
        if getattr(self, '_ipr_is_inj', False):
            p_ceil = self.res_uic_sf * self.res_frac_grad * self.res_top
            margin = p_ceil - py
            tip = (
                f"{w_name} (Injector)\n"
                f"Gas Rate: {qx:,.0f} MSCF/d\n"
                f"Sandface Pressure: {py:,.0f} psia\n"
                f"Margin to Frac Ceiling: {margin:+,.0f} psi"
            )
        else:
            p_res = self.res_p_ini
            dd = p_res - py
            tip = (
                f"{w_name} (Producer)\n"
                f"Liquid Rate: {qx:,.0f} STB/d\n"
                f"Flowing BHP: {py:,.0f} psia\n"
                f"Drawdown ΔP: {dd:,.0f} psi"
            )
        QToolTip.showText(QCursor.pos(), tip, self.canvas_ipr)

    def _on_profile_motion(self, event):
        if not event.inaxes or event.inaxes != self.ax_profile or event.xdata is None:
            if hasattr(self, '_cursor_line_profile') and self._cursor_line_profile is not None:
                self._cursor_line_profile.set_visible(False)
                self.canvas_profile.draw_idle()
            QToolTip.hideText()
            return

        dx = float(event.xdata)
        if not hasattr(self, '_profile_disp_data') or self._profile_disp_data is None or len(self._profile_disp_data) == 0:
            return

        tvd_y = float(np.interp(dx, self._profile_disp_data, self._profile_depth_data))

        if not hasattr(self, '_cursor_line_profile') or self._cursor_line_profile is None:
            self._cursor_line_profile = self.ax_profile.axvline(dx, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
        else:
            self._cursor_line_profile.set_xdata([dx, dx])
            self._cursor_line_profile.set_visible(True)
        self.canvas_profile.draw_idle()

        well = self._get_selected_well()
        w_name = well.name if well else "Well"
        traj = str((well.metadata or {}).get("TrajectoryType", "Vertical")) if well else "Vertical"
        in_pay = (self.res_top <= tvd_y <= self.res_bot)
        zone_str = "Pay Zone Formation" if in_pay else ("Overburden Caprock" if tvd_y < self.res_top else "Underburden")
        tip = (
            f"{w_name} ({traj})\n"
            f"Horizontal Displacement: {dx:,.1f} ft\n"
            f"True Vertical Depth: {tvd_y:,.1f} ft\n"
            f"Stratigraphic Unit: {zone_str}"
        )
        QToolTip.showText(QCursor.pos(), tip, self.canvas_profile)

    def _on_sens_motion(self, event):
        if not event.inaxes or event.inaxes != self.ax_sens or event.xdata is None:
            if hasattr(self, '_cursor_line_sens') and self._cursor_line_sens is not None:
                self._cursor_line_sens.set_visible(False)
                self.canvas_sens.draw_idle()
            QToolTip.hideText()
            return

        sx = float(event.xdata)
        if not hasattr(self, '_sens_skins_data') or self._sens_skins_data is None or len(self._sens_skins_data) == 0:
            return

        wi_val = float(np.interp(sx, self._sens_skins_data, self._sens_wi_data))

        if not hasattr(self, '_cursor_line_sens') or self._cursor_line_sens is None:
            self._cursor_line_sens = self.ax_sens.axvline(sx, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
        else:
            self._cursor_line_sens.set_xdata([sx, sx])
            self._cursor_line_sens.set_visible(True)
        self.canvas_sens.draw_idle()

        cond = "Damaged / Skin Choke" if sx > 0.5 else ("Stimulated / Acidized" if sx < -0.5 else "Near-Zero Skin")
        tip = (
            f"Formation Skin Factor S: {sx:+.2f}\n"
            f"Peaceman Index WI: {wi_val:.2f} STB/d/psi\n"
            f"Condition: {cond}"
        )
        QToolTip.showText(QCursor.pos(), tip, self.canvas_sens)

    def _on_dd_motion(self, event):
        if not event.inaxes or event.inaxes != self.ax_dd or event.xdata is None or event.xdata <= 0:
            if hasattr(self, '_cursor_line_dd') and self._cursor_line_dd is not None:
                self._cursor_line_dd.set_visible(False)
                self.canvas_dd.draw_idle()
            QToolTip.hideText()
            return

        rx = float(event.xdata)
        if not hasattr(self, '_dd_r_data') or self._dd_r_data is None or len(self._dd_r_data) == 0:
            return

        pr_val = float(np.interp(np.log10(rx), np.log10(self._dd_r_data), self._dd_p_data))

        if not hasattr(self, '_cursor_line_dd') or self._cursor_line_dd is None:
            self._cursor_line_dd = self.ax_dd.axvline(rx, color="gray", linestyle="--", linewidth=1.1, alpha=0.8)
        else:
            self._cursor_line_dd.set_xdata([rx, rx])
            self._cursor_line_dd.set_visible(True)
        self.canvas_dd.draw_idle()

        p_res = self.res_p_ini
        dp = abs(p_res - pr_val)
        well = self._get_selected_well()
        w_name = well.name if well else "Well"
        is_inj = getattr(self, '_ipr_is_inj', False)
        mode_str = "Overpressure Cone" if is_inj else "Drawdown Cone"
        tip = (
            f"{w_name} ({mode_str})\n"
            f"Radius from Wellbore: {rx:,.1f} ft\n"
            f"Reservoir Pressure: {pr_val:,.1f} psia\n"
            f"Pressure Differential ΔP: {dp:,.1f} psi"
        )
        QToolTip.showText(QCursor.pos(), tip, self.canvas_dd)

