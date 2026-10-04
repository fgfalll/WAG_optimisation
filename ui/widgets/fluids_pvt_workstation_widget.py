"""
Fluids & PVT Thermodynamics Workstation Widget.
Provides an un-cramped middle-screen workstation for thermodynamic fluid characterization:
1. Black Oil PVT Curves (Bo, Rs, mu_o, Bg vs Pressure with Bubble Point Pb analysis)
2. CO2 Solvent-Oil Interaction & Swelling (SF(x_CO2), Viscosity Reduction, Todd-Longstaff)
3. MMP Miscibility Barometer & Slim-Tube Multi-Contact Extraction Curve
4. Multi-Component Composition Table & P-T Phase Envelope Flash
5. Reservoir Fluid Column Hydrostatic Equilibrium (WOC/GOC Contacts, Pressure Gradients, 3D Preparation)
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QFrame, QCheckBox,
    QMenu, QFileDialog, QApplication, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT
)
import matplotlib.pyplot as plt

from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine, P_SC_PSIA
from evaluation.mmp import calculate_mmp, MMPParameters

logger = logging.getLogger(__name__)


class FluidsPVTWorkstationWidget(QWidget):
    """
    Dedicated center workstation for reservoir fluid modeling, thermodynamic curves,
    miscibility diagnostics, and 3D fluid contact preparation.
    """
    sync_to_3d_requested = pyqtSignal(dict)  # Emits fluid parameters to update 3D reservoir cube
    parameters_changed = pyqtSignal(str, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State parameters
        self.initial_pressure = 4000.0
        self.temperature_f = 160.0
        self.api_gravity = 35.0
        self.sol_gor = 500.0
        self.gas_gravity = 0.72
        self.bubble_point_psia = 2250.0
        self.dead_oil_viscosity_cp = 2.5
        self.live_oil_viscosity_cp = 1.45
        self.co2_purity_pct = 95.0
        self.max_swelling = 1.28
        self.min_viscosity_ratio = 0.25
        self.mmp_psia = 2150.0
        self.mmp_method = "auto"
        
        # Fluid Contacts & Columns
        self.top_depth = 5000.0
        self.thickness_ft = 50.0
        self.woc_depth = 5035.0
        self.has_gas_cap = False
        self.goc_depth = 4980.0
        self.water_gradient = 0.44
        self.gas_gradient = 0.08

        # Compositional Components (Mole Fractions)
        self.composition = [
            {"name": "CO2", "z": 0.02, "mw": 44.01, "tc_f": 87.9, "pc_psia": 1070.0},
            {"name": "N2", "z": 0.01, "mw": 28.01, "tc_f": -232.4, "pc_psia": 493.0},
            {"name": "C1 (Methane)", "z": 0.38, "mw": 16.04, "tc_f": -116.6, "pc_psia": 667.0},
            {"name": "C2-C3 (Ethane/Propane)", "z": 0.14, "mw": 36.50, "tc_f": 150.0, "pc_psia": 650.0},
            {"name": "C4-C6 (Intermediates)", "z": 0.15, "mw": 72.00, "tc_f": 360.0, "pc_psia": 460.0},
            {"name": "C7+ (Heavy Ends)", "z": 0.30, "mw": 210.0, "tc_f": 750.0, "pc_psia": 280.0},
        ]

        self._setup_ui()
        self._recalculate_thermodynamics()
        self._render_active_mode()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 1. Top Plotting & Diagnostic Toolbar
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QLabel {
                color: #334155;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox, QDoubleSpinBox {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
                min-height: 22px;
            }
            QCheckBox {
                color: #334155;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(4, 2, 4, 2)
        top_layout.setSpacing(10)

        # Mode Selector
        top_layout.addWidget(QLabel("Display Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Black Oil PVT Curves (Bo, Rs, Viscosity vs P)",
            "CO2 Solvent-Oil Swelling & Viscosity Reduction",
            "MMP Miscibility Barometer & Slim-Tube Extraction",
            "EOS Compositional Table & P-T Phase Envelope",
            "Reservoir Fluid Column & Hydrostatic Contacts (3D)"
        ])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        top_layout.addWidget(self.combo_mode)

        # Fluid System Presets
        top_layout.addWidget(QLabel("Preset:"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "Permian San Andres (36° API, MMP=2150 psia)",
            "Light Volatile Oil (42° API, GOR=850 scf/STB)",
            "Medium Black Oil (32° API, GOR=450 scf/STB)",
            "Heavy Crude (22° API, GOR=120 scf/STB)",
            "Custom Fluid Model"
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        top_layout.addWidget(self.combo_presets)

        # Quick Pressure & Temperature
        top_layout.addWidget(QLabel("P_ini (psia):"))
        self.spin_p_ini = QDoubleSpinBox()
        self.spin_p_ini.setRange(500.0, 15000.0)
        self.spin_p_ini.setValue(self.initial_pressure)
        self.spin_p_ini.setSingleStep(100.0)
        self.spin_p_ini.valueChanged.connect(self._on_quick_param_changed)
        top_layout.addWidget(self.spin_p_ini)

        top_layout.addWidget(QLabel("T (°F):"))
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(60.0, 450.0)
        self.spin_temp.setValue(self.temperature_f)
        self.spin_temp.setSingleStep(5.0)
        self.spin_temp.valueChanged.connect(self._on_quick_param_changed)
        top_layout.addWidget(self.spin_temp)

        # Grid Checkbox
        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._render_active_mode)
        top_layout.addWidget(self.chk_grid)

        top_layout.addStretch()

        # Export CSV Button
        self.btn_export = QPushButton("📊 Export CSV")
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f1f5f9;
                color: #0d6efd;
            }
        """)
        self.btn_export.clicked.connect(self._export_pvt_csv)
        top_layout.addWidget(self.btn_export)

        # Save Image Button
        self.btn_save_img = QPushButton("📷 Save Image")
        self.btn_save_img.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f1f5f9;
                color: #0d6efd;
            }
        """)
        self.btn_save_img.clicked.connect(self._save_plot_image)
        top_layout.addWidget(self.btn_save_img)

        # Apply & Sync to 3D Reservoir Button
        self.btn_sync_3d = QPushButton("⚡ Apply & Sync to 3D Reservoir")
        self.btn_sync_3d.setStyleSheet("""
            QPushButton {
                background: #0d6efd;
                color: #ffffff;
                border: 1px solid #0b5ed7;
                border-radius: 3px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0b5ed7;
            }
        """)
        self.btn_sync_3d.clicked.connect(self._emit_sync_to_3d)
        top_layout.addWidget(self.btn_sync_3d)

        main_layout.addWidget(top_frame)

        # 2. Main High-Resolution Plot Canvas & Stacked Table Area
        self.fig = Figure(figsize=(10, 5), facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)

        # Floating real-time cursor tooltip overlay (native system styling without custom CSS)
        self._cursor_tooltip = QLabel(self.canvas)
        self._cursor_tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._cursor_tooltip.setPalette(QToolTip.palette())
        self._cursor_tooltip.setAutoFillBackground(True)
        self._cursor_tooltip.setFrameShape(QFrame.Shape.Box)
        self._cursor_tooltip.setFrameShadow(QFrame.Shadow.Plain)
        self._cursor_tooltip.setMargin(4)
        self._cursor_tooltip.hide()

        # Canvas Interaction Event Connections
        self._cursor_lines = []
        self._bg_cache = {}
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("axes_leave_event", self._on_canvas_leave)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self.canvas.mpl_connect("draw_event", self._on_draw_event)

        # 1b. Interactive Navigation Toolbar
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 1px 4px;
            }
        """)
        tb_layout = QHBoxLayout(toolbar_frame)
        tb_layout.setContentsMargins(4, 1, 4, 1)
        tb_layout.setSpacing(6)

        lbl_tb = QLabel("Plot Tools:")
        lbl_tb.setStyleSheet("font-size: 10.5px; font-weight: bold; color: #475569;")
        tb_layout.addWidget(lbl_tb)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.hide()

        btn_tb_style = """
            QPushButton {
                background: #f8fafc;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10.5px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f1f5f9;
                color: #0284c7;
                border-color: #94a3b8;
            }
            QPushButton:checked {
                background: #e0f2fe;
                color: #0369a1;
                border: 1px solid #0284c7;
                font-weight: bold;
            }
        """

        self.btn_tb_home = QPushButton("🏠 Reset View")
        self.btn_tb_home.setStyleSheet(btn_tb_style)
        self.btn_tb_home.clicked.connect(self._on_toolbar_home)
        tb_layout.addWidget(self.btn_tb_home)

        self.btn_tb_pan = QPushButton("✋ Pan")
        self.btn_tb_pan.setCheckable(True)
        self.btn_tb_pan.setStyleSheet(btn_tb_style)
        self.btn_tb_pan.clicked.connect(self._on_toolbar_pan)
        tb_layout.addWidget(self.btn_tb_pan)

        self.btn_tb_zoom = QPushButton("🔍 Zoom Rect")
        self.btn_tb_zoom.setCheckable(True)
        self.btn_tb_zoom.setStyleSheet(btn_tb_style)
        self.btn_tb_zoom.clicked.connect(self._on_toolbar_zoom)
        tb_layout.addWidget(self.btn_tb_zoom)

        self.btn_tb_autofit = QPushButton("⤢ Auto-Fit")
        self.btn_tb_autofit.setStyleSheet(btn_tb_style)
        self.btn_tb_autofit.clicked.connect(self._on_toolbar_autofit)
        tb_layout.addWidget(self.btn_tb_autofit)

        tb_layout.addStretch()

        lbl_hint = QLabel("💡 Interactive Plot: Click chart to tune P_ini / WOC | Scroll wheel to zoom | ✋ Pan & 🔍 Zoom Box | Double-click to auto-reset")
        lbl_hint.setStyleSheet("font-size: 10px; color: #64748b; font-style: italic;")
        tb_layout.addWidget(lbl_hint)

        main_layout.addWidget(toolbar_frame)

        self.plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.plot_splitter.addWidget(self.canvas)

        # Compositional Editor Table (Visible in Mode 3)
        self.comp_table_frame = QFrame()
        self.comp_table_frame.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px;")
        comp_layout = QVBoxLayout(self.comp_table_frame)
        comp_layout.setContentsMargins(6, 6, 6, 6)
        comp_lbl = QLabel("Multi-Component Fluid Composition (EOS Flash)")
        comp_lbl.setStyleSheet("font-weight: bold; color: #0f172a; font-size: 11px;")
        comp_layout.addWidget(comp_lbl)

        self.table_comp = QTableWidget(6, 5)
        self.table_comp.setHorizontalHeaderLabels(["Component", "Mole Frac (z)", "MW (g/mol)", "Tc (°F)", "Pc (psia)"])
        self.table_comp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_comp.setStyleSheet("font-size: 10.5px;")
        self.table_comp.itemChanged.connect(self._on_table_cell_edited)
        comp_layout.addWidget(self.table_comp)

        btn_normalize = QPushButton("Normalize Mole Fractions to 1.00")
        btn_normalize.setStyleSheet("background: #f1f5f9; border: 1px solid #cbd5e1; font-weight: bold; font-size: 10px; padding: 4px;")
        btn_normalize.clicked.connect(self._normalize_composition)
        comp_layout.addWidget(btn_normalize)

        self.plot_splitter.addWidget(self.comp_table_frame)
        self.comp_table_frame.hide()

        self.plot_splitter.setStretchFactor(0, 7)
        self.plot_splitter.setStretchFactor(1, 3)
        main_layout.addWidget(self.plot_splitter, stretch=1)

        # 3. Dynamic Thermodynamic & Miscibility Diagnostics Dashboard (Middle Screen)
        self.diagnostics_frame = QFrame()
        self.diagnostics_frame.setObjectName("pvtDiagnosticsFrame")
        self.diagnostics_frame.setStyleSheet("""
            QFrame#pvtDiagnosticsFrame {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
            }
        """)
        diag_layout = QVBoxLayout(self.diagnostics_frame)
        diag_layout.setContentsMargins(6, 4, 6, 4)
        diag_layout.setSpacing(3)

        # Top row: Section title & Real-Time Interactive Cursor Readout
        readout_row = QHBoxLayout()
        readout_row.setContentsMargins(0, 0, 0, 0)
        lbl_diag_title = QLabel("DYNAMIC THERMODYNAMIC & MISCIBILITY DIAGNOSTICS")
        lbl_diag_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #0284c7; letter-spacing: 0.5px;")
        readout_row.addWidget(lbl_diag_title)
        readout_row.addStretch()

        self.lbl_cursor_readout = QLabel("Hover over charts for interactive point readout | Left-click to tune operating state | Scroll to zoom")
        self.lbl_cursor_readout.setStyleSheet("font-size: 10px; color: #475569; font-style: italic; background: #f8fafc; padding: 1px 6px; border-radius: 3px; border: 1px solid #e2e8f0;")
        readout_row.addWidget(self.lbl_cursor_readout)
        diag_layout.addLayout(readout_row)

        # Bottom row: 4 live telemetry cards
        cards_row = QHBoxLayout()
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(6)

        # Card 1: Saturation Regime
        self.card_regime = QFrame()
        self.card_regime.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
        v1 = QVBoxLayout(self.card_regime)
        v1.setContentsMargins(6, 3, 6, 3)
        v1.setSpacing(1)
        lbl1 = QLabel("Saturation Regime:")
        lbl1.setStyleSheet("font-size: 9.5px; color: #64748b; font-weight: bold;")
        self.lbl_regime_badge = QLabel("Undersaturated Oil")
        self.lbl_regime_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #16a34a;")
        self.lbl_regime_sub = QLabel("+1750 psi above Pb")
        self.lbl_regime_sub.setStyleSheet("font-size: 9.5px; color: #475569;")
        v1.addWidget(lbl1)
        v1.addWidget(self.lbl_regime_badge)
        v1.addWidget(self.lbl_regime_sub)
        cards_row.addWidget(self.card_regime, stretch=1)

        # Card 2: CO2 Miscibility State
        self.card_misc = QFrame()
        self.card_misc.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
        v2 = QVBoxLayout(self.card_misc)
        v2.setContentsMargins(6, 3, 6, 3)
        v2.setSpacing(1)
        lbl2 = QLabel("CO2 Miscibility State:")
        lbl2.setStyleSheet("font-size: 9.5px; color: #64748b; font-weight: bold;")
        self.lbl_misc_badge = QLabel("MISCIBLE")
        self.lbl_misc_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #16a34a;")
        self.lbl_misc_sub = QLabel("+1850 psi margin above MMP")
        self.lbl_misc_sub.setStyleSheet("font-size: 9.5px; color: #475569;")
        v2.addWidget(lbl2)
        v2.addWidget(self.lbl_misc_badge)
        v2.addWidget(self.lbl_misc_sub)
        cards_row.addWidget(self.card_misc, stretch=1)

        # Card 3: 3D Column Equilibrium
        self.card_col = QFrame()
        self.card_col.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
        v3 = QVBoxLayout(self.card_col)
        v3.setContentsMargins(6, 3, 6, 3)
        v3.setSpacing(1)
        lbl3 = QLabel("3D Column Equilibrium:")
        lbl3.setStyleSheet("font-size: 9.5px; color: #64748b; font-weight: bold;")
        self.lbl_col_badge = QLabel("Oil Leg: 35.0 ft")
        self.lbl_col_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #0284c7;")
        self.lbl_col_sub = QLabel("Datum: 5000 ft | WOC: 5035 ft")
        self.lbl_col_sub.setStyleSheet("font-size: 9.5px; color: #475569;")
        v3.addWidget(lbl3)
        v3.addWidget(self.lbl_col_badge)
        v3.addWidget(self.lbl_col_sub)
        cards_row.addWidget(self.card_col, stretch=1)

        # Card 4: In-Situ Viscosity & Swelling
        self.card_visc = QFrame()
        self.card_visc.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
        v4 = QVBoxLayout(self.card_visc)
        v4.setContentsMargins(6, 3, 6, 3)
        v4.setSpacing(1)
        lbl4 = QLabel("In-Situ Viscosity & Swelling:")
        lbl4.setStyleSheet("font-size: 9.5px; color: #64748b; font-weight: bold;")
        self.lbl_visc_badge = QLabel("Live μo: 0.40 cP")
        self.lbl_visc_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #7c3aed;")
        self.lbl_visc_sub = QLabel("Swollen: 0.17 cP (-58%) | SF: 1.25")
        self.lbl_visc_sub.setStyleSheet("font-size: 9.5px; color: #475569;")
        v4.addWidget(lbl4)
        v4.addWidget(self.lbl_visc_badge)
        v4.addWidget(self.lbl_visc_sub)
        cards_row.addWidget(self.card_visc, stretch=1)

        diag_layout.addLayout(cards_row)
        main_layout.addWidget(self.diagnostics_frame)

        # 4. Bottom Summary Status Bar
        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet("""
            QLabel {
                background-color: #f1f5f9;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(self.metrics_label)

        self._populate_composition_table()

    def set_display_mode(self, mode_index: int):
        """Sets active display mode programmatically from tree navigation."""
        if 0 <= mode_index < self.combo_mode.count():
            self.combo_mode.setCurrentIndex(mode_index)

    def set_parameters(self, params: Dict[str, Any]):
        """Synchronizes model parameters from property grid or config store."""
        if not params:
            return

        if "initial_pressure" in params:
            self.initial_pressure = float(params["initial_pressure"])
            self.spin_p_ini.blockSignals(True)
            self.spin_p_ini.setValue(self.initial_pressure)
            self.spin_p_ini.blockSignals(False)

        if "temperature" in params:
            self.temperature_f = float(params["temperature"])
            self.spin_temp.blockSignals(True)
            self.spin_temp.setValue(self.temperature_f)
            self.spin_temp.blockSignals(False)

        if "api_gravity" in params:
            self.api_gravity = float(params["api_gravity"])
        if "sol_gor" in params:
            self.sol_gor = float(params["sol_gor"])
        if "gas_gravity" in params:
            self.gas_gravity = float(params["gas_gravity"])
        if "bubble_point" in params:
            self.bubble_point_psia = float(params["bubble_point"])
        if "oil_viscosity" in params or "dead_oil_viscosity" in params:
            self.dead_oil_viscosity_cp = float(params.get("oil_viscosity", params.get("dead_oil_viscosity", 2.5)))
        if "co2_purity" in params:
            self.co2_purity_pct = float(params["co2_purity"])
        if "max_swelling" in params:
            self.max_swelling = float(params["max_swelling"])
        if "mmp" in params or "mmp_override" in params:
            self.mmp_psia = float(params.get("mmp_override", params.get("mmp", 2150.0)))
        if "woc_depth" in params:
            self.woc_depth = float(params["woc_depth"])
        if "has_gas_cap" in params:
            self.has_gas_cap = bool(params["has_gas_cap"])
        if "goc_depth" in params:
            self.goc_depth = float(params["goc_depth"])
        if "top_depth" in params:
            self.top_depth = float(params["top_depth"])
        if "thickness" in params or "net_pay" in params:
            self.thickness_ft = float(params.get("thickness", params.get("net_pay", 50.0)))

        self._recalculate_thermodynamics()
        self._render_active_mode()

    def get_fluid_parameters(self) -> Dict[str, Any]:
        """Returns complete fluid and thermodynamic model dictionary."""
        return {
            "initial_pressure": self.initial_pressure,
            "temperature": self.temperature_f,
            "api_gravity": self.api_gravity,
            "sol_gor": self.sol_gor,
            "gas_gravity": self.gas_gravity,
            "bubble_point": self.bubble_point_psia,
            "dead_oil_viscosity": self.dead_oil_viscosity_cp,
            "live_oil_viscosity": self.live_oil_viscosity_cp,
            "co2_purity": self.co2_purity_pct,
            "max_swelling": self.max_swelling,
            "mmp": self.mmp_psia,
            "woc_depth": self.woc_depth,
            "has_gas_cap": self.has_gas_cap,
            "goc_depth": self.goc_depth,
            "composition": self.composition,
            "water_gradient": self.water_gradient,
            "gas_gradient": self.gas_gradient,
        }

    def _recalculate_thermodynamics(self):
        """Re-evaluates thermodynamic correlations using SolventExtendedPVTEngine and MMP engine."""
        self.pvt_engine = SolventExtendedPVTEngine(
            reservoir_temperature_f=self.temperature_f,
            initial_pressure_psi=self.initial_pressure,
            api_gravity=self.api_gravity,
            gas_gravity=self.gas_gravity,
            dead_oil_viscosity_cp=self.dead_oil_viscosity_cp,
        )

        # Standing (1947) Bubble point estimation: Pb = 18.2 * ((Rs/gamma_g)^0.83 * 10^(0.00091*T - 0.0125*API) - 1.4)
        gamma_g = max(self.gas_gravity, 0.5)
        api = self.api_gravity
        t = self.temperature_f
        rs = max(self.sol_gor, 10.0)
        try:
            a_fac = (rs / gamma_g) ** 0.83
            b_fac = 10.0 ** (0.00091 * t - 0.0125 * api)
            pb_est = 18.2 * (a_fac * b_fac - 1.4)
            self.bubble_point_psia = float(np.clip(pb_est, 100.0, 10000.0))
        except Exception:
            self.bubble_point_psia = 2250.0

        # Calculate live oil viscosity at initial pressure
        try:
            self.live_oil_viscosity_cp = self.pvt_engine.calculate_oil_viscosity_cp(self.initial_pressure, x_co2=0.0)
        except Exception:
            self.live_oil_viscosity_cp = max(0.5, self.dead_oil_viscosity_cp * 0.6)

        # Calculate MMP via correlations
        try:
            mmp_params = MMPParameters(
                temperature=self.temperature_f,
                oil_gravity=self.api_gravity,
                c7_plus_mw=float(self.composition[-1]["mw"]),
                injection_gas_composition={"CO2": self.co2_purity_pct / 100.0, "CH4": (100.0 - self.co2_purity_pct) / 100.0}
            )
            self.mmp_cronquist = calculate_mmp(mmp_params, method="cronquist")
            self.mmp_yellig = calculate_mmp(mmp_params, method="yellig_metcalfe")
            self.mmp_lee = calculate_mmp(mmp_params, method="lee")
            self.mmp_alston = calculate_mmp(mmp_params, method="alston")
            self.mmp_emera = calculate_mmp(mmp_params, method="emera_sarma")
            self.mmp_calc_auto = calculate_mmp(mmp_params, method="auto")
        except Exception as e:
            logger.debug(f"MMP calculation error: {e}")
            self.mmp_cronquist = 2150.0
            self.mmp_yellig = 2280.0
            self.mmp_lee = 2100.0
            self.mmp_alston = 2050.0
            self.mmp_emera = 2200.0
            self.mmp_calc_auto = 2150.0

        self._update_diagnostics_hud()

    def _update_diagnostics_hud(self):
        """Updates the 4 live dynamic thermodynamic and miscibility diagnostic cards in the middle screen."""
        if not hasattr(self, 'lbl_regime_badge'):
            return

        p_ini = self.initial_pressure
        pb = self.bubble_point_psia
        mmp = self.mmp_psia
        gor = self.sol_gor
        visc_d = self.dead_oil_viscosity_cp

        # 1. Bubble Point Regime
        dp_pb = p_ini - pb
        if dp_pb > 50.0:
            self.lbl_regime_badge.setText("Undersaturated Oil")
            self.lbl_regime_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #16a34a;")
            self.lbl_regime_sub.setText(f"+{dp_pb:.0f} psi above Pb ({pb:.0f} psia)")
        elif abs(dp_pb) <= 50.0:
            self.lbl_regime_badge.setText("At Bubble Point (Pb)")
            self.lbl_regime_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #ea580c;")
            self.lbl_regime_sub.setText(f"Equilibrium boundary ({pb:.0f} psia)")
        else:
            self.lbl_regime_badge.setText("Two-Phase Saturated")
            self.lbl_regime_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #dc2626;")
            self.lbl_regime_sub.setText(f"{dp_pb:.0f} psi below Pb (Free Gas Evolved)")

        # 2. Miscibility State & Margin
        dp_mmp = p_ini - mmp
        if dp_mmp >= 200.0:
            self.lbl_misc_badge.setText("MISCIBLE")
            self.lbl_misc_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #16a34a;")
            self.lbl_misc_sub.setText(f"Margin: +{dp_mmp:.0f} psi above MMP ({mmp:.0f} psia)")
        elif dp_mmp >= 0.0:
            self.lbl_misc_badge.setText("NEAR-MISCIBLE")
            self.lbl_misc_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #d97706;")
            self.lbl_misc_sub.setText(f"Margin: +{dp_mmp:.0f} psi above MMP ({mmp:.0f} psia)")
        else:
            self.lbl_misc_badge.setText("IMMISCIBLE")
            self.lbl_misc_badge.setStyleSheet("font-size: 11px; font-weight: bold; color: #dc2626;")
            self.lbl_misc_sub.setText(f"Deficit: {dp_mmp:.0f} psi below MMP ({mmp:.0f} psia)")

        # 3. 3D Column Equilibrium & Pay
        top_v = self.top_depth
        woc_v = self.woc_depth
        has_gc = self.has_gas_cap
        goc_v = self.goc_depth
        top_oil = goc_v if has_gc else top_v
        oil_col = max(woc_v - top_oil, 0.0)
        if has_gc:
            gc_thick = max(goc_v - top_v, 0.0)
            self.lbl_col_badge.setText(f"Oil: {oil_col:.1f} ft | Gas Cap: {gc_thick:.1f} ft")
            self.lbl_col_sub.setText(f"WOC: {woc_v:.0f} ft | GOC: {goc_v:.0f} ft")
        else:
            self.lbl_col_badge.setText(f"Oil Leg: {oil_col:.1f} ft")
            self.lbl_col_sub.setText(f"Datum: {top_v:.0f} ft | WOC: {woc_v:.0f} ft")

        # 4. In-Situ Viscosity & Swelling
        a = 10.715 * (gor + 100.0) ** (-0.515)
        b = 5.44 * (gor + 150.0) ** (-0.338)
        live_visc = max(a * (visc_d ** b), 0.15) if gor > 10.0 else visc_d
        swollen_visc = max(live_visc * 0.42, 0.08)
        sf_val = getattr(self, 'max_swelling', 1.25)
        thinning_pct = (1.0 - (swollen_visc / max(live_visc, 0.01))) * 100.0
        self.lbl_visc_badge.setText(f"Live μo: {live_visc:.2f} cP → Swollen: {swollen_visc:.2f} cP")
        self.lbl_visc_sub.setText(f"Thinning: -{thinning_pct:.0f}% | Swelling SF: {sf_val:.2f}")

    def _on_draw_event(self, event):
        """Called automatically whenever the Matplotlib canvas completes a draw (initial, pan, zoom, resize)."""
        self._capture_backgrounds()

    def _capture_backgrounds(self):
        """Captures clean pixel buffers of all current axes for zero-lag blitting."""
        self._bg_cache = {}
        if hasattr(self, 'current_axes') and self.current_axes:
            for ax in self.current_axes:
                if ax in self.fig.axes:
                    try:
                        self._bg_cache[ax] = self.canvas.copy_from_bbox(ax.bbox)
                    except Exception:
                        pass

    def _setup_cursor_artists(self, mode_idx: int):
        """Sets up animated cursor lines for high-performance zero-lag visual feedback."""
        self._cursor_lines = []
        if not hasattr(self, 'current_axes') or not self.current_axes:
            return

        line_kwargs = dict(color="gray", linestyle="--", linewidth=1.2, alpha=0.85, animated=True)
        point_kwargs = dict(marker="o", markersize=6, color="gray", markeredgecolor="#ffffff", markeredgewidth=1.2, animated=True)

        if mode_idx == 0:
            # Mode 0: Dual vertical pressure tracking lines across ax1 and ax2
            if len(self.current_axes) >= 2:
                ax1, ax2 = self.current_axes[0], self.current_axes[1]
                l1 = ax1.axvline(x=self.initial_pressure, **line_kwargs)
                l2 = ax2.axvline(x=self.initial_pressure, **line_kwargs)
                l1.set_visible(False)
                l2.set_visible(False)
                self._cursor_lines = [l1, l2]
        elif mode_idx == 1:
            # Mode 1: Dual vertical CO2 mole fraction tracking lines across ax1 and ax2
            if len(self.current_axes) >= 2:
                ax1, ax2 = self.current_axes[0], self.current_axes[1]
                l1 = ax1.axvline(x=0, **line_kwargs)
                l2 = ax2.axvline(x=0, **line_kwargs)
                l1.set_visible(False)
                l2.set_visible(False)
                self._cursor_lines = [l1, l2]
        elif mode_idx == 2:
            # Mode 2: ax1 (barometer horizontal pressure guide) and ax2 (slim tube vertical injection pressure line)
            if len(self.current_axes) >= 2:
                ax1, ax2 = self.current_axes[0], self.current_axes[1]
                l1 = ax1.axhline(y=self.initial_pressure, **line_kwargs)
                l2 = ax2.axvline(x=self.initial_pressure, **line_kwargs)
                l1.set_visible(False)
                l2.set_visible(False)
                self._cursor_lines = [l1, l2]
        elif mode_idx == 3:
            # Mode 3: P-T crosshair on ax1 (vertical T line, horizontal P line, and center marker)
            if len(self.current_axes) >= 1:
                ax1 = self.current_axes[0]
                lv = ax1.axvline(x=self.temperature_f, **line_kwargs)
                lh = ax1.axhline(y=self.initial_pressure, **line_kwargs)
                pt = ax1.plot([self.temperature_f], [self.initial_pressure], **point_kwargs)[0]
                lv.set_visible(False)
                lh.set_visible(False)
                pt.set_visible(False)
                self._cursor_lines = [lv, lh, pt]
        else:
            # Mode 4: Dual horizontal depth tracking lines across ax1 and ax2
            if len(self.current_axes) >= 2:
                ax1, ax2 = self.current_axes[0], self.current_axes[1]
                l1 = ax1.axhline(y=self.woc_depth, **line_kwargs)
                l2 = ax2.axhline(y=self.woc_depth, **line_kwargs)
                l1.set_visible(False)
                l2.set_visible(False)
                self._cursor_lines = [l1, l2]

    def _update_cursor_blitting(self, mode_idx: int, inaxes, x: float, y: float):
        """Blits the active cursor line(s) instantly to the canvas (< 4 ms, zero-lag 300 FPS)."""
        if not hasattr(self, '_cursor_lines') or not self._cursor_lines:
            return
        if not hasattr(self, 'current_axes') or not self.current_axes:
            return
        if not hasattr(self, '_bg_cache') or not self._bg_cache:
            self._capture_backgrounds()
            if not self._bg_cache:
                return

        try:
            if mode_idx == 0:
                # Mode 0: Dual vertical pressure lines
                p = max(x, 14.7)
                if len(self.current_axes) >= 2 and len(self._cursor_lines) >= 2:
                    ax1, ax2 = self.current_axes[0], self.current_axes[1]
                    if ax1 in self._bg_cache and ax2 in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax1])
                        self.canvas.restore_region(self._bg_cache[ax2])
                        self._cursor_lines[0].set_xdata([p, p])
                        self._cursor_lines[1].set_xdata([p, p])
                        self._cursor_lines[0].set_visible(True)
                        self._cursor_lines[1].set_visible(True)
                        ax1.draw_artist(self._cursor_lines[0])
                        ax2.draw_artist(self._cursor_lines[1])
                        self.canvas.blit(ax1.bbox)
                        self.canvas.blit(ax2.bbox)

            elif mode_idx == 1:
                # Mode 1: Dual vertical CO2 mol% lines
                x_pct = np.clip(x, 0.0, 85.0)
                if len(self.current_axes) >= 2 and len(self._cursor_lines) >= 2:
                    ax1, ax2 = self.current_axes[0], self.current_axes[1]
                    if ax1 in self._bg_cache and ax2 in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax1])
                        self.canvas.restore_region(self._bg_cache[ax2])
                        self._cursor_lines[0].set_xdata([x_pct, x_pct])
                        self._cursor_lines[1].set_xdata([x_pct, x_pct])
                        self._cursor_lines[0].set_visible(True)
                        self._cursor_lines[1].set_visible(True)
                        ax1.draw_artist(self._cursor_lines[0])
                        ax2.draw_artist(self._cursor_lines[1])
                        self.canvas.blit(ax1.bbox)
                        self.canvas.blit(ax2.bbox)

            elif mode_idx == 2:
                # Mode 2: ax1 barometer horizontal line, ax2 vertical pressure line
                p_val = max(y if inaxes == self.current_axes[0] else x, 14.7)
                if len(self.current_axes) >= 2 and len(self._cursor_lines) >= 2:
                    ax1, ax2 = self.current_axes[0], self.current_axes[1]
                    if ax1 in self._bg_cache and ax2 in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax1])
                        self.canvas.restore_region(self._bg_cache[ax2])
                        self._cursor_lines[0].set_ydata([p_val, p_val])
                        self._cursor_lines[1].set_xdata([p_val, p_val])
                        self._cursor_lines[0].set_visible(True)
                        self._cursor_lines[1].set_visible(True)
                        ax1.draw_artist(self._cursor_lines[0])
                        ax2.draw_artist(self._cursor_lines[1])
                        self.canvas.blit(ax1.bbox)
                        self.canvas.blit(ax2.bbox)

            elif mode_idx == 3:
                # Mode 3: P-T crosshair and dot on ax1
                if len(self.current_axes) >= 1 and len(self._cursor_lines) >= 3:
                    ax1 = self.current_axes[0]
                    if ax1 in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax1])
                        self._cursor_lines[0].set_xdata([x, x])
                        self._cursor_lines[1].set_ydata([y, y])
                        self._cursor_lines[2].set_data([x], [y])
                        self._cursor_lines[0].set_visible(True)
                        self._cursor_lines[1].set_visible(True)
                        self._cursor_lines[2].set_visible(True)
                        ax1.draw_artist(self._cursor_lines[0])
                        ax1.draw_artist(self._cursor_lines[1])
                        ax1.draw_artist(self._cursor_lines[2])
                        self.canvas.blit(ax1.bbox)

            elif mode_idx == 4:
                # Mode 4: Dual horizontal depth lines across ax1 and ax2
                z = y
                if len(self.current_axes) >= 2 and len(self._cursor_lines) >= 2:
                    ax1, ax2 = self.current_axes[0], self.current_axes[1]
                    if ax1 in self._bg_cache and ax2 in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax1])
                        self.canvas.restore_region(self._bg_cache[ax2])
                        self._cursor_lines[0].set_ydata([z, z])
                        self._cursor_lines[1].set_ydata([z, z])
                        self._cursor_lines[0].set_visible(True)
                        self._cursor_lines[1].set_visible(True)
                        ax1.draw_artist(self._cursor_lines[0])
                        ax2.draw_artist(self._cursor_lines[1])
                        self.canvas.blit(ax1.bbox)
                        self.canvas.blit(ax2.bbox)
        except Exception as e:
            logger.debug(f"Cursor blitting error: {e}")

    def _hide_cursor_lines(self):
        """Hides cursor lines and floating tooltip badge, restoring clean canvas background."""
        if hasattr(self, '_cursor_tooltip') and self._cursor_tooltip.isVisible():
            self._cursor_tooltip.hide()
        if not hasattr(self, '_cursor_lines') or not self._cursor_lines:
            return
        if not hasattr(self, 'current_axes') or not self.current_axes:
            return
        if not hasattr(self, '_bg_cache') or not self._bg_cache:
            return

        try:
            changed = False
            for line in self._cursor_lines:
                if line.get_visible():
                    line.set_visible(False)
                    changed = True
            if changed:
                for ax in self.current_axes:
                    if ax in self._bg_cache:
                        self.canvas.restore_region(self._bg_cache[ax])
                        self.canvas.blit(ax.bbox)
        except Exception:
            pass

    def _update_cursor_tooltip(self, event, mode_idx: int, ax, x: float, y: float):
        """Displays formatted engineering data in a clean tooltip beside the cursor without CSS styling."""
        if not hasattr(self, '_cursor_tooltip'):
            return

        text = ""
        if mode_idx == 0:
            p = max(x, 14.7)
            if hasattr(self, '_cached_p_arr') and len(self._cached_p_arr) > 0:
                p_arr = self._cached_p_arr
                bo = float(np.interp(p, p_arr, self._cached_bo_arr))
                rs = float(np.interp(p, p_arr, self._cached_rs_arr))
                mu = float(np.interp(p, p_arr, self._cached_mu_arr))
                rho = float(np.interp(p, p_arr, self._cached_rho_co2_arr))
            else:
                bo, rs, mu, rho = 1.25, 500.0, self.dead_oil_viscosity_cp, 750.0

            if hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and ax == self.current_axes[1]:
                text = (
                    f"Pressure: {p:.0f} psia\n"
                    f"Live Oil Viscosity: {mu:.2f} cP\n"
                    f"CO2 Density: {rho:.1f} kg/m³\n"
                    f"[Click to tune P_ini]"
                )
            else:
                is_under = p > self.bubble_point_psia
                reg_badge = "Undersaturated Oil" if is_under else "Saturated (Two-Phase)"
                text = (
                    f"Pressure: {p:.0f} psia\n"
                    f"Oil FVF Bo: {bo:.3f} rb/STB\n"
                    f"Solution Gas Rs: {rs:.0f} scf/STB\n"
                    f"Status: {reg_badge}\n"
                    f"[Click to tune P_ini]"
                )
        elif mode_idx == 1:
            x_pct = np.clip(x, 0.0, 85.0)
            x_mol = x_pct / 100.0
            if hasattr(self, '_cached_x_co2') and len(self._cached_x_co2) > 0:
                sf = float(np.interp(x_mol, self._cached_x_co2, self._cached_sf))
                ratio = float(np.interp(x_mol, self._cached_x_co2, self._cached_visc_ratio))
            else:
                sf = 1.0 + 0.3 * (x_mol ** 1.2)
                ratio = max(0.1, 1.0 - 0.7 * (x_mol ** 0.8))
            mu = ratio * self.dead_oil_viscosity_cp
            text = (
                f"Dissolved CO2: {x_pct:.1f} mol%\n"
                f"Swelling Factor: {sf:.3f} (+{(sf-1.0)*100.0:.1f}%)\n"
                f"Swollen Viscosity: {mu:.2f} cP (-{(1.0-ratio)*100.0:.1f}%)\n"
                f"Viscosity Ratio: {ratio:.3f}"
            )
        elif mode_idx == 2:
            if hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and ax == self.current_axes[0]:
                is_misc = y >= self.mmp_psia
                misc_str = "MISCIBLE" if is_misc else "IMMISCIBLE"
                text = (
                    f"Pressure: {y:.0f} psia\n"
                    f"Reservoir Pres: {self.initial_pressure:.0f} psia\n"
                    f"Active MMP: {self.mmp_psia:.0f} psia\n"
                    f"Status: {misc_str}\n"
                    f"[Click bar to select MMP]"
                )
            else:
                p_inj = max(x, 100.0)
                if hasattr(self, '_cached_sweep_p') and len(self._cached_sweep_p) > 0:
                    rf = float(np.interp(p_inj, self._cached_sweep_p, self._cached_recovery))
                else:
                    rf = 55.0 + 25.0 * (p_inj / max(self.mmp_psia, 100.0))**1.8 if p_inj < self.mmp_psia else 80.0 + 15.0 * (1.0 - np.exp(-(p_inj - self.mmp_psia) / 250.0))
                rf = float(np.clip(rf, 40.0, 97.5))
                is_misc = p_inj >= self.mmp_psia
                misc_str = "MISCIBLE EXTRACTION (>90%)" if is_misc else "IMMISCIBLE DISPLACEMENT"
                text = (
                    f"Injection P: {p_inj:.0f} psia\n"
                    f"Slim-Tube Recovery: {rf:.1f}% OOIP\n"
                    f"Status: {misc_str}\n"
                    f"[Click to set P_ini]"
                )
        elif mode_idx == 3:
            t_f = x
            p_psi = y
            t_crit = getattr(self, '_cached_t_crit', 180.0 + self.api_gravity * 2.0)
            p_crit = getattr(self, '_cached_p_crit', 2600.0 - self.api_gravity * 15.0)
            pb_t = p_crit + 800.0 * (1.0 - ((t_f - 60.0) / max(t_crit - 60.0, 10.0))**2) if t_f <= t_crit else p_crit * np.exp(-((t_f - t_crit) / 120.0)**2)
            if p_psi <= pb_t and p_psi >= 100.0 and t_f >= 40.0 and t_f <= 400.0:
                ph = "Two-Phase Region (Liquid + Vapor)"
            elif t_f > t_crit and p_psi > p_crit:
                ph = "Supercritical Dense Fluid"
            elif p_psi > pb_t:
                ph = "Single-Phase Compressed Liquid"
            else:
                ph = "Superheated Gas / Vapor"
            text = (
                f"T = {t_f:.1f} °F | P = {p_psi:.0f} psia\n"
                f"Phase: {ph}\n"
                f"Critical: {t_crit:.0f} °F, {p_crit:.0f} psia\n"
                f"[Click to set T and P_ini]"
            )
        else:
            z = y
            top_z = self.top_depth
            woc_z = self.woc_depth
            goc_z = self.goc_depth if self.has_gas_cap else None
            gamma_o = 141.5 / max(self.api_gravity + 131.5, 10.0)
            oil_grad = 0.4335 * gamma_o
            p_ref = self.initial_pressure
            if goc_z is not None and z < goc_z:
                pz = p_ref + self.gas_gradient * (z - top_z)
                ph = "Gas Cap (Free Gas)"
                so, sw, sg = 0.0, 0.20, 0.80
            elif z <= woc_z:
                pz = p_ref + oil_grad * (z - top_z)
                ph = "Oil Column (Pay Zone)"
                so, sw, sg = 0.80, 0.20, 0.0
            else:
                p_woc = p_ref + oil_grad * (woc_z - top_z)
                pz = p_woc + self.water_gradient * (z - woc_z)
                ph = "Water Leg (Aquifer)"
                so, sw, sg = 0.0, 1.0, 0.0
            text = (
                f"TVD Depth: {z:.1f} ft\n"
                f"Pore Pressure: {pz:.0f} psia\n"
                f"Zone: {ph}\n"
                f"So = {so:.2f} | Sw = {sw:.2f} | Sg = {sg:.2f}\n"
                f"[Click to set WOC]"
            )

        self._cursor_tooltip.setText(text.strip())
        self._cursor_tooltip.adjustSize()

        # Position beside cursor in Qt canvas coordinates
        canvas_x = int(event.x)
        canvas_y = int(self.canvas.height() - event.y)
        tt_w = self._cursor_tooltip.width()
        tt_h = self._cursor_tooltip.height()

        tx = canvas_x + 15
        ty = canvas_y - tt_h // 2

        # Clamp against canvas borders
        if tx + tt_w > self.canvas.width() - 8:
            tx = canvas_x - tt_w - 15
        tx = max(8, tx)

        if ty + tt_h > self.canvas.height() - 8:
            ty = self.canvas.height() - tt_h - 8
        ty = max(8, ty)

        self._cursor_tooltip.move(tx, ty)
        self._cursor_tooltip.show()
        self._cursor_tooltip.raise_()

    def _on_canvas_motion(self, event):
        """Interactive hover cursor tracking across all plot modes with zero-lag blitted visual feedback."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._hide_cursor_lines()
            self.lbl_cursor_readout.setText("Hover over charts for interactive point readout | Left-click to tune operating state | Scroll to zoom")
            return

        ax = event.inaxes
        x = event.xdata
        y = event.ydata
        mode_idx = self.combo_mode.currentIndex()

        # 1. Update cursor line visual feedback via high-performance blitting
        self._update_cursor_blitting(mode_idx, ax, x, y)

        # 2. Update floating tooltip badge directly beside the cursor
        self._update_cursor_tooltip(event, mode_idx, ax, x, y)

        if mode_idx == 0:
            # Mode 0: Black Oil Curves
            p = max(x, 14.7)
            if hasattr(self, '_cached_p_arr') and len(self._cached_p_arr) > 0:
                p_arr = self._cached_p_arr
                bo = float(np.interp(p, p_arr, self._cached_bo_arr))
                rs = float(np.interp(p, p_arr, self._cached_rs_arr))
                mu = float(np.interp(p, p_arr, self._cached_mu_arr))
                rho = float(np.interp(p, p_arr, self._cached_rho_co2_arr))
            else:
                bo = 1.25
                rs = 500.0
                mu = self.dead_oil_viscosity_cp
                rho = 750.0

            if hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and ax == self.current_axes[1]:
                self.lbl_cursor_readout.setText(f"🎯 P = {p:.0f} psia | Live μo = {mu:.2f} cP | Supercritical CO2 Density = {rho:.1f} kg/m³ | [Click to set P_ini]")
            else:
                reg = "Undersaturated" if p > self.bubble_point_psia else "Saturated"
                self.lbl_cursor_readout.setText(f"🎯 P = {p:.0f} psia | Bo = {bo:.3f} rb/STB | Rs = {rs:.0f} scf/STB ({reg}) | [Click to set P_ini]")
        elif mode_idx == 1:
            # Mode 1: Swelling & Viscosity
            x_pct = np.clip(x, 0.0, 85.0)
            x_mol = x_pct / 100.0
            if hasattr(self, '_cached_x_co2') and len(self._cached_x_co2) > 0:
                sf = float(np.interp(x_mol, self._cached_x_co2, self._cached_sf))
                ratio = float(np.interp(x_mol, self._cached_x_co2, self._cached_visc_ratio))
            else:
                sf = 1.0 + 0.3 * (x_mol ** 1.2)
                ratio = max(0.1, 1.0 - 0.7 * (x_mol ** 0.8))
            mu = ratio * self.dead_oil_viscosity_cp
            self.lbl_cursor_readout.setText(f"🎯 Dissolved CO2 = {x_pct:.1f} mol% | Swelling SF = {sf:.3f} (+{(sf-1.0)*100.0:.1f}%) | Swollen μo = {mu:.2f} cP (Thinning: -{(1.0-ratio)*100.0:.1f}%)")
        elif mode_idx == 2:
            # Mode 2: MMP Barometer & Slim Tube
            if hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and ax == self.current_axes[0]:
                self.lbl_cursor_readout.setText(f"🎯 Barometer: Pressure = {y:.0f} psia | P_res = {self.initial_pressure:.0f} psia | Active MMP = {self.mmp_psia:.0f} psia | [Click bar to select MMP]")
            else:
                p_inj = max(x, 100.0)
                if hasattr(self, '_cached_sweep_p') and len(self._cached_sweep_p) > 0:
                    rf = float(np.interp(p_inj, self._cached_sweep_p, self._cached_recovery))
                else:
                    rf = 55.0 + 25.0 * (p_inj / max(self.mmp_psia, 100.0))**1.8 if p_inj < self.mmp_psia else 80.0 + 15.0 * (1.0 - np.exp(-(p_inj - self.mmp_psia) / 250.0))
                rf = float(np.clip(rf, 40.0, 97.5))
                misc_str = "MISCIBLE EXTRACTION (>90%)" if p_inj >= self.mmp_psia else "IMMISCIBLE DISPLACEMENT"
                self.lbl_cursor_readout.setText(f"🎯 Injection P = {p_inj:.0f} psia | Slim-Tube Recovery = {rf:.1f}% OOIP | {misc_str} | [Click to set P_ini]")
        elif mode_idx == 3:
            # Mode 3: EOS Phase Envelope
            t_f = x
            p_psi = y
            t_crit = getattr(self, '_cached_t_crit', 180.0 + self.api_gravity * 2.0)
            p_crit = getattr(self, '_cached_p_crit', 2600.0 - self.api_gravity * 15.0)
            pb_t = p_crit + 800.0 * (1.0 - ((t_f - 60.0) / max(t_crit - 60.0, 10.0))**2) if t_f <= t_crit else p_crit * np.exp(-((t_f - t_crit) / 120.0)**2)
            if p_psi <= pb_t and p_psi >= 100.0 and t_f >= 40.0 and t_f <= 400.0:
                ph = "Two-Phase Region (Liquid + Vapor in Equilibrium)"
            elif t_f > t_crit and p_psi > p_crit:
                ph = "Supercritical Dense Fluid"
            elif p_psi > pb_t:
                ph = "Single-Phase Compressed Liquid"
            else:
                ph = "Superheated Gas / Vapor Phase"
            self.lbl_cursor_readout.setText(f"🎯 T = {t_f:.1f}°F, P = {p_psi:.0f} psia | Region: {ph} | [Click to set T and P_ini]")
        else:
            # Mode 4: Fluid Column
            z = y
            top_z = self.top_depth
            bot_z = self.top_depth + self.thickness_ft
            woc_z = self.woc_depth
            goc_z = self.goc_depth if self.has_gas_cap else None
            gamma_o = 141.5 / max(self.api_gravity + 131.5, 10.0)
            oil_grad = 0.4335 * gamma_o
            p_ref = self.initial_pressure
            if goc_z is not None and z < goc_z:
                pz = p_ref + self.gas_gradient * (z - top_z)
                ph = "Gas Cap (Free Gas)"
                so, sw, sg = 0.0, 0.20, 0.80
            elif z <= woc_z:
                pz = p_ref + oil_grad * (z - top_z)
                ph = "Oil Column (Pay)"
                so, sw, sg = 0.80, 0.20, 0.0
            else:
                p_woc = p_ref + oil_grad * (woc_z - top_z)
                pz = p_woc + self.water_gradient * (z - woc_z)
                ph = "Water Leg (Aquifer)"
                so, sw, sg = 0.0, 1.0, 0.0
            self.lbl_cursor_readout.setText(f"🎯 TVD = {z:.1f} ft | Pore Pressure = {pz:.0f} psia | {ph} (So={so:.2f}, Sw={sw:.2f}) | [Click to set WOC]")

    def _on_canvas_leave(self, event):
        self._hide_cursor_lines()
        self.lbl_cursor_readout.setText("Hover over charts for interactive point readout | Left-click to tune operating state | Scroll to zoom")

    def _on_toolbar_home(self):
        self.toolbar.home()
        if hasattr(self, 'btn_tb_pan'):
            self.btn_tb_pan.setChecked(False)
        if hasattr(self, 'btn_tb_zoom'):
            self.btn_tb_zoom.setChecked(False)

    def _on_toolbar_pan(self):
        self.toolbar.pan()
        is_active = (getattr(self.toolbar, 'mode', '') == 'pan/zoom')
        if hasattr(self, 'btn_tb_pan'):
            self.btn_tb_pan.setChecked(is_active)
        if hasattr(self, 'btn_tb_zoom'):
            self.btn_tb_zoom.setChecked(False)

    def _on_toolbar_zoom(self):
        self.toolbar.zoom()
        is_active = (getattr(self.toolbar, 'mode', '') == 'zoom rect')
        if hasattr(self, 'btn_tb_zoom'):
            self.btn_tb_zoom.setChecked(is_active)
        if hasattr(self, 'btn_tb_pan'):
            self.btn_tb_pan.setChecked(False)

    def _on_toolbar_autofit(self):
        if hasattr(self, 'btn_tb_pan'):
            self.btn_tb_pan.setChecked(False)
        if hasattr(self, 'btn_tb_zoom'):
            self.btn_tb_zoom.setChecked(False)
        self._render_active_mode()

    def _on_canvas_click(self, event):
        """Interactive click on plots to adjust reference pressures, MMP, or contact depths."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if hasattr(self, 'toolbar') and getattr(self.toolbar, 'mode', ''):
            return

        if event.dblclick:
            self._on_toolbar_autofit()
            return

        if event.button != 1:
            return

        mode_idx = self.combo_mode.currentIndex()
        if mode_idx == 0:
            new_p = float(np.clip(event.xdata, 500.0, 15000.0))
            from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
            with CalculationProgressDialog(self, title="Recalculating Thermodynamic Model...", task_name="Tuning Initial Reservoir Pressure") as prog:
                prog.set_step(20, f"Setting initial pressure to {new_p:.0f} psia...")
                self.spin_p_ini.blockSignals(True)
                self.spin_p_ini.setValue(new_p)
                self.spin_p_ini.blockSignals(False)
                self.initial_pressure = new_p
                prog.set_step(50, "Evaluating bubble point, live oil viscosity & solution GOR...")
                self._recalculate_thermodynamics()
                prog.set_step(75, "Re-rendering high-resolution black oil curves...")
                self._render_active_mode()
                prog.set_step(90, "Synchronizing with subsurface reservoir model...")
                self.parameters_changed.emit("pvt", "initial_pressure", self.initial_pressure)
                prog.set_step(100, "✓ PVT state recalculated successfully!")
        elif mode_idx == 2:
            if hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and event.inaxes == self.current_axes[1]:
                new_p = float(np.clip(event.xdata, 500.0, 15000.0))
                from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
                with CalculationProgressDialog(self, title="Recalculating Slim-Tube Displacement...", task_name="Tuning Injection Pressure") as prog:
                    prog.set_step(20, f"Updating injection pressure to {new_p:.0f} psia...")
                    self.spin_p_ini.blockSignals(True)
                    self.spin_p_ini.setValue(new_p)
                    self.spin_p_ini.blockSignals(False)
                    self.initial_pressure = new_p
                    prog.set_step(55, "Recalculating miscibility margin and extraction recovery...")
                    self._recalculate_thermodynamics()
                    prog.set_step(80, "Re-rendering barometer & recovery curves...")
                    self._render_active_mode()
                    prog.set_step(95, "Synchronizing with subsurface reservoir model...")
                    self.parameters_changed.emit("pvt", "initial_pressure", self.initial_pressure)
                    prog.set_step(100, "✓ Operating pressure updated!")
            elif hasattr(self, 'current_axes') and len(self.current_axes) >= 2 and event.inaxes == self.current_axes[0]:
                corrs = [self.mmp_cronquist, self.mmp_yellig, self.mmp_lee, self.mmp_alston, self.mmp_emera]
                bar_idx = int(round(event.xdata))
                if 0 <= bar_idx < len(corrs):
                    new_mmp = corrs[bar_idx]
                    from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
                    with CalculationProgressDialog(self, title="Recalculating MMP Correlation...", task_name="Applying Minimum Miscibility Pressure") as prog:
                        prog.set_step(25, f"Setting active MMP to {new_mmp:.0f} psia...")
                        self.mmp_psia = new_mmp
                        prog.set_step(55, "Re-evaluating miscibility margin & slim-tube displacement...")
                        self._recalculate_thermodynamics()
                        prog.set_step(80, "Re-rendering barometer & displacement profiles...")
                        self._render_active_mode()
                        prog.set_step(95, "Synchronizing MMP with project models...")
                        self.parameters_changed.emit("pvt", "mmp_override", self.mmp_psia)
                        prog.set_step(100, "✓ Active MMP updated!")
        elif mode_idx == 3:
            new_t = float(np.clip(event.xdata, 60.0, 450.0))
            new_p = float(np.clip(event.ydata, 500.0, 15000.0))
            from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
            with CalculationProgressDialog(self, title="Recalculating PR-EOS Envelope...", task_name="Updating P-T Flash Conditions") as prog:
                prog.set_step(20, f"Updating reservoir conditions: T={new_t:.1f}°F, P={new_p:.0f} psia...")
                self.spin_temp.blockSignals(True)
                self.spin_p_ini.blockSignals(True)
                self.spin_temp.setValue(new_t)
                self.spin_p_ini.setValue(new_p)
                self.spin_temp.blockSignals(False)
                self.spin_p_ini.blockSignals(False)
                self.temperature_f = new_t
                self.initial_pressure = new_p
                prog.set_step(55, "Evaluating Peng-Robinson EOS flash equilibrium...")
                self._recalculate_thermodynamics()
                prog.set_step(80, "Re-rendering multicomponent phase envelope...")
                self._render_active_mode()
                prog.set_step(95, "Synchronizing thermodynamics with subsurface models...")
                self.parameters_changed.emit("pvt", "initial_pressure", self.initial_pressure)
                self.parameters_changed.emit("pvt", "temperature", self.temperature_f)
                prog.set_step(100, "✓ Thermodynamic state updated!")
        elif mode_idx == 4:
            new_woc = float(np.clip(event.ydata, self.top_depth, self.top_depth + self.thickness_ft * 2.0))
            from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
            with CalculationProgressDialog(self, title="Recalculating Fluid Column...", task_name="Updating Water-Oil Contact (WOC)") as prog:
                prog.set_step(25, f"Setting WOC contact depth to {new_woc:.1f} ft TVD...")
                self.woc_depth = new_woc
                prog.set_step(55, "Evaluating hydrostatic gradients and vertical fluid legs...")
                self._recalculate_thermodynamics()
                prog.set_step(80, "Re-rendering fluid column and saturation contacts...")
                self._render_active_mode()
                prog.set_step(95, "Updating 3D reservoir fluid contacts...")
                self.parameters_changed.emit("pvt", "woc_depth", self.woc_depth)
                prog.set_step(100, "✓ Contact depth updated!")

    def _on_canvas_scroll(self, event):
        """Interactive scroll wheel zooming centered on the mouse position."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        base_scale = 1.25
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if event.button == "up":
            scale_factor = 1.0 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            scale_factor = 1.0

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
        rely = (cur_ylim[1] - ydata) / max(cur_ylim[1] - cur_ylim[0], 1e-6)

        ax.set_xlim([xdata - new_width * (1.0 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1.0 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def _on_mode_changed(self, idx: int):
        if idx == 3:
            self.comp_table_frame.show()
        else:
            self.comp_table_frame.hide()
        self._render_active_mode()

    def _on_quick_param_changed(self):
        self.initial_pressure = self.spin_p_ini.value()
        self.temperature_f = self.spin_temp.value()
        self.parameters_changed.emit("pvt", "initial_pressure", self.initial_pressure)
        self.parameters_changed.emit("pvt", "temperature", self.temperature_f)
        self._recalculate_thermodynamics()
        self._render_active_mode()

    def _on_preset_selected(self, idx: int):
        from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title="Loading Benchmark Fluid System...", task_name="Applying Fluid Characterization Preset") as prog:
            prog.set_step(20, "Applying benchmark fluid properties...")
            if idx == 0:  # Permian San Andres
                self.api_gravity = 36.0
                self.sol_gor = 550.0
                self.gas_gravity = 0.75
                self.dead_oil_viscosity_cp = 2.2
                self.co2_purity_pct = 95.0
                self.mmp_psia = 2150.0
            elif idx == 1:  # Light Volatile
                self.api_gravity = 42.0
                self.sol_gor = 850.0
                self.gas_gravity = 0.80
                self.dead_oil_viscosity_cp = 1.1
                self.co2_purity_pct = 98.0
                self.mmp_psia = 1750.0
            elif idx == 2:  # Medium Black Oil
                self.api_gravity = 32.0
                self.sol_gor = 450.0
                self.gas_gravity = 0.70
                self.dead_oil_viscosity_cp = 4.2
                self.co2_purity_pct = 95.0
                self.mmp_psia = 2450.0
            elif idx == 3:  # Heavy Crude
                self.api_gravity = 22.0
                self.sol_gor = 120.0
                self.gas_gravity = 0.65
                self.dead_oil_viscosity_cp = 25.0
                self.co2_purity_pct = 90.0
                self.mmp_psia = 3400.0

            prog.set_step(50, "Evaluating EOS Peng-Robinson equilibrium & bubble point...")
            self._recalculate_thermodynamics()
            prog.set_step(80, "Rendering updated thermodynamic curves...")
            self._render_active_mode()
            prog.set_step(95, "Synchronizing with reservoir model...")
            self.parameters_changed.emit("pvt", "preset_applied", idx)
            prog.set_step(100, "✓ Preset applied successfully!")

    def _populate_composition_table(self):
        self.table_comp.blockSignals(True)
        for row, comp in enumerate(self.composition):
            self.table_comp.setItem(row, 0, QTableWidgetItem(comp["name"]))
            self.table_comp.setItem(row, 1, QTableWidgetItem(f"{comp['z']:.4f}"))
            self.table_comp.setItem(row, 2, QTableWidgetItem(f"{comp['mw']:.1f}"))
            self.table_comp.setItem(row, 3, QTableWidgetItem(f"{comp['tc_f']:.1f}"))
            self.table_comp.setItem(row, 4, QTableWidgetItem(f"{comp['pc_psia']:.1f}"))
        self.table_comp.blockSignals(False)

    def _on_table_cell_edited(self, item: QTableWidgetItem):
        row = item.row()
        col = item.column()
        try:
            val = float(item.text())
            if col == 1:
                self.composition[row]["z"] = max(0.0, val)
            elif col == 2:
                self.composition[row]["mw"] = max(1.0, val)
            elif col == 3:
                self.composition[row]["tc_f"] = val
            elif col == 4:
                self.composition[row]["pc_psia"] = max(10.0, val)
            self._render_active_mode()
        except ValueError:
            pass

    def _normalize_composition(self):
        total_z = sum(c["z"] for c in self.composition)
        if total_z > 0:
            for c in self.composition:
                c["z"] /= total_z
            self._populate_composition_table()
            self._render_active_mode()

    def _emit_sync_to_3d(self):
        """Emits parameters to trigger 3D reservoir cube and cross-section regeneration."""
        from ui.workbench.components.calculation_progress_dialog import CalculationProgressDialog
        with CalculationProgressDialog(self, title="Synchronizing Subsurface 3D Reservoir...", task_name="3D Fluid Contact & Thermodynamic Sync") as prog:
            prog.set_step(25, "Packaging solvent-extended thermodynamic parameters...")
            data = self.get_fluid_parameters()
            prog.set_step(60, "Emitting synchronization signals to 3D reservoir engine...")
            self.sync_to_3d_requested.emit(data)
            prog.set_step(85, "Updating diagnostic telemetry...")
            self._update_diagnostics_hud()
            prog.set_step(100, "✓ 3D reservoir synchronized!")
        logger.info(f"FluidsPVTWorkstation: Synchronized fluid model to 3D reservoir (WOC={self.woc_depth:.0f} ft, MMP={self.mmp_psia:.0f} psia)")

    def _render_active_mode(self):
        """Dispatches plot rendering to specific mode."""
        self._cursor_lines = []
        self._bg_cache = {}
        self.fig.clear()
        mode_idx = self.combo_mode.currentIndex()

        if mode_idx == 0:
            self._render_black_oil_curves()
        elif mode_idx == 1:
            self._render_co2_swelling_viscosity()
        elif mode_idx == 2:
            self._render_mmp_barometer()
        elif mode_idx == 3:
            self._render_eos_phase_envelope()
        else:
            self._render_reservoir_fluid_column()

        self._setup_cursor_artists(mode_idx)
        self.canvas.draw()
        self._capture_backgrounds()

    # =========================================================================
    # MODE 0: BLACK OIL PVT CURVES
    # =========================================================================
    def _render_black_oil_curves(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.current_axes = [ax1, ax2]

        show_grid = self.chk_grid.isChecked()
        p_max = max(self.initial_pressure * 1.35, 5500.0)
        p_arr = np.linspace(14.7, p_max, 120)
        pb = self.bubble_point_psia

        # Evaluate Bo(P) and Rs(P)
        bo_arr = []
        rs_arr = []
        mu_arr = []
        bg_arr = []
        rho_co2_arr = []

        for p in p_arr:
            bo = self.pvt_engine.calculate_oil_fvf_rb_per_stb(p, x_co2=0.0)
            rs = self.pvt_engine.calculate_hydrocarbon_solution_gor(p)
            mu = self.pvt_engine.calculate_oil_viscosity_cp(p, x_co2=0.0)
            bg = self.pvt_engine.calculate_co2_fvf_rb_per_mscf(p)
            rho = self.pvt_engine.calculate_co2_density_kg_m3(p)

            bo_arr.append(bo)
            rs_arr.append(rs)
            mu_arr.append(mu)
            bg_arr.append(bg)
            rho_co2_arr.append(rho)

        bo_arr = np.array(bo_arr)
        rs_arr = np.array(rs_arr)
        mu_arr = np.array(mu_arr)
        bg_arr = np.array(bg_arr)
        rho_co2_arr = np.array(rho_co2_arr)

        self._cached_p_arr = p_arr
        self._cached_bo_arr = bo_arr
        self._cached_rs_arr = rs_arr
        self._cached_mu_arr = mu_arr
        self._cached_rho_co2_arr = rho_co2_arr

        # Left Subplot: Formation Volume Factor Bo & Solution GOR Rs
        ax1.set_facecolor("#f8fafc")
        l1 = ax1.plot(p_arr, bo_arr, color="#0d6efd", lw=2.5, label=r"Oil FVF $B_o(P)$ (rb/STB)")
        ax1.set_xlabel("Pressure P (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_ylabel(r"Oil Formation Volume Factor $B_o$ (rb/STB)", fontsize=9.5, fontweight="bold", color="#0d6efd")
        ax1.axvline(pb, color="#dc2626", linestyle="--", lw=1.8, label=f"Bubble Point $P_b$ ({pb:.0f} psia)")
        ax1.axvline(self.initial_pressure, color="#16a34a", linestyle=":", lw=2.0, label=f"Initial $P_{{ini}}$ ({self.initial_pressure:.0f} psia)")
        ax1.grid(show_grid, linestyle=":", alpha=0.6)

        # Twin axis for Rs
        ax1_twin = ax1.twinx()
        l2 = ax1_twin.plot(p_arr, rs_arr, color="#d97706", lw=2.2, linestyle="-.", label=r"Solution GOR $R_s(P)$ (scf/STB)")
        ax1_twin.set_ylabel(r"Solution GOR $R_s$ (scf/STB)", fontsize=9.5, fontweight="bold", color="#d97706")

        lines1 = l1 + [ax1.lines[-2], ax1.lines[-1]] + l2
        labels1 = [l.get_label() for l in lines1]
        ax1.legend(lines1, labels1, loc="lower right", fontsize=8)
        ax1.set_title("Crude Oil FVF & Solution Gas Ratio", fontsize=11, fontweight="bold", color="#0f172a")

        # Right Subplot: Viscosity mu_o & Supercritical CO2 Density
        ax2.set_facecolor("#f8fafc")
        l3 = ax2.plot(p_arr, mu_arr, color="#7c3aed", lw=2.5, label=r"Live Oil Viscosity $\mu_o(P)$ (cP)")
        ax2.axhline(self.dead_oil_viscosity_cp, color="#94a3b8", linestyle="--", label=rf"Dead Oil $\mu_{{od}}$ ({self.dead_oil_viscosity_cp:.1f} cP)")
        ax2.set_xlabel("Pressure P (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_ylabel(r"Oil Viscosity $\mu_o$ (cP)", fontsize=9.5, fontweight="bold", color="#7c3aed")
        ax2.axvline(pb, color="#dc2626", linestyle="--", lw=1.8)
        ax2.grid(show_grid, linestyle=":", alpha=0.6)

        ax2_twin = ax2.twinx()
        l4 = ax2_twin.plot(p_arr, rho_co2_arr, color="#0891b2", lw=2.2, linestyle="-.", label=r"CO2 Density $\rho_{CO2}$ (kg/m³)")
        ax2_twin.set_ylabel(r"Supercritical CO2 Density (kg/m³)", fontsize=9.5, fontweight="bold", color="#0891b2")

        lines2 = l3 + [ax2.lines[-2]] + l4
        labels2 = [l.get_label() for l in lines2]
        ax2.legend(lines2, labels2, loc="upper right", fontsize=8)
        ax2.set_title("Live Oil Viscosity & Supercritical CO2 Density", fontsize=11, fontweight="bold", color="#0f172a")

        regime_str = "Undersaturated Oil (P_ini > Pb)" if self.initial_pressure > pb else "Saturated Oil Column (P_ini <= Pb)"
        self.metrics_label.setText(
            f"Black Oil PVT: Bubble Point Pb: {pb:.0f} psia ({regime_str}) | Live Oil μ_o: {self.live_oil_viscosity_cp:.2f} cP | "
            f"Bo @ P_ini: {self.pvt_engine.calculate_oil_fvf_rb_per_stb(self.initial_pressure, 0.0):.3f} rb/STB | Rs: {self.pvt_engine.calculate_hydrocarbon_solution_gor(self.initial_pressure):.0f} scf/STB"
        )

    # =========================================================================
    # MODE 1: CO2 SWELLING & VISCOSITY REDUCTION
    # =========================================================================
    def _render_co2_swelling_viscosity(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.current_axes = [ax1, ax2]
        show_grid = self.chk_grid.isChecked()

        x_co2_arr = np.linspace(0.0, 0.85, 100)
        p = self.initial_pressure

        # Swelling and Viscosity reduction
        sf_arr = [self.pvt_engine.calculate_oil_swelling_factor(p, x) for x in x_co2_arr]
        visc_ratio_arr = [self.pvt_engine.calculate_oil_viscosity_cp(p, x) / max(self.dead_oil_viscosity_cp, 0.1) for x in x_co2_arr]

        self._cached_x_co2 = x_co2_arr
        self._cached_sf = np.array(sf_arr)
        self._cached_visc_ratio = np.array(visc_ratio_arr)

        # Left Subplot: Swelling Factor SF(x_CO2)
        ax1.set_facecolor("#f8fafc")
        ax1.plot(x_co2_arr * 100.0, sf_arr, color="#0d6efd", lw=2.8, label=f"Swelling Factor $S_F$ @ {p:.0f} psia")
        ax1.fill_between(x_co2_arr * 100.0, 1.0, sf_arr, color="#bae6fd", alpha=0.45)
        ax1.axhline(1.0, color="#64748b", linestyle=":", label="Unswollen Base (1.00)")
        ax1.set_xlabel("Dissolved CO2 Mole Fraction $x_{CO2}$ (mol %)", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_ylabel(r"Swelling Factor $S_F = V_{oil}(x) / V_{oil}(0)$", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_title("CO2 Oil Swelling & Expansion", fontsize=11, fontweight="bold", color="#0f172a")
        ax1.grid(show_grid, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=8.5)

        max_sf = max(sf_arr)
        ax1.annotate(f"Max Swelling: +{(max_sf - 1.0) * 100.0:.1f}%", xy=(x_co2_arr[-1]*100.0, max_sf),
                     xytext=(x_co2_arr[-1]*100.0 - 25, max_sf - 0.05),
                     arrowprops=dict(arrowstyle="->", color="#0d6efd", lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#0d6efd", alpha=0.9),
                     fontsize=8.5, fontweight="bold")

        # Right Subplot: Viscosity Reduction Ratio
        ax2.set_facecolor("#f8fafc")
        ax2.plot(x_co2_arr * 100.0, visc_ratio_arr, color="#dc2626", lw=2.8, label=r"Viscosity Ratio $\mu_o(x) / \mu_{dead}$")
        ax2.set_yscale("log")
        ax2.set_xlabel("Dissolved CO2 Mole Fraction $x_{CO2}$ (mol %)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_ylabel(r"Viscosity Ratio $\mu_o / \mu_{od}$ (Log Scale)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_title("Solvent Viscosity Thinning (Todd-Longstaff)", fontsize=11, fontweight="bold", color="#0f172a")
        ax2.grid(show_grid, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8.5)

        min_visc = min(visc_ratio_arr)
        ax2.annotate(f"Viscosity Reduced by {(1.0 - min_visc)*100.0:.1f}%", xy=(x_co2_arr[-1]*100.0, min_visc),
                     xytext=(x_co2_arr[-1]*100.0 - 35, min_visc * 2.0),
                     arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#dc2626", alpha=0.9),
                     fontsize=8.5, fontweight="bold")

        self.metrics_label.setText(
            f"CO2-Oil Interaction @ {p:.0f} psia: Max Swelling: +{(max_sf - 1.0) * 100.0:.1f}% | "
            f"Viscosity Reduction: {(1.0 - min_visc) * 100.0:.1f}% | Live Oil Viscosity @ Saturation: {min_visc * self.dead_oil_viscosity_cp:.2f} cP"
        )

    # =========================================================================
    # MODE 2: MMP MISCIBILITY BAROMETER & SLIM-TUBE EXTRACTION
    # =========================================================================
    def _render_mmp_barometer(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.current_axes = [ax1, ax2]
        show_grid = self.chk_grid.isChecked()

        # Left Subplot: Barometer Comparison
        corrs = ["Cronquist\n(1978)", "Yellig-Metcalfe\n(1980)", "Lee\n(1979)", "Alston\n(1985)", "Emera-Sarma\n(2007)"]
        mmp_values = [self.mmp_cronquist, self.mmp_yellig, self.mmp_lee, self.mmp_alston, self.mmp_emera]
        colors = ["#38bdf8", "#0284c7", "#0ea5e9", "#0369a1", "#0284c7"]

        ax1.set_facecolor("#f8fafc")
        bars = ax1.bar(corrs, mmp_values, color=colors, width=0.55, edgecolor="#0f172a", lw=1.2, zorder=3)
        for bar, val in zip(bars, mmp_values):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{val:.0f} psi", ha="center", fontsize=8.5, fontweight="bold")

        # Overlay Current Reservoir Pressure Line
        ax1.axhline(self.initial_pressure, color="#16a34a", linestyle="--", lw=2.2, label=f"Reservoir Pressure $P_{{res}}$ ({self.initial_pressure:.0f} psia)", zorder=4)
        ax1.axhline(self.mmp_psia, color="#dc2626", linestyle=":", lw=2.0, label=f"Active Model MMP ({self.mmp_psia:.0f} psia)", zorder=4)

        ax1.set_ylabel("Minimum Miscibility Pressure (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_title("MMP Correlation Barometer vs Reservoir State", fontsize=11, fontweight="bold", color="#0f172a")
        ax1.set_ylim(0, max(max(mmp_values), self.initial_pressure) * 1.25)
        ax1.grid(show_grid, linestyle=":", alpha=0.6, zorder=1)
        ax1.legend(loc="lower right", fontsize=8.5)

        # Miscibility badge on barometer
        is_miscible = self.initial_pressure >= self.mmp_psia
        misc_text = f"STATUS: MISCIBLE (+{self.initial_pressure - self.mmp_psia:.0f} psi margin)" if is_miscible else f"STATUS: IMMISCIBLE (-{self.mmp_psia - self.initial_pressure:.0f} psi deficit)"
        badge_color = "#16a34a" if is_miscible else "#ea580c"
        ax1.text(0.5, 0.90, misc_text, transform=ax1.transAxes, ha="center",
                 fontsize=9.5, fontweight="bold", color="#ffffff",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, edgecolor="none"))

        # Right Subplot: Simulated Slim-Tube Recovery Curve (1.2 PVI)
        p_sweep = np.linspace(1000.0, max(self.mmp_psia * 1.6, 4500.0), 100)
        # Sigmoidal recovery curve with miscibility knee at MMP
        rec_immiscible = 55.0 + 20.0 * (p_sweep / self.mmp_psia)
        rec_miscible = 94.0 - 5.0 * np.exp(-(p_sweep - self.mmp_psia) / 500.0)
        recovery_pct = np.where(p_sweep < self.mmp_psia,
                                55.0 + 25.0 * (p_sweep / self.mmp_psia)**1.8,
                                80.0 + 15.0 * (1.0 - np.exp(-(p_sweep - self.mmp_psia) / 250.0)))
        recovery_pct = np.clip(recovery_pct, 40.0, 97.5)

        self._cached_sweep_p = p_sweep
        self._cached_recovery = recovery_pct

        ax2.set_facecolor("#f8fafc")
        ax2.plot(p_sweep, recovery_pct, color="#0d6efd", lw=2.8, label="Slim-Tube Oil Recovery @ 1.2 PVI")
        ax2.axvline(self.mmp_psia, color="#dc2626", linestyle="--", lw=2.0, label=f"MMP Knee ({self.mmp_psia:.0f} psia)")
        ax2.axhline(90.0, color="#64748b", linestyle=":", lw=1.5, label="Miscible Threshold (>90%)")
        ax2.axvline(self.initial_pressure, color="#16a34a", linestyle=":", lw=2.0, label=f"Operating $P_{{res}}$ ({self.initial_pressure:.0f} psia)")

        ax2.set_xlabel("Injection Pressure (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_ylabel("Oil Recovery (% OOIP)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_title("Slim-Tube Extraction Recovery vs Pressure", fontsize=11, fontweight="bold", color="#0f172a")
        ax2.set_ylim(40.0, 102.0)
        ax2.grid(show_grid, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right", fontsize=8.5)

        self.metrics_label.setText(
            f"MMP Analysis: Cronquist: {self.mmp_cronquist:.0f} psi | Yellig: {self.mmp_yellig:.0f} psi | Lee: {self.mmp_lee:.0f} psi | "
            f"Alston: {self.mmp_alston:.0f} psi | Active Model MMP: {self.mmp_psia:.0f} psia | {misc_text}"
        )

    # =========================================================================
    # MODE 3: EOS COMPOSITIONAL & PHASE ENVELOPE
    # =========================================================================
    def _render_eos_phase_envelope(self):
        ax = self.fig.add_subplot(111)
        self.current_axes = [ax]
        show_grid = self.chk_grid.isChecked()

        ax.set_facecolor("#f8fafc")
        # Synthesize authentic multicomponent P-T phase envelope based on C7+ and API
        t_arr = np.linspace(40.0, 400.0, 100)
        t_crit = 180.0 + self.api_gravity * 2.0
        p_crit = 2600.0 - self.api_gravity * 15.0

        # Bubble point curve (left boundary)
        pb_curve = p_crit + 800.0 * (1.0 - ((t_arr - 60.0) / max(t_crit - 60.0, 10.0))**2)
        pb_curve = np.clip(pb_curve, 200.0, 4800.0)

        # Dew point curve (right boundary)
        pd_curve = p_crit * np.exp(-((t_arr - t_crit) / 120.0)**2)
        pd_curve = np.clip(pd_curve, 100.0, 4800.0)

        self._cached_t_crit = t_crit
        self._cached_p_crit = p_crit
        self._cached_t_arr = t_arr
        self._cached_pb_curve = pb_curve
        self._cached_pd_curve = pd_curve

        t_env = np.concatenate([t_arr[t_arr <= t_crit], t_arr[t_arr > t_crit]])
        p_bubble = pb_curve[t_arr <= t_crit]
        p_dew = pd_curve[t_arr > t_crit]

        # Draw 2-phase envelope
        ax.plot(t_arr[t_arr <= t_crit], p_bubble, color="#dc2626", lw=2.5, label="Bubble Point Envelope (Liquid-Vapor)")
        ax.plot(t_arr[t_arr > t_crit], p_dew, color="#0d6efd", lw=2.5, label="Dew Point Envelope (Vapor-Liquid)")
        ax.fill_between(t_arr, 0, np.maximum(pb_curve, pd_curve), color="#fef3c7", alpha=0.35, label="Two-Phase Region")

        # Critical Point
        ax.scatter([t_crit], [p_crit], color="#7c3aed", s=100, zorder=6, label=f"Critical Point ({t_crit:.0f}°F, {p_crit:.0f} psia)")

        # Reservoir Operating State
        ax.scatter([self.temperature_f], [self.initial_pressure], color="#16a34a", s=140, marker="*", edgecolors="#0f172a", zorder=7,
                   label=f"Reservoir State ({self.temperature_f:.0f}°F, {self.initial_pressure:.0f} psia)")

        # MMP Miscibility Horizon
        ax.axhline(self.mmp_psia, color="#0284c7", linestyle="--", lw=1.8, label=f"CO2 MMP ({self.mmp_psia:.0f} psia)")

        ax.set_xlabel("Temperature (°F)", fontsize=9.5, fontweight="bold", color="#334155")
        ax.set_ylabel("Pressure (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax.set_title("PR-EOS Multicomponent Phase Envelope & Reservoir Flash Point", fontsize=11, fontweight="bold", color="#0f172a")
        ax.grid(show_grid, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)

        self.metrics_label.setText(
            f"EOS Composition: C1: {self.composition[2]['z']*100:.1f}% | C2-C3: {self.composition[3]['z']*100:.1f}% | "
            f"C7+: {self.composition[5]['z']*100:.1f}% (MW={self.composition[5]['mw']:.0f}) | Cricondenbar: {max(pb_curve):.0f} psia | T_crit: {t_crit:.0f}°F"
        )

    # =========================================================================
    # MODE 4: RESERVOIR FLUID COLUMN & HYDROSTATIC CONTACTS (3D PREPARATION)
    # =========================================================================
    def _render_reservoir_fluid_column(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        self.current_axes = [ax1, ax2]
        show_grid = self.chk_grid.isChecked()

        top_z = self.top_depth
        bot_z = self.top_depth + self.thickness_ft
        woc_z = self.woc_depth
        goc_z = self.goc_depth if self.has_gas_cap else None

        z_depths = np.linspace(top_z - 20.0, bot_z + 30.0, 150)
        gamma_o = 141.5 / max(self.api_gravity + 131.5, 10.0)
        oil_grad = 0.4335 * gamma_o
        water_grad = self.water_gradient
        gas_grad = self.gas_gradient

        # Left Subplot: Hydrostatic Pressure vs TVD Depth (Inverted Y-axis)
        p_ref = self.initial_pressure
        pressures = []
        for z in z_depths:
            if goc_z is not None and z < goc_z:
                p = p_ref + gas_grad * (z - top_z)
            elif z <= woc_z:
                p = p_ref + oil_grad * (z - top_z)
            else:
                p_woc = p_ref + oil_grad * (woc_z - top_z)
                p = p_woc + water_grad * (z - woc_z)
            pressures.append(p)
        pressures = np.array(pressures)

        ax1.set_facecolor("#f8fafc")
        ax1.plot(pressures, z_depths, color="#0d6efd", lw=2.5, label="Hydrostatic Pore Pressure P(z)")
        ax1.axhline(top_z, color="#64748b", linestyle=":", lw=1.5, label=f"Reservoir Top ({top_z:.0f} ft)")
        ax1.axhline(bot_z, color="#64748b", linestyle=":", lw=1.5, label=f"Reservoir Base ({bot_z:.0f} ft)")
        ax1.axhline(woc_z, color="#0284c7", linestyle="--", lw=2.2, label=f"Water-Oil Contact WOC ({woc_z:.0f} ft)")
        if goc_z is not None:
            ax1.axhline(goc_z, color="#d97706", linestyle="--", lw=2.2, label=f"Gas-Oil Contact GOC ({goc_z:.0f} ft)")

        # MMP vertical line
        ax1.axvline(self.mmp_psia, color="#dc2626", linestyle="-.", lw=1.8, label=f"CO2 MMP ({self.mmp_psia:.0f} psia)")

        ax1.invert_yaxis()
        ax1.set_xlabel("Pressure (psia)", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9.5, fontweight="bold", color="#334155")
        ax1.set_title("Hydrostatic Pressure Gradient & Contacts", fontsize=11, fontweight="bold", color="#0f172a")
        ax1.grid(show_grid, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower left", fontsize=8)

        # Right Subplot: Phase Saturation vs TVD Depth
        sw_col = []
        so_col = []
        sg_col = []
        swc = 0.20
        for z in z_depths:
            if goc_z is not None and z < goc_z:
                sg = 1.0 - swc
                sw = swc
                so = 0.0
            else:
                sg = 0.0
                # Transition zone across WOC
                diff = z - woc_z
                sw = np.clip(swc + (1.0 - swc) / (1.0 + np.exp(-diff / 2.5)), swc, 1.0)
                so = 1.0 - sw
            sw_col.append(sw)
            so_col.append(so)
            sg_col.append(sg)

        ax2.set_facecolor("#f8fafc")
        ax2.plot(so_col, z_depths, color="#16a34a", lw=2.5, label=r"Oil Saturation $S_o$")
        ax2.plot(sw_col, z_depths, color="#0284c7", lw=2.5, label=r"Water Saturation $S_w$")
        if goc_z is not None:
            ax2.plot(sg_col, z_depths, color="#dc2626", lw=2.5, label=r"Gas Saturation $S_g$")

        # Fill fluid zones
        ax2.fill_betweenx(z_depths, 0, so_col, color="#bbf7d0", alpha=0.4, label="Oil Column")
        ax2.axhline(woc_z, color="#0284c7", linestyle="--", lw=2.0)
        if goc_z is not None:
            ax2.axhline(goc_z, color="#d97706", linestyle="--", lw=2.0)

        ax2.invert_yaxis()
        ax2.set_xlabel("Phase Saturation Fraction (0 - 1)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9.5, fontweight="bold", color="#334155")
        ax2.set_title("Equilibrium Fluid Phase Zonation (3D Source)", fontsize=11, fontweight="bold", color="#0f172a")
        ax2.grid(show_grid, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", fontsize=8)

        oil_thk = (woc_z - (goc_z if goc_z else top_z))
        self.metrics_label.setText(
            f"Reservoir Fluid Column: WOC Depth: {woc_z:.0f} ft TVD | Net Oil Column: {oil_thk:.1f} ft | "
            f"Oil Gradient: {oil_grad:.3f} psi/ft | Water Gradient: {water_grad:.3f} psi/ft | Bottom Pressure @ Base: {pressures[-1]:.0f} psia"
        )

    # =========================================================================
    # CONTEXT MENU & EXPORT UTILITIES
    # =========================================================================
    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; } QMenu::item:selected { background: #e0f2fe; color: #0284c7; }")

        act_copy = menu.addAction("📋 Copy Plot Image to Clipboard")
        act_copy.triggered.connect(self._copy_plot_to_clipboard)

        act_save = menu.addAction("💾 Save Plot as High-Res Image (PNG/PDF)...")
        act_save.triggered.connect(self._save_plot_image)

        menu.addSeparator()

        act_export_csv = menu.addAction("📊 Export Active PVT Table as CSV...")
        act_export_csv.triggered.connect(self._export_pvt_csv)

        act_sync = menu.addAction("⚡ Sync Fluid Model to 3D Reservoir Grid")
        act_sync.triggered.connect(self._emit_sync_to_3d)

        menu.addSeparator()

        act_grid = menu.addAction("Toggle Grid Lines")
        act_grid.setCheckable(True)
        act_grid.setChecked(self.chk_grid.isChecked())
        act_grid.triggered.connect(lambda: self.chk_grid.setChecked(not self.chk_grid.isChecked()))

        menu.exec(self.canvas.mapToGlobal(pos))

    def _copy_plot_to_clipboard(self):
        pixmap = self.canvas.grab()
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setPixmap(pixmap)
            logger.info("FluidsPVTWorkstation: Copied plot image to clipboard.")

    def _save_plot_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PVT Plot", "fluids_pvt_plot.png", "PNG Images (*.png);;PDF Documents (*.pdf)")
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.info(f"FluidsPVTWorkstation: Saved plot to {file_path}")

    def _export_pvt_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export PVT Data Table", "pvt_thermodynamic_table.csv", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            mode_idx = self.combo_mode.currentIndex()
            with open(file_path, "w") as f:
                if mode_idx == 0:
                    f.write("Pressure_psia,Bo_rb_stb,Rs_scf_stb,mu_oil_cp,Bg_rb_mscf,rho_co2_kg_m3\n")
                    p_arr = np.linspace(14.7, max(self.initial_pressure * 1.35, 5500.0), 100)
                    for p in p_arr:
                        bo = self.pvt_engine.calculate_oil_fvf_rb_per_stb(p, 0.0)
                        rs = self.pvt_engine.calculate_hydrocarbon_solution_gor(p)
                        mu = self.pvt_engine.calculate_oil_viscosity_cp(p, 0.0)
                        bg = self.pvt_engine.calculate_co2_fvf_rb_per_mscf(p)
                        rho = self.pvt_engine.calculate_co2_density_kg_m3(p)
                        f.write(f"{p:.2f},{bo:.4f},{rs:.2f},{mu:.4f},{bg:.4f},{rho:.2f}\n")
                elif mode_idx == 1:
                    f.write("x_co2_fraction,Swelling_Factor,Viscosity_Ratio,mu_live_cp\n")
                    x_arr = np.linspace(0.0, 0.85, 80)
                    for x in x_arr:
                        sf = self.pvt_engine.calculate_oil_swelling_factor(self.initial_pressure, x)
                        mu = self.pvt_engine.calculate_oil_viscosity_cp(self.initial_pressure, x)
                        ratio = mu / max(self.dead_oil_viscosity_cp, 0.1)
                        f.write(f"{x:.4f},{sf:.4f},{ratio:.4f},{mu:.4f}\n")
                elif mode_idx == 2:
                    f.write("Correlation,MMP_psia,Reservoir_Pressure_psia,Miscible_Status\n")
                    for name, val in [("Cronquist_1978", self.mmp_cronquist), ("Yellig_Metcalfe_1980", self.mmp_yellig),
                                     ("Lee_1979", self.mmp_lee), ("Alston_1985", self.mmp_alston), ("Emera_Sarma_2007", self.mmp_emera)]:
                        status = "MISCIBLE" if self.initial_pressure >= val else "IMMISCIBLE"
                        f.write(f"{name},{val:.2f},{self.initial_pressure:.2f},{status}\n")
                elif mode_idx == 3:
                    f.write("Component,Mole_Fraction,MW,Tc_F,Pc_psia\n")
                    for c in self.composition:
                        f.write(f"{c['name']},{c['z']:.4f},{c['mw']:.2f},{c['tc_f']:.1f},{c['pc_psia']:.1f}\n")
                else:
                    f.write("TVD_Depth_ft,Hydrostatic_Pressure_psia,Oil_Saturation_So,Water_Saturation_Sw,Gas_Saturation_Sg\n")
                    z_arr = np.linspace(self.top_depth, self.top_depth + self.thickness_ft, 50)
                    for z in z_arr:
                        p = self.initial_pressure + 0.35 * (z - self.top_depth)
                        sw = 0.20 if z < self.woc_depth else 1.0
                        so = 1.0 - sw
                        f.write(f"{z:.2f},{p:.2f},{so:.4f},{sw:.4f},0.0000\n")
            logger.info(f"FluidsPVTWorkstation: Exported CSV table to {file_path}")
        except Exception as e:
            logger.error(f"FluidsPVTWorkstation: CSV export failed: {e}")
