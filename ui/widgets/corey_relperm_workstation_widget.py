"""
Corey Relative Permeability & Two-Phase Displacement Workstation Widget
Part of Subsurface Studio Workbench.

Provides expansive, full-resolution visualization of:
- Water-Oil and Gas-Liquid Corey Rel-Perm curves with Craig's Rule wettability diagnosis.
- Buckley-Leverett Fractional Flow (fw) with Welge tangent construction and shock front (Swf).
- Comparative wettability sensitivity curves.
- Full context menu for copying, saving, and exporting tables.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QCheckBox, QFrame, QMenu,
    QFileDialog, QApplication, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QAction, QCursor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class CoreyRelPermWorkstationWidget(QWidget):
    """
    Dedicated Center Workstation for Corey Relative Permeability & Displacement Modeling.
    Provides full-screen interactive plots and analysis tools for reservoir engineers.
    """
    parameters_updated = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Core Parameters
        self.swc = 0.20
        self.sorw = 0.20
        self.sgc = 0.05
        self.sorg = 0.15
        self.krw0 = 0.30
        self.kro0 = 0.85
        self.krg0 = 0.60
        self.nw = 2.5
        self.now = 2.0
        self.ng = 2.0
        self.visc_ratio = 2.0  # mu_o / mu_w
        self.wettability_preset = "Strongly Water-Wet (Sandstone)"
        self.relperm_model = "Modified Stone I (Standard CO2 EOR)"

        self._setup_ui()
        self.render_plots()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 1. Clean Non-Squished Top Toolbar (General Plotting Settings)
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

        # Display Mode Selector
        top_layout.addWidget(QLabel("Display Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Dual-System (Water-Oil & Gas-Liquid)",
            "Fractional Flow & Welge Shock Front (Displacement)",
            "Comparative Wettability Shift (Sensitivity)"
        ])
        self.combo_mode.currentIndexChanged.connect(lambda _: self.render_plots())
        top_layout.addWidget(self.combo_mode)

        # Scale Selector
        top_layout.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear Scale", "Semi-Log Scale (log10 kr)"])
        self.combo_scale.currentIndexChanged.connect(lambda _: self.render_plots())
        top_layout.addWidget(self.combo_scale)

        # Oil/Water Viscosity Ratio for Fractional Flow
        self.lbl_visc = QLabel("μo/μw Ratio:")
        top_layout.addWidget(self.lbl_visc)
        self.spin_visc = QDoubleSpinBox()
        self.spin_visc.setRange(0.1, 500.0)
        self.spin_visc.setValue(2.0)
        self.spin_visc.setSingleStep(0.5)
        self.spin_visc.valueChanged.connect(lambda v: self._on_visc_changed(v))
        top_layout.addWidget(self.spin_visc)

        # Checkboxes
        self.chk_crossover = QCheckBox("Crossover & Craig's Rule")
        self.chk_crossover.setChecked(True)
        self.chk_crossover.toggled.connect(lambda _: self.render_plots())
        top_layout.addWidget(self.chk_crossover)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(lambda _: self.render_plots())
        top_layout.addWidget(self.chk_grid)

        top_layout.addStretch()

        # Action Buttons
        self.btn_export = QPushButton("📊 Export CSV")
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #0f172a;
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
        self.btn_export.clicked.connect(self._export_curves_csv)
        top_layout.addWidget(self.btn_export)

        self.btn_save_img = QPushButton("📷 Save Image")
        self.btn_save_img.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #0f172a;
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

        main_layout.addWidget(top_frame)

        # 2. Main High-Resolution Plot Canvas
        self.fig = Figure(figsize=(10, 5), facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        main_layout.addWidget(self.canvas, stretch=1)

        # 3. Bottom Metrics Summary Bar
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

    def _on_visc_changed(self, val: float):
        self.visc_ratio = max(float(val), 0.01)
        if self.combo_mode.currentIndex() == 1:
            self.render_plots()

    def set_parameters(self, params: Dict[str, Any]):
        """Synchronizes parameters from project data or property grid."""
        if not params:
            return
        self.swc = float(params.get("s_wc", self.swc))
        self.sorw = float(params.get("s_orw", self.sorw))
        self.sgc = float(params.get("s_gc", self.sgc))
        self.sorg = float(params.get("s_org", self.sorg))
        self.krw0 = float(params.get("krw0", self.krw0))
        self.kro0 = float(params.get("kro0", self.kro0))
        self.krg0 = float(params.get("krg0", self.krg0))
        self.nw = float(params.get("n_w", self.nw))
        self.now = float(params.get("n_ow", params.get("n_o", self.now)))
        self.ng = float(params.get("n_g", self.ng))
        self.wettability_preset = str(params.get("wettability_preset", self.wettability_preset))
        self.relperm_model = str(params.get("relperm_model", self.relperm_model))

        # Check for viscosity ratio in PVT if available
        if "oil_viscosity_cp" in params and "water_viscosity_cp" in params:
            mu_o = float(params.get("oil_viscosity_cp", 1.0))
            mu_w = float(params.get("water_viscosity_cp", 0.5))
            if mu_w > 0:
                self.visc_ratio = mu_o / mu_w
                self.spin_visc.setValue(self.visc_ratio)

        self.render_plots()

    def render_plots(self):
        """Renders comprehensive middle-screen relative permeability and displacement curves."""
        mode_idx = self.combo_mode.currentIndex()
        is_log = (self.combo_scale.currentIndex() == 1)
        show_grid = self.chk_grid.isChecked()
        show_crossover = self.chk_crossover.isChecked()

        self.fig.clear()

        # Water-oil calculations
        swc = self.swc
        sorw = self.sorw
        krw0 = self.krw0
        kro0 = self.kro0
        nw = self.nw
        now = self.now

        delta_sw = max(1.0 - swc - sorw, 1e-4)
        sw_arr = np.linspace(swc, 1.0 - sorw, 150)
        swn = (sw_arr - swc) / delta_sw
        krw = krw0 * (swn ** nw)
        krow = kro0 * ((1.0 - swn) ** now)

        # Crossover saturation Sw*
        diff = np.abs(krw - krow)
        cross_idx = int(np.argmin(diff))
        sw_cross = float(sw_arr[cross_idx])
        kr_cross = float(krw[cross_idx])

        if sw_cross > 0.52:
            wet_diagnosis = "Water-Wet State"
            wet_color = "#16a34a"
        elif sw_cross < 0.48:
            wet_diagnosis = "Oil-Wet State"
            wet_color = "#ea580c"
        else:
            wet_diagnosis = "Intermediate / Mixed-Wet"
            wet_color = "#0284c7"

        if mode_idx == 0:
            # --- MODE 0: Dual-System (Water-Oil & Gas-Liquid) ---
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            # Left: Water-Oil System
            ax1.set_facecolor("#f8fafc")
            ax1.axvspan(swc, 1.0 - sorw, color="#e0f2fe", alpha=0.4, label=f"Mobile Window (ΔSw={delta_sw*100:.1f}%)")
            ax1.plot(sw_arr, krw, color="#0d6efd", lw=2.6, label=f"krw (Water, nw={nw:.2f})")
            ax1.plot(sw_arr, krow, color="#16a34a", lw=2.6, label=f"krow (Oil, now={now:.2f})")

            # Endpoints
            ax1.scatter([swc, 1.0 - sorw], [0.0, krw0], color="#0d6efd", s=45, zorder=6)
            ax1.scatter([swc, 1.0 - sorw], [kro0, 0.0], color="#16a34a", s=45, zorder=6)
            ax1.axvline(swc, color="#64748b", linestyle=":", lw=1.2, label=f"Swc = {swc:.2f}")
            ax1.axvline(1.0 - sorw, color="#64748b", linestyle="--", lw=1.2, label=f"1 - Sorw = {1.0-sorw:.2f}")

            if show_crossover:
                ax1.scatter([sw_cross], [kr_cross], color="#d97706", s=80, edgecolors="#ffffff", linewidths=1.5, zorder=7)
                ax1.annotate(
                    f"Sw* = {sw_cross:.2f}\n({wet_diagnosis})",
                    xy=(sw_cross, kr_cross),
                    xytext=(sw_cross + 0.04, kr_cross + 0.12),
                    fontsize=8.5, fontweight="bold", color=wet_color,
                    arrowprops=dict(arrowstyle="->", color=wet_color, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor=wet_color, alpha=0.9)
                )

            ax1.set_title("Water-Oil Relative Permeability (Corey)", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Water Saturation Sw (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_ylabel("Relative Permeability kr (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_xlim(0.0, 1.0)
            if is_log:
                ax1.set_yscale("log")
                ax1.set_ylim(1e-4, 1.2)
            else:
                ax1.set_ylim(-0.02, max(krw0, kro0) * 1.15)
            ax1.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax1.legend(fontsize=8, loc="upper right" if not is_log else "lower left")

            # Right: Gas-Liquid System
            sgc = self.sgc
            sorg = self.sorg
            krg0 = self.krg0
            ng = self.ng
            delta_sg = max(1.0 - sgc - sorg - swc, 1e-4)
            sg_arr = np.linspace(sgc, 1.0 - sorg - swc, 150)
            sgn = (sg_arr - sgc) / delta_sg
            krg = krg0 * (sgn ** ng)
            krog = kro0 * ((1.0 - sgn) ** ng)

            ax2.set_facecolor("#f8fafc")
            ax2.axvspan(sgc, 1.0 - sorg - swc, color="#fef3c7", alpha=0.4, label=f"Gas Mobile Window ({delta_sg*100:.1f}%)")
            ax2.plot(sg_arr, krg, color="#dc2626", lw=2.6, label=f"krg (Gas/CO2, ng={ng:.2f})")
            ax2.plot(sg_arr, krog, color="#7c3aed", lw=2.6, label=f"krog (Oil-Gas, nog={ng:.2f})")

            ax2.scatter([sgc, 1.0 - sorg - swc], [0.0, krg0], color="#dc2626", s=45, zorder=6)
            ax2.scatter([sgc, 1.0 - sorg - swc], [kro0, 0.0], color="#7c3aed", s=45, zorder=6)
            ax2.axvline(sgc, color="#64748b", linestyle=":", lw=1.2, label=f"Sgc = {sgc:.2f}")
            ax2.axvline(1.0 - sorg - swc, color="#64748b", linestyle="--", lw=1.2, label=f"1 - Sorg - Swc = {1.0-sorg-swc:.2f}")

            ax2.set_title("Gas-Liquid / Solvent Relative Permeability", fontsize=11, fontweight="bold", color="#0f172a")
            ax2.set_xlabel("Gas / CO2 Saturation Sg (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_ylabel("Relative Permeability kr (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_xlim(0.0, 1.0)
            if is_log:
                ax2.set_yscale("log")
                ax2.set_ylim(1e-4, 1.2)
            else:
                ax2.set_ylim(-0.02, max(krg0, kro0) * 1.15)
            ax2.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax2.legend(fontsize=8, loc="upper right" if not is_log else "lower left")

            # Update Metrics
            self.metrics_label.setText(
                f"Mobile PV ΔSw: {delta_sw*100:.1f}% | Crossover Sw*: {sw_cross:.2f} ({wet_diagnosis}) | "
                f"krw0: {krw0:.2f} | kro0: {kro0:.2f} | krg0: {krg0:.2f} | nw: {nw:.2f}, now: {now:.2f}, ng: {ng:.2f}"
            )

        elif mode_idx == 1:
            # --- MODE 1: Fractional Flow & Welge Shock Front ---
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            # Buckley-Leverett Fractional Flow: fw = 1 / [1 + (kro/krw) * (mu_w/mu_o)]
            visc_ratio = max(self.visc_ratio, 0.01)  # mu_o / mu_w
            fw = np.zeros_like(sw_arr)
            for i in range(len(sw_arr)):
                if krw[i] <= 1e-6:
                    fw[i] = 0.0
                elif krow[i] <= 1e-6:
                    fw[i] = 1.0
                else:
                    fw[i] = 1.0 / (1.0 + (krow[i] / krw[i]) / visc_ratio)

            # Welge Tangent from (Swc, 0)
            # Find maximum slope: (fw - 0) / (Sw - Swc)
            slopes = np.zeros_like(sw_arr)
            for i in range(len(sw_arr)):
                if sw_arr[i] > swc + 0.01:
                    slopes[i] = fw[i] / (sw_arr[i] - swc)
                else:
                    slopes[i] = 0.0

            tan_idx = int(np.argmax(slopes))
            swf = float(sw_arr[tan_idx])
            fw_front = float(fw[tan_idx])
            max_slope = float(slopes[tan_idx])

            # Average saturation at breakthrough: Sw_avg = Swc + 1 / slope
            sw_avg_bt = min(swc + (1.0 / max(max_slope, 1e-4)), 1.0 - sorw)
            ed_bt = (sw_avg_bt - swc) / max(1.0 - swc, 1e-4) * 100.0

            # Left Subplot: Fractional Flow
            ax1.set_facecolor("#f8fafc")
            ax1.plot(sw_arr, fw, color="#0d6efd", lw=2.6, label=f"fw(Sw) [μo/μw={visc_ratio:.1f}]")

            # Welge Tangent Line
            tan_x = np.array([swc, sw_avg_bt])
            tan_y = np.array([0.0, 1.0])
            ax1.plot(tan_x, tan_y, color="#dc2626", linestyle="--", lw=2.0, label="Welge Tangent Line")
            ax1.scatter([swf], [fw_front], color="#dc2626", s=75, zorder=6, label=f"Shock Front (Swf = {swf:.2f})")
            ax1.scatter([sw_avg_bt], [1.0], color="#16a34a", s=75, zorder=6, label=f"Avg at BT (Sw_bt = {sw_avg_bt:.2f})")

            ax1.axvline(swc, color="#64748b", linestyle=":", lw=1.2, label=f"Swc = {swc:.2f}")
            ax1.axhline(1.0, color="#cbd5e1", linestyle=":", lw=1.0)

            ax1.annotate(
                f"Front Swf: {swf:.2f}\nRecovery at BT: {ed_bt:.1f}% OOIP",
                xy=(swf, fw_front),
                xytext=(swf - 0.22, fw_front + 0.15),
                fontsize=8.5, fontweight="bold", color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#dc2626", alpha=0.9)
            )

            ax1.set_title("Buckley-Leverett Fractional Flow Curve fw(Sw)", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Water Saturation Sw (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_ylabel("Water Fractional Flow fw (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_xlim(0.0, 1.0)
            ax1.set_ylim(-0.02, 1.05)
            ax1.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax1.legend(fontsize=8, loc="lower right")

            # Right Subplot: Advancing Saturation Shock Profile
            ax2.set_facecolor("#f8fafc")
            xd = np.linspace(0.0, 1.0, 120)
            sw_prof = np.zeros_like(xd)
            front_pos = 0.55  # normalized breakthrough snapshot

            for i, x in enumerate(xd):
                if x <= front_pos:
                    frac = 1.0 - (x / front_pos)
                    sw_prof[i] = swf + (1.0 - sorw - swf) * (frac ** 0.5)
                else:
                    sw_prof[i] = swc

            ax2.plot(xd, sw_prof, color="#0d6efd", lw=2.6, label="Saturation Profile Sw(x)")
            ax2.fill_between(xd, swc, sw_prof, color="#bae6fd", alpha=0.4, label="Invaded Water Bank")
            ax2.axvline(front_pos, color="#dc2626", linestyle="--", lw=2.0, label=f"Shock Front x_D = {front_pos:.2f}")

            ax2.set_title("Advancing Saturation Shock Profile Sw(x_D)", fontsize=11, fontweight="bold", color="#0f172a")
            ax2.set_xlabel("Dimensionless Distance x_D (Injector to Producer)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_ylabel("Water Saturation Sw (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_xlim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.0)
            ax2.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax2.legend(fontsize=8, loc="upper right")

            self.metrics_label.setText(
                f"Welge Front Swf: {swf:.2f} | Avg Sw at BT: {sw_avg_bt:.2f} | "
                f"Frontal Recovery ED: {ed_bt:.1f}% OOIP | Oil/Water Viscosity Ratio μo/μw: {visc_ratio:.1f}"
            )

        else:
            # --- MODE 2: Comparative Wettability Shift (Sensitivity Analysis) ---
            ax1 = self.fig.add_subplot(111)
            ax1.set_facecolor("#f8fafc")

            # 1. Strongly Water-Wet (Sandstone)
            sw_ww = np.linspace(0.20, 0.80, 100)
            swn_ww = (sw_ww - 0.20) / 0.60
            krw_ww = 0.25 * (swn_ww ** 3.0)
            kro_ww = 0.85 * ((1.0 - swn_ww) ** 1.8)
            ax1.plot(sw_ww, krw_ww, color="#0d6efd", lw=2.2, linestyle="-", label="Water-Wet krw (Swc=0.20, krw0=0.25)")
            ax1.plot(sw_ww, kro_ww, color="#0d6efd", lw=2.2, linestyle="--", label="Water-Wet krow (Sorw=0.20, kro0=0.85)")

            # 2. Strongly Oil-Wet (Carbonate/Bitumen)
            sw_ow = np.linspace(0.12, 0.70, 100)
            swn_ow = (sw_ow - 0.12) / 0.58
            krw_ow = 0.75 * (swn_ow ** 1.8)
            kro_ow = 0.40 * ((1.0 - swn_ow) ** 3.2)
            ax1.plot(sw_ow, krw_ow, color="#dc2626", lw=2.2, linestyle="-", label="Oil-Wet krw (Swc=0.12, krw0=0.75)")
            ax1.plot(sw_ow, kro_ow, color="#dc2626", lw=2.2, linestyle="--", label="Oil-Wet krow (Sorw=0.30, kro0=0.40)")

            # 3. Active Current Model
            ax1.plot(sw_arr, krw, color="#16a34a", lw=3.0, linestyle="-", label=f"Current Model krw ({self.wettability_preset})")
            ax1.plot(sw_arr, krow, color="#16a34a", lw=3.0, linestyle="--", label="Current Model krow")
            ax1.scatter([sw_cross], [kr_cross], color="#d97706", s=85, zorder=6, label=f"Current Crossover Sw*={sw_cross:.2f}")

            ax1.set_title("Wettability Shift Comparison (Craig's Rule Benchmarks)", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Water Saturation Sw (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_ylabel("Relative Permeability kr (fraction)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_xlim(0.0, 1.0)
            if is_log:
                ax1.set_yscale("log")
                ax1.set_ylim(1e-4, 1.2)
            else:
                ax1.set_ylim(-0.02, 1.05)
            ax1.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax1.legend(fontsize=8, loc="upper right" if not is_log else "lower left")

            self.metrics_label.setText(
                f"Current Model: {self.wettability_preset} | Crossover Sw*: {sw_cross:.2f} ({wet_diagnosis}) | "
                f"Craig's Rule Criterion: Sw* > 0.52 (Water-Wet), Sw* < 0.48 (Oil-Wet)"
            )

        self.canvas.draw()

    def _show_context_menu(self, pos: QPoint):
        """Right-click context menu for rich interaction and quick plot exporting."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                color: #0f172a;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item {
                padding: 4px 20px 4px 20px;
                border-radius: 2px;
            }
            QMenu::item:selected {
                background-color: #0d6efd;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #e2e8f0;
                margin: 4px 6px;
            }
        """)

        act_copy = menu.addAction("📋 Copy Plot Image to Clipboard")
        act_save = menu.addAction("💾 Save Plot as Image (PNG/PDF)...")
        act_csv = menu.addAction("📊 Export Relative Permeability Table (CSV)...")
        menu.addSeparator()

        act_log = menu.addAction("🔄 Toggle Semi-Log / Linear Scale")
        act_grid = menu.addAction("▦ Toggle Grid")
        menu.addSeparator()
        act_reset = menu.addAction("🔍 Reset Default View")

        action = menu.exec(self.canvas.mapToGlobal(pos))
        if action == act_copy:
            self._copy_plot_to_clipboard()
        elif action == act_save:
            self._save_plot_image()
        elif action == act_csv:
            self._export_curves_csv()
        elif action == act_log:
            cur_idx = self.combo_scale.currentIndex()
            self.combo_scale.setCurrentIndex(1 if cur_idx == 0 else 0)
        elif action == act_grid:
            self.chk_grid.setChecked(not self.chk_grid.isChecked())
        elif action == act_reset:
            self.combo_mode.setCurrentIndex(0)
            self.combo_scale.setCurrentIndex(0)
            self.render_plots()

    def _copy_plot_to_clipboard(self):
        pixmap = self.canvas.grab()
        QApplication.clipboard().setPixmap(pixmap)
        logger.info("Copied rel-perm plot to clipboard")

    def _save_plot_image(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Rel-Perm Plot Image", "relperm_curves.png", "PNG Images (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        if filepath:
            self.fig.savefig(filepath, dpi=300, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
            logger.info(f"Saved plot image to {filepath}")

    def _export_curves_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Rel-Perm Table", "relperm_table.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return

        sw_arr = np.linspace(self.swc, 1.0 - self.sorw, 100)
        delta_sw = max(1.0 - self.swc - self.sorw, 1e-4)
        swn = (sw_arr - self.swc) / delta_sw
        krw = self.krw0 * (swn ** self.nw)
        krow = self.kro0 * ((1.0 - swn) ** self.now)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# CO2 EOR Optimizer - Corey Relative Permeability Table\n")
            f.write(f"# Swc={self.swc}, Sorw={self.sorw}, krw0={self.krw0}, kro0={self.kro0}, nw={self.nw}, now={self.now}\n")
            f.write("Sw,Swn,krw,krow\n")
            for i in range(len(sw_arr)):
                f.write(f"{sw_arr[i]:.4f},{swn[i]:.4f},{krw[i]:.6f},{krow[i]:.6f}\n")
        logger.info(f"Exported rel-perm table to {filepath}")
