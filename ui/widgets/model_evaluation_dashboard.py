"""
Model Evaluation Dashboard & Workstation for CO2 EOR Optimizer.

Provides a unified, research-grade evaluation environment displaying the
5 coupled pillars of the Single Integrated Shared Earth Model:
1. Geology & Reservoir Structure (3D wireframe, wells, layers, fault planes)
2. Petrophysics & Relative Permeability (Water-oil and gas-oil Corey curves)
3. Fluid Thermodynamics & PVT (Bo, Viscosity, Rs, MMP miscibility)
4. Wellbore Hydraulics & Deliverability (Composite Vogel-Darcy IPR, Peaceman WI)
5. Geomechanics & Caprock Containment (Mohr-Coulomb failure envelope, stress path)
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon, QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


class ModelEvaluationDashboard(QWidget):
    """Integrated Shared Earth Model Evaluation Workstation."""

    def __init__(self, project_data: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.project_data = project_data or {}
        self._setup_ui()
        self.update_data(self.project_data)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Top Control & Header Bar
        header_layout = QHBoxLayout()
        title_label = QLabel("Single Integrated Shared Earth Model — Evaluation Workstation")
        title_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1e3d59;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.audit_btn = QPushButton(QIcon.fromTheme("system-run"), "Run Pre-Flight Physical Audit")
        self.audit_btn.setStyleSheet(
            "padding: 6px 14px; font-weight: bold; background-color: #007bff; color: white; border-radius: 4px;"
        )
        self.audit_btn.clicked.connect(self._launch_audit_dialog)
        header_layout.addWidget(self.audit_btn)

        self.refresh_btn = QPushButton(QIcon.fromTheme("view-refresh"), "Refresh Diagnostics")
        self.refresh_btn.clicked.connect(lambda: self.update_data(self.project_data))
        header_layout.addWidget(self.refresh_btn)

        main_layout.addLayout(header_layout)

        # Tabbed Workspace
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_overview_tab(), "1. Multi-Domain Overview")
        self.tabs.addTab(self._create_relperm_tab(), "2. Petrophysics & Rel-Perm")
        self.tabs.addTab(self._create_pvt_tab(), "3. PVT Thermodynamics & MMP")
        self.tabs.addTab(self._create_deliverability_tab(), "4. Well Hydraulics & IPR")
        self.tabs.addTab(self._create_geomechanics_tab(), "5. Geomechanics & Containment")

        main_layout.addWidget(self.tabs)

    def _create_overview_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D Reservoir Structure Canvas
        left_box = QGroupBox("Reservoir Structure & 3D Well Network")
        left_layout = QVBoxLayout(left_box)
        self.overview_fig_3d = Figure(figsize=(6, 5))
        self.overview_canvas_3d = FigureCanvas(self.overview_fig_3d)
        self.overview_ax_3d = self.overview_fig_3d.add_subplot(111, projection='3d')
        left_layout.addWidget(self.overview_canvas_3d)
        splitter.addWidget(left_box)

        # Right: Key Performance Summary Cards
        right_box = QGroupBox("Coupled Pillar State Summary")
        right_layout = QVBoxLayout(right_box)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Coupled Pillar", "Key State Metric", "Health Value"])
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.summary_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.summary_table)

        splitter.addWidget(right_box)
        splitter.setSizes([550, 450])

        layout.addWidget(splitter)
        return panel

    def _create_relperm_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.relperm_fig = Figure(figsize=(8, 5), tight_layout=True)
        self.relperm_canvas = FigureCanvas(self.relperm_fig)
        self.ax_wo = self.relperm_fig.add_subplot(121)
        self.ax_go = self.relperm_fig.add_subplot(122)
        layout.addWidget(self.relperm_canvas)
        return panel

    def _create_pvt_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.pvt_fig = Figure(figsize=(8, 5), tight_layout=True)
        self.pvt_canvas = FigureCanvas(self.pvt_fig)
        self.ax_bo = self.pvt_fig.add_subplot(121)
        self.ax_mu = self.pvt_fig.add_subplot(122)
        layout.addWidget(self.pvt_canvas)
        return panel

    def _create_deliverability_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.ipr_fig = Figure(figsize=(8, 5), tight_layout=True)
        self.ipr_canvas = FigureCanvas(self.ipr_fig)
        self.ax_ipr_prod = self.ipr_fig.add_subplot(121)
        self.ax_ipr_inj = self.ipr_fig.add_subplot(122)
        layout.addWidget(self.ipr_canvas)
        return panel

    def _create_geomechanics_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.geomech_fig = Figure(figsize=(8, 5), tight_layout=True)
        self.geomech_canvas = FigureCanvas(self.geomech_fig)
        self.ax_stress = self.geomech_fig.add_subplot(121)
        self.ax_mohr = self.geomech_fig.add_subplot(122)
        layout.addWidget(self.geomech_canvas)
        return panel

    def update_data(self, project_data: Dict[str, Any]):
        self.project_data = project_data or {}
        try:
            self._render_overview_tab()
            self._render_relperm_tab()
            self._render_pvt_tab()
            self._render_deliverability_tab()
            self._render_geomechanics_tab()
        except Exception as e:
            logger.error(f"Error updating ModelEvaluationDashboard: {e}", exc_info=True)

    def _render_overview_tab(self):
        self.overview_ax_3d.clear()
        res_data = self.project_data.get("reservoir_data")
        manual_inputs = self.project_data.get("manual_inputs", {})
        well_data_list = self.project_data.get("well_data_list", [])

        length_ft = float(getattr(res_data, "length_ft", manual_inputs.get("length", 2000.0)) or 2000.0)
        area_acres = float(getattr(res_data, "area_acres", manual_inputs.get("area", 100.0)) or 100.0)
        thickness_ft = float(getattr(res_data, "thickness_ft", manual_inputs.get("thickness", 50.0)) or 50.0)
        width_ft = (area_acres * 43560.0) / max(length_ft, 1.0)
        top_depth = 5000.0
        if well_data_list and getattr(well_data_list[0], "depths", None) is not None and len(well_data_list[0].depths) > 0:
            top_depth = float(well_data_list[0].depths[0])
        base_depth = top_depth + thickness_ft

        # Wireframe box
        corners = np.array([
            [0, 0, top_depth], [length_ft, 0, top_depth],
            [length_ft, width_ft, top_depth], [0, width_ft, top_depth],
            [0, 0, base_depth], [length_ft, 0, base_depth],
            [length_ft, width_ft, base_depth], [0, width_ft, base_depth]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for e in edges:
            p1, p2 = corners[e[0]], corners[e[1]]
            self.overview_ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="#6c757d", linestyle="--", linewidth=1.0)

        # Wells
        for well in well_data_list:
            is_inj = "inj" in str(getattr(well, "name", "")).lower() or "injector" in str(getattr(well, "metadata", {}).get("type", "")).lower()
            color = "#007bff" if is_inj else "#dc3545"
            marker = "^" if is_inj else "o"
            if getattr(well, "well_path", None) is not None and len(well.well_path) > 0:
                wp = np.asarray(well.well_path)
                wx, wy = wp[:, 0], wp[:, 1]
                wz = wp[:, 2] if wp.shape[1] > 2 else np.linspace(top_depth, base_depth, len(wp))
            else:
                sx = float(getattr(well, "metadata", {}).get("SurfaceX", length_ft * 0.5))
                sy = float(getattr(well, "metadata", {}).get("SurfaceY", width_ft * 0.5))
                wx, wy = np.array([sx, sx]), np.array([sy, sy])
                wz = np.array([top_depth, base_depth])
            self.overview_ax_3d.plot(wx, wy, wz, color=color, linewidth=2.5)
            self.overview_ax_3d.scatter([wx[0]], [wy[0]], [wz[0]], color=color, marker=marker, s=50)
            self.overview_ax_3d.text(wx[0], wy[0], wz[0], f" {well.name}", color=color, fontsize=8, fontweight="bold")

        self.overview_ax_3d.set_title("3D Reservoir & Well Placement", fontsize=10, fontweight="bold")
        self.overview_ax_3d.set_xlabel("X (ft)", fontsize=8)
        self.overview_ax_3d.set_ylabel("Y (ft)", fontsize=8)
        self.overview_ax_3d.set_zlabel("TVD (ft)", fontsize=8)
        self.overview_ax_3d.set_zlim(base_depth + 30.0, top_depth - 30.0)
        self.overview_canvas_3d.draw()

        # Update Summary Table
        ooip = float(getattr(res_data, "ooip_stb", manual_inputs.get("ooip_stb", 1e6)) or 1e6)
        v_dp = float(getattr(res_data, "v_dp_coefficient", manual_inputs.get("v_dp_coefficient", 0.5)) or 0.5)
        mmp_val = self.project_data.get("mmp_value", 2000.0)
        p_res = float(getattr(res_data, "initial_pressure", manual_inputs.get("initial_pressure", 4000.0)) or 4000.0)

        summary_rows = [
            ("1. Reservoir Geometry", "OOIP Volumetrics", f"{ooip:,.0f} STB"),
            ("1. Reservoir Geology", "Dykstra-Parsons (V_DP)", f"{v_dp:.3f}"),
            ("2. PVT Fluid State", "Reservoir Pressure (Pres)", f"{p_res:.0f} psia"),
            ("2. PVT Fluid State", "Minimum Miscibility (MMP)", f"{float(mmp_val):.1f} psia" if mmp_val else "N/A"),
            ("3. Petrophysics", "Mobile Phase Window (1-Swc-Sorw)", f"{1.0 - float(manual_inputs.get('s_wc', 0.2)) - float(manual_inputs.get('s_orw', 0.25)):.2f}"),
            ("4. Well Network", "Total Active Wells", f"{len(well_data_list)} wells"),
            ("5. Geomechanics", "EPA Class VI Ceiling (0.90 Pfrac)", f"{0.90 * p_res * 1.5:.0f} psia"),
        ]

        self.summary_table.setRowCount(len(summary_rows))
        for r, (pillar, metric, val) in enumerate(summary_rows):
            self.summary_table.setItem(r, 0, QTableWidgetItem(pillar))
            self.summary_table.setItem(r, 1, QTableWidgetItem(metric))
            val_item = QTableWidgetItem(val)
            val_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            self.summary_table.setItem(r, 2, val_item)

    def _render_relperm_tab(self):
        self.ax_wo.clear()
        self.ax_go.clear()
        manual_inputs = self.project_data.get("manual_inputs", {})

        s_wc = float(manual_inputs.get("s_wc", 0.20))
        s_orw = float(manual_inputs.get("s_orw", 0.25))
        s_gc = float(manual_inputs.get("s_gc", 0.05))
        n_w = float(manual_inputs.get("n_w", 2.0))
        n_ow = float(manual_inputs.get("n_ow", 2.0))
        n_o = float(manual_inputs.get("n_o", 2.0))
        n_g = float(manual_inputs.get("n_g", 2.0))
        k_ro0 = float(manual_inputs.get("k_ro_0", 0.8))
        k_rg0 = float(manual_inputs.get("k_rg_0", 0.3))
        k_rw0 = float(manual_inputs.get("k_rw_0", 0.3))

        # Water-Oil
        sw = np.linspace(0.0, 1.0, 100)
        denom_wo = max(1.0 - s_wc - s_orw, 1e-4)
        s_wn = np.clip((sw - s_wc) / denom_wo, 0.0, 1.0)
        krw = np.where(sw < s_wc, 0.0, np.where(sw > 1.0 - s_orw, k_rw0, k_rw0 * (s_wn ** n_w)))
        krow = np.where(sw < s_wc, k_ro0, np.where(sw > 1.0 - s_orw, 0.0, k_ro0 * ((1.0 - s_wn) ** n_ow)))

        self.ax_wo.plot(sw, krw, label=r"$k_{rw}$ (Water)", color="#1f77b4", lw=2)
        self.ax_wo.plot(sw, krow, label=r"$k_{row}$ (Oil)", color="#2ca02c", lw=2)
        self.ax_wo.set_title("Water-Oil Relative Permeability", fontsize=10, fontweight="bold")
        self.ax_wo.set_xlabel("Water Saturation ($S_w$)", fontsize=9)
        self.ax_wo.set_ylabel("Relative Permeability", fontsize=9)
        self.ax_wo.grid(True, linestyle=":", alpha=0.6)
        self.ax_wo.legend(fontsize=8)

        # Gas-Oil
        sg = np.linspace(0.0, 1.0, 100)
        denom_go = max(1.0 - s_wc - s_gc, 1e-4)
        s_gn = np.clip((sg - s_gc) / denom_go, 0.0, 1.0)
        krg = np.where(sg < s_gc, 0.0, np.where(sg > 1.0 - s_wc, k_rg0, k_rg0 * (s_gn ** n_g)))
        krog = np.where(sg < s_gc, k_ro0, np.where(sg > 1.0 - s_wc, 0.0, k_ro0 * ((1.0 - s_gn) ** n_o)))

        self.ax_go.plot(sg, krg, label=r"$k_{rg}$ (Gas/CO$_2$)", color="#d62728", lw=2)
        self.ax_go.plot(sg, krog, label=r"$k_{rog}$ (Oil)", color="#17becf", lw=2)
        self.ax_go.set_title("Gas-Oil Relative Permeability", fontsize=10, fontweight="bold")
        self.ax_go.set_xlabel("Gas Saturation ($S_g$)", fontsize=9)
        self.ax_go.set_ylabel("Relative Permeability", fontsize=9)
        self.ax_go.grid(True, linestyle=":", alpha=0.6)
        self.ax_go.legend(fontsize=8)

        self.relperm_canvas.draw()

    def _render_pvt_tab(self):
        self.ax_bo.clear()
        self.ax_mu.clear()
        manual_inputs = self.project_data.get("manual_inputs", {})
        p_res = float(manual_inputs.get("initial_pressure", 4000.0))
        mmp_val = float(self.project_data.get("mmp_value", 2000.0) or 2000.0)
        api = float(manual_inputs.get("api_gravity", 35.0))

        p_range = np.linspace(500.0, 6000.0, 100)
        # Standing Bo approximation
        bo_base = float(manual_inputs.get("boi", 1.2))
        bo_curve = bo_base * (1.0 + 1.2e-5 * (p_range - p_res))

        self.ax_bo.plot(p_range, bo_curve, color="#2ca02c", lw=2, label="$B_o(P)$")
        self.ax_bo.axvline(p_res, color="blue", linestyle="--", label=f"$P_{{res}}={p_res:.0f}$ psia")
        self.ax_bo.axvline(mmp_val, color="red", linestyle=":", label=f"MMP={mmp_val:.0f} psia")
        self.ax_bo.set_title("Oil Formation Volume Factor ($B_o$)", fontsize=10, fontweight="bold")
        self.ax_bo.set_xlabel("Pressure (psia)", fontsize=9)
        self.ax_bo.set_ylabel("Bo (rb/STB)", fontsize=9)
        self.ax_bo.grid(True, linestyle=":", alpha=0.6)
        self.ax_bo.legend(fontsize=8)

        # Viscosity reduction with pressure and dissolved CO2
        mu_dead = float(manual_inputs.get("oil_viscosity_cp", 1.0))
        mu_live = mu_dead * np.exp(-0.0003 * p_range)
        self.ax_mu.plot(p_range, mu_live, color="#ff7f0e", lw=2, label=r"$\mu_o(P)$")
        self.ax_mu.axvline(p_res, color="blue", linestyle="--", label=f"$P_{{res}}={p_res:.0f}$ psia")
        self.ax_mu.axvline(mmp_val, color="red", linestyle=":", label=f"MMP={mmp_val:.0f} psia")
        self.ax_mu.set_title("Live Oil Viscosity Reduction", fontsize=10, fontweight="bold")
        self.ax_mu.set_xlabel("Pressure (psia)", fontsize=9)
        self.ax_mu.set_ylabel("Viscosity (cP)", fontsize=9)
        self.ax_mu.grid(True, linestyle=":", alpha=0.6)
        self.ax_mu.legend(fontsize=8)

        self.pvt_canvas.draw()

    def _render_deliverability_tab(self):
        self.ax_ipr_prod.clear()
        self.ax_ipr_inj.clear()
        manual_inputs = self.project_data.get("manual_inputs", {})
        p_res = float(manual_inputs.get("initial_pressure", 4000.0))
        j_base = 5.0  # nominal PI STB/d/psi

        # Composite Vogel-Darcy Producer IPR
        pwf_prod = np.linspace(500.0, p_res, 100)
        pb = p_res * 0.75  # bubble point estimate
        q_prod = np.zeros_like(pwf_prod)
        for idx, pwf in enumerate(pwf_prod):
            if pwf >= pb:
                q_prod[idx] = j_base * (p_res - pwf)
            else:
                q_b = j_base * (p_res - pb)
                vogel_factor = 1.0 - 0.2 * (pwf / pb) - 0.8 * ((pwf / pb)**2)
                q_prod[idx] = q_b + (j_base * pb / 1.8) * vogel_factor

        self.ax_ipr_prod.plot(q_prod, pwf_prod, color="#dc3545", lw=2.2, label="Producer Vogel-Darcy IPR")
        self.ax_ipr_prod.set_title("Producer Inflow Performance (IPR)", fontsize=10, fontweight="bold")
        self.ax_ipr_prod.set_xlabel("Liquid Production Rate (STB/d)", fontsize=9)
        self.ax_ipr_prod.set_ylabel("Bottomhole Flowing Pressure (psia)", fontsize=9)
        self.ax_ipr_prod.grid(True, linestyle=":", alpha=0.6)
        self.ax_ipr_prod.legend(fontsize=8)

        # Injector Performance
        p_frac = p_res * 1.5
        p_uic = 0.90 * p_frac
        pwf_inj = np.linspace(p_res, p_uic, 100)
        j_inj = 4.0
        q_inj = j_inj * (pwf_inj - p_res)

        self.ax_ipr_inj.plot(q_inj, pwf_inj, color="#007bff", lw=2.2, label="CO2 Injector Line")
        self.ax_ipr_inj.axhline(p_uic, color="red", linestyle="--", label=f"EPA Class VI Limit ({p_uic:.0f} psi)")
        self.ax_ipr_inj.set_title("CO2 Injector Injectivity (II)", fontsize=10, fontweight="bold")
        self.ax_ipr_inj.set_xlabel("Injection Rate (RB/d or MSCF/d)", fontsize=9)
        self.ax_ipr_inj.set_ylabel("Sandface Injection Pressure (psia)", fontsize=9)
        self.ax_ipr_inj.grid(True, linestyle=":", alpha=0.6)
        self.ax_ipr_inj.legend(fontsize=8)

        self.ipr_canvas.draw()

    def _render_geomechanics_tab(self):
        self.ax_stress.clear()
        self.ax_mohr.clear()
        manual_inputs = self.project_data.get("manual_inputs", {})
        p_res = float(manual_inputs.get("initial_pressure", 4000.0))

        # Stress Path Evolution
        dp_array = np.linspace(0.0, 1500.0, 50)
        gamma_h = 0.65  # Poroelastic stress coupling coefficient
        sigma_h = (p_res * 1.2) + gamma_h * dp_array
        p_pore = p_res + dp_array

        self.ax_stress.plot(p_pore, sigma_h, color="#6f42c1", lw=2, label=r"Total $\sigma_{h,min}(P)$")
        self.ax_stress.plot(p_pore, sigma_h - p_pore, color="#17a2b8", linestyle="--", label=r"Effective $\sigma'_{h,min}(P)$")
        self.ax_stress.set_title("Poroelastic Stress Evolution", fontsize=10, fontweight="bold")
        self.ax_stress.set_xlabel("Pore Pressure (psia)", fontsize=9)
        self.ax_stress.set_ylabel("Stress (psi)", fontsize=9)
        self.ax_stress.grid(True, linestyle=":", alpha=0.6)
        self.ax_stress.legend(fontsize=8)

        # Mohr Circle & Failure Envelope
        sigma_1_eff = p_res * 1.4 - p_res
        sigma_3_eff = p_res * 1.1 - p_res
        center = (sigma_1_eff + sigma_3_eff) / 2.0
        radius = (sigma_1_eff - sigma_3_eff) / 2.0

        theta = np.linspace(0, np.pi, 100)
        mohr_x = center + radius * np.cos(theta)
        mohr_y = radius * np.sin(theta)

        # Mohr-Coulomb failure line: tau = S0 + mu * sigma_n
        s0 = 500.0  # cohesion
        mu_f = 0.60  # friction
        sn_line = np.linspace(0, sigma_1_eff * 1.2, 100)
        tau_line = s0 + mu_f * sn_line

        self.ax_mohr.plot(mohr_x, mohr_y, color="#fd7e14", lw=2, label="Effective Mohr Circle")
        self.ax_mohr.plot(sn_line, tau_line, color="#dc3545", linestyle="--", lw=1.8, label="Mohr-Coulomb Failure Envelope")
        self.ax_mohr.set_title("Mohr-Coulomb Fault Stability", fontsize=10, fontweight="bold")
        self.ax_mohr.set_xlabel("Effective Normal Stress $\sigma'_n$ (psi)", fontsize=9)
        self.ax_mohr.set_ylabel("Shear Stress $\tau$ (psi)", fontsize=9)
        self.ax_mohr.grid(True, linestyle=":", alpha=0.6)
        self.ax_mohr.legend(fontsize=8)

        self.geomech_canvas.draw()

    def _launch_audit_dialog(self):
        from ui.dialogs.pre_flight_audit_dialog import PreFlightAuditDialog
        dialog = PreFlightAuditDialog(self.project_data, parent=self)
        dialog.exec()
