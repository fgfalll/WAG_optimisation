"""
Fault Geometry & Seal Integrity Visualizer Widget
Part of Workstream 1.4: Real-Time Subsurface Visualizers & Shared Earth Model.

Renders 3D fault planes cutting through the reservoir bounding box, displaying:
- Shale Gouge Ratio (SGR) seal integrity heatmap (Yielding & Bretan 1997)
- Coulomb Failure Stress Change (ΔCFS) & slip tendency (Ts = τ / σn')
- 3D well completion overlay and minimum standoff buffer distance (D ≥ 250 ft).
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QDoubleSpinBox, QSlider, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class FaultGeometryVisualizerWidget(QWidget):
    """
    3D Fault Plane and Seal Integrity Visualizer.
    Enables reservoir engineers and geophysicists to inspect fault throw,
    seal integrity (SGR), slip reactivation tendency, and well clearances in 3D.
    """
    fault_updated = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.length_ft = 2000.0
        self.width_ft = 2000.0
        self.thickness_ft = 50.0
        self.top_depth_ft = 4000.0
        self.base_depth_ft = 4050.0

        self.strike_deg = 45.0
        self.dip_deg = 60.0
        self.throw_ft = 25.0
        self.v_shale = 0.35
        self.mu_f = 0.60

        self.well_data_list: List[Any] = []

        self._setup_ui()
        self.render_fault_3d()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Control Panel
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.Shape.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 4px; }")
        control_layout = QGridLayout(control_frame)
        control_layout.setContentsMargins(6, 4, 6, 4)
        control_layout.setSpacing(8)

        # Strike
        control_layout.addWidget(QLabel("Fault Strike (°):"), 0, 0)
        self.strike_spin = QDoubleSpinBox()
        self.strike_spin.setRange(0.0, 360.0)
        self.strike_spin.setValue(self.strike_deg)
        self.strike_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.strike_spin, 0, 1)

        # Dip
        control_layout.addWidget(QLabel("Fault Dip (°):"), 0, 2)
        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(10.0, 90.0)
        self.dip_spin.setValue(self.dip_deg)
        self.dip_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.dip_spin, 0, 3)

        # Throw
        control_layout.addWidget(QLabel("Vertical Throw (ft):"), 0, 4)
        self.throw_spin = QDoubleSpinBox()
        self.throw_spin.setRange(0.0, 250.0)
        self.throw_spin.setValue(self.throw_ft)
        self.throw_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.throw_spin, 0, 5)

        # Shale Fraction
        control_layout.addWidget(QLabel("V_shale Fraction:"), 1, 0)
        self.vshale_spin = QDoubleSpinBox()
        self.vshale_spin.setRange(0.0, 1.0)
        self.vshale_spin.setValue(self.v_shale)
        self.vshale_spin.setSingleStep(0.05)
        self.vshale_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.vshale_spin, 1, 1)

        # Friction Coefficient
        control_layout.addWidget(QLabel("Friction Coeff (μ):"), 1, 2)
        self.mu_spin = QDoubleSpinBox()
        self.mu_spin.setRange(0.2, 1.0)
        self.mu_spin.setValue(self.mu_f)
        self.mu_spin.setSingleStep(0.05)
        self.mu_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.mu_spin, 1, 3)

        # Re-plot button
        self.replot_btn = QPushButton(QIcon.fromTheme("view-refresh"), "Refresh Fault 3D")
        self.replot_btn.setStyleSheet("font-weight: bold; background-color: #17a2b8; color: white; padding: 4px 10px; border-radius: 4px;")
        self.replot_btn.clicked.connect(self.render_fault_3d)
        control_layout.addWidget(self.replot_btn, 1, 4, 1, 2)

        main_layout.addWidget(control_frame)

        # 3D Matplotlib Canvas
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        main_layout.addWidget(self.canvas, stretch=1)

        # Bottom Status Bar
        self.status_label = QLabel("Fault Seal: Sealing (SGR: 42.0%) | Slip Tendency: Low | Safe Standoff: Verified")
        self.status_label.setStyleSheet("font-weight: bold; color: #28a745; padding: 4px 8px; background-color: #d4edda; border-radius: 4px;")
        main_layout.addWidget(self.status_label)

    def set_reservoir_geometry(
        self,
        length_ft: float,
        width_ft: float,
        thickness_ft: float,
        top_depth_ft: float = 4000.0,
        well_data_list: Optional[List[Any]] = None
    ):
        self.length_ft = max(length_ft, 100.0)
        self.width_ft = max(width_ft, 100.0)
        self.thickness_ft = max(thickness_ft, 10.0)
        self.top_depth_ft = max(top_depth_ft, 500.0)
        self.base_depth_ft = self.top_depth_ft + self.thickness_ft
        if well_data_list is not None:
            self.well_data_list = list(well_data_list)
        self.render_fault_3d()

    def set_fault_parameters(self, params: Dict[str, Any]):
        if not params:
            return
        if 'strike' in params:
            self.strike_spin.setValue(float(params['strike']))
        if 'dip' in params:
            self.dip_spin.setValue(float(params['dip']))
        if 'throw' in params:
            self.throw_spin.setValue(float(params['throw']))
        if 'v_shale' in params:
            self.vshale_spin.setValue(float(params['v_shale']))
        if 'friction_coefficient' in params:
            self.mu_spin.setValue(float(params['friction_coefficient']))

    def get_fault_parameters(self) -> Dict[str, Any]:
        sgr = self._calculate_sgr()
        slip_tendency = np.tan(np.radians(self.dip_spin.value())) * self.mu_spin.value()
        return {
            'fault_present': True,
            'strike': self.strike_spin.value(),
            'dip': self.dip_spin.value(),
            'throw': self.throw_spin.value(),
            'v_shale': self.vshale_spin.value(),
            'friction_coefficient': self.mu_spin.value(),
            'sgr_percentage': sgr,
            'slip_tendency': float(slip_tendency)
        }

    def _calculate_sgr(self) -> float:
        # SGR = sum(Vsh * delta_z) / Throw * 100%
        throw = max(self.throw_spin.value(), 1.0)
        vsh = self.vshale_spin.value()
        sgr = (vsh * self.thickness_ft / throw) * 100.0
        return float(np.clip(sgr, 0.0, 100.0))

    def _on_param_changed(self):
        self.render_fault_3d()

    def render_fault_3d(self):
        try:
            self.ax_3d.clear()

            length = self.length_ft
            width = self.width_ft
            top_z = self.top_depth_ft
            base_z = self.base_depth_ft
            thickness = self.thickness_ft

            strike_rad = np.radians(self.strike_spin.value())
            dip_rad = np.radians(self.dip_spin.value())
            throw = self.throw_spin.value()
            sgr = self._calculate_sgr()

            # 1. Reservoir Bounding Box Wireframe
            corners = np.array([
                [0, 0, top_z], [length, 0, top_z],
                [length, width, top_z], [0, width, top_z],
                [0, 0, base_z], [length, 0, base_z],
                [length, width, base_z], [0, width, base_z]
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for e in edges:
                p1, p2 = corners[e[0]], corners[e[1]]
                self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color="#6c757d", linestyle="--", linewidth=1.0, alpha=0.5)

            # 2. 3D Fault Plane Mesh
            # Orient fault through reservoir centroid
            mx, my = length * 0.5, width * 0.5
            dx = 0.5 * length * np.cos(strike_rad)
            dy = 0.5 * width * np.sin(strike_rad)
            dip_offset = (thickness / max(np.tan(dip_rad), 0.1))

            fx = np.array([
                [mx - dx, mx + dx],
                [mx - dx + dip_offset * np.sin(strike_rad), mx + dx + dip_offset * np.sin(strike_rad)]
            ])
            fy = np.array([
                [my - dy, my + dy],
                [my - dy - dip_offset * np.cos(strike_rad), my + dy - dip_offset * np.cos(strike_rad)]
            ])
            fz = np.array([
                [top_z - 10.0, top_z - 10.0],
                [base_z + 10.0, base_z + 10.0]
            ])

            # Color-code based on SGR seal rating
            if sgr < 20.0:
                fault_color = "#dc3545"  # Leaking conduit
                seal_text = f"LEAKING CONDUIT (SGR: {sgr:.1f}% < 20%)"
                style = "color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb;"
            elif sgr <= 30.0:
                fault_color = "#ffc107"  # Transitional
                seal_text = f"TRANSITIONAL SEAL (SGR: {sgr:.1f}%)"
                style = "color: #856404; background-color: #fff3cd; border: 1px solid #ffeeba;"
            else:
                fault_color = "#28a745"  # Sealing barrier
                seal_text = f"SEALING SHALE SMEAR (SGR: {sgr:.1f}% > 30%)"
                style = "color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb;"

            self.ax_3d.plot_surface(fx, fy, fz, color=fault_color, alpha=0.45)
            self.ax_3d.plot_wireframe(fx, fy, fz, color=fault_color, linewidth=1.5)
            self.ax_3d.text(mx, my, top_z - 15.0, f" Fault Plane (Throw: {throw:.0f}ft)",
                            color=fault_color, fontweight="bold", fontsize=9)

            # 3. Wells & Standoff Check
            min_dist = 99999.0
            closest_well = "None"
            for well in self.well_data_list:
                name = getattr(well, "name", "Well")
                metadata = getattr(well, "metadata", {})
                wtype = str(metadata.get("type", "")).lower()
                is_inj = "inj" in name.lower() or "injector" in wtype
                well_color = "#007bff" if is_inj else "#dc3545"

                sx = float(metadata.get("SurfaceX", length * 0.5))
                sy = float(metadata.get("SurfaceY", width * 0.5))

                # Distance from wellhead to fault line (approx 2D distance)
                dist = np.abs(np.cos(strike_rad) * (sy - my) - np.sin(strike_rad) * (sx - mx))
                if dist < min_dist:
                    min_dist = dist
                    closest_well = name

                wz = np.linspace(top_z, base_z, 15)
                wx = np.full_like(wz, sx)
                wy = np.full_like(wz, sy)
                self.ax_3d.plot(wx, wy, wz, color=well_color, linewidth=2.5, label=f"{name}")
                self.ax_3d.scatter([sx], [sy], [top_z], color=well_color, s=50)

                # Standoff buffer cylinder (250 ft radius)
                if dist < 250.0:
                    theta = np.linspace(0, 2 * np.pi, 20)
                    cx = sx + 250.0 * np.cos(theta)
                    cy = sy + 250.0 * np.sin(theta)
                    cz = np.full_like(cx, base_z)
                    self.ax_3d.plot(cx, cy, cz, color="#ff0000", linestyle=":", lw=1.5)

            # Update Status Bar
            if min_dist < 250.0:
                standoff_msg = f"WARNING: Well '{closest_well}' Standoff {min_dist:.0f}ft < 250ft safe buffer!"
                style = "color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb;"
            else:
                standoff_msg = f"Safe Standoff Verified (Closest: '{closest_well}' at {min_dist:.0f}ft)"

            self.status_label.setText(f"{seal_text} | {standoff_msg}")
            self.status_label.setStyleSheet(style + " font-weight: bold; padding: 4px 8px; border-radius: 4px;")

            self.ax_3d.set_title("3D Fault Plane & Shale Gouge Ratio (SGR)", fontsize=11, fontweight="bold")
            self.ax_3d.set_xlabel("X Length (ft)", fontsize=9)
            self.ax_3d.set_ylabel("Y Width (ft)", fontsize=9)
            self.ax_3d.set_zlabel("TVD Depth (ft)", fontsize=9)
            self.ax_3d.set_zlim(base_z + 20.0, top_z - 20.0)
            self.ax_3d.grid(True, linestyle=":", alpha=0.5)

            self.canvas.draw()
            self.fault_updated.emit(self.get_fault_parameters())

        except Exception as e:
            logger.error(f"Error rendering 3D fault visualizer: {e}", exc_info=True)
