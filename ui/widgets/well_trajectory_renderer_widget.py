"""
Interactive 3D Well Trajectory & Pattern Renderer Widget
========================================================

High-performance 3D visualization canvas for well architecture, 3D trajectories
(Vertical, Horizontal, Deviated S-Curve), completed perforation intervals,
inter-well sweep connectivity vectors, and pattern geometries.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QFrame,
)
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.data_models import WellData
from core.engine_surrogate.well_mechanics import (
    calculate_vertical_perforation_overlap,
    calculate_interwell_transmissibility,
)


class WellTrajectoryRendererWidget(QWidget):
    """
    Dedicated 3D canvas displaying well trajectories, perforations,
    and inter-well communication vectors.
    """

    wellSelected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.wells: List[WellData] = []
        self.reservoir_params: Dict[str, Any] = {}
        self.highlighted_well_name: Optional[str] = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(4)

        # Toolbar
        toolbar_frame = QFrame()
        toolbar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        toolbar_frame.setStyleSheet("QFrame { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; }")
        tb_layout = QHBoxLayout(toolbar_frame)
        tb_layout.setContentsMargins(6, 4, 6, 4)
        tb_layout.setSpacing(6)

        title_lbl = QLabel(self.tr("3D Trajectories & Pattern:"))
        title_lbl.setStyleSheet("font-weight: bold; color: #1e3d59;")
        tb_layout.addWidget(title_lbl)

        btn_iso = QPushButton("3D Iso")
        btn_iso.setToolTip("Isometric 3D view (elev=28, azim=-55)")
        btn_iso.clicked.connect(lambda: self._set_camera(elev=28, azim=-55))
        tb_layout.addWidget(btn_iso)

        btn_top = QPushButton("Top (XY)")
        btn_top.setToolTip("Map view from above (elev=90, azim=-90)")
        btn_top.clicked.connect(lambda: self._set_camera(elev=90, azim=-90))
        tb_layout.addWidget(btn_top)

        btn_side = QPushButton("Side (XZ)")
        btn_side.setToolTip("Cross-section view (elev=0, azim=0)")
        btn_side.clicked.connect(lambda: self._set_camera(elev=0, azim=0))
        tb_layout.addWidget(btn_side)

        tb_layout.addSpacing(6)

        self.chk_perfs = QCheckBox("Perforations")
        self.chk_perfs.setChecked(True)
        self.chk_perfs.toggled.connect(self.render)
        tb_layout.addWidget(self.chk_perfs)

        self.chk_vectors = QCheckBox("Inter-Well Connectors")
        self.chk_vectors.setChecked(True)
        self.chk_vectors.setToolTip("Show sweep lines colored by vertical perforation overlap")
        self.chk_vectors.toggled.connect(self.render)
        tb_layout.addWidget(self.chk_vectors)

        self.chk_drainage = QCheckBox("Drainage Radii")
        self.chk_drainage.setChecked(True)
        self.chk_drainage.toggled.connect(self.render)
        tb_layout.addWidget(self.chk_drainage)

        tb_layout.addStretch()

        self.btn_refresh = QPushButton(QIcon.fromTheme("view-refresh"), "Refresh")
        self.btn_refresh.clicked.connect(self.render)
        tb_layout.addWidget(self.btn_refresh)

        main_layout.addWidget(toolbar_frame)

        # 3D Matplotlib Canvas
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        main_layout.addWidget(self.canvas, stretch=1)

    def _set_camera(self, elev: float, azim: float):
        if self.ax:
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw_idle()

    def set_wells_data(self, wells: List[WellData], reservoir_params: Optional[Dict[str, Any]] = None):
        """Update wells and reservoir parameters, then re-render."""
        self.wells = list(wells) if wells else []
        self.reservoir_params = dict(reservoir_params) if reservoir_params else {}
        self.render()

    def highlight_well(self, well_name: Optional[str]):
        """Highlight a specific well in the 3D canvas."""
        self.highlighted_well_name = well_name
        self.render()

    def render(self):
        """Perform full 3D rendering of the well trajectories and patterns."""
        self.ax.clear()

        # Geometry dimensions
        res_len = float(self.reservoir_params.get("length_ft", 2000.0) or 2000.0)
        res_width = float(self.reservoir_params.get("width_ft", 1000.0) or 1000.0)
        top_depth = float(self.reservoir_params.get("depth_ft", 1000.0) or 1000.0)
        thick = float(self.reservoir_params.get("net_pay_ft", 100.0) or 100.0)
        base_depth = top_depth + thick

        # Bounding box of reservoir
        corners = np.array([
            [0, 0, top_depth], [res_len, 0, top_depth],
            [res_len, res_width, top_depth], [0, res_width, top_depth],
            [0, 0, base_depth], [res_len, 0, base_depth],
            [res_len, res_width, base_depth], [0, res_width, base_depth]
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for e in edges:
            p1, p2 = corners[e[0]], corners[e[1]]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                         color="#adb5bd", linestyle=":", linewidth=0.9, alpha=0.7)

        # Reservoir top & base surfaces
        xx, yy = np.meshgrid(np.linspace(0, res_len, 6), np.linspace(0, res_width, 6))
        self.ax.plot_surface(xx, yy, np.full_like(xx, top_depth), color="#17a2b8", alpha=0.08)
        self.ax.plot_surface(xx, yy, np.full_like(xx, base_depth), color="#6c757d", alpha=0.08)

        if not self.wells:
            self.ax.text(
                res_len * 0.5, res_width * 0.5, (top_depth + base_depth) * 0.5,
                "No wells loaded\n(Use 'Add Well' to insert well architectures)",
                color="#6c757d", ha="center", va="center", fontsize=9, style="italic"
            )
            self._finalize_axes(top_depth, base_depth, res_len, res_width)
            self.canvas.draw_idle()
            return

        # Separate injectors & producers for connectivity lines
        injectors_pts = []
        producers_pts = []

        z_min_all = top_depth
        z_max_all = base_depth
        n_wells = len(self.wells)
        area_acres = (res_len * res_width) / 43560.0
        drainage_radius = min(np.sqrt((area_acres * 43560.0) / (np.pi * max(n_wells, 1))), res_len * 0.35)

        for well in self.wells:
            w_meta = well.metadata or {}
            w_type = str(w_meta.get("type", "")).lower()
            if not w_type:
                w_type = "injector" if "inj" in well.name.lower() or "injector" in str(w_meta.get("status", "")).lower() else "producer"
            is_inj = "inj" in w_type
            role_tag = "INJ" if is_inj else "PROD"

            traj_type = str(w_meta.get("trajectory_type", "Vertical"))
            is_highlighted = (self.highlighted_well_name == well.name)

            # Trajectory coordinates
            if hasattr(well, "get_trajectory_points"):
                pts = well.get_trajectory_points(top_depth, base_depth, res_len, res_width)
            else:
                pts = np.array([[res_len * 0.5, res_width * 0.5, top_depth], [res_len * 0.5, res_width * 0.5, base_depth]])

            wx, wy, wz = pts[:, 0], pts[:, 1], pts[:, 2]
            z_min_all = min(z_min_all, float(np.min(wz)))
            z_max_all = max(z_max_all, float(np.max(wz)))

            # Color scheme
            if is_inj:
                base_color = "#007bff"  # Vibrant blue
                marker = "^"
            else:
                base_color = "#28a745"  # Forest green
                marker = "o"

            line_width = 4.0 if is_highlighted else 2.6
            line_alpha = 1.0 if is_highlighted else 0.85
            color = "#ff5722" if is_highlighted else base_color

            # 1. Main wellbore trajectory
            self.ax.plot(wx, wy, wz, color=color, linewidth=line_width, alpha=line_alpha, label=f"{well.name} ({role_tag})")

            # 2. Wellhead marker
            self.ax.scatter([wx[0]], [wy[0]], [wz[0]], color=color, s=80, marker=marker, edgecolors="black", linewidths=1.2, zorder=5)

            # 3. Label with trajectory details
            traj_desc = f" ({traj_type})" if "horiz" in traj_type.lower() else ""
            self.ax.text(wx[0], wy[0], wz[0] - (thick * 0.08), f" {well.name}{traj_desc}", color=color, fontsize=8, fontweight="bold")

            # 4. Highlighted perforations
            perfs = getattr(well, "perforations", []) or [
                [p.get("top", 0), p.get("bottom", 0)] for p in getattr(well, "perforation_properties", [])
            ]
            if self.chk_perfs.isChecked() and perfs:
                for p in perfs:
                    if len(p) >= 2:
                        ptop, pbot = min(p[0], p[1]), max(p[0], p[1])
                        perf_mask = (wz >= ptop) & (wz <= pbot)
                        if np.any(perf_mask):
                            self.ax.plot(wx[perf_mask], wy[perf_mask], wz[perf_mask], color="#ffc107", linewidth=6.5, alpha=0.90, zorder=4)

            # 5. Drainage circle
            if self.chk_drainage.isChecked():
                theta = np.linspace(0, 2 * np.pi, 28)
                cx = wx[-1] + drainage_radius * np.cos(theta)
                cy = wy[-1] + drainage_radius * np.sin(theta)
                cz = np.full_like(cx, wz[-1])
                self.ax.plot(cx, cy, cz, color=color, linestyle=":", linewidth=1.1, alpha=0.45)

            # Save sandface center for inter-well vectors
            mid_pt = (float(wx[-1]), float(wy[-1]), float(wz[-1]))
            if is_inj:
                injectors_pts.append((well, mid_pt, perfs))
            else:
                producers_pts.append((well, mid_pt, perfs))

        # 6. Inter-Well Pattern Connectors
        if self.chk_vectors.isChecked() and injectors_pts and producers_pts:
            for inj_well, inj_pt, inj_perfs in injectors_pts:
                for prod_well, prod_pt, prod_perfs in producers_pts:
                    # Calculate overlap ratio
                    _, omega = calculate_vertical_perforation_overlap(inj_perfs, prod_perfs)
                    
                    if omega >= 0.20:
                        vec_color = "#28a745"  # Green: valid sweep
                        vec_style = "--"
                        vec_width = 1.6
                    elif omega > 0:
                        vec_color = "#fd7e14"  # Orange: low overlap warning
                        vec_style = "-."
                        vec_width = 1.3
                    else:
                        vec_color = "#dc3545"  # Red: 0% overlap
                        vec_style = ":"
                        vec_width = 1.0

                    self.ax.plot(
                        [inj_pt[0], prod_pt[0]],
                        [inj_pt[1], prod_pt[1]],
                        [inj_pt[2], prod_pt[2]],
                        color=vec_color,
                        linestyle=vec_style,
                        linewidth=vec_width,
                        alpha=0.65,
                    )

        self._finalize_axes(z_min_all, z_max_all, res_len, res_width)
        self.canvas.draw_idle()

    def _finalize_axes(self, z_min: float, z_max: float, res_len: float, res_width: float):
        self.ax.set_title("3D Well Trajectories & Pattern Sweep", fontsize=10, fontweight="bold", pad=8)
        self.ax.set_xlabel("X Length (ft)", fontsize=8, labelpad=2)
        self.ax.set_ylabel("Y Width (ft)", fontsize=8, labelpad=2)
        self.ax.set_zlabel("TVD Depth (ft)", fontsize=8, labelpad=2)
        self.ax.tick_params(labelsize=7)

        # Standard reservoir engineering convention: TVD depth increases downward
        pad_z = max((z_max - z_min) * 0.15, 10.0)
        self.ax.set_zlim(z_max + pad_z, max(0.0, z_min - pad_z))
        self.ax.set_xlim(0, max(res_len, 100.0))
        self.ax.set_ylim(0, max(res_width, 100.0))
        self.ax.grid(True, linestyle=":", alpha=0.5)
