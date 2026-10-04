"""
Geostatistics Visualizer Widget
Part of Subsurface Studio Workbench.

Renders high-resolution 1D semivariogram model curves against physical experimental lags
and produces fast 2D/3D spatial Sequential Gaussian Simulation (SGS) realizations with
petrophysical histograms, Dykstra-Parsons V_DP, and well location overlays.
Features rich plotting controls in the top toolbar and a right-click context menu.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QFrame, QMenu,
    QFileDialog, QApplication, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QAction, QCursor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from core.geology.geostatistical_modeling import (
    create_geostatistical_grid, calculate_variogram, theoretical_variogram
)

logger = logging.getLogger(__name__)


class GeostatisticsVisualizerWidget(QWidget):
    """
    Interactive Geostatistics and Spatial Heterogeneity Visualizer.
    Top bar provides clean plotting settings and visualization utilities,
    while physical reservoir parameters are managed via the contextual side panel.
    """
    realization_updated = pyqtSignal(dict)
    regenerate_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.nx = 50
        self.ny = 50
        self.nz = 10
        self.length_ft = 2000.0
        self.width_ft = 2000.0
        self.thickness_ft = 50.0
        self.base_perm = 100.0
        self.base_poro = 0.20
        self.well_data_list: List[Any] = []

        # Model Parameters (Synced from Side Panel)
        self.variogram_type = "spherical"
        self.range_major = 1200.0
        self.range_minor = 600.0
        self.range_vert = 20.0
        self.sill_val = 1.00
        self.nugget_val = 0.05
        self.aniso_ratio = 2.0
        self.azimuth_deg = 45.0
        self.random_seed = 42
        self.current_v_dp = 0.65

        # Generated Fields
        self.current_k_field: Optional[np.ndarray] = None
        self.current_phi_field: Optional[np.ndarray] = None
        self.show_colorbar = True

        self._setup_ui()
        self.generate_realization()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 1. Clean Non-Squished Top Toolbar (Plotting Settings & Current Plot Utilities)
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
            QComboBox, QSpinBox {
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
            "Spatial Realization & Semivariogram",
            "Directional Variogram Rose & Anisotropy Ellipse",
            "Realization & Permeability Histogram (QC)",
            "3D Layer Z Slices"
        ])
        self.combo_mode.currentIndexChanged.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.combo_mode)

        # Property Selector
        top_layout.addWidget(QLabel("Property:"))
        self.combo_prop = QComboBox()
        self.combo_prop.addItems(["Permeability (PERMX)", "Porosity (PORO)"])
        self.combo_prop.currentIndexChanged.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.combo_prop)

        # Colormap Palette
        top_layout.addWidget(QLabel("Palette:"))
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(["turbo", "viridis", "plasma", "coolwarm", "jet"])
        self.combo_palette.currentIndexChanged.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.combo_palette)

        # Layer Z Selector
        self.lbl_layer = QLabel("Layer Z:")
        top_layout.addWidget(self.lbl_layer)
        self.spin_layer = QSpinBox()
        self.spin_layer.setRange(1, max(self.nz, 1))
        self.spin_layer.setValue(1)
        self.spin_layer.valueChanged.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.spin_layer)

        # Scale
        top_layout.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Logarithmic", "Linear"])
        self.combo_scale.currentIndexChanged.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.combo_scale)

        # Checkboxes
        self.chk_wells = QCheckBox("Wells")
        self.chk_wells.setChecked(True)
        self.chk_wells.toggled.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.chk_wells)

        self.chk_grid = QCheckBox("Grid")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(lambda _: self._render_plots())
        top_layout.addWidget(self.chk_grid)

        top_layout.addStretch()

        # Action Buttons
        self.btn_export = QPushButton("💾 Export...")
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
        self.btn_export.clicked.connect(self._save_plot_image)
        top_layout.addWidget(self.btn_export)

        self.btn_regen = QPushButton("🔄 Regenerate Realization")
        self.btn_regen.setStyleSheet("""
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
        self.btn_regen.clicked.connect(self.generate_realization)
        top_layout.addWidget(self.btn_regen)

        main_layout.addWidget(top_frame)

        # 2. Main High-Resolution Plot Canvas
        self.fig = Figure(figsize=(10, 5), facecolor="#ffffff", tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        main_layout.addWidget(self.canvas, stretch=1)

        # Default axes for compatibility with external references
        self.ax_variogram = self.fig.add_subplot(1, 2, 1)
        self.ax_realization = self.fig.add_subplot(1, 2, 2)

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

    def set_grid_dimensions(self, nx: int, ny: int, nz: int = 10,
                            length_ft: float = 2000.0, width_ft: float = 2000.0, thickness_ft: float = 50.0,
                            base_perm: float = 100.0, base_poro: float = 0.20):
        self.nx = max(int(nx), 5)
        self.ny = max(int(ny), 5)
        self.nz = max(int(nz), 1)
        self.length_ft = max(float(length_ft), 10.0)
        self.width_ft = max(float(width_ft), 10.0)
        self.thickness_ft = max(float(thickness_ft), 1.0)
        self.base_perm = max(float(base_perm), 0.01)
        self.base_poro = max(float(base_poro), 0.01)
        self.spin_layer.setRange(1, self.nz)

    def set_well_data(self, well_data_list: List[Any]):
        self.well_data_list = list(well_data_list)

    def set_parameters(self, params: Dict[str, Any]):
        """Synchronizes parameters from side panel or active project."""
        if not params:
            return
        if 'variogram_type' in params:
            self.variogram_type = str(params['variogram_type']).lower()
        if 'variogram_range_major' in params:
            self.range_major = float(params['variogram_range_major'])
        elif 'range' in params and params['range'] is not None:
            self.range_major = float(params['range'])

        if 'variogram_range_minor' in params:
            self.range_minor = float(params['variogram_range_minor'])
        if 'variogram_range_vert' in params:
            self.range_vert = float(params['variogram_range_vert'])
        if 'variogram_azimuth_deg' in params:
            self.azimuth_deg = float(params['variogram_azimuth_deg'])

        if 'sill_variance' in params:
            self.sill_val = float(params['sill_variance'])
        elif 'sill' in params and params['sill'] is not None:
            self.sill_val = float(params['sill'])

        if 'nugget_effect' in params:
            self.nugget_val = float(params['nugget_effect'])
        elif 'nugget' in params and params['nugget'] is not None:
            self.nugget_val = float(params['nugget'])

        self.aniso_ratio = self.range_major / max(self.range_minor, 1.0)

        if 'geostat_seed' in params:
            self.random_seed = int(params['geostat_seed'])
        elif 'random_seed' in params and params['random_seed'] is not None:
            self.random_seed = int(params['random_seed'])

        if 'length' in params:
            self.length_ft = float(params['length'])
        if 'area' in params and 'length' in params:
            area = float(params['area'])
            self.width_ft = (area * 43560.0) / max(self.length_ft, 1.0)
        if 'nx' in params:
            self.nx = int(params['nx'])
        if 'ny' in params:
            self.ny = int(params['ny'])
        if 'nz' in params:
            self.nz = int(params['nz'])
            self.spin_layer.setRange(1, self.nz)
        if 'perm' in params:
            self.base_perm = float(params['perm'])
        if 'poro' in params:
            self.base_poro = float(params['poro'])

        self.generate_realization()

    def generate_realization(self):
        """Generates 2D/3D geostatistical field using SGS with physical scale matching."""
        try:
            nx, ny = self.nx, self.ny
            dx = self.length_ft / max(nx, 1)

            # Effective range in cell units for the 2D solver
            range_cells = max(self.range_major / dx, 1.5)

            norm_field = create_geostatistical_grid((nx, ny), {
                'variogram_type': self.variogram_type,
                'range': range_cells,
                'sill': self.sill_val,
                'nugget': self.nugget_val,
                'anisotropy_ratio': self.aniso_ratio,
                'random_seed': self.random_seed,
            })

            # Permeability field (log-normal distribution centered on base_perm)
            k_field = self.base_perm * np.exp(1.8 * (norm_field - 0.5))
            self.current_k_field = np.clip(k_field, 0.001, 10000.0)

            # Porosity field (correlated to permeability via Kozeny-Carman)
            phi_field = self.base_poro * (self.current_k_field / max(self.base_perm, 1e-4)) ** 0.25
            self.current_phi_field = np.clip(phi_field, 0.01, 0.40)

            # Calculate Dykstra-Parsons V_DP
            k_flat = np.sort(self.current_k_field.flatten())
            k_50 = float(np.percentile(k_flat, 50))
            k_84_1 = float(np.percentile(k_flat, 15.9))
            self.current_v_dp = float(np.clip((k_50 - k_84_1) / max(k_50, 1e-4), 0.0, 0.95))

            self._render_plots()

            self.realization_updated.emit({
                'variogram_type': self.variogram_type,
                'range_major': self.range_major,
                'range_minor': self.range_minor,
                'sill': self.sill_val,
                'nugget': self.nugget_val,
                'anisotropy_ratio': self.aniso_ratio,
                'random_seed': self.random_seed,
                'v_dp': self.current_v_dp
            })

        except Exception as e:
            logger.error(f"Error generating geostatistics realization: {e}", exc_info=True)

    def _render_plots(self):
        """Renders middle-screen dual/triple plots according to active display mode."""
        if self.current_k_field is None:
            return

        mode_idx = self.combo_mode.currentIndex()
        prop_str = self.combo_prop.currentText()
        palette = self.combo_palette.currentText()
        is_log = (self.combo_scale.currentIndex() == 0)
        show_wells = self.chk_wells.isChecked()
        show_grid = self.chk_grid.isChecked()

        self.fig.clear()

        # Active scalar field
        is_perm = "Permeability" in prop_str
        active_field = self.current_k_field if is_perm else self.current_phi_field
        unit_str = "mD" if is_perm else "fraction"
        prop_title = "Permeability" if is_perm else "Porosity"

        nx, ny = self.nx, self.ny
        dx = self.length_ft / max(nx, 1)
        dy = self.width_ft / max(ny, 1)

        v_dp = self.current_v_dp
        koval_hk = 10.0 ** (v_dp / max(1.0 - v_dp, 0.05))
        mean_val = float(np.mean(active_field))

        if mode_idx == 0:
            # --- MODE 0: Spatial Realization & Semivariogram ---
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            self.ax_variogram = ax1
            self.ax_realization = ax2

            # Left Plot: Experimental vs Fitted Semivariogram Model (Physical Scale in Feet!)
            norm_field = (active_field - np.mean(active_field)) / max(np.std(active_field), 1e-4)
            lags_cells, exp_gamma_norm = calculate_variogram(norm_field)
            lags_ft = lags_cells * dx
            exp_gamma = exp_gamma_norm * (self.sill_val + self.nugget_val)

            # Physical Theoretical Curve in Feet
            max_h = max(float(np.max(lags_ft)), self.range_major * 1.5, 100.0)
            h_ft = np.linspace(0.1, max_h, 150)
            theo_gamma = theoretical_variogram(h_ft, self.variogram_type, self.sill_val, self.range_major, self.nugget_val)

            ax1.set_facecolor("#f8fafc")
            ax1.scatter(lags_ft, exp_gamma, color="#0f172a", s=35, zorder=5, label=r"Experimental $\hat{\gamma}(h)$")
            ax1.plot(h_ft, theo_gamma, color="#0d6efd", lw=2.5, zorder=4,
                     label=f"Fit ({self.variogram_type.capitalize()} a={self.range_major:.0f} ft)")

            total_sill = self.sill_val + self.nugget_val
            ax1.axhline(total_sill, color="#dc2626", linestyle="--", lw=1.6, label=f"Sill ({total_sill:.2f})")
            ax1.axhline(self.nugget_val, color="#d97706", linestyle=":", lw=1.4, label=f"Nugget c0 ({self.nugget_val:.2f})")
            ax1.axvline(self.range_major, color="#16a34a", linestyle="--", lw=1.4, label=f"Major Range a={self.range_major:.0f} ft")

            ax1.set_title("Experimental vs Model Semivariogram", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Lag Distance h (ft)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_ylabel(r"Semivariance $\gamma(h)$", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_xlim(0.0, max_h)
            ax1.set_ylim(-0.02, total_sill * 1.25)
            ax1.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax1.legend(fontsize=8, loc="lower right")

            # Right Plot: Spatial SGS Realization Heatmap
            ax2.set_facecolor("#f8fafc")
            extent = [0, self.length_ft, 0, self.width_ft]

            if is_perm and is_log:
                from matplotlib.colors import LogNorm
                norm = LogNorm(vmin=max(float(np.min(active_field)), 0.1), vmax=float(np.max(active_field)))
                im = ax2.imshow(active_field.T, origin="lower", extent=extent, cmap=palette, norm=norm, aspect="auto")
            else:
                im = ax2.imshow(active_field.T, origin="lower", extent=extent, cmap=palette, aspect="auto")

            if self.show_colorbar:
                cbar = self.fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                cbar.set_label(f"{prop_title} ({unit_str})", fontsize=8.5, fontweight="bold")
                cbar.ax.tick_params(labelsize=8)

            # Well overlays
            if show_wells and self.well_data_list:
                for well in self.well_data_list:
                    name = getattr(well, "name", "Well")
                    metadata = getattr(well, "metadata", {})
                    wtype = str(metadata.get("type", "")).lower()
                    is_inj = "inj" in name.lower() or "injector" in wtype
                    color = "#00ffff" if is_inj else "#ff0055"
                    marker = "^" if is_inj else "o"
                    sx = float(metadata.get("SurfaceX", self.length_ft * 0.5))
                    sy = float(metadata.get("SurfaceY", self.width_ft * 0.5))
                    ax2.scatter(sx, sy, color=color, s=90, marker=marker, edgecolors="#ffffff", linewidths=1.5, zorder=6)
                    ax2.text(sx + self.length_ft * 0.02, sy + self.width_ft * 0.02, name,
                             color="#ffffff", fontsize=8.5, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.6, edgecolor="none"))

            # Directional Strike Rose / Azimuth Arrow in Corner
            rad = np.radians(90.0 - self.azimuth_deg)
            arrow_len = self.length_ft * 0.12
            cx_ar = self.length_ft * 0.88
            cy_ar = self.width_ft * 0.15
            dx_ar = arrow_len * np.cos(rad)
            dy_ar = arrow_len * np.sin(rad)
            ax2.annotate("", xy=(cx_ar + dx_ar, cy_ar + dy_ar), xytext=(cx_ar - dx_ar, cy_ar - dy_ar),
                         arrowprops=dict(arrowstyle="<->", color="#ffffff", lw=2.2))
            ax2.text(cx_ar, cy_ar - self.width_ft * 0.06, f"Strike {self.azimuth_deg:.0f}°",
                     color="#ffffff", fontsize=7.5, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.7, edgecolor="none"))

            ax2.set_title(f"SGS {prop_title} Realization ({nx}x{ny} cells)", fontsize=11, fontweight="bold", color="#0f172a")
            ax2.set_xlabel("Reservoir X (ft)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_ylabel("Reservoir Y (ft)", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.grid(show_grid, linestyle=":", alpha=0.4, color="#ffffff")

            self.metrics_label.setText(
                f"SGS Realization: {self.variogram_type.capitalize()} Model | Range a_maj: {self.range_major:.0f} ft (a_min: {self.range_minor:.0f} ft) | "
                f"Anisotropy: {self.aniso_ratio:.1f}:1.0 @ {self.azimuth_deg:.0f}° | "
                f"Dykstra-Parsons V_DP: {v_dp:.3f} | Koval H_K: {koval_hk:.2f} | Mean: {mean_val:.1f} {unit_str}"
            )

        elif mode_idx == 1:
            # --- MODE 1: Directional Variogram Rose & Anisotropy Ellipse ---
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122, projection="polar")
            self.ax_variogram = ax1
            self.ax_realization = ax2

            # Left: Directional Variograms (Major Strike vs Minor Dip)
            max_h = max(self.range_major * 1.5, 100.0)
            h_ft = np.linspace(0.1, max_h, 150)
            theo_major = theoretical_variogram(h_ft, self.variogram_type, self.sill_val, self.range_major, self.nugget_val)
            theo_minor = theoretical_variogram(h_ft, self.variogram_type, self.sill_val, self.range_minor, self.nugget_val)

            ax1.set_facecolor("#f8fafc")
            ax1.plot(h_ft, theo_major, color="#0d6efd", lw=2.5, label=f"Major Axis (Strike {self.azimuth_deg:.0f}°, a={self.range_major:.0f} ft)")
            ax1.plot(h_ft, theo_minor, color="#dc2626", lw=2.5, linestyle="--", label=f"Minor Axis (Dip {self.azimuth_deg+90:.0f}°, a={self.range_minor:.0f} ft)")
            ax1.axhline(self.sill_val + self.nugget_val, color="#64748b", linestyle=":", lw=1.5, label=f"Sill ({self.sill_val + self.nugget_val:.2f})")

            ax1.set_title("Directional Semivariogram Continuity", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Lag Distance h (ft)", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_ylabel(r"Semivariance $\gamma(h)$", fontsize=9.5, fontweight="bold", color="#334155")
            ax1.set_xlim(0.0, max_h)
            ax1.set_ylim(-0.02, (self.sill_val + self.nugget_val) * 1.25)
            ax1.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax1.legend(fontsize=8, loc="lower right")

            # Right: Polar Anisotropy Range Ellipse
            theta_pol = np.linspace(0, 2 * np.pi, 200)
            rot = np.radians(90.0 - self.azimuth_deg)
            # Parametric ellipse
            x_el = self.range_major * np.cos(theta_pol)
            y_el = self.range_minor * np.sin(theta_pol)
            # Rotate by azimuth
            x_rot = x_el * np.cos(rot) - y_el * np.sin(rot)
            y_rot = x_el * np.sin(rot) + y_el * np.cos(rot)
            r_pol = np.sqrt(x_rot**2 + y_rot**2)
            ang_pol = np.arctan2(y_rot, x_rot)

            ax2.plot(ang_pol, r_pol, color="#0d6efd", lw=2.5, label=f"Anisotropy Envelope ({self.aniso_ratio:.1f}:1.0)")
            ax2.fill(ang_pol, r_pol, color="#bae6fd", alpha=0.4)
            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_title(f"Spatial Continuity Range Rose (Strike {self.azimuth_deg:.0f}°)", fontsize=11, fontweight="bold", color="#0f172a")
            ax2.grid(show_grid, linestyle=":", alpha=0.6)
            ax2.legend(fontsize=8, loc="upper right")

            self.metrics_label.setText(
                f"Directional Anisotropy: Major Range {self.range_major:.0f} ft @ {self.azimuth_deg:.0f}° | "
                f"Minor Range {self.range_minor:.0f} ft @ {self.azimuth_deg+90:.0f}° | Ratio: {self.aniso_ratio:.2f}"
            )

        elif mode_idx == 2:
            # --- MODE 2: Realization & Permeability Histogram QC ---
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            self.ax_realization = ax1
            self.ax_variogram = ax2

            # Left: Realization
            extent = [0, self.length_ft, 0, self.width_ft]
            im = ax1.imshow(active_field.T, origin="lower", extent=extent, cmap=palette, aspect="auto")
            if self.show_colorbar:
                cbar = self.fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar.set_label(f"{prop_title} ({unit_str})", fontsize=8.5, fontweight="bold")
            ax1.set_title(f"SGS {prop_title} Realization", fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Reservoir X (ft)", fontsize=9.5, fontweight="bold")
            ax1.set_ylabel("Reservoir Y (ft)", fontsize=9.5, fontweight="bold")

            # Right: Statistical Histogram & CDF
            flat_data = active_field.flatten()
            p10 = float(np.percentile(flat_data, 10))
            p50 = float(np.percentile(flat_data, 50))
            p90 = float(np.percentile(flat_data, 90))

            ax2.set_facecolor("#f8fafc")
            counts, bins, _ = ax2.hist(flat_data, bins=35, density=True, color="#0d6efd", alpha=0.65, edgecolor="#ffffff", label="Realization Sample Distribution")

            # PDF line
            ax2.axvline(p50, color="#16a34a", linestyle="-", lw=2.0, label=f"P50 Median = {p50:.1f} {unit_str}")
            ax2.axvline(p10, color="#dc2626", linestyle="--", lw=1.5, label=f"P10 Low = {p10:.1f} {unit_str}")
            ax2.axvline(p90, color="#d97706", linestyle="--", lw=1.5, label=f"P90 High = {p90:.1f} {unit_str}")
            ax2.axvline(mean_val, color="#7c3aed", linestyle=":", lw=1.8, label=f"Mean = {mean_val:.1f} {unit_str}")

            ax2.set_title(f"Petrophysical QC Distribution ({prop_title})", fontsize=11, fontweight="bold", color="#0f172a")
            ax2.set_xlabel(f"{prop_title} ({unit_str})", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.set_ylabel("Probability Density", fontsize=9.5, fontweight="bold", color="#334155")
            ax2.grid(show_grid, linestyle=":", alpha=0.6, color="#cbd5e1")
            ax2.legend(fontsize=8, loc="upper right")

            self.metrics_label.setText(
                f"QC Statistics: Mean {mean_val:.1f} {unit_str} | P10: {p10:.1f} | P50: {p50:.1f} | P90: {p90:.1f} | "
                f"Std Dev: {float(np.std(flat_data)):.2f} | Dykstra-Parsons V_DP: {v_dp:.3f}"
            )

        else:
            # --- MODE 3: 3D Layer Z Slice ---
            ax1 = self.fig.add_subplot(111)
            self.ax_realization = ax1
            self.ax_variogram = ax1
            layer_idx = self.spin_layer.value()
            dz = self.thickness_ft / max(self.nz, 1)
            layer_z = (layer_idx - 0.5) * dz

            extent = [0, self.length_ft, 0, self.width_ft]
            im = ax1.imshow(active_field.T, origin="lower", extent=extent, cmap=palette, aspect="auto")
            if self.show_colorbar:
                cbar = self.fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                cbar.set_label(f"{prop_title} ({unit_str})", fontsize=8.5, fontweight="bold")

            ax1.set_title(f"Geostatistical Slice - Layer {layer_idx} of {self.nz} (Depth Offset: {layer_z:.1f} ft)",
                          fontsize=11, fontweight="bold", color="#0f172a")
            ax1.set_xlabel("Reservoir X (ft)", fontsize=9.5, fontweight="bold")
            ax1.set_ylabel("Reservoir Y (ft)", fontsize=9.5, fontweight="bold")
            ax1.grid(show_grid, linestyle=":", alpha=0.4, color="#ffffff")

            self.metrics_label.setText(
                f"3D Slicing: Layer {layer_idx}/{self.nz} | ΔZ: {dz:.1f} ft | "
                f"Slice Mean {prop_title}: {mean_val:.1f} {unit_str} | Dykstra-Parsons V_DP: {v_dp:.3f}"
            )

        self.canvas.draw()

    def _show_context_menu(self, pos: QPoint):
        """Right-click context menu for geostatistical plots."""
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
        act_csv = menu.addAction("📊 Export Realization Grid (CSV)...")
        menu.addSeparator()

        act_cbar = menu.addAction("🎨 Toggle Colorbar")
        act_grid = menu.addAction("▦ Toggle Grid")
        menu.addSeparator()

        act_regen = menu.addAction("🔄 Regenerate Realization")
        act_reset = menu.addAction("🔍 Reset Default View")

        action = menu.exec(self.canvas.mapToGlobal(pos))
        if action == act_copy:
            self._copy_plot_to_clipboard()
        elif action == act_save:
            self._save_plot_image()
        elif action == act_csv:
            self._export_grid_csv()
        elif action == act_cbar:
            self.show_colorbar = not self.show_colorbar
            self._render_plots()
        elif action == act_grid:
            self.chk_grid.setChecked(not self.chk_grid.isChecked())
        elif action == act_regen:
            self.generate_realization()
        elif action == act_reset:
            self.combo_mode.setCurrentIndex(0)
            self._render_plots()

    def _copy_plot_to_clipboard(self):
        pixmap = self.canvas.grab()
        QApplication.clipboard().setPixmap(pixmap)
        logger.info("Copied geostatistics plot to clipboard")

    def _save_plot_image(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Geostatistics Figure", "geostatistics_realization.png",
            "PNG Images (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        if filepath:
            self.fig.savefig(filepath, dpi=300, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
            logger.info(f"Saved geostatistics figure to {filepath}")

    def _export_grid_csv(self):
        if self.current_k_field is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Realization Grid", "geostat_realization_grid.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return
        np.savetxt(filepath, self.current_k_field, delimiter=",", fmt="%.4f")
        logger.info(f"Exported geostatistical grid to {filepath}")
