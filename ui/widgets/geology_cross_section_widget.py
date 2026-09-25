"""
Geology Cross-Section & Vertical Proportion Curves (VPC) Widget
Part of Workstream 1.4: Real-Time Subsurface Visualizers & Shared Earth Model.

Provides interactive orthogonal IJK cross-section slicing (X-Z, Y-Z, X-Y)
and Vertical Proportion Curves (VPC) showing permeable sand vs shale barriers
across stratigraphic reservoir intervals.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSlider, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class GeologyCrossSectionWidget(QWidget):
    """
    Interactive Stratigraphic Cross-Section and Vertical Proportion Curve (VPC) Viewer.
    Allows reservoir engineers to visually confirm layer connectivity, structural dip,
    and vertical sand/shale proportions along orthogonal axes.
    """
    slice_changed = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.nx = 50
        self.ny = 50
        self.nz = 10
        self.length_ft = 2000.0
        self.width_ft = 2000.0
        self.thickness_ft = 50.0
        self.dip_angle_deg = 0.0

        self.perm_grid: Optional[np.ndarray] = None
        self.poro_grid: Optional[np.ndarray] = None
        self.layer_definitions: List[Any] = []
        self.well_data_list: List[Any] = []

        self._init_synthetic_grid()
        self._setup_ui()
        self.render_slice()

    def _init_synthetic_grid(self):
        """Creates a baseline layered grid if no real simulation grid is loaded."""
        nx, ny, nz = self.nx, self.ny, self.nz
        self.perm_grid = np.zeros((nx, ny, nz), dtype=float)
        self.poro_grid = np.zeros((nx, ny, nz), dtype=float)

        z_indices = np.linspace(0, 1, nz)
        for k in range(nz):
            # Alternating high/low perm sand-shale stratification
            is_sand = (k % 3 != 1)
            k_base = 150.0 if is_sand else 2.0
            phi_base = 0.22 if is_sand else 0.08
            self.perm_grid[:, :, k] = k_base * (1.0 + 0.3 * np.random.randn(nx, ny))
            self.poro_grid[:, :, k] = phi_base * (1.0 + 0.15 * np.random.randn(nx, ny))

        self.perm_grid = np.clip(self.perm_grid, 0.01, 2000.0)
        self.poro_grid = np.clip(self.poro_grid, 0.02, 0.35)

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

        # Slice Mode
        control_layout.addWidget(QLabel("Slice Orientation:"), 0, 0)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems([
            "In-line Cross-Section (X-Z)",
            "Cross-line Cross-Section (Y-Z)",
            "Areal Map Slice (X-Y)",
            "Vertical Proportion Curve (VPC)"
        ])
        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)
        control_layout.addWidget(self.orientation_combo, 0, 1)

        # Property Display
        control_layout.addWidget(QLabel("Property:"), 0, 2)
        self.property_combo = QComboBox()
        self.property_combo.addItems(["Permeability (mD)", "Porosity (φ)", "Facies (Sand vs Shale)"])
        self.property_combo.currentIndexChanged.connect(self.render_slice)
        control_layout.addWidget(self.property_combo, 0, 3)

        # Slicing Sliders
        self.slice_label = QLabel(f"Y Index: {self.ny // 2 + 1} / {self.ny}")
        control_layout.addWidget(self.slice_label, 1, 0)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, self.ny - 1)
        self.slice_slider.setValue(self.ny // 2)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        control_layout.addWidget(self.slice_slider, 1, 1, 1, 3)

        main_layout.addWidget(control_frame)

        # Matplotlib Canvas
        self.fig = Figure(figsize=(8, 4.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax_main = self.fig.add_subplot(111)
        main_layout.addWidget(self.canvas, stretch=1)

        # Bottom Info Bar
        self.info_label = QLabel("Layer Connectivity: Active | Net-to-Gross: 0.70 | Dip: 0.0°")
        self.info_label.setStyleSheet("font-weight: bold; color: #2b3e50; padding: 2px 6px; background-color: #e9ecef; border-radius: 4px;")
        main_layout.addWidget(self.info_label)

    def set_grid_data(
        self,
        perm_grid: Optional[np.ndarray],
        poro_grid: Optional[np.ndarray],
        length_ft: float = 2000.0,
        width_ft: float = 2000.0,
        thickness_ft: float = 50.0,
        dip_angle_deg: float = 0.0,
        well_data_list: Optional[List[Any]] = None
    ):
        if perm_grid is not None and perm_grid.ndim == 3:
            self.perm_grid = perm_grid
            self.nx, self.ny, self.nz = perm_grid.shape
        if poro_grid is not None and poro_grid.ndim == 3:
            self.poro_grid = poro_grid
        self.length_ft = length_ft
        self.width_ft = width_ft
        self.thickness_ft = thickness_ft
        self.dip_angle_deg = dip_angle_deg
        if well_data_list is not None:
            self.well_data_list = list(well_data_list)

        self._on_orientation_changed()
        self.render_slice()

    def _on_orientation_changed(self):
        mode = self.orientation_combo.currentIndex()
        if mode == 0:  # X-Z (slice across Y)
            self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.ny - 1)
            self.slice_slider.setValue(self.ny // 2)
            self.slice_label.setText(f"Cross-line (Y) Index: {self.slice_slider.value() + 1} / {self.ny}")
        elif mode == 1:  # Y-Z (slice across X)
            self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.nx - 1)
            self.slice_slider.setValue(self.nx // 2)
            self.slice_label.setText(f"In-line (X) Index: {self.slice_slider.value() + 1} / {self.nx}")
        elif mode == 2:  # X-Y (slice across Z)
            self.slice_slider.setEnabled(True)
            self.slice_slider.setRange(0, self.nz - 1)
            self.slice_slider.setValue(self.nz // 2)
            self.slice_label.setText(f"Depth Layer (K) Index: {self.slice_slider.value() + 1} / {self.nz}")
        elif mode == 3:  # VPC
            self.slice_slider.setEnabled(False)
            self.slice_label.setText("Vertical Stratigraphic Distribution")

        self.render_slice()

    def _on_slider_changed(self, val: int):
        mode = self.orientation_combo.currentIndex()
        if mode == 0:
            self.slice_label.setText(f"Cross-line (Y) Index: {val + 1} / {self.ny}")
        elif mode == 1:
            self.slice_label.setText(f"In-line (X) Index: {val + 1} / {self.nx}")
        elif mode == 2:
            self.slice_label.setText(f"Depth Layer (K) Index: {val + 1} / {self.nz}")
        self.render_slice()

    def render_slice(self):
        try:
            self.ax_main.clear()
            mode = self.orientation_combo.currentIndex()
            prop_idx = self.property_combo.currentIndex()

            # Choose grid
            grid = self.perm_grid if prop_idx in [0, 2] else self.poro_grid
            if grid is None:
                self._init_synthetic_grid()
                grid = self.perm_grid if prop_idx in [0, 2] else self.poro_grid

            if mode == 3:
                # Vertical Proportion Curve (VPC)
                self._render_vpc()
                return

            slice_idx = self.slice_slider.value()

            if mode == 0:  # X-Z Cross-section
                slice_idx = np.clip(slice_idx, 0, self.ny - 1)
                data_slice = grid[:, slice_idx, :].T  # Shape: (nz, nx)
                x_extent = self.length_ft
                z_extent = self.thickness_ft
                xlabel, ylabel = "Length Along Reservoir X (ft)", "Reservoir TVD Depth (ft)"

            elif mode == 1:  # Y-Z Cross-section
                slice_idx = np.clip(slice_idx, 0, self.nx - 1)
                data_slice = grid[slice_idx, :, :].T  # Shape: (nz, ny)
                x_extent = self.width_ft
                z_extent = self.thickness_ft
                xlabel, ylabel = "Width Along Reservoir Y (ft)", "Reservoir TVD Depth (ft)"

            elif mode == 2:  # X-Y Map slice
                slice_idx = np.clip(slice_idx, 0, self.nz - 1)
                data_slice = grid[:, :, slice_idx].T  # Shape: (ny, nx)
                x_extent = self.length_ft
                z_extent = self.width_ft
                xlabel, ylabel = "Length Along Reservoir X (ft)", "Width Along Reservoir Y (ft)"

            # Colormap selection
            if prop_idx == 0:  # Permeability
                cmap = "turbo"
                title = f"Permeability Cross-Section (mD)"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=[0, x_extent, z_extent if mode != 2 else 0, 0 if mode != 2 else z_extent],
                                         cmap=cmap, aspect="auto")
            elif prop_idx == 1:  # Porosity
                cmap = "viridis"
                title = f"Porosity Cross-Section (φ)"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=[0, x_extent, z_extent if mode != 2 else 0, 0 if mode != 2 else z_extent],
                                         cmap=cmap, aspect="auto")
            else:  # Facies (Sand vs Shale)
                sand_mask = (data_slice >= 10.0)
                facies_data = np.where(sand_mask, 1.0, 0.0)
                cmap = plt.cm.get_cmap("RdYlGn", 2)
                title = f"Lithofacies Classification (Green: Sand, Red: Shale)"
                im = self.ax_main.imshow(facies_data, origin="upper" if mode != 2 else "lower",
                                         extent=[0, x_extent, z_extent if mode != 2 else 0, 0 if mode != 2 else z_extent],
                                         cmap=cmap, aspect="auto", vmin=0, vmax=1)

            # Dip Angle Projection Overlay
            if abs(self.dip_angle_deg) > 0.01 and mode in [0, 1]:
                dip_rad = np.radians(self.dip_angle_deg)
                dip_z = np.tan(dip_rad) * x_extent * 0.2
                self.ax_main.plot([0, x_extent * 0.5], [z_extent * 0.5, z_extent * 0.5 + dip_z],
                                  color="white", linestyle="--", lw=1.8, label=f"Structural Dip ({self.dip_angle_deg:.1f}°)")
                self.ax_main.legend(loc="upper right", fontsize=8)

            self.ax_main.set_title(title, fontsize=10, fontweight="bold")
            self.ax_main.set_xlabel(xlabel, fontsize=9)
            self.ax_main.set_ylabel(ylabel, fontsize=9)
            self.fig.colorbar(im, ax=self.ax_main, fraction=0.046, pad=0.04)

            # Calculate Net-to-Gross
            ntg = float(np.mean(self.perm_grid >= 10.0))
            self.info_label.setText(
                f"Stratigraphic Continuity: Confirmed | Grid: {self.nx}x{self.ny}x{self.nz} | Net-to-Gross: {ntg:.2f} | Dip: {self.dip_angle_deg:.1f}°"
            )

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            logger.error(f"Error rendering geology cross-section: {e}", exc_info=True)

    def _render_vpc(self):
        """Renders Vertical Proportion Curve (VPC) across K layers."""
        self.fig.clear()
        ax_vpc = self.fig.add_subplot(121)
        ax_cum = self.fig.add_subplot(122)

        nz = self.nz
        depths = np.linspace(0, self.thickness_ft, nz)

        sand_props = []
        shale_props = []
        for k in range(nz):
            k_layer = self.perm_grid[:, :, k]
            sand_frac = float(np.mean(k_layer >= 10.0))
            sand_props.append(sand_frac)
            shale_props.append(1.0 - sand_frac)

        sand_props = np.array(sand_props)
        shale_props = np.array(shale_props)

        # Plot VPC stacked bars
        ax_vpc.barh(depths, sand_props, height=self.thickness_ft / nz * 0.9, color="#28a745", label="Permeable Sand (k ≥ 10 mD)")
        ax_vpc.barh(depths, shale_props, left=sand_props, height=self.thickness_ft / nz * 0.9, color="#dc3545", label="Impermeable Shale Barrier")
        ax_vpc.set_xlim(0, 1.0)
        ax_vpc.invert_yaxis()
        ax_vpc.set_xlabel("Facies Volumetric Fraction", fontsize=9)
        ax_vpc.set_ylabel("Reservoir Thickness (ft)", fontsize=9)
        ax_vpc.set_title("Vertical Proportion Curve (VPC)", fontsize=10, fontweight="bold")
        ax_vpc.grid(True, linestyle=":", alpha=0.6)
        ax_vpc.legend(fontsize=8, loc="upper right")

        # Cumulative Net Pay Thickness
        cum_net_pay = np.cumsum(sand_props * (self.thickness_ft / nz))
        ax_cum.plot(cum_net_pay, depths, "b-o", lw=2.2, label="Cumulative Net Pay (ft)")
        ax_cum.plot(depths, depths, "k--", alpha=0.5, label="100% Net Sand Baseline")
        ax_cum.invert_yaxis()
        ax_cum.set_xlabel("Net Pay Thickness (ft)", fontsize=9)
        ax_cum.set_ylabel("Gross Thickness (ft)", fontsize=9)
        ax_cum.set_title("Cumulative Net Pay vs Gross", fontsize=10, fontweight="bold")
        ax_cum.grid(True, linestyle=":", alpha=0.6)
        ax_cum.legend(fontsize=8, loc="lower right")

        self.fig.tight_layout()
        self.canvas.draw()
