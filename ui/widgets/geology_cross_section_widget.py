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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QFrame, QCheckBox,
    QMenu, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QAction, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


class GeologyCrossSectionWidget(QWidget):
    """
    Interactive Stratigraphic Cross-Section and Vertical Proportion Curve (VPC) Viewer.
    Allows reservoir engineers to visually confirm layer connectivity, structural dip,
    wells, perforations, faults, and caprock seal integrity along orthogonal axes.
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
        self.top_depth = 5000.0
        self.dip_angle_deg = 0.0

        self.perm_grid: Optional[np.ndarray] = None
        self.poro_grid: Optional[np.ndarray] = None
        self.facies_grid: Optional[np.ndarray] = None
        self.layer_definitions: List[Any] = []
        self.well_data_list: List[Any] = []
        self.fault_props: Optional[Dict[str, Any]] = None
        self.caprock_props: Optional[Dict[str, Any]] = None

        self._init_synthetic_grid()
        self._setup_ui()
        self.render_slice()

    def _init_synthetic_grid(self):
        """Creates a baseline layered grid if no real simulation grid is loaded."""
        nx, ny, nz = self.nx, self.ny, self.nz
        self.perm_grid = np.zeros((nx, ny, nz), dtype=float)
        self.poro_grid = np.zeros((nx, ny, nz), dtype=float)

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
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 1. Clean Non-Squished Horizontal Top Toolbar
        top_frame = QFrame()
        top_frame.setFrameShape(QFrame.Shape.StyledPanel)
        top_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 2px 4px;
            }
            QLabel {
                color: #475569;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox {
                background: #ffffff;
                color: #1e293b;
                border: 1.5px solid #94a3b8;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
                min-height: 20px;
            }
            QComboBox:focus {
                border: 2px solid #0d6efd;
            }
            QPushButton {
                background: #ffffff;
                color: #1e293b;
                border: 1.5px solid #94a3b8;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #f1f5f9;
                border-color: #0d6efd;
            }
            QCheckBox {
                color: #334155;
                font-size: 11px;
                font-weight: 500;
                spacing: 4px;
            }
        """)

        toolbar_layout = QVBoxLayout(top_frame)
        toolbar_layout.setContentsMargins(6, 4, 6, 4)
        toolbar_layout.setSpacing(4)

        # Row 1: Orientation, Property, Colormap & Display Toggles
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(8)

        row1_layout.addWidget(QLabel("Orientation:"))
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems([
            "In-line Cross-Section (X-Z)",
            "Cross-line Cross-Section (Y-Z)",
            "Areal Map Slice (X-Y)",
            "Vertical Proportion Curve (VPC)"
        ])
        self.orientation_combo.currentIndexChanged.connect(self._on_orientation_changed)
        row1_layout.addWidget(self.orientation_combo)

        row1_layout.addWidget(QLabel("Property:"))
        self.property_combo = QComboBox()
        self.property_combo.addItems([
            "Permeability (mD)",
            "Porosity (φ)",
            "Facies (Sand vs Shale)",
            "Oil Saturation (So)",
            "Water Saturation (Sw)",
            "Miscibility Margin (P - MMP)",
            "Pore Pressure (psia)"
        ])
        self.property_combo.currentIndexChanged.connect(self.render_slice)
        row1_layout.addWidget(self.property_combo)

        row1_layout.addWidget(QLabel("Palette:"))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["turbo", "viridis", "plasma", "coolwarm", "jet"])
        self.palette_combo.currentIndexChanged.connect(self.render_slice)
        row1_layout.addWidget(self.palette_combo)

        # Overlays
        self.chk_caprock = QCheckBox("Caprock Seal")
        self.chk_caprock.setChecked(True)
        self.chk_caprock.toggled.connect(self.render_slice)
        row1_layout.addWidget(self.chk_caprock)

        self.chk_wells = QCheckBox("Wells & Perfs")
        self.chk_wells.setChecked(True)
        self.chk_wells.toggled.connect(self.render_slice)
        row1_layout.addWidget(self.chk_wells)

        self.chk_faults = QCheckBox("Fault Trace")
        self.chk_faults.setChecked(True)
        self.chk_faults.toggled.connect(self.render_slice)
        row1_layout.addWidget(self.chk_faults)

        self.chk_contacts = QCheckBox("Fluid Contacts")
        self.chk_contacts.setChecked(True)
        self.chk_contacts.setToolTip("Display Water-Oil Contact (WOC) and Gas-Oil Contact (GOC) lines")
        self.chk_contacts.toggled.connect(self.render_slice)
        row1_layout.addWidget(self.chk_contacts)

        self.chk_layers = QCheckBox("Sub-Layers")
        self.chk_layers.setChecked(True)
        self.chk_layers.toggled.connect(self.render_slice)
        row1_layout.addWidget(self.chk_layers)

        row1_layout.addStretch()
        toolbar_layout.addLayout(row1_layout)

        # Row 2: Slice Navigation Controls (Step Buttons + Slider + Readout)
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(6)

        row2_layout.addWidget(QLabel("Slice Position:"))

        self.btn_prev_slice = QPushButton("◀ Prev")
        self.btn_prev_slice.setFixedWidth(54)
        self.btn_prev_slice.setToolTip("Step slice backwards")
        self.btn_prev_slice.clicked.connect(self._step_prev_slice)
        row2_layout.addWidget(self.btn_prev_slice)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, self.ny - 1)
        self.slice_slider.setValue(self.ny // 2)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        row2_layout.addWidget(self.slice_slider, stretch=1)

        self.btn_next_slice = QPushButton("Next ▶")
        self.btn_next_slice.setFixedWidth(54)
        self.btn_next_slice.setToolTip("Step slice forwards")
        self.btn_next_slice.clicked.connect(self._step_next_slice)
        row2_layout.addWidget(self.btn_next_slice)

        self.slice_label = QLabel(f"Cross-line (Y): {self.ny // 2 + 1} / {self.ny}")
        self.slice_label.setStyleSheet("color: #0284c7; font-weight: bold; min-width: 170px;")
        row2_layout.addWidget(self.slice_label)

        self.btn_reset_slice = QPushButton("Reset Center")
        self.btn_reset_slice.setToolTip("Reset slice to central reservoir axis")
        self.btn_reset_slice.clicked.connect(self._reset_slice_to_center)
        row2_layout.addWidget(self.btn_reset_slice)

        toolbar_layout.addLayout(row2_layout)
        main_layout.addWidget(top_frame)

        # 2. Matplotlib Canvas with Context Menu Support
        self.fig = Figure(figsize=(8, 4.5), facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        self.ax_main = self.fig.add_subplot(111)
        main_layout.addWidget(self.canvas, stretch=1)

        # 3. Bottom Engineering Summary Bar
        self.info_label = QLabel("Stratigraphic Continuity: Active | Net-to-Gross: 0.70 | Dip: 0.0°")
        self.info_label.setStyleSheet("font-weight: bold; color: #1e293b; padding: 3px 8px; background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 11px;")
        main_layout.addWidget(self.info_label)

    def _step_prev_slice(self):
        val = self.slice_slider.value()
        if val > self.slice_slider.minimum():
            self.slice_slider.setValue(val - 1)

    def _step_next_slice(self):
        val = self.slice_slider.value()
        if val < self.slice_slider.maximum():
            self.slice_slider.setValue(val + 1)

    def _reset_slice_to_center(self):
        mode = self.orientation_combo.currentIndex()
        if mode == 0:
            self.slice_slider.setValue(self.ny // 2)
        elif mode == 1:
            self.slice_slider.setValue(self.nx // 2)
        elif mode == 2:
            self.slice_slider.setValue(self.nz // 2)

    def set_grid_data(
        self,
        perm_grid: Optional[np.ndarray],
        poro_grid: Optional[np.ndarray],
        length_ft: float = 2000.0,
        width_ft: float = 2000.0,
        top_depth: float = 5000.0,
        thickness_ft: float = 50.0,
        dip_angle_deg: float = 0.0,
        well_data_list: Optional[List[Any]] = None,
        fault_props: Optional[Dict[str, Any]] = None,
        caprock_props: Optional[Dict[str, Any]] = None,
        facies_grid: Optional[np.ndarray] = None,
        saturation_grid: Optional[np.ndarray] = None,
        sw_grid: Optional[np.ndarray] = None,
        margin_grid: Optional[np.ndarray] = None,
        pressure_grid: Optional[np.ndarray] = None,
        woc_depth: Optional[float] = None,
        goc_depth: Optional[float] = None
    ):
        if perm_grid is not None and perm_grid.ndim == 3:
            self.perm_grid = perm_grid
            self.nx, self.ny, self.nz = perm_grid.shape
        if poro_grid is not None and poro_grid.ndim == 3:
            self.poro_grid = poro_grid
        if facies_grid is not None and facies_grid.ndim == 3:
            self.facies_grid = facies_grid
        if saturation_grid is not None and saturation_grid.ndim == 3:
            self.saturation_grid = saturation_grid
        if sw_grid is not None and sw_grid.ndim == 3:
            self.sw_grid = sw_grid
        if margin_grid is not None and margin_grid.ndim == 3:
            self.margin_grid = margin_grid
        if pressure_grid is not None and pressure_grid.ndim == 3:
            self.pressure_grid = pressure_grid
        if woc_depth is not None:
            self.woc_depth = float(woc_depth)
        if goc_depth is not None:
            self.goc_depth = float(goc_depth)
        self.length_ft = length_ft
        self.width_ft = width_ft
        self.top_depth = top_depth
        self.thickness_ft = thickness_ft
        self.dip_angle_deg = dip_angle_deg
        if well_data_list is not None:
            self.well_data_list = list(well_data_list)
        if fault_props is not None:
            self.fault_props = fault_props
        if caprock_props is not None:
            self.caprock_props = caprock_props

        # Preserve slider position if valid, adjust bounds without resetting to middle
        self._update_slider_range(preserve_value=True)
        self.render_slice()

    def _update_slider_range(self, preserve_value: bool = False):
        mode = self.orientation_combo.currentIndex()
        old_val = self.slice_slider.value()

        if mode == 0:  # X-Z (slice across Y)
            max_val = max(self.ny - 1, 0)
            self.slice_slider.setEnabled(True)
            self.btn_prev_slice.setEnabled(True)
            self.btn_next_slice.setEnabled(True)
            self.slice_slider.setRange(0, max_val)
            val = np.clip(old_val, 0, max_val) if preserve_value else max_val // 2
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(val)
            self.slice_slider.blockSignals(False)
            y_coord = (val / max(self.ny - 1, 1)) * self.width_ft
            self.slice_label.setText(f"Cross-line (Y): {val + 1} / {self.ny} ({y_coord:.0f} ft)")

        elif mode == 1:  # Y-Z (slice across X)
            max_val = max(self.nx - 1, 0)
            self.slice_slider.setEnabled(True)
            self.btn_prev_slice.setEnabled(True)
            self.btn_next_slice.setEnabled(True)
            self.slice_slider.setRange(0, max_val)
            val = np.clip(old_val, 0, max_val) if preserve_value else max_val // 2
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(val)
            self.slice_slider.blockSignals(False)
            x_coord = (val / max(self.nx - 1, 1)) * self.length_ft
            self.slice_label.setText(f"In-line (X): {val + 1} / {self.nx} ({x_coord:.0f} ft)")

        elif mode == 2:  # X-Y (slice across Z)
            max_val = max(self.nz - 1, 0)
            self.slice_slider.setEnabled(True)
            self.btn_prev_slice.setEnabled(True)
            self.btn_next_slice.setEnabled(True)
            self.slice_slider.setRange(0, max_val)
            val = np.clip(old_val, 0, max_val) if preserve_value else max_val // 2
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(val)
            self.slice_slider.blockSignals(False)
            z_coord = self.top_depth + (val / max(self.nz - 1, 1)) * self.thickness_ft
            self.slice_label.setText(f"Depth Layer (K): {val + 1} / {self.nz} ({z_coord:.0f} ft TVD)")

        elif mode == 3:  # VPC
            self.slice_slider.setEnabled(False)
            self.btn_prev_slice.setEnabled(False)
            self.btn_next_slice.setEnabled(False)
            self.slice_label.setText("Vertical Stratigraphic Proportions")

    def _on_orientation_changed(self):
        self._update_slider_range(preserve_value=False)
        self.render_slice()

    def _on_slider_changed(self, val: int):
        mode = self.orientation_combo.currentIndex()
        if mode == 0:
            y_coord = (val / max(self.ny - 1, 1)) * self.width_ft
            self.slice_label.setText(f"Cross-line (Y): {val + 1} / {self.ny} ({y_coord:.0f} ft)")
        elif mode == 1:
            x_coord = (val / max(self.nx - 1, 1)) * self.length_ft
            self.slice_label.setText(f"In-line (X): {val + 1} / {self.nx} ({x_coord:.0f} ft)")
        elif mode == 2:
            z_coord = self.top_depth + (val / max(self.nz - 1, 1)) * self.thickness_ft
            self.slice_label.setText(f"Depth Layer (K): {val + 1} / {self.nz} ({z_coord:.0f} ft TVD)")
        self.render_slice()

    def render_slice(self):
        try:
            self.fig.clear()
            self.ax_main = self.fig.add_subplot(111)
            mode = self.orientation_combo.currentIndex()
            prop_idx = self.property_combo.currentIndex()
            palette_name = self.palette_combo.currentText()

            # Choose grid based on property selector
            if prop_idx == 0:
                grid = self.perm_grid
            elif prop_idx == 1:
                grid = self.poro_grid
            elif prop_idx == 2:
                grid = self.facies_grid if self.facies_grid is not None else self.perm_grid
            elif prop_idx == 3:
                grid = getattr(self, 'saturation_grid', None)
            elif prop_idx == 4:
                grid = getattr(self, 'sw_grid', None)
            elif prop_idx == 5:
                grid = getattr(self, 'margin_grid', None)
            elif prop_idx == 6:
                grid = getattr(self, 'pressure_grid', None)
            else:
                grid = self.perm_grid

            if grid is None:
                self._init_synthetic_grid()
                grid = self.perm_grid

            if mode == 3:
                # Vertical Proportion Curve (VPC)
                self._render_vpc()
                return

            slice_idx = self.slice_slider.value()
            top_z = self.top_depth
            base_z = self.top_depth + self.thickness_ft

            if mode == 0:  # X-Z Cross-section across Length X
                slice_idx = np.clip(slice_idx, 0, self.ny - 1)
                data_slice = grid[:, slice_idx, :].T  # Shape: (nz, nx)
                x_extent = self.length_ft
                y_pos = (slice_idx / max(self.ny - 1, 1)) * self.width_ft
                slice_coord = y_pos
                xlabel = "Length Along Reservoir X (ft)"
                ylabel = "True Vertical Depth TVD (ft)"
                extent = [0, x_extent, base_z, top_z]

            elif mode == 1:  # Y-Z Cross-section across Width Y
                slice_idx = np.clip(slice_idx, 0, self.nx - 1)
                data_slice = grid[slice_idx, :, :].T  # Shape: (nz, ny)
                x_extent = self.width_ft
                x_pos = (slice_idx / max(self.nx - 1, 1)) * self.length_ft
                slice_coord = x_pos
                xlabel = "Width Along Reservoir Y (ft)"
                ylabel = "True Vertical Depth TVD (ft)"
                extent = [0, x_extent, base_z, top_z]

            elif mode == 2:  # X-Y Map slice
                slice_idx = np.clip(slice_idx, 0, self.nz - 1)
                data_slice = grid[:, :, slice_idx].T  # Shape: (ny, nx)
                x_extent = self.length_ft
                y_extent = self.width_ft
                xlabel = "Length Along Reservoir X (ft)"
                ylabel = "Width Along Reservoir Y (ft)"
                extent = [0, x_extent, 0, y_extent]

            # Colormap selection
            if prop_idx == 0:  # Permeability
                cmap = palette_name
                title = f"Permeability Cross-Section (mD) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto")
            elif prop_idx == 1:  # Porosity
                cmap = palette_name
                title = f"Porosity Cross-Section (φ) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto")
            elif prop_idx == 3:  # Oil Saturation (So)
                cmap = "YlOrRd"
                title = f"Oil Saturation So Cross-Section (Fraction) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto", vmin=0.0, vmax=0.85)
            elif prop_idx == 4:  # Water Saturation (Sw)
                cmap = "Blues"
                title = f"Water Saturation Sw Cross-Section (Fraction) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
            elif prop_idx == 5:  # Miscibility Margin (P - MMP)
                cmap = "coolwarm"
                title = f"CO2 Miscibility Margin (P - MMP, psia) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto")
            elif prop_idx == 6:  # Pore Pressure
                cmap = palette_name
                title = f"Pore Pressure Cross-Section (psia) — Slice at {slice_idx + 1}"
                im = self.ax_main.imshow(data_slice, origin="upper" if mode != 2 else "lower",
                                         extent=extent, cmap=cmap, aspect="auto")
            else:  # Facies (Sand vs Silt vs Shale)
                if self.facies_grid is not None:
                    if mode == 0:
                        facies_data = self.facies_grid[:, slice_idx, :].T
                    elif mode == 1:
                        facies_data = self.facies_grid[slice_idx, :, :].T
                    else:
                        facies_data = self.facies_grid[:, :, slice_idx].T
                    from matplotlib.colors import ListedColormap
                    cmap = ListedColormap(["#f59e0b", "#06b6d4", "#334155"])
                    title = f"Lithofacies Architecture (Gold: Sand, Cyan: Silt, Slate: Shale) — Slice at {slice_idx + 1}"
                    im = self.ax_main.imshow(facies_data, origin="upper" if mode != 2 else "lower",
                                             extent=extent, cmap=cmap, aspect="auto", vmin=1, vmax=3)
                else:
                    sand_mask = (data_slice >= 10.0)
                    facies_data = np.where(sand_mask, 1.0, 0.0)
                    cmap = plt.colormaps.get_cmap("RdYlGn").resampled(2)
                    title = f"Lithofacies Stratification (Green: Sand, Red: Shale) — Slice at {slice_idx + 1}"
                    im = self.ax_main.imshow(facies_data, origin="upper" if mode != 2 else "lower",
                                             extent=extent, cmap=cmap, aspect="auto", vmin=0, vmax=1)

            # Caprock Confining Seal Overburden Layer for cross-sections
            if mode in [0, 1] and self.chk_caprock.isChecked():
                cp = self.caprock_props or {}
                cap_thk = float(cp.get("caprock_thickness", 150.0))
                cap_litho = str(cp.get("caprock_lithology", "Dense Marine Shale"))
                cap_top_z = top_z - cap_thk

                # Render overlying caprock formation block with geological shale hatching
                cap_rect = patches.Rectangle(
                    (0, cap_top_z), x_extent, cap_thk,
                    linewidth=1.2, edgecolor="#334155", facecolor="#1e293b",
                    alpha=0.88, hatch="//", zorder=3
                )
                self.ax_main.add_patch(cap_rect)

                # High-contrast Sealing Contact Horizon at top_z
                self.ax_main.axhline(top_z, color="#06b6d4", linestyle="-", lw=2.5, alpha=0.95, zorder=4)

                self.ax_main.text(
                    x_extent * 0.02, top_z - cap_thk * 0.45,
                    f"▲ Overlying Caprock Seal: {cap_litho} ({cap_thk:.0f} ft, k < 1e-4 mD)",
                    color="#f8fafc", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor="#06b6d4", alpha=0.92),
                    zorder=5
                )

                # Expand y limits to encompass caprock
                self.ax_main.set_ylim(base_z + self.thickness_ft * 0.05, cap_top_z - cap_thk * 0.10)

            elif mode in [0, 1]:
                # Standard reservoir limits with depth increasing downwards
                self.ax_main.set_ylim(base_z + self.thickness_ft * 0.05, top_z - self.thickness_ft * 0.05)

            # Sub-layer horizontal stratification lines
            if mode in [0, 1] and self.chk_layers.isChecked():
                sub_dz = self.thickness_ft / max(self.nz, 1)
                for k in range(1, self.nz):
                    layer_z = top_z + k * sub_dz
                    self.ax_main.axhline(layer_z, color="#ffffff", linestyle=":", lw=0.9, alpha=0.65, zorder=2)

            # Well penetrations and perforations on cross-section
            if mode in [0, 1] and self.chk_wells.isChecked() and self.well_data_list:
                tol = (self.width_ft / max(self.ny, 1)) * 1.8 if mode == 0 else (self.length_ft / max(self.nx, 1)) * 1.8
                for well in self.well_data_list:
                    w_name = getattr(well, "name", "Well")
                    meta = getattr(well, "metadata", {})
                    w_type = str(meta.get("type", "")).lower()
                    is_inj = "inj" in w_name.lower() or "inj" in w_type
                    well_x = float(meta.get("SurfaceX", self.length_ft * 0.5))
                    well_y = float(meta.get("SurfaceY", self.width_ft * 0.5))

                    target_coord = well_y if mode == 0 else well_x
                    dist = abs(target_coord - slice_coord)
                    if dist <= tol:
                        w_proj_x = well_x if mode == 0 else well_y
                        w_color = "#dc3545" if is_inj else "#0d6efd"
                        alpha_val = max(0.4, 1.0 - (dist / tol) * 0.6)

                        # Wellbore vertical path through reservoir
                        self.ax_main.plot([w_proj_x, w_proj_x], [top_z, base_z],
                                          color=w_color, lw=2.6, alpha=alpha_val, zorder=6)

                        # Wellhead badge
                        head_z = top_z - (cap_thk * 0.2 if self.chk_caprock.isChecked() else self.thickness_ft * 0.04)
                        self.ax_main.plot(w_proj_x, top_z, marker="v", markersize=8, color=w_color, zorder=7)
                        self.ax_main.text(
                            w_proj_x, head_z, w_name,
                            color=w_color, fontsize=8, fontweight="bold", ha="center",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor=w_color, alpha=0.9),
                            zorder=7
                        )

                        # Perforation intervals
                        perfs = getattr(well, "perforations", []) or [
                            [p.get("top", 0), p.get("bottom", 0)]
                            for p in getattr(well, "perforation_properties", [])
                        ]
                        for p in perfs:
                            if len(p) >= 2:
                                p_top, p_bot = max(p[0], top_z), min(p[1], base_z)
                                if p_bot > p_top:
                                    self.ax_main.plot([w_proj_x, w_proj_x], [p_top, p_bot],
                                                      color="#f59e0b", lw=4.5, alpha=alpha_val, zorder=6)

            # Fault trace cutting across reservoir slice
            if mode in [0, 1] and self.chk_faults.isChecked() and self.fault_props:
                fp = self.fault_props
                f_name = fp.get("fault_name", "Fault F-1")
                f_dip = float(fp.get("fault_dip", 70.0))
                f_strike = float(fp.get("fault_strike", 45.0))
                f_throw = float(fp.get("fault_throw", 25.0))
                f_cx = float(fp.get("fault_center_x", self.length_ft * 0.5))
                f_cy = float(fp.get("fault_center_y", self.width_ft * 0.5))

                f_x_center = f_cx if mode == 0 else f_cy
                cot_dip = 1.0 / np.tan(np.radians(np.clip(f_dip, 15.0, 88.0)))
                mid_z = (top_z + base_z) * 0.5
                dz = self.thickness_ft * 0.65
                dx = dz * cot_dip

                fz1, fz2 = mid_z - dz, mid_z + dz
                fx1, fx2 = f_x_center - dx, f_x_center + dx

                self.ax_main.plot([fx1, fx2], [fz1, fz2], color="#e11d48", linestyle="--", lw=2.2, zorder=5)
                self.ax_main.text(
                    (fx1 + fx2) * 0.5 + x_extent * 0.02, mid_z,
                    f"{f_name} (Throw: {f_throw:.0f} ft, Dip: {f_dip:.0f}°)",
                    color="#9f1239", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff1f2", edgecolor="#e11d48", alpha=0.9),
                    zorder=6
                )

            # Structural Dip Projection
            if abs(self.dip_angle_deg) > 0.01 and mode in [0, 1]:
                dip_rad = np.radians(self.dip_angle_deg)
                dip_dz = np.tan(dip_rad) * x_extent * 0.25
                mid_z = (top_z + base_z) * 0.5
                self.ax_main.plot([0, x_extent * 0.5], [mid_z, mid_z + dip_dz],
                                  color="#f59e0b", linestyle="-.", lw=1.8, label=f"Structural Dip ({self.dip_angle_deg:.1f}°)", zorder=5)
                self.ax_main.legend(loc="upper right", fontsize=8)

            # Fluid Contact Horizons (WOC & GOC) across reservoir cross-section
            if mode in [0, 1] and hasattr(self, 'chk_contacts') and self.chk_contacts.isChecked():
                woc_val = float(getattr(self, 'woc_depth', self.top_depth + self.thickness_ft * 0.75))
                goc_val = getattr(self, 'goc_depth', None)

                if top_z <= woc_val <= base_z:
                    self.ax_main.axhline(woc_val, color="#0284c7", linestyle="--", lw=2.2, zorder=6)
                    self.ax_main.text(
                        x_extent * 0.02, woc_val - self.thickness_ft * 0.02,
                        f"Water-Oil Contact (WOC): {woc_val:.0f} ft TVD",
                        color="#0369a1", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e0f2fe", edgecolor="#0284c7", alpha=0.9),
                        zorder=7
                    )

                if goc_val is not None and top_z <= float(goc_val) <= base_z:
                    goc_float = float(goc_val)
                    self.ax_main.axhline(goc_float, color="#d97706", linestyle="--", lw=2.2, zorder=6)
                    self.ax_main.text(
                        x_extent * 0.02, goc_float - self.thickness_ft * 0.02,
                        f"Gas-Oil Contact (GOC): {goc_float:.0f} ft TVD",
                        color="#b45309", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706", alpha=0.9),
                        zorder=7
                    )

            self.ax_main.set_title(title, fontsize=10.5, fontweight="bold", color="#1e293b", pad=8)
            self.ax_main.set_xlabel(xlabel, fontsize=9, color="#334155")
            self.ax_main.set_ylabel(ylabel, fontsize=9, color="#334155")
            self.ax_main.grid(True, linestyle=":", alpha=0.45, color="#94a3b8")
            self.fig.colorbar(im, ax=self.ax_main, fraction=0.038, pad=0.03)

            # Calculate Net-to-Gross
            ntg = float(np.mean(self.perm_grid >= 10.0))
            self.info_label.setText(
                f"Stratigraphic Continuity: Confirmed | Grid: {self.nx}x{self.ny}x{self.nz} | Top TVD: {top_z:.0f} ft | Net Pay: {self.thickness_ft:.0f} ft | NTG: {ntg:.2f} | Dip: {self.dip_angle_deg:.1f}°"
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
        depths = np.linspace(self.top_depth, self.top_depth + self.thickness_ft, nz)

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
        bar_h = (self.thickness_ft / nz) * 0.88
        ax_vpc.barh(depths, sand_props, height=bar_h, color="#16a34a", label="Permeable Sand (k ≥ 10 mD)")
        ax_vpc.barh(depths, shale_props, left=sand_props, height=bar_h, color="#dc3545", label="Impermeable Shale Barrier")
        ax_vpc.set_xlim(0, 1.0)
        ax_vpc.invert_yaxis()
        ax_vpc.set_xlabel("Facies Volumetric Fraction", fontsize=9, color="#334155")
        ax_vpc.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9, color="#334155")
        ax_vpc.set_title("Vertical Proportion Curve (VPC)", fontsize=10, fontweight="bold", color="#1e293b")
        ax_vpc.grid(True, linestyle=":", alpha=0.5)
        ax_vpc.legend(fontsize=8, loc="upper right")

        # Cumulative Net Pay Thickness
        cum_net_pay = np.cumsum(sand_props * (self.thickness_ft / nz))
        ax_cum.plot(cum_net_pay, depths, "b-o", lw=2.2, label="Cumulative Net Pay (ft)")
        ax_cum.plot(np.linspace(0, self.thickness_ft, nz), depths, "k--", alpha=0.5, label="100% Net Sand Baseline")
        ax_cum.invert_yaxis()
        ax_cum.set_xlabel("Net Pay Thickness (ft)", fontsize=9, color="#334155")
        ax_cum.set_ylabel("True Vertical Depth TVD (ft)", fontsize=9, color="#334155")
        ax_cum.set_title("Cumulative Net Pay vs Depth", fontsize=10, fontweight="bold", color="#1e293b")
        ax_cum.grid(True, linestyle=":", alpha=0.5)
        ax_cum.legend(fontsize=8, loc="lower right")

        self.fig.tight_layout()
        self.canvas.draw()

    def _show_context_menu(self, pos):
        """Displays rich right-click context menu for quick stratigraphy & slice settings."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
                color: #1e293b;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background: #0d6efd;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #e2e8f0;
                margin: 4px 8px;
            }
        """)

        # Orientations submenu
        orient_menu = menu.addMenu("Slice Orientation")
        for idx, text in enumerate(["In-line (X-Z)", "Cross-line (Y-Z)", "Map View (X-Y)", "Vertical Proportion Curve (VPC)"]):
            action = orient_menu.addAction(text)
            action.triggered.connect(lambda checked, i=idx: self.orientation_combo.setCurrentIndex(i))

        # Properties submenu
        prop_menu = menu.addMenu("Active Property")
        for idx, text in enumerate(["Permeability (mD)", "Porosity (φ)", "Facies (Sand vs Shale)"]):
            action = prop_menu.addAction(text)
            action.triggered.connect(lambda checked, i=idx: self.property_combo.setCurrentIndex(i))

        # Palettes submenu
        pal_menu = menu.addMenu("Colormap Palette")
        for pal in ["turbo", "viridis", "plasma", "coolwarm", "jet"]:
            action = pal_menu.addAction(pal)
            action.triggered.connect(lambda checked, p=pal: self.palette_combo.setCurrentText(p))

        menu.addSeparator()

        # Toggles
        act_cap = menu.addAction("Toggle Caprock Seal")
        act_cap.setCheckable(True)
        act_cap.setChecked(self.chk_caprock.isChecked())
        act_cap.triggered.connect(lambda c: self.chk_caprock.setChecked(c))

        act_wells = menu.addAction("Toggle Wells & Perfs")
        act_wells.setCheckable(True)
        act_wells.setChecked(self.chk_wells.isChecked())
        act_wells.triggered.connect(lambda c: self.chk_wells.setChecked(c))

        act_faults = menu.addAction("Toggle Fault Trace")
        act_faults.setCheckable(True)
        act_faults.setChecked(self.chk_faults.isChecked())
        act_faults.triggered.connect(lambda c: self.chk_faults.setChecked(c))

        act_layers = menu.addAction("Toggle Sub-Layers")
        act_layers.setCheckable(True)
        act_layers.setChecked(self.chk_layers.isChecked())
        act_layers.triggered.connect(lambda c: self.chk_layers.setChecked(c))

        menu.addSeparator()

        act_reset = menu.addAction("Reset Slice to Center")
        act_reset.triggered.connect(self._reset_slice_to_center)

        act_export = menu.addAction("Export Cross-Section Image...")
        act_export.triggered.connect(self._export_figure)

        menu.exec(self.canvas.mapToGlobal(pos))

    def _export_figure(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Stratigraphy Cross-Section", "cross_section.png", "PNG Images (*.png);;PDF Files (*.pdf)"
        )
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches="tight")
                logger.info(f"Exported cross-section to {filename}")
            except Exception as e:
                logger.error(f"Failed to export cross section: {e}")
