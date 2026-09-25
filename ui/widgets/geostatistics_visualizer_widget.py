"""
Geostatistics Visualizer Widget
Part of Workstream 1.4: Real-Time Subsurface Visualizers & Shared Earth Model.

Renders 1D semi-variogram model curves against experimental lags and produces
fast 2D/3D spatial Sequential Gaussian Simulation (SGS) realizations with
petrophysical histograms, Dykstra-Parsons V_DP, and well location overlays.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QSplitter, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QIcon

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.geology.geostatistical_modeling import (
    create_geostatistical_grid, calculate_variogram, theoretical_variogram
)

logger = logging.getLogger(__name__)


class GeostatisticsVisualizerWidget(QWidget):
    """
    Interactive Geostatistics and Spatial Heterogeneity Visualizer.
    Provides 1D semivariogram curve fitting alongside 2D/3D SGS realization heatmaps.
    """
    realization_updated = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.nx = 50
        self.ny = 50
        self.base_perm = 100.0
        self.base_poro = 0.20
        self.well_data_list: List[Any] = []
        self.current_k_field: Optional[np.ndarray] = None
        self.current_v_dp: float = 0.65

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Control Toolbar
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.Shape.StyledPanel)
        control_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 4px; }")
        control_layout = QGridLayout(control_frame)
        control_layout.setContentsMargins(6, 4, 6, 4)
        control_layout.setSpacing(8)

        # Variogram Type
        control_layout.addWidget(QLabel("Variogram:"), 0, 0)
        self.variogram_combo = QComboBox()
        self.variogram_combo.addItems(["spherical", "exponential", "gaussian", "matern", "cubic"])
        self.variogram_combo.currentIndexChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.variogram_combo, 0, 1)

        # Range
        control_layout.addWidget(QLabel("Range (ft):"), 0, 2)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(10.0, 50000.0)
        self.range_spin.setValue(500.0)
        self.range_spin.setSingleStep(50.0)
        self.range_spin.setDecimals(1)
        self.range_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.range_spin, 0, 3)

        # Sill
        control_layout.addWidget(QLabel("Sill (σ²):"), 0, 4)
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.01, 10.0)
        self.sill_spin.setValue(1.0)
        self.sill_spin.setSingleStep(0.1)
        self.sill_spin.setDecimals(2)
        self.sill_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.sill_spin, 0, 5)

        # Nugget
        control_layout.addWidget(QLabel("Nugget (c₀):"), 1, 0)
        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 5.0)
        self.nugget_spin.setValue(0.05)
        self.nugget_spin.setSingleStep(0.02)
        self.nugget_spin.setDecimals(2)
        self.nugget_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.nugget_spin, 1, 1)

        # Anisotropy
        control_layout.addWidget(QLabel("Anisotropy (X/Y):"), 1, 2)
        self.aniso_spin = QDoubleSpinBox()
        self.aniso_spin.setRange(0.1, 20.0)
        self.aniso_spin.setValue(1.0)
        self.aniso_spin.setSingleStep(0.2)
        self.aniso_spin.setDecimals(2)
        self.aniso_spin.valueChanged.connect(self._on_param_changed)
        control_layout.addWidget(self.aniso_spin, 1, 3)

        # Random Seed
        control_layout.addWidget(QLabel("Seed:"), 1, 4)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        control_layout.addWidget(self.seed_spin, 1, 5)

        # Action Buttons
        self.generate_btn = QPushButton(QIcon.fromTheme("view-refresh"), "Generate SGS Realization")
        self.generate_btn.setStyleSheet("font-weight: bold; background-color: #007bff; color: white; padding: 4px 10px; border-radius: 4px;")
        self.generate_btn.clicked.connect(self.generate_realization)
        control_layout.addWidget(self.generate_btn, 0, 6, 2, 1)

        main_layout.addWidget(control_frame)

        # Dual-Panel Figure Canvas
        self.fig = Figure(figsize=(8, 4.5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax_variogram = self.fig.add_subplot(121)
        self.ax_realization = self.fig.add_subplot(122)
        main_layout.addWidget(self.canvas, stretch=1)

        # Metrics Summary Bar
        self.metrics_label = QLabel("Dykstra-Parsons V_DP: -- | Koval H_K: -- | Mean Perm: -- mD")
        self.metrics_label.setStyleSheet("font-weight: bold; color: #2b3e50; padding: 2px 6px; background-color: #e9ecef; border-radius: 4px;")
        main_layout.addWidget(self.metrics_label)

    def set_grid_dimensions(self, nx: int, ny: int, base_perm: float = 100.0, base_poro: float = 0.20):
        self.nx = max(int(nx), 5)
        self.ny = max(int(ny), 5)
        self.base_perm = max(float(base_perm), 0.01)
        self.base_poro = max(float(base_poro), 0.01)

    def set_well_data(self, well_data_list: List[Any]):
        self.well_data_list = list(well_data_list)

    def set_parameters(self, params: Dict[str, Any]):
        if not params:
            return
        if 'variogram_type' in params:
            idx = self.variogram_combo.findText(str(params['variogram_type']).lower())
            if idx >= 0:
                self.variogram_combo.setCurrentIndex(idx)
        if 'range' in params and params['range'] is not None:
            self.range_spin.setValue(float(params['range']))
        if 'sill' in params and params['sill'] is not None:
            self.sill_spin.setValue(float(params['sill']))
        if 'nugget' in params and params['nugget'] is not None:
            self.nugget_spin.setValue(float(params['nugget']))
        if 'anisotropy_ratio' in params and params['anisotropy_ratio'] is not None:
            self.aniso_spin.setValue(float(params['anisotropy_ratio']))
        if 'random_seed' in params and params['random_seed'] is not None:
            self.seed_spin.setValue(int(params['random_seed']))

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'variogram_type': self.variogram_combo.currentText(),
            'range': self.range_spin.value(),
            'sill': self.sill_spin.value(),
            'nugget': self.nugget_spin.value(),
            'anisotropy_ratio': self.aniso_spin.value(),
            'random_seed': self.seed_spin.value(),
            'grid_resolution': (self.nx, self.ny),
            'v_dp': self.current_v_dp
        }

    def _on_param_changed(self):
        # Update variogram theoretical curve dynamically
        self._plot_variogram_only()

    def _plot_variogram_only(self):
        try:
            vtype = self.variogram_combo.currentText()
            range_val = self.range_spin.value()
            sill_val = self.sill_spin.value()
            nugget_val = self.nugget_spin.value()

            self.ax_variogram.clear()
            h = np.linspace(0.1, max(range_val * 2.5, 20.0), 120)
            theo_gamma = theoretical_variogram(h, vtype, sill_val, range_val, nugget_val)

            self.ax_variogram.plot(h, theo_gamma, "b-", lw=2.2, label=f"Model ({vtype.capitalize()})")
            self.ax_variogram.axhline(sill_val + nugget_val, color="red", linestyle="--", alpha=0.7, label=f"Sill ({sill_val + nugget_val:.2f})")
            self.ax_variogram.axvline(range_val, color="green", linestyle=":", alpha=0.8, label=f"Range a = {range_val:.0f} ft")

            self.ax_variogram.set_title("Theoretical Semivariogram Model", fontsize=10, fontweight="bold")
            self.ax_variogram.set_xlabel("Lag Distance h (ft)", fontsize=9)
            self.ax_variogram.set_ylabel(r"Semivariance $\gamma(h)$", fontsize=9)
            self.ax_variogram.grid(True, linestyle=":", alpha=0.6)
            self.ax_variogram.legend(fontsize=8, loc="lower right")
            self.canvas.draw()
        except Exception as e:
            logger.error(f"Error updating variogram curve: {e}", exc_info=True)

    def generate_realization(self):
        """Generates full 2D SGS grid and experimental variogram fit."""
        try:
            vtype = self.variogram_combo.currentText()
            range_val = self.range_spin.value()
            sill_val = self.sill_spin.value()
            nugget_val = self.nugget_spin.value()
            aniso_val = self.aniso_spin.value()
            seed_val = self.seed_spin.value()

            nx, ny = self.nx, self.ny

            norm_field = create_geostatistical_grid((nx, ny), {
                'variogram_type': vtype,
                'range': max(range_val / 20.0, 2.0),
                'sill': sill_val,
                'nugget': nugget_val,
                'anisotropy_ratio': aniso_val,
                'random_seed': seed_val,
            })

            # Scale to permeability log-normal distribution
            k_field = self.base_perm * np.exp(2.0 * (norm_field - 0.5))
            self.current_k_field = k_field

            # Compute Dykstra-Parsons Coefficient V_DP
            k_flat = np.sort(k_field.flatten())
            k_50 = float(np.percentile(k_flat, 50))
            k_84_1 = float(np.percentile(k_flat, 15.9))
            v_dp = float(np.clip((k_50 - k_84_1) / max(k_50, 1e-4), 0.0, 0.95))
            self.current_v_dp = v_dp
            koval_hk = 10.0 ** (v_dp / max(1.0 - v_dp, 0.05))

            # 1. Experimental variogram calculation
            lags, exp_gamma = calculate_variogram(norm_field)
            h_dense = np.linspace(0.1, max(float(np.max(lags)), 1.0), 100)
            theo_gamma = theoretical_variogram(h_dense, vtype, sill_val, max(range_val / 20.0, 2.0), nugget_val)

            self.ax_variogram.clear()
            self.ax_variogram.plot(lags, exp_gamma, "ko", markersize=5, label=r"Experimental $\hat{\gamma}(h)$")
            self.ax_variogram.plot(h_dense, theo_gamma, "b-", lw=2.2, label=f"Fit ({vtype.capitalize()})")
            self.ax_variogram.axhline(sill_val + nugget_val, color="red", linestyle="--", alpha=0.7, label=f"Sill ({sill_val + nugget_val:.2f})")
            self.ax_variogram.set_title("Experimental vs Model Variogram", fontsize=10, fontweight="bold")
            self.ax_variogram.set_xlabel("Lag Distance h (cells)", fontsize=9)
            self.ax_variogram.set_ylabel(r"Semivariance $\gamma(h)$", fontsize=9)
            self.ax_variogram.grid(True, linestyle=":", alpha=0.6)
            self.ax_variogram.legend(fontsize=8, loc="lower right")

            # 2. Permeability Realization Heatmap
            self.ax_realization.clear()
            im = self.ax_realization.imshow(k_field.T, origin="lower", cmap="viridis", aspect="auto")
            self.ax_realization.set_title(f"SGS Permeability Realization ({nx}x{ny})", fontsize=10, fontweight="bold")
            self.ax_realization.set_xlabel("Grid X (cells)", fontsize=9)
            self.ax_realization.set_ylabel("Grid Y (cells)", fontsize=9)

            # Well overlays if available
            for well in self.well_data_list:
                name = getattr(well, "name", "Well")
                metadata = getattr(well, "metadata", {})
                wtype = str(metadata.get("type", "")).lower()
                is_inj = "inj" in name.lower() or "injector" in wtype
                color = "#00ffff" if is_inj else "#ff0055"
                marker = "^" if is_inj else "o"
                sx = float(metadata.get("SurfaceX", nx * 0.5))
                sy = float(metadata.get("SurfaceY", ny * 0.5))
                cx = int(np.clip(sx / 2000.0 * nx if sx > nx else sx, 0, nx - 1))
                cy = int(np.clip(sy / 2000.0 * ny if sy > ny else sy, 0, ny - 1))
                self.ax_realization.scatter(cx, cy, color=color, s=80, marker=marker, edgecolors="white", linewidths=1.5)
                self.ax_realization.text(cx + 1, cy + 1, name, color="white", fontsize=8, fontweight="bold")

            self.fig.tight_layout()
            self.canvas.draw()

            # Update Metrics
            mean_k = float(np.mean(k_field))
            self.metrics_label.setText(
                f"Dykstra-Parsons V_DP: {v_dp:.3f} | Koval Factor H_K: {koval_hk:.2f} | Mean Perm: {mean_k:.1f} mD (P50: {k_50:.1f} mD)"
            )

            self.realization_updated.emit(self.get_parameters())

        except Exception as e:
            logger.error(f"Error generating geostatistics realization: {e}", exc_info=True)
