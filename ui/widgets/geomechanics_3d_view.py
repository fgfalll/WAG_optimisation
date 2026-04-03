import logging
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QIcon, QAction

logger = logging.getLogger(__name__)

try:
    from pyvistaqt import QtInteractor
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista or pyvistaqt not available. 3D visualization will be limited.")


class Geomechanics3DView(QWidget):
    """
    Interactive 3D visualization widget for geomechanics data using PyVista.

    Features:
    - Structured grid visualization (reservoir grids, simulation meshes)
    - Scalar field visualization (stress, strain, pressure, porosity, permeability)
    - Vector field visualization (displacement, velocity, stress tensors)
    - Contour slicing and volume rendering
    - Interactive camera controls (rotate, zoom, pan)
    - Multiple color maps and visualization presets
    - Well trajectory visualization
    """

    data_updated = pyqtSignal(dict)
    status_message = pyqtSignal(str, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        if not PYVISTA_AVAILABLE:
            self._create_fallback_ui()
            return

        self._setup_ui()
        self._setup_plotter()
        self._connect_signals()

        self.grid_data: Optional[pv.StructuredGrid] = None
        self.current_scalar: str = ""
        self.current_vectors: Optional[np.ndarray] = None
        self.well_trajectories: List[Dict] = []

        self._show_welcome_message()

    def _create_fallback_ui(self):
        """Create a simple UI when PyVista is not available."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info_label = QLabel(
            "PyVista is required for 3D visualization.\nPlease install: pip install pyvista pyvistaqt"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 40px;
                font-size: 14px;
                color: #333;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

        layout.addWidget(info_label)
        self.plotter = None

    def _setup_ui(self):
        """Set up the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        self.viewer_container = QWidget()
        viewer_layout = QVBoxLayout(self.viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self.viewer_container)
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        viewer_layout.addWidget(self.plotter)
        main_layout.addWidget(self.viewer_container)

        status_bar = self._create_status_bar()
        main_layout.addWidget(status_bar)

    def _create_toolbar(self) -> QWidget:
        """Create the visualization toolbar."""
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #e8e8e8;
                border-bottom: 1px solid #ccc;
                padding: 4px;
            }
            QPushButton {
                min-width: 80px;
                padding: 4px 8px;
                background-color: #f5f5f5;
                border: 1px solid #bbb;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #d0d0d0;
            }
            QComboBox {
                min-width: 120px;
            }
        """)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Scalar Field:"))
        self.scalar_combo = QComboBox()
        self.scalar_combo.setPlaceholderText("Select field...")
        layout.addWidget(self.scalar_combo)

        layout.addWidget(QLabel("Color Map:"))
        self.colormap_combo = QComboBox()
        self._populate_colormaps()
        layout.addWidget(self.colormap_combo)

        layout.addStretch(1)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip("Reset camera to initial position")
        layout.addWidget(self.reset_view_btn)

        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.setToolTip("Save screenshot of current view")
        layout.addWidget(self.screenshot_btn)

        layout.addWidget(QLabel("|"))

        self.show_axes_check = QPushButton("Axes")
        self.show_axes_check.setCheckable(True)
        self.show_axes_check.setChecked(True)
        self.show_axes_check.setToolTip("Toggle axes visibility")
        layout.addWidget(self.show_axes_check)

        self.show_outline_check = QPushButton("Outline")
        self.show_outline_check.setCheckable(True)
        self.show_outline_check.setChecked(True)
        self.show_outline_check.setToolTip("Toggle grid outline")
        layout.addWidget(self.show_outline_check)

        return toolbar

    def _populate_colormaps(self):
        """Populate colormap dropdown with available color maps."""
        colormaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "jet",
            "turbo",
            "coolwarm",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "seismic",
            "rainbow",
            "nipy_spectral",
            "gray",
            "bone",
            "pink",
            "spring",
            "summer",
            "autumn",
            "winter",
        ]
        self.colormap_combo.addItems(colormaps)
        self.colormap_combo.setCurrentText("viridis")

    def _create_status_bar(self) -> QWidget:
        """Create status bar for displaying info."""
        status = QWidget()
        status.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
                border-top: 1px solid #ccc;
                padding: 2px 8px;
            }
            QLabel {
                font-size: 11px;
                color: #666;
            }
        """)

        layout = QHBoxLayout(status)
        layout.setContentsMargins(8, 2, 8, 2)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        return status

    def _setup_plotter(self):
        """Initialize the PyVista plotter with default settings."""
        if self.plotter is None:
            return

        self.plotter.background_color = "#ffffff"
        self.plotter.add_axes(interactive=True, line_width=2)
        self.plotter.enable_eye_dome_lighting()

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        self.reset_view_btn.clicked.connect(self._reset_view)
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        self.show_axes_check.toggled.connect(self._toggle_axes)
        self.show_outline_check.toggled.connect(self._toggle_outline)
        self.scalar_combo.currentIndexChanged.connect(self._on_scalar_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)

    def _show_welcome_message(self):
        """Show welcome message in the viewer."""
        if self.plotter is None:
            return

        self.status_label.setText("Load data to visualize 3D geomechanics model")

    def _reset_view(self):
        """Reset camera to initial position."""
        if self.plotter is None:
            return

        self.plotter.reset_camera()
        self.status_label.setText("View reset")

    def _take_screenshot(self):
        """Take a screenshot of the current view."""
        if self.plotter is None:
            return

        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "geomechanics_view.png", "PNG Images (*.png);;All Files (*)"
        )

        if file_path:
            try:
                self.plotter.screenshot(file_path)
                self.status_label.setText(f"Screenshot saved: {file_path}")
            except Exception as e:
                logger.error(f"Failed to save screenshot: {e}")
                self.status_label.setText("Failed to save screenshot")

    def _toggle_axes(self, visible: bool):
        """Toggle axes visibility."""
        if self.plotter is None:
            return

        if visible:
            self.plotter.add_axes(interactive=True, line_width=2)
        else:
            self.plotter.remove_actor("Axes")

    def _toggle_outline(self, visible: bool):
        """Toggle grid outline visibility."""
        if self.plotter is None or self.grid_data is None:
            return

        if visible:
            self.plotter.add_mesh(
                self.grid_data.outline(), color="black", line_width=1, name="outline"
            )
        else:
            self.plotter.remove_actor("outline")

    def _on_scalar_changed(self, index: int):
        """Handle scalar field selection change."""
        if self.plotter is None:
            return

        scalar_name = self.scalar_combo.currentText()
        if scalar_name and self.grid_data:
            self._update_visualization()
            self.status_label.setText(f"Displaying: {scalar_name}")

    def _on_colormap_changed(self, colormap: str):
        """Handle colormap selection change."""
        if self.plotter is None:
            return

        if self.grid_data and self.current_scalar:
            self._update_visualization()

    def clear(self):
        """Clear the visualization."""
        if self.plotter is None:
            return

        self.plotter.clear()
        self._setup_plotter()
        self.grid_data = None
        self.current_scalar = ""
        self.current_vectors = None
        self.well_trajectories = []
        self._show_welcome_message()

    def set_grid_data(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        cell_data: Optional[Dict[str, np.ndarray]] = None,
        point_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Set structured grid data for visualization.

        Args:
            x: X coordinates array (1D or 3D meshgrid)
            y: Y coordinates array (1D or 3D meshgrid)
            z: Z coordinates array (1D or 3D meshgrid)
            cell_data: Dictionary of cell-centered scalar fields
            point_data: Dictionary of point-centered scalar fields
        """
        if self.plotter is None:
            return

        try:
            if pv is None:
                logger.warning("PyVista not available")
                return

            if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
            else:
                xx, yy, zz = x, y, z

            self.grid_data = pv.StructuredGrid(xx, yy, zz)

            if cell_data:
                for name, values in cell_data.items():
                    self.grid_data.cell_data[name] = values

            if point_data:
                for name, values in point_data.items():
                    self.grid_data.point_data[name] = values

            self._populate_scalar_fields()
            self._update_visualization()

            dims = self.grid_data.dimensions
            self.info_label.setText(
                f"Grid: {dims[0]}×{dims[1]}×{dims[2]} = {self.grid_data.n_cells} cells"
            )
            self.status_label.setText("Grid data loaded successfully")

        except Exception as e:
            logger.error(f"Failed to create grid: {e}")
            self.status_label.setText(f"Error loading grid: {str(e)}")

    def set_well_trajectories(self, trajectories: List[Dict[str, Any]]):
        """
        Set well trajectories for visualization.

        Args:
            trajectories: List of dictionaries with keys:
                - name: Well name
                - md: Measured depth array
                - tvd: True vertical depth array
                - x: X coordinate array
                - y: Y coordinate array
                - radius: Well radius (optional)
        """
        if self.plotter is None:
            return

        self.well_trajectories = trajectories
        self._update_well_visualization()

    def _update_well_visualization(self):
        """Update well trajectory visualization."""
        if self.plotter is None or not self.well_trajectories:
            return

        for well in self.well_trajectories:
            if all(k in well for k in ["x", "y", "md"]):
                try:
                    # Create smooth well trajectory using PyVista Spline
                    # Following PyVista best practice: use n_points parameter for interpolation
                    # Reference: https://docs.pyvista.org/examples/00-load/create_spline.html
                    points = np.column_stack([well["x"], well["y"], well["md"]])
                    n_interp = max(100, 2 * len(points))  # Ensure minimum smoothness
                    trajectory = pv.Spline(points, n_points=n_interp)

                    radius = well.get("radius", 50.0)
                    tube = trajectory.tube(radius)

                    self.plotter.add_mesh(
                        tube,
                        color=self._get_well_color(well.get("name", "")),
                        name=f"well_{well.get('name', 'unknown')}",
                        show_scalar_bar=False,
                    )
                except Exception as e:
                    logger.warning(f"Failed to visualize well {well.get('name', 'unknown')}: {e}")

    def _get_well_color(self, well_name: str) -> str:
        """Get color for well based on name."""
        colors = {
            "injection": "blue",
            "production": "red",
            "observer": "green",
            "monitoring": "purple",
        }

        name_lower = well_name.lower()
        for key, color in colors.items():
            if key in name_lower:
                return color

        return "black"

    def _populate_scalar_fields(self):
        """Populate scalar field dropdown with available fields."""
        self.scalar_combo.blockSignals(True)
        self.scalar_combo.clear()

        if self.grid_data:
            available_scalars = []
            for name in self.grid_data.cell_data.keys():
                available_scalars.append(name)
            for name in self.grid_data.point_data.keys():
                available_scalars.append(name)

            self.scalar_combo.addItems(available_scalars)

            if available_scalars:
                self.scalar_combo.setCurrentIndex(0)

        self.scalar_combo.blockSignals(False)

    def _update_visualization(self):
        """Update the current visualization based on selected options."""
        if self.plotter is None or self.grid_data is None:
            return

        self.plotter.clear()

        self._setup_plotter()

        if self.show_outline_check.isChecked():
            self._toggle_outline(True)

        self.current_scalar = self.scalar_combo.currentText()
        colormap = self.colormap_combo.currentText()

        if self.current_scalar:
            try:
                if self.current_scalar in self.grid_data.cell_data:
                    data = self.grid_data.cell_data[self.current_scalar]
                elif self.current_scalar in self.grid_data.point_data:
                    data = self.grid_data.point_data[self.current_scalar]
                else:
                    data = None

                if data is not None:
                    self.plotter.add_mesh(
                        self.grid_data,
                        scalars=data,
                        cmap=colormap,
                        show_scalar_bar=True,
                        scalar_bar_args={
                            "title": self.current_scalar,
                            "vertical": True,
                            "width": 0.3,
                            "height": 0.6,
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to render scalar field: {e}")
                self.plotter.add_mesh(self.grid_data, color="lightgray")

        self._update_well_visualization()

    def set_camera_position(
        self, position: Tuple[float, float, float], focal_point: Tuple[float, float, float]
    ):
        """Set camera position for the view."""
        if self.plotter is None:
            return

        self.plotter.camera_position = [position, focal_point, (0, 0, 1)]

    def set_subplot(self, *args, **kwargs):
        """Pass through for subplot configuration."""
        if self.plotter is None:
            return

        if hasattr(self.plotter, "subplot"):
            self.plotter.subplot(*args, **kwargs)

    def export_to_html(self, filename: str):
        """Export current view to interactive HTML file."""
        if self.plotter is None:
            return

        try:
            from pyvista import export_plottermesh

            export_plottermesh(filename, self.plotter)
            self.status_label.setText(f"Exported to: {filename}")
        except Exception as e:
            logger.error(f"Failed to export: {e}")
            self.status_label.setText(f"Export failed: {str(e)}")

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        if self.plotter and hasattr(self.plotter, "resize"):
            self.plotter.resizeEvent(event)
