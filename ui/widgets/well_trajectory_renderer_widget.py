"""
Interactive 3D Well Trajectory & Pattern Renderer Widget (PyVista Engine)
========================================================================

High-performance 3D visualization canvas powered exclusively by PyVista
(VTK/OpenGL) for well architecture, 3D trajectories (Vertical, Horizontal,
Deviated S-Curve), completed perforation intervals, inter-well sweep
connectivity vectors, and pattern geometries.
"""

from typing import List, Optional, Dict, Any
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from core.data_models import WellData
from ui.workbench.components.pyvista_reservoir_canvas import PyVistaReservoirCanvas


class WellTrajectoryRendererWidget(QWidget):
    """
    Dedicated 3D canvas displaying well trajectories, perforations,
    and inter-well communication vectors powered exclusively by PyVista.
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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # High-performance PyVista 3D Subsurface Canvas
        self.canvas = PyVistaReservoirCanvas(self)
        self.canvas.well_clicked.connect(self.wellSelected.emit)
        main_layout.addWidget(self.canvas)

    def set_wells_data(self, wells: List[WellData], reservoir_params: Optional[Dict[str, Any]] = None):
        """Update wells and reservoir parameters, then re-render in PyVista 3D."""
        self.wells = list(wells) if wells else []
        self.reservoir_params = dict(reservoir_params) if reservoir_params else {}
        self.render()

    def highlight_well(self, well_name: Optional[str]):
        """Highlight/isolate a specific well in the PyVista 3D canvas."""
        self.highlighted_well_name = well_name
        self.canvas.set_isolated_well(well_name)

    def render(self):
        """Perform full hardware-accelerated PyVista 3D rendering."""
        res_len = float(self.reservoir_params.get("length_ft", 2000.0) or 2000.0)
        res_width = float(self.reservoir_params.get("width_ft", 1000.0) or 1000.0)
        top_depth = float(self.reservoir_params.get("depth_ft", 1000.0) or 1000.0)
        thick = float(self.reservoir_params.get("net_pay_ft", 100.0) or 100.0)
        perm = float(self.reservoir_params.get("perm_md", 100.0) or 100.0)
        poro = float(self.reservoir_params.get("porosity", 0.20) or 0.20)

        self.canvas.render_subsurface_model(
            nx=int(self.reservoir_params.get("nx", 30)),
            ny=int(self.reservoir_params.get("ny", 30)),
            nz=int(self.reservoir_params.get("nz", 6)),
            length_ft=res_len,
            width_ft=res_width,
            top_depth=top_depth,
            thickness_ft=thick,
            perm_base=perm,
            poro_base=poro,
            well_data_list=self.wells
        )
        if self.highlighted_well_name:
            self.canvas.set_isolated_well(self.highlighted_well_name)
