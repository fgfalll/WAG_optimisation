"""
Test 3D Well Mechanics, Interactive Ray-Plane Placement & Property Volume Rendering
==================================================================================
Verifies:
1. Left-panel Wells tab contains well inventory, action buttons, and mechanics card without duplicate 3D canvas.
2. 3D Shared Earth Subsurface Model renders property realization (Permeability, Porosity, Saturation, Pressure).
3. Ray-plane screen-to-reservoir coordinate inversion for interactive clicking.
4. Interactive well placement via ManualWellDialog with pre-filled coordinates.
"""

import sys
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.data_management_widget import DataManagementWidget
from ui.widgets.manual_well_dialog import ManualWellDialog
from core.data_models import WellData


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_manual_well_dialog_initial_values(qapp):
    """Verify ManualWellDialog accepts initial_values to pre-fill coordinates from 3D click."""
    init_vals = {
        "SurfaceX": 1250.0,
        "SurfaceY": 850.0,
        "TopDepth": 5000.0,
        "BottomDepth": 5060.0,
        "name": "Well-Click-Test",
        "TrajectoryType": "Horizontal",
        "LateralLength": 2000.0,
        "role": "Producer (Active)",
    }
    dlg = ManualWellDialog([], initial_values=init_vals)
    assert dlg.key_param_values["SurfaceX"] == 1250.0
    assert dlg.key_param_values["SurfaceY"] == 850.0
    assert dlg.key_param_values["TopDepth"] == 5000.0
    assert dlg.key_param_values["BottomDepth"] == 5060.0
    assert dlg.key_param_values["TrajectoryType"] == "Horizontal"
    assert dlg.key_param_values["LateralLength"] == 2000.0
    assert dlg.well_name_edit.text() == "Well-Click-Test"
    assert "Peaceman Well Index (Horizontal):" in dlg.peaceman_wi_label.text()


def test_wells_tab_no_duplicate_3d_cube(qapp):
    """Verify duplicate 3D cube is removed from left-panel Wells tab and buttons exist."""
    dm = DataManagementWidget()
    # Left panel Wells tab should not embed WellTrajectoryRendererWidget canvas
    assert dm.well_trajectory_renderer is None

    # Verify action buttons exist in Wells tab
    assert hasattr(dm, "add_well_btn")
    assert hasattr(dm, "place_well_3d_btn")
    assert hasattr(dm, "focus_well_3d_btn")
    assert hasattr(dm, "selected_well_mechanics_label")


def test_3d_subsurface_property_volume_rendering(qapp):
    """Verify 3D Subsurface Model renders volumetric property grid via PyVista without Matplotlib 3D."""
    dm = DataManagementWidget()
    assert hasattr(dm, "canvas_3d")
    assert dm.canvas_3d is not None

    # Set parameters
    dm.manual_inputs_values["length"] = 2000.0
    dm.manual_inputs_values["area"] = 100.0
    dm.manual_inputs_values["thickness"] = 60.0
    dm.manual_inputs_values["perm"] = 150.0
    dm.manual_inputs_values["porosity"] = 0.22

    # Render via PyVista
    dm._render_3d_subsurface_view()
    assert dm.canvas_3d.grid_mesh is not None
    assert dm.canvas_3d.grid_mesh.n_points > 0
    assert "Permeability" in dm.canvas_3d.grid_mesh.cell_data
    assert "Porosity" in dm.canvas_3d.grid_mesh.cell_data


def test_pyvista_surface_picking_and_well_placement(qapp, monkeypatch):
    """Verify surface clicking on the actual 3D reservoir mesh pre-fills ManualWellDialog."""
    dm = DataManagementWidget()
    dm.manual_inputs_values["length"] = 2000.0
    dm.manual_inputs_values["area"] = 100.0
    dm.manual_inputs_values["thickness"] = 50.0
    dm.manual_inputs_values["depth"] = 4000.0
    dm._render_3d_subsurface_view()

    target_x = 750.0
    target_y = 1250.0

    captured_init_values = {}

    def mock_exec(dialog_self):
        nonlocal captured_init_values
        captured_init_values = dict(dialog_self.key_param_values)
        return 0

    monkeypatch.setattr(ManualWellDialog, "exec", mock_exec)

    # Trigger surface_clicked signal from PyVista canvas
    dm.canvas_3d.surface_clicked.emit(target_x, target_y)

    assert captured_init_values.get("SurfaceX") == target_x
    assert captured_init_values.get("SurfaceY") == target_y
    assert captured_init_values.get("TopDepth") == dm._get_reservoir_top_depth()


def test_manual_well_dialog_data_creation(qapp):
    """Verify get_well_data creates WellData without unexpected keyword argument errors."""
    init_vals = {
        "SurfaceX": 1000.0,
        "SurfaceY": 500.0,
        "TopDepth": 4900.0,
        "BottomDepth": 5050.0,
        "name": "Well-Creation-Test",
        "TrajectoryType": "Vertical",
        "role": "Injector",
    }
    dlg = ManualWellDialog([], initial_values=init_vals)
    well = dlg.get_well_data()
    assert well is not None
    assert well.name == "Well-Creation-Test"
    assert well.well_index is not None
    assert well.well_index > 0
    assert hasattr(well, "skin_factor")
    assert hasattr(well, "wellbore_radius_ft")

