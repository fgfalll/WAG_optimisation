"""
Comprehensive unit and integration tests for Project Save and Load functionality.

Validates:
1. OptimizationEngine results property setter and getter.
2. ProjectFileHandler round-trip serialization/deserialization with nested dataclasses and 1D arrays.
3. Backwards compatibility loading of existing project files (e.g. test.tphd).
4. DataManagementWidget UI state flush (get_current_project_data) and restoration (load_project_data).
5. OptimizationWidget adoption of loaded optimization results.
"""

# IMPORTANT: Must import QtWebEngineWidgets before any QApplication instance is created
try:
    import PyQt6.QtWebEngineWidgets
except ImportError:
    pass

import os
from pathlib import Path
import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication

from core.data_models import (
    ReservoirData,
    PVTProperties,
    WellData,
    EOSModelParameters,
    LayerDefinition,
    GeostatisticalParams,
)
from core.optimisation_engine import OptimizationEngine
from utils.config_manager import ConfigManager
from utils.project_file_handler import (
    save_project_to_tphd,
    load_project_from_tphd,
)
from ui.data_management_widget import DataManagementWidget
from ui.optimization_widget import OptimizationWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure single QApplication instance for Qt widgets."""
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        QWebEngineView.setHtml = lambda self, *args, **kwargs: None
    except Exception:
        pass
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def create_sample_models():
    """Helper to create minimal valid reservoir and PVT data."""
    res = ReservoirData(
        grid={"PERMX": np.ones((5, 5, 3)), "ACTNUM": np.ones((5, 5, 3))},
        pvt_tables={},
        ooip_stb=20_000_000.0,
        initial_pressure=3200.0,
        temperature=160.0,
        thickness_ft=60.0,
        area_acres=640.0,
        average_porosity=0.20,
        average_permeability=80.0,
    )
    pvt = PVTProperties(
        temperature=160.0,
        gas_specific_gravity=0.72,
        api_gravity=35.0,
        oil_viscosity_cp=1.8,
        co2_viscosity=np.array([0.03]),
    )
    return res, pvt


def test_optimization_engine_results_setter_and_getter():
    """Verify OptimizationEngine.results can be set directly from loaded project data."""
    res, pvt = create_sample_models()
    engine = OptimizationEngine(reservoir=res, pvt=pvt)
    assert engine.results is None

    dummy_results = {
        "best_parameters": {"injection_rate": 8500.0, "wag_ratio": 1.5},
        "best_objective_value": 1.25e7,
        "convergence_history": [1.0e7, 1.1e7, 1.25e7],
        "detailed_metrics": {"oil_recovery_stb": 4.5e6, "co2_stored_tons": 8.5e5},
    }

    # Setting results should succeed without AttributeError
    engine.results = dummy_results
    assert engine.results == dummy_results


def test_project_file_handler_roundtrip_with_nested_dataclasses(tmp_path):
    """Verify round-trip serialization with nested dataclasses and 1D grid arrays."""
    save_file = tmp_path / "test_roundtrip.tphd"

    # Create reservoir data with nested dataclasses and 1D PERMX array (common in flattened grids)
    eos = EOSModelParameters(
        eos_type="PR",
        component_names=["C1", "CO2"],
        component_properties=np.array([
            [0.7, 190.6, 46.0, 0.011, 16.04],
            [0.3, 304.2, 73.8, 0.225, 44.01],
        ]),
        binary_interaction_coeffs=np.array([
            [0.0, 0.1],
            [0.1, 0.0],
        ]),
    )
    layers = [
        LayerDefinition(thickness=25.0, porosity=0.22, permeability_multiplier=1.2),
        LayerDefinition(thickness=40.0, porosity=0.18, permeability_multiplier=0.8),
    ]
    geostat = GeostatisticalParams(variogram_type="spherical", range=1200.0)

    grid = {
        "PERMX": np.linspace(10.0, 150.0, 5000),  # 1D flattened array
        "PORO": np.linspace(0.12, 0.28, 5000),
        "ACTNUM": np.ones(5000, dtype=int),
    }

    res = ReservoirData(
        grid=grid,
        pvt_tables={},
        ooip_stb=30_000_000.0,
        initial_pressure=3400.0,
        temperature=175.0,
        thickness_ft=65.0,
        area_acres=640.0,
        average_porosity=0.20,
        average_permeability=82.5,
        rock_compressibility=3.5e-6,
        dip_angle=4.5,
        density_contrast=0.25,
        interfacial_tension=18.0,
        eos_model=eos,
        layer_definitions=layers,
        geostatistical_params=geostat,
    )

    pvt = PVTProperties(
        temperature=175.0,
        gas_specific_gravity=0.72,
        api_gravity=36.0,
        oil_viscosity_cp=2.1,
        gas_viscosity_cp=0.025,
        c7_plus_fraction=0.35,
        co2_solubility_scm_per_bbl=85.0,
        co2_viscosity=np.array([0.035]),
    )

    wells = [
        WellData(
            name="PROD-01",
            depths=np.array([0.0, 8000.0]),
            properties={},
            units={},
            metadata={"status": "Producer", "type": "producer"},
            well_path=np.array([[0.0, 0.0, 0.0], [500.0, 500.0, 8000.0]]),
        ),
        WellData(
            name="INJ-01",
            depths=np.array([0.0, 8000.0]),
            properties={},
            units={},
            metadata={"status": "Injector", "type": "injector"},
            well_path=np.array([[0.0, 0.0, 0.0], [1500.0, 1500.0, 8000.0]]),
        ),
    ]

    opt_results = {
        "best_parameters": {"injection_rate": 9000.0},
        "best_objective_value": 4.5e7,
    }

    manual_inputs = {
        "area": 640.0,
        "thickness": 65.0,
        "poro": 0.20,
        "perm": 82.5,
        "sw": 0.25,
        "so": 0.75,
        "p_init": 3400.0,
        "temp": 175.0,
        "api": 36.0,
        "gas_grav": 0.72,
        "visc": 2.1,
    }

    data_to_save = {
        "schema_version": "1.1",
        "application_version": "0.8.5-alpha",
        "project_name": "TestRoundtrip",
        "reservoir_data": res,
        "pvt_properties": pvt,
        "optimization_results": opt_results,
        "well_data_list": wells,
        "manual_inputs": manual_inputs,
    }

    # Save project
    success = save_project_to_tphd(data_to_save, save_file)
    assert success is True
    assert os.path.exists(save_file)

    # Load project
    loaded = load_project_from_tphd(save_file)

    assert loaded["reservoir_data"] is not None
    loaded_res = loaded["reservoir_data"]
    assert isinstance(loaded_res, ReservoirData)
    assert loaded_res.ooip_stb == 30_000_000.0
    assert loaded_res.initial_pressure == 3400.0
    assert loaded_res.dip_angle == 4.5
    assert loaded_res.density_contrast == 0.25
    assert loaded_res.interfacial_tension == 18.0

    # Verify nested dataclasses are reconstructed
    assert isinstance(loaded_res.eos_model, EOSModelParameters)
    assert loaded_res.eos_model.eos_type == "PR"
    assert loaded_res.eos_model.component_names == ["C1", "CO2"]
    assert np.allclose(loaded_res.eos_model.component_properties, eos.component_properties)

    assert isinstance(loaded_res.layer_definitions, list)
    assert len(loaded_res.layer_definitions) == 2
    assert isinstance(loaded_res.layer_definitions[0], LayerDefinition)
    assert loaded_res.layer_definitions[0].thickness == 25.0
    assert loaded_res.layer_definitions[0].porosity == 0.22

    assert isinstance(loaded_res.geostatistical_params, GeostatisticalParams)
    assert loaded_res.geostatistical_params.variogram_type == "spherical"
    assert loaded_res.geostatistical_params.range == 1200.0

    # Verify PVT
    loaded_pvt = loaded["pvt_properties"]
    assert isinstance(loaded_pvt, PVTProperties)
    assert loaded_pvt.api_gravity == 36.0
    assert loaded_pvt.gas_viscosity_cp == 0.025
    assert loaded_pvt.c7_plus_fraction == 0.35
    assert loaded_pvt.co2_solubility_scm_per_bbl == 85.0

    # Verify Wells
    loaded_wells = loaded["well_data_list"]
    assert len(loaded_wells) == 2
    assert loaded_wells[0].name == "PROD-01"

    # Verify Results & Manual Inputs
    assert loaded["optimization_results"] == opt_results
    assert loaded["manual_inputs"] == manual_inputs


def test_load_existing_test_tphd_project():
    """Verify backwards-compatible loading of the repository's test.tphd file."""
    test_file = Path("test.tphd")
    if not test_file.exists():
        pytest.skip("test.tphd not found in repository root")

    loaded = load_project_from_tphd(test_file)
    assert loaded is not None
    assert loaded["reservoir_data"] is not None
    assert isinstance(loaded["reservoir_data"], ReservoirData)
    assert loaded["pvt_properties"] is not None
    assert isinstance(loaded["pvt_properties"], PVTProperties)
    assert loaded["optimization_results"] is not None

    # Check nested dataclass deserialization from legacy untyped dicts
    res = loaded["reservoir_data"]
    if res.eos_model is not None:
        assert isinstance(res.eos_model, EOSModelParameters)
    if res.layer_definitions:
        assert all(isinstance(layer, LayerDefinition) for layer in res.layer_definitions)
    if res.geostatistical_params is not None:
        assert isinstance(res.geostatistical_params, GeostatisticalParams)


def test_data_management_widget_save_and_load(qapp):
    """Test DataManagementWidget UI population, flush to project data, and safe loading of 1D PERMX."""
    config_mgr = ConfigManager()
    widget = DataManagementWidget(parent=None, config_manager=config_mgr)

    # 1. Test get_current_project_data
    widget.manual_inputs_widgets['area'].set_value(750.0)
    widget.manual_inputs_widgets['thickness'].set_value(80.0)
    widget.manual_inputs_widgets['perm'].set_value(110.0)
    widget.manual_inputs_widgets['dip_angle'].set_value(6.0)
    widget.manual_inputs_widgets['density_contrast'].set_value(0.30)
    widget.manual_inputs_widgets['interfacial_tension'].set_value(22.0)
    widget.manual_inputs_widgets['api_gravity'].set_value(39.0)
    widget.manual_inputs_widgets['gas_viscosity_cp'].set_value(0.018)

    project_data = widget.get_current_project_data()
    assert "reservoir_data" in project_data
    assert "pvt_properties" in project_data
    assert "manual_inputs" in project_data
    assert "well_data_list" in project_data

    cur_res = project_data["reservoir_data"]
    assert cur_res.area_acres == 750.0
    assert cur_res.thickness_ft == 80.0
    assert cur_res.average_permeability == 110.0
    assert cur_res.dip_angle == 6.0
    assert cur_res.density_contrast == 0.30
    assert cur_res.interfacial_tension == 22.0

    cur_pvt = project_data["pvt_properties"]
    assert cur_pvt.api_gravity == 39.0
    assert cur_pvt.gas_viscosity_cp == 0.018

    # 2. Test load_project_data with 1D PERMX array (flattened grid)
    # This previously triggered IndexError: too many indices for array: array is 1-dimensional, but 3 were indexed
    res_with_1d_perm = ReservoirData(
        grid={"PERMX": np.array([125.0, 130.0, 135.0]), "ACTNUM": np.ones(3)},
        pvt_tables={},
        ooip_stb=20_000_000.0,
        initial_pressure=3100.0,
        temperature=160.0,
        thickness_ft=70.0,
        area_acres=500.0,
        average_porosity=0.21,
        average_permeability=125.0,
        dip_angle=3.5,
        density_contrast=0.28,
        interfacial_tension=19.5,
    )
    pvt_sample = PVTProperties(
        temperature=160.0,
        gas_specific_gravity=0.70,
        api_gravity=41.0,
        oil_viscosity_cp=1.5,
        gas_viscosity_cp=0.019,
        c7_plus_fraction=0.28,
        co2_solubility_scm_per_bbl=75.0,
    )
    manual_in = {
        "area": 500.0,
        "thickness": 70.0,
        "perm": 125.0,
        "poro": 0.21,
        "sw": 0.22,
        "so": 0.78,
        "p_init": 3100.0,
        "temperature": 160.0,
        "api_gravity": 41.0,
        "gas_specific_gravity": 0.70,
        "oil_viscosity_cp": 1.5,
    }
    wells_sample = [
        WellData(
            name="W1",
            depths=np.array([0.0, 1000.0]),
            properties={},
            units={},
            metadata={"type": "producer", "status": "Producer"},
            well_path=np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 1000.0]]),
        ),
        WellData(
            name="W2",
            depths=np.array([0.0, 1000.0]),
            properties={},
            units={},
            metadata={"type": "injector", "status": "Injector"},
            well_path=np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 1000.0]]),
        ),
    ]

    # Loading must succeed without error
    widget.load_project_data({
        "reservoir_data": res_with_1d_perm,
        "pvt_properties": pvt_sample,
        "manual_inputs": manual_in,
        "well_data_list": wells_sample,
    })

    # Check UI widgets were restored
    assert float(widget.manual_inputs_widgets['area'].get_value()) == 500.0
    assert float(widget.manual_inputs_widgets['thickness'].get_value()) == 70.0
    assert float(widget.manual_inputs_widgets['perm'].get_value()) == 125.0
    assert float(widget.manual_inputs_widgets['dip_angle'].get_value()) == 3.5
    assert float(widget.manual_inputs_widgets['density_contrast'].get_value()) == 0.28
    assert float(widget.manual_inputs_widgets['interfacial_tension'].get_value()) == 19.5
    assert float(widget.manual_inputs_widgets['api_gravity'].get_value()) == 41.0
    assert float(widget.manual_inputs_widgets['gas_viscosity_cp'].get_value()) == 0.019
    assert float(widget.manual_inputs_widgets['c7_plus_fraction'].get_value()) == 0.28
    assert float(widget.manual_inputs_widgets['co2_solubility_scm_per_bbl'].get_value()) == 75.0

    # Check well list widget populated
    assert widget.well_list_widget.count() == 2


def test_optimization_widget_loads_results(qapp):
    """Test OptimizationWidget adopts loaded results and updates plots."""
    config_mgr = ConfigManager()
    res, pvt = create_sample_models()
    engine = OptimizationEngine(reservoir=res, pvt=pvt)
    dummy_results = {
        "best_parameters": {"injection_rate": 8000.0, "wag_ratio": 1.2},
        "best_objective_value": 3.2e7,
        "convergence_history": [2.5e7, 3.0e7, 3.2e7],
        "detailed_metrics": {"oil_recovery_stb": 3.8e6, "co2_stored_tons": 7.2e5},
    }
    engine.results = dummy_results

    opt_widget = OptimizationWidget(config_manager=config_mgr, parent=None)
    opt_widget.update_engine(engine)

    assert opt_widget.current_results == dummy_results
    assert opt_widget.export_button.isEnabled()

    # Call update_graphs
    opt_widget.update_graphs(dummy_results)
    assert opt_widget.results_tabs.currentIndex() == 2  # Summary tab
