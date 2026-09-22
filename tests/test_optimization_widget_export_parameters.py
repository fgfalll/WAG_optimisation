"""
Unit tests for comprehensive input parameter capture in OptimizationWidget and RunDataExporter.
"""

import pytest
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from core.data_models import (
    ReservoirData,
    PVTProperties,
    EORParameters,
    EconomicParameters,
    OperationalParameters,
    CO2StorageParameters,
    WellData,
)
from core.optimisation_engine import OptimizationEngine
from utils.config_manager import ConfigManager
from utils.run_exporter import RunDataExporter
from ui.optimization_widget import OptimizationWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_engine():
    res = ReservoirData(
        grid={"ACTNUM": np.ones((5, 5, 3))},
        pvt_tables={},
        ooip_stb=25_000_000.0,
        initial_pressure=3200.0,
        temperature=165.0,
        thickness_ft=65.0,
        area_acres=640.0,
        average_porosity=0.22,
        average_permeability=85.0,
        rock_compressibility=3.5e-6,
    )
    pvt = PVTProperties(
        temperature=165.0,
        gas_specific_gravity=0.75,
        api_gravity=38.0,
        oil_viscosity_cp=1.8,
        co2_viscosity=np.array([0.03]),
    )
    eor = EORParameters(
        injection_rate=8000.0,
        target_pressure_psi=3100.0,
        caprock_fracture_pressure_psi=5200.0,
    )
    econ = EconomicParameters(
        oil_price_usd_per_bbl=75.0,
        co2_purchase_cost_usd_per_tonne=45.0,
        capex_usd=25_000_000.0,
    )
    ops = OperationalParameters(
        project_lifetime_years=12,
        time_resolution="monthly",
    )
    co2 = CO2StorageParameters(
        structural_trapping_factor=0.25,
        residual_trapping_factor=0.35,
    )
    well = WellData(
        name="INJ-01",
        depths=np.array([5000.0, 5050.0]),
        properties={"perm": np.array([85.0, 85.0])},
        units={"perm": "mD"},
        metadata={"type": "injector"},
    )
    well_prod = WellData(
        name="PROD-01",
        depths=np.array([5000.0, 5050.0]),
        properties={"perm": np.array([85.0, 85.0])},
        units={"perm": "mD"},
        metadata={"type": "producer"},
    )

    engine = OptimizationEngine(
        reservoir=res,
        pvt=pvt,
        eor_params_instance=eor,
        economic_params_instance=econ,
        operational_params_instance=ops,
        co2_storage_params_instance=co2,
        well_data_list=[well, well_prod],
    )
    return engine


def test_optimization_widget_get_current_input_parameters(qapp, sample_engine):
    """Verify that OptimizationWidget._get_current_input_parameters extracts all categories."""
    config_mgr = ConfigManager()
    widget = OptimizationWidget(config_manager=config_mgr)
    widget.set_engine(sample_engine)

    params = widget._get_current_input_parameters()

    # Verify all expected categories exist
    expected_categories = [
        "Optimization Setup",
        "Optimization Search Bounds",
        "General & Reservoir",
        "Reservoir Parameters",
        "Fluid & PVT Properties",
        "Operational Parameters",
        "EOR Parameters",
        "Economic Parameters",
        "CO2 Storage Parameters",
        "Well Configuration",
        "Genetic Algorithm",
        "Bayesian Optimization",
    ]
    for cat in expected_categories:
        assert cat in params, f"Category '{cat}' missing from _get_current_input_parameters"
        assert len(params[cat]) > 0, f"Category '{cat}' is unexpectedly empty"

    # Verify specific enriched properties
    gen = params["General & Reservoir"]
    assert gen["OOIP (STB)"] == 25_000_000.0
    assert gen["Project Lifetime (years)"] == 12
    assert gen["Initial Pressure (psi)"] == 3200.0
    assert gen["Temperature (°F)"] == 165.0
    assert gen["Average Permeability (mD)"] == 85.0
    assert gen["Thickness (ft)"] == 65.0

    # Verify search bounds
    bounds = params["Optimization Search Bounds"]
    assert any("pressure" in k for k in bounds), "Pressure bounds missing"
    assert any("rate" in k for k in bounds), "Rate bounds missing"

    # Verify well configuration
    wells = params["Well Configuration"]
    assert wells["total_wells"] == 2
    assert wells["injector_count"] == 1
    assert wells["producer_count"] == 1
    assert "INJ-01 (injector)" in wells["wells"]
    assert "PROD-01 (producer)" in wells["wells"]


def test_run_exporter_with_enriched_parameters(tmp_path, qapp, sample_engine):
    """Verify RunDataExporter properly formats and records the enriched input parameters."""
    config_mgr = ConfigManager()
    widget = OptimizationWidget(config_manager=config_mgr)
    widget.set_engine(sample_engine)

    inputs = widget._get_current_input_parameters()

    results = {
        "method": "single_simulation",
        "objective_name": "npv",
        "objective_function_value": 350_000_000.0,
        "final_metrics": {
            "npv": 350_000_000.0,
            "recovery_factor": 0.40,
            "breakthrough_time_years": 2.5,
            "co2_utilization": 0.08,
            "storage_efficiency": 0.95,
            "ecology_compliant": True,
        },
        "optimized_params": {
            "rate": 8000.0,
            "pressure": 3100.0,
            "plateau_duration_fraction": 0.25,
            "ramp_up_fraction": 0.1,
            "wellbore_pressure": 1600.0,
            "max_production_rate_stbd": 12000.0,
        },
        "optimized_profiles": {
            "yearly_oil_stb": [1_000_000.0] * 12,
            "yearly_water_stb": [100_000.0] * 12,
            "yearly_total_gas_mscf": [500_000.0] * 12,
            "yearly_co2_produced_mscf": [50_000.0] * 12,
            "yearly_hc_gas_produced_mscf": [450_000.0] * 12,
            "time_vector": np.linspace(0, 12 * 365.25, 145),
        },
        "material_balance_analysis": {
            "summary_statistics": {
                "total_injected_tonne": 500_000.0,
                "total_gross_injected_tonne": 500_000.0,
                "total_purchased_tonne": 480_000.0,
                "total_recycled_tonne": 20_000.0,
                "total_produced_tonne": 22_000.0,
                "total_net_stored_tonne": 475_000.0,
                "total_leakage_tonne": 3_000.0,
                "avg_storage_efficiency": 0.95,
            }
        },
    }

    export_dir = RunDataExporter.export(
        results=results,
        engine=sample_engine,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    assert (export_dir / "full_run_data.json").exists()
    assert (export_dir / "run_manifest.json").exists()
    assert (export_dir / "results_summary.txt").exists()
    assert (export_dir / "run_evaluation_report.md").exists()

    # Read and inspect full_run_data.json
    import json
    with open(export_dir / "full_run_data.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)

    assert "input_parameters" in full_data
    exported_inputs = full_data["input_parameters"]
    assert "Optimization Setup" in exported_inputs
    assert "Optimization Search Bounds" in exported_inputs
    assert "Reservoir Parameters" in exported_inputs
    assert "Fluid & PVT Properties" in exported_inputs
    assert "Operational Parameters" in exported_inputs
    assert "Well Configuration" in exported_inputs

    # Inspect results_summary.txt
    txt_content = (export_dir / "results_summary.txt").read_text(encoding="utf-8")
    assert "=== SIMULATION & OPTIMIZATION INPUT PARAMETERS ===" in txt_content
    assert "--- Optimization & Engine Setup ---" in txt_content
    assert "--- Parameter Search Space Bounds ---" in txt_content
    assert "--- Reservoir Characterization ---" in txt_content
    assert "--- Well Configuration ---" in txt_content

    # Inspect run_evaluation_report.md
    report_content = (export_dir / "run_evaluation_report.md").read_text(encoding="utf-8")
    assert "**Initial Reservoir Pressure**" in report_content
    assert "**Formation Net Thickness**" in report_content
    assert "**Well Infrastructure**" in report_content
