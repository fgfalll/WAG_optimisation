"""
Unit tests for RunDataExporter
==============================
Verifies generation of all 9 standardized AI-agent and human export artifacts:
- README.md (Entrypoint, health status, warnings/errors registry)
- run_manifest.json (Lightweight machine-readable manifest)
- run_evaluation_report.md (Technical markdown report)
- cash_flows_yearly.csv (Financial schedule)
- convergence_history.csv (Optimization progression)
- summary_yearly.csv (Standardized yearly profiles with units)
- summary_monthly.csv (Monthly profiles when present)
- results_summary.txt (Clean human text without array dumps)
- full_run_data.json (Cleaned full dataset)
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from utils.run_exporter import RunDataExporter


@pytest.fixture
def mock_run_data():
    """Generate minimal valid mock run results and input parameters."""
    n_years = 5
    years = np.arange(1, n_years + 1)
    oil = np.array([100000.0, 90000.0, 80000.0, 70000.0, 60000.0])
    co2_purch = np.array([50000.0, 40000.0, 30000.0, 20000.0, 10000.0])
    co2_rec = np.array([0.0, 10000.0, 20000.0, 30000.0, 40000.0])

    results = {
        "method": "genetic_algorithm",
        "objective_name": "npv",
        "objective_function_value": 4500000.0,
        "recovery_factor": 0.42,
        "npv": 5200000.0,
        "optimized_params_final_clipped": {
            "pressure": 2400.0,
            "rate": 4500.0,
            "plateau_duration_fraction": 0.75,
            "wellbore_pressure": 1600.0,
        },
        "final_metrics": {
            "npv": 5200000.0,
            "recovery_factor": 0.42,
            "breakthrough_time_years": 1.85,
            "breakthrough_impact_factor": 0.865,
            "storage_efficiency": 0.65,
            "co2_utilization": 0.35,
            "ecology_compliant": True,
        },
        "optimized_profiles": {
            "yearly_oil_stb": oil,
            "annual_oil_stb": oil,
            "yearly_water_stb": oil * 0.1,
            "yearly_total_gas_mscf": co2_purch + co2_rec,
            "yearly_co2_produced_mscf": co2_rec,
            "yearly_hc_gas_produced_mscf": co2_rec * 0.2,
            "yearly_co2_purchased_mscf": co2_purch,
            "yearly_co2_recycled_mscf": co2_rec,
            "yearly_co2_injected_mscf": co2_purch + co2_rec,
            "yearly_water_injected_bbl": oil * 0.5,
            "yearly_pressure": np.array([2400.0, 2380.0, 2360.0, 2350.0, 2340.0]),
        },
        "material_balance_analysis": {
            "summary_statistics": {
                "total_injected_tonne": 15000.0,
                "total_produced_tonne": 5000.0,
                "total_recycled_tonne": 4900.0,
                "total_net_stored_tonne": 9950.0,
                "total_leakage_tonne": 50.0,
                "avg_storage_efficiency": 0.88,
            }
        },
        "ga_statistics": {
            "total_duration_seconds": 45.2,
            "total_evaluations": 150,
            "best_fitness_history": [1000000.0, 2500000.0, 4500000.0],
        },
    }

    input_parameters = {
        "General & Reservoir": {
            "OOIP (STB)": 1000000.0,
            "MMP (Calculated, psi)": 2100.0,
            "Average Porosity": 0.22,
            "Project Lifetime (years)": 5,
        },
        "Economic Parameters": {
            "oil_price_usd_per_bbl": 75.0,
            "co2_purchase_cost_usd_per_tonne": 45.0,
            "co2_recycle_cost_usd_per_tonne": 12.0,
            "co2_storage_credit_usd_per_tonne": 30.0,
            "operating_cost_usd_per_bbl": 6.0,
            "capex_usd": 1000000.0,
            "discount_rate_fraction": 0.10,
        },
        "EOR Parameters": {
            "co2_density_tonne_per_mscf": 0.053,
            "caprock_fracture_pressure_psi": 5000.0,
            "caprock_safety_factor": 0.90,
        },
    }

    return results, input_parameters


def test_export_generates_all_core_files(tmp_path, mock_run_data):
    """Ensure all required artifact files are generated with valid formatting."""
    results, inputs = mock_run_data

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    assert export_dir.is_dir()

    expected_files = [
        "README.md",
        "run_manifest.json",
        "run_evaluation_report.md",
        "cash_flows_yearly.csv",
        "convergence_history.csv",
        "summary_yearly.csv",
        "results_summary.txt",
        "full_run_data.json",
    ]

    for fname in expected_files:
        fpath = export_dir / fname
        assert fpath.exists(), f"Missing expected artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact {fname} is empty"


def test_run_manifest_content_and_schema(tmp_path, mock_run_data):
    """Verify run_manifest.json schema, boolean types, and key values."""
    results, inputs = mock_run_data

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    with open(export_dir / "run_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "2.0.0"
    assert manifest["metadata"]["optimizer_method"] == "genetic_algorithm"
    assert manifest["metadata"]["total_evaluations"] == 150

    kpis = manifest["key_performance_indicators"]
    assert kpis["economic_npv_usd"] == 5200000.0
    assert kpis["recovery_factor_percent"] == 42.0
    assert isinstance(kpis["ecology_compliant"], bool)
    assert kpis["ecology_compliant"] is True

    carbon = manifest["carbon_accounting_tonnes"]
    assert carbon["total_injected_tonne"] == 15000.0
    assert carbon["total_net_stored_tonne"] == 9950.0

    dvars = manifest["decision_variables"]
    assert "pressure" in dvars
    assert dvars["pressure"]["value"] == 2400.0
    assert dvars["pressure"]["unit"] == "psia"


def test_cash_flow_table_integrity(tmp_path, mock_run_data):
    """Verify year-by-year cashflow schedule and discounting calculations."""
    results, inputs = mock_run_data

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    df = pd.read_csv(export_dir / "cash_flows_yearly.csv")
    assert len(df) == 5
    assert list(df["Year"]) == [1, 2, 3, 4, 5]

    # Year 1 CAPEX check
    assert df.loc[0, "CAPEX_USD"] == 1000000.0
    assert df.loc[1, "CAPEX_USD"] == 0.0

    # Discount factor progression check (mid-year: (1.1)^(-0.5) ~ 0.9535)
    assert np.isclose(df.loc[0, "Discount_Factor"], 0.9535, atol=1e-3)
    assert df.loc[4, "Discount_Factor"] < df.loc[0, "Discount_Factor"]


def test_results_summary_has_no_numpy_array_dumps(tmp_path, mock_run_data):
    """Verify that results_summary.txt is human-readable and clean of raw numpy dumps."""
    results, inputs = mock_run_data

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    content = (export_dir / "results_summary.txt").read_text(encoding="utf-8")
    assert "=== CO₂ EOR OPTIMIZATION RUN SUMMARY ===" in content
    assert "Final Economic NPV: $5,200,000.00" in content
    assert "array([" not in content
    assert "np.float64" not in content


def test_readme_and_diagnostics_detection(tmp_path, mock_run_data):
    """Verify that README.md highlights warnings and health status."""
    results, inputs = mock_run_data

    # Introduce a geomechanical fracture violation
    results["optimized_params_final_clipped"]["pressure"] = 4800.0
    inputs["EOR Parameters"]["caprock_fracture_pressure_psi"] = 5000.0
    inputs["EOR Parameters"]["caprock_safety_factor"] = 0.90  # safe ceiling = 4500 psi

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    readme_content = (export_dir / "README.md").read_text(encoding="utf-8")
    assert "CRITICAL ISSUES DETECTED" in readme_content
    assert "GEO_FRAC_EXCEEDED" in readme_content


def test_export_against_real_session_archive(tmp_path):
    """Verify export against actual archived session run in logs/."""
    real_archive = Path("logs/Export-hybrid-ga-bo-20260916-112837/full_run_data.json")
    if not real_archive.exists():
        pytest.skip("Real session archive not found in logs/")

    with open(real_archive, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    export_dir = RunDataExporter.export(
        results=raw_data["full_results"],
        input_parameters=raw_data["input_parameters"],
        target_dir=tmp_path,
    )

    assert (export_dir / "README.md").exists()
    assert (export_dir / "run_manifest.json").exists()
    assert (export_dir / "run_evaluation_report.md").exists()
    assert (export_dir / "cash_flows_yearly.csv").exists()
    assert (export_dir / "convergence_history.csv").exists()
    assert (export_dir / "summary_yearly.csv").exists()
    assert (export_dir / "summary_monthly.csv").exists()  # Monthly profiles present!

    # Check that monthly CSV has 181 months (15 years)
    df_m = pd.read_csv(export_dir / "summary_monthly.csv")
    assert len(df_m) == 181
    assert "Oil_Production_STB" in df_m.columns
    assert "CO2_Injected_MSCF" in df_m.columns


def test_export_handles_bayesian_optimization_object(tmp_path, mock_run_data):
    """Ensure BayesianOptimization and other non-serializable objects do not crash export."""
    results, inputs = mock_run_data

    # Create a mock BayesianOptimization object
    class BayesianOptimization:
        def __init__(self):
            self.res = [{"target": 123.45, "params": {"x": 1.0}}]
            self.max = {"target": 123.45, "params": {"x": 1.0}}

    class ArbitraryUnserializable:
        pass

    results["bayes_opt_obj"] = BayesianOptimization()
    results["arbitrary_obj"] = ArbitraryUnserializable()
    results["method"] = "bayesian_gp"

    export_dir = RunDataExporter.export(
        results=results,
        input_parameters=inputs,
        target_dir=tmp_path,
    )

    full_json_path = export_dir / "full_run_data.json"
    assert full_json_path.exists()
    assert full_json_path.stat().st_size > 0

    with open(full_json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["full_results"]["bayes_opt_obj"] == "<BayesianOptimization not serialized>"
    assert loaded["full_results"]["arbitrary_obj"] == "<ArbitraryUnserializable not serialized>"

