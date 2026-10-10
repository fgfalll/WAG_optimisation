"""
Physical Invariants and Regression Prevention Test Suite.

Ensures that:
1. Gross CO2 utilization falls within established SPE/DOE benchmarks (3.0 - 15.0 MSCF/STB).
2. Closed-loop carbon mass balance is strictly preserved across all annual/monthly profiles.
3. CO2 recycling actively displaces purchased CO2 post-breakthrough in both profiles and cash flows.
4. Exported CSVs (summary_yearly.csv, cash_flows_yearly.csv) maintain cross-table consistency.
5. Geomechanical safety ceilings (EPA Class VI UIC standard) auto-clamp injection pressure.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.data_models import (
    EORParameters,
    ReservoirData,
    OperationalParameters,
    EconomicParameters,
    PVTProperties,
    EOSModelParameters,
)
from core.optimisation_engine import OptimizationEngine
from utils.run_exporter import RunDataExporter


@pytest.fixture
def standard_setup(eos_model_params):
    """Create standard reservoir and operational parameters for testing."""
    res_data = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={},
        area_acres=160.0,
        thickness_ft=50.0,
        average_porosity=0.18,
        average_permeability=50.0,
        initial_water_saturation=0.25,
        initial_pressure=3000.0,
        temperature=150.0,
        ooip_stb=6_982_200.0,
        oil_fvf=1.2,
        eos_model=eos_model_params,
    )
    res_data.ooip_stb = res_data.calculate_ooip_from_physics()

    eor_params = EORParameters(
        injection_rate=5000.0,
        target_pressure_psi=3000.0,
        max_pressure_psi=4950.0,
        caprock_fracture_pressure_psi=5500.0,
        caprock_safety_factor=0.90,
        co2_recycling_fraction=0.95,
    )

    op_params = OperationalParameters(
        project_lifetime_years=15,
        time_resolution="monthly",
    )

    econ_params = EconomicParameters(
        oil_price_usd_per_bbl=70.0,
        co2_purchase_cost_usd_per_tonne=35.0,
        co2_recycle_cost_usd_per_tonne=10.0,
    )

    return res_data, eor_params, op_params, econ_params


class TestPhysicalInvariants:
    """Test physical invariants that must never be broken by numerical approximations."""

    def test_geomechanical_safety_clamping(self):
        """Verify that max_pressure_psi is automatically clamped to 90% of fracture ceiling."""
        eor = EORParameters(
            max_pressure_psi=6000.0,
            caprock_fracture_pressure_psi=5500.0,
            caprock_safety_factor=0.90,
        )
        # Expected safe ceiling is 0.90 * 5500 = 4950.0 psi
        assert eor.max_pressure_psi <= 4950.0
        assert np.isclose(eor.max_pressure_psi, 4950.0, atol=1e-3)

    def test_gross_co2_utilization_benchmark_bounds(self, standard_setup):
        """Verify that gross utilization is within realistic SPE bounds (3.0 - 15.0 MSCF/STB)."""
        res_data, eor_params, op_params, econ_params = standard_setup
        engine = OptimizationEngine(
            reservoir=res_data,
            pvt=PVTProperties(),
            eor_params_instance=eor_params,
            operational_params_instance=op_params,
            economic_params_instance=econ_params,
        )

        results = engine.run_single_simulation()
        assert results is not None
        assert "gross_utilization_mscf_per_stb" in results

        gur = results["gross_utilization_mscf_per_stb"]
        rf = results["recovery_factor"]

        # SPE benchmark bounds: gross utilization should be realistic
        assert 3.0 <= gur <= 15.0, f"Gross utilization {gur:.2f} MSCF/STB is out of physical bounds [3.0, 15.0]"
        assert 0.05 <= rf <= 0.50, f"Recovery factor {rf:.2%} is out of physical expected range for throughput"

    def test_closed_loop_co2_recycling_mass_balance(self, standard_setup):
        """Verify that Injected == Purchased + Recycled across all yearly and monthly profiles."""
        res_data, eor_params, op_params, econ_params = standard_setup
        engine = OptimizationEngine(
            reservoir=res_data,
            pvt=PVTProperties(),
            eor_params_instance=eor_params,
            operational_params_instance=op_params,
            economic_params_instance=econ_params,
        )

        results = engine.run_single_simulation()
        profiles = results["optimized_profiles"]

        # 1. Yearly profile conservation
        yearly_purch = profiles["yearly_co2_purchased_mscf"]
        yearly_rec = profiles["yearly_co2_recycled_mscf"]
        yearly_inj = profiles["yearly_co2_injected_mscf"]
        yearly_prod = profiles["yearly_co2_produced_mscf"]

        assert len(yearly_purch) == op_params.project_lifetime_years
        assert len(yearly_rec) == op_params.project_lifetime_years
        assert len(yearly_inj) == op_params.project_lifetime_years

        # Purchased + Recycled must equal Gross Injected for every single year
        np.testing.assert_allclose(
            yearly_purch + yearly_rec,
            yearly_inj,
            rtol=1e-4,
            err_msg="Yearly CO2 conservation violated: Purchased + Recycled != Injected",
        )

        # 2. Recycling gating: when produced CO2 > 0, recycled CO2 must be > 0
        has_prod = yearly_prod > 10.0  # MSCF
        if np.any(has_prod):
            assert np.all(yearly_rec[has_prod] > 0.0), "Recycled CO2 is zero despite active CO2 production!"
            assert np.all(yearly_purch[has_prod] < yearly_inj[has_prod]), "Purchased CO2 was not reduced by recycling!"

        # 3. Monthly profile conservation
        monthly_purch = profiles["monthly_co2_purchased_mscf"]
        monthly_rec = profiles["monthly_co2_recycled_mscf"]
        monthly_inj = profiles["monthly_co2_injected_mscf"]

        np.testing.assert_allclose(
            monthly_purch + monthly_rec,
            monthly_inj,
            rtol=1e-4,
            err_msg="Monthly CO2 conservation violated: Purchased + Recycled != Injected",
        )

    def test_export_csv_cross_table_consistency(self, standard_setup):
        """Verify that exported summary_yearly.csv and cash_flows_yearly.csv contain active recycling."""
        res_data, eor_params, op_params, econ_params = standard_setup
        engine = OptimizationEngine(
            reservoir=res_data,
            pvt=PVTProperties(),
            eor_params_instance=eor_params,
            operational_params_instance=op_params,
            economic_params_instance=econ_params,
        )

        results = engine.run_single_simulation()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = RunDataExporter(output_root_dir=tmpdir)
            export_path = exporter.export_run(
                results=results,
                engine=engine,
                input_parameters={"General & Reservoir": {}, "EOR Parameters": {}},
                plots_generator=None,
            )

            summary_csv = Path(export_path) / "summary_yearly.csv"
            cash_flows_csv = Path(export_path) / "cash_flows_yearly.csv"

            assert summary_csv.exists(), "summary_yearly.csv was not generated"
            assert cash_flows_csv.exists(), "cash_flows_yearly.csv was not generated"

            df_summary = pd.read_csv(summary_csv)
            df_cash = pd.read_csv(cash_flows_csv)

            # Check column existence
            assert "CO2_Purchased_MSCF" in df_summary.columns
            assert "CO2_Recycled_MSCF" in df_summary.columns
            assert "CO2_Injected_MSCF" in df_summary.columns

            # Conservation in CSV
            np.testing.assert_allclose(
                df_summary["CO2_Purchased_MSCF"] + df_summary["CO2_Recycled_MSCF"],
                df_summary["CO2_Injected_MSCF"],
                rtol=1e-4,
                err_msg="summary_yearly.csv violated Purchased + Recycled == Injected",
            )

            # Verify that recycling is non-zero in summary_yearly.csv post-breakthrough
            if df_summary["CO2_Produced_MSCF"].sum() > 0:
                assert df_summary["CO2_Recycled_MSCF"].sum() > 0, "summary_yearly.csv has 0.0 CO2_Recycled_MSCF despite production!"

            # Verify cash flows match summary
            np.testing.assert_allclose(
                df_cash["CO2_Purchased_MSCF"],
                df_summary["CO2_Purchased_MSCF"],
                rtol=1e-4,
                err_msg="cash_flows_yearly.csv CO2_Purchased does not match summary_yearly.csv",
            )
