"""
Unit tests for standardized 4-stream reservoir output delivery in SurrogateEngine.

Verifies:
- Stream 1: Crude Oil (Daily STB/d, cumulative STB, annual STB, monthly STB, oil FVF)
- Stream 2: Natural Gas (Hydrocarbon sales gas separate from CO2, solution gas GOR)
- Stream 3: Water (Produced brine, cumulative bbl, dynamic water cut)
- Stream 4: Injection Agent (Gross injected, fresh purchased, recycled, WAG water)
- Facility compressor capacity limit enforcement
- State and integrity profiles (pressure, sandface pressure, VRR, geomechanics)
"""

import numpy as np
import pytest
from core.data_models import (
    EORParameters,
    ReservoirData,
    FluidProperties,
    OperationalParameters,
    EconomicParameters,
)
from core.engine_surrogate.surrogate_engine import SurrogateEngine


@pytest.fixture
def base_scenario():
    """Build a standard CO2 EOR scenario fixture."""
    res = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={},
        ooip_stb=5_000_000.0,
        initial_pressure=2200.0,
        temperature=150.0,
        average_porosity=0.20,
        average_permeability=50.0,
        initial_water_saturation=0.25,
        thickness_ft=100.0,
        area_acres=640.0,
    )
    fluid = FluidProperties(
        oil_density_ref=850.0,
        oil_viscosity_ref=0.0035,
        water_density_ref=1000.0,
        water_viscosity_ref=0.0006,
        oil_fvf_ref=1.25,
        oil_fvf=1.25,
    )
    eor = EORParameters(
        injection_rate=5000.0,  # 5000 MSCFD
        recycle_compressor_capacity_mscfd=2000.0,  # 2000 MSCFD limit
        facility_availability=0.95
    )
    ops = OperationalParameters(
        project_lifetime_years=10
    )
    econ = EconomicParameters(
        oil_price_usd_per_bbl=70.0,
        co2_purchase_cost_usd_per_tonne=30.0,
        co2_recycle_cost_usd_per_tonne=10.0
    )
    return res, fluid, eor, ops, econ


def test_standardized_4_streams_in_surrogate_results(base_scenario):
    """Verify that evaluate_scenario delivers all 4 fluid streams and state profiles."""
    res, fluid, eor, ops, econ = base_scenario
    engine = SurrogateEngine()
    results = engine.evaluate_scenario(
        reservoir_data=res,
        eor_params=eor,
        operational_params=ops,
        economic_params=econ
    )
    assert "profiles" in results
    p = results["profiles"]

    # Stream 1: Crude Oil
    assert "oil_profile" in p or "oil_production_rate" in p
    assert "cumulative_oil_bbl" in p
    assert "annual_oil_stb" in p
    assert "monthly_oil_stb" in p
    assert "oil_fvf_profile" in p
    assert np.all(p["oil_profile"] >= 0.0)
    assert np.all(p["oil_fvf_profile"] >= 1.0)
    assert p["cumulative_oil_bbl"][-1] > 0.0

    # Stream 2: Natural Gas (Hydrocarbon Sales Gas)
    assert "hydrocarbon_gas_sales_mscfd" in p
    assert "solution_gas_profile" in p
    assert "annual_hydrocarbon_gas_sales_mscf" in p
    assert "cumulative_hydrocarbon_gas_sales_mscf" in p
    assert np.all(p["hydrocarbon_gas_sales_mscfd"] >= 0.0)
    assert p["cumulative_hydrocarbon_gas_sales_mscf"][-1] > 0.0

    # Stream 3: Water (Formation Brine)
    assert "water_profile" in p or "water_production_rate" in p
    assert "water_cut_profile" in p
    assert "cumulative_water_bbl" in p
    assert "annual_water_bbl" in p
    assert np.all((p["water_cut_profile"] >= 0.0) & (p["water_cut_profile"] <= 1.0))

    # Stream 4: Injection Agent (CO2 & WAG Water)
    assert "injection_profile" in p
    assert "co2_purchased_mscfd" in p
    assert "co2_recycled_mscfd" in p
    assert "co2_gas_profile" in p
    assert "annual_co2_purchased_mscf" in p
    assert "annual_co2_recycled_mscf" in p
    assert "annual_co2_injected_mscf" in p

    # State & Integrity tracks
    assert "pressure" in p
    assert "sandface_injection_pressure" in p
    assert "vrr_local" in p
    assert "fault_slip_tendency" in p
    assert "caprock_tensile_margin" in p
    assert "caprock_shear_margin" in p
    assert "x_co2_liquid" in p
    assert "y_co2_vapor" in p
    assert "saturation_oil" in p
    assert "saturation_water" in p
    assert "saturation_gas" in p


def test_compressor_capacity_bottleneck(base_scenario):
    """Verify that recycle compressor capacity bottleneck is strictly respected."""
    res, fluid, eor, ops, econ = base_scenario
    # Set compressor capacity to 1000 MSCFD with 95% availability -> max 950 MSCFD
    eor.recycle_compressor_capacity_mscfd = 1000.0
    eor.facility_availability = 0.95
    max_allowed_recycled = 1000.0 * 0.95

    engine = SurrogateEngine()
    results = engine.evaluate_scenario(
        reservoir_data=res,
        eor_params=eor,
        operational_params=ops,
        economic_params=econ
    )
    p = results["profiles"]

    recycled_rates = p["co2_recycled_mscfd"]
    assert np.all(recycled_rates <= max_allowed_recycled + 1e-3), (
        f"Recycle rate exceeded compressor throughput bottleneck: max={np.max(recycled_rates)}"
    )


def test_closed_loop_injection_balance(base_scenario):
    """Verify gross injected rate equals purchased plus recycled rates."""
    res, fluid, eor, ops, econ = base_scenario
    engine = SurrogateEngine()
    results = engine.evaluate_scenario(
        reservoir_data=res,
        eor_params=eor,
        operational_params=ops,
        economic_params=econ
    )
    p = results["profiles"]

    injected = p["injection_profile"]
    purchased = p["co2_purchased_mscfd"]
    recycled = p["co2_recycled_mscfd"]

    np.testing.assert_allclose(
        injected,
        purchased + recycled,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Gross injected CO2 must equal purchased + recycled across all timesteps"
    )
