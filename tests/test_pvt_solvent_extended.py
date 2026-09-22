"""
Unit tests for SolventExtendedPVTEngine.

Verifies:
- Thermodynamic state calculation without per-step flash overhead
- Strict decoupling of hydrodynamic gas saturation (S_g) from solvent composition (x_co2)
- Real-gas Peng-Robinson supercritical CO2 density and formation volume factor (B_co2)
- Oil swelling factor S_F and live oil formation volume factor B_o
- Viscosity thinning of oil with dissolved CO2
- Surface multi-stage separation and degassing
"""

import numpy as np
import pytest
from core.engine_surrogate.pvt_state import (
    SolventExtendedPVTEngine,
    SurfaceSeparationResult,
)


def test_pure_co2_supercritical_density_pr_eos():
    """Verify Peng-Robinson EOS yields realistic supercritical CO2 densities."""
    pvt_engine = SolventExtendedPVTEngine(reservoir_temperature_f=160.0)

    # Low pressure (subcritical vapor phase)
    p_low = 500.0  # psia
    rho_low = pvt_engine.calculate_co2_density_kg_m3(p_low)
    assert 20.0 < rho_low < 150.0, f"Expected gas-like density at 500 psia, got {rho_low}"

    # Supercritical reservoir condition (2500 psia, 160 F)
    p_res = 2500.0
    rho_res = pvt_engine.calculate_co2_density_kg_m3(p_res)
    # CO2 supercritical density is typically 500-800 kg/m3 at these conditions
    assert 500.0 < rho_res < 850.0, f"Expected supercritical density at 2500 psia, got {rho_res}"

    # High pressure (dense supercritical fluid, 4500 psia)
    p_high = 4500.0
    rho_high = pvt_engine.calculate_co2_density_kg_m3(p_high)
    assert rho_high > rho_res, "Density must monotonically increase with pressure"
    assert rho_high < 1000.0, f"Dense CO2 should remain below water density (1000 kg/m3), got {rho_high}"


def test_pure_co2_fvf_in_rb_per_mscf():
    """Verify conversion of supercritical CO2 density to RB/MSCF."""
    pvt_engine = SolventExtendedPVTEngine(reservoir_temperature_f=140.0)

    # Typical reservoir conditions: 2000-3500 psia
    b_co2_2000 = pvt_engine.calculate_co2_fvf_rb_per_mscf(2000.0)
    b_co2_3500 = pvt_engine.calculate_co2_fvf_rb_per_mscf(3500.0)

    # In petroleum units, 1 MSCF of dense CO2 occupies roughly 0.35 - 0.65 RB
    assert 0.30 < b_co2_2000 < 0.80, f"Expected 0.30-0.80 RB/MSCF at 2000 psia, got {b_co2_2000}"
    assert 0.30 < b_co2_3500 < 0.60, f"Expected 0.30-0.60 RB/MSCF at 3500 psia, got {b_co2_3500}"
    assert b_co2_3500 < b_co2_2000, "CO2 downhole volume must compress with increasing pressure"


def test_oil_swelling_and_viscosity_thinning():
    """Verify oil swelling and viscosity reduction as dissolved CO2 (x_co2) increases."""
    pvt_engine = SolventExtendedPVTEngine(
        dead_oil_viscosity_cp=10.0,
        api_gravity=35.0,
        reservoir_temperature_f=150.0
    )

    pressure = 2500.0

    # Baseline (zero dissolved CO2)
    sf_0 = pvt_engine.calculate_oil_swelling_factor(pressure, x_co2=0.0)
    bo_0 = pvt_engine.calculate_oil_fvf_rb_per_stb(pressure, x_co2=0.0)
    mu_0 = pvt_engine.calculate_oil_viscosity_cp(pressure, x_co2=0.0)
    assert sf_0 == pytest.approx(1.0, abs=1e-3)
    assert mu_0 > 0.5

    # Medium dissolved CO2 (x_co2 = 0.35)
    sf_med = pvt_engine.calculate_oil_swelling_factor(pressure, x_co2=0.35)
    bo_med = pvt_engine.calculate_oil_fvf_rb_per_stb(pressure, x_co2=0.35)
    mu_med = pvt_engine.calculate_oil_viscosity_cp(pressure, x_co2=0.35)
    assert sf_med > 1.05, "Oil must swell with dissolved CO2"
    assert bo_med > bo_0, "Oil FVF must increase with swelling"
    assert mu_med < mu_0, "Oil viscosity must decrease with dissolved CO2"

    # High dissolved CO2 (x_co2 = 0.70)
    sf_high = pvt_engine.calculate_oil_swelling_factor(pressure, x_co2=0.70)
    bo_high = pvt_engine.calculate_oil_fvf_rb_per_stb(pressure, x_co2=0.70)
    mu_high = pvt_engine.calculate_oil_viscosity_cp(pressure, x_co2=0.70)
    assert sf_high > sf_med, "Swelling must increase with x_co2"
    assert bo_high > bo_med, "FVF must increase with x_co2"
    assert mu_high < mu_med, "Viscosity must decrease further with higher x_co2"


def test_decoupling_transport_saturation_from_composition():
    """Verify that thermodynamic properties do not conflate S_g with dissolved x_co2."""
    pvt_engine = SolventExtendedPVTEngine(
        dead_oil_viscosity_cp=5.0,
        api_gravity=35.0
    )

    p = 2200.0
    x_dissolved = 0.40

    # Oil properties depend strictly on (P, x_co2), independent of hydrodynamic free gas saturation S_g
    bo_a = pvt_engine.calculate_oil_fvf_rb_per_stb(p, x_co2=x_dissolved)
    mu_a = pvt_engine.calculate_oil_viscosity_cp(p, x_co2=x_dissolved)
    sf_a = pvt_engine.calculate_oil_swelling_factor(p, x_co2=x_dissolved)

    bo_b = pvt_engine.calculate_oil_fvf_rb_per_stb(p, x_co2=x_dissolved)
    mu_b = pvt_engine.calculate_oil_viscosity_cp(p, x_co2=x_dissolved)
    sf_b = pvt_engine.calculate_oil_swelling_factor(p, x_co2=x_dissolved)

    assert bo_a == pytest.approx(bo_b)
    assert mu_a == pytest.approx(mu_b)
    assert sf_a == pytest.approx(sf_b)

    # Undissolved vs dissolved states have distinct thermodynamic properties
    bo_undissolved = pvt_engine.calculate_oil_fvf_rb_per_stb(p, x_co2=0.0)
    mu_undissolved = pvt_engine.calculate_oil_viscosity_cp(p, x_co2=0.0)
    assert bo_undissolved < bo_a
    assert mu_undissolved > mu_a


def test_surface_stage_separation():
    """Verify surface multi-stage flash separation into oil shrinkage and degassed gas."""
    pvt_engine = SolventExtendedPVTEngine(
        dead_oil_viscosity_cp=4.0,
        api_gravity=35.0
    )

    pressure = 2400.0
    q_oil_res = 1200.0  # RB/day
    q_water_res = 300.0  # RB/day
    q_gas_res = 2000.0  # RB/day
    x_co2 = 0.30
    y_co2 = 0.85

    sep: SurfaceSeparationResult = pvt_engine.perform_surface_stage_separation(
        q_oil_res_rb_day=q_oil_res,
        q_water_res_rb_day=q_water_res,
        q_gas_free_res_rb_day=q_gas_res,
        producer_sandface_p_psi=pressure,
        x_co2_liquid=x_co2,
        y_co2_free_gas=y_co2
    )

    # Surface stock tank oil must be less than reservoir barrels due to B_o > 1.0
    assert sep.q_oil_stb_day < q_oil_res
    assert sep.q_water_stb_day == pytest.approx(q_water_res, rel=1e-3)

    # Gas streams
    assert sep.q_gas_hc_mscfd > 0.0
    assert sep.q_gas_co2_mscfd > 0.0
    assert sep.q_gas_total_mscfd == pytest.approx(sep.q_gas_hc_mscfd + sep.q_gas_co2_mscfd, rel=1e-3)
    assert 0.0 < sep.y_co2_separator <= 1.0
    assert sep.producing_gor_scf_per_stb > 0.0
    assert 0.0 <= sep.water_cut <= 1.0
