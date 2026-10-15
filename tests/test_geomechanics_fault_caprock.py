"""
Unit tests for GeomechanicsFaultModel and containment evaluation.

Verifies:
- Horizontal in-situ stress path evolution with reservoir pressure (gamma_h)
- Effective normal stress on critically oriented fault planes
- Mohr-Coulomb slip tendency and critical reactivation pore pressure
- Caprock tensile and shear safety margins
- Dynamic subsurface leakage fluxes when thresholds are breached
- EPA Class VI UIC 90% formation fracture pressure ceiling
"""

import pytest
from core.engine_surrogate.geomechanics_fault import (
    GeomechanicsFaultModel,
    GeomechanicalState,
)


def test_stress_path_evolution():
    """Verify that total horizontal stress increases with reservoir pore pressure."""
    geo = GeomechanicsFaultModel(
        initial_pressure_psi=2000.0,
        depth_ft=5000.0,
        overburden_gradient_psi_per_ft=1.0,
        horizontal_stress_ratio_k0=0.70,
        biot_coefficient=1.0,
        poissons_ratio=0.25
    )

    # Initial state at 2000 psi
    state_init: GeomechanicalState = geo.evaluate_state(current_pressure_psi=2000.0)
    assert state_init.sigma_v_psi == pytest.approx(5000.0, rel=1e-3)
    assert state_init.sigma_h_min_psi == pytest.approx(3500.0, rel=1e-3)

    # Pressure elevated to 2800 psi (delta_p = +800 psi)
    state_pressurized: GeomechanicalState = geo.evaluate_state(current_pressure_psi=2800.0)
    # With nu=0.25, gamma_h = ((1 - 2*0.25) / (1 - 0.25)) * 1.0 = 0.5 / 0.75 = 0.667
    expected_delta_sigma_h = 800.0 * (0.5 / 0.75)
    assert state_pressurized.sigma_h_min_psi > state_init.sigma_h_min_psi
    assert state_pressurized.sigma_h_min_psi == pytest.approx(3500.0 + expected_delta_sigma_h, rel=1e-2)

    # Effective stress must decrease as pore pressure rises
    assert state_pressurized.sigma_h_eff_psi < state_init.sigma_h_eff_psi


def test_mohr_coulomb_fault_slip_tendency():
    """Verify Mohr-Coulomb fault slip tendency increases with pressure toward reactivation."""
    geo = GeomechanicsFaultModel(
        initial_pressure_psi=2200.0,
        depth_ft=6000.0,
        fault_dip_deg=60.0,
        fault_friction_coefficient=0.60
    )

    # At normal initial pressure, fault should be stable (slip tendency < 0.60)
    res_normal: GeomechanicalState = geo.evaluate_state(current_pressure_psi=2200.0)
    assert res_normal.is_fault_reactivated is False
    assert res_normal.slip_tendency < 0.60
    assert res_normal.fault_leakage_rate_tonne_day == 0.0

    # Near critical reactivation pressure
    p_crit = res_normal.p_crit_fault_reactivation_psi
    assert p_crit > 2200.0, "Critical reactivation pressure must exceed normal pressure"

    # Severely overpressured reservoir (above critical pressure)
    res_overpressured: GeomechanicalState = geo.evaluate_state(current_pressure_psi=p_crit + 500.0)
    assert res_overpressured.slip_tendency > 0.60 or res_overpressured.pore_pressure_psi >= p_crit
    assert res_overpressured.is_fault_reactivated is True
    assert res_overpressured.fault_leakage_rate_tonne_day > 0.0, (
        "Slip-reactivated fault must yield non-zero geological leakage flux"
    )


def test_caprock_integrity_and_tensile_failure():
    """Verify caprock safety margins and leakage behavior under high injection pressure."""
    geo = GeomechanicsFaultModel(
        initial_pressure_psi=2500.0,
        depth_ft=5000.0,
        caprock_fracture_pressure_psi=4250.0,
        caprock_tensile_strength_psi=300.0,
        caprock_cohesion_psi=500.0,
        caprock_safety_factor=0.90
    )

    # Safe injection pressure below safe ceiling (4250 * 0.90 = 3825 psi)
    res_safe: GeomechanicalState = geo.evaluate_state(current_pressure_psi=3500.0)
    assert res_safe.is_caprock_breached is False
    assert res_safe.caprock_tensile_margin_psi > 0.0
    assert res_safe.caprock_shear_margin_psi > 0.0
    assert res_safe.caprock_leakage_rate_tonne_day == 0.0

    # Overpressure breaching fracture gradient (e.g. 4400 psia > 4250 psia)
    res_breach: GeomechanicalState = geo.evaluate_state(current_pressure_psi=4400.0)
    assert res_breach.is_caprock_breached is True
    assert res_breach.caprock_leakage_rate_tonne_day > 0.0, (
        "Breached caprock must produce dynamic leakage flux"
    )


def test_epa_class_vi_safe_ceiling():
    """Verify EPA Class VI UIC 90% fracture ceiling enforcement."""
    geo = GeomechanicsFaultModel(
        depth_ft=6000.0,
        caprock_fracture_pressure_psi=4800.0,
        caprock_safety_factor=0.90
    )

    assert geo.p_safe_ceiling == pytest.approx(0.90 * 4800.0, rel=1e-3)
    assert geo.p_safe_ceiling == pytest.approx(4320.0, abs=1.0)
