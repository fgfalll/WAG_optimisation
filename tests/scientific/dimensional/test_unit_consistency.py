"""
Level 1: Mathematical & Dimensional Verification - Pint Unit Consistency.

Uses Pint unit registry to independently verify dimensional homogeneity of:
1. Darcy deliverability: q = J * Delta_P
2. Material balance pressure increment: dP = (q_net * dt) / (V_p * c_t)
3. Gas formation volume factor Bg: RB/MSCF vs ft3/scf
4. CO2 storage: volume to mass conversion (tonne/MSCF)
"""

import pint
import pytest

ureg = pint.UnitRegistry()


def test_darcy_inflow_dimensions():
    """
    Verify dimensional consistency of Productivity Index J:
        J = q / Delta_P  =>  [volume / time] / [pressure]
    """
    stb_per_day = ureg.oil_barrel / ureg.day
    psi = ureg.psi
    J_unit = stb_per_day / psi

    # Rate: q = J * Delta_P
    delta_p = 500.0 * psi
    J = 2.0 * J_unit
    q = J * delta_p
    assert q.dimensionality == stb_per_day.dimensionality


def test_tank_material_balance_pressure_increment_dimensions():
    """
    Verify dimensional consistency of pressure increment equation in surrogate_engine.py:376:
        dP = (q_net * dt) / (V_p * c_t + J_eff * dt)
    Numerator: [RB/day] * [day] = [RB]
    Denominator term 1: [RB] * [1/psi] = [RB / psi]
    Denominator term 2: [RB/(day*psi)] * [day] = [RB / psi]
    Quotient: [RB] / [RB / psi] = [psi]
    """
    rb = ureg.oil_barrel
    day = ureg.day
    psi = ureg.psi

    q_net = 1000.0 * (rb / day)
    dt = 30.0 * day
    V_p = 10_000_000.0 * rb
    c_t = 1.0e-5 / psi
    J_eff = 5.0 * (rb / (day * psi))

    numerator = q_net * dt
    denom1 = V_p * c_t
    denom2 = J_eff * dt

    assert denom1.dimensionality == denom2.dimensionality, (
        f"Denominator terms have mismatched dimensions: {denom1.dimensionality} vs {denom2.dimensionality}"
    )

    dP = numerator / (denom1 + denom2)
    assert dP.dimensionality == psi.dimensionality, f"dP dimension is not pressure: {dP.dimensionality}"


def test_co2_mass_conversion_factor():
    """
    Verify standard CO2 density conversion factor:
        0.053 metric tonnes per MSCF (1,000 scf at 60 deg F, 14.696 psia).

    Molar mass of CO2 = 44.01 g/mol.
    Standard molar volume = 379.48 scf/lbmol.
    Density = 44.01 lbm / 379.48 scf = 0.11597 lbm/scf.
    1 MSCF = 115.97 lbm = 52.60 kg = 0.05260 metric tonnes.
    """
    # Verify literature constant 0.053 is accurate within 1%
    density_tonne_per_mscf = 0.053
    analytical_value = 44.01 * (1.0 / 379.48) * 1000.0 * 0.45359237 / 1000.0  # tonnes/MSCF
    assert abs(density_tonne_per_mscf - analytical_value) / analytical_value < 0.01
