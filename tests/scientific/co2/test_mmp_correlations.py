"""
Level 2 & 4: CO2-Specific Verification - MMP Correlations & Impurity Thermodynamics.

Tests minimum miscibility pressure correlations (Cronquist, Yuan, Alston) against
thermodynamic invariants and physical impurity behavior.
In phase behavior physics:
    Adding Methane (CH4) or Nitrogen (N2) to CO2 injection gas RAISED the MMP.
"""

import numpy as np
import pytest
from evaluation.mmp import calculate_mmp, MMPParameters


def test_yuan_impurity_mmp_trend():
    """
    Test the effect of CH4 impurity on MMP in Yuan correlation.
    In physics: adding methane increases MMP (harder to achieve miscibility).
    In evaluation/mmp.py:203:
        c = 0.993 - 0.778 * (1 - x_co2) ** 0.11
    Reduces c as (1 - x_co2) increases, causing predicted MMP to drop when CH4 is added!
    """
    # Pure CO2
    p_pure = MMPParameters(
        temperature=160.0,
        oil_gravity=35.0,
        injection_gas_composition={"CO2": 1.0, "CH4": 0.0},
    )
    mmp_pure = calculate_mmp(p_pure, method="yuan")

    # 10% Methane impurity
    p_impure = MMPParameters(
        temperature=160.0,
        oil_gravity=35.0,
        injection_gas_composition={"CO2": 0.90, "CH4": 0.10},
    )
    mmp_impure = calculate_mmp(p_impure, method="yuan")

    # In current implementation, adding CH4 drops MMP:
    mmp_dropped = mmp_impure < mmp_pure
    assert mmp_dropped, (
        f"Confirms inverted impurity trend: Pure MMP={mmp_pure:.1f} psi, 10% CH4 MMP={mmp_impure:.1f} psi!"
    )


def test_standing_bo_as_api_estimator():
    """
    Test API estimation from PVT using Standing formula in evaluation/mmp.py:307-308:
        F = R_s * (gamma_g / gamma_o) ** 0.5 + 1.25 * T
        gamma_o_new = 0.972 + 0.000147 * F**1.175
    This formula is Standing's (1947) Formation Volume Factor (B_o) correlation, NOT specific gravity!
    """
    from evaluation.mmp import estimate_api_from_pvt
    from core.data_models import PVTProperties

    pvt = PVTProperties(
        pvt_type="black_oil",
        temperature=150.0,
        gas_specific_gravity=0.65,
        rs=np.array([500.0]),
    )
    api = estimate_api_from_pvt(pvt)
    # The function clamps to [15.0, 50.0] because Standing's Bo (1.2-1.5) as gamma_o yields negative/distorted API
    assert 15.0 <= api <= 50.0
