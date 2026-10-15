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
    Verifies that the impurity penalty factor correctly increases MMP.
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

    # Adding CH4 must increase MMP (thermodynamic invariant):
    assert mmp_impure > mmp_pure, (
        f"MMP must increase with CH4 impurity: Pure MMP={mmp_pure:.1f} psi, 10% CH4 MMP={mmp_impure:.1f} psi"
    )


def test_alston_impurity_mmp_trend():
    """
    Test the effect of CH4 impurity on MMP in Alston correlation.
    In physics: adding methane lowers pseudo-critical temperature and increases MMP.
    """
    p_pure = MMPParameters(
        temperature=160.0,
        oil_gravity=35.0,
        c7_plus_mw=190.0,
        injection_gas_composition={"CO2": 1.0, "CH4": 0.0},
    )
    mmp_pure = calculate_mmp(p_pure, method="alston")

    p_impure = MMPParameters(
        temperature=160.0,
        oil_gravity=35.0,
        c7_plus_mw=190.0,
        injection_gas_composition={"CO2": 0.90, "CH4": 0.10},
    )
    mmp_impure = calculate_mmp(p_impure, method="alston")

    assert mmp_impure > mmp_pure, (
        f"Alston MMP must increase with CH4: Pure MMP={mmp_pure:.1f} psi, 10% CH4 MMP={mmp_impure:.1f} psi"
    )


def test_standing_bo_as_api_estimator():
    """
    Test API estimation from PVT properties.
    Ensures estimated API gravity falls in the realistic range (30-40 °API for typical crude)
    and does not collapse to the 15.0 °API boundary clamp.
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
    assert 25.0 <= api <= 45.0, f"Expected realistic crude API (25-45), got {api:.1f}"
