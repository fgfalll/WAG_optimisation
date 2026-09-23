"""
Level 1: Mathematical Verification - Singularity, Division by Zero & Overflow.

Tests mathematical domain boundaries, singular limits, and potential runtime crashes
in analytical and empirical formulations.
"""

import numpy as np
import pytest
from evaluation.mmp import calculate_mmp, MMPParameters
from core.engine_surrogate.analytical_models import PhDHybridSurrogate, KovalRecoveryModel


def test_v_dp_near_unity_singularity():
    """
    Test Koval heterogeneity factor behavior as Dykstra-Parsons coefficient V_DP -> 1.0.

    H_k = 1 / (1 - V_DP)^2. As V_DP -> 1.0, H_k -> infinity.
    Code must safely guard against ZeroDivisionError and floating point overflow.
    """
    model = PhDHybridSurrogate()
    # At V_DP = 0.999, (1 - 0.999)^2 = 1e-6, H_k = 1e6
    rf = model.calculate_recovery(v_dp=0.999, pressure=3000.0, mmp=2500.0)
    assert np.isfinite(rf), "Recovery factor was not finite at extreme V_DP -> 1.0"
    assert 0.0 <= rf <= 1.0, f"Recovery factor out of physical bounds: {rf}"


def test_cronquist_mmp_singularity_at_55_api():
    """
    Test Cronquist MMP correlation at oil_gravity >= 55.0 deg API.

    Verifies elimination of SCI-FLAW-13 (ad-hoc (55 - API) singularity).
    Authentic Cronquist (1978) correlation uses MW_C5+ derived from API gravity or C7+ MW:
        MMP = 15.988 * (Temperature ^ Y)
    At API = 55.0 and API = 56.0, Cronquist predicts positive, finite, real MMP values
    and lighter oil (API=56) has lower MMP than API=55.
    """
    # Case 1: Exactly 55 API
    params_55 = MMPParameters(temperature=150.0, oil_gravity=55.0)
    mmp_55 = calculate_mmp(params_55, method="cronquist")
    assert np.isfinite(mmp_55) and not isinstance(mmp_55, complex)
    assert 500.0 < mmp_55 < 3000.0, f"Expected physical MMP at API=55, got {mmp_55}"

    # Case 2: 56 API (Light volatile condensate)
    params_56 = MMPParameters(temperature=150.0, oil_gravity=56.0)
    mmp_56 = calculate_mmp(params_56, method="cronquist")
    assert np.isfinite(mmp_56) and not isinstance(mmp_56, complex)
    assert 500.0 < mmp_56 < 3000.0, f"Expected physical MMP at API=56, got {mmp_56}"

    # In physics: higher API (lighter oil) has lower MMP
    assert mmp_56 < mmp_55, f"Expected MMP(56 API) < MMP(55 API), got {mmp_56} vs {mmp_55}"


def test_mobility_ratio_unit_limit_singularity():
    """
    Test behavior when mobility ratio M = 1.0 in (M - 1) denominators.

    In profile_generator_fast.py:948:
        frac_flow_co2 = koval_factor / (koval_factor + (mobility_ratio - 1) * 0.5)
    When M = 1.0: (M - 1) = 0.0, denominator is koval_factor / koval_factor = 1.0.
    In surrogate_models.py:164-182:
        Craig areal sweep efficiency has a discontinuous jump at M = 1.0.
    """
    from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency

    ea_below = calculate_areal_sweep_efficiency(mobility_ratio=0.9999)
    ea_above = calculate_areal_sweep_efficiency(mobility_ratio=1.0001)

    # Document whether there is a discontinuity
    step = abs(ea_below - ea_above)
    # The current code has: if M <= 1: return 1.0, but at M=1.0001 it returns ~0.517
    # This test proves the presence of the 48% cliff
    assert step > 0.40, f"Expected known discontinuous step at M=1.0, found step={step}"
