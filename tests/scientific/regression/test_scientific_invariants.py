"""
Level 1 & 2: Property-Based Scientific Invariant Testing (Hypothesis).

Uses Hypothesis to generate thousands of physically valid input parameter combinations
and verifies that fundamental mathematical and physical invariants are preserved.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from core.engine_surrogate.analytical_models import PhDHybridSurrogate


@settings(max_examples=50, deadline=None)
@given(
    pressure=st.floats(min_value=1000.0, max_value=6000.0),
    mmp=st.floats(min_value=1500.0, max_value=4000.0),
    v_dp=st.floats(min_value=0.1, max_value=0.9),
    s_wi=st.floats(min_value=0.1, max_value=0.4),
    hcpvi=st.floats(min_value=0.1, max_value=5.0),
)
def test_hypothesis_recovery_factor_physical_invariants(pressure, mmp, v_dp, s_wi, hcpvi):
    """
    Hypothesis property test:
    For ANY physically valid reservoir parameters:
        1. Recovery factor RF must be strictly in [0.0, 1.0].
        2. RF must be a finite float (no NaN, Inf, or complex numbers).
        3. RF must not exceed 1.0 - S_wi.
    """
    surrogate = PhDHybridSurrogate()
    rf = surrogate.calculate_recovery(
        pressure=pressure,
        mmp=mmp,
        v_dp=v_dp,
        s_wi=s_wi,
        hcpvi=hcpvi,
        viscosity_oil=2.0,
        viscosity_inj=0.04,
    )
    assert np.isfinite(rf), f"Non-finite recovery factor: {rf}"
    assert 0.0 <= rf <= 1.0, f"Recovery factor outside [0, 1]: {rf}"
    assert rf <= (1.0 - s_wi + 1e-5), f"Recovery factor exceeds initial oil in place: {rf} > 1 - {s_wi}"
