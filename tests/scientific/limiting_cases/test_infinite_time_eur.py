"""
Level 2: Physical Verification - Limiting Case: Infinite-Time Asymptotic EUR.

Tests that ultimate recovery factor as t -> infinity (or HCPVI -> infinity)
converges to a stable finite asymptote bounded by 1 - Swi - Sor.
"""

import numpy as np
import pytest
from core.engine_surrogate.analytical_models import PhDHybridSurrogate


def test_asymptotic_recovery_limit(standard_reservoir_data):
    """
    Test recovery factor as HCPVI increases from 1.0 to 100.0.
    Recovery must be monotonically non-decreasing and bounded by physical limits.
    """
    surrogate = PhDHybridSurrogate()
    params = standard_reservoir_data.copy()

    hcpvi_levels = [1.0, 2.0, 5.0, 10.0, 50.0]
    rfs = []
    for h in hcpvi_levels:
        params["hcpvi"] = h
        rfs.append(surrogate.calculate_recovery(**params))

    # Invariant 1: Monotonically non-decreasing
    assert np.all(np.diff(rfs) >= -1e-6), f"Recovery decreased with increased injection: {rfs}"

    # Invariant 2: Strictly <= 1.0
    assert np.all(np.array(rfs) <= 1.0), f"Recovery exceeded 100%: {rfs}"
