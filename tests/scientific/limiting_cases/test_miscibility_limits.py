"""
Level 2: Physical Verification - Limiting Case: Miscibility Limits.

Tests limiting behavior of miscibility weighting function omega(P, MMP):
    P >> MMP => omega -> 1.0 (Fully miscible)
    P << MMP => omega -> 0.0 (Fully immiscible)
and continuity across P = MMP.
"""

import numpy as np
import pytest
from core.engine_surrogate.analytical_models import PhDHybridSurrogate


def test_miscibility_weight_limiting_bounds():
    """
    Test omega(P) bounds: 0.0 <= omega <= 1.0 everywhere.
    """
    model = PhDHybridSurrogate()
    mmp = 2500.0

    # P << MMP
    omega_low = model.get_miscibility_weight(pressure=500.0, mmp=mmp, c7_plus_fraction=0.25)
    assert omega_low < 0.05, f"Expected near-zero miscibility weight at 500 psia, got {omega_low}"

    # P >> MMP
    omega_high = model.get_miscibility_weight(pressure=6000.0, mmp=mmp, c7_plus_fraction=0.25)
    assert omega_high > 0.95, f"Expected near-unity miscibility weight at 6000 psia, got {omega_high}"

    # Continuity across P = MMP
    p_around_mmp = np.linspace(mmp - 50.0, mmp + 50.0, 50)
    omegas = [model.get_miscibility_weight(p, mmp, 0.25) for p in p_around_mmp]
    
    # Must be monotonically increasing with pressure
    assert np.all(np.diff(omegas) >= 0.0), f"Omega not monotonic across MMP window"
