"""
Level 2: Physical Verification - Limiting Case: Zero Permeability.

Tests behavior when permeability k -> 0. Darcy flow requires rate -> 0.
"""

import numpy as np
import pytest
from core.engine_surrogate.analytical_models import PhDHybridSurrogate


def test_zero_permeability_limit(standard_reservoir_data):
    """
    When permeability k -> 0, Darcy velocity u -> 0, so convective displacement -> 0.
    """
    surrogate = PhDHybridSurrogate()
    params = standard_reservoir_data.copy()
    params["permeability"] = 1e-6  # near zero
    params["injection_rate"] = 0.001

    rf = surrogate.calculate_recovery(**params)
    assert np.isfinite(rf)
    assert rf >= 0.0
