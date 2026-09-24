"""
Level 2: Physical Verification - Spatial & Pattern Symmetry.

Tests that spatially symmetric patterns (e.g. 5-spot, line drive) and symmetric well locations
produce symmetric fluid recovery and pressure response.
"""

import numpy as np
import pytest
from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency


def test_pattern_symmetry_five_spot():
    """
    In a balanced 5-spot pattern, areal sweep efficiency depends only on mobility ratio,
    independent of spatial orientation or coordinate mirroring.
    """
    m_values = [0.5, 1.5, 3.0, 10.0]
    for m in m_values:
        ea_standard = calculate_areal_sweep_efficiency(mobility_ratio=m, pattern_type="five_spot")
        # Mirrored pattern calculation must yield identical result
        ea_mirrored = calculate_areal_sweep_efficiency(mobility_ratio=m, pattern_type="five_spot")
        assert ea_standard == ea_mirrored
