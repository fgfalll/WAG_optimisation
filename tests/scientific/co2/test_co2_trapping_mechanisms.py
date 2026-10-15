"""
Level 2 & 4: CO2-Specific Verification - Trapping Mechanisms & Inverted Trapping.

Tests structural, residual (capillary), and solubility trapping equations.
In physics:
    Higher residual/critical gas saturation (S_gc or S_gr) MUST increase capillary trapping.
In surrogate_models.py:238-241:
    gas_trapping = 1.0 - s_gc
    trapping_eff = max_displacement * gas_trapping
    Higher S_gc reduces calculated trapping efficiency!
"""

import numpy as np
import pytest
from core.engine_surrogate.surrogate_models import calculate_trapping_efficiency


def test_inverted_critical_gas_trapping():
    """
    Test trapping efficiency calculation with increasing critical gas saturation S_gc.
    In reservoir physics, higher S_gc means more gas is permanently trapped in pores.
    In surrogate_models.py:238-241, trapping_eff is multiplied by (1 - S_gc), so it drops!
    """
    s_wi = 0.25
    s_or = 0.20

    # Test with low S_gc (5%) vs high S_gc (25%)
    eff_low_sgc = calculate_trapping_efficiency(s_wi=s_wi, s_or=s_or, s_gc=0.05)
    eff_high_sgc = calculate_trapping_efficiency(s_wi=s_wi, s_or=s_or, s_gc=0.25)

    # In current implementation, higher S_gc yields lower trapping efficiency!
    assert eff_high_sgc < eff_low_sgc, (
        f"Confirms inverted gas trapping: S_gc=0.05 yields {eff_low_sgc:.3f}, but S_gc=0.25 yields {eff_high_sgc:.3f}!"
    )
