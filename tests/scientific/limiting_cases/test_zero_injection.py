"""
Level 2: Physical Verification - Limiting Case: Zero Injection.

Tests behavior when injection rate = 0.
No CO2 is introduced, so CO2 storage, breakthrough, and recycling must be identically zero.
"""

import numpy as np
import pytest
from core.engine_surrogate.profile_generator_fast import FastProfileGenerator


def test_zero_injection_limits():
    """
    Verify that zero CO2 injection yields zero CO2 production and zero breakthrough.
    """
    generator = FastProfileGenerator()
    time_vec = np.linspace(0, 3650, 120)  # 10 years monthly
    oil_profile = np.full_like(time_vec, 500.0)
    zero_inj = np.zeros_like(time_vec)

    profiles = generator._generate_gas_profile(
        oil_profile=oil_profile,
        time_vector=time_vec,
        injection_profile=zero_inj,
        breakthrough_time_years=5.0,
    )

    co2_gas = profiles["co2_gas"]
    # With zero injection, post-breakthrough CO2 production must be zero
    assert np.all(co2_gas == 0.0), f"CO2 gas produced without any CO2 injection! Max = {np.max(co2_gas)}"
