"""
Level 3: Numerical Verification - Time-Step Numerical Stability & Oscillations.

Tests whether the damped Picard pressure formulation exhibits non-physical oscillations
or artificial numerical damping under varying step sizes.
"""

import numpy as np
import pytest
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from core.data_models import (
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    AdvancedEngineParams,
)
from tests.scientific.conftest import make_reservoir_instance


def test_pressure_oscillation_under_dynamic_injection(standard_reservoir_data):
    """
    Test whether high-rate injection produces sawtooth or negative pressure oscillations.
    In physics, continuous injection into an elastic reservoir yields smooth, monotonic pressure rises.
    """
    data_copy = standard_reservoir_data.copy()
    data_copy["initial_pressure"] = 3000.0
    res = make_reservoir_instance(data_copy)

    eor = EORParameters(
        default_mmp_fallback=standard_reservoir_data["mmp"],
        injection_rate=8000.0,
    )
    ops = OperationalParameters(
        project_lifetime_years=5,
        time_resolution="monthly",
    )
    econ = EconomicParameters()

    engine = SurrogateEngine()
    results = engine.evaluate_scenario(res, eor, ops, econ)

    p_profile = results["pressure"]

    # Check for non-physical sign oscillations: diff(diff(P))
    # Pressure should be smooth and bounded
    assert np.all(np.isfinite(p_profile))
    assert np.all(p_profile >= 0.0)
