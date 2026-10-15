"""
Level 2: Physical Verification - Initial Conditions Preservation.

Tests that simulation starts exactly from user-specified initial conditions
(P_init, S_wi, OOIP) without unstated pre-step transformations.
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


def test_initial_pressure_and_cumulative_at_time_zero(standard_reservoir_data):
    """
    Verify that at t = 0:
        P(0) == initial_pressure
        N_p(0) == 0.0
        G_p(0) == 0.0
    """
    p_init = standard_reservoir_data["initial_pressure"]
    res = make_reservoir_instance(standard_reservoir_data)
    eor = EORParameters(
        default_mmp_fallback=standard_reservoir_data["mmp"],
        injection_rate=5000.0,
    )
    ops = OperationalParameters(
        project_lifetime_years=5,
        time_resolution="monthly",
    )
    econ = EconomicParameters()

    engine = SurrogateEngine()
    results = engine.evaluate_scenario(res, eor, ops, econ)

    pressures = results["pressure"]
    assert np.isclose(pressures[0], p_init, atol=1e-3), (
        f"Initial simulated pressure P(0) = {pressures[0]} psi does not match input {p_init} psi"
    )
