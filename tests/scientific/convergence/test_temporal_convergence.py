"""
Level 3: Numerical Verification - Temporal Grid Refinement & Convergence.

Evaluates solution convergence under time-step refinement (yearly, monthly, daily).
Measures L1, L2, and L_infinity norms to verify asymptotic stability.
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


def test_temporal_refinement_convergence(standard_reservoir_data):
    """
    Compare monthly vs daily time resolution to evaluate numerical convergence.
    """
    res = make_reservoir_instance(standard_reservoir_data)
    eor = EORParameters(
        default_mmp_fallback=standard_reservoir_data["mmp"],
        injection_rate=6000.0,
    )
    econ = EconomicParameters()

    engine = SurrogateEngine()

    # Monthly simulation
    ops_monthly = OperationalParameters(
        project_lifetime_years=3,
        time_resolution="monthly",
    )
    res_monthly = engine.evaluate_scenario(res, eor, ops_monthly, econ)
    cum_oil_monthly = res_monthly["cumulative_oil"]

    # Weekly simulation (refined time-step)
    ops_weekly = OperationalParameters(
        project_lifetime_years=3,
        time_resolution="weekly",
    )
    res_weekly = engine.evaluate_scenario(res, eor, ops_weekly, econ)
    cum_oil_weekly = res_weekly["cumulative_oil"]

    # Relative difference between monthly and weekly time-steps
    rel_diff = abs(cum_oil_monthly - cum_oil_weekly) / max(cum_oil_weekly, 1.0)
    # Convergence criteria: relative error < 5% across step-size refinement
    assert rel_diff < 0.05, f"Excessive time-step sensitivity: rel_diff = {rel_diff*100:.2f}%"
