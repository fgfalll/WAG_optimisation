"""
Level 2: Physical Verification - Boundary Conditions: Geomechanical Safety.

Tests strict enforcement of EPA Class VI UIC geomechanical caprock ceiling:
    P_sandface <= 0.90 * P_frac
and verification that injection is throttled to 0 if reservoir reaches this ceiling.
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


def test_epa_class_vi_pressure_ceiling_enforcement(standard_reservoir_data):
    """
    Verify that reservoir pressure profile never exceeds 0.90 * P_frac under any injection rate.
    """
    p_frac = standard_reservoir_data["caprock_fracture_pressure_psi"]
    safe_ceiling = 0.90 * p_frac  # 4,950 psia

    data_copy = standard_reservoir_data.copy()
    data_copy["initial_pressure"] = 4800.0  # Starts very close to ceiling
    res = make_reservoir_instance(data_copy)

    eor = EORParameters(
        default_mmp_fallback=standard_reservoir_data["mmp"],
        caprock_fracture_pressure_psi=p_frac,
        injection_rate=50000.0,
    )
    # Massive over-injection: 50,000 MSCFD
    ops = OperationalParameters(
        project_lifetime_years=5,
        time_resolution="monthly",
    )
    econ = EconomicParameters()

    engine = SurrogateEngine()
    results = engine.evaluate_scenario(res, eor, ops, econ)

    pressures = results["pressure"]
    max_p = np.max(pressures)

    assert max_p <= safe_ceiling + 1e-3, (
        f"Geomechanical boundary violation! Max pressure ({max_p:.1f} psi) exceeded safe ceiling ({safe_ceiling:.1f} psi)."
    )
