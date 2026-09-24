"""
Unit tests for unphysical chromosome penalties, Class E modifier removal,
and Plotly dependency enforcement.
"""

import math
import numpy as np
import pytest
import plotly.graph_objects as go

from core.objectives.wrapper import ObjectiveFunctions
from core.data_models import (
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    AdvancedEngineParams,
    ReservoirData,
)


def test_no_class_e_storage_efficiency_fallback():
    """Verify that Class E artificial storage modifier is eradicated.
    
    When storage_params is None and profiles lacks storage_efficiency,
    the wrapper must return NaN, NEVER synthesizing artificial 0.3 or 0.5 * (RF / 0.35).
    """
    op = OperationalParameters(time_resolution="yearly")
    eor = EORParameters()
    res = ReservoirData(grid={}, pvt_tables={})
    adv = AdvancedEngineParams()
    obj_calc = ObjectiveFunctions(op, eor, res, adv)

    # profiles has npv but lacks storage_efficiency and no storage params
    results = obj_calc._calculate_objective_functions(
        profiles={"npv": 1e6},
        recovery_factor=0.35,
        econ_params=EconomicParameters(),
        storage_params=None,
    )

    storage_eff = results["storage_efficiency"]
    assert math.isnan(storage_eff), f"Expected NaN storage efficiency, got {storage_eff}"
    assert results["storage_metrics"]["method"] == "unphysical_or_missing_data"


def test_no_magic_co2_utilization_fallback():
    """Verify that CO2 utilization returns NaN instead of 1e6 magic number on missing/empty data."""
    op = OperationalParameters(time_resolution="yearly")
    eor = EORParameters()
    res = ReservoirData(grid={}, pvt_tables={})
    adv = AdvancedEngineParams()
    obj_calc = ObjectiveFunctions(op, eor, res, adv)

    # profiles has npv but lacks co2 purchase data
    results = obj_calc._calculate_objective_functions(
        profiles={"npv": 5e5},
        recovery_factor=0.25,
        econ_params=EconomicParameters(),
        storage_params=None,
    )

    util = results["co2_utilization"]
    assert math.isnan(util), f"Expected NaN CO2 utilization, got {util}"


def test_ga_chromosome_nan_pruned_with_failure_penalty():
    """Verify that NaN or infinite objective values receive the full FAILURE_PENALTY."""
    from core.optimisation_engine import OptimizationEngine

    engine = OptimizationEngine.__new__(OptimizationEngine)
    engine.advanced_engine_params = AdvancedEngineParams(failure_penalty=-1e12)
    engine.chosen_objective = "storage_efficiency"
    engine.eor_params = EORParameters()
    engine.operational_params = OperationalParameters(time_resolution="yearly")
    engine.economic_params = EconomicParameters()

    # Mock evaluate_for_analysis returning NaN storage_efficiency
    engine.evaluate_for_analysis = lambda *args, **kwargs: {
        "storage_efficiency": float("nan"),
        "recovery_factor": 0.3,
        "npv": 1e6,
        "simulation_mode": "co2_eor",
    }
    engine._sanitize_and_discretize_parameters = lambda p: p
    engine._check_parameter_constraints = lambda p: (True, [], 0.0)

    score = engine._objective_function_wrapper(co2_injection_rate=5000.0)
    assert score == -1e12, f"Expected full FAILURE_PENALTY (-1e12), got {score}"


def test_ga_chromosome_breakthrough_violation_full_penalty():
    """Verify breakthrough constraint violation returns full FAILURE_PENALTY without dilution."""
    from core.optimisation_engine import OptimizationEngine

    engine = OptimizationEngine.__new__(OptimizationEngine)
    engine.advanced_engine_params = AdvancedEngineParams(
        failure_penalty=-1e12,
        breakthrough_time_min_years=1.0,
        containment_critical_threshold=0.0,  # avoid containment tripping
    )
    engine.chosen_objective = "npv"
    engine.eor_params = EORParameters(max_pressure_psi=4000.0)
    engine.operational_params = OperationalParameters(time_resolution="yearly")
    engine.economic_params = EconomicParameters()

    # Evaluation with breakthrough time < min_breakthrough_time (e.g. 0.2 years < 1.0)
    engine.evaluate_for_analysis = lambda *args, **kwargs: {
        "npv": 5e6,
        "recovery_factor": 0.35,
        "storage_efficiency": 0.6,
        "simulation_mode": "co2_eor",
        "breakthrough_time_years": 0.2,
        "yearly_pressure": np.array([2000.0, 2100.0]),
    }
    engine._sanitize_and_discretize_parameters = lambda p: p
    engine._check_parameter_constraints = lambda p: (True, [], 0.0)

    score = engine._objective_function_wrapper(co2_injection_rate=5000.0)
    # Must be exactly FAILURE_PENALTY (-1e12), not -8e11 (diluted *0.8)
    assert score == -1e12, f"Expected -1e12, got {score}"


def test_plotly_is_real_dependency():
    """Verify that analysis.material_balance imports the genuine Plotly module, not a mock."""
    from analysis.material_balance import go as mb_go, make_subplots as mb_make_subplots
    import plotly.graph_objects as real_go
    from plotly.subplots import make_subplots as real_make_subplots

    # Must be the exact same class/module
    assert mb_go.Figure is real_go.Figure
    assert mb_make_subplots is real_make_subplots
    fig = mb_go.Figure()
    assert isinstance(fig, real_go.Figure)
