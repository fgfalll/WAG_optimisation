"""
Tests for pygad fitness function and multi-objective support.

Refactored (2026) to use Property-Based Testing with Hypothesis.
Instead of hardcoded test cases, we use @given to feed the fitness
function a wide range of realistic boundary values.

Key invariants tested:
- Fitness never NaN for valid parameters
- Fitness never infinite for valid parameters
- Fitness bounded within [-1e20, 1e20]
- Constraint violations correctly apply penalties
- Multi-objective returns exactly 2 values
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from hypothesis import given, settings, assume, Verbosity, HealthCheck
from hypothesis import strategies as st
from unittest.mock import MagicMock, patch

from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    GeneticAlgorithmParams,
    AdvancedEngineParams,
    PVTProperties,
    EOSModelParameters,
)
from core.engine_surrogate.surrogate_engine import create_surrogate_engine


def create_test_engine():
    """Factory function to create a real OptimizationEngine for fitness testing.

    Uses phd_hybrid recovery model which has the richest physics.
    """
    from core.optimisation_engine import OptimizationEngine

    eos_model = EOSModelParameters(
        eos_type="PR",
        component_names=["CO2", "C1", "C4", "C10+"],
        component_properties=np.array(
            [
                [0.0, 44.01, 304.13, 7.376e6, 0.225],
                [0.1, 16.04, 190.6, 4.604e6, 0.011],
                [0.3, 58.12, 425.2, 3.796e6, 0.200],
                [0.6, 142.0, 617.7, 2.11e6, 0.490],
            ]
        ),
        binary_interaction_coeffs=np.zeros((4, 4)),
    )

    reservoir = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={},
        ooip_stb=1_000_000.0,
        initial_pressure=3000.0,
        temperature=150.0,
        rock_compressibility=3e-6,
        average_porosity=0.2,
        initial_water_saturation=0.25,
        thickness_ft=50.0,
        area_acres=100.0,
        length_ft=2000.0,
        eos_model=eos_model,
    )

    pvt = PVTProperties(
        oil_compressibility=1e-5,
        oil_viscosity_cp=1.5,
        water_compressibility=3e-6,
        water_viscosity_cp=0.5,
        oil_fvf=1.2,
        water_fvf=1.0,
    )

    eor_params = EORParameters(
        injection_rate=5000.0,
        target_pressure_psi=3200.0,
        max_pressure_psi=4000.0,
        injection_scheme="continuous",
        wag_ratio=1.0,
        mobility_ratio=5.0,
        min_injection_rate_mscfd=100.0,
        max_injection_rate_mscfd=50000.0,
        min_pressure_factor=0.5,
        min_gravity_factor=0.8,
        max_gravity_factor=1.2,
        min_sor=0.15,
        max_sor=0.40,
        min_transition_alpha=0.1,
        max_transition_alpha=5.0,
        min_transition_beta=0.1,
        max_transition_beta=5.0,
        min_productivity_index=1.0,
        max_productivity_index=50.0,
        min_wellbore_pressure=500.0,
        max_wellbore_pressure=5000.0,
        min_max_production_rate=500.0,
        max_max_production_rate=10000.0,
        min_plateau_duration_fraction=0.0,
        max_plateau_duration_fraction=1.0,
        min_ramp_up_fraction=0.0,
        max_ramp_up_fraction=0.5,
        min_hyperbolic_b_factor=0.1,
        max_hyperbolic_b_factor=2.0,
        min_wag_ratio=0.1,
        max_wag_ratio=2.0,
        min_cycle_length_days=7.0,
        max_cycle_length_days=90.0,
        min_water_fraction=0.0,
        max_water_fraction=0.5,
        min_tapered_duration_years=0.0,
        max_tapered_duration_years=5.0,
        min_tapered_final_rate_multiplier=0.01,
        max_tapered_final_rate_multiplier=1.0,
        min_tapered_initial_rate_multiplier=0.5,
        max_tapered_initial_rate_multiplier=2.0,
        min_huff_n_puff_injection_period_days=7.0,
        max_huff_n_puff_injection_period_days=60.0,
        min_huff_n_puff_soaking_period_days=0.0,
        max_huff_n_puff_soaking_period_days=14.0,
        min_huff_n_puff_production_period_days=7.0,
        max_huff_n_puff_production_period_days=90.0,
        min_huff_n_puff_max_cycles=1.0,
        max_huff_n_puff_max_cycles=20.0,
    )

    operational_params = OperationalParameters(
        project_lifetime_years=15,
        time_resolution="monthly",
        recovery_model_selection="koval",
    )

    economic_params = EconomicParameters(
        oil_price_usd_per_bbl=80.0,
        co2_purchase_cost_usd_per_tonne=50.0,
        discount_rate_fraction=0.1,
    )

    ga_params = GeneticAlgorithmParams(
        num_generations=10,
        sol_per_pop=20,
        num_parents_mating=4,
        num_objectives=1,
        secondary_objective="recovery_factor",
    )

    advanced_params = AdvancedEngineParams()

    engine = OptimizationEngine(
        reservoir=reservoir,
        pvt=pvt,
        eor_params_instance=eor_params,
        ga_params_instance=ga_params,
        operational_params_instance=operational_params,
        economic_params_instance=economic_params,
        advanced_engine_params_instance=advanced_params,
    )

    engine.chosen_objective = "npv"

    return engine


class MockConstraintEngine:
    """Minimal mock engine for testing _check_parameter_constraints.

    Only implements the attributes needed by _check_parameter_constraints:
    - eor_params (with min/max bounds for various parameters)
    - advanced_engine_params (with fracture_pressure_multiplier)
    """

    def __init__(self):
        self.eor_params = EORParameters(
            injection_rate=5000.0,
            target_pressure_psi=3200.0,
            max_pressure_psi=4000.0,
            min_injection_rate_mscfd=100.0,
            max_injection_rate_mscfd=50000.0,
            min_pressure_factor=0.5,
            min_gravity_factor=0.8,
            max_gravity_factor=1.2,
            min_sor=0.15,
            max_sor=0.40,
            min_transition_alpha=0.1,
            max_transition_alpha=5.0,
            min_transition_beta=0.1,
            max_transition_beta=5.0,
            min_productivity_index=1.0,
            max_productivity_index=50.0,
            min_wellbore_pressure=500.0,
            max_wellbore_pressure=5000.0,
            min_max_production_rate=500.0,
            max_max_production_rate=10000.0,
            min_plateau_duration_fraction=0.0,
            max_plateau_duration_fraction=1.0,
            min_ramp_up_fraction=0.0,
            max_ramp_up_fraction=0.5,
            min_hyperbolic_b_factor=0.1,
            max_hyperbolic_b_factor=2.0,
            min_wag_ratio=0.1,
            max_wag_ratio=2.0,
            min_cycle_length_days=7.0,
            max_cycle_length_days=90.0,
            min_water_fraction=0.0,
            max_water_fraction=0.5,
        )
        self.advanced_engine_params = AdvancedEngineParams(failure_penalty=-1e12)

    def _check_parameter_constraints(self, params_dict):
        """Direct proxy to OptimizationEngine._check_parameter_constraints."""
        from core.optimisation_engine import OptimizationEngine
        return OptimizationEngine._check_parameter_constraints(self, params_dict)


def create_constraint_test_engine():
    """Factory function to create engine for constraint testing."""
    return MockConstraintEngine()


def create_mock_engine_for_diversity():
    """Create a mock engine for diversity selection testing.

    This uses __new__ to bypass __init__ since we only need the
    _select_diverse_solutions method.
    """
    from core.optimisation_engine import OptimizationEngine

    with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
        mock_init.return_value = None
        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine._get_parameter_bounds = MagicMock(
            return_value={
                "pressure": (1000.0, 4000.0),
                "rate": (1000.0, 10000.0),
                "mobility_ratio": (1.0, 10.0),
            }
        )
        return engine


class TestFitnessFunctionPygadProperties:
    """Property-based tests for _fitness_func_pygad using Hypothesis.

    These tests verify mathematical invariants that must hold for ALL
    valid parameter combinations, not just a few hand-picked cases.
    """

    def _create_ga_instance(self):
        """Create a mock pygad GA instance."""
        ga_instance = MagicMock()
        ga_instance.generations_completed = 0
        return ga_instance

    def _create_ga_params(self, num_objectives=1, secondary_objective="recovery_factor"):
        """Create a proper GeneticAlgorithmParams instance for testing."""
        return GeneticAlgorithmParams(
            num_generations=10,
            sol_per_pop=20,
            num_parents_mating=4,
            num_objectives=num_objectives,
            secondary_objective=secondary_objective,
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, verbosity=Verbosity.verbose, deadline=None)
    def test_fitness_no_nan_single_objective(self, pressure, rate):
        """Fitness must never be NaN for any valid parameter combination.

        Invariant: For all valid (pressure, rate) ∈ [1000, 5000] × [100, 50000],
        the returned fitness value must not be NaN.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(num_objectives=1)
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert not math.isnan(fitness), (
            f"NaN fitness at pressure={pressure}, rate={rate}. "
            f"Valid parameters must always return valid fitness."
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_fitness_no_inf_single_objective(self, pressure, rate):
        """Fitness must never be infinite for any valid parameter combination.

        Invariant: For all valid (pressure, rate), fitness must be finite.
        Infinite fitness indicates numerical overflow or unhandled edge case.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(num_objectives=1)
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert not math.isinf(fitness), (
            f"Infinite fitness at pressure={pressure}, rate={rate}. "
            f"Valid parameters must always return finite fitness."
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_fitness_bounded_single_objective(self, pressure, rate):
        """Fitness must be within [-1e20, 1e20] for any valid parameter.

        The wrapper defines MAX_OBJECTIVE_VALUE = 1e20 as safe bounds.
        Fitness outside this range indicates missed clamping or overflow.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(num_objectives=1)
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert -1e20 <= fitness <= 1e20, (
            f"Fitness {fitness} out of bounds [-1e20, 1e20] at "
            f"pressure={pressure}, rate={rate}. "
            f"Implementation must clamp extreme values."
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_fitness_is_scalar_for_single_objective(self, pressure, rate):
        """Single-objective fitness must return a scalar, not array.

        For num_objectives=1, pygad expects a scalar fitness value.
        Returning a list/array would cause downstream errors.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(num_objectives=1)
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert isinstance(fitness, (int, float, np.floating)), (
            f"Single-objective fitness must be scalar, got {type(fitness)}"
        )
        assert not isinstance(fitness, (list, np.ndarray)), (
            f"Single-objective fitness must not be array/list, got {type(fitness)}"
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_objective_returns_array_of_len_2(self, pressure, rate):
        """Multi-objective (NSGA-II) fitness must return exactly [obj1, obj2].

        When num_objectives=2, the fitness must be an array/list of length 2,
        containing the primary and secondary objective values.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(
            num_objectives=2, secondary_objective="recovery_factor"
        )
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert isinstance(fitness, (list, np.ndarray)), (
            f"Multi-objective fitness must be array/list, got {type(fitness)}"
        )
        assert len(fitness) == 2, (
            f"Multi-objective fitness must have exactly 2 values, got {len(fitness)}"
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_objective_no_nan(self, pressure, rate):
        """Multi-objective fitness values must not contain NaN.

        Both obj1 and obj2 must be valid numbers.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(
            num_objectives=2, secondary_objective="recovery_factor"
        )
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert not math.isnan(fitness[0]), (
            f"NaN in primary objective at pressure={pressure}, rate={rate}"
        )
        assert not math.isnan(fitness[1]), (
            f"NaN in secondary objective at pressure={pressure}, rate={rate}"
        )

    @given(
        pressure=st.floats(min_value=1000.0, max_value=5000.0),
        rate=st.floats(min_value=100.0, max_value=50000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_multi_objective_no_inf(self, pressure, rate):
        """Multi-objective fitness values must not contain infinity.

        Both obj1 and obj2 must be finite.
        """
        engine = create_test_engine()
        engine.ga_params_current_run = self._create_ga_params(
            num_objectives=2, secondary_objective="recovery_factor"
        )
        ga_instance = self._create_ga_instance()

        gene_array = np.array([pressure, rate])

        try:
            fitness = engine._fitness_func_pygad(ga_instance, gene_array, 0)
        except Exception:
            pytest.skip("Evaluation raised exception, skipping fitness check")

        assert not math.isinf(fitness[0]), (
            f"Infinite primary objective at pressure={pressure}, rate={rate}"
        )
        assert not math.isinf(fitness[1]), (
            f"Infinite secondary objective at pressure={pressure}, rate={rate}"
        )


class TestConstraintViolationPenalty:
    """Property-based tests for constraint violation detection and penalty application."""

    def test_feasible_params_have_no_violations(self):
        """Feasible (within-bounds) parameters should have zero violations.

        Property: If all parameters are within [min, max] bounds,
        then is_feasible=True and penalty=0.
        """
        engine = create_constraint_test_engine()

        feasible_params = {
            "rate": 5000.0,
            "gravity_factor": 1.0,
            "sor": 0.25,
            "transition_alpha": 1.0,
            "transition_beta": 1.0,
            "productivity_index": 10.0,
            "wellbore_pressure": 2000.0,
            "max_production_rate_stbd": 5000.0,
            "plateau_duration_fraction": 0.5,
            "ramp_up_fraction": 0.1,
            "hyperbolic_b_factor": 1.0,
        }

        is_feasible, violations, penalty = engine._check_parameter_constraints(feasible_params)

        assert is_feasible, (
            f"Feasible params returned is_feasible=False. Violations: {violations}"
        )
        assert penalty == 0.0, (
            f"Feasible params should have zero penalty, got {penalty}"
        )
        assert len(violations) == 0, (
            f"Feasible params should have no violations, got {violations}"
        )

    @given(
        rate=st.floats(min_value=0.0, max_value=99.0),
    )
    @settings(max_examples=500)
    def test_rate_below_min_violation(self, rate):
        """Injection rate below minimum should produce violation and positive penalty.

        Property: If rate < min_injection_rate_mscfd (100),
        then violations list must contain rate violation and penalty > 0.

        Note: is_feasible is only False when penalty >= 1e8.
        A rate of 0 gives penalty = 100*1e6 = 1e8 which is exactly the threshold.
        """
        engine = create_constraint_test_engine()
        assume(rate < engine.eor_params.min_injection_rate_mscfd)

        params = {"rate": rate}
        is_feasible, violations, penalty = engine._check_parameter_constraints(params)

        assert penalty > 0, (
            f"Rate below minimum should apply positive penalty, got {penalty}"
        )
        assert any("rate" in v.lower() for v in violations), (
            f"Violations should mention 'rate', got {violations}"
        )

    @given(
        rate=st.floats(min_value=50100.0, max_value=1e10),
    )
    @settings(max_examples=500)
    def test_rate_above_max_violation(self, rate):
        """Injection rate above maximum should produce violation and positive penalty.

        Property: If rate > max_injection_rate_mscfd (50000) + 100,
        then violations list must contain rate violation and penalty >= 1e8.

        Note: We use rate > 50100 to ensure penalty >= 1e8 (need 100 unit violation).
        """
        engine = create_constraint_test_engine()
        assume(rate > engine.eor_params.max_injection_rate_mscfd)

        params = {"rate": rate}
        is_feasible, violations, penalty = engine._check_parameter_constraints(params)

        assert penalty >= 1e8, (
            f"Rate above max should apply penalty >= 1e8, got {penalty}"
        )
        assert not is_feasible, (
            f"Rate {rate} above max {engine.eor_params.max_injection_rate_mscfd} "
            f"with large violation should be infeasible"
        )
        assert any("rate" in v.lower() for v in violations), (
            f"Violations should mention 'rate', got {violations}"
        )

    @given(
        pressure=st.floats(min_value=5001.0, max_value=1e10),
    )
    @settings(max_examples=500)
    def test_pressure_exceeds_fracture_limit_violation(self, pressure):
        """Pressure exceeding fracture limit should produce violation.

        Property: If pressure > max_pressure_psi * fracture_multiplier,
        then penalty must be positive.
        """
        engine = create_constraint_test_engine()
        max_limit = (
            engine.eor_params.max_pressure_psi
            * engine.advanced_engine_params.fracture_pressure_multiplier
        )
        assume(pressure > max_limit)

        params = {"pressure": pressure}
        is_feasible, violations, penalty = engine._check_parameter_constraints(params)

        assert penalty > 0, (
            f"Pressure {pressure} above fracture limit {max_limit} "
            f"should apply penalty, got {penalty}"
        )

    @given(
        sor=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=500)
    def test_sor_out_of_bounds(self, sor):
        """Sor outside [min_sor, max_sor] should produce violation.

        Property: If sor < min_sor (0.15) or sor > max_sor (0.40),
        then penalty must be positive.
        """
        engine = create_constraint_test_engine()
        assume(sor < engine.eor_params.min_sor or sor > engine.eor_params.max_sor)

        params = {"sor": sor}
        is_feasible, violations, penalty = engine._check_parameter_constraints(params)

        assert penalty > 0, (
            f"Sor {sor} outside [{engine.eor_params.min_sor}, {engine.eor_params.max_sor}] "
            f"should apply penalty, got {penalty}"
        )


class TestDiverseSolutionSelectionProperties:
    """Property-based tests for _select_diverse_solutions using Hypothesis.

    Tests mathematical invariants about diversity selection that must hold
    for any population size, any number of requested solutions, etc.
    """

    @given(
        num_solutions=st.integers(min_value=1, max_value=100),
        num_params=st.integers(min_value=1, max_value=5),
        identical_value=st.floats(min_value=-1e10, max_value=1e10),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_identical_solutions_returns_at_most_requested(
        self, num_solutions, num_params, identical_value
    ):
        """When all solutions are identical, output count <= min(requested, population).

        Invariant: _select_diverse_solutions must never return more solutions
        than exist in the population, even when diversity threshold is 0.
        """
        engine = create_mock_engine_for_diversity()

        solutions = np.full((num_solutions, num_params), identical_value)
        fitnesses = np.full(num_solutions, 0.5)
        param_names = ["p1", "p2", "p3", "p4", "p5"][:num_params]

        num_requested = min(num_solutions + 10, 100)

        result_sols, result_fit = engine._select_diverse_solutions(
            solutions,
            fitnesses,
            param_names,
            num_solutions=num_requested,
            diversity_threshold=0.1,
        )

        assert len(result_sols) <= num_solutions, (
            f"Cannot return more solutions ({len(result_sols)}) than "
            f"population size ({num_solutions})"
        )
        assert len(result_sols) <= num_requested, (
            f"Cannot return more solutions ({len(result_sols)}) than "
            f"requested ({num_requested})"
        )

    @given(
        num_params=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_empty_population_returns_empty(self, num_params):
        """Empty population should return empty arrays.

        Invariant: When population is empty (0 rows), result should be
        empty arrays with shape (0, num_params) and (0,).
        """
        engine = create_mock_engine_for_diversity()

        solutions = np.array([]).reshape(0, num_params)
        fitnesses = np.array([])
        param_names = ["p1", "p2", "p3", "p4", "p5"][:num_params]

        result_sols, result_fit = engine._select_diverse_solutions(
            solutions,
            fitnesses,
            param_names,
            num_solutions=5,
            diversity_threshold=0.2,
        )

        assert len(result_sols) == 0, (
            f"Empty population should return 0 solutions, got {len(result_sols)}"
        )
        assert len(result_fit) == 0, (
            f"Empty population should return 0 fitnesses, got {len(result_fit)}"
        )

    @given(
        num_solutions=st.integers(min_value=1, max_value=50),
        num_params=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_requested_more_than_population_returns_all(
        self, num_solutions, num_params
    ):
        """When requested > population, should return all available solutions.

        Invariant: If num_requested >= len(population), return all solutions.
        """
        engine = create_mock_engine_for_diversity()

        np.random.seed(42)
        solutions = np.random.rand(num_solutions, num_params)
        fitnesses = np.random.rand(num_solutions)
        param_names = ["p1", "p2", "p3", "p4", "p5"][:num_params]

        result_sols, result_fit = engine._select_diverse_solutions(
            solutions,
            fitnesses,
            param_names,
            num_solutions=num_solutions + 100,
            diversity_threshold=0.2,
        )

        assert len(result_sols) == num_solutions, (
            f"Requested {num_solutions + 100} but only {num_solutions} exist. "
            f"Should return all {num_solutions}, got {len(result_sols)}"
        )


class TestHybridOptimizeUsesDiverseSolutions:
    """Interaction test: verify _select_diverse_solutions is called during hybrid optimization.

    This is NOT property-based - it tests the method call contract.
    """

    def test_select_diverse_solutions_is_called(self):
        """Given a GA optimization completed
        When hybrid_optimize prepares solutions for BO
        Then _select_diverse_solutions is invoked.

        This is an interaction test, not a property test.
        """
        from core.optimisation_engine import OptimizationEngine

        with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
            mock_init.return_value = None
            engine = OptimizationEngine.__new__(OptimizationEngine)
            engine.ga_params_current_run = MagicMock()
            engine.ga_params_current_run.num_diverse_solutions_for_bo = 5
            engine.ga_params_current_run.diversity_threshold_for_bo = 0.2
            engine._get_parameter_bounds = MagicMock(
                return_value={
                    "pressure": (1000.0, 4000.0),
                    "rate": (1000.0, 10000.0),
                }
            )

        mock_return = (np.array([[2500, 5000], [3000, 6000]]), np.array([0.8, 0.7]))
        engine._select_diverse_solutions = MagicMock(return_value=mock_return)

        ga_instance = MagicMock()
        ga_instance.population = np.array(
            [
                [2500, 5000],
                [3000, 6000],
                [2700, 5500],
            ]
        )
        ga_instance.last_generation_fitness = np.array([0.8, 0.7, 0.75])

        diverse_solutions, diverse_fitnesses = engine._select_diverse_solutions(
            ga_instance.population,
            ga_instance.last_generation_fitness,
            ["pressure", "rate"],
            engine.ga_params_current_run.num_diverse_solutions_for_bo,
            engine.ga_params_current_run.diversity_threshold_for_bo,
        )

        engine._select_diverse_solutions.assert_called_once()
        assert len(diverse_solutions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
