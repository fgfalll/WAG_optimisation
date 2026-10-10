"""
Tests for NSGA-II related methods: _dominates and _extract_pareto_front.

Best Practices (2026):
- Follow AAA (Arrange, Act, Assert) pattern
- Use descriptive test names with given-when-then structure
- Property-based tests for edge case coverage
- Integration tests for actual optimization flows
"""

import pytest
import numpy as np
import sys
import os
import logging
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


class TestDominates:
    """Test _dominates functionality for Pareto dominance (minimization)."""

    @pytest.fixture
    def engine(self):
        """Create a mock optimization engine with _dominates method."""
        from core.optimisation_engine import OptimizationEngine

        with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
            mock_init.return_value = None
            return OptimizationEngine.__new__(OptimizationEngine)

    @pytest.mark.parametrize(
        "obj1,obj2,expected",
        [
            ([0.3, 0.8], [0.5, 0.9], True),  # Better in both
            ([0.5, 0.8], [0.5, 0.9], True),  # Equal in obj1, better in obj2
            ([0.3, 0.9], [0.5, 0.8], False),  # Better in obj1, worse in obj2
            ([0.5, 0.9], [0.5, 0.9], False),  # Equal in both (not strictly better)
            ([0.5, 0.95], [0.5, 0.9], False),  # Equal in obj1, worse in obj2
            ([0.3], [0.5], True),  # Single objective
        ],
    )
    def test_dominates_parametrized(self, engine, obj1, obj2, expected):
        """Test domination with various objective pairs (minimization)."""
        result = engine._dominates(obj1, obj2)

        logger.info(f"[DOMINATES] {obj1} vs {obj2} -> {result} (expected: {expected})")
        assert result is expected

    def test_dominates_numpy_arrays(self, engine):
        """Test domination with numpy array inputs."""
        obj1 = np.array([0.3, 0.8])
        obj2 = np.array([0.5, 0.9])
        result = engine._dominates(obj1, obj2)

        logger.info(f"[DOMINATES] numpy: {obj1} vs {obj2} -> {result}")
        assert result is True

    def test_dominates_zero_dimensional_array(self, engine):
        """Test domination with 0-d numpy arrays (scalar wrapped)."""
        obj1 = np.float64(0.3)
        obj2 = np.float64(0.5)
        result = engine._dominates(obj1, obj2)

        logger.info(f"[DOMINATES] 0-d arrays: {obj1} vs {obj2} -> {result}")
        assert result is True


class TestExtractParetoFront:
    """Test _extract_pareto_front functionality for NSGA-II."""

    @pytest.fixture
    def engine(self):
        """Create a mock optimization engine with _extract_pareto_front method."""
        from core.optimisation_engine import OptimizationEngine

        with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
            mock_init.return_value = None
            return OptimizationEngine.__new__(OptimizationEngine)

    @pytest.fixture
    def sample_bi_objective_solutions(self) -> tuple:
        """Fixture: Sample bi-objective solutions for Pareto testing."""
        solutions = np.array(
            [
                [100, 200],  # Solution 0: obj=[0.3, 0.6]
                [200, 100],  # Solution 1: obj=[0.6, 0.3]
                [150, 150],  # Solution 2: obj=[0.45, 0.45]
                [100, 210],  # Solution 3: obj=[0.4, 0.7] - dominated
            ]
        )
        fitnesses = np.array(
            [
                [0.3, 0.6],
                [0.6, 0.3],
                [0.45, 0.45],
                [0.4, 0.7],
            ]
        )
        param_names = ["p1", "p2"]
        return solutions, fitnesses, param_names

    def test_empty_population_returns_empty_list(self, engine):
        """Given an empty population
        When extracting Pareto front
        Then an empty list is returned."""
        logger.info("[PARETO] Testing empty population")
        pareto = engine._extract_pareto_front(
            np.array([]).reshape(0, 3), np.array([]), ["p1", "p2", "p3"]
        )

        logger.info(f"[PARETO] Empty result: {pareto}")
        assert pareto == []

    def test_single_solution_returns_that_solution(self, engine):
        """Given a single solution
        When extracting Pareto front
        Then that solution is returned."""
        solutions = np.array([[100, 200, 300]])
        fitnesses = np.array([[0.5, 0.6]])
        param_names = ["p1", "p2", "p3"]

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        logger.info(f"[PARETO] Single solution: {len(pareto)} solutions")
        logger.info(f"[PARETO] Solution: {pareto[0] if pareto else 'None'}")

        assert len(pareto) == 1
        assert pareto[0]["params"] == {"p1": 100, "p2": 200, "p3": 300}
        assert pareto[0]["objectives"] == [0.5, 0.6]

    def test_non_dominated_solutions_extracted(self, engine, sample_bi_objective_solutions):
        """Given a mixed population with dominated and non-dominated solutions
        When extracting Pareto front
        Then only non-dominated solutions are returned."""
        solutions, fitnesses, param_names = sample_bi_objective_solutions

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        logger.info(f"[PARETO] Population: {len(solutions)} solutions")
        logger.info(f"[PARETO] Fitnesses:\n{fitnesses}")
        logger.info(f"[PARETO] Pareto front: {len(pareto)} solutions")

        for i, p in enumerate(pareto):
            logger.info(f"[PARETO]   [{i}] idx={p['solution_index']}, obj={p['objectives']}")

        assert len(pareto) == 3
        pareto_indices = [p["solution_index"] for p in pareto]
        assert 0 in pareto_indices  # Best obj1
        assert 1 in pareto_indices  # Best obj2
        assert 2 in pareto_indices  # Middle trade-off
        assert 3 not in pareto_indices  # Dominated solution

    def test_returns_list_of_dicts_with_correct_keys(self, engine):
        """Given valid solutions
        When extracting Pareto front
        Then each entry contains params, objectives, and solution_index."""
        solutions = np.array([[100, 200], [200, 100]])
        fitnesses = np.array([[0.3, 0.8], [0.8, 0.3]])
        param_names = ["p1", "p2"]

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        logger.info(f"[PARETO] Dict keys test: {len(pareto)} solutions")
        for i, p in enumerate(pareto):
            logger.info(f"[PARETO]   [{i}] keys={list(p.keys())}")

        assert len(pareto) == 2
        for p in pareto:
            assert "params" in p
            assert "objectives" in p
            assert "solution_index" in p
            assert isinstance(p["params"], dict)
            assert isinstance(p["objectives"], list)

    def test_single_objective_best_solution_returned(self, engine):
        """Given single-objective fitness values
        When extracting Pareto front
        Then the best (minimum) solution is returned."""
        solutions = np.array([[100], [200], [150]])
        fitnesses = np.array([0.3, 0.5, 0.4])
        param_names = ["p1"]

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        logger.info(
            f"[PARETO] Single objective: best_idx={pareto[0]['solution_index'] if pareto else 'None'}"
        )
        logger.info(f"[PARETO] Fitnesses: {fitnesses}")

        assert len(pareto) == 1
        assert pareto[0]["solution_index"] == 0  # Best (0.3)


class TestParetoFrontProperties:
    """Property-based tests for Pareto front extraction."""

    @pytest.fixture
    def engine(self):
        """Create engine for property tests."""
        from core.optimisation_engine import OptimizationEngine

        with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
            mock_init.return_value = None
            return OptimizationEngine.__new__(OptimizationEngine)

    def test_pareto_front_always_subset_of_population(self, engine):
        """Property: Pareto front size cannot exceed population size."""
        population_sizes = [1, 5, 10, 20, 50]

        logger.info("[PARETO] Property: subset of population")
        for n in population_sizes:
            solutions = np.random.rand(n, 3)
            fitnesses = np.random.rand(n, 2)
            param_names = ["x", "y", "z"]

            pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

            logger.info(f"[PARETO]   n={n}, pareto_size={len(pareto)}")
            assert len(pareto) <= n

    def test_pareto_front_contains_unique_solutions(self, engine):
        """Property: Pareto front contains no duplicate solution indices."""
        solutions = np.random.rand(10, 3)
        fitnesses = np.random.rand(10, 2)
        param_names = ["x", "y", "z"]

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        indices = [p["solution_index"] for p in pareto]
        logger.info(f"[PARETO] Unique indices: {len(indices)} == {len(set(indices))}")
        assert len(indices) == len(set(indices))

    def test_no_solution_in_pareto_dominates_another(self, engine):
        """Property: No solution in Pareto front dominates another."""
        solutions = np.random.rand(20, 2)
        fitnesses = np.random.rand(20, 2)
        param_names = ["x", "y"]

        pareto = engine._extract_pareto_front(solutions, fitnesses, param_names)

        logger.info(f"[PARETO] No-dominance check: {len(pareto)} solutions")
        dominated_pairs = []
        for i, sol_i in enumerate(pareto):
            for j, sol_j in enumerate(pareto):
                if i != j:
                    if engine._dominates(sol_i["objectives"], sol_j["objectives"]):
                        dominated_pairs.append((i, j))

        if dominated_pairs:
            logger.warning(f"[PARETO] Found dominated pairs: {dominated_pairs}")
        assert len(dominated_pairs) == 0, f"Found dominated pairs: {dominated_pairs}"

    def test_symmetry_of_dominance(self, engine):
        """Property: If A dominates B, then B does not dominate A."""
        solutions = np.random.rand(10, 2)
        fitnesses = np.random.rand(10, 2)

        logger.info("[PARETO] Symmetry test: checking dominance pairs")
        asymmetry_found = []
        for i in range(10):
            for j in range(10):
                if i != j:
                    dom_ij = engine._dominates(fitnesses[i], fitnesses[j])
                    dom_ji = engine._dominates(fitnesses[j], fitnesses[i])
                    if dom_ij and dom_ji:
                        asymmetry_found.append((i, j))

        if asymmetry_found:
            logger.warning(f"[PARETO] Asymmetric pairs found: {asymmetry_found}")
        assert len(asymmetry_found) == 0, f"Asymmetric dominance: {asymmetry_found}"


class TestDominanceTransitivity:
    """Test dominance relationship transitivity properties."""

    @pytest.fixture
    def engine(self):
        """Create engine for transitivity tests."""
        from core.optimisation_engine import OptimizationEngine

        with patch("core.optimisation_engine.OptimizationEngine.__init__") as mock_init:
            mock_init.return_value = None
            return OptimizationEngine.__new__(OptimizationEngine)

    def test_dominance_transitivity(self, engine):
        """Property: If A dominates B and B dominates C, then A dominates C (for minimization)."""
        obj_a = [0.2, 0.2]
        obj_b = [0.4, 0.4]
        obj_c = [0.6, 0.6]

        logger.info(f"[DOMINATES] A={obj_a}, B={obj_b}, C={obj_c}")
        logger.info(f"[DOMINATES] A dominates B: {engine._dominates(obj_a, obj_b)}")
        logger.info(f"[DOMINATES] B dominates C: {engine._dominates(obj_b, obj_c)}")
        logger.info(f"[DOMINATES] A dominates C: {engine._dominates(obj_a, obj_c)}")

        assert engine._dominates(obj_a, obj_b) is True
        assert engine._dominates(obj_b, obj_c) is True
        assert engine._dominates(obj_a, obj_c) is True

    def test_self_dominance_is_false(self, engine):
        """Property: A solution does not dominate itself."""
        obj = [0.5, 0.5]
        result = engine._dominates(obj, obj)

        logger.info(f"[DOMINATES] self-dominance: {obj} vs {obj} -> {result}")
        assert result is False
