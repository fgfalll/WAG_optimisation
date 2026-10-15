"""
Tests for _select_diverse_solutions method.

This method selects diverse solutions from a population for Bayesian Optimization
initialization, ensuring good parameter space coverage.
"""

import pytest
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


class TestSelectDiverseSolutions:
    """Test _select_diverse_solutions functionality."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock optimization engine with _select_diverse_solutions method."""
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

    def test_select_diverse_solutions_returns_all_when_population_small(self, mock_engine):
        """Test that all solutions are returned when population <= num_solutions."""
        solutions = np.array([[1500, 5000, 5], [2000, 6000, 6]])
        fitnesses = np.array([0.8, 0.7])
        param_names = ["pressure", "rate", "mobility_ratio"]
        num_solutions = 5
        diversity_threshold = 0.2

        logger.info(f"[DIVERSE] Small pop test: {len(solutions)} solutions, need {num_solutions}")
        logger.info(f"[DIVERSE] Solutions:\n{solutions}")
        logger.info(f"[DIVERSE] Fitnesses: {fitnesses}")

        result_sols, result_fit = mock_engine._select_diverse_solutions(
            solutions,
            fitnesses,
            param_names,
            num_solutions=num_solutions,
            diversity_threshold=diversity_threshold,
        )

        logger.info(f"[DIVERSE] Result: {len(result_sols)} solutions selected")
        logger.info(f"[DIVERSE] Result solutions:\n{result_sols}")

        assert len(result_sols) == 2
        np.testing.assert_array_almost_equal(result_sols, solutions)

    def test_select_diverse_solutions_selects_by_diversity(self, mock_engine):
        """Test that solutions are selected by spatial diversity."""
        solutions = np.array(
            [
                [1500, 5000, 5],  # best fitness
                [1500, 5000, 5],  # same as best
                [3000, 8000, 8],  # far away
                [2500, 6500, 6.5],  # intermediate
            ]
        )
        fitnesses = np.array([0.9, 0.8, 0.7, 0.75])
        param_names = ["pressure", "rate", "mobility_ratio"]

        logger.info(f"[DIVERSE] Diversity test: {len(solutions)} solutions, selecting 3")
        logger.info(f"[DIVERSE] Solutions:\n{solutions}")
        logger.info(f"[DIVERSE] Fitnesses: {fitnesses}")

        result_sols, result_fit = mock_engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=3, diversity_threshold=0.1
        )

        logger.info(f"[DIVERSE] Selected: {len(result_sols)} solutions")
        logger.info(f"[DIVERSE] Result solutions:\n{result_sols}")

        assert len(result_sols) == 3
        first_result = result_sols[0]
        matches_first = np.array_equal(first_result, solutions[0]) or np.array_equal(
            first_result, solutions[1]
        )
        assert matches_first

    def test_select_diverse_solutions_normalizes_parameters(self, mock_engine):
        """Test that parameter normalization works correctly."""
        solutions = np.array(
            [
                [1000, 1000],  # min values
                [4000, 10000],  # max values
                [2500, 5500],  # middle values
            ]
        )
        fitnesses = np.array([0.5, 0.6, 0.7])
        param_names = ["pressure", "rate"]

        logger.info(f"[DIVERSE] Normalization test: {len(solutions)} solutions")
        logger.info(f"[DIVERSE] Solutions:\n{solutions}")
        logger.info(f"[DIVERSE] Bounds: pressure=(1000,4000), rate=(1000,10000)")

        result_sols, _ = mock_engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=2, diversity_threshold=0.1
        )

        logger.info(f"[DIVERSE] Result: {len(result_sols)} solutions")
        logger.info(f"[DIVERSE] Result:\n{result_sols}")

        assert len(result_sols) == 2

    def test_select_diverse_solutions_handles_identical_solutions(self, mock_engine):
        """Test handling when all solutions are identical."""
        solutions = np.array(
            [
                [2500, 5500],
                [2500, 5500],
                [2500, 5500],
            ]
        )
        fitnesses = np.array([0.5, 0.5, 0.5])
        param_names = ["pressure", "rate"]

        logger.info(f"[DIVERSE] Identical solutions test: {len(solutions)} identical")
        logger.info(f"[DIVERSE] Solutions:\n{solutions}")
        logger.info(f"[DIVERSE] Fitnesses: {fitnesses}")

        result_sols, result_fit = mock_engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=2, diversity_threshold=0.1
        )

        logger.info(f"[DIVERSE] Result: {len(result_sols)} solutions")
        logger.info(f"[DIVERSE] Result:\n{result_sols}")

        assert len(result_sols) == 2

    def test_select_diverse_solutions_empty_population(self, mock_engine):
        """Test handling of empty population."""
        solutions = np.array([]).reshape(0, 2)
        fitnesses = np.array([])
        param_names = ["pressure", "rate"]

        logger.info("[DIVERSE] Empty population test")

        result_sols, result_fit = mock_engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=5, diversity_threshold=0.2
        )

        logger.info(f"[DIVERSE] Empty result: {len(result_sols)} solutions")

        assert len(result_sols) == 0

    def test_select_diverse_solutions_with_dict_and_tuple_bounds(self, mock_engine):
        """Test that _select_diverse_solutions handles mixed bounds (dict and tuple) without unpacking error."""
        mock_engine._get_parameter_bounds = MagicMock(
            return_value={
                "pressure": (1000.0, 4000.0),
                "rate": (1000.0, 10000.0),
                "allow_well_conversion": {"low": 0, "high": 1, "step": 1},
                "shut_in_mode": {"low": 0, "high": 2},
            }
        )
        solutions = np.array(
            [
                [1500, 5000, 0, 1],
                [3000, 8000, 1, 2],
                [2000, 6000, 0, 0],
            ]
        )
        fitnesses = np.array([0.9, 0.7, 0.8])
        param_names = ["pressure", "rate", "allow_well_conversion", "shut_in_mode"]

        result_sols, result_fit = mock_engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=2, diversity_threshold=0.1
        )

        assert len(result_sols) == 2
        assert len(result_fit) == 2

    def test_select_diverse_solutions_with_real_engine_bounds(self):
        """Test _select_diverse_solutions using the real engine's _get_parameter_bounds."""
        from core.optimisation_engine import OptimizationEngine
        from core.data_models import EORParameters, ReservoirData

        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine.eor_params = EORParameters(
            injection_scheme="continuous",
            injection_scheme_locked=False,
        )
        engine.reservoir = MagicMock()
        engine._mmp_value = 2000.0
        engine._base_eor_params = engine.eor_params
        engine._base_reservoir_data = engine.reservoir

        bounds = engine._get_parameter_bounds()
        param_names = list(bounds.keys())

        # Generate a dummy population with correct number of genes
        n_genes = len(param_names)
        dummy_sol_1 = [1.0] * n_genes
        dummy_sol_2 = [2.0] * n_genes
        dummy_sol_3 = [3.0] * n_genes
        solutions = np.array([dummy_sol_1, dummy_sol_2, dummy_sol_3])
        fitnesses = np.array([0.9, 0.8, 0.7])

        # Must execute without ValueError: too many values to unpack (expected 2)
        result_sols, result_fit = engine._select_diverse_solutions(
            solutions, fitnesses, param_names, num_solutions=2, diversity_threshold=0.1
        )

        assert len(result_sols) == 2
        assert len(result_fit) == 2

