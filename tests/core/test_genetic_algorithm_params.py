"""
Tests for GeneticAlgorithmParams dataclass validation.

Best Practices (2026):
- Arrange, Act, Assert (AAA) pattern
- Descriptive test names
- Parametrized tests for coverage
- Property-based validation
"""

import pytest
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_models import GeneticAlgorithmParams

logger = logging.getLogger(__name__)


class TestGeneticAlgorithmParamsDefaults:
    """Test default parameter values."""

    def test_default_values_are_correct(self):
        """Given no arguments
        When creating GeneticAlgorithmParams
        Then default values are set correctly."""
        params = GeneticAlgorithmParams()

        logger.info(
            f"[GA_PARAMS] Defaults: num_gen={params.num_generations}, "
            f"sol_per_pop={params.sol_per_pop}, num_obj={params.num_objectives}"
        )

        assert params.num_generations == 80, f"Expected 80, got {params.num_generations}"
        assert params.sol_per_pop == 80, f"Expected 80, got {params.sol_per_pop}"
        assert params.num_parents_mating == 10, f"Expected 10, got {params.num_parents_mating}"
        assert params.num_objectives == 1, f"Expected 1, got {params.num_objectives}"
        assert params.secondary_objective == "recovery_factor"
        assert params.num_diverse_solutions_for_bo == 15
        assert params.diversity_threshold_for_bo == 0.20


class TestGeneticAlgorithmParamsValidation:
    """Test validation rules for GeneticAlgorithmParams."""

    @pytest.mark.parametrize(
        "num_objectives,expected",
        [
            (1, 1),
            (2, 2),
        ],
    )
    def test_valid_num_objectives(self, num_objectives, expected):
        """Given valid num_objectives values
        When creating params
        Then they are accepted."""
        params = GeneticAlgorithmParams(num_objectives=num_objectives)

        logger.info(f"[GA_PARAMS] num_objectives={num_objectives} -> {params.num_objectives}")
        assert params.num_objectives == expected

    def test_invalid_num_objectives_raises(self):
        """Given num_objectives=3
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing invalid num_objectives=3")
        with pytest.raises(ValueError, match="Number of objectives must be 1"):
            GeneticAlgorithmParams(num_objectives=3)

    @pytest.mark.parametrize(
        "secondary_obj",
        [
            "npv",
            "recovery_factor",
            "co2_utilization",
            "storage_efficiency",
        ],
    )
    def test_valid_secondary_objectives(self, secondary_obj):
        """Given valid secondary_objective values
        When creating bi-objective params
        Then they are accepted."""
        params = GeneticAlgorithmParams(num_objectives=2, secondary_objective=secondary_obj)

        logger.info(f"[GA_PARAMS] secondary_objective='{secondary_obj}' accepted")
        assert params.secondary_objective == secondary_obj

    def test_invalid_secondary_objective_raises(self):
        """Given invalid secondary_objective with bi-objective
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing invalid secondary_objective='invalid'")
        with pytest.raises(ValueError, match="Secondary objective must be one of"):
            GeneticAlgorithmParams(num_objectives=2, secondary_objective="invalid_objective")


class TestGeneticAlgorithmParamsFactory:
    """Test factory methods for GeneticAlgorithmParams."""

    def test_from_config_dict_creates_valid_instance(self):
        """Given a config dictionary
        When creating params via from_config_dict
        Then all values are correctly set."""
        config = {
            "num_generations": 50,
            "num_objectives": 2,
            "secondary_objective": "co2_utilization",
            "num_diverse_solutions_for_bo": 10,
        }

        logger.info(f"[GA_PARAMS] from_config_dict: {config}")

        params = GeneticAlgorithmParams.from_config_dict(config)

        assert params.num_generations == 50
        assert params.num_objectives == 2
        assert params.secondary_objective == "co2_utilization"
        assert params.num_diverse_solutions_for_bo == 10

    def test_from_config_dict_with_defaults(self):
        """Given a partial config dictionary
        When creating params
        Then missing values use defaults."""
        config = {"num_generations": 30}

        logger.info(f"[GA_PARAMS] partial config: {config}")

        params = GeneticAlgorithmParams.from_config_dict(config)

        assert params.num_generations == 30
        assert params.sol_per_pop == 80  # Default
        assert params.num_objectives == 1  # Default


class TestGeneticAlgorithmParamsBoundaries:
    """Boundary value tests for GeneticAlgorithmParams."""

    def test_num_generations_at_lower_boundary(self):
        """Given num_generations=10 (minimum valid)
        When creating params
        Then no error is raised."""
        params = GeneticAlgorithmParams(num_generations=10)
        logger.info(f"[GA_PARAMS] num_generations lower bound=10 -> OK")
        assert params.num_generations == 10

    def test_num_generations_at_upper_boundary(self):
        """Given num_generations=1000 (maximum valid)
        When creating params
        Then no error is raised."""
        params = GeneticAlgorithmParams(num_generations=1000)
        logger.info(f"[GA_PARAMS] num_generations upper bound=1000 -> OK")
        assert params.num_generations == 1000

    def test_num_generations_below_lower_boundary(self):
        """Given num_generations=5 (below minimum)
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing num_generations=5 (below min)")
        with pytest.raises(ValueError, match="Generations must be between"):
            GeneticAlgorithmParams(num_generations=5)

    def test_num_generations_above_upper_boundary(self):
        """Given num_generations=1001 (above maximum)
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing num_generations=1001 (above max)")
        with pytest.raises(ValueError, match="Generations must be between"):
            GeneticAlgorithmParams(num_generations=1001)

    def test_sol_per_pop_at_lower_boundary(self):
        """Given sol_per_pop=10 (minimum valid)
        When creating params with valid num_parents_mating
        Then no error is raised."""
        params = GeneticAlgorithmParams(sol_per_pop=10, num_parents_mating=2)
        logger.info(f"[GA_PARAMS] sol_per_pop lower bound=10 -> OK")
        assert params.sol_per_pop == 10

    def test_sol_per_pop_below_lower_boundary(self):
        """Given sol_per_pop=5 (below minimum)
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing sol_per_pop=5 (below min)")
        with pytest.raises(ValueError, match="Population Size must be between"):
            GeneticAlgorithmParams(sol_per_pop=5)

    def test_sol_per_pop_above_upper_boundary(self):
        """Given sol_per_pop=1001 (above maximum)
        When creating params
        Then ValueError is raised."""
        logger.info("[GA_PARAMS] Testing sol_per_pop=1001 (above max)")
        with pytest.raises(ValueError, match="Population Size must be between"):
            GeneticAlgorithmParams(sol_per_pop=1001)

    def test_num_parents_mating_valid_range(self):
        """Given valid num_parents_mating range
        When creating params
        Then no error is raised."""
        params = GeneticAlgorithmParams(num_parents_mating=2, sol_per_pop=100)
        logger.info(f"[GA_PARAMS] num_parents_mating=2, sol_per_pop=100 -> OK")
        assert params.num_parents_mating == 2
