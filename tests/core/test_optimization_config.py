"""
Tests for optimization configuration and method selection.
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestOptimizationConfig:
    """Test optimization configuration from base_config.json."""

    @pytest.fixture
    def config_data(self):
        """Load configuration data."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "base_config.json",
        )
        with open(config_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def optimization_config(self, config_data):
        """Get the optimization section from ui_config."""
        return config_data.get("ui_config", {}).get("optimization", {})

    def test_optimization_methods_configured(self, optimization_config):
        """Test that only GA, BO, Hybrid, NSGA-II methods are configured."""
        methods = optimization_config["methods"]

        expected_methods = {
            "Genetic Algorithm (GA)": "optimize_genetic_algorithm",
            "Bayesian Optimization (BO)": "optimize_bayesian",
            "Hybrid (GA -> BO)": "hybrid_optimize",
            "NSGA-II": "optimize_nsga_2",
            "Hybrid NSGA-II -> BO": "hybrid_nsga2_bo",
        }

        for name, key in expected_methods.items():
            assert name in methods, f"Missing method: {name}"
            assert methods[name] == key, (
                f"Wrong key for {name}: expected {key}, got {methods[name]}"
            )

    def test_no_psde_methods(self, optimization_config):
        """Test that PSO and DE methods are removed."""
        methods = optimization_config["methods"]

        assert "Particle Swarm Optimization (PSO)" not in methods
        assert "Differential Evolution (DE)" not in methods

    def test_objectives_configured(self, optimization_config):
        """Test that all objectives are properly configured."""
        objectives = optimization_config["objectives"]

        expected_objectives = {
            "Net Present Value (NPV)": "npv",
            "Recovery Factor (RF)": "recovery_factor",
            "CO2 Utilization": "co2_utilization",
            "Storage Efficiency": "storage_efficiency",
        }

        for name, key in expected_objectives.items():
            assert name in objectives, f"Missing objective: {name}"
            assert objectives[name] == key

    def test_secondary_objectives_configured(self, optimization_config):
        """Test that secondary objectives are configured for NSGA-II."""
        secondary = optimization_config["secondary_objectives"]

        expected_secondary = {
            "Net Present Value (NPV)": "npv",
            "Recovery Factor (RF)": "recovery_factor",
            "CO2 Utilization": "co2_utilization",
            "Storage Efficiency": "storage_efficiency",
        }

        for name, key in expected_secondary.items():
            assert name in secondary, f"Missing secondary objective: {name}"
            assert secondary[name] == key


class TestOptimizationMethods:
    """Test optimization method availability."""

    def test_nsga2_method_exists(self):
        """Test that optimize_nsga_2 method exists on OptimizationEngine."""
        from core.optimisation_engine import OptimizationEngine

        assert hasattr(OptimizationEngine, "optimize_nsga_2")
        assert callable(getattr(OptimizationEngine, "optimize_nsga_2"))

    def test_hybrid_nsga2_bo_method_exists(self):
        """Test that hybrid_nsga2_bo method exists on OptimizationEngine."""
        from core.optimisation_engine import OptimizationEngine

        assert hasattr(OptimizationEngine, "hybrid_nsga2_bo")
        assert callable(getattr(OptimizationEngine, "hybrid_nsga2_bo"))

    def test_select_diverse_solutions_method_exists(self):
        """Test that _select_diverse_solutions method exists."""
        from core.optimisation_engine import OptimizationEngine

        assert hasattr(OptimizationEngine, "_select_diverse_solutions")
