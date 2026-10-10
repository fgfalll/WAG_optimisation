"""
Unit and integration tests for direct single simulation execution in OptimizationEngine
and OptimizationWidget.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.optimisation_engine import OptimizationEngine
from core.data_models import (
    ReservoirData,
    PVTProperties,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    GeneticAlgorithmParams,
    BayesianOptimizationParams,
    CO2StorageParameters,
    ProfileParameters,
    AdvancedEngineParams,
)


@pytest.fixture
def test_engine(
    reservoir_data,
    pvt_data,
    eor_params,
    operational_params,
    economic_params,
    ga_params,
    bo_params,
    eos_model_params,
):
    """Creates a real OptimizationEngine instance for testing single simulation."""
    reservoir_data.eos_model = eos_model_params
    reservoir_data.ooip_stb = reservoir_data.calculate_ooip_from_physics()
    engine = OptimizationEngine(
        reservoir=reservoir_data,
        pvt=pvt_data,
        eor_params_instance=eor_params,
        ga_params_instance=ga_params,
        bo_params_instance=bo_params,
        economic_params_instance=economic_params,
        operational_params_instance=operational_params,
        profile_params_instance=ProfileParameters(),
        advanced_engine_params_instance=AdvancedEngineParams(),
        co2_storage_params_instance=CO2StorageParameters(),
    )
    return engine


class TestSingleSimulation:
    """Test suite for direct forward simulation execution."""

    def test_run_single_simulation_returns_valid_results(self, test_engine):
        """Verify that run_single_simulation executes forward simulation and returns standard results."""
        results = test_engine.run_single_simulation()

        assert isinstance(results, dict)
        assert results.get("method") == "single_simulation"
        assert "objective_function_value" in results
        assert isinstance(results["objective_function_value"], (float, int, np.floating))
        assert not np.isnan(results["objective_function_value"])

        # Check final metrics
        final_metrics = results.get("final_metrics", {})
        assert "recovery_factor" in final_metrics
        assert "npv" in final_metrics
        assert final_metrics["recovery_factor"] >= 0.0

        # Check profiles
        profiles = results.get("optimized_profiles", {})
        assert profiles is not None
        assert "yearly_time_years" in profiles or "time_vector" in profiles
        assert "well_schedule" in profiles

        # Check simulation statistics
        stats = results.get("simulation_statistics", {})
        assert stats.get("status") == "success"
        assert stats.get("evaluation_time_seconds", 0) >= 0.0

    def test_run_single_simulation_with_custom_params(self, test_engine):
        """Verify that run_single_simulation accepts and applies custom operational overrides."""
        custom_params = {
            "rate": 7500.0,
            "pressure": 3400.0,
            "plateau_duration_fraction": 0.4,
        }
        results = test_engine.run_single_simulation(custom_params=custom_params)

        assert results.get("method") == "single_simulation"
        clipped_params = results.get("optimized_params_final_clipped", {})
        assert np.isclose(clipped_params.get("rate"), 7500.0, rtol=1e-2)
        assert np.isclose(clipped_params.get("pressure"), 3400.0, rtol=1e-2)

    def test_run_single_simulation_callback(self, test_engine):
        """Verify that progress callback is called during single simulation."""
        messages = []

        def callback(msg: str):
            messages.append(msg)

        results = test_engine.run_single_simulation(text_progress_callback=callback)

        assert len(messages) >= 1
        assert any("simulation" in m.lower() for m in messages)
        assert results.get("method") == "single_simulation"


class TestOptimizationWidgetSimulationButton:
    """Test suite for UI button state and integration in OptimizationWidget."""

    @pytest.mark.usefixtures("qapp")
    def test_simulation_button_initialization_and_toggle(self, test_engine):
        """Verify that run_simulation_button is present, has correct text, and toggles with engine state."""
        from utils.config_manager import ConfigManager
        from ui.optimization_widget import OptimizationWidget

        config_manager = ConfigManager()
        widget = OptimizationWidget(config_manager=config_manager)

        # Initially without engine, button should be disabled
        assert hasattr(widget, "run_simulation_button")
        assert widget.run_simulation_button.text() == "Run Simulation"
        assert not widget.run_simulation_button.isEnabled()

        # Update with engine -> button should be enabled
        widget.update_engine(test_engine)
        assert widget.run_simulation_button.isEnabled()

        # Reset engine to None -> button should be disabled
        widget.update_engine(None)
        assert not widget.run_simulation_button.isEnabled()
