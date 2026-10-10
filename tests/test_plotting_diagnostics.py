"""
Unit tests for plotting diagnostics and failure penalty filtering in PlottingManager.
Verifies:
1. Well schedule graph generation in BO and fallback generation.
2. Objective vs parameter (BO) filtering of failure penalties (-1e12) and proper physical scaling.
3. Objective space scatter resolution of hybrid GA-BO and cumulative max best trajectory.
4. Pareto front handling for single-objective vs multi-objective hybrid runs.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import plotly.graph_objects as go

from core.plotting_manager import PlottingManager
from core.data_models import EORParameters, OperationalParameters


@pytest.fixture
def mock_engine():
    """Mock optimization engine with standard parameters and plotting manager."""
    engine = MagicMock()
    engine.chosen_objective = "npv"
    engine.eor_params = EORParameters()
    engine.operational_params = OperationalParameters()
    engine.multi_objective_names = ["NPV", "Recovery Factor"]
    return engine


def test_plot_well_schedule_from_profiles(mock_engine):
    """Test that plot_well_schedule renders correctly when well_schedule is present."""
    pm = PlottingManager(mock_engine)
    results = {
        "optimized_profiles": {
            "well_schedule": {
                "injection_scheme": "wag",
                "wells": [
                    {
                        "well_name": "Inj_1",
                        "operations": [
                            {"phase": "gas_injection", "start_day": 0, "duration_days": 30, "rate_mscfd": 2500},
                            {"phase": "water_injection", "start_day": 30, "duration_days": 30, "rate_mscfd": 1500},
                        ],
                    }
                ],
            }
        }
    }
    fig = pm.plot_well_schedule(results)
    assert isinstance(fig, go.Figure)
    assert "Well Schedule" in fig.layout.title.text
    assert len(fig.data) > 0


def test_plot_well_schedule_fallback_generation(mock_engine):
    """Test that plot_well_schedule generates fallback schedule when well_schedule is missing."""
    mock_engine._generate_well_schedule_from_params.return_value = {
        "injection_scheme": "continuous",
        "wells": [
            {
                "well_name": "Inj_Default",
                "operations": [
                    {"phase": "injection", "start_day": 0, "duration_days": 365, "rate_mscfd": 2000}
                ],
            }
        ],
    }
    pm = PlottingManager(mock_engine)
    results = {
        "optimized_profiles": {}  # missing well_schedule
    }
    fig = pm.plot_well_schedule(results)
    assert isinstance(fig, go.Figure)
    assert "Well Schedule" in fig.layout.title.text
    assert len(fig.data) > 0
    mock_engine._generate_well_schedule_from_params.assert_called_once()


def test_plot_objective_vs_parameter_filters_failure_penalty(mock_engine):
    """Test that plot_objective_vs_parameter isolates -1e12 failure penalties from y-axis scale."""
    pm = PlottingManager(mock_engine)

    # Simulate BO evaluations: 3 feasible points (~$1M - $1.5M) and 2 failed points (-1e12)
    results = {
        "bayes_opt_obj": MagicMock(
            res=[
                {"params": {"injection_rate": 1000}, "target": 1.2e6},
                {"params": {"injection_rate": 2000}, "target": 1.5e6},
                {"params": {"injection_rate": 2500}, "target": 1.3e6},
                {"params": {"injection_rate": 4000}, "target": -1e12},  # constraint violation (breakthrough)
                {"params": {"injection_rate": 5000}, "target": -1e12},  # constraint violation
            ]
        )
    }

    fig = pm.plot_objective_vs_parameter("injection_rate", results)
    assert isinstance(fig, go.Figure)

    # Verify Y-axis range is bounded by physical values (~1.2e6 - 1.5e6), NOT -1e12
    yaxis_range = fig.layout.yaxis.range
    assert yaxis_range is not None
    assert yaxis_range[0] > -1e9  # Min y-axis should be close to 1.2e6, never -1e12
    assert yaxis_range[1] >= 1.5e6

    # Verify separate feasible and infeasible traces
    trace_names = [t.name for t in fig.data]
    assert "Feasible Evaluations" in trace_names
    assert any("Infeasible" in name for name in trace_names)


def test_plot_objective_space_scatter_hybrid_and_best_trajectory(mock_engine):
    """Test that plot_objective_space_scatter handles hybrid GA-BO and tracks cumulative max for BO."""
    pm = PlottingManager(mock_engine)

    # Mock GA instance
    mock_ga = MagicMock()
    mock_ga.all_fitness = [
        np.array([1.1e6, 1.15e6, -1e12]),  # Gen 1 with a failure penalty
        np.array([1.2e6, 1.25e6, 1.3e6]),   # Gen 2
    ]
    mock_ga.best_solutions_fitness = [1.15e6, 1.3e6]

    # Mock BO results with mixed feasible and failure penalties
    results = {
        "method": "hybrid_ga_bo",
        "ga_full_results_for_hybrid": {
            "pygad_instance": mock_ga,
        },
        "bayes_opt_obj": MagicMock(
            res=[
                {"params": {}, "target": -1e12},  # Iter 1: constraint violation
                {"params": {}, "target": 1.35e6}, # Iter 2: better
                {"params": {}, "target": -1e12},  # Iter 3: violation
                {"params": {}, "target": 1.45e6}, # Iter 4: best
            ]
        ),
    }

    fig = pm.plot_objective_space_scatter(results)
    assert isinstance(fig, go.Figure)

    # Check y-axis range does not drop to -1e12
    yaxis_range = fig.layout.yaxis.range
    assert yaxis_range is not None
    assert yaxis_range[0] > -1e9

    # Check traces
    trace_names = [t.name for t in fig.data]
    assert "GA Solutions" in trace_names
    assert "GA Best Fitness" in trace_names
    assert "BO Feasible Solutions" in trace_names
    assert "BO Best So Far" in trace_names

    # Check that BO Best So Far tracks cumulative max: should start at 1.35e6 and rise to 1.45e6
    bo_best_trace = next(t for t in fig.data if t.name == "BO Best So Far")
    assert all(val > 1.3e6 for val in bo_best_trace.y)
    assert bo_best_trace.y[-1] == 1.45e6


def test_plot_pareto_front_single_vs_multi_objective(mock_engine):
    """Test plot_pareto_front messaging for single-objective and rendering for multi-objective."""
    pm = PlottingManager(mock_engine)

    # Single-objective run: hybrid_ga_bo
    single_res = {
        "method": "hybrid_ga_bo",
    }
    fig_single = pm.plot_pareto_front(single_res)
    assert "single-objective" in fig_single.layout.title.text.lower()

    # Multi-objective run: hybrid_nsga2_bo with pareto_front
    multi_res = {
        "method": "hybrid_nsga2_bo",
        "pareto_front": [
            {"params": {}, "objectives": [1.2e6, 0.35]},
            {"params": {}, "objectives": [1.4e6, 0.38]},
        ],
    }
    fig_multi = pm.plot_pareto_front(multi_res)
    assert "Pareto Front" in fig_multi.layout.title.text
    trace_names = [t.name for t in fig_multi.data]
    assert "Pareto Front" in trace_names


def test_plot_well_schedule_formatting_and_no_duplicate_legend(mock_engine):
    """Test that well schedule bars use correct width/center and deduplicate legend entries."""
    pm = PlottingManager(mock_engine)
    results = {
        "optimized_profiles": {
            "well_schedule": {
                "injection_scheme": "continuous",
                "wells": [
                    {
                        "well_name": "Well-Injector-1",
                        "well_type": "injector",
                        "operations": [
                            {"phase": "injection", "start_day": 0, "duration_days": 365, "rate_mscfd": 2500},
                            {"phase": "injection", "start_day": 365, "duration_days": 365, "rate_mscfd": 2500},
                            {"phase": "injection", "start_day": 730, "duration_days": 365, "rate_mscfd": 2500},
                            {"phase": "idle", "start_day": 1095, "duration_days": 60, "rate_mscfd": 0},
                        ],
                    }
                ],
            }
        }
    }
    fig = pm.plot_well_schedule(results)
    assert isinstance(fig, go.Figure)
    assert fig.layout.barmode == "overlay"

    # Verify that 'injection' legend entry appears at most once
    legend_shown = [t.name for t in fig.data if getattr(t, "showlegend", False)]
    assert legend_shown.count("Injection") <= 1
    assert any("Idle" in name for name in legend_shown)

    # Verify bar width and center
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) >= 3
    first_bar = bar_traces[0]
    assert first_bar.x[0] == 182.5  # center of 0 to 365
    assert first_bar.width[0] == 365  # width is 365, NOT collapsed to 0.8 days!


def test_plot_coverage_trend_min_max(mock_engine):
    """Test that plot_coverage renders Trend, Min, and Max with shaded envelope."""
    pm = PlottingManager(mock_engine)

    # Multi-generation coverage history
    results = {
        "method": "genetic_algorithm",
        "ga_statistics": {
            "coverage_history": [
                {"generation": 1, "min": 0.35, "max": 0.60, "mean": 0.48, "std": 0.08},
                {"generation": 2, "min": 0.42, "max": 0.72, "mean": 0.58, "std": 0.09},
                {"generation": 3, "min": 0.50, "max": 0.85, "mean": 0.70, "std": 0.07},
            ]
        },
    }

    fig = pm.plot_coverage(results)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Max Coverage" in trace_names
    assert "Min Coverage" in trace_names
    assert "Trend (Mean Coverage)" in trace_names
    assert "Min-Max Coverage Span" in trace_names


def test_plot_euclidean_distance_matrix(mock_engine):
    """Test that plot_euclidean_distance_matrix computes symmetric distance matrix for Phase 2."""
    mock_engine._get_parameter_bounds.return_value = {
        "injection_rate": (1000.0, 5000.0),
        "wag_ratio": (0.5, 3.0),
        "cycle_length_days": (15.0, 120.0),
    }
    pm = PlottingManager(mock_engine)

    # Phase 2 diverse points transferred to BO
    results = {
        "method": "hybrid_ga_bo",
        "diverse_points_for_bo": [
            {"params": {"injection_rate": 1500.0, "wag_ratio": 1.0, "cycle_length_days": 30.0}},
            {"params": {"injection_rate": 3500.0, "wag_ratio": 2.0, "cycle_length_days": 60.0}},
            {"params": {"injection_rate": 4500.0, "wag_ratio": 2.5, "cycle_length_days": 90.0}},
        ]
    }

    fig = pm.plot_euclidean_distance_matrix(results)
    assert isinstance(fig, go.Figure)
    heatmap = fig.data[0]
    assert isinstance(heatmap, go.Heatmap)
    z_mat = np.array(heatmap.z)
    assert z_mat.shape == (3, 3)
    # Check diagonal is zero
    assert np.allclose(np.diag(z_mat), 0.0)
    # Check symmetry
    assert np.allclose(z_mat, z_mat.T)
    # Check off-diagonal distance > 0
    assert z_mat[0, 1] > 0.0

