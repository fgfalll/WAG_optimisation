import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any
from copy import deepcopy

logger = logging.getLogger(__name__)


class PlottingManager:
    def __init__(self, engine):
        self.engine = engine

    def plot_optimization_convergence(self, results_to_plot=None) -> go.Figure:
        source = results_to_plot or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")
        fig = go.Figure()
        title = (
            f"Optimization Convergence for {self.engine.chosen_objective.replace('_', ' ').title()}"
        )

        last_eval = 0

        ga_res = source.get(
            "ga_full_results_for_hybrid",
            source if source.get("method") == "genetic_algorithm" else None,
        )
        if ga_res:
            history = None
            sol_per_pop = None
            avg_history = None
            std_history = None

            if ga_instance := ga_res.get("pygad_instance"):
                history = ga_instance.best_solutions_fitness
                sol_per_pop = ga_instance.sol_per_pop
                worst_history = None
                median_history = None
                if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness:
                    filtered_fitness = []
                    for fitness in ga_instance.all_fitness:
                        arr = np.array(fitness)
                        valid_arr = arr[arr > -1e9]
                        filtered_fitness.append(valid_arr if len(valid_arr) > 0 else arr)
                    avg_history = [float(np.mean(f)) for f in filtered_fitness]
                    std_history = [float(np.std(f)) for f in filtered_fitness]
                    worst_history = [float(np.min(f)) for f in filtered_fitness]
                    median_history = [float(np.median(f)) for f in filtered_fitness]
            elif "ga_statistics" in ga_res:
                history = ga_res["ga_statistics"].get("max_fitness_history") or ga_res["ga_statistics"].get("best_fitness_history")
                avg_history = ga_res["ga_statistics"].get("avg_fitness_history")
                std_history = ga_res["ga_statistics"].get("std_fitness_history")
                sol_per_pop = ga_res["ga_statistics"].get("population_size")
                worst_history = ga_res["ga_statistics"].get("min_fitness_history")
                median_history = None

            if history and sol_per_pop:
                evals = np.arange(1, len(history) + 1) * sol_per_pop

                # 1. Shaded Min-Max envelope if min fitness is recorded
                if worst_history and len(worst_history) == len(history):
                    fig.add_trace(
                        go.Scatter(
                            x=evals,
                            y=history,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=evals,
                            y=worst_history,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(31, 119, 180, 0.12)",
                            name="Min-Max Envelope",
                            hoverinfo="skip",
                        )
                    )

                # 2. Max (Best) Fitness trace
                fig.add_trace(
                    go.Scatter(
                        x=evals,
                        y=history,
                        mode="lines+markers",
                        name="GA Best (Max) Fitness",
                        line=dict(color="#1f77b4", width=2.5),
                        marker=dict(size=5),
                    )
                )

                # 3. Min (Worst) Fitness trace
                if worst_history:
                    fig.add_trace(
                        go.Scatter(
                            x=evals,
                            y=worst_history,
                            mode="lines",
                            name="GA Worst (Min) Fitness",
                            line=dict(color="#d62728", dash="dot", width=1.5),
                        )
                    )

                # 4. Trend (Mean) Fitness trace
                if avg_history:
                    fig.add_trace(
                        go.Scatter(
                            x=evals,
                            y=avg_history,
                            mode="lines",
                            name="GA Trend (Mean Fitness)",
                            line=dict(color="#ff7f0e", width=2.5),
                        )
                    )
                    if std_history and not worst_history:
                        fig.add_trace(
                            go.Scatter(
                                x=evals,
                                y=np.array(avg_history) + np.array(std_history),
                                fill="tonexty",
                                mode="lines",
                                line=dict(color="rgba(255,165,0,0.2)"),
                                name="Std Dev",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=evals,
                                y=np.array(avg_history) - np.array(std_history),
                                fill="tonexty",
                                mode="lines",
                                line=dict(color="rgba(255,165,0,0.2)"),
                                showlegend=False,
                            )
                        )

                if median_history:
                    fig.add_trace(
                        go.Scatter(
                            x=evals,
                            y=median_history,
                            mode="lines",
                            name="GA Median Fitness",
                            line=dict(color="purple", dash="dash"),
                        )
                    )
                last_eval = evals[-1] if evals.size > 0 else 0

        all_plot_y = []
        if ga_res and history:
            all_plot_y.extend([float(h) for h in history if float(h) > -1e9])

        if "bayes_opt_obj" in source and hasattr(source["bayes_opt_obj"], "y"):
            bo_y_raw = np.array(source["bayes_opt_obj"].y)
            bo_x_raw = np.arange(last_eval + 1, last_eval + 1 + len(bo_y_raw))
            feasible_mask = bo_y_raw > -1e9
            if np.any(feasible_mask):
                fig.add_trace(
                    go.Scatter(
                        x=bo_x_raw[feasible_mask],
                        y=bo_y_raw[feasible_mask],
                        mode="markers",
                        marker=dict(size=7, color="#2ca02c"),
                        name="BO Evaluations",
                    )
                )
                all_plot_y.extend(bo_y_raw[feasible_mask])

            infeasible_mask = ~feasible_mask
            if np.any(infeasible_mask):
                baseline_y = min(all_plot_y) if all_plot_y else 0.0
                fig.add_trace(
                    go.Scatter(
                        x=bo_x_raw[infeasible_mask],
                        y=[baseline_y] * int(np.sum(infeasible_mask)),
                        mode="markers",
                        marker=dict(symbol="x", color="red", size=6),
                        name=f"BO Infeasible / Pruned ({int(np.sum(infeasible_mask))})",
                        hovertemplate="Eval %{x}<br>Status: Infeasible (Constraint Violated)<extra></extra>",
                    )
                )

        if all_plot_y:
            y_min = float(min(all_plot_y))
            y_max = float(max(all_plot_y))
            y_pad = max((y_max - y_min) * 0.1, 1.0)
            fig.update_layout(yaxis=dict(range=[y_min - y_pad, y_max + y_pad]))

        fig.update_layout(
            title_text=title,
            xaxis_title_text="Function Evaluations",
            yaxis_title_text="Objective Value",
        )
        return fig

    def plot_parameter_sensitivity(self, param_name, results_to_use=None) -> go.Figure:
        source = results_to_use or self.engine._results
        if not (source and "optimized_params_final_clipped" in source):
            return go.Figure().update_layout(title_text="No optimized parameters available.")

        opt_base = source["optimized_params_final_clipped"]
        all_bounds = self.engine._get_parameter_bounds()

        if param_name not in opt_base:
            return go.Figure().update_layout(title_text=f"Parameter '{param_name}' not in results.")

        curr_val = opt_base[param_name]
        range_multiplier = self.engine.advanced_engine_params.sensitivity_plot_range_multiplier

        b_entry = all_bounds.get(
            param_name, (curr_val * (1 - range_multiplier), curr_val * (1 + range_multiplier))
        )
        if isinstance(b_entry, dict):
            low_bound, high_bound = b_entry["low"], b_entry["high"]
        else:
            low_bound, high_bound = b_entry[0], b_entry[1]
        test_values = np.linspace(low_bound, high_bound, 10)
        objective_values = []

        for test_val in test_values:
            test_params = opt_base.copy()
            test_params[param_name] = test_val
            eval_result = self.engine.evaluate_for_analysis(test_params)
            obj_val = eval_result.get(self.engine.chosen_objective, 0.0)
            objective_values.append(obj_val)

        fig = go.Figure(go.Scatter(x=test_values, y=objective_values, mode="lines+markers"))
        fig.update_layout(
            title_text=f"Sensitivity: {param_name.replace('_', ' ').title()}",
            xaxis_title=param_name.replace("_", " ").title(),
            yaxis_title=self.engine.chosen_objective.replace("_", " ").title(),
        )
        return fig

    def plot_production_profiles(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Generates a plot of production profiles from optimization results."""
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        profiles = source.get("optimized_profiles", {})
        if not profiles:
            return go.Figure().update_layout(title_text="No profile data found in results.")

        resolution = self.engine.operational_params.time_resolution
        profile_key = f"{resolution}_oil_stb"
        if profile_key not in profiles:
            return go.Figure().update_layout(title_text=f"No data for '{resolution}' resolution.")

        # Use actual time values from profiles if available, otherwise use integer indices
        time_key = f"{resolution}_time_years"
        if time_key in profiles:
            time_steps = np.array(profiles[time_key])
        elif "time_vector" in profiles:
            tv = np.array(profiles["time_vector"])
            time_steps = tv / 365.25 if tv[-1] > 100 else tv  # Convert days to years if needed
        else:
            time_steps = np.arange(1, len(profiles[profile_key]) + 1)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=time_steps, y=profiles[f"{resolution}_oil_stb"], name="Oil (bbl)"),
            secondary_y=False,
        )

        water_inj_key = f"{resolution}_water_injected_bbl"
        if water_inj_key in profiles:
            fig.add_trace(
                go.Scatter(x=time_steps, y=profiles[water_inj_key], name="Water Inj (bbl)"),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=time_steps, y=profiles[f"{resolution}_co2_injected_mscf"], name="CO2 Inj (MSCF)"
            ),
            secondary_y=True,
        )

        # Add volumetric sweep efficiency
        sweep_key = f"{resolution}_volumetric_sweep"
        if sweep_key in profiles:
            sweep_profile = profiles[sweep_key]
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=sweep_profile,
                    name="Volumetric Sweep",
                    yaxis="y3",
                    line=dict(color="purple", dash="dot"),
                )
            )

            avg_sweep = np.mean(sweep_profile)
            std_sweep = np.std(sweep_profile)

            fig.update_layout(
                yaxis3=dict(
                    title="Volumetric Sweep Efficiency",
                    overlaying="y",
                    side="right",
                    position=0.9,
                    showgrid=False,
                    range=[0, 1],
                ),
                annotations=[
                    dict(
                        x=0.95,
                        y=0.05,
                        xref="paper",
                        yref="paper",
                        text=f"Avg Sweep: {avg_sweep:.2f}<br>Std Dev: {std_sweep:.2f}",
                        showarrow=False,
                        align="left",
                        bordercolor="black",
                        borderwidth=1,
                    )
                ],
            )

        fig.update_layout(
            title_text=f"Optimized {resolution.title()} Profiles",
            xaxis_title=f"Project {resolution.title()}",
            barmode="group",
        )
        return fig

    def plot_objective_vs_parameter(
        self, param_name: str, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Plots objective values against a specific parameter for Bayesian optimization results."""
        source = results_to_use or self.engine._results

        bo_res = None
        if source and "bayes_opt_obj" in source and hasattr(source["bayes_opt_obj"], "res"):
            bo_res = source["bayes_opt_obj"].res
        elif source and "bayes_opt_obj_res" in source:
            bo_res = source["bayes_opt_obj_res"]

        if not bo_res:
            return go.Figure().update_layout(
                title_text="No Bayesian optimization results available."
            )

        feasible_params = []
        feasible_targets = []
        infeasible_params = []
        infeasible_targets = []

        for res in bo_res:
            if "params" in res and param_name in res["params"]:
                p_val = float(res["params"][param_name])
                t_val = float(res.get("target", 0.0))
                if t_val > -1e9:
                    feasible_params.append(p_val)
                    feasible_targets.append(t_val)
                else:
                    infeasible_params.append(p_val)
                    infeasible_targets.append(t_val)

        if not feasible_params and not infeasible_params:
            return go.Figure().update_layout(
                title_text=f"Parameter '{param_name}' not found in results."
            )

        fig = go.Figure()
        obj_display = self.engine.chosen_objective.replace("_", " ").title()

        if feasible_params:
            sort_idx = np.argsort(feasible_params)
            f_x_sorted = np.array(feasible_params)[sort_idx]
            f_y_sorted = np.array(feasible_targets)[sort_idx]

            fig.add_trace(
                go.Scatter(
                    x=f_x_sorted,
                    y=f_y_sorted,
                    mode="markers",
                    name="Feasible Evaluations",
                    marker=dict(
                        color="#1f77b4",
                        size=9,
                        line=dict(width=1, color="#0c3d66"),
                    ),
                    hovertemplate=(
                        f"{param_name.replace('_', ' ').title()}: %{{x:.4f}}<br>"
                        f"{obj_display}: %{{y:,.2f}}<extra></extra>"
                    ),
                )
            )

            # Trend line if at least 3 unique feasible points exist
            unique_x = np.unique(f_x_sorted)
            if len(unique_x) >= 3:
                try:
                    poly_deg = 2 if len(unique_x) >= 5 else 1
                    poly_coeffs = np.polyfit(f_x_sorted, f_y_sorted, poly_deg)
                    x_trend = np.linspace(f_x_sorted[0], f_x_sorted[-1], 50)
                    y_trend = np.polyval(poly_coeffs, x_trend)
                    fig.add_trace(
                        go.Scatter(
                            x=x_trend,
                            y=y_trend,
                            mode="lines",
                            line=dict(color="#1f77b4", width=2, dash="dash"),
                            name="Trend",
                        )
                    )
                except Exception:
                    pass

            y_min = float(np.min(feasible_targets))
            y_max = float(np.max(feasible_targets))
            y_range = max(y_max - y_min, 1.0)

            # Highlight best point
            best_idx = int(np.argmax(feasible_targets))
            fig.add_trace(
                go.Scatter(
                    x=[feasible_params[best_idx]],
                    y=[feasible_targets[best_idx]],
                    mode="markers",
                    name="Best Found",
                    marker=dict(symbol="star", color="#2ca02c", size=14, line=dict(color="black", width=1.5)),
                    hovertemplate=(
                        f"<b>Best Found</b><br>"
                        f"{param_name.replace('_', ' ').title()}: %{{x:.4f}}<br>"
                        f"{obj_display}: %{{y:,.2f}}<extra></extra>"
                    ),
                )
            )

            # If there are infeasible points, mark them at the baseline without corrupting the y-axis
            if infeasible_params:
                y_baseline = y_min - 0.05 * y_range
                fig.add_trace(
                    go.Scatter(
                        x=infeasible_params,
                        y=[y_baseline] * len(infeasible_params),
                        mode="markers",
                        name=f"Infeasible / Pruned ({len(infeasible_params)})",
                        marker=dict(symbol="x", color="#d62728", size=8, line=dict(width=2)),
                        hovertemplate=(
                            f"{param_name.replace('_', ' ').title()}: %{{x:.4f}}<br>"
                            f"Status: Infeasible (Constraint Violated / Pruned)<extra></extra>"
                        ),
                    )
                )
                fig.update_layout(yaxis=dict(range=[y_min - 0.12 * y_range, y_max + 0.08 * y_range]))
            else:
                fig.update_layout(yaxis=dict(range=[y_min - 0.08 * y_range, y_max + 0.08 * y_range]))
        else:
            fig.add_trace(
                go.Scatter(
                    x=infeasible_params,
                    y=infeasible_targets,
                    mode="markers",
                    name="Infeasible Evaluations",
                    marker=dict(symbol="x", color="#d62728", size=8),
                )
            )

        fig.update_layout(
            title_text=f"{obj_display} vs. {param_name.replace('_', ' ').title()} (Bayesian Optimization)",
            xaxis_title=param_name.replace("_", " ").title(),
            yaxis_title=obj_display,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_co2_performance_summary_table(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Generates a table summarizing CO2 performance metrics, using material balance analysis if available."""
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        mb_analysis = source.get("material_balance_analysis") or {}
        mb_stats = mb_analysis.get("summary_statistics", {})
        final_metrics = source.get("final_metrics", {})

        co2_density_tonne_per_mscf = self.engine.eor_params.co2_density_tonne_per_mscf

        if mb_stats:
            # Use data from material balance analysis for consistency
            total_injected_tonne = mb_stats.get("total_injected_tonne", 0)
            total_produced_tonne = mb_stats.get("total_produced_tonne", 0)
            total_stored_tonne = mb_stats.get("total_net_stored_tonne", 0)
            avg_efficiency = mb_stats.get("avg_storage_efficiency", 0)

            # Convert tonnes back to MSCF for display consistency in the table
            total_injected_mscf = (
                total_injected_tonne / co2_density_tonne_per_mscf
                if co2_density_tonne_per_mscf > 0
                else 0
            )
            total_produced_mscf = (
                total_produced_tonne / co2_density_tonne_per_mscf
                if co2_density_tonne_per_mscf > 0
                else 0
            )
        else:
            # Fallback to original method if material balance data is not present
            profiles = source.get("optimized_profiles", {})
            op_params = source.get("operational_parameters", self.engine.operational_params)
            resolution = op_params.time_resolution

            total_injected_mscf = np.sum(profiles.get(f"{resolution}_co2_injected_mscf", 0))
            total_produced_mscf = np.sum(profiles.get(f"{resolution}_co2_produced_mscf", 0))
            total_stored_tonne = final_metrics.get("total_co2_stored_tonne", 0)
            avg_efficiency = final_metrics.get("avg_storage_efficiency", 0)

        co2_utilization = final_metrics.get("co2_utilization", 0)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["Metric", "Value", "Units"]),
                    cells=dict(
                        values=[
                            [
                                "Total CO2 Injected",
                                "Total CO2 Produced",
                                "Total CO2 Stored",
                                "Average Storage Efficiency",
                                "CO2 Utilization",
                            ],
                            [
                                f"{total_injected_mscf:,.0f}",
                                f"{total_produced_mscf:,.0f}",
                                f"{total_stored_tonne:,.0f}",
                                f"{avg_efficiency:.2%}",
                                f"{co2_utilization:.2f}",
                            ],
                            ["MSCF", "MSCF", "tonnes", "%", "MSCF/stb"],
                        ]
                    ),
                )
            ]
        )
        fig.update_layout(title_text="CO2 Performance Summary")
        return fig

    def plot_ga_objective_distribution(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        source = results_to_use or self.engine._results
        if not (source and "pygad_instance" in source):
            return go.Figure().update_layout(title_text="No GA results to plot.")

        ga_instance = source["pygad_instance"]
        objectives = ga_instance.last_generation_fitness

        avg_obj = np.mean(objectives)
        std_obj = np.std(objectives)

        fig = go.Figure(data=[go.Histogram(x=objectives, nbinsx=20)])
        fig.update_layout(
            title_text="GA Population Objective Value Distribution",
            xaxis_title="Objective Value",
            yaxis_title="Frequency",
            annotations=[
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Avg: {avg_obj:.3f}<br>Std: {std_obj:.3f}",
                    showarrow=False,
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                )
            ],
        )
        return fig

    def plot_co2_breakthrough(
        self,
        years: np.ndarray,
        gor_profile: np.ndarray,
        breakthrough_time: float,
        co2_production: Optional[np.ndarray] = None,
        saturation_profile: Optional[np.ndarray] = None,
    ) -> go.Figure:
        """
        Plot CO2 breakthrough analysis with GOR evolution.

        Args:
            years: Time array in years
            gor_profile: Gas-Oil Ratio evolution (scf/stb)
            breakthrough_time: Theoretical breakthrough time (years)
            co2_production: Optional CO2 production rate array (MSCFD)
            saturation_profile: Optional CO2 saturation profile at producer

        Returns:
            Plotly Figure with breakthrough analysis
        """
        try:
            from visualizations.breakthrough_plots import BreakthroughVisualizer

            visualizer = BreakthroughVisualizer()
            return visualizer.plot_breakthrough_comprehensive(
                years=years,
                gor_profile=gor_profile,
                breakthrough_time=breakthrough_time,
                co2_production=co2_production,
                saturation_profile=saturation_profile,
            )
        except ImportError:
            logger.debug("Optional visualizations module not installed.")
            return go.Figure().update_layout(title_text="Breakthrough visualization not available")

    def plot_fault_mechanics(self, fault_history) -> go.Figure:
        """
        Plot fault stability analysis over time.

        Args:
            fault_history: List of dictionaries containing fault state over time
                Each dict should have keys:
                - time_years: Time in years
                - failure_ratio: Ratio of shear stress to shear strength
                - cumulative_slip: Total accumulated slip (mm)
                - transmissibility: Fault transmissibility multiplier

        Returns:
            Plotly Figure with fault mechanics analysis
        """
        try:
            from visualizations.fault_mechanics_plots import FaultMechanicsVisualizer

            visualizer = FaultMechanicsVisualizer()
            return visualizer.plot_fault_stability_comprehensive(fault_history)
        except ImportError:
            logger.debug("Optional visualizations module not installed.")
            return go.Figure().update_layout(title_text="Fault mechanics visualization not available")

    def plot_scenario_comparison(
        self, results_with_fault: Dict[str, Any], results_without_fault: Dict[str, Any]
    ) -> go.Figure:
        """
        Plot side-by-side comparison of scenarios (with/without fault mechanics).

        Args:
            results_with_fault: Simulation results with fault mechanics enabled
            results_without_fault: Simulation results without fault mechanics

        Returns:
            Plotly Figure with side-by-side comparison
        """
        try:
            from visualizations.comparison_plots import ScenarioComparisonVisualizer

            visualizer = ScenarioComparisonVisualizer()
            return visualizer.plot_scenario_comparison(results_with_fault, results_without_fault)
        except ImportError:
            logger.debug("Optional visualizations module not installed.")
            return go.Figure().update_layout(title_text="Scenario comparison visualization not available")

    def plot_fault_effect_analysis(
        self, results_with_fault: Dict[str, Any], results_without_fault: Dict[str, Any]
    ) -> go.Figure:
        """
        Plot fault effect analysis with delta curves.

        Args:
            results_with_fault: Simulation results with fault mechanics
            results_without_fault: Simulation results without fault mechanics

        Returns:
            Plotly Figure with fault effect analysis
        """
        try:
            from visualizations.comparison_plots import ScenarioComparisonVisualizer

            visualizer = ScenarioComparisonVisualizer()
            return visualizer.plot_fault_effect_analysis(results_with_fault, results_without_fault)
        except ImportError:
            logger.debug("Optional visualizations module not installed.")
            return go.Figure().update_layout(title_text="Fault effect analysis visualization not available")

    def plot_objective_space_scatter(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Plot all evaluated solutions in objective space, color-coded by generation.
        Shows convergence quality and solution diversity.
        """
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        fig = go.Figure()
        method = source.get("method", "")
        is_hybrid = "hybrid" in method.lower()
        is_nsga = "nsga" in method.lower()

        if is_nsga:
            ga_res = source.get("nsga2_full_results_for_hybrid", source)
        elif is_hybrid:
            ga_res = (
                source.get("ga_full_results_for_hybrid")
                or source.get("nsga2_full_results_for_hybrid")
                or source
            )
        else:
            ga_res = source.get(
                "ga_full_results_for_hybrid",
                source if method == "genetic_algorithm" else None,
            )

        all_valid_y = []

        if ga_res and (ga_instance := ga_res.get("pygad_instance")):
            if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness:
                all_fitness = ga_instance.all_fitness
                n_generations = len(all_fitness)

                filtered_gens = []
                filtered_objs = []
                filtered_colors = []

                for gen_idx, fitness_array in enumerate(all_fitness):
                    norm_gen = gen_idx / max(n_generations - 1, 1)
                    for fit_val in fitness_array:
                        val = float(np.asarray(fit_val).flat[0]) if hasattr(fit_val, "__len__") else float(fit_val)
                        if val > -1e9:
                            filtered_gens.append(gen_idx + 1)
                            filtered_objs.append(val)
                            filtered_colors.append(norm_gen)

                if filtered_objs:
                    all_valid_y.extend(filtered_objs)
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_gens,
                            y=filtered_objs,
                            mode="markers",
                            marker=dict(
                                color=filtered_colors,
                                colorscale="RdYlBu_r",
                                colorbar=dict(title="GA Gen", x=1.02),
                                size=6,
                                opacity=0.7,
                            ),
                            text=[f"Gen {g}" for g in filtered_gens],
                            hovertemplate="Generation %{text}<br>Objective: %{y:,.2f}<extra></extra>",
                            name="GA Solutions",
                        )
                    )

                if hasattr(ga_instance, "best_solutions_fitness"):
                    best_fitness = [
                        float(np.asarray(f).flat[0]) if hasattr(f, "__len__") else float(f)
                        for f in ga_instance.best_solutions_fitness
                    ]
                    valid_best = [f for f in best_fitness if f > -1e9]
                    if valid_best:
                        all_valid_y.extend(valid_best)
                        best_gens = np.arange(1, len(best_fitness) + 1)
                        fig.add_trace(
                            go.Scatter(
                                x=best_gens,
                                y=best_fitness,
                                mode="lines",
                                line=dict(color="black", width=3),
                                name="GA Best Fitness",
                            )
                        )

        bo_res = None
        if "bayes_opt_obj" in source and hasattr(source["bayes_opt_obj"], "res"):
            bo_res = source["bayes_opt_obj"].res
        elif "bayes_opt_obj_res" in source:
            bo_res = source["bayes_opt_obj_res"]

        if bo_res and len(bo_res) > 0:
            n_bo = len(bo_res)
            bo_raw_objectives = [float(res.get("target", 0.0)) for res in bo_res]
            bo_iterations = np.arange(1, n_bo + 1)

            feasible_bo_iters = []
            feasible_bo_objs = []
            feasible_bo_colors = []
            infeasible_bo_iters = []

            for i, target in enumerate(bo_raw_objectives, 1):
                if target > -1e9:
                    feasible_bo_iters.append(i)
                    feasible_bo_objs.append(target)
                    feasible_bo_colors.append((i - 1) / max(n_bo - 1, 1))
                else:
                    infeasible_bo_iters.append(i)

            if feasible_bo_objs:
                all_valid_y.extend(feasible_bo_objs)
                fig.add_trace(
                    go.Scatter(
                        x=feasible_bo_iters,
                        y=feasible_bo_objs,
                        mode="markers",
                        marker=dict(
                            color=feasible_bo_colors,
                            colorscale="Viridis",
                            colorbar=dict(title="BO Iter", x=1.12),
                            size=8,
                            symbol="diamond",
                        ),
                        text=[f"BO Iter {i}" for i in feasible_bo_iters],
                        hovertemplate="Iteration %{text}<br>Objective: %{y:,.2f}<extra></extra>",
                        name="BO Feasible Solutions",
                    )
                )

            # Cumulative best-so-far for BO (MAXIMIZATION, tracking feasible best)
            bo_best_y = []
            current_max = None
            for tgt in bo_raw_objectives:
                if tgt > -1e9:
                    if current_max is None or tgt > current_max:
                        current_max = tgt
                bo_best_y.append(current_max)

            valid_best_iters = [i for i, b in zip(bo_iterations, bo_best_y) if b is not None]
            valid_best_vals = [b for b in bo_best_y if b is not None]
            if valid_best_vals:
                fig.add_trace(
                    go.Scatter(
                        x=valid_best_iters,
                        y=valid_best_vals,
                        mode="lines",
                        line=dict(color="#2ca02c", width=2.5),
                        name="BO Best So Far",
                    )
                )

            if infeasible_bo_iters:
                y_base = min(all_valid_y) if all_valid_y else 0.0
                fig.add_trace(
                    go.Scatter(
                        x=infeasible_bo_iters,
                        y=[y_base] * len(infeasible_bo_iters),
                        mode="markers",
                        marker=dict(symbol="x", color="#d62728", size=7),
                        name=f"BO Infeasible / Pruned ({len(infeasible_bo_iters)})",
                        hovertemplate="Iteration %{x}<br>Status: Infeasible (Constraint Violated / Pruned)<extra></extra>",
                    )
                )

        if all_valid_y:
            y_min = float(min(all_valid_y))
            y_max = float(max(all_valid_y))
            y_pad = max((y_max - y_min) * 0.1, 1.0)
            fig.update_layout(yaxis=dict(range=[y_min - y_pad, y_max + y_pad]))

        fig.update_layout(
            title_text="Objective Space Scatter",
            xaxis_title_text="Generation / Iteration",
            yaxis_title_text=self.engine.chosen_objective.replace("_", " ").title(),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_pareto_front(self, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Plot Pareto front with all evaluated solutions as background.
        For multi-objective optimization (NSGA-II, Hybrid).
        """
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        method = source.get("method", "")
        is_nsga = "nsga" in method.lower()
        is_hybrid_nsga = method == "hybrid_nsga2_bo"
        has_pareto = bool(
            source.get("pareto_front")
            or source.get("nsga2_full_results_for_hybrid", {}).get("pareto_front")
        )

        if not (is_nsga or is_hybrid_nsga or has_pareto):
            method_display = method.replace("_", " ").title() if method else "Current"
            return go.Figure().update_layout(
                title_text=(
                    f"Pareto Front is only available for multi-objective optimization (NSGA-II or Hybrid NSGA-II + BO). "
                    f"'{method_display}' is a single-objective optimization method."
                )
            )

        pareto_front = source.get("pareto_front", [])
        if not pareto_front:
            nsga2_res = source.get("nsga2_full_results_for_hybrid", {})
            pareto_front = nsga2_res.get("pareto_front", [])

        if not pareto_front:
            return go.Figure().update_layout(title_text="No Pareto front data available.")

        fig = go.Figure()

        if is_hybrid_nsga or (is_nsga and "nsga2_full_results_for_hybrid" in source):
            nsga2_res = source.get("nsga2_full_results_for_hybrid", {})
            ga_instance = nsga2_res.get("pygad_instance")
            if ga_instance and hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness:
                all_fitness = ga_instance.all_fitness
                n_generations = len(all_fitness)

                for gen_idx, fitness_array in enumerate(all_fitness):
                    normalized_gen = gen_idx / max(n_generations - 1, 1)
                    obj1_vals = [f[0] if len(f) > 0 else 0 for f in fitness_array]
                    obj2_vals = [f[1] if len(f) > 1 else 0 for f in fitness_array]

                    fig.add_trace(
                        go.Scatter(
                            x=obj1_vals,
                            y=obj2_vals,
                            mode="markers",
                            marker=dict(
                                color=normalized_gen,
                                colorscale="RdYlBu_r",
                                size=5,
                                opacity=0.5,
                            ),
                            showlegend=False,
                            hovertemplate="Gen %{customdata}<br>Obj1: %{x:.4f}<br>Obj2: %{y:.4f}<extra></extra>",
                            customdata=[gen_idx] * len(obj1_vals),
                        )
                    )
        else:
            ga_res = source.get(
                "ga_full_results_for_hybrid",
                source if method == "genetic_algorithm" else None,
            )
            ga_instance = ga_res.get("pygad_instance") if ga_res else None
            if ga_instance and hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness:
                all_fitness = ga_instance.all_fitness
                n_generations = len(all_fitness)

                for gen_idx, fitness_array in enumerate(all_fitness):
                    normalized_gen = gen_idx / max(n_generations - 1, 1)
                    obj1_vals = [f[0] if len(f) > 0 else 0 for f in fitness_array]
                    obj2_vals = [f[1] if len(f) > 1 else 0 for f in fitness_array]

                    fig.add_trace(
                        go.Scatter(
                            x=obj1_vals,
                            y=obj2_vals,
                            mode="markers",
                            marker=dict(
                                color=normalized_gen,
                                colorscale="RdYlBu_r",
                                size=5,
                                opacity=0.5,
                            ),
                            showlegend=False,
                        )
                    )

        obj1_pareto = [sol["objectives"][0] for sol in pareto_front]
        obj2_pareto = [sol["objectives"][1] for sol in pareto_front]

        sorted_indices = np.argsort(obj1_pareto)
        obj1_sorted = [obj1_pareto[i] for i in sorted_indices]
        obj2_sorted = [obj2_pareto[i] for i in sorted_indices]

        fig.add_trace(
            go.Scatter(
                x=obj1_sorted,
                y=obj2_sorted,
                mode="lines+markers",
                line=dict(color="green", width=3),
                marker=dict(color="green", size=10, symbol="star"),
                name="Pareto Front",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=obj1_pareto,
                y=obj2_pareto,
                mode="markers",
                marker=dict(
                    color="green", size=12, symbol="star", line=dict(color="black", width=1)
                ),
                text=[
                    f"Obj1: {o1:.4f}<br>Obj2: {o2:.4f}" for o1, o2 in zip(obj1_pareto, obj2_pareto)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Pareto Solutions",
            )
        )

        obj1_name = (
            self.engine.multi_objective_names[0]
            if hasattr(self.engine, "multi_objective_names") and self.engine.multi_objective_names
            else "Objective 1"
        )
        obj2_name = (
            self.engine.multi_objective_names[1]
            if hasattr(self.engine, "multi_objective_names")
            and len(self.engine.multi_objective_names) > 1
            else "Objective 2"
        )

        fig.update_layout(
            title_text=f"Pareto Front - {method.replace('_', ' ').title()}",
            xaxis_title_text=obj1_name.replace("_", " ").title(),
            yaxis_title_text=obj2_name.replace("_", " ").title(),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_well_schedule(self, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Plot well operation schedule as Gantt-style stepped rate timeline.

        Shows all operations for all wells with:
        - Color-coded operational phases
        - Proper time intervals along X-axis (0 to project life)
        - Bar heights proportional to rate
        - Zero-rate operations (soak, idle) visually indicated with hatched pattern and diamond markers
        - Deduplicated phase legend
        - Per-well subplots
        """
        from plotly.subplots import make_subplots

        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        profiles = source.get("optimized_profiles", {})
        schedule = profiles.get("well_schedule", {})
        wells = schedule.get("wells", [])

        op_params = source.get("operational_parameters") or self.engine.operational_params
        eor_params = source.get("eor_parameters") or self.engine.eor_params
        project_life_years = getattr(op_params, "project_lifetime_years", 30)
        project_life_days = project_life_years * 365.25

        if not wells and hasattr(self.engine, "_generate_well_schedule_from_params"):
            try:
                monthly_time = np.linspace(0, project_life_days, int(project_life_years * 12) + 1)
                schedule = self.engine._generate_well_schedule_from_params(eor_params, op_params, monthly_time)
                wells = schedule.get("wells", [])
            except Exception as e:
                logger.warning(f"Failed to generate fallback well schedule: {e}")

        if not wells:
            return go.Figure().update_layout(title_text="No well schedule data available.")

        scheme = schedule.get("injection_scheme", getattr(eor_params, "injection_scheme", "unknown"))

        phase_colors = {
            "injection": "#2E86AB",
            "gas_injection": "#2E86AB",
            "water_injection": "#1E6F5C",
            "soaking": "#F6BD60",
            "production": "#84A98C",
            "idle": "#9B9B9B",
            "pulse": "#E63946",
            "pause": "#9B9B9B",
            "tapered": "#7B68EE",
            "swag": "#20B2AA",
        }

        fig = make_subplots(
            rows=len(wells),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[f"{w['well_name']} ({w.get('well_type', 'Well').title()})" for w in wells],
        )

        added_legend_phases = set()

        for row_idx, well in enumerate(wells, 1):
            ops = well.get("operations", [])
            rates = [o.get("rate_mscfd", 0) for o in ops]
            max_well_rate = max(rates) if rates and max(rates) > 0 else 1000.0
            nominal_idle_height = max(max_well_rate * 0.05, 50.0)

            for op in ops:
                phase = op.get("phase", "operation")
                color = phase_colors.get(phase, "#808080")
                rate = float(op.get("rate_mscfd", 0.0))
                start_day = float(op.get("start_day", 0.0))
                duration = float(op.get("duration_days", 1.0))
                center_day = start_day + duration / 2.0
                end_day = start_day + duration

                phase_display = phase.replace("_", " ").title()
                show_leg = phase not in added_legend_phases

                if rate > 0:
                    fig.add_trace(
                        go.Bar(
                            x=[center_day],
                            y=[rate],
                            width=[duration],
                            base=[0],
                            marker_color=color,
                            marker_line_color="rgba(0, 0, 0, 0.2)",
                            marker_line_width=1,
                            name=phase_display,
                            legendgroup=phase,
                            showlegend=show_leg,
                            hovertemplate=(
                                f"<b>{well['well_name']}</b><br>"
                                f"Phase: {phase_display}<br>"
                                f"Start: Day {start_day:,.1f} ({start_day/365.25:.2f} yrs)<br>"
                                f"End: Day {end_day:,.1f} ({end_day/365.25:.2f} yrs)<br>"
                                f"Duration: {duration:,.1f} days<br>"
                                f"Rate: {rate:,.0f} MSCFD<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )
                    added_legend_phases.add(phase)
                else:
                    # Zero-rate operation (soak, idle, pause)
                    fig.add_trace(
                        go.Bar(
                            x=[center_day],
                            y=[nominal_idle_height],
                            width=[duration],
                            base=[0],
                            marker_color=color,
                            marker_line_color="#555555",
                            marker_line_width=1,
                            marker_pattern_shape="/",
                            opacity=0.45,
                            name=f"{phase_display} (Zero Rate)",
                            legendgroup=phase,
                            showlegend=show_leg,
                            hovertemplate=(
                                f"<b>{well['well_name']}</b><br>"
                                f"Phase: {phase_display} (Idle/Soak)<br>"
                                f"Start: Day {start_day:,.1f} ({start_day/365.25:.2f} yrs)<br>"
                                f"End: Day {end_day:,.1f} ({end_day/365.25:.2f} yrs)<br>"
                                f"Duration: {duration:,.1f} days<br>"
                                f"Rate: 0 MSCFD<extra></extra>"
                            ),
                        ),
                        row=row_idx,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[center_day],
                            y=[nominal_idle_height / 2.0],
                            mode="markers",
                            marker_symbol="diamond",
                            marker_size=8,
                            marker_color=color,
                            marker_line_color="black",
                            marker_line_width=1,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_idx,
                        col=1,
                    )
                    added_legend_phases.add(phase)

            # Set Y-axis title per well row
            wtype = well.get("well_type", "").lower()
            unit_str = "Inj Rate (MSCFD)" if "inj" in wtype else "Prod Rate (MSCFD)"
            fig.update_yaxes(title_text=unit_str, row=row_idx, col=1)

        # Set X-axis range across all rows
        fig.update_xaxes(
            title_text="Project Timeline (Days)",
            range=[0, max(project_life_days, 365.25)],
            row=len(wells),
            col=1,
        )

        fig.update_layout(
            title_text=f"Well Schedule - {str(scheme).replace('_', ' ').title()} Scheme",
            height=max(350, 200 * len(wells)),
            barmode="overlay",
            showlegend=True,
            legend=dict(
                title_text="Operation Phase",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        return fig

    def plot_coverage(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Plot areal sweep coverage evolution across generations showing Trend, Min, and Max,
        along with the population distribution.
        """
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        ga_res = (
            source.get("ga_full_results_for_hybrid")
            or (source if source.get("method") in ["genetic_algorithm", "hybrid_ga_bo", "nsga_2", "hybrid_nsga2_bo"] or "pygad_instance" in source or "ga_statistics" in source else None)
            or source
        )

        from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency

        # Check for coverage_history recorded across generations
        cov_history = None
        if "ga_statistics" in ga_res and "coverage_history" in ga_res["ga_statistics"]:
            cov_history = ga_res["ga_statistics"]["coverage_history"]
        elif ga_instance := ga_res.get("pygad_instance"):
            if hasattr(ga_instance, "coverage_history") and ga_instance.coverage_history:
                cov_history = ga_instance.coverage_history

        # Also get population sweep efficiencies for distribution
        sweep_efficiencies = []
        if ga_instance := ga_res.get("pygad_instance"):
            population = getattr(ga_instance, "population", None)
            if population is not None and len(population) > 0:
                param_names = list(self.engine._get_parameter_bounds().keys())
                mu_oil = getattr(getattr(self.engine, "pvt", None), "oil_viscosity_cp", None) or 1.5
                mu_co2 = getattr(getattr(self.engine, "pvt", None), "gas_viscosity_cp", None) or 0.05
                base_mr = mu_oil / max(mu_co2, 1e-6)

                for individual in population:
                    params_dict = {name: val for name, val in zip(param_names, individual)}
                    m_ratio = params_dict.get("mobility_ratio", base_mr)
                    sweep_efficiencies.append(float(calculate_areal_sweep_efficiency(m_ratio)))

        if not cov_history and not sweep_efficiencies:
            return go.Figure().update_layout(title_text="No coverage / sweep data available to plot.")

        # If we have multi-generation coverage history, build a multi-panel figure
        if cov_history and len(cov_history) > 1:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.15,
                subplot_titles=[
                    "Areal Sweep Coverage Evolution Across Generations",
                    "Final Population Coverage Distribution",
                ],
            )

            gens = [d.get("generation", i + 1) for i, d in enumerate(cov_history)]
            mins = [d.get("min", 0.0) for d in cov_history]
            maxs = [d.get("max", 1.0) for d in cov_history]
            means = [d.get("mean", 0.5) for d in cov_history]

            # Shaded min-max envelope
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=maxs,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=mins,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(44, 160, 44, 0.15)",
                    name="Min-Max Coverage Span",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

            # Max Coverage
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=maxs,
                    mode="lines+markers",
                    name="Max Coverage",
                    line=dict(color="#2ca02c", width=2.5),
                    marker=dict(size=6),
                    hovertemplate="Gen %{x}<br>Max Coverage: %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Trend (Mean Coverage)
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=means,
                    mode="lines",
                    name="Trend (Mean Coverage)",
                    line=dict(color="#ff7f0e", width=3),
                    hovertemplate="Gen %{x}<br>Trend (Mean): %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Min Coverage
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=mins,
                    mode="lines+markers",
                    name="Min Coverage",
                    line=dict(color="#1f77b4", width=2, dash="dot"),
                    marker=dict(size=5),
                    hovertemplate="Gen %{x}<br>Min Coverage: %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.update_xaxes(title_text="Generation", row=1, col=1)
            fig.update_yaxes(title_text="Areal Sweep (0-1)", range=[0, 1.05], row=1, col=1)

            # Row 2: Distribution Histogram
            dist_vals = sweep_efficiencies if sweep_efficiencies else means
            fig.add_trace(
                go.Histogram(
                    x=dist_vals,
                    nbinsx=20,
                    marker_color="#84a98c",
                    name="Population Distribution",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            min_val = float(np.min(dist_vals))
            max_val = float(np.max(dist_vals))
            mean_val = float(np.mean(dist_vals))

            fig.add_vline(x=min_val, line_dash="dash", line_color="#1f77b4", row=2, col=1)
            fig.add_vline(x=mean_val, line_dash="solid", line_color="#ff7f0e", line_width=2.5, row=2, col=1)
            fig.add_vline(x=max_val, line_dash="dash", line_color="#2ca02c", row=2, col=1)

            fig.update_xaxes(title_text="Areal Sweep Efficiency (Coverage)", range=[0, 1.0], row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            fig.update_layout(
                title_text="Optimization Coverage Analysis: Trend, Min, Max",
                height=650,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            return fig
        else:
            # Single generation distribution with explicit Min, Max, Trend lines
            dist_vals = sweep_efficiencies or [0.5]
            min_val = float(np.min(dist_vals))
            max_val = float(np.max(dist_vals))
            mean_val = float(np.mean(dist_vals))
            std_val = float(np.std(dist_vals))

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=dist_vals,
                    nbinsx=20,
                    marker_color="#84a98c",
                    name="Coverage Distribution",
                )
            )

            fig.add_vline(
                x=min_val,
                line_dash="dash",
                line_color="#1f77b4",
                line_width=2,
                annotation_text=f"Min: {min_val:.3f}",
                annotation_position="top left",
            )
            fig.add_vline(
                x=mean_val,
                line_dash="solid",
                line_color="#ff7f0e",
                line_width=3,
                annotation_text=f"Trend (Mean): {mean_val:.3f}",
                annotation_position="top",
            )
            fig.add_vline(
                x=max_val,
                line_dash="dash",
                line_color="#2ca02c",
                line_width=2,
                annotation_text=f"Max: {max_val:.3f}",
                annotation_position="top right",
            )

            # Shaded span between Min and Max
            fig.add_vrect(
                x0=min_val,
                x1=max_val,
                fillcolor="rgba(44, 160, 44, 0.1)",
                layer="below",
                line_width=0,
            )

            fig.update_layout(
                title_text="GA Population Areal Sweep Efficiency Coverage (Trend, Min, Max)",
                xaxis_title="Areal Sweep Efficiency (Coverage)",
                yaxis_title="Frequency",
                annotations=[
                    dict(
                        x=0.98,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=(
                            f"<b>Coverage Summary</b><br>"
                            f"Trend (Mean): {mean_val:.3f}<br>"
                            f"Min Coverage: {min_val:.3f}<br>"
                            f"Max Coverage: {max_val:.3f}<br>"
                            f"Std Dev: {std_val:.3f}"
                        ),
                        showarrow=False,
                        align="left",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#333333",
                        borderwidth=1,
                    )
                ],
            )
            return fig

    def plot_euclidean_distance_matrix(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """
        Plot the Euclidean distance matrix in normalized parameter space [0, 1]^d.
        Visualizes candidate diversity for Phase 2 transfer from GA to Bayesian Optimization.
        """
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        bounds = self.engine._get_parameter_bounds()
        param_names = list(bounds.keys())

        # 1. Identify candidate solutions to compare
        candidate_dicts = []
        labels = []
        source_name = "Candidate"

        # Check for Phase 2 diverse solutions
        diverse_pts = source.get("diverse_points_for_bo")
        if not diverse_pts and "ga_full_results_for_hybrid" in source:
            diverse_pts = source["ga_full_results_for_hybrid"].get("diverse_points_for_bo")

        if diverse_pts and len(diverse_pts) >= 2:
            source_name = "Phase 2 BO Seed"
            for i, pt in enumerate(diverse_pts):
                candidate_dicts.append(pt["params"])
                labels.append(f"Seed {i+1}")
        elif pareto := source.get("pareto_front"):
            source_name = "Pareto Candidate"
            for i, sol in enumerate(pareto[:15]):  # limit to top 15 for readable matrix
                candidate_dicts.append(sol["params"])
                labels.append(f"Pareto {i+1}")
        elif ga_instance := source.get("pygad_instance") or source.get("ga_full_results_for_hybrid", {}).get("pygad_instance"):
            pop = getattr(ga_instance, "population", None)
            if pop is not None and len(pop) >= 2:
                source_name = "Population"
                # Select top 12 unique individuals
                last_fit = getattr(ga_instance, "last_generation_fitness", None)
                if last_fit is not None and len(last_fit) == len(pop):
                    sorted_idxs = np.argsort(last_fit)[::-1]
                    seen = set()
                    selected_idxs = []
                    for idx in sorted_idxs:
                        tpl = tuple(np.round(pop[idx], 3))
                        if tpl not in seen:
                            seen.add(tpl)
                            selected_idxs.append(idx)
                        if len(selected_idxs) >= 12:
                            break
                    for rank, idx in enumerate(selected_idxs, 1):
                        p_dict = {k: v for k, v in zip(param_names, pop[idx])}
                        candidate_dicts.append(p_dict)
                        labels.append(f"Rank {rank}")
                else:
                    for i in range(min(12, len(pop))):
                        p_dict = {k: v for k, v in zip(param_names, pop[i])}
                        candidate_dicts.append(p_dict)
                        labels.append(f"Sol {i+1}")
        elif bo_obj := source.get("bayes_opt_obj"):
            if hasattr(bo_obj, "res") and len(bo_obj.res) >= 2:
                source_name = "BO Evaluation"
                sorted_res = sorted([r for r in bo_obj.res if r.get("target", -1e12) > -1e9], key=lambda x: x["target"], reverse=True)
                for i, r in enumerate(sorted_res[:12], 1):
                    candidate_dicts.append(r["params"])
                    labels.append(f"BO Best {i}")

        if len(candidate_dicts) < 2:
            return go.Figure().update_layout(
                title_text="Insufficient candidate points to compute Euclidean distance matrix (need >= 2)."
            )

        # 2. Normalize parameters to [0, 1] range
        normalized_matrix = []
        for p in candidate_dicts:
            norm_vec = []
            for k in param_names:
                b_entry = bounds.get(k, (0.0, 1.0))
                if isinstance(b_entry, dict):
                    low, high = b_entry.get("low", 0.0), b_entry.get("high", 1.0)
                else:
                    low, high = b_entry[0], b_entry[1]
                val = p.get(k, (low + high) / 2.0)
                if isinstance(val, (str, bytes)):
                    norm_val = 0.5
                else:
                    norm_val = (float(val) - low) / (high - low) if high > low else 0.5
                norm_vec.append(np.clip(norm_val, 0.0, 1.0))
            normalized_matrix.append(norm_vec)

        norm_arr = np.array(normalized_matrix)  # shape (N, d)
        n_candidates, d_dims = norm_arr.shape

        # 3. Calculate pairwise Euclidean distance matrix
        diff = norm_arr[:, np.newaxis, :] - norm_arr[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))

        # Compute summary metrics (excluding diagonal)
        mask = ~np.eye(n_candidates, dtype=bool)
        off_diag = distances[mask] if np.any(mask) else np.array([0.0])
        mean_dist = float(np.mean(off_diag))
        min_dist = float(np.min(off_diag))
        max_dist = float(np.max(off_diag))

        # Format cell text for heatmap
        text_matrix = [[f"{distances[i, j]:.2f}" for j in range(n_candidates)] for i in range(n_candidates)]

        fig = go.Figure(
            data=go.Heatmap(
                z=distances,
                x=labels,
                y=labels,
                colorscale="Viridis",
                text=text_matrix,
                texttemplate="%{text}",
                textfont={"size": 11},
                colorbar=dict(title="Distance"),
                hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Euclidean Distance: %{z:.3f}<extra></extra>",
            )
        )

        title = f"Phase 2: Euclidean Distance Matrix ({source_name} Diversity)"
        subtitle = (
            f"<i>N = {n_candidates} Candidates, {d_dims} Dimensions | "
            f"Mean Distance: {mean_dist:.3f} | Min Distance: {min_dist:.3f} | Max Distance: {max_dist:.3f}</i>"
        )

        fig.update_layout(
            title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.5, xanchor="center"),
            xaxis=dict(title="Candidate Index", tickangle=-45),
            yaxis=dict(title="Candidate Index", autorange="reversed"),
            width=700,
            height=650,
        )
        return fig

    def plot_ga_objective_distribution(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        source = results_to_use or getattr(self.engine, "_results", None)
        if not (source and "pygad_instance" in source):
            return go.Figure().update_layout(title_text="No GA results to plot.")

        ga_instance = source["pygad_instance"]
        objectives = ga_instance.last_generation_fitness

        avg_obj = np.mean(objectives)
        std_obj = np.std(objectives)

        fig = go.Figure(data=[go.Histogram(x=objectives, nbinsx=20)])
        fig.update_layout(
            title_text="GA Population Objective Value Distribution",
            xaxis_title="Objective Value",
            yaxis_title="Frequency",
            annotations=[
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Avg: {avg_obj:.3f}<br>Std: {std_obj:.3f}",
                    showarrow=False,
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                )
            ],
        )
        return fig

    def plot_hybrid_model_analysis(self) -> go.Figure:
        """Generates a plot showing the interplay of miscible, immiscible, and hybrid recovery models."""
        try:
            from deprecated.core.simulation.recovery_models import (
                MiscibleRecoveryModel,
                ImmiscibleRecoveryModel,
                SigmoidTransition,
            )
        except ImportError:
            return go.Figure().update_layout(title_text="Recovery models not available")

        import dataclasses
        mmp = getattr(self.engine, "mmp", None) or getattr(self.engine.eor_params, "default_mmp_fallback", 2500.0)
        pressure_ratios = np.linspace(0.5, 2.0, 50)
        pressures = pressure_ratios * mmp

        miscible_rf = []
        immiscible_rf = []
        weights = []

        miscible_model = MiscibleRecoveryModel()
        immiscible_model = ImmiscibleRecoveryModel()
        transition = SigmoidTransition()

        base_params = dataclasses.asdict(self.engine.eor_params) if hasattr(self.engine, "eor_params") else {}

        for p in pressures:
            params = base_params.copy()
            params["pressure"] = p
            params["mmp"] = mmp
            miscible_rf.append(miscible_model.calculate(**params))
            immiscible_rf.append(immiscible_model.calculate(**params))
            weights.append(transition.evaluate(p / mmp, getattr(self.engine.pvt, "c7_plus_fraction", 0.3)))

        hybrid_rf = np.array(weights) * np.array(miscible_rf) + (1 - np.array(weights)) * np.array(
            immiscible_rf
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=pressure_ratios, y=immiscible_rf, mode="lines", name="Immiscible RF")
        )
        fig.add_trace(
            go.Scatter(x=pressure_ratios, y=miscible_rf, mode="lines", name="Miscible RF")
        )
        fig.add_trace(
            go.Scatter(
                x=pressure_ratios,
                y=hybrid_rf,
                mode="lines",
                name="Hybrid RF",
                line=dict(color="black", width=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pressure_ratios,
                y=weights,
                mode="lines",
                name="Miscible Weight",
                line=dict(dash="dot"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title_text="Hybrid Recovery Model Analysis",
            xaxis_title="Pressure / MMP Ratio",
            yaxis_title="Recovery Factor",
            yaxis=dict(range=[0, 1]),
            yaxis2=dict(
                title="Miscible Weight", overlaying="y", side="right", range=[0, 1], showgrid=False
            ),
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    def plot_breakthrough_mechanism_analysis(self) -> go.Figure:
        """Generates a bar chart comparing breakthrough times from different models using Surrogate Physics."""
        try:
            from analysis.breakthrough_physics import CO2BreakthroughPhysics
            bt_physics = CO2BreakthroughPhysics()

            import dataclasses
            reservoir = getattr(self.engine, "reservoir", None)
            eor_params = getattr(self.engine, "eor_params", None)

            reservoir_params = {
                "v_dp_coefficient": getattr(eor_params, "v_dp_coefficient", 0.5) if eor_params else 0.5,
                "area_acres": reservoir.area_acres if reservoir else 100.0,
                "porosity": getattr(self.engine, "avg_porosity", 0.2),
                "thickness_ft": reservoir.thickness_ft if reservoir else 50.0,
                "permeability": np.mean(reservoir.grid.get("PERMX", [100.0]))
                if reservoir and getattr(reservoir, "grid", None)
                else 100.0,
            }
            eor_params_for_bt = dataclasses.asdict(eor_params) if eor_params else {}

            eos_model = getattr(self.engine, "eos_model_instance", None) or (
                getattr(reservoir, "eos_model", None) if reservoir else None
            )
            bt_koval = bt_physics.calculate_breakthrough_time(
                reservoir_params, eor_params_for_bt, eos_model=eos_model
            )

            rho_oil = getattr(eor_params, "oil_density", 50.0) if eor_params else 50.0
            rho_co2 = getattr(eor_params, "co2_density", 44.0) if eor_params else 44.0
            gravity_mult = 1.0 / (1.0 + 0.1 * abs(rho_oil - rho_co2))
            bt_gravity = bt_koval * gravity_mult
            bt_final = bt_koval

            mechanisms = ["Analytical (Koval)", "Gravity Scaling", "Final Surrogate"]
            times = [bt_koval, bt_gravity, bt_final]

            fig = go.Figure(
                [go.Bar(x=mechanisms, y=times, text=[f"{t:.2f} y" for t in times], textposition="auto")]
            )
            fig.update_layout(
                title_text="PhD Verification: Breakthrough Analysis by Mechanism (Surrogate)",
                yaxis_title="Breakthrough Time (years)",
                template="plotly_white",
            )
            return fig
        except Exception as e:
            logger.warning(f"Failed to generate breakthrough mechanism analysis plot: {e}")
            return go.Figure()
