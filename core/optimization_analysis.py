"""
Comprehensive Optimization Analysis Module
Provides advanced analysis tools for optimization results including sensitivity analysis,
parameter evolution, convergence analysis, and multi-algorithm comparison
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express import scatter_3d, parallel_coordinates
import plotly.express as px

logger = logging.getLogger(__name__)

@dataclass
class SensitivityResult:
    """Results of sensitivity analysis"""
    parameter_name: str
    sensitivity_coefficient: float
    confidence_interval: Tuple[float, float]
    correlation_coefficient: float
    p_value: float
    elasticity: float
    standardized_coefficient: float

@dataclass
class ConvergenceMetrics:
    """Convergence analysis metrics"""
    algorithm_name: str
    convergence_rate: float
    final_objective_value: float
    convergence_iteration: int
    stability_metric: float
    efficiency_score: float
    computational_cost: int

@dataclass
class ParameterEvolution:
    """Parameter evolution during optimization"""
    parameter_name: str
    initial_value: float
    final_value: float
    evolution_path: List[float]
    convergence_time: int
    variance_trend: List[float]
    improvement_percentage: float

@dataclass
class AlgorithmComparison:
    """Comparison between different optimization algorithms"""
    algorithm_names: List[str]
    performance_scores: List[float]
    convergence_speeds: List[float]
    solution_quality: List[float]
    robustness_scores: List[float]
    computational_efficiency: List[float]

class OptimizationAnalyzer:
    """Advanced analysis tools for optimization results"""

    def __init__(self):
        self.analysis_cache = {}
        self.scaler = StandardScaler()

    def analyze_sensitivity(self,
                          optimization_results: Dict[str, Any],
                          parameter_names: List[str],
                          method: str = "sobol") -> Dict[str, SensitivityResult]:
        """
        Perform comprehensive sensitivity analysis

        Args:
            optimization_results: Results from optimization run
            parameter_names: List of parameters to analyze
            method: Sensitivity analysis method ("sobol", "morris", "regression", "correlation")

        Returns:
            Dictionary of sensitivity results for each parameter
        """
        try:
            logger.info(f"Starting sensitivity analysis using {method} method")

            # Extract data for analysis
            objective_history = optimization_results.get("objective_history", [])
            parameter_history = optimization_results.get("parameter_history", {})

            sensitivity_results = {}

            for param_name in parameter_names:
                if param_name not in parameter_history:
                    logger.warning(f"Parameter {param_name} not found in optimization results")
                    continue

                param_values = np.array(parameter_history[param_name])
                obj_values = np.array(objective_history[-len(param_values):])

                # Perform sensitivity analysis based on method
                if method == "correlation":
                    result = self._correlation_sensitivity(param_values, obj_values, param_name)
                elif method == "regression":
                    result = self._regression_sensitivity(param_values, obj_values, param_name)
                elif method == "morris":
                    result = self._morris_sensitivity(param_values, obj_values, param_name)
                else:  # Default to correlation-based analysis
                    result = self._correlation_sensitivity(param_values, obj_values, param_name)

                sensitivity_results[param_name] = result

            logger.info(f"Sensitivity analysis completed for {len(sensitivity_results)} parameters")
            return sensitivity_results

        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {}

    def _correlation_sensitivity(self, param_values: np.ndarray, obj_values: np.ndarray,
                               param_name: str) -> SensitivityResult:
        """Correlation-based sensitivity analysis"""
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(param_values, obj_values)

        # Calculate sensitivity coefficient (slope of standardized values)
        param_std = (param_values - np.mean(param_values)) / np.std(param_values)
        obj_std = (obj_values - np.mean(obj_values)) / np.std(obj_values)
        sensitivity_coefficient = np.cov(param_std, obj_std)[0, 1]

        # Calculate elasticity (percent change in objective per percent change in parameter)
        elasticity = self._calculate_elasticity(param_values, obj_values)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(param_values, obj_values)

        return SensitivityResult(
            parameter_name=param_name,
            sensitivity_coefficient=sensitivity_coefficient,
            confidence_interval=confidence_interval,
            correlation_coefficient=correlation,
            p_value=p_value,
            elasticity=elasticity,
            standardized_coefficient=sensitivity_coefficient
        )

    def _regression_sensitivity(self, param_values: np.ndarray, obj_values: np.ndarray,
                              param_name: str) -> SensitivityResult:
        """Regression-based sensitivity analysis"""
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(param_values, obj_values)

        # Calculate confidence interval for slope
        n = len(param_values)
        t_critical = stats.t.ppf(0.975, n - 2)
        margin_error = t_critical * std_err
        confidence_interval = (slope - margin_error, slope + margin_error)

        # Calculate elasticity
        elasticity = self._calculate_elasticity(param_values, obj_values)

        return SensitivityResult(
            parameter_name=param_name,
            sensitivity_coefficient=slope,
            confidence_interval=confidence_interval,
            correlation_coefficient=r_value,
            p_value=p_value,
            elasticity=elasticity,
            standardized_coefficient=slope * np.std(param_values) / np.std(obj_values)
        )

    def _morris_sensitivity(self, param_values: np.ndarray, obj_values: np.ndarray,
                           param_name: str) -> SensitivityResult:
        """Morris method sensitivity analysis (simplified)"""
        # Calculate elementary effects
        elementary_effects = []
        for i in range(1, len(param_values)):
            effect = (obj_values[i] - obj_values[i-1]) / (param_values[i] - param_values[i-1])
            elementary_effects.append(effect)

        # Calculate sensitivity metrics
        mu_star = np.mean(np.abs(elementary_effects))
        sigma = np.std(elementary_effects)
        mu = np.mean(elementary_effects)

        # Approximate correlation
        correlation, p_value = stats.pearsonr(param_values, obj_values)

        return SensitivityResult(
            parameter_name=param_name,
            sensitivity_coefficient=mu_star,
            confidence_interval=(mu - 1.96*sigma/np.sqrt(len(elementary_effects)),
                              mu + 1.96*sigma/np.sqrt(len(elementary_effects))),
            correlation_coefficient=correlation,
            p_value=p_value,
            elasticity=self._calculate_elasticity(param_values, obj_values),
            standardized_coefficient=mu_star
        )

    def _calculate_elasticity(self, param_values: np.ndarray, obj_values: np.ndarray) -> float:
        """Calculate elasticity coefficient"""
        # Use log-log regression for elasticity
        valid_indices = (param_values > 0) & (obj_values > 0)
        if np.sum(valid_indices) < 2:
            return 0.0

        log_param = np.log(param_values[valid_indices])
        log_obj = np.log(obj_values[valid_indices])

        slope, _, _, _, _ = stats.linregress(log_param, log_obj)
        return slope

    def _calculate_confidence_interval(self, param_values: np.ndarray, obj_values: np.ndarray,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        n = len(param_values)
        if n < 3:
            return (0.0, 0.0)

        correlation, _ = stats.pearsonr(param_values, obj_values)
        fisher_z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)

        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_critical * se

        ci_lower = np.tanh(fisher_z - margin)
        ci_upper = np.tanh(fisher_z + margin)

        return (ci_lower, ci_upper)

    def analyze_convergence(self, optimization_results: Dict[str, Any],
                          algorithm_name: str = "Unknown") -> ConvergenceMetrics:
        """
        Analyze convergence characteristics of optimization run

        Args:
            optimization_results: Results from optimization run
            algorithm_name: Name of the optimization algorithm

        Returns:
            ConvergenceMetrics object with analysis results
        """
        try:
            objective_history = optimization_results.get("convergence_history", [])
            if not objective_history:
                # Fallback to other possible keys
                objective_history = optimization_results.get("objective_history", [])
                objective_history = optimization_results.get("fitness_history", objective_history)

            if len(objective_history) < 5:
                logger.warning("Insufficient data for convergence analysis")
                return ConvergenceMetrics(
                    algorithm_name=algorithm_name,
                    convergence_rate=0.0,
                    final_objective_value=objective_history[-1] if objective_history else 0.0,
                    convergence_iteration=len(objective_history),
                    stability_metric=0.0,
                    efficiency_score=0.0,
                    computational_cost=optimization_results.get("iterations", 0)
                )

            objective_values = np.array(objective_history)

            # Calculate convergence rate (improvement per iteration)
            improvements = np.diff(objective_values)
            convergence_rate = np.mean(np.abs(improvements))

            # Find convergence iteration (point where improvements become minimal)
            threshold = 0.01 * np.std(objective_values)
            convergence_indices = np.where(np.abs(improvements) < threshold)[0]
            convergence_iteration = convergence_indices[0] + 1 if len(convergence_indices) > 0 else len(objective_values)

            # Calculate stability metric (variance in final iterations)
            final_window = min(20, len(objective_values) // 4)
            final_values = objective_values[-final_window:]
            stability_metric = 1.0 / (1.0 + np.var(final_values))

            # Calculate efficiency score
            total_improvement = abs(objective_values[-1] - objective_values[0])
            max_possible_improvement = abs(objective_values[-1] - np.min(objective_values))
            efficiency_score = total_improvement / max(max_possible_improvement, 1e-6)

            return ConvergenceMetrics(
                algorithm_name=algorithm_name,
                convergence_rate=convergence_rate,
                final_objective_value=objective_values[-1],
                convergence_iteration=convergence_iteration,
                stability_metric=stability_metric,
                efficiency_score=efficiency_score,
                computational_cost=optimization_results.get("iterations", len(objective_values))
            )

        except Exception as e:
            logger.error(f"Error in convergence analysis: {e}")
            return ConvergenceMetrics(
                algorithm_name=algorithm_name,
                convergence_rate=0.0,
                final_objective_value=0.0,
                convergence_iteration=0,
                stability_metric=0.0,
                efficiency_score=0.0,
                computational_cost=0
            )

    def analyze_parameter_evolution(self, optimization_results: Dict[str, Any],
                                  parameter_name: str) -> ParameterEvolution:
        """
        Analyze evolution of a specific parameter during optimization

        Args:
            optimization_results: Results from optimization run
            parameter_name: Name of parameter to analyze

        Returns:
            ParameterEvolution object with analysis results
        """
        try:
            parameter_history = optimization_results.get("parameter_history", {})
            if parameter_name not in parameter_history:
                raise ValueError(f"Parameter {parameter_name} not found in results")

            param_values = np.array(parameter_history[parameter_name])
            objective_history = optimization_results.get("convergence_history", [])
            if len(objective_history) > len(param_values):
                objective_history = objective_history[-len(param_values):]

            # Calculate basic metrics
            initial_value = param_values[0]
            final_value = param_values[-1]
            improvement_percentage = ((final_value - initial_value) / abs(initial_value)) * 100

            # Calculate convergence time (iteration where parameter stabilizes)
            convergence_threshold = 0.01 * np.std(param_values)
            differences = np.diff(param_values)
            stable_indices = np.where(np.abs(differences) < convergence_threshold)[0]
            convergence_time = stable_indices[0] + 1 if len(stable_indices) > 0 else len(param_values)

            # Calculate variance trend (sliding window variance)
            window_size = max(5, len(param_values) // 10)
            variance_trend = []
            for i in range(len(param_values) - window_size + 1):
                window = param_values[i:i + window_size]
                variance_trend.append(np.var(window))

            return ParameterEvolution(
                parameter_name=parameter_name,
                initial_value=initial_value,
                final_value=final_value,
                evolution_path=param_values.tolist(),
                convergence_time=convergence_time,
                variance_trend=variance_trend,
                improvement_percentage=improvement_percentage
            )

        except Exception as e:
            logger.error(f"Error in parameter evolution analysis: {e}")
            return ParameterEvolution(
                parameter_name=parameter_name,
                initial_value=0.0,
                final_value=0.0,
                evolution_path=[],
                convergence_time=0,
                variance_trend=[],
                improvement_percentage=0.0
            )

    def compare_algorithms(self, results_list: List[Dict[str, Any]],
                          algorithm_names: List[str]) -> AlgorithmComparison:
        """
        Compare performance of different optimization algorithms

        Args:
            results_list: List of optimization results for each algorithm
            algorithm_names: Names of algorithms

        Returns:
            AlgorithmComparison object with comparison results
        """
        try:
            if len(results_list) != len(algorithm_names):
                raise ValueError("Number of results must match number of algorithm names")

            performance_scores = []
            convergence_speeds = []
            solution_quality = []
            robustness_scores = []
            computational_efficiency = []

            for i, (results, name) in enumerate(zip(results_list, algorithm_names)):
                # Analyze each algorithm
                convergence_metrics = self.analyze_convergence(results, name)

                # Performance score (normalized objective value)
                final_obj = convergence_metrics.final_objective_value
                performance_scores.append(final_obj)

                # Convergence speed (iterations to convergence)
                convergence_speeds.append(convergence_metrics.convergence_iteration)

                # Solution quality (stability and efficiency)
                solution_quality.append(convergence_metrics.stability_metric * convergence_metrics.efficiency_score)

                # Robustness score (inverse of variance in final performance)
                objective_history = results.get("convergence_history", [])
                if len(objective_history) >= 10:
                    final_variance = np.var(objective_history[-10:])
                    robustness_scores.append(1.0 / (1.0 + final_variance))
                else:
                    robustness_scores.append(0.5)

                # Computational efficiency (performance per computational cost)
                cost = convergence_metrics.computational_cost
                efficiency = final_obj / max(cost, 1)
                computational_efficiency.append(efficiency)

            return AlgorithmComparison(
                algorithm_names=algorithm_names,
                performance_scores=performance_scores,
                convergence_speeds=convergence_speeds,
                solution_quality=solution_quality,
                robustness_scores=robustness_scores,
                computational_efficiency=computational_efficiency
            )

        except Exception as e:
            logger.error(f"Error in algorithm comparison: {e}")
            return AlgorithmComparison(
                algorithm_names=algorithm_names,
                performance_scores=[],
                convergence_speeds=[],
                solution_quality=[],
                robustness_scores=[],
                computational_efficiency=[]
            )

    def create_sensitivity_plot(self, sensitivity_results: Dict[str, SensitivityResult]) -> go.Figure:
        """Create comprehensive sensitivity analysis plot"""
        try:
            if not sensitivity_results:
                # Create empty plot
                fig = go.Figure()
                fig.add_annotation(
                    text="No sensitivity analysis data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16)
                )
                return fig

            # Prepare data
            param_names = list(sensitivity_results.keys())
            sensitivity_coeffs = [r.sensitivity_coefficient for r in sensitivity_results.values()]
            correlations = [abs(r.correlation_coefficient) for r in sensitivity_results.values()]
            elasticities = [abs(r.elasticity) for r in sensitivity_results.values()]

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Sensitivity Coefficients", "Correlation Coefficients",
                              "Elasticity Values", "Comprehensive Comparison"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "scatter3d"}]]
            )

            # Sensitivity coefficients bar chart
            fig.add_trace(
                go.Bar(x=param_names, y=sensitivity_coeffs, name="Sensitivity Coeff",
                       marker_color='rgb(55, 83, 109)'),
                row=1, col=1
            )

            # Correlation coefficients bar chart
            fig.add_trace(
                go.Bar(x=param_names, y=correlations, name="Correlation Coeff",
                       marker_color='rgb(26, 118, 255)'),
                row=1, col=2
            )

            # Elasticity values bar chart
            fig.add_trace(
                go.Bar(x=param_names, y=elasticities, name="Elasticity",
                       marker_color='rgb(219, 64, 82)'),
                row=2, col=1
            )

            # 3D scatter plot comparing all metrics
            fig.add_trace(
                go.Scatter3d(
                    x=sensitivity_coeffs,
                    y=correlations,
                    z=elasticities,
                    mode='markers+text',
                    text=param_names,
                    textposition="top center",
                    marker=dict(
                        size=8,
                        color=sensitivity_coeffs,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sensitivity", x=1.02)
                    ),
                    name="Parameters"
                ),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title_text="Comprehensive Sensitivity Analysis",
                showlegend=False,
                height=800
            )

            # Update axes labels
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=1, col=1)
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=1, col=2)
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=2, col=1)

            fig.update_yaxes(title_text="Sensitivity Coefficient", row=1, col=1)
            fig.update_yaxes(title_text="Correlation Coefficient", row=1, col=2)
            fig.update_yaxes(title_text="Elasticity", row=2, col=1)

            fig.update_layout(
                scene=dict(
                    xaxis_title="Sensitivity Coefficient",
                    yaxis_title="Correlation Coefficient",
                    zaxis_title="Elasticity"
                )
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating sensitivity plot: {e}")
            return go.Figure()

    def create_convergence_analysis_plot(self, convergence_results: List[ConvergenceMetrics]) -> go.Figure:
        """Create convergence analysis comparison plot"""
        try:
            if not convergence_results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No convergence data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16)
                )
                return fig

            algorithm_names = [r.algorithm_name for r in convergence_results]
            convergence_rates = [r.convergence_rate for r in convergence_results]
            efficiency_scores = [r.efficiency_score for r in convergence_results]
            stability_metrics = [r.stability_metric for r in convergence_results]
            computational_costs = [r.computational_cost for r in convergence_results]

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Convergence Rate", "Efficiency Score",
                              "Stability Metric", "Computational Cost")
            )

            # Convergence rate
            fig.add_trace(
                go.Bar(x=algorithm_names, y=convergence_rates,
                       marker_color='rgb(55, 83, 109)', name="Convergence Rate"),
                row=1, col=1
            )

            # Efficiency score
            fig.add_trace(
                go.Bar(x=algorithm_names, y=efficiency_scores,
                       marker_color='rgb(26, 118, 255)', name="Efficiency"),
                row=1, col=2
            )

            # Stability metric
            fig.add_trace(
                go.Bar(x=algorithm_names, y=stability_metrics,
                       marker_color='rgb(219, 64, 82)', name="Stability"),
                row=2, col=1
            )

            # Computational cost (inverted for better visualization)
            normalized_costs = [1.0 / (1.0 + cost/1000) for cost in computational_costs]
            fig.add_trace(
                go.Bar(x=algorithm_names, y=normalized_costs,
                       marker_color='rgb(46, 125, 50)', name="Efficiency (normalized)"),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title_text="Algorithm Convergence Analysis Comparison",
                showlegend=False,
                height=600
            )

            # Update axes
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(tickangle=45, row=i, col=j)

            return fig

        except Exception as e:
            logger.error(f"Error creating convergence analysis plot: {e}")
            return go.Figure()

    def create_parameter_evolution_plot(self, evolution_results: List[ParameterEvolution]) -> go.Figure:
        """Create parameter evolution analysis plot"""
        try:
            if not evolution_results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No parameter evolution data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16)
                )
                return fig

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Parameter Evolution Paths", "Improvement Percentages",
                              "Convergence Times", "Variance Trends")
            )

            # Parameter evolution paths
            for i, evolution in enumerate(evolution_results):
                if evolution.evolution_path:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(evolution.evolution_path))),
                            y=evolution.evolution_path,
                            mode='lines+markers',
                            name=evolution.parameter_name,
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )

            # Improvement percentages
            param_names = [e.parameter_name for e in evolution_results]
            improvements = [e.improvement_percentage for e in evolution_results]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]

            fig.add_trace(
                go.Bar(x=param_names, y=improvements, marker_color=colors, name="Improvement %"),
                row=1, col=2
            )

            # Convergence times
            convergence_times = [e.convergence_time for e in evolution_results]
            fig.add_trace(
                go.Bar(x=param_names, y=convergence_times,
                       marker_color='rgb(55, 83, 109)', name="Convergence Time"),
                row=2, col=1
            )

            # Variance trends (average variance for each parameter)
            avg_variances = [np.mean(e.variance_trend) if e.variance_trend else 0 for e in evolution_results]
            fig.add_trace(
                go.Bar(x=param_names, y=avg_variances,
                       marker_color='rgb(26, 118, 255)', name="Avg Variance"),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title_text="Parameter Evolution Analysis",
                height=800,
                showlegend=True
            )

            # Update axes labels
            fig.update_xaxes(title_text="Iteration", row=1, col=1)
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=1, col=2)
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=2, col=1)
            fig.update_xaxes(title_text="Parameters", tickangle=45, row=2, col=2)

            fig.update_yaxes(title_text="Parameter Value", row=1, col=1)
            fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
            fig.update_yaxes(title_text="Convergence Time", row=2, col=1)
            fig.update_yaxes(title_text="Average Variance", row=2, col=2)

            return fig

        except Exception as e:
            logger.error(f"Error creating parameter evolution plot: {e}")
            return go.Figure()

    def create_algorithm_comparison_plot(self, comparison_results: AlgorithmComparison) -> go.Figure:
        """Create comprehensive algorithm comparison plot"""
        try:
            if not comparison_results.algorithm_names:
                fig = go.Figure()
                fig.add_annotation(
                    text="No algorithm comparison data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16)
                )
                return fig

            # Normalize scores for better comparison
            def normalize_scores(scores):
                min_val, max_val = min(scores), max(scores)
                if max_val - min_val < 1e-6:
                    return [1.0] * len(scores)
                return [(s - min_val) / (max_val - min_val) for s in scores]

            algorithms = comparison_results.algorithm_names

            # Normalize different metrics
            perf_normalized = normalize_scores(comparison_results.performance_scores)
            conv_normalized = normalize_scores([1.0/s for s in comparison_results.convergence_speeds])  # Lower is better
            quality_normalized = normalize_scores(comparison_results.solution_quality)
            robust_normalized = normalize_scores(comparison_results.robustness_scores)
            eff_normalized = normalize_scores(comparison_results.computational_efficiency)

            # Create radar chart data
            categories = ['Performance', 'Convergence Speed', 'Solution Quality',
                         'Robustness', 'Computational Efficiency']

            fig = go.Figure()

            # Add traces for each algorithm
            for i, algorithm in enumerate(algorithms):
                values = [
                    perf_normalized[i],
                    conv_normalized[i],
                    quality_normalized[i],
                    robust_normalized[i],
                    eff_normalized[i]
                ]
                values += [values[0]]  # Close the radar chart

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=algorithm,
                    line=dict(width=2)
                ))

            # Update layout for radar chart
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title_text="Algorithm Performance Comparison (Radar Chart)",
                showlegend=True,
                height=600
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating algorithm comparison plot: {e}")
            return go.Figure()

    def generate_comprehensive_report(self, optimization_results: Dict[str, Any],
                                    algorithm_name: str = "Unknown") -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        try:
            report = {
                "algorithm_name": algorithm_name,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "convergence_analysis": {},
                "parameter_analysis": {},
                "sensitivity_analysis": {},
                "performance_summary": {}
            }

            # Convergence analysis
            convergence_metrics = self.analyze_convergence(optimization_results, algorithm_name)
            report["convergence_analysis"] = {
                "convergence_rate": convergence_metrics.convergence_rate,
                "final_objective_value": convergence_metrics.final_objective_value,
                "convergence_iteration": convergence_metrics.convergence_iteration,
                "stability_metric": convergence_metrics.stability_metric,
                "efficiency_score": convergence_metrics.efficiency_score,
                "computational_cost": convergence_metrics.computational_cost
            }

            # Parameter analysis
            parameter_history = optimization_results.get("parameter_history", {})
            report["parameter_analysis"] = {}
            for param_name in parameter_history.keys():
                evolution = self.analyze_parameter_evolution(optimization_results, param_name)
                report["parameter_analysis"][param_name] = {
                    "initial_value": evolution.initial_value,
                    "final_value": evolution.final_value,
                    "improvement_percentage": evolution.improvement_percentage,
                    "convergence_time": evolution.convergence_time,
                    "final_variance": np.mean(evolution.variance_trend) if evolution.variance_trend else 0
                }

            # Sensitivity analysis (if applicable)
            if parameter_history:
                sensitivity_results = self.analyze_sensitivity(
                    optimization_results,
                    list(parameter_history.keys()),
                    method="correlation"
                )
                report["sensitivity_analysis"] = {
                    param_name: {
                        "sensitivity_coefficient": result.sensitivity_coefficient,
                        "correlation_coefficient": result.correlation_coefficient,
                        "elasticity": result.elasticity,
                        "p_value": result.p_value
                    }
                    for param_name, result in sensitivity_results.items()
                }

            # Performance summary
            report["performance_summary"] = {
                "total_improvement": 0,  # Would be calculated from initial/final values
                "optimization_efficiency": convergence_metrics.efficiency_score,
                "convergence_quality": convergence_metrics.stability_metric,
                "computational_efficiency": convergence_metrics.final_objective_value / max(convergence_metrics.computational_cost, 1)
            }

            return report

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e)}

# Utility functions for advanced analysis
def calculate_hypervolume_indicator(pareto_front: List[Tuple[float, float]],
                                  reference_point: Tuple[float, float]) -> float:
    """Calculate hypervolume indicator for multi-objective optimization"""
    try:
        if not pareto_front:
            return 0.0

        # Sort by first objective (assuming minimization)
        sorted_front = sorted(pareto_front, key=lambda x: x[0])

        hypervolume = 0.0
        current_y = reference_point[1]

        for obj1, obj2 in sorted_front:
            if obj2 < current_y:
                hypervolume += (current_y - obj2) * (reference_point[0] - obj1)
                current_y = obj2

        return hypervolume

    except Exception as e:
        logger.error(f"Error calculating hypervolume indicator: {e}")
        return 0.0

def calculate_pareto_dominance_count(solutions: List[List[float]],
                                   target_idx: int) -> int:
    """Count how many solutions are dominated by a target solution"""
    try:
        if target_idx >= len(solutions):
            return 0

        target = solutions[target_idx]
        dominance_count = 0

        for i, solution in enumerate(solutions):
            if i == target_idx:
                continue

            # Check if target dominates solution (better in all objectives)
            if all(t <= s for t, s in zip(target, solution)) and any(t < s for t, s in zip(target, solution)):
                dominance_count += 1

        return dominance_count

    except Exception as e:
        logger.error(f"Error calculating Pareto dominance count: {e}")
        return 0