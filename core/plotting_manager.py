import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any
from copy import deepcopy
from analysis.profiler_refactored import ProductionProfiler

class PlottingManager:
    def __init__(self, engine):
        self.engine = engine

    def plot_optimization_convergence(self, results_to_plot=None) -> go.Figure:
        source = results_to_plot or self.engine._results
        if not source: return go.Figure().update_layout(title_text="No results to plot.")
        fig = go.Figure()
        title = f'Optimization Convergence for {self.engine.chosen_objective.replace("_"," ").title()}'
        
        last_eval = 0
        
        ga_res = source.get('ga_full_results_for_hybrid', source if source.get('method') == 'genetic_algorithm' else None)
        if ga_res:
            history = None
            sol_per_pop = None
            avg_history = None
            std_history = None

            if (ga_instance := ga_res.get('pygad_instance')):
                history = ga_instance.best_solutions_fitness
                sol_per_pop = ga_instance.sol_per_pop
            elif 'ga_statistics' in ga_res:
                history = ga_res['ga_statistics'].get('best_fitness_history')
                avg_history = ga_res['ga_statistics'].get('avg_fitness_history')
                std_history = ga_res['ga_statistics'].get('std_fitness_history')
                sol_per_pop = ga_res['ga_statistics'].get('population_size')

            if history and sol_per_pop:
                evals = np.arange(1, len(history) + 1) * sol_per_pop
                fig.add_trace(go.Scatter(x=evals, y=history, mode='lines', name='GA Best Fitness'))
                if avg_history and std_history:
                    fig.add_trace(go.Scatter(x=evals, y=avg_history, mode='lines', name='GA Avg Fitness', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=evals, y=np.array(avg_history) + np.array(std_history), fill='tonexty', mode='lines', line=dict(color='rgba(255,165,0,0.2)'), name='Std Dev'))
                    fig.add_trace(go.Scatter(x=evals, y=np.array(avg_history) - np.array(std_history), fill='tonexty', mode='lines', line=dict(color='rgba(255,165,0,0.2)'), showlegend=False))
                last_eval = evals[-1] if evals.size > 0 else 0

        if source.get('method') == 'pso' and 'pso_cost_history' in source:
            history = -np.array(source['pso_cost_history'])
            evals = np.arange(1, len(history) + 1) * self.engine.pso_params_default_config.n_particles
            fig.add_trace(go.Scatter(x=evals, y=history, mode='lines', name='PSO Best Fitness'))
            
        if 'bayes_opt_obj' in source and hasattr(source['bayes_opt_obj'], 'y'):
             bo_y = source['bayes_opt_obj'].y
             bo_x = np.arange(last_eval + 1, last_eval + 1 + len(bo_y))
             fig.add_trace(go.Scatter(x=bo_x, y=bo_y, mode='markers', name='BO Evaluations'))
        
        fig.update_layout(title_text=title, xaxis_title_text='Function Evaluations', yaxis_title_text='Objective Value')
        return fig

    def plot_parameter_sensitivity(self, param_name, results_to_use=None) -> go.Figure:
        source = results_to_use or self.engine._results
        if not (source and 'optimized_params_final_clipped' in source):
            return go.Figure().update_layout(title_text="No optimized parameters available.")
        
        opt_base = source['optimized_params_final_clipped']
        all_bounds = self.engine._get_parameter_bounds()
        
        if param_name not in opt_base: return go.Figure().update_layout(title_text=f"Parameter '{param_name}' not in results.")

        curr_val = opt_base[param_name]
        range_multiplier = self.engine.advanced_engine_params.sensitivity_plot_range_multiplier

        low_bound, high_bound = all_bounds.get(param_name, (curr_val * (1 - range_multiplier), curr_val * (1 + range_multiplier)))
        test_values = np.linspace(low_bound, high_bound, 10)
        objective_values = []

        for test_val in test_values:
            test_params = opt_base.copy()
            test_params[param_name] = test_val
            eval_result = self.engine.evaluate_for_analysis(test_params)
            obj_val = eval_result.get(self.engine.chosen_objective, 0.0)
            objective_values.append(obj_val)

        fig = go.Figure(go.Scatter(x=test_values, y=objective_values, mode='lines+markers'))
        fig.update_layout(
            title_text=f"Sensitivity: {param_name.replace('_', ' ').title()}",
            xaxis_title=param_name.replace('_', ' ').title(),
            yaxis_title=self.engine.chosen_objective.replace('_', ' ').title()
        )
        return fig

    def plot_production_profiles(self, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Generates a plot of production profiles from optimization results."""
        source = results_to_use or self.engine._results
        if not (source and 'optimized_params_final_clipped' in source):
            return go.Figure().update_layout(title_text="No results to plot.")
        
        final_params_dict = source['optimized_params_final_clipped']
        rf = source.get('final_metrics', {}).get('recovery_factor', 0.0)
        
        temp_eor_params = deepcopy(self.engine.eor_params)
        for key, value in final_params_dict.items():
            if hasattr(temp_eor_params, key):
                setattr(temp_eor_params, key, value)

        profiler = ProductionProfiler(self.engine.reservoir, self.engine.pvt, temp_eor_params, self.engine.operational_params, self.engine.profile_params)
        profiles = profiler.generate_all_profiles(ooip_stb=self.engine.reservoir.ooip_stb)
        
        resolution = self.engine.operational_params.time_resolution
        profile_key = f'{resolution}_oil_stb'
        if profile_key not in profiles:
            return go.Figure().update_layout(title_text=f"No data for '{resolution}' resolution.")

        time_steps = np.arange(1, len(profiles[profile_key]) + 1)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=time_steps, y=profiles[f'{resolution}_oil_stb'], name='Oil (bbl)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_steps, y=profiles[f'{resolution}_water_injected_bbl'], name='Water Inj (bbl)'), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_steps, y=profiles[f'{resolution}_co2_injected_mscf'], name='CO2 Inj (MSCF)'), secondary_y=True)

        # Add volumetric sweep efficiency
        sweep_key = f'{resolution}_volumetric_sweep'
        if sweep_key in profiles:
            sweep_profile = profiles[sweep_key]
            fig.add_trace(go.Scatter(x=time_steps, y=sweep_profile, name='Volumetric Sweep', yaxis='y3', line=dict(color='purple', dash='dot')))
            
            avg_sweep = np.mean(sweep_profile)
            std_sweep = np.std(sweep_profile)
            
            fig.update_layout(
                yaxis3=dict(
                    title="Volumetric Sweep Efficiency",
                    overlaying="y",
                    side="right",
                    position=0.9,
                    showgrid=False,
                    range=[0, 1]
                ),
                annotations=[
                    dict(
                        x=0.95, y=0.05, xref='paper', yref='paper',
                        text=f'Avg Sweep: {avg_sweep:.2f}<br>Std Dev: {std_sweep:.2f}',
                        showarrow=False, align='left', bordercolor='black', borderwidth=1
                    )
                ]
            )

        
        fig.update_layout(title_text=f'Optimized {resolution.title()} Profiles', xaxis_title=f'Project {resolution.title()}', barmode='group')
        return fig

    def plot_objective_vs_parameter(self, param_name: str, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Plots objective values against a specific parameter for Bayesian optimization results."""
        source = results_to_use or self.engine._results
        
        bo_res = None
        if source and 'bayes_opt_obj' in source and hasattr(source['bayes_opt_obj'], 'res'):
            bo_res = source['bayes_opt_obj'].res
        elif source and 'bayes_opt_obj_res' in source:
            bo_res = source['bayes_opt_obj_res']

        if not bo_res:
            return go.Figure().update_layout(title_text="No Bayesian optimization results available.")
            
        param_values = [res['params'][param_name] for res in bo_res if param_name in res['params']]
        target_values = [res['target'] for res in bo_res if param_name in res['params']]
        if not param_values:
            return go.Figure().update_layout(title_text=f"Parameter '{param_name}' not found in results.")
        
        fig = go.Figure(go.Scatter(x=param_values, y=target_values, mode='markers'))
        obj_display = self.engine.chosen_objective.replace('_', ' ').title()
        fig.update_layout(
            title_text=f'{obj_display} vs. {param_name.replace("_", " ").title()}',
            xaxis_title=param_name.replace('_', ' ').title(),
            yaxis_title=obj_display
        )
        return fig

    def plot_co2_performance_summary_table(self, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Generates a table summarizing CO2 performance metrics, using material balance analysis if available."""
        source = results_to_use or self.engine._results
        if not source:
            return go.Figure().update_layout(title_text="No results to plot.")

        mb_stats = source.get('material_balance_analysis', {}).get('summary_statistics', {})
        final_metrics = source.get('final_metrics', {})
        
        co2_density_tonne_per_mscf = self.engine.eor_params.co2_density_tonne_per_mscf

        if mb_stats:
            # Use data from material balance analysis for consistency
            total_injected_tonne = mb_stats.get('total_injected_tonne', 0)
            total_produced_tonne = mb_stats.get('total_produced_tonne', 0)
            total_stored_tonne = mb_stats.get('total_net_stored_tonne', 0)
            avg_efficiency = mb_stats.get('avg_storage_efficiency', 0)
            
            # Convert tonnes back to MSCF for display consistency in the table
            total_injected_mscf = total_injected_tonne / co2_density_tonne_per_mscf if co2_density_tonne_per_mscf > 0 else 0
            total_produced_mscf = total_produced_tonne / co2_density_tonne_per_mscf if co2_density_tonne_per_mscf > 0 else 0
        else:
            # Fallback to original method if material balance data is not present
            profiles = source.get('optimized_profiles', {})
            op_params = source.get('operational_parameters', self.engine.operational_params)
            resolution = op_params.time_resolution
            
            total_injected_mscf = np.sum(profiles.get(f'{resolution}_co2_injected_mscf', 0))
            total_produced_mscf = np.sum(profiles.get(f'{resolution}_co2_produced_mscf', 0))
            total_stored_tonne = final_metrics.get('total_co2_stored_tonne', 0)
            avg_efficiency = final_metrics.get('avg_storage_efficiency', 0)

        co2_utilization = final_metrics.get('co2_utilization', 0)

        fig = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Value', 'Units']),
            cells=dict(values=[
                ['Total CO2 Injected', 'Total CO2 Produced', 'Total CO2 Stored', 'Average Storage Efficiency', 'CO2 Utilization'],
                [f'{total_injected_mscf:,.0f}', f'{total_produced_mscf:,.0f}', f'{total_stored_tonne:,.0f}', f'{avg_efficiency:.2%}', f'{co2_utilization:.2f}'],
                ['MSCF', 'MSCF', 'tonnes', '%', 'MSCF/stb']
            ]))
        ])
        fig.update_layout(title_text='CO2 Performance Summary')
        return fig

    def plot_ga_objective_distribution(self, results_to_use: Optional[Dict[str, Any]] = None) -> go.Figure:
        source = results_to_use or self.engine._results
        if not (source and 'pygad_instance' in source):
            return go.Figure().update_layout(title_text="No GA results to plot.")

        ga_instance = source['pygad_instance']
        objectives = ga_instance.last_generation_fitness

        avg_obj = np.mean(objectives)
        std_obj = np.std(objectives)

        fig = go.Figure(data=[go.Histogram(x=objectives, nbinsx=20)])
        fig.update_layout(
            title_text='GA Population Objective Value Distribution',
            xaxis_title='Objective Value',
            yaxis_title='Frequency',
            annotations=[
                dict(
                    x=0.95, y=0.95, xref='paper', yref='paper',
                    text=f'Avg: {avg_obj:.3f}<br>Std: {std_obj:.3f}',
                    showarrow=False, align='left', bordercolor='black', borderwidth=1
                )
            ]
        )
        return fig

    def plot_co2_breakthrough(
        self,
        years: np.ndarray,
        gor_profile: np.ndarray,
        breakthrough_time: float,
        co2_production: Optional[np.ndarray] = None,
        saturation_profile: Optional[np.ndarray] = None
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
        from visualizations.breakthrough_plots import BreakthroughVisualizer

        visualizer = BreakthroughVisualizer()
        return visualizer.plot_breakthrough_comprehensive(
            years=years,
            gor_profile=gor_profile,
            breakthrough_time=breakthrough_time,
            co2_production=co2_production,
            saturation_profile=saturation_profile
        )

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
        from visualizations.fault_mechanics_plots import FaultMechanicsVisualizer

        visualizer = FaultMechanicsVisualizer()
        return visualizer.plot_fault_stability_comprehensive(fault_history)

    def plot_scenario_comparison(
        self,
        results_with_fault: Dict[str, Any],
        results_without_fault: Dict[str, Any]
    ) -> go.Figure:
        """
        Plot side-by-side comparison of scenarios (with/without fault mechanics).

        Args:
            results_with_fault: Simulation results with fault mechanics enabled
            results_without_fault: Simulation results without fault mechanics

        Returns:
            Plotly Figure with side-by-side comparison
        """
        from visualizations.comparison_plots import ScenarioComparisonVisualizer

        visualizer = ScenarioComparisonVisualizer()
        return visualizer.plot_scenario_comparison(results_with_fault, results_without_fault)

    def plot_fault_effect_analysis(
        self,
        results_with_fault: Dict[str, Any],
        results_without_fault: Dict[str, Any]
    ) -> go.Figure:
        """
        Plot fault effect analysis with delta curves.

        Args:
            results_with_fault: Simulation results with fault mechanics
            results_without_fault: Simulation results without fault mechanics

        Returns:
            Plotly Figure with fault effect analysis
        """
        from visualizations.comparison_plots import ScenarioComparisonVisualizer

        visualizer = ScenarioComparisonVisualizer()
        return visualizer.plot_fault_effect_analysis(results_with_fault, results_without_fault)
