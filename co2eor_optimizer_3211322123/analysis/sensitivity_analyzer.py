import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Union, Any, Tuple, Callable
from copy import deepcopy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Ensure project root is in sys.path for robust imports
try:
    project_root_path = Path(__file__).resolve().parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
except NameError:
    # Fallback for environments where __file__ is not defined
    project_root_path = Path.cwd()
    if str(project_root_path) not in sys.path:
         sys.path.insert(0, str(project_root_path))

# Core application imports
try:
    from co2eor_optimizer.core.optimisation_engine import OptimizationEngine
    from co2eor_optimizer.core.data_models import (
        ReservoirData, PVTProperties, EORParameters, EconomicParameters,
        OperationalParameters, ProfileParameters, EOSModelParameters
    )
    from co2eor_optimizer.core.eos_models import PengRobinsonEOS, SoaveRedlichKwongEOS
    from co2eor_optimizer.core.recovery_models import recovery_factor
    from co2eor_optimizer.config_manager import config_manager
except ImportError as e:
    logging.critical(f"CRITICAL: Could not import core project modules for SensitivityAnalyzer: {e}. "
                     "This module will not be functional.")
    # Define dummy types to allow the script to be parsed without crashing
    OptimizationEngine, ReservoirData, PVTProperties, EORParameters, EconomicParameters = Any, Any, Any, Any, Any
    OperationalParameters, ProfileParameters, EOSModelParameters = Any, Any, Any
    PengRobinsonEOS, SoaveRedlichKwongEOS, recovery_factor = Any, Any, None
    class DummyConfigManagerForSA:
        _is_loaded_from_file = False
        def get(self, key, default=None): return default
        def get_section(self, key): return {}
        def load_config(self, path): pass
    config_manager = DummyConfigManagerForSA()

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    def __init__(self,
                 engine: 'OptimizationEngine',
                 base_case_opt_results: Optional[Dict[str, Any]] = None):

        if not isinstance(engine, OptimizationEngine):
            raise TypeError("SensitivityAnalyzer requires a valid OptimizationEngine instance.")
        self.engine = engine

        # Store deepcopies of all base data models from the engine for pristine state restoration
        self._base_reservoir_data_for_reopt = deepcopy(engine.reservoir)
        self._base_pvt_for_reopt = deepcopy(engine.pvt)
        self._base_eor_params_for_reopt = deepcopy(engine.eor_params)
        self._base_econ_params_for_reopt = deepcopy(engine.economic_params)
        self._base_op_params_for_reopt = deepcopy(engine.operational_params)
        self._base_profile_params_for_reopt = deepcopy(engine.profile_params)
        self._base_eos_params_for_reopt: Optional[EOSModelParameters] = deepcopy(engine.reservoir.eos_model) if engine.reservoir.eos_model else None

        # Resolve the fixed EOR and Economic parameters for non-reoptimization runs
        self.base_eor_params_dict, self.base_econ_params_instance = \
            self._resolve_base_case_parameters(base_case_opt_results)

        self.sensitivity_run_data: List[Dict[str, Any]] = []

    def get_configurable_parameters(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Returns a dictionary of parameters that can be varied in sensitivity analysis,
        structured for UI consumption. The inner value is a list of tuples to
        preserve order and ensure compatibility with UI components.
        """
        # FIX: Changed the inner data structure from a dictionary to a list of tuples.
        # This resolves a "too many values to unpack" error in the consuming UI widget,
        # which likely expects an iterable of 2-item tuples rather than a dict's .items() view.
        return {
            "Economic": [
                ("econ.oil_price_usd_per_bbl", "Oil Price ($/bbl)"),
                ("econ.co2_purchase_cost_usd_per_tonne", "CO2 Purchase Cost ($/tonne)"),
                ("econ.discount_rate_fraction", "Discount Rate (fraction)")
            ],
            "Operational (EOR)": [
                ("eor.pressure", "Injection Pressure (psi)"),
                ("eor.rate", "Injection Rate (bpd)"),
                ("eor.mobility_ratio", "Mobility Ratio")
            ],
            "Reservoir / Fluid": [
                ("reservoir.avg_porosity", "Average Porosity (fraction)"),
                ("fluid.mmp_value", "MMP (psi)")
            ],
            "Recovery Model": [
                ("model.v_dp_coefficient", "V_DP Coefficient (Koval)"),
                ("model.kv_factor", "Kv Factor (Miscible)")
            ]
        }

    def _resolve_base_case_parameters(self,
                                     opt_results: Optional[Dict[str, Any]] = None
                                     ) -> Tuple[Dict[str, float], 'EconomicParameters']:
        """
        Determines the base EOR operational parameters and economic parameters to use
        for sensitivity runs where re-optimization is NOT performed.
        It prioritizes optimized results if provided, otherwise derives from the engine's defaults.
        """
        base_eor_dict = {}
        if opt_results and 'optimized_params_final_clipped' in opt_results:
            base_eor_dict.update(opt_results['optimized_params_final_clipped'])
            logger.info("SA: Using EOR parameters from provided optimization results as base case.")
        else:
            logger.info("SA: Deriving base EOR parameters from engine's default settings.")
            ga_bounds = self.engine._get_ga_parameter_bounds()
            eor_dc = self.engine.eor_params
            for param, (low, high) in ga_bounds.items():
                if hasattr(eor_dc, param):
                    base_eor_dict[param] = np.clip(getattr(eor_dc, param), low, high)
                else:
                    base_eor_dict[param] = (low + high) / 2.0

        base_econ_instance = deepcopy(self.engine.economic_params)
        logger.info(f"SA: Resolved base EOR params: { {k: f'{v:.2f}' for k, v in base_eor_dict.items()} }")
        return base_eor_dict, base_econ_instance

    def _parse_param_path(self, param_path: str) -> Tuple[Optional[str], str]:
        if '.' in param_path:
            parts = param_path.split('.', 1)
            return parts[0].lower(), parts[1]
        return None, param_path

    def _evaluate_scenario(self,
                           current_eor_params_dict: Dict[str, float],
                           current_econ_params_instance: 'EconomicParameters',
                           objectives_to_calc: List[str],
                           temp_eos_params_override: Optional['EOSModelParameters'] = None,
                           temp_reservoir_overrides: Optional[Dict[str, Any]] = None,
                           temp_fluid_overrides: Optional[Dict[str, Any]] = None,
                           temp_model_param_overrides: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, float]:
        """
        Core evaluation function. This is the heart of the sensitivity analysis,
        calculating objectives for a single, specific scenario.
        """
        if recovery_factor is None:
            raise ImportError("The 'recovery_factor' function is not available.")
        results: Dict[str, float] = {}

        temp_eos_model = None
        if temp_eos_params_override:
            try:
                eos_type = temp_eos_params_override.eos_type.lower()
                if eos_type == 'peng-robinson':
                    temp_eos_model = PengRobinsonEOS(temp_eos_params_override)
                elif eos_type == 'soave-redlich-kwong':
                    temp_eos_model = SoaveRedlichKwongEOS(temp_eos_params_override)
            except Exception as e:
                logger.error(f"Failed to instantiate temporary EOS model for evaluation: {e}")
        elif self.engine.eos_model_instance:
            temp_eos_model = self.engine.eos_model_instance

        pressure = current_eor_params_dict['pressure']
        fluid_props_at_p = {}
        if temp_eos_model:
            try:
                fluid_props_at_p = temp_eos_model.calculate_properties(pressure, self.engine.pvt.temperature)
            except Exception as e_eos_calc:
                logger.warning(f"EOS calculation failed at P={pressure:.2f} psi: {e_eos_calc}")

        mmp_to_use = temp_fluid_overrides.get('mmp_value', self.engine.mmp) if temp_fluid_overrides else self.engine.mmp
        mmp_to_use = mmp_to_use or config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)

        if temp_reservoir_overrides and 'avg_porosity' in temp_reservoir_overrides:
            porosity_to_use = temp_reservoir_overrides['avg_porosity']
        else:
            porosity_to_use = self.engine.avg_porosity

        rf_call_kwargs = {**self.engine._recovery_model_init_kwargs}
        if temp_model_param_overrides:
            rf_call_kwargs.update(temp_model_param_overrides)

        if 'oil_viscosity_cp' in fluid_props_at_p:
            rf_call_kwargs['mu_oil'] = fluid_props_at_p['oil_viscosity_cp']
        if 'gas_viscosity_cp' in fluid_props_at_p:
            rf_call_kwargs['mu_co2'] = fluid_props_at_p['gas_viscosity_cp']
        
        rf_call_kwargs.update(current_eor_params_dict)

        effective_co2_rate_for_rf = current_eor_params_dict['rate']
        if self.engine.eor_params.injection_scheme == 'wag':
            water_frac = current_eor_params_dict.get('water_fraction', 0.5)
            effective_co2_rate_for_rf *= (1.0 - water_frac)
            rf_call_kwargs['water_fraction'] = water_frac
            
        current_rf_val = recovery_factor(
            pressure, effective_co2_rate_for_rf, porosity_to_use, mmp_to_use,
            model=self.engine.recovery_model, **rf_call_kwargs
        )
        if "recovery_factor" in objectives_to_calc:
            results["recovery_factor"] = current_rf_val

        if any(obj in objectives_to_calc for obj in ["npv", "co2_utilization"]):
            original_econ_params = self.engine.economic_params
            self.engine.economic_params = current_econ_params_instance
            try:
                annual_oil, co2_inj, water_inj, water_disp = \
                    self.engine._calculate_annual_profiles(current_rf_val, current_eor_params_dict)
                if "npv" in objectives_to_calc:
                    results["npv"] = self.engine._calculate_npv(annual_oil, co2_inj, water_inj, water_disp)
                if "co2_utilization" in objectives_to_calc:
                    results["co2_utilization"] = self.engine._calculate_co2_utilization_factor(annual_oil, co2_inj)
            finally:
                self.engine.economic_params = original_econ_params # Restore

        return results

    def run_one_way_sensitivity(self,
                                params_to_vary: Dict[str, Union[Tuple[float, float, Optional[str]], List[float]]],
                                num_steps: int = 11,
                                objectives: List[str] = ['npv', 'recovery_factor']) -> pd.DataFrame:
        self.sensitivity_run_data = []
        all_sensitivity_points = []
        valid_objectives = {"npv", "recovery_factor", "co2_utilization"}
        objectives_to_calc = [obj for obj in objectives if obj in valid_objectives]
        if not objectives_to_calc: return pd.DataFrame()

        base_case_objectives = self._evaluate_scenario(self.base_eor_params_dict, self.base_econ_params_instance, objectives_to_calc)
        logger.info(f"SA Base Case Objectives: {base_case_objectives}")
        all_sensitivity_points.append({'parameter_varied': "BaseCase", 'parameter_value': "Base", **base_case_objectives})

        for param_path, variation_spec in params_to_vary.items():
            logger.info(f"Running one-way sensitivity for: {param_path}")
            param_category, param_name = self._parse_param_path(param_path)

            if isinstance(variation_spec, list):
                test_values = [float(v) for v in variation_spec]
            else:
                min_v, max_v = float(variation_spec[0]), float(variation_spec[1])
                scale = variation_spec[2].lower() if len(variation_spec) == 3 else 'linear'
                test_values = np.logspace(np.log10(min_v), np.log10(max_v), num_steps) if scale == 'log' and min_v > 0 else np.linspace(min_v, max_v, num_steps)

            for test_value in test_values:
                temp_eor_dict = self.base_eor_params_dict.copy()
                temp_econ_instance = deepcopy(self.base_econ_params_instance)
                temp_eos_override, temp_res_over, temp_fluid_over, temp_model_over = None, None, None, None

                try:
                    if param_category == 'eor': temp_eor_dict[param_name] = test_value
                    elif param_category == 'econ': setattr(temp_econ_instance, param_name, test_value)
                    elif param_category == 'reservoir': temp_res_over = {param_name: test_value}
                    elif param_category == 'fluid': temp_fluid_over = {param_name: test_value}
                    elif param_category == 'model': temp_model_over = {param_name: test_value}
                    elif param_category == 'eos':
                        if self._base_eos_params_for_reopt is None:
                            logger.warning(f"Cannot vary '{param_path}'; no base EOS model defined."); continue
                        temp_eos_override = deepcopy(self._base_eos_params_for_reopt)
                        path_parts = param_name.split('.')
                        target_attr = getattr(temp_eos_override, path_parts[0])
                        if isinstance(target_attr, np.ndarray) and len(path_parts) == 3:
                            target_attr[int(path_parts[1]), int(path_parts[2])] = test_value
                        else:
                            raise ValueError(f"Unsupported EOS parameter path format: {param_name}")
                    else:
                        logger.warning(f"Unknown category or param '{param_path}'. Skipping."); continue
                except Exception as e_apply:
                    logger.error(f"Error applying sensitivity value for '{param_path}' = {test_value}: {e_apply}"); continue

                evaluated_objectives = self._evaluate_scenario(
                    temp_eor_dict, temp_econ_instance, objectives_to_calc,
                    temp_eos_override, temp_res_over, temp_fluid_over, temp_model_over
                )
                all_sensitivity_points.append({'parameter_varied': param_path, 'parameter_value': test_value, **evaluated_objectives})

        self.sensitivity_run_data = [p for p in all_sensitivity_points if p['parameter_varied'] != 'BaseCase']
        return pd.DataFrame(all_sensitivity_points)


    def run_reoptimization_sensitivity(self,
                                     primary_param_to_vary: str,
                                     variation_values: List[float],
                                     optimization_method_on_engine: str = "hybrid_optimize",
                                     target_optimized_output_keys: Optional[List[str]] = None,
                                     objectives_at_optimum: Optional[List[str]] = None
                                     ) -> pd.DataFrame:
        logger.info(f"Starting Re-optimization Sensitivity for '{primary_param_to_vary}' using '{optimization_method_on_engine}'.")
        results_list = []
        if target_optimized_output_keys is None: target_optimized_output_keys = ['pressure', 'rate']
        if objectives_at_optimum is None: objectives_at_optimum = [self.engine.chosen_objective, 'final_recovery_factor_reported']

        param_category, param_name = self._parse_param_path(primary_param_to_vary)

        for i, p_value in enumerate(variation_values):
            logger.info(f"Re-optimizing (run {i+1}/{len(variation_values)}) for {primary_param_to_vary} = {p_value}")

            current_reservoir = deepcopy(self._base_reservoir_data_for_reopt)
            current_econ_params = deepcopy(self._base_econ_params_for_reopt)
            engine_init_overrides = {}

            try:
                if param_category == 'econ':
                    setattr(current_econ_params, param_name, p_value)
                elif param_category == 'fluid' and param_name == 'mmp_value':
                    engine_init_overrides['mmp_init_override'] = p_value
                elif param_category == 'reservoir' and param_name == 'avg_porosity':
                    engine_init_overrides['avg_porosity_init_override'] = p_value
                elif param_category == 'eos':
                    if self._base_eos_params_for_reopt is None:
                        logger.warning(f"Cannot vary '{primary_param_to_vary}'; no base EOS model defined."); continue
                    
                    temp_eos_params = deepcopy(self._base_eos_params_for_reopt)
                    path_parts = param_name.split('.')
                    target_attr = getattr(temp_eos_params, path_parts[0])
                    if isinstance(target_attr, np.ndarray) and len(path_parts) == 3:
                        target_attr[int(path_parts[1]), int(path_parts[2])] = p_value
                    else:
                        raise ValueError(f"Unsupported EOS param path: {param_name}")
                    current_reservoir.eos_model = temp_eos_params
                else:
                    logger.error(f"Re-optimization for category '{param_category}' is not implemented. Skipping."); continue
            except Exception as e_apply:
                logger.error(f"Error applying re-opt sensitivity value for '{primary_param_to_vary}'={p_value}: {e_apply}"); continue

            try:
                temp_engine = OptimizationEngine(
                    reservoir=current_reservoir,
                    pvt=deepcopy(self._base_pvt_for_reopt),
                    eor_params_instance=deepcopy(self._base_eor_params_for_reopt),
                    economic_params_instance=current_econ_params,
                    operational_params_instance=deepcopy(self._base_op_params_for_reopt),
                    profile_params_instance=deepcopy(self._base_profile_params_for_reopt),
                    **engine_init_overrides
                )

                opt_func: Callable = getattr(temp_engine, optimization_method_on_engine)
                opt_results = opt_func()

                result_row: Dict[str, Any] = {primary_param_to_vary: p_value}
                if 'optimized_params_final_clipped' in opt_results:
                    for key in target_optimized_output_keys:
                        result_row[f"opt_{key}"] = opt_results['optimized_params_final_clipped'].get(key)
                for obj_key in objectives_at_optimum:
                    result_row[obj_key] = opt_results.get(obj_key)
                results_list.append(result_row)
            except Exception as e_reopt:
                logger.error(f"Re-optimization failed for {primary_param_to_vary}={p_value}: {e_reopt}", exc_info=True)
                results_list.append({primary_param_to_vary: p_value, 'error': str(e_reopt)})

        return pd.DataFrame(results_list)

    def run_two_way_sensitivity(self,
                                param1_path: str, param1_values: List[float],
                                param2_path: str, param2_values: List[float],
                                objectives: List[str] = ['npv']) -> pd.DataFrame:
        logger.info(f"Starting Two-Way Sensitivity for '{param1_path}' and '{param2_path}'.")
        results_list = []
        objectives_to_calc = [obj for obj in ["npv", "recovery_factor", "co2_utilization"] if obj in objectives]
        if not objectives_to_calc: return pd.DataFrame()

        p1_cat, p1_name = self._parse_param_path(param1_path)
        p2_cat, p2_name = self._parse_param_path(param2_path)

        for val1 in param1_values:
            for val2 in param2_values:
                temp_eor_dict = self.base_eor_params_dict.copy()
                temp_econ_instance = deepcopy(self.base_econ_params_instance)
                temp_eos_override, temp_res_over, temp_fluid_over, temp_model_over = None, None, None, None

                if p1_cat == 'eor': temp_eor_dict[p1_name] = val1
                elif p1_cat == 'econ': setattr(temp_econ_instance, p1_name, val1)
                
                if p2_cat == 'eor': temp_eor_dict[p2_name] = val2
                elif p2_cat == 'econ': setattr(temp_econ_instance, p2_name, val2)

                eval_results = self._evaluate_scenario(
                    temp_eor_dict, temp_econ_instance, objectives_to_calc,
                    temp_eos_override, temp_res_over, temp_fluid_over, temp_model_over
                )
                results_list.append({param1_path: val1, param2_path: val2, **eval_results})
        
        return pd.DataFrame(results_list)

    def plot_contour_chart(self, two_way_df: pd.DataFrame,
                             param1_col: str, param2_col: str, objective_col: str,
                             title: Optional[str] = None) -> Optional[go.Figure]:
        if not all(col in two_way_df.columns for col in [param1_col, param2_col, objective_col]):
            logger.error("Missing required columns for contour plot.")
            return None
        
        try:
            pivot_df = two_way_df.pivot(index=param1_col, columns=param2_col, values=objective_col)
            fig = go.Figure(data=[
                go.Contour(
                    z=pivot_df.values, x=pivot_df.columns.values, y=pivot_df.index.values,
                    colorscale='Viridis', colorbar=dict(title=objective_col.replace('_',' ').title()),
                    contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10,color='white'))
                )
            ])
            fig.update_layout(
                title=title or f"Contour Plot: {objective_col.replace('_',' ').title()}",
                xaxis_title=param2_col, yaxis_title=param1_col
            )
            return fig
        except Exception as e:
            logger.error(f"Error generating contour plot: {e}", exc_info=True)
            return None

    def plot_tornado_chart(self,
                           sensitivity_df: pd.DataFrame,
                           objective_col: str,
                           title: Optional[str] = None) -> Optional[go.Figure]:
        if sensitivity_df.empty or objective_col not in sensitivity_df.columns:
            logger.warning("Cannot generate tornado chart: DataFrame empty or objective column missing.")
            return None
        base_case_row = sensitivity_df[sensitivity_df['parameter_varied'] == 'BaseCase']
        if base_case_row.empty:
            logger.error(f"BaseCase row not found. Tornado chart needs it for reference.")
            return None
        base_objective_value = base_case_row.iloc[0][objective_col]
        if pd.isna(base_objective_value):
            logger.error(f"BaseCase objective value is NaN. Cannot create tornado chart.")
            return None

        results_df = sensitivity_df[sensitivity_df['parameter_varied'] != 'BaseCase'].copy()
        summary_data = []
        for param_path in results_df['parameter_varied'].unique():
            subset = results_df[results_df['parameter_varied'] == param_path]
            if subset[objective_col].isnull().all(): continue
            min_val = subset[objective_col].min(skipna=True)
            max_val = subset[objective_col].max(skipna=True)
            summary_data.append({'parameter': param_path, 'low': min_val, 'high': max_val, 'range': abs(max_val - min_val)})
        
        if not summary_data: return None
        summary_df = pd.DataFrame(summary_data).sort_values(by='range', ascending=False).dropna(subset=['low', 'high'])
        if summary_df.empty: return None

        fig = go.Figure()
        fig.add_trace(go.Bar(y=summary_df['parameter'], x=summary_df['low'] - base_objective_value, name='Min Impact', orientation='h', marker_color='crimson'))
        fig.add_trace(go.Bar(y=summary_df['parameter'], x=summary_df['high'] - base_objective_value, name='Max Impact', orientation='h', marker_color='forestgreen'))
        fig.update_layout(
            barmode='overlay', title_text=title or f"Tornado Chart for {objective_col.replace('_',' ').title()}",
            xaxis_title_text=f"Change in {objective_col.replace('_',' ').title()} (from Base: {base_objective_value:.3e})",
            yaxis_title_text="Parameter", yaxis=dict(autorange="reversed"), height=max(400, len(summary_df) * 40)
        )
        return fig

    def plot_spider_chart(self,
                          sensitivity_df: pd.DataFrame,
                          objective_col: str,
                          title: Optional[str] = None,
                          normalize_radial: bool = False) -> Optional[go.Figure]:
        if sensitivity_df.empty or objective_col not in sensitivity_df.columns: return None
        base_case_row = sensitivity_df[sensitivity_df['parameter_varied'] == 'BaseCase']
        if base_case_row.empty: return None
        base_val = base_case_row.iloc[0][objective_col]
        if pd.isna(base_val): return None
        results_df = sensitivity_df[sensitivity_df['parameter_varied'] != 'BaseCase'].copy()
        params = results_df['parameter_varied'].unique()
        if not params.size: return None
        
        low_vals, high_vals = [], []
        for param in params:
            subset = results_df[results_df['parameter_varied'] == param]
            if subset[objective_col].isnull().all():
                low_vals.append(np.nan); high_vals.append(np.nan)
                continue
            low_vals.append(subset.loc[subset['parameter_value'].idxmin(), objective_col])
            high_vals.append(subset.loc[subset['parameter_value'].idxmax(), objective_col])

        fig = go.Figure()
        r_low, r_high, r_base = np.array(low_vals), np.array(high_vals), np.array([base_val] * len(params))
        axis_title, base_legend = objective_col.replace('_',' ').title(), f'Base ({base_val:.3e})'

        if normalize_radial and abs(base_val) > 1e-9:
            r_low, r_high = (r_low - base_val) / base_val * 100, (r_high - base_val) / base_val * 100
            r_base = np.zeros_like(r_base)
            axis_title, base_legend = f"% Change in {axis_title}", 'Base (0% Change)'
        
        fig.add_trace(go.Scatterpolar(r=r_low, theta=params, fill='toself', name='Low Setting'))
        fig.add_trace(go.Scatterpolar(r=r_high, theta=params, fill='toself', name='High Setting'))
        fig.add_trace(go.Scatterpolar(r=r_base, theta=params, mode='lines', name=base_legend, line=dict(dash='dash')))
        fig.update_layout(polar=dict(radialaxis=dict(title_text=axis_title)), title=title or f"Spider Chart for {objective_col.replace('_',' ').title()}")
        return fig