import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from copy import deepcopy
import chaospy  # For probability distributions and sampling
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# --- Project-level imports with robust path handling ---
try:
    project_root_path = Path(__file__).resolve().parent.parent
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
except NameError:
    project_root_path = Path.cwd()
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))

try:
    from core.optimisation_engine import OptimizationEngine
    from core.data_models import (
        ReservoirData, PVTProperties, EORParameters, EconomicParameters,
        OperationalParameters, ProfileParameters, EOSModelParameters
    )
    from config_manager import config_manager
    from core.recovery_models import recovery_factor  # Import directly for UQ eval
except ImportError as e:
    logging.critical(f"CRITICAL: UQ_Engine: Could not import core project modules: {e}. UQ Engine will not be functional.")
    OptimizationEngine = Any  # type: ignore
    ReservoirData, PVTProperties, EORParameters, EconomicParameters = Any, Any, Any, Any
    OperationalParameters, ProfileParameters, EOSModelParameters = Any, Any, Any
    recovery_factor = None  # type: ignore
    class DummyConfigManagerForUQRobust:
        _is_loaded_from_file = False
        def get(self, key, default=None): return default if default is not None else {} if isinstance(key, str) and key.endswith("Settings") else None
        def get_section(self, key): return {}
    config_manager = DummyConfigManagerForUQRobust()  # type: ignore

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class UncertaintyQuantificationEngine:
    def __init__(self,
                 base_engine_for_config: 'OptimizationEngine',
                 uq_settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the UncertaintyQuantificationEngine.

        Args:
            base_engine_for_config: An instance of OptimizationEngine. This engine provides
                                    the baseline configuration and non-varying data.
            uq_settings: Dictionary of UQ settings. If None, loads from config_manager.
        """
        self.base_engine_for_config = base_engine_for_config
        if uq_settings is None:
            self.uq_settings = config_manager.get_section("UncertaintyQuantificationSettings") or {}
        else:
            self.uq_settings = uq_settings

        if not self.uq_settings.get("enabled", False):
            logger.info("UQ Engine initialized, but UQ is disabled in settings.")
            self._is_enabled = False
            return
        self._is_enabled = True

        self.param_definitions: List[Dict[str, Any]] = self.uq_settings.get("parameters", [])
        if not self.param_definitions:
            logger.warning("UQ is enabled but no uncertain parameters are defined in settings.")
            self._is_enabled = False
            return
            
        self.distributions_map: Dict[str, chaospy.Distribution] = {}
        self.param_names_in_dist_order: List[str] = []
        marginal_distributions_list = []

        for param_def in self.param_definitions:
            full_name = param_def["name"]
            dist_type = param_def["distribution"]
            dist_params = param_def["params"]
            param_def.setdefault('scope', self._infer_scope_from_name(full_name))
            param_def.setdefault('internal_name', self._infer_internal_name(full_name))

            try:
                dist = self._get_chaospy_distribution(dist_type, dist_params)
                self.distributions_map[full_name] = dist
                marginal_distributions_list.append(dist)
                self.param_names_in_dist_order.append(full_name)
            except Exception as e:
                logger.error(f"Error creating distribution for parameter '{full_name}': {e}. Skipping.")
        
        if not marginal_distributions_list:
            logger.error("No valid marginal distributions created. UQ Engine cannot proceed.")
            self._is_enabled = False
            return

        correlation_settings = self.uq_settings.get("correlations")
        if correlation_settings and isinstance(correlation_settings, dict) and \
           'matrix' in correlation_settings and 'param_order' in correlation_settings:
            try:
                corr_matrix = np.array(correlation_settings['matrix'])
                if corr_matrix.shape != (len(marginal_distributions_list), len(marginal_distributions_list)):
                    logger.error("Correlation matrix dimensions do not match number of uncertain parameters. Ignoring correlations.")
                    self.joint_distribution = chaospy.J(*marginal_distributions_list)
                else:
                    self.joint_distribution = chaospy.Nataf(chaospy.J(*marginal_distributions_list), corr_matrix) # type: ignore
                    logger.info("Created joint distribution with Nataf transform for correlations.")
            except Exception as e_corr:
                logger.error(f"Error applying correlations: {e_corr}. Using independent distributions.")
                self.joint_distribution = chaospy.J(*marginal_distributions_list)
        else:
            self.joint_distribution = chaospy.J(*marginal_distributions_list)
            logger.info("Created joint distribution assuming independent parameters.")

        logger.info(f"UQ Engine initialized with {len(self.distributions_map)} uncertain parameters.")

    def _infer_scope_from_name(self, full_name: str) -> str:
        if '.' in full_name: return full_name.split('.', 1)[0]
        if hasattr(self.base_engine_for_config.economic_params, full_name): return 'economic'
        if full_name == 'avg_porosity': return 'reservoir'
        if full_name == 'mmp_value': return 'fluid'
        return 'unknown'

    def _infer_internal_name(self, full_name: str) -> str:
        if '.' in full_name: return full_name.split('.', 1)[1]
        return full_name

    def _get_chaospy_distribution(self, dist_name: str, params: List[Any]) -> chaospy.Distribution:
        dist_name_lower = dist_name.lower()
        cp_dist_map = {
            "normal": chaospy.Normal,
            "uniform": chaospy.Uniform,
            "lognormal": chaospy.LogNormal,
            "triangular": chaospy.Triangle,
            "beta": chaospy.Beta,
        }
        constructor = cp_dist_map.get(dist_name_lower)
        if constructor:
            try:
                return constructor(*params)
            except TypeError as e:
                raise ValueError(f"Incorrect parameters for chaospy.{dist_name}: {params}. Error: {e}")
        else:
            raise ValueError(f"Unsupported chaospy distribution type: {dist_name}")

    def _generate_samples(self, num_samples: int, sampling_rule: str = "latin_hypercube") -> Optional[np.ndarray]:
        if not self._is_enabled or self.joint_distribution is None:
            logger.error("UQ Engine not enabled or no joint distribution. Cannot generate samples.")
            return None
        try:
            return self.joint_distribution.sample(size=num_samples, rule=sampling_rule)
        except Exception as e:
            logger.error(f"Error generating samples using chaospy: {e}", exc_info=True)
            return None

    def _param_def_by_full_name(self, full_name: str) -> Optional[Dict[str, Any]]:
        return next((pd for pd in self.param_definitions if pd["name"] == full_name), None)

    def run_mc_for_fixed_strategy(self,
                                  fixed_eor_params_dict: Dict[str, float],
                                  target_objectives: List[str] = ['npv', 'recovery_factor'],
                                  num_samples: Optional[int] = None) -> Optional[pd.DataFrame]:
        if not self._is_enabled: return None
        n_samples = num_samples or self.uq_settings.get("num_samples_mc", 1000)
        sampling_rule = self.uq_settings.get("mc_sampling_rule", "latin_hypercube")
        logger.info(f"Running MC (Fixed Strategy) with {n_samples} samples ({sampling_rule}) for obj: {target_objectives}.")

        samples_array = self._generate_samples(n_samples, sampling_rule)
        if samples_array is None: return None

        results_list = []
        base_econ_params = deepcopy(self.base_engine_for_config.economic_params)

        for i in range(n_samples):
            sampled_params_this_run = {name: val for name, val in zip(self.param_names_in_dist_order, samples_array[:, i])}
            
            # Use the engine's public evaluation API for a clean interface
            results = self.base_engine_for_config.evaluate_for_analysis(
                eor_operational_params_dict=fixed_eor_params_dict,
                economic_params_override=self._get_sampled_econ_instance(base_econ_params, sampled_params_this_run),
                avg_porosity_override=sampled_params_this_run.get('reservoir.avg_porosity'),
                mmp_override=sampled_params_this_run.get('fluid.mmp_value'),
                recovery_model_init_kwargs_override=self._get_sampled_model_kwargs(sampled_params_this_run),
                target_objectives=target_objectives
            )

            results_list.append({**sampled_params_this_run, **results})
            if (i + 1) % max(1, n_samples // 20) == 0:
                logger.info(f"MC Fixed Strategy: Processed {i+1}/{n_samples} samples.")
        return pd.DataFrame(results_list)

    def _get_sampled_econ_instance(self, base_instance: 'EconomicParameters', samples: Dict[str, float]) -> 'EconomicParameters':
        instance = deepcopy(base_instance)
        for full_name, value in samples.items():
            param_def = self._param_def_by_full_name(full_name)
            if param_def and param_def['scope'] == 'economic':
                setattr(instance, param_def['internal_name'], value)
        return instance

    def _get_sampled_model_kwargs(self, samples: Dict[str, float]) -> Dict[str, Any]:
        kwargs = {}
        for full_name, value in samples.items():
            param_def = self._param_def_by_full_name(full_name)
            if param_def and param_def['scope'] == 'model':
                kwargs[param_def['internal_name']] = value
        return kwargs

    def run_mc_for_optimization_under_uncertainty(self,
                                                optimization_method_name: str = "hybrid_optimize",
                                                target_optimized_output_keys: Optional[List[str]] = None,
                                                objectives_at_optimum: Optional[List[str]] = None,
                                                num_samples: Optional[int] = None) -> Optional[pd.DataFrame]:
        if not self._is_enabled: return None
        n_samples = num_samples or self.uq_settings.get("num_samples_mc_reopt", 100)
        sampling_rule = self.uq_settings.get("mc_sampling_rule", "latin_hypercube")
        logger.info(f"Running MC (Opt Under Uncertainty) with {n_samples} samples ({sampling_rule}) using optimizer '{optimization_method_name}'.")

        samples_array = self._generate_samples(n_samples, sampling_rule)
        if samples_array is None: return None

        results_list = []
        if target_optimized_output_keys is None: target_optimized_output_keys = ['pressure', 'rate']
        if objectives_at_optimum is None: objectives_at_optimum = [self.base_engine_for_config.chosen_objective, 'final_recovery_factor_reported']

        # Get pristine base data models for deep copying in the loop
        base_reservoir_ref = self.base_engine_for_config._base_reservoir_data_for_reopt
        base_pvt_ref = self.base_engine_for_config._base_pvt_data_for_reopt
        base_eor_params_ref = self.base_engine_for_config._base_eor_params_for_reopt
        base_econ_params_ref = self.base_engine_for_config._base_economic_params_for_reopt
        base_op_params_ref = self.base_engine_for_config._base_operational_params_for_reopt
        base_profile_params_ref = self.base_engine_for_config._base_profile_params_for_reopt

        for i in range(n_samples):
            sampled_params_this_run = {name: val for name, val in zip(self.param_names_in_dist_order, samples_array[:, i])}
            logger.info(f"Opt Under UQ - Sample {i+1}/{n_samples}: Sampled Params { {k: f'{v:.3f}' for k,v in sampled_params_this_run.items()} }")

            # Create temporary data model instances for this run
            current_reservoir = deepcopy(base_reservoir_ref)
            current_pvt = deepcopy(base_pvt_ref)
            current_econ_params = deepcopy(base_econ_params_ref)
            
            # --- Apply sampled values to the copies ---
            engine_init_overrides = {}
            for full_name, value in sampled_params_this_run.items():
                param_def = self._param_def_by_full_name(full_name)
                if not param_def: continue
                scope, internal_name = param_def['scope'], param_def['internal_name']

                if scope == 'economic':
                    setattr(current_econ_params, internal_name, value)
                elif scope == 'reservoir':
                    if internal_name == 'avg_porosity': engine_init_overrides['avg_porosity_init_override'] = value
                    elif internal_name == 'ooip_stb': current_reservoir.ooip_stb = value
                elif scope == 'fluid':
                    if internal_name == 'mmp_value': engine_init_overrides['mmp_init_override'] = value
                elif scope == 'model':
                    if 'recovery_model_init_kwargs_override' not in engine_init_overrides:
                        engine_init_overrides['recovery_model_init_kwargs_override'] = {}
                    engine_init_overrides['recovery_model_init_kwargs_override'][internal_name] = value
                elif scope == 'eos' and current_reservoir.eos_model:
                    self._apply_eos_param_override(current_reservoir.eos_model, internal_name, value)

            try:
                temp_engine = OptimizationEngine(
                    reservoir=current_reservoir, pvt=current_pvt,
                    eor_params_instance=deepcopy(base_eor_params_ref),
                    economic_params_instance=current_econ_params,
                    operational_params_instance=deepcopy(base_op_params_ref),
                    profile_params_instance=deepcopy(base_profile_params_ref),
                    **engine_init_overrides
                )
                
                opt_func: Callable = getattr(temp_engine, optimization_method_name)
                opt_results = opt_func()

                result_row = {**sampled_params_this_run}
                if 'optimized_params_final_clipped' in opt_results:
                    for key in target_optimized_output_keys:
                        result_row[f"opt_{key}"] = opt_results['optimized_params_final_clipped'].get(key)
                for obj_key in objectives_at_optimum:
                    result_row[obj_key] = opt_results.get(obj_key)
                results_list.append(result_row)
            except Exception as e_reopt:
                logger.error(f"Error during re-optimization (sample {i}): {e_reopt}", exc_info=True)
                results_list.append({**sampled_params_this_run, 'error': str(e_reopt)})
            
            if (i + 1) % max(1, n_samples // 10) == 0:
                logger.info(f"MC Opt Under UQ: Processed {i+1}/{n_samples} re-optimizations.")

        return pd.DataFrame(results_list)

    def _apply_eos_param_override(self, eos_params_instance: 'EOSModelParameters', internal_name_path: str, value: float):
        """Applies a sampled value to a parameter within the EOSModelParameters numpy array."""
        try:
            parts = internal_name_path.split('.')
            # Example: 'component_properties.C7+.MW'
            if parts[0] == 'component_properties' and len(parts) == 3:
                comp_name, prop_key = parts[1], parts[2]
                comp_array = eos_params_instance.component_properties
                
                # Find row index for the component name
                row_idx_list = np.where(comp_array[:, 0] == comp_name)[0]
                if not row_idx_list.size:
                    logger.warning(f"EOS override: Component '{comp_name}' not found.")
                    return
                row_idx = row_idx_list[0]
                
                # Find column index for the property key
                prop_map = {'Name': 0, 'zi': 1, 'MW': 2, 'Tc': 3, 'Pc': 4, 'omega': 5, 's_V': 6}
                if prop_key not in prop_map:
                    logger.warning(f"EOS override: Property '{prop_key}' not recognized.")
                    return
                col_idx = prop_map[prop_key]
                
                comp_array[row_idx, col_idx] = value
                logger.debug(f"Applied EOS override: {comp_name}.{prop_key} set to {value}")

            # Example: 'binary_interaction_coeffs.0.1' for kij between components 0 and 1
            elif parts[0] == 'binary_interaction_coeffs' and len(parts) == 3:
                row, col = int(parts[1]), int(parts[2])
                eos_params_instance.binary_interaction_coeffs[row, col] = value
                eos_params_instance.binary_interaction_coeffs[col, row] = value # Ensure symmetry
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to apply EOS override for '{internal_name_path}': {e}")


    def analyze_uq_results(self, uq_results_df: pd.DataFrame, objective_col: str) -> Optional[Dict[str, Any]]:
        if uq_results_df.empty or objective_col not in uq_results_df.columns:
            logger.warning(f"DataFrame empty or objective '{objective_col}' not found for UQ analysis.")
            return None
        
        data = uq_results_df[objective_col].dropna()
        if data.empty:
            logger.warning(f"No valid data for objective '{objective_col}' after dropna.")
            return None

        stats = {
            'mean': data.mean(), 'std_dev': data.std(), 'min': data.min(), 'max': data.max(),
            'P10': data.quantile(0.10), 'P50': data.quantile(0.50), 'P90': data.quantile(0.90),
            'count': data.count()
        }
        logger.info(f"UQ Stats for '{objective_col}': Mean={stats['mean']:.3e}, StdDev={stats['std_dev']:.3e}, "
                    f"P10={stats['P10']:.3e}, P50={stats['P50']:.3e}, P90={stats['P90']:.3e}")
        return stats

    def plot_uq_distribution(self, uq_results_df: pd.DataFrame, objective_col: str,
                               title: Optional[str] = None) -> Optional[go.Figure]:
        if uq_results_df.empty or objective_col not in uq_results_df.columns:
            logger.warning("Cannot plot UQ distribution: DataFrame empty or objective column missing.")
            return None
        
        data = uq_results_df[objective_col].dropna()
        if data.empty: return None

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram (PDF)", "Cumulative Distribution (CDF)"))
        fig.add_trace(go.Histogram(x=data, name='PDF', histnorm='probability density'), row=1, col=1)
        sorted_data = np.sort(data)
        y_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig.add_trace(go.Scatter(x=sorted_data, y=y_cdf, name='CDF', mode='lines'), row=1, col=2)
        fig.update_layout(title_text=title or f"Uncertainty Distribution for {objective_col.replace('_',' ').title()}", showlegend=False)
        fig.update_xaxes(title_text=objective_col.replace('_',' ').title(), row=1, col=1)
        fig.update_xaxes(title_text=objective_col.replace('_',' ').title(), row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Probability", row=1, col=2)
        return fig

    def run_polynomial_chaos_analysis(self,
                                      fixed_eor_params_dict: Dict[str, float],
                                      target_objective: str = 'npv',
                                      poly_order: Optional[int] = None,
                                      quad_order_offset: int = 1
                                     ) -> Optional[Dict[str, Any]]:
        if not self._is_enabled or self.joint_distribution is None or not self.param_names_in_dist_order:
            logger.warning("UQ/PCE disabled or not configured. Cannot run PCE.")
            return None
        if recovery_factor is None:
            logger.error("recovery_factor function not available. PCE cannot run.")
            return None

        p_order = poly_order if poly_order is not None else self.uq_settings.get("pce_polynomial_order", 2)
        q_rule = self.uq_settings.get("pce_quadrature_rule", "gaussian")
        q_order = p_order + quad_order_offset
        
        logger.info(f"Running PCE Analysis for '{target_objective}': PolyOrder={p_order}, QuadOrder={q_order} ({q_rule}).")

        try:
            poly_basis = chaospy.generate_expansion(p_order, self.joint_distribution)
            nodes_uq, weights_uq = chaospy.generate_quadrature(q_order, self.joint_distribution, rule=q_rule) # type: ignore
            logger.info(f"PCE: Generated {nodes_uq.shape[1]} quadrature nodes.")
        except Exception as e_setup_pce:
            logger.error(f"Error setting up PCE basis/quadrature: {e_setup_pce}", exc_info=True)
            return None

        model_evals_at_nodes = []
        base_econ_params = deepcopy(self.base_engine_for_config.economic_params)
        
        for i in range(nodes_uq.shape[1]):
            sampled_params_for_node = {name: val for name, val in zip(self.param_names_in_dist_order, nodes_uq[:, i])}
            
            # Use the evaluation API, similar to the MC method
            results_at_node = self.base_engine_for_config.evaluate_for_analysis(
                eor_operational_params_dict=fixed_eor_params_dict,
                economic_params_override=self._get_sampled_econ_instance(base_econ_params, sampled_params_for_node),
                avg_porosity_override=sampled_params_for_node.get('reservoir.avg_porosity'),
                mmp_override=sampled_params_for_node.get('fluid.mmp_value'),
                recovery_model_init_kwargs_override=self._get_sampled_model_kwargs(sampled_params_for_node),
                target_objectives=[target_objective]
            )
            model_evals_at_nodes.append(results_at_node.get(target_objective, np.nan))

        model_evals_arr = np.array(model_evals_at_nodes)
        valid_mask = ~np.isnan(model_evals_arr)
        if not np.any(valid_mask):
            logger.error("PCE: All model evaluations resulted in NaN. Cannot fit PCE.")
            return None
        
        try:
            poly_chaos_expansion = chaospy.fit_quadrature(
                poly_basis, nodes_uq[:, valid_mask], weights_uq[valid_mask], model_evals_arr[valid_mask]
            )
        except Exception as e_fit_pce:
            logger.error(f"Failed to fit PCE model: {e_fit_pce}. Check for NaNs or ill-conditioning.", exc_info=True)
            return None

        pce_results: Dict[str, Any] = {"target_objective": target_objective, "poly_order": p_order, "num_quad_nodes_used": np.sum(valid_mask)}
        try:
            pce_results["mean_pce"] = chaospy.E(poly_chaos_expansion, self.joint_distribution)
            pce_results["variance_pce"] = chaospy.Var(poly_chaos_expansion, self.joint_distribution)
            pce_results["std_dev_pce"] = np.sqrt(pce_results["variance_pce"])
            pce_results["sobol_main_effects"] = {name: val for name, val in zip(self.param_names_in_dist_order, chaospy.Sens_m(poly_chaos_expansion, self.joint_distribution))}
            pce_results["sobol_total_effects"] = {name: val for name, val in zip(self.param_names_in_dist_order, chaospy.Sens_t(poly_chaos_expansion, self.joint_distribution))}
            pce_results["poly_chaos_expansion_coeffs"] = poly_chaos_expansion.coefficients.tolist()
            logger.info(f"PCE Analysis for '{target_objective}': Mean={pce_results['mean_pce']:.3e}, StdDev={pce_results['std_dev_pce']:.3e}")
        except Exception as e_post_pce:
            logger.error(f"Error during PCE post-processing (stats, Sobol): {e_post_pce}", exc_info=True)
        
        return pce_results