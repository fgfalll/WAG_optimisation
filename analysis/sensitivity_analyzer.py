import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Union, Any, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict, fields

from core.optimisation_engine import OptimizationEngine
from core.data_models import (
    EconomicParameters,
    EORParameters,
    OperationalParameters,
    ProfileParameters,
    EOSModelParameters,
)
from core.simulation.recovery_models import recovery_factor
from config_manager import ConfigManager
from core.unified_engine.physics.eos import CubicEOS

EOS_MODELS_AVAILABLE = True

# Optional import for Sobol analysis
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logging.warning("SALib library not found. Sobol sensitivity analysis will be unavailable.")


logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    def __init__(self, engine: "OptimizationEngine"):
        from core.optimisation_engine import OptimizationEngine

        """
        Initializes the analyzer with an OptimizationEngine instance.

        It intelligently determines the base parameters for sensitivity analysis:
        1. If the engine has optimization results, it uses the optimized parameters.
        2. Otherwise, it uses the engine's current default parameters.
        """
        if not isinstance(engine, OptimizationEngine):
            raise TypeError(
                f"SensitivityAnalyzer requires a valid OptimizationEngine instance, but got {type(engine)}"
            )
        self.engine = engine

        # Progress callback for reporting progress to UI
        self._progress_callback: Optional[Callable[[int, str], None]] = None

        # Initialize config manager for configurable parameters
        try:
            self.config_manager = ConfigManager(config_dir_path="config", require_config=False)
            if not self.config_manager.is_loaded:
                self.config_manager.load_configs_from_directory("config")
        except Exception as e:
            logger.warning(f"Failed to initialize ConfigManager: {e}. Using default values.")
            self.config_manager = None

        # Store deepcopies of all base data models for pristine state restoration
        self._base_reservoir_data_for_reopt = deepcopy(self.engine.reservoir)
        self._base_pvt_for_reopt = deepcopy(self.engine.pvt)
        self._base_eor_params_for_reopt = deepcopy(self.engine.eor_params)
        self._base_econ_params_for_reopt = deepcopy(self.engine.economic_params)
        self._base_op_params_for_reopt = deepcopy(self.engine.operational_params)
        self._base_profile_params_for_reopt = deepcopy(self.engine.profile_params)
        self._base_advanced_engine_params_for_reopt = deepcopy(self.engine.advanced_engine_params)
        self._base_eos_params_for_reopt: Optional[EOSModelParameters] = (
            deepcopy(self.engine.reservoir.eos_model) if self.engine.reservoir.eos_model else None
        )

        # Base parameters are now resolved on-demand in each `run_*` method
        # to ensure they reflect the latest state of the engine.

        self.sensitivity_run_data: List[Dict[str, Any]] = []
        self.report_runs: List[Dict[str, Any]] = []
        self.last_objective: Optional[str] = None
        self._eos_cache = {}
        self.results: Optional[Dict[str, Any]] = None

    def set_progress_callback(self, callback: Optional[Callable[[int, str], None]]) -> None:
        """
        Set a callback function for reporting progress during analysis.

        Args:
            callback: A callable that takes (progress_percent, message) arguments.
                      Set to None to disable progress reporting.
        """
        self._progress_callback = callback

    def _report_progress(self, percent: int, message: str) -> None:
        """Internal method to report progress if a callback is set."""
        if self._progress_callback is not None:
            self._progress_callback(percent, message)

    def get_configurable_parameters(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Returns a dictionary of parameters that can be varied in sensitivity analysis,
        structured for UI consumption. This includes optimized operational parameters,
        key economic inputs, and important reservoir/model properties.
        """
        # Optimized parameters are implicitly included as they are part of the base case
        # that gets varied. The user selects them from the UI which is populated by this.

        optimized_params = [
            ("eor.pressure", "Injection Pressure (psi)"),
            ("eor.rate", "Injection Rate (bpd)"),
        ]
        if self.engine._base_eor_params.injection_scheme == "wag":
            optimized_params.append(("eor.WAG_ratio", "WAG Ratio"))

        input_params = {
            "Economic": [
                ("econ.oil_price_usd_per_bbl", "Oil Price ($/bbl)"),
                ("econ.co2_purchase_cost_usd_per_tonne", "CO2 Purchase Cost ($/tonne)"),
                ("econ.co2_recycle_cost_usd_per_tonne", "CO2 Recycle Cost ($/tonne)"),
                ("econ.discount_rate_fraction", "Discount Rate (fraction)"),
                ("econ.carbon_tax_usd_per_tonne", "Carbon Tax ($/tonne)"),
            ],
            "Reservoir / Fluid": [
                ("reservoir.avg_porosity", "Average Porosity (fraction)"),
                ("reservoir.geostatistical_params", "Geostatistical Parameters"),
                ("fluid.mmp_value", "MMP (psi)"),
                ("reservoir.ooip_stb", "Original Oil In Place (STB)"),
                ("fluid.co2_solubility_scm_per_bbl", "CO2 Solubility (scm/bbl)"),
            ],
            "Recovery Model": [
                ("model.v_dp_coefficient", "V_DP Coefficient (Koval)"),
                ("model.mobility_ratio", "Mobility Ratio"),
                ("model.gravity_factor", "Gravity Factor (Miscible)"),
                ("model.sor", "Residual Oil Saturation (Sor)"),
                ("model.kv_kh_ratio", "Kv/Kh Ratio"),
            ],
            "CO2 Storage": [
                ("co2_storage.storage_capacity_tonne", "Storage Capacity (tonne)"),
                ("co2_storage.leakage_rate_fraction", "Leakage Rate (fraction)"),
                (
                    "co2_storage.monitoring_cost_usd_per_tonne",
                    "Monitoring Cost ($/tonne)",
                ),
            ],
        }

        # Combine optimized and input parameters for the UI
        all_params = {"Optimized Parameters": optimized_params, **input_params}
        return all_params

    def _resolve_base_case_parameters(
        self,
    ) -> Tuple[Dict[str, float], "EconomicParameters"]:
        """
        Determines the base EOR operational parameters and economic parameters to use
        for sensitivity runs where re-optimization is NOT performed.
        It prioritizes the engine's stored optimization results if they exist,
        otherwise it derives from the engine's defaults.
        """
        base_eor_dict = {}
        if self.engine.results and "optimized_params_final_clipped" in self.engine.results:
            logger.info("SA: Using EOR parameters from engine's optimization results as base case.")
            base_eor_dict.update(self.engine.results["optimized_params_final_clipped"])
        else:
            logger.info("SA: Deriving base EOR parameters from engine's default settings.")
            ga_bounds = self.engine._get_parameter_bounds()
            eor_dc = self.engine._base_eor_params
            for param, (low, high) in ga_bounds.items():
                if hasattr(eor_dc, param):
                    base_val = getattr(eor_dc, param)
                    base_eor_dict[param] = np.clip(base_val, low, high)
                else:
                    base_eor_dict[param] = (low + high) / 2.0

        base_econ_instance = deepcopy(self.engine._base_economic_params)
        logger.info(
            f"SA: Resolved base EOR params: { {k: f'{v:.2f}' for k, v in base_eor_dict.items() if isinstance(v, (float, int))} } "
        )
        return base_eor_dict, base_econ_instance

    def _parse_param_path(self, param_path: str) -> Tuple[Optional[str], str]:
        if "." in param_path:
            parts = param_path.split(".", 1)
            return parts[0].lower(), parts[1]
        return None, param_path

    def _get_config_value(self, key_path: str, default: Any) -> Any:
        if self.config_manager and hasattr(self.config_manager, "get"):
            try:
                return self.config_manager.get(key_path, default)
            except Exception as e:
                logger.warning(
                    f"Error getting config value '{key_path}': {e}. Using default: {default}"
                )
        return default

    def _get_eos_cache_key(self, eos_model: "EOSModelParameters") -> Tuple:
        """Creates a hashable key from an EOSModelParameters instance."""
        key_parts = [eos_model.eos_type]
        for field in fields(eos_model):
            value = getattr(eos_model, field.name)
            if isinstance(value, np.ndarray):
                key_parts.append(value.tobytes())  # More efficient than tuple for hashing
            elif isinstance(value, (list, dict)):
                key_parts.append(str(value))  # Fallback for other unhashable types
            else:
                key_parts.append(value)
        return tuple(key_parts)

    def _cached_eos_calculation(
        self, eos_model: "EOSModelParameters", pressure: float, temperature: float
    ) -> Dict[str, float]:
        """
        Cache EOS calculations to improve performance.

        BUG FIX: The original cache key `id(eos_model)` was incorrect, as it's based on memory
        address, not content. This new implementation uses a key based on the actual
        parameter values within the EOS model, allowing the cache to work as intended.
        """
        cache_key = (self._get_eos_cache_key(eos_model), pressure, temperature)
        if cache_key not in self._eos_cache:
            try:
                eos_type = eos_model.eos_type.lower()
                if eos_type == "peng-robinson":
                    concrete_eos_model = PengRobinsonEOS(eos_model)
                elif eos_type == "soave-redlich-kwong":
                    concrete_eos_model = SoaveRedlichKwongEOS(eos_model)
                else:
                    raise ValueError(f"Unknown EOS type '{eos_model.eos_type}'")

                self._eos_cache[cache_key] = concrete_eos_model.calculate_properties(
                    pressure, temperature
                )
            except Exception as e:
                logger.error(f"Failed to instantiate or calculate with EOS model for caching: {e}")
                self._eos_cache[cache_key] = {}  # Cache failure to avoid re-trying
        return self._eos_cache[cache_key]

    def _apply_parameter_overrides(
        self, param_path: str, value: Any, param_holders: Dict[str, Any]
    ) -> bool:
        """
        Centralized helper to apply a parameter value based on its path.
        This function eliminates code duplication across all `run_*` methods.

        Args:
            param_path: The full path of the parameter (e.g., 'econ.oil_price_usd_per_bbl').
            value: The value to apply.
            param_holders: A dictionary containing the temporary data objects to modify.
                           Expected keys: 'eor', 'econ', 'eos', 'res', 'fluid', 'model'.

        Returns:
            True if applied successfully, False otherwise.
        """
        param_category, param_name = self._parse_param_path(param_path)
        try:
            if param_category == "eor":
                param_holders["eor"][param_name] = value
            elif param_category == "econ":
                setattr(param_holders["econ"], param_name, value)
            elif param_category == "reservoir":
                if param_name == "geostatistical_params":
                    param_holders["res"]["geostatistical_params"] = value
                else:
                    param_holders["res"][param_name] = value
            elif param_category == "fluid":
                param_holders["fluid"][param_name] = value
            elif param_category == "model":
                param_holders["model"][param_name] = value
            elif param_category == "eos":
                if param_holders["eos"] is None:
                    logger.warning(
                        f"Cannot vary EOS parameter '{param_path}': no base EOS model defined."
                    )
                    return False
                path_parts = param_name.split(".")
                if len(path_parts) != 3:
                    raise ValueError(f"EOS path must be 'attribute.row.col', got: {param_name}")
                attr_name, row_str, col_str = path_parts
                target_attr = getattr(param_holders["eos"], attr_name)
                row_idx, col_idx = int(row_str), int(col_str)
                target_attr[row_idx, col_idx] = value
            else:
                logger.warning(
                    f"Unknown parameter category '{param_category}' for path '{param_path}'. Skipping."
                )
                return False
            return True
        except (AttributeError, KeyError, ValueError, IndexError) as e:
            logger.error(f"Error applying sensitivity value for '{param_path}' = {value}: {e}")
            return False

    def _evaluate_scenario(
        self,
        current_eor_params_dict: Dict[str, float],
        current_econ_params_instance: "EconomicParameters",
        objectives_to_calc: List[str],
        temp_eos_params_override: Optional["EOSModelParameters"] = None,
        temp_reservoir_overrides: Optional[Dict[str, Any]] = None,
        temp_fluid_overrides: Optional[Dict[str, Any]] = None,
        temp_model_param_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Core evaluation function for a single sensitivity analysis point."""
        if recovery_factor is None:
            raise ImportError("The 'recovery_factor' function is not available.")

        eos_model_to_use = (
            temp_eos_params_override
            if temp_eos_params_override
            else self.engine.reservoir.eos_model
        )

        pressure_at_inj = current_eor_params_dict["pressure"]
        if eos_model_to_use:
            try:
                # Use the corrected caching mechanism
                self._cached_eos_calculation(
                    eos_model_to_use, pressure_at_inj, self.engine.pvt.temperature
                )
            except Exception as e_eos_calc:
                logger.warning(
                    f"EOS calculation failed at P={pressure_at_inj:.2f} psi: {e_eos_calc}"
                )

        mmp_to_use = (
            temp_fluid_overrides.get("mmp_value", self.engine.mmp)
            if temp_fluid_overrides
            else self.engine.mmp
        )
        mmp_to_use = mmp_to_use or self._get_config_value(
            "GeneralFallbacks.mmp_default_psi", 2500.0
        )

        porosity_to_use = (
            temp_reservoir_overrides.get("avg_porosity", self.engine.avg_porosity)
            if temp_reservoir_overrides
            else self.engine.avg_porosity
        )

        if temp_reservoir_overrides and "geostatistical_params" in temp_reservoir_overrides:
            geostatistical_params = temp_reservoir_overrides["geostatistical_params"]
            if isinstance(geostatistical_params, dict):
                from core.geology.geostatistical_modeling import (
                    create_geostatistical_grid,
                )

                grid = create_geostatistical_grid((50, 50), geostatistical_params)
                self.engine.reservoir.geostatistical_grid = grid

        return self.engine.evaluate_for_analysis(
            eor_operational_params_dict=current_eor_params_dict,
            economic_params_override=current_econ_params_instance,
            avg_porosity_override=porosity_to_use,
            mmp_override=mmp_to_use,
            recovery_model_init_kwargs_override=temp_model_param_overrides,
            target_objectives=objectives_to_calc,
            eos_model_override=eos_model_to_use,
            dimensional_tolerance=0.5, # Allow wider range for sensitivity analysis
        )

    def run_one_way_sensitivity(
        self,
        param_paths: List[str],
        variation_str: str,
        num_steps: Optional[int] = None,
        objectives: List[str] = ["npv", "recovery_factor"],
    ) -> pd.DataFrame:
        """Runs a one-way sensitivity analysis on a list of parameters."""
        variation_str = variation_str.strip()
        if not variation_str:
            raise ValueError("Variation string cannot be empty.")

        default_num_steps = self._get_config_value("GeneralFallbacks.sensitivity_plot_points", 25)
        num_steps = num_steps if num_steps is not None else default_num_steps

        try:
            values = [float(v.strip()) for v in variation_str.split(",") if v.strip()]
            if not values:
                raise ValueError("No valid numeric values found in variation string.")

            if len(values) == 1:
                center_val = values[0]
                lower_mult = self._get_config_value(
                    "SensitivityAnalysis.single_value_lower_multiplier", 0.8
                )
                upper_mult = self._get_config_value(
                    "SensitivityAnalysis.single_value_upper_multiplier", 1.2
                )
                min_v, max_v = (
                    (center_val * lower_mult, center_val * upper_mult)
                    if center_val != 0
                    else (
                        self._get_config_value(
                            "SensitivityAnalysis.zero_center_fallback_min", -1.0
                        ),
                        self._get_config_value("SensitivityAnalysis.zero_center_fallback_max", 1.0),
                    )
                )
                variation_spec = np.linspace(min_v, max_v, num_steps)
            elif len(values) == 2:
                variation_spec = np.linspace(min(values), max(values), num_steps)
            else:
                variation_spec = sorted(values)
        except ValueError as e:
            raise ValueError(
                f"Invalid numeric format in variation string: '{variation_str}'."
            ) from e

        base_eor_params_dict, base_econ_params_instance = self._resolve_base_case_parameters()

        all_sensitivity_points = []
        base_case_objectives = self._evaluate_scenario(
            base_eor_params_dict, base_econ_params_instance, objectives
        )
        all_sensitivity_points.append(
            {
                "parameter_varied": "BaseCase",
                "parameter_value": "Base",
                **base_case_objectives,
            }
        )

        for param_path in param_paths:
            logger.info(f"Running one-way sensitivity for: {param_path}")
            for test_value in variation_spec:
                temp_eor_dict = base_eor_params_dict.copy()
                temp_econ_instance = deepcopy(base_econ_params_instance)
                temp_eos_override = deepcopy(self._base_eos_params_for_reopt)
                temp_res_over, temp_fluid_over, temp_model_over = {}, {}, {}

                param_holders = {
                    "eor": temp_eor_dict,
                    "econ": temp_econ_instance,
                    "eos": temp_eos_override,
                    "res": temp_res_over,
                    "fluid": temp_fluid_over,
                    "model": temp_model_over,
                }

                if self._apply_parameter_overrides(param_path, test_value, param_holders):
                    evaluated_objectives = self._evaluate_scenario(
                        temp_eor_dict,
                        temp_econ_instance,
                        objectives,
                        temp_eos_override,
                        temp_res_over,
                        temp_fluid_over,
                        temp_model_over,
                    )
                    all_sensitivity_points.append(
                        {
                            "parameter_varied": param_path,
                            "parameter_value": test_value,
                            **evaluated_objectives,
                        }
                    )

        self.sensitivity_run_data = all_sensitivity_points
        self.last_objective = objectives[0] if objectives else "npv"
        return pd.DataFrame(all_sensitivity_points)

    def run_sobol(
        self,
        problem_def: Dict,
        num_samples: int = 1024,
        objectives: List[str] = ["npv"],
    ) -> Dict[str, Any]:
        """Performs Sobol variance-based sensitivity analysis."""
        if not SALIB_AVAILABLE:
            raise ImportError(
                "The 'SALib' library is required for Sobol analysis. Please install it."
            )

        param_values = saltelli.sample(problem_def, num_samples, calc_second_order=True)
        total_runs = len(param_values)
        logger.info(
            f"Starting Sobol analysis with {num_samples} samples ({total_runs} total evaluations)."
        )
        results_array = np.zeros((total_runs, len(objectives)))

        base_eor_params_dict, base_econ_params_instance = self._resolve_base_case_parameters()

        def evaluate_sample(
            sample_idx: int, sample_params: np.ndarray
        ) -> Tuple[int, Dict[str, float]]:
            eor_params_dict = base_eor_params_dict.copy()
            econ_params_inst = deepcopy(base_econ_params_instance)
            temp_eos_override = deepcopy(self._base_eos_params_for_reopt)
            temp_res_over, temp_fluid_over, temp_model_over = {}, {}, {}

            param_holders = {
                "eor": eor_params_dict,
                "econ": econ_params_inst,
                "eos": temp_eos_override,
                "res": temp_res_over,
                "fluid": temp_fluid_over,
                "model": temp_model_over,
            }

            for i, name in enumerate(problem_def["names"]):
                self._apply_parameter_overrides(name, sample_params[i], param_holders)

            return sample_idx, self._evaluate_scenario(
                eor_params_dict,
                econ_params_inst,
                objectives,
                temp_eos_override,
                temp_res_over,
                temp_fluid_over,
                temp_model_over,
            )

        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(evaluate_sample, idx, params): idx
                for idx, params in enumerate(param_values)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, result_dict = future.result()
                    for i, obj_name in enumerate(objectives):
                        results_array[idx, i] = result_dict.get(obj_name, np.nan)
                except Exception as exc:
                    logger.error(f"Sobol sample {idx} generated an exception: {exc}")
                    results_array[idx, :] = np.nan

        final_analysis = {}
        for i, obj_name in enumerate(objectives):
            Y = results_array[:, i]
            if np.isnan(Y).all():
                logger.warning(
                    f"All results for objective '{obj_name}' were NaN. Skipping Sobol analysis."
                )
                continue
            Si = sobol.analyze(problem_def, Y, print_to_console=False, calc_second_order=True)
            final_analysis[obj_name] = Si

        logger.info("Sobol analysis complete.")
        return final_analysis

    def run_reoptimization_sensitivity(
        primary_param_to_vary: str,
        variation_values_str: str,
        optimization_method_on_engine: str = "hybrid_optimize",
        target_optimized_output_keys: Optional[List[str]] = None,
        objectives_at_optimum: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Runs sensitivity by re-running the full optimization for each parameter value.
        PERFORMANCE FIX: This method no longer re-instantiates the expensive OptimizationEngine
        in a loop. It creates it once and modifies its parameters for each run.
        """
        try:
            variation_values = [float(v.strip()) for v in variation_values_str.split(",")]
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid format for re-optimization values: {e}")

        logger.info(
            f"Starting Re-optimization Sensitivity for '{primary_param_to_vary}' using '{optimization_method_on_engine}'."
        )
        results_list = []
        target_keys = target_optimized_output_keys or ["pressure", "rate"]
        obj_keys = objectives_at_optimum or [
            self.engine.chosen_objective,
            "final_recovery_factor_reported",
        ]

        # Instantiate the engine ONCE outside the loop for huge performance gain
        temp_engine = OptimizationEngine(
            reservoir=deepcopy(self._base_reservoir_data_for_reopt),
            pvt=deepcopy(self._base_pvt_for_reopt),
            eor_params_instance=deepcopy(self._base_eor_params_for_reopt),
            economic_params_instance=deepcopy(self._base_econ_params_for_reopt),
            operational_params_instance=deepcopy(self._base_op_params_for_reopt),
            profile_params_instance=deepcopy(self._base_profile_params_for_reopt),
            advanced_engine_params_instance=deepcopy(self._base_advanced_engine_params_for_reopt),
        )

        for i, p_value in enumerate(variation_values):
            logger.info(
                f"Re-optimizing (run {i + 1}/{len(variation_values)}) for {primary_param_to_vary} = {p_value}"
            )

            # Reset engine state to pristine base state before applying the change
            temp_engine.reservoir = deepcopy(self._base_reservoir_data_for_reopt)
            temp_engine.economic_params = deepcopy(self._base_econ_params_for_reopt)
            # Add any other stateful components that need resetting

            param_holders = {  # We don't use all of these, but it fits the helper's API
                "econ": temp_engine.economic_params,
                "pvt": temp_engine.pvt,
                "co2_storage": temp_engine.co2_storage_params,
                "res": {},  # Dictionary for reservoir parameter overrides
                "fluid": {},  # Dictionary for fluid parameter overrides
                "model": {},
                "eos": temp_engine.reservoir.eos_model,
                "eor": {},  # Not used in re-opt
            }

            # Apply the specific parameter change for this iteration using the centralized helper
            if not self._apply_parameter_overrides(primary_param_to_vary, p_value, param_holders):
                logger.error(
                    f"Failed to apply override for {primary_param_to_vary}={p_value}. Skipping run."
                )
                results_list.append(
                    {
                        primary_param_to_vary: p_value,
                        "error": "Parameter application failed",
                    }
                )
                continue

            # Manually update engine attributes from the override dictionaries
            if "avg_porosity" in param_holders["res"]:
                temp_engine.avg_porosity_init_override = param_holders["res"]["avg_porosity"]
            if "mmp_value" in param_holders["fluid"]:
                temp_engine.mmp_init_override = param_holders["fluid"]["mmp_value"]
            if param_holders["model"]:
                temp_engine.recovery_model_init_kwargs_override = param_holders["model"]

            temp_engine.re_initialize_dependent_components()  # Ensure engine internals are updated

            try:
                opt_func: Callable = getattr(temp_engine, optimization_method_on_engine)
                opt_results = opt_func()

                result_row: Dict[str, Any] = {primary_param_to_vary: p_value}
                if "optimized_params_final_clipped" in opt_results:
                    for key in target_keys:
                        result_row[f"opt_{key}"] = opt_results[
                            "optimized_params_final_clipped"
                        ].get(key)
                for obj_key in obj_keys:
                    result_row[obj_key] = opt_results.get(obj_key)
                results_list.append(result_row)
            except Exception as e_reopt:
                logger.error(
                    f"Re-optimization failed for {primary_param_to_vary}={p_value}: {e_reopt}",
                    exc_info=True,
                )
                results_list.append({primary_param_to_vary: p_value, "error": str(e_reopt)})

        return pd.DataFrame(results_list)

    def run_two_way_sensitivity(
        param1_path: str,
        param1_values_str: str,
        param2_path: str,
        param2_values_str: str,
        objectives: List[str] = ["npv"],
    ) -> pd.DataFrame:
        """
        Runs a two-way sensitivity analysis.
        ENHANCEMENT: Now supports all parameter categories, not just 'eor' and 'econ'.
        """
        try:
            param1_values = [float(v.strip()) for v in param1_values_str.split(",")]
            param2_values = [float(v.strip()) for v in param2_values_str.split(",")]
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid format for two-way sensitivity values: {e}")

        logger.info(f"Starting Two-Way Sensitivity for '{param1_path}' and '{param2_path}'.")
        results_list = []

        base_eor_params_dict, base_econ_params_instance = self._resolve_base_case_parameters()

        for val1 in param1_values:
            for val2 in param2_values:
                temp_eor_dict = base_eor_params_dict.copy()
                temp_econ_instance = deepcopy(base_econ_params_instance)
                temp_eos_override = deepcopy(self._base_eos_params_for_reopt)
                temp_res_over, temp_fluid_over, temp_model_over = {}, {}, {}

                param_holders = {
                    "eor": temp_eor_dict,
                    "econ": temp_econ_instance,
                    "eos": temp_eos_override,
                    "res": temp_res_over,
                    "fluid": temp_fluid_over,
                    "model": temp_model_over,
                }

                # Apply both parameter changes using the centralized helper
                self._apply_parameter_overrides(param1_path, val1, param_holders)
                self._apply_parameter_overrides(param2_path, val2, param_holders)

                eval_results = self._evaluate_scenario(
                    temp_eor_dict,
                    temp_econ_instance,
                    objectives,
                    temp_eos_override,
                    temp_res_over,
                    temp_fluid_over,
                    temp_model_over,
                )
                results_list.append({param1_path: val1, param2_path: val2, **eval_results})

        return pd.DataFrame(results_list)

    # --- PLOTTING METHODS (Largely unchanged, but benefit from more robust data) ---

    def plot_3d_surface(
        self,
        two_way_df: pd.DataFrame,
        param1_col: str,
        param2_col: str,
        objective_col: str,
        title: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """Creates an interactive 3D surface plot for two-way sensitivity results."""
        if not all(col in two_way_df.columns for col in [param1_col, param2_col, objective_col]):
            logger.error(
                f"Missing required columns for 3D plot. Got: {two_way_df.columns.tolist()}"
            )
            return None

        try:
            pivot_df = two_way_df.pivot(index=param2_col, columns=param1_col, values=objective_col)
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        z=pivot_df.values,
                        colorscale="Viridis",
                        colorbar=dict(title=objective_col.replace("_", " ").title()),
                    )
                ]
            )
            fig.update_layout(
                title=title or f"3D Surface: {objective_col.replace('_', ' ').title()}",
                scene=dict(
                    xaxis_title=param1_col,
                    yaxis_title=param2_col,
                    zaxis_title=objective_col.replace("_", " ").title(),
                ),
                autosize=True,
                margin=dict(l=65, r=50, b=65, t=90),
            )
            return fig
        except Exception as e:
            logger.error(
                f"Error generating 3D surface plot (data might not be on a grid): {e}",
                exc_info=True,
            )
            return None

    def plot_tornado_chart(
        sensitivity_df: pd.DataFrame, objective_col: str, title: Optional[str] = None
    ) -> Optional[go.Figure]:
        """Creates a tornado chart from one-way sensitivity data."""
        if sensitivity_df.empty or objective_col not in sensitivity_df.columns:
            logger.warning(
                "Cannot generate tornado chart: DataFrame is empty or objective column is missing."
            )
            return None

        base_case_row = sensitivity_df[sensitivity_df["parameter_varied"] == "BaseCase"]
        if base_case_row.empty:
            logger.error("BaseCase row not found in data. Tornado chart requires it for reference.")
            return None
        base_objective_value = base_case_row.iloc[0][objective_col]
        if pd.isna(base_objective_value):
            logger.error("BaseCase objective value is NaN. Cannot create tornado chart.")
            return None

        summary_data = []
        results_df = sensitivity_df[sensitivity_df["parameter_varied"] != "BaseCase"].copy()
        for param_path in results_df["parameter_varied"].unique():
            subset = results_df[results_df["parameter_varied"] == param_path]
            if subset[objective_col].isnull().all():
                continue
            min_val = subset[objective_col].min(skipna=True)
            max_val = subset[objective_col].max(skipna=True)
            summary_data.append(
                {
                    "parameter": param_path,
                    "low": min_val,
                    "high": max_val,
                    "range": abs(max_val - min_val),
                }
            )

        if not summary_data:
            return None
        summary_df = pd.DataFrame(summary_data).sort_values(
            by="range", ascending=True
        )  # Ascending for typical tornado look

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=summary_df["parameter"],
                x=summary_df["high"] - base_objective_value,
                base=base_objective_value,
                name="High Impact",
                orientation="h",
                marker_color="mediumseagreen",
            )
        )
        fig.add_trace(
            go.Bar(
                y=summary_df["parameter"],
                x=summary_df["low"] - base_objective_value,
                base=base_objective_value,
                name="Low Impact",
                orientation="h",
                marker_color="lightcoral",
            )
        )
        fig.update_layout(
            barmode="relative",
            title_text=title or f"Tornado Chart for {objective_col.replace('_', ' ').title()}",
            xaxis_title_text=f"Value of {objective_col.replace('_', ' ').title()}",
            yaxis_title_text="Parameter",
            yaxis=dict(autorange="reversed"),
            height=max(400, len(summary_df) * 40),
            legend=dict(x=0.01, y=0.99, borderwidth=1),
        )
        fig.add_vline(
            x=base_objective_value,
            line_width=2,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Base: {base_objective_value:.3e}",
            annotation_position="top right",
        )
        return fig

    def plot_spider_chart(
        sensitivity_df: pd.DataFrame,
        objective_col: str,
        title: Optional[str] = None,
        normalize_radial: bool = False,
    ) -> Optional[go.Figure]:
        """Creates a spider (radar) chart from one-way sensitivity data."""
        if sensitivity_df.empty or objective_col not in sensitivity_df.columns:
            return None
        base_case_row = sensitivity_df[sensitivity_df["parameter_varied"] == "BaseCase"]
        if base_case_row.empty:
            return None
        base_val = base_case_row.iloc[0][objective_col]
        if pd.isna(base_val):
            return None

        results_df = sensitivity_df[sensitivity_df["parameter_varied"] != "BaseCase"].copy()
        params = results_df["parameter_varied"].unique()
        if not params.size:
            return None

        low_vals, high_vals = [], []
        for param in params:
            subset = results_df[results_df["parameter_varied"] == param]
            if subset[objective_col].isnull().all():
                low_vals.append(np.nan)
                high_vals.append(np.nan)
                continue

            min_param_val_idx = subset["parameter_value"].idxmin()
            max_param_val_idx = subset["parameter_value"].idxmax()
            low_vals.append(subset.loc[min_param_val_idx, objective_col])
            high_vals.append(subset.loc[max_param_val_idx, objective_col])

        fig = go.Figure()
        r_low, r_high, r_base = (
            np.array(low_vals),
            np.array(high_vals),
            np.array([base_val] * len(params)),
        )
        axis_title, base_legend = (
            objective_col.replace("_", " ").title(),
            f"Base ({base_val:.3e})",
        )

        if normalize_radial and abs(base_val) > 1e-9:
            r_low, r_high = (
                (r_low - base_val) / base_val * 100,
                (r_high - base_val) / base_val * 100,
            )
            r_base = np.zeros_like(r_base)
            axis_title, base_legend = f"% Change in {axis_title}", "Base (0% Change)"

        fig.add_trace(
            go.Scatterpolar(r=r_low, theta=params, fill="toself", name="Low Setting Impact")
        )
        fig.add_trace(
            go.Scatterpolar(r=r_high, theta=params, fill="toself", name="High Setting Impact")
        )
        fig.add_trace(
            go.Scatterpolar(
                r=r_base,
                theta=params,
                mode="lines",
                name=base_legend,
                line=dict(dash="dash", color="blue"),
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(title_text=axis_title)),
            title=title or f"Spider Chart for {objective_col.replace('_', ' ').title()}",
        )
        return fig
