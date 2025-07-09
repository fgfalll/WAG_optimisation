from typing import Callable, Dict, List, Optional, Any, Tuple, Union, Type
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
from copy import deepcopy

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

try:
    from config_manager import config_manager, ConfigNotLoadedError
except ImportError:
    logging.critical("ConfigManager import failed in OptimizationEngine. Using fallback.")
    class DummyConfigManagerOptim:
        _is_loaded_from_file = False
        def get(self, key_path: str, default: Any = None) -> Any:
            if default is None and not key_path.endswith("Settings") and not key_path.endswith("Defaults"):
                pass
            return default if default is not None else {} if (isinstance(key_path, str) and (key_path.endswith("Settings") or key_path.endswith("Defaults"))) else None
        def get_section(self, section_key: str) -> Optional[Dict[str, Any]]:
             res = self.get(section_key, {}) # type: ignore
             return res if isinstance(res, dict) else {}
        @property
        def is_loaded(self) -> bool: return False
    config_manager = DummyConfigManagerOptim() # type: ignore

from .data_models import (
    ReservoirData, PVTProperties, EORParameters, GeneticAlgorithmParams,
    EconomicParameters, OperationalParameters, ProfileParameters, EOSModelParameters
)
from .recovery_models import recovery_factor
from .eos_models import EOSModel, PengRobinsonEOS, SoaveRedlichKwongEOS

try:
    from evaluation.mmp import calculate_mmp as calculate_mmp_external, MMPParameters
except ImportError:
    logging.info("evaluation.mmp module not found in OptimizationEngine. MMP calculation features may be limited.")
    calculate_mmp_external = None
    MMPParameters = None # type: ignore

DAYS_PER_YEAR = 365.25
logger = logging.getLogger(__name__)

class OptimizationEngine:
    def __init__(self,
                 reservoir: ReservoirData,
                 pvt: PVTProperties,
                 eor_params_instance: Optional[EORParameters] = None,
                 ga_params_instance: Optional[GeneticAlgorithmParams] = None,
                 economic_params_instance: Optional[EconomicParameters] = None,
                 operational_params_instance: Optional[OperationalParameters] = None,
                 profile_params_instance: Optional[ProfileParameters] = None,
                 well_analysis: Optional[Any] = None,
                 avg_porosity_init_override: Optional[float] = None,
                 mmp_init_override: Optional[float] = None,
                 recovery_model_init_kwargs_override: Optional[Dict[str, Any]] = None):

        self._base_reservoir_data_for_reopt = deepcopy(reservoir)
        self._base_pvt_data_for_reopt = deepcopy(pvt)
        
        self.reservoir = reservoir
        self.pvt = pvt
        self.well_analysis = well_analysis

        self.eor_params = eor_params_instance or EORParameters.from_config_dict(
            config_manager.get_section("EORParametersDefaults") or {}
        )
        self._base_eor_params_for_reopt = deepcopy(self.eor_params)

        self.ga_params_default_config = ga_params_instance or GeneticAlgorithmParams.from_config_dict(
            config_manager.get_section("GeneticAlgorithmParamsDefaults") or {}
        )
        self._base_ga_params_for_reopt = deepcopy(self.ga_params_default_config)
        
        self.economic_params = economic_params_instance or EconomicParameters.from_config_dict(
            config_manager.get_section("EconomicParametersDefaults") or {}
        )
        self._base_economic_params_for_reopt = deepcopy(self.economic_params)

        self.operational_params = operational_params_instance or OperationalParameters.from_config_dict(
            config_manager.get_section("OperationalParametersDefaults") or {}
        )
        self._base_operational_params_for_reopt = deepcopy(self.operational_params)

        self.profile_params = profile_params_instance or ProfileParameters.from_config_dict(
            config_manager.get_section("ProfileParametersDefaults") or {}
        )
        self._base_profile_params_for_reopt = deepcopy(self.profile_params)
        
        if self.profile_params.warn_if_defaults_used and profile_params_instance is None:
            logger.warning("Using default ProfileParameters.")

        if self.reservoir.ooip_stb == 1_000_000.0:
            self.reservoir.ooip_stb = config_manager.get("ReservoirDataDefaults.ooip_stb", self.reservoir.ooip_stb) # type: ignore
        
        self._results: Optional[Dict[str, Any]] = None
        
        self._avg_porosity_init_override: Optional[float] = avg_porosity_init_override
        self._mmp_value_init_override: Optional[float] = mmp_init_override
        
        self._mmp_value_runtime_override: Optional[float] = None
        self._avg_porosity_runtime_override: Optional[float] = None
        
        self._mmp_value: Optional[float] = self._mmp_value_init_override
        self._mmp_params_used: Optional[Any] = None

        self.recovery_model: str = config_manager.get("OptimizationEngineSettings.default_recovery_model", "hybrid") # type: ignore
        base_model_cfg_kwargs = config_manager.get_section(f"RecoveryModelKwargsDefaults.{self.recovery_model.capitalize()}") or {} # type: ignore
        self._recovery_model_init_kwargs: Dict[str, Any] = {**base_model_cfg_kwargs, **(recovery_model_init_kwargs_override or {})}

        self.chosen_objective: str = config_manager.get("OptimizationEngineSettings.default_optimization_objective", "recovery_factor").lower() # type: ignore
        self.co2_density_tonne_per_mscf: float = config_manager.get("OptimizationEngineSettings.co2_density_tonne_per_mscf", 0.053) # type: ignore
        self.npv_time_steps_per_year: int = config_manager.get("OptimizationEngineSettings.npv_time_steps_per_year", 1) # type: ignore
        if self.npv_time_steps_per_year != 1: self.npv_time_steps_per_year = 1

        self._mmp_calculator_fn = calculate_mmp_external
        self._MMPParametersDataclass = MMPParameters # type: ignore
        
        # --- NEW: EOS Model Instantiation ---
        self.eos_model_instance: Optional[EOSModel] = None
        if self.reservoir.eos_model and isinstance(self.reservoir.eos_model, EOSModelParameters):
            try:
                eos_type = self.reservoir.eos_model.eos_type.lower()
                if eos_type == 'peng-robinson':
                    self.eos_model_instance = PengRobinsonEOS(self.reservoir.eos_model)
                elif eos_type == 'soave-redlich-kwong':
                    self.eos_model_instance = SoaveRedlichKwongEOS(self.reservoir.eos_model)
                else:
                    logger.warning(f"Unsupported EOS type '{eos_type}' specified in ReservoirData. No EOS model will be used.")

                if self.eos_model_instance:
                    logger.info(f"Successfully instantiated '{self.reservoir.eos_model.eos_type}' model for dynamic property calculation.")
                    self.pvt.pvt_type = 'compositional' # Mark PVT as compositional if EOS is used
            except Exception as e:
                logger.error(f"Failed to instantiate EOS model from ReservoirData: {e}", exc_info=True)
                self.eos_model_instance = None
        # --- END: EOS Model Instantiation ---

        if self._mmp_value is None: self.calculate_mmp()

    @property
    def avg_porosity(self) -> float:
        if hasattr(self, '_avg_porosity_runtime_override') and self._avg_porosity_runtime_override is not None:
            return self._avg_porosity_runtime_override
        if self._avg_porosity_init_override is not None:
            return self._avg_porosity_init_override
        poro_arr = self.reservoir.grid.get('PORO', np.array([0.15]))
        return np.mean(poro_arr) if hasattr(poro_arr, 'size') and poro_arr.size > 0 else 0.15

    @property
    def mmp(self) -> Optional[float]:
        if hasattr(self, '_mmp_value_runtime_override') and self._mmp_value_runtime_override is not None:
            return self._mmp_value_runtime_override
        if self._mmp_value_init_override is not None: # Value provided at construction
             if self._mmp_value is None: self._mmp_value = self._mmp_value_init_override
             return self._mmp_value_init_override
        if self._mmp_value is None: self.calculate_mmp()
        return self._mmp_value

    def calculate_mmp(self, method_override: Optional[str] = None) -> float:
        default_mmp_fallback = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0) # type: ignore
        
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
            return self._mmp_value

        actual_mmp_method = method_override or config_manager.get("OptimizationEngineSettings.mmp_calculation_method", "auto") # type: ignore
        mmp_input_object: Union[PVTProperties, Any] = self.pvt
        source_description = "PVT data"

        if self.well_analysis and hasattr(self.well_analysis, 'get_average_mmp_params_for_engine'):
            try:
                avg_well_params = self.well_analysis.get_average_mmp_params_for_engine()
                mmp_input_constructor_params = {
                    'temperature': avg_well_params.get('temperature', self.pvt.temperature),
                    'oil_gravity': avg_well_params.get('oil_gravity', config_manager.get("GeneralFallbacks.api_gravity_default", 35.0)), # type: ignore
                    'c7_plus_mw': avg_well_params.get('c7_plus_mw'),
                    'injection_gas_composition': avg_well_params.get('injection_gas_composition', config_manager.get_section("GeneralFallbacks.default_injection_gas_composition") or {'CO2': 1.0}), # type: ignore
                    'pvt_data': self.pvt
                }
                mmp_input_object = self._MMPParametersDataclass(**mmp_input_constructor_params)
                source_description = "WellAnalysis average parameters"
            except Exception as e: logger.warning(f"Failed to get MMP params from WellAnalysis: {e}. Using PVT.")
        
        try:
            self._mmp_value = float(self._mmp_calculator_fn(mmp_input_object, method=actual_mmp_method)) # type: ignore
            logger.info(f"MMP calculated: {self._mmp_value:.2f} psi ('{actual_mmp_method}' from {source_description}).")
        except Exception as e:
            logger.error(f"MMP calculation failed ('{actual_mmp_method}', {source_description}): {e}. Using fallback.")
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
        return self._mmp_value

    def evaluate_for_analysis(self,
                              eor_operational_params_dict: Dict[str, float],
                              economic_params_override: Optional[EconomicParameters] = None,
                              avg_porosity_override: Optional[float] = None,
                              mmp_override: Optional[float] = None,
                              recovery_model_init_kwargs_override: Optional[Dict[str, Any]] = None,
                              target_objectives: Optional[List[str]] = None
                             ) -> Dict[str, float]:
        """
        Evaluates objectives for a given set of EOR operational parameters and optional overrides.
        This is a clean API for sensitivity/UQ engines.
        """
        results: Dict[str, float] = {}
        if target_objectives is None: target_objectives = [self.chosen_objective]

        mmp_to_use = mmp_override if mmp_override is not None else self.mmp
        if mmp_to_use is None: mmp_to_use = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0) # type: ignore
            
        porosity_to_use = avg_porosity_override if avg_porosity_override is not None else self.avg_porosity
        current_econ_params = economic_params_override if economic_params_override is not None else self.economic_params
        
        pressure = eor_operational_params_dict['pressure']
        base_injection_rate = eor_operational_params_dict['rate']
        
        # Base recovery model kwargs from engine config and any runtime overrides
        rf_call_kwargs = {**self._recovery_model_init_kwargs, **(recovery_model_init_kwargs_override or {})}

        # --- NEW: EOS Integration for Dynamic Fluid Properties ---
        if self.eos_model_instance:
            try:
                # Get dynamic properties at the current operating pressure
                fluid_props = self.eos_model_instance.calculate_properties(pressure, self.pvt.temperature)
                if fluid_props.get('status') == 'Success':
                    # Map calculated EOS properties to recovery model keyword arguments
                    if 'oil_viscosity_cp' in fluid_props:
                        rf_call_kwargs['mu_oil'] = fluid_props['oil_viscosity_cp']
                    if 'gas_viscosity_cp' in fluid_props:
                        rf_call_kwargs['mu_co2'] = fluid_props['gas_viscosity_cp']
                    # Other properties like density could be passed if models support them
                    logger.debug(f"EOS props at P={pressure:.1f}psi: mu_oil={fluid_props.get('oil_viscosity_cp', 'N/A'):.3f}cp")
                else:
                    logger.warning(f"EOS flash status not 'Success' at P={pressure}psi: {fluid_props.get('status')}. Using fallback properties.")
            except Exception as e_eos:
                logger.warning(f"EOS calculation failed at P={pressure}psi. Using fallback properties. Error: {e_eos}")
        # --- END: EOS Integration ---
        
        # Add EOR operational parameters to the recovery factor call arguments.
        # These will override any same-named keys from EOS or init_kwargs (e.g., mobility_ratio if user varies it).
        rf_call_kwargs.update({
            'v_dp_coefficient': eor_operational_params_dict.get('v_dp_coefficient', self.eor_params.v_dp_coefficient),
            'mobility_ratio': eor_operational_params_dict.get('mobility_ratio', self.eor_params.mobility_ratio)
        })

        effective_co2_rate_for_rf = base_injection_rate
        water_fraction_rf: Optional[float] = None
        if self.eor_params.injection_scheme == 'wag':
            water_fraction_rf = eor_operational_params_dict.get('water_fraction', 0.5)
            effective_co2_rate_for_rf = base_injection_rate * (1.0 - water_fraction_rf)
            if 'cycle_length_days' in eor_operational_params_dict:
                rf_call_kwargs['cycle_length_days'] = eor_operational_params_dict['cycle_length_days']
            rf_call_kwargs['water_fraction'] = water_fraction_rf

        rf_val = recovery_factor(
            pressure, effective_co2_rate_for_rf, porosity_to_use, mmp_to_use,
            model=self.recovery_model, **rf_call_kwargs
        )

        if "recovery_factor" in target_objectives: results["recovery_factor"] = rf_val

        original_econ_params_for_eval = self.economic_params
        self.economic_params = current_econ_params # Set for NPV calc
        try:
            if any(obj in target_objectives for obj in ["npv", "co2_utilization"]):
                annual_oil, co2_inj, water_inj, water_disp = self._calculate_annual_profiles(rf_val, eor_operational_params_dict)
                if "npv" in target_objectives:
                    results["npv"] = self._calculate_npv(annual_oil, co2_inj, water_inj, water_disp)
                if "co2_utilization" in target_objectives:
                    results["co2_utilization"] = self._calculate_co2_utilization_factor(annual_oil, co2_inj)
        finally:
            self.economic_params = original_econ_params_for_eval # Restore

        return results

    def _objective_function_wrapper(self, params_dict: Dict[str, float]) -> float:
        # This internal wrapper now uses the new evaluate_for_analysis method.
        eval_results = self.evaluate_for_analysis(
            eor_operational_params_dict=params_dict,
            target_objectives=[self.chosen_objective]
        )
        if self.chosen_objective == "co2_utilization" and "co2_utilization" in eval_results:
            return -eval_results["co2_utilization"]
        return eval_results.get(self.chosen_objective, -float('inf'))

    def _calculate_total_volume_for_qp(self, q_initial_peak_rate: float, years: int, profile_params: ProfileParameters, profile_type_override: Optional[str] = None) -> float:
        # ... (No changes needed)
        current_profile_type = profile_type_override or profile_params.oil_profile_type; plateau_frac = profile_params.plateau_duration_fraction_of_life
        plateau_years = int(np.floor(years * plateau_frac)) if "plateau" in current_profile_type else 0; plateau_years = max(0, min(plateau_years, years)); decline_years = years - plateau_years
        total_volume_calculated = 0.0
        if plateau_years > 0: total_volume_calculated += q_initial_peak_rate * plateau_years
        if decline_years > 0:
            q_decline_starts_at = q_initial_peak_rate; current_q_at_start_of_decline_year = q_decline_starts_at
            min_econ_rate = (profile_params.min_economic_rate_fraction_of_peak * q_initial_peak_rate) if profile_params.min_economic_rate_fraction_of_peak is not None else 0.0
            if current_profile_type == "plateau_exponential_decline":
                Di_annual = profile_params.initial_decline_rate_annual_fraction or 0.1
                for _ in range(decline_years):
                    if current_q_at_start_of_decline_year < min_econ_rate: break
                    total_volume_calculated += current_q_at_start_of_decline_year; current_q_at_start_of_decline_year *= (1.0 - Di_annual)
            elif current_profile_type == "plateau_hyperbolic_decline":
                Di_nominal, b = profile_params.initial_decline_rate_annual_fraction or 0.1, profile_params.hyperbolic_b_factor or 0.5
                q_i_decline = q_decline_starts_at
                for t_year_in_decline in range(1, decline_years + 1):
                    if current_q_at_start_of_decline_year < min_econ_rate: break
                    t_s, t_e = float(t_year_in_decline - 1), float(t_year_in_decline); vol_this_year = 0.0
                    if b == 0: vol_this_year = (q_i_decline / Di_nominal) * (np.exp(-Di_nominal * t_s) - np.exp(-Di_nominal * t_e)) if Di_nominal > 1e-6 else q_i_decline
                    elif abs(b - 1.0) < 1e-6: vol_this_year = (q_i_decline / Di_nominal) * np.log((1.0 + Di_nominal * t_e) / (1.0 + Di_nominal * t_s)) if Di_nominal > 1e-6 else q_i_decline
                    else:
                        if Di_nominal > 1e-6: term_start,term_end = np.power(1.0+b*Di_nominal*t_s,(1.0-b)/b),np.power(1.0+b*Di_nominal*t_e,(1.0-b)/b); vol_this_year=(q_i_decline/(Di_nominal*(1.0-b)))*(term_start-term_end)
                        else: vol_this_year = q_i_decline
                    total_volume_calculated += vol_this_year
                    current_q_at_start_of_decline_year = q_i_decline / np.power(1.0 + b * Di_nominal * t_e, 1.0/b) if b > 1e-6 else q_i_decline * np.exp(-Di_nominal * t_e)
        return total_volume_calculated

    def _generate_oil_production_profile(self, total_oil_to_produce_stb: float) -> np.ndarray:
        # ... (No changes needed)
        years = self.operational_params.project_lifetime_years; annual_oil_profile = np.zeros(years)
        if total_oil_to_produce_stb <= 1e-9 or years == 0: return annual_oil_profile
        profile_type = self.profile_params.oil_profile_type
        if profile_type == "linear_distribution": annual_oil_profile[:] = total_oil_to_produce_stb / years
        elif profile_type == "plateau_linear_decline":
            plateau_years = int(np.floor(years * self.profile_params.plateau_duration_fraction_of_life)); plateau_years = max(0, min(plateau_years, years)); decline_years = years - plateau_years
            if plateau_years >= years or decline_years <=0 : annual_oil_profile[:] = total_oil_to_produce_stb / years
            else:
                denominator = plateau_years + decline_years / 2.0; plateau_rate = total_oil_to_produce_stb / denominator if denominator > 1e-6 else (total_oil_to_produce_stb / years)
                annual_oil_profile[:plateau_years] = plateau_rate
                if decline_years > 0:
                    for i in range(decline_years): annual_oil_profile[plateau_years + i] = plateau_rate * (1.0 - (i + 0.5) / decline_years)
        elif profile_type == "plateau_exponential_decline" or profile_type == "plateau_hyperbolic_decline":
            objective_for_solver = lambda q_peak: self._calculate_total_volume_for_qp(q_peak, years, self.profile_params, profile_type) - total_oil_to_produce_stb
            q_low_bound = total_oil_to_produce_stb / max(1, years); plateau_dur_eff = max(1.0, years * self.profile_params.plateau_duration_fraction_of_life if "plateau" in profile_type else 1.0)
            q_high_bound = max(total_oil_to_produce_stb / plateau_dur_eff * 2.0, q_low_bound * 1.5)
            if objective_for_solver(q_low_bound) * objective_for_solver(q_high_bound) >= 0:
                test_low, test_high = objective_for_solver(q_low_bound), objective_for_solver(q_high_bound)
                if test_low > 0 and test_high > 0: q_high_bound *=5 
                elif test_low < 0 and test_high < 0: q_low_bound /= 5
                if objective_for_solver(q_low_bound) * objective_for_solver(q_high_bound) >= 0:
                    logger.warning(f"Brentq bounds [{q_low_bound:.2e},{q_high_bound:.2e}] fail for {profile_type}, f(low)={test_low:.2e}, f(high)={objective_for_solver(q_high_bound):.2e}. Target {total_oil_to_produce_stb:.2e}. Fallback linear.")
                    annual_oil_profile[:] = total_oil_to_produce_stb / years; current_sum = np.sum(annual_oil_profile);
                    if current_sum > 1e-9 and abs(current_sum - total_oil_to_produce_stb) > 1e-6 * total_oil_to_produce_stb : annual_oil_profile *= (total_oil_to_produce_stb / current_sum)
                    return annual_oil_profile
            try: q_peak_optimized = brentq(objective_for_solver, q_low_bound, q_high_bound, xtol=1e-3 * q_low_bound, rtol=1e-5)
            except ValueError as e: logger.warning(f"Brentq failed for {profile_type}: {e}. Fallback linear."); annual_oil_profile[:] = total_oil_to_produce_stb / years
            else:
                plateau_years_opt = int(np.floor(years * self.profile_params.plateau_duration_fraction_of_life)) if "plateau" in profile_type else 0; plateau_years_opt = max(0, min(plateau_years_opt, years)); decline_years_opt = years - plateau_years_opt
                if plateau_years_opt > 0: annual_oil_profile[:plateau_years_opt] = q_peak_optimized
                if decline_years_opt > 0:
                    q_decline_starts_at_opt = q_peak_optimized; current_q_start_decline_opt = q_decline_starts_at_opt
                    min_econ_rate_opt = (self.profile_params.min_economic_rate_fraction_of_peak * q_peak_optimized) if self.profile_params.min_economic_rate_fraction_of_peak is not None else 0.0
                    if profile_type == "plateau_exponential_decline":
                        Di_annual_opt = self.profile_params.initial_decline_rate_annual_fraction or 0.1
                        for t_d_idx in range(decline_years_opt):
                            year_idx_profile = plateau_years_opt + t_d_idx
                            if current_q_start_decline_opt < min_econ_rate_opt: annual_oil_profile[year_idx_profile] = 0.0; continue
                            annual_oil_profile[year_idx_profile] = current_q_start_decline_opt; current_q_start_decline_opt *= (1.0 - Di_annual_opt)
                    elif profile_type == "plateau_hyperbolic_decline":
                        Di_nom_opt, b_opt = self.profile_params.initial_decline_rate_annual_fraction or 0.1, self.profile_params.hyperbolic_b_factor or 0.5
                        q_i_dec_opt = q_decline_starts_at_opt; current_q_hyper = q_i_dec_opt
                        for t_yr_dec_opt in range(1, decline_years_opt + 1):
                            year_idx_profile = plateau_years_opt + t_yr_dec_opt - 1
                            if current_q_hyper < min_econ_rate_opt: annual_oil_profile[year_idx_profile] = 0.0; continue
                            t_s_opt, t_e_opt = float(t_yr_dec_opt - 1), float(t_yr_dec_opt); vol_yr_opt = 0.0
                            if b_opt == 0: vol_yr_opt = (q_i_dec_opt / Di_nom_opt) * (np.exp(-Di_nom_opt * t_s_opt) - np.exp(-Di_nom_opt * t_e_opt)) if Di_nom_opt > 1e-6 else q_i_dec_opt
                            elif abs(b_opt - 1.0) < 1e-6: vol_yr_opt = (q_i_dec_opt / Di_nom_opt) * np.log((1.0 + Di_nom_opt * t_e_opt) / (1.0 + Di_nom_opt * t_s_opt)) if Di_nom_opt > 1e-6 else q_i_dec_opt
                            else:
                                if Di_nom_opt > 1e-6: term_s_opt,term_e_opt = np.power(1.0+b_opt*Di_nom_opt*t_s_opt,(1.0-b_opt)/b_opt),np.power(1.0+b_opt*Di_nom_opt*t_e_opt,(1.0-b_opt)/b_opt); vol_yr_opt=(q_i_dec_opt/(Di_nom_opt*(1.0-b_opt)))*(term_s_opt-term_e_opt)
                                else: vol_yr_opt = q_i_dec_opt
                            annual_oil_profile[year_idx_profile] = vol_yr_opt
                            current_q_hyper = q_i_dec_opt / np.power(1.0 + b_opt * Di_nom_opt * t_e_opt, 1.0/b_opt) if b_opt > 1e-6 else q_i_dec_opt * np.exp(-Di_nom_opt * t_e_opt)
        elif profile_type == "custom_fractions":
            fractions = self.profile_params.oil_annual_fraction_of_total
            if fractions is None or len(fractions) != years: logger.warning(f"Custom oil fractions error. Fallback linear."); annual_oil_profile[:] = total_oil_to_produce_stb / years
            else: norm_factor = sum(fractions); norm_fractions = np.array(fractions) / norm_factor if norm_factor > 1e-6 else np.ones(years)/years; annual_oil_profile = norm_fractions * total_oil_to_produce_stb
        else: logger.error(f"Unknown oil profile: {profile_type}. Fallback linear."); annual_oil_profile[:] = total_oil_to_produce_stb / years
        current_sum = np.sum(annual_oil_profile)
        if current_sum > 1e-9 and abs(current_sum - total_oil_to_produce_stb) > 1e-6 * total_oil_to_produce_stb : annual_oil_profile *= (total_oil_to_produce_stb / current_sum)
        return annual_oil_profile

    def _generate_injection_profiles(self, primary_injectant_daily_rate: float, wag_water_fraction_of_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        # ... (No changes needed)
        years = self.operational_params.project_lifetime_years; annual_co2_inj, annual_h2o_inj = np.zeros(years), np.zeros(years)
        if years == 0: return annual_co2_inj, annual_h2o_inj
        eff_co2_rate, eff_h2o_rate = 0.0, 0.0
        if self.eor_params.injection_scheme == 'wag':
            if wag_water_fraction_of_time is None: wag_water_fraction_of_time = 0.0
            eff_co2_rate = primary_injectant_daily_rate * (1.0 - wag_water_fraction_of_time)
            if self.eor_params.WAG_ratio is not None and self.eor_params.WAG_ratio >= 0 and wag_water_fraction_of_time > 1e-6:
                if (1.0 - wag_water_fraction_of_time) <= 1e-6: h2o_rate_cycle = primary_injectant_daily_rate
                else: h2o_rate_cycle = (self.eor_params.WAG_ratio * primary_injectant_daily_rate * (1.0 - wag_water_fraction_of_time)) / wag_water_fraction_of_time
                eff_h2o_rate = h2o_rate_cycle * wag_water_fraction_of_time
        else: eff_co2_rate = primary_injectant_daily_rate
        annual_co2_inj[:] = eff_co2_rate * DAYS_PER_YEAR; annual_h2o_inj[:] = eff_h2o_rate * DAYS_PER_YEAR
        return annual_co2_inj, annual_h2o_inj

    def _calculate_annual_profiles(self, current_recovery_factor: float, optimized_params_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # ... (No changes needed)
        total_oil = self.reservoir.ooip_stb * current_recovery_factor; oil_profile = self._generate_oil_production_profile(total_oil)
        primary_rate = optimized_params_dict['rate']; wag_frac_time = None
        if self.eor_params.injection_scheme == 'wag':
            wag_frac_time = optimized_params_dict.get('water_fraction', (self.eor_params.min_water_fraction + self.eor_params.max_water_fraction) / 2.0)
        co2_inj_profile, h2o_inj_profile = self._generate_injection_profiles(primary_rate, wag_frac_time)
        return oil_profile, co2_inj_profile, h2o_inj_profile, h2o_inj_profile.copy()

    def _calculate_npv(self, annual_oil_stb: np.ndarray, annual_co2_mscf: np.ndarray, annual_water_injected_bbl: np.ndarray, annual_water_disposed_bbl: np.ndarray) -> float:
        # ... (No changes needed)
        npv = 0.0; econ = self.economic_params
        if len(annual_oil_stb) == 0: return 0.0
        for i in range(len(annual_oil_stb)):
            rev = annual_oil_stb[i]*econ.oil_price_usd_per_bbl; opex=annual_oil_stb[i]*econ.operational_cost_usd_per_bbl_oil
            co2_t = annual_co2_mscf[i]*self.co2_density_tonne_per_mscf; co2_cost = co2_t*econ.co2_purchase_cost_usd_per_tonne + annual_co2_mscf[i]*econ.co2_injection_cost_usd_per_mscf
            h2o_cost = annual_water_injected_bbl[i]*econ.water_injection_cost_usd_per_bbl + annual_water_disposed_bbl[i]*econ.water_disposal_cost_usd_per_bbl
            ncf = rev - (opex + co2_cost + h2o_cost); npv += ncf / ((1+econ.discount_rate_fraction)**(i+1))
        return npv

    def _calculate_co2_utilization_factor(self, annual_oil_stb: np.ndarray, annual_co2_mscf: np.ndarray) -> float:
        # ... (No changes needed)
        tot_oil, tot_co2_mscf = np.sum(annual_oil_stb), np.sum(annual_co2_mscf)
        if tot_oil <= 1e-6: return float('inf')
        if tot_co2_mscf <= 1e-6: return 0.0
        return (tot_co2_mscf * self.co2_density_tonne_per_mscf) / tot_oil

    def _get_ga_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        # ... (No changes needed)
        mmp_val = self.mmp or config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0); rate_max_factor = config_manager.get("OptimizationEngineSettings.genetic_algorithm.rate_bound_factor_max", 1.5) # type: ignore
        min_p = np.clip(self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi > mmp_val else mmp_val*1.01, mmp_val*1.01, self.eor_params.max_pressure_psi-1.0) # type: ignore
        b = {'pressure':(min_p, self.eor_params.max_pressure_psi), 'rate':(self.eor_params.min_injection_rate_bpd, self.eor_params.injection_rate*rate_max_factor), 'v_dp_coefficient':(0.3,0.8), 'mobility_ratio':(1.2,20.0)}
        if self.eor_params.injection_scheme == 'wag': b['cycle_length_days']=(self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days); b['water_fraction']=(self.eor_params.min_water_fraction, self.eor_params.max_water_fraction)
        return b

    def optimize_genetic_algorithm(self, ga_params_to_use: Optional[GeneticAlgorithmParams] = None) -> Dict[str, Any]:
        # ... (No changes needed)
        cfg = ga_params_to_use or self.ga_params_default_config; logger.info(f"GA: Obj '{self.chosen_objective}', Gens {cfg.generations}, Pop {cfg.population_size}")
        pop = self._initialize_population_ga(cfg.population_size, cfg); best_sol = pop[0].copy() if pop else {}; best_fit = -np.inf
        for gen in range(cfg.generations):
            with ProcessPoolExecutor() as ex: fits = list(ex.map(partial(self._evaluate_individual_ga, current_ga_config=cfg), pop))
            idx_best_gen = np.argmax(fits)
            if fits[idx_best_gen] > best_fit: best_fit = fits[idx_best_gen]; best_sol = pop[idx_best_gen].copy()
            parents = self._tournament_selection_ga(pop, fits, cfg); offspring = self._crossover_ga(parents, cfg); pop = self._mutate_ga(offspring, cfg)
            if (gen+1)%10==0 or gen==cfg.generations-1: logger.info(f"GA Gen {gen+1} BestFit: Gen:{fits[idx_best_gen]:.4e} Overall:{best_fit:.4e}")
        bounds = self._get_ga_parameter_bounds(); final_params = self._get_complete_params_from_ga_individual(best_sol, bounds)
        with ProcessPoolExecutor() as ex: final_fits = list(ex.map(partial(self._evaluate_individual_ga, current_ga_config=cfg), pop))
        top_sols = [{'params':self._get_complete_params_from_ga_individual(p,bounds), 'fitness':f} for p,f in sorted(zip(pop,final_fits),key=lambda x:x[1],reverse=True)[:min(len(pop),cfg.elite_count or 1)]]
        mmp_r, poro_r = self.mmp, self.avg_porosity; eff_rate = final_params['rate']*(1.0-final_params.get('water_fraction',0) if self.eor_params.injection_scheme=='wag' else 1.0)
        rf_kwargs = {**self._recovery_model_init_kwargs, **final_params}
        final_rf = recovery_factor(final_params['pressure'], eff_rate, poro_r, mmp_r if mmp_r is not None else 2500.0, model=self.recovery_model, **rf_kwargs)
        self._results = {'optimized_params_final_clipped':final_params, 'objective_function_value':best_fit, 'chosen_objective':self.chosen_objective, 'final_recovery_factor_reported':final_rf,
                         'mmp_psi':mmp_r, 'method':'genetic_algorithm', 'generations':cfg.generations, 'population_size':cfg.population_size, 'avg_porosity_used':poro_r, 'top_ga_solutions_from_final_pop':top_sols}
        logger.info(f"GA done. Best obj ({self.chosen_objective}): {best_fit:.4e}"); return self._results

    def _evaluate_individual_ga(self, individual_dict: Dict[str, float], current_ga_config: GeneticAlgorithmParams) -> float:
        param_bounds = self._get_ga_parameter_bounds(); complete_params = self._get_complete_params_from_ga_individual(individual_dict, param_bounds)
        return self._objective_function_wrapper(complete_params)
    def _initialize_population_ga(self, population_size: int, current_ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        # ... (No changes needed)
        return [{name:random.uniform(low,high) for name,(low,high) in self._get_ga_parameter_bounds().items()} for _ in range(population_size)]
    def _get_complete_params_from_ga_individual(self, individual_dict: Dict[str, float], param_bounds_for_clipping: Dict[str, Tuple[float,float]] ) -> Dict[str, float]:
        # ... (No changes needed)
        return {name:np.clip(individual_dict.get(name,random.uniform(low,high)),low,high) for name,(low,high) in param_bounds_for_clipping.items()}
    def _tournament_selection_ga(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        # ... (No changes needed)
        sel, pop_len = [], len(population)
        if ga_config.elite_count > 0 and pop_len > 0: sel.extend(population[idx].copy() for idx in np.argsort(fitnesses)[-ga_config.elite_count:])
        for _ in range(pop_len - len(sel)):
            if pop_len==0: break; t_size=min(ga_config.tournament_size,pop_len);
            if t_size <=0: continue
            sel.append(population[max(random.sample(range(pop_len),t_size), key=lambda i:fitnesses[i])].copy())
        return sel
    def _crossover_ga(self, parent_population: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        # ... (No changes needed)
        off, n_p = [], len(parent_population);
        if n_p==0: return []
        random.shuffle(parent_population)
        for i in range(0,n_p-1,2):
            p1,p2=parent_population[i],parent_population[i+1]; c1,c2=p1.copy(),p2.copy()
            if random.random()<ga_config.crossover_rate:
                alpha=ga_config.blend_alpha_crossover
                for k in set(p1.keys())&set(p2.keys()): v1,v2=p1[k],p2[k]; c1[k]=alpha*v1+(1-alpha)*v2; c2[k]=(1-alpha)*v1+alpha*v2
            off.extend([c1,c2])
        if len(off)<n_p and n_p>0: off.extend(parent_population[len(off):])
        return off[:n_p]
    def _mutate_ga(self, population_to_mutate: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        # ... (No changes needed)
        mut, p_b = [], self._get_ga_parameter_bounds()
        for ind in population_to_mutate:
            m_ind = ind.copy()
            if random.random()<ga_config.mutation_rate:
                gene = random.choice(list(m_ind.keys()))
                if gene in p_b:
                    l,h=p_b[gene]; curr=m_ind.get(gene,(l+h)/2.0); rng=h-l
                    sig=max(rng*ga_config.mutation_strength_factor if rng>1e-9 else (abs(curr)*0.1+1e-6), 1e-7)
                    m_ind[gene]=np.clip(curr+random.gauss(0,sig),l,h)
            mut.append(m_ind)
        return mut

    def optimize_bayesian(self, n_iter_override: Optional[int]=None, init_points_override: Optional[int]=None, method_override: Optional[str]=None, initial_solutions_from_ga: Optional[List[Dict[str,Any]]]=None) -> Dict[str,Any]:
        # ... (No changes needed)
        cfg = config_manager.get_section("OptimizationEngineSettings.bayesian_optimizer") or {}; n_i,init_r,bo_m = n_iter_override or cfg.get("n_iter",40), init_points_override or cfg.get("init_points_random",8), method_override or cfg.get("default_method","gp") # type: ignore
        logger.info(f"BayesOpt: Obj '{self.chosen_objective}', Method '{bo_m}', Iter {n_i}, InitRand {init_r}")
        p_bounds_d = self._get_ga_parameter_bounds(); space_skopt=[Real(l,h,name=n) for n,(l,h) in p_bounds_d.items()]; p_names_ord=[d.name for d in space_skopt]
        @use_named_args(space_skopt)
        def obj_skopt(**p): return -self._objective_function_wrapper(p)
        def obj_bayes_lib(**p): return self._objective_function_wrapper(p)
        best_p_bo:Dict[str,float]={}; final_obj_bo:float=-np.inf; n_ga_pts=0
        if bo_m=='gp':
            x0,y0 = ([] if not initial_solutions_from_ga else [list(s['params'].values()) for s in initial_solutions_from_ga]), ([] if not initial_solutions_from_ga else [-s['fitness'] for s in initial_solutions_from_ga]); n_ga_pts=len(x0)
            tot_evals = (n_ga_pts+n_i) if x0 else (init_r+n_i); init_skopt_rand = 0 if x0 else init_r
            logger.info(f"gp_minimize: {n_ga_pts} GA pts, {init_skopt_rand} add'l rand, {n_i} BO iter. Total: {tot_evals}")
            res_sk = gp_minimize(func=obj_skopt,dimensions=space_skopt,x0=x0 or None,y0=y0 or None,n_calls=tot_evals,n_initial_points=init_skopt_rand,random_state=config_manager.get("GeneralFallbacks.random_seed",42),verbose=cfg.get("verbose_skopt",True)) # type: ignore
            best_p_bo={n:v for n,v in zip(p_names_ord,res_sk.x)}; final_obj_bo=-res_sk.fun
        elif bo_m=='bayes':
            pb_bayes={n:(l,h) for n,(l,h) in p_bounds_d.items()}; bayes_o=BayesianOptimization(f=obj_bayes_lib,pbounds=pb_bayes,random_state=config_manager.get("GeneralFallbacks.random_seed",42),verbose=cfg.get("verbose_bayes_opt",2)) # type: ignore
            if initial_solutions_from_ga:
                for sol in initial_solutions_from_ga:
                    try:
                        p_reg = {k: np.clip(v, pb_bayes[k][0], pb_bayes[k][1]) for k,v in sol['params'].items() if k in pb_bayes}
                        if len(p_reg) == len(pb_bayes):
                            bayes_o.register(params=p_reg, target=sol['fitness'])
                            n_ga_pts += 1
                    except Exception as e:
                        logger.error(f"Error registering GA sol: {e}")
            bayes_o.maximize(init_points=init_r,n_iter=n_i); best_p_bo=bayes_o.max['params']; final_obj_bo=bayes_o.max['target']
        else: raise ValueError(f"Unsupported Bayes method: {bo_m}")
        mmp_r_bo,poro_r_bo = self.mmp, self.avg_porosity; eff_rate_bo=best_p_bo['rate']*(1.0-best_p_bo.get('water_fraction',0) if self.eor_params.injection_scheme=='wag' and 'water_fraction' in best_p_bo else 1.0)
        rec_kw_bo={**self._recovery_model_init_kwargs,**best_p_bo}; final_rf_bo=recovery_factor(best_p_bo['pressure'],eff_rate_bo,poro_r_bo,mmp_r_bo if mmp_r_bo is not None else 2500.0,model=self.recovery_model,**rec_kw_bo)
        self._results={'optimized_params_final_clipped':best_p_bo, 'objective_function_value':final_obj_bo, 'chosen_objective':self.chosen_objective, 'final_recovery_factor_reported':final_rf_bo,
                       'mmp_psi':mmp_r_bo, 'method':f'bayesian_{bo_m}', 'iterations_bo_actual':n_i, 'initial_points_bo_random_requested':init_r, 'initial_points_from_ga_used':n_ga_pts, 'avg_porosity_used':poro_r_bo}
        logger.info(f"Bayesian ({bo_m}) done. Best obj ({self.chosen_objective}): {final_obj_bo:.4e}"); return self._results

    def hybrid_optimize(self, ga_params_override: Optional[GeneticAlgorithmParams]=None) -> Dict[str,Any]:
        # ... (No changes needed)
        cfg_h = config_manager.get_section("OptimizationEngineSettings.hybrid_optimizer") or {}; ga_p_h:GeneticAlgorithmParams # type: ignore
        if ga_params_override: ga_p_h=ga_params_override
        else:
            src=cfg_h.get("ga_config_source","default_ga_params")
            if src=="hybrid_specific" and "ga_params_hybrid" in cfg_h: ga_p_h=GeneticAlgorithmParams.from_config_dict(cfg_h["ga_params_hybrid"])
            else: ga_p_h=self.ga_params_default_config
        bo_i_h,bo_init_h,n_elites_h,bo_m_h = cfg_h.get("bo_iterations_in_hybrid",20),cfg_h.get("bo_random_initial_points_in_hybrid",5),cfg_h.get("num_ga_elites_to_bo",3),cfg_h.get("bo_method_in_hybrid","gp")
        logger.info(f"Hybrid Opt: Obj '{self.chosen_objective}'. GA: Gens:{ga_p_h.generations}, Pop:{ga_p_h.population_size}")
        ga_res=self.optimize_genetic_algorithm(ga_params_to_use=ga_p_h); init_bo_sols:Optional[List[Dict[str,Any]]]=None
        if n_elites_h>0 and 'top_ga_solutions_from_final_pop' in ga_res and ga_res['top_ga_solutions_from_final_pop']:
            init_bo_sols = ga_res['top_ga_solutions_from_final_pop'][:min(n_elites_h, len(ga_res['top_ga_solutions_from_final_pop']))]
            logger.info(f"Hybrid BO: Seeding with {len(init_bo_sols)} elite(s) from GA.")
        logger.info(f"Hybrid BO: Method '{bo_m_h}', Iter {bo_i_h}, RandInit (if needed) {bo_init_h}")
        bo_res=self.optimize_bayesian(n_iter_override=bo_i_h,init_points_override=bo_init_h,method_override=bo_m_h,initial_solutions_from_ga=init_bo_sols)
        self._results={**bo_res, 'ga_full_results_for_hybrid':ga_res, 'method':f'hybrid_ga(g{ga_p_h.generations})_bo(m:{bo_m_h},i:{bo_i_h},e:{len(init_bo_sols or [])})'}
        logger.info(f"Hybrid opt done. Final obj ({self.chosen_objective}): {self._results.get('objective_function_value',0.0):.4e}"); return self._results

    def optimize_recovery(self) -> Dict[str, Any]: # Gradient Descent for Pressure
        # ... (No changes needed)
        cfg_g=config_manager.get_section("OptimizationEngineSettings.gradient_descent_optimizer") or {}; max_i,tol,lr,p_p = cfg_g.get("max_iter",100),cfg_g.get("tolerance",1e-4),cfg_g.get("learning_rate",50.0),cfg_g.get("pressure_perturbation",10.0) # type: ignore
        logger.info(f"GradDesc (Pressure Only): Obj '{self.chosen_objective}'")
        mmp_v_gd,poro_gd = self.mmp or 2500.0, self.avg_porosity
        base_p_gd={'rate':self.eor_params.injection_rate,'v_dp_coefficient':self.eor_params.v_dp_coefficient,'mobility_ratio':self.eor_params.mobility_ratio}
        if self.eor_params.injection_scheme=='wag': base_p_gd['water_fraction']=(self.eor_params.min_water_fraction+self.eor_params.max_water_fraction)/2.0; base_p_gd['cycle_length_days']=(self.eor_params.min_cycle_length_days+self.eor_params.max_cycle_length_days)/2.0
        curr_p=np.clip(self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi>mmp_v_gd else mmp_v_gd*1.01, mmp_v_gd*1.01, self.eor_params.max_pressure_psi)
        last_obj,conv,done_iter = -np.inf,False,0
        for i in range(max_i):
            done_iter=i+1; eval_p={**base_p_gd,'pressure':curr_p}; curr_obj=self._objective_function_wrapper(eval_p)
            if i>0 and abs(curr_obj-last_obj)<(tol*abs(last_obj) if last_obj!=0 else tol): conv=True;break
            pert_p={**base_p_gd,'pressure':curr_p+p_p}; obj_plus_p=self._objective_function_wrapper(pert_p)
            grad=(obj_plus_p-curr_obj)/p_p
            if abs(grad)<1e-9: conv=True;break
            eff_lr=lr; curr_p=np.clip(curr_p+eff_lr*grad, mmp_v_gd*1.01, self.eor_params.max_pressure_psi); last_obj=curr_obj
        final_p_gd={**base_p_gd,'pressure':curr_p}; eff_co2_gd=final_p_gd['rate']*(1.0-final_p_gd.get('water_fraction',0) if self.eor_params.injection_scheme=='wag' else 1.0)
        rec_kw_gd={**self._recovery_model_init_kwargs,**final_p_gd}; final_rf_gd=recovery_factor(final_p_gd['pressure'],eff_co2_gd,poro_gd,mmp_v_gd,model=self.recovery_model,**rec_kw_gd)
        self._results={'optimized_params_final_clipped':final_p_gd,'objective_function_value':last_obj,'chosen_objective':self.chosen_objective,
                       'final_recovery_factor_reported':final_rf_gd,'mmp_psi':mmp_v_gd,'iterations':done_iter,'converged':conv,'avg_porosity_used':poro_gd,'method':'gradient_descent_pressure'}
        logger.info(f"GradDesc done. Obj ({self.chosen_objective}): {last_obj:.4e}"); return self._results

    def optimize_wag(self) -> Dict[str, Any]: # Iterative Grid Search
        # ... (No changes needed)
        cfg_w=config_manager.get_section("OptimizationEngineSettings.wag_optimizer") or {}; ref_c,grid_p,range_r = cfg_w.get("refinement_cycles",5),cfg_w.get("grid_search_points_per_dim",5),cfg_w.get("range_reduction_factor",0.5) # type: ignore
        logger.info(f"WAG Opt (Grid Search): Obj '{self.chosen_objective}'")
        mmp_v_w,poro_w = self.mmp or 2500.0, self.avg_porosity
        min_cl,max_cl=self.eor_params.min_cycle_length_days,self.eor_params.max_cycle_length_days; min_wf,max_wf=self.eor_params.min_water_fraction,self.eor_params.max_water_fraction
        fixed_p_w={'rate':self.eor_params.injection_rate,'v_dp_coefficient':self.eor_params.v_dp_coefficient,'mobility_ratio':self.eor_params.mobility_ratio,
                   'pressure':np.clip(self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi>mmp_v_w else mmp_v_w*1.1, mmp_v_w*1.01,self.eor_params.max_pressure_psi)}
        best_w_sol={**fixed_p_w,'cycle_length_days':(min_cl+max_cl)/2.0,'water_fraction':(min_wf+max_wf)/2.0}; best_obj_w=-np.inf
        for cyc in range(ref_c):
            logger.info(f"WAG Opt Cycle {cyc+1}: CL [{min_cl:.1f}-{max_cl:.1f}], WF [{min_wf:.2f}-{max_wf:.2f}]"); improved=False
            for wf_v in np.linspace(min_wf,max_wf,num=grid_p):
                for cl_v in np.linspace(min_cl,max_cl,num=grid_p):
                    eval_p_w={**fixed_p_w,'water_fraction':wf_v,'cycle_length_days':cl_v}; obj_gp=self._objective_function_wrapper(eval_p_w)
                    if obj_gp>best_obj_w: best_obj_w=obj_gp; best_w_sol.update({'cycle_length_days':cl_v,'water_fraction':wf_v,'pressure':eval_p_w['pressure']}); improved=True
            if not improved and cyc>0: logger.info("WAG grid search converged."); break
            wf_rng_n=(max_wf-min_wf)*range_r/2.0; min_wf=max(self.eor_params.min_water_fraction,best_w_sol['water_fraction']-wf_rng_n); max_wf=min(self.eor_params.max_water_fraction,best_w_sol['water_fraction']+wf_rng_n)
            cl_rng_n=(max_cl-min_cl)*range_r/2.0; min_cl=max(self.eor_params.min_cycle_length_days,best_w_sol['cycle_length_days']-cl_rng_n); max_cl=min(self.eor_params.max_cycle_length_days,best_w_sol['cycle_length_days']+cl_rng_n)
        eff_co2_w=best_w_sol['rate']*(1.0-best_w_sol['water_fraction']); rec_kw_w={**self._recovery_model_init_kwargs,**best_w_sol}; final_rf_w=recovery_factor(best_w_sol['pressure'],eff_co2_w,poro_w,mmp_v_w,model=self.recovery_model,**rec_kw_w)
        self._results={'optimized_params_final_clipped':best_w_sol,'objective_function_value':best_obj_w,'chosen_objective':self.chosen_objective,
                       'final_recovery_factor_reported':final_rf_w,'mmp_psi':mmp_v_w,'avg_porosity_used':poro_w,'method':'iterative_grid_search_wag'}
        logger.info(f"WAG Opt done. Best obj ({self.chosen_objective}): {best_obj_w:.4e}"); return self._results
    
    def check_mmp_constraint(self, pressure: float) -> bool:
        mmp_check = self.mmp; return pressure >= mmp_check if mmp_check is not None else False

    @property
    def results(self) -> Optional[Dict[str, Any]]: return self._results

    def set_recovery_model(self, model_name: str, **kwargs_for_model_init_override: Any):
        valid_models = ['simple', 'miscible', 'immiscible', 'hybrid', 'koval', 'layered']
        model_name_lower = model_name.lower()
        if model_name_lower not in valid_models: raise ValueError(f"Unknown recovery model: {model_name}")
        self.recovery_model = model_name_lower
        base_init_cfg = config_manager.get_section(f"RecoveryModelKwargsDefaults.{self.recovery_model.capitalize()}") or {} # type: ignore
        self._recovery_model_init_kwargs = {**base_init_cfg, **kwargs_for_model_init_override}
        logger.info(f"Recovery model set to '{self.recovery_model}'. Init kwargs: {self._recovery_model_init_kwargs}")

    # --- Plotting Methods ---
    def plot_mmp_profile(self) -> Optional[go.Figure]:
        if not (self.well_analysis and hasattr(self.well_analysis, 'calculate_mmp_profile')): return None
        try:
            profile_data = self.well_analysis.calculate_mmp_profile(); assert isinstance(profile_data, dict)
            if not all(k in profile_data for k in ['depths','mmp']) or not profile_data['depths'].size or not profile_data['mmp'].size: return None
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=profile_data['mmp'], y=profile_data['depths'], name='MMP (psi)', line_color='blue'), secondary_y=False)
            if 'temperature' in profile_data and profile_data['temperature'].size > 0: fig.add_trace(go.Scatter(x=profile_data['temperature'], y=profile_data['depths'], name='Temp (°F)', line_color='red'), secondary_y=True)
            fig.update_layout(title_text='MMP vs Depth Profile', yaxis_title_text='Depth (ft)', legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)); fig.update_yaxes(title_text="Depth (ft)", secondary_y=False, autorange="reversed")
            if 'temperature' in profile_data and profile_data['temperature'].size > 0: fig.update_yaxes(title_text="Temp Axis", secondary_y=True, autorange="reversed", overlaying='y', side='right', showticklabels=True); fig.update_xaxes(title_text="Value (MMP or Temp)")
            else: fig.update_xaxes(title_text="MMP (psi)")
            return fig
        except Exception as e: logger.error(f"Error generating MMP plot: {e}", exc_info=True); return None

    def plot_optimization_convergence(self, results_to_plot: Optional[Dict[str,Any]] = None) -> Optional[go.Figure]:
        # ... (No changes needed)
        source = results_to_plot or self._results;
        if not source: return None
        obj_val, obj_name, method = source.get('objective_function_value'), source.get('chosen_objective','Obj'), source.get('method','unknown')
        if obj_val is None: return None
        fig = go.Figure(); title = f'Opt Outcome ({method}) for {obj_name.replace("_"," ").title()}'; y_title = f'{obj_name.replace("_"," ").title()} Val'; x_title='Opt Steps (Conceptual)'; num_steps=1
        if 'ga_full_results_for_hybrid' in source and isinstance(source['ga_full_results_for_hybrid'], dict):
            ga_res = source['ga_full_results_for_hybrid']; ga_gens, ga_obj = ga_res.get('generations',0), ga_res.get('objective_function_value')
            if ga_gens > 0 and ga_obj is not None: fig.add_trace(go.Scatter(x=[ga_gens], y=[ga_obj], mode='markers+text', name='GA End', text="GA")); num_steps=ga_gens
        if 'iterations_bo_actual' in source:
            bo_iters=source.get('iterations_bo_actual',0); bo_start=num_steps+1; bo_end=bo_start+bo_iters-1
            fig.add_trace(go.Scatter(x=[bo_end if bo_iters>0 else bo_start], y=[obj_val], mode='markers+text', name='BO/Final', text="BO/Final")); num_steps=bo_end if bo_iters>0 else bo_start
        elif 'iterations' in source: total_iters=source.get('iterations',1); fig.add_trace(go.Scatter(x=[total_iters], y=[obj_val], mode='markers+text', name='Final', text="Final")); num_steps=total_iters
        else: fig.add_trace(go.Scatter(x=[1], y=[obj_val], mode='markers+text', name='Final Obj Val', text="Final"))
        fig.update_layout(title_text=title, xaxis_title_text=x_title, yaxis_title_text=y_title); return fig

    def plot_parameter_sensitivity(self, param_name_for_sensitivity: str, results_to_use_for_plot: Optional[Dict[str,Any]]=None) -> Optional[go.Figure]:
        # ... (No changes needed)
        source = results_to_use_for_plot or self._results; num_pts_sens = config_manager.get("OptimizationEngineSettings.sensitivity_plot_points", 20) # type: ignore
        if not (source and 'optimized_params_final_clipped' in source and isinstance(source['optimized_params_final_clipped'], dict)): return None
        opt_base = source['optimized_params_final_clipped'].copy(); obj_name_sens = source.get('chosen_objective', 'Objective Value')
        if param_name_for_sensitivity not in opt_base:
            # Check if it's a parameter that isn't optimized by GA (e.g. oil_price)
            if not hasattr(self.economic_params, param_name_for_sensitivity):
                 logger.warning(f"Sensitivity plot: Parameter '{param_name_for_sensitivity}' not found in optimized params or economic params.")
                 return None
            else:
                 # It's an economic parameter, proceed but note it's not from the optimizer's direct output dict
                 curr_opt_val_param = getattr(self.economic_params, param_name_for_sensitivity)
                 # Bounds for non-optimized params will be simple percentages
                 low_b=curr_opt_val_param*0.8; high_b=curr_opt_val_param*1.2
        else:
             curr_opt_val_param = opt_base[param_name_for_sensitivity]
             bounds_ref = self._get_ga_parameter_bounds(); low_b,high_b=0.0,0.0
             if param_name_for_sensitivity in bounds_ref: g_low,g_high=bounds_ref[param_name_for_sensitivity]; sweep_ext=(g_high-g_low)*0.20; low_b=max(g_low,curr_opt_val_param-sweep_ext); high_b=min(g_high,curr_opt_val_param+sweep_ext)
             else: low_b=curr_opt_val_param*0.8; high_b=curr_opt_val_param*1.2
        
        if high_b<=low_b+1e-9: delta=abs(curr_opt_val_param*0.05)+1e-3; low_b=curr_opt_val_param-delta; high_b=curr_opt_val_param+delta;
        if param_name_for_sensitivity in self._get_ga_parameter_bounds(): g_l,g_h=self._get_ga_parameter_bounds()[param_name_for_sensitivity]; low_b=max(g_l,low_b); high_b=min(g_h,high_b)
        if high_b<=low_b+1e-9: logger.warning(f"Invalid sweep range for '{param_name_for_sensitivity}'. Abort plot."); return None
        
        param_vals_sweep=np.linspace(low_b,high_b,num_pts_sens); obj_vals_sens=[]

        for p_sweep in param_vals_sweep:
            temp_eor_params = opt_base.copy()
            temp_econ_params = deepcopy(self.economic_params)
            # Check if the sensitivity parameter is an economic one
            if hasattr(temp_econ_params, param_name_for_sensitivity):
                setattr(temp_econ_params, param_name_for_sensitivity, p_sweep)
            else:
                temp_eor_params[param_name_for_sensitivity] = p_sweep
            
            eval_res = self.evaluate_for_analysis(
                eor_operational_params_dict=temp_eor_params, 
                economic_params_override=temp_econ_params,
                target_objectives=[obj_name_sens]
            )
            obj_vals_sens.append(eval_res.get(obj_name_sens, np.nan))

        fig=go.Figure(); fig.add_trace(go.Scatter(x=param_vals_sweep,y=obj_vals_sens,mode='lines+markers',name=f'{param_name_for_sensitivity} effect'))
        fig.add_vline(x=curr_opt_val_param,line=dict(width=2,dash="dash",color="green"),annotation_text="Opt. Value",annotation_position="top left")
        title_param_disp=param_name_for_sensitivity.replace("_psi"," (psi)").replace("_days"," (days)").replace("_bpd"," (bpd)").replace("_"," ").title(); y_title_disp=obj_name_sens.replace("_"," ").title()
        fig.update_layout(title_text=f'{y_title_disp} vs. {title_param_disp}',xaxis_title_text=title_param_disp,yaxis_title_text=y_title_disp); return fig