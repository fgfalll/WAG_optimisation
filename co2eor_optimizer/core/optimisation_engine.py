from typing import Callable, Dict, List, Optional, Any, Tuple, Union, Type
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
from copy import deepcopy
import dataclasses

# [ADDED] Import for NPV calculation and a fallback
try:
    from numpy_financial import npv
except ImportError:
    logging.warning("numpy_financial not found. Using a manual NPV calculation. "
                    "For better performance, please `pip install numpy_financial`.")
    def npv(rate, values):
        """Manual NPV calculation if numpy_financial is not available."""
        # Ensure values is a numpy array
        values = np.atleast_1d(values)
        # Create an array of time periods [0, 1, 2, ...]
        t = np.arange(len(values))
        # Calculate discount factors for each period
        discount_factors = (1 + rate) ** t
        # Calculate present value of each cash flow
        present_values = values / discount_factors
        # Sum them up to get the Net Present Value
        return np.sum(present_values)

from bayes_opt import BayesianOptimization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from co2eor_optimizer.analysis.well_analysis import WellAnalysis

from .data_models import (
    ReservoirData, PVTProperties, EORParameters, GeneticAlgorithmParams,
    BayesianOptimizationParams, EconomicParameters, OperationalParameters,
    ProfileParameters, EOSModelParameters
)
from .recovery_models import recovery_factor
from .eos_models import EOSModel, PengRobinsonEOS, SoaveRedlichKwongEOS

try:
    from co2eor_optimizer.evaluation.mmp import calculate_mmp, MMPParameters
except ImportError:
    logging.info("evaluation.mmp module not found in OptimizationEngine. MMP calculation features may be limited.")
    calculate_mmp_external = None
    MMPParameters = None # type: ignore
else:
    calculate_mmp_external = calculate_mmp

DAYS_PER_YEAR = 365.25
logger = logging.getLogger(__name__)

# --- [NEW] Helper function for parallel processing ---
# This function must be defined at the top level of the module to be pickle-able
def _run_objective_wrapper(objective_func: Callable, params_dict: Dict[str, float]) -> float:
    """A top-level helper to call the objective function with kwargs for multiprocessing."""
    # This function is crucial for ProcessPoolExecutor to work on all platforms (e.g., Windows),
    # as it cannot pickle instance methods directly.
    # The objective_func will be a partial function with `self` already bound.
    return objective_func(**params_dict)

class OptimizationEngine:
    RELAXABLE_CONSTRAINTS = {
        'porosity': {
            'description': 'Average reservoir porosity (v/v)',
            'range_factor': 0.20,
            'type': 'reservoir'
        },
        'ooip_stb': {
            'description': 'Original Oil In Place (STB)',
            'range_factor': 0.25,
            'type': 'reservoir'
        },
        'v_dp_coefficient': {
            'description': 'Dykstra-Parsons coefficient for heterogeneity',
            'range_factor': 0.35,
            'type': 'eor'
        },
        'mobility_ratio': {
            'description': 'Mobility Ratio (M)',
            'range_factor': 0.50,
            'type': 'eor'
        },
        'WAG_ratio': {
            'description': 'Water-Alternating-Gas Ratio',
            'range_factor': 0.60,
            'type': 'eor'
        },
        'gravity_factor': {
            'description': 'Gravity factor in miscible recovery model',
            'range_factor': 0.75,
            'type': 'eor'
        },
        'sor': {
            'description': 'Residual Oil Saturation for immiscible model',
            'range_factor': 0.20,
            'type': 'eor'
        },
        'transition_alpha': {
            'description': 'Transition center for hybrid recovery model',
            'range_factor': 0.15,
            'type': 'eor'
        },
        'transition_beta': {
            'description': 'Transition steepness for hybrid recovery model',
            'range_factor': 0.50,
            'type': 'eor'
        }
    }

    def __init__(self,
                 reservoir: ReservoirData,
                 pvt: PVTProperties,
                 eor_params_instance: Optional[EORParameters] = None,
                 ga_params_instance: Optional[GeneticAlgorithmParams] = None,
                 bo_params_instance: Optional[BayesianOptimizationParams] = None,
                 economic_params_instance: Optional[EconomicParameters] = None,
                 operational_params_instance: Optional[OperationalParameters] = None,
                 profile_params_instance: Optional[ProfileParameters] = None,
                 well_data_list: Optional[List[Any]] = None,
                 avg_porosity_init_override: Optional[float] = None,
                 mmp_init_override: Optional[float] = None,
                 recovery_model_init_kwargs_override: Optional[Dict[str, Any]] = None,
                 skip_auto_calculations: bool = False):
        self._skip_auto_calculations = skip_auto_calculations
        self._base_reservoir_data = deepcopy(reservoir)
        self._base_pvt_data = deepcopy(pvt)
        self._base_eor_params = deepcopy(eor_params_instance or EORParameters())
        self._base_economic_params = deepcopy(economic_params_instance or EconomicParameters())
        self._base_operational_params = deepcopy(operational_params_instance or OperationalParameters())
        
        self.well_analysis: Optional[WellAnalysis] = None
        if well_data_list and pvt:
            logger.info(f"Engine creating internal WellAnalysis object using representative well: '{well_data_list[0].name}'")
            self.well_analysis = WellAnalysis(well_data=well_data_list[0], pvt_data=pvt)
        
        self._unlocked_params_for_current_run: List[str] = []
        
        self.reset_to_base_state()
        
        self.ga_params_default_config = ga_params_instance or GeneticAlgorithmParams()
        self.bo_params_default_config = bo_params_instance or BayesianOptimizationParams()
        self.profile_params = profile_params_instance or ProfileParameters()
        
        if self.profile_params.warn_if_defaults_used and profile_params_instance is None:
            logger.warning("Using default ProfileParameters.")

        self._results: Optional[Dict[str, Any]] = None
        self._avg_porosity_init_override: Optional[float] = avg_porosity_init_override
        self._mmp_value_init_override: Optional[float] = mmp_init_override
        self._mmp_value: Optional[float] = self._mmp_value_init_override
        
        self.recovery_model: str = "hybrid" # Hardcode a sensible default
        self._recovery_model_init_kwargs: Dict[str, Any] = recovery_model_init_kwargs_override or {}

        self.chosen_objective: str = "recovery_factor" # Default objective
        self.co2_density_tonne_per_mscf: float = 0.053 # Default density
        
        self._mmp_calculator_fn = calculate_mmp_external
        self._MMPParametersDataclass = MMPParameters
        
        self._chaos_state: float = random.random()
        self._best_fitness_history: List[float] = []
        self._is_stagnated: bool = False

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
                    self.pvt.pvt_type = 'compositional'
            except Exception as e:
                logger.error(f"Failed to instantiate EOS model from ReservoirData: {e}", exc_info=True)
                self.eos_model_instance = None


    def reset_to_base_state(self):
        logger.info("Resetting OptimizationEngine to its base state.")
        self.reservoir = deepcopy(self._base_reservoir_data)
        self.pvt = deepcopy(self._base_pvt_data)
        self.eor_params = deepcopy(self._base_eor_params)
        self.economic_params = deepcopy(self._base_economic_params)
        self.operational_params = deepcopy(self._base_operational_params)
        
        default_model = 'hybrid'
        self.recovery_model = getattr(self.operational_params, 'recovery_model_selection', default_model)
        
        if not hasattr(self.operational_params, 'recovery_model_selection'):
            logger.warning(
                f"OperationalParameters is missing 'recovery_model_selection'. Falling back to default: '{self.recovery_model}'."
            )
        else:
            logger.info(f"Recovery model dynamically selected: '{self.recovery_model}'")

        self._unlocked_params_for_current_run = []
        self._best_fitness_history = []
        self._chaos_state = random.random()
        self._is_stagnated = False

    def prepare_for_rerun_with_unlocked_params(self, params_to_unlock: List[str]):
        self.reset_to_base_state()
        self._unlocked_params_for_current_run = [p for p in params_to_unlock if p in self.RELAXABLE_CONSTRAINTS]
        logger.info(f"Engine prepared for re-run. Unlocked parameters: {self._unlocked_params_for_current_run}")

    @property
    def avg_porosity(self) -> float:
        if self._avg_porosity_init_override is not None:
            return self._avg_porosity_init_override
        poro_arr = self.reservoir.grid.get('PORO', np.array([0.15]))
        return np.mean(poro_arr) if hasattr(poro_arr, 'size') and poro_arr.size > 0 else 0.15

    @property
    def mmp(self) -> Optional[float]:
        if self._mmp_value_init_override is not None:
             if self._mmp_value is None: self._mmp_value = self._mmp_value_init_override
             return self._mmp_value_init_override
        
        if self._mmp_value is None:
            if not self._skip_auto_calculations:
                logger.warning("MMP property accessed but value is None. Calculating on-demand.")
                self.calculate_mmp()
            else:
                logger.warning("MMP property accessed but value is None and auto-calculation is skipped.")
        return self._mmp_value
    
    @property
    def results(self) -> Optional[Dict[str, Any]]:
        return self._results

    def _refresh_operational_parameters(self):
        logger.debug("Operational parameters are managed by MainWindow state.")

    def update_parameters(self, new_params: Dict[str, Any]):
        logger.debug("Updating engine's internal parameters from optimizer.")
        dataclasses_to_update = [
            ("eor_params", self.eor_params),
            ("economic_params", self.economic_params)
        ]
        for dc_name, dc_instance in dataclasses_to_update:
            current_params_dict = dataclasses.asdict(dc_instance)
            updated = False
            for key, value in new_params.items():
                if hasattr(dc_instance, key):
                    current_params_dict[key] = value
                    updated = True
            if updated:
                try:
                    setattr(self, dc_name, dc_instance.__class__(**current_params_dict))
                    logger.debug(f"Successfully updated {dc_name}.")
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to update {dc_name}: {e}")

    def calculate_mmp(self, method_override: Optional[str] = None) -> float:
        """
        Calculates MMP. Prioritizes an externally provided override value.
        If no override is present, it calculates a representative value using
        the best available data source (WellAnalysis > PVT).
        """
        if self._mmp_value_init_override is not None:
            logger.info(f"Using externally provided MMP override value: {self._mmp_value_init_override:.2f} psi.")
            self._mmp_value = self._mmp_value_init_override
            return self._mmp_value

        default_mmp_fallback = 2500.0
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
            logger.warning("MMP calculation dependencies not found. Using fallback value.")
            return self._mmp_value

        actual_mmp_method = method_override or "auto"
        mmp_input_object: Union[PVTProperties, Any] = self.pvt
        source_description = "PVT data"

        if self.well_analysis and hasattr(self.well_analysis, 'get_average_mmp_params_for_engine'):
            try:
                avg_well_params = self.well_analysis.get_average_mmp_params_for_engine()
                if avg_well_params:
                    mmp_input_constructor_params = {
                        'temperature': avg_well_params.get('temperature', self.pvt.temperature),
                        'oil_gravity': avg_well_params.get('oil_gravity', 35.0),
                        'c7_plus_mw': avg_well_params.get('c7_plus_mw'),
                        'injection_gas_composition': avg_well_params.get('injection_gas_composition', {'CO2': 1.0}),
                        'pvt_data': self.pvt
                    }
                    mmp_input_object = self._MMPParametersDataclass(**mmp_input_constructor_params)
                    source_description = f"WellAnalysis average parameters (from well: {self.well_analysis.well_data.name})"
            except Exception as e: 
                logger.warning(f"Failed to get average MMP parameters from WellAnalysis: {e}. Falling back to PVT data.", exc_info=True)
        
        try:
            mmp_calc_value = float(self._mmp_calculator_fn(mmp_input_object, method=actual_mmp_method))
            logger.info(f"MMP calculated: {mmp_calc_value:.2f} psi (method: '{actual_mmp_method}', source: {source_description}).")
            self._mmp_value = mmp_calc_value
        except Exception as e:
            logger.error(f"MMP calculation failed (method: '{actual_mmp_method}', source: {source_description}): {e}. Using fallback.", exc_info=True)
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
        
        return self._mmp_value

    def evaluate_for_analysis(self,
                              eor_operational_params_dict: Dict[str, float],
                              economic_params_override: Optional[EconomicParameters] = None,
                              avg_porosity_override: Optional[float] = None,
                              mmp_override: Optional[float] = None,
                              ooip_override: Optional[float] = None,
                              recovery_model_init_kwargs_override: Optional[Dict[str, Any]] = None,
                              target_objectives: Optional[List[str]] = None
                             ) -> Dict[str, float]:
        results: Dict[str, float] = {}
        if target_objectives is None: target_objectives = [self.chosen_objective]

        mmp_to_use = mmp_override if mmp_override is not None else self.mmp
        if mmp_to_use is None: mmp_to_use = 2500.0
            
        porosity_to_use = avg_porosity_override if avg_porosity_override is not None else self.avg_porosity
        current_econ_params = economic_params_override if economic_params_override is not None else self.economic_params
        ooip_to_use = ooip_override if ooip_override is not None else self.reservoir.ooip_stb
        
        pressure = eor_operational_params_dict['pressure']
        base_injection_rate = eor_operational_params_dict['rate']
        
        rf_call_kwargs = {**self._recovery_model_init_kwargs, **(recovery_model_init_kwargs_override or {})}

        if self.eos_model_instance:
            try:
                fluid_props = self.eos_model_instance.calculate_properties(pressure, self.pvt.temperature)
                if fluid_props.get('status') == 'Success':
                    if 'oil_viscosity_cp' in fluid_props: rf_call_kwargs['mu_oil'] = fluid_props['oil_viscosity_cp']
                    if 'gas_viscosity_cp' in fluid_props: rf_call_kwargs['mu_co2'] = fluid_props['gas_viscosity_cp']
            except Exception as e_eos:
                logger.warning(f"EOS calculation failed at P={pressure}psi. Using fallback properties. Error: {e_eos}")
        
        # Pass all optimizable parameters to the kwargs dict for maximum flexibility
        rf_call_kwargs.update(eor_operational_params_dict)

        # --- [FIX] ---
        # Remove keys that are passed as explicit positional arguments to recovery_factor
        # to avoid the "multiple values for argument" TypeError.
        # The signature is: recovery_factor(pressure, rate, porosity, mmp, ...)
        rf_call_kwargs.pop('pressure', None)
        rf_call_kwargs.pop('rate', None)
        rf_call_kwargs.pop('porosity', None) # Also remove porosity in case it's an unlocked param

        effective_co2_rate_for_rf = base_injection_rate
        if self.eor_params.injection_scheme == 'wag':
            water_fraction_rf = eor_operational_params_dict.get('water_fraction', 0.5)
            effective_co2_rate_for_rf = base_injection_rate * (1.0 - water_fraction_rf)

        rf_val = recovery_factor(
            pressure,
            effective_co2_rate_for_rf,
            porosity_to_use,
            mmp_to_use,
            model=self.recovery_model,
            **rf_call_kwargs
        )

        if "recovery_factor" in target_objectives: results["recovery_factor"] = rf_val

        original_econ_params_for_eval = self.economic_params
        self.economic_params = current_econ_params
        try:
            if any(obj in target_objectives for obj in ["npv", "co2_utilization"]):
                (annual_oil, co2_purchased, co2_recycled, _co2_total_inj, 
                 water_inj, water_disp) = self._calculate_annual_profiles(rf_val, eor_operational_params_dict, ooip_to_use)

                if "npv" in target_objectives:
                    results["npv"] = self._calculate_npv(annual_oil, co2_purchased, co2_recycled, water_inj, water_disp)
                if "co2_utilization" in target_objectives:
                    results["co2_utilization"] = self._calculate_co2_utilization_factor(annual_oil, co2_purchased)
        finally:
            self.economic_params = original_econ_params_for_eval

        return results

    def _objective_function_wrapper(self, **params_dict: float) -> float:
        eval_params = params_dict.copy()
        
        porosity_override = eval_params.pop('porosity', None)
        ooip_override = eval_params.pop('ooip_stb', None)
        
        temp_econ_params = deepcopy(self.economic_params)
        is_econ_updated = False
        if 'oil_price_usd_per_bbl' in eval_params:
            temp_econ_params.oil_price_usd_per_bbl = eval_params.pop('oil_price_usd_per_bbl')
            is_econ_updated = True
        if 'co2_purchase_cost_usd_per_tonne' in eval_params:
            temp_econ_params.co2_purchase_cost_usd_per_tonne = eval_params.pop('co2_purchase_cost_usd_per_tonne')
            is_econ_updated = True
        
        target_name = self.operational_params.target_objective_name
        target_value = self.operational_params.target_objective_value
        
        if target_name and target_value is not None and target_value > 0:
            eval_results = self.evaluate_for_analysis(
                eor_operational_params_dict=eval_params,
                economic_params_override=temp_econ_params if is_econ_updated else None,
                avg_porosity_override=porosity_override, ooip_override=ooip_override,
                target_objectives=[target_name]
            )
            current_value = eval_results.get(target_name, 0.0)
            base_objective = -((current_value - target_value) ** 2)
        else:
            eval_results = self.evaluate_for_analysis(
                eor_operational_params_dict=eval_params,
                economic_params_override=temp_econ_params if is_econ_updated else None,
                avg_porosity_override=porosity_override, ooip_override=ooip_override,
                target_objectives=[self.chosen_objective]
            )
            objective_value = eval_results.get(self.chosen_objective, -float('inf'))
            base_objective = -objective_value if self.chosen_objective == "co2_utilization" else objective_value
        
        penalty = 0.0
        pressure = eval_params.get('pressure', 0.0)
        max_pressure = self.eor_params.max_pressure_psi
        if pressure > max_pressure:
            penalty += 1000 * (pressure - max_pressure)
        
        return base_objective - penalty

    def _calculate_total_volume_for_qp(self, q_initial_peak_rate: float, years: int, profile_params: ProfileParameters, profile_type_override: Optional[str] = None) -> float:
        return 0.0

    def _generate_oil_production_profile(self, total_oil_to_produce_stb: float) -> np.ndarray:
        project_life_days = int(self.operational_params.project_lifetime_years * DAYS_PER_YEAR)
        if project_life_days <= 0 or total_oil_to_produce_stb <= 0:
            return np.zeros(project_life_days)

        p = self.profile_params
        
        plateau_frac = p.plateau_duration_fraction_of_life or 0.3
        remaining_frac = 1.0 - plateau_frac
        ramp_up_frac = remaining_frac / 2.0

        t_ramp_up_days = int(ramp_up_frac * project_life_days)
        t_plateau_days = int(plateau_frac * project_life_days)
        t_decline_days = project_life_days - t_ramp_up_days - t_plateau_days

        if t_decline_days < 0:
            t_decline_days = 0
            t_plateau_days = project_life_days - t_ramp_up_days

        denominator = (0.5 * t_ramp_up_days) + t_plateau_days + (0.5 * t_decline_days)
        if denominator <= 0: return np.zeros(project_life_days)
        
        peak_rate = total_oil_to_produce_stb / denominator
        
        ramp_up = np.linspace(0, peak_rate, t_ramp_up_days) if t_ramp_up_days > 0 else np.array([])
        plateau = np.full(t_plateau_days, peak_rate) if t_plateau_days > 0 else np.array([])
        decline = np.linspace(peak_rate, 0, t_decline_days) if t_decline_days > 0 else np.array([])
        
        profile = np.concatenate((ramp_up, plateau, decline))
        
        if len(profile) < project_life_days:
            profile = np.pad(profile, (0, project_life_days - len(profile)), 'constant')
        
        return profile[:project_life_days]
    
    def _generate_injection_profiles(self, primary_injectant_daily_rate: float, wag_water_fraction_of_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        project_life_days = int(self.operational_params.project_lifetime_years * DAYS_PER_YEAR)
        
        if self.eor_params.injection_scheme == 'wag' and wag_water_fraction_of_time is not None:
            co2_rate = primary_injectant_daily_rate * (1.0 - wag_water_fraction_of_time)
            water_rate = primary_injectant_daily_rate * wag_water_fraction_of_time
        else:
            co2_rate = primary_injectant_daily_rate
            water_rate = 0.0

        co2_inj_profile = np.full(project_life_days, co2_rate)
        water_inj_profile = np.full(project_life_days, water_rate)
        
        return co2_inj_profile, water_inj_profile

    def _calculate_annual_profiles(self, current_recovery_factor: float, optimized_params_dict: Dict[str, float], ooip_stb: float) -> Tuple[np.ndarray, ...]:
        total_oil = ooip_stb * current_recovery_factor
        daily_oil = self._generate_oil_production_profile(total_oil)
        
        project_life_days = int(self.operational_params.project_lifetime_years * DAYS_PER_YEAR)
        if len(daily_oil) != project_life_days:
             daily_oil.resize(project_life_days, refcheck=False)
        
        inj_rate = optimized_params_dict.get('rate', 0.0)
        water_frac = optimized_params_dict.get('water_fraction', 0.0) if self.eor_params.injection_scheme == 'wag' else None
        daily_co2_inj, daily_water_inj = self._generate_injection_profiles(inj_rate, water_frac)

        gor_factor = np.linspace(0, 1, project_life_days)
        daily_co2_recycled = daily_oil * self.eor_params.gas_oil_ratio_at_breakthrough * gor_factor
        daily_co2_purchased = np.maximum(0, daily_co2_inj - daily_co2_recycled)
        
        wc_factor = np.linspace(0, 1, project_life_days)**2
        daily_water_produced = daily_oil * self.eor_params.water_cut_bwow * wc_factor
        
        num_years = self.operational_params.project_lifetime_years
        annual_profiles: List[np.ndarray] = []
        for daily_profile in [daily_oil, daily_co2_purchased, daily_co2_recycled, daily_co2_inj, daily_water_inj, daily_water_produced]:
            annual_total = np.array([np.sum(daily_profile[int(i*DAYS_PER_YEAR):int((i+1)*DAYS_PER_YEAR)]) for i in range(num_years)])
            annual_profiles.append(annual_total)
            
        return tuple(annual_profiles)

    def _calculate_npv(self,
                       annual_oil_stb: np.ndarray,
                       annual_co2_purchased_mscf: np.ndarray,
                       annual_co2_recycled_mscf: np.ndarray,
                       annual_water_injected_bbl: np.ndarray,
                       annual_water_disposed_bbl: np.ndarray) -> float:
        if len(annual_oil_stb) == 0: return -1e12

        p = self.economic_params
        
        revenue = annual_oil_stb * p.oil_price_usd_per_bbl
        
        cost_co2_purchase = annual_co2_purchased_mscf * self.co2_density_tonne_per_mscf * p.co2_purchase_cost_usd_per_tonne
        cost_co2_recycle = annual_co2_recycled_mscf * self.co2_density_tonne_per_mscf * p.co2_recycle_cost_usd_per_tonne
        cost_water_inject = annual_water_injected_bbl * p.water_injection_cost_usd_per_bbl
        cost_water_dispose = annual_water_disposed_bbl * p.water_disposal_cost_usd_per_bbl
        
        opex_fixed = p.fixed_opex_usd_per_year
        opex_variable = annual_oil_stb * p.variable_opex_usd_per_bbl
        
        cash_flow = revenue - (cost_co2_purchase + cost_co2_recycle + cost_water_inject + cost_water_dispose + opex_fixed + opex_variable)
        
        cash_flow_with_capex = np.insert(cash_flow, 0, -p.capex_usd)
        
        npv_result = npv(p.discount_rate_fraction, cash_flow_with_capex)
        return float(npv_result)


    def _calculate_co2_utilization_factor(self, annual_oil_stb: np.ndarray, annual_co2_purchased_mscf: np.ndarray) -> float:
        total_oil_produced = np.sum(annual_oil_stb)
        total_co2_purchased = np.sum(annual_co2_purchased_mscf)
        
        if total_oil_produced > 1e-6:
            return total_co2_purchased / total_oil_produced
        else:
            return 1e12

    def _get_ga_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        mmp_val = self.mmp or 2500.0
        min_p = np.clip(self.eor_params.target_pressure_psi, mmp_val * 1.01, self.eor_params.max_pressure_psi - 1.0)
        
        b = {'pressure': (min_p, self.eor_params.max_pressure_psi),
             'rate': (self.eor_params.min_injection_rate_bpd, self.eor_params.max_injection_rate_bpd)}
        
        # Always optimize a wide range of physics-based EOR parameters
        b['v_dp_coefficient'] = (0.3, 0.8)
        b['mobility_ratio'] = (0.8, 5.0)
        b['gravity_factor'] = (0.05, 0.5)
        b['sor'] = (0.15, 0.35)
        
        if self.recovery_model == 'hybrid':
            b['transition_alpha'] = (0.8, 1.2)
            b['transition_beta'] = (10.0, 40.0)

        if self.eor_params.injection_scheme == 'wag': 
            b['cycle_length_days'] = (self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days)
            b['water_fraction'] = (self.eor_params.min_water_fraction, self.eor_params.max_water_fraction)
            b['WAG_ratio'] = (0.1, 5.0)
            
        for param_key in self._unlocked_params_for_current_run:
            constraint_info = self.RELAXABLE_CONSTRAINTS.get(param_key)
            if not constraint_info: continue

            base_val = None
            if constraint_info['type'] == 'reservoir':
                base_val = getattr(self._base_reservoir_data, param_key, None)
                if base_val is None and param_key == 'porosity': 
                    poro_arr = self._base_reservoir_data.grid.get('PORO', np.array([0.15]))
                    base_val = np.mean(poro_arr) if hasattr(poro_arr, 'size') and poro_arr.size > 0 else 0.15
            elif constraint_info['type'] == 'economic':
                base_val = getattr(self._base_economic_params, param_key, None)
            elif constraint_info['type'] == 'eor':
                base_val = getattr(self._base_eor_params, param_key, None)
            
            if base_val is not None:
                range_factor = constraint_info['range_factor']
                b[param_key] = (base_val * (1 - range_factor), base_val * (1 + range_factor))
                logger.info(f"Re-run: Overriding bounds for unlocked param '{param_key}' to: ({b[param_key][0]:.3g}, {b[param_key][1]:.3g})")

        return b

    def _get_objective_name_for_logging(self) -> str:
        target_name = self.operational_params.target_objective_name
        target_value = self.operational_params.target_objective_value
        if target_name and target_value is not None and target_value > 0:
            return f"Match Target ({target_name.replace('_', ' ').title()} = {target_value:.3f})"
        return self.chosen_objective.replace('_', ' ').title()

    def _handle_target_miss_reporting(self, final_eval_results: Dict[str, float], final_results_dict: Dict[str, Any], handle_target_miss_flag: bool) -> Dict[str, Any]:
        target_name = self.operational_params.target_objective_name
        target_value = self.operational_params.target_objective_value
        
        final_results_dict['target_was_unreachable'] = False
        final_results_dict['target_objective_name_in_run'] = target_name
        final_results_dict['target_objective_value_in_run'] = target_value
        final_results_dict['unlocked_params_in_run'] = self._unlocked_params_for_current_run
        
        is_target_mode = target_name and target_value is not None and target_value > 0
        
        if is_target_mode:
            final_achieved = final_eval_results.get(target_name, 0.0)
            final_results_dict['final_target_value_achieved'] = final_achieved
            relative_error = abs(final_achieved - target_value) / (abs(target_value) + 1e-9)
            
            if relative_error > 0.05 and handle_target_miss_flag:
                logger.warning(f"Target for {target_name} ({target_value:.3f}) is unreachable. Closest value: {final_achieved:.4f}.")
                final_results_dict['target_was_unreachable'] = True

        return final_results_dict

    def _get_complete_params_from_ga_individual(self, individual_dict: Dict[str, float], param_bounds_for_clipping: Dict[str, Tuple[float,float]] ) -> Dict[str, float]:
        return {name:np.clip(individual_dict.get(name,random.uniform(low,high)),low,high) for name,(low,high) in param_bounds_for_clipping.items()}

    def _apply_fitness_sharing(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config: GeneticAlgorithmParams, bounds: Dict[str, Tuple[float, float]]) -> List[float]:
        pop_size = len(population)
        if pop_size == 0:
            return []

        normalized_pop = []
        param_names = list(bounds.keys())
        for ind in population:
            norm_ind = {}
            for name in param_names:
                low, high = bounds[name]
                val = ind.get(name, low)
                norm_ind[name] = (val - low) / (high - low) if high > low else 0.0
            normalized_pop.append(norm_ind)

        niche_counts = np.ones(pop_size)
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                dist_sq = sum((normalized_pop[i][name] - normalized_pop[j][name])**2 for name in param_names)
                dist = np.sqrt(dist_sq)

                if dist < ga_config.sharing_sigma_threshold:
                    sh = 1.0 - (dist / ga_config.sharing_sigma_threshold)
                    niche_counts[i] += sh
                    niche_counts[j] += sh
        
        adjusted_fitnesses = [fit / count for fit, count in zip(fitnesses, niche_counts)]
        return adjusted_fitnesses

    def _inject_random_individuals(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config: GeneticAlgorithmParams, bounds: Dict[str, Tuple[float, float]]) -> Tuple[List[Dict[str, float]], List[float]]:
        num_to_replace = int(ga_config.population_size * ga_config.random_injection_rate)
        if num_to_replace == 0:
            return population, fitnesses
            
        worst_indices = np.argsort(fitnesses)[:num_to_replace]
        
        for idx in worst_indices:
            population[idx] = {name: random.uniform(low, high) for name, (low, high) in bounds.items()}
            fitnesses[idx] = -np.inf 
            
        logger.debug(f"GA injected {num_to_replace} new random individuals.")
        return population, fitnesses

    def _tournament_selection_ga(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        selected, pop_len = [], len(population)
        
        elite_count = ga_config.elite_count 
        if elite_count > 0 and pop_len > 0: 
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            selected.extend(deepcopy(population[idx]) for idx in elite_indices)
        
        while len(selected) < pop_len:
            if pop_len == 0: break
            t_size = min(ga_config.tournament_size, pop_len)
            if t_size <= 0: continue
            
            tournament_indices = random.sample(range(pop_len), t_size)
            winner_index = max(tournament_indices, key=lambda i: fitnesses[i])
            selected.append(deepcopy(population[winner_index]))
        return selected

    def _crossover_ga(self, parent_population: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        offspring, n_parents = [], len(parent_population)
        if n_parents == 0: return []
        random.shuffle(parent_population)
        
        elites = parent_population[:ga_config.elite_count]
        mating_pool = parent_population[ga_config.elite_count:]
        offspring.extend(deepcopy(elites))
        
        n_mating = len(mating_pool)
        for i in range(0, n_mating - 1, 2):
            p1, p2 = mating_pool[i], mating_pool[i+1]
            c1, c2 = deepcopy(p1), deepcopy(p2)
            if random.random() < ga_config.crossover_rate:
                alpha = ga_config.blend_alpha_crossover
                for k in set(p1.keys()) & set(p2.keys()):
                    v1, v2 = p1[k], p2[k]
                    c1[k] = alpha * v1 + (1 - alpha) * v2
                    c2[k] = (1 - alpha) * v1 + alpha * v2
            offspring.extend([c1, c2])
            
        if len(offspring) < n_parents: offspring.extend(deepcopy(parent_population[len(offspring):]))
        return offspring[:n_parents]

    def _mutate_ga(self, population_to_mutate: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        mutated_pop, param_bounds = [], self._get_ga_parameter_bounds()
        
        elites = population_to_mutate[:ga_config.elite_count]
        non_elites = population_to_mutate[ga_config.elite_count:]
        mutated_pop.extend(deepcopy(elites))

        for ind in non_elites:
            mutated_ind = deepcopy(ind)
            if random.random() < ga_config.mutation_rate:
                gene_to_mutate = random.choice(list(mutated_ind.keys()))
                if gene_to_mutate in param_bounds:
                    low, high = param_bounds[gene_to_mutate]
                    current_val = mutated_ind.get(gene_to_mutate, (low + high) / 2.0)
                    gene_range = high - low
                    
                    sigma = max(gene_range * ga_config.mutation_strength_factor, 1e-7)
                    
                    if ga_config.use_chaotic_mutation:
                        self._chaos_state = ga_config.chaos_map_r * self._chaos_state * (1 - self._chaos_state)
                        chaotic_factor = 0.5 + self._chaos_state
                        sigma *= chaotic_factor

                    mutation = random.gauss(0, sigma)
                    mutated_ind[gene_to_mutate] = np.clip(current_val + mutation, low, high)
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def _adapt_ga_parameters(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config_mutable: GeneticAlgorithmParams, base_ga_config: GeneticAlgorithmParams):
        # --- [MODIFIED] New stagnation detection and logging logic ---
        is_stagnation_period = False
        if len(self._best_fitness_history) > base_ga_config.stagnation_generations_limit:
            recent_best = self._best_fitness_history[-1]
            past_best = self._best_fitness_history[-base_ga_config.stagnation_generations_limit]
            if np.isclose(recent_best, past_best):
                is_stagnation_period = True

        if is_stagnation_period and not self._is_stagnated:
            logger.warning(f"GA stagnation detected: Best fitness unchanged for {base_ga_config.stagnation_generations_limit} generations. Activating adaptive measures.")
            self._is_stagnated = True
        elif not is_stagnation_period and self._is_stagnated:
            logger.info("GA has overcome stagnation. Deactivating adaptive measures.")
            self._is_stagnated = False
        
        if len(fitnesses) > 1:
            fitness_std_dev = np.std(fitnesses)
            fitness_range = np.max(fitnesses) - np.min(fitnesses)
            diversity_metric = fitness_std_dev / (fitness_range + 1e-9)
        else:
            diversity_metric = 0.0

        if base_ga_config.adaptive_mutation_enabled:
            current_rate = ga_config_mutable.mutation_rate
            new_rate = current_rate
            if self._is_stagnated:
                new_rate = min(current_rate * 1.5, base_ga_config.max_mutation_rate)
                if not np.isclose(new_rate, current_rate):
                    logger.info(f"Adaptive Action: Increasing mutation rate from {current_rate:.4f} to {new_rate:.4f}.")
            elif diversity_metric < 0.1:
                new_rate = min(current_rate * 1.1, base_ga_config.max_mutation_rate)
            else:
                new_rate = max(current_rate * 0.95, base_ga_config.mutation_rate)
            ga_config_mutable.mutation_rate = np.clip(new_rate, base_ga_config.min_mutation_rate, base_ga_config.max_mutation_rate)

        if base_ga_config.dynamic_elitism_enabled:
            current_elites = ga_config_mutable.elite_count
            new_elites = current_elites
            if self._is_stagnated:
                new_elites = max(base_ga_config.min_elite_count, current_elites - 1)
                if new_elites != current_elites:
                    logger.info(f"Adaptive Action: Reducing elite count from {current_elites} to {new_elites}.")
            else:
                new_elites = base_ga_config.elite_count
            ga_config_mutable.elite_count = new_elites

    def _smart_select_recovery_model(self):
        """
        Automatically selects 'koval' model if its specific parameters are being optimized
        and the current model is the default 'hybrid' to prevent stagnation.
        """
        if hasattr(self.operational_params, 'recovery_model_selection'):
            return 

        if self.recovery_model != 'hybrid':
            return

        optimization_params = self._get_ga_parameter_bounds().keys()
        if 'v_dp_coefficient' in optimization_params or 'mobility_ratio' in optimization_params:
            logger.info("Auto-switching to 'koval' recovery model because its parameters are being optimized.")
            self.recovery_model = 'koval'

    def optimize_genetic_algorithm(self, ga_params_override: Optional[GeneticAlgorithmParams] = None, handle_target_miss: bool = False, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        self._refresh_operational_parameters()
        self._smart_select_recovery_model()
        
        base_cfg = ga_params_override or self.ga_params_default_config
        current_cfg = deepcopy(base_cfg)

        objective_for_log = self._get_objective_name_for_logging()
        
        progress_callback = kwargs.get('progress_callback')
        worker_is_running_check = kwargs.get('worker_is_running_check', lambda: True)
        
        logger.info(f"GA (Advanced): Obj '{objective_for_log}', Gens {base_cfg.generations}, Pop {base_cfg.population_size}")
        bounds = self._get_ga_parameter_bounds()
        pop = [{name: random.uniform(low, high) for name, (low, high) in bounds.items()} for _ in range(base_cfg.population_size)]
        
        best_sol = pop[0].copy() if pop else {}
        best_fit = -np.inf
        
        evaluate_individual_partial = partial(self._objective_function_wrapper)

        for gen in range(base_cfg.generations):
            if not worker_is_running_check():
                logger.info(f"GA run cancelled by worker at generation {gen+1}.")
                break
            
            with ProcessPoolExecutor() as executor:
                future_fitnesses = [executor.submit(_run_objective_wrapper, evaluate_individual_partial, p) for p in pop]
                fitnesses = [f.result() for f in future_fitnesses]

            idx_best_this_gen = np.argmax(fitnesses)
            if fitnesses[idx_best_this_gen] > best_fit:
                best_fit = fitnesses[idx_best_this_gen]
                best_sol = pop[idx_best_this_gen].copy()
            self._best_fitness_history.append(best_fit)

            if progress_callback:
                progress_callback({
                    'generation': gen + 1, 'best_fitness': best_fit,
                    'avg_fitness': np.mean(fitnesses), 'worst_fitness': np.min(fitnesses)
                })

            if current_cfg.use_fitness_sharing:
                fitnesses = self._apply_fitness_sharing(pop, fitnesses, current_cfg, bounds)
            
            if current_cfg.random_injection_rate > 0:
                pop, fitnesses = self._inject_random_individuals(pop, fitnesses, current_cfg, bounds)

            parents = self._tournament_selection_ga(pop, fitnesses, current_cfg)
            offspring = self._crossover_ga(parents, current_cfg)
            pop = self._mutate_ga(offspring, current_cfg)
            self._adapt_ga_parameters(pop, fitnesses, current_cfg, base_cfg)

            if (gen+1)%10==0 or gen==base_cfg.generations-1: 
                logger.info(f"GA Gen {gen+1} BestFit: Gen:{fitnesses[idx_best_this_gen]:.4e} Overall:{best_fit:.4e}")
        
        final_params = {name: np.clip(best_sol.get(name, 0), low, high) for name, (low, high) in bounds.items()}
        self.update_parameters(final_params)
        
        with ProcessPoolExecutor() as executor:
            final_future_fitnesses = [executor.submit(_run_objective_wrapper, evaluate_individual_partial, p) for p in pop]
            final_fitnesses = [f.result() for f in final_future_fitnesses]
        
        sorted_final_pop = sorted(zip(pop, final_fitnesses), key=lambda x: x[1], reverse=True)
        top_solutions = [{'params': p, 'fitness': f} for p, f in sorted_final_pop[:base_cfg.elite_count]]

        final_eval = self.evaluate_for_analysis(final_params, target_objectives=['recovery_factor', 'npv', 'co2_utilization'])
        
        self._results = {
            'optimized_params_final_clipped': final_params, 'objective_function_value': best_fit,
            'chosen_objective': objective_for_log, 'final_recovery_factor_reported': final_eval.get('recovery_factor', 0.0),
            'final_npv_reported': final_eval.get('npv'), 'mmp_psi': self.mmp, 'method': 'genetic_algorithm',
            'generations': base_cfg.generations, 'population_size': base_cfg.population_size,
            'top_ga_solutions_from_final_pop': top_solutions
        }
        
        self._results = self._handle_target_miss_reporting(final_eval, self._results, handle_target_miss)
        logger.info(f"GA done. Best obj ({objective_for_log}): {best_fit:.4e}")
        return self._results

    def optimize_bayesian(self, n_iter_override: Optional[int]=None, init_points_override: Optional[int]=None, method_override: Optional[str]=None, initial_solutions_from_ga: Optional[List[Dict[str,Any]]]=None, handle_target_miss: bool = False, **kwargs) -> Dict[str,Any]:
        self.reset_to_base_state()
        self._refresh_operational_parameters()
        self._smart_select_recovery_model()
        bo_params = self.bo_params_default_config
        n_i = n_iter_override if n_iter_override is not None else bo_params.n_iterations
        init_r = init_points_override if init_points_override is not None else bo_params.n_initial_points
        objective_for_log = self._get_objective_name_for_logging()
        
        logger.info(f"BayesOpt: Obj '{objective_for_log}', Iter {n_i}, InitRand {init_r}")
        p_bounds_d = self._get_ga_parameter_bounds()
        
        pb_bayes = {n: (l, h) for n, (l, h) in p_bounds_d.items()}
        
        bayes_o = BayesianOptimization(f=self._objective_function_wrapper, pbounds=pb_bayes, random_state=42, verbose=2)
        
        if initial_solutions_from_ga:
            logger.info(f"Probing {len(initial_solutions_from_ga)} initial points from GA for Bayesian Optimization.")
            for sol in initial_solutions_from_ga:
                params_to_probe = sol.get('params', {})
                if all(name in params_to_probe for name in pb_bayes.keys()):
                     bayes_o.probe(params=params_to_probe, lazy=True)
                else:
                    logger.warning("Skipping GA solution for BO probing due to mismatched parameters.")

        bayes_o.maximize(init_points=init_r, n_iter=n_i)
        
        best_p_bo = bayes_o.max['params']
        final_obj_bo = bayes_o.max['target']
        
        self.update_parameters(best_p_bo)
        final_eval = self.evaluate_for_analysis(best_p_bo, target_objectives=['recovery_factor', 'npv', 'co2_utilization'])
        
        self._results = {
            'optimized_params_final_clipped': best_p_bo, 'objective_function_value': final_obj_bo,
            'chosen_objective': objective_for_log, 'final_recovery_factor_reported': final_eval.get('recovery_factor', 0.0),
            'final_npv_reported': final_eval.get('npv'), 'mmp_psi': self.mmp, 'method': f'bayesian_gp',
            'iterations_bo_actual': n_i, 'initial_points_bo_random_requested': init_r,
        }
        
        self._results = self._handle_target_miss_reporting(final_eval, self._results, handle_target_miss)
        logger.info(f"Bayesian done. Best obj ({objective_for_log}): {final_obj_bo:.4e}")
        return self._results

    def hybrid_optimize(self, ga_params_override: Optional[GeneticAlgorithmParams]=None, n_iter_override: Optional[int]=None, init_points_override: Optional[int]=None, handle_target_miss: bool = False, **kwargs) -> Dict[str,Any]:
        ga_p_h = ga_params_override or self.ga_params_default_config
        
        logger.info(f"Hybrid Opt: Starting GA Phase. Gens:{ga_p_h.generations}, Pop:{ga_p_h.population_size}")
        ga_res = self.optimize_genetic_algorithm(ga_params_override=ga_p_h, handle_target_miss=False, **kwargs)
        
        bo_i_h = n_iter_override if n_iter_override is not None else self.bo_params_default_config.n_iterations
        bo_init_h = init_points_override if init_points_override is not None else self.bo_params_default_config.n_initial_points

        init_bo_sols = ga_res.get('top_ga_solutions_from_final_pop')

        logger.info(f"Hybrid Opt: Starting BO Phase. Iter {bo_i_h}, RandInit {bo_init_h}")
        bo_res = self.optimize_bayesian(n_iter_override=bo_i_h, init_points_override=bo_init_h, handle_target_miss=handle_target_miss, initial_solutions_from_ga=init_bo_sols, **kwargs)
        
        self._results = {**bo_res, 'ga_full_results_for_hybrid': ga_res, 'method': 'hybrid_ga_bo'}
        logger.info(f"Hybrid opt done. Final obj ({self._results.get('chosen_objective')}): {self._results.get('objective_function_value'):.4e}")
        return self._results

    def plot_mmp_profile(self) -> Optional[go.Figure]:
        if not (self.well_analysis and hasattr(self.well_analysis, 'calculate_mmp_profile')): return None
        try:
            profile_data = self.well_analysis.calculate_mmp_profile()
            if not all(k in profile_data for k in ['depths','mmp']) or not profile_data['depths'].size or not profile_data['mmp'].size: return None
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=profile_data['mmp'], y=profile_data['depths'], name='MMP (psi)', line_color='blue'), secondary_y=False)
            if 'temperature' in profile_data and profile_data['temperature'].size > 0: fig.add_trace(go.Scatter(x=profile_data['temperature'], y=profile_data['depths'], name='Temp (°F)', line_color='red'), secondary_y=True)
            fig.update_layout(title_text='MMP vs Depth Profile', yaxis_title_text='Depth (ft)', legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            fig.update_yaxes(title_text="Depth (ft)", secondary_y=False, autorange="reversed")
            if 'temperature' in profile_data and profile_data['temperature'].size > 0: fig.update_yaxes(title_text="Temp Axis", secondary_y=True, autorange="reversed", overlaying='y', side='right', showticklabels=True); fig.update_xaxes(title_text="Value (MMP or Temp)")
            else: fig.update_xaxes(title_text="MMP (psi)")
            return fig
        except Exception as e: 
            logger.error(f"Error generating MMP plot: {e}", exc_info=True)
            return None

    def plot_optimization_convergence(self, results_to_plot: Optional[Dict[str,Any]] = None) -> Optional[go.Figure]:
        source = results_to_plot or self._results
        if not source: return None
        fig = go.Figure()

        title = f'Optimization Convergence for {source.get("chosen_objective","Obj").replace("_"," ").title()}'
        y_title = f'Objective Value'
        x_title='Evaluations'
        
        ga_results = source.get('ga_full_results_for_hybrid') or (source if source.get('method') == 'genetic_algorithm' else None)
        if ga_results and 'best_fitness_history' in ga_results:
            history = ga_results['best_fitness_history']
            gen_count = len(history)
            pop_size = ga_results.get('population_size', 1)
            evals = np.arange(1, gen_count + 1) * pop_size
            fig.add_trace(go.Scatter(x=evals, y=history, mode='lines', name='GA Best Fitness'))
            last_ga_eval = evals[-1]
        else:
            last_ga_eval = 0

        if 'bayes_opt_obj' in source:
             bo = source['bayes_opt_obj']
             bo_x = np.arange(last_ga_eval + 1, last_ga_eval + 1 + len(bo.Y))
             fig.add_trace(go.Scatter(x=bo_x, y=bo.Y, mode='markers', name='BO Evaluations'))
        else:
            obj_val = source.get('objective_function_value')
            if obj_val is not None:
                fig.add_trace(go.Scatter(x=[last_ga_eval + 1], y=[obj_val], mode='markers', name='Final Value', marker=dict(size=12, symbol='star')))

        fig.update_layout(title_text=title, xaxis_title_text=x_title, yaxis_title_text=y_title)
        return fig


    def plot_parameter_sensitivity(self, param_name_for_sensitivity: str, results_to_use_for_plot: Optional[Dict[str,Any]]=None) -> Optional[go.Figure]:
        source = results_to_use_for_plot or self._results
        num_pts_sens = 20
        if not (source and 'optimized_params_final_clipped' in source and isinstance(source['optimized_params_final_clipped'], dict)): return None
        
        opt_base = source['optimized_params_final_clipped'].copy()
        obj_name_sens = self.chosen_objective
        
        if param_name_for_sensitivity in self.RELAXABLE_CONSTRAINTS and param_name_for_sensitivity in opt_base:
            curr_opt_val_param = opt_base[param_name_for_sensitivity]
            bounds_ref = self._get_ga_parameter_bounds()
            low_b, high_b = bounds_ref.get(param_name_for_sensitivity, (curr_opt_val_param * 0.8, curr_opt_val_param * 1.2))
        elif param_name_for_sensitivity in opt_base:
            curr_opt_val_param = opt_base[param_name_for_sensitivity]
            bounds_ref = self._get_ga_parameter_bounds()
            low_b, high_b = bounds_ref[param_name_for_sensitivity]
        elif hasattr(self.economic_params, param_name_for_sensitivity):
            curr_opt_val_param = getattr(self.economic_params, param_name_for_sensitivity)
            low_b, high_b = curr_opt_val_param * 0.8, curr_opt_val_param * 1.2
        else:
            logger.warning(f"Sensitivity plot: Parameter '{param_name_for_sensitivity}' not found.")
            return None

        param_vals_sweep=np.linspace(low_b, high_b, num_pts_sens)
        obj_vals_sens=[]

        for p_sweep in param_vals_sweep:
            temp_params_for_eval = opt_base.copy()
            
            if param_name_for_sensitivity in temp_params_for_eval:
                temp_params_for_eval[param_name_for_sensitivity] = p_sweep
            
            temp_econ_override = deepcopy(self.economic_params)
            if hasattr(temp_econ_override, param_name_for_sensitivity):
                 setattr(temp_econ_override, param_name_for_sensitivity, p_sweep)
            else:
                 temp_econ_override = None

            eval_res = self.evaluate_for_analysis(
                eor_operational_params_dict=temp_params_for_eval, 
                economic_params_override=temp_econ_override,
                target_objectives=[obj_name_sens]
            )
            obj_vals_sens.append(eval_res.get(obj_name_sens, np.nan))

        fig=go.Figure()
        fig.add_trace(go.Scatter(x=param_vals_sweep, y=obj_vals_sens, mode='lines+markers', name=f'{param_name_for_sensitivity} effect'))
        fig.add_vline(x=curr_opt_val_param, line=dict(width=2, dash="dash", color="green"), annotation_text="Opt. Value", annotation_position="top left")
        title_param_disp = param_name_for_sensitivity.replace("_"," ").title()
        y_title_disp = obj_name_sens.replace("_"," ").title()
        fig.update_layout(title_text=f'{y_title_disp} vs. {title_param_disp}', xaxis_title_text=title_param_disp, yaxis_title_text=y_title_disp)
        return fig