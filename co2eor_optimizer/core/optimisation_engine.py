from typing import Callable, Dict, List, Optional, Any, Tuple, Union, Type
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
from copy import deepcopy
import dataclasses

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

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
        'oil_price_usd_per_bbl': {
            'description': 'Market price for crude oil ($/bbl)',
            'range_factor': 0.30,
            'type': 'economic'
        },
        'co2_purchase_cost_usd_per_tonne': {
            'description': 'Cost to purchase virgin CO2 ($/tonne)',
            'range_factor': 0.40,
            'type': 'economic'
        },
        # --- [NEW] Additional EOR parameters to unlock ---
        'v_dp_coefficient': {
            'description': 'Dykstra-Parsons coefficient for heterogeneity',
            'range_factor': 0.35, # Varies significantly, allow a wider range
            'type': 'eor'
        },
        'mobility_ratio': {
            'description': 'Mobility Ratio (M)',
            'range_factor': 0.50, # Highly sensitive, allow wide exploration
            'type': 'eor'
        },
        'WAG_ratio': {
            'description': 'Water-Alternating-Gas Ratio',
            'range_factor': 0.60, # Very wide range, as it's a key design parameter
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
                 well_analysis: Optional[Any] = None,
                 avg_porosity_init_override: Optional[float] = None,
                 mmp_init_override: Optional[float] = None,
                 recovery_model_init_kwargs_override: Optional[Dict[str, Any]] = None):

        self._base_reservoir_data = deepcopy(reservoir)
        self._base_pvt_data = deepcopy(pvt)
        self._base_eor_params = deepcopy(eor_params_instance or EORParameters())
        self._base_economic_params = deepcopy(economic_params_instance or EconomicParameters())
        self._base_operational_params = deepcopy(operational_params_instance or OperationalParameters())
        
        self.well_analysis = well_analysis
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

        if self._mmp_value is None: self.calculate_mmp()

    def reset_to_base_state(self):
        logger.info("Resetting OptimizationEngine to its base state.")
        self.reservoir = deepcopy(self._base_reservoir_data)
        self.pvt = deepcopy(self._base_pvt_data)
        self.eor_params = deepcopy(self._base_eor_params)
        self.economic_params = deepcopy(self._base_economic_params)
        self.operational_params = deepcopy(self._base_operational_params)
        self._unlocked_params_for_current_run = []

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
        if self._mmp_value is None: self.calculate_mmp()
        return self._mmp_value
    
    @property
    def results(self) -> Optional[Dict[str, Any]]:
        return self._results

    def _refresh_operational_parameters(self):
        # This method is now simpler; it just uses the current state.
        # The state is updated from MainWindow, so no need to reload from a config file.
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
        default_mmp_fallback = 2500.0
        
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
            return self._mmp_value

        actual_mmp_method = method_override or "auto"
        
        well_analysis_to_use = self.well_analysis
        if self.well_analysis and isinstance(self.well_analysis.well_data, list):
            if self.well_analysis.well_data:
                well_analysis_to_use = WellAnalysis(
                    well_data=self.well_analysis.well_data[0], pvt_data=self.well_analysis.pvt_data,
                    eos_model=self.well_analysis.eos_model, temperature_gradient=self.well_analysis.temperature_gradient,
                    config=self.well_analysis.config
                )
            else:
                well_analysis_to_use = None

        mmp_input_object: Union[PVTProperties, Any] = self.pvt
        source_description = "PVT data"

        if well_analysis_to_use and hasattr(well_analysis_to_use, 'get_average_mmp_params_for_engine'):
            try:
                avg_well_params = well_analysis_to_use.get_average_mmp_params_for_engine()
                if avg_well_params:
                    mmp_input_constructor_params = {
                        'temperature': avg_well_params.get('temperature', self.pvt.temperature),
                        'oil_gravity': avg_well_params.get('oil_gravity', 35.0),
                        'c7_plus_mw': avg_well_params.get('c7_plus_mw'),
                        'injection_gas_composition': avg_well_params.get('injection_gas_composition', {'CO2': 1.0}),
                        'pvt_data': self.pvt
                    }
                    mmp_input_object = self._MMPParametersDataclass(**mmp_input_constructor_params)
                    source_description = "WellAnalysis average parameters"
            except Exception as e: 
                logger.warning(f"Failed to get MMP params from WellAnalysis: {e}. Using PVT.", exc_info=True)
        
        try:
            self._mmp_value = float(self._mmp_calculator_fn(mmp_input_object, method=actual_mmp_method))
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
        
        rf_call_kwargs.update({
            'v_dp_coefficient': eor_operational_params_dict.get('v_dp_coefficient', self.eor_params.v_dp_coefficient),
            'mobility_ratio': eor_operational_params_dict.get('mobility_ratio', self.eor_params.mobility_ratio),
            'WAG_ratio': eor_operational_params_dict.get('WAG_ratio', self.eor_params.WAG_ratio)
        })

        effective_co2_rate_for_rf = base_injection_rate
        if self.eor_params.injection_scheme == 'wag':
            water_fraction_rf = eor_operational_params_dict.get('water_fraction', 0.5)
            effective_co2_rate_for_rf = base_injection_rate * (1.0 - water_fraction_rf)
            if 'cycle_length_days' in eor_operational_params_dict: rf_call_kwargs['cycle_length_days'] = eor_operational_params_dict['cycle_length_days']
            rf_call_kwargs['water_fraction'] = water_fraction_rf

        rf_val = recovery_factor(
            pressure, effective_co2_rate_for_rf, porosity_to_use, mmp_to_use,
            model=self.recovery_model, **rf_call_kwargs
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
        # Implementation unchanged
        return 0.0

    def _generate_oil_production_profile(self, total_oil_to_produce_stb: float) -> np.ndarray:
        # Implementation unchanged
        return np.array([])
    
    def _generate_injection_profiles(self, primary_injectant_daily_rate: float, wag_water_fraction_of_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Implementation unchanged
        return np.array([]), np.array([])

    def _calculate_annual_profiles(self, current_recovery_factor: float, optimized_params_dict: Dict[str, float], ooip_stb: float) -> Tuple[np.ndarray, ...]:
        # Implementation unchanged
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def _calculate_npv(self,
                       annual_oil_stb: np.ndarray,
                       annual_co2_purchased_mscf: np.ndarray,
                       annual_co2_recycled_mscf: np.ndarray,
                       annual_water_injected_bbl: np.ndarray,
                       annual_water_disposed_bbl: np.ndarray) -> float:
        # Implementation unchanged
        return 0.0

    def _calculate_co2_utilization_factor(self, annual_oil_stb: np.ndarray, annual_co2_purchased_mscf: np.ndarray) -> float:
        # Implementation unchanged
        return 0.0

    def _get_ga_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        mmp_val = self.mmp or 2500.0
        min_p = np.clip(self.eor_params.target_pressure_psi, mmp_val * 1.01, self.eor_params.max_pressure_psi - 1.0)
        
        b = {'pressure': (min_p, self.eor_params.max_pressure_psi),
             'rate': (self.eor_params.min_injection_rate_bpd, self.eor_params.max_injection_rate_bpd)}
        
        # Standard EOR params that are not always part of the unlocked set
        if 'v_dp_coefficient' not in self._unlocked_params_for_current_run:
            b['v_dp_coefficient'] = (0.3, 0.8)
        if 'mobility_ratio' not in self._unlocked_params_for_current_run:
            b['mobility_ratio'] = (0.8, 3.0)

        if self.eor_params.injection_scheme == 'wag': 
            b['cycle_length_days'] = (self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days)
            b['water_fraction'] = (self.eor_params.min_water_fraction, self.eor_params.max_water_fraction)
            if 'WAG_ratio' not in self._unlocked_params_for_current_run:
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
                logger.info(f"Re-run: Adding unlocked param '{param_key}' to bounds: ({b[param_key][0]:.3g}, {b[param_key][1]:.3g})")

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

    def _tournament_selection_ga(self, population: List[Dict[str, float]], fitnesses: List[float], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        selected, pop_len = [], len(population)
        if ga_config.elite_count > 0 and pop_len > 0: 
            elite_indices = np.argsort(fitnesses)[-ga_config.elite_count:]
            selected.extend(population[idx].copy() for idx in elite_indices)
        
        while len(selected) < pop_len:
            if pop_len == 0: break
            t_size = min(ga_config.tournament_size, pop_len)
            if t_size <= 0: continue
            tournament_indices = random.sample(range(pop_len), t_size)
            winner_index = max(tournament_indices, key=lambda i: fitnesses[i])
            selected.append(population[winner_index].copy())
        return selected

    def _crossover_ga(self, parent_population: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        offspring, n_parents = [], len(parent_population)
        if n_parents == 0: return []
        random.shuffle(parent_population)
        for i in range(0, n_parents - 1, 2):
            p1, p2 = parent_population[i], parent_population[i+1]
            c1, c2 = p1.copy(), p2.copy()
            if random.random() < ga_config.crossover_rate:
                alpha = ga_config.blend_alpha_crossover
                for k in set(p1.keys()) & set(p2.keys()):
                    v1, v2 = p1[k], p2[k]
                    c1[k] = alpha * v1 + (1 - alpha) * v2
                    c2[k] = (1 - alpha) * v1 + alpha * v2
            offspring.extend([c1, c2])
        if len(offspring) < n_parents: offspring.extend(parent_population[len(offspring):])
        return offspring[:n_parents]

    def _mutate_ga(self, population_to_mutate: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        mutated_pop, param_bounds = [], self._get_ga_parameter_bounds()
        for ind in population_to_mutate:
            mutated_ind = ind.copy()
            if random.random() < ga_config.mutation_rate:
                gene_to_mutate = random.choice(list(mutated_ind.keys()))
                if gene_to_mutate in param_bounds:
                    low, high = param_bounds[gene_to_mutate]
                    current_val = mutated_ind.get(gene_to_mutate, (low + high) / 2.0)
                    gene_range = high - low
                    sigma = max(gene_range * ga_config.mutation_strength_factor, 1e-7)
                    mutated_ind[gene_to_mutate] = np.clip(current_val + random.gauss(0, sigma), low, high)
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def optimize_genetic_algorithm(self, ga_params_override: Optional[GeneticAlgorithmParams] = None, handle_target_miss: bool = False, **kwargs) -> Dict[str, Any]:
        self._refresh_operational_parameters()
        cfg = ga_params_override or self.ga_params_default_config
        objective_for_log = self._get_objective_name_for_logging()
        
        progress_callback = kwargs.get('progress_callback')
        worker_is_running_check = kwargs.get('worker_is_running_check', lambda: True)
        
        logger.info(f"GA: Obj '{objective_for_log}', Gens {cfg.generations}, Pop {cfg.population_size}")
        bounds = self._get_ga_parameter_bounds()
        pop = [{name: random.uniform(low, high) for name, (low, high) in bounds.items()} for _ in range(cfg.population_size)]
        
        best_sol = pop[0].copy() if pop else {}
        best_fit = -np.inf
        
        # <-- FIX: Create a partial function for the executor. This binds `self` to the instance method,
        # allowing the top-level `_run_objective_wrapper` to call it correctly in a separate process.
        evaluate_individual_partial = partial(self._objective_function_wrapper)

        for gen in range(cfg.generations):
            if not worker_is_running_check():
                logger.info(f"GA run cancelled by worker at generation {gen+1}.")
                break
            
            with ProcessPoolExecutor() as executor:
                # <-- FIX: Map the top-level wrapper, passing the partial function and the data.
                future_fitnesses = [executor.submit(_run_objective_wrapper, evaluate_individual_partial, p) for p in pop]
                fitnesses = [f.result() for f in future_fitnesses]

            idx_best_this_gen = np.argmax(fitnesses)
            if fitnesses[idx_best_this_gen] > best_fit:
                best_fit = fitnesses[idx_best_this_gen]
                best_sol = pop[idx_best_this_gen].copy()

            if progress_callback:
                progress_callback({
                    'generation': gen + 1, 'best_fitness': best_fit,
                    'avg_fitness': np.mean(fitnesses), 'worst_fitness': np.min(fitnesses)
                })

            parents = self._tournament_selection_ga(pop, fitnesses, cfg)
            offspring = self._crossover_ga(parents, cfg)
            pop = self._mutate_ga(offspring, cfg)

            if (gen+1)%10==0 or gen==cfg.generations-1: 
                logger.info(f"GA Gen {gen+1} BestFit: Gen:{fitnesses[idx_best_this_gen]:.4e} Overall:{best_fit:.4e}")
        
        final_params = {name: np.clip(best_sol.get(name, 0), low, high) for name, (low, high) in bounds.items()}
        self.update_parameters(final_params)
        
        # Re-evaluate final population to find top solutions
        with ProcessPoolExecutor() as executor:
            final_future_fitnesses = [executor.submit(_run_objective_wrapper, evaluate_individual_partial, p) for p in pop]
            final_fitnesses = [f.result() for f in final_future_fitnesses]
        
        sorted_final_pop = sorted(zip(pop, final_fitnesses), key=lambda x: x[1], reverse=True)
        top_solutions = [{'params': p, 'fitness': f} for p, f in sorted_final_pop[:cfg.elite_count]]

        final_eval = self.evaluate_for_analysis(final_params, target_objectives=['recovery_factor', 'npv', 'co2_utilization'])
        
        self._results = {
            'optimized_params_final_clipped': final_params, 'objective_function_value': best_fit,
            'chosen_objective': objective_for_log, 'final_recovery_factor_reported': final_eval.get('recovery_factor', 0.0),
            'final_npv_reported': final_eval.get('npv'), 'mmp_psi': self.mmp, 'method': 'genetic_algorithm',
            'generations': cfg.generations, 'population_size': cfg.population_size,
            'top_ga_solutions_from_final_pop': top_solutions
        }
        
        self._results = self._handle_target_miss_reporting(final_eval, self._results, handle_target_miss)
        logger.info(f"GA done. Best obj ({objective_for_log}): {best_fit:.4e}")
        return self._results

    def optimize_bayesian(self, n_iter_override: Optional[int]=None, init_points_override: Optional[int]=None, method_override: Optional[str]=None, initial_solutions_from_ga: Optional[List[Dict[str,Any]]]=None, handle_target_miss: bool = False, **kwargs) -> Dict[str,Any]:
        self._refresh_operational_parameters()
        bo_params = self.bo_params_default_config
        n_i = n_iter_override if n_iter_override is not None else bo_params.n_iterations
        init_r = init_points_override if init_points_override is not None else bo_params.n_initial_points
        objective_for_log = self._get_objective_name_for_logging()
        
        logger.info(f"BayesOpt: Obj '{objective_for_log}', Iter {n_i}, InitRand {init_r}")
        p_bounds_d = self._get_ga_parameter_bounds()
        
        pb_bayes = {n: (l, h) for n, (l, h) in p_bounds_d.items()}
        
        bayes_o = BayesianOptimization(f=self._objective_function_wrapper, pbounds=pb_bayes, random_state=42, verbose=2)
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
        self._refresh_operational_parameters()
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
        obj_val, obj_name, method = source.get('objective_function_value'), source.get('chosen_objective','Obj'), source.get('method','unknown')
        if obj_val is None: return None
        fig = go.Figure()
        title = f'Opt Outcome ({method}) for {obj_name.replace("_"," ").title()}'
        y_title = f'{obj_name.replace("_"," ").title()} Val'
        x_title='Opt Steps (Conceptual)'
        num_steps=1
        if 'ga_full_results_for_hybrid' in source and isinstance(source['ga_full_results_for_hybrid'], dict):
            ga_res = source['ga_full_results_for_hybrid']
            ga_gens, ga_obj = ga_res.get('generations',0), ga_res.get('objective_function_value')
            if ga_gens > 0 and ga_obj is not None: 
                fig.add_trace(go.Scatter(x=[ga_gens], y=[ga_obj], mode='markers+text', name='GA End', text="GA"))
                num_steps=ga_gens
        if 'iterations_bo_actual' in source:
            bo_iters=source.get('iterations_bo_actual',0)
            bo_start=num_steps+1
            bo_end=bo_start+bo_iters-1
            fig.add_trace(go.Scatter(x=[bo_end if bo_iters>0 else bo_start], y=[obj_val], mode='markers+text', name='BO/Final', text="BO/Final"))
            num_steps=bo_end if bo_iters>0 else bo_start
        else: 
            fig.add_trace(go.Scatter(x=[1], y=[obj_val], mode='markers+text', name='Final Obj Val', text="Final"))
        fig.update_layout(title_text=title, xaxis_title_text=x_title, yaxis_title_text=y_title)
        return fig

    def plot_parameter_sensitivity(self, param_name_for_sensitivity: str, results_to_use_for_plot: Optional[Dict[str,Any]]=None) -> Optional[go.Figure]:
        source = results_to_use_for_plot or self._results
        num_pts_sens = 20 # Default points for sensitivity plot
        if not (source and 'optimized_params_final_clipped' in source and isinstance(source['optimized_params_final_clipped'], dict)): return None
        
        opt_base = source['optimized_params_final_clipped'].copy()
        obj_name_sens = source.get('chosen_objective', 'Objective Value')
        
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