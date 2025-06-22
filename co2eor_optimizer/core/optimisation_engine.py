from typing import Callable, Dict, List, Optional, Any, Tuple, Union, Type
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# config_manager.py is in the project root, one level above 'core/'
try:
    from config_manager import config_manager, ConfigNotLoadedError
except ImportError:
    logging.critical(
        "ConfigManager could not be imported from core/optimization_engine.py. "
        "Ensure config_manager.py is in the project root and project root is in PYTHONPATH."
    )
    class DummyConfigManager:
        def get(self, key_path: str, default: Any = None) -> Any:
            if default is None:
                 raise ConfigNotLoadedError(f"DummyConfig: Critical key '{key_path}' access attempted.")
            return default
        def get_section(self, section_key: str) -> Optional[Dict[str, Any]]:
             res = self.get(section_key, {})
             return res if isinstance(res, dict) else {}
        @property
        def is_loaded(self) -> bool: return False
    config_manager = DummyConfigManager()

# Imports from other modules within the 'core' package
from .data_models import (
    ReservoirData, PVTProperties, EORParameters, GeneticAlgorithmParams
)
from .recovery_models import recovery_factor

# Imports for MMP calculation (evaluation package is in project_root/evaluation/)
# This import assumes project_root is in sys.path
try:
    from evaluation.mmp import calculate_mmp as calculate_mmp_external, MMPParameters
except ImportError:
    logging.critical(
        "evaluation.mmp modules failed to import into OptimizationEngine. "
        "MMP calculation will be limited."
    )
    calculate_mmp_external = None # type: ignore
    MMPParameters = None # type: ignore


class OptimizationEngine:
    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties,
                 eor_params_instance: Optional[EORParameters] = None,
                 ga_params_instance: Optional[GeneticAlgorithmParams] = None,
                 well_analysis: Optional[Any] = None): # well_analysis can be any type from outside core
        if not config_manager.is_loaded:
            logging.critical("OptimizationEngine: ConfigManager reports no configuration loaded.")

        self.reservoir = reservoir
        self.pvt = pvt

        self.eor_params = eor_params_instance or EORParameters.from_config_dict(
            config_manager.get_section("EORParametersDefaults") or {}
        )
        self.ga_params_default_config = ga_params_instance or GeneticAlgorithmParams.from_config_dict(
            config_manager.get_section("GeneticAlgorithmParamsDefaults") or {}
        )

        self.well_analysis = well_analysis
        self._results: Optional[Dict[str, Any]] = None
        self._mmp_value: Optional[float] = None
        self._mmp_params_used: Optional[Any] = None # Stores MMPParameters or PVTProperties

        self.recovery_model: str = config_manager.get("OptimizationEngineSettings.default_recovery_model", "hybrid")
        self._recovery_model_init_kwargs: Dict[str, Any] = config_manager.get_section(
            f"RecoveryModelKwargsDefaults.{self.recovery_model.capitalize()}"
        ) or {}
        
        self._mmp_calculator_fn: Optional[Callable[[Union[PVTProperties, Any], str], float]] = calculate_mmp_external
        self._MMPParametersDataclass: Optional[Type[Any]] = MMPParameters # Type of MMPParameters
        self.calculate_mmp()

    def calculate_mmp(self, method_override: Optional[str] = None) -> float:
        default_mmp_fallback = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            logging.warning("MMP calculator or MMPParameters not available. Returning fallback MMP.")
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
            return self._mmp_value

        actual_mmp_method = method_override or config_manager.get("OptimizationEngineSettings.mmp_calculation_method", "auto")
        source_description = "PVT data"
        self._mmp_params_used = self.pvt # Default to PVT

        if self.well_analysis and hasattr(self.well_analysis, 'get_average_mmp_params_for_engine'):
            try:
                avg_well_params = self.well_analysis.get_average_mmp_params_for_engine()
                mmp_input_constructor_params = {
                    'temperature': avg_well_params.get('temperature', self.pvt.temperature),
                    'oil_gravity': avg_well_params.get('oil_gravity', config_manager.get("GeneralFallbacks.api_gravity_default", 35.0)),
                    'c7_plus_mw': avg_well_params.get('c7_plus_mw'),
                    'injection_gas_composition': avg_well_params.get('injection_gas_composition',
                        config_manager.get_section("GeneralFallbacks.default_injection_gas_composition") or {'CO2': 1.0}),
                    'pvt_data': self.pvt
                }
                self._mmp_params_used = self._MMPParametersDataclass(**mmp_input_constructor_params)
                source_description = "WellAnalysis average parameters"
            except Exception as e:
                logging.warning(f"Failed to get MMP params from WellAnalysis: {e}. Using PVT data.")
        
        try:
            calculated_mmp_value = float(self._mmp_calculator_fn(self._mmp_params_used, method=actual_mmp_method))
            self._mmp_value = calculated_mmp_value
            logging.info(f"MMP calculated: {self._mmp_value:.2f} psi via '{actual_mmp_method}' from {source_description}.")
        except Exception as e:
            logging.error(f"MMP calculation failed ('{actual_mmp_method}', {source_description}): {e}. Using fallback.")
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
        
        return self._mmp_value # Guarantees a float due to fallback

    def optimize_recovery(self) -> Dict[str, Any]:
        cfg_grad = config_manager.get_section("OptimizationEngineSettings.gradient_descent_optimizer") or {}
        max_iter = cfg_grad.get("max_iter", 100); tol = cfg_grad.get("tolerance", 1e-4)
        learning_rate = cfg_grad.get("learning_rate", 50.0); pressure_perturbation = cfg_grad.get("pressure_perturbation", 10.0)

        mmp_val = self.mmp # Uses property that calls calculate_mmp if needed
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0) # Final fallback

        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))
        
        initial_pressure = self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi > mmp_val else mmp_val * 1.05
        current_pressure = np.clip(initial_pressure, mmp_val * 1.01, self.eor_params.max_pressure_psi)
        injection_rate_val = self.eor_params.injection_rate

        call_kwargs_for_recovery = {**self._recovery_model_init_kwargs,
                                    'v_dp_coefficient': self.eor_params.v_dp_coefficient,
                                    'mobility_ratio': self.eor_params.mobility_ratio}
        previous_recovery = 0.0; current_recovery = 0.0; converged = False; iterations_done = 0

        for i in range(max_iter):
            iterations_done = i + 1
            current_recovery = recovery_factor(current_pressure, injection_rate_val, avg_porosity, mmp_val,
                                               model=self.recovery_model, **call_kwargs_for_recovery)
            if i > 5 and abs(current_recovery - previous_recovery) < tol: converged = True; break
            
            recovery_plus_perturb = recovery_factor(current_pressure + pressure_perturbation, injection_rate_val, avg_porosity, mmp_val,
                                                  model=self.recovery_model, **call_kwargs_for_recovery)
            gradient = (recovery_plus_perturb - current_recovery) / pressure_perturbation
            if abs(gradient) < 1e-7: converged = True; break
            current_pressure = np.clip(current_pressure + learning_rate * gradient, mmp_val * 1.01, self.eor_params.max_pressure_psi)
            previous_recovery = current_recovery
            
        self._results = {'optimized_params': {'injection_rate': injection_rate_val, 'target_pressure_psi': current_pressure,
                                             'v_dp_coefficient': self.eor_params.v_dp_coefficient, 'mobility_ratio': self.eor_params.mobility_ratio},
                         'mmp_psi': mmp_val, 'iterations': iterations_done, 'final_recovery': current_recovery,
                         'converged': converged, 'avg_porosity': avg_porosity, 'method': 'gradient_descent_pressure'}
        return self._results

    def optimize_bayesian(self, n_iter_override: Optional[int] = None, init_points_override: Optional[int] = None,
                        method_override: Optional[str] = None,
                        initial_solutions_from_ga: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        cfg_bo = config_manager.get_section("OptimizationEngineSettings.bayesian_optimizer") or {}
        n_iter = n_iter_override if n_iter_override is not None else cfg_bo.get("n_iter", 40)
        init_pts_random = init_points_override if init_points_override is not None else cfg_bo.get("init_points_random", 8)
        bo_method = method_override or cfg_bo.get("default_method", "gp")
        rate_max_factor = cfg_bo.get("rate_bound_factor_max", 1.5)

        mmp_val = self.mmp
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))
        
        min_pressure_bound = np.clip(mmp_val * 1.01, (getattr(self.eor_params, 'min_target_pressure_psi', mmp_val *1.01)), self.eor_params.max_pressure_psi - 1.0)

        space_dims = [Real(min_pressure_bound, self.eor_params.max_pressure_psi, name='pressure'),
                      Real(self.eor_params.min_injection_rate_bpd, self.eor_params.injection_rate * rate_max_factor, name='rate'),
                      Real(0.3, 0.8, name='v_dp_coefficient'), Real(1.2, 20.0, name='mobility_ratio')]
        if self.eor_params.injection_scheme == 'wag':
            space_dims.extend([Real(self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days, name='cycle_length_days'),
                               Real(self.eor_params.min_water_fraction, self.eor_params.max_water_fraction, name='water_fraction')])
        param_names_in_order = [dim.name for dim in space_dims]

        @use_named_args(space_dims)
        def objective_function_bo(**params_bo):
            eff_rate = params_bo['rate'] * (1.0 - params_bo['water_fraction'] if self.eor_params.injection_scheme == 'wag' and 'water_fraction' in params_bo else 1.0)
            rec = recovery_factor(params_bo['pressure'], eff_rate, avg_porosity, mmp_val, # type: ignore
                                  model=self.recovery_model, **{**self._recovery_model_init_kwargs, **params_bo})
            return -rec if bo_method == 'gp' else rec

        best_params_bo: Dict[str, float] = {}; final_rec_bo: float = 0.0; num_ga_pts_used_bo: int = 0

        if bo_method == 'gp':
            x0_dicts_skopt: List[Dict[str, float]] = []; y0_skopt: List[float] = []
            if initial_solutions_from_ga:
                for ga_sol in initial_solutions_from_ga:
                    valid_sol_params = {dim.name: np.clip(ga_sol['params'][dim.name], dim.low, dim.high) for dim in space_dims if dim.name in ga_sol['params']}
                    if len(valid_sol_params) == len(space_dims): # Ensure all dims covered
                        x0_dicts_skopt.append(valid_sol_params); y0_skopt.append(-ga_sol['fitness'])
            num_ga_pts_used_bo = len(x0_dicts_skopt)
            total_calls = num_ga_pts_used_bo + init_pts_random + n_iter
            if initial_solutions_from_ga and total_calls <= num_ga_pts_used_bo: total_calls = num_ga_pts_used_bo + init_pts_random +1
            
            logging.info(f"gp_minimize: {num_ga_pts_used_bo} GA pts, {init_pts_random} random, {n_iter} BO iter. Total: {total_calls}.")
            # Pass list of dicts directly to x0 if skopt handles it with use_named_args
            # Otherwise, convert x0_dicts_skopt to list of lists:
            # x0_lists_skopt = [[d[name] for name in param_names_in_order] for d in x0_dicts_skopt]
            res_skopt = gp_minimize(objective_function_bo, space_dims, x0=x0_dicts_skopt or None, y0=y0_skopt or None,
                                   n_calls=total_calls, n_initial_points=init_pts_random, 
                                   random_state=config_manager.get("GeneralFallbacks.random_seed", 42), verbose=True) # type: ignore
            best_params_bo = dict(zip(param_names_in_order, res_skopt.x)); final_rec_bo = -res_skopt.fun
        
        elif bo_method == 'bayes':
            pbounds = {dim.name: (dim.low, dim.high) for dim in space_dims}
            opt_bayes = BayesianOptimization(f=objective_function_bo, pbounds=pbounds, random_state=config_manager.get("GeneralFallbacks.random_seed", 42), verbose=2)
            if initial_solutions_from_ga:
                for ga_sol in initial_solutions_from_ga:
                    params_reg = {p_name: np.clip(ga_sol['params'][p_name],pbounds[p_name][0],pbounds[p_name][1]) for p_name in pbounds if p_name in ga_sol['params']}
                    if len(params_reg) == len(pbounds):
                        try: opt_bayes.register(params=params_reg, target=ga_sol['fitness']); num_ga_pts_used_bo+=1
                        except Exception as e: logging.error(f"Error registering GA sol with bayes_opt: {e}")
            opt_bayes.maximize(init_points=init_pts_random, n_iter=n_iter)
            best_params_bo = opt_bayes.max['params']; final_rec_bo = opt_bayes.max['target']
        else: raise ValueError(f"Unsupported Bayesian method: {bo_method}.")

        opt_params_std = {'injection_rate': best_params_bo.get('rate'), 'target_pressure_psi': best_params_bo.get('pressure'),
                          'cycle_length_days': best_params_bo.get('cycle_length_days'), 'water_fraction': best_params_bo.get('water_fraction'),
                          'v_dp_coefficient': best_params_bo.get('v_dp_coefficient'), 'mobility_ratio': best_params_bo.get('mobility_ratio')}
        
        self._results = {'optimized_params': opt_params_std, 'mmp_psi': mmp_val, 'method': f'bayesian_{bo_method}',
                         'iterations_bo_actual': n_iter, 'initial_points_bo_random': init_pts_random,
                         'initial_points_from_ga_used': num_ga_pts_used_bo, 'final_recovery': final_rec_bo,
                         'avg_porosity': avg_porosity, 'converged': True}
        return self._results

    def hybrid_optimize(self, ga_params_override: Optional[GeneticAlgorithmParams] = None) -> Dict[str, Any]:
        cfg_hyb = config_manager.get_section("OptimizationEngineSettings.hybrid_optimizer") or \
                  {"ga_config_source": "default_ga_params", "bo_iterations_in_hybrid": 20,
                   "bo_random_initial_points_in_hybrid": 5, "num_ga_elites_to_bo": 3, "bo_method_in_hybrid": "gp"}

        ga_params_hyb = ga_params_override if ga_params_override else \
                        (GeneticAlgorithmParams.from_config_dict(cfg_hyb.get("ga_params_hybrid",{})) \
                         if cfg_hyb.get("ga_config_source") == "hybrid_specific" and cfg_hyb.get("ga_params_hybrid") \
                         else self.ga_params_default_config)

        bo_iter = cfg_hyb.get("bo_iterations_in_hybrid", 20); bo_init_rand = cfg_hyb.get("bo_random_initial_points_in_hybrid", 5)
        num_elites = cfg_hyb.get("num_ga_elites_to_bo", 3); bo_method = cfg_hyb.get("bo_method_in_hybrid", "gp")
        
        logging.info(f"Hybrid: GA(G:{ga_params_hyb.generations},P:{ga_params_hyb.population_size}) -> BO(M:{bo_method},I:{bo_iter},E:{num_elites})")
        ga_res = self.optimize_genetic_algorithm(ga_params_to_use=ga_params_hyb)
        
        top_ga_sols = ga_res.get('top_ga_solutions_from_final_pop', [])
        init_bo_sols = top_ga_sols[:min(num_elites, len(top_ga_sols))] if num_elites > 0 and top_ga_sols else None

        bo_res = self.optimize_bayesian(n_iter_override=bo_iter, init_points_override=bo_init_rand,
                                        method_override=bo_method, initial_solutions_from_ga=init_bo_sols)
        
        self._results = {**bo_res, 'ga_full_results': ga_res, 
                         'method': f'hybrid_ga(g{ga_params_hyb.generations})_bo(i{bo_iter},e{len(init_bo_sols or [])})_m({bo_method})'}
        logging.info(f"Hybrid opt. done. Final recovery: {self._results.get('final_recovery', 0.0):.4f}")
        return self._results

    def _get_ga_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        mmp_val = self.mmp
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        rate_max_f = config_manager.get("OptimizationEngineSettings.genetic_algorithm.rate_bound_factor_max", 1.5)
        min_p_ga = np.clip(mmp_val * 1.01, getattr(self.eor_params,'target_pressure_psi', mmp_val*1.01) if getattr(self.eor_params,'target_pressure_psi',0) > mmp_val else mmp_val*1.01, self.eor_params.max_pressure_psi -1.0)

        bounds = {'pressure': (min_p_ga, self.eor_params.max_pressure_psi),
                  'rate': (self.eor_params.min_injection_rate_bpd, self.eor_params.injection_rate * rate_max_f),
                  'v_dp_coefficient': (0.3, 0.8), 'mobility_ratio': (1.2, 20.0)}
        if self.eor_params.injection_scheme == 'wag':
            bounds['cycle_length_days'] = (self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days)
            bounds['water_fraction'] = (self.eor_params.min_water_fraction, self.eor_params.max_water_fraction)
        return bounds

    def _initialize_population_ga(self, pop_size: int, ga_conf: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        pop = []; bounds = self._get_ga_parameter_bounds(); p_names = list(bounds.keys())
        for _ in range(pop_size):
            ind = {p_name: random.uniform(bounds[p_name][0], bounds[p_name][1]) for p_name in p_names}
            pop.append(ind)
        return pop

    def _get_complete_params_from_ga_individual(self, ind_dict: Dict[str, float], bounds_clip: Dict[str, Tuple[float,float]]) -> Dict[str, float]:
        return {p_name: np.clip(ind_dict.get(p_name, random.uniform(low, high)), low, high) for p_name, (low,high) in bounds_clip.items()}

    def _evaluate_individual_ga(self, ind_dict: Dict[str, float], avg_poro: float, mmp: float, model: str, init_kwargs: Dict[str,Any], ga_conf: GeneticAlgorithmParams) -> float:
        params = self._get_complete_params_from_ga_individual(ind_dict, self._get_ga_parameter_bounds())
        eff_rate = params['rate'] * (1.0 - params.get('water_fraction',0) if self.eor_params.injection_scheme == 'wag' else 1.0)
        return recovery_factor(params['pressure'], eff_rate, avg_poro, mmp, model=model, **{**init_kwargs, **params})

    def _tournament_selection_ga(self, pop: List[Dict[str, float]], fits: List[float], conf: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        sel_pop = []
        if conf.elite_count > 0 and pop: sel_pop.extend(pop[i].copy() for i in np.argsort(fits)[-conf.elite_count:])
        for _ in range(len(pop) - len(sel_pop)):
            if not pop: break
            sel_pop.append(pop[max(random.sample(range(len(pop)), conf.tournament_size), key=lambda i: fits[i])].copy())
        return sel_pop

    def _crossover_ga(self, parents: List[Dict[str, float]], conf: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        offspr = []; n_par = len(parents)
        if n_par == 0: return []
        for i in range(0, n_par -1, 2):
            p1, p2 = parents[i], parents[i+1]; c1, c2 = p1.copy(), p2.copy()
            if random.random() < conf.crossover_rate:
                alpha = conf.blend_alpha_crossover
                for gene in set(p1.keys()) | set(p2.keys()):
                    v1, v2 = p1.get(gene), p2.get(gene)
                    if v1 is not None and v2 is not None:
                        c1[gene] = alpha * v1 + (1-alpha) * v2; c2[gene] = (1-alpha) * v1 + alpha * v2
            offspr.extend([c1,c2])
        if n_par % 2 == 1: offspr.append(parents[-1].copy())
        return offspr[:n_par]

    def _mutate_ga(self, pop_mut: List[Dict[str, float]], conf: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        mut_pop = []; bounds = self._get_ga_parameter_bounds()
        for ind in pop_mut:
            mut_ind = ind.copy()
            if random.random() < conf.mutation_rate:
                gene = random.choice(list(mut_ind.keys()))
                if gene in bounds:
                    low, high = bounds[gene]; curr = mut_ind.get(gene, (low+high)/2.0)
                    std_dev = (high-low) * conf.mutation_strength_factor if (high-low) > 1e-9 else 0.1*abs(curr)+1e-6
                    mut_ind[gene] = np.clip(curr + random.gauss(0, std_dev), low, high)
            mut_pop.append(mut_ind)
        return mut_pop

    def optimize_genetic_algorithm(self, ga_params_to_use: Optional[GeneticAlgorithmParams] = None) -> Dict[str, Any]:
        ga_conf = ga_params_to_use or self.ga_params_default_config
        mmp_val = self.mmp
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        avg_poro = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))

        pop = self._initialize_population_ga(ga_conf.population_size, ga_conf)
        best_sol_all = pop[0].copy() if pop else {}; best_fit_all = -np.inf
        
        for gen in range(ga_conf.generations):
            with ProcessPoolExecutor() as executor:
                fits = list(executor.map(partial(self._evaluate_individual_ga, avg_porosity=avg_poro, mmp=mmp_val, # type: ignore
                                                  model_name=self.recovery_model, recovery_model_init_kwargs=self._recovery_model_init_kwargs,
                                                  current_ga_config=ga_conf), pop)) # Renamed args for clarity
            
            best_idx_gen = np.argmax(fits)
            if fits[best_idx_gen] > best_fit_all: best_fit_all = fits[best_idx_gen]; best_sol_all = pop[best_idx_gen].copy()
            
            parents = self._tournament_selection_ga(pop, fits, ga_conf); offspring = self._crossover_ga(parents, ga_conf)
            pop = self._mutate_ga(offspring, ga_conf)
            if (gen+1)%10==0 or gen==ga_conf.generations-1: logging.info(f"GA Gen {gen+1}: BestFit Gen {fits[best_idx_gen]:.4f} Overall {best_fit_all:.4f}")
        
        with ProcessPoolExecutor() as executor:
            final_fits = list(executor.map(partial(self._evaluate_individual_ga, avg_porosity=avg_poro, mmp=mmp_val, # type: ignore
                                                     model_name=self.recovery_model, recovery_model_init_kwargs=self._recovery_model_init_kwargs,
                                                     current_ga_config=ga_conf), pop)) # Renamed args
        
        final_pop_sorted = sorted(zip(pop, final_fits), key=lambda x:x[1], reverse=True)
        top_sols_n = min(len(final_pop_sorted), ga_conf.elite_count if ga_conf.elite_count > 0 else 1)
        ga_bounds = self._get_ga_parameter_bounds() # For completing individuals
        top_sols_bo = [{'params': self._get_complete_params_from_ga_individual(p, ga_bounds), 'fitness': f} for p,f in final_pop_sorted[:top_sols_n]]
        
        opt_params_ga = self._get_complete_params_from_ga_individual(best_sol_all, ga_bounds)
        eff_rate_ga = opt_params_ga['rate'] * (1.0 - opt_params_ga.get('water_fraction',0) if self.eor_params.injection_scheme == 'wag' else 1.0)

        self._results = {'optimized_params': {'injection_rate': eff_rate_ga, 'target_pressure_psi': opt_params_ga.get('pressure'),
                                             'cycle_length_days': opt_params_ga.get('cycle_length_days'), 'water_fraction': opt_params_ga.get('water_fraction'),
                                             'v_dp_coefficient': opt_params_ga.get('v_dp_coefficient'), 'mobility_ratio': opt_params_ga.get('mobility_ratio')},
                         'mmp_psi': mmp_val, 'method': 'genetic_algorithm', 'generations': ga_conf.generations,
                         'population_size': ga_conf.population_size, 'final_recovery': best_fit_all, 'avg_porosity': avg_poro, 
                         'converged': True, 'best_solution_dict_raw_ga': best_sol_all, 'top_ga_solutions_from_final_pop': top_sols_bo}
        logging.info(f"GA opt. done. Best recovery: {best_fit_all:.4f}")
        return self._results

    def optimize_wag(self) -> Dict[str, Any]:
        cfg_wag = config_manager.get_section("OptimizationEngineSettings.wag_optimizer") or {}
        ref_cycles = cfg_wag.get("refinement_cycles", 5); grid_pts = cfg_wag.get("grid_search_points_per_dim", 5)
        range_reduct = cfg_wag.get("range_reduction_factor", 0.5)
        
        mmp_val = self.mmp
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        avg_poro = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))

        min_cl, max_cl = self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days
        min_wf, max_wf = self.eor_params.min_water_fraction, self.eor_params.max_water_fraction

        best_wag = {'cycle_length_days': (min_cl+max_cl)/2, 'water_fraction': (min_wf+max_wf)/2,
                    'pressure': np.clip(mmp_val*1.05, getattr(self.eor_params,'target_pressure_psi',mmp_val*1.05) if getattr(self.eor_params,'target_pressure_psi',0)>mmp_val else mmp_val*1.05, self.eor_params.max_pressure_psi-1),
                    'v_dp_coefficient': self.eor_params.v_dp_coefficient, 'mobility_ratio': self.eor_params.mobility_ratio, 'recovery': -np.inf}

        for cycle in range(ref_cycles):
            logging.info(f"WAG Cycle {cycle+1}: CL [{min_cl:.1f}-{max_cl:.1f}], WF [{min_wf:.2f}-{max_wf:.2f}]")
            better_found = False; best_rec_cycle = best_wag['recovery']
            for wf_val in np.linspace(min_wf, max_wf, num=grid_pts):
                for cl_val in np.linspace(min_cl, max_cl, num=grid_pts):
                    eff_rate = self.eor_params.injection_rate * (1.0 - wf_val)
                    kwargs_rec = {**self._recovery_model_init_kwargs, **best_wag, 'cycle_length_days':cl_val, 'water_fraction':wf_val}
                    rec = recovery_factor(best_wag['pressure'], eff_rate, avg_poro, mmp_val, model=self.recovery_model, **kwargs_rec) # type: ignore
                    if rec > best_rec_cycle: best_rec_cycle=rec; best_wag.update({'cycle_length_days':cl_val, 'water_fraction':wf_val, 'recovery':rec}); better_found=True
            if not better_found and cycle > 0: logging.info("WAG converged early."); break
            
            range_wf_new = (max_wf - min_wf) * range_reduct / 2.0; min_wf = max(self.eor_params.min_water_fraction, best_wag['water_fraction']-range_wf_new); max_wf = min(self.eor_params.max_water_fraction, best_wag['water_fraction']+range_wf_new)
            range_cl_new = (max_cl - min_cl) * range_reduct / 2.0; min_cl = max(self.eor_params.min_cycle_length_days, best_wag['cycle_length_days']-range_cl_new); max_cl = min(self.eor_params.max_cycle_length_days, best_wag['cycle_length_days']+range_cl_new)

        opt_p_wag = self._optimize_pressure_for_wag(best_wag['water_fraction'], best_wag['cycle_length_days'], avg_poro, mmp_val, best_wag) # type: ignore
        best_wag['pressure'] = opt_p_wag
        final_eff_rate = self.eor_params.injection_rate * (1.0 - best_wag['water_fraction'])
        final_rec_wag = recovery_factor(best_wag['pressure'], final_eff_rate, avg_poro, mmp_val, model=self.recovery_model, **{**self._recovery_model_init_kwargs, **best_wag}) # type: ignore
        best_wag['recovery'] = final_rec_wag

        wag_opt_std = {'optimal_cycle_length_days': best_wag.get('cycle_length_days'), 'optimal_water_fraction': best_wag.get('water_fraction'),
                       'optimal_target_pressure_psi': best_wag.get('pressure'), 'injection_rate': self.eor_params.injection_rate,
                       'v_dp_coefficient': best_wag.get('v_dp_coefficient'), 'mobility_ratio': best_wag.get('mobility_ratio')}
        self._results = {'wag_optimized_params': wag_opt_std, 'mmp_psi': mmp_val, 'estimated_recovery': best_wag['recovery'],
                         'avg_porosity': avg_poro, 'method': 'iterative_grid_search_wag'}
        return self._results

    def _optimize_pressure_for_wag(self, wf_val: float, cl_val: float, avg_poro_val: float, mmp_val: float, context: Dict[str,Any]) -> float:
        cfg_p = config_manager.get_section("OptimizationEngineSettings.wag_optimizer") or {}
        max_it = cfg_p.get("max_iter_per_param_pressure_opt",20); lr = cfg_p.get("pressure_opt_learning_rate",20.0)
        tol = cfg_p.get("pressure_opt_tolerance",1e-4); pert = cfg_p.get("pressure_opt_perturbation",10.0)
        p_factor = cfg_p.get("pressure_constraint_factor_vs_mmp_max",1.75)

        eff_rate = self.eor_params.injection_rate * (1.0 - wf_val)
        curr_p = np.clip(context.get('pressure',mmp_val*1.05), mmp_val*1.01, min(self.eor_params.max_pressure_psi, mmp_val*p_factor))
        best_p = curr_p; best_rec_p = -np.inf; prev_rec_p = -np.inf

        for _ in range(max_it):
            kwargs_rec = {**self._recovery_model_init_kwargs, **context, 'water_fraction':wf_val, 'cycle_length_days':cl_val}
            curr_rec_p = recovery_factor(curr_p, eff_rate, avg_poro_val, mmp_val, model=self.recovery_model, **kwargs_rec)
            if curr_rec_p > best_rec_p: best_rec_p=curr_rec_p; best_p=curr_p
            if abs(curr_rec_p - prev_rec_p) < tol: break
            prev_rec_p = curr_rec_p
            rec_pert_p = recovery_factor(curr_p+pert, eff_rate, avg_poro_val, mmp_val, model=self.recovery_model, **kwargs_rec)
            grad_p = (rec_pert_p - curr_rec_p) / pert
            if abs(grad_p) < 1e-7: break
            curr_p = np.clip(curr_p + lr*grad_p, mmp_val*1.01, min(self.eor_params.max_pressure_psi, mmp_val*p_factor))
        return best_p

    def check_mmp_constraint(self, pressure: float) -> bool:
        mmp = self.mmp
        if mmp is None: return False
        return pressure >= mmp

    @property
    def results(self) -> Optional[Dict[str, Any]]: return self._results
    @property
    def mmp(self) -> Optional[float]:
        if self._mmp_value is None: self.calculate_mmp()
        return self._mmp_value

    def set_recovery_model(self, model_name: str, **kwargs_init_override: Any):
        valid = ['simple', 'miscible', 'immiscible', 'hybrid', 'koval']; name_low = model_name.lower()
        if name_low not in valid: raise ValueError(f"Unknown model: {model_name}. Valid: {valid}")
        self.recovery_model = name_low
        base_cfg = config_manager.get_section(f"RecoveryModelKwargsDefaults.{name_low.capitalize()}") or {}
        self._recovery_model_init_kwargs = {**base_cfg, **kwargs_init_override}
        logging.info(f"Recovery model: '{self.recovery_model}'. Init kwargs: {self._recovery_model_init_kwargs}")

    def plot_mmp_profile(self) -> Optional[go.Figure]:
        if not (self.well_analysis and hasattr(self.well_analysis, 'calculate_mmp_profile')):
            logging.warning("WellAnalysis unavailable/no 'calculate_mmp_profile'."); return None
        try:
            profile = self.well_analysis.calculate_mmp_profile() # type: ignore
            if not (isinstance(profile,dict) and all(k in profile for k in ['depths','mmp']) and profile['depths'].size and profile['mmp'].size):
                logging.warning("MMP profile data incomplete/empty."); return None
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=profile['mmp'], y=profile['depths'], name='MMP (psi)'), secondary_y=False)
            if 'temperature' in profile and profile['temperature'].size:
                fig.add_trace(go.Scatter(x=profile['temperature'], y=profile['depths'], name='Temp (°F)'), secondary_y=True)
            fig.update_layout(title_text='MMP vs Depth', yaxis_title_text='Depth (ft)', legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            fig.update_yaxes(title_text="Depth (ft)", secondary_y=False, autorange="reversed")
            if 'temperature' in profile and profile['temperature'].size:
                 fig.update_yaxes(title_text="Temp Axis", secondary_y=True, autorange="reversed", overlaying='y', side='right', showticklabels=True); fig.update_xaxes(title_text="Value")
            else: fig.update_xaxes(title_text="MMP (psi)")
            return fig
        except Exception as e: logging.error(f"Error plotting MMP profile: {e}", exc_info=True); return None

    def plot_optimization_convergence(self, results_plot: Optional[Dict[str,Any]] = None) -> Optional[go.Figure]:
        data = results_plot or self._results
        if not data: logging.warning("No results for convergence plot."); return None
        fig=go.Figure(); method=data.get('method','unknown'); final_rec=data.get('final_recovery')
        if final_rec is None: logging.warning(f"No 'final_recovery' in results for {method}."); return None
        
        steps_x=0; title=f'Opt. Outcome ({method})'
        if 'ga_full_results' in data and isinstance(data['ga_full_results'],dict):
            ga_res=data['ga_full_results']; ga_gens=ga_res.get('generations',0); ga_rec=ga_res.get('final_recovery')
            if ga_gens>0 and ga_rec is not None: fig.add_trace(go.Scatter(x=np.arange(1,ga_gens+1),y=np.full(ga_gens,ga_rec),name='GA Best',mode='lines',line=dict(dash='dot')))
            steps_x+=ga_gens; title='Hybrid Opt. Outcome (GA->BO)'
        
        bo_iters=data.get('iterations_bo_actual',0)
        if bo_iters > 0:
            bo_start=steps_x+1; bo_end=steps_x+bo_iters
            fig.add_trace(go.Scatter(x=np.arange(bo_start,bo_end+1),y=np.full(bo_iters,final_rec),name='BO Final',mode='lines+markers'))
            steps_x+=bo_iters
        elif 'ga' not in method.lower() and 'gradient' not in method.lower(): steps_x=data.get('iterations',1)
        if not fig.data: fig.add_trace(go.Scatter(x=[max(1,steps_x)],y=[final_rec],name='Final Rec.',mode='markers',marker=dict(size=10)))
        fig.update_layout(title_text=title, xaxis_title_text='Opt. Steps (Conceptual)', yaxis_title_text='Recovery Factor')
        return fig

    def plot_parameter_sensitivity(self, param_name: str, results_plot: Optional[Dict[str,Any]]=None) -> Optional[go.Figure]:
        res_data = results_plot or self._results
        n_pts = config_manager.get("OptimizationEngineSettings.sensitivity_plot_points", 20)
        if not (res_data and 'optimized_params' in res_data and isinstance(res_data['optimized_params'], dict)):
            logging.warning("No optimized params for sensitivity plot."); return None
        mmp_val = self.mmp
        if mmp_val is None: mmp_val = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)

        opt_params = res_data['optimized_params'].copy()
        avg_poro = res_data.get('avg_porosity', np.mean(self.reservoir.grid.get('PORO', [0.15])))
        if param_name not in opt_params or opt_params[param_name] is None:
            logging.warning(f"Param '{param_name}' not in opt_params/is None."); return None
        
        opt_val = opt_params[param_name]; bounds_lookup = self._get_ga_parameter_bounds()
        low_b, high_b = opt_val*0.8, opt_val*1.2 # Default sweep
        if param_name in bounds_lookup:
            g_low, g_high = bounds_lookup[param_name]; sweep_h = (g_high-g_low)*0.2
            p_low = max(g_low, opt_val-sweep_h); p_high = min(g_high, opt_val+sweep_h)
        elif param_name == 'target_pressure_psi':
            p_low = max(mmp_val*1.01, opt_val*0.8); p_high = min(self.eor_params.max_pressure_psi, opt_val*1.2)
        else: p_low,p_high = low_b,high_b
        if p_high <= p_low: p_low=opt_val*0.95; p_high=opt_val*1.05
        if abs(p_high-p_low)<1e-6: p_low-=(abs(opt_val*0.05)+1e-3); p_high+=(abs(opt_val*0.05)+1e-3)

        p_sweep = np.linspace(p_low,p_high,n_pts); rec_sens = []
        for val_sw in p_sweep:
            temp_params = opt_params.copy(); temp_params[param_name] = val_sw
            eval_p = temp_params.get('target_pressure_psi', self.eor_params.target_pressure_psi)
            eval_r_base = temp_params.get('injection_rate', self.eor_params.injection_rate)
            eff_rate_sw = eval_r_base * (1.0-temp_params.get('water_fraction',0) if self.eor_params.injection_scheme=='wag' and 'water_fraction' in temp_params else 1.0)
            kwargs_sens = {**self._recovery_model_init_kwargs, **temp_params}
            rec_sens.append(recovery_factor(eval_p, eff_rate_sw, avg_poro, mmp_val, model=self.recovery_model, **kwargs_sens)) # type: ignore

        fig=go.Figure(); fig.add_trace(go.Scatter(x=p_sweep,y=rec_sens,mode='lines+markers',name=f'{param_name} sensitivity'))
        fig.add_vline(x=opt_val, line=dict(width=2,dash="dash",color="green"), name="Optimal Value")
        title_p_name = param_name.replace("_psi"," (psi)").replace("_days"," (days)").replace("_bpd"," (bpd)").replace("_"," ").capitalize()
        fig.update_layout(title_text=f'Recovery Factor vs {title_p_name}', xaxis_title_text=title_p_name, yaxis_title_text='Recovery Factor')
        return fig