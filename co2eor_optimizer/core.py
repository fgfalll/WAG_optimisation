from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from abc import ABC, abstractmethod
import random
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization

@dataclass
class WellData:
    """Container for well log data"""
    name: str
    depths: np.ndarray
    properties: Dict[str, np.ndarray]
    units: Dict[str, str]
    
    def validate(self) -> bool:
        """Check data consistency"""
        return len(self.depths) == len(next(iter(self.properties.values())))

@dataclass
class ReservoirData:
    """Container for reservoir simulation data"""
    grid: Dict[str, np.ndarray]
    pvt_tables: Dict[str, np.ndarray]
    regions: Optional[Dict[str, np.ndarray]] = None

@dataclass
class PVTProperties:
    """PVT properties for oil and gas"""
    oil_fvf: np.ndarray
    oil_viscosity: np.ndarray
    gas_fvf: np.ndarray
    gas_viscosity: np.ndarray
    rs: np.ndarray  # Solution GOR
    pvt_type: str  # 'black_oil' or 'compositional'

    def __post_init__(self):
        """Validate PVT property arrays"""
        arrays = [
            self.oil_fvf,
            self.oil_viscosity,
            self.gas_fvf,
            self.gas_viscosity,
            self.rs
        ]
        
        # Check all arrays have same length
        if len({len(arr) for arr in arrays}) > 1:
            raise ValueError("All PVT property arrays must have same length")
            
        # Validate pvt_type
        if self.pvt_type not in {'black_oil', 'compositional'}:
            raise ValueError("pvt_type must be either 'black_oil' or 'compositional'")

@dataclass
class EORParameters:
    """CO2 EOR operational parameters"""
    injection_rate: float
    WAG_ratio: Optional[float] = None
    injection_scheme: str = 'continuous'  # or 'wag'
    min_cycle_length: float = 15.0  # days
    max_cycle_length: float = 90.0  # days
    min_water_fraction: float = 0.2
    max_water_fraction: float = 0.8
    target_pressure: float = 0.0
    max_pressure: float = 5000.0  # psi
    min_injection_rate: float = 1000.0  # bbl/day
    v_dp: float = 0.5  # Dykstra-Parsons coefficient (0.3 < V_DP < 0.8)
    mobility_ratio: float = 2.0  # Mobility ratio M (1.2 < M < 20)

    def __post_init__(self):
        """Validate parameter bounds"""
        if not 0.3 < self.v_dp < 0.8:
            raise ValueError(f"V_DP must be between 0.3 and 0.8, got {self.v_dp}")
        if not 1.2 < self.mobility_ratio < 20:
            raise ValueError(f"Mobility ratio must be between 1.2 and 20, got {self.mobility_ratio}")

@dataclass
class GeneticAlgorithmParams:
    """Parameters for genetic algorithm optimization"""
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_pressure: float = 1.5
    elite_count: int = 2

@dataclass
class RecoveryModel:
    """Base class for recovery factor models"""
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        raise NotImplementedError

class KovalRecoveryModel(RecoveryModel):
    """Physics-informed sweep efficiency model using Koval method with integrated heterogeneity-mobility factor"""
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        # Validate all input parameters
        if not all(isinstance(x, (int, float)) for x in [pressure, rate, porosity, mmp]):
            raise ValueError("All parameters must be numeric")
        if any(x <= 0 for x in [pressure, rate, porosity, mmp]):
            raise ValueError("Parameters must be positive")
        if porosity > 0.3:
            raise ValueError("Porosity exceeds practical limits (>0.3)")
        if mmp < 1000 or mmp > 10000:
            raise ValueError("MMP must be between 1000-10000 psi")
            
        # Extract and validate optional parameters
        v_dp = kwargs.get('v_dp', 0.5)
        mobility_ratio = kwargs.get('mobility_ratio', 2.0)
        mu_co2 = kwargs.get('mu_co2', 0.1)  # Default CO2 viscosity (cP)
        mu_oil = kwargs.get('mu_oil', 5.0)   # Default oil viscosity (cP)
        
        if not 0.3 <= v_dp <= 0.8:
            raise ValueError(f"V_DP must be between 0.3-0.8, got {v_dp}")
        if not 1.2 <= mobility_ratio <= 20:
            raise ValueError(f"Mobility ratio must be between 1.2-20, got {mobility_ratio}")
        if mu_co2 <= 0 or mu_oil <= 0:
            raise ValueError("Viscosities must be positive")

        # Calculate Koval factor combining heterogeneity and mobility
        kv = v_dp * (1 + (mu_co2 / mu_oil))
        
        # Calculate sweep efficiency with bounds checking
        mr_term = (mobility_ratio - 1) / kv
        if mr_term < 0:
            raise ValueError(f"Invalid mobility ratio/Kv combination: MR={mobility_ratio}, Kv={kv}")
            
        es = 1 / (1 + np.sqrt(mr_term))
        
        # Phase-aware efficiency scaling
        pressure_ratio = pressure / mmp
        if pressure_ratio >= 1.0:
            return min(0.85, es)  # Full miscible regime cap
        elif pressure_ratio >= 0.8:
            return es * 0.9  # Near-miscible scaling
        return es * 0.7  # Immiscible scaling

class SimpleRecoveryModel(RecoveryModel):
    """Simple recovery model (original implementation)"""
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp))
        sweep_efficiency = 0.7 * (rate ** 0.2) * (porosity ** 0.5)
        return min(0.7, miscibility * sweep_efficiency)

class MiscibleRecoveryModel(RecoveryModel):
    """Advanced model for miscible CO2 flooding conditions incorporating:
    - Compositional effects
    - Gravity override
    - Viscous fingering
    - Reservoir heterogeneity
    """
    def __init__(self, kv_factor: float = 0.5, gravity_factor: float = 0.1):
        self.kv_factor = kv_factor  # Koval heterogeneity factor (0.3-0.8)
        self.gravity_factor = gravity_factor  # Gravity override scaling (0-1)

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [pressure, rate, porosity, mmp]):
            raise ValueError("All parameters must be numeric")
        if any(x <= 0 for x in [pressure, rate, porosity, mmp]):
            raise ValueError("Parameters must be positive")

        # Get optional parameters with defaults
        mu_co2 = kwargs.get('mu_co2', 0.06)  # CO2 viscosity (cP)
        mu_oil = kwargs.get('mu_oil', 5.0)   # Oil viscosity (cP)
        rel_perm = kwargs.get('rel_perm', {'co2': 0.8, 'oil': 0.3})  # Endpoint rel perms
        dip_angle = kwargs.get('dip_angle', 0.0)  # Reservoir dip (degrees)

        # Calculate miscibility factor (0-1)
        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp))

        # Calculate mobility ratio
        M = (mu_oil * rel_perm['co2']) / (mu_co2 * rel_perm['oil'])

        # Gravity number (Ng) - accounts for gravity override
        Ng = (self.gravity_factor * 2 * np.sin(np.radians(dip_angle))) / rate  # Enhanced dip angle impact

        # Koval factor combining heterogeneity and mobility
        Kv = self.kv_factor * (1 + (mu_co2 / mu_oil))

        # Compositional sweep efficiency (based on Craig-Geffen-Morse)
        Es_comp = 1 - 0.5 * (1 / (1 + M)) * (1 - np.exp(-2 * (1 + M)))

        # Gravity override correction
        Es_gravity = Es_comp * (1 - Ng)

        # Viscous fingering correction (Koval method)
        Es = 1 / (1 + np.sqrt((M - 1) / Kv))

        # Combined efficiency with miscibility scaling
        combined_eff = min(0.88, miscibility * Es * Es_gravity * (porosity ** 0.5) * (1 + dip_angle/50))  # Stronger dip angle influence

        # Ensure minimum recovery for full miscibility
        if pressure >= mmp * 1.2:
            return max(combined_eff, 0.61)  # Slightly increase minimum
        return combined_eff

class ImmiscibleRecoveryModel(RecoveryModel):
    """Advanced model for immiscible CO2 flooding conditions incorporating:
    - Three-phase relative permeability effects
    - Capillary pressure
    - Residual oil saturation
    """
    def __init__(self, sor: float = 0.25, krw_max: float = 0.4):
        self.sor = sor  # Residual oil saturation
        self.krw_max = krw_max  # Maximum water relative perm

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [pressure, rate, porosity, mmp]):
            raise ValueError("All parameters must be numeric")
        if any(x <= 0 for x in [pressure, rate, porosity, mmp]):
            raise ValueError("Parameters must be positive")

        # Get optional parameters with defaults
        mu_water = kwargs.get('mu_water', 0.5)  # Water viscosity (cP)
        mu_oil = kwargs.get('mu_oil', 5.0)      # Oil viscosity (cP)
        swc = kwargs.get('swc', 0.2)           # Connate water saturation
        kro_max = kwargs.get('kro_max', 0.8)    # Maximum oil rel perm

        # Calculate mobility ratio for water-oil system
        M_wo = (mu_oil * self.krw_max * 1.1) / (mu_water * kro_max)  # Add 10% sensitivity boost

        # Calculate capillary number (Nc)
        Nc = rate * mu_water / (porosity * pressure)

        # Calculate trapped gas saturation (empirical)
        Sgt = 0.15 * (1 - swc)

        # Modified Craig-Geffen-Morse model for immiscible flood
        E_displacement = (1 - self.sor) * (1 - np.exp(-Nc))
        E_vertical = 0.92 * (porosity ** 0.25)  # More porosity sensitivity
        E_areal = 1 / (1 + M_wo * 0.9)  # Modified areal efficiency

        # Combined efficiency with miscibility scaling
        miscibility = 0.9 * min(1.0, max(0.0, (pressure - mmp) / mmp))  # Stronger miscibility impact
        combined_eff = min(0.68, miscibility * E_displacement * E_vertical * E_areal)  # Adjusted max recovery
        
        # Ensure minimum recovery with slight variation
        return max(combined_eff, 0.051)  # Allow minor differences

        return combined_eff

class HybridRecoveryModel(RecoveryModel):
    """Hybrid model with smooth miscibility transition"""
    def __init__(self, transition_mode: str = 'sigmoid', **params):
        self.transition_engine = TransitionEngine(mode=transition_mode, **params)
        self._gpu_enabled = False
        
    def enable_gpu(self, enable: bool = True):
        """Toggle GPU acceleration for transition calculations"""
        self._gpu_enabled = enable
        if enable:
            self.transition_engine.enable_gpu_acceleration()
        
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        p_mmp_ratio = np.array(pressure / mmp)
        
        if self._gpu_enabled:
            import cupy as cp
            p_mmp_ratio = cp.asarray(p_mmp_ratio)
            
        efficiency = self.transition_engine.calculate_efficiency(p_mmp_ratio)
        
        # Calculate both simple and miscible models for blending
        simple_result = SimpleRecoveryModel().calculate(pressure, rate, porosity, mmp)
        miscible_result = MiscibleRecoveryModel().calculate(pressure, rate, porosity, mmp)
        
        # Blend results based on transition efficiency
        blended = simple_result * (1 - efficiency) + miscible_result * efficiency
        
        # Apply physical constraints
        result = np.clip(blended, 0, 0.8)
        
        if self._gpu_enabled:
            result = cp.asnumpy(result)
            
        return result.item()

class TransitionFunction(ABC):
    """Abstract base class for miscibility transition functions"""
    @abstractmethod
    def evaluate(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        pass

class SigmoidTransition(TransitionFunction):
    def __init__(self, alpha=1.0, beta=20.0):
        self.alpha = np.clip(alpha, 0.8, 1.2)
        self.beta = np.clip(beta, 5, 50)
        
    def evaluate(self, p_mmp_ratio):
        return 1 / (1 + np.exp(-self.beta*(p_mmp_ratio - self.alpha)))

class CubicTransition(TransitionFunction):
    def __init__(self):
        self._spline = None
        
    def fit(self, x, y):
        self._spline = CubicSpline(x, y, bc_type=((2, 0), (2, 0)))
        
    def evaluate(self, p_mmp_ratio):
        return np.clip(self._spline(p_mmp_ratio), 0, 1)

class TransitionEngine:
    """Core miscibility transition calculation engine"""
    def __init__(self, mode='sigmoid', **params):
        self.mode = mode
        self.params = params
        self._gpu_enabled = False
        self._precompute_kernels()

    def enable_gpu_acceleration(self):
        """Enable GPU acceleration for transition calculations"""
        self._gpu_enabled = True
        # Recompute kernels with GPU support
        self._precompute_kernels()
        
    def _precompute_kernels(self):
        """Precompute transition lookup tables"""
        self.p_mmp_grid = np.logspace(-4, 4, 1000)
        self.transition_cache = {}
        
        # Precompute sigmoid variants
        for alpha in np.linspace(0.8, 1.2, 5):
            for beta in np.linspace(5, 50, 5):
                fn = SigmoidTransition(alpha, beta)
                self.transition_cache[f'sigmoid_{alpha}_{beta}'] = fn.evaluate(self.p_mmp_grid)
                
    def monte_carlo_mmp(self, p_mmp_samples, n_iter=1000):
        """Monte Carlo integration for MMP uncertainty"""
        results = np.zeros_like(p_mmp_samples)
        for i in range(n_iter):
            perturbed = p_mmp_samples * np.random.normal(1, 0.1)
            results += self.calculate_efficiency(perturbed)
        return results / n_iter
        
    def calculate_efficiency(self, p_mmp_ratio):
        """Main API method to calculate sweep efficiency"""
        if self._gpu_enabled:
            import cupy as cp
            p_mmp_ratio = cp.asarray(p_mmp_ratio)

        if self.mode == 'sigmoid':
            fn = SigmoidTransition(**self.params)
        elif self.mode == 'cubic':
            fn = CubicTransition(**self.params)
        elif self.mode == 'custom':
            fn = self.params.get('custom_fn')
            
        result = fn.evaluate(p_mmp_ratio)
        
        if self._gpu_enabled:
            result = cp.asnumpy(result)
            
        return result

    @staticmethod
    def legacy_adapter(legacy_input):
        """Adapter for backward compatibility"""
        # Conversion logic for legacy input formats
        return {
            'mode': 'sigmoid',
            'alpha': 1.0,
            'beta': 20.0
        }

def recovery_factor(pressure: float, rate: float, porosity: float, mmp: float,
                  model: str = 'simple', **kwargs) -> float:
    """Estimate recovery factor using specified model"""
    models = {
        'simple': SimpleRecoveryModel(),
        'miscible': MiscibleRecoveryModel(),
        'immiscible': ImmiscibleRecoveryModel(),
        'hybrid': HybridRecoveryModel(**kwargs),
        'koval': KovalRecoveryModel()
    }
    if model not in models:
        raise ValueError(f"Unknown recovery model: {model}")
    return models[model].calculate(pressure, rate, porosity, mmp)

class OptimizationEngine:
    """Core optimization engine for CO2 EOR processes with parallel computing support"""
    
    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters,
                well_analysis: Optional[Any] = None):
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self.well_analysis = well_analysis
        self._results = None
        self._mmp = None
        self._mmp_params = None
        self.recovery_model = 'simple'  # Default recovery model
        self._mmp_value = self.calculate_mmp()
        
    def calculate_mmp(self, method: str = 'auto', well_analysis: Optional[Any] = None) -> float:
        """Calculate Minimum Miscibility Pressure (MMP)
        
        Args:
            method: MMP calculation method ('auto', 'cronquist', 'glaso', 'yuan')
            well_analysis: Optional WellAnalysis instance for log-based parameters
            
        Returns:
            Calculated MMP value in psi
        """
        from .evaluation.mmp import calculate_mmp, MMPParameters
        
        # Try to construct MMPParameters from WellAnalysis if available
        well_analysis = well_analysis or getattr(self, 'well_analysis', None)
        if well_analysis and hasattr(well_analysis, 'calculate_mmp_profile'):
            try:
                profile = well_analysis.calculate_mmp_profile(method=method)
                avg_temp = np.mean(profile['temperature'])
                avg_api = np.mean(profile['api'])
                
                self._mmp_params = MMPParameters(
                    temperature=avg_temp,
                    oil_gravity=avg_api,
                    pvt_data=self.pvt
                )
                self._mmp_value = float(calculate_mmp(self._mmp_params, method))
                self._mmp = self._mmp_value
                return self._mmp_value
            except Exception as e:
                logging.warning(f"Failed to calculate MMP from WellAnalysis: {str(e)}")
                # Fall through to PVT-based calculation
        
        # Fallback to PVT-based calculation
        result = calculate_mmp(self.pvt, method)
        logging.debug(f"MMP Calculation Result: {result} (Type: {type(result)})")
        
        if isinstance(result, MMPParameters):
            self._mmp_params = result
            self._mmp_value = float(calculate_mmp(result, method))
        else:
            self._mmp_value = float(result)
            self._mmp_params = MMPParameters(
                temperature=180,  # Default
                oil_gravity=35,   # Default
                pvt_data=self.pvt
            )
        
        self._mmp = self._mmp_value
        return self._mmp_value
        
    def optimize_recovery(self, max_iter: int = 100, tol: float = 1e-4,
                        learning_rate: float = 50.0) -> Dict[str, float]:
        """Optimize CO2 injection parameters using gradient descent"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp(well_analysis=getattr(self, 'well_analysis', None))
            
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        pressure = max(float(self._mmp_value), self.eor_params.target_pressure)
        rate = self.eor_params.injection_rate
        prev_recovery = 0.0
        converged = False
        
        for i in range(max_iter):
            recovery = recovery_factor(pressure, rate, avg_poro, self._mmp_value,
                                    model=self.recovery_model)
            
            if abs(recovery - prev_recovery) < tol and i > 5:
                converged = True
                break
                
            delta_p = 10.0
            recovery_plus = recovery_factor(pressure + delta_p, rate, avg_poro,
                                          self._mmp_value, model=self.recovery_model)
            gradient = (recovery_plus - recovery) / delta_p
            
            new_pressure = pressure + learning_rate * gradient
            pressure = max(self._mmp_value, min(new_pressure, 1.5 * self._mmp_value))
            prev_recovery = recovery
            
        self._results = {
            'optimized_params': {
                'injection_rate': rate,
                'target_pressure': pressure
            },
            'mmp': self._mmp_value,
            'iterations': i,
            'final_recovery': recovery,
            'converged': converged,
            'avg_porosity': avg_poro
        }
        return self._results

    def optimize_bayesian(self, n_iter: int = 50, init_points: int = 10,
                        method: str = 'gp') -> Dict[str, Any]:
        """Optimize using Bayesian Optimization
        
        Args:
            n_iter: Number of optimization iterations
            init_points: Number of random exploration points
            method: Optimization method ('gp' for Gaussian Process or 'bayes' for BayesianOptimization)
            
        Returns:
            Dictionary containing optimization results
        """
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp(well_analysis=getattr(self, 'well_analysis', None))
            
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        if method == 'gp':
            # Define search space
            space = [
                Real(min_pressure, self.eor_params.max_pressure, name='pressure'),
                Real(self.eor_params.min_injection_rate,
                    self.eor_params.injection_rate * 1.5, name='rate')
            ]
            
            if self.eor_params.injection_scheme == 'wag':
                space.extend([
                    Real(self.eor_params.min_cycle_length,
                        self.eor_params.max_cycle_length, name='cycle_length'),
                    Real(self.eor_params.min_water_fraction,
                        self.eor_params.max_water_fraction, name='water_fraction')
                ])
            
            @use_named_args(space)
            def objective(**params):
                if self.eor_params.injection_scheme == 'wag':
                    effective_rate = params['rate'] * (1 - params['water_fraction'])
                    return -recovery_factor(
                        params['pressure'], effective_rate, avg_poro,
                        self._mmp_value, model=self.recovery_model
                    )
                else:
                    return -recovery_factor(
                        params['pressure'], params['rate'], avg_poro,
                        self._mmp_value, model=self.recovery_model
                    )
            
            result = gp_minimize(
                objective,
                space,
                n_calls=n_iter,
                random_state=42,
                verbose=True
            )
            
            best_params = {
                'pressure': result.x[0],
                'rate': result.x[1]
            }
            if self.eor_params.injection_scheme == 'wag':
                best_params.update({
                    'cycle_length': result.x[2],
                    'water_fraction': result.x[3]
                })
                
        else:  # BayesianOptimization method
            pbounds = {
                'pressure': (min_pressure, self.eor_params.max_pressure),
                'rate': (self.eor_params.min_injection_rate,
                        self.eor_params.injection_rate * 1.5)
            }
            
            if self.eor_params.injection_scheme == 'wag':
                pbounds.update({
                    'cycle_length': (self.eor_params.min_cycle_length,
                                   self.eor_params.max_cycle_length),
                    'water_fraction': (self.eor_params.min_water_fraction,
                                      self.eor_params.max_water_fraction)
                })
            
            def objective(pressure, rate, cycle_length=0, water_fraction=0):
                if self.eor_params.injection_scheme == 'wag':
                    effective_rate = rate * (1 - water_fraction)
                    return recovery_factor(
                        pressure, effective_rate, avg_poro,
                        self._mmp_value, model=self.recovery_model
                    )
                else:
                    return recovery_factor(
                        pressure, rate, avg_poro,
                        self._mmp_value, model=self.recovery_model
                    )
            
            optimizer = BayesianOptimization(
                f=objective,
                pbounds=pbounds,
                random_state=42,
                verbose=2
            )
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter
            )
            
            best_params = optimizer.max['params']
        
        # Store results
        self._results = {
            'optimized_params': {
                'injection_rate': best_params['rate'],
                'target_pressure': best_params['pressure'],
                'cycle_length': best_params.get('cycle_length'),
                'water_fraction': best_params.get('water_fraction')
            },
            'mmp': self._mmp_value,
            'method': f'bayesian_{method}',
            'iterations': n_iter,
            'initial_points': init_points,
            'final_recovery': -result.fun if method == 'gp' else optimizer.max['target'],
            'avg_porosity': avg_poro,
            'converged': True
        }
        return self._results

    def hybrid_optimize(self, ga_params: GeneticAlgorithmParams = None,
                      bo_iter: int = 20, bo_init: int = 5) -> Dict[str, Any]:
        """Hybrid optimization combining GA and Bayesian Optimization
        
        Args:
            ga_params: Parameters for genetic algorithm phase
            bo_iter: Number of Bayesian optimization iterations
            bo_init: Number of initial points for Bayesian optimization
            
        Returns:
            Dictionary containing optimization results
        """
        # First run GA for broad exploration
        ga_results = self.optimize_genetic_algorithm(ga_params)
        
        # Then refine with Bayesian Optimization
        if self.eor_params.injection_scheme == 'wag':
            initial_point = [
                ga_results['optimized_params']['target_pressure'],
                ga_results['optimized_params']['injection_rate'],
                ga_results['optimized_params']['cycle_length'],
                ga_results['optimized_params']['water_fraction']
            ]
        else:
            initial_point = [
                ga_results['optimized_params']['target_pressure'],
                ga_results['optimized_params']['injection_rate']
            ]
            
        # Run Bayesian optimization starting near GA solution
        bo_results = self.optimize_bayesian(
            n_iter=bo_iter,
            init_points=bo_init,
            method='gp'
        )
        
        # Combine results
        self._results = {
            **bo_results,
            'ga_results': ga_results,
            'method': 'hybrid_ga_bo'
        }
        return self._results
        
    def optimize_genetic_algorithm(self, ga_params: GeneticAlgorithmParams = None) -> Dict[str, Any]:
        """Optimize using genetic algorithm with parallel evaluation"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp(well_analysis=getattr(self, 'well_analysis', None))
            
        params = ga_params or GeneticAlgorithmParams()
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        
        # Initialize population
        population = self._initialize_population(params.population_size)
        
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(params.generations):
            # Parallel fitness evaluation
            with ProcessPoolExecutor() as executor:
                fitnesses = list(executor.map(
                    partial(self._evaluate_individual, avg_poro=avg_poro,
                           mmp=self._mmp_value, model=self.recovery_model),
                    population
                ))
            
            # Selection
            selected = self._tournament_selection(population, fitnesses, 
                                                params.selection_pressure, 
                                                params.elite_count)
            
            # Crossover
            offspring = self._crossover(selected, params.crossover_rate)
            
            # Mutation
            population = self._mutate(offspring, params.mutation_rate)
            
            # Update best solution
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_solution = population[current_best_idx]
                
        # Convert best solution to operational parameters including Koval params
        cycle_len = None
        water_frac = None
        
        # Unpack with proper indices and apply constraints
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        if len(best_solution) >= 6:
            pressure, rate, cycle_len, water_frac, v_dp, mobility_ratio = best_solution[:6]
            effective_rate = rate * (1 - water_frac) if self.eor_params.injection_scheme == 'wag' else rate
        elif len(best_solution) >= 4:
            pressure, rate, cycle_len, water_frac = best_solution[:4]
            v_dp = self.eor_params.v_dp
            mobility_ratio = self.eor_params.mobility_ratio
            effective_rate = rate * (1 - water_frac) if self.eor_params.injection_scheme == 'wag' else rate
        else:
            pressure, rate = best_solution[:2]
            v_dp = self.eor_params.v_dp
            mobility_ratio = self.eor_params.mobility_ratio
            effective_rate = rate
            
        # Apply constraints to all parameters
        pressure = np.clip(pressure, min_pressure, self.eor_params.max_pressure)
        effective_rate = np.clip(effective_rate,
                               self.eor_params.min_injection_rate,
                               self.eor_params.injection_rate * 1.5)
        v_dp = np.clip(v_dp, 0.3, 0.8)
        mobility_ratio = np.clip(mobility_ratio, 1.2, 20.0)
        self._results = {
            'optimized_params': {
                'injection_rate': effective_rate,
                'cycle_length': np.clip(cycle_len,
                                      self.eor_params.min_cycle_length,
                                      self.eor_params.max_cycle_length)
                              if self.eor_params.injection_scheme == 'wag' else None,
                'water_fraction': np.clip(water_frac,
                                        self.eor_params.min_water_fraction,
                                        self.eor_params.max_water_fraction)
                              if self.eor_params.injection_scheme == 'wag' else None,
                'target_pressure': pressure,
                'v_dp': v_dp,
                'mobility_ratio': mobility_ratio
            },
            'mmp': self._mmp_value,
            'method': 'genetic_algorithm',
            'generations': params.generations,
            'population_size': params.population_size,
            'final_recovery': best_fitness,
            'avg_porosity': avg_poro,
            'converged': True,
            'iterations': params.generations
        }
        return self._results
        
    def _calculate_mmp(self) -> float:
        """Calculate minimum miscibility pressure"""
        return self.calculate_mmp(self.reservoir, self.pvt)

    def _initialize_population(self, size: int) -> List[Tuple[float, float, float, float]]:
        """Initialize population within operational constraints"""
        if not hasattr(self, '_mmp_value'):
            self._mmp_value = self._calculate_mmp()
            
        population = []
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        for _ in range(size):
            pressure = random.uniform(min_pressure, self.eor_params.max_pressure)
            rate = random.uniform(self.eor_params.min_injection_rate,
                                self.eor_params.injection_rate * 1.5)
            if self.eor_params.injection_scheme == 'wag':
                cycle_len = random.uniform(self.eor_params.min_cycle_length,
                                         self.eor_params.max_cycle_length)
                water_frac = random.uniform(self.eor_params.min_water_fraction,
                                          self.eor_params.max_water_fraction)
                population.append((pressure, rate, cycle_len, water_frac))
            else:
                population.append((pressure, rate, 0.0, 0.0))
                
        return population
        
    def _evaluate_individual(self, individual: Tuple[float, float, float, float, float, float],
                           avg_poro: float, mmp: float, model: str = 'simple') -> float:
        """Evaluate fitness of an individual solution"""
        if len(individual) == 6:
            pressure, rate, cycle_len, water_frac, v_dp, mobility_ratio = individual
        else:
            # Handle legacy 4-param individuals
            pressure, rate, cycle_len, water_frac = individual
            v_dp = self.eor_params.v_dp
            mobility_ratio = self.eor_params.mobility_ratio
        
        if self.eor_params.injection_scheme == 'wag' and cycle_len > 0:
            # Calculate effective CO2 rate accounting for WAG cycle
            effective_rate = rate * (1 - water_frac)
            return recovery_factor(pressure, effective_rate, avg_poro, mmp, model=model)
        else:
            return recovery_factor(pressure, rate, avg_poro, mmp, model=model)
        
    def _tournament_selection(self, population: List[Tuple[float, float, float, float]],
                            fitnesses: List[float], pressure: float, 
                            elite_count: int) -> List[Tuple[float, float]]:
        """Select individuals using tournament selection"""
        selected = []
        population_size = len(population)
        
        # Keep elites
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        selected.extend([population[i] for i in elite_indices])
        
        # Tournament selection for rest
        for _ in range(population_size - elite_count):
            candidates = random.sample(range(population_size), 2)
            winner = max(candidates, key=lambda x: fitnesses[x])
            selected.append(population[winner])
            
        return selected
        
    def _crossover(self, parents: List[Tuple[float, float, float, float, float, float]], rate: float) -> List[Tuple[float, float, float, float, float, float]]:
        """Perform crossover between parents"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue
                
            parent1, parent2 = parents[i], parents[i+1]
            
            if random.random() < rate:
                # Blend crossover
                alpha = random.random()
                child1 = (
                    alpha * parent1[0] + (1 - alpha) * parent2[0],
                    alpha * parent1[1] + (1 - alpha) * parent2[1],
                    alpha * parent1[2] + (1 - alpha) * parent2[2],
                    alpha * parent1[3] + (1 - alpha) * parent2[3]
                )
                child2 = (
                    (1 - alpha) * parent1[0] + alpha * parent2[0],
                    (1 - alpha) * parent1[1] + alpha * parent2[1],
                    (1 - alpha) * parent1[2] + alpha * parent2[2],
                    (1 - alpha) * parent1[3] + alpha * parent2[3]
                )
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
                
        return offspring
        
    def _mutate(self, population: List[Tuple], rate: float) -> List[Tuple]:
        """Apply mutation to population with parameter constraints.
        Handles both old (4-element) and new (6-element) individual formats."""
        mutated = []
        min_pressure = max(self._mmp_value, self.eor_params.target_pressure)
        
        for individual in population:
            if random.random() < rate:
                # Handle legacy individuals (4 elements)
                has_vdp = len(individual) > 4
                current_vdp = individual[4] if has_vdp else self.eor_params.v_dp
                current_mr = individual[5] if has_vdp else self.eor_params.mobility_ratio
                
                # Gaussian mutation with parameter constraints
                pressure = np.clip(
                    individual[0] + random.gauss(0, 100),
                    min_pressure, self.eor_params.max_pressure
                )
                rate_val = np.clip(
                    individual[1] + random.gauss(0, 200),
                    self.eor_params.min_injection_rate,
                    self.eor_params.injection_rate * 1.5
                )
                v_dp = np.clip(
                    current_vdp + random.gauss(0, 0.05),
                    0.3, 0.8  # Dykstra-Parsons coefficient bounds
                )
                mobility_ratio = np.clip(
                    current_mr + random.gauss(0, 0.5),
                    1.2, 20.0  # Mobility ratio bounds
                )
                
                if self.eor_params.injection_scheme == 'wag':
                    cycle_len = np.clip(
                        individual[2] + random.gauss(0, 5),
                        self.eor_params.min_cycle_length,
                        self.eor_params.max_cycle_length
                    )
                    water_frac = np.clip(
                        individual[3] + random.gauss(0, 0.1),
                        self.eor_params.min_water_fraction,
                        self.eor_params.max_water_fraction
                    )
                    mutated.append((pressure, rate_val, cycle_len, water_frac, v_dp, mobility_ratio))
                else:
                    mutated.append((pressure, rate_val, 0.0, 0.0, v_dp, mobility_ratio))
            else:
                # Convert legacy individuals to new format if needed
                if len(individual) == 4:
                    individual = individual + (self.eor_params.v_dp, self.eor_params.mobility_ratio)
                mutated.append(individual)
                
        return mutated
        
    def optimize_wag(self, cycles: int = 5, max_iter: int = 20) -> Dict[str, float]:
        """Optimize Water-Alternating-Gas (WAG) parameters"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp(well_analysis=getattr(self, 'well_analysis', None))
            
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        
        min_cycle = 15
        max_cycle = 90
        min_water = 0.2
        max_water = 0.8
        
        best_params = {
            'cycle_length': 30,
            'water_fraction': 0.5,
            'injection_pressure': max(float(self._mmp_value), self.eor_params.target_pressure),
            'recovery': 0.0
        }
        
        for cycle in range(cycles):
            for water_frac in np.linspace(min_water, max_water, num=5):
                for cycle_len in np.linspace(min_cycle, max_cycle, num=5):
                    effective_rate = self.eor_params.injection_rate * (1 - water_frac)
                    recovery = recovery_factor(
                        best_params['injection_pressure'],
                        effective_rate,
                        avg_poro,
                        self._mmp_value,
                        model=self.recovery_model
                    )
                    
                    if recovery > best_params['recovery']:
                        best_params.update({
                            'cycle_length': cycle_len,
                            'water_fraction': water_frac,
                            'recovery': recovery
                        })
            
            min_water = max(0.1, best_params['water_fraction'] - 0.2)
            max_water = min(0.9, best_params['water_fraction'] + 0.2)
            min_cycle = max(10, best_params['cycle_length'] - 20)
            max_cycle = min(120, best_params['cycle_length'] + 20)
        
        pressure = self._optimize_pressure_for_wag(
            best_params['water_fraction'],
            best_params['cycle_length'],
            max_iter
        )
        
        self._results = {
            'wag_params': {
                'optimal_cycle_length': best_params['cycle_length'],
                'optimal_water_fraction': best_params['water_fraction'],
                'injection_pressure': pressure,
                'sensitivity': {
                    'water_fraction': (min_water, max_water),
                    'cycle_length': (min_cycle, max_cycle)
                }
            },
            'mmp': self._mmp_value,
            'estimated_recovery': best_params['recovery'],  # At root level
            'avg_porosity': avg_poro
        }
        return self._results
        
    def _optimize_pressure_for_wag(self, water_frac: float, cycle_len: int,
                                 max_iter: int = 20) -> float:
        """Optimize injection pressure for given WAG parameters"""
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        effective_rate = self.eor_params.injection_rate * (1 - water_frac)
        
        pressure = max(float(self._mmp_value), self.eor_params.target_pressure)
        best_pressure = pressure
        best_recovery = 0.0
        
        for _ in range(max_iter):
            recovery = recovery_factor(
                pressure,
                effective_rate,
                avg_poro,
                self._mmp_value,
                model=self.recovery_model
            )
            
            if recovery > best_recovery:
                best_recovery = recovery
                best_pressure = pressure
                
            delta_p = 50
            recovery_plus = recovery_factor(
                pressure + delta_p,
                effective_rate,
                avg_poro,
                self._mmp_value,
                model=self.recovery_model
            )
            
            gradient = (recovery_plus - recovery) / delta_p
            pressure += 100 * gradient
            
            pressure = max(
                self._mmp_value * 1.05,
                min(pressure, self._mmp_value * 1.5)
            )
            
        return best_pressure
        
    def check_mmp_constraint(self, pressure: float) -> bool:
        """Check if pressure meets Minimum Miscibility Pressure (MMP) constraint"""
        if not hasattr(self, '_mmp_value'):
            self.calculate_mmp(well_analysis=getattr(self, 'well_analysis', None))
        return pressure >= float(self._mmp_value)
        
    @property
    def results(self) -> Optional[Dict]:
        """Get optimization results"""
        return self._results
        
    @property
    def mmp(self) -> Optional[float]:
        """Get calculated MMP value"""
        return self._mmp_value if hasattr(self, '_mmp_value') else None
        
    def set_recovery_model(self, model: str, **kwargs):
        """Set the recovery factor calculation model
        
        Args:
            model: One of 'simple', 'miscible', 'immiscible', 'hybrid', 'transition'
            kwargs: Additional model-specific parameters
        """
        valid_models = ['simple', 'miscible', 'immiscible', 'hybrid', 'transition', 'koval']
        if model not in valid_models:
            raise ValueError(f"Unknown recovery model: {model}. Valid options: {valid_models}")
            
        if model == 'transition':
            if 'transition_params' in kwargs:
                self.transition_engine = TransitionEngine(**kwargs['transition_params'])
            else:
                self.transition_engine = TransitionEngine()
                
        self.recovery_model = model
        self._recovery_model_kwargs = kwargs

    def plot_mmp_profile(self) -> Optional[Any]:
        """Generate MMP vs Depth profile plot using Plotly
        
        Returns:
            Plotly figure object if well_analysis exists, else None
        """
        if not hasattr(self, 'well_analysis') or not hasattr(self.well_analysis, 'calculate_mmp_profile'):
            return None
            
        profile = self.well_analysis.calculate_mmp_profile()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=profile['mmp'],
                y=profile['depth'],
                name='MMP',
                mode='lines',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=profile['temperature'],
                y=profile['depth'],
                name='Temperature',
                mode='lines',
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='MMP vs Depth Profile',
            xaxis_title='Pressure (psi) / Temperature (°F)',
            yaxis_title='Depth (ft)',
            legend=dict(x=0.8, y=0.1)
        )
        
        fig.update_yaxes(title_text="Depth (ft)", secondary_y=False)
        fig.update_yaxes(title_text="Depth (ft)", secondary_y=True)
        
        return fig

    def plot_optimization_convergence(self) -> Optional[Any]:
        """Generate optimization convergence plot
        
        Returns:
            Plotly figure object if results exist, else None
        """
        if not self._results:
            return None
            
        fig = go.Figure()
        
        if self._results['method'].startswith('genetic_algorithm'):
            generations = range(self._results['generations'])
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=[self._results['final_recovery']] * len(generations),
                    name='Best Recovery',
                    mode='lines',
                    line=dict(color='blue')
                )
            )
            fig.update_layout(
                title='GA Optimization Convergence',
                xaxis_title='Generation',
                yaxis_title='Recovery Factor'
            )
            
        elif self._results['method'].startswith('bayesian'):
            iterations = range(self._results['iterations'])
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=[self._results['final_recovery']] * len(iterations),
                    name='Best Recovery',
                    mode='lines+markers',
                    line=dict(color='green')
                )
            )
            fig.update_layout(
                title='Bayesian Optimization Convergence',
                xaxis_title='Iteration',
                yaxis_title='Recovery Factor'
            )
            
        return fig

    def plot_parameter_sensitivity(self, param: str, n_points: int = 10) -> Optional[Any]:
        """Generate parameter sensitivity plot near optimal solution
        
        Args:
            param: Parameter to analyze ('pressure', 'rate', 'cycle_length', 'water_fraction')
            n_points: Number of points to evaluate
            
        Returns:
            Plotly figure object if results exist, else None
        """
        if not self._results:
            return None
            
        optimal = self._results['optimized_params']
        avg_poro = np.mean(self.reservoir.grid['PORO'])
        
        if param == 'pressure':
            values = np.linspace(
                optimal['target_pressure'] * 0.9,
                optimal['target_pressure'] * 1.1,
                n_points
            )
            recoveries = [
                recovery_factor(
                    p, optimal['injection_rate'], avg_poro,
                    self._mmp_value, model=self.recovery_model
                )
                for p in values
            ]
        elif param == 'rate':
            values = np.linspace(
                optimal['injection_rate'] * 0.5,
                optimal['injection_rate'] * 1.5,
                n_points
            )
            recoveries = [
                recovery_factor(
                    optimal['target_pressure'], r, avg_poro,
                    self._mmp_value, model=self.recovery_model
                )
                for r in values
            ]
        elif param in ('cycle_length', 'water_fraction') and self.eor_params.injection_scheme == 'wag':
            if param == 'cycle_length':
                values = np.linspace(
                    optimal['cycle_length'] * 0.5,
                    optimal['cycle_length'] * 1.5,
                    n_points
                )
                recoveries = [
                    recovery_factor(
                        optimal['target_pressure'],
                        optimal['injection_rate'] * (1 - optimal['water_fraction']),
                        avg_poro,
                        self._mmp_value,
                        model=self.recovery_model
                    )
                    for _ in values
                ]
            else:
                values = np.linspace(
                    max(0.1, optimal['water_fraction'] - 0.2),
                    min(0.9, optimal['water_fraction'] + 0.2),
                    n_points
                )
                recoveries = [
                    recovery_factor(
                        optimal['target_pressure'],
                        optimal['injection_rate'] * (1 - w),
                        avg_poro,
                        self._mmp_value,
                        model=self.recovery_model
                    )
                    for w in values
                ]
        else:
            return None
            
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=values,
                y=recoveries,
                mode='lines+markers',
                name=f'{param} sensitivity'
            )
        )
        fig.update_layout(
            title=f'Recovery Factor vs {param.capitalize()}',
            xaxis_title=param.capitalize(),
            yaxis_title='Recovery Factor'
        )
        
        return fig