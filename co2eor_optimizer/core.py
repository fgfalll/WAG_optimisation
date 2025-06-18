from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Any, Tuple, Union, Type
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from abc import ABC, abstractmethod
import random

from scipy.interpolate import CubicSpline
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from bayes_opt import BayesianOptimization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming config_manager.py is in the same directory or accessible via PYTHONPATH
# The application's main entry point is responsible for calling config_manager.load_config()
try:
    from config_manager import config_manager, ConfigNotLoadedError
except ImportError:
    logging.critical(
        "ConfigManager could not be imported. Please ensure config_manager.py is accessible. "
        "Application functionality will be severely limited or fail."
    )
    # Create a dummy config_manager to prevent immediate crashes if accessed before proper load.
    # However, if config is required, this will eventually lead to ConfigNotLoadedError.
    class DummyConfigManager:
        def get(self, key_path: str, default: Any = None) -> Any:
            if default is None: # Simulate raising error if key must exist
                 raise ConfigNotLoadedError(f"DummyConfig: Critical key '{key_path}' access attempted before load.")
            return default
        def get_section(self, section_key: str) -> Optional[Dict[str, Any]]:
             res = self.get(section_key, {})
             return res if isinstance(res, dict) else {}

        def load_config(self, file_path: str) -> None:
            logging.error("DummyConfigManager: load_config called but not functional.")
        @property
        def is_loaded(self) -> bool: return False
    config_manager = DummyConfigManager()


# Check for GPU availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logging.warning("cupy not installed. GPU acceleration will be disabled.")

# Helper to create dataclass instances from dict, only using keys present in the dataclass
def from_dict_to_dataclass(cls, data: Dict[str, Any]):
    field_names = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

@dataclass
class WellData:
    name: str
    depths: np.ndarray
    properties: Dict[str, np.ndarray]
    units: Dict[str, str]

    def validate(self) -> bool:
        if not self.properties: return True # No properties to validate against depths
        if not hasattr(self.depths, 'size') or self.depths.size == 0:
            # If depths is empty, all property arrays must also be empty
            return all(not hasattr(prop_array, 'size') or prop_array.size == 0 for prop_array in self.properties.values())
        return all(len(self.depths) == len(prop_array) for prop_array in self.properties.values())

@dataclass
class ReservoirData:
    grid: Dict[str, np.ndarray]
    pvt_tables: Dict[str, np.ndarray]
    regions: Optional[Dict[str, np.ndarray]] = None
    runspec: Optional[Dict[str, Any]] = None
    faults: Optional[Dict[str, Any]] = None

    def set_faults_data(self, faults_data: Dict[str, Any]) -> None:
        self.faults = faults_data

@dataclass
class PVTProperties:
    oil_fvf: np.ndarray
    oil_viscosity: np.ndarray
    gas_fvf: np.ndarray
    gas_viscosity: np.ndarray
    rs: np.ndarray  # Solution GOR
    pvt_type: str  # 'black_oil' or 'compositional'
    gas_specific_gravity: float
    temperature: float  # Reservoir temperature in °F

    def __post_init__(self):
        arrays = [self.oil_fvf, self.oil_viscosity, self.gas_fvf, self.gas_viscosity, self.rs]
        # Check if any array is not None and then check lengths
        non_none_arrays = [arr for arr in arrays if arr is not None and hasattr(arr, '__len__')]
        if non_none_arrays:
            if len({len(arr) for arr in non_none_arrays}) > 1:
                raise ValueError("All provided PVT property arrays must have the same length.")
        if self.pvt_type not in {'black_oil', 'compositional'}:
            raise ValueError("pvt_type must be either 'black_oil' or 'compositional'")
        if not (isinstance(self.gas_specific_gravity, (int, float)) and 0.5 <= self.gas_specific_gravity <= 1.2):
            raise ValueError(f"Gas specific gravity must be between 0.5-1.2, got {self.gas_specific_gravity}")
        if not (isinstance(self.temperature, (int, float)) and 50 <= self.temperature <= 400):
            raise ValueError(f"Temperature must be between 50-400°F, got {self.temperature}")

@dataclass
class EORParameters:
    injection_rate: float = 5000.0 # Default fallback if not in config
    WAG_ratio: Optional[float] = None
    injection_scheme: str = 'continuous'
    min_cycle_length_days: float = 15.0
    max_cycle_length_days: float = 90.0
    min_water_fraction: float = 0.2
    max_water_fraction: float = 0.8
    target_pressure_psi: float = 0.0
    max_pressure_psi: float = 6000.0
    min_injection_rate_bpd: float = 1000.0
    v_dp_coefficient: float = 0.55
    mobility_ratio: float = 2.5

    def __post_init__(self):
        if not (isinstance(self.v_dp_coefficient, (int, float)) and 0.3 <= self.v_dp_coefficient <= 0.8):
            raise ValueError(f"V_DP coefficient must be between 0.3-0.8, got {self.v_dp_coefficient}")
        if not (isinstance(self.mobility_ratio, (int, float)) and 1.2 <= self.mobility_ratio <= 20):
            raise ValueError(f"Mobility ratio must be between 1.2-20, got {self.mobility_ratio}")

    @classmethod
    def from_config_dict(cls, config_eor_params_dict: Dict[str, Any], **kwargs):
        # Start with class defaults
        defaults = {f.name: f.default for f in fields(cls) if f.default is not field.MISSING}
        # Update with values from the passed dictionary
        defaults.update(config_eor_params_dict)
        # Update with any explicit kwargs (highest precedence)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclass
class GeneticAlgorithmParams:
    population_size: int = 60 # Default fallback
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    # selection_pressure: float = 1.5 # Not currently used in tournament selection
    elite_count: int = 3
    tournament_size: int = 3 # Used in tournament selection
    blend_alpha_crossover: float = 0.5 # For blend crossover
    mutation_strength_factor: float = 0.1 # Relative strength of mutation

    @classmethod
    def from_config_dict(cls, config_ga_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in fields(cls) if f.default is not field.MISSING}
        defaults.update(config_ga_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclass
class RecoveryModel(ABC):
    @abstractmethod
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        pass

    def _validate_inputs(self, pressure: float, rate: float, porosity: float, mmp: float) -> None:
        if not all(isinstance(x, (int, float)) for x in [pressure, rate, porosity, mmp]):
            raise ValueError("All parameters (pressure, rate, porosity, mmp) must be numeric.")
        if any(x <= 0 for x in [pressure, porosity, mmp]) or rate < 0: # rate can be 0
            raise ValueError("Pressure, porosity, MMP must be positive. Rate must be non-negative.")

class KovalRecoveryModel(RecoveryModel):
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        v_dp = kwargs.get('v_dp_coefficient', config_manager.get("RecoveryModelKwargsDefaults.Koval.v_dp_coefficient", 0.5))
        mobility_ratio_param = kwargs.get('mobility_ratio', config_manager.get("RecoveryModelKwargsDefaults.Koval.mobility_ratio", 2.0))
        mu_co2 = kwargs.get('mu_co2', config_manager.get("RecoveryModelKwargsDefaults.Koval.mu_co2", 0.1))
        mu_oil = kwargs.get('mu_oil', config_manager.get("RecoveryModelKwargsDefaults.Koval.mu_oil", 5.0))

        if porosity > 0.4: logging.warning(f"Koval: Porosity {porosity} > 0.4 seems high.")
        if mmp < 500 or mmp > 15000: logging.warning(f"Koval: MMP {mmp} psi is outside typical 1000-10000 range.")
        if not (0.3 <= v_dp <= 0.8): raise ValueError(f"Koval: V_DP must be between 0.3-0.8, got {v_dp}")
        if not (1.2 <= mobility_ratio_param <= 20): raise ValueError(f"Koval: Mobility ratio param must be between 1.2-20, got {mobility_ratio_param}")
        if mu_co2 <= 0 or mu_oil <= 0: raise ValueError("Koval: Viscosities must be positive.")

        kv_effective = v_dp * (1 + (mu_co2 / mu_oil))
        if kv_effective == 0: return 0.0 # Avoid division by zero
        mr_term = (mobility_ratio_param - 1) / kv_effective
        if mr_term < 0: # Typically means M < 1, which implies favorable displacement
            es = 1.0 # Or some high value capped later
        else:
            es = 1 / (1 + np.sqrt(mr_term))

        pressure_ratio = pressure / mmp if mmp > 0 else 10.0 # Avoid div by zero, assume highly miscible
        if pressure_ratio >= 1.0: return min(0.85, es)
        elif pressure_ratio >= 0.8: return es * 0.9
        return es * 0.7

class SimpleRecoveryModel(RecoveryModel):
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        safe_rate = max(rate, 1e-6) # Avoid log(0) or power of zero issues if rate is 0
        sweep_efficiency = 0.7 * (safe_rate ** 0.2) * (porosity ** 0.5)
        return min(0.7, miscibility * sweep_efficiency)

class MiscibleRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        # These are __init__ params, set once per model instance
        self.kv_factor = model_init_kwargs.get('kv_factor', config_manager.get("RecoveryModelKwargsDefaults.Miscible.kv_factor", 0.5))
        self.gravity_factor = model_init_kwargs.get('gravity_factor', config_manager.get("RecoveryModelKwargsDefaults.Miscible.gravity_factor", 0.1))
        if not (0.3 <= self.kv_factor <= 0.8): raise ValueError(f"Miscible: kv_factor must be 0.3-0.8, got {self.kv_factor}")
        if not (0 <= self.gravity_factor <= 1): raise ValueError(f"Miscible: gravity_factor must be 0-1, got {self.gravity_factor}")
        self.model_runtime_defaults = model_init_kwargs # Store all for fallback in calculate

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        # Runtime kwargs (from optimizer) take precedence over model_runtime_defaults, then hardcoded fallbacks
        mu_co2 = kwargs.get('mu_co2', self.model_runtime_defaults.get('mu_co2', 0.06))
        mu_oil = kwargs.get('mu_oil', self.model_runtime_defaults.get('mu_oil', 5.0))
        rel_perm = kwargs.get('rel_perm', self.model_runtime_defaults.get('rel_perm', {'co2': 0.8, 'oil': 0.3}))
        dip_angle = kwargs.get('dip_angle', self.model_runtime_defaults.get('dip_angle', 0.0))

        if mu_co2 <= 0 or mu_oil <= 0: raise ValueError("Miscible: Viscosities must be positive.")
        if not (isinstance(rel_perm, dict) and 'co2' in rel_perm and 'oil' in rel_perm and
                isinstance(rel_perm['co2'], (int, float)) and rel_perm['co2'] > 0 and
                isinstance(rel_perm['oil'], (int, float)) and rel_perm['oil'] > 0):
            raise ValueError("Miscible: rel_perm must be a dict with positive 'co2' and 'oil' numeric values.")

        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        M = (mu_oil * rel_perm['co2']) / (mu_co2 * rel_perm['oil'])
        safe_rate = max(rate, 1e-6)
        Ng = (self.gravity_factor * 2 * np.sin(np.radians(dip_angle))) / safe_rate
        Kv_effective = self.kv_factor * (1 + (mu_co2 / mu_oil)) # kv_factor is from __init__
        
        # Craig-Geffen-Morse for compositional sweep efficiency
        Es_comp = 0.0
        if M > -1.0: # Avoid issues with log/exp for M <= -1
            Es_comp = 1.0 - 0.5 * (1.0 / (1.0 + M)) * (1.0 - np.exp(-2.0 * (1.0 + M)))
        else: # Handle unstable M values, indicating very unfavorable conditions
            Es_comp = 0.1 # Assign a low efficiency

        Es_gravity = Es_comp * (1.0 - min(max(Ng, -1.0), 1.0)) # Ensure Ng effect is bounded

        Es_fingering = 1.0 # Default for M <= 1 (favorable)
        if M > 1.0 and Kv_effective > 1e-9: # Unfavorable, apply Koval for fingering
            Es_fingering = 1.0 / (1.0 + np.sqrt((M - 1.0) / Kv_effective))
        elif M > 1.0 and Kv_effective <= 1e-9: # Unfavorable M but near zero Kv (highly heterogeneous)
             Es_fingering = 0.1 # Low efficiency

        combined_eff = miscibility * Es_fingering * Es_gravity * (porosity ** 0.5) * (1.0 + dip_angle / 50.0)
        combined_eff = min(0.88, max(0.0, combined_eff))
        return max(combined_eff, 0.61) if pressure >= mmp * 1.2 else combined_eff

class ImmiscibleRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        self.sor = model_init_kwargs.get('sor', config_manager.get("RecoveryModelKwargsDefaults.Immiscible.sor", 0.25))
        self.krw_max = model_init_kwargs.get('krw_max', config_manager.get("RecoveryModelKwargsDefaults.Immiscible.krw_max", 0.4))
        if not (0 <= self.sor < 1): raise ValueError(f"Immiscible: sor must be 0-1, got {self.sor}")
        if not (0 < self.krw_max <= 1): raise ValueError(f"Immiscible: krw_max must be 0-1, got {self.krw_max}")
        self.model_runtime_defaults = model_init_kwargs

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        mu_water = kwargs.get('mu_water', self.model_runtime_defaults.get('mu_water', 0.5))
        mu_oil = kwargs.get('mu_oil', self.model_runtime_defaults.get('mu_oil', 5.0))
        swc = kwargs.get('swc', self.model_runtime_defaults.get('swc', 0.2))
        kro_max = kwargs.get('kro_max', self.model_runtime_defaults.get('kro_max', 0.8))

        if mu_water <=0 or mu_oil <=0: raise ValueError("Immiscible: Viscosities must be positive.")
        if not (0 <= swc < 1): raise ValueError(f"Immiscible: swc must be 0-1, got {swc}")
        if not (0 < kro_max <= 1): raise ValueError(f"Immiscible: kro_max must be 0-1, got {kro_max}")

        M_wo = (mu_oil * self.krw_max * 1.1) / (mu_water * kro_max) # krw_max is from __init__
        safe_pressure = max(pressure, 1e-6) # Avoid division by zero
        Nc = rate * mu_water / (porosity * safe_pressure) # Capillary number
        
        E_displacement = (1.0 - self.sor) * (1.0 - np.exp(-Nc)) # sor is from __init__
        E_vertical = 0.92 * (porosity ** 0.25)
        
        E_areal = 0.0
        if (1.0 + M_wo * 0.9) > 1e-9: # Avoid division by zero for highly unfavorable M_wo
             E_areal = 1.0 / (1.0 + M_wo * 0.9)
        else:
             E_areal = 0.1 # Low areal sweep for very high M_wo
             
        miscibility_factor = 0.9 * min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        combined_eff = miscibility_factor * E_displacement * E_vertical * E_areal
        combined_eff = min(0.68, max(0.0, combined_eff))
        return max(combined_eff, 0.051)

class HybridRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs): # Expects kwargs for transition and sub-models
        # Transition Engine parameters
        te_config = { # Start with hardcoded defaults for TE params
            'mode': 'sigmoid', 'alpha': 1.0, 'beta': 20.0, 'use_gpu': False,
            'x_points': [0.5, 0.8, 1.0, 1.2, 1.5], 'y_points': [0.05, 0.2, 0.5, 0.8, 0.95]
        }
        # Update with defaults from global config for "Hybrid" model (which might contain TE settings)
        hybrid_global_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Hybrid") or {}
        te_config.update(hybrid_global_defaults) # Overwrites hardcoded with global if keys match
        # Update with specific model_init_kwargs passed for this instance (highest precedence for TE)
        te_config.update(model_init_kwargs)
        self.transition_engine = TransitionEngine(
            mode=te_config['mode'], use_gpu=te_config['use_gpu'],
            alpha=te_config['alpha'], beta=te_config['beta'], # For sigmoid
            x_points=te_config.get('x_points'), y_points=te_config.get('y_points') # For cubic
        )
        
        # Sub-model parameters
        # Start with global defaults for sub-models
        miscible_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Miscible") or {}
        immiscible_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Immiscible") or {}
        # Allow model_init_kwargs to provide a complete dict for sub-model params (e.g. "miscible_params": {...})
        final_miscible_params = {**miscible_defaults, **model_init_kwargs.get('miscible_params', {})}
        final_immiscible_params = {**immiscible_defaults, **model_init_kwargs.get('immiscible_params', {})}
        
        self.miscible_model = MiscibleRecoveryModel(**final_miscible_params)
        self.immiscible_model = ImmiscibleRecoveryModel(**final_immiscible_params)

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        p_mmp_ratio = np.array(pressure / mmp if mmp > 0 else 10.0) # Assume highly miscible if mmp is zero/invalid
        transition_weight = self.transition_engine.calculate_efficiency(p_mmp_ratio)
        
        # Pass runtime kwargs to sub-models
        immiscible_result = self.immiscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        miscible_result = self.miscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        
        blended_recovery = immiscible_result * (1.0 - transition_weight) + miscible_result * transition_weight
        result = np.clip(blended_recovery, 0.0, 0.9) # Max recovery capped at 0.9
        return result.item() if isinstance(result, np.ndarray) else float(result)

class TransitionFunction(ABC):
    @abstractmethod
    def evaluate(self, p_mmp_ratio: np.ndarray) -> np.ndarray: pass

class SigmoidTransition(TransitionFunction):
    def __init__(self, alpha: float = 1.0, beta: float = 20.0):
        self.alpha = np.clip(alpha, 0.5, 1.5)
        self.beta = np.clip(beta, 5.0, 50.0)
    def evaluate(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.beta * (p_mmp_ratio - self.alpha)))

class CubicTransition(TransitionFunction):
    def __init__(self, x_points: Optional[List[float]] = None, y_points: Optional[List[float]] = None):
        self._spline: Optional[CubicSpline] = None
        # Fallback to hardcoded defaults if config or explicit params not provided
        default_x = [0.5, 0.8, 1.0, 1.2, 1.5]
        default_y = [0.05, 0.2, 0.5, 0.8, 0.95]
        
        final_x_list = x_points if x_points is not None else default_x
        final_y_list = y_points if y_points is not None else default_y
        
        self.fit(np.array(final_x_list, dtype=float), np.array(final_y_list, dtype=float))

    def fit(self, x_points_arr: np.ndarray, y_points_arr: np.ndarray):
        if len(x_points_arr) < 2 or len(y_points_arr) < 2:
            raise ValueError("CubicTransition requires at least 2 points for x and y.")
        if len(x_points_arr) != len(y_points_arr):
            raise ValueError("CubicTransition: x_points and y_points must have the same length.")
        
        sorted_indices = np.argsort(x_points_arr)
        x_sorted = x_points_arr[sorted_indices]
        y_sorted = y_points_arr[sorted_indices]
        # Use 'clamped' boundary condition if first/last derivatives are known/assumed (e.g., 0)
        # Otherwise, 'natural' is a common choice. Let's assume natural for now.
        try:
            self._spline = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        except ValueError as e:
            logging.error(f"Failed to create CubicSpline (x={x_sorted}, y={y_sorted}): {e}")
            raise # Re-raise after logging

    def evaluate(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        if self._spline is None:
            raise RuntimeError("CubicTransition spline not fitted. Call fit() or ensure constructor params are valid.")
        return np.clip(self._spline(p_mmp_ratio), 0.0, 1.0)

class TransitionEngine:
    def __init__(self, mode: str = 'sigmoid', use_gpu: bool = False, **params: Any):
        self.mode = mode.lower()
        self.params = params # Store all passed params (alpha, beta, x_points, y_points, custom_fn)
        self._gpu_enabled = False
        self._transition_function: Optional[TransitionFunction] = None
        self._setup_transition_function()
        if use_gpu:
            self.enable_gpu_acceleration()

    def _setup_transition_function(self):
        if self.mode == 'sigmoid':
            self._transition_function = SigmoidTransition(
                alpha=self.params.get('alpha', 1.0),
                beta=self.params.get('beta', 20.0)
            )
        elif self.mode == 'cubic':
            self._transition_function = CubicTransition(
                x_points=self.params.get('x_points'), # Will use its own defaults if None
                y_points=self.params.get('y_points')  # Will use its own defaults if None
            )
        elif self.mode == 'custom' and 'custom_fn' in self.params and isinstance(self.params['custom_fn'], TransitionFunction):
            self._transition_function = self.params['custom_fn']
        else:
            logging.warning(
                f"Invalid mode '{self.mode}' or missing/invalid params for TransitionEngine. "
                "Defaulting to sigmoid transition with alpha=1.0, beta=20.0."
            )
            self._transition_function = SigmoidTransition(alpha=1.0, beta=20.0)
            self.mode = 'sigmoid'

    def enable_gpu_acceleration(self):
        if not CUPY_AVAILABLE:
            logging.warning("cupy is not installed. GPU acceleration for TransitionEngine will be disabled.")
            self._gpu_enabled = False
            return
        self._gpu_enabled = True
        logging.info("GPU acceleration for TransitionEngine enabled (conceptual). Actual GPU use in evaluate depends on TransitionFunction impl.")

    def calculate_efficiency(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        if self._transition_function is None:
            # This should not happen if _setup_transition_function is called in __init__
            logging.error("TransitionEngine function not initialized. Defaulting to a sigmoid.")
            self._setup_transition_function() # Attempt to re-initialize
            if self._transition_function is None: # Still fails
                 raise RuntimeError("TransitionEngine function failed to initialize.")

        # GPU acceleration is conceptual here; the actual function evaluation might be on CPU
        # unless the specific TransitionFunction (e.g., SigmoidTransition) is rewritten for cupy.
        if self._gpu_enabled and CUPY_AVAILABLE:
            # Example if _transition_function.evaluate could handle cupy arrays directly:
            # p_mmp_ratio_gpu = cp.asarray(p_mmp_ratio)
            # result_gpu = self._transition_function.evaluate(p_mmp_ratio_gpu)
            # return cp.asnumpy(result_gpu)
            logging.debug("TransitionEngine: GPU mode active, but evaluation logic is function-dependent.")
        
        return self._transition_function.evaluate(p_mmp_ratio)

def recovery_factor(pressure: float, rate: float, porosity: float, mmp: float,
                  model: str = 'simple', **kwargs_for_model_calculate_and_init) -> float:
    """
    Estimates recovery factor using specified model.

    Args:
        pressure (float): Current pressure.
        rate (float): Current injection rate.
        porosity (float): Average porosity.
        mmp (float): Minimum Miscibility Pressure.
        model (str): Name of the recovery model to use (e.g., 'hybrid', 'koval').
        **kwargs_for_model_calculate_and_init:
            Additional keyword arguments. These are:
            1. Passed to the selected model's `calculate` method (e.g., v_dp_coefficient, mobility_ratio).
            2. Potentially used to override default __init__ parameters for the model if
               `OptimizationEngine.set_recovery_model` was not used or if specific overrides
               for this single call are intended (e.g., `miscible_params={...}`).
    """
    model_name_lower = model.lower()
    
    # Get model-specific __init__ defaults from global config
    # These are the base defaults for constructing the model instance.
    model_init_cfg_key = f"RecoveryModelKwargsDefaults.{model_name_lower.capitalize()}"
    base_init_params = config_manager.get_section(model_init_cfg_key) or {}

    # Allow kwargs_for_model_calculate_and_init to override __init__ params
    # This supports passing, e.g., `hybrid_params={"alpha": 0.9}` to this function.
    # The key format like `f"{model_name_lower}_params"` is a convention.
    specific_init_overrides = kwargs_for_model_calculate_and_init.get(f"{model_name_lower}_params", {})
    general_init_overrides = kwargs_for_model_calculate_and_init.get("model_init_kwargs", {})
    
    final_model_init_kwargs = {**base_init_params, **specific_init_overrides, **general_init_overrides}

    model_constructors = {
        'simple': SimpleRecoveryModel,
        'miscible': MiscibleRecoveryModel,
        'immiscible': ImmiscibleRecoveryModel,
        'hybrid': HybridRecoveryModel,
        'koval': KovalRecoveryModel
    }
    constructor = model_constructors.get(model_name_lower)
    if not constructor:
        raise ValueError(f"Unknown recovery model name: {model_name_lower}. Valid: {list(model_constructors.keys())}")
    
    # Instantiate the model with its specific init parameters
    # SimpleRecoveryModel and KovalRecoveryModel might not take specific __init__ kwargs beyond base.
    if model_name_lower in ['miscible', 'immiscible', 'hybrid']:
        # These models have more complex __init__ methods expecting specific kwargs
        selected_model_instance = constructor(**final_model_init_kwargs)
    else:
        # Simple and Koval can be instantiated without specific kwargs if their __init__ is minimal
        selected_model_instance = constructor() # Or constructor(**final_model_init_kwargs) if they also use it

    # Pass all runtime kwargs (v_dp_coefficient, mobility_ratio, etc.) to the `calculate` method.
    # The `calculate` methods of the models are responsible for picking up the kwargs they need.
    return selected_model_instance.calculate(pressure, rate, porosity, mmp, **kwargs_for_model_calculate_and_init)


class OptimizationEngine:
    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties,
                 eor_params_instance: Optional[EORParameters] = None,
                 ga_params_instance: Optional[GeneticAlgorithmParams] = None,
                 well_analysis: Optional[Any] = None):
        """
        Initializes the OptimizationEngine.
        Configuration is expected to be loaded into the global `config_manager`
        before this engine is instantiated.
        """
        if not config_manager.is_loaded:
            # This check is crucial if config_manager was initialized with require_config=False
            # or if load_config was missed in the main application setup.
            logging.critical("OptimizationEngine: ConfigManager reports no configuration loaded. "
                             "Default values will be heavily relied upon, or errors may occur.")
            # Depending on strictness, one might raise ConfigNotLoadedError here too.
            # For now, allow it to proceed and fail later if a critical config is missing.

        self.reservoir = reservoir
        self.pvt = pvt

        if eor_params_instance:
            self.eor_params = eor_params_instance
        else:
            self.eor_params = EORParameters.from_config_dict(
                config_manager.get_section("EORParametersDefaults") or {}
            )
        
        if ga_params_instance:
            self.ga_params_default_config = ga_params_instance # Used as base for GA runs
        else:
            self.ga_params_default_config = GeneticAlgorithmParams.from_config_dict(
                config_manager.get_section("GeneticAlgorithmParamsDefaults") or {}
            )

        self.well_analysis = well_analysis
        self._results: Optional[Dict[str, Any]] = None
        self._mmp_value: Optional[float] = None
        self._mmp_params_used: Optional[Any] = None # Stores the MMPParameters instance or PVT used

        # Set default recovery model and its __init__ kwargs from config
        self.recovery_model: str = config_manager.get("OptimizationEngineSettings.default_recovery_model", "hybrid")
        # These are kwargs for the __init__ of the selected recovery model
        self._recovery_model_init_kwargs: Dict[str, Any] = config_manager.get_section(
            f"RecoveryModelKwargsDefaults.{self.recovery_model.capitalize()}"
        ) or {}
        
        # MMP Calculation Setup
        self._mmp_calculator_fn: Optional[Callable[[Union[PVTProperties, Any], str], float]] = None
        self._MMPParametersDataclass: Optional[Type[Any]] = None
        try:
            from evaluation.mmp import calculate_mmp as calculate_mmp_external, MMPParameters
            self._mmp_calculator_fn = calculate_mmp_external
            self._MMPParametersDataclass = MMPParameters
        except ImportError:
            logging.critical(
                "evaluation.mmp modules (calculate_mmp, MMPParameters) failed to import. "
                "MMP calculation features will be limited, relying on fallbacks."
            )
        self.calculate_mmp()

    def calculate_mmp(self, method_override: Optional[str] = None) -> float:
        """
        Calculates or retrieves the Minimum Miscibility Pressure (MMP).
        Uses well_analysis for parameters if available, otherwise PVT data.
        Relies on `evaluation.mmp.calculate_mmp`.
        """
        default_mmp_fallback = config_manager.get("GeneralFallbacks.mmp_default_psi", 2500.0)
        
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            logging.warning("MMP calculator dependencies not available. Returning fallback MMP value.")
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
            return self._mmp_value

        # Determine method: override, then config, then auto
        actual_mmp_method = method_override if method_override else \
                            config_manager.get("OptimizationEngineSettings.mmp_calculation_method", "auto")

        mmp_input_constructor_params: Dict[str, Any] = {}
        source_description = "PVT data"

        if self.well_analysis and hasattr(self.well_analysis, 'get_average_mmp_params_for_engine'):
            try:
                # Assume get_average_mmp_params_for_engine returns a dict suitable for MMPParameters
                avg_well_params = self.well_analysis.get_average_mmp_params_for_engine()
                mmp_input_constructor_params = {
                    'temperature': avg_well_params.get('temperature', self.pvt.temperature),
                    'oil_gravity': avg_well_params.get('oil_gravity', config_manager.get("GeneralFallbacks.api_gravity_default", 35.0)),
                    'c7_plus_mw': avg_well_params.get('c7_plus_mw'),
                    'injection_gas_composition': avg_well_params.get(
                        'injection_gas_composition',
                        config_manager.get_section("GeneralFallbacks.default_injection_gas_composition") or {'CO2': 1.0}
                    ),
                    'pvt_data': self.pvt # Always pass PVT for context or API estimation if needed
                }
                self._mmp_params_used = self._MMPParametersDataclass(**mmp_input_constructor_params)
                source_description = "WellAnalysis average parameters"
            except Exception as e:
                logging.warning(f"Failed to get MMP parameters from WellAnalysis: {e}. Using PVT data directly.")
                self._mmp_params_used = self.pvt # Fallback to using PVTProperties object directly
        else:
            self._mmp_params_used = self.pvt

        try:
            calculated_mmp_value = float(self._mmp_calculator_fn(self._mmp_params_used, method=actual_mmp_method))
            self._mmp_value = calculated_mmp_value
            logging.info(f"MMP calculated: {self._mmp_value:.2f} psi using method '{actual_mmp_method}' from {source_description}.")
        except Exception as e:
            logging.error(f"MMP calculation failed using '{actual_mmp_method}' and {source_description}: {e}. Using fallback MMP.")
            self._mmp_value = self._mmp_value if self._mmp_value is not None else default_mmp_fallback
        
        return self._mmp_value

    def optimize_recovery(self) -> Dict[str, Any]: # Simplified Gradient Descent
        cfg_grad = config_manager.get_section("OptimizationEngineSettings.gradient_descent_optimizer") or {}
        max_iter = cfg_grad.get("max_iter", 100)
        tol = cfg_grad.get("tolerance", 1e-4)
        learning_rate = cfg_grad.get("learning_rate", 50.0)
        pressure_perturbation = cfg_grad.get("pressure_perturbation", 10.0)

        if self._mmp_value is None: self.calculate_mmp() # Ensure MMP is available
        mmp_val = self._mmp_value # Should not be None here due to calculate_mmp's fallback

        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))
        
        # Initial pressure: respect target pressure if above MMP, else slightly above MMP, capped by max pressure
        initial_pressure = self.eor_params.target_pressure_psi
        if initial_pressure <= mmp_val:
            initial_pressure = mmp_val * 1.05
        current_pressure = np.clip(initial_pressure, mmp_val * 1.01, self.eor_params.max_pressure_psi)
        
        injection_rate_val = self.eor_params.injection_rate # Fixed for this simple optimizer

        # Runtime kwargs for recovery_factor, including those being optimized if this was multi-param
        call_kwargs_for_recovery = {
            **self._recovery_model_init_kwargs, # Model's own init params
            'v_dp_coefficient': self.eor_params.v_dp_coefficient, # Runtime param from EORParameters
            'mobility_ratio': self.eor_params.mobility_ratio      # Runtime param from EORParameters
        }

        previous_recovery = 0.0
        current_recovery = 0.0
        converged = False
        iterations_done = 0

        for i in range(max_iter):
            iterations_done = i + 1
            current_recovery = recovery_factor(
                current_pressure, injection_rate_val, avg_porosity, mmp_val,
                model=self.recovery_model, **call_kwargs_for_recovery
            )
            if i > 5 and abs(current_recovery - previous_recovery) < tol:
                converged = True
                break
            
            recovery_plus_perturb = recovery_factor(
                current_pressure + pressure_perturbation, injection_rate_val, avg_porosity, mmp_val,
                model=self.recovery_model, **call_kwargs_for_recovery
            )
            gradient = (recovery_plus_perturb - current_recovery) / pressure_perturbation
            
            if abs(gradient) < 1e-7: # Negligible gradient
                converged = True; break

            new_pressure_val = current_pressure + learning_rate * gradient
            current_pressure = np.clip(new_pressure_val, mmp_val * 1.01, self.eor_params.max_pressure_psi)
            previous_recovery = current_recovery
            
        self._results = {
            'optimized_params': {
                'injection_rate': injection_rate_val, # Was fixed
                'target_pressure_psi': current_pressure,
                'v_dp_coefficient': self.eor_params.v_dp_coefficient, # Report fixed values
                'mobility_ratio': self.eor_params.mobility_ratio      # Report fixed values
            },
            'mmp_psi': mmp_val,
            'iterations': iterations_done,
            'final_recovery': current_recovery,
            'converged': converged,
            'avg_porosity': avg_porosity,
            'method': 'gradient_descent_pressure'
        }
        return self._results

    def optimize_bayesian(self, n_iter_override: Optional[int] = None, init_points_override: Optional[int] = None,
                        method_override: Optional[str] = None,
                        initial_solutions_from_ga: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        cfg_bo_standalone = config_manager.get_section("OptimizationEngineSettings.bayesian_optimizer") or {}
        n_iter = n_iter_override if n_iter_override is not None else cfg_bo_standalone.get("n_iter", 40)
        init_pts_random = init_points_override if init_points_override is not None else cfg_bo_standalone.get("init_points_random", 8)
        bo_method = method_override if method_override is not None else cfg_bo_standalone.get("default_method", "gp")
        rate_max_factor = cfg_bo_standalone.get("rate_bound_factor_max", 1.5)

        if self._mmp_value is None: self.calculate_mmp()
        mmp_val = self._mmp_value

        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))
        
        # Define search space dimensions (parameter names must match EORParameters or GA output)
        min_pressure_bound = np.clip(
            mmp_val * 1.01, # Ensure slightly above MMP
            self.eor_params.min_injection_rate_bpd if hasattr(self.eor_params, 'min_target_pressure_psi') else mmp_val * 1.01, # Hypothetical min target P
            self.eor_params.max_pressure_psi - 1.0 # Ensure it's below max
        )

        space_dims = [
            Real(min_pressure_bound, self.eor_params.max_pressure_psi, name='pressure'), # Renamed for clarity in BO
            Real(self.eor_params.min_injection_rate_bpd, self.eor_params.injection_rate * rate_max_factor, name='rate'),
            Real(0.3, 0.8, name='v_dp_coefficient'),
            Real(1.2, 20.0, name='mobility_ratio')
        ]
        if self.eor_params.injection_scheme == 'wag':
            space_dims.extend([
                Real(self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days, name='cycle_length_days'),
                Real(self.eor_params.min_water_fraction, self.eor_params.max_water_fraction, name='water_fraction')
            ])
        param_names_in_order = [dim.name for dim in space_dims]

        @use_named_args(space_dims) # For skopt
        def objective_function_for_bo(**params_dict_from_bo):
            # params_dict_from_bo contains keys matching `name` in space_dims
            eff_rate = params_dict_from_bo['rate']
            if self.eor_params.injection_scheme == 'wag' and 'water_fraction' in params_dict_from_bo:
                eff_rate *= (1.0 - params_dict_from_bo['water_fraction'])
            
            # Pass all params from BO iteration and model's init defaults to recovery_factor
            call_kwargs = {**self._recovery_model_init_kwargs, **params_dict_from_bo}
            
            recovery = recovery_factor(
                params_dict_from_bo['pressure'], eff_rate, avg_porosity, mmp_val,
                model=self.recovery_model, **call_kwargs
            )
            return -recovery if bo_method == 'gp' else recovery # skopt minimizes, bayes_opt maximizes

        best_params_from_bo: Dict[str, float] = {}
        final_recovery_val: float = 0.0
        num_ga_points_used_in_bo: int = 0

        if bo_method == 'gp':
            x0_param_dicts_for_skopt: List[Dict[str, float]] = []
            y0_objective_values_for_skopt: List[float] = []

            if initial_solutions_from_ga:
                for ga_sol in initial_solutions_from_ga:
                    current_ga_params_dict = {}
                    is_valid_solution = True
                    for dim in space_dims: # Ensure all required dimensions are present and clip
                        if dim.name in ga_sol['params']:
                            current_ga_params_dict[dim.name] = np.clip(ga_sol['params'][dim.name], dim.low, dim.high)
                        else:
                            logging.warning(f"GA solution missing '{dim.name}' for BO. Skipping this GA point.")
                            is_valid_solution = False; break
                    if is_valid_solution:
                        x0_param_dicts_for_skopt.append(current_ga_params_dict)
                        y0_objective_values_for_skopt.append(-ga_sol['fitness']) # skopt minimizes
            num_ga_points_used_in_bo = len(x0_param_dicts_for_skopt)

            # n_calls is total evaluations.
            total_skopt_calls = num_ga_points_used_in_bo + init_pts_random + n_iter
            if initial_solutions_from_ga and total_skopt_calls <= num_ga_points_used_in_bo: # Ensure BO runs
                 total_skopt_calls = num_ga_points_used_in_bo + init_pts_random + 1


            logging.info(f"Running gp_minimize: {num_ga_points_used_in_bo} points from GA, "
                         f"{init_pts_random} random initial BO points, {n_iter} BO iterations. "
                         f"Total skopt calls: {total_skopt_calls}.")

            skopt_result = gp_minimize(
                objective_function_for_bo, space_dims,
                x0=x0_param_dicts_for_skopt if x0_param_dicts_for_skopt else None, # Pass list of dicts
                y0=y0_objective_values_for_skopt if y0_objective_values_for_skopt else None,
                n_calls=total_skopt_calls,
                n_initial_points=init_pts_random, # skopt uses these if x0 is too small or None
                random_state=config_manager.get("GeneralFallbacks.random_seed", 42),
                verbose=True
            )
            best_params_from_bo = {name: val for name, val in zip(param_names_in_order, skopt_result.x)}
            final_recovery_val = -skopt_result.fun
        
        elif bo_method == 'bayes':
            pbounds_for_bayesopt = {dim.name: (dim.low, dim.high) for dim in space_dims}
            bayes_optimizer = BayesianOptimization(
                f=objective_function_for_bo, # Already adapted for maximization
                pbounds=pbounds_for_bayesopt,
                random_state=config_manager.get("GeneralFallbacks.random_seed", 42),
                verbose=2
            )
            num_registered_ga_points = 0
            if initial_solutions_from_ga:
                logging.info(f"Registering {len(initial_solutions_from_ga)} GA solutions with bayes_opt.")
                for ga_sol_idx, ga_sol in enumerate(initial_solutions_from_ga):
                    params_to_register = {}
                    valid_for_registration = True
                    for param_name in pbounds_for_bayesopt.keys():
                        if param_name in ga_sol['params']:
                            # Clip to bounds, bayes_opt can be strict
                            low_b, high_b = pbounds_for_bayesopt[param_name]
                            params_to_register[param_name] = np.clip(ga_sol['params'][param_name], low_b, high_b)
                        else:
                            logging.warning(f"GA solution {ga_sol_idx} missing '{param_name}' for bayes_opt registration. Skipping.")
                            valid_for_registration = False; break
                    if valid_for_registration:
                        try:
                            bayes_optimizer.register(params=params_to_register, target=ga_sol['fitness'])
                            num_registered_ga_points += 1
                        except Exception as e_reg: # More general catch for bayes_opt issues
                            logging.error(f"Error registering GA solution {params_to_register} with bayes_opt: {e_reg}")
            num_ga_points_used_in_bo = num_registered_ga_points

            bayes_optimizer.maximize(init_points=init_pts_random, n_iter=n_iter)
            best_params_from_bo = bayes_optimizer.max['params']
            final_recovery_val = bayes_optimizer.max['target']
        else:
            raise ValueError(f"Unsupported Bayesian optimization method: {bo_method}. Choose 'gp' or 'bayes'.")

        # Standardize parameter names in the output for consistency with EORParameters fields
        optimized_params_standardized = {
            'injection_rate': best_params_from_bo.get('rate'),
            'target_pressure_psi': best_params_from_bo.get('pressure'),
            'cycle_length_days': best_params_from_bo.get('cycle_length_days'), # Will be None if not WAG
            'water_fraction': best_params_from_bo.get('water_fraction'),     # Will be None if not WAG
            'v_dp_coefficient': best_params_from_bo.get('v_dp_coefficient'),
            'mobility_ratio': best_params_from_bo.get('mobility_ratio')
        }
        
        self._results = {
            'optimized_params': optimized_params_standardized,
            'mmp_psi': mmp_val,
            'method': f'bayesian_{bo_method}',
            'iterations_bo_actual': n_iter,
            'initial_points_bo_random': init_pts_random,
            'initial_points_from_ga_used': num_ga_points_used_in_bo,
            'final_recovery': final_recovery_val,
            'avg_porosity': avg_porosity,
            'converged': True # BO typically runs for fixed iterations or converges on its own terms
        }
        return self._results

    def hybrid_optimize(self, ga_params_override: Optional[GeneticAlgorithmParams] = None) -> Dict[str, Any]:
        cfg_hyb_base = config_manager.get_section("OptimizationEngineSettings.hybrid_optimizer")
        if not cfg_hyb_base: # Fallback if hybrid_optimizer section is missing in config
            logging.warning("Hybrid optimizer config not found. Using hardcoded defaults for hybrid strategy.")
            cfg_hyb_base = {
                "ga_config_source": "default_ga_params",
                "bo_iterations_in_hybrid": 20, # Default BO iterations
                "bo_random_initial_points_in_hybrid": 5, # Default BO random starts
                "num_ga_elites_to_bo": 3, # Default number of GA elites
                "bo_method_in_hybrid": "gp" # Default BO method
            }

        actual_ga_params_for_hybrid: GeneticAlgorithmParams
        if ga_params_override:
            actual_ga_params_for_hybrid = ga_params_override
            logging.info("Hybrid GA phase: Using explicit ga_params_override.")
        else:
            ga_config_source = cfg_hyb_base.get("ga_config_source", "default_ga_params")
            if ga_config_source == "hybrid_specific":
                ga_params_hybrid_dict = cfg_hyb_base.get("ga_params_hybrid", {})
                if ga_params_hybrid_dict:
                    actual_ga_params_for_hybrid = GeneticAlgorithmParams.from_config_dict(ga_params_hybrid_dict)
                    logging.info("Hybrid GA phase: Using 'hybrid_specific' GA parameters from config.")
                else:
                    logging.warning("Hybrid GA phase: 'hybrid_specific' chosen but no config found. Using default engine GA params.")
                    actual_ga_params_for_hybrid = self.ga_params_default_config
            else: # "default_ga_params" or any other value
                actual_ga_params_for_hybrid = self.ga_params_default_config
                logging.info("Hybrid GA phase: Using default engine GA parameters.")

        bo_iter_hyb = cfg_hyb_base.get("bo_iterations_in_hybrid", 20)
        bo_init_rand_hyb = cfg_hyb_base.get("bo_random_initial_points_in_hybrid", 5)
        num_ga_elites_hyb = cfg_hyb_base.get("num_ga_elites_to_bo", 3)
        bo_method_hyb = cfg_hyb_base.get("bo_method_in_hybrid", "gp")
        
        logging.info(f"Hybrid Run Starting: "
                     f"GA phase (Gens: {actual_ga_params_for_hybrid.generations}, Pop: {actual_ga_params_for_hybrid.population_size}, Elites: {actual_ga_params_for_hybrid.elite_count}) "
                     f"-> BO phase (Method: {bo_method_hyb}, Iters: {bo_iter_hyb}, RandomStarts: {bo_init_rand_hyb}, ElitesFromGA: {num_ga_elites_hyb})")
        
        ga_results = self.optimize_genetic_algorithm(ga_params_to_use=actual_ga_params_for_hybrid)

        initial_bo_solutions_from_ga_list: Optional[List[Dict[str, Any]]] = None
        if num_ga_elites_hyb > 0 and 'top_ga_solutions_from_final_pop' in ga_results:
            top_ga_sols_list = ga_results['top_ga_solutions_from_final_pop']
            if top_ga_sols_list:
                initial_bo_solutions_from_ga_list = top_ga_sols_list[:min(num_ga_elites_hyb, len(top_ga_sols_list))]
                logging.info(f"Passing {len(initial_bo_solutions_from_ga_list)} elite solutions from GA to BO phase.")
            else:
                logging.warning("GA reported no top solutions; BO will start purely with its random initial points.")
        else:
            logging.info("BO phase will start purely with its random initial points (no elites from GA requested or available).")

        bo_results = self.optimize_bayesian(
            n_iter_override=bo_iter_hyb,
            init_points_override=bo_init_rand_hyb,
            method_override=bo_method_hyb,
            initial_solutions_from_ga=initial_bo_solutions_from_ga_list
        )
        
        self._results = {
            **bo_results,
            'ga_full_results': ga_results,
            'method': f'hybrid_ga(g{actual_ga_params_for_hybrid.generations})_bo(i{bo_iter_hyb},e{len(initial_bo_solutions_from_ga_list) if initial_bo_solutions_from_ga_list else 0})_m({bo_method_hyb})'
        }
        logging.info(f"Hybrid optimization completed. Final recovery: {self._results.get('final_recovery', 0.0):.4f}")
        return self._results

    def _get_ga_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Helper to get GA parameter bounds based on EORParameters and config."""
        if self._mmp_value is None: self.calculate_mmp() # Ensure MMP is fresh
        mmp_val = self._mmp_value
        rate_max_factor_ga = config_manager.get("OptimizationEngineSettings.genetic_algorithm.rate_bound_factor_max", # New config key
                                             config_manager.get("OptimizationEngineSettings.bayesian_optimizer.rate_bound_factor_max", 1.5))

        min_pressure_ga = np.clip(mmp_val * 1.01, self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi > mmp_val else mmp_val * 1.01, self.eor_params.max_pressure_psi -1.0)

        bounds = {
            'pressure': (min_pressure_ga, self.eor_params.max_pressure_psi),
            'rate': (self.eor_params.min_injection_rate_bpd, self.eor_params.injection_rate * rate_max_factor_ga),
            'v_dp_coefficient': (0.3, 0.8),
            'mobility_ratio': (1.2, 20.0)
        }
        if self.eor_params.injection_scheme == 'wag':
            bounds['cycle_length_days'] = (self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days)
            bounds['water_fraction'] = (self.eor_params.min_water_fraction, self.eor_params.max_water_fraction)
        return bounds

    def _initialize_population_ga(self, population_size: int, current_ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        population = []
        param_bounds = self._get_ga_parameter_bounds()
        param_names_for_ga = list(param_bounds.keys()) # These are the genes

        for _ in range(population_size):
            individual: Dict[str, float] = {}
            for param_name in param_names_for_ga:
                low, high = param_bounds[param_name]
                individual[param_name] = random.uniform(low, high)
            population.append(individual)
        return population

    def _get_complete_params_from_ga_individual(self, individual_dict: Dict[str, float],
                                              param_bounds_for_clipping: Dict[str, Tuple[float,float]] ) -> Dict[str, float]:
        """Ensures all expected parameters are present and clipped for an individual from GA."""
        complete_params = {}
        for param_name, (low, high) in param_bounds_for_clipping.items():
            val = individual_dict.get(param_name, random.uniform(low, high)) # Fallback if missing, though shouldn't happen
            complete_params[param_name] = np.clip(val, low, high)
        return complete_params

    def _evaluate_individual_ga(self, individual_dict: Dict[str, float], avg_porosity: float, mmp: float,
                                model_name: str, recovery_model_init_kwargs: Dict[str, Any],
                                current_ga_config: GeneticAlgorithmParams) -> float: # current_ga_config might not be needed here
        
        param_bounds = self._get_ga_parameter_bounds() # Get fresh bounds for clipping
        # Ensure the individual has all necessary parameters and they are within bounds
        eval_params = self._get_complete_params_from_ga_individual(individual_dict, param_bounds)

        effective_rate = eval_params['rate']
        if self.eor_params.injection_scheme == 'wag' and 'water_fraction' in eval_params:
            effective_rate *= (1.0 - eval_params['water_fraction'])
        
        # Combine model's __init__ kwargs with the current individual's (runtime) parameters
        all_call_kwargs_for_recovery = {**recovery_model_init_kwargs, **eval_params}
        
        return recovery_factor(
            eval_params['pressure'], effective_rate, avg_porosity, mmp,
            model=model_name, **all_call_kwargs_for_recovery
        )

    def _tournament_selection_ga(self, population: List[Dict[str, float]], fitnesses: List[float],
                               ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        selected_population = []
        population_len = len(population)
        
        # Elitism: copy elite_count best individuals directly
        if ga_config.elite_count > 0 and population_len > 0 :
            elite_indices = np.argsort(fitnesses)[-ga_config.elite_count:]
            for idx in elite_indices:
                selected_population.append(population[idx].copy()) # Use .copy()

        # Tournament selection for the rest
        num_to_select_via_tournament = population_len - len(selected_population)
        for _ in range(num_to_select_via_tournament):
            if population_len == 0: break # Should not happen if pop is managed
            tournament_competitor_indices = random.sample(range(population_len), ga_config.tournament_size)
            winner_idx = max(tournament_competitor_indices, key=lambda idx: fitnesses[idx])
            selected_population.append(population[winner_idx].copy()) # Use .copy()
            
        return selected_population

    def _crossover_ga(self, parent_population: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        offspring_population = []
        num_parents = len(parent_population)
        if num_parents == 0: return []

        # Ensure we have pairs for crossover; last one might pass through if odd number
        for i in range(0, num_parents -1 , 2): # Iterate to second to last if num_parents is odd
            parent1 = parent_population[i]
            parent2 = parent_population[i+1]
            child1, child2 = parent1.copy(), parent2.copy() # Start with copies

            if random.random() < ga_config.crossover_rate:
                # Blend crossover (Arithmetic crossover)
                alpha = ga_config.blend_alpha_crossover # Could be randomized: random.random()
                all_genes = set(parent1.keys()) | set(parent2.keys()) # All possible gene names
                for gene_key in all_genes:
                    val_p1 = parent1.get(gene_key)
                    val_p2 = parent2.get(gene_key)

                    if val_p1 is not None and val_p2 is not None:
                        child1[gene_key] = alpha * val_p1 + (1.0 - alpha) * val_p2
                        child2[gene_key] = (1.0 - alpha) * val_p1 + alpha * val_p2
                    # If one parent is missing a gene (shouldn't happen with robust init/mutation),
                    # the child inherits from the parent that has it. copy() already handled this.
            offspring_population.extend([child1, child2])
        
        # If there's an odd number of parents, the last one passes through to maintain population size
        if num_parents % 2 == 1:
            offspring_population.append(parent_population[-1].copy())
            
        return offspring_population[:num_parents] # Ensure population size is maintained

    def _mutate_ga(self, population_to_mutate: List[Dict[str, float]], ga_config: GeneticAlgorithmParams) -> List[Dict[str, float]]:
        mutated_population = []
        param_bounds = self._get_ga_parameter_bounds() # Get current bounds

        for individual_dict in population_to_mutate:
            mutated_individual = individual_dict.copy()
            if random.random() < ga_config.mutation_rate:
                # Mutate one randomly chosen gene
                gene_to_mutate = random.choice(list(mutated_individual.keys()))
                
                if gene_to_mutate in param_bounds:
                    low_bound, high_bound = param_bounds[gene_to_mutate]
                    current_value = mutated_individual.get(gene_to_mutate, (low_bound + high_bound) / 2.0)
                    
                    # Gaussian mutation: strength relative to the parameter's range
                    mutation_range = (high_bound - low_bound)
                    # Avoid zero range if low_bound == high_bound (shouldn't happen for Real spaces)
                    mutation_std_dev = mutation_range * ga_config.mutation_strength_factor if mutation_range > 1e-9 else 0.1 * abs(current_value) + 1e-6
                    
                    mutated_value = current_value + random.gauss(0, mutation_std_dev)
                    mutated_individual[gene_to_mutate] = np.clip(mutated_value, low_bound, high_bound)
            mutated_population.append(mutated_individual)
        return mutated_population

    def optimize_genetic_algorithm(self, ga_params_to_use: Optional[GeneticAlgorithmParams] = None) -> Dict[str, Any]:
        current_ga_config = ga_params_to_use if ga_params_to_use else self.ga_params_default_config
        
        if self._mmp_value is None: self.calculate_mmp()
        mmp_val = self._mmp_value
        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))

        current_population = self._initialize_population_ga(current_ga_config.population_size, current_ga_config)
        
        best_solution_overall_dict = current_population[0].copy() if current_population else {}
        best_fitness_overall = -np.inf
        
        param_bounds_for_ga = self._get_ga_parameter_bounds() # For completing individuals

        for generation in range(current_ga_config.generations):
            # Parallel fitness evaluation
            with ProcessPoolExecutor() as executor:
                # Pass necessary arguments for evaluation
                fitness_values = list(executor.map(
                    partial(self._evaluate_individual_ga, avg_poro=avg_porosity, mmp=mmp_val,
                            model_name=self.recovery_model,
                            recovery_model_init_kwargs=self._recovery_model_init_kwargs,
                            current_ga_config=current_ga_config), # Pass the config
                    current_population
                ))
            
            current_gen_best_idx = np.argmax(fitness_values)
            if fitness_values[current_gen_best_idx] > best_fitness_overall:
                best_fitness_overall = fitness_values[current_gen_best_idx]
                best_solution_overall_dict = current_population[current_gen_best_idx].copy()
            
            selected_parents = self._tournament_selection_ga(current_population, fitness_values, current_ga_config)
            offspring = self._crossover_ga(selected_parents, current_ga_config)
            current_population = self._mutate_ga(offspring, current_ga_config)
            
            if (generation + 1) % 10 == 0 or generation == current_ga_config.generations -1 :
                logging.info(f"GA Gen {generation+1}/{current_ga_config.generations} - Current Gen Best Fitness: {fitness_values[current_gen_best_idx]:.4f} - Overall Best Fitness: {best_fitness_overall:.4f}")
        
        # Get top solutions from the final population
        with ProcessPoolExecutor() as executor:
            final_population_fitnesses = list(executor.map(
                 partial(self._evaluate_individual_ga, avg_poro=avg_porosity, mmp=mmp_val,
                         model_name=self.recovery_model,
                         recovery_model_init_kwargs=self._recovery_model_init_kwargs,
                         current_ga_config=current_ga_config),
                current_population
            ))

        sorted_final_population_details = sorted(
            zip(current_population, final_population_fitnesses),
            key=lambda x: x[1], # Sort by fitness
            reverse=True      # Higher fitness is better
        )
        
        num_top_solutions_to_return = min(
            len(sorted_final_population_details),
            current_ga_config.elite_count if current_ga_config.elite_count > 0 else 1
        )
        top_ga_solutions_for_bo = [
            {'params': self._get_complete_params_from_ga_individual(ind_params, param_bounds_for_ga), 'fitness': ind_fitness}
            for ind_params, ind_fitness in sorted_final_population_details[:num_top_solutions_to_return]
        ]
        
        # Prepare standardized output for the best solution found by GA
        final_optimized_params_from_ga = self._get_complete_params_from_ga_individual(best_solution_overall_dict, param_bounds_for_ga)
        effective_injection_rate_ga = final_optimized_params_from_ga['rate']
        if self.eor_params.injection_scheme == 'wag' and 'water_fraction' in final_optimized_params_from_ga:
            effective_injection_rate_ga *= (1.0 - final_optimized_params_from_ga['water_fraction'])

        self._results = {
            'optimized_params': { # Standardized keys matching EORParameters fields
                'injection_rate': effective_injection_rate_ga, # This is effective rate if WAG
                'target_pressure_psi': final_optimized_params_from_ga.get('pressure'),
                'cycle_length_days': final_optimized_params_from_ga.get('cycle_length_days'),
                'water_fraction': final_optimized_params_from_ga.get('water_fraction'),
                'v_dp_coefficient': final_optimized_params_from_ga.get('v_dp_coefficient'),
                'mobility_ratio': final_optimized_params_from_ga.get('mobility_ratio')
            },
            'mmp_psi': mmp_val,
            'method': 'genetic_algorithm',
            'generations': current_ga_config.generations,
            'population_size': current_ga_config.population_size,
            'final_recovery': best_fitness_overall,
            'avg_porosity': avg_porosity,
            'converged': True, # Placeholder, GA runs for fixed generations
            'best_solution_dict_raw_ga': best_solution_overall_dict, # Raw dict from GA
            'top_ga_solutions_from_final_pop': top_ga_solutions_for_bo # For hybrid optimizer
        }
        logging.info(f"Genetic Algorithm optimization completed. Best recovery: {best_fitness_overall:.4f}")
        return self._results

    def optimize_wag(self) -> Dict[str, Any]:
        cfg_wag_opt = config_manager.get_section("OptimizationEngineSettings.wag_optimizer") or {}
        refinement_cycles = cfg_wag_opt.get("refinement_cycles", 5)
        grid_points_per_dim = cfg_wag_opt.get("grid_search_points_per_dim", 5)
        range_reduction_factor = cfg_wag_opt.get("range_reduction_factor", 0.5)
        
        if self._mmp_value is None: self.calculate_mmp()
        mmp_val = self._mmp_value
        avg_porosity = np.mean(self.reservoir.grid.get('PORO', [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)]))

        # Initial ranges for WAG parameters
        current_min_cycle_len = self.eor_params.min_cycle_length_days
        current_max_cycle_len = self.eor_params.max_cycle_length_days
        current_min_water_frac = self.eor_params.min_water_fraction
        current_max_water_frac = self.eor_params.max_water_fraction

        # Initialize best WAG parameters dict using EORParameters field names for optimized values
        best_wag_params_dict = {
            'cycle_length_days': (current_min_cycle_len + current_max_cycle_len) / 2.0,
            'water_fraction': (current_min_water_frac + current_max_water_frac) / 2.0,
            'pressure': np.clip(mmp_val * 1.05, self.eor_params.target_pressure_psi if self.eor_params.target_pressure_psi > mmp_val else mmp_val*1.05, self.eor_params.max_pressure_psi -1.0),
            'v_dp_coefficient': self.eor_params.v_dp_coefficient, # Keep these fixed from EOR params for WAG opt
            'mobility_ratio': self.eor_params.mobility_ratio,     # Keep these fixed
            'recovery': -np.inf
        }

        for cycle_num in range(refinement_cycles):
            logging.info(f"WAG Optimization Cycle {cycle_num + 1}/{refinement_cycles}: "
                         f"CL Range [{current_min_cycle_len:.1f}-{current_max_cycle_len:.1f}], "
                         f"WF Range [{current_min_water_frac:.2f}-{current_max_water_frac:.2f}]")
            
            found_better_in_cycle = False
            current_best_recovery_in_cycle = best_wag_params_dict['recovery']

            water_frac_range_iter = np.linspace(current_min_water_frac, current_max_water_frac, num=grid_points_per_dim)
            cycle_len_range_iter = np.linspace(current_min_cycle_len, current_max_cycle_len, num=grid_points_per_dim)

            for wf_iter_val in water_frac_range_iter:
                for cl_iter_val in cycle_len_range_iter:
                    effective_inj_rate_iter = self.eor_params.injection_rate * (1.0 - wf_iter_val)
                    
                    # Construct kwargs for recovery_factor, using current best_wag_params_dict as base
                    # and overriding with iteration-specific WAG params.
                    call_kwargs_for_wag_iter = {
                        **self._recovery_model_init_kwargs, # Model's __init__ params
                        **best_wag_params_dict,            # Current best WAG params (incl. pressure, v_dp, M)
                        'cycle_length_days': cl_iter_val,  # Iteration specific
                        'water_fraction': wf_iter_val      # Iteration specific
                    }
                    
                    recovery_iter = recovery_factor(
                        best_wag_params_dict['pressure'], # Use current best pressure
                        effective_inj_rate_iter, avg_porosity, mmp_val,
                        model=self.recovery_model, **call_kwargs_for_wag_iter
                    )
                    if recovery_iter > current_best_recovery_in_cycle:
                        current_best_recovery_in_cycle = recovery_iter
                        best_wag_params_dict.update({
                            'cycle_length_days': cl_iter_val,
                            'water_fraction': wf_iter_val,
                            'recovery': recovery_iter
                        })
                        found_better_in_cycle = True
            
            if not found_better_in_cycle and cycle_num > 0: # Allow first cycle to always refine range
                logging.info("WAG optimization converged early as no improvement found in cycle.")
                break
            
            # Refine search ranges around the new best found
            range_wf = (current_max_water_frac - current_min_water_frac) * range_reduction_factor / 2.0
            current_min_water_frac = max(self.eor_params.min_water_fraction, best_wag_params_dict['water_fraction'] - range_wf)
            current_max_water_frac = min(self.eor_params.max_water_fraction, best_wag_params_dict['water_fraction'] + range_wf)
            
            range_cl = (current_max_cycle_len - current_min_cycle_len) * range_reduction_factor / 2.0
            current_min_cycle_len = max(self.eor_params.min_cycle_length_days, best_wag_params_dict['cycle_length_days'] - range_cl)
            current_max_cycle_len = min(self.eor_params.max_cycle_length_days, best_wag_params_dict['cycle_length_days'] + range_cl)

        # Optimize pressure for the best WAG cycle length and water fraction found
        optimized_pressure_for_wag = self._optimize_pressure_for_wag(
            best_wag_params_dict['water_fraction'], best_wag_params_dict['cycle_length_days'],
            avg_porosity, mmp_val, best_wag_params_dict # Pass full context
        )
        best_wag_params_dict['pressure'] = optimized_pressure_for_wag
        
        # Final recovery with optimized pressure
        final_effective_rate_wag = self.eor_params.injection_rate * (1.0 - best_wag_params_dict['water_fraction'])
        final_call_kwargs_wag = {**self._recovery_model_init_kwargs, **best_wag_params_dict}
        final_recovery_wag = recovery_factor(
            best_wag_params_dict['pressure'], final_effective_rate_wag, avg_porosity, mmp_val,
            model=self.recovery_model, **final_call_kwargs_wag
        )
        best_wag_params_dict['recovery'] = final_recovery_wag

        # Prepare results with standardized keys for 'optimized_params'
        # Matching EORParameters field names where appropriate for consistency
        wag_optimized_params_standardized = {
            'optimal_cycle_length_days': best_wag_params_dict.get('cycle_length_days'),
            'optimal_water_fraction': best_wag_params_dict.get('water_fraction'),
            'optimal_target_pressure_psi': best_wag_params_dict.get('pressure'),
            'injection_rate': self.eor_params.injection_rate, # Report the base rate used
            'v_dp_coefficient': best_wag_params_dict.get('v_dp_coefficient'), # Was fixed
            'mobility_ratio': best_wag_params_dict.get('mobility_ratio')     # Was fixed
        }

        self._results = {
            'wag_optimized_params': wag_optimized_params_standardized,
            'mmp_psi': mmp_val,
            'estimated_recovery': best_wag_params_dict['recovery'],
            'avg_porosity': avg_porosity,
            'method': 'iterative_grid_search_wag'
        }
        return self._results

    def _optimize_pressure_for_wag(self, water_fraction_val: float, cycle_length_val: float,
                                 avg_porosity_val: float, mmp_val: float,
                                 current_best_wag_params_context: Dict[str, Any]) -> float:
        cfg_wag_p_opt = config_manager.get_section("OptimizationEngineSettings.wag_optimizer") or {}
        max_iterations = cfg_wag_p_opt.get("max_iter_per_param_pressure_opt", 20)
        learning_rate_p = cfg_wag_p_opt.get("pressure_opt_learning_rate", 20.0)
        tolerance_p = cfg_wag_p_opt.get("pressure_opt_tolerance", 1e-4)
        perturbation_p = cfg_wag_p_opt.get("pressure_opt_perturbation", 10.0)
        # Max pressure constraint relative to MMP for this sub-optimization
        pressure_constraint_factor = cfg_wag_p_opt.get("pressure_constraint_factor_vs_mmp_max", 1.75)

        effective_rate_for_pressure_opt = self.eor_params.injection_rate * (1.0 - water_fraction_val)
        
        # Initial pressure guess
        current_pressure_p_opt = np.clip(
            current_best_wag_params_context.get('pressure', mmp_val * 1.05), # Start from context if available
            mmp_val * 1.01,
            min(self.eor_params.max_pressure_psi, mmp_val * pressure_constraint_factor) # Upper bound
        )
        
        best_pressure_found = current_pressure_p_opt
        best_recovery_for_pressure = -np.inf
        previous_recovery_for_pressure = -np.inf

        for _ in range(max_iterations):
            # Construct call_kwargs, ensuring the current WAG params and pressure are used
            call_kwargs_pressure_opt = {
                **self._recovery_model_init_kwargs,
                **current_best_wag_params_context, # Base context (v_dp, M)
                'water_fraction': water_fraction_val, # Specific WAG param for this pressure opt
                'cycle_length_days': cycle_length_val, # Specific WAG param
                # Pressure is passed directly to recovery_factor, not via kwargs here
            }
            
            current_recovery_p = recovery_factor(
                current_pressure_p_opt, effective_rate_for_pressure_opt, avg_porosity_val, mmp_val,
                model=self.recovery_model, **call_kwargs_pressure_opt
            )
            if current_recovery_p > best_recovery_for_pressure:
                best_recovery_for_pressure = current_recovery_p
                best_pressure_found = current_pressure_p_opt
            
            if abs(current_recovery_p - previous_recovery_for_pressure) < tolerance_p:
                break
            previous_recovery_for_pressure = current_recovery_p
            
            recovery_plus_perturb_p = recovery_factor(
                current_pressure_p_opt + perturbation_p, effective_rate_for_pressure_opt, avg_porosity_val, mmp_val,
                model=self.recovery_model, **call_kwargs_pressure_opt
            )
            gradient_p_opt = (recovery_plus_perturb_p - current_recovery_p) / perturbation_p
            
            if abs(gradient_p_opt) < 1e-7: # Negligible gradient
                break
                
            new_pressure_val_p = current_pressure_p_opt + learning_rate_p * gradient_p_opt
            current_pressure_p_opt = np.clip(
                new_pressure_val_p,
                mmp_val * 1.01,
                min(self.eor_params.max_pressure_psi, mmp_val * pressure_constraint_factor) # Apply max constraint
            )
        return best_pressure_found

    def check_mmp_constraint(self, pressure: float) -> bool:
        if self._mmp_value is None: self.calculate_mmp()
        # self._mmp_value should always be populated by calculate_mmp due to fallbacks
        return pressure >= self._mmp_value

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        return self._results

    @property
    def mmp(self) -> Optional[float]:
        if self._mmp_value is None:
            self.calculate_mmp() # Ensure it's calculated if accessed directly
        return self._mmp_value

    def set_recovery_model(self, model_name: str, **kwargs_for_model_init_override: Any):
        """
        Sets the recovery factor calculation model for the engine.

        Args:
            model_name (str): Name of the model (e.g., 'hybrid', 'koval').
            **kwargs_for_model_init_override: Keyword arguments to override the default
                                             __init__ parameters for this model instance.
                                             These will be merged with global defaults from config.
        """
        valid_models = ['simple', 'miscible', 'immiscible', 'hybrid', 'koval']
        model_name_lower = model_name.lower()
        if model_name_lower not in valid_models:
            raise ValueError(f"Unknown recovery model: {model_name}. Valid options: {valid_models}")
            
        self.recovery_model = model_name_lower
        
        # Start with global config defaults for this model's __init__
        base_init_config_params = config_manager.get_section(
            f"RecoveryModelKwargsDefaults.{self.recovery_model.capitalize()}"
        ) or {}
        
        # Merge with any overrides passed directly to this method
        self._recovery_model_init_kwargs = {**base_init_config_params, **kwargs_for_model_init_override}
        logging.info(f"Recovery model set to '{self.recovery_model}'. "
                     f"Effective __init__ kwargs for new instances: {self._recovery_model_init_kwargs}")

    def plot_mmp_profile(self) -> Optional[go.Figure]:
        if not (self.well_analysis and hasattr(self.well_analysis, 'calculate_mmp_profile')):
            logging.warning("WellAnalysis not available or does not support 'calculate_mmp_profile' for MMP plot.")
            return None
        try:
            profile_data = self.well_analysis.calculate_mmp_profile()
            if not isinstance(profile_data, dict) or not all(k in profile_data for k in ['depths', 'mmp']):
                logging.warning("MMP profile data from WellAnalysis is incomplete or not a dictionary.")
                return None
            if not profile_data['depths'].size or not profile_data['mmp'].size:
                 logging.warning("MMP profile data arrays are empty.")
                 return None

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=profile_data['mmp'], y=profile_data['depths'], name='MMP (psi)', mode='lines', line=dict(color='blue')),
                secondary_y=False
            )
            if 'temperature' in profile_data and profile_data['temperature'].size:
                fig.add_trace(
                    go.Scatter(x=profile_data['temperature'], y=profile_data['depths'], name='Temperature (°F)', mode='lines', line=dict(color='red')),
                    secondary_y=True
                )
            fig.update_layout(
                title_text='MMP vs Depth Profile',
                yaxis_title_text='Depth (ft)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Depth (ft)", secondary_y=False, autorange="reversed")
            if 'temperature' in profile_data and profile_data['temperature'].size:
                 fig.update_yaxes(title_text="Temperature Axis", secondary_y=True, autorange="reversed", overlaying='y', side='right', showticklabels=True)
                 fig.update_xaxes(title_text="Value") # Single x-axis title
            else:
                 fig.update_xaxes(title_text="MMP (psi)")
            return fig
        except Exception as e:
            logging.error(f"Error generating MMP vs Depth profile plot: {e}", exc_info=True)
            return None

    def plot_optimization_convergence(self, results_to_plot: Optional[Dict[str,Any]] = None) -> Optional[go.Figure]:
        # This is a simplified plot. For detailed convergence, history must be stored.
        plot_data_source = results_to_plot if results_to_plot is not None else self._results
        if not plot_data_source:
            logging.warning("No results available for convergence plot.")
            return None

        fig = go.Figure()
        method_name = plot_data_source.get('method', 'unknown_method')
        final_recovery_value = plot_data_source.get('final_recovery')

        if final_recovery_value is None:
             logging.warning(f"No 'final_recovery' in results for method '{method_name}'. Cannot plot convergence.")
             return None

        # Simplified: show a line at the final recovery value.
        # For GA: number of generations. For BO: number of iterations.
        # For Hybrid: Could try to combine GA generations and BO iterations on x-axis.
        
        num_steps_x_axis = 0
        plot_title = f'Optimization Outcome ({method_name})'

        if 'ga_full_results' in plot_data_source and isinstance(plot_data_source['ga_full_results'], dict): # Hybrid
            ga_res_data = plot_data_source['ga_full_results']
            ga_generations = ga_res_data.get('generations', 0)
            ga_final_rec = ga_res_data.get('final_recovery')
            if ga_generations > 0 and ga_final_rec is not None:
                fig.add_trace(go.Scatter(x=np.arange(1, ga_generations + 1), y=np.full(ga_generations, ga_final_rec),
                                         name=f'GA Phase Best (End)', mode='lines', line=dict(color='blue', dash='dot')))
            num_steps_x_axis += ga_generations
            plot_title = f'Hybrid Opt. Outcome (GA->BO)'
        
        bo_iterations = plot_data_source.get('iterations_bo_actual', 0)
        if bo_iterations > 0 : # Standalone BO or BO part of Hybrid
            # For hybrid, BO iterations start after GA generations
            bo_start_step = num_steps_x_axis + 1
            bo_end_step = num_steps_x_axis + bo_iterations
            fig.add_trace(go.Scatter(x=np.arange(bo_start_step, bo_end_step + 1),
                                     y=np.full(bo_iterations, final_recovery_value), # Show BO's final recovery
                                     name='BO Phase Final', mode='lines+markers', line=dict(color='green')))
            num_steps_x_axis += bo_iterations
        elif 'ga' not in method_name.lower() and 'gradient' not in method_name.lower(): # e.g. standalone BO with 0 iter but has result
             num_steps_x_axis = plot_data_source.get('iterations', 1) # Fallback for other methods

        if not fig.data : # If no specific phases plotted, just plot the final point
            if num_steps_x_axis == 0: num_steps_x_axis = 1 # ensure at least one point on x-axis
            fig.add_trace(go.Scatter(x=[num_steps_x_axis], y=[final_recovery_value], name='Final Recovery', mode='markers', marker=dict(size=10)))

        fig.update_layout(
            title_text=plot_title,
            xaxis_title_text='Optimization Steps (Generations/Iterations - Conceptual)',
            yaxis_title_text='Recovery Factor'
        )
        return fig

    def plot_parameter_sensitivity(self, param_name_for_sensitivity: str,
                                 results_to_use_for_plot: Optional[Dict[str,Any]]=None) -> Optional[go.Figure]:
        # `param_name_for_sensitivity` should match keys in `optimized_params` (e.g. 'target_pressure_psi')
        
        source_results = results_to_use_for_plot if results_to_use_for_plot is not None else self._results
        num_points_sensitivity = config_manager.get("OptimizationEngineSettings.sensitivity_plot_points", 20)

        if not source_results or 'optimized_params' not in source_results or not isinstance(source_results['optimized_params'], dict):
            logging.warning("No optimized parameters available for sensitivity plot.")
            return None
        if self._mmp_value is None: self.calculate_mmp()
        mmp_val = self._mmp_value

        optimized_params_base = source_results['optimized_params'].copy()
        avg_porosity_for_sens = source_results.get('avg_porosity',
                                             np.mean(self.reservoir.grid.get('PORO',
                                             [config_manager.get("GeneralFallbacks.porosity_default_fraction", 0.15)])))

        if param_name_for_sensitivity not in optimized_params_base or optimized_params_base[param_name_for_sensitivity] is None:
            logging.warning(f"Parameter '{param_name_for_sensitivity}' not found in optimized_params or is None. Cannot plot sensitivity.")
            return None
        
        current_optimal_value_for_param = optimized_params_base[param_name_for_sensitivity]
        
        # Determine bounds for sensitivity sweep (more robustly)
        # Use EORParameters for min/max if param matches one of its fields
        param_bounds_lookup = self._get_ga_parameter_bounds() # Re-use GA bounds logic as it's comprehensive
        
        low_bound_sens, high_bound_sens = current_optimal_value_for_param * 0.8, current_optimal_value_for_param * 1.2 # Default range
        if param_name_for_sensitivity in param_bounds_lookup:
            low_bound_sens, high_bound_sens = param_bounds_lookup[param_name_for_sensitivity]
            # Center the sweep around optimal value if possible, within global bounds
            sweep_range_half = (high_bound_sens - low_bound_sens) * 0.2 # e.g., sweep 20% of total range around opt
            param_range_low = max(low_bound_sens, current_optimal_value_for_param - sweep_range_half)
            param_range_high = min(high_bound_sens, current_optimal_value_for_param + sweep_range_half)
        elif param_name_for_sensitivity == 'target_pressure_psi': # Specific handling if not in param_bounds_lookup
            param_range_low = max(mmp_val * 1.01, current_optimal_value_for_param * 0.8)
            param_range_high = min(self.eor_params.max_pressure_psi, current_optimal_value_for_param * 1.2)
        else: # Generic fallback: +/- 20% of optimal value
            param_range_low = current_optimal_value_for_param * 0.8
            param_range_high = current_optimal_value_for_param * 1.2

        if param_range_high <= param_range_low: # Ensure valid range
            param_range_low = current_optimal_value_for_param * 0.95
            param_range_high = current_optimal_value_for_param * 1.05
        if abs(param_range_high - param_range_low) < 1e-6: # Handle case where optimal is at a boundary or range is tiny
             param_range_low -= (abs(current_optimal_value_for_param * 0.05) + 1e-3)
             param_range_high += (abs(current_optimal_value_for_param * 0.05) + 1e-3)


        parameter_values_for_sweep = np.linspace(param_range_low, param_range_high, num_points_sensitivity)
        recovery_values_sensitivity = []

        for p_sweep_val in parameter_values_for_sweep:
            temp_params_for_eval = optimized_params_base.copy()
            temp_params_for_eval[param_name_for_sensitivity] = p_sweep_val
            
            # Reconstruct necessary inputs for recovery_factor from temp_params_for_eval
            eval_pressure = temp_params_for_eval.get('target_pressure_psi', self.eor_params.target_pressure_psi)
            eval_base_rate = temp_params_for_eval.get('injection_rate', self.eor_params.injection_rate)
            
            effective_rate_sens = eval_base_rate
            if self.eor_params.injection_scheme == 'wag' and 'water_fraction' in temp_params_for_eval:
                wf_sens = temp_params_for_eval.get('water_fraction', self.eor_params.min_water_fraction)
                effective_rate_sens = eval_base_rate * (1.0 - wf_sens)
            
            # Pass all params from temp_params_for_eval (which includes the varied one)
            # and model's __init__ defaults to recovery_factor
            call_kwargs_for_sensitivity = {**self._recovery_model_init_kwargs, **temp_params_for_eval}
            
            recovery_values_sensitivity.append(
                recovery_factor(eval_pressure, effective_rate_sens, avg_porosity_for_sens, mmp_val,
                                model=self.recovery_model, **call_kwargs_for_sensitivity)
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=parameter_values_for_sweep, y=recovery_values_sensitivity, mode='lines+markers',
                                 name=f'{param_name_for_sensitivity} sensitivity'))
        fig.add_vline(x=current_optimal_value_for_param, line_width=2, line_dash="dash", line_color="green", name="Optimal Value")
        
        # Improve title readability
        title_param_name = param_name_for_sensitivity.replace("_psi"," (psi)").replace("_days"," (days)") \
                                                  .replace("_bpd"," (bpd)").replace("_", " ").capitalize()
        fig.update_layout(
            title_text=f'Recovery Factor vs {title_param_name}',
            xaxis_title_text=title_param_name,
            yaxis_title_text='Recovery Factor'
        )
        return fig