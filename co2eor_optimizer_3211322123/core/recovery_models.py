from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union
import numpy as np
import logging
from abc import ABC, abstractmethod

from scipy.interpolate import CubicSpline

# config_manager.py is in the project root, one level above 'core/'
# If project root is in sys.path, this direct import will work.
try:
    from config_manager import config_manager, ConfigNotLoadedError
except ImportError:
    logging.critical(
        "ConfigManager could not be imported from core/recovery_models.py. "
        "Ensure config_manager.py is in the project root and project root is in PYTHONPATH. "
        "Functionality will be severely limited."
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


# Check for GPU availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None # type: ignore
    CUPY_AVAILABLE = False
    # logging.warning("cupy not installed. GPU acceleration will be disabled in recovery models.")

@dataclass
class RecoveryModel(ABC):
    @abstractmethod
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        pass

    def _validate_inputs(self, pressure: float, rate: float, porosity: float, mmp: float) -> None:
        if not all(isinstance(x, (int, float)) for x in [pressure, rate, porosity, mmp]):
            raise ValueError("All parameters (pressure, rate, porosity, mmp) must be numeric.")
        if any(x <= 0 for x in [pressure, porosity, mmp]) or rate < 0:
            raise ValueError("Pressure, porosity, MMP must be positive. Rate must be non-negative.")

class KovalRecoveryModel(RecoveryModel):
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        v_dp = kwargs.get('v_dp_coefficient', config_manager.get("RecoveryModelKwargsDefaults.Koval.v_dp_coefficient", 0.5))
        mobility_ratio_param = kwargs.get('mobility_ratio', config_manager.get("RecoveryModelKwargsDefaults.Koval.mobility_ratio", 2.0))
        mu_co2 = kwargs.get('mu_co2', config_manager.get("RecoveryModelKwargsDefaults.Koval.mu_co2", 0.1))
        mu_oil = kwargs.get('mu_oil', config_manager.get("RecoveryModelKwargsDefaults.Koval.mu_oil", 5.0))

        if porosity > 0.4: logging.debug(f"Koval: Porosity {porosity} > 0.4 seems high.")
        if mmp < 500 or mmp > 15000: logging.debug(f"Koval: MMP {mmp} psi is outside typical 1000-10000 range.")
        if not (0.3 <= v_dp <= 0.8): raise ValueError(f"Koval: V_DP must be 0.3-0.8, got {v_dp}")
        if not (1.2 <= mobility_ratio_param <= 20): raise ValueError(f"Koval: Mobility ratio param must be 1.2-20, got {mobility_ratio_param}")
        if mu_co2 <= 0 or mu_oil <= 0: raise ValueError("Koval: Viscosities must be positive.")

        kv_effective = v_dp * (1 + (mu_co2 / mu_oil))
        if kv_effective == 0: return 0.0
        mr_term = (mobility_ratio_param - 1) / kv_effective
        es = 1.0 / (1 + np.sqrt(mr_term)) if mr_term >= 0 else 1.0

        pressure_ratio = pressure / mmp if mmp > 0 else 10.0
        if pressure_ratio >= 1.0: return min(0.85, es)
        elif pressure_ratio >= 0.8: return es * 0.9
        return es * 0.7

class SimpleRecoveryModel(RecoveryModel):
    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        safe_rate = max(rate, 1e-6)
        sweep_efficiency = 0.7 * (safe_rate ** 0.2) * (porosity ** 0.5)
        return min(0.7, miscibility * sweep_efficiency)

class MiscibleRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        self.kv_factor = model_init_kwargs.get('kv_factor', config_manager.get("RecoveryModelKwargsDefaults.Miscible.kv_factor", 0.5))
        self.gravity_factor = model_init_kwargs.get('gravity_factor', config_manager.get("RecoveryModelKwargsDefaults.Miscible.gravity_factor", 0.1))
        if not (0.3 <= self.kv_factor <= 0.8): raise ValueError(f"Miscible: kv_factor must be 0.3-0.8, got {self.kv_factor}")
        if not (0 <= self.gravity_factor <= 1): raise ValueError(f"Miscible: gravity_factor must be 0-1, got {self.gravity_factor}")
        self.model_runtime_defaults = model_init_kwargs

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
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
        Kv_effective = self.kv_factor * (1 + (mu_co2 / mu_oil))
        
        Es_comp = 1.0 - 0.5 * (1.0 / (1.0 + M)) * (1.0 - np.exp(-2.0 * (1.0 + M))) if M > -1.0 else 0.1
        Es_gravity = Es_comp * (1.0 - min(max(Ng, -1.0), 1.0))
        Es_fingering = 1.0
        if M > 1.0 and Kv_effective > 1e-9:
            Es_fingering = 1.0 / (1.0 + np.sqrt((M - 1.0) / Kv_effective))
        elif M > 1.0 and Kv_effective <= 1e-9:
             Es_fingering = 0.1

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

        M_wo = (mu_oil * self.krw_max * 1.1) / (mu_water * kro_max)
        safe_pressure = max(pressure, 1e-6)
        Nc = rate * mu_water / (porosity * safe_pressure)
        
        E_displacement = (1.0 - self.sor) * (1.0 - np.exp(-Nc))
        E_vertical = 0.92 * (porosity ** 0.25)
        E_areal = 1.0 / (1.0 + M_wo * 0.9) if (1.0 + M_wo * 0.9) > 1e-9 else 0.1
             
        miscibility_factor = 0.9 * min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        combined_eff = miscibility_factor * E_displacement * E_vertical * E_areal
        combined_eff = min(0.68, max(0.0, combined_eff))
        return max(combined_eff, 0.051)

class HybridRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        te_config = {
            'mode': 'sigmoid', 'alpha': 1.0, 'beta': 20.0, 'use_gpu': False,
            'x_points': [0.5, 0.8, 1.0, 1.2, 1.5], 'y_points': [0.05, 0.2, 0.5, 0.8, 0.95]
        }
        hybrid_global_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Hybrid") or {}
        te_config.update(hybrid_global_defaults)
        te_config.update(model_init_kwargs) 
        self.transition_engine = TransitionEngine(
            mode=te_config.get('mode','sigmoid'), 
            use_gpu=te_config.get('use_gpu', False), 
            alpha=te_config.get('alpha', 1.0), beta=te_config.get('beta', 20.0),
            x_points=te_config.get('x_points'), y_points=te_config.get('y_points')
        )
        
        miscible_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Miscible") or {}
        immiscible_defaults = config_manager.get_section("RecoveryModelKwargsDefaults.Immiscible") or {}
        
        # model_init_kwargs might contain 'miscible_params' and 'immiscible_params' dicts
        final_miscible_params = {**miscible_defaults, **model_init_kwargs.get('miscible_params', {})}
        final_immiscible_params = {**immiscible_defaults, **model_init_kwargs.get('immiscible_params', {})}
        
        self.miscible_model = MiscibleRecoveryModel(**final_miscible_params)
        self.immiscible_model = ImmiscibleRecoveryModel(**final_immiscible_params)

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        p_mmp_ratio = np.array(pressure / mmp if mmp > 0 else 10.0) # Ensure mmp > 0
        transition_weight = self.transition_engine.calculate_efficiency(p_mmp_ratio)
        
        immiscible_result = self.immiscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        miscible_result = self.miscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        
        blended_recovery = immiscible_result * (1.0 - transition_weight) + miscible_result * transition_weight
        result = np.clip(blended_recovery, 0.0, 0.9)
        return result.item() if isinstance(result, np.ndarray) else float(result)

@dataclass
class LayerDefinition:
    pv_fraction: float
    perm_factor: float
    porosity: float

class LayeredRecoveryModel(RecoveryModel):
    def __init__(self,
                 layer_definitions: Optional[List[Union[LayerDefinition, Dict[str, Any]]]] = None,
                 base_model_type: str = 'simple',
                 **base_model_init_kwargs: Any):
        
        parsed_layer_definitions: List[LayerDefinition] = []
        if layer_definitions is None:
            # Try to load from config first
            default_layer_defs_list = config_manager.get("RecoveryModelKwargsDefaults.Layered.default_layer_definitions", None)
            if default_layer_defs_list and isinstance(default_layer_defs_list, list):
                try:
                    parsed_layer_definitions = [LayerDefinition(**ld_dict) for ld_dict in default_layer_defs_list]
                    logging.info("LayeredRecoveryModel: Using default layer definitions from config.")
                except TypeError as e:
                    logging.error(f"Error parsing default_layer_definitions from config: {e}. Using hardcoded defaults.")
                    parsed_layer_definitions = [
                        LayerDefinition(pv_fraction=0.4, perm_factor=2.5, porosity=0.22),
                        LayerDefinition(pv_fraction=0.6, perm_factor=0.5, porosity=0.18)
                    ]
            else: # Fallback if not in config or not a list
                parsed_layer_definitions = [
                    LayerDefinition(pv_fraction=0.4, perm_factor=2.5, porosity=0.22),
                    LayerDefinition(pv_fraction=0.6, perm_factor=0.5, porosity=0.18)
                ]
                logging.info("LayeredRecoveryModel: Using hardcoded default 2-layer definition.")
        else:
            if all(isinstance(ld, dict) for ld in layer_definitions):
                parsed_layer_definitions = [LayerDefinition(**ld) for ld in layer_definitions] # type: ignore
            elif all(isinstance(ld, LayerDefinition) for ld in layer_definitions):
                 parsed_layer_definitions = layer_definitions # type: ignore
            else:
                raise ValueError("layer_definitions must be a list of LayerDefinition objects or a list of dicts.")
        self.layer_definitions = parsed_layer_definitions

        total_pv_fraction = sum(ld.pv_fraction for ld in self.layer_definitions)
        if not np.isclose(total_pv_fraction, 1.0):
            if 0.9 < total_pv_fraction < 1.1 and total_pv_fraction > 1e-6 :
                logging.warning(f"Normalizing layer pv_fractions from sum {total_pv_fraction:.3f} to 1.0.")
                for ld_item in self.layer_definitions: ld_item.pv_fraction /= total_pv_fraction
            else:
                raise ValueError(f"Sum of layer pv_fractions ({total_pv_fraction:.3f}) must be close to 1.0 or normalizable.")

        self.base_model_type = base_model_type.lower()
        # These are kwargs for the base_model's constructor, e.g. for MiscibleModel's kv_factor
        self.base_model_constructor_kwargs = base_model_init_kwargs

        model_constructors: Dict[str, Callable[..., RecoveryModel]] = {
            'simple': SimpleRecoveryModel, 'miscible': MiscibleRecoveryModel,
            'immiscible': ImmiscibleRecoveryModel, 'koval': KovalRecoveryModel,
        }
        if self.base_model_type not in model_constructors:
             raise ValueError(f"Unsupported base_model_type for LayeredRecoveryModel: {self.base_model_type}")

        self._base_model_instance = model_constructors[self.base_model_type](**self.base_model_constructor_kwargs)

    def calculate(self, pressure: float, total_rate: float, avg_porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, total_rate, avg_porosity, mmp)

        total_weighted_recovery = 0.0
        
        # Calculate sum of (perm_factor * pv_fraction) for rate distribution
        # This weight determines how much of the total_rate goes to each layer
        rate_distribution_weights = np.array([ld.perm_factor * ld.pv_fraction for ld in self.layer_definitions])
        sum_of_weights = np.sum(rate_distribution_weights)

        if sum_of_weights <= 1e-9: # If all perm_factors or pv_fractions are zero
            # Fallback: Distribute rate purely by pv_fraction
            rate_distribution_weights = np.array([ld.pv_fraction for ld in self.layer_definitions])
            sum_of_weights = np.sum(rate_distribution_weights) # Should be ~1.0
            if sum_of_weights <= 1e-9: # All pv_fractions are zero (edge case)
                logging.warning("LayeredRecoveryModel: All pv_fractions are zero. Returning zero recovery.")
                return 0.0
            logging.warning("LayeredRecoveryModel: Sum of perm_factor*pv_fraction is zero. Distributing rate by pv_fraction only.")

        for i, layer_def in enumerate(self.layer_definitions):
            layer_rate_fraction = rate_distribution_weights[i] / sum_of_weights
            layer_rate = total_rate * layer_rate_fraction
            
            # The **kwargs passed to the main calculate are runtime params for the base model
            # e.g. v_dp_coefficient, mobility_ratio if base model is Koval
            layer_rf = self._base_model_instance.calculate(
                pressure, layer_rate, layer_def.porosity, mmp, **kwargs 
            )
            total_weighted_recovery += layer_rf * layer_def.pv_fraction # Weight RF by layer's PV fraction
            
        return np.clip(total_weighted_recovery, 0.0, 0.95)

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
        try:
            self._spline = CubicSpline(x_sorted, y_sorted, bc_type='natural')
        except ValueError as e:
            logging.error(f"Failed to create CubicSpline (x={x_sorted}, y={y_sorted}): {e}")
            raise

    def evaluate(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        if self._spline is None:
            raise RuntimeError("CubicTransition spline not fitted.")
        return np.clip(self._spline(p_mmp_ratio), 0.0, 1.0)

class TransitionEngine:
    def __init__(self, mode: str = 'sigmoid', use_gpu: bool = False, **params: Any):
        self.mode = mode.lower()
        self.params = params
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
                x_points=self.params.get('x_points'),
                y_points=self.params.get('y_points')
            )
        elif self.mode == 'custom' and 'custom_fn' in self.params and isinstance(self.params['custom_fn'], TransitionFunction):
            self._transition_function = self.params['custom_fn']
        else:
            # logging.warning(f"Invalid mode '{self.mode}' or params for TransitionEngine. Defaulting to sigmoid.")
            self._transition_function = SigmoidTransition(alpha=1.0, beta=20.0)
            self.mode = 'sigmoid'

    def enable_gpu_acceleration(self):
        if not CUPY_AVAILABLE:
            self._gpu_enabled = False; return
        self._gpu_enabled = True

    def calculate_efficiency(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        if self._transition_function is None:
            # logging.error("TransitionEngine function not initialized. Defaulting to sigmoid.")
            self._setup_transition_function()
            if self._transition_function is None: raise RuntimeError("Transition function init failed.")
        
        # if self._gpu_enabled and CUPY_AVAILABLE:
            # logging.debug("TransitionEngine: GPU mode active.")
        
        return self._transition_function.evaluate(p_mmp_ratio)

def recovery_factor(pressure: float, rate: float, porosity: float, mmp: float,
                  model: str = 'simple', **kwargs_for_model_calculate_and_init) -> float:
    model_name_lower = model.lower()
    
    # Get model-specific initialization parameters from config
    # Example: RecoveryModelKwargsDefaults.Layered will have 'layer_definitions', 'base_model_type', 'base_model_init_kwargs'
    # Example: RecoveryModelKwargsDefaults.Miscible will have 'kv_factor', 'gravity_factor'
    model_config_key = f"RecoveryModelKwargsDefaults.{model_name_lower.capitalize()}"
    config_init_params = config_manager.get_section(model_config_key) or {}

    # Allow runtime overrides for model initialization parameters
    # These are passed within kwargs_for_model_calculate_and_init as a dict under keys like 'layered_params' or 'miscible_params'
    # Or generically as 'model_init_kwargs'
    specific_model_init_overrides = kwargs_for_model_calculate_and_init.get(f"{model_name_lower}_params", {})
    general_model_init_overrides = kwargs_for_model_calculate_and_init.get("model_init_kwargs", {})
    
    # Combine config defaults with runtime overrides for initialization
    # Runtime overrides take precedence
    final_model_constructor_args = {**config_init_params, **specific_model_init_overrides, **general_model_init_overrides}

    model_constructors: Dict[str, Callable[..., RecoveryModel]] = {
        'simple': SimpleRecoveryModel, 'miscible': MiscibleRecoveryModel,
        'immiscible': ImmiscibleRecoveryModel, 'hybrid': HybridRecoveryModel,
        'koval': KovalRecoveryModel,
        'layered': LayeredRecoveryModel 
    }
    constructor = model_constructors.get(model_name_lower)
    if not constructor:
        raise ValueError(f"Unknown recovery model: {model_name_lower}. Valid: {list(model_constructors.keys())}")
    
    # Instantiate the model using the combined constructor arguments
    selected_model_instance = constructor(**final_model_constructor_args)

    # Parameters for the .calculate() method are directly in kwargs_for_model_calculate_and_init
    # These include v_dp_coefficient, mobility_ratio, mu_co2, etc.
    # We don't need to filter them here as the calculate methods use .get() or expect them.
    return selected_model_instance.calculate(pressure, rate, porosity, mmp, **kwargs_for_model_calculate_and_init)