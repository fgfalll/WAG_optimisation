from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union
import numpy as np
import logging
from abc import ABC, abstractmethod

from scipy.interpolate import CubicSpline

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
    def __init__(self, v_dp_coefficient: float = 0.5, mobility_ratio: float = 2.0, **kwargs):
        self.v_dp_coefficient = v_dp_coefficient
        self.mobility_ratio_param = mobility_ratio
        self.pres_ratio_threshold = kwargs.get("pressure_ratio_threshold", 0.8)
        self.full_misc_multiplier = kwargs.get("full_miscibility_multiplier", 1.0)
        self.near_misc_multiplier = kwargs.get("near_miscibility_multiplier", 0.9)
        self.immisc_multiplier = kwargs.get("immiscible_multiplier", 0.7)
        self.max_rf_cap = kwargs.get("max_rf_cap", 0.85)
        
        # Validate inputs during initialization
        if not (0.3 <= self.v_dp_coefficient <= 0.8):
            raise ValueError(f"Koval: V_DP must be 0.3-0.8, got {self.v_dp_coefficient}")
        if not (1.2 <= self.mobility_ratio_param <= 20):
            raise ValueError(f"Koval: Mobility ratio param must be 1.2-20, got {self.mobility_ratio_param}")

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        mu_co2 = kwargs.get('mu_co2', 0.1)
        mu_oil = kwargs.get('mu_oil', 5.0)

        # Physics-based Koval factor (Koval, 1963 SPE-654-PA)
        kv_effective = self.v_dp_coefficient * (1 + np.sqrt(mu_oil / mu_co2))
        if kv_effective == 0:
            return 0.0
            
        # Mobility ratio term
        mr_term = (self.mobility_ratio_param - 1) / kv_effective
        
        # Corrected sweep efficiency (Lake, 1989)
        es = 1.0 - 0.5 * (1 - np.exp(-2 * mr_term)) / (1 + mr_term) if mr_term >= 0 else 1.0

        # Miscibility effects
        pressure_ratio = pressure / mmp if mmp > 0 else 10.0
        if pressure_ratio >= 1.0:
            return min(self.max_rf_cap, es * self.full_misc_multiplier)
        elif pressure_ratio >= self.pres_ratio_threshold:
            return es * self.near_misc_multiplier
        return es * self.immisc_multiplier

class SimpleRecoveryModel(RecoveryModel):
    def __init__(self, sweep_base_factor: float = 0.7, rate_exponent: float = 0.2, 
                 porosity_exponent: float = 0.5, max_rf_cap: float = 0.7, **kwargs):
        self.sweep_base = sweep_base_factor
        self.rate_exp = rate_exponent
        self.poro_exp = porosity_exponent
        self.max_rf_cap = max_rf_cap

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)

        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        safe_rate = max(rate, 1e-6)
        sweep_efficiency = self.sweep_base * (safe_rate ** self.rate_exp) * (porosity ** self.poro_exp)
        return min(self.max_rf_cap, miscibility * sweep_efficiency)

class MiscibleRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        self.kv_factor = model_init_kwargs.get('kv_factor', 0.5)
        self.gravity_factor = model_init_kwargs.get('gravity_factor', 0.1)
        self.dip_angle_divisor = model_init_kwargs.get("dip_angle_divisor", 50.0)
        self.max_rf_cap = model_init_kwargs.get("max_rf_cap", 0.88)
        
        if not (0.3 <= self.kv_factor <= 0.8):
            raise ValueError(f"Miscible: kv_factor must be 0.3-0.8, got {self.kv_factor}")
        if not (0 <= self.gravity_factor <= 1):
            raise ValueError(f"Miscible: gravity_factor must be 0-1, got {self.gravity_factor}")
        self.model_runtime_defaults = model_init_kwargs

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        mu_co2 = kwargs.get('mu_co2', self.model_runtime_defaults.get('mu_co2', 0.06))
        mu_oil = kwargs.get('mu_oil', self.model_runtime_defaults.get('mu_oil', 5.0))
        rel_perm = kwargs.get('rel_perm', self.model_runtime_defaults.get('rel_perm', {'co2': 0.8, 'oil': 0.3}))
        dip_angle = kwargs.get('dip_angle', self.model_runtime_defaults.get('dip_angle', 0.0))
        k = kwargs.get('permeability', 100)  # md, required for gravity number
        density_diff = kwargs.get('density_diff', 0.3)  # g/cc, required for gravity number
        
        # Validate inputs
        if mu_co2 <= 0 or mu_oil <= 0:
            raise ValueError("Miscible: Viscosities must be positive.")
        if not (isinstance(rel_perm, dict) and 'co2' in rel_perm and 'oil' in rel_perm and
                isinstance(rel_perm['co2'], (int, float)) and rel_perm['co2'] > 0 and
                isinstance(rel_perm['oil'], (int, float)) and rel_perm['oil'] > 0):
            raise ValueError("Miscible: rel_perm must be a dict with positive 'co2' and 'oil' numeric values.")
        if k <= 0:
            raise ValueError("Miscible: Permeability must be positive.")
        if density_diff <= 0:
            raise ValueError("Miscible: Density difference must be positive.")

        # Physics-based gravity number (Green-Willhite, 1998)
        g = 980  # cm/s²
        area = kwargs.get('cross_sectional_area', 1e6)  # cm²
        v = (rate * 1e6) / (area * 86400)  # Convert m³/day to cm/s
        Ng = (k * 9.869e-9 * density_diff * g * np.sin(np.radians(dip_angle))) / (mu_co2 * v)

        miscibility = min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        M = (mu_oil * rel_perm['co2']) / (mu_co2 * rel_perm['oil'])
        
        # Koval effective factor
        Kv_effective = self.kv_factor * (1 + np.sqrt(mu_oil / mu_co2))
        
        # Gravity sweep impact (SPE 169044)
        Es_gravity = 1.0 - min(max(Ng, 0.0), 0.8)  # Cap at 80% reduction
        
        # Fingering effect (Lake, 1989)
        Es_fingering = 1.0
        if M > 1.0 and Kv_effective > 1e-9:
            mr_term = (M - 1.0) / Kv_effective
            Es_fingering = 1.0 - 0.5 * (1 - np.exp(-2 * mr_term)) / (1 + mr_term)
        elif M > 1.0:
            Es_fingering = 0.1

        combined_eff = miscibility * Es_fingering * Es_gravity * (porosity ** 0.5) * (1.0 + dip_angle / self.dip_angle_divisor)
        return min(self.max_rf_cap, max(0.0, combined_eff))

class ImmiscibleRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        self.sor = model_init_kwargs.get('sor', 0.25)
        self.krw_max = model_init_kwargs.get('krw_max', 0.4)
        self.m_wo_factor = model_init_kwargs.get("m_wo_factor", 1.1)
        self.e_areal_factor = model_init_kwargs.get("e_areal_factor", 0.9)
        self.e_vertical_base = model_init_kwargs.get("e_vertical_base_factor", 0.92)
        self.e_vertical_poro_exp = model_init_kwargs.get("e_vertical_porosity_exponent", 0.25)
        self.miscibility_scaling = model_init_kwargs.get("miscibility_scaling_factor", 0.9)
        self.max_rf_cap = model_init_kwargs.get("max_rf_cap", 0.68)
        self.min_rf_floor = model_init_kwargs.get("min_rf_floor", 0.051)
        
        if not (0 <= self.sor < 1):
            raise ValueError(f"Immiscible: sor must be 0-1, got {self.sor}")
        if not (0 < self.krw_max <= 1):
            raise ValueError(f"Immiscible: krw_max must be 0-1, got {self.krw_max}")
        self.model_runtime_defaults = model_init_kwargs

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        mu_water = kwargs.get('mu_water', self.model_runtime_defaults.get('mu_water', 0.5))
        mu_oil = kwargs.get('mu_oil', self.model_runtime_defaults.get('mu_oil', 5.0))
        swc = kwargs.get('swc', self.model_runtime_defaults.get('swc', 0.2))
        kro_max = kwargs.get('kro_max', self.model_runtime_defaults.get('kro_max', 0.8))
        interfacial_tension = kwargs.get('interfacial_tension', 30.0)  # dynes/cm
        area = kwargs.get('cross_sectional_area', 1e6)  # cm²
        temp = kwargs.get('temperature', 50.0)  # °C

        # Validate inputs
        if mu_water <=0 or mu_oil <=0:
            raise ValueError("Immiscible: Viscosities must be positive.")
        if not (0 <= swc < 1):
            raise ValueError(f"Immiscible: swc must be 0-1, got {swc}")
        if not (0 < kro_max <= 1):
            raise ValueError(f"Immiscible: kro_max must be 0-1, got {kro_max}")
        if interfacial_tension <= 0:
            raise ValueError("Immiscible: Interfacial tension must be positive.")
        if area <= 0:
            raise ValueError("Immiscible: Cross-sectional area must be positive.")

        # Temperature-dependent SOR (SPE 180924)
        sor_temp_adjusted = self.sor * (1 - 0.005 * max(0, temp - 50))
        
        # Correct capillary number (Lake, 1989)
        v = (rate * 1e6) / (area * 86400)  # Convert m³/day to cm/s
        Nc = (v * mu_water) / interfacial_tension

        M_wo = (mu_oil * self.krw_max * self.m_wo_factor) / (mu_water * kro_max)
        
        E_displacement = (1.0 - sor_temp_adjusted) * (1.0 - np.exp(-Nc))
        E_vertical = self.e_vertical_base * (porosity ** self.e_vertical_poro_exp)
        E_areal = 1.0 / (1.0 + M_wo * self.e_areal_factor) if (1.0 + M_wo * self.e_areal_factor) > 1e-9 else 0.1
             
        miscibility_factor = self.miscibility_scaling * min(1.0, max(0.0, (pressure - mmp) / mmp if mmp > 0 else 1.0))
        combined_eff = miscibility_factor * E_displacement * E_vertical * E_areal
        combined_eff = min(self.max_rf_cap, max(0.0, combined_eff))
        return max(combined_eff, self.min_rf_floor)

class HybridRecoveryModel(RecoveryModel):
    def __init__(self, **model_init_kwargs):
        te_config = {
            'mode': 'sigmoid', 'alpha': 1.0, 'beta': 20.0,
            'x_points': [0.5, 0.8, 1.0, 1.2, 1.5], 'y_points': [0.05, 0.2, 0.5, 0.8, 0.95]
        }
        te_config.update(model_init_kwargs)
        self.transition_engine = TransitionEngine(
            mode=te_config.get('mode','sigmoid'), 
            alpha=te_config.get('alpha', 1.0), beta=te_config.get('beta', 20.0),
            x_points=te_config.get('x_points'), y_points=te_config.get('y_points')
        )
        
        self.max_rf_cap = model_init_kwargs.get("max_rf_cap", 0.9)
        
        miscible_defaults = {
            "kv_factor": 0.5, "gravity_factor": 0.1, "dip_angle_divisor": 50.0, "max_rf_cap": 0.88
        }
        immiscible_defaults = {
            "sor": 0.25, "krw_max": 0.4, "m_wo_factor": 1.1, "e_areal_factor": 0.9,
            "e_vertical_base_factor": 0.92, "e_vertical_porosity_exponent": 0.25,
            "miscibility_scaling_factor": 0.9, "max_rf_cap": 0.68, "min_rf_floor": 0.051
        }
        final_miscible_params = {**miscible_defaults, **model_init_kwargs.get('miscible_params', {})}
        final_immiscible_params = {**immiscible_defaults, **model_init_kwargs.get('immiscible_params', {})}
        
        self.miscible_model = MiscibleRecoveryModel(**final_miscible_params)
        self.immiscible_model = ImmiscibleRecoveryModel(**final_immiscible_params)

    def calculate(self, pressure: float, rate: float, porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, rate, porosity, mmp)
        p_mmp_ratio = np.array(pressure / mmp if mmp > 0 else 10.0)
        transition_weight = self.transition_engine.calculate_efficiency(p_mmp_ratio)
        
        immiscible_result = self.immiscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        miscible_result = self.miscible_model.calculate(pressure, rate, porosity, mmp, **kwargs)
        
        blended_recovery = immiscible_result * (1.0 - transition_weight) + miscible_result * transition_weight
        result = np.clip(blended_recovery, 0.0, self.max_rf_cap)
        return result.item() if isinstance(result, np.ndarray) else float(result)

@dataclass
class LayerDefinition:
    pv_fraction: float
    perm_factor: float
    porosity: float
    thickness: float = 1.0  # Added for transmissibility calculation

class LayeredRecoveryModel(RecoveryModel):
    def __init__(self,
                 layer_definitions: Optional[List[Union[LayerDefinition, Dict[str, Any]]]] = None,
                 base_model_type: str = 'simple',
                 **base_model_init_kwargs: Any):
        
        default_layer_defs_list = [
            {'pv_fraction':0.4, 'perm_factor':2.5, 'porosity':0.22, 'thickness':10.0},
            {'pv_fraction':0.6, 'perm_factor':0.5, 'porosity':0.18, 'thickness':10.0}
        ]

        parsed_layer_definitions: List[LayerDefinition] = []
        if layer_definitions is None:
            parsed_layer_definitions = [LayerDefinition(**ld) for ld in default_layer_defs_list]
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
            if 0.9 < total_pv_fraction < 1.1 and total_pv_fraction > 1e-6:
                logging.warning(f"Normalizing layer pv_fractions from sum {total_pv_fraction:.3f} to 1.0.")
                for ld_item in self.layer_definitions:
                    ld_item.pv_fraction /= total_pv_fraction
            else:
                raise ValueError(f"Sum of layer pv_fractions ({total_pv_fraction:.3f}) must be close to 1.0 or normalizable.")

        self.base_model_type = base_model_type.lower()
        self.base_model_constructor_kwargs = base_model_init_kwargs
        self.max_rf_cap = base_model_init_kwargs.get("max_rf_cap", 0.95)

        model_constructors: Dict[str, Callable[..., RecoveryModel]] = {
            'simple': SimpleRecoveryModel, 'miscible': MiscibleRecoveryModel,
            'immiscible': ImmiscibleRecoveryModel, 'koval': KovalRecoveryModel,
        }
        if self.base_model_type not in model_constructors:
             raise ValueError(f"Unsupported base_model_type for LayeredRecoveryModel: {self.base_model_type}")

        self._base_model_instance = model_constructors[self.base_model_type](**self.base_model_constructor_kwargs)

    def calculate(self, pressure: float, total_rate: float, avg_porosity: float, mmp: float, **kwargs) -> float:
        self._validate_inputs(pressure, total_rate, avg_porosity, mmp)

        # Physics-based rate allocation using transmissibility (kh)
        kh_values = np.array([ld.perm_factor * ld.thickness for ld in self.layer_definitions])
        total_kh = np.sum(kh_values)
        
        if total_kh <= 1e-9:
            logging.warning("LayeredRecoveryModel: Total kh is zero. Distributing rate by pv_fraction.")
            rate_distribution_weights = np.array([ld.pv_fraction for ld in self.layer_definitions])
        else:
            rate_distribution_weights = kh_values / total_kh

        total_weighted_recovery = 0.0
        for i, layer_def in enumerate(self.layer_definitions):
            layer_rate = total_rate * rate_distribution_weights[i]
            layer_rf = self._base_model_instance.calculate(
                pressure, layer_rate, layer_def.porosity, mmp, **kwargs 
            )
            total_weighted_recovery += layer_rf * layer_def.pv_fraction
            
        return min(self.max_rf_cap, max(0.0, total_weighted_recovery))

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
    def __init__(self, mode: str = 'sigmoid', **params: Any):
        self.mode = mode.lower()
        self.params = params
        self._transition_function: Optional[TransitionFunction] = None
        self._setup_transition_function()

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
            self._transition_function = SigmoidTransition(alpha=1.0, beta=20.0)
            self.mode = 'sigmoid'

    def calculate_efficiency(self, p_mmp_ratio: np.ndarray) -> np.ndarray:
        if self._transition_function is None:
            self._setup_transition_function()
            if self._transition_function is None:
                raise RuntimeError("Transition function init failed.")
        
        return self._transition_function.evaluate(p_mmp_ratio)

def recovery_factor(pressure: float, rate: float, porosity: float, mmp: float,
                  model: str = 'simple', **kwargs_for_model_calculate_and_init) -> float:
    model_name_lower = model.lower()
      
    final_model_constructor_args = kwargs_for_model_calculate_and_init

    model_constructors: Dict[str, Callable[..., RecoveryModel]] = {
        'simple': SimpleRecoveryModel, 'miscible': MiscibleRecoveryModel,
        'immiscible': ImmiscibleRecoveryModel, 'hybrid': HybridRecoveryModel,
        'koval': KovalRecoveryModel, 'layered': LayeredRecoveryModel 
    }

    constructor = model_constructors.get(model_name_lower)
    if not constructor:
        raise ValueError(f"Unknown recovery model: {model_name_lower}. Valid: {list(model_constructors.keys())}")
    
    selected_model_instance = constructor(**final_model_constructor_args)

    return selected_model_instance.calculate(pressure, rate, porosity, mmp, **kwargs_for_model_calculate_and_init)