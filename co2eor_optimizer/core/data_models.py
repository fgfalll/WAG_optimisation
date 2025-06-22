from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

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
    injection_rate: float = 5000.0
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
        defaults = {f.name: f.default for f in fields(cls) if f.default is not field.MISSING}
        defaults.update(config_eor_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclass
class GeneticAlgorithmParams:
    population_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_count: int = 3
    tournament_size: int = 3
    blend_alpha_crossover: float = 0.5
    mutation_strength_factor: float = 0.1

    @classmethod
    def from_config_dict(cls, config_ga_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in fields(cls) if f.default is not field.MISSING}
        defaults.update(config_ga_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)