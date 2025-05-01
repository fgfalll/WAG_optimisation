from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class WellData:
    """Container for well log data"""
    name: str
    depths: np.ndarray
    properties: Dict[str, np.ndarray]  # {'GR': array, 'RHOB': array, etc.}
    units: Dict[str, str]
    
    def validate(self) -> bool:
        """Check data consistency"""
        return len(self.depths) == len(next(iter(self.properties.values())))

@dataclass
class ReservoirData:
    """Container for reservoir simulation data"""
    grid: Dict[str, np.ndarray]  # {'PORO': array, 'PERMX': array, etc.}
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

@dataclass
class EORParameters:
    """CO2 EOR operational parameters"""
    injection_rate: float
    WAG_ratio: Optional[float] = None
    injection_scheme: str = 'continuous'  # or 'wag'
    target_pressure: float = 0.0