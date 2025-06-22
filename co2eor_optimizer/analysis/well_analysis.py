"""
Well analysis module for CO2 EOR optimization

IMPROVED: This module has been refactored to eliminate hardcoded values
by introducing a configuration class. It is now more robust and flexible.
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

from core.data_models import WellData, PVTProperties
from evaluation.mmp import calculate_mmp, MMPParameters

# Configure logging to see informational messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class WellAnalysisConfig:
    """
    Configuration for WellAnalysis to avoid hardcoded values.

    This dataclass holds physical constants and default parameters used in the
    well analysis calculations, making them easily configurable.
    """
    surface_temp: float = 70.0  # Default surface temperature in °F
    default_api_gravity: float = 32.0  # Default API gravity if RHOB log is not available
    default_reservoir_temp: float = 212.0 # Default reservoir temperature if no gradient or PVT data is available
    default_gas_composition: Dict[str, float] = field(default_factory=lambda: {'CO2': 1.0})

@dataclass
class WellAnalysis:
    """
    Integrates well log data with EOR analysis using a configurable and
    vectorized approach.
    """
    well_data: WellData
    pvt_data: Optional[PVTProperties] = None
    temperature_gradient: Optional[float] = None  # °F/ft
    config: WellAnalysisConfig = field(default_factory=WellAnalysisConfig)

    def calculate_mmp_profile(self,
                            method: str = 'auto',
                            gas_composition: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Calculate MMP profile vs depth using well log data.

        This method uses vectorized calculations for temperature and API gravity profiles
        for improved performance.
        """
        if not hasattr(self.well_data, 'depths') or self.well_data.depths.size == 0:
            raise ValueError("WellData must contain depth information")

        # Use the provided gas composition or fall back to the one in the config
        final_gas_comp = gas_composition or self.config.default_gas_composition

        # 1. Calculate the full profiles for temperature and API gravity vectorially
        api_profile = self._estimate_api_from_logs()
        temp_profile = self._calculate_temperature_profile()
        
        mmp_values = np.zeros_like(self.well_data.depths)

        # 2. Loop to call the MMP calculator for each depth point.
        # A loop is appropriate here because calculate_mmp in mmp.py has internal
        # logic ('auto' method selection) that operates on single parameter sets.
        for i in range(len(self.well_data.depths)):
            params = MMPParameters(
                temperature=temp_profile[i],
                oil_gravity=api_profile[i],
                injection_gas_composition=final_gas_comp,
                pvt_data=self.pvt_data
                # c7_plus_mw would be passed here if available from data
            )
            mmp_values[i] = calculate_mmp(params, method=method)

        return {
            'depths': self.well_data.depths,
            'mmp': mmp_values,
            'temperature': temp_profile,
            'api': api_profile
        }

    def _estimate_api_from_logs(self) -> np.ndarray:
        """
        Estimate API gravity from density log (RHOB) in a vectorized manner.
        Falls back to a configurable default if the log is not present.
        """
        if 'RHOB' not in self.well_data.properties:
            logging.warning(
                f"RHOB log not found. Using default API gravity from config: "
                f"{self.config.default_api_gravity}°"
            )
            return np.full(len(self.well_data.depths), self.config.default_api_gravity)

        rhob = self.well_data.properties['RHOB']
        
        # Prevent division by zero or invalid values
        rhob[rhob <= 0] = 1.0 # Assuming water density as a fallback for invalid data
        
        api = (141.5 / rhob) - 131.5  # Simple conversion
        
        # Clip values to the valid range for MMP correlations [cite: mmp.py]
        return np.clip(api, 15, 50)

    def _calculate_temperature_profile(self) -> np.ndarray:
        """
        Calculate temperature vs depth using a gradient in a vectorized manner.
        Uses more intelligent, configurable defaults if no gradient is provided.
        """
        if self.temperature_gradient is None:
            # First priority: Use reservoir temperature from PVT data if available.
            if self.pvt_data and hasattr(self.pvt_data, 'temperature'):
                logging.info(
                    "No temperature gradient provided. Using constant reservoir "
                    f"temperature from PVT data: {self.pvt_data.temperature}°F"
                )
                return np.full(len(self.well_data.depths), self.pvt_data.temperature)
            
            # Second priority: Use the default from the config object.
            logging.warning(
                "No temperature gradient or PVT temperature available. Using "
                f"default from config: {self.config.default_reservoir_temp}°F"
            )
            return np.full(len(self.well_data.depths), self.config.default_reservoir_temp)
        
        # Use the configurable surface temperature for the gradient calculation.
        ref_temp = self.config.surface_temp
        
        # Assume the first depth point is the reference for the gradient calculation
        # This is a common approach when a specific reference depth is not provided.
        ref_depth = self.well_data.depths[0]
        
        return ref_temp + self.temperature_gradient * (self.well_data.depths - ref_depth)

    def find_miscible_zones(self,
                          pressure: float,
                          gas_composition: Optional[Dict] = None,
                          method: str = 'auto') -> Dict[str, np.ndarray]:
        """
        Identify reservoir zones where the injection pressure will achieve miscibility.
        This calculation is now based on the improved, configurable profile calculation.
        """
        profile = self.calculate_mmp_profile(method=method, gas_composition=gas_composition)
        is_miscible = pressure >= profile['mmp']

        return {
            'depths': profile['depths'],
            'is_miscible': is_miscible,
            'mmp': profile['mmp'],
            'temperature': profile['temperature']
        }