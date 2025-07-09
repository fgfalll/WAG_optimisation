"""
Well analysis module for CO2 EOR optimization.

This module is fully integrated with the compositional EOS models. It provides
a hierarchical approach to determining fluid and thermal properties for MMP
calculations:
1.  **EOS Model (Compositional):** If an EOS model is provided, it is the
    highest authority. API gravity is calculated from the fluid composition
    by flashing at standard conditions.
2.  **Explicit Metadata:** Manually entered data on the well (e.g., a single
    API gravity or reservoir temperature) is used next.
3.  **Log-Derived Properties:** Well logs (e.g., RHOB for density/API) are
    used if higher-priority data is absent.
4.  **PVT/Configuration Fallbacks:** Finally, data from PVT models or
    configurable default values are used.

This ensures the most physically accurate data is always used when available.
"""
import sys
import os
from typing import Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

import numpy as np

# This block ensures that the project root is in the path,
# allowing for correct imports of other modules like core and evaluation.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_models import WellData, PVTProperties
from evaluation.mmp import calculate_mmp, MMPParameters

# Use a forward reference for EOSModel to prevent circular import errors,
# as eos_models.py might also import from core.data_models.
if TYPE_CHECKING:
    from core.eos_models import EOSModel

logger = logging.getLogger(__name__)

# Constants for standard conditions (60°F, 1 atm)
STD_TEMP_F = 60.0
STD_PRESS_PSIA = 14.696

@dataclass
class WellAnalysisConfig:
    """
    Configuration for WellAnalysis to avoid hardcoded values.

    This dataclass holds physical constants and default parameters used in the
    well analysis calculations, making them easily configurable.
    """
    surface_temp: float = 70.0  # Default surface temperature in °F
    default_api_gravity: float = 32.0  # Default API gravity if no other source is available
    default_reservoir_temp: float = 212.0 # Default reservoir temperature if no other source is available
    default_gas_composition: Dict[str, float] = field(default_factory=lambda: {'CO2': 1.0})

@dataclass
class WellAnalysis:
    """
    Integrates well log data, PVT properties, and compositional EOS models
    for comprehensive EOR analysis using a configurable and vectorized approach.
    """
    well_data: WellData
    pvt_data: Optional[PVTProperties] = None
    eos_model: Optional['EOSModel'] = None
    temperature_gradient: Optional[float] = None  # °F/ft
    config: WellAnalysisConfig = field(default_factory=WellAnalysisConfig)

    def _get_api_from_eos(self) -> Optional[float]:
        """
        Calculates the stock tank oil API gravity from the compositional EOS model.

        This is done by performing a flash calculation at standard conditions to find
        the density of the resulting liquid phase. Returns None if the calculation fails.
        """
        if self.eos_model is None:
            return None

        try:
            logger.info("Calculating API gravity from EOS model at standard conditions.")
            # Flash the fluid at standard temperature and pressure
            std_cond_props = self.eos_model.calculate_properties(
                pressure_psia=STD_PRESS_PSIA,
                temperature_F=STD_TEMP_F
            )

            # Check if the flash was successful and resulted in a liquid phase
            if std_cond_props.get('status') != 'Success' or 'oil_density_kg_m3' not in std_cond_props:
                logger.warning("EOS flash at standard conditions failed or did not yield a liquid phase.")
                return None

            # oil_density_kg_m3 is rho_o
            rho_o = std_cond_props['oil_density_kg_m3']
            # Density of water is ~1000 kg/m^3
            specific_gravity_oil = rho_o / 1000.0
            
            if specific_gravity_oil <= 0:
                return None
                
            api_gravity = (141.5 / specific_gravity_oil) - 131.5
            logger.info(f"API gravity from EOS model: {api_gravity:.2f}°")
            return api_gravity

        except Exception as e:
            logger.error(f"Error calculating API from EOS model: {e}", exc_info=True)
            return None

    def _estimate_api_profile(self) -> np.ndarray:
        """
        Estimates the API gravity profile using the best available data source.

        The priority for data sources is:
        1.  Compositional EOS Model
        2.  Explicit 'API' value in WellData metadata
        3.  RHOB (density) log from WellData properties
        4.  Default API gravity from the configuration
        """
        # --- Priority 1: Use EOS model if available ---
        api_from_eos = self._get_api_from_eos()
        if api_from_eos is not None:
            logger.info(f"Using constant API gravity profile from EOS model for '{self.well_data.name}'.")
            return np.full(len(self.well_data.depths), api_from_eos)

        # --- Priority 2: Check for explicit metadata ---
        if 'API' in self.well_data.metadata:
            api_val = self.well_data.metadata['API']
            logger.info(f"Using explicit API gravity from well metadata for '{self.well_data.name}': {api_val}°")
            return np.full(len(self.well_data.depths), api_val)

        # --- Priority 3: Use RHOB log if available ---
        if 'RHOB' in self.well_data.properties and self.well_data.properties['RHOB'].size > 0:
            logger.info(f"Estimating API gravity profile from RHOB log for '{self.well_data.name}'.")
            rhob = self.well_data.properties['RHOB'].copy()
            rhob[rhob <= 0] = 1.0  # Use water density (g/cc) as a fallback for invalid data
            api = (141.5 / rhob) - 131.5
            return np.clip(api, 15, 50) # Clip to valid range for MMP correlations

        # --- Priority 4: Fallback to configurable default ---
        logging.warning(
            f"For well '{self.well_data.name}', no reliable source for API gravity found. "
            f"Using default from config: {self.config.default_api_gravity}°"
        )
        return np.full(len(self.well_data.depths), self.config.default_api_gravity)

    def _calculate_temperature_profile(self) -> np.ndarray:
        """
        Calculates temperature vs depth with a clear priority.
        1. User-provided temperature from metadata (assumed constant).
        2. Gradient calculation if a gradient is provided.
        3. Constant temperature from PVT data.
        4. Configurable default value.
        """
        # Priority 1: Explicit metadata
        if 'Temperature' in self.well_data.metadata:
            temp_val = self.well_data.metadata['Temperature']
            logger.info(f"Using explicit constant temperature from well metadata for '{self.well_data.name}': {temp_val}°F")
            return np.full(len(self.well_data.depths), temp_val)

        # Priority 2: Temperature gradient
        if self.temperature_gradient is not None:
            logger.info(f"Calculating temperature profile for '{self.well_data.name}' with gradient: {self.temperature_gradient}°F/ft")
            ref_temp = self.config.surface_temp
            ref_depth = self.well_data.depths[0] if self.well_data.depths.size > 0 else 0
            return ref_temp + self.temperature_gradient * (self.well_data.depths - ref_depth)
        
        # Priority 3: PVT data
        if self.pvt_data and hasattr(self.pvt_data, 'temperature'):
            logger.info(f"Using constant temperature from PVT data for '{self.well_data.name}': {self.pvt_data.temperature}°F")
            return np.full(len(self.well_data.depths), self.pvt_data.temperature)
            
        # Priority 4: Configurable default
        logger.warning(f"No temperature source found. Using default from config: {self.config.default_reservoir_temp}°F")
        return np.full(len(self.well_data.depths), self.config.default_reservoir_temp)

    def calculate_mmp_profile(self,
                            method: str = 'auto',
                            gas_composition: Optional[Dict] = None,
                            c7_plus_mw_override: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Calculates a depth-based MMP profile using the best available data.

        This method vectorizes the temperature and API gravity profile calculations
        and then iterates to calculate MMP at each depth, as the external MMP
        calculator operates on single-point parameter sets.
        """
        if not hasattr(self.well_data, 'depths') or self.well_data.depths.size == 0:
            raise ValueError("WellData must contain depth information to calculate a profile.")

        final_gas_comp = gas_composition or self.config.default_gas_composition
        api_profile = self._estimate_api_profile()
        temp_profile = self._calculate_temperature_profile()
        mmp_values = np.zeros_like(self.well_data.depths)

        # FIXED: Prioritize the UI override, then fall back to metadata.
        # This handles cases where override is 0.0 or other falsy values correctly.
        c7_plus_mw = c7_plus_mw_override if c7_plus_mw_override is not None else self.well_data.metadata.get('C7_plus_MW')

        for i in range(len(self.well_data.depths)):
            params = MMPParameters(
                temperature=temp_profile[i],
                oil_gravity=api_profile[i],
                injection_gas_composition=final_gas_comp,
                c7_plus_mw=c7_plus_mw,
                pvt_data=self.pvt_data # Pass PVT data for potential use in MMP correlations
            )
            mmp_values[i] = calculate_mmp(params, method=method)

        return {
            'depths': self.well_data.depths,
            'mmp': mmp_values,
            'temperature': temp_profile,
            'api': api_profile
        }

    def find_miscible_zones(self,
                          pressure: float,
                          gas_composition: Optional[Dict] = None,
                          method: str = 'auto') -> Dict[str, np.ndarray]:
        """
        Identifies reservoir zones where a given injection pressure achieves miscibility.
        """
        profile = self.calculate_mmp_profile(method=method, gas_composition=gas_composition)
        is_miscible = pressure >= profile['mmp']

        return {
            'depths': profile['depths'],
            'is_miscible': is_miscible,
            'mmp': profile['mmp'],
            'temperature': profile['temperature']
        }

    def get_average_mmp_params_for_engine(self, pay_zone_depths: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Calculates average fluid and thermal properties over a specified depth interval
        to provide a representative set of parameters for the OptimizationEngine.

        Args:
            pay_zone_depths: An optional tuple (start_depth, end_depth) defining the pay zone.
                             If None, averages over the entire well depth range.

        Returns:
            A dictionary of average parameters suitable for constructing an MMPParameters object.
        """
        profiles = self.calculate_mmp_profile()
        depths, temps, apis = profiles['depths'], profiles['temperature'], profiles['api']

        if depths.size == 0:
            logger.warning("Cannot get average MMP params; no depth data in well.")
            return {}

        if pay_zone_depths:
            start_depth, end_depth = min(pay_zone_depths), max(pay_zone_depths)
            mask = (depths >= start_depth) & (depths <= end_depth)
            if not np.any(mask):
                logger.warning(f"Pay zone {pay_zone_depths} ft resulted in no data points. Averaging over full well.")
                mask = np.full_like(depths, True, dtype=bool)
        else:
            mask = np.full_like(depths, True, dtype=bool)

        avg_params = {
            'temperature': np.mean(temps[mask]),
            'oil_gravity': np.mean(apis[mask]),
            'c7_plus_mw': self.well_data.metadata.get('C7_plus_MW'),
            'injection_gas_composition': self.config.default_gas_composition
        }
        logger.info(f"Calculated average parameters for OptimizationEngine: {avg_params}")
        return avg_params