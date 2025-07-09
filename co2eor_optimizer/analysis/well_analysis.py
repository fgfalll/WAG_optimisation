"""
Well analysis module for CO2 EOR optimization.

This module is fully integrated with the compositional EOS models. It provides
a hierarchical approach to determining fluid and thermal properties for MMP
calculations:

This ensures the most physically accurate data is always used when available.
"""
import sys
import os
from typing import Dict, Optional, Any, Tuple, TYPE_CHECKING, Callable
from dataclasses import dataclass, field
import logging

import numpy as np

from co2eor_optimizer.core.data_models import WellData, PVTProperties
from co2eor_optimizer.evaluation.mmp import calculate_mmp, MMPParameters

# Use a forward reference for EOSModel to prevent circular import errors,
# as eos_models.py might also import from core.data_models.
if TYPE_CHECKING:
    from co2eor_optimizer.core.eos_models import EOSModel

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
            std_cond_props = self.eos_model.calculate_properties(
                pressure_psia=STD_PRESS_PSIA,
                temperature_F=STD_TEMP_F
            )
            if std_cond_props.get('status') != 'Success' or 'oil_density_kg_m3' not in std_cond_props:
                logger.warning("EOS flash at standard conditions failed or did not yield a liquid phase.")
                return None
            rho_o = std_cond_props['oil_density_kg_m3']
            specific_gravity_oil = rho_o / 1000.0
            if specific_gravity_oil <= 0: return None
            api_gravity = (141.5 / specific_gravity_oil) - 131.5
            logger.info(f"API gravity from EOS model: {api_gravity:.2f}°")
            return api_gravity
        except Exception as e:
            logger.error(f"Error calculating API from EOS model: {e}", exc_info=True)
            return None

    def _estimate_api_profile(self) -> np.ndarray:
        """
        Estimates a continuous API gravity profile along the well path using the best available data source.
        """
        api_from_eos = self._get_api_from_eos()
        if api_from_eos is not None:
            return np.full(len(self.well_data.depths), api_from_eos)

        if 'API' in self.well_data.metadata:
            api_val = self.well_data.metadata['API']
            return np.full(len(self.well_data.depths), api_val)

        if 'RHOB' in self.well_data.properties and self.well_data.properties['RHOB'].size > 0:
            rhob = self.well_data.properties['RHOB'].copy()
            rhob[rhob <= 0] = 1.0
            api = (141.5 / rhob) - 131.5
            return np.clip(api, 15, 50)

        return np.full(len(self.well_data.depths), self.config.default_api_gravity)

    def _calculate_temperature_profile(self) -> np.ndarray:
        """
        Calculates a continuous temperature profile vs depth with a clear priority.
        """
        if 'Temperature' in self.well_data.metadata:
            temp_val = self.well_data.metadata['Temperature']
            return np.full(len(self.well_data.depths), temp_val)

        if self.temperature_gradient is not None:
            ref_temp = self.config.surface_temp
            ref_depth = self.well_data.depths[0] if self.well_data.depths.size > 0 else 0
            return ref_temp + self.temperature_gradient * (self.well_data.depths - ref_depth)
        
        if self.pvt_data and hasattr(self.pvt_data, 'temperature'):
            return np.full(len(self.well_data.depths), self.pvt_data.temperature)
            
        return np.full(len(self.well_data.depths), self.config.default_reservoir_temp)

    def calculate_mmp_profile(self,
                            method: str = 'auto',
                            gas_composition: Optional[Dict] = None,
                            c7_plus_mw_override: Optional[float] = None,
                            progress_callback: Optional[Callable[[int, int], None]] = None,
                            is_stop_requested: Optional[Callable[[], bool]] = None,
                            ) -> Dict[str, np.ndarray]:
        """
        Calculates a depth-based MMP profile using the best available data.

        The method establishes base profiles for temperature and API gravity using the
        standard data hierarchy (EOS, metadata, logs, etc.). If perforation-specific
        data is available, it then overwrites the values in the base profiles for
        the corresponding depth intervals. This ensures that high-priority perforation
        data is used for those specific zones while still providing a continuous
        profile for the entire wellbore for visualization.

        Args:
            method (str): The MMP correlation to use ('auto', 'cronquist', etc.).
            gas_composition (Optional[Dict]): Injection gas mole fractions.
            c7_plus_mw_override (Optional[float]): Manual override for C7+ MW.
            progress_callback (Optional[Callable]): Callback for progress updates.
            is_stop_requested (Optional[Callable]): Callback to check for cancellation.

        ENHANCED: Now accepts optional callbacks to report progress and check for
        cancellation requests, making it suitable for use in worker threads.
        """
        if not hasattr(self.well_data, 'depths') or self.well_data.depths.size == 0:
            raise ValueError("WellData must contain depth information to calculate a profile.")

        final_gas_comp = gas_composition or self.config.default_gas_composition
        api_profile = self._estimate_api_profile()
        temp_profile = self._calculate_temperature_profile()

        total_points = len(self.well_data.depths)
        mmp_values = np.full(total_points, np.nan)
        c7_plus_mw = c7_plus_mw_override if c7_plus_mw_override is not None else self.well_data.metadata.get('C7_plus_MW')

        # If perforation data is available, use it to overwrite the base profiles.
        # This gives perforation data the highest priority for its specific intervals.
        is_perforated_well = bool(self.well_data.perforation_properties)
        if is_perforated_well:
            logger.info("Applying perforation-specific properties to MMP profile.")
            # OPTIMIZATION: Only calculate MMP at the top and bottom of each perforation.
            indices_to_process = set()
            for perf in self.well_data.perforation_properties:
                # Apply properties to the entire perforation interval for the plot
                mask = (self.well_data.depths >= perf['top']) & (self.well_data.depths <= perf['bottom'])
                if 'api' in perf: api_profile[mask] = perf['api']
                if 'temp' in perf: temp_profile[mask] = perf['temp']

                # Find the specific indices in the depth array for the perf boundaries
                top_idx = np.abs(self.well_data.depths - perf['top']).argmin()
                bottom_idx = np.abs(self.well_data.depths - perf['bottom']).argmin()
                indices_to_process.add(top_idx)
                indices_to_process.add(bottom_idx)
            
            points_to_loop = sorted(list(indices_to_process))
            logger.info(f"Optimized calculation for {len(points_to_loop)} points at perforation boundaries.")

        else: # If not a perforated well, calculate for the entire wellbore
            points_to_loop = range(total_points)

        for i in points_to_loop:

            # Check for cancellation request inside the loop
            if is_stop_requested and is_stop_requested():
                logger.info("MMP profile calculation cancelled by user request.")
                # Return partial results
                return {
                    'depths': self.well_data.depths[:i], 'mmp': mmp_values[:i],
                    'temperature': temp_profile[:i], 'api': api_profile[:i]
                }

            params = MMPParameters(
                temperature=temp_profile[i],
                oil_gravity=api_profile[i],
                injection_gas_composition=final_gas_comp,
                c7_plus_mw=c7_plus_mw,
                pvt_data=self.pvt_data
            )
            mmp_values[i] = calculate_mmp(params, method=method)
            
            # Report progress
            if progress_callback:
                progress_callback(i + 1, total_points)

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

    def get_average_mmp_params_for_engine(self) -> Dict[str, Any]:
        """
        Calculates average fluid and thermal properties to provide a representative
        set of parameters for the OptimizationEngine.
        """
        logger.info(f"Getting average MMP parameters for well '{self.well_data.name}'.")

        if self.well_data.perforation_properties:
            logger.info("Using thickness-weighted average from defined perforations.")
            perfs = self.well_data.perforation_properties
            
            total_thickness = 0
            weighted_api_sum = 0
            weighted_temp_sum = 0

            for p in perfs:
                thickness = p['bottom'] - p['top']
                if thickness <= 0: continue
                total_thickness += thickness
                weighted_api_sum += p['api'] * thickness
                weighted_temp_sum += p['temp'] * thickness
            
            if total_thickness > 0:
                avg_params = {
                    'temperature': weighted_temp_sum / total_thickness,
                    'oil_gravity': weighted_api_sum / total_thickness,
                    'c7_plus_mw': self.well_data.metadata.get('C7_plus_MW'),
                    'injection_gas_composition': self.config.default_gas_composition
                }
                logger.info(f"Calculated perforation-weighted avg parameters: {avg_params}")
                return avg_params

        logger.warning(
            f"Well '{self.well_data.name}': No valid perforations found. Averaging properties over the entire wellbore."
        )
        # --- OPTIMIZATION: Avoid calculating the full MMP profile just for averages ---
        # Instead, get the underlying property profiles directly. This is much faster
        # and avoids spamming the log with MMP calculation messages.
        api_profile = self._estimate_api_profile()
        temp_profile = self._calculate_temperature_profile()

        if api_profile.size == 0:
            logger.error("Cannot get average MMP params; no depth data in well.")
            return {}

        avg_params = {
            'temperature': np.mean(temp_profile),
            'oil_gravity': np.mean(api_profile),
            'c7_plus_mw': self.well_data.metadata.get('C7_plus_MW'),
            'injection_gas_composition': self.config.default_gas_composition
        }
        logger.info(f"Calculated full-wellbore average parameters: {avg_params}")
        return avg_params