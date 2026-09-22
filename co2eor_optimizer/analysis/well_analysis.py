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
        Calculates a depth-based MMP profile. If perforations are defined, the
        calculation is performed for the perforated intervals to generate a
        continuous profile connecting them. Otherwise, a profile for the entire
        wellbore is generated.
        """
        final_gas_comp = gas_composition or self.config.default_gas_composition
        c7_plus_mw = c7_plus_mw_override if c7_plus_mw_override is not None else self.well_data.metadata.get('C7_plus_MW')
        is_perforated_well = bool(self.well_data.perforation_properties)

        if is_perforated_well:
            logger.info("Perforated well detected. Calculating connected MMP profile at perforation boundaries.")
            depth_points, api_points, temp_points, mmp_points = [], [], [], []
            perfs = self.well_data.perforation_properties
            num_perfs = len(perfs)

            for i, perf in enumerate(perfs):
                if is_stop_requested and is_stop_requested():
                    logger.info("MMP profile calculation cancelled by user request.")
                    return {'depths': np.array(depth_points), 'mmp': np.array(mmp_points),
                            'temperature': np.array(temp_points), 'api': np.array(api_points)}

                top, bottom, api, temp = perf.get('top'), perf.get('bottom'), perf.get('api'), perf.get('temp')
                if any(v is None for v in [top, bottom, api, temp]):
                    logger.warning(f"Skipping a perforation due to missing properties (top, bottom, api, or temp).")
                    continue
                try:
                    params = MMPParameters(
                        temperature=temp, oil_gravity=api, injection_gas_composition=final_gas_comp,
                        c7_plus_mw=c7_plus_mw, pvt_data=self.pvt_data)
                    mmp_val = calculate_mmp(params, method=method)
                    
                    depth_points.extend([top, bottom])
                    api_points.extend([api, api])
                    temp_points.extend([temp, temp])
                    mmp_points.extend([mmp_val, mmp_val])
                except Exception as e:
                    logger.error(f"Failed to calculate MMP for perforation {top}-{bottom}: {e}")
                    continue
                
                if progress_callback: progress_callback(i + 1, num_perfs)
            
            return {'depths': np.array(depth_points), 'mmp': np.array(mmp_points),
                    'temperature': np.array(temp_points), 'api': np.array(api_points)}

        else:  # Non-perforated well logic
            if not hasattr(self.well_data, 'depths') or self.well_data.depths.size == 0:
                raise ValueError("WellData must contain depth information to calculate a continuous profile.")
            
            total_points = len(self.well_data.depths)
            logger.info("Non-perforated well. Calculating continuous MMP profile.")
            api_profile = self._estimate_api_profile()
            temp_profile = self._calculate_temperature_profile()
            mmp_values = np.full(total_points, np.nan)
            
            props = np.stack((temp_profile, api_profile), axis=-1)
            nan_mask = np.isnan(props)
            nan_changes = np.any(np.diff(nan_mask, axis=0), axis=1)
            both_valid_mask = ~nan_mask[:-1] & ~nan_mask[1:]
            value_diff = np.diff(props, axis=0)
            value_changes = np.any((value_diff != 0) & both_valid_mask, axis=1)
            prop_changes = np.concatenate(([True], nan_changes | value_changes))
            indices_to_calculate = np.where(prop_changes)[0]
            
            logger.info(f"Optimized MMP calculation for {len(indices_to_calculate)} unique property blocks.")
            for i, current_idx in enumerate(indices_to_calculate):
                if is_stop_requested and is_stop_requested():
                    logger.info("MMP profile calculation cancelled by user request.")
                    mmp_values[current_idx:] = np.nan
                    return {'depths': self.well_data.depths, 'mmp': mmp_values, 'temperature': temp_profile, 'api': api_profile}
                
                end_idx = indices_to_calculate[i+1] if (i + 1) < len(indices_to_calculate) else total_points
                if np.isnan(temp_profile[current_idx]) or np.isnan(api_profile[current_idx]):
                    if progress_callback: progress_callback(end_idx, total_points)
                    continue

                params = MMPParameters(
                    temperature=temp_profile[current_idx], oil_gravity=api_profile[current_idx],
                    injection_gas_composition=final_gas_comp, c7_plus_mw=c7_plus_mw, pvt_data=self.pvt_data)
                mmp_values[current_idx:end_idx] = calculate_mmp(params, method=method)
                if progress_callback: progress_callback(end_idx, total_points)
                
            return {'depths': self.well_data.depths, 'mmp': mmp_values, 'temperature': temp_profile, 'api': api_profile}


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

        If perforations are defined, it calculates a thickness-weighted average
        of properties from the perforated intervals. Otherwise, it averages
        properties over the entire wellbore.
        """
        logger.info(f"Getting average MMP parameters for well '{self.well_data.name}'.")

        if self.well_data.perforation_properties:
            logger.info("Using thickness-weighted average from defined perforations.")
            perfs = self.well_data.perforation_properties
            
            total_thickness = 0
            weighted_api_sum = 0
            weighted_temp_sum = 0

            for p in perfs:
                thickness = p.get('bottom', 0) - p.get('top', 0)
                if thickness <= 0: continue
                total_thickness += thickness
                weighted_api_sum += p.get('api', self.config.default_api_gravity) * thickness
                weighted_temp_sum += p.get('temp', self.config.default_reservoir_temp) * thickness
            
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

        if api_profile.size == 0 or temp_profile.size == 0:
            logger.error("Cannot get average MMP params; no depth data in well.")
            return {}

        avg_params = {
            'temperature': np.mean(temp_profile[~np.isnan(temp_profile)]),
            'oil_gravity': np.mean(api_profile[~np.isnan(api_profile)]),
            'c7_plus_mw': self.well_data.metadata.get('C7_plus_MW'),
            'injection_gas_composition': self.config.default_gas_composition
        }
        logger.info(f"Calculated full-wellbore average parameters: {avg_params}")
        return avg_params