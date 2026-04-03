"""
Refactored Production Profiler for CO2 EOR Simulation
Uses modular components for better organization and maintainability.
"""

import numpy as np
import logging
from typing import Optional

from core.data_models import EORParameters, OperationalParameters, ProfileParameters, ReservoirData, PVTProperties
from core.simulation.profile_generator import ProfileGenerator

logger = logging.getLogger(__name__)


class ProductionProfiler:
    """
    Generates production and injection profiles using a 1D fractional flow model.
    This refactored version uses modular components for better organization.
    """

    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters, 
                 op_params: OperationalParameters, profile_params: ProfileParameters, 
                 initial_pressure_override: Optional[float] = None):
        """
        Initializes the profiler with the necessary reservoir and fluid data.
        
        Args:
            reservoir: Reservoir data including geological properties
            pvt: PVT properties for fluids
            eor_params: Enhanced oil recovery parameters
            op_params: Operational parameters
            profile_params: Profile generation parameters
            initial_pressure_override: Optional override for initial pressure
        """
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self.op_params = op_params
        self.profile_params = profile_params
        self.initial_pressure_override = initial_pressure_override
        
        # Create default CCUS parameters since they are required by ProfileGenerator
        from core.data_models import CCUSParameters
        ccus_params = CCUSParameters()

        # Initialize the profile generator
        self.profile_generator = ProfileGenerator(
            reservoir, pvt, eor_params, op_params, profile_params, ccus_params, initial_pressure_override=initial_pressure_override
        )

    def generate_all_profiles(self, ooip_stb: float, **kwargs) -> dict:
        """
        Generates production and injection profiles using modular components.
        
        Args:
            ooip_stb: Original oil in place (STB)
            **kwargs: Additional arguments passed to the profile generator
            
        Returns:
            Dictionary containing all production and injection profiles
        """
        logger.info(f"Generating production profiles for OOIP: {ooip_stb:.0f} STB")
        
        try:
            profiles = self.profile_generator.generate_all_profiles(ooip_stb, **kwargs)
            logger.info("Production profiles generated successfully")
            return profiles
        except Exception as e:
            logger.error(f"Error generating production profiles: {str(e)}")
            raise

    @property
    def areal_sweep_efficiency(self) -> float:
        """Delegates areal sweep efficiency to the inner profile generator."""
        if hasattr(self, 'profile_generator') and hasattr(self.profile_generator, 'areal_sweep_efficiency'):
            return self.profile_generator.areal_sweep_efficiency
        return 0.0 # Default or fallback

    # Backward compatibility methods
    def _calculate_geology_enhanced_sweep_efficiency(self) -> float:
        """Backward compatibility method - now handled by GeologyEngine"""
        from core.geology import GeologyEngine
        geology_engine = GeologyEngine(self.reservoir, self.eor_params)
        return geology_engine.calculate_geology_enhanced_sweep_efficiency()

    def _calculate_geology_factor(self) -> float:
        """Backward compatibility method - now handled by GeologyEngine"""
        from core.geology import GeologyEngine
        geology_engine = GeologyEngine(self.reservoir, self.eor_params)
        return geology_engine._calculate_geology_factor()

    def _calculate_heterogeneity_index(self) -> float:
        """Backward compatibility method - now handled by GeologyEngine"""
        from core.geology import GeologyEngine
        geology_engine = GeologyEngine(self.reservoir, self.eor_params)
        return geology_engine.calculate_heterogeneity_index()

    def _get_geology_based_permeability_modifier(self) -> float:
        """Backward compatibility method - now handled by GeologyEngine"""
        from core.geology import GeologyEngine
        geology_engine = GeologyEngine(self.reservoir, self.eor_params)
        return geology_engine.get_geology_based_permeability_modifier()

    def _calculate_geology_injection_factor(self) -> float:
        """Backward compatibility method - now handled by GeologyEngine"""
        from core.geology import GeologyEngine
        geology_engine = GeologyEngine(self.reservoir, self.eor_params)
        return geology_engine.calculate_geology_injection_factor()

    def _relative_permeability(self, S_co2: np.ndarray) -> tuple:
        """Backward compatibility method - now handled by PhysicsEngine"""
        from core.physics import PhysicsEngine
        physics_engine = PhysicsEngine(self.eor_params)
        return physics_engine.relative_permeability(S_co2)

    def _fractional_flow(self, S_co2: np.ndarray) -> np.ndarray:
        """Backward compatibility method - now handled by PhysicsEngine"""
        from core.physics import PhysicsEngine
        physics_engine = PhysicsEngine(self.eor_params)
        return physics_engine.fractional_flow(S_co2)

    def _relative_permeability_water(self, S_w: np.ndarray) -> tuple:
        """Backward compatibility method - now handled by PhysicsEngine"""
        from core.physics import PhysicsEngine
        physics_engine = PhysicsEngine(self.eor_params)
        return physics_engine.relative_permeability_water(S_w)

    def _water_fractional_flow(self, S_w: np.ndarray, current_pressure: float) -> np.ndarray:
        """Backward compatibility method - now handled by PhysicsEngine"""
        from core.physics import PhysicsEngine
        physics_engine = PhysicsEngine(self.eor_params)
        return physics_engine.water_fractional_flow(S_w, current_pressure, self.pvt)

    def _welge_tangent(self) -> tuple:
        """Backward compatibility method - now handled by PhysicsEngine"""
        from core.physics import PhysicsEngine
        physics_engine = PhysicsEngine(self.eor_params)
        return physics_engine.welge_tangent()

    def _setup_injection_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                              project_life_days: int, geology_enhanced_injection_rate: float, 
                              B_gas: float, daily_hnp_cycle: np.ndarray):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes.setup_injection_scheme(
            daily_co2_inj, daily_water_inj, project_life_days,
            geology_enhanced_injection_rate, B_gas, daily_hnp_cycle
        )

    def _implement_wag_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                            co2_inj_rate_mscfd: float, water_inj_rate_bpd: float,
                            project_life_days: int):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes._implement_wag_scheme(
            daily_co2_inj, daily_water_inj, co2_inj_rate_mscfd, water_inj_rate_bpd, project_life_days
        )

    def _implement_huff_n_puff_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                                    co2_inj_rate_mscfd: float, project_life_days: int, daily_hnp_cycle: np.ndarray):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes._implement_huff_n_puff_scheme(
            daily_co2_inj, daily_water_inj, co2_inj_rate_mscfd, project_life_days, daily_hnp_cycle
        )

    def _implement_swag_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                             co2_inj_rate_mscfd: float, water_inj_rate_bpd: float,
                             project_life_days: int):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes._implement_swag_scheme(
            daily_co2_inj, daily_water_inj, co2_inj_rate_mscfd, water_inj_rate_bpd, project_life_days
        )

    def _implement_tapered_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                                co2_inj_rate_mscfd: float, project_life_days: int):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes._implement_tapered_scheme(
            daily_co2_inj, daily_water_inj, co2_inj_rate_mscfd, project_life_days
        )

    def _implement_pulsed_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                               co2_inj_rate_mscfd: float, project_life_days: int):
        """Backward compatibility method - now handled by InjectionSchemes"""
        from core.simulation.injection_schemes import InjectionSchemes
        injection_schemes = InjectionSchemes(self.eor_params)
        injection_schemes._implement_pulsed_scheme(
            daily_co2_inj, daily_water_inj, co2_inj_rate_mscfd, project_life_days
        )

    def _resample_profile(self, daily_data: np.ndarray, resolution: str, key: str = None) -> np.ndarray:
        """Backward compatibility method - now handled by ProfilerUtils"""
        from core.utils.profiler_utils import ProfilerUtils
        return ProfilerUtils.resample_profile(
            daily_data, resolution, key, self.op_params.project_lifetime_years
        )


# Factory function for backward compatibility
def create_production_profiler(reservoir: ReservoirData, pvt: PVTProperties, 
                              eor_params: EORParameters, op_params: OperationalParameters,
                              profile_params: ProfileParameters, 
                              initial_pressure_override: Optional[float] = None) -> ProductionProfiler:
    """
    Factory function to create a ProductionProfiler instance.
    
    Args:
        reservoir: Reservoir data
        pvt: PVT properties
        eor_params: EOR parameters
        op_params: Operational parameters
        profile_params: Profile parameters
        initial_pressure_override: Optional initial pressure override
        
    Returns:
        ProductionProfiler instance
    """
    return ProductionProfiler(
        reservoir, pvt, eor_params, op_params, profile_params, initial_pressure_override
    )