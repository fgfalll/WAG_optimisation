"""
Utilities Module for CO2 EOR Simulation
Handles helper functions, constants, and data processing utilities.
"""

import numpy as np
from typing import Dict, Any

# Constants - use PhysicalConstants for consistency
try:
    from core.data_models import PhysicalConstants
    _PHYS_CONSTANTS = PhysicalConstants()
    EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
except ImportError:
    EPSILON = 1e-9  # Fallback for standalone usage

DAYS_PER_YEAR = 365
FT3_PER_BBL = 5.61458


class ProfilerUtils:
    """
    Utilities for CO2 EOR simulation.
    Handles helper functions, constants, and data processing utilities.
    """

    # Constants
    DAYS_PER_YEAR = 365
    FT3_PER_BBL = 5.61458
    EPSILON = EPSILON
    
    @staticmethod
    def resample_profile(daily_data: np.ndarray, resolution: str, key: str = None, 
                        project_lifetime_years: int = None) -> np.ndarray:
        """
        Resamples a daily profile to a coarser time resolution.
        
        Args:
            daily_data: Daily data array
            resolution: Target resolution ("yearly", "quarterly", "monthly", "weekly", "daily")
            key: Data key for determining aggregation method
            project_lifetime_years: Project lifetime in years (required for yearly resolution)
            
        Returns:
            Resampled data array
        """
        if resolution == "daily":
            return daily_data
            
        days_in_period = {
            "yearly": DAYS_PER_YEAR,
            "quarterly": DAYS_PER_YEAR / 4.0,
            "monthly": DAYS_PER_YEAR / 12.0,
            "weekly": 7
        }
        
        period_days = days_in_period.get(resolution, DAYS_PER_YEAR)
        if resolution == "yearly":
            if project_lifetime_years is None:
                raise ValueError("project_lifetime_years is required for yearly resolution")
            num_periods = project_lifetime_years
        else:
            num_periods = int(np.floor(len(daily_data) / period_days))
        
        # Determine if this is pressure data (should be averaged) or other data (should be summed)
        # Pressure and state variables should be averaged, while rates/volumes should be summed
        is_pressure_data = key and 'pressure' in key.lower()
        is_state_variable = key and any(var in key.lower() for var in ['saturation', 'temperature', 'ratio', 'efficiency'])
        
        if is_pressure_data or is_state_variable:
            # For pressure and state variables, use average instead of sum
            resampled = np.array([
                np.mean(daily_data[int(i*period_days) : int((i+1)*period_days)])
                for i in range(num_periods)
            ])
        else:
            # For production/injection data (rates, volumes), use sum
            resampled = np.array([
                np.sum(daily_data[int(i*period_days) : int((i+1)*period_days)])
                for i in range(num_periods)
            ])
        
        return resampled
    
    @staticmethod
    def calculate_pore_volume(reservoir) -> float:
        """
        Calculate pore volume from reservoir data.
        
        Args:
            reservoir: ReservoirData object
            
        Returns:
            Pore volume in barrels
        """
        if reservoir.geostatistical_grid is not None:
            pore_volume_bbl = (reservoir.length_ft * 
                              reservoir.cross_sectional_area_acres * 43560 * 
                              np.mean(reservoir.geostatistical_grid)) / FT3_PER_BBL
        else:
            pore_volume_bbl = (reservoir.length_ft * 
                              reservoir.cross_sectional_area_acres * 43560 * 
                              np.mean(reservoir.grid.get('PORO', 0.2))) / FT3_PER_BBL
        
        return pore_volume_bbl
    
    @staticmethod
    def normalize_saturations(s_oil: float, s_gas: float, s_water: float) -> tuple:
        """
        Normalize saturations to sum to 1 with minimum values.

        FIX: S_gc is a FLOW threshold (mobility=0), NOT a STATE threshold.
        Saturation can be exactly 0.0 before a phase arrives - only epsilon-level
        minimum for numerical stability is needed.

        Args:
            s_oil: Oil saturation
            s_gas: Gas saturation
            s_water: Water saturation

        Returns:
            Tuple of normalized (s_oil, s_gas, s_water)
        """
        # Use epsilon-level minimum for numerical stability only
        # S_gc is a FLOW threshold (mobility=0), NOT a STATE threshold
        # Use ULTRA epsilon for saturation-specific numerical stability
        from core.data_models import PhysicalConstants
        _PHYS_CONSTANTS = PhysicalConstants()
        SATURATION_EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_ULTRA
        s_oil = max(s_oil, SATURATION_EPSILON)  # Allow oil to be effectively zero
        s_gas = max(s_gas, SATURATION_EPSILON)  # Allow gas to be effectively zero (FIX: no 1% floor)
        s_water = max(s_water, SATURATION_EPSILON)  # Allow water to be effectively zero
        
        # Normalize saturations to sum to 1
        total_s = s_oil + s_gas + s_water
        if total_s > 0:
            s_oil = s_oil / total_s
            s_gas = s_gas / total_s
            s_water = s_water / total_s
        
        return s_oil, s_gas, s_water
    
    @staticmethod
    def calculate_enhanced_productivity_index(base_pi: float, permeability_modifier: float, 
                                            heterogeneity_index: float) -> float:
        """
        Calculate enhanced productivity index with geology factors.
        
        Args:
            base_pi: Base productivity index
            permeability_modifier: Permeability modifier from geology
            heterogeneity_index: Heterogeneity index
            
        Returns:
            Enhanced productivity index
        """
        return base_pi * permeability_modifier * (1.0 - heterogeneity_index * 0.2)
    
    @staticmethod
    def calculate_huff_n_puff_co2_production(total_injected_co2: float, total_produced_co2: float,
                                           pore_volume_bbl: float, B_gas: float, 
                                           cycles_completed: int) -> tuple:
        """
        Calculate CO2 production parameters for Huff-n-Puff scheme.
        
        Args:
            total_injected_co2: Total injected CO2 (MSCF)
            total_produced_co2: Total produced CO2 (MSCF)
            pore_volume_bbl: Pore volume in barrels
            B_gas: Gas formation volume factor
            cycles_completed: Number of completed cycles
            
        Returns:
            Tuple of (co2_saturation_near_well, f_co2_at_producer)
        """
        # Calculate remaining CO2 in reservoir that could be produced
        remaining_co2_in_reservoir = total_injected_co2 - total_produced_co2
        
        # Enhanced CO2 saturation calculation with realistic physics
        # In Huff-n-Puff, CO2 saturation near wellbore can be very high during production
        co2_volume_in_reservoir_rb = remaining_co2_in_reservoir * B_gas
        pore_volume_near_well = pore_volume_bbl * 0.4  # 40% of pore volume accessible
        co2_saturation_near_well = min(0.85, co2_volume_in_reservoir_rb / pore_volume_near_well)
        co2_saturation_near_well = np.clip(co2_saturation_near_well, 0.4, 0.85)  # Increased minimum from 0.3 to 0.4
        
        # Enhanced cycle efficiency with realistic physics
        base_efficiency = 0.5  # Increased from 0.4 to 0.5
        improvement_rate = 0.15  # Increased from 0.12 to 0.15
        cycle_efficiency = min(0.95, base_efficiency + cycles_completed * improvement_rate)  # Increased cap to 0.95
        
        # Additional boost for early production phases to increase CO2 recycle ratio
        early_cycle_boost = 1.2 if cycles_completed <= 2 else 1.0  # 20% boost for early cycles
        
        return co2_saturation_near_well, cycle_efficiency, early_cycle_boost
    
    @staticmethod
    def assemble_final_profiles(daily_oil_stb: np.ndarray, daily_co2_inj: np.ndarray,
                              daily_water_inj: np.ndarray, daily_water_prod_bbl: np.ndarray,
                              daily_co2_prod_mscf: np.ndarray, daily_pressure: np.ndarray,
                              daily_pore_volumes_injected: np.ndarray,
                              co2_recycling_efficiency_fraction: float,
                              daily_injector_bhp: np.ndarray = None,
                              daily_producer_bhp: np.ndarray = None,
                              daily_injector_porosity: np.ndarray = None,
                              daily_injector_permeability: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Assemble final production profiles with CO2 recycling calculations.
        
        Args:
            daily_oil_stb: Daily oil production (STB)
            daily_co2_inj: Daily CO2 injection (MSCF)
            daily_water_inj: Daily water injection (bbl)
            daily_water_prod_bbl: Daily water production (bbl)
            daily_co2_prod_mscf: Daily CO2 production (MSCF)
            daily_pressure: Daily reservoir pressure (psi)
            daily_pore_volumes_injected: Daily pore volumes injected
            co2_recycling_efficiency_fraction: CO2 recycling efficiency
            daily_injector_bhp: Daily injector bottom-hole pressure (psi)
            daily_producer_bhp: Daily producer bottom-hole pressure (psi)
            daily_injector_porosity: Daily injector porosity
            daily_injector_permeability: Daily injector permeability
            
        Returns:
            Dictionary of assembled profiles
        """
        daily_co2_recycled = daily_co2_prod_mscf * co2_recycling_efficiency_fraction
        daily_co2_purchased = np.maximum(0, daily_co2_inj - daily_co2_recycled)
        
        profiles = {
            'daily_oil_stb': daily_oil_stb,
            'daily_co2_injected_mscf': daily_co2_inj,
            'daily_water_injected_bbl': daily_water_inj,
            'daily_co2_purchased_mscf': daily_co2_purchased,
            'daily_co2_recycled_mscf': daily_co2_recycled,
            'daily_water_produced_bbl': daily_water_prod_bbl,
            'daily_co2_produced_mscf': daily_co2_prod_mscf,
            'daily_pressure': daily_pressure,
            'daily_pore_volumes_injected': daily_pore_volumes_injected,
            'yearly_oil_stb': ProfilerUtils.resample_profile(daily_oil_stb, 'yearly', 'oil_stb', int(len(daily_oil_stb) / DAYS_PER_YEAR)),
            'yearly_co2_injected_mscf': ProfilerUtils.resample_profile(daily_co2_inj, 'yearly', 'co2_injected', int(len(daily_co2_inj) / DAYS_PER_YEAR)),
            'yearly_water_injected_bbl': ProfilerUtils.resample_profile(daily_water_inj, 'yearly', 'water_injected', int(len(daily_water_inj) / DAYS_PER_YEAR)),
            'yearly_water_produced_bbl': ProfilerUtils.resample_profile(daily_water_prod_bbl, 'yearly', 'water_produced', int(len(daily_water_prod_bbl) / DAYS_PER_YEAR)),
            'yearly_co2_produced_mscf': ProfilerUtils.resample_profile(daily_co2_prod_mscf, 'yearly', 'co2_produced', int(len(daily_co2_prod_mscf) / DAYS_PER_YEAR)),
            'yearly_co2_purchased_mscf': ProfilerUtils.resample_profile(daily_co2_purchased, 'yearly', 'co2_purchased', int(len(daily_co2_purchased) / DAYS_PER_YEAR)),
            'yearly_co2_recycled_mscf': ProfilerUtils.resample_profile(daily_co2_recycled, 'yearly', 'co2_recycled', int(len(daily_co2_recycled) / DAYS_PER_YEAR)),
        }

        if daily_injector_bhp is not None:
            profiles['daily_injector_bhp'] = daily_injector_bhp
        if daily_producer_bhp is not None:
            profiles['daily_producer_bhp'] = daily_producer_bhp
        if daily_injector_porosity is not None:
            profiles['daily_injector_porosity'] = daily_injector_porosity
        if daily_injector_permeability is not None:
            profiles['daily_injector_permeability'] = daily_injector_permeability
        
        return profiles