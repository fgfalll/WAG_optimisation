"""
Pressure Dynamics Module for CO2 EOR Simulation
Handles reservoir pressure calculations, compressibility, and pressure control.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Numerical constants - use PhysicalConstants for consistency
try:
    from core.data_models import PhysicalConstants
    _PHYS_CONSTANTS = PhysicalConstants()
    EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
except ImportError:
    EPSILON = 1e-9  # Fallback for standalone usage

# Physical constants for unit conversion
try:
    from core.data_models import PhysicalConstants
    _PHYS_CONSTANTS = PhysicalConstants()
except ImportError:
    # Fallback if data_models not available
    class _FallbackConstants:
        PSI_TO_PA = 6894.76
        DEFAULT_POROSITY = 0.20
        DEFAULT_PERMEABILITY_MD = 100.0
    _PHYS_CONSTANTS = _FallbackConstants()


class PressureDynamics:
    """
    Pressure dynamics engine for CO2 EOR simulation.
    Handles reservoir pressure calculations, compressibility, and pressure control.
    """
    
    def __init__(self, reservoir, eor_params, pvt_props, pore_volume_bbl, initial_pressure):
        """
        Initialize pressure dynamics engine.
        
        Args:
            reservoir: ReservoirData object
            eor_params: EORParameters object
            pvt_props: PVTProperties object
            pore_volume_bbl: Pore volume in barrels
            initial_pressure: Initial reservoir pressure (psi)
        """
        self.reservoir = reservoir
        self.eor_params = eor_params
        self.pvt_props = pvt_props
        self.pore_volume_bbl = pore_volume_bbl
        self.initial_pressure = initial_pressure

    @staticmethod
    def validate_pressure_units(pressure_value: float) -> float:
        """
        Validate and convert pressure to PSI if needed.

        If pressure > 1e7, assume it's in Pascals and convert to PSI.
        This handles the common error of passing Pascal values where PSI is expected.

        Args:
            pressure_value: Pressure value to validate

        Returns:
            Pressure in PSI (converted if necessary)
        """
        if pressure_value > 1e7:  # Likely Pascals
            converted = pressure_value / _PHYS_CONSTANTS.PSI_TO_PA
            logger.warning(
                f"Pressure {pressure_value:.1f} appears to be in Pascals. "
                f"Converting to PSI: {converted:.1f} psi"
            )
            return converted
        return pressure_value

    def validate_pressure_bounds(self, pressure: np.ndarray) -> np.ndarray:
        """
        Ensure pressures are within physically realistic bounds.

        Realistic reservoir pressures typically range from:
        - Minimum: ~14.7 psi (atmospheric) to ~500 psi (shallow)
        - Maximum: ~15,000-20,000 psi (very deep/high pressure)

        Args:
            pressure: Array of pressure values (psi)

        Returns:
            Pressure array clipped to realistic bounds
        """
        MIN_REALISTIC_PRESSURE = 14.7  # atmospheric pressure in psi
        MAX_REALISTIC_PRESSURE = 20000.0  # ~30,000 ft depth equivalent

        original_min = np.min(pressure)
        original_max = np.max(pressure)

        validated = np.clip(pressure, MIN_REALISTIC_PRESSURE, MAX_REALISTIC_PRESSURE)

        # Log warnings if values were clipped
        if original_min < MIN_REALISTIC_PRESSURE:
            logger.warning(
                f"Pressure values below minimum {MIN_REALISTIC_PRESSURE} psi detected. "
                f"Min: {original_min:.1f} psi. Clipping to minimum."
            )
        if original_max > MAX_REALISTIC_PRESSURE:
            logger.warning(
                f"Pressure values above maximum {MAX_REALISTIC_PRESSURE} psi detected. "
                f"Max: {original_max:.1f} psi. Clipping to maximum."
            )

        return validated

    def validate_initial_pressure(self, initial_pressure: float) -> float:
        """
        Validate initial pressure for simulation startup.

        Args:
            initial_pressure: Initial pressure value to validate

        Returns:
            Validated initial pressure in PSI
        """
        # Convert from Pascal if needed
        pressure_psi = self.validate_pressure_units(initial_pressure)

        # Create single-element array for bounds checking
        pressure_array = np.array([pressure_psi])
        validated_array = self.validate_pressure_bounds(pressure_array)

        return validated_array[0]

    def calculate_effective_compressibility(self, s_oil: float, s_gas: float, s_water: float) -> float:
        """
        Calculate effective compressibility based on current saturations.
        
        Args:
            s_oil: Oil saturation (fraction)
            s_gas: Gas saturation (fraction)
            s_water: Water saturation (fraction)
            
        Returns:
            Effective compressibility (1/psi)
        """
        c_eff = (self.pvt_props.oil_compressibility * s_oil + 
                self.pvt_props.gas_compressibility * s_gas + 
                self.pvt_props.water_compressibility * s_water + 
                self.reservoir.rock_compressibility)
        
        return c_eff
    
    def calculate_pressure_change(self, current_pressure: float, inj_vol_rb: float, 
                                total_production_volume: float, Vp_ceff: float,
                                is_injection_phase: bool, is_production_phase: bool,
                                injection_scheme: str) -> float:
        """
        Calculate pressure change based on injection/production volumes and current state.
        
        Args:
            current_pressure: Current reservoir pressure (psi)
            inj_vol_rb: Injection volume in reservoir barrels
            total_production_volume: Total production volume in reservoir barrels
            Vp_ceff: Pore volume times effective compressibility
            is_injection_phase: Whether current phase is injection
            is_production_phase: Whether current phase is production
            injection_scheme: Current injection scheme
            
        Returns:
            Pressure change (psi)
        """
        if injection_scheme == 'huff_n_puff':
            return self._calculate_huff_n_puff_pressure_change(
                current_pressure, inj_vol_rb, total_production_volume, Vp_ceff,
                is_injection_phase, is_production_phase
            )
        else:
            return self._calculate_standard_pressure_change(
                current_pressure, inj_vol_rb, Vp_ceff
            )
    
    def _calculate_huff_n_puff_pressure_change(self, current_pressure: float, inj_vol_rb: float,
                                             total_production_volume: float, Vp_ceff: float,
                                             is_injection_phase: bool, is_production_phase: bool) -> float:
        """
        Calculate pressure change for Huff-n-Puff scheme.
        
        Args:
            current_pressure: Current reservoir pressure (psi)
            inj_vol_rb: Injection volume in reservoir barrels
            total_production_volume: Total production volume in reservoir barrels
            Vp_ceff: Pore volume times effective compressibility
            is_injection_phase: Whether current phase is injection
            is_production_phase: Whether current phase is production
            
        Returns:
            Pressure change (psi)
        """
        if is_injection_phase:
            # Injection phase: pressure increases due to injection
            pressure_increase = (inj_vol_rb / Vp_ceff) if Vp_ceff > EPSILON else 0
            # Limit pressure increase to prevent excessive buildup but allow realistic values
            pressure_increase = min(pressure_increase, self.eor_params.huff_n_puff.max_pressure_increase_psi_day)  # Increased from 50 to 100 psi/day
            logger.debug(f"Huff-n-Puff injection - Pressure increase: {pressure_increase:.1f} psi")
            return pressure_increase

        elif is_production_phase:
            # Production phase: pressure decreases due to production
            pressure_decline = (total_production_volume / Vp_ceff) if Vp_ceff > EPSILON else 0
            # Ensure minimum pressure decline during production phases
            pressure_decline = max(pressure_decline, self.eor_params.huff_n_puff.min_pressure_decline_psi_day)  # Minimum 5 psi/day decline during production
            pressure_decline = min(pressure_decline, self.eor_params.huff_n_puff.max_pressure_decline_psi_day)  # Increased from 40 to 80 psi/day
            logger.debug(f"Huff-n-Puff production - Pressure decline: {pressure_decline:.1f} psi")
            return -pressure_decline

        else:
            # Soaking phase: minimal pressure change
            logger.debug(f"Huff-n-Puff soaking - Pressure decline: {self.eor_params.huff_n_puff.soaking_pressure_decline_psi_day} psi")
            return -self.eor_params.huff_n_puff.soaking_pressure_decline_psi_day  # Small pressure decline during soaking (increased from 1.0)
    
    def _calculate_standard_pressure_change(self, current_pressure: float, inj_vol_rb: float,
                                          Vp_ceff: float) -> float:
        """
        Calculate pressure change for standard injection schemes.
        
        Args:
            current_pressure: Current reservoir pressure (psi)
            inj_vol_rb: Injection volume in reservoir barrels
            Vp_ceff: Pore volume times effective compressibility
            
        Returns:
            Pressure change (psi)
        """
        if Vp_ceff > EPSILON:
            return inj_vol_rb / Vp_ceff
        else:
            return 0.0
    
    def apply_bhp_control(self, current_pressure: float, inj_vol_rb: float, Vp_ceff: float,
                         daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray, day: int) -> float:
        """
        Apply BHP-based well control for physics-based injection rate calculation.

        Uses productivity index (PI) equation: Q = PI * (BHP_target - P_reservoir)
        This preserves mass conservation by letting physics determine the flow rate
        rather than arbitrarily scaling injection.

        Args:
            current_pressure: Current reservoir pressure (psi)
            inj_vol_rb: Injection volume in reservoir barrels
            Vp_ceff: Pore volume times effective compressibility
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            day: Current day index

        Returns:
            Adjusted injection volume in reservoir barrels
        """
        # Use BHP control if enabled
        if not self.eor_params.use_bhp_control:
            # Fall back to simplified pressure limit check (no arbitrary scaling)
            if Vp_ceff > EPSILON and (current_pressure + (inj_vol_rb / Vp_ceff)) > self.eor_params.max_pressure_psi:
                # Calculate maximum allowable injection volume to stay under pressure limit
                max_inj_vol = (self.eor_params.max_pressure_psi - current_pressure) * Vp_ceff
                # Apply linear reduction without minimum scaling factor
                if max_inj_vol < 0:
                    return 0.0
                return min(inj_vol_rb, max_inj_vol)
            return inj_vol_rb

        # BHP-based control: Calculate actual injection based on pressure difference
        injector_bhp_target = self.eor_params.injector_target_bhp_psi
        pressure_drawdown = injector_bhp_target - current_pressure

        # If reservoir pressure is at or above target BHP, injection stops naturally
        if pressure_drawdown <= 0:
            logger.debug(f"Day {day}: BHP control - Reservoir pressure ({current_pressure:.1f} psi) >= "
                       f"target BHP ({injector_bhp_target:.1f} psi). Injection stopped.")
            return 0.0

        # Calculate productivity index-based injection rate
        # Q = PI * (BHP - P_reservoir)
        # Using productivity_index from EORParameters as PI
        productivity_index = self.eor_params.productivity_index
        bhp_controlled_rate = productivity_index * pressure_drawdown

        # Get total injection rate (CO2 + water)
        total_target_rate = daily_co2_inj[day] + daily_water_inj[day]

        if total_target_rate > EPSILON:
            # Scale injection proportionally to BHP-limited rate
            scaling_factor = min(1.0, bhp_controlled_rate / total_target_rate)

            daily_co2_inj[day] *= scaling_factor
            daily_water_inj[day] *= scaling_factor

            # Recalculate adjusted injection volume
            adjusted_inj_vol_rb = (daily_co2_inj[day] * self._get_gas_fvf()) + (daily_water_inj[day] * 1.0)

            if day % 100 == 0 or scaling_factor < 0.9:
                logger.debug(f"Day {day}: BHP control - Pressure drawdown={pressure_drawdown:.1f} psi, "
                           f"Scaling factor={scaling_factor:.4f}, Current pressure={current_pressure:.1f} psi")

            return adjusted_inj_vol_rb
        else:
            return inj_vol_rb

    def apply_pressure_control(self, current_pressure: float, inj_vol_rb: float, Vp_ceff: float,
                             daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray, day: int) -> float:
        """
        Legacy method for backward compatibility. Uses BHP-based control.

        Args:
            current_pressure: Current reservoir pressure (psi)
            inj_vol_rb: Injection volume in reservoir barrels
            Vp_ceff: Pore volume times effective compressibility
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            day: Current day index

        Returns:
            Adjusted injection volume in reservoir barrels
        """
        return self.apply_bhp_control(current_pressure, inj_vol_rb, Vp_ceff,
                                   daily_co2_inj, daily_water_inj, day)
    
    def _get_gas_fvf(self) -> float:
        """
        Get gas formation volume factor.
        
        Returns:
            Gas FVF (rb/MSCF)
        """
        return self.pvt_props.gas_fvf_rb_per_mscf
    
    def get_physical_minimum_pressure(self) -> float:
        """
        Get physical minimum pressure for the reservoir.

        Returns the atmospheric pressure as the physical lower bound, avoiding artificial
        constraints that prevent natural pressure depletion.

        Returns:
            Minimum pressure (psi) - atmospheric pressure (14.7 psi)
        """
        # Physical minimum: atmospheric pressure (14.7 psi at sea level)
        # This is the true physical lower bound - below this, fluids would vaporize
        return 14.7

    def get_fracture_pressure(self, reservoir_depth_ft: Optional[float] = None) -> float:
        """
        Calculate fracture pressure based on reservoir depth.

        Uses typical overburden gradient of 1.0 psi/ft to estimate fracture pressure.

        Args:
            reservoir_depth_ft: Reservoir depth in feet. If None, estimates from initial pressure.

        Returns:
            Fracture pressure (psi)
        """
        # Estimate depth from initial pressure using typical gradient (0.45 psi/ft)
        if reservoir_depth_ft is None:
            reservoir_depth_ft = self.initial_pressure / 0.45

        # Fracture pressure ~ overburden stress = depth * overburden gradient
        # Typical overburden gradient: 1.0 psi/ft
        fracture_pressure = reservoir_depth_ft * 1.0

        return fracture_pressure

    def clamp_pressure(self, pressure: float) -> float:
        """
        Clamp pressure to physical bounds only.

        Removes artificial floors and allows natural pressure depletion based on
        material balance. Only enforces true physical constraints.

        Args:
            pressure: Pressure to clamp

        Returns:
            Clamped pressure (psi)
        """
        # Use only physical constraints
        min_pressure = self.get_physical_minimum_pressure()
        max_pressure = self.eor_params.max_pressure_psi

        # Log if pressure is being clipped at physical bounds
        original_pressure = pressure
        clamped_pressure = np.clip(pressure, min_pressure, max_pressure)

        if original_pressure < min_pressure:
            logger.warning(f"Pressure ({original_pressure:.1f} psi) below atmospheric ({min_pressure:.1f} psi). "
                         f"Clamping to atmospheric pressure.")
        elif original_pressure > max_pressure:
            logger.warning(f"Pressure ({original_pressure:.1f} psi) exceeds maximum ({max_pressure:.1f} psi). "
                         f"Clamping to maximum pressure.")

        return clamped_pressure