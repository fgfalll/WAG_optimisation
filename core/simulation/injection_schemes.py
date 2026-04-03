"""
Injection Schemes Module for CO2 EOR Simulation
Handles different injection patterns: WAG, Huff-n-Puff, SWAG, tapered, and pulsed.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Configurable constants - can be overridden by parameters
DAYS_PER_YEAR = 365


class InjectionSchemes:
    """
    Injection schemes engine for CO2 EOR simulation.
    Handles different injection patterns: WAG, Huff-n-Puff, SWAG, tapered, and pulsed.
    """
    
    def __init__(self, eor_params):
        """
        Initialize injection schemes engine with EOR parameters.

        Args:
            eor_params: EORParameters object containing injection scheme parameters
        """
        self.eor_params = eor_params

        # Make days per year configurable
        self.days_per_year = getattr(eor_params, 'days_per_year', DAYS_PER_YEAR)
    
    def setup_injection_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                             project_life_days: int, geology_enhanced_injection_rate: float, 
                             B_gas: float, daily_hnp_cycle: np.ndarray):
        """
        Sets up injection scheme patterns before the main simulation loop.
        This isolates injection logic from the time-stepping simulation.
        
        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            project_life_days: Total project lifetime in days
            geology_enhanced_injection_rate: Geology-enhanced injection rate
            B_gas: Gas formation volume factor
            daily_hnp_cycle: Array for Huff-n-Puff cycle tracking
        """
        # --- Injection rates (Corrected Units) with Geology Enhancement ---
        gas_inj_rate_mscfd = geology_enhanced_injection_rate

        # Determine injection scheme based on configuration
        scheme = self.eor_params.injection_scheme
        is_wag = scheme == 'wag' and self.eor_params.WAG_ratio > 0
        is_continuous = scheme == 'continuous'
        is_huff_n_puff = scheme == 'huff_n_puff'
        is_swag = scheme == 'swag'
        is_tapered = scheme == 'tapered'
        is_pulsed = scheme == 'pulsed'
        
        # Volumetric calculations
        co2_inj_mscf_per_day = gas_inj_rate_mscfd
        co2_inj_rb_per_day = co2_inj_mscf_per_day * B_gas

        if is_wag or is_swag:
            # WAG ratio is water volume / gas volume at reservoir conditions
            # Convert CO2 injection rate from MSCF/day to reservoir barrels/day
            # Then apply WAG ratio to get water injection rate in barrels/day
            water_inj_rate_bpd = co2_inj_rb_per_day * self.eor_params.WAG_ratio
            logger.debug(f"WAG injection rates: CO2={co2_inj_mscf_per_day:.1f} MSCF/day, Water={water_inj_rate_bpd:.1f} bbl/day, WAG ratio={self.eor_params.WAG_ratio:.2f}")
        else:
            water_inj_rate_bpd = 0.0
            
        # Apply injection scheme
        if is_wag:
            self._implement_wag_scheme(daily_co2_inj, daily_water_inj, co2_inj_mscf_per_day,
                                     water_inj_rate_bpd, project_life_days)
        elif is_huff_n_puff:
            self._implement_huff_n_puff_scheme(daily_co2_inj, daily_water_inj, co2_inj_mscf_per_day,
                                             project_life_days, daily_hnp_cycle)
        elif is_swag:
            self._implement_swag_scheme(daily_co2_inj, daily_water_inj, co2_inj_mscf_per_day,
                                       water_inj_rate_bpd, project_life_days)
        elif is_tapered:
            self._implement_tapered_scheme(daily_co2_inj, daily_water_inj, co2_inj_mscf_per_day,
                                         project_life_days)
        elif is_pulsed:
            self._implement_pulsed_scheme(daily_co2_inj, daily_water_inj, co2_inj_mscf_per_day,
                                         project_life_days)
        else: # Continuous
            daily_co2_inj.fill(co2_inj_mscf_per_day)
            daily_water_inj.fill(water_inj_rate_bpd)
    
    def _implement_wag_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                            co2_inj_rate_mscfd: float, water_inj_rate_bpd: float,
                            project_life_days: int):
        """
        Implements Enhanced Water-Alternating-Gas (WAG) injection scheme with optimized mobility control.
        Addresses early breakthrough and poor sweep efficiency issues identified in field analysis.

        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            co2_inj_rate_mscfd: CO2 injection rate in MSCF/day
            water_inj_rate_bpd: Water injection rate in bbl/day
            project_life_days: Total project lifetime in days
        """
        # Enhanced WAG parameters based on expert analysis recommendations
        # Use shorter cycles initially to establish mobility control, then longer cycles
        min_cycle_length = getattr(self.eor_params, 'min_wag_cycle_length', 30)
        min_cycle_days = max(min_cycle_length, self.eor_params.min_cycle_length_days)
        max_cycle_days = self.eor_params.max_cycle_length_days

        # Optimized cycle strategy for improved sweep efficiency
        initial_cycles = getattr(self.eor_params, 'initial_wag_cycles', 3)  # Short initial cycles for mobility control
        initial_cycle_length = getattr(self.eor_params, 'initial_wag_cycle_length', 45)  # days per phase initially
        standard_cycle_length = getattr(self.eor_params, 'standard_wag_cycle_length', 90)  # days per phase for main production

        # Calculate WAG ratio based on mobility control requirements
        # Higher water ratio improves sweep but reduces CO2 storage
        mobility_factor = getattr(self.eor_params, 'mobility_ratio_factor', 0.001)
        mobility_ratio = co2_inj_rate_mscfd * mobility_factor  # Approximate mobility ratio

        high_mobility_threshold = getattr(self.eor_params, 'high_mobility_threshold', 2.0)
        max_wag_ratio = getattr(self.eor_params, 'max_enhanced_wag_ratio', 2.0)
        wag_ratio_enhancement = getattr(self.eor_params, 'wag_ratio_enhancement_factor', 1.5)

        if mobility_ratio > high_mobility_threshold:
            # High mobility ratio needs more water for control
            enhanced_wag_ratio = min(max_wag_ratio, self.eor_params.WAG_ratio * wag_ratio_enhancement)
        else:
            enhanced_wag_ratio = self.eor_params.WAG_ratio

        # Recalculate water injection rate with enhanced WAG ratio
        default_b_gas = getattr(self.eor_params, 'default_gas_fvf', 0.005)
        B_gas = default_b_gas  # Approximate gas formation volume factor
        co2_inj_rb_per_day = co2_inj_rate_mscfd * B_gas
        enhanced_water_rate_bpd = co2_inj_rb_per_day * enhanced_wag_ratio

        logger.info(f"Enhanced WAG parameters:")
        logger.info(f"  CO2 rate: {co2_inj_rate_mscfd:.1f} MSCF/day")
        logger.info(f"  Water rate: {enhanced_water_rate_bpd:.1f} bbl/day")
        logger.info(f"  WAG ratio: {enhanced_wag_ratio:.2f} (enhanced from {self.eor_params.WAG_ratio:.2f})")
        logger.info(f"  Initial cycles: {initial_cycles} @ {initial_cycle_length} days each")
        logger.info(f"  Standard cycles: {standard_cycle_length} days each")

        # Implement enhanced WAG cycling strategy
        current_day = 0
        cycle_count = 0
        is_co2_phase = True  # Start with CO2 injection

        while current_day < project_life_days:
            cycle_count += 1

            # Determine cycle length based on cycle number
            if cycle_count <= initial_cycles * 2:  # Each WAG cycle has 2 phases
                cycle_length_days = initial_cycle_length
            else:
                cycle_length_days = standard_cycle_length

            if is_co2_phase:
                # CO2 injection phase - gradually taper rate to improve front stability
                phase_end_day = min(current_day + cycle_length_days, project_life_days)
                days_in_phase = phase_end_day - current_day

                if days_in_phase > 0:
                    # Configurable tapering for better front stability
                    taper_percentage = getattr(self.eor_params, 'co2_taper_percentage', 0.1)  # 10% default
                    min_taper_factor = getattr(self.eor_params, 'min_co2_taper_factor', 0.85)

                    for day_offset in range(int(days_in_phase)):
                        day_idx = current_day + day_offset
                        taper_factor = 1.0 - (taper_percentage * day_offset / days_in_phase)
                        tapered_rate = co2_inj_rate_mscfd * max(min_taper_factor, taper_factor)
                        daily_co2_inj[day_idx] = tapered_rate
                        daily_water_inj[day_idx] = 0.0

                    current_day = phase_end_day
                    logger.debug(f"CO2 phase #{(cycle_count+1)//2}: days {current_day-cycle_length_days}-{current_day-1}")

            else:
                # Water injection phase - maintain constant rate for mobility control
                phase_end_day = min(current_day + cycle_length_days, project_life_days)
                days_in_phase = phase_end_day - current_day

                if days_in_phase > 0:
                    daily_co2_inj[current_day:phase_end_day] = 0.0
                    daily_water_inj[current_day:phase_end_day] = enhanced_water_rate_bpd
                    current_day = phase_end_day
                    logger.debug(f"Water phase #{cycle_count//2}: days {current_day-cycle_length_days}-{current_day-1}")

            is_co2_phase = not is_co2_phase  # Alternate phases

        # Final summary statistics
        co2_injection_days = np.sum(daily_co2_inj > 0)
        water_injection_days = np.sum(daily_water_inj > 0)
        total_co2_injected = np.sum(daily_co2_inj)
        total_water_injected = np.sum(daily_water_inj)

        logger.info(f"Enhanced WAG scheme completed:")
        logger.info(f"  Total cycles: {cycle_count // 2}")
        logger.info(f"  CO2 injection days: {co2_injection_days}")
        logger.info(f"  Water injection days: {water_injection_days}")
        logger.info(f"  Total CO2: {total_co2_injected:.0f} MSCF")
        logger.info(f"  Total water: {total_water_injected:.0f} bbl")
        logger.info(f"  Average CO2 rate: {total_co2_injected/co2_injection_days:.1f} MSCF/day")
        logger.info(f"  Average water rate: {total_water_injected/water_injection_days:.1f} bbl/day")
    
    def _implement_huff_n_puff_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                                    co2_inj_rate_mscfd: float, project_life_days: int, daily_hnp_cycle: np.ndarray):
        """
        Implements Huff-n-Puff (cyclic) injection scheme.
        Alternates between injection, soaking, and production periods.
        Continues cycling throughout the project lifetime.
        
        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            co2_inj_rate_mscfd: CO2 injection rate in MSCF/day
            project_life_days: Total project lifetime in days
            daily_hnp_cycle: Array for Huff-n-Puff cycle tracking
        """
        # Use Huff-n-Puff parameter object if available, otherwise fall back to individual parameters
        if self.eor_params.huff_n_puff:
            hnp_params = self.eor_params.huff_n_puff
            inject_days = hnp_params.injection_period_days
            soak_days = hnp_params.soaking_period_days
            prod_days = hnp_params.production_period_days
            max_cycles = hnp_params.max_cycles
        else:
            # Fallback to individual parameters (for backward compatibility)
            inject_days = getattr(self.eor_params, 'huff_n_puff_injection_period_days', 30)
            soak_days = getattr(self.eor_params, 'huff_n_puff_soaking_period_days', 7)
            prod_days = getattr(self.eor_params, 'huff_n_puff_production_period_days', 60)
            max_cycles = getattr(self.eor_params, 'huff_n_puff_max_cycles', 10)
        
        current_day = 0
        cycle_count = 0
        
        # Continue cycling until project end or max cycles reached
        while current_day < project_life_days and cycle_count < max_cycles:
            # Injection phase
            inject_end = min(current_day + inject_days, project_life_days)
            daily_co2_inj[current_day:inject_end] = co2_inj_rate_mscfd
            daily_hnp_cycle[current_day:inject_end] = cycle_count
            current_day = inject_end
            
            # Soaking phase (no injection)
            if current_day < project_life_days and soak_days > 0:
                soak_end = min(current_day + soak_days, project_life_days)
                daily_co2_inj[current_day:soak_end] = 0.0
                daily_hnp_cycle[current_day:soak_end] = cycle_count
                current_day = soak_end
            
            # Production phase (no injection)
            if current_day < project_life_days and prod_days > 0:
                prod_end = min(current_day + prod_days, project_life_days)
                daily_co2_inj[current_day:prod_end] = 0.0
                daily_hnp_cycle[current_day:prod_end] = cycle_count
                current_day = prod_end
                cycle_count += 1
        
        # If we haven't reached max cycles, continue cycling
        if cycle_count < max_cycles and current_day < project_life_days:
            # Calculate remaining cycles
            remaining_cycles = max_cycles - cycle_count
            cycle_length = inject_days + soak_days + prod_days
            
            for cycle in range(remaining_cycles):
                if current_day >= project_life_days:
                    break
                    
                # Injection phase
                inject_end = min(current_day + inject_days, project_life_days)
                daily_co2_inj[current_day:inject_end] = co2_inj_rate_mscfd
                daily_hnp_cycle[current_day:inject_end] = cycle_count
                current_day = inject_end
                
                # Soaking phase
                if current_day < project_life_days and soak_days > 0:
                    soak_end = min(current_day + soak_days, project_life_days)
                    daily_co2_inj[current_day:soak_end] = 0.0
                    daily_hnp_cycle[current_day:soak_end] = cycle_count
                    current_day = soak_end
                
                # Production phase
                if current_day < project_life_days and prod_days > 0:
                    prod_end = min(current_day + prod_days, project_life_days)
                    daily_co2_inj[current_day:prod_end] = 0.0
                    daily_hnp_cycle[current_day:prod_end] = cycle_count
                    current_day = prod_end
        
        logger.debug(f"Huff-n-Puff scheme implemented: {cycle_count} cycles completed, {current_day} days populated out of {project_life_days} total days")
        logger.debug(f"Huff-n-Puff cycle structure: Inject={inject_days}d, Soak={soak_days}d, Prod={prod_days}d")
    
    def _implement_swag_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                             co2_inj_rate_mscfd: float, water_inj_rate_bpd: float,
                             project_life_days: int):
        """
        Implements Simultaneous Water and Gas (SWAG) injection scheme.
        Injects both CO2 and water simultaneously throughout the project.
        
        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            co2_inj_rate_mscfd: CO2 injection rate in MSCF/day
            water_inj_rate_bpd: Water injection rate in bbl/day
            project_life_days: Total project lifetime in days
        """
        # Continuous simultaneous injection
        daily_co2_inj.fill(co2_inj_rate_mscfd)
        daily_water_inj.fill(water_inj_rate_bpd)
        
        # Apply mixing efficiency factor if needed
        mixing_efficiency = self.eor_params.swag_mixing_efficiency
        if mixing_efficiency < 1.0:
            daily_co2_inj *= mixing_efficiency
            daily_water_inj *= mixing_efficiency
    
    def _implement_tapered_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                                co2_inj_rate_mscfd: float, project_life_days: int):
        """
        Implements tapered injection scheme with gradually decreasing rates.
        
        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            co2_inj_rate_mscfd: CO2 injection rate in MSCF/day
            project_life_days: Total project lifetime in days
        """
        initial_multiplier = self.eor_params.tapered_initial_rate_multiplier
        final_multiplier = self.eor_params.tapered_final_rate_multiplier
        taper_years = self.eor_params.tapered_duration_years
        taper_function = self.eor_params.tapered_function
        
        taper_days = int(taper_years * self.days_per_year)
        
        # Calculate tapered rates
        for day in range(min(taper_days, project_life_days)):
            progress = day / max(1, taper_days - 1)
            
            if taper_function == "linear":
                multiplier = initial_multiplier - progress * (initial_multiplier - final_multiplier)
            elif taper_function == "exponential":
                exponential_decay_factor = getattr(self.eor_params, 'exponential_decay_factor', 3.0)
                exponent = -progress * exponential_decay_factor  # Configurable exponential decay factor
                multiplier = initial_multiplier * np.exp(exponent)
                multiplier = max(final_multiplier, multiplier)
            else:  # logarithmic
                log_progress_factor = getattr(self.eor_params, 'log_progress_factor', 10)
                log_denominator = np.log1p(log_progress_factor)
                # Prevent division by zero
                if log_denominator > 0:
                    multiplier = initial_multiplier - np.log1p(progress * log_progress_factor) * (initial_multiplier - final_multiplier) / log_denominator
                else:
                    multiplier = final_multiplier  # Default to final multiplier if denominator is zero
                multiplier = max(final_multiplier, multiplier)
            
            daily_co2_inj[day] = co2_inj_rate_mscfd * multiplier
        
        # Fill remaining days with final rate
        if taper_days < project_life_days:
            daily_co2_inj[taper_days:] = co2_inj_rate_mscfd * final_multiplier
    
    def _implement_pulsed_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                               co2_inj_rate_mscfd: float, project_life_days: int):
        """
        Implements pulsed injection scheme with intermittent high-intensity pulses.
        
        Args:
            daily_co2_inj: Array for daily CO2 injection rates
            daily_water_inj: Array for daily water injection rates
            co2_inj_rate_mscfd: CO2 injection rate in MSCF/day
            project_life_days: Total project lifetime in days
        """
        pulse_days = self.eor_params.pulsed_pulse_duration_days
        pause_days = self.eor_params.pulsed_pause_duration_days
        intensity_multiplier = self.eor_params.pulsed_intensity_multiplier
        
        cycle_days = pulse_days + pause_days
        total_cycles = max(1, project_life_days // cycle_days)
        
        current_day = 0
        cycle_count = 0
        
        while current_day < project_life_days and cycle_count < total_cycles:
            # Pulse phase - high intensity injection
            pulse_end = min(current_day + pulse_days, project_life_days)
            daily_co2_inj[current_day:pulse_end] = co2_inj_rate_mscfd * intensity_multiplier
            current_day = pulse_end
            
            # Pause phase - no injection
            if current_day < project_life_days:
                pause_end = min(current_day + pause_days, project_life_days)
                daily_co2_inj[current_day:pause_end] = 0.0
                current_day = pause_end
                cycle_count += 1
        
        # Fill remaining days with continuous injection
        if current_day < project_life_days:
            daily_co2_inj[current_day:] = co2_inj_rate_mscfd