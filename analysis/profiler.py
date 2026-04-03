import numpy as np
from scipy.interpolate import interp1d
import logging
from typing import Optional

from core.data_models import EORParameters, OperationalParameters, ProfileParameters, ReservoirData, PVTProperties, PhysicalConstants
from core.Phys_engine_full.eos_models import ReservoirFluid

_PHYS_CONSTANTS = PhysicalConstants()
DAYS_PER_YEAR = 365
FT3_PER_BBL = 5.61458
EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
logger = logging.getLogger(__name__)

class ProductionProfiler:
    """
    Generates production and injection profiles using a 1D fractional flow model.
    This model simulates the displacement of oil by CO2 based on physical principles,
    providing a more realistic prediction of reservoir performance than a simple tank model.
    """

    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters, op_params: OperationalParameters, profile_params: ProfileParameters, initial_pressure_override: Optional[float] = None):
        """
        Initializes the profiler with the necessary reservoir and fluid data.
        """
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self.op_params = op_params
        self.profile_params = profile_params
        self.initial_pressure = initial_pressure_override if initial_pressure_override is not None else self.reservoir.initial_pressure
        self.temperature_F = self.pvt.temperature

        self.reservoir_fluid = None
        if self.reservoir.eos_model:
            try:
                self.reservoir_fluid = ReservoirFluid(self.reservoir.eos_model)
            except Exception as e:
                logger.warning(f"Failed to initialize ReservoirFluid: {e}")

        if self.reservoir.geostatistical_grid is not None:
            self.pore_volume_bbl = (self.reservoir.length_ft * 
                                    self.reservoir.cross_sectional_area_acres * 43560 * 
                                    np.mean(self.reservoir.geostatistical_grid)) / FT3_PER_BBL
        else:
            self.pore_volume_bbl = (self.reservoir.length_ft * 
                                    self.reservoir.cross_sectional_area_acres * 43560 * 
                                    np.mean(self.reservoir.grid.get('PORO', 0.2))) / FT3_PER_BBL
        
        self.mobility_ratio = self.eor_params.mobility_ratio

        # --- Enhanced Sweep Efficiency Calculation with Geology Integration ---
        self.areal_sweep_efficiency = self._calculate_geology_enhanced_sweep_efficiency()

    def _calculate_geology_enhanced_sweep_efficiency(self) -> float:
        """
        Calculate sweep efficiency enhanced with geological parameters.
        Incorporates rock type, depositional environment, and structural complexity.
        """
        M = self.mobility_ratio
        
        # Base sweep efficiency from mobility ratio
        if M <= 1.0:
            base_efficiency = 1.0
        else:
            if M <= 10:
                base_efficiency = 0.5 + 0.4 * np.log10(M) / (M - 1)
            else:
                base_efficiency = np.exp(-0.1 * (M - 10)) * (0.546 + 0.0357/M)
        
        # Apply geological modifiers
        geology_factor = self._calculate_geology_factor()
        enhanced_efficiency = base_efficiency * geology_factor
        
        return np.clip(enhanced_efficiency, 0.1, 1.0)

    def _calculate_geology_factor(self) -> float:
        """
        Calculate geology factor based on reservoir geological characteristics.
        """
        factor = 1.0
        
        # Rock type modifier
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                factor *= 1.1  # Sandstone typically has better sweep
            elif rock_type == 'carbonate':
                factor *= 0.9  # Carbonates often have more heterogeneity
            elif rock_type == 'shale':
                factor *= 0.7  # Shale has poor sweep characteristics
        
        # Depositional environment modifier
        depositional_env = getattr(self.reservoir, 'depositional_environment', None)
        if depositional_env:
            if depositional_env in ['fluvial', 'deltaic']:
                factor *= 1.05  # Channelized systems can have good sweep if properly targeted
            elif depositional_env == 'aeolian':
                factor *= 0.95  # Dune systems can have complex flow patterns
            elif depositional_env in ['deep_marine', 'shallow_marine']:
                factor *= 1.0  # Marine systems typically have moderate sweep
        
        # Structural complexity modifier
        structural_complexity = getattr(self.reservoir, 'structural_complexity', None)
        if structural_complexity:
            if structural_complexity == 'simple':
                factor *= 1.1
            elif structural_complexity == 'moderate':
                factor *= 1.0
            elif structural_complexity == 'complex':
                factor *= 0.9
            elif structural_complexity == 'very_complex':
                factor *= 0.8
        
        # Geostatistical heterogeneity modifier
        if hasattr(self.reservoir, 'geostatistical_grid') and self.reservoir.geostatistical_grid is not None:
            heterogeneity = self._calculate_heterogeneity_index()
            # Higher heterogeneity reduces sweep efficiency
            heterogeneity_penalty = 1.0 - (heterogeneity * 0.3)  # Up to 30% penalty for high heterogeneity
            factor *= np.clip(heterogeneity_penalty, 0.7, 1.0)
        
        return factor

    def _calculate_heterogeneity_index(self) -> float:
        """
        Calculate heterogeneity index from geostatistical grid.
        Higher values indicate more heterogeneous reservoir.
        """
        if self.reservoir.geostatistical_grid is None:
            return 0.0
        
        grid = self.reservoir.geostatistical_grid
        if grid.size == 0:
            return 0.0
        
        # Calculate coefficient of variation as heterogeneity measure
        std_dev = np.std(grid)
        mean_val = np.mean(grid)
        
        if mean_val > 0:
            return std_dev / mean_val
        else:
            return 0.0

    def _get_geology_based_permeability_modifier(self) -> float:
        """
        Get permeability modifier based on geological characteristics.
        """
        modifier = 1.0
        
        # Rock type permeability modifiers
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                modifier *= 1.2  # Sandstone typically has higher permeability
            elif rock_type == 'carbonate':
                modifier *= 0.8  # Carbonates can have variable permeability
            elif rock_type == 'shale':
                modifier *= 0.3  # Shale has very low permeability
        
        return modifier

    def _calculate_geology_injection_factor(self) -> float:
        """
        Calculate injection rate factor based on geological characteristics.
        """
        factor = 1.0
        
        # Rock type injection modifiers
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                factor *= 1.1  # Sandstone can typically handle higher injection rates
            elif rock_type == 'carbonate':
                factor *= 0.9  # Carbonates may have fracture concerns
            elif rock_type == 'shale':
                factor *= 0.6  # Shale has very low injectivity
        
        # Structural complexity modifier
        structural_complexity = getattr(self.reservoir, 'structural_complexity', None)
        if structural_complexity:
            if structural_complexity == 'complex':
                factor *= 0.8
            elif structural_complexity == 'very_complex':
                factor *= 0.6
        
        return np.clip(factor, 0.5, 1.5)

    def _relative_permeability(self, S_co2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates oil and CO2 relative permeabilities using the Corey-Brooks model.
        
        Args:
            S_co2: Array of CO2 saturation values (fraction).
            
        Returns:
            A tuple of (k_ro, k_rg), where k_ro is the relative permeability to oil
            and k_rg is the relative permeability to CO2.
        """
        S_or = self.eor_params.sor
        S_gc = self.eor_params.s_gc
        n_o = self.eor_params.n_o
        n_g = self.eor_params.n_g
        
        # Normalize saturation for the Corey-Brooks model
        # Add EPSILON to denominator to avoid division by zero if S_or + S_gc >= 1
        S_star = (S_co2 - S_gc) / (1 - S_or - S_gc + EPSILON)
        S_star = np.clip(S_star, 0, 1)
        
        k_ro = (1 - S_star)**n_o
        k_rg = S_star**n_g
        
        return k_ro, k_rg

    def _fractional_flow(self, S_co2: np.ndarray) -> np.ndarray:
        """
        Calculates the fractional flow of CO2 as a function of CO2 saturation.
        
        f_co2 = (1 + (k_ro / k_rg) * (mu_co2 / mu_oil))^-1
        
        Args:
            S_co2: Array of CO2 saturation values.
            
        Returns:
            Array of corresponding CO2 fractional flow values.
        """
        k_ro, k_rg = self._relative_permeability(S_co2)
        
        # Avoid division by zero for k_rg at S_co2 <= S_gc
        k_rg[k_rg < EPSILON] = EPSILON
        
        # Avoid division by zero in the fractional flow calculation
        denominator = 1 + (k_ro / k_rg) * (1 / max(self.mobility_ratio, EPSILON))
        denominator[denominator < EPSILON] = EPSILON
        
        f_co2 = 1 / denominator
        return f_co2

    def _relative_permeability_water(self, S_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S_wc = self.eor_params.s_wc
        S_orw = self.eor_params.s_orw
        n_w = self.eor_params.n_w
        n_ow = self.eor_params.n_ow

        S_wn = (S_w - S_wc) / (1 - S_wc - S_orw + EPSILON)
        S_wn = np.clip(S_wn, 0, 1)

        k_rw = S_wn**n_w
        k_row = (1 - S_wn)**n_ow
        return k_row, k_rw

    def _water_fractional_flow(self, S_w: np.ndarray, current_pressure: float) -> np.ndarray:
        """Calculates the fractional flow of water based on saturation and pressure."""
        k_row, k_rw = self._relative_permeability_water(S_w)
        mu_w = 0.5  # cP
        
        # Interpolate oil viscosity at the current pressure
        pressure_range = self.pvt.pressure_points
        mu_oil = np.interp(current_pressure, pressure_range, self.pvt.oil_viscosity)
        
        k_rw[k_rw < 1e-9] = 1e-9

        f_w = (1 + (k_row / k_rw) * (mu_w / mu_oil))**-1
        return f_w

    def _welge_tangent(self) -> tuple[float, float, float]:
        """
        Performs the Welge tangent construction to find the CO2 saturation at the
        shock front (S_wf), the average CO2 saturation behind the front (S_w_avg),
        and the slope of the fractional flow curve at the front.
        This is key to Buckley-Leverett theory.
        
        Returns:
            A tuple of (S_wf, S_w_avg, df_dS_at_front).
        """
        S_gc = self.eor_params.s_gc
        S_or = self.eor_params.sor
        S_co2_range = np.linspace(S_gc, 1 - S_or, 500)
        f_co2 = self._fractional_flow(S_co2_range)
        
        # Welge tangent construction: find the tangent to the fractional flow curve
        # that starts at (S_gc, 0). The slope of this tangent gives the shock velocity.
        # The slope is given by f_co2 / (S_co2 - S_gc)
        denominator = S_co2_range - S_gc
        # Avoid division by zero at the first point
        denominator[denominator < EPSILON] = EPSILON
        tangent_slope = f_co2 / denominator
        
        front_idx = np.argmax(tangent_slope)
        df_dS_at_front = tangent_slope[front_idx]
        S_wf = S_co2_range[front_idx]
        f_wf = f_co2[front_idx]

        if df_dS_at_front < EPSILON:
            S_w_avg = S_wf
        else:
            S_w_avg = S_wf + (1 - f_wf) / df_dS_at_front
        
        S_w_avg = np.clip(S_w_avg, 0, 1 - S_or)
        
        # Debug logging for breakthrough analysis
        logger.debug(f"Welge tangent: S_wf={S_wf:.4f}, S_w_avg={S_w_avg:.4f}, df_dS_at_front={df_dS_at_front:.6f}")
        logger.debug(f"Fractional flow at front: f_wf={f_wf:.4f}")
        logger.debug(f"Critical saturations: S_gc={S_gc:.4f}, S_or={S_or:.4f}")
        
        return S_wf, S_w_avg, df_dS_at_front

    def generate_all_profiles(self, ooip_stb: float, **kwargs) -> dict:
        """
        Generates production and injection profiles using a 1D fractional flow model
        with pressure support, productivity index, and dynamic well control for CO2 storage.
        """
        # Input validation
        if ooip_stb <= 0:
            raise ValueError(f"OOIP must be positive, got {ooip_stb}")
        if self.pore_volume_bbl <= 0:
            raise ValueError(f"Pore volume must be positive, got {self.pore_volume_bbl}")
            
        project_life_days = int(self.op_params.project_lifetime_years * DAYS_PER_YEAR)
        
        # --- Geology-Enhanced Fluid and Rock Properties ---
        c_oil = 10e-6  # 1/psi
        c_gas = 1e-3   # 1/psi
        c_water = 3e-6 # 1/psi
        B_oil = 1.2    # rb/stb
        
        B_gas = 5.0 # Default value
        if self.reservoir_fluid:
            try:
                temp_K = (self.temperature_F - 32) * 5/9 + 273.15
                pressure_Pa = self.initial_pressure * 6894.76
                B_gas = self.reservoir_fluid.get_bgas_rb_per_mscf(temp_K, pressure_Pa)
                logger.info(f"Calculated B_gas: {B_gas:.4f} rb/MSCF")
            except Exception as e:
                logger.warning(f"Could not calculate B_gas, falling back to default 5.0. Error: {e}")

        B_water = 1.0  # rb/stb
        
        # Apply geology-based modifiers
        permeability_modifier = self._get_geology_based_permeability_modifier()
        heterogeneity_index = self._calculate_heterogeneity_index()
        
        # Enhanced productivity index calculation with geology factors
        base_productivity_index = 1.0  # Default value
        enhanced_pi = base_productivity_index * permeability_modifier * (1.0 - heterogeneity_index * 0.2)
        
        # Geology-aware injection rate adjustment
        geology_injection_factor = self._calculate_geology_injection_factor()
        geology_enhanced_injection_rate = self.eor_params.injection_rate * geology_injection_factor

        # --- Initialization ---
        cumulative_oil_produced = 0.0
        recoverable_oil_stb = ooip_stb * (1 - self.eor_params.sor)
        current_pressure = self.initial_pressure
        
        daily_oil_stb = np.zeros(project_life_days)
        daily_co2_prod_mscf = np.zeros(project_life_days)
        daily_water_prod_bbl = np.zeros(project_life_days)
        daily_pressure = np.zeros(project_life_days)
        well_status = np.full(project_life_days, 'producer')  # Track well status: 'producer' or 'injector'
        daily_co2_inj = np.zeros(project_life_days)
        daily_water_inj = np.zeros(project_life_days)
        daily_hnp_cycle = np.zeros(project_life_days, dtype=int)


        # --- Pre-loop Injection Scheme Setup ---
        # Initialize injection arrays and apply injection scheme patterns
        self._setup_injection_scheme(daily_co2_inj, daily_water_inj, project_life_days,
                                   geology_enhanced_injection_rate, B_gas, daily_hnp_cycle)

        # --- Time-stepping Simulation ---
        for day in range(project_life_days):

            # 1. Check if well should be converted to injector
            should_convert = bool(round(self.eor_params.allow_well_conversion))
            if should_convert and well_status[day-1] == 'producer':
                # Conversion based on time
                if day >= self.eor_params.well_conversion_day:
                    well_status[day:] = 'injector'
                    continue

                # Conversion based on production rate
                if day > 30:
                    avg_recent_production = np.mean(daily_oil_stb[max(0, day-30):day])
                    if avg_recent_production < self.eor_params.well_shut_in_threshold_bpd:
                        well_status[day:] = 'injector'
                        continue

            # 3. Update Reservoir Pressure
            # --- Saturations and Compressibility ---
            s_gas = ((np.sum(daily_co2_inj[:day]) - np.sum(daily_co2_prod_mscf[:day])) * B_gas) / self.pore_volume_bbl if self.pore_volume_bbl > 0 else 0
            s_water = (np.sum(daily_water_inj[:day]) - np.sum(daily_water_prod_bbl[:day])) / self.pore_volume_bbl if self.pore_volume_bbl > 0 else 0
            s_gas = np.clip(s_gas, 0, 1)
            s_water = np.clip(s_water, 0, 1)
            s_oil = np.clip(1 - s_gas - s_water, 0, 1)
            
            # Ensure minimum saturations to avoid numerical instability
            s_oil = max(s_oil, 0.01)  # Minimum oil saturation
            s_gas = max(s_gas, 0.01)  # Minimum gas saturation
            s_water = max(s_water, 0.01)  # Minimum water saturation
            
            # Normalize saturations to sum to 1
            total_s = s_oil + s_gas + s_water
            if total_s > 0:
                s_oil = s_oil / total_s
                s_gas = s_gas / total_s
                s_water = s_water / total_s
            
            c_eff = c_oil * s_oil + c_gas * s_gas + c_water * s_water + self.reservoir.rock_compressibility
            Vp_ceff = self.pore_volume_bbl * c_eff
            
            # Ensure minimum effective compressibility to avoid division by zero
            Vp_ceff = max(Vp_ceff, self.pore_volume_bbl * 1e-9)

            # --- Dynamic Injection Control (Replaces old shutdown logic) ---
            inj_vol_rb = (daily_co2_inj[day] * B_gas) + (daily_water_inj[day] * B_water)

            # If the calculated next pressure step will exceed the maximum pressure...
            if Vp_ceff > EPSILON and (current_pressure + (inj_vol_rb / Vp_ceff)) > self.eor_params.max_pressure_psi:
                # ...calculate the maximum allowable injection volume to stay under the pressure limit.
                max_inj_vol = (self.eor_params.max_pressure_psi - current_pressure) * Vp_ceff
                
                # Scale back the gas and water injection proportionally
                original_total_inj = inj_vol_rb if inj_vol_rb > EPSILON else 1.0
                scaling_factor = max(0, max_inj_vol / original_total_inj)
                
                # Ensure minimum injection to prevent premature termination
                min_scaling_factor = 0.01  # 1% minimum injection to continue simulation
                scaling_factor = max(scaling_factor, min_scaling_factor)
                
                daily_co2_inj[day] *= scaling_factor
                daily_water_inj[day] *= scaling_factor
                
                # Recalculate the injection volume for the pressure update
                inj_vol_rb = (daily_co2_inj[day] * B_gas) + (daily_water_inj[day] * B_water)
                
                # Debug logging for pressure control
                if day % 100 == 0 or scaling_factor < 0.1:
                    logger.debug(f"Day {day}: Pressure control activated. Scaling factor={scaling_factor:.4f}, Current pressure={current_pressure:.1f} psi")

            # Now, update the pressure with the (potentially scaled-back) injection volume
            PI = max(0, self.eor_params.productivity_index)
            BHP = self.eor_params.wellbore_pressure

            # For Huff-n-Puff, differentiate between injection and production phases
            # During injection phases, pressure should increase; during production, it should decrease
            if self.eor_params.injection_scheme == 'huff_n_puff':
                # Enhanced Huff-n-Puff pressure dynamics to fix critical flaws
                is_injection_phase = daily_co2_inj[day] > 0
                is_production_phase = daily_co2_inj[day] == 0 and day > 0  # Not injection and not first day
                
                if is_injection_phase:
                    # Injection phase: pressure increases due to injection
                    pressure_increase = (inj_vol_rb / Vp_ceff) if Vp_ceff > EPSILON else 0
                    # Limit pressure increase to prevent excessive buildup but allow realistic values
                    pressure_increase = min(pressure_increase, 100.0)  # Increased from 50 to 100 psi/day
                    current_pressure += pressure_increase
                    logger.debug(f"Day {day}: Huff-n-Puff injection - Pressure increase: {pressure_increase:.1f} psi")
                elif is_production_phase:
                    # Production phase: pressure decreases due to production
                    # Calculate total production volume (oil + water + CO2)
                    oil_production_volume = daily_oil_stb[day] * B_oil if day < len(daily_oil_stb) else 0
                    water_production_volume = daily_water_prod_bbl[day] * B_water if day < len(daily_water_prod_bbl) else 0
                    co2_production_volume = daily_co2_prod_mscf[day] * B_gas if day < len(daily_co2_prod_mscf) else 0
                    total_production_volume = oil_production_volume + water_production_volume + co2_production_volume
                    
                    pressure_decline = (total_production_volume / Vp_ceff) if Vp_ceff > EPSILON else 0
                    # Ensure minimum pressure decline during production phases
                    pressure_decline = max(pressure_decline, 5.0)  # Minimum 5 psi/day decline during production
                    pressure_decline = min(pressure_decline, 80.0)  # Increased from 40 to 80 psi/day
                    current_pressure -= pressure_decline
                    logger.debug(f"Day {day}: Huff-n-Puff production - Pressure decline: {pressure_decline:.1f} psi")
                else:
                    # Soaking phase: minimal pressure change
                    current_pressure -= 2.0  # Small pressure decline during soaking (increased from 1.0)
                    logger.debug(f"Day {day}: Huff-n-Puff soaking - Pressure decline: 2.0 psi")
            else:
                # Standard pressure update for other schemes
                prod_potential_at_new_p = 0.0 if well_status[day] == 'injector' else PI
                numerator = current_pressure * Vp_ceff + inj_vol_rb + prod_potential_at_new_p * BHP
                denominator = Vp_ceff + prod_potential_at_new_p

                if denominator > EPSILON:
                    current_pressure = numerator / denominator
                else:
                    current_pressure += (inj_vol_rb / Vp_ceff) if Vp_ceff > EPSILON else 0
            
            # Ensure pressure stays within reasonable bounds with better minimum pressure
            min_pressure = max(1600, self.initial_pressure * 0.35)  # Increased from 1500 to 1600, from 30% to 35% of initial
            current_pressure = np.clip(current_pressure, min_pressure, self.eor_params.max_pressure_psi)
            
            if not np.isfinite(current_pressure):
                current_pressure = daily_pressure[day-1] if day > 0 else self.initial_pressure

            # 2. Calculate Production/Injection based on current well status and pressure
            if well_status[day] == 'producer':
                # --- Enhanced Production Calculation using Darcy's Law ---
                # Total fluid production potential governed by Productivity Index
                total_flow_rate_bpd = enhanced_pi * (current_pressure - self.eor_params.wellbore_pressure)
                total_flow_rate_bpd = max(0, total_flow_rate_bpd)

                # --- Ramp-up logic for initial production ---
                ramp_up_days = 90
                if day < ramp_up_days:
                    ramp_up_factor = (day + 1) / ramp_up_days
                    total_flow_rate_bpd *= ramp_up_factor
                
                # Ensure minimum production when injection is reduced due to pressure control
                # This prevents premature production termination
                min_production_factor = 0.05  # 5% of peak production
                peak_production = enhanced_pi * (self.initial_pressure - self.eor_params.wellbore_pressure)
                if day > ramp_up_days and total_flow_rate_bpd < (peak_production * min_production_factor):
                    total_flow_rate_bpd = peak_production * min_production_factor
                    if day % 100 == 0:
                        logger.debug(f"Day {day}: Applying minimum production factor. Current pressure={current_pressure:.1f}, Production={total_flow_rate_bpd:.1f}")

                # --- Calculate fractional flows based on current saturations ---
                cumulative_inj_vol_rb_at_day = (np.sum(daily_co2_inj[:day]) * B_gas) + np.sum(daily_water_inj[:day])
                PVI_current = cumulative_inj_vol_rb_at_day / self.pore_volume_bbl if self.pore_volume_bbl > 0 else 0
                S_wf, S_w_avg, df_dS_at_front = self._welge_tangent()
                PVI_bt = 1.0 / df_dS_at_front if df_dS_at_front > EPSILON else np.inf

                # Calculate CO2 fractional flow based on breakthrough physics
                f_co2_at_producer = 0.0  # Before breakthrough
                
                # Special handling for Huff-n-Puff: CO2 production occurs during production phases
                if self.eor_params.injection_scheme == 'huff_n_puff':
                    # In Huff-n-Puff, CO2 is produced during production phases after injection
                    is_production_phase = daily_co2_inj[day] == 0 and day > 0
                    if is_production_phase:
                        # Enhanced CO2 production physics to fix critical flaws
                        total_injected_co2 = np.sum(daily_co2_inj[:day])
                        total_produced_co2 = np.sum(daily_co2_prod_mscf[:day])
                        
                        # Calculate remaining CO2 in reservoir that could be produced
                        remaining_co2_in_reservoir = total_injected_co2 - total_produced_co2
                        
                        # Enhanced CO2 saturation calculation with realistic physics
                        # In Huff-n-Puff, CO2 saturation near wellbore can be very high during production
                        co2_volume_in_reservoir_rb = remaining_co2_in_reservoir * B_gas
                        pore_volume_near_well = self.pore_volume_bbl * 0.4  # 40% of pore volume accessible
                        co2_saturation_near_well = min(0.85, co2_volume_in_reservoir_rb / pore_volume_near_well)
                        co2_saturation_near_well = np.clip(co2_saturation_near_well, 0.4, 0.85)  # Increased minimum from 0.3 to 0.4
                        
                        # Calculate fractional flow based on actual CO2 saturation
                        f_co2_at_producer = self._fractional_flow(np.array([co2_saturation_near_well]))[0]
                        
                        # Enhanced cycle efficiency with realistic physics
                        # Calculate cycles completed based on injection pattern
                        cycles_completed = daily_hnp_cycle[day] if day < len(daily_hnp_cycle) else 0
                        base_efficiency = 0.5  # Increased from 0.4 to 0.5
                        improvement_rate = 0.15  # Increased from 0.12 to 0.15
                        cycle_efficiency = min(0.95, base_efficiency + cycles_completed * improvement_rate)  # Increased cap to 0.95
                        f_co2_at_producer *= cycle_efficiency
                        
                        # Ensure realistic CO2 production during production phases
                        # In Huff-n-Puff, CO2 fractional flow should be significant
                        f_co2_at_producer = max(f_co2_at_producer, 0.4)  # Increased from 0.3 to 0.4
                        f_co2_at_producer = min(f_co2_at_producer, 0.85)  # Increased cap from 0.8 to 0.85
                        
                        # Additional boost for early production phases to increase CO2 recycle ratio
                        if cycles_completed <= 2:  # First few cycles
                            f_co2_at_producer *= 1.2  # 20% boost for early cycles
                        
                        logger.debug(f"Day {day}: Huff-n-Puff CO2 production - f_co2={f_co2_at_producer:.3f}, cycle={cycles_completed}")
                
                elif PVI_current >= PVI_bt:
                    # After breakthrough, CO2 fractional flow increases based on saturation
                    pvi_past_bt = PVI_current - PVI_bt
                    # Use more physically-based model for CO2 breakthrough
                    # CO2 fractional flow should increase gradually after breakthrough
                    co2_cut_aggressiveness = 0.8  # Reduced from 1.5 to make breakthrough more gradual
                    f_co2_at_producer = np.clip(1.0 - np.exp(-co2_cut_aggressiveness * pvi_past_bt), 0, 0.6)  # Reduced max from 0.8 to 0.6
                    
                    # Also consider the CO2 saturation at the producer
                    # After breakthrough, CO2 saturation increases gradually
                    S_co2_at_producer = np.clip(S_wf + (1 - self.eor_params.sor - S_wf) * (1 - np.exp(-0.3 * pvi_past_bt)), S_wf, 1 - self.eor_params.sor)
                    # Calculate fractional flow based on actual CO2 saturation
                    f_co2_from_saturation = self._fractional_flow(np.array([S_co2_at_producer]))[0]
                    # Use the maximum of the two methods
                    f_co2_at_producer = max(f_co2_at_producer, f_co2_from_saturation)

                # Track key variables for debugging breakthrough
                if day % 100 == 0 or (PVI_current >= PVI_bt and f_co2_at_producer > 0.1):
                    logger.debug(f"Day {day}: PVI_current={PVI_current:.4f}, PVI_bt={PVI_bt:.4f}, f_co2_at_producer={f_co2_at_producer:.4f}")
                    logger.debug(f"Breakthrough status: {'YES' if PVI_current >= PVI_bt else 'NO'}, df_dS_at_front={df_dS_at_front:.6f}")

                # Calculate water fractional flow based on current water saturation
                # Calculate water saturation at producer using Buckley-Leverett theory
                # Water saturation should be based on the displacement front, not just net injection
                if PVI_current < PVI_bt:
                    # Before breakthrough, water saturation at producer is connate water saturation
                    S_w_at_producer = self.eor_params.s_wc
                else:
                    # After breakthrough, water saturation increases gradually
                    pvi_past_bt = PVI_current - PVI_bt
                    # Use more realistic water saturation increase after breakthrough
                    S_w_at_producer = np.clip(
                        self.eor_params.s_wc + (1 - self.eor_params.s_orw - self.eor_params.s_wc) *
                        (1 - np.exp(-0.5 * pvi_past_bt)),
                        self.eor_params.s_wc, 1 - self.eor_params.s_orw
                    )
                
                f_water_at_producer = self._water_fractional_flow(np.array([S_w_at_producer]), current_pressure)[0]
                
                # Debug logging for water saturation
                if day % 100 == 0 or (PVI_current >= PVI_bt and f_water_at_producer > 0.1):
                    logger.debug(f"Day {day}: PVI={PVI_current:.4f}, S_w={S_w_at_producer:.4f}, f_water={f_water_at_producer:.4f}")

                # --- Water Cut Shutdown Logic ---
                if f_water_at_producer > self.eor_params.water_cut_bwow:
                    logger.warning(f"Day {day}: Water cut ({f_water_at_producer:.2f}) exceeds limit ({self.eor_params.water_cut_bwow:.2f}). Setting production to zero.")
                    # Instead of breaking, set production to zero but continue simulation
                    daily_oil_stb[day] = 0.0
                    daily_water_prod_bbl[day] = 0.0
                    daily_co2_prod_mscf[day] = 0.0
                    # Continue to next day without breaking
                    continue

                # Calculate oil fractional flow (remaining fraction)
                f_oil_at_producer = np.clip(1.0 - f_co2_at_producer - f_water_at_producer, 0, 1)

                # --- Calculate individual fluid rates from total flow rate ---
                oil_rate_stb = total_flow_rate_bpd * f_oil_at_producer * self.areal_sweep_efficiency
                co2_rate_rb = total_flow_rate_bpd * f_co2_at_producer
                water_rate_bbl = total_flow_rate_bpd * f_water_at_producer

                daily_oil_stb[day] = oil_rate_stb
                daily_water_prod_bbl[day] = water_rate_bbl
                daily_co2_prod_mscf[day] = co2_rate_rb / B_gas if B_gas > 0 else 0
                
            else:
                daily_oil_stb[day] = 0.0
                daily_water_prod_bbl[day] = 0.0
                daily_co2_prod_mscf[day] = 0.0
            
            daily_pressure[day] = np.clip(current_pressure, 500, self.eor_params.max_pressure_psi)
            cumulative_oil_produced += daily_oil_stb[day]

        # --- Calculate Volumetric Sweep (now correctly Pore Volumes Injected) ---
        cumulative_inj_vol_rb = np.cumsum(daily_co2_inj * B_gas + daily_water_inj * B_water)
        daily_pore_volumes_injected = cumulative_inj_vol_rb / self.pore_volume_bbl if self.pore_volume_bbl > 0 else np.zeros(project_life_days)

        # --- Assemble final profiles ---
        daily_co2_recycled = daily_co2_prod_mscf * self.profile_params.co2_recycling_efficiency_fraction
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
        }
        
        resolution = self.op_params.time_resolution
        resampled_profiles = {}
        for key, daily_data in profiles.items():
            res_key = key.replace('daily', resolution)
            resampled_profiles[res_key] = self._resample_profile(daily_data, resolution, key)

        return {**profiles, **resampled_profiles}

    def _resample_profile(self, daily_data: np.ndarray, resolution: str, key: str = None) -> np.ndarray:
        """Resamples a daily profile to a coarser time resolution."""
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
            num_periods = self.op_params.project_lifetime_years
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

    def _setup_injection_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                              project_life_days: int, geology_enhanced_injection_rate: float, B_gas: float, daily_hnp_cycle: np.ndarray):
        """
        Sets up injection scheme patterns before the main simulation loop.
        This isolates injection logic from the time-stepping simulation.
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
        Implements Water-Alternating-Gas (WAG) injection scheme.
        Alternates between CO2 injection and water injection cycles.
        """
        min_cycle_days = self.eor_params.min_cycle_length_days
        max_cycle_days = self.eor_params.max_cycle_length_days
        
        # Use average cycle length if min and max are different
        if min_cycle_days != max_cycle_days:
            cycle_length_days = int((min_cycle_days + max_cycle_days) // 2)
        else:
            cycle_length_days = int(min_cycle_days)
        
        # Calculate cycle parameters - each full cycle includes both CO2 and water phases
        co2_phase_days = cycle_length_days
        water_phase_days = cycle_length_days
        
        # Implement WAG cycling - alternate between CO2 and water injection
        current_day = 0
        is_co2_phase = True  # Start with CO2 injection
        
        while current_day < project_life_days:
            if is_co2_phase:
                # CO2 injection phase
                phase_end_day = min(current_day + co2_phase_days, project_life_days)
                days_in_phase = phase_end_day - current_day
                if days_in_phase > 0:
                    daily_co2_inj[current_day:phase_end_day] = co2_inj_rate_mscfd
                    daily_water_inj[current_day:phase_end_day] = 0.0
                    current_day = phase_end_day
            else:
                # Water injection phase
                phase_end_day = min(current_day + water_phase_days, project_life_days)
                days_in_phase = phase_end_day - current_day
                if days_in_phase > 0:
                    daily_co2_inj[current_day:phase_end_day] = 0.0
                    daily_water_inj[current_day:phase_end_day] = water_inj_rate_bpd
                    current_day = phase_end_day
            
            is_co2_phase = not is_co2_phase  # Alternate phases
        
        logger.debug(f"WAG scheme implemented: {current_day} days populated out of {project_life_days} total days")
        logger.debug(f"WAG cycles: CO2 phase={co2_phase_days} days, Water phase={water_phase_days} days")
        
        # Debug: Check the actual injection patterns
        co2_injection_days = np.sum(daily_co2_inj > 0)
        water_injection_days = np.sum(daily_water_inj > 0)
        logger.debug(f"WAG injection pattern: CO2 days={co2_injection_days}, Water days={water_injection_days}")

    def _implement_huff_n_puff_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                                    co2_inj_rate_mscfd: float, project_life_days: int, daily_hnp_cycle: np.ndarray):
        """
        Implements Huff-n-Puff (cyclic) injection scheme.
        Alternates between injection, soaking, and production periods.
        Continues cycling throughout the project lifetime.
        """
        inject_days = self.eor_params.huff_n_puff_injection_period_days
        soak_days = self.eor_params.huff_n_puff_soaking_period_days
        prod_days = self.eor_params.huff_n_puff_production_period_days
        max_cycles = self.eor_params.huff_n_puff_max_cycles
        
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
        """
        initial_multiplier = self.eor_params.tapered_initial_rate_multiplier
        final_multiplier = self.eor_params.tapered_final_rate_multiplier
        taper_years = self.eor_params.tapered_duration_years
        taper_function = self.eor_params.tapered_function
        
        taper_days = int(taper_years * DAYS_PER_YEAR)
        
        # Calculate tapered rates
        for day in range(min(taper_days, project_life_days)):
            progress = day / max(1, taper_days - 1)
            
            if taper_function == "linear":
                multiplier = initial_multiplier - progress * (initial_multiplier - final_multiplier)
            elif taper_function == "exponential":
                exponent = -progress * 3  # Exponential decay factor
                multiplier = initial_multiplier * np.exp(exponent)
                multiplier = max(final_multiplier, multiplier)
            else:  # logarithmic
                multiplier = initial_multiplier - np.log1p(progress * 10) * (initial_multiplier - final_multiplier) / np.log1p(10)
                multiplier = max(final_multiplier, multiplier)
            
            daily_co2_inj[day] = co2_inj_rate_mscfd * multiplier
        
        # Fill remaining days with final rate
        if taper_days < project_life_days:
            daily_co2_inj[taper_days:] = co2_inj_rate_mscfd * final_multiplier

    def _implement_pulsed_scheme(self, daily_co2_inj: np.ndarray, daily_water_inj: np.ndarray,
                               co2_inj_rate_mscfd: float, project_life_days: int):
        """
        Implements pulsed injection scheme with intermittent high-intensity pulses.
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