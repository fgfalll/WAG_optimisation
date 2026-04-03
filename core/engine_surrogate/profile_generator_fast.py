"""
Fast Production Profile Generator for Surrogate Engine
====================================================

Generates production profiles using parameterized shapes
instead of numerical simulation for ultra-fast evaluation.

Performance: O(n) where n is the number of time steps (~100-365 points)
Target: < 0.1ms for full profile generation
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-10


class FastProfileGenerator:
    """
    Fast production profile generator using parameterized shapes.

    Uses plateau + decline models instead of numerical simulation
    for ultra-fast profile generation.
    """

    def __init__(self, model_type: str = "plateau_decline"):
        """
        Initialize the profile generator.

        Args:
            model_type: Type of profile model
                - "plateau_decline": Plateau followed by exponential decline
                - "arps": Arps decline curves (exponential, harmonic, hyperbolic)
                - "logistic": Logistic growth model
        """
        self.model_type = model_type

    def generate_profile(
        self,
        ooip: float,
        recovery_factor: float,
        injection_rate: float,
        project_lifetime: int = 15,
        **params
    ) -> Dict[str, np.ndarray]:
        """
        Generate production profiles.

        Args:
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor (0-1)
            injection_rate: CO2 injection rate (MSCFD)
            project_lifetime: Project lifetime (years)
            **params: Additional parameters for specific models

        Returns:
            Dictionary with:
                - oil_profile: Oil production rate (STB/day)
                - water_profile: Water production rate (STB/day)
                - gas_profile: Gas production rate (MSCFD)
                - injection_profile: CO2 injection rate (MSCFD)
                - time_vector: Time points (days)
        """
        # Time discretization
        n_years = project_lifetime
        n_points = int(n_years * 12)  # Monthly output
        time_vector = np.linspace(0, n_years * 365.25, n_points)

        # Select profile model
        if self.model_type == "plateau_decline":
            oil_profile = self._plateau_decline_profile(
                time_vector, ooip, recovery_factor, **params
            )
        elif self.model_type == "arps":
            oil_profile = self._arps_decline_profile(
                time_vector, ooip, recovery_factor, **params
            )
        elif self.model_type == "logistic":
            oil_profile = self._logistic_profile(
                time_vector, ooip, recovery_factor, **params
            )
        else:
            # Default to plateau decline
            oil_profile = self._plateau_decline_profile(
                time_vector, ooip, recovery_factor, **params
            )

        # Generate injection profile FIRST
        injection_profile = self._generate_injection_profile(
            time_vector, injection_rate, **params
        )

        # Update params for dependent profiles
        params['injection_profile'] = injection_profile
        params['ooip'] = ooip
        params['recovery_factor'] = recovery_factor

        # Generate water and gas profiles based on oil profile
        water_profile = self._generate_water_profile(oil_profile, time_vector, **params)
        gas_profile = self._generate_gas_profile(oil_profile, time_vector, **params)

        return {
            "oil_profile": oil_profile,
            "water_profile": water_profile,
            "gas_profile": gas_profile,
            "injection_profile": injection_profile,
            "time_vector": time_vector,
        }

    def _plateau_decline_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        plateau_fraction: float = 0.3,
        decline_rate: float = 0.15,
        ramp_up_fraction: float = 0.1,
        **params
    ) -> np.ndarray:
        """
        Generate plateau + exponential decline profile.

        Args:
            time_vector: Time points (days)
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor
            plateau_fraction: Fraction of time at plateau (0-1)
            decline_rate: Annual decline rate (fraction)
            ramp_up_fraction: Fraction of time for ramp-up (0-1)
        """
        n_points = len(time_vector)
        total_years = time_vector[-1] / 365.25

        # Calculate cumulative production target
        ultimate_recovery = ooip * recovery_factor

        # Time periods
        ramp_up_days = total_years * 365.25 * ramp_up_fraction
        plateau_days = total_years * 365.25 * plateau_fraction
        decline_start = ramp_up_days + plateau_days

        # Estimate peak rate from material balance
        # Simplified: Q_peak * (plateau_duration + 1/decline_rate) = ultimate_recovery
        effective_plateau_years = plateau_fraction * total_years
        peak_rate_estimate = ultimate_recovery / (
            effective_plateau_years + 1.0 / decline_rate
        ) / 365.25

        # Generate profile
        oil_profile = np.zeros(n_points)

        for i, t in enumerate(time_vector):
            if t < ramp_up_days:
                # Ramp-up phase: linear increase
                oil_profile[i] = peak_rate_estimate * (t / ramp_up_days)
            elif t < decline_start:
                # Plateau phase: constant rate
                oil_profile[i] = peak_rate_estimate
            else:
                # Decline phase: exponential decline
                decline_time_years = (t - decline_start) / 365.25
                oil_profile[i] = peak_rate_estimate * np.exp(-decline_rate * decline_time_years)

        # Scale to match ultimate recovery
        # oil_profile is STB/day, dt is days → sum * dt = STB (cumulative)
        cumulative_production = np.sum(oil_profile) * (time_vector[1] - time_vector[0])
        scale_factor = ultimate_recovery / cumulative_production if cumulative_production > 0 else 1.0

        return np.maximum(oil_profile * scale_factor, 0)

    def _arps_decline_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        decline_type: str = "hyperbolic",
        initial_decline: float = 0.2,
        b_factor: float = 0.5,
        qi: Optional[float] = None,
        **params
    ) -> np.ndarray:
        """
        Generate Arps decline curve profile.

        Args:
            time_vector: Time points (days)
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor
            decline_type: "exponential", "harmonic", or "hyperbolic"
            initial_decline: Initial annual decline rate (fraction)
            b_factor: Hyperbolic exponent (0=b for exponential, 1=harmonic)
            qi: Initial rate (STB/day), auto-calculated if None
        """
        n_points = len(time_vector)
        time_years = time_vector / 365.25

        # Calculate ultimate recovery
        ultimate_recovery = ooip * recovery_factor

        # Auto-calculate initial rate if not provided
        if qi is None:
            # Estimate from material balance
            # For exponential: qi/D = cumulative
            # For hyperbolic: qi^(1-b) / (D * (1-b)) = cumulative
            if decline_type == "exponential":
                qi = ultimate_recovery * initial_decline / 365.25
            elif decline_type == "harmonic":
                qi = ultimate_recovery * initial_decline / 365.25
            else:  # hyperbolic
                if abs(1 - b_factor) < 0.01:
                    qi = ultimate_recovery * initial_decline / 365.25
                else:
                    qi = (ultimate_recovery * initial_decline * (1 - b_factor) / 365.25) ** (1 / (1 - b_factor))

        # Generate decline profile
        oil_profile = np.zeros(n_points)

        for i, t in enumerate(time_years):
            if t <= 0:
                oil_profile[i] = qi
            elif decline_type == "exponential":
                oil_profile[i] = qi * np.exp(-initial_decline * t)
            elif decline_type == "harmonic":
                oil_profile[i] = qi / (1 + initial_decline * t)
            else:  # hyperbolic
                oil_profile[i] = qi / (1 + b_factor * initial_decline * t) ** (1 / b_factor)

        return np.maximum(oil_profile, 0)

    def _logistic_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        growth_rate: float = 0.5,
        midpoint: float = 0.3,
        **params
    ) -> np.ndarray:
        """
        Generate logistic growth profile (S-curve).

        Args:
            time_vector: Time points (days)
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor
            growth_rate: Logistic growth rate
            midpoint: Midpoint of growth (fraction of total time)
        """
        n_points = len(time_vector)
        time_years = time_vector / 365.25
        total_years = time_years[-1]

        # Calculate ultimate recovery
        ultimate_recovery = ooip * recovery_factor

        # Normalize time to [0, 1]
        t_normalized = time_years / total_years

        # Logistic cumulative production: L / (1 + exp(-k*(t - t0)))
        cumulative_fraction = 1.0 / (1.0 + np.exp(-growth_rate * (t_normalized - midpoint)))

        # Differentiate to get rate
        dt = time_years[1] - time_years[0]
        cumulative_profile = ultimate_recovery * cumulative_fraction
        oil_profile = np.gradient(cumulative_profile, dt)

        return np.maximum(oil_profile, 0)

    def _generate_water_profile(
        self,
        oil_profile: np.ndarray,
        time_vector: np.ndarray,
        **params
    ) -> np.ndarray:
        """
        Generate water production profile using fractional flow sensitivity.
        
        Ref: Buckley-Leverett (1942). Water cut depends on mobility ratio 
        and cumulative recovery.
        """
        n_points = len(oil_profile)
        mobility_ratio = params.get("mobility_ratio", 2.0)
        
        # Approximate water cut evolution: f_w = 1 / (1 + (1/M) * (1-Sw)/Sw)
        # Simplified for proxy: fw = (cumulative_recovery / max_recovery)^gamma
        # where gamma is sensitive to mobility ratio.
        gamma = max(0.5, 2.0 / (mobility_ratio ** 0.5))
        
        dt = np.diff(time_vector, prepend=0)
        cumulative_oil = np.cumsum(oil_profile * dt)
        total_oil = np.sum(oil_profile * dt) + EPSILON
        
        recovery_fraction = cumulative_oil / total_oil
        water_cut = recovery_fraction ** gamma
        
        # In a gas flood (non-WAG), water production should not increase like a waterflood
        # If no water is being injected, water cut usually stays near initial levels
        injection_scheme = params.get("injection_scheme", "continuous")
        if injection_scheme.lower() != "wag":
            # For gas flood, use a much slower water cut growth or keep it at start
            water_cut_start = params.get("water_cut_start", 0.0)
            water_cut = water_cut_start + 0.1 * (water_cut - water_cut_start)
            
        water_cut = np.clip(water_cut, params.get("water_cut_start", 0.0), 0.98)

        # liquid = oil / (1 - fw)
        liquid_profile = oil_profile / (1.0 - water_cut + 1e-6)
        water_profile = liquid_profile * water_cut

        return np.maximum(water_profile, 0)

    def _generate_gas_profile(
        self,
        oil_profile: np.ndarray,
        time_vector: np.ndarray,
        **params
    ) -> np.ndarray:
        """
        Generate gas (CO2) production profile using simple zero-dimensional mass balance.
        Produced Gas = Solution Gas Component + Recycled/Breakthrough CO2
        Mass is conserved: Total Gas Produced cannot exceed Total Injected + Initial Gas - Trapped Gas.
        """
        n_points = len(oil_profile)
        time_years = time_vector / 365.25
        dt = np.diff(time_vector, prepend=0)
        
        initial_gor = params.get("initial_gor", 500.0)
        breakthrough_time_years = params.get("breakthrough_time", 3.0)
        
        # Base solution gas production (MSCFD)
        solution_gas = oil_profile * initial_gor * 1e-3 
        
        injection_profile = params.get('injection_profile', None)
        if injection_profile is None:
            return solution_gas
            
        produced_gas = np.copy(solution_gas)
        breakthrough_mask = time_years > breakthrough_time_years
        
        # Trapping efficiency: fraction of injected CO2 permanently retained in reservoir
        trapping_eff = params.get("trapping_efficiency", 0.5) 
        
        # Fraction of injected gas that returns over time after breakthrough
        if np.any(breakthrough_mask):
            time_since_bt = time_years - breakthrough_time_years
            time_since_bt[time_since_bt < 0] = 0
            
            # Use a fractional flow approach for returning CO2 mass
            # As time goes on, a larger fraction of injection cycles directly to production
            max_recycle_frac = 1.0 - trapping_eff
            recycle_growth_rate = 1.5 # 1/years
            recycle_frac = max_recycle_frac * (1.0 - np.exp(-recycle_growth_rate * time_since_bt))
            
            # Convert injection profile (res-bbl/day) to MSCFD for gas profile
            # Bg ~ 0.5 rb/MSCF → 1 rb ~ 2 MSCF
            bg_inv = params.get("mscf_per_res_bbl", 2.0)
            recycled_gas = (injection_profile * bg_inv) * recycle_frac
            produced_gas += recycled_gas

        return np.maximum(produced_gas, 0)

    def _generate_injection_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        injection_scheme: str = "continuous",
        wag_ratio: float = 1.0,
        **params
    ) -> np.ndarray:
        """
        Generate CO2 injection profile.

        Args:
            time_vector: Time points (days)
            base_injection_rate: Base injection rate (MSCFD)
            injection_scheme: "continuous", "WAG", or "tapered"
            wag_ratio: Water-alternating-gas ratio
        """
        n_points = len(time_vector)

        scheme_lower = injection_scheme.lower()
        if scheme_lower == "continuous":
            # Constant rate injection
            injection_profile = np.ones(n_points) * base_injection_rate

        elif scheme_lower == "wag":
            # Water-Alternating-Gas
            # Cycle length from params or default to 90 days.
            cycle_length = params.get("cycle_length_days", 90.0)
            if cycle_length <= 0:
                cycle_length = 90.0
                
            # Ratio of water cycle time vs CO2 cycle time
            water_frac = wag_ratio / (1.0 + wag_ratio)
            gas_phase_limit = 1.0 - water_frac
            
            cycle_phase = (time_vector % cycle_length) / cycle_length
            injection_profile = np.where(cycle_phase < gas_phase_limit, base_injection_rate, 0)

        elif scheme_lower == "huff_n_puff":
            # Cyclic injection/soak/production
            inj_period = params.get("huff_n_puff_injection_period_days", 30.0)
            soak_period = params.get("huff_n_puff_soaking_period_days", 15.0)
            prod_period = params.get("huff_n_puff_production_period_days", 45.0)
            max_cycles = params.get("huff_n_puff_max_cycles", 10)
            
            cycle_length = inj_period + soak_period + prod_period
            if cycle_length <= 0:
                cycle_length = 90.0
                inj_period = 30.0
                
            cycle_phase = (time_vector % cycle_length)
            active_cycles = np.floor(time_vector / cycle_length)
            
            # Injection happens during the inj_period and only up to max_cycles
            injection_profile = np.where(
                (cycle_phase < inj_period) & (active_cycles < max_cycles), 
                base_injection_rate, 0
            )

        elif scheme_lower == "swag":
            simultaneous = params.get("swag_simultaneous_injection", True)
            wgr = params.get("swag_water_gas_ratio", 1.0)
            
            gas_fraction = 1.0 / (1.0 + wgr)
            if simultaneous:
                injection_profile = np.ones(n_points) * base_injection_rate * gas_fraction
            else:
                cycle_length = params.get("cycle_length_days", 90.0)
                if cycle_length <= 0: cycle_length = 90.0
                cycle_phase = (time_vector % cycle_length) / cycle_length
                injection_profile = np.where(cycle_phase < gas_fraction, base_injection_rate, 0)

        elif scheme_lower == "tapered":
            # Tapered injection (decreasing over time)
            initial_mult = params.get("tapered_initial_rate_multiplier", 2.0)
            final_mult = params.get("tapered_final_rate_multiplier", 0.5)
            duration_years = params.get("tapered_duration_years", 5.0)
            func_type = params.get("tapered_function", "linear").lower()
            
            time_years = time_vector / 365.25
            t_normalized = np.clip(time_years / max(duration_years, 0.001), 0.0, 1.0)
            
            if func_type == "exponential":
                taper_factor = initial_mult * np.exp(np.log(final_mult / max(initial_mult, 1e-6)) * t_normalized)
            elif func_type == "logarithmic":
                taper_factor = initial_mult + (final_mult - initial_mult) * (1.0 - (1.0 - t_normalized)**2)
            else:  # linear
                taper_factor = initial_mult + (final_mult - initial_mult) * t_normalized
                
            injection_profile = base_injection_rate * taper_factor

        elif scheme_lower == "pulsed":
            pulse_days = params.get("pulsed_pulse_duration_days", 15.0)
            pause_days = params.get("pulsed_pause_duration_days", 15.0)
            intensity = params.get("pulsed_intensity_multiplier", 2.0)
            
            cycle_length = pulse_days + pause_days
            if cycle_length <= 0:
                cycle_length = 30.0
                pulse_days = 15.0
                
            cycle_phase = (time_vector % cycle_length)
            injection_profile = np.where(cycle_phase < pulse_days, base_injection_rate * intensity, 0)

        else:
            # Default to continuous
            injection_profile = np.ones(n_points) * base_injection_rate

        return np.maximum(injection_profile, 0)


def generate_fast_profile(
    ooip: float,
    recovery_factor: float,
    injection_rate: float,
    project_lifetime: int = 15,
    profile_type: str = "plateau_decline",
) -> Dict[str, np.ndarray]:
    """
    Convenience function to generate fast production profiles.

    Args:
        ooip: Original oil in place (STB)
        recovery_factor: Ultimate recovery factor (0-1)
        injection_rate: CO2 injection rate (MSCFD)
        project_lifetime: Project lifetime (years)
        profile_type: Type of profile model

    Returns:
        Dictionary with production profiles
    """
    generator = FastProfileGenerator(model_type=profile_type)
    return generator.generate_profile(
        ooip=ooip,
        recovery_factor=recovery_factor,
        injection_rate=injection_rate,
        project_lifetime=project_lifetime,
    )
