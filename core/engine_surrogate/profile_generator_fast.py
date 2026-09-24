"""
Fast Production Profile Generator for Surrogate Engine
====================================================

Generates production profiles using parameterized shapes
instead of numerical simulation for ultra-fast evaluation.

Performance: O(n) where n is the number of time steps (~100-365 points)
Target: < 0.1ms for full profile generation
"""

from typing import Dict, Any, Optional, List, Tuple
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

    def __init__(
        self,
        model_type: str = "plateau_decline",
        trapping_params: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the profile generator.

        Args:
            model_type: Type of profile model
                - "plateau_decline": Plateau followed by exponential decline
                - "arps": Arps decline curves (exponential, harmonic, hyperbolic)
                - "logistic": Logistic growth model
            trapping_params: Optional dict of trapping efficiencies from
                SurrogateEngine. Keys: structural, residual, solubility, mineral, total.
                If None, gas profile generation falls back to legacy key lookups.
        """
        self.model_type = model_type
        self._trapping_params = trapping_params
        if trapping_params is None:
            logger.warning(
                "FastProfileGenerator initialized without trapping_params. "
                "Gas profile will fall back to legacy key lookups. "
                "Pass trapping_params from SurrogateEngine for physics-consistent results."
            )

    @staticmethod
    def calculate_composite_ipr_deliverability(
        p_res: float,
        p_wf: float,
        mmp: float,
        pi: float,
    ) -> float:
        """
        Calculates maximum single-well liquid deliverability using Composite Vogel-Darcy IPR.
        - Above MMP (P_wf >= MMP): Linear Darcy miscible flow.
        - Below MMP (P_wf < MMP <= P_res): Two-phase near-wellbore flashing via Composite Vogel.
        - Saturated (P_res < MMP): Classic Vogel two-phase inflow.

        Ref: Vogel (1968), Standing (1971), Beggs (1991) 'Production Optimization'.
        """
        if p_res <= 0 or pi <= 0:
            return 0.0

        p_wf_effective = max(0.0, min(float(p_wf), float(p_res)))

        # Regime 1: Entire drainage volume is above MMP (Single-phase miscible Darcy flow)
        if p_wf_effective >= mmp:
            return float(pi * (p_res - p_wf_effective))

        # Regime 2: Reservoir is above MMP, but near-wellbore drawdown flashes below MMP
        if p_res >= mmp:
            q_mmp = pi * (p_res - mmp)
            ratio = p_wf_effective / max(mmp, 1.0)
            # Composite Vogel addition below bubble point/MMP:
            q_two_phase = (pi * mmp / 1.8) * max(0.0, 1.0 - 0.2 * ratio - 0.8 * (ratio**2))
            return float(q_mmp + q_two_phase)

        # Regime 3: Entire reservoir is below MMP (Saturated immiscible two-phase inflow)
        ratio = p_wf_effective / max(p_res, 1.0)
        return float((pi * p_res / 1.8) * max(0.0, 1.0 - 0.2 * ratio - 0.8 * (ratio**2)))

    def generate_profile(
        self,
        ooip: float,
        recovery_factor: float,
        injection_rate: float,
        project_lifetime: int = 15,
        time_resolution: str = "monthly",
        trapping_params: Optional[Dict[str, float]] = None,
        **params,
    ) -> Dict[str, np.ndarray]:
        """
        Generate production profiles with proper WAG physics.

        Args:
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor (0-1)
            injection_rate: CO2 injection rate (MSCFD)
            project_lifetime: Project lifetime (years)
            time_resolution: Resolution of time steps ("monthly" or "yearly")
            trapping_params: Optional dict of trapping efficiencies from
                SurrogateEngine. Keys: structural, residual, solubility, mineral, total.
                Takes precedence over instance-level self._trapping_params.
            **params: Additional parameters for specific models

        Returns:
            Dictionary with:
                - oil_profile: Oil production rate (STB/day)
                - water_profile: Water production rate (STB/day)
                - gas_profile: Gas production rate (MSCFD)
                - injection_profile: CO2 injection rate (MSCFD)
                - time_vector: Time points (days)
        """
        n_years = float(project_lifetime)
        n_points = int(n_years * 12) + 1  # Monthly, including t=0

        time_vector = np.linspace(0, n_years * 365.25, n_points)

        tp = trapping_params if trapping_params is not None else self._trapping_params
        params["_trapping_params"] = tp

        injection_scheme = str(params.get("injection_scheme", "continuous")).lower()

        # === Phase 4.4: Ensure n_injectors in params ===
        if "n_injectors" not in params:
            params["n_injectors"] = 1
        if "n_producers" not in params:
            params["n_producers"] = 1
        if "active_injectors" not in params:
            params["active_injectors"] = params["n_injectors"]

        # Debug: log params that affect well counts
        logger.info(f"PGF input: n_inj={params.get('n_injectors')}, n_prod={params.get('n_producers')}, act={params.get('active_injectors')}, scheme={params.get('injection_scheme')}")

        # === Phase 3: Physical Boundary Checks ===
        n_injectors = params.get("n_injectors", 1)
        n_producers = params.get("n_producers", 1)
        active_injectors = params.get("active_injectors", n_injectors)

        # Storage mode: ensure active_injectors >= 1 even if n_injectors is 0
        if injection_scheme == "storage" and active_injectors == 0:
            active_injectors = max(params.get("n_injectors", 0), 1)
            params["active_injectors"] = active_injectors
            logger.info(f"Storage mode override: set active_injectors={active_injectors}")

        # Log well counts for debugging
        logger.info(f"PGF well counts: scheme={injection_scheme}, n_inj={n_injectors}, n_prod={n_producers}, active_inj={active_injectors}")

        # Zero-producer check: no production without producers (except HnP which has same-well production, or storage mode)
        if n_producers == 0 and injection_scheme != "huff_n_puff" and injection_scheme != "storage":
            logger.warning(f"No producers ({n_producers}) and scheme={injection_scheme} - returning zero profiles")
            return {
                "oil_profile": np.zeros(n_points),
                "water_profile": np.zeros(n_points),
                "gas_profile": np.zeros(n_points),
                "co2_gas_profile": np.zeros(n_points),
                "solution_gas_profile": np.zeros(n_points),
                "injection_profile": np.zeros(n_points),
                "water_injection_profile": np.zeros(n_points),
                "time_vector": time_vector,
            }

        # Zero-injector check: no injection without active injectors (storage mode overrides to use 1)
        if active_injectors == 0:
            logger.warning(f"No active injectors ({active_injectors}) - returning zero injection")
            logger.warning(f"Scheme={injection_scheme}, n_injectors={n_injectors}, params_active={params.get('active_injectors')}")
            return {
                "oil_profile": np.zeros(n_points),
                "water_profile": np.zeros(n_points),
                "gas_profile": np.zeros(n_points),
                "co2_gas_profile": np.zeros(n_points),
                "solution_gas_profile": np.zeros(n_points),
                "injection_profile": np.zeros(n_points),
                "water_injection_profile": np.zeros(n_points),
                "time_vector": time_vector,
            }

        if injection_scheme == "huff_n_puff":
            oil_profile = self._generate_huff_n_puff_production_profile(
                time_vector, ooip, recovery_factor, **params
            )
        else:
            if self.model_type == "plateau_decline":
                oil_profile = self._plateau_decline_profile(
                    time_vector, ooip, recovery_factor, **params
                )
            elif self.model_type == "arps":
                oil_profile = self._arps_decline_profile(
                    time_vector, ooip, recovery_factor, **params
                )
            elif self.model_type == "logistic":
                oil_profile = self._logistic_profile(time_vector, ooip, recovery_factor, **params)
            else:
                oil_profile = self._plateau_decline_profile(
                    time_vector, ooip, recovery_factor, **params
                )

        # === Phase 4.1: Scale injection rate by active injector count (field-scale) ===
        field_injection_rate = injection_rate * params.get("active_injectors", 1)
        co2_injection, water_injection = self._generate_injection_profile(
            time_vector, field_injection_rate, **params
        )

        params["injection_profile"] = co2_injection
        params["water_injection_profile"] = water_injection
        params["ooip"] = ooip
        params["recovery_factor"] = recovery_factor

        water_profile = self._generate_water_profile(oil_profile, time_vector, **params)
        gas_profiles = self._generate_gas_profile(oil_profile, time_vector, **params)

        if injection_scheme == "wag":
            oil_profile = self._apply_wag_oil_modulation(
                oil_profile, time_vector, co2_injection, water_injection, **params
            )
            water_profile = self._apply_wag_water_modulation(
                water_profile, time_vector, water_injection, oil_profile, **params
            )

        n_producers = params.get("n_producers", 1)
        if n_producers == 0 and injection_scheme != "huff_n_puff" and injection_scheme != "storage":
            return {
                "oil_profile": np.zeros_like(oil_profile),
                "water_profile": np.zeros_like(water_profile),
                "gas_profile": np.zeros_like(gas_profiles["total_gas"]),
                "co2_gas_profile": np.zeros_like(gas_profiles["co2_gas"]),
                "solution_gas_profile": np.zeros_like(gas_profiles["solution_gas"]),
                "injection_profile": co2_injection,
                "time_vector": time_vector,
            }

        oil_profile, water_profile, co2_gas_validated = self._validate_profile_constraints(
            oil_profile, water_profile, gas_profiles["co2_gas"], time_vector, **params
        )

        gas_profiles["co2_gas"] = co2_gas_validated

        return {
            "oil_profile": oil_profile,
            "water_profile": water_profile,
            "gas_profile": gas_profiles["total_gas"],
            "co2_gas_profile": co2_gas_validated,
            "solution_gas_profile": gas_profiles["solution_gas"],
            "injection_profile": co2_injection,
            "water_injection_profile": water_injection,
            "time_vector": time_vector,
        }

    def _validate_profile_constraints(
        self,
        oil_profile: np.ndarray,
        water_profile: np.ndarray,
        co2_gas_profile: np.ndarray,
        time_vector: np.ndarray,
        **params,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate and enforce physical constraints on production profiles.

        Checks:
        1. Max production rate constraint
        2. Minimum production rate (shut-in threshold)
        3. Water-oil ratio consistency with recovery stage
        4. CO2 production must only occur after breakthrough

        Returns:
            Tuple of (oil_profile, water_profile, co2_gas_profile) after validation
        """
        max_rate = params.get("max_production_rate_stbd", 0.0)
        min_rate = params.get("min_production_rate_stbd", 0.0)
        breakthrough_time_years = params.get("breakthrough_time_years", 5.0)

        if max_rate > 0:
            oil_profile = np.clip(oil_profile, 0, max_rate)
            water_profile = np.clip(water_profile, 0, max_rate)

        if min_rate > 0:
            oil_profile = np.where(oil_profile < min_rate, 0, oil_profile)
            water_profile = np.where(water_profile < min_rate, 0, water_profile)

        time_years = time_vector / 365.25
        pre_bt_mask = time_years < breakthrough_time_years
        if np.any(pre_bt_mask) and np.any(co2_gas_profile > 0):
            co2_gas_profile = np.where(pre_bt_mask, 0, co2_gas_profile)

        liquid_profile = oil_profile + water_profile
        if np.any(liquid_profile > 0):
            water_cut = np.divide(
                water_profile,
                liquid_profile,
                out=np.zeros_like(water_profile),
                where=liquid_profile > 0,
            )
            water_cut = np.clip(water_cut, 0, 0.95)
            water_profile = liquid_profile * water_cut
            oil_profile = liquid_profile * (1 - water_cut)

        return oil_profile, water_profile, co2_gas_profile

    def _apply_wag_oil_modulation(
        self,
        oil_profile: np.ndarray,
        time_vector: np.ndarray,
        co2_injection: np.ndarray,
        water_injection: np.ndarray,
        **params,
    ) -> np.ndarray:
        """
        Apply WAG phase mobility buffering to oil production.

        Physics (Stalkup, 1983; Caudle & Dyes, 1958; Lake, 1989):
        During gas cycles, solvent lowers IFT and mobilizes oil according to phase mobility.
        During water cycles, water buffers total mobility and stabilizes displacement.
        Conservation of mass is strictly enforced: cumulative oil is preserved.
        """
        if not np.any(co2_injection > 0) and not np.any(water_injection > 0):
            return oil_profile

        dt = np.diff(time_vector, prepend=0)
        total_cum_before = float(np.sum(oil_profile * dt))
        if total_cum_before <= 0:
            return oil_profile

        mu_o = float(params.get("viscosity_oil", 2.0))
        mu_g = float(params.get("co2_viscosity", 0.05))
        mu_w = float(params.get("water_viscosity", 1.0))

        # Relative phase mobilities from endpoints
        k_rg = float(params.get("k_rg_0", 0.3))
        k_rw = float(params.get("k_rw_0", 0.2))
        k_ro = float(params.get("k_ro_0", 0.8))

        lambda_g = k_rg / max(mu_g, 1e-6)
        lambda_w = k_rw / max(mu_w, 1e-6)
        lambda_o = k_ro / max(mu_o, 1e-6)

        # Mobility contrast determines physical phase response amplitude
        contrast = (lambda_g - lambda_w) / max(lambda_o + lambda_g + lambda_w, 1e-6)
        amp = float(np.clip(contrast * 0.1, -0.15, 0.15))

        gas_phase = co2_injection > 0
        water_phase = water_injection > 0

        modulated = oil_profile.copy()
        modulated = np.where(gas_phase, modulated * (1.0 + amp), modulated)
        modulated = np.where(water_phase, modulated * (1.0 - amp), modulated)
        modulated = np.maximum(modulated, 0.0)

        # Strictly re-normalize to preserve mass conservation (total cumulative oil)
        total_cum_after = float(np.sum(modulated * dt))
        if total_cum_after > 0:
            modulated *= (total_cum_before / total_cum_after)

        return modulated

    def _apply_wag_water_modulation(
        self,
        water_profile: np.ndarray,
        time_vector: np.ndarray,
        water_injection: np.ndarray,
        oil_profile: np.ndarray,
        **params,
    ) -> np.ndarray:
        """
        Apply WAG phase response to water production.

        Physics (Buckley-Leverett, 1942; Welge, 1952):
        Water production evolves smoothly according to fractional flow and injection response.
        Mass conservation is strictly enforced.
        """
        if not np.any(water_injection > 0):
            return water_profile

        dt = np.diff(time_vector, prepend=0)
        total_cum_before = float(np.sum(water_profile * dt))
        if total_cum_before <= 0:
            return water_profile

        mu_w = float(params.get("water_viscosity", 1.0))
        mu_o = float(params.get("viscosity_oil", 2.0))
        mobility_ratio_wo = (params.get("k_rw_0", 0.2) / max(mu_w, 1e-6)) / (
            params.get("k_ro_0", 0.8) / max(mu_o, 1e-6)
        )
        amp = float(np.clip(0.05 * mobility_ratio_wo, 0.0, 0.15))

        water_phase = water_injection > 0
        gas_phase = ~water_phase

        modulated = water_profile.copy()
        modulated = np.where(water_phase, modulated * (1.0 + amp), modulated)
        modulated = np.where(gas_phase, modulated * (1.0 - amp), modulated)
        modulated = np.maximum(modulated, 0.0)

        total_cum_after = float(np.sum(modulated * dt))
        if total_cum_after > 0:
            modulated *= (total_cum_before / total_cum_after)

        return modulated

    def _plateau_decline_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        plateau_fraction: float = 0.3,
        decline_rate: Optional[float] = None,
        ramp_up_fraction: float = 0.1,
        **params,
    ) -> np.ndarray:
        """
        Generate plateau + exponential decline profile with optional post-breakthrough acceleration.

        Physics basis (literature-derived):
        - Pre-breakthrough: stable displacement, constant rate plateau
        - Post-breakthrough: accelerated decline due to CO2 channeling and mobility contrast
        - Acceleration derived from Koval (1963) fractional flow theory

        Reference decline rates (from CO2-EOR literature):
        - Low permeability (< 50 mD): 0.05-0.12 per year
        - Medium permeability (50-200 mD): 0.12-0.20 per year
        - High permeability (> 200 mD): 0.20-0.35 per year

        Args:
            time_vector: Time points (days)
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor
            plateau_fraction: Fraction of time at plateau (0-1)
            decline_rate: Annual decline rate (fraction), obtained from params if not provided
            ramp_up_fraction: Fraction of time for ramp-up (0-1)
            **params: Additional parameters including:
                - base_decline_rate: Physics-based base decline rate
                - breakthrough_time_years: CO2 breakthrough time for post-BT acceleration
                - mobility_ratio: For post-BT acceleration calculation
        """
        n_points = len(time_vector)
        total_years = time_vector[-1] / 365.25

        if decline_rate is None:
            decline_rate = params.get("decline_rate", params.get("base_decline_rate", 0.15))

        base_decline = params.get("base_decline_rate", decline_rate)
        breakthrough_time = params.get("breakthrough_time_years", 5.0)
        mobility_ratio = params.get("mobility_ratio", 3.0)

        calculate_dynamic = params.get("use_dynamic_decline", True) and breakthrough_time > 0

        ultimate_recovery = ooip * recovery_factor

        if calculate_dynamic:
            time_years = time_vector / 365.25
            post_bt_acceleration = 1.0 + (mobility_ratio - 1.0) / (mobility_ratio + 1.0) * 0.5
            post_bt_acceleration = np.clip(post_bt_acceleration, 1.0, 2.0)
            bt_years = breakthrough_time
        else:
            time_years = np.zeros(n_points)
            post_bt_acceleration = 1.0
            bt_years = total_years + 1.0

        effective_plateau_years = plateau_fraction * total_years
        effective_ramp_years = ramp_up_fraction * total_years
        effective_decline_years = max(
            total_years - effective_plateau_years - effective_ramp_years, 0.1
        )

        if calculate_dynamic:
            ramp_up_days = total_years * 365.25 * ramp_up_fraction
            plateau_days = total_years * 365.25 * plateau_fraction
            decline_start = ramp_up_days + plateau_days

            ramp_and_plateau_integral = effective_plateau_years
            bt_idx = int(np.searchsorted(time_vector, decline_start))
            if bt_idx < n_points and bt_idx > 0:
                time_before_bt = (
                    time_years[bt_idx] - ramp_up_fraction * total_years - effective_plateau_years
                )
                if time_before_bt > 0:
                    ramp_and_plateau_integral = (ramp_up_fraction + plateau_fraction) * total_years
                else:
                    ramp_and_plateau_integral = time_years[bt_idx]

                post_bt_years = total_years - time_years[bt_idx]
                if post_bt_years > 0:
                    post_bt_integral = (
                        1.0 - np.exp(-base_decline * post_bt_acceleration * post_bt_years)
                    ) / (base_decline * post_bt_acceleration)
                    ramp_and_plateau_integral = time_years[bt_idx] + post_bt_integral

        # === Deliverability & Composite Vogel-Darcy IPR Constraints ===
        pi = float(
            params.get("locked_productivity_index")
            or params.get("productivity_index")
            or 5.0
        )
        p_wf = float(
            params.get("wellbore_pressure")
            or params.get("min_producer_bhp_psi")
            or 1500.0
        )
        p_res = float(
            params.get("pressure")
            or params.get("target_pressure_psi")
            or 2500.0
        )
        mmp = float(params.get("mmp") or 2065.0)
        n_producers = int(params.get("n_producers", 1))
        if n_producers < 1:
            n_producers = 1

        # Pattern-density deliverability scaling for field development:
        # If user left single-well fallback (n_producers == 1) on a field with large acreage,
        # scale deliverability by standard pattern spacing (40 acres for moderate perm, 20 for tight).
        area_acres = float(params.get("area_acres") or params.get("acreage") or 0.0)
        if area_acres <= 0.0:
            w_ft = float(params.get("width_ft", 1000.0))
            l_ft = float(params.get("length_ft", 2000.0))
            area_acres = (w_ft * l_ft) / 43560.0

        perm_md = float(params.get("permeability", 100.0))
        pattern_spacing_acres = 40.0 if perm_md >= 20.0 else 20.0
        n_patterns = max(1, int(round(area_acres / pattern_spacing_acres)))
        effective_producers = max(n_producers, n_patterns)

        # Single-well deliverability via Composite Vogel-Darcy IPR
        q_ipr_per_well = self.calculate_composite_ipr_deliverability(
            p_res=p_res, p_wf=p_wf, mmp=mmp, pi=pi
        )
        field_ipr_capacity = q_ipr_per_well * effective_producers

        # User/optimizer max production rate limit
        user_max_rate = params.get("max_production_rate_stbd", 0.0)
        if user_max_rate > 0:
            rate_ceiling = min(field_ipr_capacity, user_max_rate * effective_producers)
        else:
            rate_ceiling = field_ipr_capacity

        # Check for dynamic reservoir pressure profile vector
        dyn_p_res = params.get("pressure_profile")
        has_dynamic_pressure = (
            dyn_p_res is not None
            and hasattr(dyn_p_res, "__len__")
            and len(dyn_p_res) == n_points
        )

        if has_dynamic_pressure:
            dyn_ceilings = np.zeros(n_points)
            for idx in range(n_points):
                p_i = float(dyn_p_res[idx])
                q_i = self.calculate_composite_ipr_deliverability(
                    p_res=p_i, p_wf=p_wf, mmp=mmp, pi=pi
                ) * effective_producers
                dyn_ceilings[idx] = min(q_i, user_max_rate * effective_producers) if user_max_rate > 0 else q_i
        else:
            dyn_ceilings = np.full(n_points, rate_ceiling)

        if calculate_dynamic:
            # Field-level rate from ultimate recovery without duplicate * n_producers multiplier
            peak_rate_estimate = ultimate_recovery / max(ramp_and_plateau_integral, 0.1) / 365.25
        else:
            decline_integral = (
                1.0 - np.exp(-decline_rate * effective_decline_years)
            ) / decline_rate
            peak_rate_estimate = (
                ultimate_recovery / (effective_plateau_years + decline_integral) / 365.25
            )

        # Bound peak rate by physical field deliverability and user ceiling
        peak_rate_estimate = min(peak_rate_estimate, rate_ceiling)

        ramp_up_days = total_years * 365.25 * ramp_up_fraction
        plateau_days = total_years * 365.25 * plateau_fraction
        decline_start = ramp_up_days + plateau_days

        oil_profile = np.zeros(n_points)

        for i, t in enumerate(time_vector):
            if t < ramp_up_days:
                base_rate = peak_rate_estimate * (t / ramp_up_days)
            elif t < decline_start:
                base_rate = peak_rate_estimate
            else:
                decline_time_years = (t - decline_start) / 365.25
                if calculate_dynamic:
                    time_from_bt = max(
                        0,
                        decline_time_years
                        - (breakthrough_time - (ramp_up_fraction + plateau_fraction) * total_years),
                    )
                    if time_from_bt > 0:
                        current_decline = (
                            base_decline
                            * post_bt_acceleration
                            * (1.0 + 0.2 * (1.0 - np.exp(-time_from_bt / 3.0)))
                        )
                    else:
                        current_decline = base_decline
                    base_rate = peak_rate_estimate * np.exp(
                        -current_decline * decline_time_years
                    )
                else:
                    base_rate = peak_rate_estimate * np.exp(-decline_rate * decline_time_years)

            # Enforce dynamic deliverability constraint at this time step
            oil_profile[i] = min(base_rate, dyn_ceilings[i])

        # Strict physical mass conservation: normalize cumulative production to ultimate recovery
        # while respecting the physical dynamic deliverability ceiling
        dt = np.diff(time_vector, prepend=0)
        total_oil = float(np.sum(oil_profile * dt))
        if total_oil > 0 and ultimate_recovery > 0:
            scale = ultimate_recovery / total_oil
            scaled_profile = oil_profile * scale
            oil_profile = np.minimum(scaled_profile, dyn_ceilings)

        return np.maximum(oil_profile, 0)

    def _arps_decline_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        decline_type: str = "hyperbolic",
        initial_decline: float = 0.2,
        b_factor: float = 0.5,
        qi: Optional[float] = None,
        **params,
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
                    qi = (ultimate_recovery * initial_decline * (1 - b_factor) / 365.25) ** (
                        1 / (1 - b_factor)
                    )

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
        **params,
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

    def _generate_huff_n_puff_production_profile(
        self,
        time_vector: np.ndarray,
        ooip: float,
        recovery_factor: float,
        **params,
    ) -> np.ndarray:
        """
        Generate phase-aware Huff-n-Puff production profile.

        In HnP, the same well cycles through:
        1. Injection phase - CO2 injected, NO oil production
        2. Soaking phase - CO2 diffuses into oil, NO production
        3. Production phase - Oil produced, NO injection

        This method generates production ONLY during phase 3.

        Args:
            time_vector: Time points (days)
            ooip: Original oil in place (STB)
            recovery_factor: Ultimate recovery factor
            **params: Huff-n-Puff parameters including:
                - huff_n_puff_injection_period_days: default 30
                - huff_n_puff_soaking_period_days: default 15
                - huff_n_puff_production_period_days: default 45
                - huff_n_puff_max_cycles: default 10
                - huff_n_puff_production_rate_fraction: fraction of injection rate for production

        Returns:
            Oil production rate profile (STB/day) - zero during injection/soak phases
        """
        n_points = len(time_vector)
        time_years = time_vector / 365.25
        total_years = time_years[-1]

        # HnP parameters
        inj_period = params.get("huff_n_puff_injection_period_days", 30.0)
        soak_period = params.get("huff_n_puff_soaking_period_days", 15.0)
        prod_period = params.get("huff_n_puff_production_period_days", 45.0)
        max_cycles = params.get("huff_n_puff_max_cycles", 10)

        # Production rate as fraction of injection rate
        # This is physics-based: production rate from same well can't exceed injection rate
        prod_rate_fraction = params.get("huff_n_puff_production_rate_fraction", 0.3)

        cycle_length = inj_period + soak_period + prod_period
        if cycle_length <= 0:
            cycle_length = 90.0
            inj_period = 30.0
            soak_period = 15.0
            prod_period = 45.0

        # Calculate ultimate recovery and expected production rate
        ultimate_recovery = ooip * recovery_factor
        injection_rate = params.get("injection_rate", 5000.0)  # MSCFD
        co2_density = params.get("co2_density_tonne_per_mscf", 0.05297)

        # Convert injection rate to oil production rate estimate
        # Rule of thumb: 1 MSCF CO2 produces ~1-2 bbl oil in HnP
        oil_per_co2_ratio = params.get("hnp_oil_per_co2_ratio", 1.5)
        peak_production_rate = injection_rate * co2_density * oil_per_co2_ratio * prod_rate_fraction

        # Build phase-aware production profile
        oil_profile = np.zeros(n_points)
        gas_profile = np.zeros(n_points)
        water_profile = np.zeros(n_points)

        # Total active production time across all cycles
        total_active_prod_time = prod_period * max_cycles
        total_cycle_time = cycle_length * max_cycles

        # Expected total production over active periods
        expected_total_production = peak_production_rate * total_active_prod_time

        # Scale factor to match ultimate recovery
        if expected_total_production > 0:
            recovery_scale = ultimate_recovery / expected_total_production
        else:
            recovery_scale = 1.0

        for i, t in enumerate(time_vector):
            cycle_num = int(t / cycle_length)
            if cycle_num >= max_cycles:
                break

            phase_in_cycle = t % cycle_length

            # Phase 1: Injection (0 to inj_period) - NO production
            # Phase 2: Soaking (inj_period to inj_period + soak_period) - NO production
            # Phase 3: Production (inj_period + soak_period to cycle_length) - PRODUCE

            prod_phase_start = inj_period + soak_period
            if phase_in_cycle >= prod_phase_start:
                # We're in the production phase
                phase_time = phase_in_cycle - prod_phase_start

                # Production declines within the production period (simulate well cleanup)
                decay_rate = params.get("huff_n_puff_production_decay_rate", 0.02)
                instantaneous_rate = peak_production_rate * np.exp(-decay_rate * phase_time)

                # Scale to match overall recovery target
                oil_profile[i] = instantaneous_rate * recovery_scale

                # Associated gas and water during production
                initial_gor = params.get("initial_gor", 200.0)
                gas_profile[i] = instantaneous_rate * initial_gor / 1000.0 * recovery_scale
                water_cut = params.get("water_cut_start", 0.1)
                water_profile[i] = instantaneous_rate * water_cut * recovery_scale

        return np.maximum(oil_profile, 0)

    def _generate_water_profile(
        self, oil_profile: np.ndarray, time_vector: np.ndarray, **params
    ) -> np.ndarray:
        """
        Generate water production profile using fractional flow sensitivity.

        Ref: Buckley-Leverett (1942). Water cut depends on mobility ratio
        and cumulative recovery. For WAG schemes, water production responds
        to injection cycles (higher during water injection phases).
        """
        n_points = len(oil_profile)
        mobility_ratio = params.get("mobility_ratio", 2.0)

        gamma = max(0.5, 2.0 / (mobility_ratio**0.5))

        dt = np.diff(time_vector, prepend=0)
        cumulative_oil = np.cumsum(oil_profile * dt)
        total_oil = np.sum(oil_profile * dt) + EPSILON

        recovery_fraction = cumulative_oil / total_oil
        water_cut = recovery_fraction**gamma

        injection_scheme = str(params.get("injection_scheme", "continuous"))
        if injection_scheme.lower() != "wag":
            water_cut_start = params.get("water_cut_start", 0.0)
            water_cut = water_cut_start + 0.1 * (water_cut - water_cut_start)
        else:
            water_cut_start = params.get("water_cut_start", 0.05)
            water_inj_profile = params.get("water_injection_profile", None)

            if water_inj_profile is not None and np.any(water_inj_profile > 0):
                max_water_inj = max(np.max(water_inj_profile), 1.0)
                water_inj_factor = water_inj_profile / max_water_inj

                wag_water_enhancement = params.get("wag_water_enhancement", 1.8)
                base_water_cut = water_cut_start + 0.2 * (water_cut - water_cut_start)
                water_cut = base_water_cut * (
                    1.0 + water_inj_factor * (wag_water_enhancement - 1.0)
                )
            else:
                water_cut = water_cut_start + 0.2 * (water_cut - water_cut_start)

        water_cut = np.clip(water_cut, max(0.0, water_cut_start), 0.95)

        liquid_profile = oil_profile / (1.0 - water_cut + 1e-6)
        water_profile = liquid_profile * water_cut

        return np.maximum(water_profile, 0)

    def _generate_gas_profile(
        self, oil_profile: np.ndarray, time_vector: np.ndarray, **params
    ) -> Dict[str, np.ndarray]:
        """
        Unified gas production model based on Koval (1963) fractional flow and CO2 trapping physics.

        Trapping mechanisms (research-based):
        - Structural: Dominant initially, CO2 rises to caprock
        - Residual: 15-30% becomes immobile droplets (residual_gas_trapping_fraction)
        - Solubility: 2-5% dissolves in brine (solubility_trapping_fraction)
        - Mineral: Negligible in 15-year EOR timeframe

        Physics:
        - Pre-BT: Solution gas = oil_rate × GOR × (1 - pre_bt_co2_frac)
                         + dissolved CO2 = oil_rate × GOR × pre_bt_co2_frac
        - Post-BT: Solution gas + CO2 from cumulative stored using fractional flow
                   CO2_rate = cumulative_stored × koval_factor × (1 - total_trapping)

        Args:
            oil_profile: Oil production rate (STB/day)
            time_vector: Time points (days)
            **params: initial_gor, breakthrough_time_years, mobility_ratio,
                     injection_profile, koval_factor_multiplier, etc.

        Returns:
            Dictionary with gas profiles (all in MSCFD):
                - 'total_gas': Combined gas production
                - 'co2_gas': CO2 production only (post-breakthrough)
                - 'solution_gas': Total solution gas (HC + dissolved CO2)
                - 'solution_hc': Hydrocarbon solution gas only
                - 'solution_co2': Dissolved CO2 in solution gas
        """
        n_points = len(oil_profile)
        time_years = time_vector / 365.25

        initial_gor = params.get("initial_gor", 200.0)
        breakthrough_time_years = params.get("breakthrough_time_years", 5.0)
        mobility_ratio = params.get("mobility_ratio", 5.0)
        koval_mult = params.get("koval_factor_multiplier", 1.0)
        pre_bt_co2_frac = params.get("pre_breakthrough_co2_fraction_in_solution_gas", 0.05)
        tp = params.get("_trapping_params")

        residual_trap = (
            tp.get("residual")
            if tp else params.get("residual_gas_trapping_fraction", 0.20)
        )
        solubility_trap = (
            tp.get("solubility")
            if tp else params.get("solubility_trapping_fraction", 0.03)
        )
        structural_trap = (
            tp.get("structural")
            if tp else params.get("structural_trapping_factor", 0.0)
        )
        co2_rate_const = params.get("co2_production_rate_constant", 0.2)
        injection_profile = params.get("injection_profile", None)

        total_trapping = float(np.clip(residual_trap + solubility_trap + structural_trap, 0.0, 0.95))

        # Authentic Koval (1963) Viscous Fingering & Heterogeneity Factor
        v_dp = float(params.get("v_dp", params.get("v_dp_coefficient", 0.5)))
        h_koval = 1.0 / max(1.0 - min(v_dp, 0.95), 0.05) ** 2
        m_eff = max(float(mobility_ratio), 1.0)
        e_eff = (0.78 + 0.22 * (m_eff ** 0.25)) ** 4
        koval_factor = float(np.clip(h_koval * e_eff * koval_mult, 1.0, 50.0))

        # Koval fractional flow: as fingering/mobility worsens (K increases), gas fractional flow increases monotonically
        # f_g = K * S / (1 + S * (K - 1)), with reference displacement saturation S_ref = 0.40
        s_ref = 0.40
        frac_flow_co2 = (koval_factor * s_ref) / (1.0 + s_ref * (koval_factor - 1.0))
        frac_flow_co2 = float(np.clip(frac_flow_co2, 0.15, 0.95))

        solution_gas_total = oil_profile * initial_gor * 1e-3

        solution_hc = solution_gas_total * (1.0 - pre_bt_co2_frac)
        solution_co2 = solution_gas_total * pre_bt_co2_frac

        co2_gas = np.zeros(n_points)

        if injection_profile is not None and np.any(injection_profile > 0):
            dt_days = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 1.0
            cumulative_co2_injected = np.cumsum(injection_profile * dt_days)

            post_bt_mask = time_years > breakthrough_time_years

            if np.any(post_bt_mask):
                bt_idx = int(np.searchsorted(time_years, breakthrough_time_years))
                bt_idx = min(bt_idx, n_points - 1)

                time_since_bt = time_years - breakthrough_time_years
                time_since_bt = np.maximum(time_since_bt, 0.0)

                co2_growth = 1.0 - np.exp(-co2_rate_const * time_since_bt)

                target_co2_recycled_mscf = params.get("target_co2_recycled_mscf")
                if target_co2_recycled_mscf is not None and target_co2_recycled_mscf > 0:
                    dt = np.diff(time_vector, prepend=0)
                    shape_integral = float(np.sum(co2_growth[bt_idx:] * dt[bt_idx:]))
                    if shape_integral > 0:
                        co2_gas[bt_idx:] = co2_growth[bt_idx:] * (target_co2_recycled_mscf / shape_integral)
                    else:
                        co2_gas[bt_idx:] = injection_profile[bt_idx:] * (1.0 - total_trapping) * frac_flow_co2 * co2_growth[bt_idx:]
                else:
                    for i in range(bt_idx, n_points):
                        co2_rate_available = injection_profile[i] * (1.0 - total_trapping)
                        co2_gas[i] = co2_rate_available * frac_flow_co2 * co2_growth[i]
                        max_possible = injection_profile[i] * (1.0 - total_trapping)
                        co2_gas[i] = min(co2_gas[i], max_possible)

        total_gas = solution_hc + solution_co2 + co2_gas

        return {
            "total_gas": np.maximum(total_gas, 0),
            "co2_gas": np.maximum(co2_gas, 0),
            "solution_gas": np.maximum(solution_hc + solution_co2, 0),
            "solution_hc": np.maximum(solution_hc, 0),
            "solution_co2": np.maximum(solution_co2, 0),
        }

    def _generate_injection_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        injection_scheme: str = "continuous",
        **params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CO2 and water injection profiles.

        This is the single source of truth for all injection scheme shaping.
        Consolidates physics from the former injection_schemes.py module.

        Args:
            time_vector: Time points (days)
            base_injection_rate: Base CO2 injection rate (MSCFD)
            injection_scheme: "continuous", "wag", "huff_n_puff", "swag", "tapered", "pulsed"
            **params: Scheme-specific parameters

        Returns:
            Tuple of (co2_injection_profile, water_injection_profile)
            For non-WAG schemes, water_injection_profile is zeros
        """
        n_points = len(time_vector)
        scheme_lower = str(injection_scheme).lower()
        water_injection_profile = np.zeros(n_points)

        if scheme_lower == "continuous":
            injection_profile = np.ones(n_points) * base_injection_rate

        elif scheme_lower == "wag":
            injection_profile, water_injection_profile = self._generate_wag_profile(
                time_vector, base_injection_rate, **params
            )

        elif scheme_lower == "huff_n_puff":
            injection_profile = self._generate_huff_n_puff_profile(
                time_vector, base_injection_rate, **params
            )

        elif scheme_lower == "swag":
            injection_profile, water_injection_profile = self._generate_swag_profile(
                time_vector, base_injection_rate, **params
            )

        elif scheme_lower == "tapered":
            injection_profile = self._generate_tapered_profile(
                time_vector, base_injection_rate, **params
            )

        elif scheme_lower == "pulsed":
            injection_profile = self._generate_pulsed_profile(
                time_vector, base_injection_rate, **params
            )

        elif scheme_lower == "storage":
            injection_profile = np.ones(n_points) * base_injection_rate

        else:
            injection_profile = np.ones(n_points) * base_injection_rate

        return np.maximum(injection_profile, 0), water_injection_profile

    def _generate_wag_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        **params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced WAG injection with mobility-ratio-based optimization.

        Physics:
        - Initial short cycles for mobility control establishment
        - Standard longer cycles for main production phase
        - CO2 tapering within each gas phase for front stability
        - Mobility-ratio-adaptive WAG ratio
        """
        n_points = len(time_vector)
        water_injection_profile = np.zeros(n_points)

        wag_ratio = params.get("wag_ratio", 1.0)
        initial_cycle_length = params.get("initial_wag_cycle_length", 45)
        standard_cycle_length = params.get("standard_wag_cycle_length", 90)
        initial_cycles = params.get("initial_wag_cycles", 3)
        mobility_factor = params.get("mobility_ratio_factor", 0.001)
        high_mobility_threshold = params.get("high_mobility_threshold", 2.0)
        max_wag_ratio = params.get("max_enhanced_wag_ratio", 2.0)
        wag_ratio_enhancement = params.get("wag_ratio_enhancement_factor", 1.5)
        co2_taper_percentage = params.get("co2_taper_percentage", 0.1)
        min_co2_taper_factor = params.get("min_co2_taper_factor", 0.85)
        default_b_gas = params.get("default_gas_fvf", 0.005)

        mobility_ratio = base_injection_rate * mobility_factor
        if mobility_ratio > high_mobility_threshold:
            enhanced_wag_ratio = min(max_wag_ratio, wag_ratio * wag_ratio_enhancement)
        else:
            enhanced_wag_ratio = wag_ratio

        co2_inj_rb_per_day = base_injection_rate * default_b_gas
        enhanced_water_rate_bpd = co2_inj_rb_per_day * enhanced_wag_ratio

        water_frac = enhanced_wag_ratio / (1.0 + enhanced_wag_ratio)
        gas_phase_limit = 1.0 - water_frac

        injection_profile = np.zeros(n_points)

        for i, t_day in enumerate(time_vector):
            cycle_count = i // standard_cycle_length
            cycle_day = i % standard_cycle_length

            if cycle_count < initial_cycles * 2:
                cycle_length = initial_cycle_length
                current_cycle_day = i % initial_cycle_length
            else:
                cycle_length = standard_cycle_length
                current_cycle_day = cycle_day

            cycle_phase = current_cycle_day / cycle_length if cycle_length > 0 else 0

            if cycle_phase < gas_phase_limit:
                is_gas_phase = True
            else:
                is_gas_phase = False

            if is_gas_phase:
                taper_factor = 1.0 - (co2_taper_percentage * cycle_phase / max(gas_phase_limit, 1e-6))
                taper_factor = max(min_co2_taper_factor, taper_factor)
                injection_profile[i] = base_injection_rate * taper_factor
                water_injection_profile[i] = 0.0
            else:
                injection_profile[i] = 0.0
                water_injection_profile[i] = enhanced_water_rate_bpd

        return injection_profile, water_injection_profile

    def _generate_huff_n_puff_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        **params,
    ) -> np.ndarray:
        """
        Huff-n-Puff injection with proper soak period support.

        Phases: Injection -> Soaking -> Production
        """
        n_points = len(time_vector)
        inj_period = params.get("huff_n_puff_injection_period_days", 30.0)
        soak_period = params.get("huff_n_puff_soaking_period_days", 7.0)
        prod_period = params.get("huff_n_puff_production_period_days", 60.0)
        max_cycles = params.get("huff_n_puff_max_cycles", 10)

        cycle_length = inj_period + soak_period + prod_period
        if cycle_length <= 0:
            cycle_length = 97.0
            inj_period = 30.0
            soak_period = 7.0
            prod_period = 60.0

        cycle_phase = (time_vector % cycle_length) / max(cycle_length, 1.0)
        cycle_number = np.floor(time_vector / cycle_length)
        injection_profile = np.where(
            (cycle_phase < inj_period / max(cycle_length, 1.0)) & (cycle_number < max_cycles),
            base_injection_rate,
            0.0,
        )

        return injection_profile

    def _generate_swag_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        **params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SWAG injection with simultaneous or alternating modes.

        Physics:
        - Mixing efficiency reduces effective CO2 injection
        - Can operate in simultaneous (both phases) or alternating mode
        """
        n_points = len(time_vector)
        simultaneous = params.get("swag_simultaneous_injection", True)
        wgr = params.get("swag_water_gas_ratio", 1.0)
        mixing_efficiency = params.get("swag_mixing_efficiency", 0.9)
        cycle_length_days = params.get("swag_cycle_length_days", 30)

        gas_fraction = 1.0 / (1.0 + wgr)
        effective_co2_rate = base_injection_rate * gas_fraction * mixing_efficiency

        water_rate_bpd = base_injection_rate * wgr * params.get("default_gas_fvf", 0.005) * 1000

        if simultaneous:
            injection_profile = np.ones(n_points) * effective_co2_rate
            water_injection_profile = np.ones(n_points) * water_rate_bpd
        else:
            cycle_length = max(1, cycle_length_days)
            injection_profile = np.zeros(n_points)
            water_injection_profile = np.zeros(n_points)

            for i, t_day in enumerate(time_vector):
                day_in_cycle = int(t_day % cycle_length)
                if day_in_cycle < gas_fraction * cycle_length:
                    injection_profile[i] = effective_co2_rate
                    water_injection_profile[i] = 0.0
                else:
                    injection_profile[i] = 0.0
                    water_injection_profile[i] = water_rate_bpd

        return injection_profile, water_injection_profile

    def _generate_tapered_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        **params,
    ) -> np.ndarray:
        """
        Tapered injection with linear, exponential, or logarithmic decay.
        """
        initial_mult = params.get("tapered_initial_rate_multiplier", 2.0)
        final_mult = params.get("tapered_final_rate_multiplier", 0.5)
        duration_years = params.get("tapered_duration_years", 5.0)
        func_type = params.get("tapered_function", "linear").lower()

        time_years = time_vector / 365.25
        t_normalized = np.clip(time_years / max(duration_years, 0.001), 0.0, 1.0)

        if func_type == "exponential":
            exponential_decay_factor = params.get("exponential_decay_factor", 3.0)
            taper_factor = initial_mult * np.exp(-exponential_decay_factor * t_normalized)
            taper_factor = np.maximum(taper_factor, final_mult)
        elif func_type == "logarithmic":
            log_progress_factor = params.get("log_progress_factor", 10)
            log_denominator = np.log1p(log_progress_factor)
            if log_denominator > 0:
                taper_factor = initial_mult - np.log1p(t_normalized * log_progress_factor) * (
                    initial_mult - final_mult
                ) / log_denominator
            else:
                taper_factor = final_mult
            taper_factor = np.maximum(taper_factor, final_mult)
        else:
            taper_factor = initial_mult + (final_mult - initial_mult) * t_normalized

        injection_profile = base_injection_rate * taper_factor
        min_floor = base_injection_rate * params.get("tapered_min_rate_floor", 0.05)
        injection_profile = np.maximum(injection_profile, min_floor)

        return injection_profile

    def _generate_pulsed_profile(
        self,
        time_vector: np.ndarray,
        base_injection_rate: float,
        **params,
    ) -> np.ndarray:
        """
        Pulsed injection with intermittent high-intensity pulses.
        """
        pulse_days = params.get("pulsed_pulse_duration_days", 15.0)
        pause_days = params.get("pulsed_pause_duration_days", 15.0)
        intensity = params.get("pulsed_intensity_multiplier", 2.0)

        cycle_length = pulse_days + pause_days
        if cycle_length <= 0:
            cycle_length = 30.0
            pulse_days = 15.0

        injection_profile = np.zeros(n_points := len(time_vector))

        for i, t_day in enumerate(time_vector):
            day_in_cycle = t_day % cycle_length
            if day_in_cycle < pulse_days:
                injection_profile[i] = base_injection_rate * intensity
            else:
                injection_profile[i] = 0.0

        return injection_profile

    def generate_fast_profile(
        self,
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
