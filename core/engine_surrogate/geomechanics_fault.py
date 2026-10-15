"""
Geomechanical Fault Stability and Caprock Integrity Engine for CO2-EOR.

Provides physics-based modeling of:
1. Reservoir in-situ stress state and poroelastic stress path:
   Delta_sigma_h = gamma_h * Delta_P, where gamma_h = ((1 - 2*nu) / (1 - nu)) * alpha.
2. Caprock integrity assessment:
   - Tensile hydraulic micro-fracturing limit: P_crit,tensile = sigma_h,min + T_0.
   - Mohr-Coulomb shear failure envelope: tau_max vs tau_crit(sigma_m').
   - Containment safety margins and dynamic caprock breach leakage flux.
3. Fault plane stress resolution & fault slip tendency (Ts):
   - Normal stress sigma_n and shear stress tau resolved on fault plane (dip theta, strike alpha).
   - Effective normal stress sigma_n' = sigma_n - alpha * P.
   - Slip tendency Ts = tau / sigma_n'.
   - Critical fault reactivation pressure: P_crit,fault = (sigma_n - (tau - S_0)/mu_f) / alpha.
   - Dynamic fault breach leakage flux when Ts >= mu_f.
4. Closed-loop leakage mass accounting (in metric tonnes).

References:
- Zoback, M.D. (2010) Reservoir Geomechanics, Cambridge Univ. Press.
- Jaeger, Cook, & Zimmerman (2007) Fundamentals of Rock Mechanics.
- Streit, J.E. & Hillis, R.R. (2004) Energy, 29(9-10), 1445-1456 (Fault reactivation in CO2 storage).
- Rutqvist, J. (2012) Int. J. Greenh. Gas Control, 11, S130-S141 (Caprock geomechanics).
"""

from dataclasses import dataclass
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeomechanicalState:
    """Instantaneous geomechanical evaluation state."""
    pore_pressure_psi: float
    sigma_v_psi: float
    sigma_h_min_psi: float
    sigma_v_eff_psi: float
    sigma_h_eff_psi: float

    # Caprock metrics
    p_crit_tensile_psi: float
    caprock_tensile_margin_psi: float
    caprock_shear_margin_psi: float
    caprock_safety_margin_fraction: float
    is_caprock_breached: bool
    caprock_leakage_rate_tonne_day: float

    # Fault metrics
    sigma_n_fault_psi: float
    tau_fault_psi: float
    sigma_n_eff_fault_psi: float
    slip_tendency: float
    p_crit_fault_reactivation_psi: float
    is_fault_reactivated: bool
    fault_leakage_rate_tonne_day: float

    # Total geomechanical leakage
    total_geomech_leakage_rate_tonne_day: float


class GeomechanicsFaultModel:
    """
    Evaluates reservoir in-situ stress evolution, caprock integrity, and fault slip reactivation.
    """

    def __init__(
        self,
        depth_ft: float = 5000.0,
        initial_pressure_psi: float = 3000.0,
        overburden_gradient_psi_per_ft: float = 1.0,
        horizontal_stress_ratio_k0: float = 0.75,
        poissons_ratio: float = 0.25,
        biot_coefficient: float = 0.80,
        # Caprock parameters
        caprock_fracture_pressure_psi: Optional[float] = None,
        caprock_tensile_strength_psi: float = 200.0,
        caprock_cohesion_psi: float = 400.0,
        caprock_friction_angle_deg: float = 30.0,
        caprock_safety_factor: float = 0.90,
        # Fault parameters
        fault_dip_deg: float = 60.0,
        fault_strike_deg: float = 0.0,
        fault_friction_coefficient: float = 0.60,
        fault_cohesion_psi: float = 0.0,
        fault_transmissibility_multiplier: float = 0.0,
    ):
        self.depth = float(max(depth_ft, 500.0))
        self.p_init = float(max(initial_pressure_psi, 14.7))
        self.overburden_grad = float(overburden_gradient_psi_per_ft)
        self.k0 = float(np.clip(horizontal_stress_ratio_k0, 0.40, 1.20))
        self.nu = float(np.clip(poissons_ratio, 0.10, 0.45))
        self.alpha = float(np.clip(biot_coefficient, 0.50, 1.00))

        # Poroelastic stress coupling coefficient: gamma_h = ((1 - 2*nu) / (1 - nu)) * alpha
        self.gamma_h = float(((1.0 - 2.0 * self.nu) / (1.0 - self.nu)) * self.alpha)

        # In-situ vertical total stress (overburden)
        self.sigma_v0 = self.depth * self.overburden_grad
        # In-situ minimum horizontal total stress at initial pressure
        self.sigma_h0 = self.sigma_v0 * self.k0

        # Caprock parameters
        self.t0 = float(max(caprock_tensile_strength_psi, 0.0))
        self.c0_cap = float(max(caprock_cohesion_psi, 0.0))
        self.phi_cap = float(np.radians(caprock_friction_angle_deg))
        self.safety_factor = float(caprock_safety_factor)

        # Caprock fracture limit (default from in-situ stress if not explicitly provided)
        if caprock_fracture_pressure_psi is not None and caprock_fracture_pressure_psi > 0:
            self.p_frac_caprock = float(caprock_fracture_pressure_psi)
        else:
            self.p_frac_caprock = self.sigma_h0 + self.t0

        # Safe injection ceiling (e.g. 90% of fracture limit)
        self.p_safe_ceiling = self.p_frac_caprock * self.safety_factor

        # Fault parameters
        self.theta_fault = float(np.radians(fault_dip_deg))
        self.mu_f = float(max(fault_friction_coefficient, 0.10))
        self.s0_fault = float(max(fault_cohesion_psi, 0.0))
        self.fault_trans = float(np.clip(fault_transmissibility_multiplier, 0.0, 1.0))

        # Baseline critical pore pressure for fault reactivation (from initial in-situ stress state)
        cos_t = np.cos(self.theta_fault)
        sin_t = np.sin(self.theta_fault)
        sigma_n0 = self.sigma_v0 * (cos_t**2) + self.sigma_h0 * (sin_t**2)
        tau0 = 0.5 * abs(self.sigma_v0 - self.sigma_h0) * np.sin(2.0 * self.theta_fault)
        if self.mu_f > 0 and self.alpha > 0:
            self.p_crit_fault = float((sigma_n0 - (tau0 - self.s0_fault) / self.mu_f) / self.alpha)
        else:
            self.p_crit_fault = 1e6

    def evaluate_state(self, current_pressure_psi: float) -> GeomechanicalState:
        """
        Evaluate full geomechanical state, caprock failure, and fault slip at given pore pressure.

        Args:
            current_pressure_psi: Current average reservoir / injector sandface pore pressure.

        Returns:
            GeomechanicalState with stresses, safety margins, slip tendencies, and leakage rates.
        """
        P = float(max(current_pressure_psi, 14.7))
        delta_p = P - self.p_init

        # 1. Stress Path Evolution
        sigma_v = self.sigma_v0  # Overburden remains essentially constant
        sigma_h = self.sigma_h0 + self.gamma_h * delta_p  # Poroelastic horizontal stress increase

        sigma_v_eff = sigma_v - self.alpha * P
        sigma_h_eff = sigma_h - self.alpha * P

        # 2. Caprock Integrity (Tensile & Shear Failure)
        # Tensile fracturing threshold: pore pressure exceeds minimum principal stress + tensile strength
        p_crit_tensile = sigma_h + self.t0
        tensile_margin = p_crit_tensile - P

        # Mohr-Coulomb shear failure in caprock
        tau_max = 0.5 * abs(sigma_v_eff - sigma_h_eff)
        sigma_m_eff = 0.5 * (sigma_v_eff + sigma_h_eff)
        tau_crit = self.c0_cap + max(0.0, sigma_m_eff) * np.tan(self.phi_cap)
        shear_margin = tau_crit - tau_max

        # Caprock safety margin fraction relative to safe ceiling
        margin_frac = (self.p_safe_ceiling - P) / max(self.p_safe_ceiling - self.p_init, 100.0)

        is_caprock_breached = (P > self.p_frac_caprock) or (tensile_margin < 0.0) or (shear_margin < 0.0)

        # Dynamic caprock leakage flux (metric tonnes / day)
        # Darcy leakage flux across damaged micro-fractures proportional to overpressure
        if P > self.p_safe_ceiling:
            overpressure = P - self.p_safe_ceiling
            # Base rate: 0.05 tonne/day per psi overpressure above safe limit
            caprock_leakage = 0.05 * (overpressure / 100.0) ** 1.5
            if is_caprock_breached:
                caprock_leakage *= 5.0
        else:
            caprock_leakage = 0.0

        # 3. Fault Plane Stress Resolution & Slip Tendency
        # Normal and shear stress on planar fault dipping at angle theta
        cos_t = np.cos(self.theta_fault)
        sin_t = np.sin(self.theta_fault)

        sigma_n = sigma_v * (cos_t**2) + sigma_h * (sin_t**2)
        tau_f = 0.5 * abs(sigma_v - sigma_h) * np.sin(2.0 * self.theta_fault)

        sigma_n_eff = sigma_n - self.alpha * P
        sigma_n_eff_clamped = max(sigma_n_eff, 1.0)

        # Slip tendency: ratio of shear stress to effective normal stress
        slip_tendency = tau_f / sigma_n_eff_clamped

        p_crit_fault = self.p_crit_fault
        is_fault_reactivated = (slip_tendency >= self.mu_f) or (P >= p_crit_fault)

        # Dynamic fault breach leakage flux (metric tonnes / day)
        # If fault is reactivated, shear dilation opens high-permeability flow paths
        if is_fault_reactivated and P > p_crit_fault:
            excess_p = P - p_crit_fault
            excess_slip = max(0.0, slip_tendency - self.mu_f)
            # Dilation leakage proportional to slip excess and overpressure
            fault_leakage = 0.10 * (1.0 + 10.0 * excess_slip) * (excess_p / 100.0)
        else:
            fault_leakage = 0.0

        total_leakage = caprock_leakage + fault_leakage

        return GeomechanicalState(
            pore_pressure_psi=float(P),
            sigma_v_psi=float(sigma_v),
            sigma_h_min_psi=float(sigma_h),
            sigma_v_eff_psi=float(sigma_v_eff),
            sigma_h_eff_psi=float(sigma_h_eff),
            p_crit_tensile_psi=float(p_crit_tensile),
            caprock_tensile_margin_psi=float(tensile_margin),
            caprock_shear_margin_psi=float(shear_margin),
            caprock_safety_margin_fraction=float(margin_frac),
            is_caprock_breached=bool(is_caprock_breached),
            caprock_leakage_rate_tonne_day=float(caprock_leakage),
            sigma_n_fault_psi=float(sigma_n),
            tau_fault_psi=float(tau_f),
            sigma_n_eff_fault_psi=float(sigma_n_eff),
            slip_tendency=float(slip_tendency),
            p_crit_fault_reactivation_psi=float(p_crit_fault),
            is_fault_reactivated=bool(is_fault_reactivated),
            fault_leakage_rate_tonne_day=float(fault_leakage),
            total_geomech_leakage_rate_tonne_day=float(total_leakage),
        )
