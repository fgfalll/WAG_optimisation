"""
Recovery Models for CO2-EOR Simulation
=======================================

Implements the Hybrid Oil Recovery Model from:
  "Hybrid Oil Recovery Model for Optimizing Carbon Dioxide Storage
   and Reservoir Safety" (PhD Research, Стаття 3.pdf)

All formulas are taken DIRECTLY from the paper. No data fitting.

Models implemented:
  1. Todd-Longstaff Miscibility Model      (paper Section 2.2)
  2. MiscibleRecoveryModel (Koval/Welge)   (paper Eq. RF_mis)
  3. ImmiscibleRecoveryModel (Nc/Sor)      (paper Eq. RF_im)
  4. IFT Model (exponential decay)         (paper Eq. σ(p))
  5. HybridRecoveryModel (sigmoid weight)  (paper Eq. RF_total)
  6. Supporting models (Buckley-Leverett, Dykstra-Parsons, Koval)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Optional, Type
from copy import deepcopy

try:
    from core.data_models import LayerDefinition, PhysicalConstants
except ImportError:
    from ..data_models import LayerDefinition, PhysicalConstants

import numpy as np
from scipy.optimize import fsolve

# Use centralized constants from PhysicalConstants
_PERM_CONSTANTS = PhysicalConstants()
PERM_MD_TO_M2 = _PERM_CONSTANTS.MD_TO_M2    # 9.869233e-16 mD to m²
PERM_MD_TO_CM2 = _PERM_CONSTANTS.MD_TO_CM2  # 9.869233e-13 mD to cm²

VISC_CP_TO_POISE = 0.01
RATE_STB_DAY_TO_CM3_S = 1.84013
GRAVITY_ACCEL_CM_S2 = 981.0
EPSILON = 1e-9  # Using hardcoded value for backward compatibility

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: TODD-LONGSTAFF MISCIBILITY MODEL
# Paper: Section 2.2 — Miscibility Model (Todd-Longstaff)
# =============================================================================

class ToddLongstaffMixingModel:
    """
    Todd-Longstaff miscibility model for CO2-EOR.

    Computes effective viscosities that account for partial miscibility
    between CO2 (solvent) and reservoir oil.

    Reference formulas (paper Section 2.2):
      f_s  = S_g / (S_g + S_o)                          [solvent fraction]
      μ_m  = (f_s·μ_g^(-1/4) + (1-f_s)·μ_o^(-1/4))^(-4) [mixture viscosity]
      μ_oe = μ_m^ω · μ_o^(1-ω)                          [effective oil viscosity]
      μ_ge = μ_m^ω · μ_g^(1-ω)                          [effective gas viscosity]

    where ω (omega) is the Todd-Longstaff mixing parameter (0 = immiscible,
    1 = fully miscible). Paper Table 3: ω = 0.7 for CO2-EOR.
    """

    def __init__(self, omega: float = 0.7):
        """
        Args:
            omega: Todd-Longstaff mixing parameter (paper Table 3: 0.7)
        """
        # paper specifies 0 ≤ ω ≤ 1
        self.omega = float(np.clip(omega, 0.0, 1.0))

    def mixture_viscosity(
        self,
        mu_oil: float,
        mu_co2: float,
        s_oil: float,
        s_gas: float,
    ) -> float:
        """
        Compute mixture viscosity μ_m using Todd-Longstaff formula.

        Args:
            mu_oil: Oil viscosity (cP)
            mu_co2: CO2/gas viscosity (cP)
            s_oil:  Oil saturation (fraction)
            s_gas:  Gas/CO2 saturation (fraction)

        Returns:
            Mixture viscosity μ_m (cP)
        """
        mu_oil = max(mu_oil, EPSILON)
        mu_co2 = max(mu_co2, EPSILON)
        s_total = s_oil + s_gas
        if s_total < EPSILON:
            return mu_oil  # no solvent present → pure oil viscosity

        # Solvent fraction in hydrocarbon mixture (paper Eq.: f_s = S_g/(S_g+S_o))
        f_s = np.clip(s_gas / s_total, 0.0, 1.0)

        # Mixture viscosity: μ_m = (f_s·μ_g^(-1/4) + (1-f_s)·μ_o^(-1/4))^(-4)
        inner = f_s * (mu_co2 ** (-0.25)) + (1.0 - f_s) * (mu_oil ** (-0.25))
        mu_m = inner ** (-4.0)
        return float(np.clip(mu_m, EPSILON, 1e6))

    def effective_oil_viscosity(
        self,
        mu_oil: float,
        mu_co2: float,
        s_oil: float,
        s_gas: float,
    ) -> float:
        """
        Compute effective oil viscosity μ_oe.

        μ_oe = μ_m^ω · μ_o^(1-ω)   (paper Eq. for effective oil viscosity)

        Args:
            mu_oil: Oil viscosity (cP)
            mu_co2: CO2 viscosity (cP)
            s_oil:  Oil saturation
            s_gas:  Gas saturation

        Returns:
            Effective oil viscosity μ_oe (cP)
        """
        mu_m = self.mixture_viscosity(mu_oil, mu_co2, s_oil, s_gas)
        mu_oe = (mu_m ** self.omega) * (mu_oil ** (1.0 - self.omega))
        return float(np.clip(mu_oe, EPSILON, 1e6))

    def effective_gas_viscosity(
        self,
        mu_oil: float,
        mu_co2: float,
        s_oil: float,
        s_gas: float,
    ) -> float:
        """
        Compute effective CO2 viscosity μ_ge.

        μ_ge = μ_m^ω · μ_g^(1-ω)   (paper Eq. for effective gas viscosity)
        """
        mu_m = self.mixture_viscosity(mu_oil, mu_co2, s_oil, s_gas)
        mu_ge = (mu_m ** self.omega) * (mu_co2 ** (1.0 - self.omega))
        return float(np.clip(mu_ge, EPSILON, 1e6))

    def effective_mobility_ratio(
        self,
        mu_oil: float,
        mu_co2: float,
        s_oil: float,
        s_gas: float,
    ) -> float:
        """
        Effective mobility ratio M_eff = μ_o / μ_oe.

        Used in Koval factor calculation for miscible RF.
        """
        mu_oe = self.effective_oil_viscosity(mu_oil, mu_co2, s_oil, s_gas)
        return float(mu_oil / max(mu_oe, EPSILON))


# =============================================================================
# SECTION 2: IFT MODEL
# Paper: Section 2.3 — IFT and Capillary Number
# σ(p) = σ_0 · exp(-λ·(p - MMP)) for p ≥ MMP
# σ(p) = σ_0                       for p < MMP
# =============================================================================

def interfacial_tension(
    pressure_psi: float,
    mmp_psi: float,
    sigma_0: float = 20.0,
    lambda_ift: float = 0.001,
) -> float:
    """
    Interfacial tension between CO2 and oil (mN/m).

    Paper Eq.: σ(p) = σ_0 · exp(-λ·(p - MMP))  for p ≥ MMP
               σ(p) = σ_0                         for p < MMP

    Args:
        pressure_psi: Reservoir pressure (psi)
        mmp_psi:      Minimum miscibility pressure (psi)
        sigma_0:      Base IFT at MMP (mN/m), paper default ≈ 20 mN/m
        lambda_ift:   IFT decay rate (psi⁻¹), paper ≈ 0.001 psi⁻¹

    Returns:
        IFT σ (mN/m)
    """
    if pressure_psi >= mmp_psi:
        delta_p = pressure_psi - mmp_psi
        sigma = sigma_0 * np.exp(-lambda_ift * delta_p)
    else:
        sigma = sigma_0
    return float(np.clip(sigma, 0.0, sigma_0))


# =============================================================================
# SECTION 3: CAPILLARY NUMBER AND SOR REDUCTION
# Paper: Section 2.3
# N_c = v · μ_CO2 / σ
# S_or* / S_or_base = 1 / (1 + (N_c / N_c_crit)^B)
# =============================================================================

def capillary_number(
    interstitial_velocity_cm_s: float,
    mu_co2_poise: float,
    sigma_mn_m: float,
) -> float:
    """
    Capillary number N_c = v · μ_CO2 / σ (dimensionless).

    Paper Eq.: N_c = v·μ_CO2 / σ_CO2-oil

    Note: Consistent CGS units required. σ in mN/m → dyne/cm (1 mN/m = 1 dyne/cm).

    Args:
        interstitial_velocity_cm_s: Interstitial (pore) velocity (cm/s)
        mu_co2_poise:               CO2 viscosity (poise = g/cm/s)
        sigma_mn_m:                 IFT in mN/m (= dyne/cm in CGS)

    Returns:
        Capillary number N_c (dimensionless)
    """
    sigma_dyne_cm = max(sigma_mn_m, EPSILON)  # 1 mN/m = 1 dyne/cm
    nc = (interstitial_velocity_cm_s * mu_co2_poise) / sigma_dyne_cm
    return float(np.clip(nc, 0.0, 1.0))


def residual_oil_saturation(
    sor_base: float,
    nc: float,
    nc_crit: float = 1e-5,
    b_exp: float = 0.5,
) -> float:
    """
    Reduced residual oil saturation S_or* from paper formula.

    Paper Eq.: S_or* / S_or_base = 1 / (1 + (N_c/N_c_crit)^B)

    Args:
        sor_base: Base residual oil saturation (immiscible flooding)
        nc:       Capillary number
        nc_crit:  Critical capillary number (paper: 1×10⁻⁵)
        b_exp:    Exponent B (paper: 0.5)

    Returns:
        Effective S_or* (fraction)
    """
    nc_ratio = nc / max(nc_crit, EPSILON)
    reduction = 1.0 / (1.0 + (nc_ratio ** b_exp))
    sor_star = sor_base * reduction
    return float(np.clip(sor_star, 0.0, sor_base))


# =============================================================================
# SECTION 4: SIGMOID TRANSITION WEIGHTING (Hybrid Model)
# Paper: Eq. ω(P_r, χ) = 1/(1+exp(-β·(P_r - α_eff(χ))))
#         α_eff(χ) = α_base + comp_sensitivity·(C7+ - C7_base)
# =============================================================================

class SigmoidTransition:
    """
    Sigmoidal weighting function for hybrid miscible/immiscible transition.

    Paper Eq.: ω(P_r, χ) = 1 / (1 + exp(-β·(P_r - α_eff(χ))))
    where:
        P_r       = P_res / MMP  (pressure ratio)
        α_eff(χ)  = α_base + comp_sensitivity·(C7+ - C7_base)
        α_base    = 1.0          (transition centre at P/MMP = 1)
        β         = 20           (steepness of transition)
        comp_sensitivity = 0.5   (compositional shift coefficient)
        C7_base   = 0.3          (reference C7+ fraction)
    """

    def __init__(
        self,
        alpha_base: float = 1.0,
        miscibility_window: float = 0.1,
        comp_sensitivity: float = 0.5,
        c7_base: float = 0.3,
        **kwargs,
    ):
        """
        Initialize with physical parameters.
        β is derived from the logit-90% transition interval (ΔPr).
        β = 4.394 / ΔPr.
        """
        # α_base = 1.0: transition centred at P/MMP = 1 (paper default)
        self.alpha_base = np.clip(alpha_base, 0.5, 1.5)
        
        # Derive beta from physical miscibility window (default 10% of MMP)
        # beta = 2 * ln(0.9/0.1) / miscibility_window
        derived_beta = 4.394 / max(miscibility_window, 0.01)
        self.beta = kwargs.get("beta", derived_beta)
        
        self.comp_sensitivity = comp_sensitivity
        self.c7_base = c7_base

    def evaluate(self, p_mmp_ratio: float, c7_plus_fraction: float) -> float:
        """
        Compute weighting factor ω ∈ [0, 1].

        Returns 0 (fully immiscible) at P << MMP,
        returns 1 (fully miscible) at P >> MMP.
        """
        p_mmp_ratio = np.clip(p_mmp_ratio, 0.0, 2.0)
        c7_plus_fraction = np.clip(c7_plus_fraction, 0.0, 1.0)

        # Compositional shift: heavier oil requires higher P/MMP for same miscibility
        comp_shift = self.comp_sensitivity * (c7_plus_fraction - self.c7_base)
        alpha_eff = self.alpha_base + comp_shift

        # Physical sigmoid: ω = 1 / (1 + exp(-beta * (Pr - alpha)))
        arg = self.beta * (p_mmp_ratio - alpha_eff)
        return float(1.0 / (1.0 + np.exp(-np.clip(arg, -700, 700))))


class TransitionEngine:
    def __init__(self, **params):
        # Default: α=1.0 (transition at P/MMP=1), 10% window
        self.transition_alpha = params.get("transition_alpha", 1.0)
        self.miscibility_window = params.get("miscibility_window", 0.1)
        # Only pass beta if explicitly set — avoid overriding derived_beta with None
        _beta = params.get("transition_beta")
        _sigmoid_kwargs = {}
        if _beta is not None:
            _sigmoid_kwargs["beta"] = _beta
        self._transition_func = SigmoidTransition(
            alpha_base=self.transition_alpha,
            miscibility_window=self.miscibility_window,
            **_sigmoid_kwargs,
        )

    def calculate_weight(self, p_mmp_ratio: float, c7_plus_fraction: float) -> float:
        return self._transition_func.evaluate(p_mmp_ratio, c7_plus_fraction)


# =============================================================================
# SECTION 5: ABSTRACT BASE CLASS
# =============================================================================

class RecoveryModel(ABC):
    def __init__(self, max_rf_cap: float = 1.0, **kwargs):
        if not (0.0 <= max_rf_cap <= 1.0):
            raise ValueError("max_rf_cap must be between 0 and 1.")
        self.max_rf_cap = max_rf_cap
        self._validate_inputs(**kwargs)

    @abstractmethod
    def calculate(self, **kwargs) -> float:
        pass

    def _bounded_calculate(self, **kwargs) -> float:
        rf = self.calculate(**kwargs)
        return np.clip(rf, 0.0, self.max_rf_cap)

    def _validate_inputs(self, **kwargs):
        if (p := kwargs.get("pressure")) is not None and p <= 0:
            raise ValueError("Pressure must be positive.")
        if (mmp := kwargs.get("mmp")) is not None and mmp <= 0:
            raise ValueError("MMP must be positive.")
        if (r := kwargs.get("rate")) is not None and r < 0:
            raise ValueError("Rate must be non-negative.")
        if (phi := kwargs.get("porosity")) is not None and not (0.01 <= phi <= 0.60):
            raise ValueError("Porosity must be between 0.01 and 0.60.")
        if (k := kwargs.get("permeability")) is not None and k <= 0:
            raise ValueError("Permeability must be positive.")
        if (mu := kwargs.get("viscosity_oil")) is not None and mu <= 0:
            raise ValueError("Oil viscosity must be positive.")
        if (mu := kwargs.get("co2_viscosity")) is not None and mu <= 0:
            raise ValueError("CO2 viscosity must be positive.")


# =============================================================================
# SECTION 6: MISCIBLE RECOVERY MODEL (Koval/Welge formula)
# Paper: RF_mis = (3K²-3K+1)/K³ · (1-S_wi)
#        K = H · E_eff
#        H = 1/(1-V_DP)²         (heterogeneity, Dykstra-Parsons)
#        E_eff = (0.78+0.22·M^0.25)^4  (Kowal effective viscosity ratio)
#        M = μ_o / μ_oe            (effective mobility ratio via Todd-Longstaff)
# =============================================================================

class MiscibleRecoveryModel(RecoveryModel):
    """
    Miscible CO2-EOR recovery factor using Koval factor and Welge formula.

    Paper Eq.:
        K      = H · E_eff
        H      = 1 / (1 - V_DP)²
        E_eff  = (0.78 + 0.22 · M_eff^0.25)^4
        RF_mis = (3K² - 3K + 1) / K³ · (1 - S_wi)

    where M_eff is computed via Todd-Longstaff mixing (paper Section 2.2).
    """

    def __init__(self, omega: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        # Todd-Longstaff mixing parameter (paper Table 3)
        self._tl = ToddLongstaffMixingModel(omega=omega)

    def calculate(self, **kwargs) -> float:
        p = kwargs

        # Reservoir parameters
        v_dp = float(np.clip(p.get("v_dp", 0.6), 0.0, 0.9999))
        s_wi = float(np.clip(p.get("s_wi", p.get("connate_water_saturation", 0.25)), 0.0, 0.8))
        mu_oil = float(p.get("viscosity_oil", p.get("mu_oil", 1.5)))
        mu_co2 = float(p.get("viscosity_inj", p.get("co2_viscosity", p.get("mu_co2", 0.05))))

        # Current phase saturations (used by Todd-Longstaff)
        s_oil = float(p.get("s_oil", p.get("oil_saturation", 1.0 - s_wi - 0.05)))
        s_gas = float(p.get("s_gas", p.get("gas_saturation", 0.05)))
        s_oil = max(s_oil, EPSILON)
        s_gas = max(s_gas, EPSILON)

        # Step 1: Effective mobility ratio via Todd-Longstaff (paper Section 2.2)
        M_eff = self._tl.effective_mobility_ratio(mu_oil, mu_co2, s_oil, s_gas)
        M_eff = max(M_eff, 1.0)  # mobility ratio ≥ 1 for unfavourable CO2 flooding

        # Step 2: Heterogeneity factor H = 1/(1-V_DP)²  (paper Eq.)
        H = 1.0 / (1.0 - v_dp) ** 2

        # Step 3: Kowal effective viscosity ratio E_eff = (0.78 + 0.22·M^0.25)^4
        E_eff = (0.78 + 0.22 * (M_eff ** 0.25)) ** 4

        # Step 4: Koval factor K = H · E_eff
        K = max(H * E_eff, 1.0 + EPSILON)

        # Step 5: Welge-based miscible RF (paper Eq. RF_mis)
        # RF_mis = (3K² - 3K + 1) / K³ · (1 - S_wi)
        welge = (3.0 * K**2 - 3.0 * K + 1.0) / (K**3)
        rf_mis = welge * (1.0 - s_wi)

        return float(np.clip(rf_mis, 0.0, 1.0))


# =============================================================================
# SECTION 7: IMMISCIBLE RECOVERY MODEL
# Paper: RF_im = (S_or_base - S_or*) / (1 - S_wi) · E_A · E_V
#        S_or* via capillary number (paper Section 2.3)
# =============================================================================

class ImmiscibleRecoveryModel(RecoveryModel):
    """
    Immiscible CO2-EOR recovery factor.

    Paper Eq.:
        N_c    = v · μ_CO2 / σ_CO2-oil
        σ(p)   = σ_0 · exp(-λ·(p-MMP)) for p≥MMP, else σ_0
        S_or*  = S_or_base / (1 + (N_c/N_c_crit)^B)
        RF_im  = (S_or_base - S_or*) / (1 - S_wi) · E_A · E_V

    where E_A = areal sweep, E_V = vertical sweep.
    """

    def __init__(self, max_rf_cap: float = 0.50, **kwargs):
        super().__init__(max_rf_cap=max_rf_cap, **kwargs)

    def calculate(self, **kwargs) -> float:
        p = kwargs

        # Fluid properties
        mu_co2_cp = float(p.get("viscosity_inj", p.get("co2_viscosity", p.get("mu_co2", 0.05))))
        mu_co2_poise = mu_co2_cp * VISC_CP_TO_POISE

        # Flow parameters
        rate_cm3_s = float(p.get("rate", 5000.0)) * RATE_STB_DAY_TO_CM3_S
        porosity = float(p.get("porosity", 0.2))
        cross_area_cm2 = float(p.get("cross_sectional_area", 10000.0))  # cm²
        interstitial_vel = rate_cm3_s / max(cross_area_cm2 * porosity, EPSILON)

        # IFT model: σ(p) = σ_0 · exp(-λ·(p-MMP))  (paper Eq.)
        pressure = float(p.get("pressure", 0.0))
        mmp = float(p.get("mmp", 2000.0))
        sigma_0 = float(p.get("sigma_0", 20.0))        # mN/m, paper default
        lambda_ift = float(p.get("lambda_ift", 0.001))  # psi⁻¹, paper default
        sigma = interfacial_tension(pressure, mmp, sigma_0, lambda_ift)

        # Capillary number N_c = v · μ_CO2 / σ  (paper Eq.)
        nc = capillary_number(interstitial_vel, mu_co2_poise, sigma)

        # Residual oil saturation reduction (paper Eq. with N_c_crit=1e-5, B=0.5)
        sor_base = float(p.get("sor", p.get("sor_base", 0.35)))
        nc_crit = float(p.get("nc_crit", 1e-5))
        b_exp = float(p.get("nc_exponent", 0.5))
        sor_star = residual_oil_saturation(sor_base, nc, nc_crit, b_exp)

        s_wi = float(np.clip(p.get("s_wi", p.get("connate_water_saturation", 0.25)), 0.0, 0.8))

        # Displacement efficiency from Sor reduction (paper Eq. RF_im)
        denominator = max(1.0 - s_wi, EPSILON)
        displacement_eff = (sor_base - sor_star) / denominator

        # Vertical sweep efficiency E_V (via Dykstra-Parsons coefficient)
        v_dp = float(np.clip(p.get("v_dp", 0.6), 0.0, 0.99))
        # Simple approximation: E_V ~ 1 - V_DP^0.7 (consistent with Johnson chart asymptote)
        e_v = 1.0 - (v_dp ** 0.7)

        # Areal sweep efficiency E_A (mobility-based, standard for CO2-EOR)
        mu_oil = float(p.get("viscosity_oil", p.get("mu_oil", 1.5)))
        mobility_ratio = mu_oil / max(mu_co2_cp, EPSILON)
        if mobility_ratio <= 1.0:
            e_a = 1.0
        elif mobility_ratio <= 10.0:
            e_a = 0.5 + 0.4 * np.log10(mobility_ratio) / (mobility_ratio - 1.0)
        else:
            e_a = 0.546 + 0.0357 / mobility_ratio
        e_a = float(np.clip(e_a, 0.1, 1.0))

        # Final immiscible RF  (paper Eq.: RF_im = (Sor_base-Sor*)/(1-Swi) · E_A · E_V)
        rf_im = displacement_eff * e_a * e_v
        return float(np.clip(rf_im, 0.0, 1.0))


# =============================================================================
# SECTION 8: BUCKLEY-LEVERETT (analytical, preserved)
# =============================================================================

class BuckleyLeverettModel(RecoveryModel):
    """
    Buckley-Leverett displacement efficiency using Welge tangent construction.
    """

    def calculate(self, **kwargs) -> float:
        mu_oil = kwargs.get("viscosity_oil")
        mu_inj = kwargs.get("viscosity_inj")
        sor = kwargs.get("sor")
        s_gc = kwargs.get("s_gc", 0.0)
        n_o = kwargs.get("n_o", 2.0)
        n_g = kwargs.get("n_g", 2.0)

        if any(v is None for v in [mu_oil, mu_inj, sor]):
            raise ValueError(
                "Buckley-Leverett model requires 'viscosity_oil', 'viscosity_inj', and 'sor'."
            )

        mobility_ratio = mu_oil / (mu_inj + EPSILON)

        def fractional_flow(s_g):
            s_star = np.clip((s_g - s_gc) / (1 - sor - s_gc), 0, 1)
            k_ro = (1 - s_star) ** n_o
            k_rg = s_star ** n_g
            return 1.0 / (1.0 + (k_ro / (k_rg + EPSILON)) * (1 / mobility_ratio))

        s_g_range = np.linspace(s_gc, 1 - sor, 500)
        f_g = fractional_flow(s_g_range)

        # Welge tangent construction to find shock front saturation
        tangent_slope = f_g / (s_g_range - s_gc + EPSILON)
        front_idx = np.argmax(tangent_slope[1:]) + 1
        s_gf = s_g_range[front_idx]

        # Recovery at breakthrough
        e_d_bt = (s_gf - s_gc) / (1 - s_gc)
        return float(np.clip(e_d_bt, 0.0, 1.0))


# =============================================================================
# SECTION 9: DYKSTRA-PARSONS (analytical, preserved)
# =============================================================================

class DykstraParsonsModel(RecoveryModel):
    """
    Vertical sweep efficiency for layered reservoir (Dykstra-Parsons).
    """

    def calculate(self, **kwargs) -> float:
        layers = kwargs.get("layer_definitions")
        mobility_ratio = kwargs.get("mobility_ratio")
        base_perm = kwargs.get("permeability")

        if not layers or mobility_ratio is None or base_perm is None:
            raise ValueError(
                "Dykstra-Parsons model requires 'layer_definitions', 'mobility_ratio', and base 'permeability'."
            )

        sorted_layers = sorted(layers, key=lambda l: l.permeability_multiplier, reverse=True)
        perms = np.array([layer.permeability_multiplier * base_perm for layer in sorted_layers])

        log_perms = np.log(perms)
        std_dev_log_k = np.std(log_perms)
        v_dp = 1 - np.exp(-std_dev_log_k)
        v_dp = np.clip(v_dp, 0.0, 0.999)

        def _dp_solver(v_dp, mobility_ratio, wor_limit=1.0):
            if v_dp <= 1e-6:
                return 1.0

            def equation_to_solve(ev):
                if ev <= 0 or ev >= 1:
                    return 1e6
                a = -0.571 - 0.614 * np.log10(mobility_ratio) - 0.083 * (np.log10(mobility_ratio)) ** 2
                b = 1.122 + 0.081 * np.log10(mobility_ratio) + 0.031 * (np.log10(mobility_ratio)) ** 2
                wor_calc = (1 / (b * (1 - v_dp) ** a)) * ((1 / ev) ** b - 1)
                return wor_calc - wor_limit

            try:
                c_guess = 0.5 + 0.05 * np.log(mobility_ratio) if mobility_ratio > 1 else 0.5
                initial_guess = 1 - v_dp ** c_guess
                (solution,) = fsolve(equation_to_solve, initial_guess)
                vertical_sweep = np.clip(solution, 0.0, 1.0)
            except Exception:
                c = 0.5 + 0.05 * np.log(mobility_ratio) if mobility_ratio > 1 else 0.5
                vertical_sweep = 1 - v_dp ** c
            return vertical_sweep

        vertical_sweep = _dp_solver(v_dp, mobility_ratio)
        soi = kwargs.get("soi", 0.8)
        sor = kwargs.get("sor", 0.25)
        displacement_sweep = (soi - sor) / soi if soi > 0 else 0.0
        return float(np.clip(vertical_sweep * displacement_sweep, 0.0, 1.0))


# =============================================================================
# SECTION 10: KOVAL MODEL (standalone, preserved for compatibility)
# =============================================================================

class KovalRecoveryModel(RecoveryModel):
    """
    Koval miscible sweep efficiency model.
    Preserved for backward compatibility; MiscibleRecoveryModel uses Koval/Welge.
    """

    def calculate(self, **kwargs) -> float:
        v_dp = np.clip(kwargs.get("v_dp", 0.5), 0.0, 0.9999)
        M = max(kwargs.get("mobility_ratio", 10.0), EPSILON)
        hk = (1.0 / (1.0 - v_dp)) ** 2
        kv = hk * (0.78 + 0.22 * M ** 0.25) ** 4
        kv = max(kv, EPSILON)
        if abs(M - 1.0) < EPSILON:
            sweep_efficiency = (
                1.0 if abs(kv - 1.0) < EPSILON else (1.0 - np.exp(1.0 - kv)) / (kv - 1.0)
            )
        else:
            c = 1.0 / (M - 1.0)
            if abs(kv - 1.0) < EPSILON:
                sweep_efficiency = (1.0 - np.exp(-c)) / c
            else:
                term1 = (1.0 - np.exp(1.0 - kv)) / (kv - 1.0)
                term2 = (1.0 - np.exp(c * (1.0 - kv))) / (c * (kv - 1.0))
                sweep_efficiency = term1 - (term1 - term2) / (M - 1.0)
        return float(np.clip(sweep_efficiency, 0.0, 1.0))


# =============================================================================
# SECTION 11: HYBRID RECOVERY MODEL
# Paper: RF_total = ω · RF_mis + (1-ω) · RF_im  (paper Eq. RF_total)
# =============================================================================

class HybridRecoveryModel(RecoveryModel):
    """
    Hybrid CO2-EOR recovery model combining miscible and immiscible regimes.

    Paper Eq.:
        RF_total(P, u) = ω(P, χ) · RF_mis(u) + (1-ω(P, χ)) · RF_im(u)

    where ω is the sigmoid weighting function (paper Section 2.1).
    """

    def __init__(
        self,
        miscible_model: RecoveryModel,
        immiscible_model: RecoveryModel,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.miscible_model = miscible_model
        self.immiscible_model = immiscible_model
        self.transition_engine = TransitionEngine(**kwargs)

    def calculate(self, **kwargs) -> float:
        mmp = kwargs.get("mmp", 2000.0)
        pressure = kwargs.get("pressure", 0.0)
        p_mmp_ratio = pressure / max(mmp, EPSILON)
        c7_plus = kwargs.get("c7_plus_fraction", 0.3)

        # Sigmoid weighting (paper Eq. ω(P_r, χ))
        w_miscible = self.transition_engine.calculate_weight(p_mmp_ratio, c7_plus)

        # Component RFs
        rf_mis = self.miscible_model._bounded_calculate(**kwargs)
        rf_im = self.immiscible_model._bounded_calculate(**kwargs)

        # Paper Eq.: RF_total = ω·RF_mis + (1-ω)·RF_im
        rf = w_miscible * rf_mis + (1.0 - w_miscible) * rf_im
        return float(np.clip(rf, 0.0, self.max_rf_cap))


# =============================================================================
# SECTION 12: PUBLIC API
# =============================================================================

def get_model_instance(model_name: str, **kwargs) -> RecoveryModel:
    """Factory function to create and return an instance of a recovery model."""
    model_map: Dict[str, Type[RecoveryModel]] = {
        "miscible": MiscibleRecoveryModel,
        "immiscible": ImmiscibleRecoveryModel,
        "koval": KovalRecoveryModel,
        "buckley_leverett": BuckleyLeverettModel,
        "dykstra_parsons": DykstraParsonsModel,
    }

    if model_name in model_map:
        return model_map[model_name](**kwargs)

    if model_name == "hybrid":
        miscible_model = MiscibleRecoveryModel(**kwargs)
        immiscible_model = ImmiscibleRecoveryModel(**kwargs)
        return HybridRecoveryModel(miscible_model, immiscible_model, **kwargs)

    raise ValueError(f"Unsupported recovery model: '{model_name}'")


def recovery_factor(model: str, **kwargs) -> float:
    """
    Public API: calculate the recovery factor using a specified model.

    Input normalization is applied to handle various key naming conventions.
    All physics formulas are from the PhD paper — no data fitting.
    """
    # --- Input normalization ---
    if "v_dp_coefficient" in kwargs:
        kwargs["v_dp"] = kwargs.pop("v_dp_coefficient")
    if "oil_viscosity" in kwargs:
        kwargs["viscosity_oil"] = kwargs.pop("oil_viscosity")
    if "oil_viscosity_cp" in kwargs:
        kwargs["viscosity_oil"] = kwargs.pop("oil_viscosity_cp")
    if "mu_oil" in kwargs:
        kwargs["viscosity_oil"] = kwargs.pop("mu_oil")

    if "viscosity_oil" not in kwargs:
        kwargs["viscosity_oil"] = 1.5  # paper Table 3 default (Wasson oil)

    if "co2_viscosity" in kwargs:
        kwargs["viscosity_inj"] = kwargs.pop("co2_viscosity")

    # Ensure mobility_ratio is calculated if not present
    if "mobility_ratio" not in kwargs:
        kwargs["mobility_ratio"] = kwargs["viscosity_oil"] / (
            kwargs.get("viscosity_inj", 0.05) + EPSILON
        )

    try:
        if model == "layered":
            layers = kwargs["layer_definitions"]
            base_model_name = kwargs.get("base_model_for_layered", "hybrid")

            layer_kh, layer_pv = [], []
            base_perm = kwargs.get("permeability")
            if base_perm is None:
                raise ValueError("Base 'permeability' must be provided for LayeredRecoveryModel.")

            for layer in layers:
                k_layer = base_perm * layer.permeability_multiplier
                h_layer = layer.thickness
                phi_layer = layer.porosity
                layer_kh.append(k_layer * h_layer)
                layer_pv.append(h_layer * phi_layer)

            total_kh, total_pv = sum(layer_kh), sum(layer_pv)
            if total_kh < EPSILON or total_pv < EPSILON:
                return 0.0

            rate_fractions = [kh / total_kh for kh in layer_kh]
            pv_fractions = [pv / total_pv for pv in layer_pv]

            total_rf = 0.0
            total_rate = kwargs.get("rate", 0)

            for i, layer in enumerate(layers):
                layer_params = deepcopy(kwargs)
                layer_params.update(layer.param_overrides)
                layer_params["rate"] = total_rate * rate_fractions[i]
                layer_params["porosity"] = layer.porosity
                layer_params["permeability"] = base_perm * layer.permeability_multiplier

                layer_rf = recovery_factor(base_model_name, **layer_params)
                total_rf += layer_rf * pv_fractions[i]

            return float(total_rf)

        model_instance = get_model_instance(model, **kwargs)
        return model_instance._bounded_calculate(**kwargs)

    except (TypeError, ValueError, KeyError) as e:
        logger.error(f"Error in recovery_factor model '{model}': {e}", exc_info=True)
        return 0.0
