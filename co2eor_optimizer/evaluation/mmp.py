"""
Minimum Miscibility Pressure (MMP) calculation module

This module provides functions to estimate MMP using several industry-standard
correlations. It includes robust input validation and can estimate required
parameters from PVT data when direct measurements are not available.
"""
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import PVTProperties


# Configure logging to ensure messages are displayed
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class MMPParameters:
    """
    Parameters for MMP calculation.

    Attributes:
        temperature: Reservoir temperature in °F (valid range: 70-300°F).
        oil_gravity: Oil API gravity (valid range: 15-50°API).
        c7_plus_mw: Molecular weight of C7+ fraction (150-300 g/mol).
        injection_gas_composition: Mole fractions of injection gas (e.g., {'CO2': 0.95, 'CH4': 0.05}).
        pvt_data: Optional PVT properties from the core module.
    """
    temperature: float
    oil_gravity: float
    c7_plus_mw: Optional[float] = None
    injection_gas_composition: Optional[dict] = None
    pvt_data: Optional[PVTProperties] = None

    def __post_init__(self):
        """Validate input parameters with warnings for near-boundary values."""
        # Temperature validation
        if not 70 <= self.temperature <= 300:
            raise ValueError(
                f"Temperature {self.temperature}°F is outside the typical valid range (70-300°F). "
                "MMP correlations are unreliable outside this range."
            )
        if self.temperature < 100 or self.temperature > 250:
            logging.warning(
                f"Temperature {self.temperature}°F is near the correlation limits. "
                "Results may have reduced accuracy."
            )

        # Oil gravity validation
        if not 15 <= self.oil_gravity <= 50:
            raise ValueError(
                f"Oil gravity {self.oil_gravity}°API is outside the typical valid range (15-50°API). "
                "MMP correlations are unreliable for heavy oils (<20°API) or condensates (>45°API)."
            )
        if self.oil_gravity < 20 or self.oil_gravity > 45:
            logging.warning(
                f"Oil gravity {self.oil_gravity}°API is near the correlation limits. "
                "Results may have reduced accuracy."
            )

        # C7+ molecular weight validation
        if self.c7_plus_mw and not 150 <= self.c7_plus_mw <= 300:
            raise ValueError(
                f"C7+ MW {self.c7_plus_mw} is outside the valid range (150-300 g/mol). "
                "Provide measured values for accurate MMP estimation."
            )

        # Gas composition validation
        if self.injection_gas_composition:
            total = sum(self.injection_gas_composition.values())
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"Gas composition fractions sum to {total:.3f}, but should be 1.0. "
                    "Please normalize the fractions before calculation."
                )

def calculate_mmp_cronquist(params: MMPParameters) -> float:
    """
    Calculate MMP using the Cronquist correlation (1978).

    This correlation is primarily valid for pure CO2 injection.
    Formula: MMP = 15.988 * (T^0.744206) * (API^0.279033)
    """
    return 15.988 * (params.temperature ** 0.744206) * (params.oil_gravity ** 0.279033)

def calculate_mmp_hybrid_gh(params: MMPParameters) -> float:
    """
    Calculate MMP using a hybrid approach.

    - Base MMP from Cronquist (1978) for pure CO2.
    - Adjustment for C7+ molecular weight from Glaso (1985).
    - Adjustment for gas composition from Yellig & Metcalfe (1980).

    This method is suitable for oils with known C7+ molecular weight and
    for injection gases with varying CO2 concentrations.
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight is required for the hybrid_gh method.")

    # 1. Base MMP from Cronquist (1978)
    mmp_base = calculate_mmp_cronquist(params)

    # 2. Adjust for C7+ molecular weight (Glaso 1985)
    mmp_adj_c7 = mmp_base * (1.0 - 0.007 * (params.c7_plus_mw - 190))

    # 3. Adjust for gas composition (Yellig & Metcalfe 1980)
    if params.injection_gas_composition:
        co2_fraction = params.injection_gas_composition.get('CO2', 0.0)
        # Linear interpolation between pure CO2 and 80% CO2 MMPs
        return mmp_adj_c7 * (1.0 - 0.25 * (1.0 - co2_fraction))

    return mmp_adj_c7

def calculate_mmp_yuan(params: MMPParameters) -> float:
    """
    Calculate MMP using the Yuan et al. correlation (2005).

    This correlation is generally better for impure CO2 injection cases.
    """
    if not params.injection_gas_composition:
        raise ValueError("Gas composition is required for the Yuan correlation.")

    co2_fraction = params.injection_gas_composition.get('CO2', 0.0)
    c1_fraction = params.injection_gas_composition.get('CH4', 0.0)

    # Yuan correlation coefficients
    a = 8.784 * (params.temperature ** 0.523)
    b = 0.147 * (params.oil_gravity ** 1.226)
    c = 1.0 - 0.8 * (1.0 - co2_fraction)  # CO2 purity factor
    d = 1.0 + 0.1 * c1_fraction  # Methane adjustment

    return a * b * c * d

def calculate_mmp(params: Union[MMPParameters, PVTProperties], method: str = 'auto') -> float:
    """
    Unified MMP calculation interface that selects the best method automatically.

    Args:
        params: An object containing the required fluid and reservoir properties.
                Can be MMPParameters or PVTProperties.
        method: The correlation to use: 'cronquist', 'hybrid_gh', 'yuan', or 'auto'.
                'auto' selects the most appropriate method based on available data.

    Returns:
        The calculated Minimum Miscibility Pressure (MMP) in psi.
    """
    mmp_params: MMPParameters

    if isinstance(params, PVTProperties):
        # If PVTProperties are passed, create MMPParameters internally.
        logging.info("PVTProperties object provided. Attempting to estimate API gravity.")
        try:
            api_gravity = estimate_api_from_pvt(params)
            logging.warning(
                "API gravity was estimated from PVT properties using Standing's correlation. "
                "This approximation has limited accuracy (±5°API). Use measured values for critical applications."
            )
            mmp_params = MMPParameters(
                temperature=params.temperature,
                oil_gravity=api_gravity,
                pvt_data=params
            )
        except Exception as e:
            raise ValueError(
                "API gravity estimation from PVT failed. Please provide a measured oil API gravity. "
                f"Original error: {str(e)}"
            )
    elif isinstance(params, MMPParameters):
        mmp_params = params
    else:
        raise TypeError(f"Unsupported type for 'params': {type(params).__name__}. Must be MMPParameters or PVTProperties.")

    # Select and run the appropriate calculation method
    if method == 'auto':
        if mmp_params.injection_gas_composition and mmp_params.injection_gas_composition.get('CO2', 0.0) < 0.9:
            logging.info("Auto-selecting Yuan correlation for impure CO2 stream.")
            return calculate_mmp_yuan(mmp_params)
        elif mmp_params.c7_plus_mw:
            logging.info("Auto-selecting Hybrid_GH correlation due to presence of C7+ MW.")
            return calculate_mmp_hybrid_gh(mmp_params)
        else:
            logging.info("Auto-selecting Cronquist correlation as a baseline.")
            return calculate_mmp_cronquist(mmp_params)
    elif method == 'cronquist':
        return calculate_mmp_cronquist(mmp_params)
    elif method == 'hybrid_gh':
        return calculate_mmp_hybrid_gh(mmp_params)
    elif method == 'yuan':
        return calculate_mmp_yuan(mmp_params)
    else:
        raise ValueError(f"Unknown MMP calculation method: '{method}'. Available methods: 'auto', 'cronquist', 'hybrid_gh', 'yuan'.")


def estimate_api_from_pvt(pvt: PVTProperties) -> float:
    """
    Estimate API gravity from PVT properties using Standing's correlation (1947).

    WARNING: This is an approximation with limited accuracy (±5°API).
    Use measured API gravity values for critical applications.

    Reference:
    Standing, M.B., 1947. A Pressure-Volume-Temperature Correlation for Mixtures
    of California Oils and Gases. Drilling and Production Practice, API, pp. 275-287.
    """
    if not all([hasattr(pvt, attr) and pvt.rs is not None and pvt.rs.size > 0 for attr in ['rs', 'gas_specific_gravity', 'temperature']]):
        raise ValueError("PVTProperties must contain 'rs', 'gas_specific_gravity', and 'temperature' data.")

    # Use properties from the PVT object
    R_s = pvt.rs[0]  # Solution GOR at standard conditions
    γ_g = pvt.gas_specific_gravity
    T = pvt.temperature

    # Iteratively solve for oil specific gravity (γ_o)
    γ_o = 0.85  # Initial guess
    converged = False
    for _ in range(15):  # Max 15 iterations for convergence
        F = R_s * (γ_g / γ_o)**0.5 + 1.25 * T
        new_γ_o = 0.972 + 0.000147 * F**1.175

        # Check for convergence
        if abs(new_γ_o - γ_o) < 1e-4:
            γ_o = new_γ_o
            converged = True
            break
        γ_o = new_γ_o

    if not converged:
        logging.warning(
            "API gravity estimation from PVT did not converge after 15 iterations. "
            "The resulting value may have reduced accuracy."
        )

    # Convert specific gravity to API and clamp to a valid range
    api = (141.5 / γ_o) - 131.5
    return max(15.0, min(50.0, api))