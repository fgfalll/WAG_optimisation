"""
Minimum Miscibility Pressure (MMP) Calculation Module

This module provides functions to estimate the Minimum Miscibility Pressure (MMP)
for gas injection projects using several industry-standard correlations.

Key Features:
- Robust input validation within specified correlation limits.
- A unified interface that can automatically select the most appropriate
  correlation based on the available data.
- The ability to estimate required parameters (like API gravity) from
  standard PVT data when direct measurements are unavailable, with clear
  warnings about the associated uncertainty.
"""
import logging
import sys
import os
from typing import Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.data_models import PVTProperties
except ImportError as e:
    raise ImportError(
        "Could not import 'PVTProperties' from 'core.py'. "
        f"Please ensure 'core.py' is located in the project root directory: '{project_root}'"
    ) from e


# Configure logging to ensure messages are displayed clearly to the user.
logging.basicConfig(level=logging.INFO, format='%(levelname)s - MMP_CALC: %(message)s')


@dataclass
class MMPParameters:
    """
    A dataclass to hold all necessary parameters for MMP calculation.

    Input validation is performed automatically upon instantiation to ensure
    that parameters fall within the typical validity ranges of the correlations.

    Attributes:
        temperature (float): Reservoir temperature in degrees Fahrenheit (°F).
            (Typical valid range: 70-300°F).
        oil_gravity (float): Oil gravity in degrees API (°API).
            (Typical valid range: 15-50°API).
        c7_plus_mw (Optional[float]): The molecular weight of the C7+ fraction
            of the reservoir fluid, in g/mol. Required for some correlations
            like 'hybrid_gh'. (Typical valid range: 150-300 g/mol).
        injection_gas_composition (Optional[Dict[str, float]]): A dictionary
            containing the mole fractions of the injection gas components.
            The sum of fractions must be 1.0. Example: {'CO2': 0.95, 'CH4': 0.05}.
        pvt_data (Optional[PVTProperties]): An optional PVTProperties object
            from the 'core' module. This is used if API gravity needs to be
            estimated.
    """
    temperature: float
    oil_gravity: float
    c7_plus_mw: Optional[float] = None
    injection_gas_composition: Optional[Dict[str, float]] = None
    pvt_data: Optional[PVTProperties] = None

    def __post_init__(self):
        """
        Performs validation of input parameters after initialization.
        Raises ValueError for out-of-range inputs and provides warnings
        for values near the boundaries of correlation validity.
        """
        # Validate Reservoir Temperature
        if not 70 <= self.temperature <= 300:
            raise ValueError(
                f"Temperature {self.temperature}°F is outside the typical valid "
                f"range (70-300°F). MMP correlations are unreliable here."
            )
        if self.temperature < 100 or self.temperature > 250:
            logging.warning(
                f"Temperature {self.temperature}°F is near correlation limits. "
                "Accuracy of the results may be reduced."
            )

        # Validate Oil API Gravity
        if not 15 <= self.oil_gravity <= 50:
            raise ValueError(
                f"Oil gravity {self.oil_gravity}°API is outside the typical "
                f"valid range (15-50°API). Correlations are not validated for "
                f"heavy oils (<20°API) or light oils/condensates (>45°API)."
            )
        if self.oil_gravity < 20 or self.oil_gravity > 45:
            logging.warning(
                f"Oil gravity {self.oil_gravity}°API is near correlation limits. "
                "Accuracy of the results may be reduced."
            )

        # Validate C7+ Molecular Weight if provided
        if self.c7_plus_mw and not 150 <= self.c7_plus_mw <= 300:
            raise ValueError(
                f"C7+ MW {self.c7_plus_mw} g/mol is outside the typical valid "
                f"range (150-300 g/mol). Please provide measured values."
            )

        # Validate Injection Gas Composition if provided
        if self.injection_gas_composition:
            total_fraction = sum(self.injection_gas_composition.values())
            if not np.isclose(total_fraction, 1.0):
                raise ValueError(
                    f"Sum of gas composition fractions is {total_fraction:.4f}, "
                    "but must be 1.0. Please normalize the fractions."
                )

def _calculate_mmp_cronquist(params: MMPParameters) -> float:
    """
    Calculates MMP using the Cronquist correlation (1978).

    This is a simple correlation primarily valid for pure CO2 injection in
    oils within the specified temperature and gravity ranges.

    Formula: MMP [psi] = 15.988 * (Temperature [°F] ^ 0.744206) * (API Gravity ^ 0.279033)
    """
    mmp = 15.988 * (params.temperature ** 0.744206) * (params.oil_gravity ** 0.279033)
    return mmp

def _calculate_mmp_hybrid_gh(params: MMPParameters) -> float:
    """
    Calculates MMP using a hybrid approach combining multiple correlations.

    1.  Base MMP is calculated using Cronquist (1978) for pure CO2.
    2.  An adjustment is made for the C7+ molecular weight based on Glaso (1985).
    3.  A simplified adjustment is made for gas impurities based on
        Yellig & Metcalfe (1980).

    This method is suitable for oils with known C7+ MW and for injection gases
    with high CO2 concentrations.
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight is required for the 'hybrid_gh' method.")

    # 1. Base MMP from Cronquist
    mmp_base = _calculate_mmp_cronquist(params)

    # 2. Adjust for C7+ molecular weight (Glaso, 1985)
    # This term reduces MMP for lighter oils (higher C7+ MW is not typical)
    # and increases it for heavier oils.
    mmp_adj_c7 = mmp_base * (1.0 - 0.007 * (params.c7_plus_mw - 190))

    # 3. Adjust for gas composition (Yellig & Metcalfe, 1980)
    if params.injection_gas_composition:
        co2_fraction = params.injection_gas_composition.get('CO2', 0.0)
        # Simplified linear interpolation for the effect of impurities.
        # Assumes MMP increases as CO2 purity decreases. A sensitivity factor
        # of 0.25 is a common simplification for hydrocarbon impurities.
        IMPURITY_SENSITIVITY_FACTOR = 0.25
        return mmp_adj_c7 / (1.0 - IMPURITY_SENSITIVITY_FACTOR * (1.0 - co2_fraction))

    return mmp_adj_c7

def _calculate_mmp_yuan(params: MMPParameters) -> float:
    """
    Calculates MMP using the Yuan et al. correlation (2005).

    This correlation is generally considered more robust for impure CO2 streams,
    as it explicitly accounts for the mole fractions of CO2 and Methane (C1).
    """
    if not params.injection_gas_composition:
        raise ValueError("Gas composition is required for the 'yuan' correlation.")

    co2_fraction = params.injection_gas_composition.get('CO2', 0.0)
    ch4_fraction = params.injection_gas_composition.get('CH4', 0.0)

    # Yuan correlation coefficients
    # A: Temperature dependency term
    a = 10**(3.356 + 0.0016 * params.temperature - 0.0000033 * params.temperature**2)
    # B: Oil composition term (C5+ Mol weight is approximated from API)
    m_c5_plus = 630 - 10.3 * params.oil_gravity
    b = (0.342 * m_c5_plus**0.36) / (0.641 * params.temperature**0.21)
    # C: Purity term (accounts for non-CO2 components)
    x_co2 = co2_fraction
    c = 0.993 - 0.778 * (1 - x_co2)**0.11
    
    # Final MMP calculation in MPa, then converted to psi
    mmp_mpa = a * (b**x_co2) * c
    return mmp_mpa * 145.038  # Convert MPa to psi

def _calculate_mmp_alston(params: MMPParameters) -> float:
    """
    Calculates MMP using the Alston et al. correlation (1985).

    This method is robust for impure gas streams containing N2, CH4, and CO2.
    It adjusts the pure-CO2 MMP based on the pseudo-critical temperature of
    the injection gas mixture.

    Requires C7+ MW for an exponent and gas composition for T_pc.
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight is required for the 'alston' method.")
    if not params.injection_gas_composition:
        raise ValueError("Gas composition is required for the 'alston' correlation.")

    # Critical temperatures of common components in Kelvin
    CRITICAL_TEMPS_K = {'CO2': 304.1, 'CH4': 190.6, 'N2': 126.2}
    
    y_co2 = params.injection_gas_composition.get('CO2', 0.0)
    y_ch4 = params.injection_gas_composition.get('CH4', 0.0)
    y_n2 = params.injection_gas_composition.get('N2', 0.0)

    # 1. Calculate pseudo-critical temperature (Tpc) of the gas mixture
    tpc_k = y_co2 * CRITICAL_TEMPS_K['CO2'] + \
            y_ch4 * CRITICAL_TEMPS_K['CH4'] + \
            y_n2 * CRITICAL_TEMPS_K['N2']

    # 2. Calculate MMP for pure CO2 using Yellig & Metcalfe (1980) as a base
    T_F = params.temperature
    mmp_pure_co2 = 1016 + 4.773*T_F - 0.00946*(T_F**2) + 0.000021*(T_F**3)

    # 3. Calculate the Alston exponent 'A'
    exponent_A = 2.41 - 0.00284 * params.c7_plus_mw

    # 4. Calculate the final MMP for the impure gas
    mmp_impure = mmp_pure_co2 * (tpc_k / CRITICAL_TEMPS_K['CO2'])**exponent_A
    return mmp_impure

# --- [UPDATED] Dictionary mapping method names to functions for UI and internal use ---
MMP_METHODS: Dict[str, Callable[[MMPParameters], float]] = {
    'cronquist': _calculate_mmp_cronquist,
    'hybrid_gh': _calculate_mmp_hybrid_gh,
    'yuan': _calculate_mmp_yuan,
    'alston': _calculate_mmp_alston,
}


def estimate_api_from_pvt(pvt: PVTProperties) -> float:
    """
    Estimates oil API gravity from PVT properties using Standing's correlation (1947).

    CRITICAL WARNING: This is a rough approximation and can have significant
    error (often ±5 °API or more). The result is highly sensitive to the
    input PVT data. It should ONLY be used for preliminary screening when no
    measured oil gravity is available.

    The function uses an iterative method to solve for oil specific gravity.
    """
    required_attrs = ['rs', 'gas_specific_gravity', 'temperature']
    if not all(hasattr(pvt, attr) and getattr(pvt, attr) is not None for attr in required_attrs):
        raise ValueError(
            "PVTProperties must contain 'rs', 'gas_specific_gravity', and "
            "'temperature' data to estimate API gravity."
        )
    if pvt.rs.size == 0:
         raise ValueError("'rs' (solution GOR) in PVTProperties cannot be empty.")

    # Use properties from the PVT object at the first data point (e.g., bubble point)
    R_s = pvt.rs[0]
    gamma_g = pvt.gas_specific_gravity
    T = pvt.temperature

    # Iteratively solve for oil specific gravity (gamma_o)
    gamma_o = 0.85  # Initial guess for a typical crude oil
    converged = False
    for i in range(20):  # Max 20 iterations for convergence
        F = R_s * (gamma_g / gamma_o)**0.5 + 1.25 * T
        gamma_o_new = 0.972 + 0.000147 * F**1.175

        if abs(gamma_o_new - gamma_o) < 1e-5:
            gamma_o = gamma_o_new
            converged = True
            break
        gamma_o = gamma_o_new

    if not converged:
        logging.warning(
            "API gravity estimation from PVT did not converge after 20 iterations. "
            "The resulting value is highly uncertain."
        )

    # Convert oil specific gravity to API and clamp to a valid range
    api = (141.5 / gamma_o) - 131.5
    return max(15.0, min(50.0, api))


def calculate_mmp(
    params: Union[MMPParameters, PVTProperties],
    method: str = 'auto'
) -> float:
    """
    Unified MMP calculation interface.

    This function can take either a pre-filled MMPParameters object or a
    PVTProperties object. If a PVTProperties object is provided, it will
    attempt to estimate the required oil API gravity.

    Args:
        params (Union[MMPParameters, PVTProperties]): An object containing the
            required fluid and reservoir properties.
        method (str): The correlation to use. Can be 'cronquist', 'hybrid_gh',
            'yuan', 'alston', or 'auto'. 'auto' mode selects the most appropriate
            method based on the available data.

    Returns:
        float: The calculated Minimum Miscibility Pressure (MMP) in psi.
    """
    mmp_params: MMPParameters

    if isinstance(params, PVTProperties):
        logging.info("PVTProperties object provided. Estimating API gravity from PVT data.")
        try:
            api_gravity = estimate_api_from_pvt(params)
            logging.critical(
                "CRITICAL WARNING: API gravity was estimated from PVT properties using "
                "Standing's correlation. This is a rough approximation with limited "
                "accuracy (error can exceed ±5°API). Use measured oil gravity "
                "for reliable MMP calculations."
            )
            # Create MMPParameters with estimated gravity and other available PVT data
            mmp_params = MMPParameters(
                temperature=params.temperature,
                oil_gravity=api_gravity,
                injection_gas_composition=getattr(params, 'injection_gas_composition', None),
                c7_plus_mw=getattr(params, 'c7_plus_mw', None),
                pvt_data=params
            )
        except Exception as e:
            raise ValueError(
                "Failed to estimate API gravity from PVTProperties. "
                f"Please provide a measured oil gravity. Original error: {e}"
            )
    elif isinstance(params, MMPParameters):
        mmp_params = params
    else:
        raise TypeError(
            f"Unsupported type for 'params': {type(params).__name__}. "
            "Must be MMPParameters or PVTProperties."
        )

    # --- [REFACTORED] Method Selection and Calculation ---
    logging.info(f"Calculating MMP with method: '{method}'.")
    if method == 'auto':
        # Intelligent selection based on data richness
        if mmp_params.c7_plus_mw and mmp_params.injection_gas_composition and \
           mmp_params.injection_gas_composition.get('CO2', 0.0) < 0.98:
            logging.info("Auto-selecting 'alston' correlation for impure gas with known C7+ MW.")
            return _calculate_mmp_alston(mmp_params)
        elif mmp_params.injection_gas_composition and mmp_params.injection_gas_composition.get('CO2', 0.0) < 0.95:
            logging.info("Auto-selecting 'yuan' correlation for impure CO2 stream.")
            return _calculate_mmp_yuan(mmp_params)
        elif mmp_params.c7_plus_mw:
            logging.info("Auto-selecting 'hybrid_gh' correlation due to presence of C7+ MW.")
            return _calculate_mmp_hybrid_gh(mmp_params)
        else:
            logging.info("Auto-selecting 'cronquist' correlation as a baseline for pure CO2.")
            return _calculate_mmp_cronquist(mmp_params)

    # Dynamic dispatch using the MMP_METHODS dictionary
    calculation_func = MMP_METHODS.get(method)
    if calculation_func:
        return calculation_func(mmp_params)
    else:
        raise ValueError(
            f"Unknown MMP calculation method: '{method}'. Available methods are: "
            f"{', '.join(MMP_METHODS.keys())}."
        )