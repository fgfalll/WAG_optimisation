import logging
import sys
import os
from typing import Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from core.data_models import PVTProperties
except ImportError as e:
    raise ImportError(
        "Could not import 'PVTProperties' from 'core.data_models'. "
        "Please ensure the package structure is correct and the project root is in sys.path."
    ) from e


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
            like 'hybrid_gh'. (Typical valid range: 50-250 g/mol).
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

        # Validate Oil API Gravity, changing hard error to a warning.
        if not 15 <= self.oil_gravity <= 50:
            logging.warning(
                f"Oil gravity {self.oil_gravity}°API is outside the typical valid "
                f"range (15-50°API). Correlations are being extrapolated and results may be inaccurate."
            )
        elif self.oil_gravity < 20 or self.oil_gravity > 45:
            logging.warning(
                f"Oil gravity {self.oil_gravity}°API is near correlation limits. "
                "Accuracy of the results may be reduced."
            )

        # Validate C7+ Molecular Weight if provided, but with a warning instead of an error.
        if self.c7_plus_mw and not 50 <= self.c7_plus_mw <= 250:
            logging.warning(
                f"C7+ MW {self.c7_plus_mw} g/mol is outside the typical valid "
                f"range (50-250 g/mol). Correlation accuracy may be significantly reduced."
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
    Calculates MMP using the published Cronquist correlation (1978).

    Formula (Cronquist 1978, DOE / CO2 Prophet):
        MMP [psia] = 15.988 * (Temperature [°F] ^ Y)
        Y = 0.744206 + 0.0011038 * MW_C5+ + 0.0015279 * Vol

    Where:
        - MW_C5+: Molecular weight of pentanes-plus fraction (g/mol).
          If c7_plus_mw is provided, MW_C5+ = max(72.0, c7_plus_mw - 20.0).
          Otherwise, estimated from API gravity via standard DOE / CO2 Prophet correlation:
          MW_C5+ = 4247.98641 * (oil_gravity ** -0.87022).
        - Vol: Mole percent of volatile components (C1 + N2) in the oil phase.
          Defaults to 0.0 if not specified.

    This published formulation is strictly positive, monotonically decreases with API
    gravity (lighter oils have lower MMP), and eliminates the singularity and complex-number
    crash at API >= 55 (SCI-FLAW-13).
    """
    # 1. Determine MW_C5+
    if params.c7_plus_mw and params.c7_plus_mw > 0:
        mw_c5_plus = max(72.0, float(params.c7_plus_mw) - 20.0)
    else:
        # Standard DOE / CO2 Prophet correlation for C5+ MW from API gravity
        api_clamped = max(10.0, float(params.oil_gravity))
        mw_c5_plus = 4247.98641 * (api_clamped ** -0.87022)

    # 2. Volatile content in reservoir oil (mole percent C1 + N2)
    # Default to 0.0 if not explicitly known
    vol_pct = 0.0

    # 3. Cronquist exponent Y
    exponent_y = 0.744206 + 0.0011038 * mw_c5_plus + 0.0015279 * vol_pct

    # 4. Pure CO2 MMP in psia (15.988 corresponds to 0.11027 MPa * 145.0 psi/MPa)
    mmp = 15.988 * (params.temperature ** exponent_y)
    return float(mmp)


def _calculate_mmp_yellig_metcalfe(params: MMPParameters) -> float:
    """
    Yellig & Metcalfe (1980) correlation for pure CO2.
    Formula (SPE 7477):
        MMP [psia] = 1833.7217 + 2.2518055·T + 0.01800674·T² - 103949.93 / T

    Note: If T < 95°F, MMP is capped at the CO2 critical/bubble-point pressure (1070 psia)
    as demonstrated by Yellig & Metcalfe (1980).
    """
    T = params.temperature
    if T < 95.0:
        # At reservoir temperatures below 95°F, near or below CO2 critical temperature (87.9°F),
        # CO2 vapor pressure / bubble point pressure governs miscibility (~1070 psia).
        return 1070.0

    # Published Yellig & Metcalfe (1980) equation
    mmp = 1833.7217 + 2.2518055 * T + 0.01800674 * (T**2) - (103949.93 / T)
    return float(max(1070.0, mmp))


def _calculate_mmp_hybrid_gh(params: MMPParameters) -> float:
    """
    Calculates MMP using a hybrid approach combining multiple correlations.

    1. Base MMP is calculated using Yellig & Metcalfe (1980) for pure CO2.
    2. An adjustment is made for the C7+ molecular weight based on Glaso (1985).
    3. An adjustment is made for gas impurities based on Yellig & Metcalfe (1980) / Sebastian (1985).

    This method is suitable for oils with known C7+ MW and for injection gases
    with varying CO2 concentrations.
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight is required for the 'hybrid_gh' method.")

    # 1. Base MMP from pure CO2 correlation
    mmp_base = _calculate_mmp_yellig_metcalfe(params)

    # 2. Adjust for C7+ molecular weight (Glaso, 1985)
    # Increases MMP for heavier oils (higher C7+ MW) and decreases for lighter oils
    mmp_adj_c7 = mmp_base * (1.0 + 0.007 * (params.c7_plus_mw - 190.0))

    # 3. Adjust for gas composition (Yellig & Metcalfe 1980, Sebastian 1985)
    if params.injection_gas_composition:
        co2_fraction = float(params.injection_gas_composition.get("CO2", 0.0))
        total_impurity = 1.0 - co2_fraction

        if total_impurity > 1e-4:
            impurity_sensitivities = {
                "CH4": 0.35,  # Methane has strong MMP-increasing effect
                "N2": 0.45,   # Nitrogen has strongest MMP-increasing effect
                "H2S": -0.10, # H2S slightly decreases MMP (favorable)
                "C2": -0.05,  # Ethane slightly decreases MMP
                "C3": -0.10,  # Propane decreases MMP
            }

            weighted_sens = 0.0
            for imp, sens in impurity_sensitivities.items():
                imp_fraction = params.injection_gas_composition.get(imp, 0.0)
                weighted_sens += sens * imp_fraction
            sensitivity_factor = weighted_sens / total_impurity
            # Bound sensitivity factor
            sensitivity_factor = max(0.10, min(0.60, sensitivity_factor))

            logger.debug(
                f"MMP impurity adjustment: CO2 fraction={co2_fraction:.2f}, "
                f"sensitivity factor={sensitivity_factor:.3f}"
            )
            return float(mmp_adj_c7 / (1.0 - sensitivity_factor * total_impurity))

    return float(mmp_adj_c7)


def _calculate_mmp_yuan(params: MMPParameters) -> float:
    """
    Calculates MMP using the Yuan et al. correlation (2005) for pure and impure CO2 streams.

    Based on analytical gas-flooding theory:
        MMP [psia] = a * (b ** x_co2) * c * 145.038

    Where:
        - a: Temperature dependency term
        - b: Oil composition term (C5+ MW from API or C7+)
        - c: Purity penalty factor. In physics, volatile impurities (CH4, N2)
          increase MMP (harder to achieve miscibility).
    """
    if not params.injection_gas_composition:
        raise ValueError("Gas composition is required for the 'yuan' correlation.")

    co2_fraction = float(params.injection_gas_composition.get("CO2", 0.0))
    x_co2 = max(0.0, min(1.0, co2_fraction))

    # Yuan correlation coefficients
    # A: Temperature dependency term
    T = params.temperature
    a = 10.0 ** (1.356 + 0.0016 * T - 0.0000033 * (T**2))

    # B: Oil composition term (C5+ MW from C7+ MW or API)
    if params.c7_plus_mw and params.c7_plus_mw > 0:
        m_c5_plus = max(72.0, float(params.c7_plus_mw) - 20.0)
    else:
        m_c5_plus = max(72.0, 630.0 - 10.3 * params.oil_gravity)
    b = (m_c5_plus ** 0.36) / (0.641 * (T ** 0.21))

    # C: Impurity term: Impurities (1 - x_co2) monotonically increase MMP
    # When x_co2 = 1.0 (pure CO2), impurity_factor = 1.0
    # When impurities (CH4, N2) are present, MMP increases
    impurity_frac = 1.0 - x_co2
    c = 1.0 + 1.25 * (impurity_frac ** 0.8)

    # Final MMP calculation in MPa, then converted to psi
    mmp_mpa = a * (b ** x_co2) * c
    return float(mmp_mpa * 145.038)


def _calculate_mmp_alston(params: MMPParameters) -> float:
    """
    Calculates MMP using the Alston et al. correlation (1985) for impure gas streams.

    Adjusts pure-CO2 MMP based on the pseudo-critical temperature of the injection gas mixture.
    In thermodynamic physics, adding lighter impurities (CH4, N2) lowers T_pc and RAISES MMP:
        MMP_impure = MMP_pure * (T_pc,CO2 / T_pc,mix) ** exponent_A

    Where:
        - T_pc,CO2 = 304.1 K (87.9°F)
        - exponent_A = max(0.5, 2.41 - 0.00284 * C7+_MW)
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight is required for the 'alston' method.")
    if not params.injection_gas_composition:
        raise ValueError("Gas composition is required for the 'alston' correlation.")

    # Critical temperatures of common components in Kelvin
    CRITICAL_TEMPS_K = {
        "CO2": 304.1,
        "CH4": 190.6,
        "N2": 126.2,
        "H2S": 373.2,
        "C2": 305.3,
        "C3": 369.8,
    }

    # 1. Calculate pseudo-critical temperature (Tpc) of the gas mixture in Kelvin
    tpc_k = 0.0
    total_y = 0.0
    for comp, frac in params.injection_gas_composition.items():
        tc = CRITICAL_TEMPS_K.get(comp.upper(), 304.1)
        tpc_k += frac * tc
        total_y += frac

    if total_y > 0:
        tpc_k /= total_y
    else:
        tpc_k = CRITICAL_TEMPS_K["CO2"]

    # 2. Calculate baseline MMP for pure CO2 using Yellig & Metcalfe (1980)
    mmp_pure_co2 = _calculate_mmp_yellig_metcalfe(params)

    # 3. Calculate the Alston exponent 'A'
    exponent_A = max(0.2, 2.41 - 0.00284 * params.c7_plus_mw)

    # 4. Impurity ratio: As T_pc drops below T_pc,CO2, miscibility pressure increases.
    # Ratio (T_pc,CO2 / T_pc_mix) ensures MMP increases with CH4/N2 impurities.
    t_ratio = CRITICAL_TEMPS_K["CO2"] / max(50.0, tpc_k)
    mmp_impure = mmp_pure_co2 * (t_ratio ** exponent_A)
    return float(mmp_impure)


# --- Dictionary mapping method names to functions for UI and internal use ---
MMP_METHODS: Dict[str, Callable[[MMPParameters], float]] = {
    "cronquist": _calculate_mmp_cronquist,
    "yellig_metcalfe": _calculate_mmp_yellig_metcalfe,
    "hybrid_gh": _calculate_mmp_hybrid_gh,
    "yuan": _calculate_mmp_yuan,
    "alston": _calculate_mmp_alston,
}


def estimate_api_from_pvt(pvt: PVTProperties) -> float:
    """
    Estimates oil API gravity from PVT properties.

    Checks direct attributes first (api_gravity, oil_density_ppg), then analytically
    inverts Standing's (1947) formation volume factor correlation, with a graceful
    fallback to a typical reservoir crude default (35.0 °API).

    References:
    - Standing, M.B. (1947). "A Pressure-Volume-Temperature Correlation
      for Mixtures of California Oils and Gases." API Drilling and Production Practice.
    - McCain, W.D. (1990). "The Properties of Petroleum Fluids."
    """
    # 1. Direct API gravity if available on PVT object
    if hasattr(pvt, "api_gravity") and pvt.api_gravity is not None:
        try:
            val = float(pvt.api_gravity)
            if 10.0 <= val <= 65.0:
                return val
        except (ValueError, TypeError):
            pass

    # 2. Check oil density in ppg if available
    oil_density_ppg = getattr(pvt, "oil_density_ppg", None)
    if oil_density_ppg is not None:
        try:
            ppg = float(oil_density_ppg)
            if 5.0 <= ppg <= 12.0:
                gamma_o = ppg / 8.337
                api = (141.5 / gamma_o) - 131.5
                return float(max(15.0, min(50.0, api)))
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    # 3. Analytical inversion of Standing's (1947) Formation Volume Factor (B_o)
    # B_o = 0.972 + 0.000147 * [ R_s * (gamma_g / gamma_o)**0.5 + 1.25 * T ]**1.175
    try:
        R_s = float(pvt.rs[0]) if (hasattr(pvt, "rs") and pvt.rs is not None and pvt.rs.size > 0) else 500.0
        gamma_g = float(getattr(pvt, "gas_specific_gravity", 0.7) or 0.7)
        T = float(getattr(pvt, "temperature", 150.0) or 150.0)

        # Determine B_o
        b_o = None
        if hasattr(pvt, "oil_fvf") and pvt.oil_fvf is not None and len(pvt.oil_fvf) > 0:
            b_o = float(pvt.oil_fvf[0])
        elif hasattr(pvt, "oil_fvf_simple") and pvt.oil_fvf_simple is not None:
            b_o = float(pvt.oil_fvf_simple)

        if b_o is not None and b_o > 1.0 and R_s > 0:
            f_val = ((b_o - 0.972) / 0.000147) ** (1.0 / 1.175)
            rem = f_val - 1.25 * T
            if rem > 0:
                sqrt_ratio = rem / R_s
                gamma_o = gamma_g / (sqrt_ratio ** 2)
                if 0.65 <= gamma_o <= 1.05:
                    api = (141.5 / gamma_o) - 131.5
                    return float(max(15.0, min(50.0, api)))
    except Exception as e:
        logger.debug(f"Analytical Standing Bo inversion failed: {e}")

    # 4. Fallback to typical reservoir crude oil gravity (35.0 °API)
    logger.info("Using standard reservoir crude oil gravity (35.0°API) as default PVT estimate.")
    return 35.0


def calculate_mmp(params: Union[MMPParameters, PVTProperties], method: str = "auto") -> float:
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

    # Debug logging to understand the type issue
    logging.info(f"calculate_mmp called with params type: {type(params)}")
    logging.info(f"params module: {type(params).__module__}")

    # Robustly check for PVTProperties type by checking class name
    # This handles cases where the class is imported from different paths (core.data_models vs co2eor_optimizer.core.data_models)
    is_pvt_properties = False
    if type(params).__name__ == "PVTProperties":
        is_pvt_properties = True
    elif isinstance(params, PVTProperties):
        is_pvt_properties = True

    if is_pvt_properties:
        api_gravity = None
        if hasattr(params, "api_gravity") and params.api_gravity is not None:
            api_gravity = params.api_gravity
            logging.info(f"Using provided API gravity: {api_gravity:.2f}°API")
        else:
            logging.info(
                "PVTProperties object provided without API gravity. Estimating API gravity from PVT data."
            )
            try:
                api_gravity = estimate_api_from_pvt(params)
                logging.critical(
                    "CRITICAL WARNING: API gravity was estimated from PVT properties using "
                    "Standing's correlation. This is a rough approximation with limited "
                    "accuracy (error can exceed ±5°API). Use measured oil gravity "
                    "for reliable MMP calculations."
                )
            except Exception as e:
                raise ValueError(
                    "Failed to estimate API gravity from PVTProperties. "
                    f"Please provide a measured oil gravity. Original error: {e}"
                )

        # Create MMPParameters with estimated gravity and other available PVT data
        mmp_params = MMPParameters(
            temperature=params.temperature,
            oil_gravity=api_gravity,
            injection_gas_composition=getattr(params, "injection_gas_composition", None),
            c7_plus_mw=getattr(params, "c7_plus_mw", None),
            pvt_data=params,
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
    if method == "auto":
        # Intelligent selection based on data richness
        if (
            mmp_params.c7_plus_mw
            and mmp_params.injection_gas_composition
            and mmp_params.injection_gas_composition.get("CO2", 0.0) < 0.98
        ):
            logging.info("Auto-selecting 'alston' correlation for impure gas with known C7+ MW.")
            return _calculate_mmp_alston(mmp_params)
        elif (
            mmp_params.injection_gas_composition
            and mmp_params.injection_gas_composition.get("CO2", 0.0) < 0.95
        ):
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
