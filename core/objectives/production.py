"""
Production objective functions.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Numerical constants - use PhysicalConstants for consistency
try:
    from core.data_models import PhysicalConstants
    _PHYS_CONSTANTS = PhysicalConstants()
    EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
except ImportError:
    EPSILON = 1e-10  # Fallback for standalone usage


def calculate_recovery_factor(
    profiles: Dict[str, np.ndarray],
    ooip_stb: float,
    time_resolution: str = "annual",
) -> float:
    """
    Calculate Oil Recovery Factor.
    
    RF = Cumulative_Oil_Produced / OOIP
    
    Args:
        profiles: Production profiles
        ooip_stb: Original oil in place (STB)
        time_resolution: Time resolution key prefix
        
    Returns:
        Recovery factor (fraction)
    """
    oil_production = profiles.get(f"{time_resolution}_oil_stb", np.array([0]))
    total_produced = np.sum(oil_production)
    
    if ooip_stb <= 0:
        return 0.0
    
    rf = total_produced / ooip_stb
    return float(np.clip(rf, 0.0, 1.0))


def calculate_co2_utilization_factor(
    oil_production: np.ndarray,
    co2_purchased_mscf: np.ndarray,
    co2_density_tonne_per_mscf: float = 0.053,
) -> float:
    """
    Calculate CO2 Utilization Factor.
    
    Utilization = Total CO2 Purchased (tonnes) / Cumulative Oil Produced (bbl)
    
    Lower values are better (more oil per unit CO2).
    
    Args:
        oil_production: Oil production profile (STB)
        co2_purchased_mscf: Purchased CO2 profile (mscf)
        co2_density_tonne_per_mscf: CO2 density conversion factor
        
    Returns:
        CO2 utilization factor (tonne/bbl)
    """
    total_oil = np.sum(oil_production)
    total_co2_tonne = np.sum(co2_purchased_mscf) * co2_density_tonne_per_mscf
    
    if total_oil < EPSILON:
        return 1e6  # Very high (bad) utilization if no oil produced
    
    return float(total_co2_tonne / total_oil)


def calculate_production_rate(
    profiles: Dict[str, np.ndarray],
    time_resolution: str = "annual",
) -> float:
    """
    Calculate Average Oil Production Rate.
    
    Args:
        profiles: Production profiles
        time_resolution: Time resolution key prefix
        
    Returns:
        Average production rate (STB/year or STB/month)
    """
    oil_production = profiles.get(f"{time_resolution}_oil_stb", np.array([0]))
    
    if len(oil_production) == 0:
        return 0.0
    
    return float(np.mean(oil_production))
