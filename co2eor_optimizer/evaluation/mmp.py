"""Minimum Miscibility Pressure (MMP) calculation module"""
import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class MMPParameters:
    """Parameters for MMP calculation"""
    temperature: float  # Reservoir temperature in °F
    oil_gravity: float  # Oil API gravity
    c7_plus_mw: Optional[float] = None  # Molecular weight of C7+ fraction
    injection_gas_composition: Optional[dict] = None  # Mole fractions {'CO2': 0.95, ...}

def calculate_mmp_cronquist(params: MMPParameters) -> float:
    """
    Calculate MMP using Cronquist correlation (1978)
    Valid for pure CO2 injection
    """
    # Cronquist correlation: MMP = 15.988*(T)^0.744206 * (API)^0.279033
    return 15.988 * (params.temperature ** 0.744206) * (params.oil_gravity ** 0.279033)

def calculate_mmp_glaso(params: MMPParameters) -> float:
    """
    Calculate MMP using Glaso correlation (1985)
    Accounts for gas composition
    """
    if not params.c7_plus_mw:
        raise ValueError("C7+ molecular weight required for Glaso correlation")
    
    """Calculate MMP using hybrid correlation approach"""
    # Base MMP from Cronquist (1978) - most reliable for pure CO2
    mmp_base = 15.988 * (params.temperature ** 0.744206) * (params.oil_gravity ** 0.279033)
    
    # Adjust for C7+ molecular weight (Glaso 1985)
    if params.c7_plus_mw:
        mmp_base *= 1.0 - 0.007 * (params.c7_plus_mw - 190)
    
    # Adjust for gas composition (Yellig & Metcalfe 1980)
    if params.injection_gas_composition:
        co2_fraction = params.injection_gas_composition.get('CO2', 0.0)
        # Linear interpolation between pure CO2 and 80% CO2 MMPs
        mmp_base *= 1.0 - 0.25 * (1.0 - co2_fraction)
    
    return mmp_base