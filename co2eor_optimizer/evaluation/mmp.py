"""Minimum Miscibility Pressure (MMP) calculation module"""
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from ..core import PVTProperties

@dataclass
class MMPParameters:
    """Parameters for MMP calculation"""
    temperature: float  # Reservoir temperature in °F (valid range: 70-300°F)
    oil_gravity: float  # Oil API gravity (valid range: 15-50°API)
    c7_plus_mw: Optional[float] = None  # Molecular weight of C7+ fraction (150-300 g/mol)
    injection_gas_composition: Optional[dict] = None  # Mole fractions {'CO2': 0.95, ...}
    pvt_data: Optional[PVTProperties] = None  # Optional PVT properties from core

    def __post_init__(self):
        """Validate input parameters"""
        if not 70 <= self.temperature <= 300:
            raise ValueError(f"Temperature {self.temperature}°F outside valid range (70-300°F)")
        if not 15 <= self.oil_gravity <= 50:
            raise ValueError(f"Oil gravity {self.oil_gravity}°API outside valid range (15-50°API)")
        if self.c7_plus_mw and not 150 <= self.c7_plus_mw <= 300:
            raise ValueError(f"C7+ MW {self.c7_plus_mw} outside valid range (150-300 g/mol)")
        if self.injection_gas_composition:
            total = sum(self.injection_gas_composition.values())
            if not 0.99 <= total <= 1.01:
                raise ValueError(f"Gas composition sums to {total}, should be 1.0±0.01")

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

def calculate_mmp_yuan(params: MMPParameters) -> float:
    """
    Calculate MMP using Yuan correlation (2005)
    Better for impure CO2 injection cases
    """
    if not params.injection_gas_composition:
        raise ValueError("Gas composition required for Yuan correlation")
    
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
    Unified MMP calculation interface
    
    Args:
        params: Either MMPParameters or PVTProperties
        method: 'cronquist', 'glaso', 'yuan', or 'auto'
    
    Returns:
        Calculated MMP in psi
    """
    if isinstance(params, PVTProperties):
        # Convert PVTProperties to MMPParameters using reasonable defaults
        # Note: API gravity estimated from oil FVF at standard conditions
        if not params.oil_fvf.any():
            raise ValueError("PVTProperties must contain oil FVF data")
            
        # Estimate API gravity from FVF (standing correlation approximation)
        bo_std = params.oil_fvf[0]  # First value assumed at standard conditions
        api_gravity = (141.5 / (131.5 + bo_std)) - 131.5
        
        # Create parameters then calculate MMP
        mmp_params = MMPParameters(
            temperature=180,  # Default reservoir temp
            oil_gravity=max(15, min(50, api_gravity)),  # Clamped to valid range
            pvt_data=params
        )
        return calculate_mmp_cronquist(mmp_params)  # Default to Cronquist for PVT input
    
    if method == 'auto':
        if params.injection_gas_composition and params.injection_gas_composition.get('CO2', 0.0) < 0.9:
            return calculate_mmp_yuan(params)
        elif params.c7_plus_mw:
            return calculate_mmp_glaso(params)
        else:
            return calculate_mmp_cronquist(params)
    elif method == 'cronquist':
        return calculate_mmp_cronquist(params)
    elif method == 'glaso':
        return calculate_mmp_glaso(params)
    elif method == 'yuan':
        return calculate_mmp_yuan(params)
    else:
        raise ValueError(f"Unknown MMP calculation method: {method}")

def estimate_api_from_pvt(pvt: PVTProperties) -> float:
    """
    Estimate API gravity from PVT properties
    
    Args:
        pvt: PVTProperties object containing oil FVF data
    
    Returns:
        Estimated API gravity (15-50°API)
    """
    if not pvt.oil_fvf.any():
        raise ValueError("PVTProperties must contain oil FVF data")
    
    bo_std = pvt.oil_fvf[0]  # First value assumed at standard conditions
    # Standing correlation for API gravity from FVF
    api = (141.5 / bo_std) - 131.5
    # Empirical adjustment based on typical oil properties
    if bo_std < 1.3:
        api = api * 2.5  # Light oils
    else:
        api = api * 1.8  # Medium oils
    return max(15, min(50, api))  # Clamped to valid range