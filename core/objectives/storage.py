"""
CO2 Storage objective functions.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_co2_storage_metrics(
    co2_purchased_mscf: np.ndarray,
    co2_recycled_mscf: np.ndarray,
    co2_produced_mscf: np.ndarray,
    co2_density_tonne_per_mscf: float = 0.053,
) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive CO2 storage metrics.
    
    Args:
        co2_purchased_mscf: Purchased CO2 volumes (mscf)
        co2_recycled_mscf: Recycled CO2 volumes (mscf)
        co2_produced_mscf: Produced CO2 volumes (mscf)
        co2_density_tonne_per_mscf: CO2 density conversion factor
        
    Returns:
        Dictionary with storage metrics arrays
    """
    co2_injected_mscf = co2_purchased_mscf + co2_recycled_mscf
    co2_injected_tonne = co2_injected_mscf * co2_density_tonne_per_mscf
    co2_produced_tonne = co2_produced_mscf * co2_density_tonne_per_mscf
    
    # Net CO2 stored = injected - produced
    net_co2_stored_tonne = co2_injected_tonne - co2_produced_tonne
    
    # Storage efficiency = net stored / injected
    with np.errstate(divide='ignore', invalid='ignore'):
        storage_efficiency = np.where(
            co2_injected_tonne > 0,
            net_co2_stored_tonne / co2_injected_tonne,
            0.0
        )
    
    # Cumulative storage
    cumulative_co2_stored_tonne = np.cumsum(net_co2_stored_tonne)
    
    return {
        'co2_injected_tonne': co2_injected_tonne,
        'co2_produced_tonne': co2_produced_tonne,
        'net_co2_stored_tonne': net_co2_stored_tonne,
        'storage_efficiency': storage_efficiency,
        'cumulative_co2_stored_tonne': cumulative_co2_stored_tonne,
    }


def calculate_co2_storage_efficiency(
    profiles: Dict[str, np.ndarray],
    time_resolution: str = "annual",
    co2_density_tonne_per_mscf: float = 0.053,
) -> float:
    """
    Calculate CO2 Storage Efficiency.
    
    Efficiency = Net CO2 Stored / Total CO2 Injected
    
    Args:
        profiles: Production and injection profiles
        time_resolution: Time resolution key prefix
        co2_density_tonne_per_mscf: CO2 density conversion factor
        
    Returns:
        Storage efficiency (fraction)
    """
    co2_purchased = profiles.get(f"{time_resolution}_co2_purchased_mscf", np.array([0]))
    co2_recycled = profiles.get(f"{time_resolution}_co2_recycled_mscf", np.array([0]))
    co2_produced = profiles.get(f"{time_resolution}_co2_produced_mscf", np.array([0]))
    
    total_injected = (np.sum(co2_purchased) + np.sum(co2_recycled)) * co2_density_tonne_per_mscf
    total_produced = np.sum(co2_produced) * co2_density_tonne_per_mscf
    
    if total_injected <= 0:
        return 0.0
    
    net_stored = total_injected - total_produced
    return float(net_stored / total_injected)


def calculate_plume_containment_score(
    profiles: Dict[str, np.ndarray],
    co2_storage_params: Optional[Any] = None,
    time_resolution: str = "annual",
) -> float:
    """
    Calculate plume containment score based on CO2 migration and leakage.
    
    Score = 1.0 - leakage_fraction
    
    Args:
        profiles: Production and injection profiles
        co2_storage_params: CO2 storage parameters
        time_resolution: Time resolution key prefix
        
    Returns:
        Containment score (0 to 1, higher is better)
    """
    if co2_storage_params is None:
        return 0.95  # Default high containment
    
    leakage_rate = getattr(co2_storage_params, 'leakage_rate_fraction', 0.001)
    
    # Consider pressure-related risk
    pressure_profile = profiles.get(f'{time_resolution}_pressure', np.array([]))
    max_pressure_psi = getattr(co2_storage_params, 'max_injection_pressure_psi', 5000.0)
    
    if len(pressure_profile) > 0:
        pressure_ratio = np.max(pressure_profile) / max_pressure_psi
        pressure_penalty = max(0.0, pressure_ratio - 0.9) * 0.5  # Penalty for > 90% max
    else:
        pressure_penalty = 0.0
    
    containment_score = 1.0 - leakage_rate - pressure_penalty
    return float(np.clip(containment_score, 0.0, 1.0))


def calculate_trapping_efficiency(
    co2_storage_params: Optional[Any] = None,
) -> float:
    """
    Calculate CO2 trapping efficiency based on trapping mechanisms.
    
    Considers structural, residual, solubility, and mineral trapping.
    
    Args:
        co2_storage_params: CO2 storage parameters
        
    Returns:
        Trapping efficiency (fraction)
    """
    if co2_storage_params is None:
        return 0.85  # Default
    
    # Weighted sum of trapping mechanisms
    structural = getattr(co2_storage_params, 'structural_trapping_fraction', 0.4)
    residual = getattr(co2_storage_params, 'residual_trapping_fraction', 0.25)
    solubility = getattr(co2_storage_params, 'solubility_trapping_fraction', 0.2)
    mineral = getattr(co2_storage_params, 'mineral_trapping_fraction', 0.0)
    
    total_trapping = structural + residual + solubility + mineral
    return float(np.clip(total_trapping, 0.0, 1.0))
