"""
CO2 Storage objective functions.
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_co2_storage_efficiency(
    profiles: Dict[str, np.ndarray],
    time_resolution: str = "annual",
    co2_density_tonne_per_mscf: float = 0.053,
    co2_storage_params: Optional[Any] = None,
) -> float:
    """
    Calculate CO2 Storage Efficiency.

    Efficiency = Net CO2 Stored / Total CO2 Injected

    Args:
        profiles: Production and injection profiles
        time_resolution: Time resolution key prefix
        co2_density_tonne_per_mscf: CO2 density conversion factor
        co2_storage_params: Optional CO2StorageParameters with CO2 fraction settings

    Returns:
        Storage efficiency (fraction), clamped to [0, 1]
    """
    co2_purchased = profiles.get(f"{time_resolution}_co2_purchased_mscf", np.array([0]))
    co2_recycled = profiles.get(f"{time_resolution}_co2_recycled_mscf", np.array([0]))
    co2_produced = profiles.get(f"{time_resolution}_co2_produced_mscf", np.array([0]))

    # Get CO2 fractions from params if available, otherwise use defaults
    if co2_storage_params is not None:
        produced_co2_fraction = getattr(co2_storage_params, "produced_co2_fraction", 0.50)
    else:
        produced_co2_fraction = 0.50

    try:
        co2_density_val = float(co2_density_tonne_per_mscf)
    except (TypeError, ValueError):
        co2_density_val = 0.053

    try:
        produced_co2_frac_val = float(produced_co2_fraction)
    except (TypeError, ValueError):
        produced_co2_frac_val = 0.50

    # Total CO2 injected (purchased + recycled)
    total_injected = float(np.sum(co2_purchased) + np.sum(co2_recycled)) * co2_density_val

    # Produced gas includes recycled CO2 and hydrocarbon solution gas
    total_produced = float(np.sum(co2_produced)) * produced_co2_frac_val * co2_density_val

    if total_injected <= 0.0:
        return 0.0

    net_stored = total_injected - total_produced

    # Clamp to [0, 1] - storage efficiency cannot be negative or exceed 100%
    return float(np.clip(net_stored / total_injected, 0.0, 1.0))


def calculate_geomechanical_containment_score(
    pressure_profile: np.ndarray,
    fracture_pressure: float,
    storage_params: Any,
) -> float:
    """
    Calculate plume containment score using PhD geomechanical formula.

    S_cont = γ_safety * [w_p * S_press + w_s * S_seal + w_t * S_struct]
    S_press = max(0, 1.0 - P̄_inj / (P_frac * λ_limit))

    Args:
        pressure_profile: Array of injection pressures (psi)
        fracture_pressure: Formation fracture pressure (psi)
        storage_params: Must have containment_* parameters from AdvancedEngineParams

    Returns:
        Containment score (0 to 1, higher is better)
    """
    gamma_safety = getattr(storage_params, "containment_safety_margin", 1.0)
    w_p = getattr(storage_params, "containment_pressure_weight", 0.5)
    w_s = getattr(storage_params, "containment_seal_weight", 0.3)
    w_t = getattr(storage_params, "containment_structure_weight", 0.2)
    lambda_limit = getattr(storage_params, "fracture_pressure_limit_fraction", 0.9)
    s_seal = getattr(storage_params, "reservoir_seal_integrity_factor", 0.9)
    s_struct = getattr(storage_params, "structural_trapping_factor", 0.85)

    if len(pressure_profile) > 0:
        avg_pressure = np.mean(pressure_profile)
        s_press = max(0.0, 1.0 - avg_pressure / (fracture_pressure * lambda_limit))
    else:
        s_press = 1.0

    s_cont = gamma_safety * (w_p * s_press + w_s * s_seal + w_t * s_struct)

    return float(np.clip(s_cont, 0.0, 1.0))
