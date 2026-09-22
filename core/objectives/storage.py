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
    with np.errstate(divide="ignore", invalid="ignore"):
        storage_efficiency = np.where(
            co2_injected_tonne > 0, net_co2_stored_tonne / co2_injected_tonne, 0.0
        )

    # Cumulative storage
    cumulative_co2_stored_tonne = np.cumsum(net_co2_stored_tonne)

    return {
        "co2_injected_tonne": co2_injected_tonne,
        "co2_produced_tonne": co2_produced_tonne,
        "net_co2_stored_tonne": net_co2_stored_tonne,
        "storage_efficiency": storage_efficiency,
        "cumulative_co2_stored_tonne": cumulative_co2_stored_tonne,
    }


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

    Notes:
        The formula accounts for the fact that not all produced gas is CO2.
        Only a fraction (typically ~50% post-breakthrough) of produced gas is
        recycled CO2. The rest is solution gas (CH4, hydrocarbons) that was
        in the reservoir before CO2 injection began.

        References:
        - OSTI-1204577 (Peck et al. 2017): ~50% of injected CO2 is produced
        - Mathiassen 2003: Koval-based fractional flow for CO2 EOR
    """
    co2_purchased = profiles.get(f"{time_resolution}_co2_purchased_mscf", np.array([0]))
    co2_recycled = profiles.get(f"{time_resolution}_co2_recycled_mscf", np.array([0]))
    co2_produced = profiles.get(f"{time_resolution}_co2_produced_mscf", np.array([0]))

    # Get CO2 fractions from params if available, otherwise use defaults
    if co2_storage_params is not None:
        produced_co2_fraction = getattr(co2_storage_params, "produced_co2_fraction", 0.50)
        pre_bt_frac = getattr(co2_storage_params, "pre_breakthrough_co2_fraction", 0.05)
    else:
        produced_co2_fraction = 0.50
        pre_bt_frac = 0.05

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

    # Produced gas is NOT pure CO2 - it includes solution gas (CH4, hydrocarbons)
    # that was in the reservoir before CO2 injection. We apply a conservative
    # CO2 fraction (50%) to account for this.
    # Reference: OSTI-1204577 shows ~50% of injected CO2 is produced (recycled),
    # meaning the other ~50% is retained in formation via trapping / dissolution.
    total_produced = float(np.sum(co2_produced)) * produced_co2_frac_val * co2_density_val

    if total_injected <= 0.0:
        return 0.0

    net_stored = total_injected - total_produced

    # Clamp to [0, 1] - storage efficiency cannot be negative or exceed 100%
    return float(np.clip(net_stored / total_injected, 0.0, 1.0))


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

    leakage_rate = getattr(co2_storage_params, "leakage_rate_fraction", 0.001)

    # Consider pressure-related risk
    pressure_profile = profiles.get(f"{time_resolution}_pressure", np.array([]))
    max_pressure_psi = getattr(co2_storage_params, "max_injection_pressure_psi", 5000.0)

    if len(pressure_profile) > 0:
        pressure_ratio = np.max(pressure_profile) / max_pressure_psi
        pressure_penalty = max(0.0, pressure_ratio - 0.9) * 0.5  # Penalty for > 90% max
    else:
        pressure_penalty = 0.0

    containment_score = 1.0 - leakage_rate - pressure_penalty
    return float(np.clip(containment_score, 0.0, 1.0))


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


def calculate_trapping_efficiency(
    trapping_from_engine: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate CO2 trapping efficiency based on trapping mechanisms.

    Considers structural, residual, solubility, and mineral trapping.

    Args:
        trapping_from_engine: Dictionary with trapping values from engine:
            - structural: structural trapping fraction
            - residual: residual trapping fraction
            - solubility: solubility trapping fraction
            - mineral: mineral trapping fraction
            - total: total trapping fraction

    Returns:
        Trapping efficiency (fraction)

    Raises:
        ValueError: If trapping_from_engine is not provided.
    """
    if trapping_from_engine is None:
        raise ValueError(
            "trapping_from_engine is required. SurrogateEngine did not return "
            "trapping efficiency data. Ensure co2_storage_params were provided "
            "and the engine executed successfully. If no trapping data is "
            "available, this objective function cannot be evaluated."
        )
    return float(np.clip(trapping_from_engine.get("total_trapping_efficiency"), 0.0, 1.0))
