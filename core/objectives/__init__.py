# Objective Functions Package for CO2-EOR Optimization

from core.objectives.base import ObjectiveType, OptimizationMetric, ObjectiveResult
from core.objectives.economic import calculate_npv, calculate_cashflow
from core.objectives.storage import (
    calculate_co2_storage_efficiency,
    calculate_plume_containment_score,
    calculate_trapping_efficiency,
    calculate_co2_storage_metrics,
)
from core.objectives.production import (
    calculate_recovery_factor,
    calculate_co2_utilization_factor,
    calculate_production_rate,
)
from core.objectives.wrapper import ObjectiveFunctions

__all__ = [
    # Base classes
    "ObjectiveType",
    "OptimizationMetric", 
    "ObjectiveResult",
    # Economic
    "calculate_npv",
    "calculate_cashflow",
    # Storage
    "calculate_co2_storage_efficiency",
    "calculate_plume_containment_score",
    "calculate_trapping_efficiency",
    "calculate_co2_storage_metrics",
    # Production
    "calculate_recovery_factor",
    "calculate_co2_utilization_factor",
    "calculate_production_rate",
    "ObjectiveFunctions",
]
