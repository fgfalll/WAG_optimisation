# Objective Functions Package for CO2-EOR Optimization

from core.objectives.economic import calculate_npv, calculate_cashflow
from core.objectives.storage import (
    calculate_co2_storage_efficiency,
    calculate_geomechanical_containment_score,
)
from core.objectives.wrapper import ObjectiveFunctions

__all__ = [
    "calculate_npv",
    "calculate_cashflow",
    "calculate_co2_storage_efficiency",
    "calculate_geomechanical_containment_score",
    "ObjectiveFunctions",
]
