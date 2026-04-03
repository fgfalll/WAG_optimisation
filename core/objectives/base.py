"""
Base classes and enums for objective functions.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


class OptimizationMetric(Enum):
    """Standard optimization metrics."""
    NPV = "net_present_value"
    RECOVERY_FACTOR = "recovery_factor"
    CO2_STORAGE = "co2_storage_efficiency"
    CO2_UTILIZATION = "co2_utilization_factor"
    PRODUCTION_RATE = "production_rate"
    CUMULATIVE_PRODUCTION = "cumulative_production"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SWEEP_EFFICIENCY = "sweep_efficiency"
    PLUME_CONTAINMENT = "plume_containment"
    TRAPPING_EFFICIENCY = "trapping_efficiency"


@dataclass
class ObjectiveResult:
    """Result of objective function evaluation."""
    value: float
    constraints_satisfied: bool = True
    constraint_violations: List[str] = field(default_factory=list)
    intermediate_values: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
