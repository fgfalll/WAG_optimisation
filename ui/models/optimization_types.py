from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List


@dataclass
class OptimizationObjective:
    """Define optimization objective configuration"""
    name: str
    description: str
    target_value: Optional[float] = None
    target_type: str = "maximize"
    weight: float = 1.0
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class OptimizationConstraints:
    """Define optimization parameter constraints"""
    parameter_name: str
    min_value: float
    max_value: float
    step_size: float = 0.01
    parameter_type: str = "continuous"
    allowed_values: List[Any] = field(default_factory=list)
