"""
Core components for the Unified Physics Engine.

Contains shared core functionality: state management, grid management,
time-stepping, and well control.
"""

from .state_manager import (
    UnifiedState,
    StateManager,
    StateHistory,
    Phase,
    Component,
)
from .grid_manager import (
    GridManager,
    GridType,
    CartesianGridManager,
    CornerPointGridManager,
    CellGeometry,
    NeighborInfo,
    create_grid_manager,
)
from .time_stepper import (
    UnifiedTimeStepper,
    TimestepStatus,
    TimestepResult,
    TimestepStatistics,
    CFLCalculator,
    create_time_stepper,
)

__all__ = [
    # State management
    "UnifiedState",
    "StateManager",
    "StateHistory",
    "Phase",
    "Component",
    # Grid management
    "GridManager",
    "GridType",
    "CartesianGridManager",
    "CornerPointGridManager",
    "CellGeometry",
    "NeighborInfo",
    "create_grid_manager",
    # Time-stepping
    "UnifiedTimeStepper",
    "TimestepStatus",
    "TimestepResult",
    "TimestepStatistics",
    "CFLCalculator",
    "create_time_stepper",
]
