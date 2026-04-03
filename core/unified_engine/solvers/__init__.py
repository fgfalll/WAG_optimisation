"""
Numerical solvers for the Unified Physics Engine.

Contains flow and pressure solver implementations that can be shared
between FastEngine and DetailedEngine.
"""

from .flow_solver import (
    FlowSolver,
    ExplicitFlowSolver,
    ImplicitFlowSolver,
    SolverStatus,
    SolverResult,
    SolverStatistics,
    create_flow_solver,
)

__all__ = [
    # Flow solvers
    "FlowSolver",
    "ExplicitFlowSolver",
    "ImplicitFlowSolver",
    "SolverStatus",
    "SolverResult",
    "SolverStatistics",
    "create_flow_solver",
]
