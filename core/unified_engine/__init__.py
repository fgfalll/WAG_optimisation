"""
Unified Physics Engine for CO2 EOR Optimizer

This package provides a unified architecture with two engines (Fast/Simple and Detailed)
that share common physics components to eliminate code duplication.

Architecture:
    - base: Abstract interfaces and configuration
    - core: Shared core components (grid, state, time-stepping, wells)
    - solvers: Numerical solver interfaces and implementations
    - physics: Modular physics components
    - engines: Concrete FastEngine and DetailedEngine implementations

Usage:
    from core.unified_engine import EngineConfig, EngineMode, create_fast_engine, create_detailed_engine

    # Create fast engine for optimization
    config = EngineConfig(mode=EngineMode.SIMPLE)
    engine = create_fast_engine(config)
    engine.initialize()
    results = engine.run()

    # Create detailed engine for final verification
    config_detailed = EngineConfig(mode=EngineMode.DETAILED)
    engine_detailed = create_detailed_engine(config_detailed)
    engine_detailed.initialize()
    detailed_results = engine_detailed.run()
"""

# Base components
from .base.engine_interface import (
    SimulationEngineInterface,
    EvaluationResult,
    BatchSimulationResult,
)
from .base.engine_config import (
    EngineConfig,
    EngineMode,
    SolverType,
    TimestepConfig,
    ModuleConfig,
    GridConfig,
    WellConfig,
    PressureSolverType,
    RelativePermeabilityModel,
    EOSModel,
)
from .base.physics_module import (
    PhysicsModule,
    CompositionalModule,
    GeomechanicsModule,
    FaultMechanicsModule,
    MineralizationModule,
    ModuleRegistry,
)

# Core components
from .core.state_manager import (
    UnifiedState,
    StateManager,
    StateHistory,
    Phase,
    Component,
)
from .core.grid_manager import (
    GridManager,
    GridType,
    CartesianGridManager,
    CornerPointGridManager,
    CellGeometry,
    NeighborInfo,
    create_grid_manager,
)
from .core.time_stepper import (
    UnifiedTimeStepper,
    TimestepStatus,
    TimestepResult,
    TimestepStatistics,
    CFLCalculator,
    create_time_stepper,
)

# Solvers
from .solvers.flow_solver import (
    FlowSolver,
    ExplicitFlowSolver,
    ImplicitFlowSolver,
    SolverStatus,
    SolverResult,
    SolverStatistics,
    create_flow_solver,
)

# Physics
from .physics.multiphase_flow import (
    MultiphaseFlowModule,
    FluidProperties,
    RockProperties,
    RelativePermeabilityModel as RelPermModel,
    DarcyFlowCalculator,
    create_multiphase_flow_module,
)

# Engine implementations
from .engines.fast_engine import FastEngine, create_fast_engine
from .engines.detailed_engine import DetailedEngine, create_detailed_engine
from .engines.wrapper import UnifiedEngineWrapper


__all__ = [
    # Base interfaces
    "SimulationEngineInterface",
    "EvaluationResult",
    "BatchSimulationResult",
    # Configuration
    "EngineConfig",
    "EngineMode",
    "SolverType",
    "TimestepConfig",
    "ModuleConfig",
    "GridConfig",
    "WellConfig",
    "PressureSolverType",
    "RelativePermeabilityModel",
    "EOSModel",
    # Physics modules
    "PhysicsModule",
    "CompositionalModule",
    "GeomechanicsModule",
    "FaultMechanicsModule",
    "MineralizationModule",
    "ModuleRegistry",
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
    # Solvers
    "FlowSolver",
    "ExplicitFlowSolver",
    "ImplicitFlowSolver",
    "SolverStatus",
    "SolverResult",
    "SolverStatistics",
    "create_flow_solver",
    # Physics
    "MultiphaseFlowModule",
    "FluidProperties",
    "RockProperties",
    "RelPermModel",
    "DarcyFlowCalculator",
    "create_multiphase_flow_module",
    # Engines
    "FastEngine",
    "create_fast_engine",
    "DetailedEngine",
    "create_detailed_engine",
]

# Version tracking
__version__ = "1.0.0"
