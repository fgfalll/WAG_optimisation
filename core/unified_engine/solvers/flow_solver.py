"""
Flow Solver Interface and Implementations for Unified Physics Engine.

Provides both explicit (fast) and implicit (robust) flow solver implementations
that can be shared between Simple and Detailed engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from enum import Enum
import time as timer

from ..base.engine_config import EngineConfig, SolverType
from ..core.state_manager import UnifiedState
from ..physics.multiphase_flow import MultiphaseFlowModule


class SolverStatus(Enum):
    """Status of solver operation."""
    SUCCESS = "success"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    SINGULAR = "singular"
    ERROR = "error"


@dataclass
class SolverResult:
    """Result from a solver operation."""
    status: SolverStatus
    solution: np.ndarray
    iterations: int
    residual_norm: float
    elapsed_time: float
    message: str = ""
    convergence_history: List[float] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Check if solver was successful."""
        return self.status in (SolverStatus.SUCCESS, SolverStatus.CONVERGED)


@dataclass
class SolverStatistics:
    """Statistics for solver performance."""
    n_calls: int = 0
    n_successful: int = 0
    n_failed: int = 0
    total_time: float = 0.0
    total_iterations: int = 0
    avg_iterations: float = 0.0
    max_iterations_single: int = 0

    def update(self, result: SolverResult) -> None:
        """Update statistics with a solver result."""
        self.n_calls += 1
        self.total_time += result.elapsed_time
        self.total_iterations += result.iterations
        self.max_iterations_single = max(
            self.max_iterations_single, result.iterations
        )
        self.avg_iterations = self.total_iterations / self.n_calls

        if result.is_successful:
            self.n_successful += 1
        else:
            self.n_failed += 1


class FlowSolver(ABC):
    """
    Abstract base class for flow solvers.

    Both explicit and implicit solvers implement this interface.
    """

    def __init__(self, config: EngineConfig, name: str = "flow_solver"):
        """
        Initialize flow solver.

        Args:
            config: Engine configuration.
            name: Solver name for diagnostics.
        """
        self.config = config
        self.name = name
        self.statistics = SolverStatistics()
        self._initialized = False

        # Physics module
        self.physics: Optional[MultiphaseFlowModule] = None

    def set_physics(self, physics: MultiphaseFlowModule) -> None:
        """Set physics module."""
        self.physics = physics

    @abstractmethod
    def initialize(self, state: UnifiedState) -> None:
        """Initialize solver with initial state."""
        self._initialized = True

    @abstractmethod
    def solve(
        self,
        state: UnifiedState,
        dt: float,
        source_terms: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """
        Solve flow equations for one timestep.

        Args:
            state: Current simulation state.
            dt: Timestep size (days).
            source_terms: Optional source/sink terms.

        Returns:
            SolverResult with updated state.
        """
        pass

    def cleanup(self) -> None:
        """Clean up solver resources."""
        self._initialized = False
        self.statistics = SolverStatistics()

    def get_required_state_fields(self) -> List[str]:
        """Get required state fields."""
        return ["pressure", "saturations"]


class ExplicitFlowSolver(FlowSolver):
    """
    Explicit flow solver for fast simulations.

    Uses explicit time integration for saturation and pressure.
    Fast but requires small timesteps for stability (CFL condition).
    """

    def __init__(self, config: EngineConfig):
        """Initialize explicit flow solver."""
        super().__init__(config, name="explicit_flow")

        # Numerical parameters
        self.cfl_number = config.timestep.cfl_safety_factor
        self.grid_spacing = 100.0  # Will be set from grid

    def initialize(self, state: UnifiedState) -> None:
        """Initialize explicit solver."""
        super().initialize(state)

        # Calculate stable timestep based on CFL
        if self.physics is not None and self.physics.rock_properties is not None:
            k_max = np.max(self.physics.rock_properties.permeability)
            phi_min = np.min(self.physics.rock_properties.porosity)
            mu = self.physics.oil_properties.viscosity

            # Simplified CFL estimate
            velocity = 6.33e-3 * k_max / mu
            dt_cfl = self.cfl_number * self.grid_spacing / velocity

            if dt_cfl < self.config.timestep.initial:
                # Suggest smaller timestep
                pass

    def solve(
        self,
        state: UnifiedState,
        dt: float,
        source_terms: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """
        Solve flow equations using explicit method.

        For saturation: S^{n+1} = S^n - (dt/V) * flux
        For pressure: Use analytical or simplified approach
        """
        start_time = timer.time()

        if self.physics is None:
            return SolverResult(
                status=SolverStatus.ERROR,
                solution=state.pressure.copy(),
                iterations=0,
                residual_norm=0.0,
                elapsed_time=0.0,
                message="Physics module not set",
            )

        try:
            # Get relative permeabilities
            kro, krw, krg = self.physics.flow_calculator.relperm_model.calculate(
                state.oil_saturation,
                state.water_saturation,
                state.gas_saturation,
            )

            # Calculate mobilities
            if self.physics.rock_properties is not None:
                k = self.physics.rock_properties.permeability
                if k.ndim == 1:
                    k = k[:, np.newaxis]

                mobility_oil = k[:, 0] * kro / self.physics.oil_properties.viscosity
                mobility_water = k[:, 0] * krw / self.physics.water_properties.viscosity
                mobility_gas = k[:, 0] * krg / self.physics.gas_properties.viscosity
            else:
                mobility_oil = kro / self.physics.oil_properties.viscosity
                mobility_water = krw / self.physics.water_properties.viscosity
                mobility_gas = krg / self.physics.gas_properties.viscosity

            total_mobility = mobility_oil + mobility_water + mobility_gas

            # Explicit pressure update (simplified - constant pressure in this version)
            new_pressure = state.pressure.copy()

            # Explicit saturation update
            # ds/dt = -1/phi * div(flux)
            if state.saturations.ndim == 2:
                new_saturations = state.saturations.copy()

                # Simple upwinding: saturation moves from high to low pressure
                # This is a very simplified explicit update
                p_grad = np.gradient(state.pressure)

                for i in range(state.n_cells):
                    # Saturation change due to flow
                    # (simplified - no spatial coupling in this basic version)
                    pass

                # For now, just copy (solver would need grid connectivity)
                new_saturations = state.saturations.copy()
            else:
                new_saturations = state.saturations.copy()

            # Update state (time will be updated by time_stepper)
            state.pressure = new_pressure
            state.saturations = new_saturations
            state.timestep = dt

            elapsed = timer.time() - start_time

            return SolverResult(
                status=SolverStatus.SUCCESS,
                solution=new_pressure,
                iterations=1,
                residual_norm=0.0,
                elapsed_time=elapsed,
                message="Explicit step completed",
            )

        except Exception as e:
            elapsed = timer.time() - start_time
            return SolverResult(
                status=SolverStatus.ERROR,
                solution=state.pressure.copy(),
                iterations=0,
                residual_norm=0.0,
                elapsed_time=elapsed,
                message=f"Explicit solver error: {str(e)}",
            )


class ImplicitFlowSolver(FlowSolver):
    """
    Implicit flow solver for robust simulations.

    Uses Newton-Raphson iteration for fully implicit treatment.
    More expensive but allows larger timesteps.
    """

    def __init__(self, config: EngineConfig):
        """Initialize implicit flow solver."""
        super().__init__(config, name="implicit_flow")

        # Solver parameters
        self.max_iterations = config.max_newton_iterations
        self.pressure_tolerance = config.tolerance_pressure
        self.saturation_tolerance = config.tolerance_saturation

    def initialize(self, state: UnifiedState) -> None:
        """Initialize implicit solver."""
        super().initialize(state)

    def solve(
        self,
        state: UnifiedState,
        dt: float,
        source_terms: Optional[np.ndarray] = None,
    ) -> SolverResult:
        """
        Solve flow equations using Newton-Raphson iteration.

        Solves: F(P, S) = 0 where F is the residual equation
        J * dx = -F, where J is the Jacobian matrix
        """
        start_time = timer.time()

        if self.physics is None:
            return SolverResult(
                status=SolverStatus.ERROR,
                solution=state.pressure.copy(),
                iterations=0,
                residual_norm=0.0,
                elapsed_time=0.0,
                message="Physics module not set",
            )

        # Initialize
        pressure = state.pressure.copy()
        saturations = state.saturations.copy()

        convergence_history = []

        for iteration in range(self.max_iterations):
            # Calculate residuals
            residual_pressure, residual_saturation = self._calculate_residuals(
                pressure, saturations, dt, source_terms
            )

            # Check convergence
            p_norm = float(np.max(np.abs(residual_pressure)))
            s_norm = float(np.max(np.abs(residual_saturation)))
            convergence_history.append(max(p_norm, s_norm))

            if p_norm < self.pressure_tolerance and s_norm < self.saturation_tolerance:
                elapsed = timer.time() - start_time

                # Update state (time will be updated by time_stepper)
                state.pressure = pressure
                state.saturations = saturations
                state.timestep = dt
                state.iteration = iteration + 1

                return SolverResult(
                    status=SolverStatus.CONVERGED,
                    solution=pressure,
                    iterations=iteration + 1,
                    residual_norm=max(p_norm, s_norm),
                    elapsed_time=elapsed,
                    message=f"Converged in {iteration + 1} iterations",
                    convergence_history=convergence_history,
                )

            # Calculate Jacobian (simplified)
            J = self._calculate_jacobian(pressure, saturations, dt)

            # Solve linear system (simplified - diagonal only)
            # In reality, would use sparse linear solver
            try:
                delta_p = residual_pressure / (np.diag(J) + 1e-10)
                delta_s = residual_saturation / (np.diag(J) + 1e-10)

                # Update
                damping = 0.5  # Damping factor
                pressure -= damping * delta_p
                saturations -= damping * delta_s

                # Clip saturations to valid range
                saturations = np.clip(saturations, 0, 1)
                if saturations.ndim == 2:
                    row_sums = np.sum(saturations, axis=1, keepdims=True)
                    saturations = saturations / np.maximum(row_sums, 1e-10)

            except np.linalg.LinAlgError:
                elapsed = timer.time() - start_time
                return SolverResult(
                    status=SolverStatus.SINGULAR,
                    solution=pressure,
                    iterations=iteration + 1,
                    residual_norm=max(p_norm, s_norm),
                    elapsed_time=elapsed,
                    message="Singular Jacobian matrix",
                    convergence_history=convergence_history,
                )

        # Max iterations reached
        elapsed = timer.time() - start_time
        return SolverResult(
            status=SolverStatus.MAX_ITERATIONS,
            solution=pressure,
            iterations=self.max_iterations,
            residual_norm=max(p_norm, s_norm),
            elapsed_time=elapsed,
            message="Maximum Newton iterations reached",
            convergence_history=convergence_history,
        )

    def _calculate_residuals(
        self,
        pressure: np.ndarray,
        saturations: np.ndarray,
        dt: float,
        source_terms: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate residual equations."""
        # Very simplified residual calculation
        # In reality, would calculate flux divergence, accumulation, etc.

        n_cells = len(pressure)

        # Pressure residual: mass balance
        residual_pressure = np.zeros(n_cells)

        # Saturation residual: phase balance
        if saturations.ndim == 2:
            residual_saturation = np.zeros_like(saturations)
        else:
            residual_saturation = np.zeros(n_cells)

        return residual_pressure, residual_saturation

    def _calculate_jacobian(
        self, pressure: np.ndarray, saturations: np.ndarray, dt: float
    ) -> np.ndarray:
        """Calculate Jacobian matrix."""
        # Simplified diagonal Jacobian
        # In reality, would calculate derivatives of residuals
        n = len(pressure)
        return np.eye(n) * 1.0


def create_flow_solver(config: "EngineConfig") -> FlowSolver:
    """
    Factory function to create appropriate flow solver.

    Args:
        config: Engine configuration.

    Returns:
        FlowSolver instance (ExplicitFlowSolver or ImplicitFlowSolver).
    """
    solver_type = config.get_effective_solver_type()

    if solver_type == SolverType.EXPLICIT:
        return ExplicitFlowSolver(config)
    elif solver_type == SolverType.IMPLICIT:
        return ImplicitFlowSolver(config)
    else:  # ADAPTIVE
        if config.mode == "simple":
            return ExplicitFlowSolver(config)
        else:
            return ImplicitFlowSolver(config)
