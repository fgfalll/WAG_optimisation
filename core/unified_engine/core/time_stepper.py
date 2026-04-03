"""
Unified Time-Stepping for the Unified Physics Engine.

Provides adaptive time-stepping that works for both Simple (explicit)
and Detailed (implicit) engine modes.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Tuple
from enum import Enum
import numpy as np
import time as timer

from ..base.engine_config import EngineConfig, SolverType, EngineMode
from .state_manager import UnifiedState


class TimestepStatus(Enum):
    """Status of a timestep attempt."""
    SUCCESS = "success"
    CONVERGED = "converged"
    FAILED = "failed"
    CUTBACK = "cutback"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class TimestepResult:
    """Result of a timestep attempt."""
    status: TimestepStatus
    dt: float
    iterations: int
    final_state: UnifiedState
    message: str = ""
    convergence_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if timestep was successful."""
        return self.status in (TimestepStatus.SUCCESS, TimestepStatus.CONVERGED)


@dataclass
class TimestepStatistics:
    """Statistics for time-stepping performance."""
    n_steps: int = 0
    n_successful: int = 0
    n_failed: int = 0
    n_cutbacks: int = 0
    total_time: float = 0.0
    dt_min: float = float("inf")
    dt_max: float = 0.0
    dt_average: float = 0.0

    def update(self, result: TimestepResult, elapsed: float) -> None:
        """Update statistics with a timestep result."""
        self.n_steps += 1
        self.total_time += elapsed
        self.dt_min = min(self.dt_min, result.dt)
        self.dt_max = max(self.dt_max, result.dt)
        self.dt_average = self.total_time / self.n_steps

        if result.is_successful:
            self.n_successful += 1
        else:
            self.n_failed += 1

        if result.status == TimestepStatus.CUTBACK:
            self.n_cutbacks += 1


class CFLCalculator:
    """
    CFL (Courant-Friedrichs-Lewy) condition calculator.

    Used for adaptive time-stepping in explicit schemes.
    """

    def __init__(self, safety_factor: float = 0.8):
        """
        Initialize CFL calculator.

        Args:
            safety_factor: Safety factor for CFL condition (< 1.0).
        """
        self.safety_factor = safety_factor

    def calculate_dt(
        self,
        permeability: np.ndarray,
        porosity: np.ndarray,
        viscosity: float,
        compressibility: float,
        grid_spacing: float,
        max_pressure_change: Optional[float] = None,
    ) -> float:
        """
        Calculate maximum stable timestep based on CFL condition.

        Args:
            permeability: Permeability field (mD).
            porosity: Porosity field (fraction).
            viscosity: Fluid viscosity (cP).
            compressibility: Total compressibility (1/psi).
            grid_spacing: Characteristic grid spacing (ft).
            max_pressure_change: Maximum allowed pressure change per step (psi).

        Returns:
            Maximum stable timestep (days).
        """
        # Calculate maximum velocity
        k_max = np.max(permeability) if permeability.ndim == 1 else np.max(permeability[:, 0])
        phi_min = np.min(porosity)

        # Darcy velocity approximation
        velocity = 6.33e-3 * k_max / viscosity  # ft/day/psi

        # CFL-based timestep
        dt_cfl = self.safety_factor * grid_spacing / velocity / compressibility

        # Pressure-based timestep
        if max_pressure_change is not None:
            dt_pressure = max_pressure_change / (velocity * compressibility)
            dt_cfl = min(dt_cfl, dt_pressure)

        return max(dt_cfl, 0.001)  # Minimum 0.001 days


class UnifiedTimeStepper:
    """
    Unified time-stepper for both Simple and Detailed engine modes.

    Provides:
    - Adaptive time-stepping based on convergence and stability
    - CFL-based control for explicit schemes
    - Newton iteration control for implicit schemes
    - Automatic cutback on failure
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize time-stepper.

        Args:
            config: Engine configuration.
        """
        self.config = config
        self.ts_config = config.timestep

        # Current timestep
        self._current_dt = self.ts_config.initial

        # Statistics
        self._statistics = TimestepStatistics()

        # Callback for convergence monitoring
        self._convergence_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # CFL calculator (for explicit schemes)
        self._cfl_calculator = CFLCalculator(
            safety_factor=self.ts_config.cfl_safety_factor
        )

    @property
    def current_dt(self) -> float:
        """Get current timestep size."""
        return self._current_dt

    @property
    def statistics(self) -> TimestepStatistics:
        """Get time-stepping statistics."""
        return self._statistics

    def set_convergence_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Set callback for convergence monitoring."""
        self._convergence_callback = callback

    def reset(self) -> None:
        """Reset time-stepper to initial state."""
        self._current_dt = self.ts_config.initial
        self._statistics = TimestepStatistics()

    def calculate_next_dt(
        self,
        state: UnifiedState,
        iteration_count: int,
        converged: bool,
        grid_spacing: float = 100.0,
        permeability: Optional[np.ndarray] = None,
        porosity: Optional[np.ndarray] = None,
        viscosity: float = 1.0,
        compressibility: float = 1e-5,
    ) -> float:
        """
        Calculate next timestep based on current performance.

        Args:
            state: Current simulation state.
            iteration_count: Number of iterations in last step.
            converged: Whether last step converged.
            grid_spacing: Characteristic grid spacing (ft).
            permeability: Permeability field (for CFL).
            porosity: Porosity field (for CFL).
            viscosity: Fluid viscosity (cP).
            compressibility: Total compressibility (1/psi).

        Returns:
            Next timestep size (days).
        """
        if not converged:
            # Cut back on failure
            new_dt = max(
                self._current_dt * self.ts_config.cutback_factor,
                self.ts_config.minimum,
            )
            return new_dt

        # Adaptive timestep based on iterations
        if not self.ts_config.adaptive:
            return self._current_dt

        # For explicit schemes, use CFL
        if (
            self.config.get_effective_solver_type() == SolverType.EXPLICIT
            and permeability is not None
            and porosity is not None
        ):
            dt_cfl = self._cfl_calculator.calculate_dt(
                permeability=permeability,
                porosity=porosity,
                viscosity=viscosity,
                compressibility=compressibility,
                grid_spacing=grid_spacing,
            )
            dt_cfl = min(dt_cfl, self.ts_config.maximum)
            return dt_cfl

        # For implicit schemes, adjust based on iterations
        target_iterations = 3  # Target Newton iterations
        if iteration_count <= target_iterations:
            # Increase timestep
            growth = min(
                self.ts_config.growth_factor ** (target_iterations - iteration_count + 1),
                self.ts_config.growth_factor,
            )
            new_dt = min(self._current_dt * growth, self.ts_config.maximum)
        elif iteration_count <= self.ts_config.max_iterations:
            # Slight decrease
            new_dt = max(
                self._current_dt / (1 + 0.1 * (iteration_count - target_iterations)),
                self.ts_config.minimum,
            )
        else:
            # Cut back
            new_dt = max(
                self._current_dt * self.ts_config.cutback_factor,
                self.ts_config.minimum,
            )

        return new_dt

    def check_convergence(
        self,
        pressure_change: np.ndarray,
        saturation_change: np.ndarray,
        iteration: int,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if Newton iterations have converged.

        Args:
            pressure_change: Pressure residual (psi).
            saturation_change: Saturation residual (fraction).
            iteration: Current iteration number.

        Returns:
            Tuple of (converged, convergence_info).
        """
        p_norm = float(np.max(np.abs(pressure_change)))
        s_norm = float(np.max(np.abs(saturation_change)))

        converged = (
            p_norm < self.config.tolerance_pressure
            and s_norm < self.config.tolerance_saturation
        )

        info = {
            "iteration": iteration,
            "pressure_norm": p_norm,
            "saturation_norm": s_norm,
            "pressure_tolerance": self.config.tolerance_pressure,
            "saturation_tolerance": self.config.tolerance_saturation,
            "converged": converged,
        }

        if self._convergence_callback is not None:
            self._convergence_callback(info)

        return converged, info

    def take_step(
        self,
        state: UnifiedState,
        step_function: Callable[[UnifiedState, float], Tuple[UnifiedState, Dict[str, Any]]],
        end_time: float,
    ) -> TimestepResult:
        """
        Attempt to take a timestep.

        Args:
            state: Current simulation state.
            step_function: Function that advances state by dt. Returns (new_state, info).
            end_time: Target end time (will not exceed).

        Returns:
            TimestepResult with status and updated state.
        """
        start_time = timer.time()

        # Determine timestep size
        dt = min(self._current_dt, end_time - state.current_time)

        # Attempt the step
        try:
            new_state, step_info = step_function(state, dt)
            elapsed = timer.time() - start_time

            # Extract convergence info
            converged = step_info.get("converged", True)
            iterations = step_info.get("iterations", 1)

            if converged:
                # Successful step
                status = TimestepStatus.SUCCESS
                message = f"Step completed in {iterations} iterations"

                # Update state time
                new_state.current_time = state.current_time + dt

                # Update timestep for next step
                self._current_dt = self.calculate_next_dt(
                    new_state,
                    iterations,
                    True,
                    **step_info.get("adaptive_params", {}),
                )
            else:
                # Non-convergence - cut back
                status = TimestepStatus.MAX_ITERATIONS
                message = "Maximum Newton iterations reached"
                new_state = state  # Revert to old state
                self._current_dt = max(
                    self._current_dt * self.ts_config.cutback_factor,
                    self.ts_config.minimum,
                )

        except Exception as e:
            # Step failed - cut back
            elapsed = timer.time() - start_time
            status = TimestepStatus.FAILED
            message = f"Step failed: {str(e)}"
            new_state = state
            step_info = {"error": str(e)}
            self._current_dt = max(
                self._current_dt * self.ts_config.cutback_factor,
                self.ts_config.minimum,
            )

        # Update statistics
        result = TimestepResult(
            status=status,
            dt=dt,
            iterations=step_info.get("iterations", 0),
            final_state=new_state,
            message=message,
            convergence_info=step_info,
        )
        self._statistics.update(result, elapsed)

        return result

    def run_to_time(
        self,
        initial_state: UnifiedState,
        end_time: float,
        step_function: Callable[[UnifiedState, float], Tuple[UnifiedState, Dict[str, Any]]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UnifiedState:
        """
        Run simulation from current time to end_time.

        Args:
            initial_state: Initial simulation state.
            end_time: Target end time (days).
            step_function: Function that advances state by dt.
            progress_callback: Optional callback for progress updates.

        Returns:
            Final simulation state.
        """
        state = initial_state
        start_time = state.current_time

        while state.current_time < end_time - 1e-6:
            # Report progress
            if progress_callback is not None:
                progress = (state.current_time - start_time) / (end_time - start_time)
                progress_callback(
                    progress,
                    f"t={state.current_time:.2f} days, dt={self._current_dt:.3f} days",
                )

            # Take a step
            result = self.take_step(state, step_function, end_time)
            state = result.final_state

            if not result.is_successful:
                # Report failure but continue with cutback
                if progress_callback is not None:
                    progress_callback(
                        (state.current_time - start_time) / (end_time - start_time),
                        f"Cutback: {result.message}",
                    )

        return state


def create_time_stepper(config: EngineConfig) -> UnifiedTimeStepper:
    """
    Factory function to create appropriate time-stepper.

    Args:
        config: Engine configuration.

    Returns:
        UnifiedTimeStepper instance.
    """
    return UnifiedTimeStepper(config)
