"""
Abstract base interface for the Unified Physics Engine.

Defines the common interface that both FastEngine and DetailedEngine must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np

from .engine_config import EngineConfig, EngineMode
from ..core.state_manager import UnifiedState


class SimulationEngineInterface(ABC):
    """
    Abstract interface for simulation engines.

    Both FastEngine and DetailedEngine inherit from this interface, ensuring
    they can be used interchangeably in the optimization framework.
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize the engine with configuration.

        Args:
            config: Engine configuration specifying mode, solvers, and modules.
        """
        self.config = config
        self._validate_config()

        # Engine state
        self._initialized = False
        self._running = False
        self._current_time = 0.0

        # Callbacks for progress reporting
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._convergence_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    @property
    def mode(self) -> EngineMode:
        """Get the engine mode."""
        return self.config.mode

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self._current_time

    @abstractmethod
    def initialize(
        self,
        grid_data: Optional[Dict[str, np.ndarray]] = None,
        initial_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the engine with grid and initial conditions.

        Args:
            grid_data: Optional grid properties (porosity, permeability, etc.)
            initial_conditions: Optional initial state (pressure, saturations, etc.)
        """
        pass

    @abstractmethod
    def run(
        self,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UnifiedState:
        """
        Run the simulation to end_time.

        Args:
            end_time: Simulation end time (days). If None, uses config.simulation_time.
            progress_callback: Optional callback for progress updates.

        Returns:
            Final UnifiedState after simulation.
        """
        pass

    @abstractmethod
    def step(self, dt: float) -> UnifiedState:
        """
        Advance simulation by one timestep.

        Args:
            dt: Timestep size (days).

        Returns:
            Updated UnifiedState.
        """
        pass

    @abstractmethod
    def get_state(self) -> UnifiedState:
        """
        Get current simulation state.

        Returns:
            Current UnifiedState.
        """
        pass

    @abstractmethod
    def set_state(self, state: UnifiedState) -> None:
        """
        Set current simulation state.

        Args:
            state: UnifiedState to set as current.
        """
        pass

    @abstractmethod
    def evaluate_well(
        self,
        well_name: str,
        control_type: str,
        value: float,
    ) -> Dict[str, float]:
        """
        Evaluate a well's performance under specified control.

        Args:
            well_name: Name of the well.
            control_type: Type of control ("rate" or "bhp").
            value: Control value (rate in STB/day or BHP in psi).

        Returns:
            Dictionary with well performance metrics.
        """
        pass

    @abstractmethod
    def get_production_rates(self) -> Dict[str, np.ndarray]:
        """
        Get current production/injection rates for all wells.

        Returns:
            Dictionary mapping well names to rate arrays [oil, gas, water].
        """
        pass

    @abstractmethod
    def get_cumulative_production(self) -> Dict[str, float]:
        """
        Get cumulative production/injection for all wells.

        Returns:
            Dictionary with cumulative oil, gas, water production.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial conditions."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulation statistics.

        Returns:
            Dictionary with timestep info, iterations, timing, etc.
        """
        pass

    def set_progress_callback(
        self, callback: Callable[[float, str], None]
    ) -> None:
        """
        Set callback for progress updates.

        Args:
            callback: Function taking (progress_fraction, message) arguments.
        """
        self._progress_callback = callback

    def set_convergence_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Set callback for convergence monitoring.

        Args:
            callback: Function taking convergence info dict.
        """
        self._convergence_callback = callback

    def _report_progress(self, progress: float, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback is not None:
            self._progress_callback(progress, message)

    def _report_convergence(self, info: Dict[str, Any]) -> None:
        """Report convergence if callback is set."""
        if self._convergence_callback is not None:
            self._convergence_callback(info)

    def _validate_config(self) -> None:
        """Validate engine configuration."""
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage in MB for different components.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        pass


class EvaluationResult:
    """
    Result from an engine evaluation.

    Contains the objective function value, constraints, and diagnostic information.
    """

    def __init__(
        self,
        objective_value: float,
        success: bool,
        state: UnifiedState,
        constraints: Optional[Dict[str, float]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        message: str = "",
    ):
        self.objective_value = objective_value
        self.success = success
        self.state = state
        self.constraints = constraints or {}
        self.diagnostics = diagnostics or {}
        self.message = message

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(objective={self.objective_value:.4g}, "
            f"success={self.success}, message='{self.message}')"
        )


class BatchSimulationResult:
    """
    Result from batch simulations (e.g., parameter sweeps, uncertainty quantification).
    """

    def __init__(
        self,
        results: List[EvaluationResult],
        parameter_sets: List[Dict[str, float]],
        total_time: float,
    ):
        self.results = results
        self.parameter_sets = parameter_sets
        self.total_time = total_time

    @property
    def n_simulations(self) -> int:
        """Number of simulations in batch."""
        return len(self.results)

    @property
    def n_successful(self) -> int:
        """Number of successful simulations."""
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        """Number of failed simulations."""
        return sum(1 for r in self.results if not r.success)

    @property
    def objective_values(self) -> np.ndarray:
        """Array of objective values from all simulations."""
        return np.array([r.objective_value for r in self.results])

    @property
    def success_mask(self) -> np.ndarray:
        """Boolean mask of successful simulations."""
        return np.array([r.success for r in self.results])

    def get_statistics(self) -> Dict[str, float]:
        """Get statistics of objective values from successful runs."""
        successful_values = self.objective_values[self.success_mask]
        if len(successful_values) == 0:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "median": np.nan,
            }
        return {
            "mean": float(np.mean(successful_values)),
            "std": float(np.std(successful_values)),
            "min": float(np.min(successful_values)),
            "max": float(np.max(successful_values)),
            "median": float(np.median(successful_values)),
        }
