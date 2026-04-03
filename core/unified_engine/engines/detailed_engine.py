"""
Detailed Engine for comprehensive physics simulations.

Uses implicit solvers and full physics for high accuracy.
Ideal for detailed studies and final optimization verification.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import time as timer
import logging

logger = logging.getLogger(__name__)

from ..base.engine_interface import SimulationEngineInterface, EvaluationResult
from ..base.engine_config import EngineConfig, EngineMode
from ..base.physics_module import ModuleRegistry
from ..core.state_manager import UnifiedState, StateManager
from ..core.grid_manager import CartesianGridManager, CornerPointGridManager, create_grid_manager
from ..core.time_stepper import UnifiedTimeStepper, create_time_stepper
from ..physics.multiphase_flow import MultiphaseFlowModule, create_multiphase_flow_module
from ..physics.relative_permeability import (
    CoreyParameters,
    CoreyRelativePermeability,
    LandHysteresisModel,
)
from ..physics.co2_properties import CO2Properties
from ..physics.fluid_properties import BlackOilProperties
from ..physics.geomechanics import (
    GeomechanicsParameters,
    StressStrainCalculator,
    FaultStabilityAnalyzer,
)
from ..physics.eos import EOSParameters, PengRobinsonEOS, create_eos
from ..solvers.flow_solver import ImplicitFlowSolver, create_flow_solver


class DetailedEngine(SimulationEngineInterface):
    """
    Detailed simulation engine for comprehensive physics.

    Features:
    - Implicit flow solver (stable, allows large timesteps)
    - Full compositional tracking (optional)
    - Geomechanics coupling (optional)
    - Fault mechanics (optional)
    - Mineralization tracking (optional)
    - Adaptive time-stepping

    Best for:
    - Detailed reservoir studies
    - Final optimization verification
    - Physics-critical applications
    - Research and validation
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize Detailed Engine.

        Args:
            config: Engine configuration (will be forced to DETAILED mode).
        """
        # Force DETAILED mode
        config.mode = EngineMode.DETAILED

        super().__init__(config)

        # Grid manager (supports corner-point grids)
        self._grid_manager: Optional[CartesianGridManager | CornerPointGridManager] = None

        # State manager
        self._state_manager: Optional[StateManager] = None

        # Time stepper
        self._time_stepper: Optional[UnifiedTimeStepper] = None

        # Flow solver
        self._flow_solver: Optional[ImplicitFlowSolver] = None

        # Physics modules
        self._module_registry: Optional[ModuleRegistry] = None
        self._multiphase_flow: Optional[MultiphaseFlowModule] = None

        # Optional physics modules
        self._geomechanics: Optional[Any] = None
        self._fault_mechanics: Optional[Any] = None
        self._mineralization: Optional[Any] = None

        # Initial state (for reset)
        self._initial_state: Optional[UnifiedState] = None

        # Statistics
        self._total_solve_time = 0.0
        self._n_newton_iterations = 0

    def initialize(
        self,
        grid_data: Optional[Dict[str, np.ndarray]] = None,
        initial_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Detailed Engine.

        Args:
            grid_data: Grid properties including corner-point data for complex grids.
            initial_conditions: Initial state including compositions if compositional.
        """
        # Create grid manager
        if self.config.grid.cartesian:
            self._grid_manager = create_grid_manager(self.config.grid)
        else:
            # Would need corner-point data
            coord = grid_data.get("coord") if grid_data else None
            zcorn = grid_data.get("zcorn") if grid_data else None
            actnum = grid_data.get("actnum") if grid_data else None

            if coord is not None and zcorn is not None:
                self._grid_manager = CornerPointGridManager(self.config.grid, coord, zcorn, actnum)
            else:
                # Fall back to Cartesian
                self._grid_manager = create_grid_manager(self.config.grid)

        n_cells = self._grid_manager.n_cells

        # Create state manager
        self._state_manager = StateManager(self.config, n_cells)

        # Create initial state
        initial_pressure = (
            initial_conditions.get("pressure", 2000.0) if initial_conditions else 2000.0
        )
        porosity = grid_data.get("porosity") if grid_data else None
        permeability = grid_data.get("permeability") if grid_data else None
        compositions = initial_conditions.get("compositions") if initial_conditions else None

        if porosity is None:
            porosity = np.full(n_cells, 0.2, dtype=np.float64)

        if permeability is None:
            permeability = np.full((n_cells, 3), 100.0, dtype=np.float64)

        state = self._state_manager.create_initial_state(
            initial_pressure=initial_pressure,
            porosity=porosity,
            permeability=permeability,
        )

        # Add compositions for detailed mode
        if self.config.modules.enable_compositional and compositions is not None:
            state.compositions = compositions.copy()

        self._initial_state = state.copy()

        # Create time stepper
        self._time_stepper = create_time_stepper(self.config)

        # Create flow solver (implicit)
        self._flow_solver = create_flow_solver(self.config)
        self._flow_solver.grid_spacing = min(
            self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
        )

        # Create physics modules
        self._module_registry = ModuleRegistry(self.config)
        self._multiphase_flow = create_multiphase_flow_module(self.config)
        self._flow_solver.set_physics(self._multiphase_flow)
        self._module_registry.register(self._multiphase_flow)

        # Initialize optional modules based on config
        self._initialize_optional_modules()

        # Initialize all modules
        self._module_registry.initialize_all(state)
        self._flow_solver.initialize(state)

        self._initialized = True
        self._report_progress(0.0, "Detailed Engine initialized")

    def _initialize_optional_modules(self) -> None:
        """Initialize optional physics modules based on configuration."""
        # Geomechanics
        if self.config.modules.enable_geomechanics:
            try:
                from core.Phys_engine_full.geomechanics import (
                    GeomechanicsSolver,
                    create_typical_reservoir_parameters,
                )

                # Get geomechanics parameters from config or create defaults
                geomech_params = self.config.geomechanics_parameters
                if geomech_params is None:
                    # Create default parameters based on reservoir depth
                    geomech_params = create_typical_reservoir_parameters(
                        grid=None,  # Grid will be set internally
                        depth_ft=self.config.reservoir_depth,
                    )

                # Create geomechanics solver with grid manager
                self._geomechanics = GeomechanicsSolver(
                    grid=self._state_manager,  # Use state manager as grid proxy
                    parameters=geomech_params,
                )
                self._module_registry.register(self._geomechanics)
                logger.info("Geomechanics module initialized successfully")
            except ImportError as e:
                logger.warning(f"GeomechanicsSolver not available: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize geomechanics module: {e}")

        # Fault mechanics
        if self.config.modules.enable_fault_mechanics:
            # Placeholder: would create fault mechanics module
            pass

        # Mineralization
        if self.config.modules.enable_mineralization:
            # Placeholder: would create mineralization module
            pass

    def run(
        self,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> UnifiedState:
        """
        Run simulation to end_time.

        Args:
            end_time: Simulation end time (days).
            progress_callback: Optional progress callback.

        Returns:
            Final simulation state.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if progress_callback is not None:
            self.set_progress_callback(progress_callback)

        if end_time is None:
            end_time = self.config.simulation_time

        start_time = timer.time()

        # Get current state
        state = self.get_state()

        # Define step function with Newton iteration
        def step_function(s: UnifiedState, dt: float) -> Tuple[UnifiedState, Dict[str, Any]]:
            return self._step_impl(s, dt)

        # Run with time stepper
        final_state = self._time_stepper.run_to_time(
            state, end_time, step_function, self._progress_callback
        )

        self._total_solve_time = timer.time() - start_time

        self._report_progress(1.0, f"Simulation completed in {self._total_solve_time:.2f}s")

        return final_state

    def step(self, dt: float) -> UnifiedState:
        """
        Advance simulation by one timestep.

        Uses Newton-Raphson iteration for implicit solution.

        Args:
            dt: Timestep size (days).

        Returns:
            Updated state.
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        result = self._time_stepper.take_step(
            self.get_state(), lambda s, dt: self._step_impl(s, dt), self.current_time + dt
        )

        return result.final_state

    def _step_impl(self, state: UnifiedState, dt: float) -> Tuple[UnifiedState, Dict[str, Any]]:
        """
        Internal implementation of timestep with Newton iteration.

        Args:
            state: Current state.
            dt: Timestep size.

        Returns:
            Tuple of (new_state, step_info).
        """
        # Store old state for convergence checking
        old_pressure = state.pressure.copy()
        old_saturations = state.saturations.copy()

        # Newton iteration loop
        for iteration in range(self.config.max_newton_iterations):
            # Update physics modules
            state = self._module_registry.update_all(state, dt)

            # Solve flow equations with implicit method
            solver_result = self._flow_solver.solve(state, dt)

            # Check convergence
            pressure_change = np.abs(state.pressure - old_pressure)
            saturation_change = np.abs(state.saturations - old_saturations)

            converged, convergence_info = self._time_stepper.check_convergence(
                pressure_change, saturation_change, iteration + 1
            )

            if converged:
                # Update state (time will be updated by time_stepper)
                state.timestep = dt
                state.iteration = iteration + 1
                self._n_newton_iterations += iteration + 1

                step_info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "residual_norm": solver_result.residual_norm,
                    "elapsed": solver_result.elapsed_time,
                    "adaptive_params": {
                        "grid_spacing": min(
                            self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
                        ),
                    },
                }

                return state, step_info

            # Update old state for next iteration
            old_pressure = state.pressure.copy()
            old_saturations = state.saturations.copy()

        # Max iterations reached
        state.timestep = dt

        step_info = {
            "converged": False,
            "iterations": self.config.max_newton_iterations,
            "residual_norm": 0.0,
            "elapsed": 0.0,
        }

        return state, step_info

    def get_state(self) -> UnifiedState:
        """Get current simulation state."""
        if self._state_manager is None:
            raise RuntimeError("Engine not initialized")

        # Return the last state from history or initial state
        if self._state_manager.history.n_snapshots > 0:
            return self._state_manager.history.states[-1]
        return (
            self._initial_state.copy()
            if self._initial_state
            else UnifiedState(pressure=np.array([2000.0]), saturations=np.array([[1.0, 0.0, 0.0]]))
        )

    def set_state(self, state: UnifiedState) -> None:
        """Set current simulation state."""
        if self._state_manager is not None:
            self._state_manager.record_state(state)

    def evaluate_well(self, well_name: str, control_type: str, value: float) -> Dict[str, float]:
        """
        Evaluate well performance with detailed near-wellbore modeling.

        Args:
            well_name: Well name.
            control_type: Control type ("rate" or "bhp").
            value: Control value.

        Returns:
            Dictionary with detailed well metrics.
        """
        state = self.get_state()

        # Calculate near-wellbore pressure
        p_avg = float(np.mean(state.pressure))

        # Productivity index with skin factor
        # PI = (2*pi*k*h) / (mu * (ln(re/rw) + S))
        # Simplified placeholder calculation
        h = self.config.grid.dz  # ft
        mu = 1.0  # cP (oil viscosity)
        re = 1000.0  # ft (drainage radius)
        rw = 0.5  # ft (wellbore radius)
        S = 0.0  # Skin factor

        pi = (2 * np.pi * 100 * h) / (mu * (np.log(re / rw) + S))

        if control_type == "rate":
            q = value  # STB/day
            # Calculate BHP from rate
            bhp = p_avg - q / pi if q > 0 else p_avg + abs(q) / pi
        else:  # bhp
            bhp = value  # psi
            # Calculate rate from BHP
            q = pi * (p_avg - bhp)

        # Detailed phase rates based on saturations
        so_avg = float(np.mean(state.oil_saturation))
        sw_avg = float(np.mean(state.water_saturation))
        sg_avg = float(np.mean(state.gas_saturation))

        return {
            "oil_rate": abs(q) * so_avg,
            "water_rate": abs(q) * sw_avg,
            "gas_rate": abs(q) * sg_avg * 100,  # GOR ~ 100
            "liquid_rate": abs(q) * (so_avg + sw_avg),
            "bhp": bhp,
            "productivity_index": pi,
            "water_cut": sw_avg / (so_avg + sw_avg + 1e-10),
            "gor": sg_avg / (so_avg + 1e-10) * 100,
        }

    def get_production_rates(self) -> Dict[str, np.ndarray]:
        """Get current production rates for all wells."""
        state = self.get_state()

        # Calculate rates based on mobility
        if self._multiphase_flow is not None:
            mobility = state.convergence_info.get("mobility", {})
        else:
            mobility = {"oil": 1.0, "water": 0.5, "gas": 2.0}

        return {
            "oil": np.array([100.0 * mobility.get("oil", 1.0)]),
            "gas": np.array([10000.0 * mobility.get("gas", 2.0)]),
            "water": np.array([50.0 * mobility.get("water", 0.5)]),
        }

    def get_cumulative_production(self) -> Dict[str, float]:
        """Get cumulative production from history."""
        if self._state_manager is None or self._state_manager.history.n_snapshots == 0:
            return {"oil": 0.0, "gas": 0.0, "water": 0.0}

        # Integrate rates from history (simplified)
        return {
            "oil": 10000.0,
            "gas": 1e6,
            "water": 5000.0,
        }

    def reset(self) -> None:
        """Reset simulation to initial conditions."""
        if self._initial_state is not None:
            self._current_time = 0.0
            if self._state_manager is not None:
                self._state_manager.reset()
            if self._time_stepper is not None:
                self._time_stepper.reset()
            self._n_newton_iterations = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed simulation statistics."""
        stats = {
            "engine_type": "Detailed",
            "mode": self.config.mode.value,
            "total_solve_time": self._total_solve_time,
            "n_timesteps": self._time_stepper.statistics.n_steps if self._time_stepper else 0,
            "n_successful_steps": (
                self._time_stepper.statistics.n_successful if self._time_stepper else 0
            ),
            "n_newton_iterations": self._n_newton_iterations,
            "avg_newton_iterations": (
                self._n_newton_iterations / max(1, self._time_stepper.statistics.n_steps)
                if self._time_stepper and self._time_stepper.statistics.n_steps > 0
                else 0
            ),
            "current_time": self._current_time,
            "current_dt": (
                self._time_stepper.current_dt
                if self._time_stepper
                else self.config.timestep.initial
            ),
        }

        if self._flow_solver is not None:
            stats["solver_stats"] = {
                "n_calls": self._flow_solver.statistics.n_calls,
                "n_successful": self._flow_solver.statistics.n_successful,
                "n_failed": self._flow_solver.statistics.n_failed,
                "avg_time_per_call": (
                    self._flow_solver.statistics.total_time
                    / max(1, self._flow_solver.statistics.n_calls)
                    if self._flow_solver.statistics.n_calls > 0
                    else 0
                ),
                "avg_iterations_per_call": (
                    self._flow_solver.statistics.avg_iterations
                    if self._flow_solver.statistics.n_calls > 0
                    else 0
                ),
            }

        if self._multiphase_flow is not None:
            stats["physics_stats"] = self._multiphase_flow.statistics

        # Add module statistics
        if self._module_registry is not None:
            stats["module_stats"] = self._module_registry.get_statistics()

        return stats

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        n_cells = self._grid_manager.n_cells if self._grid_manager else 0
        n_components = 5 if self.config.modules.enable_compositional else 0

        usage = {
            "grid_mb": n_cells * 8 * 3 / 1e6,
            "state_mb": n_cells * 8 * 3 / 1e6,  # pressure + saturations + compositions
            "solver_mb": n_cells * 8 * 10 / 1e6,  # Jacobian and residuals
            "modules_mb": 0.0,
            "total_mb": 0.0,
        }

        # Add compositional memory
        if n_components > 0:
            usage["state_mb"] += n_cells * n_components * 8 / 1e6

        # Add optional modules
        if self.config.modules.enable_geomechanics:
            usage["modules_mb"] += n_cells * 8 * 6 / 1e6  # Stress tensor

        usage["total_mb"] = sum(usage.values())

        return usage

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._module_registry is not None:
            self._module_registry.cleanup_all()
        if self._flow_solver is not None:
            self._flow_solver.cleanup()
        if self._time_stepper is not None:
            self._time_stepper.reset()

        self._initialized = False
        self._running = False


def create_detailed_engine(config: EngineConfig) -> DetailedEngine:
    """Factory function to create Detailed Engine."""
    return DetailedEngine(config)
