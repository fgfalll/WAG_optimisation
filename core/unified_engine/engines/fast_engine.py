"""
Fast Engine for screening and optimization simulations.

Uses explicit solvers and simplified physics for fast execution.
Ideal for optimization loops requiring many function evaluations.
"""

from typing import Dict, Any, Optional, Callable, List
import numpy as np
import time as timer

from ..base.engine_interface import SimulationEngineInterface, EvaluationResult
from ..base.engine_config import EngineConfig, EngineMode
from ..base.physics_module import ModuleRegistry
from ..core.state_manager import UnifiedState, StateManager
from ..core.grid_manager import CartesianGridManager, create_grid_manager
from ..core.time_stepper import UnifiedTimeStepper, create_time_stepper
from ..physics.multiphase_flow import MultiphaseFlowModule, create_multiphase_flow_module
from ..physics.relative_permeability import CoreyParameters, CoreyRelativePermeability
from ..physics.co2_properties import CO2Properties
from ..solvers.flow_solver import ExplicitFlowSolver, create_flow_solver


class FastEngine(SimulationEngineInterface):
    """
    Fast simulation engine for screening and optimization.

    Features:
    - Explicit flow solver (fast, conditionally stable)
    - Analytical pressure solution
    - Simplified multiphase flow (no compositional tracking)
    - Minimal physics overhead
    - Optimized for many sequential evaluations

    Best for:
    - Optimization loops
    - Parameter sweeps
    - Quick screening studies
    """

    def __init__(self, config: EngineConfig):
        """
        Initialize Fast Engine.

        Args:
            config: Engine configuration (will be forced to SIMPLE mode).
        """
        # Force SIMPLE mode
        config.mode = EngineMode.SIMPLE

        super().__init__(config)

        # Grid manager
        self._grid_manager: Optional[CartesianGridManager] = None

        # State manager
        self._state_manager: Optional[StateManager] = None

        # Time stepper
        self._time_stepper: Optional[UnifiedTimeStepper] = None

        # Flow solver
        self._flow_solver: Optional[ExplicitFlowSolver] = None

        # Physics modules
        self._module_registry: Optional[ModuleRegistry] = None
        self._multiphase_flow: Optional[MultiphaseFlowModule] = None

        # Initial state (for reset)
        self._initial_state: Optional[UnifiedState] = None

        # Statistics
        self._total_solve_time = 0.0
        self._n_timesteps = 0

    def initialize(
        self,
        grid_data: Optional[Dict[str, np.ndarray]] = None,
        initial_conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Fast Engine.

        Args:
            grid_data: Optional grid properties (porosity, permeability).
            initial_conditions: Optional initial state.
        """
        # Create grid manager
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

        if porosity is None:
            porosity = np.full(n_cells, 0.2, dtype=np.float64)

        if permeability is None:
            # Default 100 mD isotropic
            permeability = np.full(n_cells, 100.0, dtype=np.float64)

        state = self._state_manager.create_initial_state(
            initial_pressure=initial_pressure,
            porosity=porosity,
            permeability=permeability,
        )

        self._initial_state = state.copy()

        # Create time stepper
        self._time_stepper = create_time_stepper(self.config)

        # Create flow solver
        self._flow_solver = create_flow_solver(self.config)
        self._flow_solver.grid_spacing = min(
            self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
        )

        # Create physics modules
        self._module_registry = ModuleRegistry(self.config)
        self._multiphase_flow = create_multiphase_flow_module(self.config)
        self._flow_solver.set_physics(self._multiphase_flow)
        self._module_registry.register(self._multiphase_flow)

        # Initialize modules
        self._module_registry.initialize_all(state)
        self._flow_solver.initialize(state)

        self._initialized = True
        self._report_progress(0.0, "Fast Engine initialized")

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

        # Define step function
        def step_function(s: UnifiedState, dt: float) -> tuple[UnifiedState, Dict[str, Any]]:
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

    def _step_impl(self, state: UnifiedState, dt: float) -> tuple[UnifiedState, Dict[str, Any]]:
        """
        Internal implementation of timestep.

        Args:
            state: Current state.
            dt: Timestep size.

        Returns:
            Tuple of (new_state, step_info).
        """
        # Update physics modules
        state = self._module_registry.update_all(state, dt)

        # Solve flow equations
        solver_result = self._flow_solver.solve(state, dt)

        # Update state (time will be updated by time_stepper)
        state.timestep = dt
        state.iteration += 1

        step_info = {
            "converged": solver_result.is_successful,
            "iterations": solver_result.iterations,
            "residual_norm": solver_result.residual_norm,
            "elapsed": solver_result.elapsed_time,
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
        Evaluate well performance using productivity index based on reservoir properties.

        Uses the Darcy equation to calculate productivity index:
        PI = (0.00708 * kh) / (μ * ln(re/rw) + s)

        Args:
            well_name: Well name.
            control_type: Control type ("rate" or "bhp").
            value: Control value.

        Returns:
            Dictionary with well metrics.
        """
        state = self.get_state()

        # Average pressure
        p_avg = float(np.mean(state.pressure))

        # Calculate productivity index from reservoir properties
        pi = self._calculate_productivity_index(state)

        # Calculate rates and pressures
        if control_type == "rate":
            q = value
            bhp = p_avg + q / pi if q > 0 else p_avg - abs(q) / pi
        else:  # bhp
            bhp = value
            q = pi * (p_avg - bhp)

        # Calculate phase rates based on saturations
        saturations = np.array(state.saturations) if hasattr(state, 'saturations') else np.array([[0.7, 0.2, 0.1]])
        if saturations.ndim == 2 and saturations.shape[1] >= 3:
            s_oil = np.mean(saturations[:, 0])
            s_water = np.mean(saturations[:, 1])
            s_gas = np.mean(saturations[:, 2])
        else:
            s_oil, s_water, s_gas = 0.7, 0.2, 0.1

        # Relative permeabilities (simplified)
        kro = max(0.1, s_oil ** 2)  # Oil relative permeability
        krw = max(0.01, s_water ** 3)  # Water relative permeability
        krg = max(0.01, s_gas ** 2)  # Gas relative permeability

        # Viscosities (cP)
        mu_oil = 2.0
        mu_water = 1.0
        mu_gas = 0.08

        # Phase mobilities
        lambda_o = kro / mu_oil
        lambda_w = krw / mu_water
        lambda_g = krg / mu_gas
        lambda_total = lambda_o + lambda_w + lambda_g

        # Phase splits based on mobility
        oil_cut = lambda_o / lambda_total if lambda_total > 0 else 0.6
        water_cut = lambda_w / lambda_total if lambda_total > 0 else 0.4

        return {
            "oil_rate": q * oil_cut,
            "water_rate": q * water_cut,
            "gas_rate": q * (1 - oil_cut - water_cut) * 100,  # GOR effect
            "bhp": bhp,
            "productivity_index": pi,
        }

    def _calculate_productivity_index(self, state) -> float:
        """
        Calculate productivity index from reservoir properties.

        Based on Darcy's law with configurable skin factor and drainage radius.
        """
        # Default reservoir properties (can be overridden by config)
        perm_md = 100.0  # Permeability (mD)
        thickness_ft = 50.0  # Net thickness (ft)
        well_radius_ft = getattr(self._config, 'well_radius_ft', 0.25)  # 3 inch well
        skin_factor = getattr(self._config, 'skin_factor', 0.0)
        drainage_radius_ft = getattr(self._config, 'drainage_radius_ft', 1000.0)

        # Get permeability from state if available
        if hasattr(state, 'permeability') and state.permeability is not None:
            perm_array = np.array(state.permeability)
            if perm_array.ndim == 1:
                perm_md = float(np.mean(perm_array))
            elif perm_array.ndim == 2:
                perm_md = float(np.mean(perm_array[:, 0]))

        # Get thickness from grid if available
        if hasattr(self._grid_manager, 'total_thickness'):
            thickness_ft = self._grid_manager.total_thickness

        # Productivity index calculation
        # PI = (0.00708 * kh) / (μ * (ln(re/rw) + s)) [STB/day/psi]
        # For oil: μ ≈ 2 cP
        mu_oil = 2.0  # cP

        # Wellbore radius conversion
        rw = well_radius_ft

        # Darcy units conversion factor
        kh = perm_md * thickness_ft
        denominator = mu_oil * (np.log(drainage_radius_ft / rw) + skin_factor - 0.75)

        if denominator <= 0:
            pi = 0.1  # Fallback
        else:
            pi = (0.00708 * kh) / denominator

        # Bound PI to reasonable range
        pi = max(0.01, min(100.0, pi))

        return pi

    def get_production_rates(self) -> Dict[str, np.ndarray]:
        """Get current production rates (simplified)."""
        state = self.get_state()

        # Very simplified rate calculation
        return {
            "oil": np.array([100.0]),  # Placeholder
            "gas": np.array([10000.0]),
            "water": np.array([50.0]),
        }

    def get_cumulative_production(self) -> Dict[str, float]:
        """Get cumulative production (simplified)."""
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        stats = {
            "engine_type": "Fast",
            "mode": self.config.mode.value,
            "total_solve_time": self._total_solve_time,
            "n_timesteps": self._time_stepper.statistics.n_steps if self._time_stepper else 0,
            "n_successful_steps": (
                self._time_stepper.statistics.n_successful if self._time_stepper else 0
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
                "avg_time_per_call": (
                    self._flow_solver.statistics.total_time
                    / max(1, self._flow_solver.statistics.n_calls)
                    if self._flow_solver.statistics.n_calls > 0
                    else 0
                ),
            }

        if self._multiphase_flow is not None:
            stats["physics_stats"] = self._multiphase_flow.statistics

        return stats

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import sys

        # Rough estimates
        n_cells = self._grid_manager.n_cells if self._grid_manager else 0

        usage = {
            "grid_mb": n_cells * 8 * 3 / 1e6,  # 3 double arrays
            "state_mb": n_cells * 8 * 2 / 1e6,  # pressure + saturations
            "solver_mb": n_cells * 8 * 4 / 1e6,  # Working arrays
            "total_mb": 0.0,
        }
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


def create_fast_engine(config: "EngineConfig") -> FastEngine:
    """Factory function to create Fast Engine."""
    return FastEngine(config)
