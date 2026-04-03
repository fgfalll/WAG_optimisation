"""
Unified State Management for the Unified Physics Engine.

Provides a single state representation that works for both Simple and Detailed engines,
with optional fields that are only used in detailed mode.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np
import copy
from datetime import datetime

from ..base.engine_config import EngineMode


class Phase(Enum):
    """Fluid phases."""
    OIL = "oil"
    GAS = "gas"
    WATER = "water"
    CO2 = "co2"  # Supercritical CO2 phase


class Component(Enum):
    """Fluid components for compositional modeling."""
    CO2 = "CO2"
    C1 = "C1"      # Methane
    C2 = "C2"      # Ethane
    C3 = "C3"      # Propane
    C4 = "C4"      # Butane
    C5 = "C5"      # Pentane+
    C6 = "C6"      # Hexane+
    C7P = "C7P"    # Heptane+
    N2 = "N2"      # Nitrogen
    H2S = "H2S"    # Hydrogen sulfide


@dataclass
class StateHistory:
    """
    History tracking for simulation state.

    Stores snapshots of state at specified output intervals.
    """
    times: List[float] = field(default_factory=list)
    states: List["UnifiedState"] = field(default_factory=list)
    max_history: int = 1000  # Maximum number of snapshots to store

    def add(self, time: float, state: "UnifiedState") -> None:
        """Add a state snapshot."""
        self.times.append(time)
        self.states.append(copy.deepcopy(state))

        # Limit history size
        if len(self.times) > self.max_history:
            self.times.pop(0)
            self.states.pop(0)

    def get_at_time(self, time: float) -> Optional["UnifiedState"]:
        """Get state at specific time (interpolated if needed)."""
        if not self.times:
            return None

        if time <= self.times[0]:
            return self.states[0]
        if time >= self.times[-1]:
            return self.states[-1]

        # Find surrounding indices
        i = 0
        while i < len(self.times) - 1 and self.times[i + 1] < time:
            i += 1

        # Linear interpolation
        t0, t1 = self.times[i], self.times[i + 1]
        if t1 - t0 < 1e-10:
            return self.states[i]

        alpha = (time - t0) / (t1 - t0)
        return self.states[i].interpolate(self.states[i + 1], alpha)

    def clear(self) -> None:
        """Clear history."""
        self.times.clear()
        self.states.clear()

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots stored."""
        return len(self.times)

    def get_time_series(self, field_name: str) -> np.ndarray:
        """
        Extract a field as a time series.

        Args:
            field_name: Name of field to extract (e.g., "pressure", "oil_saturation").

        Returns:
            Array of field values over time.
        """
        values = []
        for state in self.states:
            value = getattr(state, field_name, None)
            if value is not None:
                # Use cell average or total as appropriate
                if isinstance(value, np.ndarray):
                    values.append(np.mean(value))
                else:
                    values.append(value)
        return np.array(values)


@dataclass
class UnifiedState:
    """
    Unified state representation for both Simple and Detailed engines.

    Core fields are always present. Optional fields are only used in detailed mode
    or when specific physics modules are enabled.

    Attributes:
        pressure: Cell pressures (psi), shape (n_cells,)
        saturations: Cell saturations, shape (n_cells, n_phases)
        current_time: Current simulation time (days)
        timestep: Current timestep size (days)

        # Optional fields (detailed mode)
        compositions: Component compositions, shape (n_cells, n_components)
        porosity: Cell porosities, shape (n_cells,)
        permeability: Cell permeabilities, shape (n_cells, 3) for kx, ky, kz
        temperature: Cell temperatures (°F), shape (n_cells,)
        stress: Effective stress tensor, shape (n_cells, 6) for xx, yy, zz, xy, yz, xz
        dissolved_co2: Dissolved CO2 concentration, shape (n_cells,)
        mineralized_co2: Mineralized CO2, shape (n_cells,)

        # Metadata
        mode: Engine mode (SIMPLE or DETAILED)
        iteration: Current Newton iteration count
        convergence_info: Dict with convergence metrics
    """

    # Core fields (always required)
    pressure: np.ndarray
    saturations: np.ndarray
    current_time: float = 0.0
    timestep: float = 1.0

    # Optional: Compositional tracking (detailed mode)
    compositions: Optional[np.ndarray] = None  # (n_cells, n_components)

    # Optional: Rock properties
    porosity: Optional[np.ndarray] = None  # (n_cells,)
    permeability: Optional[np.ndarray] = None  # (n_cells, 3) or (n_cells,)

    # Optional: Thermal
    temperature: Optional[np.ndarray] = None  # (n_cells,)

    # Optional: Geomechanics
    stress: Optional[np.ndarray] = None  # (n_cells, 6)

    # Optional: CO2 trapping
    dissolved_co2: Optional[np.ndarray] = None  # (n_cells,)
    mineralized_co2: Optional[np.ndarray] = None  # (n_cells,)

    # Metadata
    mode: EngineMode = EngineMode.SIMPLE
    iteration: int = 0
    convergence_info: Dict[str, Any] = field(default_factory=dict)

    # Well states
    well_states: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self):
        """Validate state dimensions."""
        n_cells = len(self.pressure)
        n_phases = self.saturations.shape[1] if self.saturations.ndim > 1 else 1

        # Validate saturations sum to 1
        if self.saturations.ndim == 2:
            sums = np.sum(self.saturations, axis=1)
            if not np.allclose(sums, 1.0, atol=0.01):
                # Normalize
                self.saturations = self.saturations / sums[:, np.newaxis]

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.pressure)

    @property
    def n_phases(self) -> int:
        """Number of phases."""
        return self.saturations.shape[1] if self.saturations.ndim > 1 else 1

    @property
    def n_components(self) -> Optional[int]:
        """Number of components (if compositional)."""
        if self.compositions is not None:
            return self.compositions.shape[1]
        return None

    @property
    def oil_saturation(self) -> np.ndarray:
        """Get oil saturation."""
        return self._get_phase_saturation(Phase.OIL)

    @property
    def gas_saturation(self) -> np.ndarray:
        """Get gas saturation."""
        return self._get_phase_saturation(Phase.GAS)

    @property
    def water_saturation(self) -> np.ndarray:
        """Get water saturation."""
        return self._get_phase_saturation(Phase.WATER)

    @property
    def co2_saturation(self) -> np.ndarray:
        """Get CO2 saturation (if tracked separately)."""
        return self._get_phase_saturation(Phase.CO2)

    def _get_phase_saturation(self, phase: Phase) -> np.ndarray:
        """Get saturation for a specific phase."""
        if self.saturations.ndim == 1:
            # Single phase - assume it's the requested phase
            return self.saturations

        # Map phase to index: oil=0, gas=1, water=2
        phase_map = {Phase.OIL: 0, Phase.GAS: 1, Phase.WATER: 2, Phase.CO2: 1}
        idx = phase_map.get(phase, 0)
        if idx < self.saturations.shape[1]:
            return self.saturations[:, idx]
        return np.zeros_like(self.pressure)

    @property
    def average_pressure(self) -> float:
        """Get average reservoir pressure."""
        return float(np.mean(self.pressure))

    @property
    def max_pressure(self) -> float:
        """Get maximum pressure."""
        return float(np.max(self.pressure))

    @property
    def min_pressure(self) -> float:
        """Get minimum pressure."""
        return float(np.min(self.pressure))

    def get_cell_pressure(self, i: int, j: int, k: int, nx: int, ny: int) -> float:
        """
        Get pressure at grid cell (i, j, k).

        Args:
            i, j, k: Grid indices.
            nx, ny: Grid dimensions.

        Returns:
            Pressure at cell (psi).
        """
        idx = k * nx * ny + j * nx + i
        return float(self.pressure[idx])

    def get_cell_saturation(
        self, i: int, j: int, k: int, nx: int, ny: int, phase: int = 0
    ) -> float:
        """
        Get saturation at grid cell (i, j, k) for a phase.

        Args:
            i, j, k: Grid indices.
            nx, ny: Grid dimensions.
            phase: Phase index (0=oil, 1=gas, 2=water).

        Returns:
            Saturation at cell (fraction).
        """
        idx = k * nx * ny + j * nx + i
        if self.saturations.ndim == 1:
            return float(self.saturations[idx])
        return float(self.saturations[idx, phase])

    def interpolate(self, other: "UnifiedState", alpha: float) -> "UnifiedState":
        """
        Interpolate between this state and another.

        Args:
            other: Other state to interpolate with.
            alpha: Interpolation parameter (0=this, 1=other).

        Returns:
            Interpolated state.
        """
        result = copy.deepcopy(self)

        # Interpolate pressure
        result.pressure = (1 - alpha) * self.pressure + alpha * other.pressure

        # Interpolate saturations
        result.saturations = (1 - alpha) * self.saturations + alpha * other.saturations

        # Interpolate time
        result.current_time = (1 - alpha) * self.current_time + alpha * other.current_time

        # Interpolate optional fields if present in both
        if self.compositions is not None and other.compositions is not None:
            result.compositions = (
                (1 - alpha) * self.compositions + alpha * other.compositions
            )

        if self.temperature is not None and other.temperature is not None:
            result.temperature = (
                (1 - alpha) * self.temperature + alpha * other.temperature
            )

        return result

    def copy(self) -> "UnifiedState":
        """Create a deep copy of the state."""
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "pressure": self.pressure.tolist(),
            "saturations": self.saturations.tolist(),
            "current_time": self.current_time,
            "timestep": self.timestep,
            "compositions": self.compositions.tolist() if self.compositions is not None else None,
            "porosity": self.porosity.tolist() if self.porosity is not None else None,
            "permeability": self.permeability.tolist() if self.permeability is not None else None,
            "temperature": self.temperature.tolist() if self.temperature is not None else None,
            "stress": self.stress.tolist() if self.stress is not None else None,
            "dissolved_co2": self.dissolved_co2.tolist() if self.dissolved_co2 is not None else None,
            "mineralized_co2": self.mineralized_co2.tolist() if self.mineralized_co2 is not None else None,
            "mode": self.mode.value,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedState":
        """Create state from dictionary."""
        return cls(
            pressure=np.array(data["pressure"]),
            saturations=np.array(data["saturations"]),
            current_time=data.get("current_time", 0.0),
            timestep=data.get("timestep", 1.0),
            compositions=np.array(data["compositions"]) if data.get("compositions") else None,
            porosity=np.array(data["porosity"]) if data.get("porosity") else None,
            permeability=np.array(data["permeability"]) if data.get("permeability") else None,
            temperature=np.array(data["temperature"]) if data.get("temperature") else None,
            stress=np.array(data["stress"]) if data.get("stress") else None,
            dissolved_co2=np.array(data["dissolved_co2"]) if data.get("dissolved_co2") else None,
            mineralized_co2=np.array(data["mineralized_co2"]) if data.get("mineralized_co2") else None,
            mode=EngineMode(data.get("mode", "simple")),
            iteration=data.get("iteration", 0),
        )

    def validate(self) -> List[str]:
        """
        Validate state and return list of errors.

        Returns:
            Empty list if valid, otherwise list of error messages.
        """
        errors = []

        # Check pressure
        if np.any(self.pressure < 0):
            errors.append("Negative pressure values detected")

        if np.any(np.isnan(self.pressure)):
            errors.append("NaN pressure values detected")

        if np.any(np.isinf(self.pressure)):
            errors.append("Inf pressure values detected")

        # Check saturations
        if self.saturations.ndim == 2:
            sums = np.sum(self.saturations, axis=1)
            if not np.allclose(sums, 1.0, atol=0.05):
                errors.append("Saturation sums deviate from 1.0")

            if np.any(self.saturations < 0):
                errors.append("Negative saturation values detected")

            if np.any(self.saturations > 1):
                errors.append("Saturation values greater than 1.0 detected")

        # Check compositions
        if self.compositions is not None:
            sums = np.sum(self.compositions, axis=1)
            if not np.allclose(sums, 1.0, atol=0.05):
                errors.append("Composition sums deviate from 1.0")

            if np.any(self.compositions < 0):
                errors.append("Negative composition values detected")

        # Check porosity
        if self.porosity is not None:
            if np.any(self.porosity < 0):
                errors.append("Negative porosity values detected")
            if np.any(self.porosity > 1):
                errors.append("Porosity values greater than 1.0 detected")

        # Check permeability
        if self.permeability is not None:
            if np.any(self.permeability < 0):
                errors.append("Negative permeability values detected")

        return errors

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the state."""
        return {
            "current_time": self.current_time,
            "timestep": self.timestep,
            "n_cells": self.n_cells,
            "n_phases": self.n_phases,
            "pressure_avg": float(np.mean(self.pressure)),
            "pressure_min": float(np.min(self.pressure)),
            "pressure_max": float(np.max(self.pressure)),
            "oil_saturation_avg": float(np.mean(self.oil_saturation)),
            "gas_saturation_avg": float(np.mean(self.gas_saturation)),
            "water_saturation_avg": float(np.mean(self.water_saturation)),
            "mode": self.mode.value,
            "iteration": self.iteration,
        }


class StateManager:
    """
    Manages simulation state including initialization, updates, and history.

    Provides methods for creating initial states, updating states, and tracking history.
    """

    def __init__(self, config, n_cells: int, n_phases: int = 3):
        """
        Initialize state manager.

        Args:
            config: Engine configuration.
            n_cells: Number of grid cells.
            n_phases: Number of fluid phases.
        """
        self.config = config
        self.n_cells = n_cells
        self.n_phases = n_phases
        self.history: StateHistory = StateHistory()
        self._initial_state: Optional[UnifiedState] = None

    def create_initial_state(
        self,
        initial_pressure: float = 2000.0,
        initial_saturations: Optional[np.ndarray] = None,
        porosity: Optional[np.ndarray] = None,
        permeability: Optional[np.ndarray] = None,
    ) -> UnifiedState:
        """
        Create initial simulation state.

        Args:
            initial_pressure: Initial reservoir pressure (psi).
            initial_saturations: Initial saturations (n_cells, n_phases).
            porosity: Cell porosities (n_cells,).
            permeability: Cell permeabilities (n_cells, 3) or (n_cells,).

        Returns:
            Initial UnifiedState.
        """
        # Initialize pressure
        pressure = np.full(self.n_cells, initial_pressure, dtype=np.float64)

        # Initialize saturations
        if initial_saturations is None:
            # Default: connate water, remaining oil
            saturations = np.zeros((self.n_cells, self.n_phases), dtype=np.float64)
            saturations[:, 2] = 0.2  # Water saturation (Swc)
            saturations[:, 0] = 0.8  # Oil saturation
        else:
            saturations = initial_saturations.copy()

        state = UnifiedState(
            pressure=pressure,
            saturations=saturations,
            current_time=0.0,
            timestep=self.config.timestep.initial,
            mode=self.config.mode,
        )

        # Add optional fields based on mode
        if self.config.is_detailed():
            if porosity is not None:
                state.porosity = porosity.copy()

            if permeability is not None:
                state.permeability = permeability.copy()

            # Initialize temperature
            state.temperature = np.full(self.n_cells, 150.0, dtype=np.float64)

        self._initial_state = state.copy()
        return state

    def reset(self) -> UnifiedState:
        """Reset to initial state."""
        if self._initial_state is None:
            raise RuntimeError("No initial state stored. Call create_initial_state first.")
        self.history.clear()
        return self._initial_state.copy()

    def record_state(self, state: UnifiedState) -> None:
        """Record a state snapshot in history."""
        self.history.add(state.current_time, state)

    def get_state_at_time(self, time: float) -> Optional[UnifiedState]:
        """Get state at a specific time from history."""
        return self.history.get_at_time(time)

    def get_time_series(self, field_name: str) -> np.ndarray:
        """Get a field as a time series."""
        return self.history.get_time_series(field_name)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.history.clear()
        self._initial_state = None
