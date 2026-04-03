"""
Base class for physics modules in the Unified Physics Engine.

All physics components (multiphase flow, geomechanics, etc.) inherit from
this base to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np

from .engine_config import EngineConfig, EngineMode
from ..core.state_manager import UnifiedState


class PhysicsModule(ABC):
    """
    Abstract base class for physics modules.

    All physics modules (multiphase flow, geomechanics, EOS, etc.) must
    implement this interface to be compatible with the unified engine.
    """

    def __init__(self, config: EngineConfig, name: str):
        """
        Initialize the physics module.

        Args:
            config: Engine configuration.
            name: Module name for logging and diagnostics.
        """
        self.config = config
        self.name = name
        self.enabled = True
        self._initialized = False

        # Module-specific statistics
        self._call_count = 0
        self._total_time = 0.0

    @property
    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "initialized": self._initialized,
            "call_count": self._call_count,
            "total_time": self._total_time,
        }

    @abstractmethod
    def initialize(self, state: UnifiedState) -> None:
        """
        Initialize the module with the simulation state.

        Args:
            state: Initial simulation state.
        """
        self._initialized = True

    @abstractmethod
    def update(self, state: UnifiedState, dt: float) -> UnifiedState:
        """
        Update the module for a timestep.

        Args:
            state: Current simulation state.
            dt: Timestep size (days).

        Returns:
            Updated simulation state.
        """
        self._call_count += 1
        return state

    def reset(self) -> None:
        """Reset module state (call count, timing, etc.)."""
        self._call_count = 0
        self._total_time = 0.0

    def cleanup(self) -> None:
        """Clean up module resources."""
        self._initialized = False

    def validate_state(self, state: UnifiedState) -> List[str]:
        """
        Validate that state has required fields for this module.

        Args:
            state: State to validate.

        Returns:
            List of error messages (empty if valid).
        """
        return []

    def get_required_state_fields(self) -> List[str]:
        """
        Get list of required state field names.

        Returns:
            List of field names that must be present in state.
        """
        return []

    def get_optional_state_fields(self) -> List[str]:
        """
        Get list of optional state field names.

        Returns:
            List of field names that may be present in state.
        """
        return []


class CompositionalModule(PhysicsModule):
    """
    Base class for compositional physics modules.

    Handles component tracking and phase behavior calculations.
    """

    def __init__(self, config: EngineConfig, name: str = "compositional"):
        super().__init__(config, name)
        self.n_components: Optional[int] = None
        self.n_phases: int = 3  # Oil, gas, water
        self.component_names: List[str] = []

    @abstractmethod
    def flash_calculation(
        self, pressure: np.ndarray, temperature: np.ndarray, composition: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform flash calculation to determine phase splits.

        Args:
            pressure: Pressure array (psi).
            temperature: Temperature array (°F).
            composition: Overall composition (n_cells, n_components).

        Returns:
            Tuple of (phase_fractions, phase_compositions, k_values)
                - phase_fractions: (n_cells, n_phases)
                - phase_compositions: (n_cells, n_phases, n_components)
                - k_values: (n_cells, n_components)
        """
        pass

    @abstractmethod
    def get_components(self) -> List[str]:
        """Get list of component names."""
        return self.component_names


class GeomechanicsModule(PhysicsModule):
    """
    Base class for geomechanics modules.

    Handles stress, strain, and porosity/permeability coupling.
    """

    def __init__(self, config: EngineConfig, name: str = "geomechanics"):
        super().__init__(config, name)
        self.youngs_modulus: Optional[float] = None
        self.poisson_ratio: Optional[float] = None

    @abstractmethod
    def calculate_stress(
        self, pressure: np.ndarray, state: UnifiedState
    ) -> Dict[str, np.ndarray]:
        """
        Calculate stress field from pressure and state.

        Args:
            pressure: Pressure field (psi).
            state: Current simulation state.

        Returns:
            Dictionary with stress components.
        """
        pass

    @abstractmethod
    def update_porosity_perm(
        self, porosity: np.ndarray, permeability: np.ndarray, stress: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update porosity and permeability based on stress.

        Args:
            porosity: Current porosity.
            permeability: Current permeability.
            stress: Stress field.

        Returns:
            Tuple of (updated_porosity, updated_permeability).
        """
        pass


class FaultMechanicsModule(PhysicsModule):
    """
    Base class for fault mechanics modules.

    Handles fault slip, reactivation, and transmissibility changes.
    """

    def __init__(self, config: EngineConfig, name: str = "fault_mechanics"):
        super().__init__(config, name)
        self.fault_properties: Dict[str, Any] = {}

    @abstractmethod
    def check_reactivation(
        self, stress: np.ndarray, pore_pressure: np.ndarray
    ) -> np.ndarray:
        """
        Check if faults are reactivated based on stress and pressure.

        Args:
            stress: Stress tensor field.
            pore_pressure: Pore pressure field.

        Returns:
            Boolean array indicating fault reactivation status.
        """
        pass

    @abstractmethod
    def update_transmissibility(
        self, transmissibility: np.ndarray, slip: np.ndarray
    ) -> np.ndarray:
        """
        Update transmissibility based on fault slip.

        Args:
            transmissibility: Current transmissibility.
            slip: Fault slip array.

        Returns:
            Updated transmissibility.
        """
        pass


class MineralizationModule(PhysicsModule):
    """
    Base class for CO2 mineralization modules.

    Handles geochemical reactions and trapping.
    """

    def __init__(self, config: EngineConfig, name: str = "mineralization"):
        super().__init__(config, name)
        self.reaction_rates: Dict[str, float] = {}

    @abstractmethod
    def calculate_trapping(
        self,
        dissolved_co2: np.ndarray,
        mineral_composition: Dict[str, np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate mineral trapping of CO2.

        Args:
            dissolved_co2: Dissolved CO2 concentration.
            mineral_composition: Dictionary of mineral volume fractions.
            dt: Timestep (days).

        Returns:
            Tuple of (trapped_co2, updated_mineral_composition).
        """
        pass

    @abstractmethod
    def get_reaction_rate(self, mineral_name: str, temperature: float) -> float:
        """
        Get reaction rate for a specific mineral.

        Args:
            mineral_name: Name of the mineral.
            temperature: Temperature (°F).

        Returns:
            Reaction rate constant.
        """
        pass


class ModuleRegistry:
    """
    Registry for managing physics modules.

    Handles module registration, initialization, and execution order.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._modules: Dict[str, PhysicsModule] = {}
        self._execution_order: List[str] = []

    def register(self, module: PhysicsModule, position: Optional[int] = None) -> None:
        """
        Register a physics module.

        Args:
            module: Physics module to register.
            position: Optional position in execution order (appended if None).
        """
        self._modules[module.name] = module

        if position is None:
            self._execution_order.append(module.name)
        else:
            self._execution_order.insert(position, module.name)

    def unregister(self, module_name: str) -> None:
        """
        Unregister a physics module.

        Args:
            module_name: Name of module to unregister.
        """
        if module_name in self._modules:
            del self._modules[module_name]
            self._execution_order.remove(module_name)

    def get(self, module_name: str) -> Optional[PhysicsModule]:
        """Get a registered module by name."""
        return self._modules.get(module_name)

    def initialize_all(self, state: UnifiedState) -> None:
        """Initialize all registered modules."""
        for name in self._execution_order:
            module = self._modules[name]
            if module.enabled:
                module.initialize(state)

    def update_all(self, state: UnifiedState, dt: float) -> UnifiedState:
        """Update all registered modules in execution order."""
        for name in self._execution_order:
            module = self._modules[name]
            if module.enabled:
                state = module.update(state, dt)
        return state

    def reset_all(self) -> None:
        """Reset all modules."""
        for module in self._modules.values():
            module.reset()

    def cleanup_all(self) -> None:
        """Clean up all modules."""
        for module in self._modules.values():
            module.cleanup()

    def get_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all modules."""
        return [self._modules[name].statistics for name in self._execution_order]

    @property
    def enabled_modules(self) -> List[str]:
        """Get list of enabled module names."""
        return [name for name in self._execution_order if self._modules[name].enabled]
