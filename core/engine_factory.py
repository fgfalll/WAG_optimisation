"""
Engine Factory for CO2-EOR Optimization
========================================

This module provides a factory interface to create and switch between
different simulation engines (simple and detailed) for optimization.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Import engine types
try:
    from .engine_simple.optimization_interface import OptimizationInterface as SimpleOptimizationInterface
    from .engine_simple.reservoir_engine import ReservoirSimulationEngine as SimpleEngine
    SIMPLE_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Simple engine not available: {e}")
    SIMPLE_ENGINE_AVAILABLE = False

# Try to import detailed engine
try:
    from .Phys_engine_full.enhanced_reservoir_simulator import EnhancedReservoirSimulator
    DETAILED_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Detailed engine not available: {e}")
    DETAILED_ENGINE_AVAILABLE = False

# Try to import surrogate engine
try:
    from .engine_surrogate.surrogate_engine import SurrogateEngineWrapper
    SURROGATE_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Surrogate engine not available: {e}")
    SURROGATE_ENGINE_AVAILABLE = False

# Try to import compositional engine
try:
    from .compositional_engine.engine_wrapper import CompositionalEngineWrapper
    COMPOSITIONAL_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Compositional engine not available: {e}")
    COMPOSITIONAL_ENGINE_AVAILABLE = False

from core.data_models import ReservoirData, EORParameters, OperationalParameters


class EngineType(Enum):
    """Available simulation engine types"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    SURROGATE = "surrogate"
    COMPOSITIONAL = "compositional"


class SimulationEngineInterface(ABC):
    """Abstract interface for all simulation engines"""

    @abstractmethod
    def evaluate_scenario(self, reservoir_data: ReservoirData,
                          eor_params: EORParameters,
                          operational_params: OperationalParameters,
                          economic_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a scenario and return comprehensive results"""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine"""
        pass

    @abstractmethod
    def validate_parameters(self, reservoir_data: ReservoirData,
                           eor_params: EORParameters) -> Dict[str, bool]:
        """Validate input parameters"""
        pass


class SimpleEngineWrapper(SimulationEngineInterface):
    """Wrapper for the simple simulation engine"""

    def __init__(self):
        if not SIMPLE_ENGINE_AVAILABLE:
            raise ImportError("Simple engine is not available")

        # Initialize simple engine
        from .engine_simple.utils import RockProperties, FluidProperties
        from .engine_simple.reservoir_engine import ReservoirSimulationEngine
        from .data_models import SimpleGrid

        # Create default simple engine configuration
        grid_params = SimpleGrid(
            nx=50, ny=50, nz=1,
            dx=20.0, dy=20.0, dz=10.0
        )
        
        # Create arrays for rock properties to match grid size
        n_cells = grid_params.nx * grid_params.ny * grid_params.nz
        poro_arr = np.full(n_cells, 0.15)
        perm_x_arr = np.full(n_cells, 100.0)
        perm_y_arr = np.full(n_cells, 100.0)
        perm_z_arr = np.full(n_cells, 10.0)
        
        self.engine = ReservoirSimulationEngine(
            grid=grid_params,
            rock=RockProperties(
                porosity=poro_arr,
                permeability_x=perm_x_arr,
                permeability_y=perm_y_arr,
                permeability_z=perm_z_arr,
                compressibility=1e-5
            ),
            fluid=FluidProperties(
                oil_viscosity_ref=1.5e-3, # 1.5 cP -> Pa·s
                water_viscosity_ref=0.5e-3, # 0.5 cP -> Pa·s
                gas_viscosity_ref=0.05e-3 # 0.05 cP -> Pa·s
            )
        )

    def evaluate_scenario(self, reservoir_data: ReservoirData,
                          eor_params: EORParameters,
                          operational_params: OperationalParameters,
                          economic_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate a scenario using the simple engine"""
        # Map main data models to simple engine parameters
        from .engine_simple.optimization_interface import OptimizationInterface
        interface = OptimizationInterface(self.engine)
        
        # Use evaluate_eor_scenario which exists in OptimizationInterface
        results = interface.evaluate_eor_scenario(
            eor_params=eor_params, 
            economic_params=economic_params
        )
        
        # Convert OptimizationResults dataclass to the dictionary format expected by OptimizationEngine
        return {
            'oil_production_rate': results.oil_production_profile,
            'water_production_rate': results.water_production_profile,
            'gas_production_rate': results.gas_production_profile,
            'oil_cumulative': results.cumulative_oil,
            'time_vector': results.time_vector,
            'co2_storage_volume': results.co2_storage_profile,
            'co2_injection': results.co2_injection_rate,
            'co2_utilization': results.co2_utilization_factor,
            'recovery_factor': results.recovery_factor,
            'sweep_efficiency': results.sweep_efficiency,
            'storage_efficiency': results.storage_efficiency,
            'pressure': results.pressure,
            'npv': results.npv,
            'objective_value': results.objective_value,
            'convergence_status': results.convergence_status,
            'simulation_time': results.simulation_time
        }

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "simple",
            "description": "Basic grid-based simulation with simplified physics",
            "available": SIMPLE_ENGINE_AVAILABLE
        }

    def validate_parameters(self, reservoir_data: ReservoirData,
                           eor_params: EORParameters) -> Dict[str, bool]:
        # Simple validation for basic engine
        return {"valid": True}


class DetailedEngineWrapper(SimulationEngineInterface):
    """Wrapper for the detailed (full physics) engine"""

    def __init__(self, **kwargs):
        if not DETAILED_ENGINE_AVAILABLE:
            raise ImportError("Detailed engine is not available")

        # Initialize detailed engine later when we have parameters
        self.engine = None
        self.advanced_params = kwargs.get('advanced_params', None)

    def evaluate_scenario(self, reservoir_data: ReservoirData,
                          eor_params: EORParameters,
                          operational_params: OperationalParameters,
                          economic_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate scenario using detailed engine"""
        try:
            # Lazy initialization or update with specific parameters
            from .Phys_engine_full.detailed_engine_adapters import (
                create_detailed_engine_params,
                convert_results_to_main_format
            )

            # Create detailed engine parameters using adapter functions
            detailed_params = create_detailed_engine_params(
                reservoir_data, eor_params, operational_params, self.advanced_params
            )

            # Initialize the detailed engine with proper parameters
            from .Phys_engine_full.enhanced_reservoir_simulator import EnhancedReservoirSimulator
            self.engine = EnhancedReservoirSimulator(
                grid=detailed_params.grid,
                fluid=detailed_params.fluid,
                rock=detailed_params.rock,
                bc=detailed_params.bc,
                params=detailed_params.sim_params
            )

            # Run detailed simulation
            simulation_results = self.engine.run_simulation()

            # Convert to unified output format
            return convert_results_to_main_format(simulation_results, economic_params)

        except Exception as e:
            logger.error(f"Detailed engine evaluation failed: {e}")
            # Fallback or error reporting
            raise

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "detailed",
            "description": "Full physics-based CCUS-EOR simulator with implicit solving",
            "available": DETAILED_ENGINE_AVAILABLE
        }

    def validate_parameters(self, reservoir_data: ReservoirData,
                           eor_params: EORParameters) -> Dict[str, bool]:
        """Validate parameters for detailed engine"""
        from .Phys_engine_full.detailed_engine_adapters import validate_for_detailed_engine
        # Enhanced checks for detailed engine using main data model attributes
        return validate_for_detailed_engine(reservoir_data, eor_params)


class SurrogateEngineWrapper(SimulationEngineInterface):
    """Wrapper for the analytical surrogate engine (PhD Project Primary Engine)"""

    def __init__(self, model_type="analytical", recovery_model_type="hybrid"):
        if not SURROGATE_ENGINE_AVAILABLE:
            raise ImportError("Surrogate engine is not available")
            
        from .engine_surrogate.surrogate_engine import SurrogateEngineWrapper as InnerWrapper
        self.engine = InnerWrapper(
            model_type=model_type,
            recovery_model_type=recovery_model_type
        )

    def evaluate_scenario(self, reservoir_data: ReservoirData,
                          eor_params: EORParameters,
                          operational_params: OperationalParameters,
                          economic_params: Optional[Dict] = None,
                          **kwargs) -> Dict[str, Any]:
        """Evaluate scenario using surrogate engine"""
        # Surrogate engine uses its own data mapping
        results = self.engine.evaluate_scenario(
            reservoir_data=reservoir_data,
            eor_params=eor_params,
            operational_params=operational_params,
            economic_params=economic_params,
            **kwargs
        )
        
        # Ensure engine info is attached
        results['engine_type'] = 'surrogate'
        return results

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "surrogate",
            "description": "Fast analytical surrogate based on PhD verified physics",
            "available": SURROGATE_ENGINE_AVAILABLE
        }

    def validate_parameters(self, reservoir_data: ReservoirData,
                           eor_params: EORParameters) -> Dict[str, bool]:
        # Surrogate is very robust, minimal validation needed
        return {"valid": True}


class CompositionalEngineWrapper(SimulationEngineInterface):
    """Wrapper for the compositional EOS-based engine"""

    def __init__(self):
        if not COMPOSITIONAL_ENGINE_AVAILABLE:
            raise ImportError("Compositional engine is not available")
            
        from .compositional_engine.engine_wrapper import CompositionalEngineWrapper as InnerWrapper
        self.engine = InnerWrapper()

    def evaluate_scenario(self, reservoir_data: ReservoirData,
                          eor_params: EORParameters,
                          operational_params: OperationalParameters,
                          economic_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate scenario using compositional engine"""
        return self.engine.evaluate_scenario(
            reservoir_data=reservoir_data,
            eor_params=eor_params,
            operational_params=operational_params,
            economic_params=economic_params
        )

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "compositional",
            "description": "Multi-component phase equilibrium based simulator",
            "available": COMPOSITIONAL_ENGINE_AVAILABLE
        }

    def validate_parameters(self, reservoir_data: ReservoirData,
                           eor_params: EORParameters) -> Dict[str, bool]:
        # Check if fluid composition is defined
        has_composition = hasattr(reservoir_data, 'fluid_composition') and reservoir_data.fluid_composition is not None
        return {"valid": has_composition, "reason": "" if has_composition else "Fluid composition required"}


class EngineFactory:
    """Factory for creating simulation engines"""

    @staticmethod
    def create_engine(engine_type: Union[str, EngineType], **kwargs) -> SimulationEngineInterface:
        """
        Create a simulation engine instance of the specified type.
        
        Args:
            engine_type: Type of engine to create ('simple', 'detailed', 'surrogate', 'compositional')
            **kwargs: Additional parameters for engine initialization
            
        Returns:
            SimulationEngineInterface: The created engine instance
        """
        # Convert string to EngineType enum if necessary
        if isinstance(engine_type, str):
            try:
                engine_type = EngineType(engine_type.lower())
            except ValueError:
                raise ValueError(f"Unknown engine type: {engine_type}")

        # Always return surrogate for PhD consistency unless explicitly overridden
        # if engine_type != EngineType.DETAILED:
        #    return SurrogateEngineWrapper(
        #        model_type=kwargs.get('model_type', 'phd_hybrid'),
        #        recovery_model_type=kwargs.get('recovery_model_type', 'hybrid')
        #    )

        if engine_type == EngineType.SIMPLE:
            if not SIMPLE_ENGINE_AVAILABLE:
                logger.warning("Simple engine requested but not available. Falling back to surrogate.")
                return SurrogateEngineWrapper()
            return SimpleEngineWrapper()

        elif engine_type == EngineType.DETAILED:
            if not DETAILED_ENGINE_AVAILABLE:
                logger.warning("Detailed engine requested but not available. Falling back to surrogate.")
                return SurrogateEngineWrapper()
            return DetailedEngineWrapper(**kwargs)

        elif engine_type == EngineType.SURROGATE:
            if not SURROGATE_ENGINE_AVAILABLE:
                raise ImportError("Surrogate engine is required but not available")
            return SurrogateEngineWrapper(
                model_type=kwargs.get('model_type', 'analytical'),
                recovery_model_type=kwargs.get('recovery_model_type', 'phd_hybrid')
            )

        elif engine_type == EngineType.COMPOSITIONAL:
            if not COMPOSITIONAL_ENGINE_AVAILABLE:
                logger.warning("Compositional engine requested but not available. Falling back to surrogate.")
                return SurrogateEngineWrapper()
            return CompositionalEngineWrapper()

        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")

    @staticmethod
    def get_available_engines() -> Dict[str, bool]:
        """Get a dictionary of available engine types and their status"""
        return {
            "simple": SIMPLE_ENGINE_AVAILABLE,
            "detailed": DETAILED_ENGINE_AVAILABLE,
            "surrogate": SURROGATE_ENGINE_AVAILABLE,
            "compositional": COMPOSITIONAL_ENGINE_AVAILABLE
        }

    @staticmethod
    def switch_engine(current_engine: SimulationEngineInterface, 
                      target_type: Union[str, EngineType]) -> SimulationEngineInterface:
        """Switch between engine types while maintaining state if possible"""
        # For now, just create a new one
        return EngineFactory.create_engine(target_type)

    @staticmethod
    def test_engine_availability(engine_type: Union[str, EngineType]) -> bool:
        """Test if a specific engine type is available for use"""
        try:
            EngineFactory.create_engine(engine_type)
            return True
        except (ValueError, ImportError):
            return False


def get_default_engine() -> str:
    """Always returns surrogate for PhD consistency."""
    return 'surrogate'
