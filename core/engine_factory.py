"""
Engine Factory for CO2-EOR Optimization
========================================

This module provides a factory interface. For this specific project,
we strictly use the analytical surrogate engine (PhD Project Primary Engine)
for all optimization evaluations.
"""

import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from core.data_models import ReservoirData, EORParameters, OperationalParameters


class EngineType(Enum):
    """Available simulation engine types - SURROGATE ONLY for PhD consistency"""

    SURROGATE = "surrogate"


class SimulationEngineInterface(ABC):
    """Abstract interface for all simulation engines"""

    @abstractmethod
    def evaluate_scenario(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
        operational_params: OperationalParameters,
        economic_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Evaluate a scenario and return comprehensive results"""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine"""
        pass

    @abstractmethod
    def validate_parameters(
        self, reservoir_data: ReservoirData, eor_params: EORParameters
    ) -> Dict[str, bool]:
        """Validate input parameters"""
        pass


class SurrogateEngineWrapper(SimulationEngineInterface):
    """Wrapper for the analytical surrogate engine (PhD Project Primary Engine)"""

    def __init__(self, model_type="analytical", recovery_model_type="hybrid"):
        try:
            from .engine_surrogate.surrogate_engine import SurrogateEngineWrapper as InnerWrapper

            self.engine = InnerWrapper(
                model_type=model_type, recovery_model_type=recovery_model_type
            )
        except ImportError as e:
            raise ImportError(f"Surrogate engine is not available: {e}")

    def evaluate_scenario(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
        operational_params: OperationalParameters,
        economic_params: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate scenario using surrogate engine"""
        results = self.engine.evaluate_scenario(
            reservoir_data=reservoir_data,
            eor_params=eor_params,
            operational_params=operational_params,
            economic_params=economic_params,
            **kwargs,
        )
        results["engine_type"] = "surrogate"
        return results

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "surrogate",
            "description": "Fast analytical surrogate based on PhD verified physics",
            "available": True,
        }

    def validate_parameters(
        self, reservoir_data: ReservoirData, eor_params: EORParameters
    ) -> Dict[str, bool]:
        return {"valid": True}


class EngineFactory:
    """Factory for creating simulation engines"""

    @staticmethod
    def create_engine(engine_type: Union[str, EngineType], **kwargs) -> SimulationEngineInterface:
        """
        Create a simulation engine instance.
        Always returns the Surrogate engine for PhD consistency.
        """
        logger.info("EngineFactory: Routing all simulation requests to Surrogate Engine.")

        # Always return surrogate for PhD consistency, ignoring the requested type
        return SurrogateEngineWrapper(
            model_type=kwargs.get("model_type", "analytical"),
            recovery_model_type=kwargs.get("recovery_model_type", "hybrid"),
        )

    @staticmethod
    def get_available_engines() -> Dict[str, bool]:
        """Get a dictionary of available engine types and their status"""
        return {"surrogate": True}

    @staticmethod
    def switch_engine(
        current_engine: SimulationEngineInterface, target_type: Union[str, EngineType]
    ) -> SimulationEngineInterface:
        """Switch engine type - only surrogate is available"""
        logger.warning("Engine switching is deprecated. Only surrogate engine is available.")
        return EngineFactory.create_engine("surrogate")

    @staticmethod
    def test_engine_availability(engine_type: Union[str, EngineType]) -> bool:
        """Test if a specific engine type is available for use"""
        return engine_type == EngineType.SURROGATE


def get_default_engine() -> str:
    """Always returns surrogate for PhD consistency."""
    return "surrogate"
