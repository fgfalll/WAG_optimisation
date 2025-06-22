import logging

# Expose key data models directly
from .data_models import (
    WellData,
    ReservoirData,
    PVTProperties,
    EORParameters,
    GeneticAlgorithmParams,
    from_dict_to_dataclass
)

# Expose key recovery model components
from .recovery_models import (
    RecoveryModel, # ABC
    KovalRecoveryModel,
    SimpleRecoveryModel,
    MiscibleRecoveryModel,
    ImmiscibleRecoveryModel,
    HybridRecoveryModel,
    TransitionEngine,
    recovery_factor # The main function
)

# Expose the main engine
from .optimization_engine import OptimizationEngine

logging.info("Core package initialized. Exposed: Data models, Recovery models, OptimizationEngine.")

__all__ = [
    # Data Models
    "WellData", "ReservoirData", "PVTProperties",
    "EORParameters", "GeneticAlgorithmParams", "from_dict_to_dataclass",
    # Recovery Models & Engine
    "RecoveryModel", "KovalRecoveryModel", "SimpleRecoveryModel",
    "MiscibleRecoveryModel", "ImmiscibleRecoveryModel", "HybridRecoveryModel",
    "TransitionEngine", "recovery_factor",
    # Optimization Engine
    "OptimizationEngine"
]