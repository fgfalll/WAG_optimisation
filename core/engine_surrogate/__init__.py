"""
Fast Surrogate Engine for CO2 EOR Optimization
==============================================

Physics-informed intermediate reservoir simulation engine for CO2 EOR.
"""

from .surrogate_models import (
    BaseSurrogateModel,
    AnalyticalSurrogate,
    create_surrogate_model,
    get_available_surrogate_models,
)
from .surrogate_engine import SurrogateEngine, SurrogateEngineWrapper
from .analytical_models import (
    AnalyticalRecoveryModel,
    BuckleyLeverettSurrogate,
    MiscibleSurrogate,
    ImmiscibleSurrogate,
    HybridSurrogate,
    PhDHybridSurrogate,
)
from .profile_generator_fast import FastProfileGenerator
from .pvt_state import SolventExtendedPVTEngine
from .geomechanics_fault import GeomechanicsFaultModel

__all__ = [
    # Base and Surrogate Models
    "BaseSurrogateModel",
    "AnalyticalSurrogate",
    "create_surrogate_model",
    "get_available_surrogate_models",
    # Main Engine
    "SurrogateEngine",
    "SurrogateEngineWrapper",
    # Physics Engines
    "FastProfileGenerator",
    "SolventExtendedPVTEngine",
    "GeomechanicsFaultModel",
    # Analytical Recovery Models
    "AnalyticalRecoveryModel",
    "BuckleyLeverettSurrogate",
    "MiscibleSurrogate",
    "ImmiscibleSurrogate",
    "HybridSurrogate",
    "PhDHybridSurrogate",
]

__version__ = "0.8.5"
