"""
Fast Surrogate Engine for CO2 EOR Optimization
==============================================

This module provides ultra-fast surrogate models for optimization screening.
Uses analytical models and response surfaces instead of numerical simulation.

Performance Targets:
- Evaluation time: < 1ms per scenario
- Accuracy: < 10% relative error vs simple engine
- Memory: < 100MB for trained models
- Speedup: 1000x faster than simple engine
"""

from .surrogate_models import (
    BaseSurrogateModel,
    AnalyticalSurrogate,
    ResponseSurfaceSurrogate,
)
from .surrogate_engine import SurrogateEngine, SurrogateEngineWrapper
from .analytical_models import (
    AnalyticalRecoveryModel,
    BuckleyLeverettSurrogate,
    MiscibleSurrogate,
    ImmiscibleSurrogate,
    HybridSurrogate,
)
from .response_surfaces import (
    PolynomialResponseSurface,
    RBFResponseSurface,
)
from .model_factory import create_surrogate_model, get_available_surrogate_models
from .feature_transformer import FeatureTransformer

__all__ = [
    # Base classes
    "BaseSurrogateModel",
    "AnalyticalSurrogate",
    "ResponseSurfaceSurrogate",
    # Main engine
    "SurrogateEngine",
    "SurrogateEngineWrapper",
    # Analytical models
    "AnalyticalRecoveryModel",
    "BuckleyLeverettSurrogate",
    "MiscibleSurrogate",
    "ImmiscibleSurrogate",
    "HybridSurrogate",
    # Response surfaces
    "PolynomialResponseSurface",
    "RBFResponseSurface",
    # Factory
    "create_surrogate_model",
    "get_available_surrogate_models",
    # Utilities
    "FeatureTransformer",
]

# Version info
__version__ = "0.1.0"
__author__ = "CO2 EOR Optimizer Team"
