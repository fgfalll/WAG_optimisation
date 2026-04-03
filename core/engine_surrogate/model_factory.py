"""
Model Factory for Surrogate Engine
==================================

Factory functions for creating surrogate models.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

from .surrogate_models import (
    BaseSurrogateModel,
    AnalyticalSurrogate,
    ResponseSurfaceSurrogate,
    create_surrogate_model as create_base_model,
)
from .analytical_models import get_analytical_model, get_available_models
from .response_surfaces import (
    PolynomialResponseSurface,
    RBFResponseSurface,
    create_response_surface,
)


def create_surrogate_model(
    model_type: str = "analytical",
    **kwargs
) -> BaseSurrogateModel:
    """
    Factory function to create surrogate models.

    Args:
        model_type: Type of surrogate model
            - "analytical": Use analytical recovery correlations
            - "response_surface": Use polynomial or RBF response surface
        **kwargs: Additional model-specific parameters

    Returns:
        BaseSurrogateModel instance

    Examples:
        >>> # Create analytical surrogate with hybrid recovery model
        >>> model = create_surrogate_model("analytical", recovery_model_type="hybrid")
        >>>
        >>> # Create polynomial response surface
        >>> model = create_surrogate_model("response_surface",
        ...                                 surface_type="polynomial",
        ...                                 degree=2)
        >>>
        >>> # Create RBF response surface
        >>> model = create_surrogate_model("response_surface",
        ...                                 surface_type="rbf",
        ...                                 function="multiquadric")
    """
    return create_base_model(model_type=model_type, **kwargs)


def get_available_surrogate_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available surrogate models.

    Returns:
        Dictionary with model information including:
            - name: Display name
            - description: Brief description
            - speed: Expected evaluation time
            - accuracy: Expected accuracy
            - requires_training: Whether training is required
    """
    return {
        "analytical": {
            "name": "Analytical Surrogate",
            "description": "Ultra-fast analytical recovery correlations",
            "speed": "0.1ms",
            "accuracy": "Screening quality (~10% error)",
            "requires_training": False,
            "recovery_models": get_available_models(),
        },
        "response_surface": {
            "name": "Response Surface",
            "description": "Polynomial/RBF interpolation of simulation data",
            "speed": "0.5ms",
            "accuracy": "Good with good training data (~5% error)",
            "requires_training": True,
            "surface_types": ["polynomial", "rbf"],
        },
    }


def get_model_config_recommendations() -> Dict[str, Any]:
    """
    Get recommended model configurations for different use cases.

    Returns:
        Dictionary with configuration recommendations
    """
    return {
        "optimization_screening": {
            "description": "Fast evaluation for optimization algorithms",
            "recommended_model": "analytical",
            "recommended_recovery": "hybrid",
            "reason": "Fastest evaluation, suitable for screening",
        },
        "sensitivity_analysis": {
            "description": "Parameter sensitivity studies",
            "recommended_model": "analytical",
            "recommended_recovery": "miscible",
            "reason": "Consistent analytical derivatives",
        },
        "uncertainty_quantification": {
            "description": "Monte Carlo uncertainty analysis",
            "recommended_model": "analytical",
            "recommended_recovery": "hybrid",
            "reason": "Fast for many evaluations",
        },
        "final_prediction": {
            "description": "Final prediction with best accuracy",
            "recommended_model": "response_surface",
            "recommended_surface": "polynomial",
            "reason": "Best accuracy with trained model",
        },
    }


def validate_surrogate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate surrogate engine configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with validation results:
            - is_valid: Overall validity
            - errors: List of error messages
            - warnings: List of warning messages
    """
    errors = []
    warnings = []

    # Check model type
    model_type = config.get("model_type", "analytical")
    if model_type not in ["analytical", "response_surface"]:
        errors.append(f"Invalid model_type: {model_type}")

    # Check recovery model type for analytical
    if model_type == "analytical":
        recovery_model = config.get("recovery_model_type", "hybrid")
        available_models = get_available_models()
        if recovery_model not in available_models:
            errors.append(f"Invalid recovery_model_type: {recovery_model}")

    # Check surface type for response surface
    if model_type == "response_surface":
        surface_type = config.get("surface_type", "polynomial")
        if surface_type not in ["polynomial", "rbf"]:
            errors.append(f"Invalid surface_type: {surface_type}")

        # Check degree for polynomial
        if surface_type == "polynomial":
            degree = config.get("degree", 2)
            if not (1 <= degree <= 5):
                warnings.append(f"Polynomial degree {degree} outside typical range [1, 5]")

        # Check RBF function
        if surface_type == "rbf":
            rbf_function = config.get("rbf_function", "multiquadric")
            valid_functions = ["multiquadric", "inverse", "gaussian", "linear", "cubic"]
            if rbf_function not in valid_functions:
                errors.append(f"Invalid rbf_function: {rbf_function}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
