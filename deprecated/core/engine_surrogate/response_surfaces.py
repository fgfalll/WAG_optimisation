"""
Response Surface Models for Surrogate Engine
============================================

Polynomial and RBF response surface models for fast prediction
after training on simulation data.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for scipy/sklearn availability
try:
    from scipy.interpolate import Rbf
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge
    SCIPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    logger.warning("scipy/sklearn not available, response surface models limited")


class ResponseSurfaceBase:
    """Base class for response surface models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.n_features = 0
        self.n_targets = 0
        self.feature_names: List[str] = []
        self.target_names: List[str] = []


class PolynomialResponseSurface(ResponseSurfaceBase):
    """
    Polynomial response surface model.

    Uses polynomial features with ridge regression for
    smooth interpolation of training data.

    Suitable for: Smooth, low-dimensional problems
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize polynomial response surface.

        Args:
            degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic)
            alpha: Ridge regularization parameter
            feature_names: Names of input features
        """
        super().__init__("polynomial_response_surface")
        self.degree = degree
        self.alpha = alpha

        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, polynomial response surface disabled")
            return

        self.poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        self.scaler = StandardScaler()
        self.models: List[Ridge] = []

        if feature_names:
            self.feature_names = feature_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train the polynomial response surface.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) or (n_samples, n_targets)
            feature_names: Optional feature names
            target_names: Optional target names
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn is required for PolynomialResponseSurface")

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Handle feature/target names
        if feature_names:
            self.feature_names = feature_names
        if target_names:
            self.target_names = target_names

        # Handle multi-target y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_targets = y.shape[1]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create polynomial features
        X_poly = self.poly_features.fit_transform(X_scaled)

        # Train a separate model for each target
        self.models = []
        for target_idx in range(self.n_targets):
            model = Ridge(alpha=self.alpha)
            model.fit(X_poly, y[:, target_idx])

            # Calculate training score
            score = model.score(X_poly, y[:, target_idx])
            logger.debug(f"Polynomial model for target {target_idx}: R² = {score:.4f}")

            self.models.append(model)

        self.is_trained = True
        logger.info(f"Trained polynomial (degree={self.degree}) response surface: "
                   f"{n_samples} samples, {n_features} features, {self.n_targets} targets")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained polynomial response surface.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_targets)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn is required for PolynomialResponseSurface")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Create polynomial features
        X_poly = self.poly_features.transform(X_scaled)

        # Predict for each target
        y_pred = np.zeros((X.shape[0], self.n_targets))
        for target_idx, model in enumerate(self.models):
            y_pred[:, target_idx] = model.predict(X_poly)

        return y_pred

    def predict_single(self, params: Dict[str, float]) -> np.ndarray:
        """
        Predict for a single parameter dictionary.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Prediction array (n_targets,)
        """
        # Convert params dict to feature vector
        X = self._params_to_features(params)
        return self.predict(X.reshape(1, -1))[0]

    def _params_to_features(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to feature vector."""
        if self.feature_names:
            # Use known feature order
            return np.array([params.get(f, 0.0) for f in self.feature_names])
        else:
            # Use default order
            default_order = [
                "injection_rate", "target_pressure_psi", "mobility_ratio",
                "porosity", "permeability", "mmp"
            ]
            return np.array([params.get(f, 0.0) for f in default_order])

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from trained models.

        Returns:
            Dictionary mapping target names to feature importance arrays
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return {}

        importance = {}
        feature_names_with_bias = self.poly_features.get_feature_names_out(self.feature_names)

        for target_idx, model in enumerate(self.models):
            target_name = self.target_names[target_idx] if target_idx < len(self.target_names) else f"target_{target_idx}"
            # Normalize coefficients by their standard deviation
            coef = model.coef_
            importance[target_name] = dict(zip(feature_names_with_bias, np.abs(coef)))

        return importance


class RBFResponseSurface(ResponseSurfaceBase):
    """
    Radial Basis Function response surface model.

    Uses RBF interpolation for exact or smooth interpolation
    of training data points.

    Suitable for: Non-smooth, higher-dimensional problems
    """

    def __init__(
        self,
        function: str = "multiquadric",
        smooth: float = 0.0,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize RBF response surface.

        Args:
            function: RBF function type
                - "multiquadric": sqrt((r/epsilon)^2 + 1)
                - "inverse": 1/sqrt((r/epsilon)^2 + 1)
                - "gaussian": exp(-(r/epsilon)^2)
                - "linear": r
                - "cubic": r^3
                - "quintic": r^5
                - "thin_plate": r^2 * log(r)
            smooth: Smoothing parameter (0 = exact interpolation)
            feature_names: Names of input features
        """
        super().__init__("rbf_response_surface")
        self.function = function
        self.smooth = smooth

        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, RBF response surface disabled")
            return

        self.scaler = StandardScaler()
        self.models: List[Any] = []

        if feature_names:
            self.feature_names = feature_names

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
    ) -> None:
        """
        Train the RBF response surface.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) or (n_samples, n_targets)
            feature_names: Optional feature names
            target_names: Optional target names
        """
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for RBFResponseSurface")

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Handle feature/target names
        if feature_names:
            self.feature_names = feature_names
        if target_names:
            self.target_names = target_names

        # Handle multi-target y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_targets = y.shape[1]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train separate RBF for each target
        self.models = []
        for target_idx in range(self.n_targets):
            # Create RBF interpolator
            # Scipy's Rbf expects separate arguments for each dimension
            rbf = Rbf(*[X_scaled[:, i] for i in range(n_features)],
                      y[:, target_idx],
                      function=self.function,
                      smooth=self.smooth)
            self.models.append(rbf)

            # Calculate training error
            y_pred = rbf(*[X_scaled[:, i] for i in range(n_features)])
            rmse = np.sqrt(np.mean((y[:, target_idx] - y_pred) ** 2))
            logger.debug(f"RBF model for target {target_idx}: RMSE = {rmse:.4f}")

        self.is_trained = True
        logger.info(f"Trained RBF (function={self.function}) response surface: "
                   f"{n_samples} samples, {n_features} features, {self.n_targets} targets")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained RBF response surface.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples, n_targets)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for RBFResponseSurface")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict for each target
        y_pred = np.zeros((X.shape[0], self.n_targets))
        for target_idx, model in enumerate(self.models):
            # Rbf prediction for all samples
            for i in range(X.shape[0]):
                y_pred[i, target_idx] = model(*X_scaled[i, :])

        return y_pred

    def predict_single(self, params: Dict[str, float]) -> np.ndarray:
        """
        Predict for a single parameter dictionary.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Prediction array (n_targets,)
        """
        # Convert params dict to feature vector
        X = self._params_to_features(params)
        return self.predict(X.reshape(1, -1))[0]

    def _params_to_features(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to feature vector."""
        if self.feature_names:
            # Use known feature order
            return np.array([params.get(f, 0.0) for f in self.feature_names])
        else:
            # Use default order
            default_order = [
                "injection_rate", "target_pressure_psi", "mobility_ratio",
                "porosity", "permeability", "mmp"
            ]
            return np.array([params.get(f, 0.0) for f in default_order])


def create_response_surface(
    surface_type: str = "polynomial",
    **kwargs
) -> ResponseSurfaceBase:
    """
    Factory function to create response surface models.

    Args:
        surface_type: Type of response surface ("polynomial" or "rbf")
        **kwargs: Additional arguments passed to model constructor

    Returns:
        ResponseSurfaceBase instance
    """
    if surface_type == "polynomial":
        return PolynomialResponseSurface(**kwargs)
    elif surface_type == "rbf":
        return RBFResponseSurface(**kwargs)
    else:
        raise ValueError(f"Unknown response surface type: {surface_type}. "
                        f"Available: ['polynomial', 'rbf']")


def get_available_response_surfaces() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available response surface types.

    Returns:
        Dictionary with response surface information
    """
    return {
        "polynomial": {
            "name": "Polynomial Response Surface",
            "description": "Global polynomial approximation with ridge regression",
            "best_for": "Smooth, low-dimensional problems",
            "requires_sklearn": True,
        },
        "rbf": {
            "name": "RBF Response Surface",
            "description": "Local radial basis function interpolation",
            "best_for": "Non-smooth, scattered data",
            "requires_scipy": True,
        },
    }
