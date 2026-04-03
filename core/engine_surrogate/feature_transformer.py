"""
Feature Transformer for Surrogate Engine
========================================

Handles parameter normalization and feature transformation
for training and prediction with response surface models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for sklearn availability
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, feature transformation limited")


class FeatureTransformer:
    """
    Feature transformer for normalizing and transforming parameters.

    Supports various scaling methods and feature transformations
    to improve surrogate model performance.
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        apply_log_transform: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize feature transformer.

        Args:
            scaling_method: Scaling method
                - "standard": StandardScaler (zero mean, unit variance)
                - "minmax": MinMaxScaler (scale to [0, 1])
                - "robust": RobustScaler (median and IQR, robust to outliers)
                - "none": No scaling
            apply_log_transform: List of feature names to apply log transform
            feature_names: Names of features
        """
        self.scaling_method = scaling_method
        self.apply_log_transform = apply_log_transform or []
        self.feature_names = feature_names or []

        # Initialize scaler
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using manual scaling")
            self.scaler = None
            self._use_sklearn = False
        else:
            self._use_sklearn = True
            if scaling_method == "standard":
                self.scaler = StandardScaler()
            elif scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            elif scaling_method == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = None
                self._use_sklearn = False

        # Manual scaling parameters (when sklearn not available)
        self._manual_scale_mean = None
        self._manual_scale_std = None
        self._manual_scale_min = None
        self._manual_scale_max = None

        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "FeatureTransformer":
        """
        Fit the transformer to data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            self (fitted transformer)
        """
        # Apply log transform if specified
        X_transformed = self._apply_log_transform(X)

        # Fit scaler
        if self._use_sklearn and self.scaler is not None:
            self.scaler.fit(X_transformed)
        elif self.scaling_method == "standard":
            self._manual_scale_mean = np.mean(X_transformed, axis=0)
            self._manual_scale_std = np.std(X_transformed, axis=0)
        elif self.scaling_method == "minmax":
            self._manual_scale_min = np.min(X_transformed, axis=0)
            self._manual_scale_max = np.max(X_transformed, axis=0)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before transform")

        # Apply log transform
        X_transformed = self._apply_log_transform(X)

        # Apply scaling
        if self._use_sklearn and self.scaler is not None:
            return self.scaler.transform(X_transformed)
        elif self.scaling_method == "standard" and self._manual_scale_mean is not None:
            return (X_transformed - self._manual_scale_mean) / (self._manual_scale_std + 1e-10)
        elif self.scaling_method == "minmax" and self._manual_scale_min is not None:
            return (X_transformed - self._manual_scale_min) / (self._manual_scale_max - self._manual_scale_min + 1e-10)
        else:
            return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit transformer and return transformed features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Transformed feature matrix
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Feature matrix in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before inverse_transform")

        # Inverse scaling
        if self._use_sklearn and self.scaler is not None:
            X_transformed = self.scaler.inverse_transform(X_scaled)
        elif self.scaling_method == "standard" and self._manual_scale_mean is not None:
            X_transformed = X_scaled * (self._manual_scale_std + 1e-10) + self._manual_scale_mean
        elif self.scaling_method == "minmax" and self._manual_scale_min is not None:
            X_transformed = X_scaled * (self._manual_scale_max - self._manual_scale_min + 1e-10) + self._manual_scale_min
        else:
            X_transformed = X_scaled

        # Inverse log transform
        X_original = self._inverse_log_transform(X_transformed)

        return X_original

    def _apply_log_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply log transform to specified features."""
        if not self.apply_log_transform:
            return X

        X_transformed = X.copy()

        for feature_name in self.apply_log_transform:
            if self.feature_names and feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                # Add small constant to avoid log(0)
                X_transformed[:, idx] = np.log(X[:, idx] + 1e-10)

        return X_transformed

    def _inverse_log_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse log transform for specified features."""
        if not self.apply_log_transform:
            return X

        X_original = X.copy()

        for feature_name in self.apply_log_transform:
            if self.feature_names and feature_name in self.feature_names:
                idx = self.feature_names.index(feature_name)
                X_original[:, idx] = np.exp(X[:, idx])

        return X_original

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names

    def set_feature_names(self, feature_names: List[str]) -> None:
        """Set feature names."""
        self.feature_names = feature_names


def create_parameter_dict(
    feature_vector: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Convert feature vector to parameter dictionary.

    Args:
        feature_vector: Feature values
        feature_names: Corresponding feature names

    Returns:
        Dictionary mapping feature names to values
    """
    return {name: value for name, value in zip(feature_names, feature_vector)}


def create_feature_matrix(
    param_dicts: List[Dict[str, float]],
    feature_names: List[str],
) -> np.ndarray:
    """
    Convert list of parameter dictionaries to feature matrix.

    Args:
        param_dicts: List of parameter dictionaries
        feature_names: Ordered list of feature names

    Returns:
        Feature matrix (n_samples, n_features)
    """
    n_samples = len(param_dicts)
    n_features = len(feature_names)

    X = np.zeros((n_samples, n_features))

    for i, params in enumerate(param_dicts):
        for j, name in enumerate(feature_names):
            X[i, j] = params.get(name, 0.0)

    return X


def get_default_feature_names() -> List[str]:
    """
    Get default feature names for CO2-EOR surrogate models.

    Returns:
        List of default feature names in order
    """
    return [
        "injection_rate",
        "target_pressure_psi",
        "mobility_ratio",
        "porosity",
        "permeability",
        "mmp",
        "WAG_ratio",
        "kv_kh_ratio",
    ]


def get_default_target_names() -> List[str]:
    """
    Get default target names for CO2-EOR surrogate models.

    Returns:
        List of default target names
    """
    return [
        "recovery_factor",
        "npv",
        "cumulative_oil",
        "co2_stored",
    ]
