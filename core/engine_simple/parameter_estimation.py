"""
Parameter Estimation using Ensemble Kalman Filter
==================================================

This module contains Ensemble Kalman Filter (EnKF) implementation
for parameter estimation in reservoir simulation.

Based on the theoretical framework from the technical specification document.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


@dataclass
class EnKFParameters:
    """Parameters for Ensemble Kalman Filter"""
    ensemble_size: int = 100              # Number of ensemble members
    inflation_factor: float = 1.0         # Covariance inflation factor
    localization_cutoff: float = None     # Distance cutoff for localization
    observation_error_variance: float = 1e-4  # Observation error variance
    parameter_error_variance: float = 1e-2  # Prior parameter error variance

    def __post_init__(self):
        """Validate EnKF parameters"""
        if self.ensemble_size <= 0:
            raise ValueError("Ensemble size must be positive")

        if self.inflation_factor <= 0:
            raise ValueError("Inflation factor must be positive")

        if self.observation_error_variance <= 0:
            raise ValueError("Observation error variance must be positive")

        if self.parameter_error_variance <= 0:
            raise ValueError("Parameter error variance must be positive")


@dataclass
class ParameterState:
    """Container for parameter state variables"""
    # Reservoir parameters
    permeability: np.ndarray = None           # Permeability field (mD)
    porosity: np.ndarray = None               # Porosity field (fraction)

    # Relative permeability parameters
    krw0: np.ndarray = None                   # Water end-point rel perm
    kro0: np.ndarray = None                   # Oil end-point rel perm
    corey_nw: np.ndarray = None               # Water Corey exponent
    corey_no: np.ndarray = None               # Oil Corey exponent

    # Fluid properties
    oil_viscosity: np.ndarray = None          # Oil viscosity (Pa·s)
    water_viscosity: np.ndarray = None        # Water viscosity (Pa·s)

    # CO₂ parameters
    co2_mmp: np.ndarray = None                # Minimum miscibility pressure (Pa)
    co2_solubility: np.ndarray = None         # CO₂ solubility factor

    # Initial conditions
    initial_pressure: np.ndarray = None       # Initial pressure (Pa)
    initial_saturation: np.ndarray = None     # Initial saturation

    def __post_init__(self):
        """Initialize parameter arrays if None"""
        # This is a placeholder - actual initialization would depend on grid size
        pass


class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter for parameter estimation in reservoir simulation
    """

    def __init__(self, params: EnKFParameters):
        """
        Initialize Ensemble Kalman Filter

        Parameters:
        -----------
        params : EnKFParameters
            EnKF configuration parameters
        """
        self.params = params
        self.ensemble = []
        self.observations = []
        self.observation_times = []

    def initialize_ensemble(self, prior_mean: Dict, prior_std: Dict,
                          grid_shape: Tuple[int, int, int]) -> List[ParameterState]:
        """
        Initialize ensemble with prior distributions

        Parameters:
        -----------
        prior_mean : dict
            Prior means for parameters
        prior_std : dict
            Prior standard deviations for parameters
        grid_shape : tuple
            Grid shape (nz, ny, nx)

        Returns:
        --------
        list : Initial ensemble
        """
        n_cells = np.prod(grid_shape)
        ensemble = []

        for i in range(self.params.ensemble_size):
            state = ParameterState()

            # Initialize reservoir parameters
            if 'permeability' in prior_mean:
                state.permeability = np.random.lognormal(
                    np.log(prior_mean['permeability']),
                    prior_std['permeability'] / prior_mean['permeability'],
                    n_cells
                ).reshape(grid_shape)

            if 'porosity' in prior_mean:
                state.porosity = np.random.normal(
                    prior_mean['porosity'],
                    prior_std['porosity'],
                    n_cells
                ).reshape(grid_shape)
                # Ensure porosity is within bounds
                state.porosity = np.clip(state.porosity, 0.01, 0.5)

            # Initialize relative permeability parameters
            if 'krw0' in prior_mean:
                state.krw0 = np.random.normal(
                    prior_mean['krw0'],
                    prior_std['krw0'],
                    self.params.ensemble_size
                )

            if 'kro0' in prior_mean:
                state.kro0 = np.random.normal(
                    prior_mean['kro0'],
                    prior_std['kro0'],
                    self.params.ensemble_size
                )

            if 'corey_nw' in prior_mean:
                state.corey_nw = np.random.normal(
                    prior_mean['corey_nw'],
                    prior_std['corey_nw'],
                    self.params.ensemble_size
                )
                state.corey_nw = np.clip(state.corey_nw, 0.5, 5.0)

            if 'corey_no' in prior_mean:
                state.corey_no = np.random.normal(
                    prior_mean['corey_no'],
                    prior_std['corey_no'],
                    self.params.ensemble_size
                )
                state.corey_no = np.clip(state.corey_no, 0.5, 5.0)

            # Initialize fluid properties
            if 'oil_viscosity' in prior_mean:
                state.oil_viscosity = np.random.normal(
                    prior_mean['oil_viscosity'],
                    prior_std['oil_viscosity'],
                    self.params.ensemble_size
                )
                state.oil_viscosity = np.clip(state.oil_viscosity, 0.0001, 0.1)

            if 'water_viscosity' in prior_mean:
                state.water_viscosity = np.random.normal(
                    prior_mean['water_viscosity'],
                    prior_std['water_viscosity'],
                    self.params.ensemble_size
                )
                state.water_viscosity = np.clip(state.water_viscosity, 0.0001, 0.1)

            ensemble.append(state)

        self.ensemble = ensemble
        return ensemble

    def forecast_step(self, forward_model: Callable,
                     time_step: float, control_inputs: Dict) -> List[ParameterState]:
        """
        Forecast step: propagate ensemble through forward model

        Parameters:
        -----------
        forward_model : callable
            Forward model function
        time_step : float
            Time step size
        control_inputs : dict
            Control inputs (injection rates, etc.)

        Returns:
        --------
        list : Forecasted ensemble
        """
        forecasted_ensemble = []

        for i, state in enumerate(self.ensemble):
            # Run forward model for each ensemble member
            try:
                forecasted_state = forward_model(state, time_step, control_inputs)
                forecasted_ensemble.append(forecasted_state)
            except Exception as e:
                warnings.warn(f"Forward model failed for ensemble member {i}: {e}")
                # Keep original state if forward model fails
                forecasted_ensemble.append(state)

        self.ensemble = forecasted_ensemble
        return forecasted_ensemble

    def analysis_step(self, observations: np.ndarray,
                     observation_operator: Callable,
                     observation_error_cov: np.ndarray = None) -> List[ParameterState]:
        """
        Analysis step: update ensemble with observations

        Parameters:
        -----------
        observations : np.ndarray
            Observation vector
        observation_operator : callable
            Function to map state to observation space
        observation_error_cov : np.ndarray
            Observation error covariance matrix

        Returns:
        --------
        list : Updated ensemble
        """
        if observation_error_cov is None:
            n_obs = len(observations)
            observation_error_cov = np.eye(n_obs) * self.params.observation_error_variance

        # Convert ensemble to matrix form
        ensemble_matrix = self._ensemble_to_matrix()
        n_state = ensemble_matrix.shape[0]

        # Calculate ensemble mean and covariance
        ensemble_mean = np.mean(ensemble_matrix, axis=1, keepdims=True)
        ensemble_anomaly = ensemble_matrix - ensemble_mean
        ensemble_cov = (ensemble_anomaly @ ensemble_anomaly.T) / (self.params.ensemble_size - 1)

        # Apply covariance inflation
        ensemble_cov *= self.params.inflation_factor

        # Generate observations for each ensemble member
        predicted_observations = []
        for state in self.ensemble:
            try:
                obs = observation_operator(state)
                predicted_observations.append(obs)
            except Exception as e:
                warnings.warn(f"Observation operator failed: {e}")
                predicted_observations.append(np.zeros_like(observations))

        predicted_obs_matrix = np.array(predicted_observations).T
        predicted_obs_mean = np.mean(predicted_obs_matrix, axis=1, keepdims=True)
        predicted_obs_anomaly = predicted_obs_matrix - predicted_obs_mean

        # Calculate cross-covariance
        cross_cov = (ensemble_anomaly @ predicted_obs_anomaly.T) / (self.params.ensemble_size - 1)

        # Predicted observation covariance
        predicted_obs_cov = (predicted_obs_anomaly @ predicted_obs_anomaly.T) / (self.params.ensemble_size - 1)

        # Kalman gain
        innovation_cov = predicted_obs_cov + observation_error_cov
        try:
            kalman_gain = cross_cov @ np.linalg.pinv(innovation_cov)
        except np.linalg.LinAlgError:
            warnings.warn("Singular innovation covariance, using pseudo-inverse")
            kalman_gain = cross_cov @ np.linalg.pinv(innovation_cov, rcond=1e-10)

        # Update ensemble
        innovation = observations - predicted_obs_mean.flatten()
        updated_ensemble_matrix = ensemble_matrix + kalman_gain @ innovation.reshape(-1, 1)

        # Add observation perturbations
        obs_perturbations = np.random.multivariate_normal(
            np.zeros(len(observations)),
            observation_error_cov,
            self.params.ensemble_size
        ).T

        # Kalman gain with perturbations
        for i in range(self.params.ensemble_size):
            innovation_i = observations + obs_perturbations[:, i] - predicted_obs_matrix[:, i]
            updated_ensemble_matrix[:, i] = ensemble_matrix[:, i] + kalman_gain @ innovation_i

        # Convert back to ensemble states
        updated_ensemble = self._matrix_to_ensemble(updated_ensemble_matrix)

        # Apply constraints and bounds
        updated_ensemble = self._apply_constraints(updated_ensemble)

        self.ensemble = updated_ensemble
        return updated_ensemble

    def _ensemble_to_matrix(self) -> np.ndarray:
        """Convert ensemble to matrix form"""
        # This is a simplified implementation
        # In practice, you would flatten all parameters
        n_params = len(self.ensemble[0].__dict__)
        matrix = np.zeros((n_params, self.params.ensemble_size))

        for i, state in enumerate(self.ensemble):
            j = 0
            for attr, value in state.__dict__.items():
                if value is not None and np.isscalar(value):
                    matrix[j, i] = value
                    j += 1

        return matrix

    def _matrix_to_ensemble(self, matrix: np.ndarray) -> List[ParameterState]:
        """Convert matrix back to ensemble states"""
        ensemble = []
        n_params = matrix.shape[0]

        for i in range(self.params.ensemble_size):
            state = ParameterState()
            j = 0
            for attr in state.__dict__.keys():
                if j < n_params:
                    setattr(state, attr, matrix[j, i])
                    j += 1
            ensemble.append(state)

        return ensemble

    def _apply_constraints(self, ensemble: List[ParameterState]) -> List[ParameterState]:
        """Apply physical constraints to ensemble members"""
        for state in ensemble:
            # Ensure positive values
            if state.permeability is not None:
                state.permeability = np.maximum(state.permeability, 0.1)  # Min 0.1 mD

            if state.porosity is not None:
                state.porosity = np.clip(state.porosity, 0.01, 0.5)

            if state.krw0 is not None:
                state.krw0 = np.clip(state.krw0, 0.0, 1.0)

            if state.kro0 is not None:
                state.kro0 = np.clip(state.kro0, 0.0, 1.0)

            if state.corey_nw is not None:
                state.corey_nw = np.maximum(state.corey_nw, 0.5)

            if state.corey_no is not None:
                state.corey_no = np.maximum(state.corey_no, 0.5)

            if state.oil_viscosity is not None:
                state.oil_viscosity = np.maximum(state.oil_viscosity, 0.0001)

            if state.water_viscosity is not None:
                state.water_viscosity = np.maximum(state.water_viscosity, 0.0001)

        return ensemble

    def get_ensemble_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate ensemble statistics

        Returns:
        --------
        dict : Parameter statistics
        """
        stats = {}

        if len(self.ensemble) == 0:
            return stats

        # Calculate statistics for each parameter
        for attr in self.ensemble[0].__dict__.keys():
            values = []
            for state in self.ensemble:
                value = getattr(state, attr)
                if value is not None and np.isscalar(value):
                    values.append(value)

            if values:
                values = np.array(values)
                stats[attr] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        return stats

    def get_best_member(self) -> ParameterState:
        """
        Get the best ensemble member (closest to ensemble mean)

        Returns:
        --------
        ParameterState : Best ensemble member
        """
        if len(self.ensemble) == 0:
            raise ValueError("Ensemble is empty")

        stats = self.get_ensemble_statistics()
        best_idx = 0
        min_distance = np.inf

        for i, state in enumerate(self.ensemble):
            distance = 0.0
            count = 0

            for attr, stat in stats.items():
                value = getattr(state, attr)
                if value is not None and np.isscalar(value):
                    distance += (value - stat['mean']) ** 2 / (stat['std'] ** 2 + 1e-10)
                    count += 1

            if count > 0:
                distance = np.sqrt(distance / count)
                if distance < min_distance:
                    min_distance = distance
                    best_idx = i

        return self.ensemble[best_idx]

    def update_parameters(self, **kwargs):
        """
        Update EnKF parameters

        Parameters:
        -----------
        **kwargs : dict
            Parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")

        # Re-validate parameters
        self.params.__post_init__()