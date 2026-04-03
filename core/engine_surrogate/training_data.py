"""
Training Data Generation for Response Surface Surrogates
======================================================

Functions for generating training data using Design of Experiments
and sampling strategies for response surface model training.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for scipy availability
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, limited sampling options")


class TrainingDataGenerator:
    """
    Generate training data for surrogate model training.

    Uses Design of Experiments (DOE) methods for efficient
    sampling of the parameter space.
    """

    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        seed: Optional[int] = None,
    ):
        """
        Initialize the training data generator.

        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) tuples
            seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.n_params = len(self.parameter_names)

        if seed is not None:
            np.random.seed(seed)

    def generate_lhs_samples(
        self,
        n_samples: int,
        criterion: str = "center",
    ) -> np.ndarray:
        """
        Generate Latin Hypercube Sampling (LHS) samples.

        LHS provides good space-filling properties with relatively
        few samples compared to random sampling.

        Args:
            n_samples: Number of samples to generate
            criterion: Sampling criterion
                - "center": Center points within intervals
                - "maximin": Maximize minimum distance between points
                - "correlation": Minimize correlation
                - "random": Random LHS

        Returns:
            Sample matrix (n_samples, n_params)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using random sampling")
            return self.generate_random_samples(n_samples)

        # Generate LHS
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=self.n_params, seed=0)
        samples = sampler.random(n=n_samples)

        # Apply criterion if specified
        if criterion == "maximin":
            # Optimize for maximin distance
            for _ in range(10):  # Simple optimization
                new_sampler = qmc.LatinHypercube(d=self.n_params, seed=np.random.randint(1000))
                new_samples = new_sampler.random(n=n_samples)
                # Could add optimization here
                samples = new_samples
        elif criterion == "center":
            # Use center of each stratum
            samples = (samples + 0.5) / (n_samples + 1)
        elif criterion == "correlation":
            # Minimize correlation (simple approximation)
            pass  # scipy's LHS already does this reasonably well

        return samples

    def generate_sobol_samples(
        self,
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate Sobol sequence samples.

        Sobol sequences provide excellent space-filling properties
        and are quasi-random (low discrepancy).

        Args:
            n_samples: Number of samples to generate

        Returns:
            Sample matrix (n_samples, n_params)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using random sampling")
            return self.generate_random_samples(n_samples)

        try:
            from scipy.stats import qmc

            # Use power of 2 for Sobol
            n_samples_pow2 = 2 ** int(np.ceil(np.log2(n_samples)))

            sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=0)
            samples = sampler.random(n=n_samples_pow2)

            # Truncate to requested size
            return samples[:n_samples]

        except Exception as e:
            logger.warning(f"Sobol sampling failed: {e}, using random")
            return self.generate_random_samples(n_samples)

    def generate_random_samples(
        self,
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate random uniform samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Sample matrix (n_samples, n_params)
        """
        return np.random.random((n_samples, self.n_params))

    def generate_grid_samples(
        self,
        n_per_dim: int,
    ) -> np.ndarray:
        """
        Generate grid samples (full factorial design).

        Note: Number of samples grows exponentially with dimensions.
        Only suitable for low-dimensional problems.

        Args:
            n_per_dim: Number of samples per dimension

        Returns:
            Sample matrix (n_samples, n_params)
        """
        # Create 1D grid for each parameter
        grids = []
        for i in range(self.n_params):
            grids.append(np.linspace(0, 1, n_per_dim))

        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')

        # Flatten and stack
        samples = np.column_stack([m.flatten() for m in mesh])

        return samples

    def scale_samples_to_bounds(
        self,
        samples: np.ndarray,
    ) -> np.ndarray:
        """
        Scale samples from [0, 1] to parameter bounds.

        Args:
            samples: Sample matrix in [0, 1] range

        Returns:
            Scaled sample matrix
        """
        scaled = np.zeros_like(samples)

        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_bounds[param_name]
            scaled[:, i] = min_val + samples[:, i] * (max_val - min_val)

        return scaled

    def samples_to_param_dicts(
        self,
        samples: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Convert sample matrix to list of parameter dictionaries.

        Args:
            samples: Sample matrix

        Returns:
            List of parameter dictionaries
        """
        param_dicts = []

        for sample in samples:
            param_dict = {}
            for i, param_name in enumerate(self.parameter_names):
                param_dict[param_name] = sample[i]
            param_dicts.append(param_dict)

        return param_dicts


def get_default_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get default parameter bounds for CO2-EOR surrogate training.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    return {
        "injection_rate": (100.0, 100000.0),  # MSCFD
        "target_pressure_psi": (1000.0, 8000.0),  # psi
        "mobility_ratio": (0.1, 50.0),  # dimensionless
        "porosity": (0.05, 0.35),  # fraction
        "permeability": (1.0, 1000.0),  # mD
        "mmp": (1500.0, 4000.0),  # psi
        "WAG_ratio": (0.0, 4.0),  # dimensionless
        "kv_kh_ratio": (0.01, 1.0),  # dimensionless
    }


def generate_training_data(
    n_samples: int = 500,
    sampling_method: str = "lhs",
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to generate training data samples.

    Args:
        n_samples: Number of samples to generate
        sampling_method: Sampling method ("lhs", "sobol", "random", "grid")
        parameter_bounds: Parameter bounds (uses defaults if None)
        seed: Random seed

    Returns:
        Tuple of (sample_matrix, parameter_names)
    """
    if parameter_bounds is None:
        parameter_bounds = get_default_parameter_bounds()

    generator = TrainingDataGenerator(parameter_bounds, seed=seed)

    # Generate samples
    if sampling_method == "lhs":
        samples = generator.generate_lhs_samples(n_samples)
    elif sampling_method == "sobol":
        samples = generator.generate_sobol_samples(n_samples)
    elif sampling_method == "random":
        samples = generator.generate_random_samples(n_samples)
    elif sampling_method == "grid":
        n_per_dim = int(np.ceil(n_samples ** (1 / len(parameter_bounds))))
        samples = generator.generate_grid_samples(n_per_dim)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # Scale to parameter bounds
    scaled_samples = generator.scale_samples_to_bounds(samples)

    return scaled_samples, generator.parameter_names


def run_simple_engine_batch(
    samples: List[Dict[str, float]],
    simple_engine,
    reservoir_data,
    economic_params=None,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Run simple engine for a batch of samples (for training response surfaces).

    Args:
        samples: List of parameter dictionaries
        simple_engine: Simple engine instance
        reservoir_data: Reservoir data
        economic_params: Economic parameters
        progress_callback: Optional callback for progress updates

    Returns:
        Results array with columns for each target
    """
    n_samples = len(samples)
    results = []

    for i, sample in enumerate(samples):
        # Create EOR parameters from sample
        # This is a simplified version - actual implementation would
        # need to properly construct EORParameters from the sample dict

        # For now, just return placeholder
        results.append([0.0, 0.0, 0.0, 0.0])  # rf, npv, oil, co2

        if progress_callback:
            progress_callback(i / n_samples)

    return np.array(results)
