
import numpy as np
import gstools as gs
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_realistic_range(shape: Tuple[int, int]) -> float:
    """
    Calculate realistic variogram range based on grid dimensions.

    For geological realism, the range should be 30-50% of the largest grid dimension.
    This ensures spatial correlation between adjacent cells while preserving
    geological heterogeneity.

    Args:
        shape: Grid dimensions (nx, ny)

    Returns:
        float: Recommended variogram range in grid units
    """
    max_dimension = max(shape)
    min_dimension = min(shape)

    # Range should be 30-50% of largest dimension for good spatial correlation
    # Use 50% for smoother appearance (was 40%, increased to reduce "bed of nails" effect)
    range_factor = 0.5

    recommended_range = max_dimension * range_factor

    # Ensure range is at least a few cells for correlation (increased from 0.3)
    min_range = min_dimension * 0.4
    recommended_range = max(recommended_range, min_range)

    logger.info(f"Calculated realistic range: {recommended_range:.1f} for grid {shape} "
                f"(max_dim={max_dimension}, range_factor={range_factor})")

    return recommended_range

def create_geostatistical_grid(shape: Tuple[int, int], params: Dict) -> np.ndarray:
    """
    Create a geostatistical grid using specified parameters.

    Args:
        shape: Grid dimensions (nx, ny)
        params: Dictionary of geostatistical parameters

    Returns:
        numpy array containing the generated grid
    """
    try:
        # Validate shape parameter
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError(f"Shape must be a tuple of 2 integers, got: {shape}")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Shape dimensions must be positive, got: {shape}")
        if shape[0] < 3 or shape[1] < 3:
            logger.warning(f"Shape dimensions {shape} are very small. Geostatistical modeling requires minimum 3x3 grid for meaningful results.")

        # Extract parameters with defaults
        variogram_type = params.get('variogram_type', 'spherical')

        # Use user-provided range if available (from facies table), otherwise auto-calculate
        if 'range' in params and params['range'] > 0:
            range_val = params['range']  # USER VALUE - preserved exactly
            logger.info(f"Using user-provided range: {range_val}")
        else:
            # Fallback: auto-calculate for geostatistical mode without facies
            range_val = calculate_realistic_range(shape)
            logger.info(f"Using auto-calculated range: {range_val} (fallback - no facies)")

        sill = params.get('sill', 0.1)      # USER VALUE - preserved
        nugget = params.get('nugget', 0.0)    # USER VALUE - preserved
        anisotropy_ratio = params.get('anisotropy_ratio', 1.0)
        anisotropy_angle = params.get('anisotropy_angle', 0.0)
        trend_type = params.get('trend_type', 'none')
        trend_parameters = params.get('trend_parameters', [0.0, 0.0])
        simulation_method = params.get('simulation_method', 'sequential_gaussian')
        random_seed = params.get('random_seed', 42)
        
        # Create variogram model
        if variogram_type == 'spherical':
            model = gs.Spherical(dim=2, var=sill, len_scale=range_val, nugget=nugget)
        elif variogram_type == 'exponential':
            model = gs.Exponential(dim=2, var=sill, len_scale=range_val, nugget=nugget)
        elif variogram_type == 'gaussian':
            model = gs.Gaussian(dim=2, var=sill, len_scale=range_val, nugget=nugget)
        elif variogram_type == 'matern':
            model = gs.Matern(dim=2, var=sill, len_scale=range_val, nugget=nugget)
        elif variogram_type == 'cubic':
            model = gs.Cubic(dim=2, var=sill, len_scale=range_val, nugget=nugget)
        else:
            raise ValueError(f"Unsupported variogram type: {variogram_type}")
        
        # Apply anisotropy - use the correct gstools API
        if anisotropy_ratio != 1.0:
            # For gstools, anisotropy is applied by setting different length scales
            len_scale_x = range_val
            len_scale_y = range_val / anisotropy_ratio
            # Recreate model with anisotropic parameters
            if variogram_type == 'spherical':
                model = gs.Spherical(dim=2, var=sill, len_scale=[len_scale_x, len_scale_y], angles=anisotropy_angle, nugget=nugget)
            elif variogram_type == 'exponential':
                model = gs.Exponential(dim=2, var=sill, len_scale=[len_scale_x, len_scale_y], angles=anisotropy_angle, nugget=nugget)
            elif variogram_type == 'gaussian':
                model = gs.Gaussian(dim=2, var=sill, len_scale=[len_scale_x, len_scale_y], angles=anisotropy_angle, nugget=nugget)
            elif variogram_type == 'matern':
                model = gs.Matern(dim=2, var=sill, len_scale=[len_scale_x, len_scale_y], angles=anisotropy_angle, nugget=nugget)
            elif variogram_type == 'cubic':
                model = gs.Cubic(dim=2, var=sill, len_scale=[len_scale_x, len_scale_y], angles=anisotropy_angle, nugget=nugget)
        
        # Create random field
        try:
            if simulation_method == 'sequential_gaussian':
                srf = gs.SRF(model, seed=random_seed)
            elif simulation_method == 'turning_bands':
                srf = gs.SRF(model, seed=random_seed, method='turning_bands')
            elif simulation_method == 'fft':
                srf = gs.SRF(model, seed=random_seed, method='fft')
            else:
                raise ValueError(f"Unsupported simulation method: {simulation_method}")

            x = np.arange(shape[0])
            y = np.arange(shape[1])
            # Generate the field using correct gstools API
            # srf.structured() expects a list of coordinate arrays
            field = srf.structured([x, y])

            # Validate that field is a proper array, not a scalar
            if not isinstance(field, np.ndarray):
                logger.error(f"gstools returned non-array type: {type(field)}. Falling back to random field.")
                np.random.seed(random_seed)
                field = np.random.rand(*shape)
            elif field.size == 1:
                logger.error(f"gstools returned scalar field. Falling back to random field.")
                np.random.seed(random_seed)
                field = np.random.rand(*shape)

        except Exception as e:
            logger.error(f"Error in gstools field generation: {e}. Falling back to random field.")
            np.random.seed(random_seed) # Ensure fallback is deterministic
            field = np.random.rand(*shape)

        # Ensure the field has the correct shape
        if field.shape != shape:
            try:
                # If shape is inverted, transpose it
                if field.shape == (shape[1], shape[0]):
                    field = field.T
                else:
                    # Try to flatten and reshape, but only if total elements match
                    if field.size == shape[0] * shape[1]:
                        field = field.flatten().reshape(shape)
                    else:
                        logger.error(f"Field size {field.size} doesn't match target size {shape[0] * shape[1]}. Falling back to random field.")
                        np.random.seed(random_seed)
                        field = np.random.rand(*shape)
            except ValueError as reshape_error:
                logger.error(f"Error reshaping field from {field.shape} to {shape}: {reshape_error}. Falling back to random field.")
                np.random.seed(random_seed)
                field = np.random.rand(*shape)
        
        # Apply trend if specified
        if trend_type != 'none':
            field = _apply_trend(field, trend_type, trend_parameters)
        
        # Normalize to [0, 1] range for porosity representation
        field = _normalize_field(field)
        
        return field
        
    except Exception as e:
        logger.error(f"Error creating geostatistical grid: {e}")
        raise

def create_facies_based_grid(shape: Tuple[int, int], facies_data: List[Dict]) -> np.ndarray:
    """
    Enhanced facies-based grid creation with advanced geological modeling.
    
    Args:
        shape: Grid dimensions (nx, ny)
        facies_data: List of facies definitions with geological parameters
        
    Returns:
        Combined grid with facies distribution
    """
    try:
        # Validate facies proportions
        total_proportion = sum(facies.get('proportion', 0) for facies in facies_data)
        if not np.isclose(total_proportion, 1.0, atol=1e-6):
            raise ValueError(f"Facies proportions must sum to 1.0 (current sum: {total_proportion})")

        # Validate shape parameter
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError(f"Shape must be a tuple of 2 integers, got: {shape}")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Shape dimensions must be positive, got: {shape}")
        if shape[0] < 3 or shape[1] < 3:
            logger.warning(f"Shape dimensions {shape} are very small. Geostatistical modeling requires minimum 3x3 grid for meaningful results.")
            # Continue anyway, but warn the user

        # Create categorical field for facies distribution
        try:
            # Use realistic range for proper spatial correlation
            realistic_range = calculate_realistic_range(shape)

            # Validate the calculated range
            if realistic_range <= 0 or not np.isfinite(realistic_range):
                logger.error(f"Calculated invalid range {realistic_range}. Using fallback range.")
                realistic_range = max(shape) * 0.5

            cat_model = gs.Gaussian(dim=2, var=1.0, len_scale=realistic_range)
            srf_cat = gs.SRF(cat_model, seed=20170520)

            # Create coordinate arrays for gstools (required API)
            # srf.structured() expects a list of coordinate arrays, not a shape tuple
            x = np.arange(shape[0])
            y = np.arange(shape[1])

            # Generate the categorical field using correct gstools API
            cat_field = srf_cat.structured([x, y])

            # Validate that field is a proper array
            if not isinstance(cat_field, np.ndarray):
                logger.error(f"gstools returned non-array type for categorical field: {type(cat_field)}. Using random field.")
                cat_field = np.random.rand(*shape)
            elif cat_field.size == 1:
                logger.error(f"gstools returned scalar categorical field. This may be caused by invalid shape {shape} or gstools version incompatibility. Using random field.")
                cat_field = np.random.rand(*shape)
            elif cat_field.shape != shape and cat_field.shape != (shape[1], shape[0]):
                logger.warning(f"gstools returned field with unexpected shape {cat_field.shape}, expected {shape}. Attempting to reshape...")
                # Try to reshape if the total size matches
                if cat_field.size == shape[0] * shape[1]:
                    cat_field = cat_field.reshape(shape)
                else:
                    logger.error(f"Cannot reshape field from {cat_field.shape} to {shape}. Using random field.")
                    cat_field = np.random.rand(*shape)

            # Ensure the categorical field has the correct shape
            if cat_field.shape != shape:
                try:
                    # If shape is inverted, transpose it
                    if cat_field.shape == (shape[1], shape[0]):
                        cat_field = cat_field.T
                    elif cat_field.size == shape[0] * shape[1]:
                        # Flatten and reshape
                        cat_field = cat_field.flatten().reshape(shape)
                    else:
                        logger.error(f"Categorical field size {cat_field.size} doesn't match target {shape[0] * shape[1]}. Using random field.")
                        cat_field = np.random.rand(*shape)
                except ValueError as reshape_error:
                    logger.error(f"Error reshaping categorical field: {reshape_error}. Using random field.")
                    cat_field = np.random.rand(*shape)

        except Exception as e:
            logger.error(f"Error creating categorical field: {e}")
            # Fallback: create a simple random field for facies distribution
            cat_field = np.random.rand(shape[0], shape[1])
        
        # Create grids for each facies with enhanced geological parameters
        facies_grids = []
        for i, facies in enumerate(facies_data):
            facies_params = facies.get('geostatistical_params', {})
            
            # Enhanced facies properties
            facies_name = facies.get('name', f'Facies_{i+1}')
            proportion = facies.get('proportion', 0.0)
            rock_type = facies.get('rock_type', 'sandstone')
            depositional_env = facies.get('depositional_environment', 'fluvial')
            
            logger.info(f"Creating grid for {facies_name} ({rock_type}, {depositional_env})")
            
            # Create geostatistical field for this facies
            field = create_geostatistical_grid(shape, facies_params)
            
            # Apply facies-specific adjustments based on geological characteristics
            field = _apply_facies_characteristics(field, rock_type, depositional_env)
            
            facies_grids.append(field)
        
        # Combine grids based on facies proportions with improved blending
        combined_grid = _combine_facies_grids(cat_field, facies_data, facies_grids)
        
        return combined_grid
        
    except Exception as e:
        logger.error(f"Error creating facies-based grid: {e}")
        raise

def _apply_trend(field: np.ndarray, trend_type: str, parameters: List[float]) -> np.ndarray:
    """Apply geological trend to the field."""
    ny, nx = field.shape
    
    if trend_type == 'linear':
        # Linear trend: parameters = [slope_x, slope_y]
        x_trend = np.linspace(0, parameters[0], nx)
        y_trend = np.linspace(0, parameters[1], ny)
        trend = x_trend[np.newaxis, :] + y_trend[:, np.newaxis]
        return field + trend
    
    elif trend_type == 'quadratic':
        # Quadratic trend: parameters = [a, b, c]
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        trend = parameters[0] * X**2 + parameters[1] * Y**2 + parameters[2] * X * Y
        return field + trend
    
    else:
        return field

def _apply_facies_characteristics(field: np.ndarray, rock_type: str, depositional_env: str) -> np.ndarray:
    """Apply geological characteristics to facies field."""
    # Rock type adjustments
    if rock_type == 'sandstone':
        # Sandstone typically has higher porosity
        field = field * 1.1
    elif rock_type == 'carbonate':
        # Carbonates often have more heterogeneous porosity
        field = field + np.random.normal(0, 0.05, field.shape)
    elif rock_type == 'shale':
        # Shale has lower porosity
        field = field * 0.7
    
    # Depositional environment adjustments
    if depositional_env == 'fluvial':
        # Fluvial deposits often show channel patterns
        field = field * (1 + 0.2 * np.sin(field * 2 * np.pi))
    elif depositional_env == 'aeolian':
        # Aeolian deposits often show dune patterns
        field = field * (1 + 0.15 * np.cos(field * 3 * np.pi))
    
    return np.clip(field, 0, 1)

def _combine_facies_grids(cat_field: np.ndarray, facies_data: List[Dict], facies_grids: List[np.ndarray]) -> np.ndarray:
    """Combine facies grids with improved blending and transition zones."""
    combined_grid = np.zeros_like(cat_field)
    proportions = [facies.get('proportion', 0.0) for facies in facies_data]
    cum_proportions = np.cumsum(proportions)
    
    for i, (facies, grid) in enumerate(zip(facies_data, facies_grids)):
        if i == 0:
            mask = cat_field <= cum_proportions[i]
        else:
            mask = (cat_field > cum_proportions[i-1]) & (cat_field <= cum_proportions[i])
        
        # Apply facies with potential transition zones
        transition_width = facies.get('transition_width', 0.05)
        if transition_width > 0 and i > 0:
            # Create smooth transition between facies
            transition_mask = _create_transition_mask(cat_field, cum_proportions[i-1],
                                                    cum_proportions[i], transition_width)
            blended_value = _blend_facies(combined_grid, grid, transition_mask)
            combined_grid[transition_mask] = blended_value[transition_mask]
        
        combined_grid[mask] = grid[mask]
    
    return combined_grid

def _create_transition_mask(cat_field: np.ndarray, lower_bound: float,
                           upper_bound: float, transition_width: float) -> np.ndarray:
    """Create mask for transition zone between facies."""
    transition_lower = upper_bound - transition_width
    transition_upper = upper_bound + transition_width
    return (cat_field >= transition_lower) & (cat_field <= transition_upper)

def _blend_facies(grid1: np.ndarray, grid2: np.ndarray, transition_mask: np.ndarray) -> np.ndarray:
    """Blend two facies grids in transition zone."""
    # Linear blending in transition zone
    blend_factor = np.linspace(0, 1, np.sum(transition_mask))
    blended = grid1[transition_mask] * (1 - blend_factor) + grid2[transition_mask] * blend_factor
    result = grid1.copy()
    result[transition_mask] = blended
    return result

def _normalize_field(field: np.ndarray) -> np.ndarray:
    """Normalize field to [0, 1] range for porosity representation."""
    field_min = np.min(field)
    field_max = np.max(field)

    if field_max - field_min > 1e-10:
        normalized = (field - field_min) / (field_max - field_min)
    else:
        normalized = np.zeros_like(field)

    return np.clip(normalized, 0, 1)

def create_facies_map(shape: Tuple[int, int], facies_data: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """
    Create a facies map showing which facies is at each grid location.

    Args:
        shape: Grid dimensions (nx, ny)
        facies_data: List of facies definitions with proportions

    Returns:
        Tuple of (facies_map, facies_info) where:
        - facies_map: 2D array with facies indices (0, 1, 2, ...) for each cell
        - facies_info: Dictionary mapping facies indices to facies names and colors
    """
    try:
        # Validate facies proportions
        total_proportion = sum(facies.get('proportion', 0) for facies in facies_data)
        if not np.isclose(total_proportion, 1.0, atol=1e-6):
            raise ValueError(f"Facies proportions must sum to 1.0 (current sum: {total_proportion})")

        # Create categorical field for facies distribution
        # Use realistic range for proper spatial correlation
        realistic_range = calculate_realistic_range(shape)
        cat_model = gs.Gaussian(dim=2, var=1.0, len_scale=realistic_range)
        srf_cat = gs.SRF(cat_model, seed=20170520)

        # Create coordinate arrays for gstools (required API)
        # srf.structured() expects a list of coordinate arrays, not a shape tuple
        x = np.arange(shape[0])
        y = np.arange(shape[1])

        # Generate the categorical field using correct gstools API
        cat_field = srf_cat.structured([x, y])
        logger.info(f"Facies map: Using categorical field with range {realistic_range:.1f}")

        # Ensure the categorical field has the correct shape
        if cat_field.shape != shape:
            if cat_field.shape == (shape[1], shape[0]):
                cat_field = cat_field.T
            else:
                cat_field = cat_field.reshape(shape)

        # Create facies map by assigning facies based on categorical field
        facies_map = np.zeros(shape, dtype=int)
        proportions = [facies.get('proportion', 0.0) for facies in facies_data]
        cum_proportions = np.cumsum(proportions)

        for i in range(len(facies_data)):
            if i == 0:
                mask = cat_field <= cum_proportions[i]
            else:
                mask = (cat_field > cum_proportions[i-1]) & (cat_field <= cum_proportions[i])
            facies_map[mask] = i

        # Create facies info with colors for visualization
        facies_colors = [
            '#FF6B6B',  # Red for facies 0
            '#4ECDC4',  # Teal for facies 1
            '#45B7D1',  # Blue for facies 2
            '#FFA07A',  # Light salmon for facies 3
            '#98D8C8',  # Mint for facies 4
            '#F7DC6F',  # Yellow for facies 5
            '#BB8FCE',  # Purple for facies 6
            '#85C1E2',  # Light blue for facies 7
        ]

        facies_info = {}
        for i, facies in enumerate(facies_data):
            facies_info[i] = {
                'name': facies.get('name', f'Facies_{i+1}'),
                'rock_type': facies.get('rock_type', 'unknown'),
                'proportion': facies.get('proportion', 0.0),
                'color': facies_colors[i % len(facies_colors)]
            }

        logger.info(f"Created facies map with {len(facies_data)} facies: "
                   f", ".join([f"{info['name']} ({info['proportion']:.0%})"
                             for info in facies_info.values()]))

        return facies_map, facies_info

    except Exception as e:
        logger.error(f"Error creating facies map: {e}")
        # Fallback: create uniform facies map
        facies_map = np.zeros(shape, dtype=int)
        facies_info = {
            0: {
                'name': 'Default_Facies',
                'rock_type': 'sandstone',
                'proportion': 1.0,
                'color': '#FF6B6B'
            }
        }
        return facies_map, facies_info

def create_facies_based_grid_with_map(shape: Tuple[int, int], facies_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Enhanced facies-based grid creation that returns both the property grid and facies map.

    Args:
        shape: Grid dimensions (nx, ny)
        facies_data: List of facies definitions with geological parameters

    Returns:
        Tuple of (property_grid, facies_map, facies_info) where:
        - property_grid: Combined porosity/property grid
        - facies_map: 2D array showing which facies is at each cell
        - facies_info: Dictionary with facies names and colors
    """
    try:
        # Validate facies proportions
        total_proportion = sum(facies.get('proportion', 0) for facies in facies_data)
        if not np.isclose(total_proportion, 1.0, atol=1e-6):
            raise ValueError(f"Facies proportions must sum to 1.0 (current sum: {total_proportion})")

        # Validate shape parameter
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError(f"Shape must be a tuple of 2 integers, got: {shape}")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Shape dimensions must be positive, got: {shape}")
        if shape[0] < 3 or shape[1] < 3:
            logger.warning(f"Shape dimensions {shape} are very small. Geostatistical modeling requires minimum 3x3 grid for meaningful results.")

        # Create categorical field for facies distribution
        # Use realistic range for proper spatial correlation
        realistic_range = calculate_realistic_range(shape)

        # Validate the calculated range
        if realistic_range <= 0 or not np.isfinite(realistic_range):
            logger.error(f"Calculated invalid range {realistic_range}. Using fallback range.")
            realistic_range = max(shape) * 0.5

        cat_model = gs.Gaussian(dim=2, var=1.0, len_scale=realistic_range)
        srf_cat = gs.SRF(cat_model, seed=20170520)

        # Create coordinate arrays for gstools (required API)
        # srf.structured() expects a list of coordinate arrays, not a shape tuple
        x = np.arange(shape[0])
        y = np.arange(shape[1])

        # Generate the categorical field using correct gstools API
        cat_field = srf_cat.structured([x, y])
        logger.info(f"create_facies_based_grid_with_map: Using categorical field with range {realistic_range:.1f}")

        # Validate that field is a proper array
        if not isinstance(cat_field, np.ndarray):
            logger.error(f"gstools returned non-array type for categorical field in create_facies_based_grid_with_map: {type(cat_field)}. Using random field.")
            cat_field = np.random.rand(*shape)
        elif cat_field.size == 1:
            logger.error(f"gstools returned scalar categorical field. This may be caused by invalid shape {shape} or gstools version incompatibility. Using random field.")
            cat_field = np.random.rand(*shape)
        elif cat_field.shape != shape and cat_field.shape != (shape[1], shape[0]):
            logger.warning(f"gstools returned field with unexpected shape {cat_field.shape}, expected {shape}. Attempting to reshape...")
            # Try to reshape if the total size matches
            if cat_field.size == shape[0] * shape[1]:
                cat_field = cat_field.reshape(shape)
            else:
                logger.error(f"Cannot reshape field from {cat_field.shape} to {shape}. Using random field.")
                cat_field = np.random.rand(*shape)

        # Ensure the categorical field has the correct shape
        if cat_field.shape != shape:
            try:
                # If shape is inverted, transpose it
                if cat_field.shape == (shape[1], shape[0]):
                    cat_field = cat_field.T
                elif cat_field.size == shape[0] * shape[1]:
                    # Flatten and reshape
                    cat_field = cat_field.flatten().reshape(shape)
                else:
                    logger.error(f"Categorical field size {cat_field.size} doesn't match target {shape[0] * shape[1]}. Using random field.")
                    cat_field = np.random.rand(*shape)
            except ValueError as reshape_error:
                logger.error(f"Error reshaping categorical field: {reshape_error}. Using random field.")
                cat_field = np.random.rand(*shape)

        # Create facies map
        facies_map = np.zeros(shape, dtype=int)
        proportions = [facies.get('proportion', 0.0) for facies in facies_data]
        cum_proportions = np.cumsum(proportions)

        for i in range(len(facies_data)):
            if i == 0:
                mask = cat_field <= cum_proportions[i]
            else:
                mask = (cat_field > cum_proportions[i-1]) & (cat_field <= cum_proportions[i])
            facies_map[mask] = i

        # Create grids for each facies
        facies_grids = []
        for i, facies in enumerate(facies_data):
            facies_params = facies.get('geostatistical_params', {})
            facies_name = facies.get('name', f'Facies_{i+1}')
            rock_type = facies.get('rock_type', 'sandstone')
            depositional_env = facies.get('depositional_environment', 'fluvial')

            logger.info(f"Creating grid for {facies_name} ({rock_type}, {depositional_env})")

            # Create geostatistical field for this facies
            field = create_geostatistical_grid(shape, facies_params)

            # Apply facies-specific characteristics
            field = _apply_facies_characteristics(field, rock_type, depositional_env)

            facies_grids.append(field)

        # Combine grids based on facies proportions
        combined_grid = _combine_facies_grids(cat_field, facies_data, facies_grids)

        # Create facies info with colors
        facies_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F1948A', '#82E0AA'
        ]

        facies_info = {}
        for i, facies in enumerate(facies_data):
            facies_info[i] = {
                'name': facies.get('name', f'Facies_{i+1}'),
                'rock_type': facies.get('rock_type', 'unknown'),
                'proportion': facies.get('proportion', 0.0),
                'color': facies_colors[i % len(facies_colors)]
            }

        return combined_grid, facies_map, facies_info

    except Exception as e:
        logger.error(f"Error creating facies-based grid with map: {e}")
        raise

def calculate_variogram(field: np.ndarray, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate experimental variogram for a field."""
    if max_lag is None:
        max_lag = min(field.shape) // 2
    
    # Simple variogram calculation (can be enhanced with gstools)
    lags = np.arange(1, max_lag + 1)
    gamma = np.zeros_like(lags, dtype=float)
    
    for i, lag in enumerate(lags):
        # Calculate variance for this lag (simplified)
        diff_sq = (field[lag:, :] - field[:-lag, :])**2
        gamma[i] = np.mean(diff_sq) / 2.0
    
    return lags, gamma
