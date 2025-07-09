import numpy as np
from typing import List, Dict, Optional
from co2eor_optimizer.core.data_models import ReservoirData, WellData

def validate_reservoir_data(reservoir_data: ReservoirData) -> List[str]:
    """
    Validates the ReservoirData object with context-aware checks for grid properties.
    """
    errors = []
    grid = reservoir_data.grid
    runspec = reservoir_data.runspec

    if not grid:
        errors.append("Grid properties dictionary is empty.")
        return errors

    dims = runspec.get('DIMENSIONS')
    if not dims or len(dims) != 3:
        errors.append(f"Invalid or missing DIMENSIONS in runspec: {dims}. Cannot validate grid shapes.")
        return errors
    
    nx, ny, nz = dims
    total_cells = nx * ny * nz

    # Define expected shapes for different keyword types
    per_cell_props = {'PORO', 'PERMX', 'PERMY', 'PERMZ', 'NTG', 'SWAT', 'PRESSURE', 'ACTNUM', 'TOPS', 'DZ'}
    per_x_props = {'DX', 'DXV'}
    per_y_props = {'DY', 'DYV'}
    per_z_props = {'DZ', 'DZV'} 

    for prop_name, value in grid.items():
        if not isinstance(value, np.ndarray):
            continue

        array = value
        prop_name_upper = prop_name.upper()
        
        # --- FIX: Corrected validation logic and error messages ---
        if prop_name_upper in per_cell_props:
            if array.size != total_cells:
                errors.append(f"'{prop_name}' (per-cell): size is {array.size}, expected {total_cells}.")
        elif prop_name_upper in per_x_props:
            if array.size != nx:
                errors.append(f"'{prop_name}' (per-X): size is {array.size}, expected {nx}.")
        elif prop_name_upper in per_y_props:
            if array.size != ny:
                errors.append(f"'{prop_name}' (per-Y): size is {array.size}, expected {ny}.")
        elif prop_name_upper in per_z_props:
            if array.size != nz and array.size != total_cells:
                 errors.append(f"'{prop_name}' (per-Z): size is {array.size}, expected {nz} or {total_cells}.")

    for table_name, table_data in reservoir_data.pvt_tables.items():
        if not isinstance(table_data, np.ndarray):
            errors.append(f"PVT table '{table_name}' must be a numpy array.")
            continue
        if table_data.ndim != 2:
            errors.append(f"PVT table '{table_name}' must be 2-dimensional.")

    return errors

def validate_well_data(well_data: WellData) -> List[str]:
    """Validate WellData object against schema and business rules"""
    errors = []
    
    if not well_data.validate():
        errors.append("Property arrays must match depth array length")
    
    if len(well_data.depths) > 1:
        depth_diffs = np.diff(well_data.depths)
        if np.any(depth_diffs <= 0):
            errors.append("Depth array must be strictly increasing")
    
    for prop_name, values in well_data.properties.items():
        if prop_name in ['GR', 'RT'] and np.any(values < 0):
            errors.append(f"Property '{prop_name}' contains negative values")
        elif prop_name == 'PORO' and (np.any(values < 0) or np.any(values > 1)):
            errors.append(f"Porosity values must be between 0-1, found {values.min()} to {values.max()}")
        elif prop_name == 'SW' and (np.any(values < 0) or np.any(values > 1)):
            errors.append(f"Water saturation values must be between 0-1, found {values.min()} to {values.max()}")
    
    for prop_name, unit in well_data.units.items():
        if not unit:
            errors.append(f"Unit missing for property '{prop_name}'")
        elif prop_name in ['GR', 'NPHI'] and 'GAPI' not in unit.upper():
            errors.append(f"Unexpected unit '{unit}' for {prop_name} - expected GAPI")
        elif prop_name == 'PORO' and 'V/V' not in unit.upper():
            errors.append(f"Unexpected unit '{unit}' for porosity - expected v/v")
    
    return errors