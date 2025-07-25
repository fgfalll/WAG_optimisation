import numpy as np
import logging
from pathlib import Path
from typing import Union

from core.data_models import ReservoirData

logger = logging.getLogger(__name__)

def _format_array_for_grdecl(
    arr: np.ndarray, 
    items_per_line: int = 8, 
    is_int_property: bool = False, 
    nan_rep: str = "NAN"
) -> str:
    """
    Formats a NumPy array into a string suitable for GRDECL output.

    Args:
        arr: NumPy array to format.
        items_per_line: Maximum number of values per line in the output string.
        is_int_property: If True, values are formatted as integers. Otherwise, as floats.
        nan_rep: The string representation for NaN (Not a Number) values.

    Returns:
        A string with formatted values, newline-separated.
    """
    if not isinstance(arr, np.ndarray):
        logger.error(f"Input to _format_array_for_grdecl must be a numpy array, got {type(arr)}")
        return ""
    if arr.size == 0:
        return ""

    lines = []
    temp_list = []
    
    # Iterate through the flattened array to format each value.
    for i_val, val in enumerate(arr.flatten()):
        if np.isnan(val):
            # Use the specified NaN representation.
            formatted_val = nan_rep
        elif is_int_property:
            try:
                # Round before converting to integer to handle floating point inaccuracies.
                formatted_val = str(int(round(val)))
            except ValueError:
                logger.warning(f"Could not convert {val} to int, using 0.")
                formatted_val = "0"
        else:
            # Smart float formatting to avoid scientific notation for whole numbers.
            if abs(val) > 1e-5 and abs(val) < 1e7 and val == round(val):
                 formatted_val = str(int(round(val)))
            else:
                 # Use general format with up to 7 significant digits.
                 formatted_val = f"{val:.7G}"

        temp_list.append(formatted_val)
        
        # Create a new line when the number of items is reached or at the end of the array.
        if (i_val + 1) % items_per_line == 0 or (i_val + 1) == arr.size:
            lines.append("  " + " ".join(temp_list)) # Indent data lines
            temp_list = []
            
    return "\n".join(lines)

def write_grdecl(
    reservoir_data: ReservoirData, 
    output_filepath: Union[str, Path], 
    nan_representation: str = "NAN"
) -> None:
    """
    Writes reservoir grid data to a GRDECL file.

    Args:
        reservoir_data: ReservoirData object containing grid information.
        output_filepath: Path to save the GRDECL file.
        nan_representation: The string to use for NaN values (e.g., 'NAN', '1*').
    """
    output_filepath = Path(output_filepath)
    if not reservoir_data.grid:
        logger.error("ReservoirData contains no grid data to write.")
        raise ValueError("No grid data available in ReservoirData object.")

    grid_data = reservoir_data.grid
    content_lines = ["-- GRDECL file generated by updated Python script", ""]

    # --- Grid dimensions ---
    nx, ny, nz = (0, 0, 0)
    if 'DIMENS' in grid_data and isinstance(grid_data['DIMENS'], (tuple, list, np.ndarray)) and len(grid_data['DIMENS']) == 3:
        nx, ny, nz = map(int, grid_data['DIMENS'])
        content_lines.append("DIMENS")
        content_lines.append(f"  {nx} {ny} {nz} /")
    elif 'SPECGRID_DIMS' in grid_data and isinstance(grid_data['SPECGRID_DIMS'], (tuple, list, np.ndarray)) and len(grid_data['SPECGRID_DIMS']) == 3:
        nx, ny, nz = map(int, grid_data['SPECGRID_DIMS'])
        lgr_name = grid_data.get('SPECGRID_LGR', 'ROOT') 
        pillar_flag = grid_data.get('SPECGRID_PILLAR_FLAG', 1)
        content_lines.append("SPECGRID")
        content_lines.append(f"  {nx} {ny} {nz} '{lgr_name}' {pillar_flag} /")
    else:
        logger.error("Grid dimensions (DIMENS or SPECGRID_DIMS) not found in ReservoirData.")
        raise ValueError("Grid dimensions not found or invalid in ReservoirData.")
    content_lines.append("")

    # --- Keyword and Data Writing ---
    
    # Configuration for items per line based on keyword.
    ITEMS_PER_LINE_MAP = {
        'COORD': 6,
        'ZCORN': 8,
        'ACTNUM': 20,
    }
    DEFAULT_ITEMS_PER_LINE = 10

    # Define the standard order for geometry keywords.
    grdecl_geometry_keywords = ['COORD', 'ZCORN', 'ACTNUM']
    
    # Find all other property keywords in the data that are numpy arrays.
    excluded_keys = ['DIMENS', 'SPECGRID_DIMS', 'SPECGRID_LGR', 'SPECGRID_PILLAR_FLAG'] + grdecl_geometry_keywords
    property_keywords = sorted([
        k for k, v in grid_data.items() 
        if k.upper() not in map(str.upper, excluded_keys) and isinstance(v, np.ndarray)
    ])
    
    # Combine geometry and property keywords for writing.
    all_keywords_to_write = grdecl_geometry_keywords + property_keywords

    for keyword in all_keywords_to_write:
        if keyword in grid_data:
            data_array = grid_data[keyword]
            if isinstance(data_array, np.ndarray):
                content_lines.append(keyword.upper())
                
                upper_keyword = keyword.upper()
                is_int_data = (upper_keyword == 'ACTNUM') # Extend this tuple for other integer properties
                
                # Use the dictionary map to get items_per_line.
                items_per_line = ITEMS_PER_LINE_MAP.get(upper_keyword, DEFAULT_ITEMS_PER_LINE)
                
                # Format the array data into a string.
                formatted_data = _format_array_for_grdecl(
                    data_array, 
                    items_per_line=items_per_line, 
                    is_int_property=is_int_data,
                    nan_rep=nan_representation
                )

                if formatted_data:
                    content_lines.append(formatted_data)
                
                content_lines.append("/")
                content_lines.append("") 
            else:
                logger.warning(f"Skipping keyword {keyword} as its data is not a NumPy array (type: {type(data_array)}).")
    
    # --- File Output ---
    try:
        # Ensure the output directory exists before writing.
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, 'w') as f:
            f.write("\n".join(content_lines))
        logger.info(f"Successfully wrote GRDECL file to {output_filepath}")
    except IOError as e:
        logger.error(f"Failed to write GRDECL file {output_filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing GRDECL file {output_filepath}: {e}")
        raise