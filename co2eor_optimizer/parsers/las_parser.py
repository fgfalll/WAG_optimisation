import lasio
from typing import Dict, Optional, Any
import numpy as np
import logging
from ..core import WellData
from dataclasses import dataclass

@dataclass
class MissingWellNameError(Exception):
    """Exception raised when LAS file is missing WELL section."""
    file_path: str
    available_sections: Dict[str, Any]
    message: str = "LAS file missing WELL section - please provide a well name"


# Standard unit conversion factors (metric to field and vice versa)
UNIT_CONVERSIONS = {
    'M': {'FT': 3.28084},
    'FT': {'M': 0.3048},
    'G/CC': {'LB/FT3': 62.428},
    'LB/FT3': {'G/CC': 0.0160185},
    'CP': {'CP': 1.0},  # No conversion
    'D': {'MD': 1000.0},
    'MD': {'D': 0.001}
}

def parse_las(file_path: str,
              depth_unit: Optional[str] = None,
              fill_value: float = np.nan,
              well_name_override: Optional[str] = None,
              engine: str = 'normal') -> WellData:
    """
    Parses a LAS file into a WellData object, with options for unit conversion
    and handling of missing data.

    Args:
        file_path (str): The path to the LAS file.
        depth_unit (Optional[str]): The target unit for the depth curve (e.g., 'M' or 'FT').
                                     If provided, the depth curve will be converted.
        fill_value (float): The value to use for missing or invalid data points.
                            Defaults to np.nan.
        well_name_override (Optional[str]): A name to assign to the well, overriding
                                             any name found in the LAS file.
        engine (str): The engine for lasio to use ('normal' or 'python').

    Returns:
        WellData: An object containing the parsed well data.

    Raises:
        IOError: If the LAS file cannot be read.
        MissingWellNameError: If a well name is not found in the file and not provided
                              via well_name_override.
        ValueError: If depth values are non-numeric and cannot be converted.
    """
    try:
        # Attempt to read the LAS file using lasio
        las = lasio.read(file_path, engine=engine)
        logging.debug(f"Successfully read LAS file: {file_path}")
        logging.debug(f"Detected sections: {list(las.sections.keys())}")
    except lasio.LASHeaderError as e:
        # Catch specific lasio header errors for more detailed logging
        logging.error(f"Header parsing error in LAS file: {file_path}. Details: {e}")
        raise IOError(f"LAS file header error: {str(e)}")
    except Exception as e:
        # Catch any other exception during file reading
        logging.error(f"Failed to read LAS file: {file_path}")
        raise IOError(f"LAS file read error: {str(e)}")

    # --- Well Name Handling (Simplified Logic) ---
    well_name = None
    if well_name_override:
        well_name = well_name_override
        # Ensure the las object has a well section for consistency if we override
        if not hasattr(las, 'well'):
            las.well = {}
        las.well['WELL'] = lasio.HeaderItem('WELL', value=well_name)
        logging.debug(f"Using well name from override: '{well_name}'")
    else:
        # Try to extract the well name from the file's header
        try:
            if hasattr(las, 'well') and las.well and hasattr(las.well, 'WELL') and las.well.WELL.value:
                well_name = str(las.well.WELL.value)
                logging.debug(f"Found well name in LAS header: '{well_name}'")
        except Exception as e:
            logging.warning(f"Could not extract well name from LAS header: {e}")

    # If no well name could be determined, raise a specific error
    if not well_name:
        available_sections = {
            'version': getattr(las, 'version', None),
            'well': getattr(las, 'well', None),
            'curves': getattr(las, 'curves', None),
            'other': getattr(las, 'other', None)
        }
        raise MissingWellNameError(
            file_path=file_path,
            available_sections=available_sections
        )

    # Handle cases where there is no curve data
    if not hasattr(las, 'curves') or not las.curves:
        logging.warning("LAS file missing curve data - initializing empty dataset.")
        # Create a minimal DEPT curve to avoid errors downstream
        las.curves = [lasio.CurveItem(mnemonic='DEPT', unit=depth_unit or '', data=np.array([], dtype=np.float64))]

    # --- Curve Data Processing ---
    properties = {}
    units = {}
    depth_units = {}
    
    for curve in las.curves:
        # Robustly clean curve data: handle NaNs, null values, and non-numeric types
        clean_data = []
        for val in curve.data:
            try:
                num = float(val)
                if np.isnan(num) or num <= -999.0:
                    clean_data.append(fill_value)
                else:
                    clean_data.append(num)
            except (ValueError, TypeError):
                # This handles strings or other non-convertible types
                clean_data.append(fill_value)
        
        data = np.array(clean_data, dtype=np.float64)
        properties[curve.mnemonic] = data
        
        # Get unit from curve, default to empty string if not available
        unit = getattr(curve, 'unit', '') or ''
        units[curve.mnemonic] = unit
        
        # Special handling for DEPT to track units for conversion
        if curve.mnemonic == 'DEPT':
            depth_units['original'] = unit
            depth_units['target'] = depth_unit if depth_unit else unit

    # --- Unit Conversion for Depth ---
    depths = las.index.copy()
    if depth_unit and depth_units.get('original') and depth_units['original'] != depth_units['target']:
        try:
            conv_factor = UNIT_CONVERSIONS[depth_units['original']][depth_units['target']]
            
            # Convert string-based depth arrays to float before multiplication
            if isinstance(depths, np.ndarray) and depths.dtype.kind in ['U', 'S']:
                try:
                    # Strip whitespace and trailing commas that can appear in some files
                    depths = np.array([float(str(x).strip().rstrip(',')) for x in depths], dtype=np.float64)
                except ValueError as e:
                    raise ValueError(f"Invalid depth value format while converting to float: {str(e)}")
            
            # Apply conversion to the depths array
            depths = np.array(depths * conv_factor, dtype=np.float64)

            # Ensure the converted depths are updated in the properties dictionary
            if 'DEPT' in properties:
                properties['DEPT'] = depths.copy()
            
            # Update the unit string to reflect the conversion
            units['DEPT'] = depth_units['target']
            
            logging.debug(f"Converted depths from {depth_units['original']} to {depth_units['target']}")
        except KeyError:
            logging.warning(f"Unsupported unit conversion: {depth_units['original']} to {depth_units['target']}. Skipping conversion.")

    # --- Final Validation and Object Creation ---
    well_data = WellData(
        name=well_name,
        depths=depths,
        properties=properties,
        units=units
    )
    
    # Check for consistent array lengths in the final object
    if not well_data.validate():
        logging.warning("Data validation failed - inconsistent array lengths found in the final WellData object.")
        
    return well_data