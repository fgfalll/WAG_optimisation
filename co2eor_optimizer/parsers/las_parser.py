import lasio
from typing import Dict, Optional, Any
import numpy as np
import logging
from ..core import WellData
from dataclasses import dataclass

@dataclass
class MissingWellNameError(Exception):
    """Exception raised when LAS file is missing WELL section"""
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
    Parse LAS file into WellData object with enhanced validation and handling.
    
    Args:
        file_path: Path to LAS file
        depth_unit: Target unit for depth (None keeps original)
        fill_value: Value to use for missing/invalid data points
        
    Returns:
        WellData object with validated and cleaned data
        
    Raises:
        ValueError: For invalid LAS files or unit conversions
        IOError: For file reading issues
    """
    try:
        las = lasio.read(file_path, engine=engine)
        logging.debug(f"Detected sections: {list(las.sections.keys())}")
    except Exception as e:
        logging.error(f"Failed to read LAS file: {file_path}")
        raise IOError(f"LAS file read error: {str(e)}")

    # Handle WELL section
    well_name = 'UNKNOWN'
    logging.debug(f"las.well exists: {hasattr(las, 'well')}, las.well value: {getattr(las, 'well', None)}")
    if not hasattr(las, 'well') or not las.well or not hasattr(las.well, 'WELL') or not las.well.WELL.value:
        if well_name_override:
            well_name = well_name_override
            if not hasattr(las, 'well'):
                las.well = {}
            las.well['WELL'] = lasio.HeaderItem('WELL', value=well_name)
        else:
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
    if not hasattr(las, 'curves') or not las.curves:
        logging.warning("LAS file missing curve data - initializing empty dataset")
        las.curves = [lasio.CurveItem(mnemonic='DEPT', data=np.array([0.0]))]

    # Get well name if not already set to DEFAULT
    if well_name == 'UNKNOWN':
        try:
            if hasattr(las.well, 'WELL'):
                well_name = str(las.well.WELL.value)
            elif isinstance(las.well, dict) and 'WELL' in las.well:
                well_name = str(las.well['WELL'].value)
            else:
                logging.warning("LAS file missing WELL identifier - using DEFAULT")
                well_name = 'DEFAULT'
        except Exception as e:
            logging.warning(f"Error getting well name: {str(e)} - using DEFAULT")
            well_name = 'DEFAULT'
    
    properties = {}
    units = {}
    depth_units = {}
    
    for curve in las.curves:
        # Handle missing data (-999 and NaN)
        # Handle various missing data indicators and invalid types
        clean_data = []
        for val in curve.data:
            try:
                num = float(val)
                if np.isnan(num) or num <= -999.0:
                    clean_data.append(fill_value)
                else:
                    clean_data.append(num)
            except (ValueError, TypeError):
                clean_data.append(fill_value)
        data = np.array(clean_data, dtype=np.float64)
        properties[curve.mnemonic] = data
        # Get unit from curve or use empty string if not available
        unit = getattr(curve, 'unit', '') or ''
        # Special handling for DEPT curve to ensure unit is set
        if curve.mnemonic == 'DEPT':
            if not unit and depth_unit:
                unit = depth_unit
        units[curve.mnemonic] = unit
        
        # Track original depth units for conversion
        if curve.mnemonic == 'DEPT':
            depth_units['original'] = unit
            depth_units['target'] = depth_unit if depth_unit else unit

    # Apply unit conversion if needed
    depths = las.index.copy()
    if depth_unit and depth_units.get('original') and depth_units['original'] != depth_units['target']:
        try:
            conv_factor = UNIT_CONVERSIONS[depth_units['original']][depth_units['target']]
            # Convert both DEPT curve and depths array
            if 'DEPT' in properties:
                properties['DEPT'] = np.array(properties['DEPT'] * conv_factor, dtype=np.float64)
            # Always update the depth unit to target unit after conversion
            units['DEPT'] = depth_units['target']
            # Convert string depths to float before multiplication
            if isinstance(depths, np.ndarray) and depths.dtype.kind in ['U', 'S']:
                try:
                    depths = np.array([float(x.strip().rstrip(',')) for x in depths], dtype=np.float64)
                except ValueError as e:
                    raise ValueError(f"Invalid depth value format: {str(e)}")
            # Apply conversion to depths array
            depths = np.array(depths * conv_factor, dtype=np.float64)
            # Ensure the WellData object gets the converted depths
            properties['DEPT'] = depths.copy()
            # Update DEPT curve data directly
            dept_curve = next(c for c in las.curves if c.mnemonic == 'DEPT')
            dept_curve.data = depths
            logging.debug(f"Converted depths from {depth_units['original']} to {depth_units['target']}")
        except KeyError:
            logging.warning(f"Unsupported unit conversion: {depth_units['original']} to {depth_units['target']}")

    well_data = WellData(
        name=well_name,
        depths=depths,
        properties=properties,
        units=units
    )
    
    if not well_data.validate():
        logging.warning("Data validation failed - inconsistent array lengths")
        
    return well_data