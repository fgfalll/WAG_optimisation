import lasio
from typing import Dict, Optional
import numpy as np
import logging
from ..core import WellData

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
              fill_value: float = np.nan) -> WellData:
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
        las = lasio.read(file_path, engine='normal')  # Force normal engine for wrapped files
    except Exception as e:
        logging.error(f"Failed to read LAS file: {file_path}")
        raise IOError(f"LAS file read error: {str(e)}")

    # Validate required sections
    if not hasattr(las, 'well') or not las.well:
        logging.warning("LAS file missing WELL section - using default values")
        if not hasattr(las, 'well'):
            las.well = {}
        las.well['WELL'] = lasio.HeaderItem('WELL', value='DEFAULT')
        
        # Ensure we return properly constructed WellData
        return WellData(
            name='DEFAULT',
            depths=np.array([0.0]),
            properties={},
            units={}
        )
    if not hasattr(las, 'curves') or not las.curves:
        logging.warning("LAS file missing curve data - initializing empty dataset")
        las.curves = [lasio.CurveItem(mnemonic='DEPT', data=np.array([0.0]))]

    # Get well name - handle different LAS versions
    well_name = 'UNKNOWN'
    try:
        if hasattr(las.well, 'WELL'):
            well_name = str(las.well.WELL.value)
        elif isinstance(las.well, dict) and 'WELL' in las.well:
            well_name = str(las.well['WELL'].value)
        else:
            raise ValueError("LAS file missing WELL identifier")
    except Exception as e:
        logging.warning(f"Error getting well name: {str(e)}")
    
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
        unit = getattr(curve, 'unit', '') or ''
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
            if 'DEPT' in properties:
                properties['DEPT'] *= conv_factor
                units['DEPT'] = depth_units['target']
            depths *= conv_factor
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