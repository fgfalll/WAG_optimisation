import lasio
from typing import Dict
import numpy as np
from ..core import WellData

def parse_las(file_path: str) -> WellData:
    """Parse LAS file into WellData object"""
    las = lasio.read(file_path)
    
    properties = {}
    units = {}
    for curve in las.curves:
        properties[curve.mnemonic] = curve.data
        units[curve.mnemonic] = curve.unit or ''

    return WellData(
        name=las.well['WELL'].value,
        depths=las.index,
        properties=properties,
        units=units
    )