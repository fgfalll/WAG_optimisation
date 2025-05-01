"""CO2 EOR Optimization Package"""
__version__ = "0.1.0"

from .core import (
    ReservoirData,
    WellData,
    PVTProperties,
    EORParameters
)

__all__ = [
    'ReservoirData',
    'WellData',
    'PVTProperties',
    'EORParameters'
]