"""
Geology module for CO2 EOR Simulation.
Contains geostatistical modeling routines.
"""

from .geostatistical_modeling import create_geostatistical_grid

__all__ = [
    "create_geostatistical_grid",
]