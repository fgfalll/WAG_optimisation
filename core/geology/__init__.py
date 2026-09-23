"""
Geology module for CO2 EOR Simulation.
Contains geostatistical modeling routines.
"""

from .geostatistical_modeling import create_geostatistical_grid

# Backward-compatible lazy import for deprecated GeologyEngine
def __getattr__(name):
    if name == "GeologyEngine":
        from deprecated.core.geology.geology_engine import GeologyEngine
        return GeologyEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "create_geostatistical_grid",
]