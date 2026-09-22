"""
Simulation tools for CO2 EOR
"""

from .profile_generator import ProfileGenerator

try:
    from core.engine_surrogate.profile_generator_fast import FastProfileGenerator
except ImportError:
    FastProfileGenerator = None

__all__ = [
    'ProfileGenerator',
    'FastProfileGenerator',
]

