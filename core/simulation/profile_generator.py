"""
Legacy profile generator module (Deprecated).
Redirects to deprecated.core.simulation.profile_generator.
"""

import warnings
from deprecated.core.simulation.profile_generator import *

warnings.warn(
    "core.simulation.profile_generator is deprecated. Use core.engine_surrogate.profile_generator_fast instead.",
    DeprecationWarning,
    stacklevel=2,
)
