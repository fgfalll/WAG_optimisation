"""
Legacy recovery models module (Deprecated).
Redirects to deprecated.core.simulation.recovery_models.
"""

import warnings
from deprecated.core.simulation.recovery_models import *

warnings.warn(
    "core.simulation.recovery_models is deprecated. Use core.engine_surrogate.analytical_models instead.",
    DeprecationWarning,
    stacklevel=2,
)
