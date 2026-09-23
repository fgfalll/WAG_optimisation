"""
Simulator export module (Deprecated).
Redirects to utils.cmg_exporter.
"""

import warnings
from utils.cmg_exporter import SimulatorExporter, SimulatorExportConfig

warnings.warn(
    "core.simulation.simulator_exporter is deprecated. Use utils.cmg_exporter instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SimulatorExporter", "SimulatorExportConfig"]
