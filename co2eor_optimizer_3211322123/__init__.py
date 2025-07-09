# This file makes 'co2eor_optimizer' a package.

__version__ = "0.1.0"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
     "OptimizationEngine", "ReservoirData", # from core
     "SensitivityAnalyzer",                # from analysis
     "config_manager",                     # from root
     "__version__"
]