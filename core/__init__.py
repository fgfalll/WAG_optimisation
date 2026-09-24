"""
Core modules for CO2 EOR Simulation
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'GeologyEngine':
        from .geology import GeologyEngine
        return GeologyEngine
    elif name == 'OptimizationEngine':
        from .optimisation_engine import OptimizationEngine
        return OptimizationEngine
    elif name == 'SurrogateEngineWrapper':
        from .engine_surrogate.surrogate_engine import SurrogateEngineWrapper
        return SurrogateEngineWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'GeologyEngine',
    'OptimizationEngine',
    'SurrogateEngineWrapper',
]
