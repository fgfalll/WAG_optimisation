"""
Core modules for CO2 EOR Simulation
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'PhysicsEngine':
        from .Phys_engine_full import PhysicsEngine
        return PhysicsEngine
    elif name == 'GeologyEngine':
        from .geology import GeologyEngine
        return GeologyEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'PhysicsEngine',
    'GeologyEngine',
]
