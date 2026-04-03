"""
EOR Reservoir Simulation Engine
===============================

A fast and reliable parameter estimation engine for EOR reservoir simulation
including CO₂ injection. This engine provides guidance for optimization algorithms
by simulating EOR scenarios and returning comprehensive performance metrics.

Main Components:
- Reservoir simulation core with multiphase flow
- CO₂-EOR physics and property calculations
- Parameter estimation using Ensemble Kalman Filter
- Storage efficiency and mass balance calculations
- Optimization interface for seamless integration

Author: Reservoir Engineering Expert
"""

from core.engine_simple.reservoir_engine import ReservoirSimulationEngine
from core.engine_simple.storage_efficiency import StorageEfficiencyCalculator
from core.engine_simple.parameter_estimation import EnsembleKalmanFilter
from core.engine_simple.optimization_interface import OptimizationInterface
from core.engine_simple.utils import ReservoirState, GridParameters, RockProperties, FluidProperties

__version__ = "1.0.0"
__author__ = "Reservoir Engineering Expert"

__all__ = [
    "ReservoirSimulationEngine",
    "StorageEfficiencyCalculator",
    "EnsembleKalmanFilter",
    "OptimizationInterface",
    "ReservoirState",
    "GridParameters",
    "RockProperties",
    "FluidProperties"
]