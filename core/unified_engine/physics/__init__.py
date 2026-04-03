"""
Shared Physics Modules for CO2 EOR Optimizer.

This package contains unified physics models used across all simulation engines.
These modules replace duplicate implementations in engine_simple and Phys_engine_full.

Modules:
- relative_permeability: Corey-Brooks relative permeability with hysteresis
- co2_properties: CO2 thermodynamics (density, viscosity, solubility)
- fluid_properties: General PVT correlations
- multiphase_flow: Transmissibility and fractional flow calculations
- eos: Equations of state (Peng-Robinson, SRK)
- geomechanics: Stress/strain and fault mechanics
"""

from .multiphase_flow import (
    MultiphaseFlowModule,
    FluidProperties,
    RockProperties,
    RelativePermeabilityModel,
    DarcyFlowCalculator,
    create_multiphase_flow_module,
)

from .relative_permeability import (
    CoreyParameters,
    CoreyRelativePermeability,
    LETParameters,
    LETRelativePermeability,
    HysteresisModel,
    LandHysteresisModel,
    create_relative_permeability_model,
)

from .co2_properties import (
    CO2Properties,
    CO2DensityCorrelation,
    CO2ViscosityCorrelation,
    CO2SolubilityCorrelation,
    create_co2_properties,
)

from .fluid_properties import (
    FluidProperties as PVTFluidProperties,
    BlackOilProperties,
    ViscosityCorrelation,
    DensityCorrelation,
    create_black_oil_properties,
)

from .geomechanics import (
    GeomechanicsParameters,
    StressStrainCalculator,
    FaultStabilityAnalyzer,
    CompactionCalculator,
    create_geomechanics_parameters,
)

from .eos import (
    EOSParameters,
    CubicEOS,
    PengRobinsonEOS,
    SoaveRedlichKwongEOS,
    PhaseEquilibriumCalculator,
    ReservoirFluid,
    create_eos,
)

__all__ = [
    # Multiphase flow
    "MultiphaseFlowModule",
    "FluidProperties",
    "RockProperties",
    "RelativePermeabilityModel",
    "DarcyFlowCalculator",
    "create_multiphase_flow_module",
    # Relative permeability
    "CoreyParameters",
    "CoreyRelativePermeability",
    "LETParameters",
    "LETRelativePermeability",
    "HysteresisModel",
    "LandHysteresisModel",
    "create_relative_permeability_model",
    # CO2 properties
    "CO2Properties",
    "CO2DensityCorrelation",
    "CO2ViscosityCorrelation",
    "CO2SolubilityCorrelation",
    "create_co2_properties",
    # Fluid properties
    "PVTFluidProperties",
    "BlackOilProperties",
    "ViscosityCorrelation",
    "DensityCorrelation",
    "create_black_oil_properties",
    # Geomechanics
    "GeomechanicsParameters",
    "StressStrainCalculator",
    "FaultStabilityAnalyzer",
    "CompactionCalculator",
    "create_geomechanics_parameters",
    # EOS
    "EOSParameters",
    "CubicEOS",
    "PengRobinsonEOS",
    "SoaveRedlichKwongEOS",
    "PhaseEquilibriumCalculator",
    "ReservoirFluid",
    "create_eos",
]
