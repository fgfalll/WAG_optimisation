"""
Fast Surrogate Engine for CO2 EOR Optimization
==============================================

Physics-informed intermediate reservoir simulation engine for CO2 EOR.
"""

from .surrogate_models import (
    BaseSurrogateModel,
    AnalyticalSurrogate,
    create_surrogate_model,
    get_available_surrogate_models,
)
from .surrogate_engine import SurrogateEngine, SurrogateEngineWrapper
from .analytical_models import (
    AnalyticalRecoveryModel,
    BuckleyLeverettSurrogate,
    MiscibleSurrogate,
    ImmiscibleSurrogate,
    HybridSurrogate,
    PhDHybridSurrogate,
)
from .profile_generator_fast import FastProfileGenerator
from .pvt_state import SolventExtendedPVTEngine
from .geomechanics_fault import GeomechanicsFaultModel
from .relative_permeability import (
    stone_1_three_phase_relperm,
    carlson_trapped_gas,
    carlson_imbibition_gas_relperm,
    corey_two_phase_relperm,
    normalize_saturations,
)
from .well_mechanics import (
    calculate_peaceman_index_horizontal,
    calculate_peaceman_index_vertical,
    calculate_vertical_perforation_overlap,
    calculate_interwell_transmissibility,
    generate_synthetic_well_trajectory,
    validate_well_network,
)

__all__ = [
    # Base and Surrogate Models
    "BaseSurrogateModel",
    "AnalyticalSurrogate",
    "create_surrogate_model",
    "get_available_surrogate_models",
    # Main Engine
    "SurrogateEngine",
    "SurrogateEngineWrapper",
    # Physics Engines
    "FastProfileGenerator",
    "SolventExtendedPVTEngine",
    "GeomechanicsFaultModel",
    "stone_1_three_phase_relperm",
    "carlson_trapped_gas",
    "carlson_imbibition_gas_relperm",
    "corey_two_phase_relperm",
    "normalize_saturations",
    # Well Mechanics
    "calculate_peaceman_index_horizontal",
    "calculate_peaceman_index_vertical",
    "calculate_vertical_perforation_overlap",
    "calculate_interwell_transmissibility",
    "generate_synthetic_well_trajectory",
    "validate_well_network",
    # Analytical Recovery Models
    "AnalyticalRecoveryModel",
    "BuckleyLeverettSurrogate",
    "MiscibleSurrogate",
    "ImmiscibleSurrogate",
    "HybridSurrogate",
    "PhDHybridSurrogate",
]

__version__ = "0.8.5"
