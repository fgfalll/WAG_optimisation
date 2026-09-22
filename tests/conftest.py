"""
Test configuration and fixtures.
Contains fixtures for optimization tests and surrogate engine tests.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt6.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    import PyQt6.QtWebEngineWidgets
except Exception:
    pass

from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    GeneticAlgorithmParams,
    BayesianOptimizationParams,
    CO2StorageParameters,
    ProfileParameters,
    PVTProperties,
    EOSModelParameters,
    EmpiricalFittingParameters,
)
from core.engine_surrogate.surrogate_engine import create_surrogate_engine


# =============================================================================
# Optimization Test Fixtures
# =============================================================================


@pytest.fixture
def mock_eos_model():
    """Create a mock EOS model."""
    eos = MagicMock()
    eos.calculate_mmp.return_value = 2500.0
    eos.calculate_z_factor.return_value = 0.75
    return eos


@pytest.fixture
def reservoir_data(mock_eos_model):
    """Create test reservoir data."""
    return ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={},
        ooip_stb=1000000.0,
        initial_pressure=3000.0,
        temperature=150.0,
        rock_compressibility=3e-6,
        average_porosity=0.2,
        initial_water_saturation=0.25,
        thickness_ft=50.0,
        area_acres=100.0,
        length_ft=2000.0,
        oil_fvf=1.2,
        eos_model=mock_eos_model,
    )


@pytest.fixture
def pvt_data():
    """Create test PVT properties."""
    return PVTProperties(
        oil_compressibility=1e-5,
        oil_viscosity_cp=1.5,
        water_compressibility=3e-6,
        water_viscosity_cp=0.5,
        water_fvf=1.0,
    )


@pytest.fixture
def eor_params():
    """Create test EOR parameters."""
    return EORParameters(
        injection_rate=5000.0,
        target_pressure_psi=3200.0,
        max_pressure_psi=4000.0,
        injection_scheme="continuous",
        wag_ratio=1.0,
        mobility_ratio=5.0,
    )


@pytest.fixture
def operational_params():
    """Create test operational parameters."""
    return OperationalParameters(
        project_lifetime_years=15,
        time_resolution="monthly",
        recovery_model_selection="hybrid",
    )


@pytest.fixture
def economic_params():
    """Create test economic parameters."""
    return EconomicParameters(
        oil_price_usd_per_bbl=80.0,
        co2_purchase_cost_usd_per_tonne=50.0,
        discount_rate_fraction=0.1,
    )


@pytest.fixture
def ga_params():
    """Create test GA parameters."""
    return GeneticAlgorithmParams(
        num_generations=10,
        sol_per_pop=20,
        num_parents_mating=4,
        num_objectives=1,
        secondary_objective="recovery_factor",
        num_diverse_solutions_for_bo=5,
        diversity_threshold_for_bo=0.2,
    )


@pytest.fixture
def ga_params_nsga2():
    """Create test GA parameters for NSGA-II (bi-objective)."""
    return GeneticAlgorithmParams(
        num_generations=10,
        sol_per_pop=20,
        num_parents_mating=4,
        num_objectives=2,
        secondary_objective="recovery_factor",
        num_diverse_solutions_for_bo=5,
        diversity_threshold_for_bo=0.2,
    )


@pytest.fixture
def bo_params():
    """Create test BO parameters."""
    return BayesianOptimizationParams(
        n_iterations=10,
        n_initial_points=3,
    )


@pytest.fixture
def co2_storage_params():
    """Create test CO2 storage parameters."""
    return CO2StorageParameters()


@pytest.fixture
def profile_params():
    """Create test profile parameters."""
    return ProfileParameters()


@pytest.fixture
def mock_surrogate_engine():
    """Create a mock surrogate engine that returns valid results."""
    mock = MagicMock()
    mock.evaluate_scenario.return_value = {
        "recovery_factor": 0.35,
        "npv": 5000000.0,
        "cumulative_oil": 350000.0,
        "co2_stored": 250000.0,
        "co2_utilization": 0.5,
        "storage_efficiency": 0.6,
        "oil_production_rate": np.ones(180) * 100.0,
        "water_production_rate": np.ones(180) * 50.0,
        "gas_production_rate": np.ones(180) * 20.0,
        "co2_injection": np.ones(180) * 5000.0,
        "pressure": np.ones(180) * 3000.0,
        "time_vector": np.arange(0, 180, 1),
        "engine_type": "surrogate",
        "constraint_violations": {},
    }
    return mock


@pytest.fixture
def mock_objective_functions():
    """Create mock objective functions."""
    mock = MagicMock()
    mock._calculate_objective_functions.return_value = {
        "recovery_factor": 0.35,
        "npv": 5000000.0,
        "co2_utilization": 0.5,
        "storage_efficiency": 0.6,
    }
    return mock


# =============================================================================
# Surrogate Engine Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cmg_parser():
    """SR3 parser instance for CMG GEM files."""
    from validation.sr3_parser import SR3Parser

    return SR3Parser()


@pytest.fixture
def phd_hybrid_engine():
    """Surrogate engine with PhD hybrid recovery model."""
    return create_surrogate_engine(
        model_type="analytical",
        recovery_model_type="phd_hybrid",
    )


@pytest.fixture
def analytical_engine():
    """Surrogate engine with default analytical model."""
    return create_surrogate_engine(
        model_type="analytical",
        recovery_model_type="hybrid",
    )


@pytest.fixture(
    params=["miscible", "immiscible", "hybrid", "koval", "buckley_leverett", "phd_hybrid"]
)
def all_recovery_engines(request):
    """Parametrized fixture for all recovery model types."""
    return create_surrogate_engine(
        model_type="analytical",
        recovery_model_type=request.param,
    )


@pytest.fixture
def standard_reservoir():
    """Standard test reservoir data."""
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=1_000_000.0,
        initial_pressure=3000.0,
        temperature=150.0,
        average_porosity=0.20,
        average_permeability=100.0,
        initial_water_saturation=0.25,
        length_ft=2000.0,
        area_acres=10.0,
        thickness_ft=50.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.25
    res.v_dp_coefficient = 0.5
    res.bg = 0.002
    return res


@pytest.fixture
def standard_eor_params():
    """Standard EOR parameters."""
    return EORParameters(
        injection_rate=5000.0,
        target_pressure_psi=3000.0,
        max_pressure_psi=6000.0,
        mobility_ratio=5.0,
        default_mmp_fallback=2500.0,
        wag_ratio=1.0,
        injection_scheme="continuous",
        default_oil_viscosity_cp=2.0,
        default_co2_viscosity_cp=0.05,
        s_gc=0.05,
        n_o=2.0,
        n_g=2.0,
        sor=0.25,
    )


@pytest.fixture
def standard_operational_params():
    """Standard operational parameters."""
    return OperationalParameters(
        project_lifetime_years=10,
        time_resolution="yearly",
        recovery_model_selection="koval",
    )


@pytest.fixture
def standard_economic_params():
    """Standard economic parameters."""
    return EconomicParameters(
        oil_price_usd_per_bbl=70.0,
        co2_purchase_cost_usd_per_tonne=50.0,
        co2_recycle_cost_usd_per_tonne=15.0,
        co2_storage_credit_usd_per_tonne=25.0,
        discount_rate_fraction=0.10,
        capex_usd=5_000_000.0,
        fixed_opex_usd_per_year=200_000.0,
        variable_opex_usd_per_bbl=5.0,
        carbon_tax_usd_per_tonne=0.0,
    )


@pytest.fixture
def gmflu001_1d_reservoir():
    """gmflu001-1d case reservoir (SPE5 Wasson 1D).

    OOIP: 263,039 STB (from SR3: cum_oil=194,873, RF=0.7409)
    Note: This is a gas cycling case, not a CO2 flood like gmflu002/003
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=263_039.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=200.0,
        initial_water_saturation=0.20,
        area_acres=5.739,
        length_ft=5000.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.0
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu001_reservoir():
    """gmflu001 case reservoir (SPE5 Wasson 3D).

    OOIP: 3,449,258 STB (from SR3: cum_oil=2,254,086, RF=0.6535)
    Note: This is a gas cycling case, not a CO2 flood like gmflu002/003
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=3_449_258.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=215.0,
        initial_water_saturation=0.20,
        area_acres=112.48,
        length_ft=3500.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.612
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu002_1d_reservoir():
    """gmflu002-1d case reservoir (SPE5 Wasson 1D CO2 Flood).

    OOIP: 4,648,655 STB (from SR3: cum_oil=3,620,543, RF=0.7788)
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=4_648_655.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=200.0,
        initial_water_saturation=0.20,
        area_acres=5.739,
        length_ft=5000.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.0
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu002_reservoir():
    """gmflu002 case reservoir (SPE5 Wasson 3D CO2 Flood).

    OOIP: 45,076,813 STB (from SR3: cum_oil=14,992,874, RF=0.3326)
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=45_076_813.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=215.0,
        initial_water_saturation=0.20,
        area_acres=112.48,
        length_ft=3500.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.612
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu003_1d_reservoir():
    """gmflu003-1d case reservoir (SPE5 Wasson 1D variant H2/He).

    OOIP: 185,699 STB (from SR3: cum_oil=57,987, RF=0.3123)
    Note: This case has H2/He components, lower recovery than gmflu002
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=185_699.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=200.0,
        initial_water_saturation=0.20,
        area_acres=5.739,
        length_ft=5000.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.0
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu003_reservoir():
    """gmflu003 case reservoir (SPE5 Wasson 3D variant H2/He).

    OOIP: 207,709 STB (from SR3: cum_oil=12,470, RF=0.0600)
    Note: This case has H2/He components, very low recovery due to composition
    """
    res = ReservoirData(
        grid={},
        pvt_tables={},
        ooip_stb=207_709.0,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=215.0,
        initial_water_saturation=0.20,
        area_acres=112.48,
        length_ft=3500.0,
        oil_fvf=1.2,
    )
    res.residual_oil_saturation = 0.39
    res.v_dp_coefficient = 0.612
    res.bg = 0.00207
    return res


@pytest.fixture
def gmflu_eor_params():
    """Standard EOR parameters for gmflu cases."""
    return EORParameters(
        injection_rate=24_837.0,
        target_pressure_psi=1430.0,
        default_mmp_fallback=1200.0,
        default_oil_viscosity_cp=1.5,
        default_co2_viscosity_cp=0.05,
        sor=0.39,
    )


@pytest.fixture
def gmflu_operational_params():
    """Standard operational parameters for gmflu cases (8 year project)."""
    return OperationalParameters(
        project_lifetime_years=8,
        time_resolution="yearly",
        recovery_model_selection="koval",
    )


@pytest.fixture
def eos_model_params():
    """EOS model parameters for gmflu cases."""
    return EOSModelParameters(
        eos_type="PR",
        component_names=["CO2", "C1", "C4", "C10+"],
        component_properties=np.array(
            [
                [0.0, 44.01, 304.13, 7.376e6, 0.225],
                [0.1, 16.04, 190.6, 4.604e6, 0.011],
                [0.3, 58.12, 425.2, 3.796e6, 0.200],
                [0.6, 142.0, 617.7, 2.11e6, 0.490],
            ]
        ),
        binary_interaction_coeffs=np.zeros((4, 4)),
    )


@pytest.fixture
def empirical_fitting_params():
    """Standard empirical fitting parameters for PhD hybrid model."""
    return EmpiricalFittingParameters(
        c7_plus_fraction=0.57,
        alpha_base=1.0,
        miscibility_window=0.011,
        breakthrough_time_years=1.5,
        trapping_efficiency=0.4,
        initial_gor_scf_per_stb=500.0,
        transverse_mixing_calibration=0.5,
        omega_tl=0.6,
        k_ro_0=0.8,
        k_rg_0=1.0,
        n_o=2.0,
        n_g=2.0,
    )
