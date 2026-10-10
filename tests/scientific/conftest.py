"""
Scientific Verification Test Suite Configuration & Fixtures.

Provides independent reference setups, standard fluid/reservoir properties,
and verification tolerances for scientific correctness testing.
"""

import pytest
import numpy as np
from typing import Dict, Any
from core.data_models import ReservoirData


def make_reservoir_instance(d: Dict[str, Any]) -> ReservoirData:
    """Helper to construct a valid ReservoirData object for tests."""
    return ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={},
        ooip_stb=float(d.get("ooip_stb", 5_000_000.0)),
        initial_pressure=float(d.get("initial_pressure", 4000.0)),
        temperature=float(d.get("temperature", 160.0)),
        average_porosity=float(d.get("porosity", 0.20)),
        average_permeability=float(d.get("permeability", 100.0)),
        initial_water_saturation=float(d.get("connate_water_saturation", d.get("s_wi", 0.25))),
        thickness_ft=float(d.get("thickness_ft", 50.0)),
        area_acres=float(d.get("area_acres", 160.0)),
        length_ft=float(d.get("length_ft", 2640.0)),
        rock_compressibility=float(d.get("rock_compressibility", 4.0e-6)),
        oil_fvf=float(d.get("oil_fvf", 1.25)),
    )


@pytest.fixture
def standard_reservoir_data() -> Dict[str, Any]:
    """Standard reference reservoir data based on SPE 5 benchmark parameters."""
    return {
        "ooip_stb": 5_000_000.0,
        "initial_pressure": 4000.0,
        "temperature": 160.0,  # deg F
        "permeability": 100.0,  # mD
        "porosity": 0.20,
        "thickness_ft": 50.0,
        "area_acres": 160.0,
        "length_ft": 2640.0,
        "connate_water_saturation": 0.25,
        "residual_oil_saturation": 0.20,
        "sor": 0.20,
        "s_wi": 0.25,
        "v_dp": 0.60,
        "rock_compressibility": 4.0e-6,
        "oil_compressibility": 1.2e-5,
        "water_compressibility": 3.0e-6,
        "gas_compressibility": 1.5e-4,
        "oil_viscosity": 2.0,  # cP
        "co2_viscosity": 0.04,  # cP
        "oil_fvf": 1.25,  # RB/STB
        "mmp": 2400.0,  # psia
        "caprock_fracture_pressure_psi": 5500.0,
        "target_pressure_psi": 3500.0,
    }


@pytest.fixture
def light_oil_params() -> Dict[str, Any]:
    """Volatile light crude with high API gravity (> 55 deg API)."""
    return {
        "temperature": 200.0,
        "oil_gravity": 56.5,
        "c7_plus_mw": 140.0,
        "c7_plus_fraction": 0.15,
        "injection_gas_composition": {"CO2": 0.95, "CH4": 0.05},
    }


@pytest.fixture
def impure_co2_streams() -> Dict[str, Dict[str, float]]:
    """CO2 injection gas streams with varying impurities."""
    return {
        "pure": {"CO2": 1.0},
        "methane_10": {"CO2": 0.90, "CH4": 0.10},
        "methane_20": {"CO2": 0.80, "CH4": 0.20},
        "nitrogen_10": {"CO2": 0.90, "N2": 0.10},
    }
