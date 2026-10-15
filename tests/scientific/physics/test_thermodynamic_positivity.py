"""
Level 2: Physics Verification - Fluid Thermodynamics & Positivity.

Tests physical invariants in PVT generation:
- Positive isothermal compressibility (co = -1/Bo * dBo/dP > 0, dBo/dP < 0)
- Physical viscosity-pressure dependence (dmu_o/dP > 0, dmu_co2/dP > 0)
"""

import numpy as np
import pytest
from core.data_integration_engine import DataIntegrationEngine


def test_oil_compressibility_positivity():
    """
    Test isothermal oil compressibility positivity:
        co = -1/Bo * dBo/dP > 0   <=>   dBo/dP < 0
    Verifies resolution of SCI-FLAW-02 across both _create_reservoir_data and _create_pvt_properties.
    """
    engine = DataIntegrationEngine()
    data = {
        "reservoir_parameters": {
            "grid_dimensions": {"nx": 5, "ny": 5, "nz": 2},
            "block_sizes": {"dx": 100.0, "dy": 100.0, "dz": 20.0},
            "temperature": 150.0,
        },
        "pvt_parameters": {
            "oil_viscosity_cp": 2.0,
            "gas_viscosity_cp": 0.02,
        },
    }
    # 1. Test ReservoirData pvt_tables
    res_data = engine._create_reservoir_data(data)
    pvt_tables = res_data.pvt_tables
    pressures = pvt_tables["PRESSURE"]
    oil_fvf = pvt_tables["OIL_FVF"]

    d_bo_dp = np.diff(oil_fvf) / np.diff(pressures)
    assert np.all(d_bo_dp < 0), f"Expected dBo/dP < 0 in pvt_tables, got: {d_bo_dp}"
    bo_mid = 0.5 * (oil_fvf[:-1] + oil_fvf[1:])
    co = -(1.0 / bo_mid) * d_bo_dp
    assert np.all(co > 0), f"Expected co > 0 in pvt_tables, got: {co}"

    # 2. Test PVTProperties object
    pvt_props = engine._create_pvt_properties(data)
    d_bo_dp2 = np.diff(pvt_props.oil_fvf) / np.diff(pvt_props.pressure_points)
    assert np.all(d_bo_dp2 < 0), f"Expected dBo/dP < 0 in pvt_props, got: {d_bo_dp2}"


def test_liquid_viscosity_pressure_derivative():
    """
    Test that undersaturated liquid and supercritical fluid viscosities
    increase with pressure:
        dmu_o/dP > 0, dmu_co2/dP > 0
    Verifies resolution of SCI-FLAW-03 across both _create_reservoir_data and _create_pvt_properties.
    """
    engine = DataIntegrationEngine()
    data = {
        "reservoir_parameters": {
            "grid_dimensions": {"nx": 5, "ny": 5, "nz": 2},
            "block_sizes": {"dx": 100.0, "dy": 100.0, "dz": 20.0},
            "temperature": 150.0,
        },
        "pvt_parameters": {
            "oil_viscosity_cp": 2.0,
            "gas_viscosity_cp": 0.02,
        },
    }
    # 1. Test ReservoirData pvt_tables
    res_data = engine._create_reservoir_data(data)
    pvt_tables = res_data.pvt_tables
    pressures = pvt_tables["PRESSURE"]
    oil_visc = pvt_tables["OIL_VISC"]
    co2_visc = pvt_tables["CO2_VISC"]

    d_mu_o_dp = np.diff(oil_visc) / np.diff(pressures)
    d_mu_co2_dp = np.diff(co2_visc) / np.diff(pressures)
    assert np.all(d_mu_o_dp > 0), f"Expected dmu_o/dP > 0 in pvt_tables, got: {d_mu_o_dp}"
    assert np.all(d_mu_co2_dp > 0), f"Expected dmu_co2/dP > 0 in pvt_tables, got: {d_mu_co2_dp}"

    # 2. Test PVTProperties object
    pvt_props = engine._create_pvt_properties(data)
    d_mu_o_dp2 = np.diff(pvt_props.oil_viscosity) / np.diff(pvt_props.pressure_points)
    d_mu_co2_dp2 = np.diff(pvt_props.co2_viscosity) / np.diff(pvt_props.pressure_points)
    assert np.all(d_mu_o_dp2 > 0), f"Expected dmu_o/dP > 0 in pvt_props, got: {d_mu_o_dp2}"
    assert np.all(d_mu_co2_dp2 > 0), f"Expected dmu_co2/dP > 0 in pvt_props, got: {d_mu_co2_dp2}"
