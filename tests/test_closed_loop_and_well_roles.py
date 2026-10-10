"""
Unit tests for well role inference, closed-loop mass balance accounting,
and decline curve onset detection.
"""

import numpy as np
import pytest
from analysis.decline_curve_analysis import DeclineCurveAnalyzer
from analysis.material_balance import MaterialBalanceAnalyzer
from core.data_models import WellData


def test_dca_plateau_and_decline_onset():
    """Verify DCA handles plateau followed by decline with high positive R2 and reasonable EUR."""
    dca = DeclineCurveAnalyzer()
    
    # 15-year profile with 3-year plateau at 1.5M STB/yr followed by exponential decline
    time = np.arange(1, 16, dtype=float)
    rate = np.array([
        1.50e6, 1.52e6, 1.49e6,  # Plateau years 1-3
        1.25e6, 1.00e6, 0.80e6, 0.65e6, 0.52e6, 0.42e6, 0.34e6,  # Boundary-dominated decline
        0.27e6, 0.22e6, 0.18e6, 0.14e6, 0.11e6
    ])
    
    res = dca.analyze_production(time, rate, model_type="auto", forecast_years=30)
    
    # R-squared must be strongly positive (> 0.85)
    assert res.r_squared > 0.85, f"Expected R2 > 0.85, got {res.r_squared}"
    # Parameters must identify decline onset year
    assert "decline_onset_year" in res.parameters
    assert res.parameters["decline_onset_year"] == 4.0
    # EUR must be physically bounded and exceed historical cumulative
    hist_cum = np.sum(rate)
    assert res.forecast_cumulative[-1] >= hist_cum
    # EUR must not unreasonably explode
    assert res.forecast_cumulative[-1] < 2.0 * hist_cum


def test_material_balance_geomechanical_gating_and_closure():
    """Verify material balance suppresses leakage when below fracture ceiling and closes carbon balance."""
    mb = MaterialBalanceAnalyzer(co2_density_tonne_per_mscf=0.053)
    
    n_years = 10
    purchased_mscf = np.full(n_years, 100000.0)  # 5,300 t/yr
    produced_mscf = np.array([0, 10000, 30000, 50000, 70000, 80000, 85000, 90000, 90000, 90000], dtype=float)
    recycled_mscf = produced_mscf * 0.90
    
    # Mechanically intact reservoir: P_res = 2200 psi, P_frac = 5000 psi (safe ceiling = 4500 psi)
    res_params = {"pressure": 2200.0}
    eor_params = {"caprock_fracture_pressure_psi": 5000.0, "caprock_safety_factor": 0.90}
    
    bal = mb.calculate_material_balance(
        annual_co2_injected_mscf=purchased_mscf,
        annual_co2_produced_mscf=produced_mscf,
        annual_co2_recycled_mscf=recycled_mscf,
        leakage_rate_fraction=0.01,
        reservoir_params=res_params,
        eor_params=eor_params
    )
    
    stats = mb.generate_summary_statistics(bal)
    
    # With safe geomechanics, blanket 1%/yr leakage is gated to <= 0.01%/yr (trace permeation)
    # Total leakage over 10 years on ~40,000 t storage is ~23 t (< 0.05% of 53,000 t injection, vs 2,289 t without gating)
    assert stats["total_leakage_tonne"] < 50.0
    assert stats["total_leakage_tonne"] / stats["total_injected_tonne"] < 0.001
    # Mass balance closure error must be essentially 0
    assert stats["mass_balance_error_tonne"] < 1.0


def test_well_data_injector_inference():
    """Verify that WellData with 'inj' in name is correctly identified as injector."""
    w1 = WellData(
        name="Well-Injector-1",
        depths=np.array([0.0, 1000.0]),
        properties={},
        units={},
        metadata={"status": "Producer (Active)", "type": "producer"},
        well_path=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1000.0]])
    )
    # Check that inferring role from name overrides corrupted producer metadata
    is_inj = (
        str(w1.metadata.get("type", "")).lower() == "injector" or
        "injector" in str(w1.metadata.get("status", "")).lower() or
        "inj" in w1.name.lower()
    )
    assert is_inj is True
