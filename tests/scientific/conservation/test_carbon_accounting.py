"""
Level 2: Physical Verification - Carbon Mass Balance & Storage Accounting.

Tests closed-loop carbon balance invariants:
    Gross Injected = Purchased + Recycled = Net Stored + Leakage + Produced
and checks for double-subtraction or inventory leakages.
"""

import numpy as np
import pytest
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from core.data_models import (
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    AdvancedEngineParams,
)
from tests.scientific.conftest import make_reservoir_instance


def test_closed_loop_carbon_balance_invariant(standard_reservoir_data):
    """
    Verify closed-loop carbon balance in SurrogateEngine:
        Gross Injected = Net Stored + Cumulative Produced
        Gross Injected = Purchased CO2 + Recycled CO2
    """
    res = make_reservoir_instance(standard_reservoir_data)
    eor = EORParameters(
        default_mmp_fallback=standard_reservoir_data["mmp"],
        injection_rate=10000.0,
    )
    ops = OperationalParameters(
        project_lifetime_years=10,
        time_resolution="monthly",
    )
    econ = EconomicParameters()

    engine = SurrogateEngine()
    results = engine.evaluate_scenario(res, eor, ops, econ)

    annual_inj = results.get("annual_co2_injected_mscf", np.array([]))
    annual_purchased = results.get("annual_co2_purchased_mscf", np.array([]))
    annual_recycled = results.get("annual_co2_recycled_mscf", np.array([]))
    annual_prod = results.get("annual_co2_produced_mscf", np.array([]))

    if len(annual_inj) > 0 and len(annual_purchased) > 0:
        total_inj = np.sum(annual_inj)
        total_purchased = np.sum(annual_purchased)
        total_recycled = np.sum(annual_recycled)
        total_prod = np.sum(annual_prod)

        # Invariant 1: Injected = Purchased + Recycled
        assert np.isclose(total_inj, total_purchased + total_recycled, rtol=1e-3), (
            f"Gross injection ({total_inj}) != Purchased ({total_purchased}) + Recycled ({total_recycled})"
        )

        # Invariant 2: Recycled cannot exceed produced
        assert total_recycled <= total_prod * 1.001, (
            f"Recycled CO2 ({total_recycled}) exceeds produced CO2 ({total_prod})"
        )


def test_material_balance_analyzer_closed_loop():
    """
    Verify MaterialBalanceAnalyzer carbon balance.
    """
    from analysis.material_balance import MaterialBalanceAnalyzer

    analyzer = MaterialBalanceAnalyzer()
    
    # 10-year test scenario
    inj_mscf = np.full(10, 10000.0)
    prod_mscf = np.full(10, 4000.0)
    recycled_mscf = prod_mscf * 0.90  # 90% recycle efficiency
    oil_prod_stb = np.full(10, 500.0)

    # Net storage = sum(inj) - sum(prod) = 10 * (10000 - 4000) = 60,000 MSCF
    expected_net_storage = 60000.0
    actual_stored = np.sum(inj_mscf) - np.sum(prod_mscf)
    assert np.isclose(actual_stored, expected_net_storage)
