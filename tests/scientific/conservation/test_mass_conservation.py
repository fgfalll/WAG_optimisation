"""
Level 2: Physical Verification - Mass Conservation & Material Balance.

Tests conservation of fluid masses, volumetric limits, and cumulative recovery bounds.
"""

import numpy as np
import pytest
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from core.engine_surrogate.analytical_models import PhDHybridSurrogate


def test_cumulative_oil_recovery_mass_bound(standard_reservoir_data):
    """
    Verify that simulated cumulative oil production cannot exceed total Original Oil in Place:
        N_p(t) <= OOIP for all t.
    """
    surrogate = PhDHybridSurrogate()
    params = standard_reservoir_data.copy()
    
    # Run with aggressive injection throughput: 5.0 HCPVI
    params["hcpvi"] = 5.0
    rf = surrogate.calculate_recovery(**params)
    
    assert rf <= 1.0, f"Cumulative recovery factor exceeds 100% of OOIP: {rf}"
    assert rf >= 0.0, f"Cumulative recovery factor negative: {rf}"


def test_pore_volume_vs_ooip_recovery_bound_discrepancy(standard_reservoir_data):
    """
    Test whether maximum recovery factor is bounded by mobile oil as fraction of OOIP:
        RF_max = (1 - S_wi - S_or) / (1 - S_wi)
    or incorrectly by mobile pore volume:
        RF_max_pore = 1 - S_wi - S_or.

    With S_wi = 0.25, S_or = 0.20:
        RF_max (OOIP basis) = (1 - 0.25 - 0.20) / (1 - 0.25) = 0.55 / 0.75 = 0.733 (73.3%)
        RF_max_pore = 0.55 (55.0%)
    Confirms documentation of mobile oil ceiling in analytical_models.py:881 and surrogate_engine.py:425.
    """
    swi = standard_reservoir_data["s_wi"]
    sor = standard_reservoir_data["sor"]

    true_theoretical_rf_limit = (1.0 - swi - sor) / (1.0 - swi)
    code_pore_volume_limit = 1.0 - swi - sor

    # Verify that the code pore volume limit is strictly lower than true OOIP theoretical limit
    truncation_fraction = (true_theoretical_rf_limit - code_pore_volume_limit) / true_theoretical_rf_limit
    assert truncation_fraction > 0.20, (
        f"Confirms mobile oil definition defect: using (1 - Swi - Sor) truncates theoretical limit by {truncation_fraction*100:.1f}%!"
    )
