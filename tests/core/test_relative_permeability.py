"""
Unit Tests for Stone I 3-Phase Relative Permeability & Carlson Gas Hysteresis
=============================================================================
"""

import pytest
import numpy as np
from core.engine_surrogate.relative_permeability import (
    normalize_saturations,
    corey_two_phase_relperm,
    stone_1_three_phase_relperm,
    carlson_trapped_gas,
    carlson_imbibition_gas_relperm,
)


def test_saturation_normalization():
    sw = 0.50
    sg = 0.20
    s_wc = 0.20
    s_orw = 0.20
    s_gc = 0.05
    sw_n, sg_n, so_n = normalize_saturations(sw, sg, s_wc, s_orw, s_gc)

    assert 0.0 <= sw_n <= 1.0
    assert 0.0 <= sg_n <= 1.0
    assert 0.0 <= so_n <= 1.0
    # Expected: sw_n = (0.50 - 0.20) / (1.0 - 0.20 - 0.20) = 0.30 / 0.60 = 0.50
    assert pytest.approx(sw_n, 1e-4) == 0.50


def test_corey_endpoints():
    s_wc = 0.20
    s_orw = 0.25
    s_gc = 0.05
    k_rw0 = 0.35
    k_ro0 = 0.85
    k_rg0 = 0.40

    # At irreducible water, krw should be 0, krow should be k_ro0
    res_wc = corey_two_phase_relperm(
        sw=s_wc, sg=s_gc, s_wc=s_wc, s_orw=s_orw, s_gc=s_gc,
        k_rw0=k_rw0, k_ro0=k_ro0, k_rg0=k_rg0
    )
    assert pytest.approx(res_wc["krw"], 1e-4) == 0.0
    assert pytest.approx(res_wc["krow"], 1e-4) == k_ro0
    assert pytest.approx(res_wc["krg"], 1e-4) == 0.0
    assert pytest.approx(res_wc["krog"], 1e-4) == k_ro0

    # At residual oil, krw should be k_rw0, krow should be 0
    sw_max = 1.0 - s_orw
    res_orw = corey_two_phase_relperm(
        sw=sw_max, sg=s_gc, s_wc=s_wc, s_orw=s_orw, s_gc=s_gc,
        k_rw0=k_rw0, k_ro0=k_ro0, k_rg0=k_rg0
    )
    assert pytest.approx(res_orw["krw"], 1e-4) == k_rw0
    assert pytest.approx(res_orw["krow"], 1e-4) == 0.0


def test_stone_1_reduction():
    # In 3-phase flow with both gas and water present, kro must be <= krow and <= krog
    s_wc = 0.20
    s_orw = 0.20
    s_gc = 0.05
    k_ro0 = 0.80

    res = stone_1_three_phase_relperm(
        sw=0.40, sg=0.20, s_wc=s_wc, s_orw=s_orw, s_gc=s_gc, k_ro0=k_ro0
    )
    assert 0.0 <= res["kro"] <= k_ro0
    assert res["krw"] > 0.0
    assert res["krg"] > 0.0


def test_carlson_hysteresis():
    s_wc = 0.20
    s_gr_max = 0.35

    # Peak gas saturation reached during gas cycle
    s_gi = 0.40
    s_gt = carlson_trapped_gas(s_gi, s_wc=s_wc, s_gr_max=s_gr_max)

    # Trapped gas must be strictly positive and <= s_gi
    assert 0.0 < s_gt < s_gi
    assert s_gt <= s_gr_max

    # Imbibition scanning curve: at sg <= s_gt, krg_imb must be 0
    krg_at_trap = carlson_imbibition_gas_relperm(
        sg=s_gt, s_gi=s_gi, s_wc=s_wc, s_gr_max=s_gr_max
    )
    assert pytest.approx(krg_at_trap, 1e-6) == 0.0

    # Above s_gt, krg_imb is positive and increases with sg
    krg_mid = carlson_imbibition_gas_relperm(
        sg=(s_gi + s_gt) / 2.0, s_gi=s_gi, s_wc=s_wc, s_gr_max=s_gr_max
    )
    assert krg_mid > 0.0
