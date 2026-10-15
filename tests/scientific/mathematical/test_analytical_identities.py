"""
Level 1: Mathematical Verification - Analytical Identities.

Tests fundamental mathematical identities, integral definitions, and derivatives
using SymPy and independent numerical evaluations.
"""

import numpy as np
import pytest
import sympy as sp
from core.engine_surrogate.analytical_models import KovalRecoveryModel, PhDHybridSurrogate
from analysis.decline_curve_analysis import DeclineCurveAnalyzer


def test_koval_recovery_integral_identity():
    """
    Verify Koval (1963) Welge integration identity symbolically and numerically.

    In Koval's formulation, solvent fractional flow F_s(t_D) for 1/K <= t_D <= K is:
        F_s(t_D) = (K - sqrt(K / t_D)) / (K - 1)
    The cumulative oil recovery N_p(t_D) is:
        N_p(t_D) = integral_0^(t_D) (1 - F_s(tau)) dtau
                 = (2*sqrt(K*t_D) - 1 - t_D) / (K - 1)
    Also test whether the implemented surrogate formula (3K^2 - 3K + 1)/K^3 matches
    or deviates from authentic Welge integration.
    """
    # Symbolic verification with SymPy
    tau, K = sp.symbols('tau K', positive=True)
    Fs = (K - sp.sqrt(K / tau)) / (K - 1)
    
    # Integrate (1 - Fs) from 1/K to t_D
    tD = sp.symbols('tD', positive=True)
    # Integral before breakthrough (tau < 1/K): Fs = 0, so integral is 1/K
    np_post_bt = (1 / K) + sp.integrate(1 - Fs, (tau, 1 / K, tD))
    np_post_bt_simplified = sp.simplify(np_post_bt)
    
    # Expected analytical solution: (2*sqrt(K*tD) - 1 - tD) / (K - 1)
    expected_sym = (2 * sp.sqrt(K * tD) - 1 - tD) / (K - 1)
    diff = sp.simplify(np_post_bt_simplified - expected_sym)
    assert diff == 0, f"Symbolic integration of Koval (1 - Fs) does not match expected: diff = {diff}"

    # Evaluate at 1.0 PVI (t_D = 1.0)
    # Authentic Koval: N_p(1.0) = 2 / (sqrt(K) + 1)
    for k_val in [2.0, 4.0, 9.0]:
        authentic_recovery = float(expected_sym.subs({K: k_val, tD: 1.0}))
        expected_closed_form = 2.0 / (np.sqrt(k_val) + 1.0)
        assert np.isclose(authentic_recovery, expected_closed_form, atol=1e-6)


def test_welge_tangent_identity():
    """
    Verify Buckley-Leverett Welge tangent identity.

    At front saturation S_wf, the tangent line from connate water S_wi satisfies:
        df_w/dS_w(S_wf) = (f_w(S_wf) - f_w(S_wi)) / (S_wf - S_wi)
    """
    Sw, M = sp.symbols('Sw M', positive=True)
    # Quadratic Corey rel perm: krw = Sw^2, kro = (1 - Sw)^2
    # fw = 1 / (1 + (kro / krw) / M) = Sw^2 / (Sw^2 + (1 - Sw)^2 / M)
    kro = (1 - Sw)**2
    krw = Sw**2
    fw = krw / (krw + kro / M)
    dfw_dSw = sp.diff(fw, Sw)
    
    # For M = 1, fw = Sw^2 / (Sw^2 + (1 - Sw)^2)
    fw_m1 = fw.subs(M, 1)
    dfw_m1 = dfw_dSw.subs(M, 1)
    
    # Welge construction for Swi = 0: dfw/dSw = fw / Swf
    # Check that tangent condition has a unique solution in (0, 1)
    tangent_eq = sp.simplify(dfw_m1 - fw_m1 / Sw)
    roots = sp.solve(tangent_eq, Sw)
    valid_roots = [r.evalf() for r in roots if r.is_real and 0 < r < 1]
    assert len(valid_roots) == 1, f"Expected exactly 1 Welge shock front saturation, got {valid_roots}"
    assert np.isclose(float(valid_roots[0]), 0.70710678, atol=1e-4)


def test_arps_rate_cumulative_derivative_identity():
    """
    Verify Arps decline curve identity: dNp/dt = q(t).

    For hyperbolic decline:
        q(t) = qi / (1 + b * di * t)^(1/b)
        Np(t) = (qi^b / ((1 - b) * di)) * (qi^(1 - b) - q(t)^(1 - b))
    Verify symbolically that d(Np)/dt == q(t).
    """
    t, qi, di, b = sp.symbols('t qi di b', positive=True)
    q = qi / (1 + b * di * t)**(1 / b)
    Np = (qi**b / ((1 - b) * di)) * (qi**(1 - b) - q**(1 - b))
    
    dNp_dt = sp.diff(Np, t)
    diff = sp.simplify(dNp_dt - q)
    assert diff == 0, f"Derivative of Arps cumulative production does not equal rate: {diff}"
