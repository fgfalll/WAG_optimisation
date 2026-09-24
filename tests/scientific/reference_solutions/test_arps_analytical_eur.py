"""
Level 5: Validation - Independent Reference Solutions: Analytical Arps Decline EUR.

Compares analytical closed-form Arps integration against numerical cumulative integration
in analysis/decline_curve_analysis.py.
"""

import numpy as np
import pytest
from analysis.decline_curve_analysis import DeclineCurveAnalyzer


def test_analytical_vs_trapezoidal_arps_eur():
    """
    Compare exact closed-form Arps hyperbolic cumulative production:
        Np(t) = [qi^b / ((1 - b) * di)] * [qi^(1 - b) - q(t)^(1 - b)]
    against numerical trapezoidal integration.
    """
    analyzer = DeclineCurveAnalyzer()
    
    qi = 1000.0  # BOPD
    di = 0.10 / 365.25  # per day (10% nominal annual)
    b = 0.50
    t_days = np.linspace(0, 3650, 120)  # 10 years

    # Numerical rate
    q = analyzer.hyperbolic_decline(t_days, qi, di, b)
    # Numerical cumulative via trapezoid
    np_numerical = analyzer.calculate_cumulative(t_days, q)

    # Exact analytical cumulative
    np_analytical = (qi**b / ((1.0 - b) * di)) * (qi**(1.0 - b) - q**(1.0 - b))

    # Numerical trapezoid should be within 0.1% of exact analytical formula
    rel_error = np.max(abs(np_numerical[1:] - np_analytical[1:]) / np_analytical[1:])
    assert rel_error < 1e-3, f"Trapezoid integration error exceeds 0.1%: {rel_error*100:.3f}%"
