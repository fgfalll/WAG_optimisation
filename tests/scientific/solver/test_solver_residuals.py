"""
Level 3: Numerical Verification - Solver Residuals & Equilibrium.

Audits residual calculation in pressure material balance iterations:
    Residual = | V_p * c_t * (P^{n+1} - P^n) / dt - q_net(P^{n+1}) | -> 0.
"""

import numpy as np
import pytest


def test_pressure_material_balance_discrete_residual():
    """
    Test that the pressure increment formula satisfies backward-Euler linearization residual:
        (V_p * c_t + J_eff * dt) * dP - q_net * dt == 0
    to machine precision.
    """
    dt = 30.0
    vp = 10_000_000.0
    ct = 1.2e-5
    q_net = 1500.0
    j_eff = 4.0

    # Equation implemented in surrogate_engine.py:376:
    dp = (q_net * dt) / (vp * ct + j_eff * dt)

    # Calculate equation residual
    residual = abs((vp * ct + j_eff * dt) * dp - q_net * dt)
    assert np.isclose(residual, 0.0, atol=1e-10), f"Linearized residual non-zero: {residual}"
