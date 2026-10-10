"""
Level 3: Numerical Verification - Method of Manufactured Solutions (MMS).

Tests spatial and temporal discretization accuracy for 1D pressure diffusion:
    c_t * (dP/dt) - (k / mu) * (d^2P/dx^2) = S(x, t)
using a manufactured analytical solution:
    P_exact(x, t) = P_0 + A * sin(pi * x / L) * exp(-lambda * t).

Verifies monotonic error reduction under grid refinement.
"""

import numpy as np
import pytest
from scipy.linalg import solve_banded


def test_mms_pressure_diffusion_convergence():
    """
    Run MMS on 1D pressure diffusion to verify spatial error convergence.
    """
    L = 1000.0  # ft
    P0 = 3000.0  # psi
    A = 500.0  # psi
    alpha = 100.0  # ft2/day hydraulic diffusivity
    lambda_t = 0.05  # 1/day
    T_end = 2.0  # days

    def p_exact(x, t):
        return P0 + A * np.sin(np.pi * x / L) * np.exp(-lambda_t * t)

    def source_term(x, t):
        term = -lambda_t + alpha * (np.pi / L)**2
        return term * A * np.sin(np.pi * x / L) * np.exp(-lambda_t * t)

    def solve_1d_fd(nx, nt):
        dx = L / (nx - 1)
        dt = T_end / nt
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T_end, nt + 1)

        P = p_exact(x, 0.0)
        r = alpha * dt / (dx**2)

        main_diag = (1.0 + 2.0 * r) * np.ones(nx)
        off_diag = -r * np.ones(nx - 1)

        main_diag[0] = 1.0
        main_diag[-1] = 1.0

        ab = np.zeros((3, nx))
        ab[0, 1:] = off_diag
        ab[1, :] = main_diag
        ab[2, :-1] = off_diag
        ab[0, 1] = 0.0
        ab[2, -2] = 0.0

        for n in range(nt):
            t_next = t[n + 1]
            rhs = P + dt * source_term(x, t_next)
            rhs[0] = p_exact(0.0, t_next)
            rhs[-1] = p_exact(L, t_next)
            P = solve_banded((1, 1), ab, rhs)

        exact_final = p_exact(x, T_end)
        l2_err = np.sqrt(np.mean((P - exact_final)**2))
        return l2_err

    # Refine grid and time step together to verify consistent error reduction
    err1 = solve_1d_fd(nx=21, nt=100)
    err2 = solve_1d_fd(nx=41, nt=400)
    err3 = solve_1d_fd(nx=81, nt=1600)

    # Invariant: error must monotonically decrease with resolution refinement
    assert err2 < err1, f"Error did not decrease with refinement: err1={err1:.3e}, err2={err2:.3e}"
    assert err3 < err2, f"Error did not decrease with refinement: err2={err2:.3e}, err3={err3:.3e}"
