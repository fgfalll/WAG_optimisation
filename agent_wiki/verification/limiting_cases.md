# Limiting Cases: Asymptotic & Degenerate Boundary Matrix

## 1. Role of Limiting Case Analysis

A rigorous scientific simulator must remain mathematically and physically well-behaved at the asymptotic boundaries and degenerate extremes of its parameter space. Testing limiting cases uncovers hidden division by zero, unphysical negative rates, non-monotonicity, and violation of thermodynamic bounds.

---

## 2. Limiting Case Test Matrix

The following matrix documents the limiting cases implemented and verified in `tests/scientific/limiting_cases/`:

| Boundary / Limiting Condition | Physical Expectation | Computational Behavior | Verification Test File | Status |
|:---|:---|:---|:---|:---|
| **Zero Permeability ($k \to 0$)** | Zero flow; no injectivity, no recovery ($q_{\text{inj}} = 0$, $RF = 0$) | Clamps rate to 0 without division by zero or NaN | `test_zero_permeability.py` | **VERIFIED** |
| **Zero Injection Rate ($q_{\text{inj}} = 0$)** | Pure primary depletion; zero EOR recovery, zero CO₂ storage | Recovery matches primary depletion; storage = 0 | `test_zero_injection.py` | **VERIFIED** |
| **Infinite Time ($t \to \infty$)** | Cumulative production converges asymptotically to EUR ($N_p \le N_{\text{mobile}}$) | Production rate decays monotonically to 0; EUR bounded | `test_infinite_time_eur.py` | **VERIFIED** |
| **Deep Immiscible Limit ($P \ll \text{MMP}$)** | Miscibility weight $\omega \to 0$; recovery governed by immiscible water/gas flood | $\omega = 0.0$; recovery transitions to immiscible baseline | `test_miscibility_limits.py` | **VERIFIED** |
| **Full Miscible Limit ($P \gg \text{MMP}$)** | Miscibility weight $\omega \to 1.0$; zero interfacial tension ($\sigma \to 0$) | $\omega = 1.0$; recovery governed by miscible displacement | `test_miscibility_limits.py` | **VERIFIED** |
| **Unit Mobility Ratio ($M = 1.0$)** | Stable piston displacement without viscous fingering | Discontinuous step in Craig areal sweep correlation | `test_singularity_and_overflow.py` | **SCIENTIFIC FLAW (SCI-FLAW-16)** |
| **Infinite Heterogeneity ($V_{\text{DP}} \to 1.0$)** | Immediate gas breakthrough; recovery efficiency collapsed | Heterogeneity exponent clamped to prevent IEEE overflow | `test_singularity_and_overflow.py` | **VERIFIED** |
| **Zero Drawdown ($\Delta P \to 0$)** | Zero well production ($q_o = 0$) | Smooth linear decrease to zero without singular behavior | `test_wellbore_drawdown_limits.py` | **VERIFIED** |

---

## 3. Detailed Limiting Case Audits

### 3.1 Zero Permeability Limit ($k = 0$)
- **Physical Test**: Injectivity $J_{\text{inj}} \propto k$ and productivity $J_{\text{prod}} \propto k$. If $k = 0$, fluid cannot move through porous media.
- **Result**: `tests/scientific/limiting_cases/test_zero_permeability.py` verifies that injectivity, production rate, and cumulative recovery evaluate cleanly to $0.0$ without raising unhandled exceptions or computing `NaN`.

### 3.2 Deep Immiscible vs Fully Miscible Asymptotes
- **Physical Test**: The miscibility weight function $\omega(P)$ transitions from immiscible displacement ($\omega = 0$) to fully miscible displacement ($\omega = 1$):
  $$\omega(P) = \begin{cases} 0 & P < P_{\text{min}} \\ \frac{P - P_{\text{min}}}{\text{MMP} - P_{\text{min}}} & P_{\text{min}} \le P < \text{MMP} \\ 1 & P \ge \text{MMP} \end{cases}$$
- **Result**: `tests/scientific/limiting_cases/test_miscibility_limits.py` confirms that $\omega(1000) = 0.0$ and $\omega(5000) = 1.0$, and the bounds $[0, 1]$ are strictly respected.
