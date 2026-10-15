# Numerical & Algorithmic Verification: Solvers, Time-Stepping, & Stability

## 1. Scope & Numerical Architecture

Level 3 Numerical Verification evaluates the discretization accuracy, iterative solver convergence, and numerical stability of time integration algorithms used across the simulation pipeline.

In the active surrogate engine (`core/engine_surrogate/`), reservoir pressure depletion and build-up are modeled via a coupled 0D Tank Material Balance ODE solved using an implicit, damped Picard formulation.

---

## 2. Tank Pressure ODE Formulation & Residual Verification

### 2.1 Discrete Equation of Motion

The discrete volume-balance equation governing average reservoir pressure $P^{n+1}$ over a time step $\Delta t$ is:

$$V_p c_t \frac{P^{n+1} - P^n}{\Delta t} = q_{\text{inj,RB}}(P^{n+1}) - q_{\text{prod,RB}}(P^{n+1})$$

where:
- $V_p = \frac{\text{OOIP} \cdot B_o}{1 - S_{wi}}$ is the reservoir pore volume (RB).
- $c_t = c_o S_o + c_w S_w + c_f$ is the total system compressibility ($\text{psi}^{-1}$).
- $q_{\text{inj,RB}} = q_{\text{CO2,inj}} \cdot B_{g,\text{dynamic}} + q_{w,\text{inj}} \cdot B_w$ is the total downhole injection rate (RB/d).
- $q_{\text{prod,RB}} = q_o B_o + q_w B_w + q_{\text{gas}} B_g$ is the total downhole reservoir voidage rate (RB/d).

Linearizing well deliverability around current reservoir pressure with effective system injectivity/productivity index $J_{\text{eff}} = J_{\text{inj}} + J_{\text{prod}}$, the discrete update is:

$$\Delta P = \frac{q_{\text{net,IPR}} \cdot \Delta t}{V_p c_t + J_{\text{eff}} \cdot \Delta t}$$

### 2.2 Algebraic Residual Evaluation

The exact discrete residual $r(P^{n+1})$ is defined as:
$$r(P^{n+1}) = (V_p c_t + J_{\text{eff}}\Delta t)\Delta P - q_{\text{net,IPR}}\Delta t$$

**Verification Test**: [`tests/scientific/solver/test_solver_residuals.py::test_pressure_material_balance_discrete_residual`](file:///d:/rep/4.6/co2eor_optimizer/tests/scientific/solver/test_solver_residuals.py) evaluates $\|r(P^{n+1})\|_\infty$.
- **Observed Result**: $\|r(P^{n+1})\|_\infty \le 10^{-12}\text{ psi}$.
- **Status**: **VERIFIED**.

---

## 3. Numerical Stability & Monotonicity Envelopes

### 3.1 Time-Step Damping & Oscillation Prevention

To prevent unphysical sawtooth pressure oscillations under abrupt injection switching (e.g., WAG cycles), the formulation incorporates numerical damping:
- **Damping Term**: $J_{\text{eff}}\Delta t > 0$ acts as an implicit diagonal regularizer, ensuring the denominator strictly exceeds $V_p c_t$.
- **Courant-Friedrichs-Lewy (CFL) Condition**: In the 1D flow solvers (`core/unified_engine/core/time_stepper.py`), time-steps are restricted by:
  $$\Delta t \le \text{CFL} \cdot \min_i \left( \frac{\Delta x_i \phi_i S_{g,i}}{u_{g,i}} \right)$$
  with $\text{CFL} = 0.15$ for TVD flux-limited front tracking.

**Verification Test**: [`tests/scientific/numerical/test_timestep_stability.py::test_pressure_oscillation_under_dynamic_injection`](file:///d:/rep/4.6/co2eor_optimizer/tests/scientific/numerical/test_timestep_stability.py) tests high-rate injection (8,000 MSCFD) into an elastic reservoir.
- **Observed Result**: Pressure profile is positive, strictly bounded, and non-oscillatory.
- **Status**: **VERIFIED**.
