# Scientific Verification Strategy: 7-Level Verification Hierarchy

## 1. Philosophical Framework

In computational reservoir engineering, software validation often suffers from circularity: simulation models with dozens of adjustable empirical parameters are "calibrated" against historical field data or commercial simulators (e.g., CMG GEM, ECLIPSE 300). When predictions match, the software is deemed "validated," even if underlying physical equations, unit conversions, or thermodynamic formulations contain severe mathematical or physical errors.

This project enforces the standard of the **American Society of Mechanical Engineers (ASME V&V 10 / V&V 20)**:
- **Verification**: The process of determining that a computational model implementation accurately represents the developer's conceptual description and specification ("solving the equations right").
- **Validation**: The process of determining the degree to which a model is an accurate representation of the real world from the perspective of the intended uses ("solving the right equations").

**Foundational Rule**: Verification must strictly precede validation. No software can be considered validated if its governing equations fail analytical verification, mass conservation, or dimensional consistency.

---

## 2. Seven-Level Verification Hierarchy

```
Level 7: Property-Based Invariant Metamorphic Testing (Hypothesis)
   ▲
Level 6: Manufactured Solutions (Method of Manufactured Solutions - MMS)
   ▲
Level 5: Grid & Time Refinement Convergence Studies (L1, L2, L_inf)
   ▲
Level 4: Independent Reference Solutions (Welge Buckley-Leverett, Arps)
   ▲
Level 3: Numerical & Algorithmic Verification (Picard residuals, stability)
   ▲
Level 2: Physical & Thermodynamic Invariants (Conservation, 2nd Law, EPA UIC)
   ▲
Level 1: Exact Mathematical Identities & Dimensional Consistency (SymPy, Pint)
```

### Level 1: Mathematical Identities & Dimensional Consistency
- **Methodology**: Exact symbolic verification using `sympy` and dimensional analysis using `pint`.
- **Target**: Derivations of Koval fractional flow, Welge tangent line equality, Arps decline rate integrals, and dimensional homogeneity of inflow equations.
- **Pass/Fail Standard**: Exact symbolic identity (simplification to 0) and zero dimensional residue ($[L^3/T]$, $[M/L/T^2]$, etc.).

### Level 2: Physical & Thermodynamic Invariants
- **Methodology**: Monotonicity checks, second-law constraints, and regulatory boundary limits.
- **Target**: Positivity of isothermal compressibility ($c_o > 0 \implies \partial B_o / \partial P < 0$), positive pressure-viscosity slope ($\partial\mu/\partial P > 0$), negative thermal expansion ($\partial\rho/\partial T < 0$), and geomechanical sandface pressure ceilings ($P \le 0.90 P_{\text{frac}}$).
- **Pass/Fail Standard**: Strict inequality preservation across the entire physical operating envelope.

### Level 3: Numerical & Algorithmic Verification
- **Methodology**: Discrete algebraic residual evaluation and time-step stability checks.
- **Target**: Explicit evaluation of algebraic residual $\|r(P^{n+1})\| = \|(V_p c_t + J_{\text{eff}}\Delta t)\Delta P - q_{\text{net}}\Delta t\|$, absence of numerical oscillations.
- **Pass/Fail Standard**: Residual norm within $10^{-6}$ and zero non-physical oscillations ($\Delta^2 P$ sign flips under smooth forcing).

### Level 4: Independent Reference Solutions
- **Methodology**: Cross-verification against independent, uncalibrated analytical solvers.
- **Target**: 1D Buckley-Leverett shock front front location via independent numerical Welge tangent construction; Arps cumulative recovery via closed-form analytical integration.
- **Pass/Fail Standard**: Relative error $< 1.0\%$ against analytical benchmarks without tuning.

### Level 5: Grid & Temporal Convergence Studies
- **Methodology**: Step refinement across multiple orders of magnitude ($\Delta t \to \Delta t/2 \to \Delta t/4$).
- **Target**: Observation of asymptotic convergence rates ($O(\Delta t)$, $O(\Delta x^2)$) and evaluation of discrete error norms ($L_1, L_2, L_\infty$).
- **Pass/Fail Standard**: Monotonic decrease in error norm and relative sensitivity $< 5\%$ between monthly and daily discretization.

### Level 6: Method of Manufactured Solutions (MMS)
- **Methodology**: Forcing an arbitrary, highly non-linear analytical field into the PDE and injecting the analytical residual as a source term $S(x,t)$.
- **Target**: 1D non-linear pressure diffusion equation:
  $$\phi c_t \frac{\partial P}{\partial t} - \frac{k}{\mu}\frac{\partial^2 P}{\partial x^2} = S_{\text{MMS}}(x,t)$$
- **Pass/Fail Standard**: Observed order of accuracy matches theoretical formal discretization order ($p \ge 1.8$ for 2nd order spatial central differences).

### Level 7: Metamorphic & Property-Based Testing
- **Methodology**: Property-based automated exploration using `hypothesis`.
- **Target**: Global invariants over tens of thousands of pseudo-random parameter combinations:
  - $0.0 \le RF \le \frac{1 - S_{wi} - S_{or}}{1 - S_{wi}}$
  - Cumulative produced carbon $\le$ Cumulative injected carbon
  - Recovery factor monotonically non-decreasing with injected volume in 1D piston drive.
- **Pass/Fail Standard**: Zero falsifying examples encountered across all hypothesis strategies.

---

## 3. Separation of Independent Verification from Calibration

To prevent "hidden calibration" (the practice of tuning empirical exponents until curves match CMG or SPE benchmarks), verification tests in `tests/scientific/` operate under three strict constraints:
1. **No Optimizer In the Loop**: Verification tests directly invoke physics and mathematics modules with explicit states, bypassing GA/PSO calibration loops.
2. **Zero Fitting Parameters**: Test assertions compare against first-principles physics (e.g., mass conservation, sympy identities), not fitted historical data.
3. **Immutability of Governing Equations During Audit**: In accordance with Rule 23, discovered scientific flaws are documented and cataloged, not silently patched or obscured.
