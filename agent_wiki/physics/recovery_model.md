# Recovery Models & PhD Hybrid Formulation

## 1. Overview of Recovery Models

The recovery model calculates the ultimate fraction of Original Oil In Place that can be produced ($RF \in [0.0, 1.0]$).

The codebase provides 5 recovery models in [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py):

| Model Name | Class | Governing Physics | Use Case |
| :--- | :--- | :--- | :--- |
| `miscible` | `MiscibleSurrogate` | Koval (1963) heterogeneity + Todd-Longstaff mixing | High-pressure floods well above MMP |
| `immiscible` | `ImmiscibleSurrogate` | Buckley-Leverett (1942) + Craig areal + Johnson vertical | Floods below MMP |
| `buckley_leverett` | `BuckleyLeverettSurrogate` | Direct 1D Welge tangent shock construction | Benchmark comparisons |
| `hybrid` | `HybridSurrogate` | Sigmoidal interpolation between miscible and immiscible | Standard transition modeling |
| `phd_hybrid` | `PhDHybridSurrogate` | Thermodynamic weighting + HCPVI mass balance constraint | **Default / Authoritative PhD Engine** |

---

## 2. Deep Dive: `PhDHybridSurrogate` & Sigmoidal Recovery Formulation

`PhDHybridSurrogate` is the **authoritative primary innovation** of this research platform. It mathematically resolves the classic "miscibility cliff" (a step change / non-differentiable jump in recovery and mobility ratio at MMP that stalls gradient and Bayesian optimizers) using a continuous, smooth, infinitely differentiable hyperbolic tangent weighting function.

### A. Mathematical Specification of the Sigmoidal Model

1. **Thermodynamic Miscibility Weight ($\omega(P)$)**:
   $$\omega(P) = 0.5 \cdot \left[ 1 + \tanh\left( \beta \cdot \left( \frac{P}{P_{MMP}} - \alpha_{\text{eff}} \right) \right) \right]$$
   Where:
   - $P$: Dynamic reservoir pressure (psia).
   - $P_{MMP}$: Minimum Miscibility Pressure (psia), calculated via Cronquist or Yellig-Metcalfe correlations.
   - $\alpha_{\text{eff}} = \alpha_{\text{base}} + \lambda_{C7+} \cdot (z_{C7+} - 0.30)$: Effective transition midpoint adjusted for heavy component fraction.
   - $\beta = \frac{4.394}{\Delta P_r}$: Transition sharpness governing the width of the near-miscible zone ($\Delta P_r \approx 0.10 - 0.15$).

2. **Differentiable Recovery Factor Interpolation**:
   $$RF_{\text{ultimate}}(P, t_D) = \omega(P) \cdot RF_{\text{miscible}}(t_D) + (1 - \omega(P)) \cdot RF_{\text{immiscible}}(t_D)$$
   Where:
   - $RF_{\text{miscible}}(t_D)$: Koval (1963) unstable miscible displacement with Todd-Longstaff effective viscosity mixing.
   - $RF_{\text{immiscible}}(t_D)$: Buckley-Leverett (1942) fractional flow displacement with two-phase Corey relative permeability and Craig-Johnson areal/vertical sweep.

3. **Smooth Todd-Longstaff Viscosity Bridging**:
   Rather than applying an arbitrary conditional cliff (`if omega > 0.01`), the Todd-Longstaff effective phase viscosities bridge smoothly:
   $$\mu_{ge} = \mu_{\text{mix}}^{\omega \cdot \omega_{TL}} \cdot \mu_g^{1 - \omega \cdot \omega_{TL}}$$
   $$\mu_{oe} = \mu_{\text{mix}}^{\omega \cdot \omega_{TL}} \cdot \mu_o^{1 - \omega \cdot \omega_{TL}}$$
   Where $\mu_{\text{mix}} = \left( 0.5 \mu_g^{-0.25} + 0.5 \mu_o^{-0.25} \right)^{-4}$.

4. **Capillary Desaturation Curve & Dynamic $S_{or}$**:
   Dynamic Interfacial Tension ($\sigma_{go}$) scales continuously with $\omega(P)$:
   $$\sigma_{go}(P) = \sigma_{\text{imm}} \cdot (1 - \omega(P)) + \sigma_{\text{mis}} \cdot \omega(P)$$
   $$N_c(P) = \frac{\mu_{\text{inj}} \cdot u}{\sigma_{go}(P)}$$
   $$S_{or}(N_c) = S_{or,\text{imm}} \cdot \left( \max\left(1.0, \, \frac{N_c}{N_{c,\text{crit}}}\right) \right)^{-m}$$

---

## 3. Dynamic Mass Balance Enforcement & Pore Volume Scaling

The recovery factor is strictly bounded by cumulative Hydrocarbon Pore Volumes Injected (HCPVI) to prevent unphysical recovery under solvent-deficient conditions:
$$RF(t) = RF_{\text{ultimate}}(P) \cdot \left( 1 - \exp\left(-\frac{\text{HCPVI}(t)}{\tau}\right) \right)$$
Where:
- $\text{HCPVI}(t) = \frac{\text{Cumulative Injected Solvent Volume (RB)}}{V_{p,\text{HC}}}$.
- Prevents the optimizer from claiming 35%+ recovery when $\text{HCPVI} < 0.20$.
- Implementation: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py).

---

## 4. Finite-Difference Numerical Gradients

For gradient-based optimization routines (BFGS, L-BFGS-B), `PhDHybridSurrogate.calculate_gradient(**params)` calculates partial derivatives via central and forward finite differencing:
$$\frac{\partial RF}{\partial P} \approx \frac{RF(P \cdot (1 + h_{\text{rel}})) - RF(P \cdot (1 - h_{\text{rel}}))}{2 \cdot P \cdot h_{\text{rel}}}$$
With $h_{\text{rel}} = 0.001$ ($0.1\%$).
- Implementation: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L894-L942).

---

## 5. Dynamic Degree of Miscibility Output (`miscibility_degree_profile`)

A core operational advantage of the Sigmoidal model is that it calculates the exact **degree of miscibility** $\omega(P_t) \in [0.0, 1.0]$ at each simulation time step:
- Fully immiscible: $\omega \approx 0.0$ ($P \ll P_{MMP}$)
- Transition / Near-miscible: $0.1 < \omega < 0.9$ ($P \approx P_{MMP}$)
- Fully miscible: $\omega \approx 1.0$ ($P \gg P_{MMP}$)

### Result Dictionary Keys
`SurrogateEngine.evaluate_scenario()` exposes these metrics directly for UI analysis and optimization objectives:
- `miscibility_degree_profile`: 1D NumPy array of $\omega(P_t)$ across each timestep in `time_vector`.
- `average_miscibility_degree`: Time-weighted average scalar degree of miscibility $\bar{\omega} = \frac{1}{T} \int_0^T \omega(P_t) dt$.
- Methods: `PhDHybridSurrogate.get_miscibility_weight(pressure, mmp, c7_plus)` and `PhDHybridSurrogate.get_last_miscibility_weight()`.

