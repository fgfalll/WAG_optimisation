# V&V Testing Framework — D5

Source: `3D THMC_docs/Стратегія верифікації та валідації (V&V Testing Framework) для 3D THMC симулятора.md`
(507 lines, 56 988 bytes). **Design intent only.**

---

## 1. Testing philosophy

D5 opens by **rejecting simple unit tests and code coverage as reliability criteria**:

> "Rust code with **100 % code coverage** that runs without panics can generate catastrophic calculation errors."

Consequences stated:
- Primitive unit tests are "entirely incapable" of guaranteeing physical plausibility and numerical stability for stiffly coupled nonlinear PDE systems with private derivatives.
- Memory-safe code (no `unsafe`, no leaks, no data races) "can produce absolute physical absurdity".
- Conventional **95–100 %** line/branch coverage is a "false criterion creating an illusion of correctness".

> ⚠️ **CONF-21.** D5 rejects coverage as a gate and then proposes **no quantitative replacement**.
> The CI has budgets and tolerances but no coverage or mutation-score requirement. This is a genuine
> methodology gap, not a stylistic one.

### 1.1 Verification vs validation

| | **VERIFICATION** | **VALIDATION** |
|---|---|---|
| Question | "Are we solving the equations correctly?" | "Are the equations we solve correct?" |
| Checks | approximation check · convergence order $O(\Delta t)$ · mass conservation $< 10^{-12}$ · M-matrix and monotonicity | correspondence to physics · absence of phantom production / utilisation · SPE benchmarks (1..11) |

Declared fundamental invariants: component-wise mass conservation · boundedness of state variables ·
thermodynamic monotonicity of Gibbs minimisation · symmetry of spatial discrete operators.

---

## 2. The 6-level stack

```
Level 6  Stress tests of extreme scenarios
Level 5  Industrial SPE benchmarks (SPE 1/3/5/9/10/11)
Level 4  Convergence order and Fixed-Stress Split
Level 3  Thermodynamic monotonicity and EOS flash
Level 2  Invariant conservation audit (mass / energy)
Level 1  Analytical benchmarks (Buckley-Leverett)
```

**Stated gate logic (sequential):** Level 1 confirms discretisation accuracy on closed analytical tests
→ Level 2 audits invariants end-to-end → Level 3 checks thermodynamic consistency → Level 4 checks
convergence order and splitting → Level 5 validates on SPE → Level 6 checks extreme and emergency regimes.

---

## 3. Level 1 — Analytical benchmarks

| Benchmark | Physical process | Control metric | Permissible deviation |
|---|---|---|---|
| **Buckley–Leverett** | 1D two-phase displacement | shock-front position | **≤ 0.1 %** over length |
| **Terzaghi** | 1D poroelastic consolidation | pore-pressure dissipation `P(t)` | **≤ 0.05 %** |
| **Mandel** | 2D/3D poroelasticity, Mandel–Cryer effect | peak centre overshoot `ΔP_center` | **≤ 0.2 %** |
| **Sneddon / KGD** | hydraulic fracture | aperture profile `w_f(x)` | **≤ 0.5 %** |
| **Avdonin** | 1D/radial non-isothermal convection+conduction | temperature profile `T(x,t)` | **≤ 0.1 %** |

**Details.**

- **Buckley–Leverett (§2.1).** 1D immiscible two-phase, homogeneous, **no capillary pressure**, constant
  injection rate, connate water $S_{wi}$. Reference: **Welge construction** from the tangent to
  $f_g(S_g) = \dfrac{1}{1 + \dfrac{k_{ro}}{k_{rg}}\dfrac{\mu_g}{\mu_o}}$.
  Requires a **TVD/WENO5** scheme in the Rust module: high shock-front resolution, no artificial
  smearing, no non-physical overshoot outside $[S_{wi}, 1 - S_{or}]$.
- **Terzaghi (§2.2).** 1D consolidation of water-saturated rock under constant load, **one-sided
  drainage**; match the exponential Fourier-series analytic solution for `P(t)` and settlement.
- **Mandel (§2.2).** Rectangular specimen instantaneously compressed between rigid impermeable plates,
  free drainage at lateral boundaries. Must additionally reproduce the **Mandel–Cryer effect** —
  temporary pore-pressure **rise above initial** at specimen centre at early times.
  **Inf-Sup / LBB requirement:** equal-order `(P_1-P_1)` violates LBB and causes false spatial pressure
  oscillations at low permeability / undrained compaction; the scheme **must** use mixed
  **Taylor–Hood `P_2-P_1`**.
- **Sneddon / KGD (§2.3).** **Sneddon:** 3D penny-shaped fracture under uniform net pressure
  $P_{net}$, $w_f(x)\propto\sqrt{1-(x/L)^2}$. **KGD** = Kristianovich–Geertsma–de Klerk, finite-height
  planar deformation under viscous injection. Deviation of the satellite-computed aperture **and the
  near-tip stress–strain state** must not exceed **0.5 %**.
- **Avdonin (§2.4).** Cold-fluid injection into a hot reservoir; solution accounts for convection
  in-reservoir and conduction through caprock and floor. **No oscillation at the thermal front**;
  **exact thermal energy balance**.

> ⚠️ **CONF-22.** The **summary table's 4th column is shifted one column relative to the 3rd** in the
> source: "за довжиною" (over length) belongs to the BL tolerance. Anyone transcribing the table must
> re-derive the column alignment.

---

## 4. Level 2 — Conservation audits

### 4.1 Component-wise mass balance to machine precision

Scope: **each of $N_c$ filtration components**, checked **at every time step**:

$$\left|\sum_k M_i^{(k)}(t) + \int_0^t Q_{prod,i}(\tau)\,d\tau - \int_0^t Q_{inj,i}(\tau)\,d\tau - \sum_k M_i^{(k)}(0)\right| < 10^{-12}$$

**Explicit tolerance disambiguation (D5 lines 182–184):**

- For the **full 3D FVM nonlinear solver** (Newton-Raphson + Krylov), $10^{-12}$ is an "unambiguous
  mathematical convergence criterion at every time step".
- The threshold **≥ 99.9 % ($10^{-3}$) applies exclusively to simplified/coarse high-level surrogate
  diagnostic models**, and is "**categorically inadmissible inside the main numerical core**".

**Stated consequences of failure:** phantom oil/gas volumes · erroneous NPV calculations ·
Newton-Raphson divergence. Missing $10^{-12}$ also distorts CO₂ utilisation to **< 1 MSCF/STB instead
of the design 5–10 MSCF/STB**.

> [!CAUTION]
> **CONF-23 — direct conflict with this repository's own gates.** D5 forbids $10^{-3}$ inside the core.
> But the **live engine is a surrogate**, and the repository's own acceptance criteria are
> `> 99.9 %` (`agent_wiki/README.md` invariant 14, `utils/run_exporter.py`), and D2 §6.2 criterion 1
> itself requires **> 99.9 %**. **Three documents in the same set disagree about which tolerance
> applies to which tier.** This must be resolved before adoption — see
> [`evaluation_plan.md`](evaluation_plan.md) §3.

### 4.2 Saturation positivity and composition closure

Must hold **on every Newton iteration**:
1. $S_\alpha \in [0,1]$ for $\alpha \in \{o,w,g\}$
2. $\sum_i z_i = 1.0$ and $\sum_\alpha S_\alpha = 1.0$
3. **Clamping is categorically forbidden** — clamping negative saturations to `0.0` or `>1.0` to `1.0`
   after the linear solve "destroys the conservativity of the discrete scheme". Membership in $[0,1]$
   must be guaranteed by the discretisation formulation and exact Jacobian computation.

> ⚠️ **CONF-24.** Forbidding clamping outright is incompatible with the iterative Newton schemes
> everywhere else in the set (line search, adaptive timestep cut, primary-variable substitution in
> D5 §7.2). Either the discretisation must be positivity-preserving by construction, or bounds handling
> must be specified. Not addressed.

### 4.3 MPFA-O operator properties

Discrete filtration operator must form an **M-matrix**: non-negative diagonal, non-positive
off-diagonal, diagonal dominance. Audit positive definiteness and **symmetry** of the diffusion
operator. Violation produces **non-physical negative pressures near wells** and **false reverse flow
against the pressure gradient**.

---

## 5. Level 3 — EOS flash monotonicity

Target: phase-split (flash) and VLE on cubic **Peng-Robinson** and **Soave–Redlich–Kwong**.

### 5.1 Gibbs free-energy monotonicity

Flash iteration must monotonically decrease $G$:

$$G^{(k+1)}(x,y) < G^{(k)}(x,y)$$

Iterative descent from *Iteration 0 (single-phase state `z`)* through *Iteration 1* to
*Global minimum `G(x,y)` (two-phase equilibrium L + V)*.

**Stated failure mode:** landing in a local minimum, or non-monotone oscillation of $G$, causes the
nonlinear solver to **cycle** and $\varphi_i^L,\varphi_i^V$ to lose physical meaning → Newton divergence.

### 5.2 Michelsen TPD multi-start test

$$TPD(y) = \sum_{i=1}^{N_c} y_i\left(\ln y_i + \ln\varphi_i(y) - \ln z_i - \ln\varphi_i(z)\right)$$

```rust
// Нульове виділення пам'яті у внутрішньому циклі SIMD Flash
pub fn evaluate_tpd_simd(
    z: &[f64],
    y_trial: &[f64],
    p: Pascal,
    t: Kelvin
) -> f64 {
    let mut tpd = 0.0;
    for i in 0..z.len() {
        tpd += y_trial[i] * (y_trial[i].ln() + fugacity_coeff_gas(y_trial, p, t, i).ln()
               - z[i].ln() - fugacity_coeff_mixture(z, p, t, i).ln());
    }
    tpd
}
```

**Search strategy:** testing at **10 000 randomly generated points** in $P$-$T$-$z$ space; stationary
points sought by solving $\nabla TPD(y) = 0$; trial seeds from **several** initial approximations —
**Wilson's K-factor estimates**

$$K_i = \frac{P_{ci}}{P}\exp\left(5.37(1+\omega_i)\left(1 - \frac{T_{ci}}{T}\right)\right)$$

pure-component vectors $e_i$, and their mixtures. If **$TPD(y) < 0$ at at least one stationary point**,
the single-phase state is unstable and the mixture must split — this prevents missing "shadow"
two-phase regions near retrograde-condensation points.

**Unit newtypes:**

```rust
# [derive(Debug, Clone, Copy, PartialEq)]
pub struct Pascal(pub f64);

# [derive(Debug, Clone, Copy, PartialEq)]
pub struct MoleFraction(pub f64);

impl MoleFraction {
    pub fn new(val: f64) -> Result<Self, NumericalDivergenceError> {
        if (0.0..=1.0).contains(&val) {
            Ok(MoleFraction(val))
        } else {
            Err(NumericalDivergenceError::UnphysicalMoleFraction(val))
        }
    }
}
```

> ⚠️ **CONF-25.** The listing claims `std::simd` with **AVX-512** and zero heap allocation, but the
> shown loop is scalar. `Kelvin`, `NumericalDivergenceError`, `fugacity_coeff_gas`,
> `fugacity_coeff_mixture` are referenced and **never declared**. `NumericalDivergenceError` has
> exactly one variant named and no `enum` definition anywhere in D5.

### 5.3 Critical-point scan

Hysteresis-free sweep of $P$ and $T$ **through the critical point and back**. Three criteria:
1. $\varphi_i^L / \varphi_i^V \to 1.0$
2. all phase functions (density, viscosity, compressibility) remain monotone and **C¹**-continuous
3. **excluded**: critical errors, overflow, `NaN`/`Inf` indeterminacies

---

## 6. Level 4 — Temporal convergence and operator splitting

### 6.1 Numerical temporal convergence order

| Quantity | Exact statement |
|---|---|
| Method | systematic time-step refinement study |
| Error norm | $\lVert E(\Delta t)\rVert = \lVert u_{num} - u_{exact}\rVert$ |
| Refinement sequence | **$\Delta t,\ \Delta t/2,\ \Delta t/4,\ \Delta t/8$** |
| Slope estimator | $\alpha = \dfrac{\log\lVert E(\Delta t_1)\rVert - \log\lVert E(\Delta t_2)\rVert}{\log\Delta t_1 - \log\Delta t_2}$ |
| Backward Euler | **$\alpha \approx 1.0$** → $O(\Delta t)$ |
| Crank–Nicolson / Radau IIA | **$\alpha \approx 2.0$** → $O(\Delta t^2)$ |

### 6.2 Fixed-stress split stability

Coupling by **Fixed-Stress Split**; stabilisation added to the filtration matrix diagonal:

$$S_{stab} = \frac{\alpha_B^2}{K_{dry}}$$

**Convergence criterion:** the sequential staggered scheme must reach full convergence in displacement
and pressure norm within **$N_{iter} \le 3\text{–}5$ subiterations per time step**
(body text; the CI table states **$N_{iter} \le 5$**).

### 6.3 Streamline ↔ FVM remapping

1. Remapping component masses between streamline and FVM grids must preserve exact total mass to **$10^{-14}$**.
2. Grid-orientation audit on symmetric **five-spot** patterns: breakthrough results **must not change when the computational grid is rotated by 45°**.

> [!CAUTION]
> **CONF-26 — spatial convergence is claimed but never tested.** D5 §1.2 promises adherence to declared
> **"temporal and spatial"** convergence order and line 79 restates it, but Level 4 is **temporal only**
> ($\mathcal{O}(\Delta t)$ throughout). There is **no** spatial grid-refinement study, **no**
> order-of-accuracy target in $h$, and **no** truncation-error quantification (no $C\Delta h^p$ leading
> constant). MPFA-O + WENO + Taylor–Hood spatial verification is therefore **entirely absent**.

---

## 7. Level 5 — SPE benchmark suite

| Benchmark | Physical problem | Key features | Control metric | Tolerance |
|---|---|---|---|---|
| **SPE 1** | 3D black-oil gas injection | 3D gas displacement, gravity segregation | GOR dynamics, `P(t)` | **≤ 0.5 %** |
| **SPE 3** | gas-condensate reservoir | retrograde condensation, liquid dropout | residual condensate, condensate production | **≤ 1.0 %** |
| **SPE 5** | compositional WAG | alternating gas/water injection | `RF`, produced-gas composition | **≤ 1.5 %** |
| **SPE 9** | water coning | high heterogeneity, fast coning | water breakthrough `t_bt`, water-cut dynamics | **≤ 0.8 %** |

**SPE 10 — large-scale upscaling.** Model: **$1.1\times10^6$ cells**, permeability contrast **up to 6 orders**
between layers. Four requirements on the Rust core:
1. lock-free parallel Jacobian assembly using **`rayon`**
2. **linear scalability** of sparse direct solvers (`faer` / `nalgebra` or `PETSc-bindings`) to **64+ cores**
3. **absence of dynamic heap allocations** inside the inner Newton loop
4. exact preservation of global rates and component masses under large-scale upscaling

**SPE 11 — geomechanics & CO₂ sequestration.** Three processes: (1) CO₂ dissolution in formation water
with density and viscosity change; (2) CO₂ phase state (supercritical / liquid / gas-like);
(3) geomechanical **caprock integrity** under increasing reservoir pressure — evaluate onset of critical
stresses to prevent non-physical technogenic filtration through impermeable strata.

> [!NOTE]
> **This repository already carries SPE 5** — see
> [`../validation/benchmarks.md`](../validation/benchmarks.md) §3 (`validation/spe5_config.py`,
> $7\times7\times3=147$ blocks, 2 100 ft, $10^\circ$ dip, $\phi=0.35$,
> $K = 500/50/200$ mD, 4 000 psia, $S_{wi}=0.16$) — plus CMG GEM cases `gmflu001`–`gmflu003` read via
> `h5py` in `validation/sr3_reader.py`. **D5 names no comparison simulator and no reference dataset
> provider.** This is the single biggest practical asset the repository already holds.

---

## 8. Level 6 — Adversarial physical stress tests

| § | Test | Scenario | Required core reaction |
|---|---|---|---|
| 7.1 | **t = 0⁺ impulse** | injection from zero to $Q_{max}$ instantaneously | adaptive **sub-stepping** immediately reduces $\Delta t$; **L-stable** integrator damps pressure oscillation without aborting |
| 7.2 | **Phase boundary appearance** | $S_g = 0 \to S_g > 0$, or $S_o > 0 \to S_o = 0$; free gas appears at $P < P_{sat}$ | **Primary Variable Substitution**: switch primary variable from mole fraction $z_i$ to phase saturation $S_g$ on appearance, back to $z_i$ on disappearance; **C¹-continuity of Jacobian elements** at the transition boundary; explicit requirement of **no Newton-Raphson cycling** |
| 7.3 | **Verma–Pruess catastrophic clogging** | salt/solid precipitation → $\phi \to \phi_c$ | $k(\phi) = k_0\left(\frac{\phi-\phi_c}{1-\phi_c}\right)^{\eta}$, $\phi_c \in [0.01, 0.05]$, $\eta \approx 3.0\text{–}8.0$; as $\phi\to\phi_c$, $k\to 0$, cell face blocked, controller shrinks $\Delta t\to 0$, **no division by zero** |
| 7.4 | **Tubing hydrate blockage** | wellbore $P$-$T$ profile exits into clathrate envelope (van der Waals–Platteeuw) | wellbore satellite reduces effective hydraulic diameter $D_{tubing}\to 0$; recomputes well index (WI) and wall-roughness coefficient, throttling inflow; reservoir core handles growing back-pressure, **redirecting filtration flows without system panic or simulation stoppage** |

---

## 9. CI/CD pipeline and budgets

```
[ Git Push / PR ]
   |
1. Compilation & static analysis (Cargo Check / Clippy / Unit)
   |
2. Levels 1-3: analytical benchmarks, mass balance (<10^-12), TPD flash monotonicity   [ < 3 minutes ]
   |
3. Levels 4-5: convergence order O(dt), Fixed-Stress split, SPE 1/3/5/9                [ < 45 minutes ]
   |
4. Levels 5-6: SPE 10 (1.1 M cells), SPE 11, stress tests                             [ < 3 hours ]
   |
[ Deploy to Production Core ]
```

| Suite | Levels | Budget | Checks |
|---|---|---|---|
| **PR Smoke** | 1–3 | **< 3 min** | complete absence of `panic!`; exact analytic solutions; mass conservation **< $10^{-12}$**; Michelsen TPD over **10 000** points |
| **Nightly Regression** | 4–5 | **< 45 min** | temporal orders **$O(\Delta t)$** and **$O(\Delta t^2)$**; Fixed-Stress stability (**$N_{iter}\le 5$**); SPE 1, 3, 5, 9 |
| **Pre-Release Stress & Scale** | 5–6 | **< 3 h** | SPE 10 (**$1.1\times10^6$** cells) on multi-core; SPE 11 geomechanics; all adversarial stress tests |

**Panic policy:** internal solver module compiled with **`panic = "abort"`**; any numerical divergence
caught by strict `Result<T, NumericalDivergenceError>`, guaranteeing deterministic test termination
without unplanned process crashes.

**Anti-pattern reference thresholds (D5 §1.1):** CO₂ utilisation **< 1 MSCF/STB** (wrong) vs design
**5–10 MSCF/STB**; NPV penalty `result *= breakthrough_impact` / artificial drop to **$-1.0\times10^{12}$**;
rate clamped to **5 000 MSCFD** instead of the **20 000–60 000 MSCFD** range.

---

## 10. Consolidated tolerance ledger

| # | Quantity | Value |
|---|---|---|
| 1 | Buckley–Leverett shock-front position | **≤ 0.1 %** |
| 2 | Terzaghi pore-pressure dissipation | **≤ 0.05 %** |
| 3 | Mandel centre overshoot | **≤ 0.2 %** |
| 4 | Sneddon/KGD aperture + near-tip stress | **≤ 0.5 %** |
| 5 | Avdonin temperature profile | **≤ 0.1 %** |
| 6 | Component-wise mass balance (full 3D FVM) | **< $10^{-12}$** |
| 7 | Mass-balance tolerance allowed in coarse surrogates only | **≥ 99.9 % ($10^{-3}$)** |
| 8 | Saturation bounds / composition closure | $S_\alpha\in[0,1]$; $\sum z_i=1$; $\sum S_\alpha=1$ |
| 9 | Streamline↔FVM remap mass preservation | **$10^{-14}$** |
| 10 | Grid rotation invariance | **45°** |
| 11 | Backward Euler log-log slope | **$\alpha\approx1.0$** |
| 12 | Crank–Nicolson / Radau IIA slope | **$\alpha\approx2.0$** |
| 13 | Refinement sequence | $\Delta t,\ \Delta t/2,\ \Delta t/4,\ \Delta t/8$ |
| 14 | Fixed-Stress split subiterations | **$N_{iter}\le3\text{–}5$** |
| 15 | SPE 1 | **≤ 0.5 %** |
| 16 | SPE 3 | **≤ 1.0 %** |
| 17 | SPE 5 | **≤ 1.5 %** |
| 18 | SPE 9 | **≤ 0.8 %** |
| 19 | SPE 10 | **$1.1\times10^6$ cells**, contrast 6 orders, **64+ cores** |
| 20 | Michelsen TPD sample size | **10 000** |
| 21 | Critical-point limit | $\varphi_i^L/\varphi_i^V\to1.0$; monotone, C¹ |
| 22 | Verma–Pruess $\phi_c$ | **[0.01, 0.05]** |
| 23 | Verma–Pruess $\eta$ | **≈ 3.0–8.0** |
| 24 | CI budgets | **< 3 min / < 45 min / < 3 h** |
| 25 | Panic policy | **`panic = "abort"`** + `Result<T, NumericalDivergenceError>` |
| 26 | Wilson K-value constant | **5.37** |
| 27 | Rejected coverage "gold standard" | 95–100 % |

---

## 11. Gaps D5 does not close

| # | Gap | Note |
|---|---|---|
| G-01 | **No reference/expected values.** Every criterion is a *relative* tolerance against an analytic or "reference" solution that is never supplied numerically. **Zero absolute expected numbers.** | Nothing is executable or assertable. |
| G-02 | **No test IDs, no test function names, no file or module paths.** Tests are prose titles only. | Not traceable. |
| G-03 | **No reference-solution provider.** "еталону" (the reference) is undefined. No commercial simulator, no published dataset, no DOI. | — |
| G-04 | **No manufactured solutions.** Zero mentions. | Standard in modern V&V; absent here. |
| G-05 | **No spatial convergence study.** See **CONF-26**. | — |
| G-06 | **No truncation-error quantification.** | — |
| G-07 | **No field data.** Only the generic phrase "core laboratory tests, analytic solutions and field development data". No field, no dataset, no history-matching target, no acceptance threshold. | — |
| G-08 | **No test-runner tooling.** No crate, no `cargo test`, no `criterion`/`proptest`/`insta`/`approx`, no file path, no CI system named. | — |
| G-09 | **No quantification of the "core" definition.** "The core" is asserted by tier, never defined by interface or component. | Feeds **CONF-23**. |
| G-10 | **10 000 random TPD points with no seed** → the test is not reproducible. | Feeds the reproducibility gap shared with D7. |