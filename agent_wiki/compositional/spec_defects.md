# Specification Defects to Fix Before M1

The `CONF-*` conflicts in [`../thmc/conflict_and_gap_register.md`](../thmc/conflict_and_gap_register.md)
were written against a design set intended for adoption. For a **new engine**, the subset below becomes
**specification defects** — errors in the document that will be transcribed into code if not fixed first.

This page lists only what must be resolved **before M1**, or before the milestone it is attached to.
The rest of the register stays relevant for **M7a–M7h** (in scope, not yet specified — see
[`build_plan.md`](build_plan.md) M7) and for the design set's own record.

---

## 1. Before M1 — gas fractional flow closure (**CONF-01**, corrected 07-10-2026)

**The sign is correct — do not "fix" it.** The design formula agrees with the classical
$f_g = \frac{M}{1+M}$ to 4 dp. The defect is the *closure*:

$$\frac{1-S_w-S_g}{S_g-S_{gc}} = \frac{S_o}{S_g-S_{gc}} \quad \text{stands in for } \frac{k_{ro}}{k_{rg}}$$

| # | Defect | Required correction |
|---|---|---|
| a | **No $S_{or}$** — measured $f_g = 0.800$ at $S_o = S_{or}$; reaches 1.0 only at $S_o=0$ | Numerator must be $(S_o - S_{or})$, normalised |
| b | **Linear, not Corey** | $\dfrac{k_{ro}}{k_{rg}} = C\cdot\dfrac{(S_o-S_{or})^{n_o}}{(S_g-S_{gc})^{n_g}}$ — **separate exponents on separate saturations** |
| c | **No water term** | Three-phase: $f_g = \lambda_g/(\lambda_g+\lambda_o+\lambda_w)$. D1 §6.1 already *claims* Stone I/II |
| d | **Unguarded at $S_g\to S_{gc}$** | Uses $1/(S_g-S_{gc})$; Corey gives $1/k_{rg}\propto (S_g-S_{gc})^{-n_g}$. Add a numerical guard |

> **Spec requirement for M3:** the fractional-flow closure must be **declared** in one place, with $S_{or}$,
> $S_{wi}$, $S_{gc}$, $S_{gr}$, the Corey exponents, and the water term — and must be **swept** over the
> mobility ratio, not tested at a single point. See
> [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) §4 for the notation
> discipline this prevents recurring.

**Do not adopt the reservoir engineer's replacement verbatim** — it applies a single combined exponent
$n_o/n_g$ to a saturation *ratio*, which is a defensible approximation but is **not** Corey and must be
documented as such.

---

## 2. Before M1 — thermodynamics is unspecified (**CONF-31**, **CONF-10**, **CONF-29**, **CONF-49**)

| ID | Gap | Required before M1 |
|---|---|---|
| **CONF-31** | **No coefficient for any named correlation.** Standing, Glaso, Beggs–Robinson, Joback-Reid, PPR78, Huron–Vidal, QSPR, Karakas–Tarik are all name-only | Source a **cited** reference dataset. **No critical property may be invented** |
| **CONF-10** | VT shift $s=f((M\omega)^{-1})$ — correlant undefined, no fit supplied | Fit the VT coefficients to reference density data; record the fit and its residuals |
| **CONF-29** | `bip_matrix` is **2 × 3 for 6 components**; a valid symmetric set needs **21** unique $k_{ij}$ — **15 absent** | Specify all 21, or specify the regression that produces them from a named dataset |
| **CONF-49** | The set *declares* MAPD density error **3–9 %** for PR/SRK | Set the **M1 gate at ≤ 2 %** and treat 3–9 % as a reason to use VT or PC-SAFT |

**✅ M1 BLOCKER RESOLVED 07-10-2026.** Both unidentifiable citations are now resolved, and **both DOIs
were fetched and verified** — authors, journal, volume and pages all match.

### 2a.1 Abudour et al. (2014) — the ">900 binary system" QSPR table

> Abudour, A. M., Mohammad, S. A., Robinson, R. L. Jr., & Gasem, K. A. M. (2014).
> *Generalized binary interaction parameters for the Peng–Robinson equation of state.*
> **Fluid Phase Equilibria**, Vol. 383, pp. 156–173.
> DOI: [`10.1016/j.fluid.2014.10.006`](https://doi.org/10.1016/j.fluid.2014.10.006) — **verified**

| Property | Value |
|---|---|
| Systems | **916 low-pressure binary VLE systems**, 10+ functional-group categories |
| Output | Temperature-independent $k_{ij}$ for **PR**, generalised via QSPR |
| Stated accuracy | PR-QSPR bubble-point predictions **within 2×** of direct experimental $k_{ij}$ regressions |
| Use | **Fallback** QSPR table where no experimental VLE regression exists |

> ⚠️ **Read that accuracy statement carefully before relying on it.** "**within 2×**" is a *factor-of-two*
> band, not a percentage. Since $k_{ij}$ values are typically $O(0.05\text{–}0.2)$, a factor of two permits
> **~100 % relative error on the interaction parameter**. Acceptable as a **fallback**; **not** adequate as
> the primary source for an engine targeting **≤ 2 %** density MAPD and **≤ 0.1 %** Buckley–Leverett front
> position.
>
> **M1 requirement:** every $k_{ij}$ carries a **`source`** field — experimental / QSPR-fallback /
> default. **An un-sourced $k_{ij}$ is a defect**, not a default. Schema requirement, not a nicety
> ([`data_architecture.md`](data_architecture.md) §4.2).

### 2a.2 Baled et al. (2012) — the HTHP density and VT database

> Baled, H., Enick, R. M., Wu, Y., McHugh, M. A., Burgess, W., Tapriyal, D., & Morreale, B. D. (2012).
> *Prediction of hydrocarbon densities at extreme conditions using volume-translated SRK and PR equations
> of state fit to high temperature, high pressure PVT data.*
> **Fluid Phase Equilibria**, Vol. 317, pp. 65–76.
> DOI: [`10.1016/j.fluid.2011.12.027`](https://doi.org/10.1016/j.fluid.2011.12.027) — **verified**

| Property | Value |
|---|---|
| Pressure | **7 – 276 MPa** (1 000 – 40 000 psi) |
| Temperature | **278 – 533 K** (40 – 500 °F) |
| Coverage | Single-phase liquid densities, **17 pure hydrocarbons** — $n$-alkanes $C_1$…$n\text{-}C_{40}H_{82}$, cycloalkanes, aromatics — plus binary hydrocarbon mixtures |
| Output | Temperature-dependent **VT parameters** for HTHP VT-PR and VT-SRK, correlated to $(M\omega)^{-1}$ |
| Stated AARD | Density MAPD **1–2 %** (VT-SRK), **1–4 %** (VT-PR) |

> ✅ **This is the primary M1 source.** Its envelope (7–276 MPa, 278–533 K) **matches the design set's
> stated HTHP range exactly**, and its **1–2 %** VT-SRK figure is *better* than the **3–9 %** the design
> set declares for bare PR/SRK (**CONF-49**).
>
> **Consequence: volume translation is not optional.** It is the difference between passing and failing
> the M1 density gate. Bare PR or bare SRK will not meet ≤ 2 % over this envelope.
>
> ⚠️ It is a **density** dataset — it does **not** supply $k_{ij}$. Those still need Abudour (2014) for
> light/non-hydrocarbon pairs, plus experimental regression for the CO₂–hydrocarbon pairs that dominate
> CO₂-EOR.

### 2a.3 M1 ingestion path

| Item | Target | Requirement |
|---|---|---|
| $k_{ij}$ fallback | Thermodynamic initialisation — defaults for non-hydrocarbon and light-hydrocarbon pairs with no experimental regression | ⚠️ Every value carries `source` |
| VT coefficients | HTHP volume-translation solver, removing density underprediction above 10 000 psi | ⚠️ VT is mandatory, not optional |
| Critical properties | $T_c$, $P_c$, $\omega$, MW per component — sourced, **never invented** | `source` field |

---

## 3. Before M2 — the flash API must be designed, not transcribed (**CONF-25**)

D5's `evaluate_tpd_simd` references `Kelvin`, `NumericalDivergenceError`, `fugacity_coeff_gas` and
`fugacity_coeff_mixture` — **none declared anywhere**. `NumericalDivergenceError` has one variant named
and no `enum` definition. The listing claims `std::simd` AV-512 vectorisation while showing a scalar
loop.

Additionally unspecified: the Rachford-Rice **bracket strategy**, root **tolerance**, **failure
policy**, and treatment of the **two-phase boundary**.

**Spec requirement:** the flash module's public API, error taxonomy, and bracketing policy must be
written down before implementation. The D5 listing is not an API.

---

## 4. Before M3 — bounds policy (**CONF-24**)

D5 §3.2 states that clamping saturations after the linear solve is **"categorically forbidden"** and
that membership in $[0,1]$ must be guaranteed by the discretisation and an exact Jacobian. The same
framework requires line search, adaptive timestep cuts, and primary-variable substitution — none of
which preserve $[0,1]$ unconditionally.

**Spec requirement:** choose one and write it down:
- **(a)** prove positivity preservation of the discretisation, or
- **(b)** define an explicit bounds-handling policy, with the deviation from "no clamping" documented.

---

## 5. Before M4 — no clamping, and no inert recovery cap (**CONF-24**, **CONF-02**)

**CONF-02:** the proposed recovery ceiling

$$RF(t) \le RF_{ult}(\bar{P}_{res,eff})\left(1-e^{-\text{HCPVI}/\tau}\right),\qquad\tau = 1.5$$

is **99.4 % inactive** at the operating point and has **no mobility-ratio dependence** — it cannot model
fingering, gravity override or channel breakthrough, and would **hide** the exact defect the new engine
must demonstrate it does not have.

**Spec requirement: do not adopt this cap.** The recovery ceiling must come from a declared sweep model —
Koval $K = H\cdot E_{\text{eff}}$ with $E_{\text{eff}} = (0.78+0.22M^{0.25})^4$ plus Dykstra–Parsons
$V_{DP}$ — so it is **mobility-sensitive by construction**. The M6 gate measures non-degenerate
sensitivity in $\mu_o$ and HCPVI.

---

## 6. Before M5 — IO format and tensor payload (**CONF-30b**, **CONF-41**, **CONF-42**, **CONF-28**)

| ID | Gap | Requirement |
|---|---|---|
| **CONF-30b** | `petekIO` is **named 3× and defined 0×** — no magic bytes, header, byte order, slab typing, offset table, or version | Design one format with a **version field** and a **round-trip test across a version bump** |
| **CONF-41** | D7's payload carries only $(K_x,K_y,K_z)$ while its upscaler must preserve off-diagonal $K_{ij}$ | Carry the **full 6-component symmetric tensor**, or drop the upscaler's mandate. Do not silently drop the off-diagonals — that is the error MPFA-O exists to prevent |
| **CONF-42** | **No NTG field** anywhere. Grid-block pore volume requires it | Specify NTG in the hand-off contract |
| **CONF-28** | No DTO versioning, no error codes, no checkpoint/restart format | Specify versioning and at least one restart format |

---

## 7. Before M5 — the economic module (**CONF-58**, **CONF-59**, **CONF-04**, **CONF-37**)

Per the **CONF-04** ruling (adjudicated 07-10-2026): the core solves conservation equations and
publishes production adjoints $\partial q_i(t)/\partial\mathbf{x}$; prices and discounting live off-core.

**But the ruling's own NPV expression must not be transcribed.** It omits four terms:

| Term | Why it matters |
|---|---|
| **CO₂ purchase cost** | Typically the dominant operating cost in a CO₂-EOR project |
| **CO₂ recycle cost** | The mechanism that makes high-recycle designs unprofitable |
| **CO₂ storage credit** | A revenue term |
| **Carbon tax on leakage** | The only term pricing containment failure |

And **CONF-59**: produced gas must arrive as **separate streams**, so **recycled CO₂ is never booked as
sales gas**.

**Spec requirement** — the economic module's full term list, with the sign of each:

$$NPV = \int\Big[q_oP_o + q_{g,\text{sales}}P_g + q_{\text{CO}_2,\text{stored}}C_{\text{credit}} - \big(q_{\text{CO}_2,\text{purch}}C_{\text{purch}} + q_{\text{CO}_2,\text{recycled}}C_{\text{recycl}} + q_wC_w + q_{\text{leak}}C_{\text{tax}}\big)\Big]e^{-rt}dt - \text{CAPEX}$$

**CONF-37:** no price deck exists in the design set — no oil price, gas price, carbon credit, OPEX split,
escalation, or discount rate. These must be **inputs with stated defaults and provenance**, not constants.

---

## 8. Whole engine — what the design set never specifies, and never will

These are not conflicts; they are **absences**. Each needs a decision that the design set cannot supply.

| Absent | Needed for |
|---|---|
| Preconditioner, sparse format, fill-reducing ordering, Krylov method (**CONF-14**) | M0.1 decision; scales to SPE 10 |
| Reference PVT/VLE dataset (**CONF-31**) | M1 |
| Seeds for every stochastic component (TPD sweep, synthetic geology — **CONF-25b**, **CONF-43**) | M2, M5+ |
| Manufactured solutions — **zero mentions** in the design set | M3, M5 spatial and temporal verification |
| Reference values for **any** V&V test (**G-01**) — D5 supplies zero absolute expected numbers | every gate |
| Spatial grid-convergence study (**CONF-26**) | M5 — propose $L_2$-norm Richardson extrapolation |
| Quantity-reliability gate (**CONF-21**) — D5 rejects code coverage and proposes nothing | every gate |
| Rust toolchain — `cargo`/`rustc`/`rustup` all absent (**B-1**) | M0 |

> **Pattern worth naming:** the design set is strongest on *physics intent* and weakest on *numerics
> specification* and *verification data*. It names correct models (Koval, Land, Killough, Barton–Bandis,
> Michelsen, Lasaga) and supplies almost no coefficients, no preconditioners, no seeds, and no reference
> values. **Budget accordingly: the engineering is in the numbers, not in the module list.**

---

## Priority summary

| Before | Must resolve |
|---|---|
| **M1** | CONF-31, CONF-10, CONF-29, CONF-49 · source a cited PVT/VLE dataset |
| **M2** | CONF-25, CONF-25b — design the flash API and add seeds |
| **M3** | CONF-01 (closure only), CONF-24 |
| **M4** | CONF-24 · hold the `1e-12` gate or renegotiate with evidence |
| **M5** | CONF-30b, CONF-41, CONF-42, CONF-28, CONF-58, CONF-59, CONF-37, CONF-14 |
| **M6** | CONF-02 — do not adopt the inertial cap; use a mobility-sensitive sweep ceiling |
| **throughout** | CONF-21 (reliability gate), G-01 (reference values), manufactured solutions |