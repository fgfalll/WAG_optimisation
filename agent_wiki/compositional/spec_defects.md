# Specification Defects to Fix Before M1

The `CONF-*` conflicts in [`../thmc/conflict_and_gap_register.md`](../thmc/conflict_and_gap_register.md)
were written against a design set intended for adoption. For a **new engine**, the subset below becomes
**specification defects** — errors in the document that will be transcribed into code if not fixed first.

This page lists only what must be resolved **before M1**, or before the milestone it is attached to.
The rest of the register stays relevant for **M7a–M7h** (in scope, not yet specified — see
[`build_plan.md`](build_plan.md) M7) and for the design set's own record.

---

## 1. Before M1 — gas fractional flow closure (**CONF-01 / CONF-63 — ✅ RESOLVED 15-10-2026 Audit**)

**The sign is correct — do not "fix" it.** The design formula agrees with the classical
$f_g = \frac{M}{1+M}$ to 4 dp. 

#### ✅ Resolved Governing Formulation (Flaw 1 Closure)
The fractional-flow closure is anchored to **normalized mobile gas saturation** $S_g^*$, eliminating the historical unphysical leakage at $S_g = S_{gc}$:

$$S_g^* = \frac{S_g - S_{gc}}{1 - S_{wi} - S_{gc} - S_{org}}$$

$$f_g(S_g^*) = \begin{cases} 
\dfrac{K_K S_g^*}{1 + S_g^*(K_K - 1)}, & S_g \ge S_{gc} \\ 
0, & S_g < S_{gc} 
\end{cases}, \quad K_K = H \cdot \left(0.78 + 0.22 M^{0.25}\right)^4$$

Three-phase relative permeability is computed via **Modified Stone II** with independent Corey exponents ($n_w, n_{ow}, n_{og}, n_g$) and Land capillary hysteresis for trapped gas $S_{gr}$.

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

## 4. Before M3 — bounds policy (**CONF-24 — ✅ RESOLVED 15-10-2026 Audit**)

D5 §3.2 states that clamping saturations after the linear solve is **"categorically forbidden"** (INV-1).

#### ✅ Resolved Bounds Policy (Option b with Exact Safeguards)
Bounds membership $z_i \in (0, 1)$ and $S_\alpha \in [0, 1]$ is strictly guaranteed by:
1. **Line search step truncation:** $\lambda \le \lambda_{max,bounds} = \min_i \left( -z_i^{(k)} / \Delta z_i \right)$ for $\Delta z_i < 0$.
2. **Adaptive time-step cutback:** If the line search cannot find an admissible step with $\lambda \ge 10^{-3}$, the timestep is cut back by factor $0.5$ without artificial post-solve clamping.
3. Positivity preservation is enforced by continuous primary variables ($z_i$) and exact flash thermodynamics rather than post-solve state manipulation.

---

## 5. Before M4 — no clamping, and no inert recovery cap (**CONF-02 — ✅ RESOLVED 15-10-2026 Audit**)

**CONF-02 CLOSED:** The empirical recovery cap $1 - e^{-\text{HCPVI}/1.5}$ and Black-Oil volumetric integrals are **completely purged**.

#### ✅ Resolved Governing Formulation (Flaw 2 Closure)
Component recovery is evaluated strictly from the conserved 3D compositional domain mass/molar integral:

$$M_i(t) = \int_\Omega \phi(x,t) \left( \sum_{\alpha \in \{o, w, g\}} S_\alpha(x,t) \rho_\alpha(x,t) x_{i,\alpha}(x,t) \right) d\Omega$$

$$RF_i(t) = \frac{M_i(0) + \int_0^t q_{i,\text{inj}}(\tau) d\tau - M_i(t) - \int_0^t q_{i,\text{leak}}(\tau) d\tau}{M_i(0)}$$

Surface hydrocarbon volumes are computed by multi-stage separator flash calculations ($T_{sep}, P_{sep}$), not downhole volume tracking. In 1D models, sweep efficiency is governed by Koval / Dykstra-Parsons curves:
$$RF(t) = E_A \cdot E_I \cdot E_D \cdot (1 - S_{wi} - S_{or})$$

---

## 6. Before M5 — IO format and tensor payload (**CONF-30b**, **CONF-41**, **CONF-42**, **CONF-28**)

| ID | Gap | Requirement | Status |
|---|---|---|---|
| **CONF-30b** | `petekIO` deleted | Direct open formats: HDF5 + Parquet / Arrow | ✅ **CLOSED** (C-127) |
| **CONF-41** | Off-diagonal $K_{ij}$ | Full 6-component symmetric tensor carried in payload | ✅ **RESOLVED** |
| **CONF-42** | NTG field | NTG specified in grid hand-off contract | ✅ **RESOLVED** |
| **CONF-28** | Checkpoint/restart | Separate, independently-committed atomic checkpoint file | ✅ **CLOSED** (C-126) |

---

## 7. Before M5 — the economic module (**CONF-58**, **CONF-59**, **CONF-04**, **CONF-37 — ✅ RESOLVED 15-10-2026 Audit**)

Per the **CONF-04** ruling (adjudicated 07-10-2026):
1. **Simulation Core:** Solves PDE conservation laws and computes **production adjoints** with respect to well controls $\mathbf{u}$: $\frac{\partial q_o}{\partial \mathbf{u}}, \frac{\partial q_{g,\text{sales}}}{\partial \mathbf{u}}, \frac{\partial q_{\text{CO2,inj}}}{\partial \mathbf{u}}, \frac{\partial q_{\text{CO2,recyc}}}{\partial \mathbf{u}}, \frac{\partial q_w}{\partial \mathbf{u}}, \frac{\partial q_{\text{leak}}}{\partial \mathbf{u}}$.
2. **Off-Core Satellite Module:** Evaluates full discounted cash flows and chains economic prices onto production adjoints.

#### ✅ Resolved Unified DCF NPV Formulation (Flaw 5 Closure)
$$\text{NPV} = \sum_{t=1}^{N_t} \Big[ q_o(t) P_o + q_{g,\text{sales}}(t) P_g + q_{\text{CO2,stored}}(t) C_{\text{credit}} - \Big( q_{\text{CO2,purch}}(t) C_{\text{purch}} + q_{\text{CO2,recycled}}(t) C_{\text{recyc}} + q_w(t) C_w + q_{\text{leak}}(t) C_{\text{tax}} \Big) - \text{OPEX}_{\text{fixed}}(t) \Big] \cdot d(t) - \text{CAPEX}$$

where:
- **Sales Gas Revenue:** $\text{Rev}_{gas}(t) = q_{g,\text{sales}}(t) \cdot P_g = \max(0, q_{prod,gas}(t) - q_{recycled}(t)) \cdot P_{gas}$ (CONF-11 / CONF-59).
- **Mid-Year Discrete Compounding:** $d(t) = (1 + r)^{-(t - 0.5)}$, eliminating continuous compounding drift (Flaw 5).

---

## 8. Whole engine — what the design set never specifies, and never will

| Absent | Needed for | Resolution (15-10-2026 Audit) | Status |
|---|---|---|---|
| Preconditioner & Krylov method (**CONF-14**) | M0.1 / SPE 10 scale | **Block-CPR Preconditioned FGMRES / BiCGStab** (Stage 1a unsymmetric AMG $\mathbf{A}_{pp}$, Stage 1b vector AMG $\mathbf{A}_{uu}$, Stage 2 Block-ILU(1)) | ✅ **RESOLVED** |
| Reference PVT/VLE dataset (**CONF-31**) | M1 | Sourced cited dataset (`D:\RAG` SPE 5 fluid, Baled et al. 2012, Abudour 2014) | ✅ **RESOLVED** |
| Seeds for stochastic components (**CONF-25b**, **CONF-43**) | M2, M5+ | Explicit fixed seeds declared in configuration schema | ✅ **RESOLVED** |
| Manufactured solutions | M3, M5 V&V | MMS for 1D/2D transport and Biot poroelasticity | ✅ **RESOLVED** |
| Reference values for V&V (**G-01**) | Every gate | Mapped to `D:\RAG\Data files\SPE5-ProbForecasting-BaseCase.txt` & SPE benchmarks | ✅ **RESOLVED** |
| Spatial grid-convergence (**CONF-26**) | M5 | $L_2$-norm Richardson extrapolation | ✅ **RESOLVED** |
| Quantity-reliability gate (**CONF-21**) | Every gate | Limit-case property testing suite and analytical oracles | ✅ **RESOLVED** |
| Rust toolchain (**B-1**) | M0 | Deferred to M0 kick-off by owner decision | ⏸️ Deferred |

---

## Priority summary (Audit Status as of 15-10-2026)

| Before | Item | Resolution Status |
|---|---|---|
| **M0** | CONF-14 (Block-CPR solver stack) | ✅ **CLOSED** in `engine_numerics.md` §7x.3 and `build_plan.md` §M0.1 |
| **M1** | CONF-31, CONF-10, CONF-29, CONF-49 ($6 \times 6$ BIP matrix, VT parameters) | ✅ **CLOSED** in `satellite_toolkit.md` §1.3 and `doc_inventory.md` |
| **M2** | CONF-25, CONF-25b (flash API error enum, TPD seed) | ✅ **CLOSED** in `engine_spec_closures.md` §7e.5 |
| **M3** | CONF-01 / CONF-63 (normalized $S_g^*$ Koval fractional flow) | ✅ **CLOSED** in `architecture_design.md` §3.2 and `spec_defects.md` §1 |
| **M4** | CONF-24 (saturation bounds via line search truncation, no clamping) | ✅ **CLOSED** in `engine_numerics.md` §7x.2 and `vv_testing_framework.md` §4.2 |
| **M4** | CONF-02 (exact domain mass integral $M_i(t)$) | ✅ **CLOSED** in `architecture_design.md` §6.1 and `spec_defects.md` §5 |
| **M5** | CONF-30b, CONF-41, CONF-42, CONF-28 (open formats, NTG, restart) | ✅ **CLOSED** in `data_architecture.md` and `engine_numerics.md` §7o.3 |
| **M5** | CONF-58, CONF-59, CONF-04, CONF-37 (Core-satellite DCF, discrete compounding) | ✅ **CLOSED** in `solver_and_numerics.md` §7.1 and `satellite_toolkit.md` §5.2 |
| **M7b** | CONF-66 (CPPM return mapping, $\theta_L = -\pi/6$ compression, $\mathbf{D}^{alg}$) | ✅ **CLOSED** in `engine_constitutive.md` §7ll.1 and `solver_and_numerics.md` §10.1 |
| **M7c** | CONF-51 (proppant closure aperture, StimLab crush curves) | ✅ **CLOSED** in `fractures_wellbore.md` §6.3.1 and `engine_constitutive.md` §7ll.3 |
| **M7h** | CONF-68 (scale-invariant skin floor) | ✅ **CLOSED** in `fractures_wellbore.md` §7.3.1 |