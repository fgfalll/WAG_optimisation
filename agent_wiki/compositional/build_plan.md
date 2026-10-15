# Compositional Engine — Build Plan (M0 → M6)

Companion to [`README.md`](README.md). Each milestone has a **gate** that must be measured, not asserted.
Gates are recorded in a dated run audit under
[`../audit/simulation_run_audits/`](../audit/simulation_run_audits/index.md) with an explicit **Verdict**.

**Rule for the whole plan:** a `Status:` line on every gate claim, citing a command and a measured
number. `agent_wiki/README.md` invariant 15 requires a measurement before any `RESOLVED` mark; the same
discipline applies here from day one.

---

## M0 — Toolchain, skeleton, and the separate gates

**This milestone exists so that M1 onward cannot be built on an unauditable foundation.**

### Tasks

| # | Task | Notes |
|---|---|---|
| 0.0 | Install Rust toolchain | ✅ **RESOLVED BY DECISION 07-10-2026** — the toolchain is installed **after the documentation is complete, at the start of development**. Rationale given: *"some crates can be changed in process"*. Consequence: **crate selection below is provisional** — record the reasoning, not just the choice |
| 0.1 | Create `crates/compositional/` | `Cargo.toml` here, **not** the repo root. Single crate; no workspace |
| 0.2 | Crate lints in `lib.rs` | `#![deny(unsafe_code)]`, `#![forbid(non_snake_case)]`, `#![deny(missing_docs)]` |
| 0.3 | `audit_comp/` module | [`register_spec.md`](register_spec.md) §6 |
| 0.4 | Separate CI job | `cargo build`, `cargo clippy -- -D warnings`, `cargo test`, `audit_comp --register validate`, `audit_comp.continuity check` |
| 0.5 | Independence gates | C-1 forbidden-import grep; C-2 FFI ban; C-3/C-4 no cross-import between `audit` and `audit_comp` |
| 0.6 | **Run-specification format** — versioned, hashable, self-contained | Resolves **CONF-30b**. ⚠️ Needed before any gate can run — see §M0.2 |
| 0.7 | Choose and benchmark the **linear algebra stack** | Resolves **CONF-14**. See §M0.1. **Provisional** — crates may change |
| 0.8 | 🔵 **Full output schema declared** — all nine domains, with a **capability declaration** mechanism | [`output_schema.md`](output_schema.md); **INV-6**. An interface is cheap to define and expensive to change |
| 0.9 | 🔵 **Write layer — HDF5/VTK-HDF + Parquet + DuckDB** | [`output_schema.md`](output_schema.md) §3, §4. ⚠️ Chunking + compression **mandatory** — 510 MB/step at SPE-10 scale |
| 0.10 | Add `pyarrow` + `duckdb` to Python deps | **Measured MISSING 08-10-2026.** Without them the satellite cannot read the engine's output |

### M0.2b 🔴 SCOPE RE-BASELINE — approved 08-10-2026

The approved master map ([`output_schema.md`](output_schema.md) §9) brings the **entire 3D THMC
programme** into scope: fully coupled thermal + geomechanics (Biot FEM) + geochemistry (Lasaga), fractures
and EDFM, wellbore drift-flux, Schwarz sub-domains, remediation, GPU flash, adjoint gradients, ES-MDA.

> **The earlier deferral is void.** M0–M6 previously scoped the compositional subset with geochemistry,
> THMC, fractures, faults, GPU and adjoint in the old "M7+" bucket. Those are now in scope. **This is not an 8–11
> person-month plan** — the old estimate covered roughly one third of approved scope.

**The tractable path — define the interface now, implement incrementally:**

| Milestone | Domains populated | Declared absent (**INV-6**) |
|---|---|---|
| M1 | none — standalone EOS/PVT | 1–9 |
| M3 | 1, 2, 3, **9 (physical part only)** | 4, 5, 6, 7, 8 |
| M4 | + compositional | 4–8 |
| M5 | + 7, 8 (grid, wells, fractures) | 4, 5, 6 |
| M6 | + 6 (5-state trapping) | 4, 5 |
| M7 | + 4, 5 (geochemistry, geomechanics) | none |

A **declared-absent** field is not a failure. Only a `FAILED` module fails the run (**INV-1**).

**Scope now in the plan:** M7a geochemistry · M7b THMC/Biot FEM · M7c fractures & EDFM · M7d fault
mechanics · M7e GPU flash · M7f adjoint gradients (needed for the RL loop, **INV-4**) · M7g Schwarz
sub-domains · M7h remediation & well control.

> ✅ **RESOLVED IN GOVERNING DESIGN (15-10-2026 Audit & Ruling 4):** The core specification blockers are now closed across [`engine_numerics.md`](engine_numerics.md), [`engine_constitutive.md`](engine_constitutive.md), [`engine_spec_closures.md`](engine_spec_closures.md), and [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md):
> - **CONF-01 & CONF-63 (Flaw 1)**: Normalized mobile gas saturation $S_g^* = \frac{S_g - S_{gc}}{1 - S_{wi} - S_{gc} - S_{org}}$ anchoring Koval fractional flow $f_g(S_g^*)$, with Modified Stone II and Land capillary hysteresis.
> - **CONF-02 (Flaw 2)**: Exact 3D compositional domain component mass integral $M_i(t) = \int_\Omega \phi \sum_\alpha S_\alpha \rho_\alpha x_{i,\alpha} d\Omega$ and dynamic $RF_i(t)$ replacing inert empirical Black-Oil cap.
> - **CONF-04 & CONF-58 (Flaw 5)**: Strict Core-Satellite boundary: core evaluates production adjoints $\partial q_i / \partial \mathbf{u}$; satellite evaluates economics with Mid-Year Discrete Compounding $(1+r)^{-(t-0.5)}$.
> - **CONF-05**: Minimum thickness threshold $\Delta z_{\text{pinch}} = 1.0\text{ mm}$; collapsed cells generate explicit Non-Neighbor Connections (NNCs) between bounding active blocks.
> - **CONF-08**: Disambiguated time-step notation ($\Delta t_{\text{macro}}$, $\delta t_{\text{zoom}}$, $\delta t_{\text{well}}$) with two-way coupling conservation tolerance $< 10^{-12}$.
> - **CONF-13**: Production Newton-Raphson solver with Armijo-Goldstein backtracking line search, quadratic interpolation fallback, and exponential damping soft-start.
> - **CONF-14 (Flaw 3)**: **Block-CPR Preconditioned FGMRES / BiCGStab** (Stage 1a unsymmetric AMG on pressure $\mathbf{A}_{pp}$, Stage 1b vector AMG on displacement $\mathbf{A}_{uu}$, Stage 2 Block-ILU(1) smoothing latched per timestep).
> - **CONF-16 & CONF-18**: Solid-volume corrected porosity update $\phi^{t+\Delta t} = \frac{\phi^t - \sum_m \Delta V_m / V_{bulk}}{1 + \varepsilon_v^{mech}}$ and normalized chemical strain tensor $\boldsymbol{\varepsilon}_{chem} = \frac{1}{3} \frac{\sum_m \Delta V_m}{V_{bulk}} \mathbf{I}$.
> - **CONF-20**: Formally reconciled CAPEX band to \$40M–\$100M (\$90.95M offshore expansion case; satellite emits `ValidityWarning::CAPEXBandExceeded`).
> - **CONF-26**: M3 spatial grid convergence benchmark suite ($N \in \{16^3, 32^3, 64^3, 128^3\}$, convergence order $p \ge 1.8$, TVD monotonic shock fronts).
> - **CONF-32**: Zero silent mutation under INV-1/INV-7; raw DTOs preserved bit-identically, adjustments emit versioned `InputAuditTrail` and `ValidityWarning`.
> - **CONF-33**: SEPD terminal exponential decline cutoff: $q(t) = q_i \exp[-(t/\tau)^n - D_\infty t]$ with $D_\infty \ge 5\%/\text{yr}$.
> - **CONF-37**: Approved baseline economic price deck (Crude \$75/bbl, Sales Gas \$3.50/MSCF, 45Q \$85/t, Purchase \$50/t, Recycle OPEX \$15/t, Disposal \$1.25/bbl, Injection \$0.75/bbl, Leakage tax \$120/t, discount rate $r=10\%$).
> - **CONF-40**: Domain 7 rel-perm payload upgrade to 11-parameter Modified Stone II + Corey closure.
> - **CONF-41**: Full 6-component symmetric permeability tensor payload `AnisotropicTensor3D` $\{K_{xx}, K_{yy}, K_{zz}, K_{xy}, K_{xz}, K_{yz}\}$.
> - **CONF-42**: Mandatory structural grid fields ($NI, NJ, NK, DX, DY, DZ / ZCORN, COORD, NTG, \phi, \mathbf{K}, Z_{\text{top}}$).
> - **CONF-46**: Backwashing law via hydrodynamic shear-stress detachment $\partial\sigma_p/\partial t = -\lambda_{\text{wash}}\|\mathbf{u}_{\text{rev}}\|\sigma_p$ coupled to Verma-Pruess.
> - **CONF-49**: PR/SRK with Baled (2012) Volume Translation mandatory; MAPD $\le 2.0\%$ across 7–276 MPa, 278–533 K.
> - **CONF-51**: Proppant pack aperture under closure $w_{res} = \max(w_{elastic}, \frac{C_{prop}}{\rho_{prop}(1-\phi_{pack})} - 2 d_{embed})$ and StimLab stress crush curves $k_f(\sigma_n')$.
> - **CONF-60**: Penalty function split: $-10^{12}$ catastrophic step penalty for solver failures; $C^2$ quadratic exterior penalty for bounded physical constraints to preserve adjoint sensitivities $\partial q_i / \partial \mathbf{u}$.
> - **CONF-66 (Flaw 4)**: Compression-positive convention ($\theta_L = -\pi/6$ compression meridian), CPPM return mapping with consistent algorithmic elastoplastic tangent $\mathbf{D}^{alg}$, and rounded apex regularisation $N_\kappa \ge \kappa_{\min} > 0$.
> - **CONF-68**: Scale-invariant skin floor $S_{min} = -\ln(r_e/r_w) + \Delta S_{margin}$ with `ValidityWarning::SkinFloorActivated` (INV-1).

### M0.1 Linear-algebra decision (RESOLVED 15-10-2026 — Block-CPR Solver Stack)

**CONF-14 CLOSED**: The linear solver stack is formally standardized as **Block-CPR Preconditioned FGMRES / BiCGStab**, eliminating the previous deadlock between M0 linear algebra and M7b non-associated elastoplasticity.

| Component | Selected Method & Crate | Role & Justification |
|---|---|---|
| **Krylov Driver** | **FGMRES / BiCGStab** (`faer` + custom Krylov) | Tolerates variable, iteration-dependent preconditioning arising from adaptive AMG hierarchies and phase-boundary transitions. |
| **Stage 1a (Pressure)** | **Unsymmetric AMG** on $\mathbf{A}_{pp}$ | Collapses elliptic pressure degrees of freedom. Uses classical Ruge-Stüben or smoothed aggregation with drop tolerance $10^{-3}$; handles non-symmetric pressure Jacobian. |
| **Stage 1b (Mechanics)** | **Vector AMG** on $\mathbf{A}_{uu}$ | Handles vector elasticity and geomechanics $[u_x, u_y, u_z]$. Near-nullspace initialized with 6 rigid body modes. Operates on stabilized elastoplastic tangent $\mathbf{D}^{alg}$. |
| **Stage 2 (Transport/Global)** | **Block-ILU(1)** | Global multi-color Block-ILU(1) smoothing with level-of-fill 1, latched per timestep to preserve cache locality. |

#### ✅ M0.1 Unblocked: Non-Symmetric Elastoplasticity Decoupled
1. Non-associated Drucker-Prager / Mohr-Coulomb ($\mathbf{D}^{alg}$ non-symmetric) is handled natively by unsymmetric AMG and FGMRES; symmetric-only Cholesky / IC(0) solvers are permanently excluded.
2. The pressure-mechanics-composition coupling is segregated by the two-stage CPR projection, ensuring quadratic Newton convergence ($L_2$ residual drop by $\ge 10^3$ per iteration) even during active plastic yielding.
3. Baseline benchmarks run against non-symmetric test matrices derived from the SPE 5 and SPE 10 coupled benchmarks.

### M0.2 Run-specification format — the item that unblocks everything else

**CONF-30b**: `petekIO` is **named 3× in the design set and defined 0×** — no magic bytes, header, byte
order, slab typing, offset table, or version. Yet **every** M-gate needs a loadable problem
specification.

Per the data-architecture decision ([`data_architecture.md`](data_architecture.md) §6.1), the spec is
built from PostgreSQL and serialised into the engine's own binary form. Required properties:

| Property | Why |
|---|---|
| **Versioned** | `spec_version` field; a spec written by an older build must be rejected, not misread |
| **Hashable** | `spec_hash` + `engine_version` + `build_hash` identify a result completely (**INV-4**) |
| **Self-contained** | Grid, fluid, PVT, SCAL, initial state, wells, schedule, numeric controls — the engine loads nothing else |
| **Deterministic** | Same spec ⇒ identical output. Required by INV-4 and every gate |
| **Provenance-carrying** | ⚠️ Every critical property and $k_{ij}$ carries a **`source`**. Un-sourced values are a defect |
| **Memory-compatible** | Layout maps onto the SoA flat-slab requirement ([`../thmc/architecture_design.md`](../thmc/architecture_design.md) §2.1) — and a PostgreSQL `float8[]` column *is* that layout |

> ⚠️ **Do not defer this to P2.** If the database migration is deferred, the engine must first carry a
> throwaway file format — which is exactly the **CONF-30b** mistake the design set already makes.

### M0 gate

| Check | Requirement |
|---|---|
| Build | `cargo build --release` clean |
| Lints | `cargo clippy -- -D warnings` clean |
| **INV-1** | Zero `unsafe`; zero `unwrap`/`expect`/`panic!` in `src/`; zero references to `core/`; **no fallback path anywhere** |
| Register | `audit_comp --register validate` passes; a seeded `COMP-01` with an `.rs` location validates |
| Continuity | `audit_comp.continuity check` proves the gate works (deliberately break a claim, watch it fail) |
| **Run spec** | Versioned, hashable, round-trips, rejects an unknown version |
| Docs | `missing_docs` clean |

---

## M1 — EOS + PVT, no flow

**Hard gate.** The engine's validity is bounded by its fluid model.

### Tasks

| # | Task |
|---|---|
| 1.1 | Pure-component properties: $T_c$, $P_c$, $\omega$ for C1, CO₂, N₂, H₂S + C2–C6 and C7+ pseudocomponent construction (Joback-Reid, PPR78) |
| 1.2 | Peng-Robinson + volume translation (VT-PR); SRK + VT |
| 1.3 | $Z$-factor, molar volume, density |
| 1.4 | Viscosity: Free Volume theory and/or f-theory |
| 1.5 | Binary interaction parameters $k_{ij}$ — **Huron-Vidal** or van der Waals mixing |
| 1.6 | Unit safety: typed units, no bare `f64` for a physical quantity |

### Specification gaps that must close before 1.1

| Gap | Problem |
|---|---|
| **CONF-31** | **No coefficient is given for any named correlation** — Standing, Glaso, Beggs–Robinson, Joback-Reid, PPR78, Huron–Vidal, QSPR, Karakas–Tarik. All are name-only |
| **CONF-10** | VT shift $s = f((M\omega)^{-1})$ — the correlant is undefined and no fit is supplied |
| **CONF-29** | D6's `bip_matrix` is **2 × 3 for 6 components** — a valid symmetric 6-component set needs **21** unique $k_{ij}$; **15 are absent** |
| **CONF-49** | The design set *declares* MAPD density error **3–9 %** for PR/SRK. That is inside the noise band of the thing being optimised |

**Required action:** source a **cited** reference dataset before 1.1. The design set names a
**"> 900 binary system" QSPR database** and **"17 pure hydrocarbons, CH₄…n-C₄₀H₈₂, Baled et al. 2012"**
— identify both, or state the substitution with a citation. **No critical property may be invented.**

### M1 gate

| Metric | Requirement | Basis |
|---|---|---|
| Pure-component $Z$ | vs. published reference, **MAPD ≤ 2 %** | D5 §4.3 target for VT models |
| Liquid density, HTHP (7–276 MPa, 278–533 K) | **MAPD ≤ 2 %** | Tighter than the design set's own 3–9 % — **CONF-49** |
| Viscosity | **MAPD ≤ 5 %** (Free Volume, with PC-SAFT `< 2 %`) | D2 §3.2 |
| Volume translation | improves density MAPD over plain PR/SRK by a **measured** margin | Justifies the VT at all |
| $k_{ij}$ | regression residuals reported against the reference VLE data | |
| Units | no unit conversion error found in review | `agent_wiki/data/units.md` conventions |

> **If M1 lands above 2 %, stop.** Do not proceed to M2. A 5 % fluid model makes every subsequent
> recovery number uncertain by ±5 %, and no amount of solver quality recovers that.

---

## M2 — Flash with analytic derivatives

### Tasks

| # | Task |
|---|---|
| 2.1 | Michelsen TPD stability test, multi-start from **Wilson K-factors** ($K_i = \frac{P_{ci}}{P}e^{5.37(1+\omega_i)(1-T_{ci}/T)}$), pure-component vectors $e_i$, and mixtures |
| 2.2 | Rachford-Rice with Newton root-find and step limiting: $f(V) = \sum_i \frac{z_i(K_i-1)}{1+V(K_i-1)} = 0$ |
| 2.3 | Fugacity coefficients $\varphi_i^L$, $\varphi_i^V$ |
| 2.4 | **Analytic derivatives** $\partial \ln\varphi_i/\partial P$, $\partial \ln\varphi_i/\partial x_j$ |
| 2.5 | Hypersensitive / supercritical handling; critical-point continuity |

### Specification gaps

| Gap | Problem |
|---|---|
| **CONF-25** | D5's `evaluate_tpd_simd` references `Kelvin`, `NumericalDivergenceError`, `fugacity_coeff_gas`, `fugacity_coeff_mixture` — **none declared**. The API must be designed, not transcribed |
| **CONF-25** | The listing claims `std::simd` AV-512 vectorisation; the shown loop is scalar |
| — | The set states **no** Rachford-Rice bracket strategy, no root tolerance, no failure policy, and no treatment of the two-phase boundary |
| **CONF-33 (pattern)** | — |

### M2 gate

| Metric | Requirement | Basis |
|---|---|---|
| **Gibbs monotonicity** | $G^{(k+1)} < G^{(k)}$ at every iteration — a **test**, not a claim | D5 §4.1 |
| **TPD sweep** | ≥ 10 000 **seeded** random $P$-$T$-$z$ points, **no missed stability boundary** | D5 §4.2. **Add a seed** — D5 has none (**CONF-25b**) |
| **Split criterion** | $TPD(y)<0$ at ≥ 1 stationary point ⇒ mixture must split | D5 §4.2 |
| **Jacobian correctness** | analytic vs. central difference, **rel. error < 1e-6** | No such check exists anywhere in the design set |
| **Critical-point limit** | $\varphi_i^L/\varphi_i^V \to 1.0$; phase functions monotone and C¹ | D5 §4.3 |
| **No non-finite** | zero `NaN`/`Inf` across the sweep | D5 §4.3 |

---

## M3 — 1D two-phase FVM, single component

The first gate that proves the **transport** discretisation, independent of thermodynamics.

### Tasks

| # | Task |
|---|---|
| 3.1 | 1D structured grid, transmissibility, `ActiveMask`, zero-volume handling |
| 3.2 | **Corey / Brooks-Corey** relative permeability with $S_{or}$, $S_{wi}$, $S_{gc}$, $S_{gr}$ |
| 3.3 | Explicit two-phase fractional flow, gas fractional-flow function with **correct closure** — see [`spec_defects.md`](spec_defects.md) §1 |
| 3.4 | Fully implicit FVM; upwind + TVD |
| 3.5 | Newton-Raphson, line search, adaptive timestep |
| 3.6 | Single-phase well with Darcy/Vogel IPR |

### Gate

| Metric | Requirement | Basis |
|---|---|---|
| **Buckley–Leverett** | shock-front position **≤ 0.1 %** of length vs. **Welge construction** | D5 §2.1 |
| BL overshoot | no value outside $[S_{wi}, 1-S_{or}]$; **no clamping** used to achieve it | D5 §3.2 forbids clamping |
| BL mobility sensitivity | $f_g$ **rises** with mobility ratio $M=\lambda_g/\lambda_o$, verified **by sweep** | `../thmc/reservoir_engineer_ruling.md` §4 |
| Temporal order | $\alpha \approx 1.0$ (Backward Euler) / $\alpha \approx 2.0$ (CN, Radau IIA) over $\Delta t, \Delta t/2, \Delta t/4, \Delta t/8$ | D5 §5.1 |
| Saturation closure | $\sum S = 1$ on **100 %** of steps, **every Newton iteration** | D5 §3.2 |
| Mass balance | residual below the tier tolerance on every step | D5 §3.1 |

> **BL is the single most important test in the plan.** It is the test the retracted CONF-01 was about.
> An independent Welge construction already exists at
> `tests/scientific/reference_solutions/test_buckley_leverett_analytical.py` — **reuse its numeric
> expectations, not its code.**

---

## M4 — 1D compositional FVM

### Tasks

| # | Task |
|---|---|
| 4.1 | Component transport; coupled component/pressure system |
| 4.2 | **Primary-variable switching** at phase appearance/disappearance, with **C¹ Jacobian continuity** |
| 4.3 | Analytic Jacobian across the switch boundary |
| 4.4 | Multiple injection-gas compositions (CO₂, CO₂+CH₄, CO₂+N₂, CO₂+H₂S) |
| 4.5 | Component-wise accumulation and closure |

### Specification gaps

| Gap | Problem |
|---|---|
| **CONF-24** | D5 §3.2 **forbids clamping** saturations after the linear solve, while the same framework requires line search, step cuts and primary-variable substitution. These cannot all hold — the bounds policy must be designed |
| **CONF-25** | Phase-boundary switching is specified behaviourally with no algorithm |
| — | No treatment of the $S_g \to S_{gc}$ singularity in the fractional flow (see `spec_defects.md` §1) |

### M4 gate

| Metric | Requirement |
|---|---|
| **Component mass balance** | **`< 1e-12`** per component per timestep — **the defining M4 gate** |
| Component closure | $\sum_i z_i = 1.0$ and $\sum_\alpha S_\alpha = 1.0$ at every Newton iteration |
| Newton robustness | **no cycling** at phase boundaries (D5 §7.2 explicit requirement) |
| Zero non-finite | no `NaN`/`Inf` across the compositional suite |
| Zero panic | no `panic!` path reachable — enforced by lint, not by inspection |
| CO₂ purity | runs with each gas composition from D6's `PVT_EOS_DTO` |

> **`1e-12` is a hard gate.** If M4 cannot hold it, the `10^-12` requirement in D5 is aspirational and
> the tolerance must be renegotiated with evidence — **not** quietly relaxed.

---

## M4.5 — 🔴 INTEGRATION SLICE (added 09-10-2026)

**This milestone exists to make disconnected modules impossible.** 🔴 It is not a feature milestone and
carries **no new physics**.

**The problem it solves.** M1–M4 are built as vertical stacks of independently-tested subsystems: EOS,
then flash, then transport, then the compositional loop. Each passes its own gate. ⚠️ **Nothing in M0–M4
requires those subsystems to work *together*** — so a solver can pass every existing gate, have every module
unit-tested, and still be unable to produce **one run**. That failure mode produces individually-correct
disconnected modules and no simulator.

### Gate

| Check | Requirement | Fails if |
|---|---|---|
| **End-to-end run** | One complete case: spec → flash → transport → compositional Newton → timestep → output written | any stage cannot be driven from the driver |
| **No orphan modules** | Every `pub` module is reachable from the driver. A module with no caller is a gate failure | an unreferenced module exists |
| **Single write path** | Output is produced by the one IO path, not by a module-local debug writer | a second write path exists |
| **Mass balance, integrated** | Component-wise `< 1e-12` **across the assembled system**, not per-subsystem | balance holds per-module but not assembled |
| **Determinism** | Same spec twice → byte-identical output | any byte differs |
| **No silent degradation** | Running the integrated case emits **no** `ValidityWarning` | a warning fires on a nominal case |

⚠️ **The "no orphan modules" check is the load-bearing one.** A module that nothing calls is not
disconnected code — it is **dead code that looks implemented**, and it will pass every other gate.

### Ruled

✅ **A module may not be marked complete until it is reachable from the driver and exercised by the
end-to-end run.** ✅ **M4.5 is a merge gate as well as a milestone** — it applies to *every* PR touching
the module graph, from M0 onward, not only at M4.5.

⚠️ **Scope of the case:** deliberately small — a 1-D or small 3-D grid, two or three components, a few
timesteps. 📌 **It is an integration test, not a benchmark**; accuracy is M3's and M4's job. Its only
question is *"does one run complete?"*

---

## M5 — 3D grid, wells, IO, reporting

### Tasks

| # | Task |
|---|---|
| 5.1 | 3D structured grid $N_x\times N_y\times N_z$; parallel residual/Jacobian assembly (Rayon) |
| 5.2 | **Zero heap allocation** in the Newton loop — a **test**, not a doc claim |
| 5.3 | Multi-well; IPR; well mass accounting |
| 5.4 | **IO format** — one format, fully specified, **with a version field** |
| 5.5 | Output: per-component rates, cumulatives, pressure/saturation fields |
| 5.6 | 🔴 **Output schema must be sufficient for training** (**INV-4**) — see §M5.1 |
| 5.7 | 🔴 **Produced gas emitted as two distinct streams** — hydrocarbon sales gas and CO₂ (**INV-3**) |

### 5.1 🔴 The output schema is decided here, not in P2

The compositional engine is the **training-data source** for a per-reservoir neural surrogate (INV-4).
That constrains the output schema:

| Requirement | Detail |
|---|---|
| **Labelled pairs** | $(x, y)$: input control vector → full output state, per run |
| **Full fields, not summaries** | Per-cell pressure, per-component saturation, per-component composition. ⚠️ A surrogate trained on profiles **cannot represent fingering or channeling** |
| **Versioned** | The dataset schema is versioned; retraining after an engine change requires knowing which engine version produced which samples |
| **Provenance per sample** | `engine_version`, `build_hash`, `run_manifest_id` |
| **Split metadata** | ⚠️ Samples carry **geological realisation** and **development pattern** identifiers — see §5.2 |

> **Why this cannot be deferred.** If the schema is decided late and proves inadequate, **every historical
> run must be re-run** at $10^3$–$10^5\times$ the surrogate's cost. Decide it at M5, alongside IO (D9).

### 5.2 🔴 Train/test leakage is a design constraint, not a later concern

Compositional runs are expensive, so there is strong pressure to reuse them across splits. If the same
well configuration appears in both train and test, reported surrogate accuracy is **meaningless** — the
network memorises the configuration rather than learning the physics.

**Required:** splits are by **geological realisation** and **development pattern**, never randomly over
samples. The identifiers must be in the schema from the start. Retrofitting is painful.

### 5.3 Economic output — **not in this engine** (**INV-3**)

Per **INV-3** and the **CONF-04** ruling, the compositional engine contains **no** economics. A separate
**economic engine** owns all field development calculations.

| Moves out | Content |
|---|---|
| NPV / IRR / payback / discounting | Economic engine |
| CAPEX / OPEX / price deck / escalation | Economic engine |
| Storage credit, carbon tax | Economic engine |
| **CONF-58** — the four CO₂ cost terms | Economic engine must carry all four |
| **CONF-59** — recycled CO₂ booked as sales gas | Resolved by 5.7: two distinct gas streams |
| **CONF-11 / CRIT-18** — gas sales contribute $0 | Economic engine |

> ⚠️ **This engine must still publish the physical quantities economics needs** — per-component
> production, injection, storage, leakage, and well state. Just no money.

### Specification gaps

| Gap | Problem |
|---|---|
| **CONF-30b** | `petekIO` is **named 3× and defined 0×** — no magic bytes, header, byte order, slab typing, or version. The format must be designed |
| **CONF-14** | No sparse format or preconditioner — see M0.1 |
| **CONF-41** | The D7 payload carries only $(K_x,K_y,K_z)$ while its upscaler is mandated to preserve off-diagonal $K_{ij}$. Must carry the **full 6-component symmetric tensor** or the mandate is dropped |
| **CONF-42** | **No NTG field** anywhere in D7's hand-off. Grid-block pore volume requires it |
| **CONF-58** | The economic module must carry **CO₂ purchase, CO₂ recycle, storage credit, carbon tax on leakage** |
| **CONF-59** | Produced gas must be **split** into sales gas and recycle CO₂; recycle must never be booked as sales |

### M5 gate

| Metric | Requirement | Basis |
|---|---|---|
| **Spatial order** | $L_2$-norm **Richardson extrapolation** grid-refinement across mesh levels; measured order | D5 §5.1 is **temporal only** — **CONF-26**. This is the engine's first spatial verification and the design set has none |
| **Grid-orientation invariance** | five-spot results unchanged when the grid is rotated **45°** | D5 §5.3 |
| M-matrix | non-negative diagonal, non-positive off-diagonal, diagonal dominance; symmetry; positive definiteness | D5 §3.3 |
| Allocation | zero heap allocation in the Newton loop | D1 §10.1 |
| IO round-trip | write → read → identical state, across a **version bump** | **CONF-28** |
| SPE 5 | **≤ 1.5 %** on RF and produced-gas composition | D5 §6.1 |

> **SPE 5 parameters already exist** at `validation/spe5_config.py` — reuse the **numbers**.
> **CMG GEM reference outputs** exist at `validation/cmg/flu/` (`gmflu001`–`gmflu003`) — these are the
> comparison simulator D5 says it lacks. Reuse the **expected values**.

---

## M6 — CO₂-EOR specifics

### Tasks

| # | Task |
|---|---|
| 6.1 | MMP calculation and miscibility-region detection |
| 6.2 | **Five-state trapping** inventory: Free / Trapped / Dissolved / Adsorbed / Mineralised — 5.1; **Mineralised deferred** (needs geochemistry, out of scope) → 4 states |
| 6.3 | Land trapping + **C¹-smooth Killough/Carlson hysteresis** |
| 6.4 | Koval sweep, **corrected closure** per `spec_defects.md` §1 |
| 6.5 | Net-utilisation accounting and reconciliation |
| 6.6 | WAG cycling |

### Gate

| Metric | Requirement |
|---|---|
| Trapping | mass balance **across all trapping transitions**, not just globally |
| Saturation closure | $\sum S = 1$ on **100 %** of steps including hysteresis reversals |
| **Net utilisation** | no "phantom utilisation"; reconciliation between injected, produced, stored |
| **Mobility sensitivity** | recovery **non-degenerate** in $\mu_o$ over 0.5 → 100 cP, and non-degenerate in HCPVI |
| MMP | matches a cited correlation set to a stated tolerance |
| Reversal | **no discontinuity** on flow reversal (C¹ hysteresis) |

> **The mobility-sensitivity gate is deliberate.** The current Python engine has a defect exactly of this
> shape — recovery exactly flat in oil viscosity (`../thmc/README.md` invariant 6, CRIT-14). The new
> engine must demonstrate non-degenerate sensitivity from M3 onward, and M6 must show it in recovery.
> **The inertial exponential cap the design set proposes (`CONF-02`) must not be adopted** — at the
> default operating point it is 99.4 % inactive and would hide exactly this.

---

## M7 — **IN SCOPE** (re-baselined 08-10-2026) — milestone bodies **not yet specified**

> 🔴 **This section previously read "M7+ — Deferred, each a separate decision", which contradicted
> §M0.2b (the re-baseline), [`vision_and_phases.md`](vision_and_phases.md) §5.2 #10, and
> [`output_schema.md`](output_schema.md) §9. Corrected 09-10-2026.**
>
> **The deferral is void.** These are approved scope. What is *missing* is specification — see the
> honesty note below, which is **not** a licence to start them early.

| # | Milestone | Content | Design source | Blocking conflicts |
|---|---|---|---|---|
| **M7a** | Geochemistry | Reactive transport, PWRI, asphaltenes, hydrates | D3 | **CONF-31** (correlation coefficients), **L-1** |
| **M7b** | THMC / geomechanics | Fully coupled thermal + Biot FEM, rate-and-state friction | D1 §3.3–3.4 | ✅ **body written 09-10-2026** — see M7b below. 🟠 **CONF-66** partially resolved (**4 blockers**), **CONF-14** (unblocks here), **CONF-18** (Bethel) |
| **M7c** | Fractures & EDFM | PKN/KGD/radial propagation, EDFM, proppant transport | D1 §7, D8 | **CONF-51** (proppant coupling), **C-113** list |
| **M7d** | Fault mechanics | Reactivation, seal→conduit transition | D1 §8.2 | not yet triaged |
| **M7e** | GPU flash | GPU-accelerated TPD/flash | D1 §4.3 | **CONF-25** (the SIMD listing is scalar) |
| **M7f** | Adjoint gradients | Adjoint for the RL loop (**INV-4**) | D1 §10.2 | not yet triaged |
| **M7g** | Schwarz sub-domains | 3D domain decomposition / zooming | D1 §2.3 | not yet triaged |
| **M7h** | Remediation & well control | Remediation works, emergency control | D4 | not yet triaged |

Also in scope, not separately numbered: wellbore drift-flux and micro-annulus (D1 §9.2) · DP/DP and
MINC (D2 §4.1) · PGS and synthetic geology (D7) · GEP (D1 §11.2) · ES-MDA (D1 §11.3).

---

## M7b — THMC / geomechanics (fully coupled thermal + Biot FEM)

**Written 09-10-2026.** 🔴 **This is the only M7 milestone with a body, and it was written last on purpose**
— it is where all 36 rulings landed, so it needed the most adjudication to specify.

⚠️ **Read [`engine_constitutive.md`](engine_constitutive.md) in full before starting.** Sections
`7x`–`7hh` are the constitutive rulings this milestone implements, and every one of them is a
**measured** correction to a submitted form.

### 7b.1 Scope

| In scope | Out of scope |
|---|---|
| Poroelastic **Biot FEM** — fully coupled, cell-local pressure | Fault reactivation (**M7d**) |
| **Thermo-mechanical** coupling | Fractures / EDFM (**M7c**) |
| Elastoplastic constitutive model — Drucker–Prager, non-associated flow | Remediation / well control (**M7h**) |
| Rate-and-state friction, implicit in $V$ (**C-72**) | GPU (**M7e**), adjoint (**M7f**), Schwarz (**M7g**) |
| Hardening / softening law, $C^1$ | Geochemistry (**M7a**) |

### 7b.2 Locked constitutive model — do not re-derive

Every item below was corrected by **measurement**. 🔴 **Re-deriving any of them reintroduces a known defect.**

| Element | Locked form | Ruling |
|---|---|---|
| Yield surface | $F=\sqrt{J_2/3}-\alpha I_1-k$, **compression-positive** | `7aa` |
| Coefficients | $\alpha=\dfrac{2\sin\phi}{3(3\mp\sin\phi)}$, $k=\dfrac{2c\cos\phi}{3\mp\sin\phi}$ | `7z`, `7aa` |
| Tensile apex | $I_1=-3c\cot\phi$, exactly MC's tensile cutoff | `7bb` |
| Stored-energy rate | $\dot E^{p,\text{stored}}=H(\bar\varepsilon_p)\bar\varepsilon_p\dot\gamma$ — ✅ **because $\mu\equiv1$**, verified twice | `7ii`, `7jj` |
| Hardening | **two-interval cubic Hermite** in $s=(\bar\varepsilon_p-\bar\varepsilon_p^{\text{peak}}+\delta)/2\delta$ | `7bb` |
| Flow rule | **non-associated**; CPPM return mapping; consistent tangent | `7aa`, `7dd` |
| Solver | **FGMRES + ILU(1)**, latched per timestep on first plastic yield | `7dd` |
| Lode anchor | $g(\theta_L)\equiv1$ at $\theta_L=-\pi/6$, **compression meridian** | `7dd` |
| Convexity floor | $N_\kappa=g^2+gg''\ge\texttt{MIN\_CURVATURE\_NUMERATOR}>0$ | `7ff`, `7gg` |
| Meridian variants | three named types, each verified — see **§7b.3a** | `7jj` |

🔴 **Three of these were wrong on first submission and correct only after measurement** — the sign
(`7aa`, which overturned **my own** C-165/C-171), the apex (`7bb`), and the convexity criterion
(`7ff`, which overturned **my own** C-203). ⚠️ **The last two inversions were mine, both times by deriving
against the wrong object.** ✅ **That is why no scalar factor in this model is typed by hand** (`7aa.7`).

### 7b.3 ✅ Blocking items — ALL CLOSED (Ruling 38, 09-10-2026)

| # | Item | Closure |
|---|---|---|
| **B-7b.1** | $\mu=\dot{\bar\varepsilon}_p/\dot\gamma$ | ✅ **$\mu\equiv1$ exactly** — verified in **two independent parameterisations** to $1.0000000000$ (**C-217**, **C-221**). Dilatancy cannot enter: $-\alpha_\psi\mathbf I$ is purely volumetric and the equivalent-strain norm sees only the deviatoric part. 📌 **So $\dot E^{p,\text{stored}}=H\bar\varepsilon_p\dot\gamma$ is exact**, and 🔴 **my own C-211 and C-215 are withdrawn** |
| **B-7b.2** | Convexity floor on the dimensionless numerator | ✅ **CLOSED** (**C-219a**) — $N_\kappa=g^2+gg''\ge\texttt{MIN\_CURVATURE\_NUMERATOR}$; endpoint step `(2·FRAC_PI_6)/n` so $\pm\pi/6$ are exact |
| **B-7b.3** | Extension-meridian coefficients | ✅ **CLOSED** (**C-222**) — 🔴 **closed by honest naming, not by changing a number.** $k=\frac{2c\cos\phi}{3+\sin\phi}$ is **verified apex-matched to $1.0000000000$**; ✅ **both** branches are apex-matched, and **only the compression branch is additionally tangent to MC** |
| **B-7b.4** | Re-entrant locus; silent clamp | ✅ **CLOSED** (**C-220**) — pre-run sweep, `Err(NonConvexYieldSurface)` under **INV-1**, **no silent clamping** under **INV-7**, clamp warning + linear-convergence fallback |

### 7b.3a 🔴 The three `MeridianType` variants — and the rule that keeps them honest

| Variant | Coefficients | What it actually matches | Verified |
|---|---|---|---|
| **`MohrCoulombCompression`** | $\alpha=\frac{2\sin\phi}{3(3-\sin\phi)}$, $k=\frac{2c\cos\phi}{3-\sin\phi}$ | **MC compression tangency _and_ apex-matched** | `7aa` — $0.0000\%$ error |
| **`ApexMatchedExtension`** | $\alpha=\frac{2\sin\phi}{3(3+\sin\phi)}$, $k=\frac{2c\cos\phi}{3+\sin\phi}$ | **Apex-matched only** — ⚠️ **not** tangent to the MC extension line (**$-40.9\%$ at $P=0$) | **`C-222`** — apex identity to $1.0000000000$ at $\phi=15°/30°/45°$ |
| **`J2EquivalentCylinder`** ⚠️ | $\alpha=0$, $k=\frac{2c}{\sqrt3}$ | $F=\sqrt{J_2/3}-k$; ⚠️ pure-shear yield $\tau=2c$ — 🔴 **not** the von Mises↔Tresca match ($\tau=c$) | **`C-224`** — $\tau=2.000000\times10^6$ vs $1.000000\times10^6$ Pa |

$$\boxed{\text{Every }\texttt{MeridianType}\text{ variant must name the property its coefficients have, verified by a test that evaluates that property.}}$$

📌 **This rule worked:** ✅ `MohrCoulombExtension` → `ApexMatchedExtension` replaced an **incorrect** name with a
**correct** one **without touching a coefficient**. 🔴 It is the same discipline as `7cc.6` — *a name or a constant
that asserts a property must be measured, not asserted.*

⚠️ **Non-blocking:** $\texttt{MIN\_CURVATURE\_NUMERATOR}=10^{-4}$ stays `SOURCE_PENDING` (**L-9**) — confirmed
non-blocking since $N_\kappa\equiv1\gg10^{-4}$ for a circular cone. ✅ **M7b must not wait on it.**

### 7b.4 Tasks

| # | Task | Notes |
|---|---|---|
| 7b.1 | Poroelastic **Biot FEM** — fully coupled, cell-local $P$ | ⚠️ **not** the old $\bar P_{res,\text{eff}}$ single-scalar coupling — **CONF-17, closed**; that quantity is a post-processed diagnostic only |
| 7b.2 | Thermo-mechanical coupling; $k_{th}$ from a cited source | ⚠️ **no cited value exists** — new literature item, same class as **L-1** |
| 7b.3 | Implement the locked constitutive model of **§7b.2** | ⚠️ from `engine_constitutive.md`, **not** from the design set |
| 7b.4 | **Return mapping** — CPPM, consistent tangent | ✅ adopted; ⚠️ DOIs pending (**L-6**) |
| 7b.5 | **Linear algebra** — FGMRES + ILU(1), latched | 🔴 **the M0.1 preconditioner decision is conditional on this** — see below |
| 7b.6 | Rate-and-state friction, **implicit in $V$** (**C-72**) | adds a **nested Newton / Schur complement** |
| 7b.7 | Compaction / porosity change from the plastic strain increment | ⚠️ **Bethel correction required** — **CONF-18**, still open |
| 7b.8 | Emit the constitutive diagnostics to output **domain 5** | per [`output_schema.md`](output_schema.md) |

### 7b.5 Gate

| Metric | Requirement | Basis |
|---|---|---|
| **Limit-case suite** | ✅ **every family reproduces its degenerate limit** — $\phi\to0\Rightarrow$ Tresca; apex in compression; $\sqrt3 F_A\equiv F_B$ | `7cc`, `7hh`. 🔴 **This gate caught two of my own inversions** |
| **Constitutive sign gate** | ✅ every $\pm$ constant evaluated at a state whose value is known independently | `7cc`. Three instances so far: $\sqrt3$, $I_1$, $\theta_L$ |
| **Dimensional gate** | ✅ zero bare literals compared against a physical quantity — **including in guard clauses** | 🔴 **10 failures in this component**; the last was in a guard, i.e. in code written *to be safe* |
| **§7s.6** | ✅ every singularity guard **and every saturating transform** paired with its `ValidityWarning` | `7dd`, `7ee`. A guard without a warning is a build failure |
| **Dissipation** | $\dot D^p=\dot\gamma\left(3\alpha_\psi c\cot\phi-H\bar\varepsilon_p\right)\ge0$, evaluated at **$\alpha_\psi=0$ as well as at realistic dilatancy** | `7ee`, `7ff`, `7ii`, `7jj`. ✅ $\mu\equiv1$ (**C-221**). ⚠️ At the apex $\boldsymbol\sigma=\mathbf{-}c\cot\phi\,\mathbf I$ (compression-positive) and $\partial g/\partial\boldsymbol\sigma\to\mathbf{-}\alpha_\psi\mathbf I$, so $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p=+\;3\alpha_\psi c\cot\phi$ — **positive**, and it vanishes **linearly** at $\alpha_\psi=0$ (**C-225**). 📌 **A gate only ever run at realistic $\alpha_\psi$ never reaches its failure branch** |
| **Convexity** | $N_\kappa\ge\texttt{MIN\_CURVATURE\_NUMERATOR}$ **sampled across the Lode sweep**, reporting the failing $\theta_L$ | `7gg`. ⚠️ the failing angle may be $\theta_L=0$ — the **primary** meridian |
| **$\mu$** | ✅ **assert $\mu\equiv1$** — a definitional identity, **not** a computed quantity | `7ii`. Measured exactly $1.0000000000$ at $J_2\in\{1,\,0.25,\,0.04,\,10^{-3}\}$ × $\alpha_\psi\in\{0,\,0.15,\,0.40\}$ |
| **Two-sided tests** | every equivalence test draws its two sides from **independent** sources | `7aa`. ⚠️ **five checks in this register could not fail** |
| **Code-block pinning** | every submitted Rust block has a `#[cfg(test)]` suite compiling the **exact** implementation | `7gg` |
| **Non-finite** | zero `NaN`/`Inf`; `is_finite()` on $f$, $\mathbf{D}^{alg}$, $\dot\gamma$, $\boldsymbol\sigma$ | `7dd`. 🔴 `f64::clamp` **returns NaN**, and NaN passes every guard |
| **C¹ hardening** | slope residual $0.00$ at all three nodes; $\max c=c_{\text{peak}}$ exactly | `7bb`. Verified |
| **Recovery factor** | $\mathrm{RF}=\dfrac{\int\rho_o^0S_o^0-\int\rho_ot^S_ot}{\int\rho_o^0S_o^0+N_{\text{influx},o}}$ — **oil** influx only | `7y`, `7aa`. **CONF-17, closed** |
| **Integration** | ✅ **M4.5's coherence gate applies here too** — geomechanics must be reachable from the driver and produce one integrated run | [`separation_doctrine.md`](separation_doctrine.md) **C-7** |

⚠️ **No numeric tolerance is invented above.** Where the design set gives one it is cited; where it does
not, the metric is a **property** (residual $\equiv0$, coverage, existence) rather than a tolerance.
📌 **A gate with an invented tolerance is not a gate.**

### 7b.6 🔴 M0.1 unblocks here, and only here

**CONF-14 stays open until M7b.** ⚠️ The design set decides the constitutive model at M7b but the linear
algebra at M0 — and a symmetric preconditioner **silently degrades** an elastoplastic Newton solve from
quadratic to linear, which is exactly the **C-35** failure already logged from the Python attempt.

✅ **Ruled:** M7b's benchmark must include **at least one non-symmetric** test matrix, and the M0.1
preconditioner decision is **conditional on this milestone's outcome** — recorded as a decision with a
benchmark, not a re-derivation.

### 7b.7 Still open in M7b's dependency set

| Conflict | State |
|---|---|
| **CONF-66** | ✅ **constitutive block CLOSED** — items 2 and 3 done, **all four M7b blockers closed** (**C-219a**, **C-220**, **C-221**, **C-222**). ⚠️ Remaining: `VonMisesExtension` → `J2EquivalentCylinder` rename (**C-224**, cosmetic) |
| **CONF-18** | 🔴 **open** — porosity updated by bare volume subtraction, no Bethel correction (task 7b.7) |
| **CONF-13** | ✅ **closed** — Armijo-Goldstein line search + in-core sparse solve (**C-78**) |
| **CONF-14** | 🟠 **open until M7b** — see **§7b.6** |
| **CONF-25** | ⚠️ **narrowed** — flash API, M2 scope |

Also in scope, not separately numbered:

🔴 **Honesty note — resolving scope did not resolve specification.** **M7b now has a body** (written
09-10-2026, below). **M7a and M7c–M7h still have no task lists, no gates, and no measured acceptance
criteria**, exactly as M1–M6 did not before [`spec_defects.md`](spec_defects.md) was written for them.
**An agent must not invent gates for those milestones.**

Each needs the same three-step treatment:

1. read the design source in [`../thmc/`](../thmc/README.md) and log every conflict as `CONF-nn`
2. record the adjudication in [`spec_corrections_log.md`](spec_corrections_log.md)
3. write the milestone body here — tasks, specification gaps, and a **measured** gate

⚠️ **M7b was written first deliberately.** All 36 rulings and `C-1…C-237` landed on the M7b constitutive
model, so it needed the most adjudication to specify — 📌 and two of its four blockers are still open, so a
body written earlier would have had to invent answers.

**Not in scope, and the one remaining phase-gated item:** FFI to Python (**coupling**) — see
[`separation_doctrine.md`](separation_doctrine.md) §6.

---

## Cross-cutting requirements

| Requirement | Applies to |
|---|---|
| **Component-wise mass balance** on every timestep | M4 onward; continuous through M7a–M7h |
| **No silent failure** — every error path is an explicit typed error, never a fallback value | all |
| **Zero `unwrap`/`expect`/`panic!`** in library code | all (enforced by lint) |
| **No clamping** of saturations or compositions to hide solver failure | M3 onward |
| **Seeded determinism** for every stochastic component | M2 (TPD sweep), M5+ |
| **One commit closes one issue**, against `audit_comp` | all |
| Every gate claim carries a **`Status:` line with a command and a measured value** | all |

## What this plan deliberately does not do

| Not done | Why |
|---|---|
| Copy the design set's code listings | They are sketches, and several are demonstrably not what they claim (see `../thmc/solver_and_numerics.md` §2) |
| Adopt the design set's fractional-flow formula as printed | **CONF-01** — the closure is wrong (`spec_defects.md` §1) |
| Adopt the HCPVI exponential cap | **CONF-02** — inert at the operating point |
| Adopt the reduced NPV expression | **CONF-58** — drops four cost terms the live engine already has |
| Reuse `audit/registry.py` | `audit_comp` is a separate implementation ([`register_spec.md`](register_spec.md) §6) |
| Import anything from `core/`, `evaluation/`, `validation/`, `utils/` | [`separation_doctrine.md`](separation_doctrine.md) §1 |
| Validate against the Python surrogate | Coupling. Use independent analytic and CMG references instead |
