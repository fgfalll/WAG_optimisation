# Document Inventory — `3D_THMC_docs/`

Measured 07-10-2026. All 8 files are **untracked** in git (`git status --porcelain` → `?? 3D_THMC_docs/`),
**Ukrainian-language**, and use **plain text with no Markdown headings** — section numbers are bare
numbered lines.

---

## 1. Master inventory

| ID | File | Lines | Bytes | Nature | Ends where |
|---|---|---|---|---|---|
| **D1** | `3D THMC Reservoir Simulator Full Architecture Design Document.md` | 490 | 68 343 | System architecture | **abruptly, mid-sentence inside §11.3** |
| **D2** | `Технічна специфікація розробки ядра draft.md` | 452 | 42 634 | Rust core technical spec, **draft** | cleanly at acceptance checklist |
| **D3** | `Доповнення 1.md` | 443 | 38 163 | Addendum 1 — water / asphaltene / hydrate | cleanly |
| **D4** | `Доповнення 2.md` | 314 | 37 420 | Addendum 2 — remediation / emergency control | cleanly |
| **D5** | `Стратегія верифікації та валідації (V&V Testing Framework) для 3D THMC симулятора.md` | 507 | 56 988 | 6-level V&V framework | cleanly at CI pipeline |
| **D6** | `Специфікація автономного сателітного інструментарію та калькуляторів (Pre- & Post-Simulation Utilities Suite).md` | 399 | 58 606 | Satellite pre/post utilities | cleanly |
| **D7** | `Специфікація генерації синтетичних геологічних даних та інтеграції з сателітними калькуляторами.md` | 177 | 35 665 | Procedural geology generator | cleanly |
| **D8** | `Hydraulic Fracturing and Complex Wellbore Architecture Specification.md` | 282 | 38 772 | HF + wellbore + EDFM + NPV unification | cleanly |

Total: **3 071 lines, 376 591 bytes**.

---

## 2. Section maps (translated; ordering carried by *sub*-numbers only)

### D1 — Full Architecture Design

| § | Title |
|---|---|
| 1.1 | Separation of concerns: In-Core Solver vs Satellite Toolkit |
| 1.2 | Physical regimes and modular configuration |
| 2.1 | Contiguous flat memory slabs, L1/L2 cache locality |
| 2.2 | 3D corner-point IJK (GRDECL) topology and geometric defects |
| 2.3 | Schwarz 3D sub-domains (full-physics 3D zooming) |
| 3.1 | FVM + MPFA-O for `AnisotropicTensor3D` |
| 3.2 | High-order convective schemes (TVD limiters, WENO3/WENO5) |
| 3.3 | FEM (`fenris`/`RustFEA`) for Biot poroelasticity |
| 3.4 | Fixed-stress split and stabilisation |
| 4.1 | Zero-hardcoded EOS engine (PR, SRK, PC-SAFT, CPA) |
| 4.2 | Phase envelope and asphaltene precipitation kinetics |
| 4.3 | Massively parallel GPU flash (`wgpu` / CUDA) |
| 4.4 | Hyper-dual automatic differentiation |
| 5.1 | Acid-leaching kinetics, Damköhler and Péclet numbers |
| 5.2 | Wormhole evolution, porosity–permeability coupling |
| 5.3 | Clay swelling and sorption swelling |
| 5.4 | Chemo-mechanical softening |
| 6.1 | Phase transfer, Stone I / Stone II three-phase relative permeability |
| 6.2 | Capillary trapping (Land) and C¹ hysteresis (Killough / Carlson) |
| 6.3 | Dissolved phase and Setschenov salting-out |
| 6.4 | Langmuir isotherm and irreversible mineral precipitation |
| 6.5 | **5-state trapping summary table** (Free / Trapped / Dissolved / Adsorbed / Mineralised) |
| 7.1 | Fracture geometry and Barton–Bandis aperture |
| 7.2 | Proppant transport, settling, Forchheimer degradation |
| 7.3 | EDFM coupling and NNC fluxes |
| 8.1 | DP/DP and MINC |
| 8.2 | Fault geomechanics: SGR, Coulomb failure stress, slip activation |
| 9.1 | Unconstrained scheduling and PID controllers |
| 9.2 | Multi-segment drift-flux hydraulics, micro-annulus |
| 9.3 | Sub-adaptive sub-stepping with 100 % mass balance |
| 10.1 | Zero-allocation parallel assembly (Rayon) |
| 10.2 | Adjoint gradient engine |
| 10.3 | Solvers stack (faer, russell_sparse, cuDSS) and line search |
| 11.1 | Pre-simulation calculators and tensor upscaling |
| 11.2 | Procedural generator (WFC, noise, erosion); PGS suite; GEP engine |
| 11.3 | Post-simulation analytics, AHM, ES-MDA ensembles — **document ends here** |

### D2 — Rust core spec (draft)

| § | Title |
|---|---|
| 1.1 | Extreme-state handling and zero-panic guarantee |
| 1.1a | Adaptive time-stepping algorithm |
| 1.2 | Basic computational mechanisms; **GEP-driven optimisation integration** |
| 2.1 | Aqueous speciation and Lasaga kinetics |
| 2.2 | Dynamic porosity & permeability evolution |
| 3.1 | THMC coupling formulation |
| 3.2 | HTHP phase state and thermodynamics |
| 4.1 | DP/DP, EDP, MINC conceptual models; Rust data structures; transfer function |
| 4.2 | Barton–Bandis fracture deformation and permeability tensors |
| 5.1 | DFN and NNC |
| 5.2 | Fault shear reactivation (ΔCFS), SGR, seal→conduit |
| 6.1 | Audit corrections, Koval model, dynamic economics |
| 6.2 | **Acceptance-criteria checklist — all boxes `[ ]`** |

### D3 — Addendum 1

| § | Title |
|---|---|
| 1.1–1.3 | Context, Rust integration, AD hand-off, discretised drift-flux momentum equation |
| 2.1–2.4 | PWRI: deep-bed TSS filtration (Iwasaki), external filter cake, OiW + Jamin effect, DTOs |
| 3.1–3.4 | Asphaltenes: AOP thermodynamics, Verma–Pruess kinetics, tubing ID degradation, DTOs |
| 4.1–4.4 | Hydrates: van der Waals–Platteeuw equilibrium, Bischoff–Englezos kinetics, Thomas rheology + smooth plug, DTOs |

### D4 — Addendum 2

| § | Title |
|---|---|
| 1 | Integration architecture; localised source terms `R_dissolution`, `R_kill` |
| 2.1–2.3 | Matrix/hydrochloric acidising, aromatic solvents, backwashing |
| 3.1–3.3 | Hydrate dissolution via THI + heating, depressurisation, conformance control |
| 4.1–4.2 | Well kill, SSSV, water hammer |
| 5.1 | Squeeze cementing and leak sealing |
| 6 | Input DTO architecture |

### D5 — V&V framework

| § | Title |
|---|---|
| 1.1 | Limits of code coverage in numerical simulation |
| 1.2 | Verification vs validation |
| 2.1–2.4 | Level 1: Buckley–Leverett, Terzaghi, Mandel, Sneddon/KGD, Avdonin |
| 3.1–3.3 | Level 2: component-wise mass balance, saturation positivity, MPFA-O M-matrix |
| 4.1–4.3 | Level 3: Gibbs monotonicity, Michelsen TPD multi-start, critical-point scan |
| 5.1–5.3 | Level 4: temporal convergence order, fixed-stress split stability, streamline remapping |
| 6.1–6.3 | Level 5: SPE 1 / 3 / 5 / 9, SPE 10 (1.1 M cells), SPE 11 |
| 7.1–7.4 | Level 6: t=0⁺ impulse, phase boundary appearance, Verma–Pruess clogging, tubing hydrate blockage |
| — | CI/CD pipeline regulation, 3 suites with wall-clock budgets |

### D6 — Satellite utilities

| § | Title |
|---|---|
| 1.1 | Modular organisation, integration stack, DTO schemas, execution modes, stability standards |
| 2.1–2.4 | PVT/EOS fitter, petrophysics/geostatistics/SCAL, geomech/well/geometry, upscaling |
| 3.1–3.3 | Format translators, DTO validation cascade, initialisation |
| 4.1–4.6 | Sweep/RF, DCA/RTA, material balance, history matching + GEP, storage/recycle analytics, economics |

### D7 — Synthetic geology

| § | Title |
|---|---|
| 1.1–1.2 | Zero-Bloat Core paradigm; 4 target use cases |
| 2.1–2.5 | fBm noise; Voronoi + DFN; WFC + L-systems; marching cubes + dual contouring; **PGS 5-step algorithm** |
| 3.1–3.6 | Porosity compaction; FZI/RQI; rel-perm + capillary; EOS fitter; GEP calibration; Archie/Gassmann/Somerton |
| 4.1–4.5 | Satellite pre-sim; DTO conversion; pattern partitioning; post-sim diagnostics; economics |

### D8 — Fracturing & wellbore

| § | Title |
|---|---|
| 1.1 | Module architecture and EDFM integration |
| 1.2 | EoS and delumping; accuracy table |
| 1.3 | Surrogate engine and GEP integration |
| 2.1–2.4 | Fracture geometry; proppant transport; conductivity degradation; Koval breakthrough |
| 3.1–3.4 | Wellbore & casing geometry; cement sheath; perforations; drift-flux |
| 4.1–4.2 | Dynamic CAPEX; unified DCF NPV; 3 numbered NPV-unification acceptance criteria |

---

## 3. Rust code blocks present in the set

| Doc | Items | Derive attributes |
|---|---|---|
| **D2** | `ComputationalError` (5 variants), `ConvergenceControl`, `StateValidator`, `NewtonRaphsonSolver` (+`step_time`, `compute_residuals_and_jacobian`, `calculate_l2_norm`), `DualPorositySystem`, `DualPermeabilitySystem`, `MincSubvolume`, `FracturedContinuumModel` | `#[derive(Debug, Clone, PartialEq)]`, `#[derive(Debug, Clone, Copy)]`, `#[derive(Debug, Clone)]` |
| **D3** | `ParticleSizeBin`, `PwriModuleInputDTO` (+`validate`), `AopPoint`, `AsphalteneModuleInputDTO` (+`validate`), `InhibitorTypeEnum`, `GasComponentFraction`, `HydrateModuleInputDTO` (+`validate`) | `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` |
| **D4** | `RemediationValidationError` (+`Display` +`std::error::Error`), `RemediationInputDTO` (+`validate`) | `…, Serialize, Deserialize]`; error also `Eq` |
| **D5** | `Pascal`, `MoleFraction` (+`new`), `evaluate_tpd_simd` | `#[derive(Debug, Clone, Copy, PartialEq)]` |
| **D8** | `SimulatorError` (+`Display` +`Error`), `FractureGeometry`, `MatrixCell`, `EDFMIntersection`, `trait FractureMatrixFlux: Send + Sync` | `#[derive(Debug, Clone, PartialEq)]`, `#[derive(Debug, Clone)]` |

> [!NOTE]
> All Rust listings are **mangled** by the source conversion: `#[derive(...)]` was written as
> `# [derive(...)]` (space after `#`), and code blocks have **no fences**. `D5` also references
> `Kelvin`, `NumericalDivergenceError`, `fugacity_coeff_gas`, `fugacity_coeff_mixture` **without ever
> declaring them**. Any agent transcribing these to real Rust must re-derive, not copy-paste.

Full verbatim field lists: [`solver_and_numerics.md`](solver_and_numerics.md), [`addenda.md`](addenda.md),
[`fractures_wellbore.md`](fractures_wellbore.md).

---

## 4. Formatting and content defects in the source documents

| # | Defect | Where | Consequence for agents |
|---|---|---|---|
| F-01 | All top-level headings render as `1.` | D1 (11×), D3, D4, D5, D6, D7 (as `1,5,6,1`), D8 | "§5" / "§6" citations are **ambiguous**. Cite by Wiki ID + topic. |
| F-02 | Ordered lists restart mid-document (`1.,2.,3.,3.,1.`) | D2 §1.1, §2.2, §5.1 | Item numbers do not map to headings |
| F-03 | `#[derive(...)]` mangled to `# [derive(...)]`; no code fences | D2, D3, D4, D5, D8 | Listing will not compile if pasted |
| F-04 | Mixed-language artefacts: Russian `вершины`, English `balances`, doubled `штучні штучні` | D1 | Editing/proofing noise |
| F-05 | Doc truncates mid-sentence | D1 line 490 (inside §11.3) | D1 has **no conclusion, no acceptance criteria, no revision history** |
| F-06 | ASCII "tables" are single run-on lines | D1, D2, D5 | Must be re-typed; risk of misreading column alignment |
| F-07 | Duplicate title suffix on D6 and D7 | both end `(PRE- & POST-SIMULATION UTILITIES SUITE)` | Mis-filing risk |
| F-08 | `ω` denotes both acentric factor and miscibility displacement factor | D6, D7 | Ambiguous symbol |
| F-09 | Section-numbering claim in D5 §1.2 promises *temporal and spatial* convergence, but Level 4 is **temporal only** | D5 | The spatial claim is unbacked |

---

## 5. What is absent from the entire set

| Absent item | Verified by |
|---|---|
| Any crate name, `Cargo.toml`, `src/` layout, `mod`/`use` path, dependency list, toolchain version | Full read of D2 (the "core development spec") |
| Any `#[cfg(test)]`, `#[test]`, `assert_*` | Full read of D1–D8 |
| `petekIO` file layout, header, magic bytes, byte order, schema version | Only 3 mentions exist (D1, D6); no layout anywhere |
| **Preconditioners** — no ILU, no AMG, no block preconditioning | grep-equivalent full read of D1, D2 |
| **Sparse storage format** — no CSR/CSC/CSF/ELL, no fill-reducing ordering | Full read of D1, D2 |
| **Any Krylov / iterative linear solver** — only direct (faer, MUMPS, UMFPACK, cuDSS) | Full read of D1, D2 |
| Reference/expected numerical values for any of the 22 V&V tests | Full read of D5 — **zero absolute expected numbers** |
| **Manufactured solutions** | Full read of D5 |
| Spatial grid-convergence / order-of-accuracy study | Full read of D5 |
| Any named comparison simulator (CMG, ECLIPSE, INTERSECT, DECIPHER) | Full read of D5 |
| Any field or dataset name, history-matching target, or acceptance threshold | Full read of D5 |
| Random seeds / determinism / reproducibility clause | Full read of D7 (**0 hits** for `seed`) |
| Grid specification in D7: no `NI/NJ/NK`, no `DX/DY/DZ`, no layering, no NTG (0 hits), no corner-point construction | Full read of D7 |
| Checkpoint / restart hand-off format | Full read of D6 |
| Any CLI contract for any tool | Full read of D6 |
| Price deck, carbon-credit price, OPEX split, IRR, payback | Full read of D6, D7, D8 |
| Units declaration / conversion table for the suite | Full read of D6 |
| PKN, KGD, DPM, cohesive-zone, phase-field, peridynamics, PFC, RBSM fracture models | Full read of D8 — D8 names **only** P3D, Planar-3D, DFN |
| Gravel pack, ICD/AICD design specs, completion types, micro-annulus mechanics | Full read of D8 |
| HF workflow: stage count, cluster spacing, fluid rheology, proppant schedule | Full read of D8 |
| `tests/test_surrogate_engine.py` (referenced by D2) | `Test-Path` → **False** |