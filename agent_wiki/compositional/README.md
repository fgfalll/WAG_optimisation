# Compositional Engine — Build Section

> [!WARNING]
> **🦀 Under massive active development. This is the 3D THMC reservoir simulator, being built from
> scratch in Rust.**
>
> 🔴 **No engine code exists.** Nothing on these pages is a report of running software — every page is a
> **specification, a correction, or an audit trail** describing **what the engine must do**.
>
> | | 🐍 Python engine | 🦀 **This engine** |
> |---|---|---|
> | Status | ✅ **Shipped** | 🔴 **Under construction — no code** |
> | Docs | [`../architecture/`](../architecture/overview.md) | these pages + [`../thmc/`](../thmc/README.md) |
>
> ⚠️ **The name is a legacy of an earlier narrower scope.** 🔴 **"Compositional" here means the *whole* 3D THMC
> simulator** — geomechanics, geochemistry, fractures and all. Taking it literally means shipping about
> **one third** of approved scope. See [§5.4](vision_and_phases.md).

**Why it exists.** The app has one engine today: a **surrogate**, built for fast screening of thousands
of candidates per optimisation run. During development it became clear that **no available simulator
covers the full spectrum of this project's tasks** — the surrogate's simplifications are acceptable for
*ranking* and unacceptable for *evaluating*. A full-physics compositional engine is needed, in Rust.

> **Prior art:** an `EngineFactory` **and a Python compositional engine were built, then removed** after
> the attempt failed. Rust is the second attempt at the same goal. See
> [`engine_invariants.md`](engine_invariants.md) §6 — including why
> `agent_wiki/development/extension_points.md` currently documents a **deleted** component.

**Current state.** Work has **not started**. This section exists so that when it does, an agent has the
design source normalised, the conflicts adjudicated, the spec gaps indexed by milestone, the build gates
measurable, and the integration/data-model work scoped rather than discovered late.

| | |
|---|---|
| **Vision and phases** | [`vision_and_phases.md`](vision_and_phases.md) — P1 develop → P2 integrate → P3 couple |
| **Mandatory invariants** | [`engine_invariants.md`](engine_invariants.md) — 🔴 **read before writing any code.** Seven non-negotiable invariants, **INV-1…INV-7**. ⚠️ **The file is ~220 KB — use its §0 index, not an end-to-end read** |
| **Status** | **No code exists.** Rust toolchain installation is **deferred by decision** until after documentation, at the start of development — *"some crates can be changed in process"* |
| **Design source** | [`../thmc/`](../thmc/README.md) — the 8 documents in `3D_THMC_docs/`, normalised |
| **What it is not** | A plan to modify `core/engine_surrogate/` — out of scope until P2 |

> [!CAUTION]
> **Read [`separation_doctrine.md`](separation_doctrine.md) before creating any file.**
> Three toolchain blockers were found that make "separate" non-trivial, and one of them means the
> existing flaw register **physically cannot record a Rust finding**:
>
> `audit/registry.py:114` — `LOCATION_RE = re.compile(r"`([\w/\\.\-]+\.py)(?::(\d+)…)?`")`
> **The location regex accepts `.py` only.** A `crates/compositional/src/flash.rs:120` location is
> rejected by the schema. Also `audit/registry.py:300` validates every location against `REPO_ROOT`,
> and `audit/continuity.py:150-189` **hard-imports** `core.engine_surrogate` and
> `core.data_models`.
>
> A separate register module is therefore **required**, not optional. See
> [`register_spec.md`](register_spec.md).

---

## 1. Scope: what "compositional engine" means here

The design set describes 5 physical regimes, 9 modules, a satellite suite, and GPU compute. That is a
24–36 person-month programme. **The compositional engine is a strict subset.** Definition of done:

### 1.1 In scope (M0 → M6)

| Capability | Requirement |
|---|---|
| **Components** | $N_c \ge 4$: C1, C2–C3, C4–C6, C7+, CO₂ (matches the D6 `PVT_EOS_DTO`) |
| **EOS** | Peng-Robinson + volume translation (VT-PR); SRK + VT as second. PR/SRK as written, not "configured generically" |
| **Flash** | Michelsen stability test (TPD) → isothermal split, Rachford-Rice with Newton root-find, **analytic derivatives** w.r.t. $P$ and $x$ |
| **Discretisation** | Structured-grid FVM, corner-point optional later, single-point transmissibility first; SoA `Vec<f64>` flat slabs |
| **Relative permeability** | Corey/Brooks-Corey per phase, **with $S_{or}$, $S_{wi}$, $S_{gc}$, $S_{gr}$**; Stone I / Stone II for three-phase |
| **Nonlinear solve** | Newton-Raphson with **analytic Jacobian** (hyper-dual or hand-derived), line search, adaptive timestep |
| **Conservation** | **Component-wise** mass balance to `< 1e-12` on every timestep — per D5 §3.1 |
| **Wells** | Single-phase well, Darcy/Vogel IPR, BHP or rate control, per-well mass accounting |
| **Geometry** | Structured grid `N_x × N_y × N_z`, `ActiveMask`, zero-volume cell handling |
| **IO** | **One** format, fully specified, with a version field (answers **CONF-30b**) |
| **Output** | Per-component rates, cumulatives, pressure/saturation fields, timestep series |

### 1.2 Scope of what may be built **early**

🔴 **Scope was re-baselined 08-10-2026 to the full 3D THMC programme** —
[`vision_and_phases.md`](vision_and_phases.md) §5.2 #10, [`build_plan.md`](build_plan.md) §M0.2b.
Geochemistry, THMC/Biot FEM, fractures, faults, GPU flash and adjoint gradients are **approved scope,
not deferred**. They sit in **M7a–M7h**, which are listed but **not yet specified** — no tasks, no gates.

**None of these may be started early.** Each is a separate V&V programme, and starting one before the
compositional base is verified reproduces exactly the failure the audit documented for the current
engine — physics accumulating on an unverified foundation. ⚠️ **"In scope" means "will be built", not
"may be begun at M1".** The M1 → M6 order is unchanged.

---

## 2. Build ladder

Each gate is **measurable and must be recorded** in a dated run audit under
[`../audit/simulation_run_audits/`](../audit/simulation_run_audits/index.md) with a **Verdict**.

| # | Milestone | Gate to pass | Est. |
|---|---|---|---|
| **M0** | Toolchain, crate skeleton, **separate register + CI**, doc spine | `cargo build` green; register validates; forbidden-import check runs | 1 wk |
| **M1** | **EOS + PVT standalone, no flow** | $Z$-factor, density, viscosity vs. **published reference data**; MAPD target **≤ 2 %** | 6–8 wk |
| **M2** | **Flash: stability + split, with analytic derivatives** | Michelsen TPD over seeded random points; Gibbs $G$ monotone descent; Jacobian vs. central difference `< 1e-6` | 6–8 wk |
| **M3** | **1D two-phase FVM, single component** | **Buckley–Leverett** front position vs. Welge construction **≤ 0.1 %**; temporal order $O(\Delta t)$ measured | 6–8 wk |
| **M4** | **1D compositional FVM** | **Component-wise** mass balance `< 1e-12` per component per step | 8–10 wk |
| **M5** | **3D grid + wells + IPR + IO + reporting** | $L_2$-norm Richardson spatial order study; five-spot grid-rotation invariance to 45° | 8–10 wk |
| **M6** | **CO₂-EOR specifics**: miscibility/MMP, trapping inventory, Koval sweep | Net-utilisation and trapping reconciliation; `Σ S = 1` on **100 %** of steps | 8–10 wk |

**🔴 The M0–M6 ladder covers roughly one third of approved scope.** The estimate below is the
*compositional core only*, retained as the M1–M6 cost basis. The full 3D THMC programme adds **M7a–M7h**,
which are **in scope but not yet specified** — so **no defensible total exists today**
([`build_plan.md`](build_plan.md) §M7). M7 (geochemistry, THMC, fractures, faults, GPU, adjoint,
Schwarz, remediation) is a separate scoping exercise, not a schedule extension.

### 2.1 Why M1 and M2 get disproportionate effort

Thermodynamics is both the **critical path** and the **weakest-specified** part of the design set:

| Spec problem | Consequence for M1/M2 |
|---|---|
| **CONF-31** — no coefficient given for *any* named correlation (Standing, Glaso, Beggs–Robinson, Joback-Reid, PPR78, Huron–Vidal, QSPR) | Critical properties must be sourced from an external, cited dataset — not invented |
| **CONF-49** — declared MAPD density error **3–9 %** for PR/SRK | If realised, recovery differences between designs sit inside fluid-model noise. **M1 must demonstrate ≤ 2 %** or the whole optimisation is noise-limited |
| **CONF-10** — volume translation $s = f((M\omega)^{-1})$ with $M$, $\omega$ undefined and no correlation | The VT coefficient is not specified. Must be fitted to reference density data and the fit recorded |
| **CONF-25** — D5's `evaluate_tpd_simd` references 4 undeclared identifiers | Flash API must be designed, not transcribed |
| **CONF-14** — no preconditioner, no sparse format, no Krylov specified | Linear algebra stack must be chosen and benchmarked in M0 |

> **The engine's validity is bounded by its EOS accuracy.** If M1 lands at 5 % MAPD, M6's recovery
> numbers carry ±5 % fluid uncertainty and the optimiser will rank noise. **M1 is a hard gate, not a
> sprint item.**

### 2.2 Reference data required before M1

The design set supplies **zero** reference values (see [`../thmc/vv_testing_framework.md`](../thmc/vv_testing_framework.md) G-01).
Before M1 starts, a cited PVT reference dataset must be in the repository:

- Pure-component $T_c$, $P_c$, $\omega$ for C1, CO₂, N₂, H₂S, and the C2–C6/C7+ pseudocomponents
- A published binary VLE dataset for $k_{ij}$ regression (the design names a **> 900-system QSPR
  database** but does not identify it — **CONF-31**)
- Liquid density at HTHP to calibrate VT-PR / VT-SRK against

### 2.3 Reuse of existing reference solutions — the one narrow exception

⚠️ The decision is "no coupling". Coupling means **code** coupling. The repository already contains
**analytic reference solutions** in `tests/scientific/`:

| Existing artefact | Usable as an independent check? |
|---|---|
| `tests/scientific/reference_solutions/test_buckley_leverett_analytical.py` — independent Welge construction via SciPy root-finding, asserts `0.20 < ed_welge < 0.90` | **Yes, as reference values only.** Gives M3 an independently computed breakthrough efficiency to compare against. |
| `tests/scientific/manufactured_solutions/test_mms_pressure_diffusion.py` — MMS convergence test | **Yes, as reference values.** Supplies the M5 spatial-order method (**CONF-26**) and a known MMS source term. |
| `tests/scientific/conservation/test_mass_conservation.py` | **Yes, as reference values** for M4's `1e-12` gate. |
| `validation/spe5_config.py` — SPE 5 parameters ($7\times7\times3$, 2 100 ft, 10° dip, $\phi=0.35$, $K=500/50/200$ mD, 4 000 psia, $S_{wi}=0.16$) | **Yes, as parameters.** |
| `validation/cmg/flu/` `gmflu001`–`gmflu003` CMG reference outputs, read via `h5py` in `validation/sr3_reader.py` | **Yes, as expected values** — this is the comparison simulator D5 says it lacks. |
| `evaluation/mmp.py` — 5 validated MMP correlations | ⚠️ **Do not link the Rust crate to this Python module.** Re-implement and cross-validate; a shared dependency would be coupling. |

**Rule:** the compositional crate imports **nothing** from this repository. Reference **values** may be
transcribed into the Rust side with attribution, and the independent Rust implementation is then
checked against them. Any file in `tests/scientific/` that imports `core.data_models` (all of them do)
**may not be linked** — only its numeric expectations may be reused.

---

## 3. Separation summary — **Phase 1 discipline, not architecture**

> Separation exists **"for development purpose to not distract"** (owner, 07-10-2026). It is a **P1**
> discipline. The runtime-code gates are **retired by ADR at the start of P2**. See
> [`vision_and_phases.md`](vision_and_phases.md) §2.

| Dimension | Policy | Scope |
|---|---|---|
| **Runtime code** | The compositional crate imports **nothing** from `core/`, `ui/`, `evaluation/`, `utils/`, `validation/`. Enforced by CI gate C-1 | **P1 only** |
| **Flaw register** | Separate module, separate file, `COMP-nn` namespace — [`register_spec.md`](register_spec.md) | Recommended **permanent** |
| **Continuity gate** | Separate module. `audit/continuity.py` hard-imports the surrogate and cannot be reused | **Permanent** — must not import `core.*` even after P2 |
| **Test suite** | Rust-native (`cargo test`). `pyproject.toml` sets `testpaths = ["tests"]`, so Rust tests are not auto-collected | **Permanent** |
| **CI** | Separate workflow. Existing gates (`pytest`, `ruff --select F821`) stay scoped to Python | **Permanent** |
| **Git discipline** | One commit closes one issue, against the **compositional** register | **Permanent** |
| **Docs** | This section + `../thmc/`. No shared claim about engine behaviour | **Permanent** |
| **Coupling** | **Deferred to P3.** Undesigned now — designing it early is how two engines end up sharing state | **P1 prohibition** |

---

## 4. Immediate blockers

| # | Blocker | Status |
|---|---|---|
| **B-1** | **No Rust toolchain on this machine** — `cargo`, `rustc`, `rustup` all `NOT FOUND` (verified 07-10-2026) | 🔴 Must be installed before M0 |
| **B-2** | Register schema rejects `.rs` locations (`audit/registry.py:114`) | 🔴 Needs the new module |
| **B-3** | Continuity gate hard-imports the surrogate (`audit/continuity.py:150-189`) | 🔴 Needs a separate module |
| **B-4** | No reference PVT dataset in the repository for M1 | 🔴 Must be sourced and cited |
| **B-5** | No CI configuration exists for any language | 🟠 Part of M0 |
| **B-6** | `audit/registry.py` is invoked by `audit/pipeline.py` and `audit/__main__.py`; the new module must not be swept into the Python gate | 🟠 Part of M0 |

---

## 5. Page index

### 5a. Mandatory — read before writing any code

| Page | Purpose |
|---|---|
| 🔄 [`HANDOVER.md`](HANDOVER.md) | **ROLLOVER GATE.** ⚠️ **Read in full on compaction, session resume, or agent handoff.** ~6 KB by design. Carries current state, the seven invariants, **§3 RETIRED — do not re-derive**, and the standing CI rules. 🔴 The docs below total 1,087 KB and **cannot be re-read linearly** |
| [`engine_invariants.md`](engine_invariants.md) | 🔴 **INV-1…INV-7** + ten standing CI rules. ⚠️ **Start at its §0 index**, which maps milestones to sections — do not read end to end |
| [`engine_spec_closures.md`](engine_spec_closures.md) | Adjudicated `CONF-*` closures, DOI resolution, provenance and literature rules (§7c–§7n) |
| [`engine_numerics.md`](engine_numerics.md) | Architecture, newtypes and the status lattice, determinism, tolerances, precision (§7o–§7w) |
| [`engine_constitutive.md`](engine_constitutive.md) | Fracture scaling, Drucker–Prager, hardening, Lode angle, dissipation — **the M7b content** (§7x–§7hh) |
| [`build_plan.md`](build_plan.md) | M0→M6 + **M4.5 integration slice** + **M7a–M7h**. Each gate carries a `Status:` line and a measured value |
| [`vision_and_phases.md`](vision_and_phases.md) | P1→P2→P3, and every decision taken. ⚠️ **§5.4 explains why "compositional" is a misleading name** |

### 5b. Interface contracts — approved or decided

| Page | Purpose |
|---|---|
| 🔵 [`output_schema.md`](output_schema.md) | **APPROVED 08-10-2026** — 9 output domains, 116 fields/cell/timestep, three temporal resolutions, persistence formats, UI mapping. ⚠️ Carries **CONF-63…68** |
| 🔵 [`data_architecture.md`](data_architecture.md) | **DECIDED** — PostgreSQL + `pgvector`, layered file substrate, run-spec format |

### 5c. Problems, gaps & research — 🔴 nothing here is resolved

| Page | Purpose |
|---|---|
| 📚 [`literature_todo.md`](literature_todo.md) | **L-1…L-9** — outstanding research, each with the 3-part acceptance standard and a search protocol. ✅ **L-1 (Karakas & Tariq α₀ table) CLOSED 09-10-2026** — 🔴 **but it exposed C-227…C-229; the perforation-skin expression is still wrong.** ⚠️ **L-8 ($S_v$/$S_{wb}$) is unobtainable** — do not gate work on it |
| 🔴 [`../thmc/conflict_and_gap_register.md`](../thmc/conflict_and_gap_register.md) | **68 conflicts** — the authoritative status. **24 closed · 0 blocking reds · 26 open** |
| 🔴 [`spec_corrections_log.md`](spec_corrections_log.md) | **The authority rule** — `3D_THMC_docs` are immutable, the wiki carries the corrected spec. **C-1…C-237 across 38 rulings**, including **three retractions of my own claims** |
| [`spec_defects.md`](spec_defects.md) | `CONF-*` items to fix **in the spec**, indexed by milestone. **M1 references verified** |
| [`data_model_gap.md`](data_model_gap.md) | Evidence that `core/data_models.py` and `config/base_config.json` cannot express a full simulation |
| [`python_attempt_postmortem.md`](python_attempt_postmortem.md) | 🔴 **Why the first attempt failed** — six numerical-method root causes, all language-independent. **Read before M2** |

### 5d. Process & discipline

| Page | Purpose |
|---|---|
| [`separation_doctrine.md`](separation_doctrine.md) | **P1-only** discipline, plus **C-7** — the coherence gate that makes disconnected modules impossible. 🔴 **Permanent, never retired** |
| [`register_spec.md`](register_spec.md) | **INV-5** — the separate flaw register. 🔴 The Python register *cannot* hold a `.rs` finding |
| [`integration_plan.md`](integration_plan.md) | **P2** engine routing, data models, value pool. ⚠️ UI/UX **out of scope** until the engine is verified |
| [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) | The 07-10-2026 adjudication, including the **CONF-01 retraction** |

---

## 6. Documentation audit — 09-10-2026

Audit of this section for internal contradiction and for anything that would let an agent build
**disconnected** modules. Findings and what was done:

### 6.1 Fixed — factual contradictions against recorded decisions

| # | Defect | Severity | Action |
|---|---|---|---|
| **A-1** | 🔴 [`build_plan.md`](build_plan.md) §M7+ read **"Deferred, each a separate decision"**, contradicting §M0.2b (*"the earlier deferral is **void**"*), [`vision_and_phases.md`](vision_and_phases.md) §5.2 #10, and [`output_schema.md`](output_schema.md) §9. An agent reading M0→M6 then reaching §M7+ would treat **most of the approved engine as out of scope** | **Critical** | ✅ Rewritten as **§M7 — IN SCOPE**, with M7a–M7h tabulated, their blocking conflicts listed, and an explicit *do not invent gates* instruction |
| **A-2** | 🔴 This README's §1.2 listed geochemistry, THMC/Biot, fractures, faults, GPU and adjoint as **"explicitly out of scope"** — the same contradiction, in the **entry-point file** | **Critical** | ✅ Rewritten as §1.2 "Scope of what may be built early", separating **in scope** from **may be started at M1** |
| **A-3** | 🔴 This README stated **"Total ≈ 8–11 person-months"**, which §M0.2b explicitly voids (*"This is not an 8–11 person-month plan"*) | High | ✅ Replaced: the estimate is now labelled *compositional core only*, and it states plainly that **no defensible total exists today** |
| **A-4** | ⚠️ This README's page index claimed *"19 corrections + 7 open branches"* — the log is **C-1…C-237 across 38 rulings** | Medium | ✅ Updated, with the register's live state |
| **A-5** | ⚠️ The invariants summary here listed **4** invariants; **INV-5** and **INV-7** were missing. `register_spec.md` (which *owns* INV-5) contains **zero** `INV-` references | Medium | ✅ Full INV-1…INV-7 list restored in both places. ⚠️ The `register_spec.md` gap is **unfixed** — see Q4 |
| **A-6** | 🔴 [`engine_invariants.md`](engine_invariants.md) had **no index**, and its section order was `§1…§9` then `§7c…§7hh` — so the "Invariant summary" sat at line 337 with **2 500 lines of logically-earlier material after it** | High | ✅ Added **§0 INDEX**, then ✅ **split into four topic files** (§-numbers preserved exactly; all 43 verified to resolve) |
| **A-7** | 🔴 **Nothing prevented disconnected modules** — no integration gate, no "module must be wired in", no end-to-end test. M1–M4 are independent stacks, each passing its own gate, with **nothing requiring them to work together** | **Critical** | ✅ **M4.5 Integration Slice** + ✅ **C-7 coherence gate** (permanent, never retired) |
| **A-8** | 🔴 `register_spec.md` **never named INV-5**, though it *is* the implementation of it; `separation_doctrine.md` mapped none of the seven | Medium | ✅ Both fixed. **73-finding** figure in `register_spec.md` verified before asserting |

### 6.2 Open — needs your decision, not an assumption

All four questions were answered on 09-10-2026 and are now implemented:

| # | Question | Decision | Where |
|---|---|---|---|
| **Q1** | Coherence enforcement | ✅ **Both** — a merge gate *and* a vertical slice | **M4.5** in [`build_plan.md`](build_plan.md); **C-7** in [`separation_doctrine.md`](separation_doctrine.md) |
| **Q2** | M7 milestone bodies | ✅ **M7b first** | ✅ **M7b body written** — [`build_plan.md`](build_plan.md) M7b. ⚠️ **M7a, M7c–M7h remain unspec'd by decision** |
| **Q3** | File size | ✅ **Split into per-topic files** | ✅ **Done** — §-numbers preserved exactly; verified all 43 resolve |
| **Q4** | Unlinked governing docs | ✅ **Add INV references to both** | ✅ **Done** — `register_spec.md` (INV-5), `separation_doctrine.md` (all seven) |

🔴 **Still open — and deliberately so:**

| # | Gap | Why it stays open |
|---|---|---|
| **R-1** | **M7a, M7c, M7d, M7e, M7f, M7g, M7h have no bodies** | By decision (Q2). §M7 says so explicitly and says *do not invent gates*. Each needs the same 3-step treatment M7b got: read source → log `CONF-nn` → adjudicate → write a **measured** gate |
| **R-2** | **M7b's four blockers are open** — $\mu$ formula, convexity-floor units, `Extension` branch, free anchors | 🔴 These are **physics**, not documentation. They are recorded as **B-7b.1…4** and the milestone gate cannot pass with any outstanding. ⚠️ **Not invented closed** |
| **R-3** | **Two pre-existing defects outside this section** — `agent_wiki/development/common_pitfalls.md` has extensive mojibake; two run-audit `audit.md` files use 3 directory-ups where 4 are needed | Reported, **not touched** — outside this section's scope |
| **R-4** | No cited $k_{th}$ for thermo-mechanical coupling (task 7b.2) | New literature item, same class as **L-1**. Declared as a gap rather than assumed |

### 6.3 Verified sound — no action taken

- **Cross-links**: all resolve across all 25 files.
- **Encoding**: clean, no mojibake.
- **Conflict register**: 68 conflicts, 72 index rows, 23 closed, 0 blocking reds.
- **Corrections log**: `C-1…C-237`, 38 rulings.
- **`build_plan.md` M0.1 blocked-on-M7b note**: consistent with `CONF-66`'s state in the register.
- **§M7 honesty note**: correctly states that all 36 rulings landed on **M7b**, ahead of its milestone.
