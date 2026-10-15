# Engine Invariants — Mandatory

**These are not preferences. They are constraints on the compositional engine's design, and violating any
of them is a defect.** Decided by the project owner, 07-10-2026.

---

## 0. INDEX — read this, then jump

> ⚠️ **This file was ~220 KB in one document on 09-10-2026 and has been split into four topic files.**
> 📌 **Section numbers were preserved exactly** — every ruling and cross-reference that cites `7c`…`7hh`
> still resolves to the same content, just in a different file.

| File | Contains | Read it when |
|---|---|---|
| **`engine_invariants.md`** (this file) | The **seven invariants**, prior art, the invariant summary, and the decisions log | **Always.** §1–§9 are short |
| [`engine_spec_closures.md`](engine_spec_closures.md) | `CONF-*` closures, DOI resolution, provenance and literature rules — **7c…7n** | Working on a spec conflict, a citation, or a correlation's provenance |
| [`engine_numerics.md`](engine_numerics.md) | Architecture, newtypes and the status lattice, determinism, tolerances, precision, cutbacks — **7o…7w** | Building the solver, IO, checkpointing, or the status/tolerance model |
| [`engine_constitutive.md`](engine_constitutive.md) | Fracture scaling, Drucker–Prager, hardening, Lode angle, dissipation — **7x…7hh** | Building **M7b** geomechanics, or **M7c** fractures |

### 0.1 The seven invariants — read all of these

| ID | Invariant |
|---|---|
| **INV-1** | **Fail loudly. Never fall back.** Typed error, stop. No fallback value, ever |
| **INV-2** | **Runs only when the user started it** — no automatic invocation |
| **INV-3** | **Simulation only** — no NPV, costs, prices |
| **INV-4** | **Output is training-ready** — labelled $(x,y)$, full fields, versioned |
| **INV-5** | **Own flaw register** — separate, `COMP-nn` namespace |
| **INV-6** | **Capability declaration ≠ failure** — `NOT_IMPLEMENTED` ≠ `FAILED` |
| **INV-7** | **Unconstrained by construction** — absurd input still gets full physical evaluation |

### 0.2 Standing rules — these bind every milestone

These are spread across the split files but apply **everywhere**. Source file given for each.

| Rule | What it forbids | Source |
|---|---|---|
| **Symbol Register Rule** | No undeclared, unvalued, or undimensioned symbol in an equation | 7k |
| **7s.6** | A singularity guard **or any saturating transform** without its paired `ValidityWarning` | 7s · 7dd · 7ee |
| **Limit-case suite** | Every constitutive family must reproduce its degenerate limit ($\phi\to0\Rightarrow$ Tresca, etc.) | 7bb · 7cc |
| **Constitutive sign gate** | Every $\pm$ constant evaluated at a state whose value is known independently | 7cc |
| **Dimensional gate** | A bare numeric literal compared against a physical quantity — *including in guard clauses* | 7w · 7y · 7ee |
| **Code-block test pinning** | A submitted code block without a `#[cfg(test)]` suite compiling the **exact** implementation | 7gg |
| **Two-sided independence** | An equivalence test whose two sides share an input | 7aa |
| **Cumulative edits** | A guarded expression rewritten without re-verifying its checklist | 7dd |
| **Zero-denominator rule** | Any expression that can divide by zero without a loud failure | 7s.5 |
| **Derive-then-assert** | A convexity/monotonicity criterion written from memory instead of derived from its defining geometry | 7ff |

### 0.3 Milestone → what to read

| Milestone | **Read at minimum** |
|---|---|
| **M0** toolchain / skeleton / register | INV-1, INV-5, INV-6 · 7k · 7p · 7q · 7r · 7s |
| **M1** EOS / PVT | INV-1 · 7o · 7p · 7u · 7t · 7s.5 · 7w |
| **M2** flash | INV-1 · 7i · 7j · 7k · 7u · 7n *(correlation provenance)* |
| **M3** 1D two-phase FVM | INV-1 · 7o · 7p · 7t · 7v · 7w · 7s.5 |
| **M4** 1D compositional | INV-1 · 7d · 7e · 7f · 7g · 7h · 7o · 7s.5 |
| **M5** 3D grid / IO / output | INV-1, INV-4 · 7c · 7d · 7q · 7r · 7s |
| **M6** CO₂-EOR | INV-1, INV-3 · 7c · 7d · 7f · 7x |
| **M7a** geochemistry | INV-1, INV-7 · 7n · 7m *(provenance and literature rules)* |
| **M7b** THMC / geomechanics | INV-1 · **the whole `engine_constitutive.md`, 7x–7hh** |
| **M7c** fractures | INV-1 · 7x ($C_K$) · 7bb (two-interval PCHIP) |
| **M7d–M7h** | INV-1 only — ⚠️ **not yet specified**, see [`build_plan.md`](build_plan.md) M7 |

⚠️ **A milestone number alone does not mean you can skip a standing rule.** Section 0.2 binds all of them.

## 1. INV-1 — Fail loudly. Never fall back.

> *"it is mandatory that if for some reason engine errors out it throws proper error and stops to
> reverting or other things permited. and this is impossible by design"*

**Required behaviour, without exception:**

| Condition | Required | Forbidden |
|---|---|---|
| Non-convergence | Return a typed error and **stop** | Revert to the surrogate, retry silently with a looser tolerance, or emit a partial result |
| Flash failure | Return `ThermodynamicFlashFailed` and **stop** | Substitute a single-phase guess |
| Conservation violation | Return an error and **stop** | Renormalise and continue |
| Timeout / resource exhaustion | Return an error and **stop** | Return the last converged step |
| IO or schema error | Return an error and **stop** | Load a default and proceed |
| Any postcondition failure | Return an error and **stop** | Warn and continue |

**Why this is not negotiable:** the compositional engine produces **full-physics evaluations that a
neural surrogate is trained on** (INV-3). A silently-degraded result is not merely imprecise — it
becomes **training data**, propagating the defect into the surrogate permanently and invisibly.

> ⚠️ **This is the failure mode to design against.** The pattern to refuse is:
> `if compositional.run() failed: return surrogate.run()`. It ships invalid results as valid, and with
> INV-3 it *also* poisons the training set. There is no safe fallback — which is precisely why the owner
> calls it impossible by design.

### 1.1 Enforcement

| Mechanism | Requirement |
|---|---|
| **Return type** | `Result<T, CompositionalError>`. Never a partially-populated output struct |
| **Error taxonomy** | Typed enum, D2 §1.1's `ComputationalError` as the starting point. Every failure mode has a variant |
| **Zero panics** | `#![deny(unsafe_code)]` + clippy `unwrap_used`/`expect_used`/`panic` denied |
| **No defaults** | No `unwrap_or_default()`, no `unwrap_or(0.0)`, no "if missing assume X" |
| **CI gate** | A test asserting that **every** error variant is reachable and that no code path returns a
  value while an error is pending |
| **Manifest** | A run manifest records engine identity **and** terminal status. A failed run produces **no**
  usable artifact |

---

## 2. INV-2 — The engine runs only when the user starts it

> *"this engine can run only if user started it"*

**Required:**

| Rule | Detail |
|---|---|
| **No automatic invocation** | Never called from an optimisation loop, a parameter sweep, a sensitivity analysis, or any background/batch path |
| **No fallback in either direction** | The surrogate must never trigger the compositional engine; the compositional engine must never trigger the surrogate |
| **No heuristic dispatch** | Cost, accuracy or convergence must not cause automatic engine selection |
| **Explicit user action only** | A named, deliberate user action starts a compositional run |
| **Cost disclosure before commit** | The UI shows expected runtime and scale **before** the user commits. Not after |

**Consequence for the optimiser:** optimisation runs use the surrogate, full stop. Since the compositional
engine is roughly $10^3$–$10^5\times$ slower, this is also the only workable arrangement — but it is a
**rule, not a performance heuristic**, so it must not be expressed as "unless it's too slow".

---

## 3. INV-3 — Simulation only. No economics.

> *"we will develop separate economic engine that will do all the field development calculations and
> compositional engine should stay as a simulation only"*

**The compositional engine contains no:** NPV · IRR · payback · CAPEX · OPEX · discounting · price deck ·
carbon credit · storage credit · carbon tax.

Its output is **physical state and rates only**: pressure, per-component saturation, per-component
composition, per-component rates and cumulatives, well states, timestep history.

### 3.1 What this resolves — and what it moves

| Item | Effect |
|---|---|
| **CONF-04** (economics exiled from core, yet given an in-core adjoint) | **Resolved by construction.** There is no economics in the engine, so there is no placement question and no NPV adjoint inside it |
| **CONF-58** (reduced NPV expression drops CO₂ purchase, recycle, storage credit, carbon tax) | **Moves** to the future economic engine. The content still stands: **that** engine must carry all four terms |
| **CONF-59** (recycled CO₂ booked as gas revenue) | **Moves** likewise. The produced-gas stream must reach the economic engine **split** into sales gas and recycle CO₂ |
| **CONF-37** (no price deck) | **Moves.** The economic engine needs a complete, sourced value pool |
| **CONF-11 / CRIT-18** (gas sales contribute $0) | **Moves.** Fixed in the economic engine, not here |

> ⚠️ **Design consequence.** The compositional engine must publish produced gas as **two distinct
> streams** — hydrocarbon sales gas and CO₂ — rather than a single combined `gas_rate`. If it publishes
> one number, the economic engine cannot avoid CONF-59, and no downstream fix can recover the split.

---

## 4. INV-4 — Output must be training-ready

> *"to 'teach' surrogate engine neural network (yet to be developed) for current reservoir per user
> request"*

The compositional engine has **two purposes**: full-physics runs, and **generating the labelled dataset
that trains a per-reservoir neural surrogate** (not yet built).

**Required:**

| Rule | Detail |
|---|---|
| **Labelled dataset emission** | Every run emits $(x, y)$ pairs: input control vector $x$ → full output state $y$ |
| **Per-reservoir** | The surrogate is trained **for the current reservoir**, not universally. Each reservoir gets its own training set |
| **Full fields, not summaries** | $y$ must include per-cell pressure, per-component saturation and per-component composition — a surrogate trained on profiles cannot represent fingering or channeling |
| **Determinism** | Same $(x, \text{reservoir}, \text{seed})$ ⇒ bit-identical $y$. Already required by the M2/M5 gates; **now also required for dataset integrity** |
| **Versioned schema** | The dataset schema is **versioned**. Retraining after an engine change requires knowing which engine version produced which samples |
| **Engine provenance per sample** | Each sample records the engine version, build hash, and run manifest id |

> 🔴 **This constraint bites in P1, not P2.** The output schema determines what can be trained on. If it
> is decided late and turns out inadequate, **every historical run must be re-run** — at ~$10^3$–$10^5\times$
> the surrogate's cost. **Decide the output schema during M5**, alongside IO (D9).

---

## 5. INV-5 — Simulation register stays separate

The compositional engine's findings use `audit_comp/` with the `COMP-nn` namespace
([`register_spec.md`](register_spec.md)). This is **recommended permanent**: two engines will have two
finding histories, and merging them would bury new-engine defects under 73 legacy ones.

---

## 6. 🔴 INV-6 — Capability declaration vs. failure (added 08-10-2026)

**The problem INV-6 solves.** The approved output schema
([`output_schema.md`](output_schema.md)) spans **nine domains** — hydrodynamics, thermodynamics,
petrophysics, geochemistry, geomechanics, 5-state trapping, fractures/faults, wellbore, diagnostics. The
build is incremental. A hydrodynamics-only run **cannot** populate $\sigma_{ij}$ or $\Delta CFS$.

Without an explicit mechanism, **an absent field is ambiguous**: the module may be *not yet
implemented*, or it may have *failed*. Those are opposite states and collapsing them would violate
**INV-1**.

### 6.1 Required: every run declares its capabilities

**Rust declaration, adopted verbatim from the 08-10-2026 ruling:**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleCapabilityState {
    Implemented,    // Executed, valid physics -> Populated
    NotImplemented, // Module deferred -> Absent, explicitly declared
    Degraded,       // Reduced fidelity/fallback -> Populated + fidelity log
    Failed,         // Solver/module crashed -> Run outcome = FAILED
}
```

| State | Meaning | Run outcome |
|---|---|---|
| `Implemented` | Module exists, ran, produced valid output | Field **populated** |
| `NotImplemented` | Module does not exist yet | Field **absent**, declared |
| `Degraded` | Module ran but under reduced fidelity | Field **populated**, `fidelity` recorded |
| `Failed` | Module exists and failed | 🔴 **Error. Run fails.** Per **INV-1** |

**A run may only terminate successfully if every module it claimed is `Implemented` or `Degraded`.** A
module in `Failed` state fails the run. There is no partial success.

> ✅ **The ruling's stated reason is the right one and is preserved:**
> *"If an unassigned field is simply zeroed out, an ML model or researcher cannot distinguish between a
> zero physical value (e.g. zero plastic strain) and an unimplemented module."*

### 6.2 Required: the capability declaration is part of the artifact

| Field | Location |
|---|---|
| `capabilities` — per-domain status | Run manifest, and the output file header |
| `schema_version` | Output file header |
| `engine_version`, `build_hash`, `spec_hash` | Manifest (already in [`output_schema.md`](output_schema.md) §2, [`data_architecture.md`](data_architecture.md) §4.5) |
| `populated_domains` | Derived; a consumer must be able to tell which of the 9 domains hold data |

> **Why this is mandatory, not convenient.** **INV-4** makes this output the training set for a neural
> surrogate. If "absent" is ambiguous, a consumer cannot tell whether a run is a **training sample** or a
> **partial failure**. Training on a partial run teaches the surrogate that unpopulated fields are
> arbitrary — which is precisely how a bad dataset becomes a bad model. **The capability declaration is
> what makes the dataset trustworthy.**

### 6.3 Interaction with the three temporal resolutions

| Tier | Populated only when |
|---|---|
| Micro-step | The step was a Newton iteration of a **converging** solve. **Failed/cut steps must not enter the training set** |
| Daily | The timestep completed |
| Monthly/yearly | ⚠️ **No economics** (**INV-3**, **CONF-64**). DCF belongs to the economic engine |

---

## 7. Prior art: the Python compositional engine was built and removed

> *"Factory was builded but because we dont have luck with python implementation of compositional engine
> we had to remove it"*

**Recorded history:** an `EngineFactory` **was implemented**, along with a Python compositional engine.
The attempt failed and **the engine and factory were deleted**.

### 6.1 Why this is in the wiki

| Reason | Detail |
|---|---|
| **`extension_points.md` documents a deleted component** | `agent_wiki/development/extension_points.md:98-117` gives step-by-step instructions for registering an engine via `EngineFactory`, including *"Modify `core/engine_factory.py` lines 105–116"*. That file **does not exist** — it was built and removed. This is worse than a path that never existed: the wiki describes a **failed approach** as if it were the correct one |
| **Rust is the second attempt at the same goal** | The reason for the language choice is recorded history, not preference |
| **The failure mode is worth not repeating** | A Python compositional engine is a large numerical workload with no advantage here: no JIT-equivalent speed story, no zero-cost abstractions, and GIL-bound parallel assembly |

### 6.2 Lessons to carry into the Rust attempt

| Lesson | Application |
|---|---|
| Don't build the abstraction before the thing | The `EngineFactory` existed with nothing valid to register. **P2 must not re-create it speculatively** — build it when a second working engine exists |
| A failed first attempt is not a reason to skip the gate | The 6-level V&V framework exists partly to prevent a silent partial success. **INV-1 (fail loudly) is the specific countermeasure** to a first attempt that appeared to work |
| Delete the failure, but record why | The Python engine is gone. Its post-mortem is not. **This section is the surviving record** |
| ⚠️ A deleted engine may have left traces | Legacy imports, config keys and doc claims may survive. `agent_wiki/architecture/source_of_truth_map.md` already records that retired engine paths were deleted rather than relocated |

> **Open question:** was there a written post-mortem for the Python attempt, and were the findings folded
> into `audit/scientific_flaws.md`? If the Python engine failed in identifiable ways, those failures are
> **requirements for the Rust engine**, not history to discard. Worth recovering.

---

## 7b. 🔴 INV-7 — Unconstrained by construction (added 08-10-2026)

> Owner requirement, verified against the design set 08-10-2026:
> *"a compositional engine implementation on rust that have unconstrained nature so it can be used for
> research using even obviously wrong input (for example rate is astonishingly high for an injection
> well) but engine should give full physical evaluation even if it means the project is failed and
> unrealistic to use as a development strategy."*

**Required:** the engine applies **no artificial constraint** to any physical quantity. It evaluates the
input as given and reports what the physics produces — including results that condemn the project.

### 7b.1 Two distinct failure modes — INV-1 and INV-7 address different things

This is the crux, and conflating them is the failure:

| | **INV-1** | **INV-7** |
|---|---|---|
| Trigger | The **numerics** cannot produce an answer — non-convergence, flash failure, conservation violation, IO error | The **input is absurd**, but the numerics can still solve it |
| Required | Typed error, **stop**, no partial artifact, no fallback | **Complete the run.** Report the physical consequence |
| Example | Flash does not converge inside a valid region | Injection at 50 000 BOPD |
| Forbidden | Renormalise, loosen tolerance, substitute a guess, revert to surrogate | Clamp the rate, truncate, smooth, or apply a penalty |

> 🔴 **They do not conflict — they are ordered.** INV-7 does not license non-convergence to be papered
> over. If 50 000 BOPD genuinely makes the system unsolvable, **INV-1 governs**: typed error, stop,
> no artifact. The difference is only in the **diagnostic**: the error must say *"cannot converge at
> rate 50 000 BOPD, pressure exceeds model validity"*, never *"clamped to 1 000 BOPD"*.

### 7b.2 Three-tier result classification — required in the run manifest

An unconstrained engine must still let a researcher distinguish *valid* from *absurd*. Required
per-run field `ValidityClass`:

| Tier | Meaning | Engine behaviour | Run status |
|---|---|---|---|
| **Validated** | Inside the V&V-verified envelope ([`vv_testing_framework.md`](../thmc/vv_testing_framework.md)) | Complete normally | `OK` |
| **Converged, outside envelope** | Solved cleanly, but outside the verified range — e.g. rate far above any benchmark | **Complete normally.** Emit a `ValidityWarning` naming the violated envelope | `OK` + warning |
| **Unsolvable** | Numerics cannot produce an answer | Typed error, **stop**, **no artifact** (**INV-1**) | `FAILED` |

> ✅ **This is what makes INV-7 safe for research.** Tier 2 gives the researcher the absurd answer they
> asked for, *plus* the knowledge that it is unverified. Tier 3 is not a soft answer — it is the refusal
> to fabricate one. A run that cannot converge must produce **nothing**, not a plausible-looking file.
>
> ⚠️ **Tier 2 must never be collapsed into Tier 1.** A warning that is routinely ignored is not a warning.

### 7b.3 Artificial constraints that must NOT survive into Rust — measured

Each of these is a **round number** imposed by the design set, not a consequence of physics. They are
the direct targets of INV-7.

| # | Artifact | Source | Status under INV-7 |
|---|---|---|---|
| 1 | **`q_well ≤ 1 000 BOPD` hard clamp** | D1 §9.1, D6 §3.6, D7 §4.3, D8 §3.3 | 🔴 **SUPERSEDED — remove.** See measurement below |
| 2 | `N_pat = Area / 40 acres`, `N_inj = N_prod = N_pat` | D1 §9.1 | ⚠️ Pattern sizing is a **scenario input**, not a constraint. Never imposed silently |
| 3 | `min_injection_rate_bpd: 1000.0` | `config/base_config.json` | 🔴 **Must not exist** in the Rust engine |
| 4 | 6 × `locked_*` fudge parameters (`locked_sor`, `locked_productivity_index`, `locked_gravity_factor`, `locked_hyperbolic_b_factor`, `locked_transition_alpha/beta`) | `config/base_config.json` | 🔴 **Fudge factors.** These are CRIT-19's fudge factors and must have no Rust equivalent |
| 5 | RF cap `RF_ult·(1 − exp(−HCPVI/1.5))` | S1 line 77 | ✅ Already **removed** (**CONF-02**) |
| 6 | `−1.0e12` failure penalty, `breakthrough_impact` multiplier | live Python engine | ✅ Already **removed** (**CONF-37**) |
| 7 | `np.clip` of sandface P to `p_safe_ceiling` | `surrogate_engine.py:468` | 🔴 **Must be modelled, not clipped.** See §7b.4 |
| 8 | CO₂ GOR `5 000–25 000 SCF/STB` | S1 | ⚠️ A **benchmark observation**, not a design constraint. Never an input bound |

### 7b.4 The 1 000 BOPD clamp is not a physical limit — measured 08-10-2026

**Steady-state vertical-well Darcy throughput at the design set's own pattern size** (D5: 40 acre →
$r_e = 227$ m, $\ln(r_e/r_w) = 7.65$; $\Delta P = 50$ MPa, $\mu = 1$ cP):

| $k$ | $h$ | $J$ (Darcy limit) | vs. the 1 000 BOPD clamp |
|---|---|---|---|
| 50 mD | 10 m | 11 011 STB/d | **11×** above |
| 100 mD | 20 m | 44 042 STB/d | **44×** above |
| 200 mD | 30 m | 132 127 STB/d | **132×** above |
| 1000 mD | 50 m | 1 101 055 STB/d | **1101×** above |

> ✅ **The clamp binds 11× to 1100× below the Darcy limit of the stated pattern.** It is a round-number
> cap, not a reservoir constraint. **CONF-54** already recorded D5 naming the `20 000–60 000 MSCFD` range
> as *the anti-pattern signature of a clamped rate* — the design set diagnoses its own defect and then
> ships a clamp anyway.
>
> **CONF-07 is therefore resolved in the direction the register only flagged as unstated:** the clamp has
> **no position in the chain at all under INV-7, because it must not exist.** This supersedes correction
> **C-22**, which had asked only where the clamp sits.

### 7b.5 Pressure limits are different — and this distinction matters

INV-7 forbids **artificial** constraints. It does **not** forbid **physical** ones.

| Limit | Class | Unconstrained behaviour |
|---|---|---|
| **1 000 BOPD** | 🔴 **Artificial** — round number, 11–1100× below Darcy | **Remove.** Rate is a scenario input; the consequence is the engine's job to compute |
| **Fracture / caprock integrity** | ✅ **Physical** — the rock actually fails | **Model it.** Fracture initiation, fault slip (`DCFS > 0`), containment loss — all emitted as physical events |
| **Conservation of mass** | ✅ **Physical** | Enforce. A violated balance is **INV-1 `FAILED`**, not a warning |

> 🔴 **The `np.clip` at `surrogate_engine.py:468` is the exact pattern INV-7 forbids.** Clipping sandface
> pressure to `0.90 × P_frac` makes the geomechanical limit *unobservable*: the run continues, the
> pressure history is smooth, and the failure that would have been reported never happens. (This is
> recorded for the Python audit — `HIGH-23`. **The Rust engine must emit the failure event instead.**)
>
> ⚠️ **Where does this leave "safe"?** A real project has permit and containment obligations. **That is a
> satellite concern, not an engine concern** — consistent with **INV-3**. The engine reports that
> containment failed; the satellite reports that the project is not permittable. **Neither masks the
> other.**

### 7b.6 Enforcement

| Mechanism | Requirement |
|---|---|
| **Input path** | No clamp, `min_*`, `locked_*` or default-bounded field on any physical quantity. Types and units validated; **magnitudes are not** |
| **Solution path** | No `clip`, `clamp`, `saturate`, `smooth` or `penalty` on a state variable, rate, or pressure |
| **Manifest** | Every run records `ValidityClass` (Tier 1/2/3), the input as **given**, and the envelope warning if Tier 2 |
| **CI gate** | A test that feeds a deliberately absurd input (e.g. 50 000 BOPD injection) and asserts the engine **completes** and the rate appears **unclamped** in the output |
| **CI gate** | A grep-based gate rejecting `locked_*`, `min_injection_rate`, and `clip(`/`clamp(` on physical fields |
| **Audit** | Any clamp reappearing is a **`COMP-nn` finding**, not a style comment (**INV-5**) |

---

## 8. Invariant summary

| ID | Invariant | Scope | Decided |
|---|---|---|---|
| **INV-1** | **Fail loudly. Never fall back.** Every error returns a typed error and stops | **Permanent, mandatory** | 07-10 |
| **INV-2** | **Runs only when the user starts it.** No automatic invocation, no dispatch heuristic | **Permanent, mandatory** | 07-10 |
| **INV-3** | **Simulation only.** No economics of any kind | **Permanent** | 07-10 |
| **INV-4** | **Output is training-ready.** Labelled $(x,y)$ pairs, full fields, versioned, provenance per sample | **Permanent** | 07-10 |
| **INV-5** | **Own flaw register**, `COMP-nn` namespace | Recommended permanent | 07-10 |
| **INV-6** | **Capability declaration vs. failure.** A run declares which domains it populated. `NOT_IMPLEMENTED` ≠ `FAILED` | **Permanent, mandatory** | 08-10 |
| **INV-7** | **Unconstrained by construction.** No clamp, min, lock, clip or penalty on physical quantities. Absurd input → full physical evaluation, even when the project fails. Three-tier `ValidityClass` | **Permanent, mandatory** | 08-10 |

---

## 9. Decisions resolved 08-10-2026

| # | Owner answer | Effect |
|---|---|---|
| 1 | **"We leave Python as is so all findings and audits will be there as a python audit"** | ✅ Confirms **INV-5**. No changes to `core/`, no changes to `audit/scientific_flaws.md`. The 73 legacy findings stay as the Python audit |
| 2 | **"complement"** — the compositional engine **complements** the surrogate | P3 coupling definitely exists. The surrogate is **not** retired |
| 3 | **`pgvector` for now** | Layered with the S4 file stack — see [`output_schema.md`](output_schema.md) §5.1 |
| 4 | Output schema "explains" the economic-engine question | ✅ **INV-3 confirmed and now precise**: the engine emits physical quantities only; the economic engine consumes them. See **CONF-64** |
| 5 | **Full output enables reinforcement / simulation learning to tune the surrogate on full physics** | ⚠️ **The economic engine is on the critical path for the RL loop** — the reward is NPV. Consistent with **INV-3**: economics is out of the *simulation* engine, and in the *learning loop* by necessity |
| 6 | `3D_THMC_docs` = **reference**; the **wiki** carries the corrected spec; all approved changes and branches recorded | [`spec_corrections_log.md`](spec_corrections_log.md) created and seeded with 19 corrections + 7 open branches |
| 7 | **"It is deliberate by user to start simlearning and user knows it is long process"** | ✅ Accepted as a known cost, not a risk to mitigate |
| 8 | ⚠️ **CONF-64…68 attribution to be confirmed** — the owner notes these "refer to our surrogate engine". **Recorded as disputed**: all four fixes are written against **S1's domains** (the compositional output schema), and none would apply to the surrogate, which has no $\epsilon_p$, no aqueous speciation vector and no additive skin decomposition. **No ruling is blocked**; the caveat is logged in [`spec_corrections_log.md`](spec_corrections_log.md) |
| 9 | ✅ **`sr3_reader.py` will be updated and expanded, with a proper Rust rewrite** | — |
| 10 | ✅ **Long load time for complex maps and fields is acceptable.** Geometry loads fine | Removes a performance constraint from the P2 UI questions |
| 11 | 🔵 **`pyarrow`/`duckdb` are leftovers, not gaps.** Most satellites are developed *after* the compositional engine, so the dependencies are added then | **No longer a P1 blocker.** Recorded, not removed |
| 12 | 🔴 **The wiki states CURRENT state.** Forward-looking specifications are **targets**, not present reality | Every spec page now carries an explicit target-vs-current banner |
| 13 | ✅ **All micro-steps retained** ("we can update it later") | Closes branch **B-6** |
| 14 | ✅ **Economic engine reads DuckDB / Parquet, never HDF5** | Closes branch **B-4** |
| 15 | ✅ **Python post-mortem supplied** — six numerical-method root causes, **all language-independent**. ⚠️ The failure was **not** execution speed | Preserved at [`python_attempt_postmortem.md`](python_attempt_postmortem.md). Yields **C-33…C-37** |
| 16 | ✅ **Unconstrained nature confirmed** — absurd input must still receive full physical evaluation, even if the project is thereby failed | 🔴 **INV-7.** Kills **CONF-07** (the 1 000 BOPD clamp — measured 11×–1100× below the Darcy limit), `min_injection_rate_bpd`, all 6 `locked_*` fudge parameters, and any `clip` on pressure. Requires three-tier `ValidityClass` |
---

