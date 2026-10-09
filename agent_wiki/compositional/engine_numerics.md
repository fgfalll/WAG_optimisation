# Numerics and Status Architecture

**Split out of [`engine_invariants.md`](engine_invariants.md) on 09-10-2026.** **Section numbers are
preserved exactly** — every cross-reference in the wiki citing sections 7o through 7w still resolves to the
same content, just in this file.

**What belongs here:** the architecture rulings, the private-newtype and status-lattice design, the
determinism triad, the three-tier tolerance scheme, the rate-form residual, precision, and timestep
cutbacks.

| Section | Subject |
|---|---|
| 7o | Architecture ruled — seven items |
| 7p | Private newtypes, cascade lattice, checkpoint file, determinism triad |
| 7q | 2x2 status matrix, atomic checkpoint, two-tier reduction |
| 7r | CONF-23 three-tier tolerances, CONF-55 remediation plan |
| 7s | Tier A normalization, component closure, typed `Unsolvable` |
| 7t | Rate-form residual and the dual-branch Tier A gate |
| 7u | Precision additions locked; one INV-1 boundary |
| 7v | Cutbacks, and the third break of the status partition |
| 7w | Dimensional repairs and two-phase hardening |

**The standing rules these sections establish — Symbol Register (7k), 7s.6, the limit-case suite, the
dimensional gate — apply to _every_ milestone, not just this file.**

---

## 7o. Architecture ruled — seven items (08-10-2026)

Corrections **C-124 … C-130**. **Five accepted outright; five defects found inside the rulings.**

### 7o.1 ✅ C-124 — compile-time newtypes, with two corrections

**Accepted**, and the reasoning is right: `uom` checks at **runtime**, and the Newton hot loop cannot
afford it against the Zero-Bloat Core.

🔴 **The code sketch defeats its own purpose:**

```rust
#[repr(transparent)] struct Pascals(pub f64);   // <-- pub defeats the point
```

`.0` returns a **bare `f64`**, freely mixable with any other `f64`. This keeps the **zero-cost** property
and **loses the type-safety** property. ✅ **The field must be private**, with a getter or `Deref` — or the
dimensional checking is theatre.

⚠️ **Dimensional analysis is not a compile-time check** with plain newtypes. It runs in `#[cfg(test)]` at
**test time**. ✅ **Rule it as a TEST gate, not a build gate** — otherwise **C-115**'s typed bounds, and
**C-88**'s Penéloux sign error, will be expected to fail a *compile* and will not.

### 7o.2 ✅ C-125 — M2 formulation resolved; C-85 closed

**Overall composition $(P, S_\alpha, z_i)$** adopted. ✅ **This closes C-85**, which was mine and had been
the highest-leverage open decision in the project.

| Claim | Verdict |
|---|---|
| State vector $1 + (N_p-1) + (N_c-1)$ per cell | ✅ **correct** — two constraints, $\sum S_\alpha = 1$ and $\sum z_i = 1$ |
| $z_i$ is $C^1$-continuous across phase boundaries | ✅ the standard argument, and sound |
| Removes variable switching and its $C^0$ residual breaks | ✅ the main win — oscillation near the critical point |
| `feos-ad` for phase derivatives | ✅ **VERIFIED REAL** |

✅ **`feos-ad` verified on crates.io:** v0.2.3 (2025-05-28), **4 346 downloads**, MIT OR Apache-2.0,
keywords `autodiff` / `equations_of_state` / `phase_equilibria`, same author and org as `feos`
(`feos-org`).

> ⚠️ **I assumed it was fabricated and was wrong.** Six prior rulings of unverifiable identifiers had
> taught me the pattern, and I generalised it. **The prior was right; the generalisation was not.** Had I
> not checked I would have filed a false `PROVENANCE` finding against a real crate.

✅ **Pinning rule:**

```toml
feos = { version = "0.10", default-features = false }   # explicit allowlist
```

⚠️ `feos` 0.8.0 shipped a **`python` feature (pyo3)** that links libpython. 0.10.1's feature list no
longer shows it — but an explicit allowlist is what **guarantees** the P1 separation doctrine, rather than
trusting a version bump not to reintroduce it.

⚠️ **$C^1$ on the primary variable $\ne$ $C^1$ on the residual.** The flash map still changes character
where the active phase set changes. **Variable switching is gone — the main prize — but near-boundary
convergence still needs the C-33 / C-72 treatment.** The ruling should not be read as "convergence is now
smooth everywhere".

### 7o.3 ✅ C-126 — DTO bus deleted; closes CONF-27 and CONF-28

**Accepted**, and this is the best architectural move in the round. The approved **two-output contract**
(HDF5 + Parquet, open formats) *already replaces* a DTO bus. Deleting it removes the IPC-vs-Zero-Bloat
contradiction outright rather than resolving it.

🔴 **But restart state cannot live inside a run artifact whose failure invalidates it.**

> You restart **failed** runs. So if `/Snapshots/t_n` sits in the run HDF5 and **INV-1 / C-51** quarantine
> failed runs, **the rule that protects the artifact destroys the restart point.**
>
> ✅ **Checkpoint must be a separate, independently-committed file**, with its own atomic write and its
> own lifecycle — deliberately outside the run artifact's commit boundary.

⚠️ The restart vector $[P,T,S_\alpha,z_i,\sigma_{ij},\mathbf u]^T$ is only valid when geomechanics is
active. ✅ It must be **driven by the INV-6 capability declaration**, not fixed — an M3 hydrodynamics
restart has no $\sigma_{ij}$ or $\mathbf u$.

### 7o.4 ✅ C-127 — `petekIO` in-memory only; closes CONF-30b by deletion

**Accepted.** Parser boundary becomes **GRDECL / RESQML / RESCUE** only.

🔴 Elegant consequence: the seven unstated items — magic bytes, header struct, byte order, slab typing,
chunk table, alignment contract, format version — **stop being requirements**, because there is no file
to specify. **Second deletion ruling; second time one has closed a conflict by removing the requirement
rather than documenting it.**

### 7o.5 🔴 C-128 — the status lattice is not a partition

**Counter-case: all active modules = `Degraded`, all inputs inside the V&V envelope.**

| Tier | Requires | Result |
|---|---|---|
| **3 `Unsolvable`** | some module `Failed`, **or** an INV-1 typed error | 🔴 fails |
| **1 `Validated`** | **all** active = `Implemented` **and** in-envelope | 🔴 fails — `Degraded` $\ne$ `Implemented` |
| **2 `ConvergedOutsideEnvelope`** | all active $\in\{$`Implemented`,`Degraded`$\}$ **and** (outside-envelope **or** $\exists$`NotImplemented`) | 🔴 fails — **neither** disjunct holds |

⇒ **No tier matches. The run is unclassified.**

✅ **Ruled: a strict priority cascade.** Tier 3 first; then Tier 1; then **Tier 2 = everything else that
completed.** The complement definition cannot leave holes.

🔴 **Second defect: Tier 2 conflates `ModuleAbsent` with envelope violation.** A declared-absent Domain 5
is **normal** for an M3 hydrodynamics run (**INV-6**) and says nothing about whether the *input* was
verified. Tagging it "outside the validated envelope" is a category error.

✅ **Ruled:** Tier 2 = **"completed with $\ge 1$ warning"**, and the warning list must distinguish:

```rust
enum ValidityWarning {
    EnvelopeExceeded { .. },                 // the INPUT was unverified
    ModuleAbsent { domain: DomainId },        // INV-6, not an input problem
    CorrelationProvenanceUnverified { .. },   // C-116
    ExtrapolatedPhasing { .. },               // C-111
    AnisotropicPerforationRawCorrelation { .. }, // C-109
}
```

### 7o.6 ✅ C-129 — determinism; closes CONF-43, but seed alone is not enough

**Accepted** — the cheapest close on the register. 🔴 **But "100 % bit reproducibility" does not follow
from a seed.** Three further requirements:

| # | Requirement | Why |
|---|---|---|
| **(a)** | **Fixed PRNG algorithm *and version*** | PCG64, ChaCha and xoshiro yield **different streams from the same seed** |
| **(b)** | **Deterministic reduction order** | Rayon's parallel `f64` reduction reorders summation, so output depends on **thread count** |
| **(c)** | `f64` behaviour pinned | Rust forbids fast-math, which helps; vector reductions still reorder |

✅ **Ruled: per-cell independent streams keyed by `hash(seed, cell_index)`.** This makes a geological
realisation independent of **thread count and traversal order** — the only way the **SPE5 cross-simulator
comparison (C-119)** can be bit-stable, and the only way **INV-4**'s samples stay re-derivable.

### 7o.7 ✅ C-130 — single crate M0–M6, workspace at M7

**Accepted**, consistent with [`separation_doctrine.md`](separation_doctrine.md).

⚠️ The **`deny(unsafe_code)` vs `russell_sparse` FFI** conflict is already recorded as breaking the lint.
It needs a scoped `#[allow]` on the FFI module or a different solver — and it is an **M0** decision
**independent of crate layout**. Keeping it in one crate does not make the FFI easier; it only makes the
lint harder to scope.

### 7o.8 📌 The transferable lesson — two deletion rulings, four conflicts

| Ruling | Method | Closed |
|---|---|---|
| **C-126** | DTO bus redundant against an approved file contract | CONF-27 + CONF-28 |
| **C-127** | `petekIO` never needed to be a file | CONF-30b |

Both worked identically: the requirement looked unimplementable because it was **underspecified**, and the
fix was to notice the requirement **did not need to exist**.

> 📌 **Specification work that ends in a deletion is cheaper than specification work that ends in a
> document** — and this register has spent many rulings producing documents. When a spec item resists
> specification, the first question should be *"what requirement would disappear if this were deleted?"*

---

## 7p. Private newtypes, cascade lattice, `.ckpt.h5`, determinism triad (08-10-2026)

Corrections **C-131 … C-135**.

### 7p.1 ✅ C-131 — private-field newtype accepted

```rust
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pascals(f64);

impl Pascals {
    #[inline(always)] pub const fn new(val: f64) -> Self { Self(val) }
    #[inline(always)] pub const fn as_f64(&self) -> f64 { self.0 }
}
```

✅ Cross-dimension arithmetic is **blocked at compile time**; conversion needs an explicit `From`/`Into`.
✅ `as_f64()` as the single named escape hatch is the right shape — **explicit and greppable**, so
boundary conversions can be audited with `rg`.

| Refinement | Detail |
|---|---|
| ⚠️ **(a)** `as_f64()` **is** an escape hatch | The gate must **count and bound its uses inside constitutive relations**, and it must not appear in the solver hot path |
| ⚠️ **(b)** `PartialOrd` is per-type, not blanket | Physically meaningful only for **scalar-ordered** quantities. Deriving it on a tensor-valued or direction-ambiguous type invites meaningless comparisons |

### 7p.2 ✅🔴 C-132 — cascade closes the reported hole, misses an adjacent state

**Hole CLOSED.** Verified: all-`Degraded` + in-envelope now fires **Tier 2** (the added
$\lor\ \exists$`Degraded` disjunct); all-`Implemented` + in-envelope fires **Tier 1**.

✅ **INV-6 decoupling accepted and correct** — `ModuleAbsent` does not participate in active-module
evaluation and does **not** downgrade `Validated` for domains that *are* active.

🔴 **A fourth state is still missing: incomplete-but-not-failed.**

| Case | Result |
|---|---|
| User stops the run (**INV-2** — *"runs only when the user starts it"*) → no `Failed`, no `TypedError`, **did not complete** | |
| Tier 3 requires a failure | 🔴 no |
| Tier 2 requires *"run completed"* | 🔴 no |
| Tier 1 requires *"run completed"* | 🔴 no |
| | ⇒ **unclassified, exactly as before** |

✅ **Ruled: add `Aborted`.** Not `Validated`, not `Unsolvable`, and per **INV-1** a partial artifact is
**never** committed to the training store.

✅ **Gate: prove exhaustiveness over the full cross-product**
$\{\text{complete},\ \text{incomplete}\}\times\{\text{failure},\ \text{no-failure}\}$ — 4 cells, 4 classes.

> 📌 **C-132 shows this was not an arithmetic slip but a structural gap.** The fix closed the case I
> found and left an adjacent one — the signature of a **missing exhaustive-case argument**, not a typo.

### 7p.3 ⚠️ C-133 — `.ckpt.h5` decoupling accepted; the checkpoint needs its own durability rule

**Accepted**, and the INV-6-driven state vector is exactly right:

| Milestone | $\mathbf{Y}_{\text{ckpt}}$ |
|---|---|
| M2 / M3 | $[P,\ T,\ S_\alpha,\ z_i,\ \text{WellStates}]^T$ |
| M7+ | $[P,\ T,\ S_\alpha,\ z_i,\ \sigma_{ij},\ \mathbf u,\ w_f,\ C_k,\ \text{WellStates}]^T$ |

🔴 **But the checkpoint's own durability has no rule** — and that is the entire purpose of the file. A
torn write (crash mid-`flush`, full disk, killed process) leaves a `.ckpt.h5` that **loads without error
and yields a corrupted restart** — **strictly worse than no restart**, because it fails silently.

✅ **Ruled:**

| Rule | Detail |
|---|---|
| **Atomic publish** | write `run_name.ckpt.h5.tmp` → **atomic rename** |
| **Commit token last** | a `commit_token` dataset holding run identity + step index, written **last** |
| **Loadable iff** | `commit_token` matches the expected run/step identity |
| **Partial files** | unlinkable **by construction**, never by inspection |

### 7p.4 ✅ C-134 — determinism triad accepted, and the float claim now measured

Per-cell streams keyed $\text{Blake3}(\text{MasterSeed}\,\Vert\,\text{CellIndex}\,\Vert\,\text{DomainID})$
with **ChaCha8** (`rand_chacha`, version pinned in `Cargo.lock`) ✅ accepted.

**Measured on $N = 10^6$ terms:**

| Summation | Result | Relative error |
|---|---|---|
| naive, original order | $500161.97345979\mathbf{54}$ | $1.29\times10^{-14}$ |
| naive, **shuffled** order | $500161.97345980\mathbf{775}$ | $1.29\times10^{-14}$ |
| **Kahan**, original order | $500161.97345980\mathbf{187}$ | **0.0** |
| **Kahan**, shuffled order | $500161.97345980\mathbf{187}$ | **0.0** |

✅ The engineer's per-operation scale is consistent — $1.29\times10^{-14} matches a $\sqrt{N}\,\varepsilon$
random walk.

🔴 **Correction to my own expectation.** I predicted Kahan would buy accuracy but **not** reproducibility.
**Measured, the opposite:** naive summation is **order-dependent** (different bit pattern, $\lvert\Delta\rvert
= 1.23\times10^{-8}$, same accuracy), while **Kahan was bit-identical across both orders** and matched
`math.fsum` exactly. ✅ **Kahan is not redundant** — it delivers accuracy **and** order-independence.

⚠️ **But the guarantee lives in the ORDER clause, not the Kahan clause.** Kahan is not *provably*
order-independent for all inputs. ✅ **The fixed strict-cell-index tree merge is what turns reproducibility
from a *likelihood* into a guarantee. Keep both; document the order clause as load-bearing.**

### 7p.5 🔴 C-135 — determinism ≠ comparability

✅ Per-cell ChaCha8 streams give **perfect reproducibility within Rust**.
🔴 **But they cannot reproduce a CMG or SPE reference realisation** — those used different PRNGs entirely.

For **C-119**'s SPE5 work there are **two distinct goals**, and the protocol conflates them:

| Goal | Meaning | Achievable by PRNG choice? |
|---|---|---|
| **(a) Reproducibility** | same input ⇒ same output, every run | ✅ **delivered** |
| **(b) Comparability** | our realisation **matches the reference** | ❌ **not achievable** |

✅ **Ruled: SPE5 comparison must use the SPE-supplied realisation, not a regenerated one**, and the V&V
framework must name the two goals separately — otherwise a legitimate PRNG change will be misdiagnosed as
solver error.

⚠️ **Tolerance-based stopping makes the iteration count data-dependent**, so reproducibility requires the
reduction order to be **strictly index-derived and never history-derived** — *including inside the
solver*, where the adaptive AMG hierarchy of **C-95** is **history-dependent by construction**.

---

## 7q. 2×2 status matrix, atomic checkpoint, two-tier reduction, goal split (08-10-2026)

Corrections **C-136 … C-139**.

### 7q.1 ✅ C-136 — the 2×2 matrix is a partition. C-132 closed.

| | **No-Failure** | **Failure / TypedError** |
|---|---|---|
| **Complete** | **Tier 1 `Validated`** (all `Implemented` + in-envelope)<br>**or Tier 2** (`Degraded` **or** out-of-envelope) | **Tier 3 `Unsolvable`** |
| **Incomplete** | **Tier 4 `Aborted`** *(new)* — user stop, timeout | **Tier 3 `Unsolvable`** |

✅ **Verified exhaustive, no gap and no overlap.** The Complete/No-Failure cell resolves because either
$\forall$`Implemented` $\wedge$ in-envelope (Tier 1), **or** its negation — which is exactly
$\exists$`Degraded` $\vee$ out-of-envelope (Tier 2).

**Artifact rule ✅ correct and consistent with C-133:** on `Aborted`, `run_name.out.h5` is **never
committed** — no partial-data leakage — while the autonomous `run_name.ckpt.h5` **is** retained for
recovery.

> 📌 **First closure in fourteen rulings where the fix was verified against the *whole* partition rather
> than the reported case — and it held.** That is the dividend of C-132's gate requirement: *prove
> exhaustiveness, do not spot-check.*

### 7q.2 ⚠️ C-137 — checkpoint protocol accepted; two gaps

Protocol accepted: `.tmp` write → BLAKE3 `CommitToken` last → `fsync()` → atomic POSIX `rename()` →
verify-on-load. Sound.

🔴 **Gap 1 — no directory `fsync()` after the rename.** POSIX durability requires fsyncing the **parent
directory** after `rename()`. `fsync()` on the temp file makes the **contents** durable but not the
**directory entry** — so a crash immediately after the rename can still lose the rename itself. This is
the classic ext4/XFS rule.

✅ **Ruled:** `fsync(tmp)` → `rename()` → **`fsync(dirfd)`**.

🔴 **Gap 2 — `unlink()` on token mismatch destroys the evidence.** Silent deletion makes post-mortem
impossible, which contradicts this project's own audit discipline (**INV-5**, `audit_comp`).
✅ **Ruled: quarantine to `run_name.ckpt.h5.corrupt`, never `unlink`.**

⚠️ **Which clause is load-bearing?** *"Token written last"* and *"rename only after a complete write"* are
**redundant as atomicity guarantees** — `rename` provides atomicity; the token detects later bit-rot.
✅ **Ruled: state it explicitly**, so a future simplification cannot cut the wrong clause.

> 📌 **C-137 is the same shape as C-136 one level down.** The protocol was verified against the
> *reported* failure — torn write — and was correct. But **crash-during-rename** is a different *instant*,
> and it is uncovered. 📌 **A protocol verified against one hazard has been verified against one hazard.**
> The durable check is to enumerate the failure **instants**, not the failure **modes**.

### 7q.3 ✅ C-138 — two-tier reduction designation accepted. C-134 closed.

Accepted verbatim, and it is the right documentation:

> *"The reduction order by static cell indices is the load-bearing element of determinism; Kahan
> summation is the load-bearing element of numerical accuracy."*

✅ Fixed index order = **determinism guarantee**; Kahan = **accuracy defence**. Both retained, roles
distinguished — consistent with the measurement (**naive order-dependent**; **Kahan bit-identical**;
`fsum` error **0.0**).

### 7q.4 ✅ C-139 — goal split accepted; official SPE files now an action item

**The AMG clause is the most valuable part.** Coarse-grid construction must use a **static adjacency
graph with deterministic index-based tie-breaking**, excluding **Rayon traversal order and memory
addresses**. ✅ That is the classic hidden nondeterminism in adaptive AMG, and naming memory addresses
as an exclusion is exactly right.

✅ FGMRES inner products and $\lVert R\rVert_2$ routed through the fixed tree — so the **tolerance-based
stopping criterion flagged in C-134 is now deterministic.**

🔴 **New ACTION ITEM (L-4):** SPE5/SPE10 tests must import the **official** grid and permeability files
(`SPE5_PERM.GRDECL` / `.HDF5`), not regenerate them.

> ⚠️ **The CMG deck found in `D:\RAG` (C-119) is NOT the official specification** — it is a CMG-format
> transcription, explicitly a *starting point*. **The official files are a separate acquisition.** Added to
> [`literature_todo.md`](literature_todo.md).

---

## 7r. CONF-23 three-tier tolerances, CONF-55 remediation plan (08-10-2026)

Corrections **C-140 … C-142**. **Both blocking reds closed.**

### 7r.1 ✅ C-140 — CONF-23 closed: three tiers are three different quantities

| Tier | Metric | Scope | Spec |
|---|---|---|---|
| **A — in-core FVM solver** | absolute per-component $\|\mathbf{R}_{m,i}\|_\infty < 10^{-12}$ | per component, **every cell, every Newton iteration** | D5 §3.1 |
| **B — remap operator** | $\|\sum M_{FVM} - \sum M_{SL}\| < 10^{-14}$ | streamline ↔ FVM remapping precision | D5 §5.3, renamed `RemapConservationPrecision` |
| **C — satellite / surrogate** | relative global $\text{MB}_{\text{closure}} \ge 99.9\,\%$ | full history $T_{\max}$ | D2 §6.2, D6 §4.3 |

✅ Tier A and Tier C **hold simultaneously** and are not in conflict: high-precision local Newton steps
accumulate to good global closure. ✅ Renaming $10^{-14}$ is right — it is **operator precision**, not a
convergence tolerance.

⚠️ **Refinement (a) — the index $i$ is overloaded** in the closure expression:

$$\text{MB}_{\text{closure}} = 1 - \frac{\left|\sum_i M_{i,\text{remaining}} + W_p + G_p + N_p - M_{i,\text{initial}} - W_i - G_i\right|}{M_{i,\text{initial}} + W_i + G_i}$$

$M_{i,\text{remaining}}$ and $M_{i,\text{initial}}$ index **component**; $W_i$, $G_i$ index **well**; and
$W_p, G_p, N_p$ carry **no component index at all**. The balance is component-total vs component-total
**only if** the production and injection terms are component-resolved. ✅ **Use $c$ for component, $w$ for
well, and state the summation.** 📌 This is the register's recurring **notation-collision** class.

🔴 **Refinement (b) — an absolute $10^{-12}$ kg/s tolerance is not $\Delta t$-invariant.** Measured, for a
fixed net flux $q$:

| $\Delta t$ | $R = q\,\Delta t$ | effective mass-per-step tolerance at $10^{-12}$ kg/s |
|---|---|---|
| 1.0 | $1.00\times10^{-2}$ kg | $1.0\times10^{-12}$ kg |
| $10^{-2}$ | $1.00\times10^{-4}$ kg | $1.0\times10^{-10}$ kg |
| $10^{-4}$ | $1.00\times10^{-6}$ kg | **$1.0\times10^{-8}$ kg** 🔴 |

**Four orders looser.** With **C-34** soft-start and adaptive $\Delta t$, **one tolerance is a different
strictness at every step.** ✅ **Ruled: normalise by the local mass rate (dimensionless relative
residual), or express in kg per timestep.**

### 7r.2 ⚠️ C-141 — exporter ruling accepted; status semantics need one field

✅ Standardise **99.9 %**; **[99.0 %, 99.9 %)** $\Rightarrow$ `ValidityWarning::MaterialBalanceSuboptimal`;
**< 99.0 %** $\Rightarrow$ quarantine from ML training. ✅ The quarantine point is right and consistent with
**C-51**'s atomic-commit rule.

🔴 **But Tier 3 from the exporter is not Tier 3 from the solver:**

| Source | Meaning |
|---|---|
| **Solver** | *"the numerics could not produce an answer"* (**INV-1**) |
| **Exporter** | *"the run solved, and the post-hoc audit found the closure inadequate"* |

**Two different events, one label** — and the manifest cannot distinguish them, which weakens the
**C-136** lattice. ✅ **Ruled: carry the source.** Either `Unsolvable{solver}` / `Unsolvable{post_hoc_audit}`,
or a separate `audit_disposition` field. **Training-set quarantine applies in both cases.**

### 7r.3 ✅ C-142 — CONF-55 closed, with one factual correction

**(A)** ✅ **`N_p/15$ Ghost Finding formally withdrawn.** ✅ The cleanest close in the register: a finding of
mine cited the wrong lines, was audited, and was **withdrawn on evidence** rather than defended.

**(B)** ✅ **D8 §4.2 reclassified as a Remediation Specification**, all criteria subjunctive, with explicit
**Gate 1** (raw physical state vectors, no hardcoded cost/penalty overrides) and **Gate 2** (unified
99.9 % validation, matching assertion bounds).

**(C)** ✅ **Dual-track `FAILURE_PENALTY` accepted** — removed by construction in Rust (**INV-7**),
retained as a legacy artifact in the Python audit.

🔴 **One factual error in (C):** *"...until the Python codebase is fully decommissioned in Milestone M6."*

Verified: **M6 is CO₂-EOR specifics** — Domain 6, 5-state trapping ([`build_plan.md`](build_plan.md)) — and
**"decommission" appears nowhere in the compositional section.** Owner decision 1 is **"Leave Python as
is"**, and the engines **complement** at P3.

✅ **Ruled: strike the M6 decommission reference.** `FAILURE_PENALTY` stays in the **Python audit** with
**no scheduled end date**.

### 7r.4 📌 Register state — no blocking reds remain

> **All eight 🔴 conflicts are closed:** CONF-03 · 11 · 19 · 23 · 43 · 55 · 58 · 59.
> Every remaining item is 🟠 material or 🟡 documentary.

✅ **Two of this round's closures came from measuring the repo, not the docs.** CONF-23's live-repo claim
was wrong in **two** places — README states **no** tolerance, and the enforced value is **99.5 %** in
`utils/run_exporter.py:548` while line **778 displays 99.9 %**; the engine core enforces **neither**.
CONF-55's line reference pointed at the **NPV block**. 📌 **Neither would surface by re-reading the
design documents.**

---

## 7s. Tier A normalization, component closure, typed `Unsolvable` (08-10-2026)

Corrections **C-143 … C-146**.

### 7s.1 ⚠️ C-143 — $\Delta t$-invariance accepted; the normalizer has a vanishing denominator

$$\text{Relative Component Residual:}\quad \frac{\lVert\mathbf{R}_{c,k}\rVert_\infty}{\sum_f \lvert\mathbf{F}_{c,k,f}\rvert + \dfrac{M_{c,k}}{\Delta t}} < 10^{-12}$$

✅ Normalising **is** right — it removes the $\Delta t$ dependence measured in **C-140b**.

🔴 **But the denominator vanishes in depleted cells and dead zones.** Measured, at
$R = 10^{-15}$ kg/s:

| Cell state | $M_{c,k}$ | $\sum_f\lvert\mathbf{F}\rvert$ | denominator | ratio |
|---|---|---|---|---|
| active producer | 5.0e1 | 1.0e−2 | 5.00e1 | **2.0e−17** ✅ |
| transition | 5.0e0 | 1.0e−3 | 5.00e0 | **2.0e−16** ✅ |
| nearly depleted | 5.0e−3 | 1.0e−5 | 5.01e−3 | **2.0e−13** ✅ |
| 🔴 **depleted / dead zone** | 1.0e−9 | 0.0 | 1.00e−9 | **1.0e−6** 🔴 |

At a **true dead zone** the denominator is **0** and the ratio is **undefined**. And meeting $10^{-12}$
*relative* in a depleted cell demands an absolute residual of **$10^{-21}$ kg/s**.

🔴 **Under INV-7 such cells are COMMON, not exceptional** — absurd rates create them deliberately.

✅ **Ruled — a mixed criterion, never a bare ratio:**

```
converged  IF   R_abs < tol_abs                     // declared global floor, e.g. 1e-14 kg/s
           OR   R_rel < 1e-12                       // where denom > 0
denom <= 0 -> absolute branch ONLY                  // by construction, never a ratio
```

⚠️ **The residual form must be DECLARED as a RATE** (kg/s). The ratio is dimensionless **only** if
$\mathbf{R} = \sum_f \mathbf{F} - \mathrm{d}M/\mathrm{d}t$; with a mass-form residual the ratio carries
units of seconds.

> 📌 **C-143 is the fifth instance of one defect class — see §7s.5.**

### 7s.2 ✅ C-144 — component-resolved closure, with two conditions

$$\mathrm{MB}_{closure,c} = 1 - \frac{\left|M_{c,total}(T_{max}) + \sum_w Q^{cum}_{p,c,w} - \sum_w Q^{cum}_{i,c,w} - M_{c,total}(0)\right|}{M_{c,total}(0) + \sum_w Q^{cum}_{i,c,w}}$$

✅ **Accepted.** $c\in[1,N_c]$, $w\in[1,N_w]$ removes the overload cleanly, and per-component checking is
**strictly better** than a total-only check: a dominant component can no longer mask a trace component's
error.

| Condition | Detail |
|---|---|
| ⚠️ **(a) condition basis must be declared** | $M_{c,total}$ is a **reservoir** quantity; $Q^{cum}_{p,c,w}$ is naturally a **surface** quantity. Mixing them without an explicit $B$-factor / density conversion **manufactures a closure error that looks like a mass-balance failure**. 📌 This is the **same gap CONF-15 retains**, and it now also gates **C-144** |
| ⚠️ **(b) per-component $99.9\,\%$ may be unattainable for trace components** | A component at $10^{-3}$ mole fraction carries so little mass that its closure sits near solver tolerance — the test then fails on **arithmetic**, not physics |

✅ **Ruled:** strict per-component test for components above a **declared mass fraction**; **mass-weighted
aggregate** for the trace remainder; **both the threshold and the fraction recorded in the manifest.**

### 7s.3 ✅ C-145 — typed `Unsolvable` split, exactly as ruled

```rust
pub enum UnsolvableSource {
    SolverFailure { error: TypedError },                                  // INV-1
    PostHocAuditFailure { metric: String, value: f64, threshold: f64 },    // exporter
}

pub enum ValidityClass {
    Validated,                                        // Tier 1
    ConvergedOutsideEnvelope(Vec<ValidityWarning>),   // Tier 2
    Unsolvable(UnsolvableSource),                     // Tier 3
    Aborted,                                          // Tier 4
}
```

✅ Provenance preserved; ✅ **both variants quarantine**; ✅ `Vec<ValidityWarning>` is consistent with
**C-128**'s requirement that warnings be **distinguishable kinds**, not a string list. **C-141 closed.**

### 7s.4 ✅ C-146 — CONF-55 locked

Withdrawal locked; **M6 decommission struck**; `FAILURE_PENALTY` retained **indefinitely** in the Python
audit scope with **no sunset date**; Python remains active for high-level proxy experiments; Rust core
strictly unconstrained under **INV-7**. ✅ Consistent with owner decision 1 and with the engines
**complementing** at P3.

### 7s.5 📌 The one rule that has now predicted five defects

| # | Where | Vanishing quantity used as divisor / comparator |
|---|---|---|
| **C-69** | Verma-Pruess clogging | $\phi_0 - \phi_c$ denominator |
| **C-102** | $\alpha_0$ phasing fit | crosses zero at $\theta = 43.5^\circ$ |
| **C-115** | symbol-register range type | `(f64,f64)` cannot express an open bound |
| **C-134** | adaptive AMG coarsening | tie-break on traversal order / address |
| **C-143** | Tier A residual normalisation | $\sum_f\lvert\mathbf{F}\rvert + M/\Delta t$ |

> 📌 **Every one: a quantity that can approach zero was used as a divisor or a comparator, with nothing
> else governing the limit.**
>
> ### ✅ The rule
> **No convergence criterion, correlation, or normalised ratio may rely on a denominator without an
> explicit branch for its zero.**
>
> 📌 Recorded once and cited by every future occurrence. It is the highest-transfer item in twenty-one
> rulings: it has now predicted five defects **after the fact**, across **four different documents**,
> with **no shared author**.

⚠️ **And the inverse pattern, worth stating too:** every fix in this register has been **directionally
correct and locally incomplete**. Normalising **is** right; `petekIO` **was** memory; the
overall-composition formulation **is** better. The recurring gap is never wrong intuition — it is
**the limit case nobody wrote down.**

---

## 7t. Rate-form residual and the dual-branch Tier A gate (08-10-2026)

Corrections **C-147 … C-149**. **The gate is accepted and verified sound.**

### 7t.1 ✅ C-147 — rate-form residual, dimensional anomaly closed

$$\mathbf{R}_{c,k} = \sum_f \mathbf{F}_{c,k,f} + \frac{M^{n+1}_{c,k}-M^{n,k}_{c,k}}{\Delta t} - Q_{c,k}\quad[\text{kg/s}]$$

✅ Sign convention self-consistent: $\mathbf{R}=0 \Rightarrow \mathrm{d}M/\mathrm{d}t = Q - \sum_f \mathbf{F}$,
i.e. $\mathbf{F}$ is net **outflow** and $Q$ net **injection**.
✅ Using $\lvert\mathbf{F}\rvert$ in the denominator while $\mathbf{F}$ stays **signed** in the residual is
correct — **magnitude for scale, sign for error**.

### 7t.2 ✅ C-148 — dual-branch gate accepted and verified

```
converged  IF  R_abs < tol_abs                                 // 1e-14 kg/s
           OR  (denom > denom_min  AND  R_abs/denom < 1e-12)   // denom_min = 1e-11 kg/s
```

**Measured — the `OR` yields $\mathbf{R} < \max(tol_{abs},\ 10^{-12}\cdot denom)$:**

| `denom` kg/s | `rel*denom` | effective tol | dominant branch |
|---|---|---|---|
| 1e−1 | 1.0e−13 | 1.0e−13 | relative |
| 1e−3 | 1.0e−15 | 1.0e−14 | **absolute** |
| 1e−6 | 1.0e−18 | 1.0e−14 | **absolute** |
| 1e−9 | 1.0e−21 | 1.0e−14 | **absolute** |

✅ Branch switch lands exactly where intended — **relative above ~$10^{-4}$ kg/s, absolute below**.
✅ **The Rust short-circuit is correct**: `denom > denom_min` is tested **before** the division, so **a
division by zero is impossible**. ✅ It resolves the case measured in **C-143** — at `denom = 1e-9`,
$R = 10^{-15}$ now **passes via the absolute branch**, where the bare ratio gave $10^{-6}$ and failed.

| Precision addition | Detail |
|---|---|
| ⚠️ **(a)** `denom_min` is a **safety guard, not an accuracy knob** | Measured: $10^{-11}$ divides without issue in `f64`. It exists **only** to stop `denom = 0`. ✅ **Record it as such so nobody tunes it** |
| ✅ **(b)** The routed-off zone is **loose by design, and bounded** | At `denom = 1e-12` the gate implies **1.0 %** local relative error. Measured worst case — **every** one of 1.1 M cells at its worst — is 1.1e−8 kg/s $\Rightarrow$ **0.35 kg/yr** against a $10^{9}$ kg reservoir, i.e. **$3.5\times10^{-10}$ relative**. **Tier A's local looseness is bounded by Tier C's global closure** |
| 🔴 **(c)** No `is_finite()` guard | Measured: `R = NaN` $\Rightarrow$ gate returns `false` $\Rightarrow$ solver iterates to max-iter then errors. **Safe but confusing.** Given three NaN generators already found (**C-69**, **C-102**, **C-143**), ✅ **add `if !r_abs.is_finite() \|\| !denom.is_finite() { return Err(TypedError::NonFiniteResidual) }`** |

### 7t.3 ⚠️ C-149 — $\Delta t$-invariance is PARTIAL; narrow the claim

| Term | $\Delta t$ behaviour |
|---|---|
| $\sum_f \lvert\mathbf{F}_{c,k,f}\rvert$ | ✅ $\Delta t$-**independent** — the original defect is fixed for throughput-dominated cells |
| $M^{n+1}_{c,k}/\Delta t$ | 🔴 **inversely proportional to $\Delta t$** — denominator grows as $\Delta t$ shrinks, so the criterion becomes **looser** at small steps in **storage-dominated** cells (shut-in, depletion) |

✅ **Ruled: narrow the claim to what is true.**

> *"$\Delta t$-invariant in throughput-dominated cells; storage-dominated cells carry an explicit $1/\Delta t$
> scaling."*

⚠️ This is a **design choice, not a defect** — loosening at small $\Delta t$ on a storage-dominated cell is
physically reasonable. But it must be **stated**, or a later reader will re-open it as an unfixed **C-140b**.

### 7t.4 📌 The rule fired again — two consecutive predictions

§7s.5's zero-denominator rule **predicted this finding**: it was raised against the proposal *before* the
code was written, and the fix was built with an explicit zero branch (`denom_min`), verified to make
division by zero impossible.

> 📌 **Two consecutive findings now derived from one recorded rule, in different subsystems —
> clogging/perforation $\rightarrow$ convergence gating — with no shared author.**

✅ **And the two-tier coverage result is the substantive one.** The gate is loose by design in dead cells
— up to **1 % local** — and the worst case across 1.1 M cells contributes **$3.5\times10^{-10}$** of
reservoir mass per year. **Tier A's local looseness is bounded by Tier C's global closure.** That is the
architecture working as designed, and it is worth stating explicitly because the gate would otherwise read
as a weakening.

---

## 7u. Precision additions locked; one INV-1 boundary (08-10-2026)

Corrections **C-150 … C-152**.

### 7u.1 ✅ C-150 — `denom_min` locked as a safety guard

Documented `#[doc = "SAFETY GUARD against div-by-zero; DO NOT TUNE"]` ✅ correct — tuning it would move
the active/dead-zone boundary and silently change which branch governs.

⚠️ **One factual overreach corrected:** the justification *"or numbers approaching $f64$ subnormal
limits"* does **not** apply. $f64$ min-normal is $\approx 2.2\times10^{-308}$, so $10^{-11}$ sits **297
orders of magnitude above** it.

> ✅ **Div-by-zero is the only real reason the guard exists.** Harmless to the design — but recorded
> accurately, so nobody later "generalises" the reasoning to a value where it would matter.

### 7u.2 🔴 C-151 — `NonFiniteResidual` must NOT trigger Δt cutback or a fallback solver

A `NaN` from a **failed flash** is a **thermodynamic failure**. **INV-1** is explicit:

| Condition | Required | Forbidden |
|---|---|---|
| Flash failure | return `ThermodynamicFlashFailed` and **stop** | substitute a single-phase guess |
| Non-convergence | return a typed error and **stop** | revert to the surrogate, retry silently with a looser tolerance |

🔴 The stated impact — *"immediate adaptive time-step cutback ($\Delta t \to \Delta t/2$) **or fallback
solver routines**"* — is **precisely the silent-retry pattern INV-1 prohibits.**

✅ **Ruled — the boundary is the *kind* of failure, not its size:**

| Error variant | Trigger | Required behaviour |
|---|---|---|
| `TypedError::NonFiniteResidual` | NaN / Inf in residual or denominator | 🔴 **Hard stop, INV-1.** A NaN means the thermodynamics or the algebra is broken; halving $\Delta t$ does not repair a broken flash |
| `NewtonDivergence` | residuals **finite**, merely above tolerance | ✅ $\Delta t$ cutback **is** legitimate — standard adaptive stepping, not a fallback |

🔴 **But the cutback must not be silent.** The manifest records **`cutback_count`** and
**`total_cutback_time`**, and a run that converged **only after** cutbacks is surfaced.

✅ **And "fallback solver routines" is struck outright** — there is no fallback solver under **INV-1**, in
any circumstance.

### 7u.3 ⚠️ C-152 — my C-149 caveat was too broad; the precise statement is narrower

I wrote that the storage term $\propto 1/\Delta t$ breaks invariance in storage-dominated cells.
🔴 **Measured, that is wrong in the common case.**

**Pure storage** (shut-in, zero flux, zero $Q$):

$$R = \frac{\mathrm{d}M}{\mathrm{d}t} = \frac{M_1-M_0}{\Delta t},\qquad \mathrm{denom} = \frac{M_1}{\Delta t}\;\Longrightarrow\; \frac{R}{\mathrm{denom}} = \frac{M_1-M_0}{M_1}$$

**The $1/\Delta t$ appears in both and cancels exactly.** Verified invariant across
$\Delta t \in [10^{0}, 10^{-8}]$ — ratio constant at $1.000000\times10^{-09}$.

**Non-invariance occurs only in the MIXED case**, where $R$ and $\mathrm{denom}$ are dominated by
**different** terms — measured, flux-dominated $R$ with storage-dominated $\mathrm{denom}$ improves **4
orders** as $\Delta t$ shrinks 4 orders.

> ✅ **Corrected specification:** *"the criterion is $\Delta t$-invariant whenever $\mathbf{R}$ and the
> denominator share a dominant term; it is scale-dependent only in the mixed regime."*
>
> ✅ **Useful corollary:** a shut-in cell at convergence has $\mathrm{d}M \to 0 \Rightarrow R \to 0$, so it
> fires the **absolute branch at iteration 1** and **never reaches the relative branch** — no cutback
> loop, no stalling.

### 7u.4 📌 One rule behind both INV-1 collisions

**C-93** (`clamp(r_D, …)`) and **C-151** ($\Delta t$ cutback / fallback solver) are different mechanisms
and **one rule**:

> **A correction that makes a failure _disappear_ rather than _reported_ is forbidden** — whether it
> hides behind a `clamp`, a halved $\Delta t$, or an alternate solver.
>
> ✅ **The test is always the same: after this, can a reader tell what happened?**
> A $\Delta t$ cutback that increments a recorded counter and appears in the manifest **passes**. One
> that quietly retries until it succeeds **fails**.

⚠️ **And C-152 is the second time this round a measurement corrected my own caveat** (first: the Kahan
result in **C-134**). I predicted the storage term broke invariance; measured, it cancels in the case I
had in mind.

> 📌 **The failure mode is consistent: I generalise from a mechanism to a rule without checking whether
> the mechanism actually propagates.** That is the same shape as the error this register was opened to
> document — which is the argument for measuring rather than reasoning.

---

## 7v. Cutbacks, and the third break of the status partition (08-10-2026)

Correction **C-153**. **C-150, C-151, C-152 accepted exactly as restated.**

✅ **Fallback solvers struck from the architecture entirely** — there is exactly one physical PDE solver.
That is the strongest available form of the ruling. ✅ The $10^{-11}$ rationale now records div-by-zero
only, citing $f64$ min-normal $2.22\times10^{-308}$. ✅ The $\Delta t$-cancellation proof and its
confinement to the mixed regime are correct.

### 7v.1 🔴 C-153 — the cutback trigger re-opens the C-136 partition

*"A run completing after cutbacks is surfaced as `ConvergedOutsideEnvelope`."*

Measured, adding a third disjunct to Tier 2 without excluding it from Tier 1:

| | |
|---|---|
| Case | `completed = TRUE`, all `Implemented`, all params **inside** envelope, `cutback_count = 3` |
| Tier 1 matches | **true** |
| Tier 2 matches | **true** |
| | 🔴 **OVERLAP** — a partition requires **exactly one** |

**Ruled — two changes:**

| # | Change | Why |
|---|---|---|
| **(a)** 🔴 **`cutback` is a WARNING, not a Tier condition** — `ValidityWarning::NewtonCutback { count: u32, total_cutback_time: f64 }` | A cutback is **not** an envelope violation. It is the solver **handling** a physical non-linearity |
| **(b)** ✅ **`Tier 2 := completed ∧ ¬Tier 1`** | The **complement** rule from **C-128** — what closed the original hole |

✅ Verified under Fix B: Tier 1 fires, Tier 2 does not — exactly one match.
⚠️ **Fix A** (also excluding `cutback_count == 0` from Tier 1) restores exclusivity but leaves **two
predicates that must be kept in sync by hand**. **Fix B is the same rule that already solved this problem
once, so reuse it.**

### 7v.2 📌 The durable change is structural

This is the **second** time the partition has been re-broken by an adjacent change — first by `Aborted`
(**C-132**), now by `cutback`.

> 📌 **A partition maintained by _listing conditions_ degrades every time a new condition arrives. A
> partition maintained by _one predicate plus its complement_ cannot.**
>
> ✅ **`ValidityClass` must derive from a single `Tier1Predicate` plus `NOT` — never from an
> independently-maintained condition list.**

```rust
fn tier1(cap: &CapabilityDeclaration, checks: &VAndVChecks) -> bool {
    cap.all_active_implemented() && checks.all_inside_envelope()   // the ONLY list
}

match (run.completed, !run.had_error) {
    (false, _)      => ValidityClass::Aborted,                       // C-132
    (true,  false)  => ValidityClass::Unsolvable(UnsolvableSource::SolverFailure { .. }), // C-145
    (true,  true) if tier1(..) => ValidityClass::Validated,          // Tier 1
    (true,  true)               => ValidityClass::ConvergedOutsideEnvelope(run.warnings.clone()),
}
```

### 7v.3 📌 And the category half — one recurring error, three rulings

A cutback is **not** an envelope violation. Putting it in the Tier predicate conflates *"your input is
unverified"* with *"the solver worked hard."*

| Ruling | The two reasons that got merged |
|---|---|
| **C-128** | `ModuleAbsent` (INV-6) vs `EnvelopeExceeded` (INV-7) |
| **C-141** | solver failure vs post-hoc audit failure |
| **C-153** | envelope violation vs adaptive cutback |

> 📌 **Three rulings, one recurring error: putting two different _reasons_ into one _status_.**
>
> ✅ **`ValidityWarning` is the home for reasons. `ValidityClass` carries only the verdict.**

---

## 7w. Dimensional repairs and two-phase hardening (08-10-2026)

Corrections **C-154 … C-158**.

### 7w.1 ✅ C-154 — CONF-51 dimensional repairs VERIFIED CORRECT

$$\mathcal{K} = \frac{4\sqrt{2/\pi}\,K_{IC}(1-\nu^2)^{3/4}}{\left(12E^3\mu\,\dfrac{Q_{inj}}{H_f}\right)^{1/4}}$$

✅ $[E'^3\mu'q_{2D}]^{1/4} = [\text{Pa}^3\cdot\text{Pa·s}\cdot\text{m}^2/\text{s}]^{1/4} = \text{Pa·m}^{1/2}$, matching
$[K_{IC}] = \text{Pa·m}^{1/2}$ $\Rightarrow$ $\mathcal{K}$ **strictly dimensionless**.
✅ **And the $(1-\nu^2)^{3/4}$ placement is self-consistent** — verified by expanding $E'^3 = E^3/(1-\nu^2)^3$
into the denominator, which reproduces the submitted numerator exactly.

✅ Both $w_0$ forms close in metres. ✅ Model-specific viscosity routing ($\mu^{1/4}$ PKN/penny,
$\mu^{1/6}$ KGD). ✅ Krieger–Dougherty $\beta = -[\eta]\phi_m = -(2.5)(0.64) = \mathbf{-1.60}$.
✅ Bandis–Lumsden–Barton (1983) DOI independently verified.

🔴 **Residual (a) — the two $w_0$ forms are not independent.** Under the volume balance
$q_{2D}t = 2L_fw_0$:

| Form | Result |
|---|---|
| Form 2, substituted | $w_0^2 = 2K_{IC}^2(1-\nu^2)^2L_f/E^2$ |
| Form 1, squared | $w_0^2 = \mathcal{C}_K^2K_{IC}^2(1-\nu^2)^2L_f/E^2$ |

**They agree only if $\mathcal{C}_K = \sqrt2 = 1.4142$.** ✅ **$\mathcal{C}_K$ is not a free constant —
Form 2 fixes it.**

⚠️ **Residual (b) — the Type-I/Type-II switch is discontinuous.** $H_f$ grows during propagation, so
$\mathcal{K}$ can cross 1 mid-growth, and the two formulas **do not agree** at $\mathcal{K}=1$.
✅ **Ruled: lock $\mathcal{C}_K = \sqrt2$ and blend over $\mathcal{K}\in[0.8,\,1.25]$ — no hard switch.**

### 7w.2 🔴 C-155 — two-phase hardening fixed, but $C^0$ not $C^1$

✅ Pre-peak parabola verified: $c(0)=c_{yield}$, $c(\bar\varepsilon_p^{peak})=c_{peak}$, $H>0$ strictly
for $x<1$ — **matches the stated modulus exactly.** ✅ Post-peak exponential $\to c_{res}$.

🔴 **Measured at the switch:**

| $\bar\varepsilon_p$ | $H = \mathrm{d}c/\mathrm{d}\bar\varepsilon_p$ |
|---|---|
| 0.0499 | +3.2e+05 |
| 0.0500 | +1.6e+02 |
| **0.0501** | **−4.455e+08** 🔴 |

**$H$ jumps by $\sim-\eta(c_{peak}-c_{res})$ at the transition.** The law is **$C^0$ but not $C^1$**, so the
consistent tangent **jumps** and the asymptotic quadratic rate is lost exactly at the switch.

✅ **Ruled: declare it, and average the tangent over the active set at the switch** — the same
`ValidityWarning`-style treatment as **C-145**; the switch is an **event**, not a silent kink.

🔴 **Drucker-Prager convention mismatch.** The submitted $F = \sqrt{J_2} + \alpha I_1 - k$ uses $\sqrt{J_2}$,
while the standard $k,\alpha$ pair **assumes** $F = \sqrt{J_2/3} + \alpha I_1 - k$. Measured at $c=1$ MPa,
$\phi=30°$: $k = 1.200\times10^6$ with the $\sqrt3$ factor, $2.078\times10^6$ without — **ratio exactly
$\sqrt3 = 1.7321$**.

✅ **Ruled: either write $F = \sqrt{J_2/3} + \alpha I_1 - k$, or drop the $\sqrt3$ from $k$ and $\alpha$.**

⚠️ **Declare** that one scalar $\bar\varepsilon_p$ drives **both** $k$ and $\alpha$. Standard isotropic DP
hardening scales $k$ **alone** with $\alpha$ fixed; varying both is a **non-proportional** law and should be
named as such.

### 7w.3 ⚠️ C-156 — Schur condensation eliminated; two residuals

✅ Eliminating static condensation of $\mathbf{K}_{uu}$ is **correct** — the ill-conditioning argument is
right ($E/(1-\nu^2)$ vs $c_t$ contrast propagating into $\mathbf{S}_p$). ✅ Block ILU(1) + FGMRES on the 2×2
THMC block is the right answer.

🔴 **Residual (a) — Cholesky on $\mathbf{K}_{uu}$ is valid only while the displacement block is ELASTIC.**
With active plasticity $\mathbf{K}_{uu}$ contains $D^{ep}$ and is **non-symmetric**.
✅ **Ruled: preconditioner selection is conditioned on whether any plastic set is active.**

🔴 **Residual (b) — §C prohibits splitting $\mathbf{A}_{pp}$ by pressure/transport, then prescribes
*CPR-AMG* on that same block — which *is* that split.** 🔴 **Third resurrection of C-79.**
✅ **Ruled: $\mathbf{A}_{pp}$ takes a single coupled preconditioner (Block ILU(1) or unsymmetric AMG).
No CPR inside it.**

### 7w.4 ⚠️ C-157 — CPPM citations plausible, DOIs outstanding

*SComputational Inelasticity* (Simo & Hughes, Springer 1998) ✅ real and canonical. Simo & Taylor
(1985/86), Abbo & Sloan (1995) ✅ plausible. ✅ **Terminology corrected** — $\mathbf{D}^{alg} =
\partial\boldsymbol\sigma_{n+1}/\partial\boldsymbol\varepsilon_{n+1}$ distinguished from the continuum
$\mathbf{D}^{ep}$ **is the right distinction** and matters for Newton.

⚠️ **All three need DOIs before M7b** ([`literature_todo.md`](literature_todo.md) §1 test 1).

### 7w.5 ✅ C-158 — CONF-17 closed

Cell-local $P$, $z_i$ as **sole** coupling inputs; $\bar{P}_{res,\text{eff}}$ demoted to post-processed
diagnostic; **zero-feedback constraint** explicit — the strongest form, since it forbids reuse rather
than merely demoting.

⚠️ **One addition:** **RF cannot be cell-local.** D2 §6.1 drove **both** miscibility *and* RF from the
scalar. Miscibility correctly becomes a field; **RF's definition must change** to a volume integral of
local quantities — $\mathrm{RF} = \int_\Omega \Phi_{prod}/\int_\Omega \Phi_{init}$ — or RF is accidentally
made local too.

### 7w.6 📌 Two promoted rules

| # | Rule | Basis |
|---|---|---|
| 1 | ✅ **Adopt the `#[cfg(test)]` dimensional unit gate** over the typed newtypes (**C-124**/**C-131**) | 🔴 **Seven** dimensional failures in this register: C-83, C-88, C-106, C-122, C-140, C-151, and now the DP $\sqrt3$ convention. ✅ **The gate would have caught all seven before review** |
| 2 | 🔴 **Name *CPR inside $\mathbf{A}_{pp}$* a banned construct in the CI gate** | **C-79 has been resurrected three times** after closure. A conflict that keeps returning is **structural**, not documentary |

> 📌 **Rule 1 is the highest-value engineering artefact this register has produced** — it converts the
> most-repeated defect class from a review finding into a build failure.

---

