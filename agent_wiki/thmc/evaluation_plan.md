# Evaluation Plan — 3D THMC Design Set

> [!CAUTION]
> **SUPERSEDED on 07-10-2026 — retained for the record.**
>
> The decision is now: **build the compositional engine first**, decoupled from the Python codebase,
> with a **separate flaw register**. The O1–O4 framing below (ADOPT / HARVEST / PILOT / REJECT) has
> been overtaken: **O1 has effectively been chosen, but narrowed to the compositional subset and with
> the coupling deferred.**
>
> **The live plan is [`../compositional/build_plan.md`](../compositional/build_plan.md)** — milestones
> M0 → M6 with measurable gates. Supporting pages:
> [`../compositional/README.md`](../compositional/README.md) (scope),
> [`../compositional/separation_doctrine.md`](../compositional/separation_doctrine.md) (decoupling),
> [`../compositional/register_spec.md`](../compositional/register_spec.md) (separate register),
> [`../compositional/spec_defects.md`](../compositional/spec_defects.md) (spec gaps by milestone).
>
> **What survives from this page:** the HARVEST deliverables in §5.1 are **not** part of the current
> task — they modify the Python engine, which is now explicitly out of scope. They return as a
> **separate** work item, and should not be merged with compositional work.
>
> **Three toolchain blockers discovered while re-planning** (verified 07-10-2026): no Rust toolchain
> (`cargo`/`rustc`/`rustup` all absent); `audit/registry.py:114` `LOCATION_RE` accepts **`.py` only**, so
> a Rust finding cannot be registered in the existing register; `audit/continuity.py:150-189`
> hard-imports the surrogate and cannot be reused.

**Question this document originally answered:** how do we, with the project manager, decide whether to
adopt, adapt, or reject the `3D_THMC_docs` design set — without writing a line of production code first?

**Status at time of writing:** proposed. Nothing in this plan was executed before it was superseded.
Companion reading: [`conflict_and_gap_register.md`](conflict_and_gap_register.md).

---

## 1. The decision this plan supports

The design set describes a **Rust** 3D THMC compositional simulator. There is no Rust code in this
repository (verified: only `pyproject.toml`), no crate, no CI, and every acceptance box in D2 §6.2 is
unchecked. The live engine is a Python 0D physics-informed surrogate in `core/engine_surrogate/`.

There are exactly **four** defensible outcomes. The plan exists to choose between them with evidence
rather than enthusiasm.

| # | Outcome | Meaning | Cost |
|---|---|---|---|
| **O1** | **ADOPT** | Build the Rust core as specified and route optimisation through it | 24–36 person-months; retires the current engine; requires all 4 🔴 conflicts resolved first |
| **O2** | **HARVEST** | Take the *physics contracts and acceptance criteria* into the existing Python engine; build no Rust | 4–8 person-months; addresses a subset of the live finding register; keeps the current optimiser path |
| **O3** | **PILOT** | Build one thin vertical slice in Rust (2D single-phase FVM + mass balance gate) to prove the toolchain, then re-decide | 3–5 person-months |
| **O4** | **REJECT** | File the design set as reference material only; keep the current engine | ~0 |

> **Recommendation for the project-manager meeting: do not choose between these in one meeting.**
> Choose **O3 (PILOT)** as a bounded experiment, and take the **O2 (HARVEST) items that are free**
> — the acceptance criteria and the physics contracts — regardless of the Rust decision. Those are
> independent of the language question and pay for themselves.

> [!NOTE]
> **Updated 07-10-2026 after the reservoir-engineer ruling**
> ([`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md)):
> - **CONF-04 is resolved** — economics placement is now decided (production adjoints from the core,
>   prices and discounting off-core). That removes one of the four Stage-0 blockers.
> - **CONF-01 is retracted** and downgraded 🔴 → 🟠. It no longer blocks adoption on physics-sign grounds.
> - **Blocking conflicts remaining: CONF-02, CONF-03, CONF-58, CONF-59.** CONF-58/59 were opened by the
>   ruling itself and must be settled **before** any CONF-04 refactor moves economics off-core —
>   otherwise the refactor will delete four working cost terms from the live engine.
> - The ruling's CONF-02 remediation **collapses to fixing CRIT-14 / CRIT-15 in the existing engine**:
>   `core/engine_surrogate/analytical_models.py:176-193` already implements exactly the Koval
>   $K_K = H\cdot E_v$ structure the engineer recommends. **Nothing needs importing from the design
>   set for CONF-02** — this shrinks the THMC scope materially.
> - **New Stage 1 item:** decide mid-year vs. end-year discounting (**CONF-62**, **+4.88 % NPV**).

---

## 2. Stage 0 — Documentation remediation (no code, 1–2 weeks)

Everything here is editing `3D_THMC_docs/` or annotating them. It must happen before any technical
evaluation, because **7 documents currently contain physics that a completed audit rejected** and
**6 documents report remediations that have not happened**.

### 2.1 Mandatory corrections

| Action | Driven by |
|---|---|
| Replace the $f_g(t)$ **closure** at all 7 occurrences: add $S_{or}$, use Corey/Brooks-Corey exponents on separate saturations, add the water phase, guard $S_g\to S_{gc}$. ⚠️ **Do not "fix" the sign** — it is correct | **CONF-01** (corrected) |
| Re-derive the HCPVI cap with provenance, or delete it; state its behaviour at the shipped default HCPVI 7.69. Engineer's proposed fix = the Koval $K_K=H\cdot E_v$ already in `analytical_models.py:176-193`, i.e. a **live-engine** fix | **CONF-02** |
| Rewrite every "penalties fully removed" statement in D1 §10.3, D2 §6.1, D4, D6 §1.2, D8 §2.4, D8 §4.2 as a **requirement**, with a `Status:` line | **CONF-03, CONF-19, CONF-55** |
| Add a hydrocarbon-gas revenue term to `NOCF`, or state explicitly why gas sales are excluded and where they are accounted | **CONF-11** |
| Define "the numerical core" by interface/component, then assign mass-balance tolerances per tier. ⚠️ Use only **sourced** numbers — `10^-6` appears nowhere in the set | **CONF-23** (corrected) |
| ~~Choose: economics in-core or off-core~~ — **DECIDED 07-10-2026**: core publishes $\partial q_i/\partial\mathbf{x}$; prices and discounting off-core | **CONF-04** ✅ resolved |
| **Record the CONF-04 ruling as an ADR** before any refactor moves economics off-core | **CONF-04, CONF-58, CONF-59** |
| **Carry CO₂ purchase cost, CO₂ recycle cost, storage credit and carbon tax into whatever off-core economic module replaces `surrogate_engine.py:624-646`** | **CONF-58** |
| **Split the produced-gas stream** so recycled CO₂ is never booked as sales gas — while *also* adding the missing sales-gas revenue (live **CRIT-18**) | **CONF-59 + CONF-11** |
| **Split invariant 12**: keep the step penalty for solver non-convergence; use smooth barriers only for bounded-physics constraints, and only where no adjoint gradient is required | **CONF-60** |
| Replace the phantom "48 tests in `test_surrogate_engine.py`" target with **one regression test per registered finding** | **CONF-61** |
| Decide mid-year vs. end-year discounting; the live engine uses end-of-year (`surrogate_engine.py:649-650`) against `+4.88 %` NPV | **CONF-62** |
| Add `seed` / `random_state` / determinism clauses to D7, adopting `GeostatisticalParams` fields | **CONF-43** |
| Add a price deck (oil price, gas price, storage credit $/tCO₂e, CO₂ purchase + recycle cost, carbon tax, OPEX split, escalation, discount rate range) | **CONF-37, CONF-58** |
| Fix the FZI derivation (`1012.7` → `1014.24`) or delete it | **CONF-39** |
| Fix `k_micro = w²/12` → `w³/12` | **CONF-47** |
| Normalise the chemical strain by a reference volume | **CONF-16** |
| Add the Bethel (solid-volume) correction to the porosity update | **CONF-18** |
| Reconcile the Net-utilisation floor to one value | **CONF-35** |
| Resolve the pinch-out disjunction | **CONF-05** |
| Reconcile the CAPEX band with the worked example | **CONF-20** |
| Repair the `bip_matrix` (21 $k_{ij}$ for 6 components) | **CONF-29** |
| Correct the DCF year-2 arithmetic | **CONF-30** |
| Fix D5's shifted tolerance-table column | **CONF-22** |
| Either carry the full 6-component `AnisotropicTensor3D` in the D7 payload or drop the upscaler's mandate | **CONF-41** |
| Reconcile D8's 20 000–60 000 MSCFD with D5's anti-pattern text | **CONF-54** |
| Add $L_2$-norm Richardson spatial grid-refinement across $N_x\times N_y\times N_z$ mesh levels (engineer's remediation for the missing spatial study) | **CONF-26** |

### 2.2 Mandatory additions (not corrections)

| Addition | Driven by |
|---|---|
| **Reference solution dataset per V&V test.** D5 supplies **zero** absolute expected values; every criterion is relative to an unsupplied reference. Without this, no test is executable. | `vv_testing_framework.md` G-01 |
| **Spatial convergence study** — D5 promises it and omits it | **CONF-26** |
| **Manufactured solutions** — zero mentions | G-04 |
| **Replacement reliability gate** — D5 rejects coverage with no substitute | **CONF-21** |
| **Preconditioner + sparse format + Krylov specification** — D5 names Krylov; D1/D2 specify none | **CONF-14** |
| **`petekIO` format specification** — named 3×, defined 0× | **CONF-30b** |
| **Unit/conversion table for the whole suite** — D6 uses MSCF, STB, MSCFD, mD, MPa, K, USD, acres with no declared convention | D6 |
| **Gradient-verification procedure for the adjoint engine** | D1 §10.2 |

### 2.3 Stage 0 exit criterion

**All 🔴 conflicts resolved or explicitly accepted as risks in writing by the project manager.**
Currently 🔴 = **CONF-02, CONF-03, CONF-58, CONF-59** (CONF-01 downgraded 🟠, CONF-04 resolved).
Until then, no code is written and no benchmark is run.

---

## 3. Stage 1 — Turn the documents into measurable acceptance criteria (no new engine, 3–4 weeks)

The design set already contains a complete, unused acceptance contract. This stage converts it into
tests against the **existing** engine, so we learn whether the physics claims are even right before
spending anything on a rewrite.

### 3.1 Promote the V&V Level-1 and Level-2 gates to live tests

| Test | Source | Where to implement | Target |
|---|---|---|---|
| **Buckley–Leverett** shock-front position, ≤ 0.1 % of Welge construction | D5 §2.1 | 1D harness over `profile_generator_fast.py` fractional flow | **this is the CONF-01 test** |
| **Terzaghi** 1D consolidation, ≤ 0.05 % | D5 §2.2 | new 1D test against the Fourier-series solution | — |
| **Mandel** centre overshoot ≤ 0.2 %, Mandel–Cryer rise reproduced | D5 §2.2 | new 2D test | — |
| **Avdonin** thermal profile ≤ 0.1 %, exact energy balance | D5 §2.4 | new 1D thermal test | — |
| **Component-wise mass balance per component per step** | D5 §3.1 | `tests/test_physical_invariants.py` (already references `cash_flows_yearly.csv`) | decide the tolerance per **CONF-23** |
| **Saturation closure** $S_o+S_w+S_g=1$ on **every** timestep | D5 §3.2 | directly tests live **CRIT-17** | currently fails 27 % of steps |
| **M-matrix / symmetry / positive-definiteness of the flow operator** | D5 §3.3 | new test | — |

**Deliverable:** a runnable `tests/thmc_vv/` suite. The first test to run is Buckley–Leverett, because
**CONF-01 predicts it will fail on mobility-ratio sensitivity.**

### 3.2 Promote the repository's existing benchmarks to the D5 tolerances

The repository already holds assets D5 says it lacks.

| D5 requirement | Asset already in repo | Work |
|---|---|---|
| SPE 5, RF + produced-gas composition ≤ 1.5 % | `validation/spe5_config.py`; `agent_wiki/validation/benchmarks.md` §3 | run it; record the delta |
| Comparison simulator | `validation/cmg/flu/` `gmflu001`–`gmflu003`, read via `h5py` in `validation/sr3_reader.py` | add SPE-1-like and SPE-3-like cases |
| MMP correlations | `evaluation/mmp.py` — 5 validated correlations | substitute for D6's unnamed "MMP calculator" |
| Geostatistical parameters incl. `random_seed`, variogram, sill, nugget, anisotropy | `core/data_models.py:358-404` | adopt as D7's missing clauses |

### 3.3 Measure the D2/D8 NPV-unification criteria against live code

| Criterion | Measurement command | Expected today |
|---|---|---|
| D8 §4.2-2: `economic_npv_usd` = `results["npv"]` = `Cumulative_NPV_USD` to **$0.01** | `pytest tests/test_physical_invariants.py -v` (references `cash_flows_yearly.csv` at lines 8, 159, 181, 184, 211) | **unknown — must be measured** |
| D8 §4.2-3: penalties removed | grep for `FAILURE_PENALTY`, `breakthrough_impact`, `-1e12` in `core/objectives/`, `core/engine_surrogate/` | **fails** (invariant 12) |
| D8 §4.2-1: no `N_p/15` averaging | read `core/engine_surrogate/surrogate_engine.py:646-651` | **passes** |
| CRIT-18: gas sales excluded from revenue | read `surrogate_engine.py:636` vs `:588-606`, `:688-689` | **fails** — 216 810 MSCF over 15 yr contribute $0 |
| D2 §6.2-2: single NPV trajectory | grep for the second NPV at `surrogate_models.py:507-530` | **shadowed duplicate present** |

### 3.4 Measure the Koval/HCPVI behaviour directly

A 20-line measurement settles **CONF-02** and previews **CONF-01**:

```
sweep μ_o ∈ [0.5, 1, 5, 100] cP at the shipped default HCPVI = 7.69
sweep HCPVI ∈ [0.01 … 10] at fixed M
report RF(HCPVI) and RF(M)
```

Expected today: RF identical to 6 dp in μ_o (**CRIT-14**, `CONFIRMED`). Expected from the THMC cap:
$1-e^{-7.69/1.5}=0.9939$, i.e. **the cap will not change the answer** at the default operating point.

### 3.5 Stage 1 exit criterion

A dated run audit in
[`../audit/simulation_run_audits/`](../audit/simulation_run_audits/index.md) with a **Verdict** of
`PASSED` / `ACCEPTABLE WITH CONDITIONS` / `FLAGGED` / `FAILED`, stating which of D1–D8's quantitative
claims are **refuted**, **confirmed**, or **untestable as written**.

Register anything that turns out to be a **live code** defect via
`python -m audit --register new <ID> …` — never by hand.

---

## 4. Stage 2 — Decision gate (project-manager meeting, 1 session)

Bring the Stage 1 measurements. Decide **O1 / O2 / O3 / O4**. The decision inputs:

| Question | Where the answer comes from |
|---|---|
| Does the THMC physics actually beat the current surrogate on our cases? | Stage 1 §3.1, §3.2 |
| Is the 0D surrogate's known-defect list closed by THMC, or by smaller fixes? | Stage 1 §3.3, §3.4 vs. the live register |
| Is the HCPVI cap a fix or another clip? | **CONF-02** measurement |
| Where does economics live? | ✅ **already decided** — CONF-04 ruling |
| Mid-year or end-year discounting? | **CONF-62** — one-line measurement, re-baselines everything |
| Is a 1 000 BOPD clamp acceptable as a design constraint? | **CONF-07** |
| Is a 9 % EoS density error acceptable for the objective? | **CONF-49** |
| Is 1 370× GPU flash worth planning for, on what hardware? | **CONF-06** |
| What does a full rewrite cost, and who maintains two engines meanwhile? | Stage 3 scoping |

**Decision-record template** (write into [`../decisions/architecture_decisions.md`](../decisions/architecture_decisions.md) as an ADR):

```
ADR-NNN — 3D THMC adoption
Status:        PROPOSED | ACCEPTED | REJECTED | SUPERSEDED
Date:          DD-MM-YYYY
Outcome:       O1 ADOPT | O2 HARVEST | O3 PILOT | O4 REJECT
Blocking conflicts resolved:  CONF-02 ☐  CONF-03 ☐  CONF-58 ☐  CONF-59 ☐
  (CONF-01 downgraded to MATERIAL 07-10-2026; CONF-04 RESOLVED by ruling)
Stage-1 verdict:            <link to run audit>
Accepted residual risks:    CONF-nn (justification)
Rejected with reasons:      CONF-nn (justification)
Rollback condition:         <measurable trigger>
```

---

## 5. Stage 3 — If and only if O2 or O3 is chosen: scope the work

### 5.1 O2 (HARVEST) — physics contracts into the existing engine

Deliverables, in dependency order, each closing a **named** live finding:

| # | Deliverable | Closes | Risk |
|---|---|---|---|
| H1 | Component-wise mass balance gate per timestep | live: `recycled ≤ produced ≤ injected` holds only by construction | 🟢 Low |
| H2 | Saturation-closure gate $\sum S = 1$ | live **CRIT-17** (fails 27 % of steps) | 🟠 Medium |
| H3 | Adjoint gradients for NPV w.r.t. controls | optimiser throughput; requires a gradient-verification test | 🟠 Medium |
| H4 | Re-derive the Koval sweep with mobility-ratio sensitivity | live **CRIT-14**, **CRIT-15** — ⚠️ this is also the **entire fix for CONF-02**: the engineer's recommended $K_K = H\cdot E_v$ is already implemented at `analytical_models.py:176-193` | 🟠 Medium |
| H5 | Add hydrocarbon-gas revenue to `annual_rev`, **splitting sales gas from recycle CO₂** | live **CRIT-18** + **CONF-59** | 🟢 Low |
| H6 | Tiered mass-balance tolerance, defined by interface; **sourced numbers only** | **CONF-23** | 🟢 Low |
| H7 | Seeded, reproducible synthetic-geology generator | **CONF-43**; `GeostatisticalParams` already exists at 27.3 % usage | 🟢 Low |
| H8 | Level-6 adversarial tests (`t=0⁺`, phase appearance, percolation clogging) | absent entirely from the live suite | 🟡 Medium |
| **H9** | **Move economics off-core per the CONF-04 ruling — carrying CO₂ purchase, CO₂ recycle, storage credit and carbon tax across** | **CONF-04, CONF-58, CONF-59** | 🟠 Medium |
| **H10** | **Decide mid-year vs. end-year discounting; re-baseline every published NPV** | **CONF-62** (+4.88 % NPV) | 🟢 Low, but re-baselines results |
| **H11** | **Split invariant 12** — step penalty for solver non-convergence; barriers only for bounded physics, and only where no adjoint gradient is required | **CONF-60** | 🟡 Medium |

**H1, H2, H5, H6, H7, H10 are cheap and language-independent. Do them first regardless of the Rust
decision.** ⚠️ **H9 must not precede H5 / CONF-58 / CONF-59**, or the refactor will delete four working
cost terms and/or book recycled CO₂ as gas revenue.

### 5.2 O3 (PILOT) — one Rust vertical slice

Scope deliberately minimal, and chosen so it settles the open questions:

| Element | Scope | Why |
|---|---|---|
| Crate | `thmc-core`, one crate | answer CONF-14 (preconditioner), CONF-28 (DTO bus) |
| Discretisation | **2D single-phase FVM**, 5-point, on a flat `Vec<f64>` SoA slab | proves memory layout, zero-allocation claim, Rayon assembly |
| Nonlinear solve | Newton + **one named preconditioner** (e.g. block-Jacobi or ILU(0)) | proves D1 §10.3 + answers CONF-14 |
| Physics | Darcy single-phase + **component-wise mass balance** to `< 1e-12` | proves the Level-2 gate is achievable |
| IO | **one** format, fully specified, with a version field | answers CONF-30b |
| Gate | 2D **five-spot** with grid rotated **45°** | D5 §5.3 — the cheapest spatial verification available |

Explicitly **out** of scope: compositional flash, geochemistry, geomechanics, fractures, wellbore,
GPU, adjoint. If the 2D slice cannot reach `1e-12` mass balance on one core in one week, the
`10^{-12}` gate in D5 is aspirational and O1 should be rejected.

---

## 6. Evaluation metrics

Track these from Stage 1 onward. Every number must come with a `Status:` line citing a measurement.

| Metric | Baseline (measured 05-10-2026) | Target | Source |
|---|---|---|---|
| Predictive validity of the active `hybrid` path | **`NOT ESTABLISHED`** | established against ≥ 2 independent references | `phd_audit.md` §C |
| Live defects in the register | **73** (13 CRIT, 19 HIGH, 16 MED, 5 LOW) + 11 from round 2 | net −10 | `audit/scientific_flaws.md` |
| RF sensitivity to oil viscosity | **exactly flat** (identical to 6 dp, μ_o 0.5→100 cP) | non-degenerate | **CRIT-14** |
| Saturation closure $\sum S = 1$ | fails on **27 %** of timesteps | 100 % | **CRIT-17** |
| Leakage term in the carbon ledger | **identically 0** in every configuration | non-degenerate or explicitly zero with justification | **HIGH-23** |
| Mass balance per component | not enforced | `< 1e-12` (core) / defined tier | D5 §3.1 |
| SPE 5 delta | not recorded | ≤ 1.5 % | D5 §6.1 |
| CMG GEM delta | not recorded | ≤ 0.5 % | D5 §6.1 analogue |
| NPV cross-artefact agreement | unmeasured | ≤ $0.01 | D8 §4.2-2 |
| Test count / coverage | 335 passing, coverage not gated | gate added | **CONF-21** |
| Wall-clock per evaluation | unrecorded | recorded | — |

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Design set is mistaken for a description of shipped code | **High** — 6 documents state unperformed remediations in the past tense | **High** | this wiki section + `doc_inventory.md` + the 🔴 banner in [`README.md`](README.md) |
| The rewritten engine inherits the current defects | High | High | Stage 1 measures the claims **against the current engine first**; `conflict_and_gap_register.md` lists which of the 4 🔴 items are inherited defects (**CONF-02** ← CRIT-15, **CONF-11** ← CRIT-18, **CONF-33** ← CRIT-19) |
| Scope explosion (8 documents, 9 modules, 4 physical regimes) | **High** | High | O3 slices to 2D single-phase; everything else is deferred by construction |
| `petekIO` becomes a second, competing substrate | Medium | Medium | require a version field + a round-trip test before adoption (**CONF-30b**) |
| Two engines maintained in parallel | High if O1 chosen | High | Stage 2 must record a rollback condition in the ADR |
| Rust expertise unavailable in team | Medium | Medium | O2 needs no Rust; decide tooling before O1/O3 |
| Gas-sales revenue added to NPV changes every published result | **Certain** | Medium | H5 must re-baseline all historical results and mark them stale |

---

## 8. What is explicitly **not** being proposed

- No source code change in this document. The plan produces measurements and an ADR.
- No new finding written by hand. Findings go through `python -m audit --register new`.
- No claim that the design set is *wrong* because it is not implemented. It is a design; several of its
  ideas (component-wise mass balance, no penalty fudging, saturation positivity, Level-6 adversarial
  tests, adjoint gradients) are **better** than what the live engine does. The evaluation question is
  cost and correctness, not worth.
- No assumption that O1 is the goal. **O2 and O3 are on the table and O4 is a legitimate answer.**

---

## 9. Immediate next actions (for the project-manager meeting)

1. Read [`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md) first, then
   [`conflict_and_gap_register.md`](conflict_and_gap_register.md) §1 and §6 (index).
2. **Acknowledge the CONF-01 retraction.** The design set does *not* re-introduce SCI-FLAW-01; the sign
   is correct. The real defect is the relative-permeability closure. Anyone who has read the earlier
   version of this register must be told.
3. **Record the CONF-04 ruling as an ADR** in
   [`../decisions/architecture_decisions.md`](../decisions/architecture_decisions.md) — but **with
   CONF-58/59 attached**, so the refactor does not delete CO₂ costs or sell recycled gas.
4. Settle **CONF-62** (mid-year vs end-year discounting) — one line of code, but it re-baselines every
   published NPV, so decide it before, not after, any economic refactor.
5. Confirm **Stage 0** ownership: who edits D1–D8, and by when.
6. Confirm **Stage 1** is acceptable as "measure first, build later".
7. Choose **O2**, **O3** or **O4** as the default trajectory, with **O1** gated behind a Stage-2 ADR.
   Note that **H4 (fixing CRIT-14/CRIT-15) is both a live-engine defect fix and the whole of CONF-02** —
   it is the highest value-per-hour item in this plan regardless of the THMC decision.
8. Book the Stage-2 decision meeting after Stage 1 produces its run audit.
9. Register any live-code defect discovered in Stage 1 via the finding CLI, and let
   `python -m audit.continuity check` adjudicate.