# 3D THMC Simulator — Wiki Section

> [!WARNING]
> **🦀 This is the design source for the Rust 3D THMC engine, which is under construction.**
> 🔴 **No Rust engine code exists.** These pages are a normalised transcription of the 8 source documents
> plus the full conflict/adjudication trail. The **corrected specification** — what the engine must actually
> do — lives in [`../compositional/`](../compositional/README.md). ✅ The **shipped** Python surrogate is
> documented under [`../architecture/`](../architecture/overview.md); ⚠️ **do not read either section as
> describing the other.**

> [!CAUTION]
> **Nothing in this section describes existing code.**
> `3D_THMC_docs/` (8 Ukrainian-language design documents, added 07-10-2026, untracked in git) specifies a
> **Rust** 3D thermo-hydro-mechano-chemical compositional simulator that **does not exist in this repository**.
> Verified 07-10-2026: `Get-ChildItem -Recurse -Include *.rs, Cargo.toml` over the working tree returns
> **only `pyproject.toml`** — there is **no Rust crate, no `Cargo.toml`, no `.rs` file, no CI config**
> anywhere. Every acceptance criterion in the core spec is **unchecked `[ ]`**.
>
> This section is a **normalised, agent-readable transcription** of those design documents. It records
> *design intent and stated requirements only*. It is **not** a source of truth for engine behaviour.
> The source of truth for engine behaviour remains
> [`architecture/source_of_truth_map.md`](../architecture/source_of_truth_map.md) →
> **`core/engine_surrogate/`**.

---

## 1. What this section is for

| Question | Answer |
|---|---|
| What do the 8 documents in `3D_THMC_docs/` actually say? | [`doc_inventory.md`](doc_inventory.md), then the per-topic pages below |
| What architecture do they propose? | [`architecture_design.md`](architecture_design.md) |
| What numerics / solver / data structures? | [`solver_and_numerics.md`](solver_and_numerics.md) |
| What does the 6-level V&V framework require? | [`vv_testing_framework.md`](vv_testing_framework.md) |
| What are the satellite (off-core) tools? | [`satellite_toolkit.md`](satellite_toolkit.md) |
| What are the two addenda (water/asphaltene/hydrate, remediation/emergency)? | [`addenda.md`](addenda.md) |
| What is specified for fracturing & wellbore? | [`fractures_wellbore.md`](fractures_wellbore.md) |
| **Where does the design conflict with the live codebase and the finding register?** | **[`conflict_and_gap_register.md`](conflict_and_gap_register.md)** |
| **What did the reservoir engineer rule on those conflicts (07-10-2026)?** | **[`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md)** |
| How do we evaluate this with the project manager? | [`evaluation_plan.md`](evaluation_plan.md) |

---

## 2. Document set (8 files, all Ukrainian, all untracked)

| Wiki ID | Source file | Size | Content |
|---|---|---|---|
| **D1** | `3D THMC Reservoir Simulator Full Architecture Design Document.md` | 68 KB | Zero-Bloat Core, `petekIO`, corner-point grid, FVM/MPFA-O, WENO, Biot FEM, EOS + GPU flash, geochemistry, 5-state trapping, fracturing, faults, wells, adjoint gradients, PGS/GEP, economics |
| **D2** | `Технічна специфікація розробки ядра draft.md` | 43 KB | Rust core spec (draft): error taxonomy, `ConvergenceControl`, Newton, THMC coupling, DP/DP/MINC, Barton–Bandis, faults, EOR remediation formulas, **6 unchecked acceptance criteria** |
| **D3** | `Доповнення 1.md` | 38 KB | Produced-water re-injection (TSS/OiW), asphaltene AOP + deposition, gas hydrates; 8 Rust DTOs |
| **D4** | `Доповнення 2.md` | 37 KB | Formation-damage remediation, acidising, solvents, backwashing, hydrate dissolution, well-kill, SSSV, squeeze cementing; 2 Rust DTOs |
| **D5** | `Стратегія верифікації та валідації (V&V Testing Framework) для 3D THMC симулятора.md` | 57 KB | 6-level V&V stack, 22 declared tests, exact tolerances, 3 CI suites with wall-clock budgets |
| **D6** | `Специфікація автономного сателітного інструментарію та калькуляторів (Pre- & Post-Simulation Utilities Suite).md` | 59 KB | Pre/post simulators, DTO bus, PGS/SCAL/rock-typing, upscaling, DCA/RTA, Havlena–Odeh, GEP, DCF economics |
| **D7** | `Специфікація генерації синтетичних геологічних даних та інтеграції з сателітними калькуляторами.md` | 36 KB | Procedural geology (fBm, Voronoi, WFC, marching cubes, PGS), property cascade, FZI rock typing, DTO hand-off |
| **D8** | `Hydraulic Fracturing and Complex Wellbore Architecture Specification.md` | 39 KB | P3D/Planar-3D/DFN, proppant transport, EDFM coupling, casing/cement/perforation, drift-flux wellbore, NPV unification criteria |

> [!WARNING]
> **Every source document has broken section numbering.** All top-level headings render as `1.`
> (a markdown auto-numbering artefact); ordered lists restart mid-document. **Cite by Wiki ID + topic,
> never by "section N".** Full defect list in [`doc_inventory.md`](doc_inventory.md) §4.

---

## 3. Mandatory reading order for agents

```
1. THIS FILE                          — status, scope, no-code warning
2. conflict_and_gap_register.md       — read this BEFORE writing any THMC code.
                                        It is where the design contradicts the live engine.
2a. reservoir_engineer_ruling.md      — the 07-10-2026 adjudication. CONF-01 is RETRACTED;
                                        CONF-04 is RESOLVED; CONF-58/59/60/61/62 are new.
3. architecture_design.md             — the proposed architecture (Zero-Bloat Core, petekIO, FVM/MPFA-O)
4. solver_and_numerics.md             — Newton / timestep / linear algebra / AD / adjoint / Rust types
5. vv_testing_framework.md            — the acceptance gates (6 levels, tolerances, CI budgets)
6. satellite_toolkit.md, addenda.md, fractures_wellbore.md   — topic detail as needed
7. evaluation_plan.md                 — the staged plan + the decision log for the project manager
```

---

## 4. Hard rules for working in this area

1. **Do not cite this section as evidence of engine behaviour.** The active engine is
   `core/engine_surrogate/`. See `agent_wiki/README.md` invariant 1.
2. **Do not treat any D1–D8 number as validated.** Every tolerance, benchmark timing and
   "accuracy %" in these documents is an *assertion in a design document*, with **no** measurement
   behind it. D5 explicitly has **zero** supplied reference/expected values for any of its 22 tests.
3. **Do not hand-write findings.** If a THMC requirement is found to be violated in live code, register
   it via `python -m audit --register new <ID> …`, never by editing `audit/scientific_flaws.md`.
   See [`../development/finding_registry.md`](../development/finding_registry.md).
4. **Do not import Rust into this repository silently.** There is no toolchain, no crate, no CI.
   Any adoption decision is a project-manager decision — see [`evaluation_plan.md`](evaluation_plan.md) §1.
5. **A `RESOLVED` mark requires a measurement.** `python -m audit.continuity check <ID>` must report
   `CONFIRMED` first. See [`../development/continuity_gate.md`](../development/continuity_gate.md).

---

## 5. The three claims in D1–D8 that matter most for this repository

These are the items where the design document is *directly talking about the Python engine*, and where
its claims are **contradicted by the live code**. They are the reason
[`conflict_and_gap_register.md`](conflict_and_gap_register.md) exists.

> [!CAUTION]
> **Update 07-10-2026 — CONF-01 retracted.** A reservoir engineer reviewed the register and the review
> showed that **CONF-01 was wrong as I originally wrote it**. Full arithmetic and the replacement
> finding: [`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md) §2 and
> [`conflict_and_gap_register.md`](conflict_and_gap_register.md) §1. Do not quote the retracted claim.

### 5.1 The proposed gas fractional flow substitutes an ad-hoc ratio for relative permeability — *corrected*

D1 §6.1, D4 §3.3, D5 §1.1, D6 §4.1, D6 §4.5, D7 §4.4 and D8 §2.4 all state

$$f_g(t) = \frac{1}{1 + \left(\frac{1 - S_w(t) - S_g(t)}{S_g(t) - S_{gc}}\right)\cdot\frac{\mu_g}{\mu_o}}$$

**The sign is correct.** The classical form is
$f_g = \frac{\lambda_g}{\lambda_g+\lambda_o} = \frac{1}{1+\frac{k_{ro}}{k_{rg}}\frac{\mu_g}{\mu_o}} = \frac{M}{1+M}$ —
same structure, same monotonicity. Measured over $\mu_g/\mu_o \in [0.1, 10]$ the design and classical
forms agree to **4 decimal places** at every point. **SCI-FLAW-01 is not re-introduced**; the shipped
Koval form at `core/engine_surrogate/analytical_models.py:176-193` and the design form agree.

**The defect is the closure.** $\frac{S_o}{S_g-S_{gc}}$ stands in for $k_{ro}/k_{rg}$ but is not one:

| # | Defect | Consequence |
|---|---|---|
| a | **No $S_{or}$** — measured $f_g = 0.800$ at $S_o = S_{or}$, never 1.0 until $S_o = 0$ | Cannot represent the depleted state; $S_{or}$ sets the terminal gas cut |
| b | **Linear, not Corey** — should be $(S_o-S_{or})^{n_o}/(S_g-S_{gc})^{n_g}$, i.e. separate exponents | Wrong curvature, wrong transition width |
| c | **No water term** — three-phase needs $\lambda_g/(\lambda_g+\lambda_o+\lambda_w)$ | Contradicts D1 §6.1's own Stone I/II claim; interacts with live **CRIT-17** |
| d | **Unguarded at $S_g \to S_{gc}$** — uses $1/(S_g-S_{gc})$ not $(S_g-S_{gc})^{-n_g}$ | Right direction, wrong rate, no numerical guard |

> **Method note, now mandatory repo practice:** three conventions of $M$ were in play in this exchange
> ($\lambda_g/\lambda_o$, $\mu_g/\mu_o$, $\mu_o/\mu_g$) and the original error survived because the test
> used a **single point** where the formulas coincide. Every "$f$ rises/falls with $M$" claim must now
> state the definition of $M$ on the same line **and** be verified by a sweep. See ruling §4.

### 5.2 The proposed recovery cap is inert at the shipped default HCPVI

D1 §6.1, D2 §6.1, D4 §3.3, D6 §4.1, D7 §4.4 all bound recovery by

$$RF(t) \le RF_{\text{ult}}(\bar{P}_{\text{res,eff}})\cdot\left(1 - e^{-\text{HCPVI}/\tau}\right),\qquad \tau = 1.5$$

At the default HCPVI of this repository (**7.69**, per `agent_wiki/README.md` invariant 6 / CRIT-15):
$1 - e^{-7.69/1.5} = 0.9939$. The cap is therefore **inactive by 99.4 %** at the default operating
point — it constrains nothing exactly where the optimiser operates. This is the same *structural*
failure mode as **CRIT-15** (default HCPVI pins the Koval sweep at its `0.95` clip for every mobility
ratio). The cap is also **dimensionally dimensionless-by-assertion**: `τ = 1.5` has no units, no
provenance and no sensitivity study in any of the 8 documents. Tracked as **CONF-02**.

### 5.3 The documents claim remediation of the Python engine that has not happened

D1 §10.3, D2 §6.1, D4, D6 §1.2, D8 §4.2 all state that artificial penalty barriers
(`result *= breakthrough_impact`, `-1.0 × 10^{12}`) have been **"fully removed"** from the simulator
code and replaced by natural NPV reduction via gas-recycle cost.

In the live repository, `FAILURE_PENALTY = -1e12` is still a **documented invariant**
(`agent_wiki/README.md` invariant 12, "Test Suite Health & Zero Silent Swallowing"). The documents
therefore describe an **intended** remediation state, not the shipped state, while citing Python
symbols (`surrogate_engine.py`, `surrogate_models.py`, `run_exporter.py`, `results["npv"]`,
`cash_flows_yearly.csv`) that **do** exist — and one Python test file,
`tests/test_surrogate_engine.py`, that **does not exist** (`Test-Path` → `False`). Tracked as **CONF-03**.

### 5.4 Economics is exiled from the core and given an in-core adjoint — **adjudicated 07-10-2026**

D1 §1.1 exiles NPV/CAPEX to the satellite suite; D1 §10.2 computes the NPV adjoint gradient inside the
core. **The reservoir engineer's ruling is adopted**: the core owns conservation equations and publishes
**production adjoints** $\partial q_i(t)/\partial\mathbf{x}$; the satellite owns the economic vector
$(P_o, P_g, C_w, \text{CAPEX}, r)$ and chains prices onto those adjoints to obtain $\partial NPV/\partial x$.
This is the only boundary under which both D1 §1.1 and D1 §10.2 are true.

> ⚠️ **Do not adopt the ruling's NPV expression verbatim.** Two new blocking conflicts came out of it:
> **CONF-58** — it drops CO₂ purchase cost, CO₂ recycle cost, storage credit and carbon tax on leakage,
> all four of which the live engine already computes at `surrogate_engine.py:624-646`; **CONF-59** —
> an unqualified $q_gP_g$ would book **recycled CO₂ as gas revenue**. The ruling is *"prices off-core"*,
> not *"fewer prices off-core"*. Tracked as **CONF-04 (resolved)** with **CONF-58/59 (new)**.

---

## 6. Page index

| Page | Purpose |
|---|---|
| [`doc_inventory.md`](doc_inventory.md) | Per-document metadata, section maps, formatting defects, what is absent |
| [`architecture_design.md`](architecture_design.md) | D1 — Zero-Bloat Core, `petekIO`, grid, discretisation, thermodynamics, geochemistry, trapping, fractures, faults, wells |
| [`solver_and_numerics.md`](solver_and_numerics.md) | D1 §10 + D2 — Newton, adaptive timestep, linear solvers, hyper-dual AD, adjoint gradients, all Rust type definitions |
| [`vv_testing_framework.md`](vv_testing_framework.md) | D5 — 6 levels, 22 tests, exact tolerances, CI budgets, gaps |
| [`satellite_toolkit.md`](satellite_toolkit.md) | D6 + D7 — pre/post calculators, DTO schemas, PGS, GEP, synthetic geology, economics |
| [`addenda.md`](addenda.md) | D3 + D4 — PWRI, asphaltenes, hydrates, remediation, well-kill, SSSV, squeeze cementing |
| [`fractures_wellbore.md`](fractures_wellbore.md) | D8 — P3D/Planar-3D/DFN, proppant, EDFM, casing/cement/perforation, drift-flux, NPV unification |
| [`conflict_and_gap_register.md`](conflict_and_gap_register.md) | **CONF-01…CONF-62** — design vs. live code vs. finding register. ⚠️ CONF-01 retracted 07-10-2026 |
| [`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md) | **Adjudication of 07-10-2026** — what was accepted, what was rejected, and the notation discipline it forced |
| [`evaluation_plan.md`](evaluation_plan.md) | Staged evaluation plan and decision log for the project manager |