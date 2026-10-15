---
name: agent-wiki
description: Mandatory entry point and gatekeeper for ALL tasks in this repository. Consult this skill and agent_wiki/ before performing any code edits, searches, investigations, or refactoring in CO2 EOR Optimizer.
---

# Agent Wiki Mandatory Gatekeeper

## 🛑 Action Requirement Before Doing Any Work

Before running search tools (`grep_search`), terminal commands (`run_command`), or editing any source code:
1. **First Tool Call**: You MUST read [`agent_wiki/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/README.md) or the specific topic document in [`agent_wiki/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki) relevant to the task.
2. **Confirm Active Engine**: Verify [`agent_wiki/architecture/source_of_truth_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/source_of_truth_map.md). Remember that 100% of simulation evaluations route strictly to `core/engine_surrogate/`. Do NOT modify `core/Phys_engine_full/`, `compositional_engine/`, or `unified_engine/` expecting them to affect optimization runs.
3. **Verify Physical Invariants & Traps**:
   - Check [`agent_wiki/development/common_pitfalls.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/common_pitfalls.md) (traps with unit conversions, NumPy 2.0, mass balance, Vogel IPR).
   - Check [`agent_wiki/development/safe_modification_rules.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/safe_modification_rules.md) for immutable invariants.
4. **Log Simulation Run Audits**: Whenever running or evaluating simulation runs, parameter sweeps, or benchmarks, create a date-stamped subfolder under [`agent_wiki/audit/simulation_run_audits/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) (`DD-MM-YYYY_<run_name>/`), save diagnostic artifacts, author `audit.md` (verdict, proposal, relevant files), and register it in `index.md`.

## Topic Navigation Map

| Topic | Relevant Wiki Document |
| :--- | :--- |
| **Wiki Overview & Reading Order** | [`agent_wiki/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/README.md) |
| **Active vs Dormant Engines** | [`agent_wiki/architecture/source_of_truth_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/source_of_truth_map.md) |
| **Simulation Run Audits** | [`agent_wiki/audit/simulation_run_audits/index.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) |
| **Common Pitfalls & Gotchas** | [`agent_wiki/development/common_pitfalls.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/common_pitfalls.md) |
| **Safe Modification Rules** | [`agent_wiki/development/safe_modification_rules.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/safe_modification_rules.md) |
| **Risk Classification** | [`agent_wiki/development/change_safety_matrix.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/change_safety_matrix.md) |
| **Physics, EOR & Reservoir Models** | [`agent_wiki/physics/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/physics/) |
| **Unit Definitions & Conversions** | [`agent_wiki/data/units.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/data/units.md) |
| **Known Limitations & Envelopes** | [`agent_wiki/validation/known_limitations.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/validation/known_limitations.md) |
| **Agent Skills Knowledge Base** | [`agent_wiki/development/agent_skills.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/agent_skills.md) |
| **3D THMC Design Set (design intent, NO code)** | [`agent_wiki/thmc/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/thmc/README.md) |
| **3D THMC — conflicts vs. live code (READ FIRST)** | [`agent_wiki/thmc/conflict_and_gap_register.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/thmc/conflict_and_gap_register.md) |
| **3D THMC — engineer ruling 07-10-2026 (CONF-01 retracted)** | [`agent_wiki/thmc/reservoir_engineer_ruling.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/thmc/reservoir_engineer_ruling.md) |
| 🔴 **COMPOSITIONAL ENGINE — CURRENT WORKSTREAM** | [`agent_wiki/compositional/README.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/README.md) |
| 🔵 **Compositional — APPROVED output schema (9 domains)** | [`agent_wiki/compositional/output_schema.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/output_schema.md) |
| 🔵 **Compositional — spec corrections log (wiki = authority)** | [`agent_wiki/compositional/spec_corrections_log.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/spec_corrections_log.md) |
| **Compositional — build plan M0→M6** | [`agent_wiki/compositional/build_plan.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/build_plan.md) |
| **Compositional — separation doctrine (no coupling)** | [`agent_wiki/compositional/separation_doctrine.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/separation_doctrine.md) |
| **Compositional — separate flaw register (`COMP-nn`)** | [`agent_wiki/compositional/register_spec.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/register_spec.md) |

## 🛑 Special Rule — Compositional Engine (current active workstream)

A **full-physics 3D compositional simulator in Rust** is being built at `crates/compositional/`,
because **no available simulator covers the full spectrum of this project's tasks**. *(An `EngineFactory`
and a Python compositional engine were built and then removed — see invariant §6.)*

1. 🔴 **Six MANDATORY invariants** —
   [`compositional/engine_invariants.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/engine_invariants.md):
   - **INV-1 — fail loudly, never fall back.** Every error returns a typed error and **stops**. No
     surrogate fallback, no partial result, no relaxed tolerance. *Impossible by design.*
   - **INV-2 — runs only when the user starts it.** Never invoked from an optimisation loop, sweep or
     background path.
   - **INV-3 — simulation only.** No NPV, costs, discounting or prices. A separate economic engine owns
     field development.
   - **INV-4 — output is training-ready.** Labelled $(x,y)$ pairs, full fields, versioned schema,
     provenance. The engine also **trains a per-reservoir neural surrogate** (not yet built).
   - **INV-6 — capability declaration vs. failure.** `NOT_IMPLEMENTED` ≠ `FAILED`. A run declares which
     of the nine output domains it populated.
2. **Two engines, two jobs — and they COMPLEMENT** (decided 08-10-2026). The **surrogate**
   (`core/engine_surrogate/`) screens and ranks at $10^3$–$10^5$ evaluations per run; the **compositional
   engine** evaluates a few candidates with solved physics. The surrogate is **not** retired.
3. 🔵 **Output schema APPROVED 08-10-2026** — [`compositional/output_schema.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/output_schema.md).
   9 domains, **116 float fields per cell per timestep**, three temporal resolutions. 🔴 **This is a
   TARGET specification — no engine exists; current state is M0.** 🔴 **The engine emits all raw values;
   satellite tools interpret and format them.** ⚠️ **Two-output contract: spatial → HDF5, time-series →
   Parquet/DuckDB. The satellite economic engine NEVER reads HDF5.** Ruling of 08-10-2026 applied:
   economics purged, Koval unified, `ε_p` out of the elastic baseline, `Fe²⁺` added, skin floor
   corrected. ⚠️ Carries **CONF-63…68**.
4. 🔴 **Python post-mortem** — [`compositional/python_attempt_postmortem.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/python_attempt_postmortem.md).
   The first attempt failed on **six numerical-method defects, not on execution speed** — all
   language-independent. **Read before M2.** Yields C-33…C-37 (Heidemann–Khalil critical solver,
   exponential soft-start, well control modes in the global Jacobian, permeability-collapse selection
   rule). ⚠️ It repeats the **retracted** "$f_g$ sign error" reason — the *action* is right, the *reason*
   is not.
5. 🔵 **Spec authority:** `3D_THMC_docs/` are **immutable references**; the **wiki carries the corrected
   spec** and states **current** state. Every deviation logged with source and line in
   [`compositional/spec_corrections_log.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/spec_corrections_log.md)
   (**46 corrections**, incl. 2 of my own framings withdrawn by measurement).
6. **Phase model** — [`compositional/vision_and_phases.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/vision_and_phases.md):
   **P1 DEVELOP** → **P2 INTEGRATE** → **P3 COUPLE**. ⚠️ **UI/UX is out of scope** until the engine is fully
   built and tested. **Python is left as-is** — legacy findings remain the Python audit. ✅ The two engines
   **complement**; the surrogate is **not** retired.
7. ⚠️ **Scope re-baselined 08-10-2026 to the FULL 3D THMC programme** (coupled thermal + geomechanics +
   geochemistry, fractures/EDFM, wellbore, Schwarz, GPU, adjoint, ES-MDA). The earlier "M7+ deferred"
   framing is **void**. **Resolving scope has not resolved specification** — CONF-13/14/25/51/16/18/47/08
   remain open.
8. **During P1 only:** never import from `core/`, `evaluation/`, `validation/`, `utils/`, `ui/`.
   **These gates retire by ADR at P2** — separation is a development discipline, not an architecture.
9. **Own flaw register** at `audit_comp/`, namespace `COMP-nn`. ⚠️ `audit/registry.py:114`
   `LOCATION_RE` accepts **`.py` only** — the existing register *cannot* record a Rust finding.
10. **Milestones M0 → M6**, each gated by a measured value:
    [`compositional/build_plan.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/build_plan.md).
    **M1 references verified** (Abudour 2014, Baled 2012 — both DOIs confirmed). Spec gaps:
    [`compositional/spec_defects.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/spec_defects.md).
11. 🔵 **Data layer: PostgreSQL + `pgvector`**, with a layered open-file substrate for simulation output —
    [`compositional/data_architecture.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/compositional/data_architecture.md).
    ✅ Standard open formats ⇒ the satellite reads them with `h5py` (already a dependency) — **no FFI, no
    Rust in the Python process**. `pyarrow`/`duckdb` are **leftovers to be added when the satellites are
    built**, not a P1 blocker.
12. ⚠️ **`agent_wiki/development/extension_points.md` is unbuildable** — `config/default_config.json`,
    `core/engine_factory.py`, `core/models/`, `core/surrogate_engine.py`, `core/analytical_models.py`
    **none exist** (`EngineFactory` was built and removed). Verify every path with `Test-Path`.
13. **Rust toolchain installation is deferred by decision** until after documentation, at the start of
    development — crate choices are provisional.
14. **The active engine for the existing application is still `core/engine_surrogate/`** (invariant 2).

## 🛑 Special Rule — 3D THMC Design Set

`3D_THMC_docs/` and `agent_wiki/thmc/` describe a **Rust** 3D THMC compositional simulator that
**does not exist** in this repository. Verified 07-10-2026: `*.rs` → 0, `Cargo.toml` → 0, CI → 0.
The active engine is still `core/engine_surrogate/` (invariant 2 above).

Before acting on anything in `agent_wiki/thmc/`:
1. Read `thmc/conflict_and_gap_register.md` and `thmc/reservoir_engineer_ruling.md`. ⚠️ **CONF-01 was
   RETRACTED 07-10-2026** — the proposed gas fractional flow has the **correct sign**; the original
   "$df_g/dM<0$" claim was a substitution error on my part, confirmed-wrong by measurement, and the
   reservoir engineer's endorsement of it repeated the same error. The real defect is the *closure*
   (no $S_{or}$, linear not Corey, no water term). **Do not repeat the retracted claim.**
   Still blocking: **CONF-02** (HCPVI cap inert at the default), **CONF-03** (documents claim
   unperformed remediations), **CONF-04** (now adjudicated).
2. Never cite a THMC tolerance, accuracy % or benchmark timing as validated. The design set's own V&V
   framework supplies **zero** absolute reference values for any of its 22 declared tests.
3. Follow `thmc/evaluation_plan.md` — **measure first** (Stage 1 converts the documents' acceptance
   criteria into measurements against the *existing* engine), **decide second** (Stage 2 ADR:
   ADOPT / HARVEST / PILOT / REJECT), **build last**.
