# CO₂ EOR Optimizer — Technical Wiki

> [!WARNING]
> **This repository is under massive active development.**
>
> **Two engines, in very different states. Never read one as a description of the other.**

| | 🐍 **Python engine** | 🦀 **Rust engine** |
|---|---|---|
| **What it is** | A **physics-informed surrogate** — a deliberately reduced-order, curve-driven reservoir model | A **full 3D THMC compositional reservoir simulator** — transport + phase equilibrium + geomechanics + geochemistry |
| **Status** | ✅ **Shipped, active.** Describes running code today | 🔴 **Under construction. No engine code exists yet** |
| **Language** | Python 3 + PyQt6 | Rust (toolchain installs at the start of M0) |
| **Role** | Screens and ranks $10^3$–$10^5$ candidates per optimisation run | Evaluates a handful of candidates per run with solved physics, **and generates the training data** for a future neural surrogate |
| **Documents here** | [architecture](architecture/overview.md), [physics](physics/reservoir_model.md), [audit](audit/technical_debt.md) | [compositional/](compositional/README.md) + [thmc/](thmc/README.md) |
| **Named** | "the surrogate" | "the **compositional engine**" = "the **3D THMC engine**" — **the same artefact**, two names |

!!! danger "Read this before citing anything about the Rust engine"
    Every page under [`compositional/`](compositional/README.md) and [`thmc/`](thmc/README.md) is a
    **specification, a correction, or an audit trail** — never a report of running software. 🔴 A tolerance,
    a milestone, or a "verified" claim on those pages describes **what the engine must do**, not what it
    does. 📌 The naming trap: *"compositional"* is a **legacy of an earlier narrower scope** when the THMC
    layers sat in a deferred bucket. 🔴 **Reading it literally — as thermodynamics and multiphase flow only —
    means shipping roughly one third of approved scope.** See
    [vision_and_phases §5.4](compositional/vision_and_phases.md).

This documentation is designed for **reservoir engineers, AI pair-programmers, scientific computing
auditors, and software architects**. It provides unambiguous, machine-readable, and scientifically grounded
guidance to enable rapid comprehension, prevent accidental regression of scientific invariants, and
eliminate modification risk.

---

## 🧭 Documentation Structure

### 🐍 Python engine — shipped

| Section | Purpose |
|---|---|
| [**Architecture**](architecture/overview.md) | Engine routing, module map, execution flow, dependency graph, [**Shared Earth Workstation**](architecture/shared_earth_workstation.md), [**Subsurface Studio**](architecture/subsurface_studio_workbench.md), [**Model QA/QC & Self-Assurance**](architecture/model_qa_qc_and_self_assurance.md) |
| [**Physics Models**](physics/reservoir_model.md) | Reservoir geometry, PVT, CO₂ properties, displacement, Koval, Todd-Longstaff |
| [**Data & Parameters**](data/inputs.md) | Input/output schemas, parameter registry, field unit definitions |
| [**Development Guide**](development/common_pitfalls.md) | Change safety matrix, common pitfalls, safe modification rules |
| [**Audit Reports**](audit/technical_debt.md) | Dead code, hardcoded values, [**data management**](audit/data_management_audit.md), [**optimization widget**](audit/optimization_widget_audit.md), [**dialogs, models, utils & widgets**](audit/dialogs_models_utils_widgets_audit.md), [**simulation run audits**](audit/simulation_run_audits/index.md), fallbacks, suspicious logic, [**resolved archive**](audit/resolved_issues.md) |
| [**Agent Showcase Checklist**](audit/agent_checklist.md) | **Automated Master Pre-Flight**: verified flaws, parameter provenances, safe harbor rules, and static diagnostic anchors |
| [**Continuity Gate**](development/continuity_gate.md) | **Autouse**: `python -m audit.continuity check` re-measures every `RESOLVED` claim, checks wiki/code drift, and enforces 1-commit-1-issue |
| [**Finding Registry**](development/finding_registry.md) | **Required**: `python -m audit --register new\|validate\|issue` — the flaw register is machine-generated; never hand-write a finding |
| [**2026-10-04 Independent Audit**](../res_audit.md) | [Software quality & anti-patterns](../res_audit.md) · [Scientific/physics audit (`phd_audit.md`)](../phd_audit.md) · [**Flaw register — 73 findings**](../audit/scientific_flaws.md) · [Parameter provenance — 112 rows](../audit/parameter_provenance.csv) |
| [**2026-10-05 Round-2 Audit**](audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md) | **Verdict `FLAGGED`** — adversarial verification of the uncommitted 05-10-2026 remediation; **11 new findings**, 8 "RESOLVED" marks reversed or downgraded |
| [**Verification**](verification/verification_strategy.md) | 7-level V&V hierarchy, conservation tests, convergence studies |
| [**Validation**](validation/benchmarks.md) | SPE 5, CMG GEM reference benchmarks |
| [**Decisions**](decisions/architecture_decisions.md) | Architecture & scientific rationale records (ADRs) |
| [**Code Reference**](code/inventory.md) | Class, function and module inventories |

### 🦀 Rust engine — in development

🔴 **Nothing in this group is running code.** Start at
[**compositional/README.md**](compositional/README.md).

| Section | Purpose |
|---|---|
| 🔄 [**Agent Handover**](compositional/HANDOVER.md) | **ROLLOVER GATE — read in full on compaction, session resume, or agent handoff.** ~9 KB (160 lines) by design. Carries current state, the seven invariants, and **§3 RETIRED — do not re-derive** (ten withdrawn claims). 🔴 The Rust-engine docs total **1,087 KB** and cannot be re-read linearly |
| [**Vision & Phases**](compositional/vision_and_phases.md) | P1 develop → P2 integrate → P3 couple; all decisions taken |
| [**Build Plan M0→M7**](compositional/build_plan.md) | Milestones with **measured** gates, incl. **M4.5 integration slice** and the **M7b** body |
| [**Mandatory Invariants**](compositional/engine_invariants.md) | 🔴 **INV-1…INV-7 + ten standing CI rules.** Start at §0 |
| [**Output Schema**](compositional/output_schema.md) | 🔵 **APPROVED** — 9 domains, 116 fields/cell/timestep |
| [**Data Architecture**](compositional/data_architecture.md) | 🔵 **DECIDED** — PostgreSQL + `pgvector` |
| [**Separation Doctrine**](compositional/separation_doctrine.md) | P1 discipline + the **C-7** coherence gate |
| [**Separate Register**](compositional/register_spec.md) | **INV-5** — `COMP-nn` namespace |

### 🔴 Problems, gaps & research

| Section | Purpose |
|---|---|
| [**Conflict & Gap Register**](thmc/conflict_and_gap_register.md) | **68 conflicts** across the 8 design docs — the authoritative status |
| [**Reservoir Engineer Ruling**](thmc/reservoir_engineer_ruling.md) | The 07-10-2026 adjudication, incl. the CONF-01 retraction |
| [**Spec Corrections Log**](compositional/spec_corrections_log.md) | **C-1…C-237** across 38 rulings — every correction and retraction |
| [**Spec Defects**](compositional/spec_defects.md) | `CONF-*` items indexed by milestone |
| [**📚 Literature TODO**](compositional/literature_todo.md) | **L-1…L-9** — outstanding research, with the acceptance standard. ✅ **L-1, L-7, L-8 closed** — ✅ the full Karakas & Tariq (1991) paper was obtained and **all five tables transcribed**, which closed **CONF-31's coefficient defect** and resolved the phasing dispute as **parameter-dependent** |
| [**Python Attempt Post-mortem**](compositional/python_attempt_postmortem.md) | Why the first attempt failed — six language-independent causes |
| [**Data-Model Gap**](compositional/data_model_gap.md) | Why `core/data_models.py` cannot express a full simulation |

### 📐 Design source

| Section | Purpose |
|---|---|
| [**3D THMC Simulator**](thmc/README.md) | **Design intent only** — 8 Ukrainian design docs (`3D_THMC_docs/`), normalised for agents: architecture, solver, 6-level V&V, satellite toolkit, addenda, fracturing/wellbore |
| [**Doc Inventory**](thmc/doc_inventory.md) | What each source document contains |

> [!CAUTION]
> **`thmc/` describes a simulator that does not exist.** `3D_THMC_docs/` (8 Ukrainian design documents,
> added 07-10-2026, untracked) specifies a **Rust** 3D THMC compositional core. Verified 07-10-2026:
> the working tree contains **no `.rs` file, no `Cargo.toml`, and no CI** — only `pyproject.toml`.
> Every acceptance criterion in the core spec is **unchecked `[ ]`**.
>
> **The active engine remains `core/engine_surrogate/`** (invariant 1 below). Nothing in `thmc/` may be
> cited as evidence of engine behaviour.
>
> **Read [`agent_wiki/thmc/conflict_and_gap_register.md`](thmc/conflict_and_gap_register.md) before
> working in this area.** Blocking conflicts are recorded there. Two carry a **2026-10-07 correction**:
> **CONF-01 was retracted** — the proposed fractional flow has the **correct sign**; the defect is in
> its *closure* (no $S_{or}$, no Corey exponents, no water term). Do not repeat the original claim.
> **CONF-02** the proposed HCPVI recovery cap is **99.4 % inert** at the shipped default HCPVI 7.69
> (structurally identical to **CRIT-15**); **CONF-03** six documents report penalty removal and NPV
> unification as *accomplished* when `FAILURE_PENALTY = -10^{12}` is still a live invariant and
> `tests/test_surrogate_engine.py` **does not exist**; **CONF-04** economics is simultaneously exiled
> from the core and given an in-core adjoint gradient — **now adjudicated** in
> [`thmc/reservoir_engineer_ruling.md`](thmc/reservoir_engineer_ruling.md).

> [!WARNING]
> **Audit status — read before trusting any "RESOLVED" mark below.**
>
> **Round 2 (`05-10-2026`, verdict `FLAGGED`)** adversarially re-audited the uncommitted remediation
> of the 17 findings the 04-10-2026 round had marked RESOLVED. Verdict: **11 CONFIRMED · 4 PARTIALLY
> RESOLVED · 1 REGRESSED (CRIT-12) · 1 CONFIRMED-but-inert (CRIT-06)**, and **11 new findings** were
> registered (CRIT-14…CRIT-21, HIGH-20/21/23/24/25/26, MED-17…MED-22) — see
> [`audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md`](audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md)
> and issues [#7](https://github.com/fgfalll/WAG_optimisation/issues/7)–[#15](https://github.com/fgfalll/WAG_optimisation/issues/15).
>
> The **live, most consequential defects** are now:
> - **CRIT-14** — a `mobility_ratio` override makes recovery **exactly independent of oil viscosity**
>   in all three recovery models (measured: RF identical to 6 decimals for μ_o = 0.5 → 100 cP).
> - **CRIT-15** — the default HCPVI (7.69) pins the Koval sweep at its 0.95 clip for **every**
>   mobility ratio.
> - **CRIT-17** — the CRIT-12 "fix" broke saturation closure: `S_o + S_w > 1` on **27 %** of timesteps.
> - **HIGH-23** — leakage is *identically zero* while the storage credit is paid on **leakage-blind**
>   stored mass: **leaked CO₂ would be paid for, not charged for**.
> - **CRIT-19 / CRIT-20** — the previously inert `gravity_factor` gene is now an active ±20 % fudge on
>   recovery, and the miscibility weight no longer depends on composition.
>
> Genuinely fixed and worth keeping: `B_g` constant (CRIT-04), Z-factor (CRIT-05), HCPVI
> dimensionality (CRIT-01), the NPV double-evaluation defect (CRIT-02), the immiscible-limb gradient
> (CRIT-07), and the `rf_max_physical` OOIP normalisation (HIGH-01).
>
> The suite reports **335 passed / 0 failed** while every defect above is live — see HIGH-18: green is
> not evidence. **Predictive validity for the active `hybrid` path remains `NOT ESTABLISHED`.**
>
> **Prior round (`04-10-2026`).** An independent four-phase audit registered **53 findings (13 CRITICAL,
> 19 HIGH, 16 MEDIUM, 5 LOW)** in [`audit/scientific_flaws.md`](../audit/scientific_flaws.md) and issued a
> **predictive-validity verdict of `NOT ESTABLISHED`** ([`phd_audit.md`](../phd_audit.md) §C). Read the register
> before treating any wiki claim below as verified: several invariants here were **not enforced by code**
> (they hold only by construction), and `verification/test_matrix.md` listed six tests that do not exist
> (MED-16).

---

## ⚡ Critical Architectural Invariants

Before reading or modifying any file in this repository, keep the following **core facts** in mind:

1. **The Single Active Simulation Engine is `core/engine_surrogate` (Intermediate Physics-Informed Simulator)**:
   The name "Surrogate" signifies an **intermediate, physics-informed reduced-order reservoir simulator** positioned between full 3D multi-block compositional numerical solvers and classical 0D material balance. It operates as a general reservoir simulator embedded with specialized CO₂ EOR expertise (solvent dissolution, oil swelling, viscosity reduction, Koval viscous fingering, Todd-Longstaff partial miscibility, and EPA Class VI geomechanical integrity). All legacy engines (`compositional_engine`, `unified_engine`, `engine_simple`) and the intermediate `EngineFactory` are **absent from the working tree** — `deprecated/`, `core/unified_engine/`, `compositional_engine/` and `core/Phys_engine_full/` all return `Test-Path = False` (corrected 2026-10-04: earlier text claimed they had been "relocated into `deprecated/`"; no such directory exists — see MED-16). All simulation evaluations route directly to `SurrogateEngineWrapper` (`core/engine_surrogate/surrogate_engine.py`).

2. **`FastProfileGenerator` is the Single Source of Truth for Profiles**:
   All production, injection, and rate profiles are synthesized via `core/engine_surrogate/profile_generator_fast.py`. `core/simulation/profile_generator.py` is a deprecated re-export shim (`from core.engine_surrogate.profile_generator_fast import FastProfileGenerator`). ⚠️ `core/simulation/injection_schemes.py`, cited here previously, **does not exist** (grep → 0 hits; corrected 2026-10-04).

3. **NPV and CO₂ Accounting are Engine-Owned** (corrected 2026-10-05 — NPV moved out of `surrogate_models.py`):
   - **CO₂ purchased vs recycled** is computed *inline* inside `SurrogateEngine.evaluate_scenario()`
     (`core/engine_surrogate/surrogate_engine.py:615-620` annual make-up, `:660-670` rate and
     cumulative profiles).
   - **NPV** is now computed *inline* inside `SurrogateEngine.evaluate_scenario()`
     (`core/engine_surrogate/surrogate_engine.py:622-651`) from the simulated annual streams, and
     republished as the `npv` profile key. ⚠️ **CRIT-02 is genuinely fixed** — measured
     `cumulative_oil_stb == OOIP × recovery_factor` to `5.8e-10` STB, and
     `Σ(oil_profile·Δt) == Σ(annual_oil_stb)` exactly. The older inline NPV in
     `surrogate_models.py:507-530` still exists but is now **shadowed**.
   - ⚠️ **The cash flow is truncated (CRIT-18).** `annual_rev` is oil + storage credit only:
     `annual_hc_gas_mscf` is allocated, accumulated and published but never referenced in the
     revenue term, so 216 810 MSCF/15 yr of gas sales contribute $0. Since `npv` is the primary
     optimisation objective, the optimiser ranks on a partial model.
   - The wrapper `core/objectives/wrapper.py:ObjectiveFunctions._calculate_objective_functions()`
     is a **consumer** (reads `profiles["npv"]`, then applies containment/remediation penalties).
   - ⚠️ There is **no** method named `_calculate_engine_npv`, `_calculate_co2_purchased_recycled`,
     `_solve_pressure_ode` or `_calculate_pressure_profile` anywhere in the repository (grep: 0 hits).
     `SurrogateEngine` exposes only `evaluate_scenario`, `_build_params_dict`, `_error_result`,
     `get_performance_stats`, `reset_performance_stats`.

4. **Tank Pressure Uses Coupled Darcy/Vogel IPR & Damped Material Balance**:
   The legacy `solve_ivp` pressure ODE has been superseded by an explicit, physics-based deliverability and material balance model in `surrogate_engine.py`:
   - Gas injection converted via dynamic formation volume factor $B_{g,\text{dynamic}}$ to reservoir barrels per day (RB/d).
   - Dynamic Koval fractional flow with Todd-Longstaff effective mobility ratio $M_e$.
   - Deliverability bounded by producer drawdown ($J_{\text{prod}}$) and injector geomechanical limits ($J_{\text{inj}}$).
   - Pressure increment: $dP = (q_{\text{net,IPR}} \cdot \Delta t) / (V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t)$.
   - Dual-pressure Voidage Replacement Ratio (VRR) tracking local injection vs production reservoir volumes.

5. **Thermodynamic Rigor via Solvent-Extended PVT (`core/engine_surrogate/pvt_state.py`)**:
   Transport saturation ($S_g$) is decoupled from composition ($x_{\text{CO2}}, y_{\text{CO2}}$). Fluid properties ($B_o, \mu_o, S_F, B_{\text{CO2}}$) evolve with dissolved solvent concentration $x_{\text{CO2}}$ and pressure without iterative per-timestep flash. Pure CO₂ supercritical density comes from Peng-Robinson EOS ($\sim 400-950\text{ kg/m}^3$, $B_{\text{CO2}} = 327.36/\rho$).
   **Verified 05-10-2026:** CRIT-04 fixed ($B_g = 5.0351$ derived vs $5.035$ used) and CRIT-05 fixed (Papay 1968; measured $Z = 0.849/0.827/0.868/0.972$ at $P_{pr}$ 2.24→6.73, a proper dense-gas dip). **Open defects:** CRIT-03 — $P_b$ is the literal constant `min(P_init, 2800)`, **no Standing correlation exists**; CRIT-21 — $c_g(\text{CO}_2)$ is an uncited power law contradicting the Peng-Robinson EOS *in the same class* by 2.6–8.3×; HIGH-20 — $B_o$ is C⁰ but not C¹ at $P_b$ ($dB_o/dP$ flips +1.26e-4 → −1.57e-5 /psi); HIGH-21 — `c_o` is `1.2e-5` here but `1e-5` in the pressure ODE.

6. **Authentic Viscous Fingering (Koval & Todd-Longstaff)** — ⚠️ **the sweep is correct but inert in the shipped configuration**:
   The Koval term is now continuous and strictly monotone in $M$ (measured max jump `7.5e-3` across $M=1$), so SCI-FLAW-01 is genuinely eliminated. **But** the default configuration drives it to its `0.95` clip: measured sweeps at the shipped HCPVI = 7.6928 are `[0.95, 0.95, 0.95, 0.95]` for $M = 1, 2, 5, 10, so mobility ratio cannot influence sweep at all (**CRIT-06** `CONFIRMED_BUT_INERT`, **CRIT-15**). Compounding this, `mobility_ratio` is now a hard-coded constant (`EORParameters.mobility_ratio`, default 5.0) that **overrides** the PVT-derived $M$, so recovery is *exactly* flat in oil viscosity (**CRIT-14**): measured RF identical to 6 dp for $\mu_o$ = 0.5 → 100 cP, where the pre-remediation path varied 0.5938 → 0.5595.

7. **Geomechanical Stress Path, Caprock & Fault Integrity (`core/engine_surrogate/geomechanics_fault.py`)**:
   In-situ horizontal stress evolves with reservoir pore pressure ($\Delta \sigma_h = \gamma_h \Delta P$). Caprock tensile and shear failure envelopes are evaluated at bottomhole injection pressures. Critically oriented faults are evaluated via Mohr-Coulomb slip tendency ($T_s = \tau / \sigma_n'$). Dynamic geological leakage occurs if sandface pressure breaches seal threshold or fault slip reactivation occurs.

8. **Comprehensive 4-Fluid-Stream Output Delivery**:
   The engine returns 4 standardized fluid streams at daily, monthly, and annual resolutions:
   - **Crude Oil**: Surface STB/day, cumulative STB, annual STB, plateau duration.
   - **Natural Gas**: Hydrocarbon sales gas (MSCF), solution gas degassing, total gas, and separate pure CO₂ stream.
   - **Water**: Formation brine, injected water, cumulative bbl, and dynamic water cut ($f_w$).
   - **Injection Agent**: CO₂ injected/purchased/recycled/stored + WAG water, constrained by facility compressor limits ($Q_{\text{recycle,max}}$) and 95% availability.

9. **Mass-Conserving WAG Mobility Buffering**:
   Crude static rate multipliers (+8% / -4%) have been replaced by physics-based phase mobility contrast $\Delta \lambda / \Sigma \lambda$ in `profile_generator_fast.py`. Cumulative oil and water production are strictly re-normalized.
   ⚠️ **Not verified.** Re-measure before relying on it — see CRIT-17: the post-rescale saturation re-synchronisation breaks $\sum S = 1$ on 27 % of timesteps.

10. **Closed-Loop Carbon Accounting Invariant**:
    $$\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Leakage} + \text{Produced}$$
    ⚠️ **The first two equalities hold. The second half is uncheckable.** Measured 05-10-2026: the gross ledger closes to machine precision (`produced + stored = injected`, residual `0.00e+00` MSCF) and `recycled ≤ produced ≤ injected` holds — but `total_leakage_tonne ≡ 0.0` in every configuration, so the `Leakage` term is structurally absent and `cum_stored = injected − produced` ignores it. **See HIGH-23.**

11. **Geomechanical Containment Safety (EPA Class VI)**:
    Sandface injection pressure is capped at $0.90 \times P_{\text{frac}}$ and injection throttles to zero at that ceiling. Overpressure incurs quadratic penalties ($10^6 \cdot (\Delta P / P_{\text{limit}})^2$).
    ⚠️ **The cap is an artefact, not a solved constraint.** Reservoir pressure is `np.clip`ped to `[p_min, p_safe_ceiling]` at `core/engine_surrogate/surrogate_engine.py:468`, so it cannot exceed the ceiling by construction — no geomechanical solve produces that bound. Leakage is identically zero, so a breach costs nothing. **See HIGH-23.**

12. **Test Suite Health & Zero Silent Swallowing**:
    Zero silent exception swallowing across core scientific modules. All physical states are logged contextually. Unphysical or violating candidates receive explicit mathematical penalties (`FAILURE_PENALTY = -10^{12}`) to kill off unviable chromosomes.

13. **Project Save/Load & State Persistence Integrity**:
    User projects (`.tphd` JSON format via `utils/project_file_handler.py`) must cleanly round-trip all reservoir geometries, PVT parameters, well patterns, tuning overrides, and optimization results:
    - **Shallow Dataclass Encoding**: `ProjectEncoder` serializes fields shallowly so nested dataclasses (`EOSModelParameters`, `LayerDefinition`, `GeostatisticalParams`) preserve `_dataclass` type annotations. Never invoke recursive `dataclasses.asdict()`.
    - **Backwards Compatibility**: `project_decoder` must dynamically reconstitute untyped dictionary representations from legacy projects into typed dataclasses.
    - **Robust Grid Ingestion**: Grid permeability deserialization in `DataManagementWidget` must support scalar, 1D flattened, and multi-dimensional grid arrays (e.g. `PERMX.flat[0]`), never assuming 3D shapes.
    - **Engine Results Restoration**: `OptimizationEngine.results` must provide `@results.setter` to ensure saved runs can be restored into engines and GUI summary plots.
    - **Verification Requirement**: Whenever data models or UI widgets are modified, agents must run `pytest tests/test_project_save_load.py -v`.

14. **Simulation Run Audit Logging & Historical Tracking Protocol**:
    Every reservoir simulation run audit, parameter sweep evaluation, or benchmark run conducted by developers or AI agents must be logged in [`agent_wiki/audit/simulation_run_audits/index.md`](audit/simulation_run_audits/index.md) within its dedicated date-stamped subfolder (`DD-MM-YYYY_<run_name>/audit.md`). Every audit entry must be marked with the date in `DD-MM-YYYY` format (e.g. `24-09-2026`) and must document:
    - **Verdict**: Clear evaluation status (`PASSED`, `ACCEPTABLE WITH CONDITIONS`, `FLAGGED`, or `FAILED`).
    - **Proposal**: Concrete actionable proposal (parameter updates, physics fixes, or operational guidelines).
    - **Relevant Files**: Markdown links to input configurations, engine modules, execution scripts, and output data.
    - **Past Runs Tracking**: Indexed in the Master Simulation Run Audits table so historical runs and past proposals remain visible across sessions.

15. **Finding & Continuity Discipline (added 05-10-2026)**:
    - **Never hand-write a finding.** `audit/scientific_flaws.md` is machine-generated and schema-validated: `python -m audit --register new <ID> …` then `python -m audit --register validate`. Severity, Category and Status are closed enumerations; a `Location` must cite a real `file:line` or `` `repo-wide` ``. See [`development/finding_registry.md`](development/finding_registry.md).
    - **Never hand-write the GitHub issue.** `python -m audit --register issue <ID>` renders it from the register record.
    - **A `RESOLVED` mark requires a measurement.** `python -m audit.continuity check <ID>` must report `CONFIRMED` before a status is written. On 05-10-2026 seventeen marks were written without one; eight needed reversing and two were regressions. See [`development/continuity_gate.md`](development/continuity_gate.md).
    - **One commit closes one issue.** `python -m audit.continuity check-commit "<subject>" "<body>"` rejects `Closes #1, #2`.

16. **The Compositional Engine Is a Second Engine, Built First and Coupled Later** (added 07-10-2026) —
    🔴 **CURRENT ACTIVE WORKSTREAM.** A full-physics **3D compositional engine in Rust** is being built
    at `crates/compositional/`, because **no available simulator covers the full spectrum of this
    project's tasks**.
    - **Prior art:** an `EngineFactory` **and a Python compositional engine were built, then removed**
      after the attempt failed. Rust is the second attempt. See
      [`compositional/engine_invariants.md`](compositional/engine_invariants.md) §6.
    - **The two engines have different jobs, not different quality levels.** The surrogate
      (`core/engine_surrogate/`) **screens and ranks** at $10^3$–$10^5$ evaluations per run. The
      compositional engine **evaluates** a small number of candidates with solved physics.
    - 🔴 **Seven mandatory invariants** — [`compositional/engine_invariants.md`](compositional/engine_invariants.md):
      **INV-1 fail loudly, never fall back** (every error returns a typed error and *stops*); **INV-2 runs
      only when the user starts it**; **INV-3 simulation only** (a separate economic engine owns field
      development); **INV-4 output is training-ready** (labelled $(x,y)$ pairs, full fields, versioned,
      provenance — the engine also **trains a per-reservoir neural surrogate**, not yet built);
      **INV-5 own flaw register** (`COMP-nn`, a separate module — the Python register cannot hold a `.rs`
      location); **INV-6 capability declaration vs. failure** (`NOT_IMPLEMENTED` ≠ `FAILED`);
      **INV-7 unconstrained** (absurd input still gets full physical evaluation).
    - 🔴 **Coherence is enforced, not assumed** — [`build_plan.md`](compositional/build_plan.md) **M4.5**
      requires an end-to-end run (spec → solve → output), and
      [`separation_doctrine.md`](compositional/separation_doctrine.md) **C-7** requires every `pub` module to
      be reachable from the driver and exercised by a test. 📌 **This is the gate against disconnected
      modules**: a crate can pass every per-module unit test and still not produce one run.
    - 📌 **The invariants are split across four topic files** (09-10-2026, §-numbers preserved exactly):
      [`engine_invariants.md`](compositional/engine_invariants.md) (the seven invariants) ·
      [`engine_spec_closures.md`](compositional/engine_spec_closures.md) (`CONF-*` closures, DOIs, provenance) ·
      [`engine_numerics.md`](compositional/engine_numerics.md) (architecture, newtypes, status lattice, tolerances) ·
      [`engine_constitutive.md`](compositional/engine_constitutive.md) (fracture, Drucker–Prager, Lode angle).
      **Read the §0 index in `engine_invariants.md` first** — it maps milestones to sections.
    - ✅ **The two engines complement.** The surrogate is **not** retired; P3 coupling definitely exists.
      **Python is left as-is** — the 73 legacy findings remain the Python audit.
    - 🔵 **Output schema APPROVED 08-10-2026** — 9 domains, **116 float fields per cell per timestep**,
      three temporal resolutions (micro-step / daily / monthly-yearly), global indexing
      $I_{global}=i+(j-1)N_x+(k-1)N_xN_y$ plus NNC arrays. Formats: **HDF5/VTK-HDF**, **RESQML 2.0.1 /
      GRDECL**, **Parquet/Arrow**, **DuckDB/SQLite**.
      🔴 **The engine emits all raw values; satellite tools interpret and format them.**
      ⚠️ **This is a TARGET specification. No engine exists** — current state is M0, not started.
      ⚠️ **Measured: 510 MB per timestep at SPE-10 scale, 51 GB for 100 steps** — ruled remedy is
      **chunked HDF5 with `zstd`** plus an **active-frame RAM cache**.
      🔴 **Two-output contract (ruled):** spatial → HDF5; time-series → Parquet/DuckDB.
      **The satellite economic engine never reads HDF5.**
      Full spec: [`compositional/output_schema.md`](compositional/output_schema.md).
    - ✅ **Ruling of 08-10-2026 applied** — economics purged (**CONF-64/65**), Koval unified (**CONF-63**),
      $\epsilon_p$ removed from the elastic baseline (**CONF-66**), $\text{Fe}^{2+}$ added to the aqueous
      species (**CONF-67**), skin floor corrected (**CONF-68**), **CONF-35 resolved** by unit conversion
      ($1$ MSCF CO₂ $\approx 0.0519$ t). ⚠️ Two of my own framings were **withdrawn by measurement** —
      see [`compositional/spec_corrections_log.md`](compositional/spec_corrections_log.md) C-37, C-38.
      Full adjudication: [`thmc/reservoir_engineer_ruling.md`](thmc/reservoir_engineer_ruling.md) §6–§9.
    - 🔴 **Python post-mortem recorded** — the first attempt failed on **six numerical-method defects**,
      **not** on execution speed. Yields five new requirements (**Heidemann–Khalil** critical solver,
      exponential soft-start, well control modes in the global Jacobian, permeability-collapse selection
      rule). [`compositional/python_attempt_postmortem.md`](compositional/python_attempt_postmortem.md).
    - 🔴 **Spec authority rule:** `3D_THMC_docs/` are **immutable references**; the **wiki carries the
      corrected specification**. Every deviation is logged with source and line in
      [`compositional/spec_corrections_log.md`](compositional/spec_corrections_log.md) —
      **19 corrections, 7 open branches**.
    - ⚠️ **Scope re-baselined 08-10-2026** to the **full 3D THMC programme** (coupled thermal +
      geomechanics + geochemistry, fractures/EDFM, wellbore, Schwarz sub-domains, GPU, adjoint, ES-MDA).
      The earlier "M7+ deferred" framing is **void**. **Resolving scope has not resolved specification** —
      CONF-13/14/25/51/16/18/47/08/63/66 all remain open.
    - ✅ **Simulation learning committed** — full output enables RL to tune the surrogate on full physics.
      A known, accepted long process. The **economic engine is on the critical path** for the reward.
    - **Phase model** — [`compositional/vision_and_phases.md`](compositional/vision_and_phases.md):
      **P1 DEVELOP** (standalone, own register — separation is a **development discipline to avoid
      distraction, not an architecture**) → **P2 INTEGRATE** → **P3 COUPLE**. P3 begins only after the
      engine is verified on all features and fronts.
    - ⚠️ **During P1 only:** do not import anything from `core/`, `evaluation/`, `validation/`, `utils/`
      or `ui/`. **These gates retire by ADR at P2.**
    - **Own flaw register** at `audit_comp/`, namespace **`COMP-nn`**. ⚠️ `audit/registry.py:114`
      `LOCATION_RE` accepts `.py` only — the existing register **cannot** record a Rust finding.
    - **No code exists.** Rust toolchain installation is **deferred by decision** until after
      documentation, at the start of development.
    - Plan: [`compositional/build_plan.md`](compositional/build_plan.md) — **M0 → M6**, each gated by a
      measured value. **M1 references verified 07-10-2026**:
      [`compositional/spec_defects.md`](compositional/spec_defects.md) §2a.
    - 🔵 **Data layer decided:** **PostgreSQL + vector store**, replacing file-based config —
      [`compositional/data_architecture.md`](compositional/data_architecture.md).
    - ⚠️ **`core/data_models.py` and `config/base_config.json` cannot express a full simulation** —
      `ReservoirData.grid` holds **only dimensions**, porosity/permeability are **scalars**, no NTG,
      transmissibility, corner-point geometry or 3D facies, the shipped fluid has **2 components and 1
      binary interaction coefficient**, PVT tables are single-element lists. Evidence:
      [`compositional/data_model_gap.md`](compositional/data_model_gap.md).
    - ⚠️ **`agent_wiki/development/extension_points.md` is unbuildable** — it instructs edits to
      `config/default_config.json`, `core/engine_factory.py`, `core/models/`, `core/surrogate_engine.py`,
      `core/analytical_models.py`, **none of which exist**. `EngineFactory` was built and removed. Verify
      every path with `Test-Path`.
    - ⚠️ **UI/UX is out of scope** until the Rust engine is fully built and tested.
    - **The active engine for the existing application remains `core/engine_surrogate/`** (invariant 1).

17. **The 3D THMC Design Set Is Design Intent, Not Implementation** (added 07-10-2026):
    - [`agent_wiki/thmc/`](thmc/README.md) is a **normalised, agent-readable transcription** of the 8 Ukrainian design documents in `3D_THMC_docs/`. They describe a **Rust** 3D THMC compositional core.
    - ⚠️ **No Rust code exists in this repository.** Verified 07-10-2026: `*.rs` → 0, `Cargo.toml` → 0, CI → 0. All 6 acceptance criteria in the core spec are **unchecked `[ ]`**.
    - **100 % of optimisation evaluations still route to `core/engine_surrogate/`.** The THMC set changes nothing about engine routing.
    - **Never cite a THMC number as validated.** Every tolerance, accuracy % and benchmark timing in `3D_THMC_docs/` is an assertion in a design document. The V&V framework supplies **zero** absolute reference values for any of its 22 declared tests.
    - **Blocking conflicts are recorded** in [`thmc/conflict_and_gap_register.md`](thmc/conflict_and_gap_register.md) — read it before any THMC work. ⚠️ **CONF-01 was RETRACTED on 07-10-2026**: the proposed fractional flow has the **correct sign**; my original claim of $df_g/dM<0$ came from a substitution error and is **withdrawn**. The real defect is the *closure* — no $S_{or}$, linear instead of Corey exponents, no water term. See [`thmc/reservoir_engineer_ruling.md`](thmc/reservoir_engineer_ruling.md). The other blockers: **CONF-02** (HCPVI cap 99.4 % inert at the shipped default HCPVI 7.69 — the CRIT-15 shape), **CONF-03** (6 documents report penalty removal and NPV unification as *accomplished*; `FAILURE_PENALTY = -10^{12}` is still live and `tests/test_surrogate_engine.py` does not exist), **CONF-04** (economics both exiled from and inside the core — **adjudicated**: core exposes $\partial q_i/\partial x$; prices applied off-core).
    - **Staged evaluation:** [`thmc/evaluation_plan.md`](thmc/evaluation_plan.md) — Stage 0 documentation remediation → Stage 1 convert the documents' acceptance criteria into measurements against the *existing* engine → Stage 2 ADR decision (ADOPT / HARVEST / PILOT / REJECT) → Stage 3 scope. Recommendation: **PILOT**, with the language-independent items taken regardless.

---

## 📖 Recommended Reading Order

1. Start with [Architecture Overview](architecture/overview.md) to understand system architecture.
2. Review [Source of Truth Map](architecture/source_of_truth_map.md) before modifying any engine or simulation code.
3. Review [Recovery Model](physics/recovery_model.md) and [Displacement Model](physics/displacement_model.md) before touching recovery or rate calculations.
4. Consult [Suspicious Logic](audit/suspicious_logic.md) and [Common Pitfalls](development/common_pitfalls.md) to avoid known failure modes.
5. Check [Change Safety Matrix](development/change_safety_matrix.md) to determine the risk tier of your task.

> If the task touches the 3D THMC design set (`3D_THMC_docs/` or [`agent_wiki/thmc/`](thmc/README.md)),
> read [`thmc/conflict_and_gap_register.md`](thmc/conflict_and_gap_register.md) **before** writing anything.
> That section describes a simulator which does not exist; treat every claim in it as unverified.
>
> 🔴 **If the task is about the compositional engine** (`crates/compositional/`, `audit_comp/`,
> [`agent_wiki/compositional/`](compositional/README.md)), start with
> [`compositional/vision_and_phases.md`](compositional/vision_and_phases.md). It is the **current active
> workstream**: a full-physics 3D compositional simulator in Rust, built **first**, coupled to the
> surrogate in a **later phase** (P3). During **P1** it is decoupled from `core/engine_surrogate/` and
> has its own flaw register (`COMP-nn`) — separation there is a **development discipline to avoid
> distraction, not an architecture**, and those gates retire at P2.
