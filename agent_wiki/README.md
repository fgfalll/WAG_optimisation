# CO₂ EOR Optimizer — Technical Wiki

Welcome to the **CO₂ EOR Optimizer Agent Wiki**. This repository of technical knowledge documents the **actual implementation, physical models, numerical approximations, execution pathways, and architectural realities** of the CO₂ Enhanced Oil Recovery Optimizer codebase (`co2eor_optimizer`, v0.8.5).

This documentation is designed for **reservoir engineers, AI pair-programmers, scientific computing auditors, and software architects**. It provides unambiguous, machine-readable, and scientifically grounded guidance to enable rapid comprehension, prevent accidental regression of scientific invariants, and eliminate modification risk.

---

## 🧭 Documentation Structure

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

---

## 📖 Recommended Reading Order

1. Start with [Architecture Overview](architecture/overview.md) to understand system architecture.
2. Review [Source of Truth Map](architecture/source_of_truth_map.md) before modifying any engine or simulation code.
3. Review [Recovery Model](physics/recovery_model.md) and [Displacement Model](physics/displacement_model.md) before touching recovery or rate calculations.
4. Consult [Suspicious Logic](audit/suspicious_logic.md) and [Common Pitfalls](development/common_pitfalls.md) to avoid known failure modes.
5. Check [Change Safety Matrix](development/change_safety_matrix.md) to determine the risk tier of your task.
