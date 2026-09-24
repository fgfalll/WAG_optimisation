# CO₂ EOR Optimizer — Technical Wiki

Welcome to the **CO₂ EOR Optimizer Agent Wiki**. This repository of technical knowledge documents the **actual implementation, physical models, numerical approximations, execution pathways, and architectural realities** of the CO₂ Enhanced Oil Recovery Optimizer codebase (`co2eor_optimizer`, v0.8.5).

This documentation is designed for **reservoir engineers, AI pair-programmers, scientific computing auditors, and software architects**. It provides unambiguous, machine-readable, and scientifically grounded guidance to enable rapid comprehension, prevent accidental regression of scientific invariants, and eliminate modification risk.

---

## 🧭 Documentation Structure

| Section | Purpose |
|---|---|
| [**Architecture**](architecture/overview.md) | Engine routing, module map, execution flow, dependency graph |
| [**Physics Models**](physics/reservoir_model.md) | Reservoir geometry, PVT, CO₂ properties, displacement, Koval, Todd-Longstaff |
| [**Data & Parameters**](data/inputs.md) | Input/output schemas, parameter registry, field unit definitions |
| [**Development Guide**](development/common_pitfalls.md) | Change safety matrix, common pitfalls, safe modification rules |
| [**Audit Reports**](audit/technical_debt.md) | Dead code, hardcoded values, [**simulation run audits**](audit/simulation_run_audits.md), fallbacks, suspicious logic, [**resolved archive**](audit/resolved_issues.md) |
| [**Verification**](verification/verification_strategy.md) | 7-level V&V hierarchy, conservation tests, convergence studies |
| [**Validation**](validation/benchmarks.md) | SPE 5, CMG GEM reference benchmarks |
| [**Decisions**](decisions/architecture_decisions.md) | Architecture & scientific rationale records (ADRs) |

---

## ⚡ Critical Architectural Invariants

Before reading or modifying any file in this repository, keep the following **core facts** in mind:

1. **The Single Active Simulation Engine is `core/engine_surrogate` (Intermediate Physics-Informed Simulator)**:
   The name "Surrogate" signifies an **intermediate, physics-informed reduced-order reservoir simulator** positioned between full 3D multi-block compositional numerical solvers and classical 0D material balance. It operates as a general reservoir simulator embedded with specialized CO₂ EOR expertise (solvent dissolution, oil swelling, viscosity reduction, Koval viscous fingering, Todd-Longstaff partial miscibility, and EPA Class VI geomechanical integrity). All legacy engines (`compositional_engine`, `unified_engine`, `engine_simple`) and the intermediate `EngineFactory` have been deprecated and relocated into `deprecated/` to eliminate confusion. All simulation evaluations route directly to `SurrogateEngineWrapper` (`core/engine_surrogate/surrogate_engine.py`).

2. **`FastProfileGenerator` is the Single Source of Truth for Profiles**:
   All production, injection, and rate profiles are synthesized via `core/engine_surrogate/profile_generator_fast.py`. The legacy classes in `core/simulation/injection_schemes.py` and `core/simulation/profile_generator.py` are deprecated wrappers.

3. **NPV and CO₂ Accounting are Engine-Owned**:
   NPV calculation and CO₂ purchased/recycled volumes are computed directly inside `core/engine_surrogate/surrogate_engine.py` (`_calculate_engine_npv` and `_calculate_co2_purchased_recycled`). The wrapper in `core/objectives/wrapper.py` is a consumer that reads engine results.

4. **Tank Pressure Uses Coupled Darcy/Vogel IPR & Damped Material Balance**:
   The legacy `solve_ivp` pressure ODE has been superseded by an explicit, physics-based deliverability and material balance model in `surrogate_engine.py`:
   - Gas injection converted via dynamic formation volume factor $B_{g,\text{dynamic}}$ to reservoir barrels per day (RB/d).
   - Dynamic Koval fractional flow with Todd-Longstaff effective mobility ratio $M_e$.
   - Deliverability bounded by producer drawdown ($J_{\text{prod}}$) and injector geomechanical limits ($J_{\text{inj}}$).
   - Pressure increment: $dP = (q_{\text{net,IPR}} \cdot \Delta t) / (V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t)$.
   - Dual-pressure Voidage Replacement Ratio (VRR) tracking local injection vs production reservoir volumes.

5. **Thermodynamic Rigor via Solvent-Extended PVT (`core/engine_surrogate/pvt_state.py`)**:
   Transport saturation ($S_g$) is strictly decoupled from composition ($x_{\text{CO2}}, y_{\text{CO2}}$). Fluid properties ($B_o, \mu_o, S_F, B_{\text{CO2}}$) evolve with dissolved solvent concentration $x_{\text{CO2}}$ and pressure without requiring costly iterative per-timestep flash equations. Pure CO₂ supercritical density is evaluated via Peng-Robinson EOS ($\sim 400-950\text{ kg/m}^3$, $B_{\text{CO2}} = 327.36 / \rho$). Degassing and surface shrinkage are tracked via multi-stage flash separation.

6. **Authentic Viscous Fingering (Koval & Todd-Longstaff)**:
   SCI-FLAW-01 has been eliminated. The fractional flow of gas $f_g$ and effective Koval factor $K = H_k \cdot E_{\text{eff}}$ strictly satisfy $\partial f_g / \partial M > 0$. An adverse mobility ratio ($M > 1$) monotonically accelerates breakthrough and increases gas channeling.

7. **Geomechanical Stress Path, Caprock & Fault Integrity (`core/engine_surrogate/geomechanics_fault.py`)**:
   In-situ horizontal stress evolves with reservoir pore pressure ($\Delta \sigma_h = \gamma_h \Delta P$). Caprock tensile and shear failure envelopes are evaluated at bottomhole injection pressures. Critically oriented faults are evaluated via Mohr-Coulomb slip tendency ($T_s = \tau / \sigma_n'$). Dynamic geological leakage occurs if sandface pressure breaches seal threshold or fault slip reactivation occurs.

8. **Comprehensive 4-Fluid-Stream Output Delivery**:
   The engine returns 4 standardized fluid streams at daily, monthly, and annual resolutions:
   - **Crude Oil**: Surface STB/day, cumulative STB, annual STB, plateau duration.
   - **Natural Gas**: Hydrocarbon sales gas (MSCF), solution gas degassing, total gas, and separate pure CO₂ stream.
   - **Water**: Formation brine, injected water, cumulative bbl, and dynamic water cut ($f_w$).
   - **Injection Agent**: CO₂ injected/purchased/recycled/stored + WAG water, constrained by facility compressor limits ($Q_{\text{recycle,max}}$) and 95% availability.

9. **Mass-Conserving WAG Mobility Buffering**:
   Crude static rate multipliers (+8% / -4%) have been replaced by physics-based phase mobility contrast $\Delta \lambda / \Sigma \lambda$ in `profile_generator_fast.py`. Cumulative oil and water production are strictly re-normalized, ensuring exact mass conservation.

10. **Closed-Loop Carbon Accounting Invariant**:
    Net CO₂ storage is strictly balanced:

    $$\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Leakage} + \text{Produced}$$

11. **Geomechanical Containment Safety (EPA Class VI)**:
    Sandface injection pressure is strictly capped at $0.90 \times P_{\text{frac}}$ ($P_{\text{safe ceiling}}$). If reservoir pressure reaches this ceiling, injection is instantly throttled to zero (Class VI UIC shut-in). Overpressure violations incur quadratic economic penalties ($10^6 \cdot (\Delta P / P_{\text{limit}})^2$).

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
    Every reservoir simulation run audit, parameter sweep evaluation, or benchmark run conducted by developers or AI agents must be logged in [`agent_wiki/audit/simulation_run_audits.md`](audit/simulation_run_audits.md). Every audit entry must be marked with the date in `DD-MM-YYYY` format (e.g. `24-09-2026`) and must document:
    - **Verdict**: Clear evaluation status (`PASSED`, `ACCEPTABLE WITH CONDITIONS`, `FLAGGED`, or `FAILED`).
    - **Proposal**: Concrete actionable proposal (parameter updates, physics fixes, or operational guidelines).
    - **Relevant Files**: Markdown links to input configurations, engine modules, execution scripts, and output data.
    - **Past Runs Tracking**: Indexed in the Master Simulation Run Audits table so historical runs and past proposals remain visible across sessions.

---

## 📖 Recommended Reading Order

1. Start with [Architecture Overview](architecture/overview.md) to understand system architecture.
2. Review [Source of Truth Map](architecture/source_of_truth_map.md) before modifying any engine or simulation code.
3. Review [Recovery Model](physics/recovery_model.md) and [Displacement Model](physics/displacement_model.md) before touching recovery or rate calculations.
4. Consult [Suspicious Logic](audit/suspicious_logic.md) and [Common Pitfalls](development/common_pitfalls.md) to avoid known failure modes.
5. Check [Change Safety Matrix](development/change_safety_matrix.md) to determine the risk tier of your task.
