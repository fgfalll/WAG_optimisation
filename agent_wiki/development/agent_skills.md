# Agent Skills Knowledge Base & Execution Standards

This document records the specialized **Agent Skills** integrated into the CO₂ EOR Optimizer development and operational environment. AI pair-programmers and autonomous agents operating on this repository must draw upon these specialized skills and their associated reference frameworks.

---

## 🧭 Active Skills Inventory

| Skill Name | Origin & Invocation | Core Domain & Focus | Key References |
|:---|:---|:---|:---|
| **`agent-wiki`** | Local: [`.agents/skills/agent-wiki/SKILL.md`](file:///d:/rep/4.6/co2eor_optimizer/.agents/skills/agent-wiki/SKILL.md) | **Mandatory Gatekeeper**: Enforces consulting `agent_wiki/README.md` before any tool action in every turn. | `agent_wiki/README.md`, `source_of_truth_map.md`, `common_pitfalls.md` |
| **`petroleum-engineer`** | Remote: `theneoai/awesome-skills` | **Domain Authority**: Senior Reservoir, Production, Drilling & EOR Engineering. Governs 5-tier Decision Hierarchy (Reserves $\to$ Rate $\to$ Cost $\to$ Risk $\to$ Value). | `references/domain.md`, `decision-frameworks.md`, `workflow.md`, `problem-signature.md`, `risks.md`, `three-layer-architecture.md` |
| **`simulation-orchestrator`** | Remote: `heshamfs/materials-simulation-skills` | **Campaign Execution**: Multi-run parameter sweeps (Grid, Linspace, LHS), nested dot-notation configs, job tracking, and result aggregation. | `scripts/sweep_generator.py`, `campaign_manager.py`, `job_tracker.py`, `result_aggregator.py`, `sweep_strategies.md` |
| **`physics-simulation`** | Remote: `omer-metin/skills-for-antigravity` | **Numerical Correctness**: ODE solvers, CFL stability conditions, Hamiltonian/symplectic integration, FEM, and particle dynamics. | `references/patterns.md` (Creation), `references/sharp_edges.md` (Diagnosis), `references/validations.md` (Audit) |

---

## 1. `agent-wiki` Gatekeeper Skill

- **Path**: [`.agents/skills/agent-wiki/SKILL.md`](file:///d:/rep/4.6/co2eor_optimizer/.agents/skills/agent-wiki/SKILL.md)
- **Role**: Mandatory repository gateway.
- **Invariant**: The agent's **VERY FIRST tool call in EVERY turn MUST be** `view_file("agent_wiki/README.md")` (or the specific relevant topic document in `agent_wiki/`).
- **Core Guardrails**:
  1. Prevents modifying dormant engines (`core/Phys_engine_full/`, `compositional_engine/`, `unified_engine/`) under the false assumption that they affect runtime optimizations.
  2. Guards against the 28 documented traps in [`common_pitfalls.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/development/common_pitfalls.md).
  3. Enforces physical mass conservation, Vogel-Darcy composite IPR limits, and EPA Class VI geomechanical constraints.

---

## 2. `petroleum-engineer` Skill

- **Invocation**: `npx skills use "https://github.com/theneoai/awesome-skills" --skill "petroleum-engineer" --full-depth`
- **Cached Location**: `C:\Users\sayno\AppData\Local\Temp\skills-use-ZZZOup\petroleum-engineer\`
- **Role**: Provides the engineering persona, physics standards, and decision-making framework for reservoir management and EOR evaluation.

### Decision Hierarchy
1. **Reserves / Volume**: Confirm initial fluids in place (STOOIP / GIIP) and volumetric sweep efficiency $E_v = E_A \times E_I$.
2. **Deliverability / Rate**: Apply Composite Vogel-Darcy IPR to model production rates clamped to reservoir drawdown capacity.
3. **Operational Cost**: Incorporate water and CO₂ injection/recycling OPEX and compression costs.
4. **Subsurface Risk**: Enforce EPA Class VI geomechanical limits ($P_{\text{sandface}} \le 0.90 \times P_{\text{frac}}$) to prevent caprock fracturing.
5. **Economic Value (NPV)**: Calculate project NPV as the final synthesized objective.

### Key References:
- `references/domain.md`: SPE reserves classification (1P/2P/3P), fluid PVT behavior, Corey/Brooks-Corey relative permeabilities, Material Balance Equation (MBE).
- `references/decision-frameworks.md`: Decision trees for artificial lift selection, EOR screening, and pattern flooding.
- `references/risks.md`: Reservoir risks (early breakthrough, viscous fingering, gravity override, asphaltene precipitation, reservoir souring).

---

## 3. `simulation-orchestrator` Skill

- **Invocation**: `npx skills use "https://github.com/heshamfs/materials-simulation-skills" --skill "simulation-orchestrator" --full-depth`
- **Cached Location**: `C:\Users\sayno\AppData\Local\Temp\skills-use-rjOTkx\simulation-orchestrator\`
- **Role**: Automates parameter study campaigns, sensitivity sweeps, and multi-run aggregations for `co2eor_optimizer`.

### Workflow & Scripts:
1. **Sweep Generation** (`scripts/sweep_generator.py`):
   - Generates parameter sets using `grid` (full factorial), `linspace` (uniform), or `lhs` (Latin Hypercube Sampling with `--seed`).
   - Supports deep dot-notation overrides (e.g. `parameters.injection.co2_rate`).
2. **Campaign Initialization** (`scripts/campaign_manager.py`):
   - Sets up campaign state, manifests, and safely quoted command templates (`--action init`, `status`, `list`).
3. **Job Tracking** (`scripts/job_tracker.py`):
   - Tracks execution states (`pending`, `running`, `completed`, `failed`) based on filesystem markers and outputs.
4. **Result Aggregation** (`scripts/result_aggregator.py`):
   - Extracts scalar metrics (e.g. `npv`, `recovery_factor`), computes statistics (min, max, mean, std, median, IQR), and isolates `best_run` (`--maximize` or default minimize).

### Verification & Safety Checklist:
- **Key Path Merge**: Ensure swept dot-notation parameters overwrite the exact nested solver key rather than leaving defaults intact.
- **Directionality**: Verify `--maximize` is passed for recovery factor and NPV.
- **Job Reconciliation**: Ensure `completed + failed == total_jobs` to catch silent failures.
- **Outlier Screening**: Apply Tukey 1.5× IQR outlier filtering on aggregated outputs.
- **Simulation Run Audit Protocol**: Every parameter sweep or simulation evaluation must be logged in [`agent_wiki/audit/simulation_run_audits.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits.md) marked as `DD-MM-YYYY` with verdict, proposal, and relevant files.

---

## 4. `physics-simulation` Skill

- **Invocation**: `npx skills use "https://github.com/omer-metin/skills-for-antigravity" --skill "physics-simulation" --full-depth`
- **Cached Location**: `C:\Users\sayno\AppData\Local\Temp\skills-use-GpkUhe\physics-simulation\`
- **Role**: Enforces mathematical rigor, numerical stability, and algorithmic correctness for physical modeling and ODE solvers.

### Reference System Architecture:
- **For Creation (`references/patterns.md`)**:
  - Base numerical integrators: Explicit Euler, RK4, Velocity Verlet (symplectic/Hamiltonian), and adaptive RK45 with Butcher tableau.
  - Rigid body dynamics: Euler rotation equations in body frame with `scipy.spatial.transform.Rotation`.
  - Finite Element Method (FEM): 1D rod/beam, 2D Constant Strain Triangle (CST), sparse matrix assembly, penalty boundary conditions.
  - Particle systems: Spatial hashing (`SpatialHash`), Hookean spring networks, damped contact forces.
- **For Diagnosis (`references/sharp_edges.md`)**:
  - **Timestep Instability**: Numerical explosion when exceeding the CFL condition ($\Delta t < \Delta x / v_{\text{max}}$) or spring stability limit ($\Delta t < 2\sqrt{m/k}$).
  - **Secular Energy Drift**: Energy non-conservation in non-symplectic integrators over long time horizons.
  - **Discrete Tunneling**: Fast-moving objects penetrating boundaries (requires Continuous Collision Detection or velocity clamping).
  - **Contact Jitter**: Undamped penalty springs (requires critical damping $c = 2\sqrt{km}$).
  - **Precision Degradation**: Precision loss in `float32` for large coordinates (enforce `float64`).
- **For Review & Audit (`references/validations.md`)**:
  - Rule `euler-for-dynamics`: Flag forward Euler on oscillatory or dynamic systems.
  - Rule `hardcoded-timestep`: Flag arbitrary $\Delta t$ without stability proof.
  - Rule `float32-simulation`: Enforce `float64` for simulation state arrays.
  - Rule `matrix-inversion-loop`: Prevent `np.linalg.inv()` inside simulation loops; use `np.linalg.solve()` or precomputation.
