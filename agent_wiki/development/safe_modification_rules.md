# Safe Modification Rules for AI Agents

To prevent accidental breakage of scientific invariants or silent corruption of optimization results, AI agents must adhere to the following **non-negotiable rules**:

---

## 1. Never Change Units Without End-to-End Dimensional Audit
- **The Rule**: If you alter the unit of an input or intermediate variable (e.g. converting MSCFD to res-bbl/day), you must trace its consumption through all derivative calculations (e.g., in `surrogate_engine._pressure_ode_system` and `objectives/wrapper.py`).
- **Rationale**: An uncoordinated unit change in injectivity caused a one-million-fold error in previous versions.

## 2. Never Add Hardcoded Calibration Factors
- **The Rule**: Never introduce unconfigurable numerical multipliers (e.g., `* 1.08` or `* 0.96`) into physical formulas or rate profiles.
- **Protocol**: If an empirical adjustment is physically justified, it must be added to [core/data_models.py:EmpiricalFittingParameters](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L2238), documented in [`agent_wiki/data/parameters.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/data/parameters.md#e-empiricalfittingparameters-line-2241), and default to an uncalibrated baseline (1.0 or physical theory).

## 3. Always Support `economic_params=None` Gracefully
- **The Rule**: `evaluate_scenario()` in `surrogate_engine.py` must never raise a breaking exception if `economic_params` is omitted.
- **Protocol**: If `economic_params is None`, fallback to `EconomicParameters()` defaults or return `npv = 0.0`. This preserves unit test compatibility across the entire repository.

## 4. Preserve Key Contract for `FastProfileGenerator`
- **The Rule**: `FastProfileGenerator.generate_profile()` is consumed by `SurrogateEngine`, `OptimizationEngine`, and UI plotting managers.
- **Mandatory Return Keys**: The dictionary returned by `generate_profile()` must always contain:
  - `time_vector`: 1D array of time in days.
  - `oil_profile`: 1D array of oil rate in STB/day.
  - `water_profile`: 1D array of water rate in bbl/day.
  - `gas_profile`: 1D array of total gas rate in MSCFD.
  - `co2_gas_profile`: 1D array of breakthrough CO₂ in MSCFD.
  - `solution_gas_profile`: 1D array of dissolved hydrocarbon gas in MSCFD.
  - `injection_profile`: 1D array of CO₂ injection in MSCFD.
  - `water_injection_profile`: 1D array of water injection in bbl/day.

## 5. Validate Any Physics Change Against Baseline Tests
- **The Rule**: After modifying any equation in `analytical_models.py`, `surrogate_engine.py`, or `profile_generator_fast.py`, run:
  ```bash
  uv run --extra dev --with hypothesis --with h5py pytest tests/ -v
  ```
  Ensure all passing tests remain passing and verify that recovery factor remains within physical bounds $[0.0, 1.0]$.

## 6. Zero Tolerance for Silent Exception Swallowing (`except Exception: pass`)
- **The Rule**: Bare `except:` or blind `except Exception: pass` blocks in core mathematical, solver, or simulation modules are strictly prohibited.
- **Protocol**: Catch specific exception types (e.g. `FloatingPointError`, `ZeroDivisionError`, `ValueError`, `RuntimeError`, `scipy.optimize.NonlinAlgError`). Always explicitly log the state variables (Pressure, Saturation, Temperature, and Composition) at the point of numerical breakdown.

## 7. No Artificial Penalty Softening or Class E Modifiers
- **The Rule**: Never invent artificial positive metrics (e.g., synthesizing non-zero storage efficiency from recovery factor alone) to cheat optimizer penalties.
- **Protocol**: If a chromosome in GA produces unphysical profiles, NaN metrics, or missing simulation data, it must return `NaN` or receive the full mathematical failure penalty (`FAILURE_PENALTY`, $-10^{12}$). Multipliers that dilute penalties (e.g. `* 0.1` or `* 0.8`) are strictly prohibited so that the optimizer naturally eliminates unviable genetic lineages.

## 8. Mandatory Visualization Dependencies
- **The Rule**: Plotly and primary visualization tools (`plotly`, `matplotlib`) are non-negotiable core requirements.
- **Protocol**: Never introduce dummy or mock classes (`class go: class Figure: pass`) that swallow chart generation. If a visualization library is absent, the system must fail loudly and inform the user during startup QA/QC.

## 9. Closed-Loop CO₂ Mass Conservation Closure Invariant
- **The Rule**: In closed-loop recycling EOR systems, gross injected volume equals net stored plus cumulative leakage plus gross produced gas:
  $$\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Total Leakage} + \text{Gross Produced}$$
- **Protocol**: Never compare gross accounted CO₂ directly against *purchased* fresh CO₂. Any audit or reporting tool (e.g. `RunDataExporter`, `material_balance.py`) must evaluate closure on a gross basis, enforcing closure $> 99.9\%$.

## 10. Complete Multi-Phase Stream Shut-In Protocol
- **The Rule**: Whenever a shut-in threshold (water cut, GOR, or environmental compliance limit) is triggered, **all** fluid production and injection streams must be synchronously tapered or zeroed.
- **Protocol**: When implementing or editing shut-in guards in `surrogate_engine.py` or `FastProfileGenerator`, always update: `water_profile`, `co2_gas_profile`, `solution_gas_profile`, `gas_profile`, and `injection_profile`. Leaving gas or CO₂ flowing while zeroing oil violates conservation of mass and incurs massive false ecology penalties.

## 11. Plotly Timeline Interval Bar Geometry
- **The Rule**: Vertical interval bar charts (`go.Bar`) representing operational timelines must calculate interval midpoints for `x` and set `width = duration`.
- **Protocol**:
  ```python
  x = [start_day + duration_days / 2.0]
  width = [duration_days]
  base = [0]
  y = [rate]
  barmode = "overlay"
  ```
  Never pass `x=[duration_days]` and `base=[start_day]`, as Plotly interprets `base` as the vertical Y-offset, collapsing multi-year timelines into a hairline stack.

## 12. Decline Curve Analysis Boundary-Dominated Regime Restriction
- **The Rule**: Arps decline models ($q(t) = q_i / (1 + b D_i t)^{1/b}$) are physically valid only in boundary-dominated decline ($dq/dt < 0$).
- **Protocol**: In `analysis/decline_curve_analysis.py`, always detect plateau onset ($q(t) < 0.95 \times q_{\text{peak}}$). If a multi-year plateau exists, preserve historical plateau rates and fit Arps parameters strictly to the declining segment $(t - t_{\text{onset}})$. Never regress Arps equations across a plateau.

## 13. Project Save/Load Serialization & State Preservation Integrity
- **The Rule**: Any changes to data models in `core/data_models.py`, serialization logic in `utils/project_file_handler.py`, or GUI tabs (`ui/data_management_widget.py`, `ui/config_widget.py`, `ui/main_window.py`) must guarantee that saving to `.tphd` and loading from `.tphd` round-trips without data loss, type stripping, or unhandled exceptions.
- **Protocol**:
  1. **Shallow Dataclass Serialization**: In `ProjectEncoder`, never use recursive `dataclasses.asdict(o)`, which strips `_dataclass` tags from nested dataclasses (`EOSModelParameters`, `LayerDefinition`, `GeostatisticalParams`). Instead, serialize fields shallowly.
  2. **Backwards Compatibility**: In `project_decoder`, convert untyped nested dicts into typed dataclasses to ensure legacy `.tphd` project files load seamlessly.
  3. **Array Shape Agnostic Ingestion**: In `DataManagementWidget.load_project_data()`, handle scalar, 1D flattened, and multi-dimensional grid arrays (e.g. `grid['PERMX'].flat[0]`), never assuming 3D indexing `[0,0,0]`.
  4. **Engine Results Accessor**: `OptimizationEngine.results` must retain `@results.setter` so that restored optimization runs populate engine state and GUI graphs.
  5. **UI State Flush Before Save**: In `MainWindow._perform_project_save()`, always query `data_management_tab.get_current_project_data()` and `config_tab.get_all_configurations()` to synchronize active widget inputs before writing to disk.
  6. **Mandatory Test**: Run `pytest tests/test_project_save_load.py -v` whenever touching data models, serialization, or UI state.

## 14. Simulation Run Audit Logging & Subfolder Workflow
- **The Rule**: Every reservoir simulation run audit, parameter sweep evaluation, or benchmark run conducted by developers or AI agents must be recorded in a dedicated subfolder under [`agent_wiki/audit/simulation_run_audits/`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) (`DD-MM-YYYY_<run_name>/`) with date format `DD-MM-YYYY`, explicit verdict, actionable proposal, diagnostic artifacts, and linked relevant files.
- **Protocol**:
  1. **Date & Directory Naming**: Every subfolder MUST be prefixed with `DD-MM-YYYY` (e.g. `24-09-2026_single_simulation_baseline/`).
  2. **Preserve Run Artifacts**: Copy key diagnostic plots (`*.png`), stream tables (`summary_*.csv`), and execution manifests (`run_manifest.json`) from `logs/` into the subfolder.
  3. **Verdict Requirement**: Assign one of the standard verdicts in `audit.md`: `PASSED`, `ACCEPTABLE WITH CONDITIONS`, `FLAGGED`, or `FAILED`.
  4. **Proposal Requirement**: Formulate a clear, actionable proposal detailing recommendations for model tuning, physical parameter adjustments, search bound updates, or operational strategies.
  5. **Relevant Files**: Provide markdown links (`file:///...`) to input configuration files, engine modules, test scripts, generated profile outputs, and logs.
  6. **Past Runs Index**: Register the entry in the Master Simulation Run Audits Index table in [`index.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/simulation_run_audits/index.md) so historical runs and recommendations can be audited across development sessions.


