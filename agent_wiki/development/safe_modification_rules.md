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

