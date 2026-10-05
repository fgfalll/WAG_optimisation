# Top 61 Traps & Common Pitfalls for AI Agents

This document highlights the most frequent misconceptions, resolved gotchas, and traps encountered when working with this codebase.

> [!IMPORTANT]
> **Read first вЂ” 2026-10-04 independent audit.** This list predates an audit that registered **53 findings
> (13 CRITICAL, 19 HIGH, 16 MEDIUM, 5 LOW; 1 already RESOLVED, HIGH-10)**, of which only a subset is represented above. Before editing any
> scientific code, consult:
> - [`audit/scientific_flaws.md`](../../audit/scientific_flaws.md) вЂ” the master flaw register (severity, category, `file:line`, evidence, remediation order);
> - [`res_audit.md`](../../res_audit.md) вЂ” software quality, anti-patterns, toolchain record, change-safety rows;
> - [`phd_audit.md`](../../phd_audit.md) вЂ” physical/mathematical reconstruction; predictive validity for the active `hybrid` path is **NOT ESTABLISHED**;
> - [`audit/parameter_provenance.csv`](../../audit/parameter_provenance.csv) вЂ” 91 parameters; **48 (53 %) are `UNKNOWN вЂ” EVIDENCE REQUIRED`**.
>
> Traps **50вЂ“53** below were added from that audit (53 covers HIGH-19, registered 2026-10-04 after the workbench commit).

---

### 1. Modifying `compositional_engine` Expecting Application Results to Change
- **Trap**: An agent is asked to "improve the simulation engine" and spends time refactoring `core/compositional_engine/`.
- **Reality**: `EngineFactory` hardwires all evaluations to `SurrogateEngineWrapper`. Changes to `compositional_engine` have **zero effect** on the main application, optimization runs, or GUI results.

### 2. Modifying `core/simulation/profile_generator.py` Instead of `profile_generator_fast.py`
- **Trap**: Editing the legacy `ProfileGenerator` in `core/simulation/`.
- **Reality**: `FastProfileGenerator` in `core/engine_surrogate/` is the authoritative source of truth for rate profiles.

### 3. Calling `evaluate_scenario()` Without `economic_params` [RESOLVED]
- **Trap**: Calling `engine.evaluate_scenario(reservoir, eor, ops)` without passing `economic_params`.
- **Historical Reality**: Crashed with `ValueError: economic_params is required for NPV calculation`.
- **Resolution**: `evaluate_scenario()` now provides `EconomicParameters()` by default when omitted.

### 4. Assuming `injection_profile` is in Reservoir Barrels
- **Trap**: Reading `profile_result["injection_profile"]` and treating it directly as res-bbl/day.
- **Reality**: It is generated in **MSCFD**. You must convert it using $B_g$ ($1000 \times B_g\text{ RB/MSCF}$) to obtain reservoir volume terms.

### 5. Trusting the `tanh` Miscibility Documentation in `PhDHybridSurrogate`
- **Trap**: Believing the docstrings and validation report claiming that `PhDHybridSurrogate` is smooth and continuously differentiable at MMP.
- **Reality**: Lines 709вЂ“712 execute a sharp piecewise exponential kink: `if pressure >= mmp: 1 - exp(...) else: 0.0`.

### 6. The Hardcoded 0.80 Recovery Factor Ceiling
- **Trap**: Expecting recovery factor to reach 0.85вЂ“0.90 during extensive high-pressure miscible COв‚‚ floods.
- **Reality**: `analytical_models.py:198` forces `np.clip(rf, 0.05, 0.80)`.

### 7. Editing `core/objectives/economic.py` for NPV Changes
- **Trap**: Modifying `core/objectives/economic.py` to change NPV calculation logic.
- **Reality**: NPV is written **inline** in `PhDHybridSurrogate.predict()`
  (`core/engine_surrogate/surrogate_models.py:507-530`) and republished as the profile key `npv`
  (`surrogate_engine.py:172`, `:692`); `core/objectives/wrapper.py:50-53` only consumes it and
  subtracts containment/remediation penalties. `core/objectives/economic.py` is imported by
  `core/objectives/__init__.py` but is never called in the evaluation path.
  вљ пёЏ Corrected 2026-10-04 (MED-15): the previously cited `surrogate_engine._calculate_engine_npv`
  **does not exist** (grep в†’ 0 hits); neither do `_calculate_co2_purchased_recycled`,
  `_solve_pressure_ode` and `_calculate_pressure_profile`. Verify method names against
  `SurrogateEngine` before citing them вЂ” it exposes only `evaluate_scenario`, `_build_params_dict`,
  `_error_result`, `get_performance_stats`, `reset_performance_stats`.

### 8. `UnboundLocalError` on `econ_params` [RESOLVED]
- **Trap**: Testing the fallback path in `OptimizationEngine.evaluate_for_analysis()` when `simulation_engine is None`.
- **Historical Reality**: Crashed with `UnboundLocalError: cannot access local variable 'econ_params'` because it was only bound in the non-None branch.
- **Resolution**: Variable scoping was unified across all evaluation branches.

### 9. Ambiguous Truth Value on Numpy Profile Arrays [RESOLVED]
- **Trap**: Writing `if profiles:` or `elif storage_params and profiles:`.
- **Historical Reality**: If `profiles` contains numpy arrays, Python raises `ValueError: The truth value of an array with more than one element is ambiguous`.
- **Resolution**: Refactored to explicit checks: `if profiles is not None and len(profiles) > 0:`.

### 10. Assuming the Engine Selection UI Dropdown Switches Solvers
- **Trap**: Believing the user can select "Compositional" or "Unified" in `EngineSelectionWidget`.
- **Reality**: `EngineFactory.get_available_engines()` returns `{"surrogate": True}` and `switch_engine()` logs a deprecation warning and returns the surrogate engine.

### 11. Missing `hypothesis` and `h5py` in Environment
- **Trap**: Running `pytest` in a minimal virtual environment without `--with hypothesis --with h5py`.
- **Reality**: Test collection fails immediately with `ModuleNotFoundError`. Always ensure the virtual environment has installed the full `requirements.txt`.

### 12. Syntax Error in `spe5_benchmark_validation.py` [RESOLVED]
- **Trap**: Running the SPE 5 benchmark script.
- **Historical Reality**: Line 249 had a missing comma in parameter definitions that prevented compilation.
- **Resolution**: Repaired comma syntax error.

### 13. Double-Counting Recycled COв‚‚ in Material Balance [RESOLVED]
- **Trap**: Using `analysis/material_balance.py` to verify net stored COв‚‚.
- **Historical Reality**: Recycled COв‚‚ was subtracted twice, causing net storage to evaluate artificially negative.
- **Resolution**: Removed redundant second subtraction; mass balance closes to machine precision.

### 14. Inconsistent Surface COв‚‚ Density Constant
- **Trap**: Assuming `PhysicalConstants.CO2_DENSITY_TONNE_PER_MSCF` is used everywhere.
- **Reality**: `PhysicalConstants` defines `0.05254`, but formulas across `analytical_models.py` and `surrogate_engine.py` hardcode `0.053`.

### 15. Moving Inline Imports to Module Level
- **Trap**: Tidying up inline imports in `engine_selection_widget.py` or `optimisation_engine.py` to top of file.
- **Reality**: Triggers circular import deadlocks between `core.engine_factory`, `core.optimisation_engine`, and `ui`.

### 16. NumPy 2.0 `np.trapz` Removal
- **Trap**: Using `np.trapz(y, x)` for profile integration (e.g. cumulative production).
- **Reality**: NumPy 2.0 removed `np.trapz`, causing immediate runtime `AttributeError: module 'numpy' has no attribute 'trapz'`. Always use `scipy.integrate.trapezoid` or `scipy.integrate.cumulative_trapezoid(y, x, initial=0.0)`.

### 17. Single-Producer Well Setup Downgrading Injection Schemes
- **Trap**: Assuming that if the user only inputs producer wells (`n_injectors == 0`), the simulation will still inject COв‚‚.
- **Reality**: Naive checks would see `n_injectors == 0` and downgrade the entire project to primary depletion, ignoring the user's selected WAG or continuous COв‚‚ scheme. `OptimizationEngine` now falls back to `n_injectors = 1` for active injection schemes to represent field-wide pattern injection.

### 18. Grid Block Metric/Field Unit Confusion (35.3x OOIP Inflation)
- **Trap**: Assuming `DataManagementWidget` block sizes `dx, dy, dz` are in meters and dividing by $0.3048$ and $4046.86$.
- **Reality**: Block sizes are in **feet**. Dividing by metric factors inflated reservoir volumes and OOIP by $35.314\times$ ($171\text{M STB}$ vs $4.86\text{M STB}$), creating unphysical multi-billion-dollar NPV estimates. Use $43,560\text{ ft}^2/\text{acre}$ and direct foot thickness.

### 19. `SurrogateEngine._evaluate_primary_production` Method Scope
- **Trap**: Adding internal engine calculation methods to `SurrogateEngineWrapper` instead of `SurrogateEngine`.
- **Reality**: Internal engine methods like `evaluate_scenario()` execute on `self` (`SurrogateEngine`), so moving methods exclusively to the wrapper causes `AttributeError: 'SurrogateEngine' object has no attribute '_evaluate_primary_production'`. Implement methods on `SurrogateEngine` and delegate from the wrapper.

### 20. 12Г— Recovery Factor Discrepancy in Simulation Validation [RESOLVED]
- **Trap**: In `analysis/data_validation.py`, integrating oil production rates assuming daily rates are sampled annually ($dt = 365.25\text{ days}$).
- **Historical Reality**: When `FastProfileGenerator` outputs 181 monthly rate points (15 years $\times 12 + 1$), integrating with $dt = 365.25$ instead of $dt = 30.4375\text{ days}$ caused calculated recovery factor to be exactly $12\times$ higher than reported (e.g. reported $0.390$ vs calculated $4.680$), failing validation checks.
- **Resolution**: `DataValidator.validate_simulation_results()` now detects sub-annual/monthly time steps from time vectors or array length and prefers direct cumulative production volumes (`cumulative_oil`) when available.

### 21. Profile Array Truncation to 15 Flat Bars [RESOLVED]
- **Trap**: Storing raw monthly arrays (181 elements) directly into `yearly_oil_stb` while setting `yearly_time_years` to 15 elements ($1 \dots 15$).
- **Historical Reality**: The plotting and export logic zipped or sliced the array to match `yearly_time_years`, truncating the 15-year simulation to the first 15 months of plateau data, displaying 15 identical flat bars.
- **Resolution**: `OptimizationEngine.evaluate_for_analysis()` now integrates monthly profiles into 15 true annual totals (`annual_oil_stb`, `yearly_oil_stb`), preserving full 181 monthly points in `monthly_oil_stb` and rate keys.

### 22. Breakthrough GOR Unit Inflation in Recycled COв‚‚ [RESOLVED]
- **Trap**: Calculating breakthrough recycled COв‚‚ using GOR in SCF/STB without dividing by 1,000 MSCF.
- **Historical Reality**: `calculate_breakthrough_aware_recycling` multiplied cumulative oil tonnes by raw GOR, producing an unphysical 21,200,000 tonnes of recycled COв‚‚ (exceeding total injected COв‚‚ by orders of magnitude).
- **Resolution**: GOR in SCF/STB is converted to MSCF by dividing by 1,000 before multiplying by COв‚‚ density, and recycled COв‚‚ is strictly capped by physical mass conservation: $\text{CO}_{2,\text{recycled}} \le \text{CO}_{2,\text{produced}}$.

### 23. COв‚‚ Utilization Fallback Penalty 1,000,000.00 [RESOLVED]
- **Trap**: Assuming objective wrappers and engines use identical dictionary key naming for annual profiles.
- **Historical Reality**: `core/objectives/wrapper.py` searched only for `"annual_co2_purchased_mscf"`, while the engine output `"yearly_co2_purchased_mscf"`. The wrapper failed silently and returned the fallback penalty $1,000,000.00$ for COв‚‚ utilization.
- **Resolution**: Both `annual_` and `yearly_` aliases are provided in engine profiles, and `wrapper.py` checks both keys.

### 24. Decline Curve Analysis Crash on Dictionary Results [RESOLVED]
- **Trap**: Assuming `DeclineCurveAnalyzer.plot_decline_curve()` only receives `DCAResult` dataclass instances.
- **Historical Reality**: The UI and optimization workers serialize DCA results as Python dictionaries, causing `plot_decline_curve` to crash with `'dict' object has no attribute 'time'`.
- **Resolution**: `plot_decline_curve` now transparently accepts both `DCAResult` dataclasses and dictionaries containing DCA forecast arrays.

### 25. Deliverability Producer Count Multiplication Bug [RESOLVED]
- **Trap**: In `FastProfileGenerator.generate_oil_profile`, calculating `peak_rate_estimate = (ultimate_recovery / ...) * n_producers`.
- **Historical Reality**: `ultimate_recovery` is already the **field-total** recoverable volume ($OOIP \times RF$). Multiplying by `n_producers` artificially inflated peak rates to 14,300 STB/d (producing 10.8% of OOIP per year from 2 wells with $PI = 5.0$). Furthermore, `elif ultimate_recovery > 0` normalization was skipped when user max rate was set, decoupling rate from volume.
- **Resolution**: Removed `* n_producers` multiplier, introduced Composite Vogel-Darcy IPR deliverability upper limit ($q_{\text{field,max}} = n_{\text{producers}} \cdot q_{\text{ipr}}(P_{\text{wf}})$), and enforced strict cumulative volume normalization so instantaneous rates and cumulative recovery remain in exact physical balance.

### 26. Unbounded Injection Pressure & Caprock Rupture [RESOLVED]
- **Trap**: Setting the optimizer parameter upper bound for `pressure` directly to `caprock_fracture_pressure_psi` (5,500 psi).
- **Historical Reality**: The optimizer pushed reservoir pressure to 5,269 psi (leaving an unsafe 230 psi margin). With near-wellbore injection overpressure $\Delta P_{\text{inj}} = q_{\text{inj}} / II \approx 200\text{ psi}$, sandface pressure shattered the seal and leaked 1,041,953 tonnes of COв‚‚. With zero carbon tax, the optimizer treated massive caprock blowouts as free energy.
- **Resolution**: Injection pressure upper bound is now legally constrained by EPA Class VI 90% UIC safety rules ($P_{\text{res,max}} \le 0.90 \times P_{\text{frac}} - \Delta P_{\text{inj}}$), and a mandatory $\$100/\text{tonne}$ environmental remediation penalty is applied to any leaked COв‚‚ regardless of carbon tax settings.

### 27. Continuous Injection Gene Space Contamination [RESOLVED]
- **Trap**: Unconditionally adding cyclic `huff_n_puff`, `wag`, and `tapered` parameters into the GA gene space when injection scheme is locked to `continuous`.
- **Historical Reality**: The optimizer wasted fitness evaluations tuning soak days, cycle counts, and slug tapers that were never executed, while the report generator displayed active parameters for inactive schemes.
- **Resolution**: `_get_parameter_bounds()` conditionally prunes cyclic parameters based on the active injection scheme, restricting the search space strictly to relevant physics.

### 28. Floating-Point Return Values for Discrete Integer Operational Modes [RESOLVED]
- **Trap**: Allowing continuous GA/BO algorithms to output floating-point values for integer operational flags like `shut_in_mode = 0.3386` and `allow_well_conversion = 0.892`.
- **Historical Reality**: Operators were presented with fractional modes that make no operational sense.
- **Resolution**: Gene spaces for discrete parameters use explicit integer steps, and `_sanitize_and_discretize_parameters()` strictly rounds operational flags to integer levels (`0`, `1`, `2`).

### 29. Class E Artificial Storage Modifiers & Penalty Dilution Hacks [RESOLVED]
- **Trap**: Synthesizing artificial storage efficiency via `default_efficiency = max(0.3, 0.5 * (RF / 0.35))` or diluting failure penalties (`FAILURE_PENALTY * 0.1`).
- **Historical Reality**: This awarded ~50% COв‚‚ storage credit to primary depletion runs and allowed invalid chromosomes to survive in the GA population.
- **Resolution**: Eradicated all Class E modifiers and penalty multipliers. Missing data returns `NaN`, and unphysical profiles receive full `FAILURE_PENALTY` ($-10^{12}$), naturally killing off unviable genetic lines.

### 30. Silent Exception Swallowing (`except Exception: pass`) in Core Math Modules [RESOLVED]
- **Trap**: Using bare `except:` or `except Exception: pass` to suppress convergence or numerical failures in flow solvers, flash calculations, or EOS evaluations.
- **Historical Reality**: Masked catastrophic numerical divergence, division by zero, or bad fluid properties, causing the engine to operate on uninitialized variables.
- **Resolution**: Replaced with specific exception types (`FloatingPointError`, `ZeroDivisionError`, `ValueError`, `RuntimeError`), logging exact state variables (Pressure, Saturation, Temperature, Composition).

### 31. Plotly Mock Classes Swallowing Chart Generation [RESOLVED]
- **Trap**: Defining mock `class go: class Figure: pass` when Plotly is missing.
- **Historical Reality**: Silently swallowed interactive charts and decline curve fits, preventing engineers from performing QA/QC on simulation results.
- **Resolution**: Eradicated all mock classes. Plotly is a strict mandatory dependency enforced at startup.

### 32. Plotly Vertical Bar Geometry Trap (Collapsed Schedule Timelines) [RESOLVED]
- **Trap**: In vertical Plotly bar charts (`go.Bar`), treating `x` as the duration and `base` as the start time (e.g. `x=[duration_days]`, `base=[start_day]`, `barmode="stack"`).
- **Historical Reality**: In Plotly vertical bars, `x` is the horizontal position along the X-axis, while `base` is the vertical offset along the Y-axis. This stacked all operations at a single point ($x = 45\text{ days}$) with tiny bar widths (0.8 days spanning 44.6 to 45.4 days), completely collapsing multi-year schedules into a hairline sliver.
- **Resolution**: Compute true horizontal interval spans: `x = [start_day + duration_days / 2.0]`, `width = [duration_days]`, `base = [0]`, `y = [rate]`, with `barmode = "overlay"`.

### 33. Legend Pollution via Cyclic Traces (Missing Legend Deduplication) [RESOLVED]
- **Trap**: Appending bar traces in a loop for each operational cycle with `name=phase` and expecting Plotly's `legendgroup` to hide duplicates.
- **Historical Reality**: `legendgroup` ties visibility toggling together, but does *not* suppress duplicate legend entries if `showlegend=True` on multiple traces. A 10-cycle schedule produced 10 duplicate "production" legend entries.
- **Resolution**: Maintain an explicit `added_legend_phases = set()` during trace generation and set `showlegend=(phase not in added_legend_phases)`.

### 34. Invisible Zero-Rate Operational Phases [RESOLVED]
- **Trap**: Representing soak, shut-in, and idle phases solely with `y=[0.0]`.
- **Historical Reality**: Zero-height bars are physically invisible in Plotly, misleading operators into thinking idle or soaking periods were deleted or skipped.
- **Resolution**: Render zero-rate phases with distinct hatched pattern bars (`marker_pattern_shape="/"`, opacity=0.45) accompanied by diamond markers on the baseline ($y = 0$).

### 35. Artificial 10-Cycle Cap in Continuous Production (`max_cycles = 10`) [RESOLVED]
- **Trap**: Reusing cyclic WAG looping logic (`while day < project_life_days and cycle < int(max_cycles)`) for continuous production.
- **Historical Reality**: For standard 45-day operational steps, continuous production was prematurely halted after 450 days (1.2 years) instead of spanning the entire field life (15+ years).
- **Resolution**: Removed the 10-cycle cap in `_ops_standard_production` so continuous and WAG production span the full project lifetime.

### 36. Closed-Loop COв‚‚ Mass Balance Trap (Purchased vs Gross Injected) [RESOLVED]
- **Trap**: Comparing gross accounted COв‚‚ ($\text{Stored} + \text{Leakage} + \text{Gross Produced}$) directly against **purchased** fresh COв‚‚.
- **Historical Reality**: In a closed-loop recycling flood, $\text{Gross Injected} = \text{Purchased} + \text{Recycled}$. Comparing gross accounted volume ($1,452,195\text{ t}$) against purchased gas ($740,490\text{ t}$) resulted in an apparent discrepancy of exactly $711,705.5\text{ t}$ (the recycled stream), erroneously flagging a mass-conserving simulation as having a 50.99% error.
- **Resolution**: Formulated mass balance strictly on a closed-loop gross basis: $\text{Gross Injected} = \text{Net Stored} + \text{Total Leakage} + \text{Gross Produced}$. Closure $> 99.9\%$ is verified.

### 37. Decline Curve Analysis Plateau Regression Breakdown ($R^2 < 0$) [RESOLVED]
- **Trap**: Fitting Arps decline models (exponential/hyperbolic) across the entire production history including multi-year flat plateaus.
- **Historical Reality**: Arps formulations assume boundary-dominated decline ($dq/dt < 0$). Fitting across a 3-year flat plateau followed by steep decline produced severe mathematical regression breakdown ($R^2 = -3.14$) and unphysical EUR overestimation (72% OOIP).
- **Resolution**: Added decline onset detection ($q(t) < 0.95 \times q_{\text{peak}}$). If a multi-year plateau exists, Arps models are fitted strictly to the declining segment $(t - t_{\text{onset}})$, yielding $R^2 > 0.95$ and physically bounded EUR forecasts.

### 38. Ecology Violation via Incomplete Stream Shut-In [RESOLVED]
- **Trap**: Only zeroing hydrocarbon gas and water streams upon well shut-in.
- **Historical Reality**: In `surrogate_engine._apply_shut_in_guard`, checks looked only for `"hydrocarbon_gas_production_rate"` and `"water_production_rate"`, ignoring `co2_gas_profile` and `solution_gas_profile`. Breakthrough COв‚‚ continued flowing at full rate for 10+ years post shut-in ($444,672\text{ t}$), generating hundreds of thousands of dollars in false ecology penalties.
- **Resolution**: Extended shut-in guards to taper and zero all phase streams: `water_profile`, `co2_gas_profile`, `solution_gas_profile`, `gas_profile`, and `injection_profile`.

### 39. Manual Well Dialog Role Inversion via Substring Matching [RESOLVED]
- **Trap**: Checking `status in ["active", "inactive"]` when UI status dropdown strings are `"Producer (Active)"` or `"Producer (Inactive)"`.
- **Historical Reality**: The substring check failed, defaulting well names to `"Well-Injector-1"`, while backend code tagged the metadata as `"producer"`. This caused injector-producer role inversion, leading the optimizer to run with 0 injectors.
- **Resolution**: Fixed combo text parsing, implemented bidirectional live typing synchronization between well name and role dropdown, and added name-based role fallback inference.

### 40. The 1,350 psi Pressure Decoupling Trap (Optimizer Variable vs Tank ODE)
- **Trap**: Believing the optimizer decision variable `pressure` reflects the actual physical reservoir operating pressure.
- **Reality**: `pressure` is an independent search parameter that `PhDHybridSurrogate` uses to calculate recovery factor and miscibility weight $\omega$. If the optimizer drives `pressure` to the upper bound ($4,450\text{ psia}$), recovery is evaluated as fully miscible at $4,450\text{ psia}$ even though the stiff material balance tank ODE simulates dynamic reservoir pressure between $3,080\text{ psia}$ and $3,335\text{ psia}$. Always check `summary_yearly.csv` or `monthly_pressure` for true physical pressure.

### 41. Catastrophic Breakthrough Penalty Wall (-1.0e12) Inducing Rate Floor Collapse
- **Trap**: Wondering why the optimizer always pins injection rate to the minimum search bound ($5,000\text{ MSCFD}$).
- **Reality**: The objective evaluator applies an unphysical hard step penalty (`-1.0e12`) whenever solvent breakthrough occurs earlier than 1.0 year. In single-well pattern configurations, any realistic field injection rate breaks through in $< 1\text{ year}$, triggering this failure wall and driving the optimizer straight into the lower rate bound.

### 42. Phantom Net Utilization (< 1 MSCF/STB) from Uncoupled Koval Surrogate
- **Trap**: Accepting simulation outputs showing net $\text{CO}_2$ utilization $< 1.0\text{ MSCF/STB}$ ($< 0.05\text{ tonne/STB}$).
- **Reality**: Published U.S. DOE/NETL benchmarks and empirical Permian Basin data establish typical net utilization at $5.0 \text{ to } 12.0\text{ MSCF/STB}$ ($0.25 \text{ to } 0.60\text{ tonne/STB}$). Net utilization $< 1.0\text{ MSCF/STB}$ indicates that the analytical recovery formula is decoupled from pore volume injection throughput, allowing the optimizer to claim field-scale recovery while injecting almost no solvent.

### 43. The Single-Well County Drain Anomaly (Field-Wide Point Voidage)
- **Trap**: Assuming field-wide injection fallbacks automatically scale well counts to maintain physical Darcy drawdowns.
- **Reality**: If the user does not define explicit injector patterns, `OptimizationEngine` falls back to `n_injectors = 1` and `n_producers = 1`. In large reservoirs (e.g. 1,354 acres, 48.5 MMbbl OOIP), this routes the entire field production ($20.3\text{ MMbbl}$) through a single producer well, sustaining an unphysical $7,270\text{ BOPD}$ plateau in a $100\text{ mD}$ formation without accounting for pattern spacing or multi-well interference.

### 44. Static Flat CAPEX Distorting Project Economics ($5M for $878M NPV)
- **Trap**: Relying on the default static \$5.0M CAPEX for field-scale EOR project evaluation.
- **Reality**: A static \$5.0M CAPEX ignores well drilling/conversion costs ($\$1.0\text{M} - \$2.0\text{M}$ per well) and gas recycling facility scaling (which scales with peak recycle capacity, $Q_{\text{peak}}^{0.65}$). On a 1,354-acre field with 20вЂ“30 wells and a gas processing plant, true CAPEX is $\$60\text{M} \text{ to } \$120\text{M}+$. A flat \$5M CAPEX produces fictitious economics (11-day payback, $\$878\text{M}$ NPV).

### 45. Omitted Signals (`engine_selection_changed`) During ConfigWidget UI Pruning [RESOLVED]
- **Trap**: Deleting signals or UI hooks from `ConfigWidget` when pruning unused fields and dropdowns.
- **Historical Reality**: In `ui/config_widget.py`, replacing the multi-engine dropdown with a static surrogate badge omitted `engine_selection_changed = pyqtSignal(str)`. When `MainWindow._setup_main_app_tabs_container()` connected to this signal, the app crashed on boot with `AttributeError`.
- **Resolution**: Kept `engine_selection_changed = pyqtSignal(str)` on `ConfigWidget` and added `hasattr` defensive guards in `ui/main_window.py`.

### 46. Root `config_manager` vs `utils.config_manager` Import Path [RESOLVED]
- **Trap**: Using `from config_manager import ConfigManager` in UI or worker modules.
- **Historical Reality**: Following root module restructuring, `ConfigManager` resides strictly in `utils/config_manager.py`. `ui/data_management_widget.py` had an outdated root import that failed, setting `PreferencesManager = None` and `ConfigManager = None` and disabling unit preferences.
- **Resolution**: Updated all import locations to `from utils.config_manager import ConfigManager` and added `set_engine_type` compatibility method on `DataManagementWidget`.

### 47. Missing Qt Translation Files Triggering Spurious Warnings [RESOLVED]
- **Trap**: Attempting to load missing `.qm` translation files and logging warnings on every non-English system launch.
- **Historical Reality**: `translations/` only contained empty placeholder `.ts` files. On Ukrainian (`uk`) system locales, `I18nManager` logged repeated `WARNING` messages for missing `app_uk.qm` and missing `app_en.qm`, despite English already being the hardcoded native in-code language.
- **Resolution**: Added `translation_path.is_file()` existence guard in `load_and_install_translator`. Missing translation binaries log at `DEBUG` level and fall back cleanly to native in-code strings without warnings.

### 48. Assuming Application Startup & UI Are Covered by Engine Tests [RESOLVED]
- **Trap**: Assuming that a 100% pass rate in `pytest tests/` guarantees the GUI application (`python main.py`) will boot.
- **Historical Reality**: The test suite focused exclusively on mathematical identities, mass conservation, physical bounds, and surrogate engine calculations. Neither `main.py`, `MainWindow`, nor `analysis/sensitivity_analyzer.py` were imported by any test in `tests/`. Relocating `core/simulation/recovery_models.py` caused `main.py` to crash on startup with `ModuleNotFoundError` during `timed_import_main_window()` despite 307 passing tests.
- **Resolution**: Added transparent backward-compatible deprecation shims (`core/simulation/recovery_models.py` and `profile_generator.py`) and created [tests/test_app_startup.py](file:///d:/rep/4.6/co2eor_optimizer/tests/test_app_startup.py) which explicitly executes `timed_import_main_window()`, `SensitivityAnalyzer`, and `ProductionProfiler` under `QApplication`, permanently guarding against startup import regressions.

### 49. Project Save/Load Silent Breakage: Stripping Dataclass Tags and 1D Grid Indexing Errors [RESOLVED]
- **Trap**: Assuming that saving a project with `ProjectEncoder` or loading it with `load_project_data()` works without verifying nested dataclasses or flattened grid arrays.
- **Historical Reality**:
  1. `ProjectEncoder.default` called `dataclasses.asdict(o)`, recursively converting nested dataclasses (`eos_model`, `layer_definitions`, `geostatistical_params`) into plain dicts and stripping `_dataclass` tags, corrupting objects on reload.
  2. `DataManagementWidget.load_project_data()` indexed `grid['PERMX'][0,0,0]`. When loading grids generated by `data_integration_engine.py` (which flatten properties into 1D arrays), Python raised an unhandled `IndexError: too many indices for array`, instantly aborting project loading and leaving UI input widgets blank.
  3. `OptimizationEngine.results` was defined as a `@property` without `@results.setter`, causing `self.optimisation_engine_instance.results = optimization_results` to raise `AttributeError: can't set attribute 'results'`.
  4. `MainWindow._perform_project_load()` evaluated `locals().get("stacked_layout_index", 0)`, leaving the user stranded on the welcome screen because `stacked_layout_index` was never in `locals()`.
- **Resolution**:
  - Refactored `ProjectEncoder` to perform shallow dictionary conversion across dataclass fields so nested objects retain `_dataclass` tags.
  - Added backwards-compatibility handling in `project_decoder` to automatically reconstruct typed dataclasses from legacy untyped dicts.
  - Used `.flat[0]` for safe permeability extraction across scalar, 1D, and 3D arrays in `load_project_data()`.
  - Added `@results.setter` to `OptimizationEngine.results`.
  - Added [tests/test_project_save_load.py](file:///d:/rep/4.6/co2eor_optimizer/tests/test_project_save_load.py) covering end-to-end roundtrip serialization, legacy compatibility, and UI restoration.

### 50. Treating a Green `pytest` as Analytical Verification
- **Trap**: Reading `36 passed` in `tests/scientific/` (or `333 passed / 0 failed` overall) as evidence that the physics is verified.
- **Reality**: 5 of the 36 scientific tests import production symbols they **never call** вЂ” `test_koval_fractional_flow_mobility_monotonicity` imports `FastProfileGenerator` and then re-derives `profile_generator_fast.py:958-969` by hand at `:38-46` вЂ” and 3 tests assert **that a defect exists** (`step > 0.40`, `eff_high_sgc < eff_low_sgc`, `truncation_fraction > 0.20`), so *fixing* the physics turns the suite red. 6 more never import production code at all. Consequence: the *Analytical Verification* rung of the evidence hierarchy reports PASS for items where the test either restates the code or pins the wrong answer. **HIGH-18.**
- **Rule**: Before trusting any test, confirm it (a) executes the symbol under test, (b) asserts the physically correct value, and (c) would fail if the defect were introduced *or* removed.

### 51. Believing the Verification Matrix (`verification/test_matrix.md`)
- **Trap**: Using the master test matrix to decide what is covered.
- **Reality**: It declares **42 test items across 16 subdirectories**; `pytest --collect-only tests/scientific` returns **36 in 15**. Six of its listed tests **do not exist anywhere in `tests/`** (`test_co2_density_thermal_expansion`, `test_cubic_eos_z_factor_bounds`, `test_phase_label_assignment`, `test_peng_robinson_fugacity_equation_structure`, `test_corey_relative_permeability_bounds`, `test_bg_discrepancy_between_modules`) and five of those cite the deleted `unified_engine/` tree; one was renamed (`вЂ¦_mobility_inversion` в†’ `вЂ¦_monotonicity`); two exist but are unlisted. **MED-16.**
- **Rule**: Regenerate coverage claims from `pytest --collect-only`, never by hand; verify every cited path with `Test-Path`.

### 52. Assuming Wiki-Stated Invariants Are Enforced by Code
- **Trap**: Reading `README.md` invariants #3/#10 (mass conservation, closed-loop carbon accounting) as runtime guarantees.
- **Reality**: $M_{\text{recycled}} \le M_{\text{produced}}$ holds **by construction**, not by enforcement вЂ” recycled is capped as `min(prodВ·О·_recycle, inj)` at `optimisation_engine.py:871-872`, so no code asserts it; and the second half is uncheckable because `total_leakage_tonne в‰Ў 0.0` (CRIT-10, HIGH-05) while `cum_stored = inj в€’ prod` (`surrogate_engine.py:543`) **ignores leakage entirely**. Measured on a 10-yr run: inj 36 525 000 MSCF = purchased 34 994 496 + recycled 1 530 504 (exact) вњ“.
- **Rule**: An invariant that cannot be violated is not an invariant that has been tested. Distinguish *holds by construction* from *asserted* вЂ” see `res_audit.md` В§2.4.

### 53. Treating a Fixed Defect as a Fixed Class
- **Trap**: Reading `333 passed / 0 failed` (and the HIGH-10 fix) as "the missing-import problem is gone".
- **Reality**: HIGH-10 was one missing `from PyQt6.QtGui import QIcon`. Two commits later the **same defect class** is back in `ui/workbench/`: `pyvista_reservoir_canvas.py:974` uses `has_active_fault`, which is bound nowhere in the repository, and `subsurface_data_viewer_widget.py:335` uses `QToolTip`, which is never imported. The first is swallowed by `except Exception` at `:1272`, so the 3-D caprock view silently renders nothing when the checkbox (turned on automatically by the `caprock` / `caprock_3d` tree entries at `subsurface_workbench_widget.py:755`, `:784`) is enabled; the second raises inside the `btn_copy_table.clicked` slot, which has no guard. Neither is reached by any test, so the suite still reports 0 failures. **HIGH-19.**
- **Rule**: `F821` is a release gate, not a lint preference вЂ” run `ruff check --select F821` before committing UI code, and never close a defect without asking whether the *pattern* still exists elsewhere. Note the counter-example: `F821` on `petrophysical_distribution.py:404` (`prev_field`) is a **false positive** (bound at `:405`, read only under `if k > 0`) вЂ” see `audit/scientific_flaws.md` В§6; a blanket "fix every F821" rule would corrupt working control flow.

### 54. Believing a "RESOLVED" Mark Before the Remediation Was Audited
- **Trap**: Reading `audit/scientific_flaws.md` line `**Status:** RESOLVED вЂ” Remediated in <file>:<line>` and treating the finding as closed.
- **Reality:** The 05-10-2026 remediation was applied **and marked RESOLVED in the same uncommitted working tree**. A round-2 adversarial re-audit of those 17 claims found **11 CONFIRMED В· 4 PARTIALLY RESOLVED В· 1 REGRESSED (CRIT-12, saturation closure now broken) В· 1 CONFIRMED-but-inert (CRIT-06, pinned at its clip)**, and registered **11 new findings**. Two of the "fixes" (CRIT-19 `gravity_factor`, CRIT-20 miscibility sigmoid) *converted harmless dead code into live un-calibrated physics*.
- **Rule:** A remediation is closed only after a **fresh measurement** shows the defect is gone вЂ” not after the edit lands. See `agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md` В§2 for the verification matrix.

### 55. Adding a Gene to the Physics Without a Physical Definition
- **Trap**: Wiring an inert optimizer gene (`gravity_factor`) into `e_v`, `vertical_eff` and `N_g` because "it should influence the result".
- **Reality:** `gravity_factor` is dimensionless, has no unit and no citation (range 0.5вЂ“1.5, `data_models.py:1028-1029`). It now scales recovery by В±20 % in three places, twice within the same hybrid path. **CRIT-19.** Note the trap in the other direction too: `N_g` at `analytical_models.py:802` already computes gravity segregation from О”ПЃ, dip, permeability and Darcy velocity вЂ” the new gene duplicates a real physical term with a fitted one.
- **Rule:** A gene may enter an equation only with (a) a physical definition, (b) a unit, (c) a literature or derivation citation. Dead code is visible; fitted fudge is not.

### 56. Overriding a Computed Transport Property with a User Constant
- **Trap**: `if "mobility_ratio" in params: m_eff = params["mobility_ratio"] else: <compute from Ој>`.
- **Reality**: `_build_params_dict` **always** supplies the key (`EORParameters.mobility_ratio`, default 5.0), so the computed branch is dead and recovery becomes exactly flat in oil viscosity вЂ” measured RF identical to 6 decimals for Ој_o = 0.5 в†’ 100 cP. **CRIT-14.** The same pattern recurs in `getattr(reservoir_data, "bo_rb_per_stb", None) or <computed>` where the attribute **does not exist** at all (`hasattr` = False; the real field is `oil_fvf`) вЂ” **CRIT-16**, a re-introduction of HIGH-04.
- **Rule:** Never let a constant shadow a derived property. Assert the override is *reachable* (`hasattr` on the real dataclass field) and add a test that the derived property still responds to its own inputs.

### 57. Using Hydrocarbon Pore Volume Where Total Pore Volume Is Required
- **Trap**: `pv_ref = ooip * bo / (1 - S_wi)` (hydrocarbon PV) used as the denominator for **both** `S_o` and `S_w`.
- **Reality**: `S_wi` is defined on **total** PV, so the two terms live on different bases and `S_o + S_w` exceeds 1 вЂ” measured on **49 of 181 timesteps (27 %)**, max 1.0428, with `S_g` silently zeroed by `np.clip`. **CRIT-17.**
- **Rule:** Saturation bookkeeping must be written against one declared PV basis, and the run must assert `ОЈ S == 1` at every step. A `np.clip` that saturates on >5 % of steps is masking a basis error, not preventing one.

### 58. Paying a Storage Credit on Leakage-Blind Mass
- **Trap**: `annual_stored = max(0, inj - prod)` used both as the reported stored mass and as the revenue base for the storage credit, while the carbon-tax penalty uses `total_leakage_tonne`.
- **Reality**: `total_leakage_tonne в‰Ў 0.0` in every configuration probed, and the only wrapper path that could ever charge leakage (`wrapper.py:112-121`, using `leakage_rate_fraction`) is now **dead** because `total_leakage_tonne` is present. So the model **pays $25/tonne for COв‚‚ it cannot retain and charges $0 for the COв‚‚ it loses.** **HIGH-23.**
- **Rule:** `M_stored = M_inj в€’ M_prod в€’ M_leaked` everywhere. A containment failure must have a non-zero price, and wiki invariants #10/#11 are **not enforced by code** вЂ” do not cite them as guarantees.

### 59. Running Only the Test Suite and Calling a Fix Verified
- **Trap**: Fixing a physics defect, re-running `pytest`, seeing `335 passed`, and writing `Status: RESOLVED`.
- **Reality:** On 05-10-2026 seventeen findings were marked RESOLVED this way and the suite stayed green throughout. Adversarial re-audit found **8 requiring reopening**, including two *regressions* (CRIT-12 broke saturation closure; CRIT-14 severed viscosity from recovery). No test detects any of them.
- **Rule:** Run `python -m audit.continuity check` before writing a RESOLVED mark, and cite the measured value in the status line. See [`continuity_gate.md`](continuity_gate.md).

### 60. Letting a Guard Be the Only Thing Preventing a NameError
- **Trap**: `if RECOVERY_MODELS_AVAILABLE: self.model = MiscibleRecoveryModel(...)` where the flag is a hard-coded `False` and the class is never imported.
- **Reality:** `analytical_models.py:96`, `:223`, `:358`. All three fire `NameError` the moment anyone "enables the literature models". This is the HIGH-10 в†’ HIGH-19 в†’ HIGH-11 recurrence, now caught automatically by the gate's `F821` check. Note the fix may be to **delete** the dead branch, not to add a guard.
- **Rule:** `ruff check --select F821` is a release gate. Keep exactly one reviewed allowlist entry вЂ” `petrophysical_distribution.py:404` (`prev_field`, bound on the previous loop iteration) is a genuine false positive; everything else must be fixed.

### 61. Writing a Formula Into a Remediation Note That the Code Does Not Implement
- **Trap:** Documenting `HCPVI = QВ·Bg / (VpВ·SoiВ·Boi)` when the code divides by `(ooipВ·Bo)/(1в€’Swi)`.
- **Reality:** Both are wrong, and by different factors. The code's denominator reduces to `V_p` (total pore volume) instead of `V_p(1в€’S_wi)` вЂ” off by exactly `1в€’S_wi`. The written formula multiplies a pore volume by a formation volume factor, which adds a spurious `rb/STB`: **pore volume is already at reservoir conditions.** Neither matches the other. Issues [#17](https://github.com/fgfalll/WAG_optimisation/issues/17), [#18](https://github.com/fgfalll/WAG_optimisation/issues/18).
- **Rule:** Every formula in a wiki page or commit body must be reproducible from a one-line algebraic identity. If you cannot reduce it to units that cancel, it is wrong.
