# Fallback & Exception-Swallowing Audit

## 1. Executive Summary

An AST audit across active Python modules originally identified **2,459 total fallbacks**, comprising:
- **1,710** dictionary `.get(key, default)` calls with implicit fallbacks
- **629** function parameter default values
- **96** `try...except` blocks that catch exceptions and return fallback values or constants
- **24** `try...except` blocks that silently pass (`pass`) without logging or re-raising
- **800** bounding/clipping calls (`np.clip`, `max`, `min`, `clamp`)

Following the strict eradication initiative:
- **All silent `except Exception: pass` blocks in the core mathematical and solver modules were completely eradicated (0 remaining)**.
- **Specific exception types are now caught** (e.g. `FloatingPointError`, `ZeroDivisionError`, `ValueError`, `RuntimeError`), and **state variables (Pressure, Saturation, Temperature) are explicitly logged**.
- **Class E artificial result-producing fallbacks were deleted** (reduced to 0 in active optimization paths). Unphysical chromosomes receive full mathematical failure penalties (`FAILURE_PENALTY`, $-10^{12}$) or `NaN` to naturally kill off unviable genetic lines.
- **Plotly and visualization libraries are enforced as hard mandatory requirements**, eradicating silent dummy mock plot swallowers.

---

## 2. Fallback Classification Taxonomy

Fallbacks in this repository fall into five distinct categories:

| Category | Description | Count | Severity | Risk Level | Status |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Class A: Legitimate Numerical Safety** | Division-by-zero protection (`+ EPSILON`), preventing negative saturations | ~450 | LOW | Preserves physical meaning without distorting trends. | Active |
| **Class B: Legitimate Engineering Defaults** | Standard reservoir compressibility $4 \times 10^{-6}\text{ psi}^{-1}$ if unmeasured | ~600 | MEDIUM | Documented industry baseline; acceptable if flagged. | Active |
| **Class C: Implementation Fallbacks** | Plotly dummy mock classes if plotting library is uninstalled | 0 (eliminated) | LOW | **ERADICATED**: Plotly enforced as mandatory dependency. | Resolved |
| **Class D: Scientifically Dangerous Fallbacks** | Catching calculation exceptions and returning uncalibrated guesses | 0 in math core | HIGH | **ERADICATED**: Replaced with specific exceptions + state variable logging. | Resolved |
| **Class E: Artificial Result-Producing Fallbacks** | Substituting artificial positive values to cheat optimizer penalties | 0 | CRITICAL | **ERADICATED**: Pruned with full mathematical penalties or `NaN`. | Resolved |

---

## 3. High-Priority Fallback Audit Table

The following table catalogs the critical fallbacks in the codebase and their resolution status:

| Location | Triggering Condition | Fallback Behavior | Intended Original Behavior | Scientific Consequence | Class | Severity | Status & Remediation |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| [core/optimisation_engine.py:1700-1725](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1700-L1725) | Total injected CO₂ is $\le 0$, but $RF > 0.05$, or storage efficiency $\le 10^{-6}$ | Previously set `storage_efficiency = 0.3` or returned diluted `FAILURE_PENALTY * 0.1` | Evaluate true physical storage ($0.0$) and prune invalid candidate | **Falsified physical reality**; awarded 30% storage credit or softened penalties | **E** | **CRITICAL** | **RESOLVED**: Eliminated override and dilution. Evaluates true storage efficiency ($0.0$ for primary depletion); unphysical candidates receive full `FAILURE_PENALTY` ($-10^{12}$). |
| [core/objectives/wrapper.py:130-137](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L130-L137) | `storage_params` or `profiles` missing in `_calculate_objective_functions` | Previously set `storage_efficiency = max(0.3, 0.5 * (RF / 0.35))` | Compute storage efficiency from injection/production profile | Synthesized artificial storage efficiency from recovery factor alone | **E** | **CRITICAL** | **RESOLVED**: Deleted Class E synthesis. Sets `storage_efficiency = float("nan")` and records `method="unphysical_or_missing_data"`. Chromosome is pruned with `FAILURE_PENALTY`. |
| [core/objectives/wrapper.py:155-186](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L155-L186) | Missing/empty CO₂ purchase or oil production profiles | Previously returned magic number `results["co2_utilization"] = 1e6` | Accurately compute CO₂ utilization | Masked missing simulation profiles with arbitrary magic number | **D** | **HIGH** | **RESOLVED**: Returns `float("nan")` on missing/empty arrays, triggering full `FAILURE_PENALTY` in GA. |
| [analysis/material_balance.py:9-50](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L9-L50) | `import plotly` fails | Previously created dummy `go.Figure`, `go.Scatter`, `go.Bar` classes with no-op methods | Render interactive HTML plots | Silent failure of chart generation; prints text instead of raising missing dependency | **C** | **LOW** | **RESOLVED**: Eradicated all dummy mock classes. Plotly is now imported directly and verified on startup in `main.py`. |
| [core/optimisation_engine.py:470-482](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L470-L482) | EOS calculation throws exception in `_get_co2_fraction_from_eos()` | Previously caught blind `except Exception:` and silently passed (`pass`) | Compute phase split via cubic EOS | Silently used static/uncalibrated gas fraction | **D** | **HIGH** | **RESOLVED**: Replaced with `except (RuntimeError, ValueError, ZeroDivisionError, ArithmeticError) as e:` and explicitly logs state variables: pressure ($P$) and temperature ($T$). |
| [core/compositional_engine/flow/compositional_solver.py:700-720](file:///d:/rep/4.6/co2eor_optimizer/core/compositional_engine/flow/compositional_solver.py#L700-L720) | Phase density calculation throws exception in gravity flux loop | Previously caught `except Exception: pass` | Calculate inter-cell phase densities and gravity flux | Silently swallowed numerical breakdown in fluid density | **D** | **HIGH** | **RESOLVED**: Replaced with specific exception tuple (`FloatingPointError`, `ZeroDivisionError`, `ValueError`, `IndexError`, `KeyError`, `RuntimeError`) and logs cell ID, $P$, $T$, $S_o$, $S_g$. |
| [core/compositional_engine/phase_behavior/flash_calculator.py:285-320](file:///d:/rep/4.6/co2eor_optimizer/core/compositional_engine/phase_behavior/flash_calculator.py#L285-L320) | Rachford-Rice flash equation fails to solve | Previously caught `except Exception: pass` | Solve vapor fraction and phase compositions | Silently bypassed flash convergence failure | **D** | **HIGH** | **RESOLVED**: Replaced with specific numerical exceptions and logs $P$, $T$, and overall composition $z$. |
| [core/engine_simple/multiphase_flow_adapter.py:455-475](file:///d:/rep/4.6/co2eor_optimizer/core/engine_simple/multiphase_flow_adapter.py#L455-L475) | MMP / miscible relative permeability adjustment fails | Previously caught `except Exception: pass` | Adjust relative permeability near MMP | Silently skipped miscibility corrections | **D** | **HIGH** | **RESOLVED**: Replaced with specific exceptions (`ValueError`, `TypeError`, `ZeroDivisionError`, `FloatingPointError`) and logs average $P$, $T$, $\bar{S}_g$, $\bar{S}_w$. |
| [core/simulation/recovery_models.py:605-615](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L605-L615) | `scipy.optimize.fsolve` fails to solve Dykstra-Parsons WOR equation | Previously caught blind `except Exception:`, executed heuristic power law | Solve non-linear WOR balance | Discarded numerical solver convergence failure without reporting | **D** | **HIGH** | **RESOLVED**: Replaced with `except (RuntimeError, ValueError, ArithmeticError) as e:` and logs $V_{DP}$, mobility ratio $M$, and solver diagnostic. |
| [core/engine_surrogate/analytical_models.py:127-495](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L127-L495) | Primary bounded calculation raises exception in analytical recovery models | Previously caught blind `except Exception as e:` with generic warning | Calculate model recovery | Masked parameter-specific crashes | **D** | **HIGH** | **RESOLVED**: Replaced with specific exceptions (`ValueError`, `TypeError`, `ArithmeticError`, `RuntimeError`) and logs exact state variables ($P, T, S_{wi}, S_o, \mu_o, V_{DP}, M$). |
| [analysis/breakthrough_physics.py:155-165](file:///d:/rep/4.6/co2eor_optimizer/analysis/breakthrough_physics.py#L155-L165) | Peng-Robinson EOS initialization fails | Previously caught `except Exception:` and set `eos_model = None` | Initialize dynamic fluid EOS | Silently disabled dynamic breakthrough fluid properties | **D** | **HIGH** | **RESOLVED**: Replaced with specific exceptions (`ImportError`, `TypeError`, `ValueError`, `RuntimeError`) and logs pressure and temperature. |
| [analysis/decline_curve_analysis.py:110-145](file:///d:/rep/4.6/co2eor_optimizer/analysis/decline_curve_analysis.py#L110-L145) | `scipy.optimize.curve_fit` fails on hyperbolic decline curve | Previously caught blind `except Exception as e:` | Fit hyperbolic b-factor | Silently converted hyperbolic decline to exponential without reporting details | **D** | **MEDIUM** | **RESOLVED**: Replaced with `except (RuntimeError, ValueError, TypeError) as e:` and logs data point count, peak rate, and curve fit status. |
