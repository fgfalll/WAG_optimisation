# Optimization Widget & Connected Subsystems Audit

> [!NOTE]
> This audit documents active findings, dead code, disconnected simulation/optimization parameters, software engineering anti-patterns, and resolution actions implemented across `ui/optimization_widget.py`, `ui/workers/optimization_worker.py`, `ui/dialogs/injection_scheme_dialog.py`, `ui/dialogs/preferences_dialog.py`, and `core/optimisation_engine.py`.

---

## 1. Subsystem Scope

| Component | File Path | Architectural Role | Audit Status |
| :--- | :--- | :--- | :--- |
| **Main Optimization Widget** | [`ui/optimization_widget.py`](../../ui/optimization_widget.py) | Central controller for algorithm selection, primary/secondary objectives, resolution, and results visualization. | **Refactored**: Dead methods removed, unsafe `eval` eliminated, resolutions constrained, NSGA-II secondary objective wired. |
| **Optimization Worker Thread** | [`ui/workers/optimization_worker.py`](../../ui/workers/optimization_worker.py) | Background `QThread` running optimization algorithms and streaming callbacks. | **Fixed**: NSGA-II positional arg crash resolved, progress callback wired, unused imports removed. |
| **Optimization Model Types** | [`ui/models/optimization_types.py`](../../ui/models/optimization_types.py) | Data models defining `OptimizationObjective` and `OptimizationConstraints`. | **Documented as Legacy**: Never used in active engine (retained for backward compatibility). |
| **Injection Scheme Modal** | [`ui/dialogs/injection_scheme_dialog.py`](../../ui/dialogs/injection_scheme_dialog.py) | Dialog for configuring WAG, Huff-n-Puff, SWAG, Tapered, and Pulsed schemes. | **Refactored**: 8 unused Qt imports removed, full reset logic implemented across all 5 schemes. |
| **Advanced Preferences** | [`ui/dialogs/preferences_dialog.py`](../../ui/dialogs/preferences_dialog.py) | Preferences configuring GPU acceleration, cache size, memory limits, and threads. | **Documented**: Clarified that CPU multi-processing is active, while GPU/cache settings are dormant. |
| **Engine Interface** | [`core/optimisation_engine.py`](../../core/optimisation_engine.py) | Backend executing GA, BO, Hybrid, and NSGA-II optimization runs. | **Fixed**: `optimize_nsga_2` calling signature fixed, `"pulsed"` added to `INJECTION_SCHEMES`. |

---

## 2. Disconnected & Dropped Parameters

### 2.1 Missing Secondary Objective UI for NSGA-II
- **Location**: `ui/optimization_widget.py`
- **Issue**: `config/base_config.json` configured `secondary_objectives` (`npv`, `recovery_factor`, `co2_utilization`, `storage_efficiency`), and `GeneticAlgorithmParams` accepted `secondary_objective`. However, `OptimizationWidget` only had a single `objective_combo`. NSGA-II defaulted to `recovery_factor`, producing a redundant objective if the primary objective was already `recovery_factor`.
- **Resolution**:
  - Added `self.secondary_objective_label` and `self.secondary_objective_combo` to `OptimizationWidget`.
  - Loaded `self.OPTIMIZATION_SECONDARY_OBJECTIVES` from configuration.
  - Dynamically toggles visibility in `_on_method_changed()` when NSGA-II or Hybrid NSGA-II -> BO is selected.
  - Enforced objective conflict validation in `_run_optimization()` to prevent running bi-objective optimization with duplicate objectives.

### 2.2 Disconnected "Pulsed" Injection Scheme
- **Location**: `core/optimisation_engine.py` vs `core/engine_surrogate/profile_generator_fast.py`
- **Issue**: `FastProfileGenerator` implemented `_generate_pulsed_profile()` and `InjectionSchemeDialog` offered "Pulsed". However, `INJECTION_SCHEMES` in `core/optimisation_engine.py` only contained 5 schemes (`["continuous", "wag", "tapered", "huff_n_puff", "swag"]`), omitting `"pulsed"`.
- **Resolution**: Added `"pulsed"` to `INJECTION_SCHEMES` and updated discrete parameter bound calculations (`len(INJECTION_SCHEMES)`).

### 2.3 Phantom Time Resolutions ("Quarterly" & "Weekly")
- **Location**: `ui/optimization_widget.py`
- **Issue**: The resolution combobox included `["Yearly", "Quarterly", "Monthly", "Weekly"]`. However, `FastProfileGenerator` only computes monthly timesteps ($n_{\text{points}} = \text{years} \times 12 + 1$) and `SurrogateEngineWrapper` only builds `yearly_` and `monthly_` keys. Selecting Quarterly or Weekly yielded empty profile dictionaries, breaking tables and plotting views.
- **Resolution**: Constrained resolution dropdown strictly to `["Yearly", "Monthly"]`.

### 2.4 Ghost Algorithm Categories in Parameter Summary
- **Location**: `ui/optimization_widget.py` (`_get_current_input_parameters`)
- **Issue**: `all_params` dictionary contained empty stub keys `"Particle Swarm Optimization": {}` and `"Differential Evolution": {}`. Neither algorithm exists in the active surrogate optimizer.
- **Resolution**: Removed these ghost categories from the input summary schema.

### 2.5 Placebo WAG & Huff-n-Puff Cycle Parameters in Scheme Dialog
- **Location**: `ui/dialogs/injection_scheme_dialog.py`
- **Issue**: `min_cycle_length_days` and `max_cycle_length_days` are edited in GUI, but `_generate_wag_profile()` in `profile_generator_fast.py` ignores them, using hardcoded `initial_wag_cycle_length = 45` and `standard_wag_cycle_length = 90`. Similarly, `huff_n_puff_cycle_length_days` is overwritten by `inj_period + soak_period + prod_period`.
- **Status**: Documented as physical engine simplification in `FastProfileGenerator`.

---

## 3. Dead Legacy Code & Obsolete Methods

### 3.1 Deleted Obsolete Methods in `OptimizationWidget` (~220 LOC)
The following unused methods were superseded by `utils/run_exporter.py` (`RunDataExporter`) and `core/plotting_manager.py` (`PlottingManager`):
- `_clean_dict_for_json(self, data_dict)`: Obsolete dictionary sanitizer.
- `_generate_detailed_txt_export(self)`: Superseded by `RunDataExporter.export()`.
- `_generate_detailed_csv_export(self)`: Superseded by `RunDataExporter.export()`.
- `_get_user_defined_params(self)`: Only called by the dead TXT export method.
- `_generate_co2_summary_html(self)`: Superseded by `PlottingManager.plot_co2_performance_summary_table()`.

### 3.2 Completely Orphaned Data Models (`ui/models/optimization_types.py`)
- `OptimizationObjective` and `OptimizationConstraints` were never imported or instantiated anywhere in the repository.
- **Action**: Retained for API backwards compatibility with clear module deprecation documentation.

---

## 4. Software Engineering & Security Remediation

### 4.1 Remote / Arbitrary Code Execution via `eval()` (Eliminated)
- **Previous Code** (`ui/optimization_widget.py:1040`):
  ```python
  elif isinstance(widget, QLineEdit) and name == "mutation":
      try:
          kwargs[name] = eval(widget.text())
      except:
          raise ValueError(self.tr("Invalid format for mutation tuple. Use (min, max)."))
  ```
- **Finding**: Calling built-in `eval()` on unsanitized user GUI text input with a bare `except:` clause is a severe security risk. Furthermore, `GeneticAlgorithmParams` has no `mutation` field (it uses `mutation_rate: float` and `mutation_percent_genes: int`).
- **Resolution**: Replaced with standard text extraction `kwargs[name] = widget.text()`.

### 4.2 Silent Exception Suppression (Fixed)
- **Previous Code** (`ui/optimization_widget.py:1022`):
  ```python
  except Exception:
      pass
  ```
- **Finding**: Swallowed all exceptions during min/max widget validation.
- **Resolution**: Replaced with contextual debug logging `logger.debug(f"Failed to update min/max validator: {e}")`.

### 4.3 Shadowed Module Import (Fixed)
- **Previous Code** (`ui/optimization_widget.py:4, 758`): Top-level import `from datetime import datetime` was shadowed inside `_append_log_message` by inline `import datetime`.
- **Resolution**: Standardized on top-level `datetime.fromtimestamp()`.

### 4.4 Incomplete Reset in `InjectionSchemeDialog` (Fixed)
- **Previous Code** (`ui/dialogs/injection_scheme_dialog.py:654`): `_reset_to_defaults` only reset WAG, leaving Huff-n-Puff, SWAG, Tapered, and Pulsed schemes without reset logic.
- **Resolution**: Completed reset implementation for all 5 injection schemes to their configuration baselines.

### 4.5 NSGA-II Calling Signature Mismatch (Fixed)
- **Previous Code** (`core/optimisation_engine.py`):
  `def optimize_nsga_2(self, nsga2_params_override, ...)` had no default value, raising `TypeError: optimize_nsga_2() missing 1 required positional argument` when called from `OptimizationWorker` or `hybrid_nsga2_bo`.
- **Resolution**: Set `nsga2_params_override: Optional[GeneticAlgorithmParams] = None` and aliased `ga_params_override` in `OptimizationWorker`.

---

## 5. Verification & Test Suite

The refactored subsystem was verified across the test suite:
- `tests/test_optimization_widget_export_parameters.py` (including new test for secondary objective visibility toggle and resolution constraints): **PASSED**
- `tests/test_project_save_load.py` (project save/load round-trip invariant): **PASSED**
- `tests/core/test_single_simulation.py`: **PASSED**
- `tests/core/test_optimization_config.py`: **PASSED**

Total: 19 passed, 0 failures.
