# Technical Debt & Software Engineering Audit

## 1. Overview of Technical Debt

The codebase shows signs of rapid iterative prototyping, multiple architectural pivots (from 3D numerical grids to fast analytical surrogates for PhD research), and legacy preservation.

This document outlines the major software engineering defects, current test health, and identified runtime risks in UI modules.

---

## 2. Test Suite Status: FULLY PASSING (299 Passed, 23 Skipped, 0 Failed)

The automated test suite in `tests/` currently achieves a **100% pass rate**:
- **299 passed**, **23 skipped**, **0 failed** (executed in 129.06s).
- All prior unit fixture regressions (`economic_params is required`, `econ_params` unbound variable, NumPy 2.0 `np.trapz` removal, and ambiguous array truth evaluation) have been resolved.
- SFT-02 (`MagicMock` OPEX broadcasting) and SFT-03 (`OptimizationWidget.set_engine` alias) resolved, achieving 0 test failures.

---

## 3. Root Directory Cleanup & Code Restructuring: COMPLETED

All loose Python modules in the root directory were reorganized into structured packages or removed:
- `config_manager.py` $\to$ [utils/config_manager.py](file:///d:/rep/4.6/co2eor_optimizer/utils/config_manager.py)
- `error_handler.py` $\to$ [utils/error_handler.py](file:///d:/rep/4.6/co2eor_optimizer/utils/error_handler.py)
- `path_utils.py` $\to$ [utils/path_utils.py](file:///d:/rep/4.6/co2eor_optimizer/utils/path_utils.py)
- `validation_manager.py` $\to$ [utils/validation_manager.py](file:///d:/rep/4.6/co2eor_optimizer/utils/validation_manager.py)
- `help_manager.py` $\to$ Decommissioned and deleted along with `ui/dialogs/parameter_help_dialog.py` and `help/*.md`.
- `cleanup_pycache.ps1` $\to$ [scripts/cleanup_pycache.ps1](file:///d:/rep/4.6/co2eor_optimizer/scripts/cleanup_pycache.ps1)
- `data_processor.py` superseded by dedicated CLI tool [scripts/process_las_data.py](file:///d:/rep/4.6/co2eor_optimizer/scripts/process_las_data.py)
- Removed orphaned and dead files (`ui/workers/data_processing_worker.py`, `report.log`, `fixed_phd_class.txt`, `test_surrogate.py`, `core/engine_surrogate/analytical_models – копія.py`, and legacy scientific justification dialogs/HTML assets in `ui/dialogs/` and `ui/assets/docs/`).

---

## 4. `pyproject.toml` Configuration Debt: RESOLVED

All historical configuration artifacts in `pyproject.toml` have been resolved:
1. **Django Plugin Removed**:
   Removed `plugins = ["mypy_django_plugin.main"]` from `[tool.mypy]`.
2. **Test Dependencies Added**:
   Added `hypothesis>=6.80` and `pytest-benchmark>=4.0` to `[project.optional-dependencies] dev` (`h5py>=3.8` is already in core `dependencies`).
3. **Modernized Ruff Configuration Syntax**:
   Nested `select` and `ignore` under `[tool.ruff.lint]`.

---

## 5. UI Presentation Layer Defects: ERADICATED (36 F821 Undefined Names Resolved)

All critical static analysis errors (`F821` undefined names, `E722` bare excepts, syntax errors) have been systematically resolved:

| File & Line | Error Code | Description | Consequence | Remediation Status |
| :--- | :--- | :--- | :--- | :--- |
| `ui/sensitivity_widget.py:450-525` | `F821` | Missing imports for `pd`, `np`, `go`, `make_subplots` | `NameError: name 'pd' is not defined` when sensitivity analysis completes | **RESOLVED**: Added imports for pandas, numpy, and plotly. |
| `ui/main_window.py:1528, 1542, 1551` | `F821` | Undefined local variable `charts` / layout indices in `_generate_report_data` | `NameError` during PDF/HTML report generation | **RESOLVED**: Fixed `report_charts` and layout/tab index extraction. |
| `ui/uq_widget.py:172, 176, 267` | `F821` | Missing PyQt6 imports `QSpinBox`, `QTextBrowser` | `NameError` during Uncertainty Quantification tab layout initialization | **RESOLVED**: Imported `QSpinBox`, `QTextBrowser` from `PyQt6.QtWidgets`. |
| `ui/widgets/log_viewer_dialog.py:275` | `F821` | Missing import `QTableWidgetItem` | `NameError` when displaying well log perforations table | **RESOLVED**: Imported `QTableWidgetItem` from `PyQt6.QtWidgets`. |
| `ui/optimization_widget.py:1774` | `F821` | Undefined `UnlockParametersDialog` | Crash when user attempts to unlock relaxable constraints | **RESOLVED**: Implemented `UnlockParametersDialog` with typed relaxation bounds. |
| `ui/optimization_widget.py:994` | `E722` | Bare `except:` in mutation tuple parsing | Catches `KeyboardInterrupt` and hides evaluation errors | **RESOLVED**: Changed to `except (ValueError, SyntaxError) as e:`. |
| `tests/validation/spe5_benchmark_validation.py:249` | SyntaxError | Missing comma after `simulation_years: float = 8.0` | Benchmark script cannot be compiled or imported by test runners | **RESOLVED**: Added trailing comma `,`. |
| `core/optimisation_engine.py` | `F821` | Undefined `EPSILON`, `SimulatorExporter` | Runtime `NameError` in optimizer paths | **RESOLVED**: Defined `EPSILON = 1e-10` and imported `SimulatorExporter`. |
| `core/data_integration_engine.py` | `F821` | Undefined `GeomechanicsParameters`, `create_geostatistical_grid` | Runtime `NameError` in geomechanics integration | **RESOLVED**: Imported `GeomechanicsParameters` and `create_geostatistical_grid`. |
| `analysis/sensitivity_analyzer.py` | `F821` | Undefined `self`, `PengRobinsonEOS`, `SoaveRedlichKwongEOS` | Crash in sensitivity matrix evaluations | **RESOLVED**: Added `self` parameter to `run_two_way_sensitivity` and handled EOS imports safely. |
| `ui/ai_assistant_widget.py` | `F821` | Undefined `AI_SERVICES_CONFIG`, `QInputDialog` | Crash in AI service credential dialogs | **RESOLVED**: Fixed `self.AI_SERVICES_CONFIG` and imported `QInputDialog`. |
| `core/validation/physical_consistency_validator.py` | `F821` | Undefined `CCUSState` | Validation crash | **RESOLVED**: Imported `CCUSState`. |
| `ui/analysis_widget.py`, `ui/config_widget.py` | `F821` | Undefined `ConfigManager` | UI settings crash | **RESOLVED**: Imported `ConfigManager`. |

---

## 6. Resolved Issues Archive

All verified historical defect fixes, eradicated hidden calibrations, and eliminated scientific flaws are archived in:
👉 [**Resolved Issues & Defect Resolution Archive (`resolved_issues.md`)**](resolved_issues.md)

