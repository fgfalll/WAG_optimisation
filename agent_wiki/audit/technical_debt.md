# Technical Debt & Software Engineering Audit

## 1. Overview of Technical Debt

The codebase shows signs of rapid iterative prototyping, multiple architectural pivots (from 3D numerical grids to fast analytical surrogates for PhD research), and legacy preservation.

This document outlines the major software engineering defects, current test health, and identified runtime risks in UI modules.

---

## 2. Test Suite Status: FULLY PASSING (258 Passed, 16 Skipped, 0 Failed)

The automated test suite in `tests/` currently achieves a **100% pass rate**:
- **258 passed**, **16 skipped**, **0 failed** (executed in 225.22s, 31% overall codebase coverage; 60–80% coverage on active `core/engine_surrogate` and `core/optimisation_engine`).
- All prior unit fixture regressions (`economic_params is required`, `econ_params` unbound variable, NumPy 2.0 `np.trapz` removal, and ambiguous array truth evaluation) have been resolved.

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

## 4. `pyproject.toml` Configuration Debt

1. **Django Plugin in Petroleum Engineering App**:
   `plugins = ["mypy_django_plugin.main"]`. This project has no Django components; this is a copied boilerplate artifact.
2. **Missing Test Dependencies**:
   `hypothesis` and `h5py` are required by tests and `sr3_reader.py`, but omitted from `[project.optional-dependencies] dev`.
3. **Deprecated Ruff Configuration Syntax**:
   Top-level options `ignore` and `select` should be nested under `[tool.ruff.lint]`.

---

## 5. UI Presentation Layer Defects (53 Ruff Errors Cataloged)

Static analysis identified 53 critical errors (`F821` undefined names, `E722` bare excepts, syntax errors) in presentation and validation modules:

| File & Line | Error Code | Description | Consequence | Remediation |
| :--- | :--- | :--- | :--- | :--- |
| `ui/sensitivity_widget.py:450-525` | `F821` | Missing imports for `pd`, `np`, `go`, `make_subplots` | `NameError: name 'pd' is not defined` when sensitivity analysis completes | Add `import pandas as pd`, `import numpy as np`, `import plotly.graph_objects as go`, `from plotly.subplots import make_subplots`. |
| `ui/main_window.py:1528, 1542, 1551` | `F821` | Undefined local variable `charts` in `_generate_report_data` | `NameError: name 'charts' is not defined` during PDF/HTML report generation | Initialize `charts: Dict[str, Any] = {}` at top of method. |
| `ui/uq_widget.py:172, 176, 267` | `F821` | Missing PyQt6 imports `QSpinBox`, `QTextBrowser` | `NameError` during Uncertainty Quantification tab layout initialization | Import `QSpinBox`, `QTextBrowser` from `PyQt6.QtWidgets`. |
| `ui/widgets/log_viewer_dialog.py:275` | `F821` | Missing import `QTableWidgetItem` | `NameError` when displaying well log perforations table | Import `QTableWidgetItem` from `PyQt6.QtWidgets`. |
| `ui/optimization_widget.py:1774` | `F821` | Undefined `UnlockParametersDialog` | Crash when user attempts to unlock relaxable constraints | Implement or import dialog class. |
| `ui/optimization_widget.py:994` | `E722` | Bare `except:` in mutation tuple parsing | Catches `KeyboardInterrupt` and hides evaluation errors | Change to `except (ValueError, SyntaxError) as e:`. |
| `tests/validation/spe5_benchmark_validation.py:249` | SyntaxError | Missing comma after `simulation_years: float = 8.0` | Benchmark script cannot be compiled or imported by test runners | Add trailing comma `,`. |
