# UI & Utility Subsystems (`ui/` & `utils/`)

## 1. User Interface Subsystem (`ui/`)

The `ui/` directory houses the PyQt6 desktop interface, data visualization widgets, background workers, and dialogs.

### Module Inventory
| Module | LOC | Primary Classes | Function | Status / Issues |
| :--- | :---: | :--- | :--- | :--- |
| `ui/main_window.py` | 1,620 | `MainWindow` | Main application shell, menus, status bars, report export. | Contains software bug: undefined `charts` variable in `_generate_report_data` (line 1528). |
| `ui/optimization_widget.py` | 1,840 | `OptimizationWidget` | Setup tab for algorithms, parameters, well counts, objective weights. | Contains bare `except:` at line 994; references undefined `UnlockParametersDialog` at line 1774. |
| `ui/sensitivity_widget.py` | 560 | `SensitivityWidget` | Interactive sensitivity analysis charts (tornado, re-optimization). | Contains software bugs: missing imports for `pd`, `np`, `go`, `make_subplots` (lines 450-525). |
| `ui/uq_widget.py` | 380 | `UQWidget` | Uncertainty quantification interface (Monte Carlo & PCE). | Contains software bugs: missing PyQt6 imports `QSpinBox`, `QTextBrowser` (lines 172, 176, 267). |
| `ui/data_management_widget.py` | 740 | `DataManagementWidget` | Petrophysical data loading, PVT properties, and LAS well log inspector. | Active and stable. |
| `ui/workers/optimization_worker.py` | 240 | `OptimizationWorker` | `QThread` background runner executing optimization without blocking GUI. | Stable worker pattern. |

---

## 2. Utility Subsystem (`utils/`)

The `utils/` directory provides infrastructure services, serialization, and hardware detection.

### Module Inventory
| Module | LOC | Primary Classes / Functions | Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `utils/run_exporter.py` | 410 | `RunDataExporter` | Serializes simulation and optimization results to JSON, CSV, Excel, and NetCDF formats. | **LOW** |
| `utils/cache_manager.py` | 220 | `CacheManager` | Disk and memory LRU caching for expensive thermodynamic property evaluations. | **LOW** |
| `utils/hardware_detector.py` | 180 | `HardwareDetector` | Detects physical CPU cores, hyperthreading, GPU availability, and RAM limits. | **LOW** |
| `utils/preferences_manager.py` | 160 | `PreferencesManager` | User preferences persistence (`~/.co2eor_optimizer/preferences.json`). | **LOW** |
| `utils/units_converter.py` | 280 | `UnitConverter` | Bidirectional conversions between Field units (STB, MSCF, psi, ft) and SI units. | **MEDIUM** |
