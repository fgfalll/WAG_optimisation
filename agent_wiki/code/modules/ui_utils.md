# UI & Utility Subsystems (`ui/` & `utils/`)

## 1. User Interface Subsystem (`ui/`)

The `ui/` directory houses the PyQt6 desktop interface, data visualization widgets, background workers, and dialogs.

### Module Inventory
| Module | LOC | Primary Classes | Function | Status / Issues |
| :--- | :---: | :--- | :--- | :--- |
| `ui/main_window.py` | 1,990 | `MainWindow` | Main application shell, menus, status bars, tab wiring, report export. | Active and stable; report generation and engine signals guarded. |
| `ui/config_widget.py` | 1,310 | `ConfigWidget` | Visual configuration editor for application dataclasses and engine constraints. | Active and stable; single surrogate engine badge, `engine_selection_changed` signal. |
| `ui/optimization_widget.py` | 2,050 | `OptimizationWidget` | Setup tab for algorithms, parameters, well counts, objective weights. | Active and stable; `UnlockParametersDialog` and typed bounds integrated. |
| `ui/sensitivity_widget.py` | 560 | `SensitivityWidget` | Interactive sensitivity analysis charts (tornado, re-optimization). | Active and stable; Plotly subplots and pandas/numpy imports verified. |
| `ui/uq_widget.py` | 380 | `UQWidget` | Uncertainty quantification interface (Monte Carlo & PCE). | Active and stable; Qt widgets verified. |
| `ui/data_management_widget.py` | 1,980 | `DataManagementWidget` | Petrophysical data loading, PVT properties, and LAS well log inspector. | Active and stable; `utils.config_manager` imports and `set_engine_type` integrated. |
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
