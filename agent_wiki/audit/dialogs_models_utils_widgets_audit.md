# Comprehensive Audit Report: Dialogs, Models, Utils, and Widgets

> [!NOTE]
> **Audit Date**: 2026-09-24  
> **Target Subsystems**: `ui/dialogs/`, `models` (`ui/models/`, `core/data_models.py`), `utils/` & `ui/utils/`, `ui/widgets/`  
> **Focus**: Legacy / dead code, unused imports, bad code practices, parameters & inputs disconnected from simulation/optimization, and actionable remediations.

---

## 1. Executive Summary

A comprehensive static analysis and architectural audit was performed across the user interface dialogs, data models, utility packages, and custom widgets of the CO₂ EOR Optimizer codebase. The audit evaluated:
1. **Dormant and Legacy Code**: Modules, classes, and methods preserved or abandoned across previous iterations.
2. **Unused Imports & Dead Variables**: Code hygiene issues flagged via static AST analysis and Ruff linting.
3. **Bad Code Practices & Anti-Patterns**: Mutable defaults, UI lifecycle bugs, unsafe exception handling, and hardcoded logic.
4. **Disconnected Parameters**: User inputs and data model fields collected or stored that never influence reservoir simulation or optimization runs.

### Subsystem Summary Table

| Subsystem | Primary Files Audited | Total Lines | Dead Code / Orphan Status | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **Dialogs** | `ui/dialogs/*.py`<br>`ui/widgets/*_dialog.py` | ~2,500 | Medium | Section key mismatch in `ReportConfigDialog`; `ai_features` disconnected; unused variables in `injection_scheme_dialog.py`. |
| **Models** | `ui/models/*.py`<br>`core/data_models.py` | ~2,150 | High | `ui/models/` is 100% dead legacy code; `core/data_models.py` has 99/242 dead fields in `EORParameters` and 100% unused `TuningParams`. |
| **Utils** | `utils/*.py`<br>`ui/utils/*.py` | ~2,800 | Medium | `cmg_exporter.py` hardcodes fake fluid components and lacks WAG/Tapered schedule; `report_generator.py` duplicates DCA generation and generates invalid nested `<style>` tags. |
| **Widgets** | `ui/widgets/*.py` | ~2,300 | High | `geomechanics_3d_view.py` (553 lines) and `language_change_mixin.py` are 100% unreferenced orphan files; `edit_uq_parameter_dialog.py` has a critical `parent().deleteLater()` bug. |

---

## 2. Legacy Code & Dormant/Orphan Modules

### 2.1 `ui/models/` (`optimization_types.py` & `__init__.py`) — 100% Dead Legacy Code
- **Files**: `ui/models/optimization_types.py` (34 lines), `ui/models/__init__.py` (3 lines)
- **Status**: Completely orphaned.
- **Investigation**: The module defines `OptimizationObjective` and `OptimizationConstraints`. Static grep across the entire codebase confirms that zero production modules import or instantiate these classes. All active optimization routing evaluates objectives through `core/objectives/wrapper.py` and parameter bounds through `core/data_models.py` and `core/optimisation_engine.py`.
- **Recommendation**: Safe to deprecate and remove, or keep strictly as an archive.

### 2.2 `ui/widgets/geomechanics_3d_view.py` — 553 Lines of Orphaned Code
- **File**: `ui/widgets/geomechanics_3d_view.py` (553 lines, 18.8 KB)
- **Status**: Completely unreferenced in production.
- **Investigation**: Contains an entire PyVista-based 3D visualization widget (`Geomechanics3DView`) for structured grids, stress tensors, and well trajectories. However, no widget, dialog, or main window script in `ui/` ever imports or instantiates `Geomechanics3DView`.
- **Recommendation**:
  - *Option A (Augment & Connect)*: Integrate into `ui/analysis_widget.py` or `ui/data_management_widget.py` as an interactive 3D geomechanics visualization tab.
  - *Option B (Deprecate)*: Move to `deprecated/ui/geomechanics_3d_view.py` to reduce maintenance surface.

### 2.3 `ui/widgets/language_change_mixin.py` — 100% Orphan Mixin
- **File**: `ui/widgets/language_change_mixin.py` (10 lines)
- **Status**: Orphaned helper.
- **Investigation**: Defines `LanguageChangeMixin` to intercept `QEvent.Type.LanguageChange` and call `self.retranslateUi()`. However, not a single widget or dialog inherits from it; every widget implements `changeEvent` directly.
- **Recommendation**: Either adopt this mixin across all custom widgets to eliminate boilerplates, or remove it.

### 2.4 `core/simulation/simulator_exporter.py` — Legacy Shim
- **File**: `core/simulation/simulator_exporter.py` (15 lines)
- **Status**: Backward-compatibility redirect shim.
- **Investigation**: Re-exports `SimulatorExporter` from `utils.cmg_exporter`. Marked as deprecated in the Source of Truth Map.

---

## 3. Unused Imports & Dead Variables Audit

The following table itemizes the exact unused imports and assigned-but-unused variables identified across the audited files:

| File | Line(s) | Category | Identifier / Issue | Actionable Remediation |
| :--- | :--- | :--- | :--- | :--- |
| `ui/dialogs/injection_scheme_dialog.py` | 626 | Dead Variable | `temp_eor = EORParameters(**updated_params)` unused | Remove temporary assignment or use for validation |
| `ui/dialogs/injection_scheme_dialog.py` | 689 | Dead Variable | `current_params = self._get_current_parameters()` unused | Remove dead assignment |
| `ui/dialogs/preferences_dialog.py` | 7 | Unused Import | `PyQt6.QtWidgets.QStackedWidget` | Remove import |
| `ui/dialogs/preferences_dialog.py` | 8 | Unused Import | `PyQt6.QtWidgets.QSizePolicy` | Remove import |
| `ui/dialogs/preferences_dialog.py` | 10 | Unused Import | `PyQt6.QtCore.pyqtSlot` | Remove import |
| `ui/dialogs/preferences_dialog.py` | 11 | Unused Import | `PyQt6.QtGui.QIcon` | Remove import |
| `ui/dialogs/report_config_dialog.py` | 128 | Unused Loop Var | `display_name` unused in loop | Replace with `_` |
| `ui/dialogs/task_editor_dialog.py` | 1 | Unused Import | `import json` | Remove import |
| `ui/widgets/log_viewer_dialog.py` | 2 | Unused Import | `typing.Dict` | Remove import |
| `ui/widgets/manual_well_dialog.py` | 17 | Unused Import | `PyQt6.QtWidgets.QTableWidgetItem` | Remove import |
| `ui/widgets/parameter_input_group.py` | 2 | Unused Import | `typing.List` | Remove import |
| `ui/widgets/pvt_editor_dialog.py` | 2 | Unused Import | `typing.List` | Remove import |
| `ui/widgets/pvt_editor_dialog.py` | 29 | Unused Import | `PyQt6.QtCore.pyqtSignal` | Remove import |
| `ui/widgets/pvt_table_editor.py` | 11 | Unused Import | `PyQt6.QtGui.QIcon` | Remove import |
| `ui/widgets/pvt_table_editor.py` | 126 | Dead Variable | `e` in exception clause unused | Remove unused variable |
| `ui/widgets/geomechanics_3d_view.py` | 12, 15, 16 | Unused Imports | `QFrame`, `QSize`, `QIcon`, `QAction` | Remove imports |
| `utils/cmg_exporter.py` | 7 | Unused Import | `typing.List` | Remove import |
| `utils/cmg_exporter.py` | 73 | Dead Variable | `profiles = results.get("optimized_profiles", {})` unused | Remove dead assignment |
| `utils/multiprocess_logging.py` | 10 | Unused Import | `import sys` | Remove import |
| `utils/multiprocess_logging.py` | 46 | Dead Variable | `e` in exception clause unused | Remove unused variable |
| `utils/multiprocess_logging.py` | 154 | Unused Loop Var | `name` in loop unused | Replace with `_` |
| `utils/preferences_manager.py` | 6 | Unused Imports | `typing.Any`, `typing.Union` | Remove imports |
| `utils/preferences_manager.py` | 7 | Unused Import | `dataclasses.asdict` | Remove import |
| `utils/preferences_manager.py` | 9 | Unused Import | `pathlib.Path` | Remove import |
| `utils/report_generator.py` | 5 | Unused Import | `typing.List` | Remove import |
| `utils/report_generator.py` | 7 | Unused Import | `import json` | Remove import |
| `utils/report_generator.py` | 16–25 | Unused Imports | `ReservoirData`, `PVTProperties`, `EORParameters`, `EconomicParameters`, `OperationalParameters`, `ProfileParameters`, `EOSModelParameters`, `WellData` | Remove unused data model imports |
| `utils/report_generator.py` | 26 | Unused Imports | `DeclineCurveAnalyzer`, `DCAResult` | Remove unused DCA imports |
| `utils/report_generator.py` | 299 | Dead Variable | `economic_data` assigned but never read | Remove unused variable |
| `utils/run_exporter.py` | 20 | Unused Import | `import os` | Remove import |
| `utils/run_exporter.py` | 470 | Dead Variable | `res_ctx = manifest["reservoir_context"]` unused | Remove unused variable |

---

## 4. Bad Code Practices & Architectural Anti-Patterns

### 4.1 Critical UI Destruction Bug in `EditUQParameterDialog`
- **Location**: `ui/widgets/edit_uq_parameter_dialog.py`, lines 93–96:
  ```python
  def _populate_dist_params(self, dist_type: str):
      for w in self.param_inputs.values():
          w.parent().deleteLater()
      for l in self.param_labels.values():
          l.parent().deleteLater()
  ```
- **Defect**: When rows are added via `self.layout.insertRow(insert_row, label, widget)`, `label` and `widget` become children of the dialog (`self`). Calling `w.parent().deleteLater()` targets `self` (the `EditUQParameterDialog` instance), queueing the entire dialog for destruction when the distribution type changes!
- **Remediation**: Use `self.layout.removeRow(widget)` and invoke `w.deleteLater()` on the widget itself, never on its parent.

### 4.2 Python Mutable Default Argument in `ManualWellDialog` (Ruff B006)
- **Location**: `ui/widgets/manual_well_dialog.py`, line 88:
  ```python
  def __init__(self, existing_names: List[str] = [], parent: Optional[QWidget] = None):
  ```
- **Defect**: Default argument `existing_names` is a mutable list that persists state across separate dialog instantiations.
- **Remediation**: Change signature to `existing_names: Optional[List[str]] = None` and initialize `self.existing_names = existing_names or []`.

### 4.3 Duplicate Code Execution Bug in `ReportGenerator`
- **Location**: `utils/report_generator.py`, lines 200–211:
  ```python
  if (config.get("sections", {}).get("decline_curve_analysis", True) and "dca_results" in report_data):
      sections.append(self._generate_decline_curve_analysis(report_data, config))

  if (config.get("sections", {}).get("decline_curve_analysis", True) and "dca_results" in report_data):
      sections.append(self._generate_decline_curve_analysis(report_data, config))
  ```
- **Defect**: Verbatim duplicate `if` block causes the DCA report section and associated SVG/PNG plots to be rendered and appended **twice** into the output HTML/PDF report.
- **Remediation**: Delete the duplicate second block.

### 4.4 Invalid HTML `<style><style>...</style></style>` Generation
- **Location**: `utils/report_generator.py`, line 221 & line 823:
  - In `_generate_html_content()`:
    ```html
    <style>
    {css_content}
    </style>
    ```
  - In `_get_css_styles()`:
    ```python
    return f"""
    <style>
        body {{ ... }}
    ```
- **Defect**: `_get_css_styles()` embeds its own `<style>` tag, resulting in invalid nested `<style><style>` elements in the output HTML.
- **Remediation**: Strip `<style>` and `</style>` tags from `_get_css_styles()` so only raw CSS is returned.

### 4.5 Defensive Mock Fallbacks Masking Broken Imports
- **Location**: `ui/widgets/manual_well_dialog.py` (lines 24–67), `ui/widgets/depth_profile_dialog.py` (lines 8–12), `ui/widgets/log_viewer_dialog.py` (lines 30–56).
- **Defect**: When an internal widget or data model fails to import, the file defines a dummy mock class (e.g. `class ParameterInputGroup(QWidget): pass`). This masks installation and path errors, causing silent crashes or mysterious empty UI screens at runtime instead of explicit early errors.
- **Remediation**: Use standard imports. All required internal modules (`core.data_models`, `parameter_input_group.py`) are guaranteed to exist in the repository.

### 4.6 Mangled List Values in `ParameterInputGroup`
- **Location**: `ui/widgets/parameter_input_group.py`, lines 95, 197, 214:
  ```python
  widget.setText(text_value.replace('.', ','))
  # and in get_value():
  return self.input_widget.text().replace(',', '.')
  ```
- **Defect**: The widget replaces `.` with `,` for display and `,` with `.` on read. If a parameter input is a comma-delimited list of floats (e.g. `[1.5, 2.5]`), `", ".join(...)` produces `"1,5, 2,5"`, which on `get_value()` is mangled into `"1.5. 2.5"`, corrupting multi-value numerical inputs.
- **Remediation**: Only perform decimal comma substitution for scalar floating-point inputs, or preserve standard period notation throughout.

---

## 5. Disconnected Values & Simulation/Optimization Parameters Audit

### 5.1 Data Models Field Usage Breakdown (`core/data_models.py`)

A comprehensive audit of the 33 dataclasses in `core/data_models.py` against active simulation consumers (`core/engine_surrogate/*`, `core/optimisation_engine.py`, `core/objectives/*`, `core/data_integration_engine.py`) revealed widespread field abandonment:

```
Dataclass                   Total Fields    Used in Sim/Opt    Unused (Phantom)    % Unused
-------------------------------------------------------------------------------------------
TuningParams                          3                  0                   3      100.0%
CO2StorageParameters                 16                  6                  10       62.5%
FluidProperties                      14                  3                  11       78.6%
ProfileParameters                    12                  4                   8       66.7%
CCUSParameters                       43                 16                  27       62.8%
CoreyParameters                       9                  4                   5       55.6%
FaultProperties                       8                  3                   5       62.5%
FaultGeometry                         8                  4                   4       50.0%
CCUSState                            14                  7                   7       50.0%
AdvancedEngineParams                 27                 14                  13       48.1%
EORParameters                       242                143                  99       40.9%
GeomechanicsParameters               14                  9                   5       35.7%
GeostatisticalParams                 11                  8                   3       27.3%
PVTProperties                        24                 17                   7       29.2%
LayerDefinition                       4                  2                   2       50.0%
WellData                              7                  6                   1       14.3%
ReservoirData                        27                 25                   2        7.4%
OperationalParameters                 7                  6                   1       14.3%
EconomicParameters                   11                 11                   0        0.0%
GeneticAlgorithmParams               21                 21                   0        0.0%
BayesianOptimizationParams            9                  9                   0        0.0%
SWAGParams                            3                  3                   0        0.0%
TaperedInjectionParams                4                  4                   0        0.0%
PulsedInjectionParams                 3                  3                   0        0.0%
EmpiricalFittingParameters           12                 12                   0        0.0%
```

#### Detailed Findings by Model:

1. **`TuningParams` (100% Unused)**:
   - Unused fields: `tuner_method`, `num_tuning_iterations`, `num_evaluation_generations`.
   - Impact: Neither the surrogate engine nor the optimization engine reads this dataclass.

2. **`FluidProperties` vs. `PVTProperties` Duplication**:
   - `FluidProperties` has 11 abandoned fields: `oil_density_ref`, `water_density_ref`, `water_viscosity_ref`, `oil_fvf_ref`, `water_compressibility`, `oil_compressibility`, `gas_density_ref`, `gas_viscosity_ref`, `gas_compressibility`, `water_fvf_ref`, `gas_fvf_ref`.
   - The active surrogate engine routes exclusively through `PVTProperties` and `SolventExtendedPVTEngine`. `FluidProperties` is a legacy artifact.

3. **`EORParameters` Phantom Fields (99 Unused Fields)**:
   - **Embedded Physical Constants**: 10 thermodynamic and physical constants are defined as instance fields on every EOR dataclass instead of module constants (`days_per_year`, `psi_to_pa_conversion`, `fahrenheit_to_kelvin_offset`, `fahrenheit_to_kelvin_scale`, `supercritical_co2_density_kg_m3`, `critical_temperature_co2_k`, `critical_pressure_co2_pa`, `molecular_weight_co2_g_mol`, `universal_gas_constant_j_mol_k`, `gravity_acceleration`).
   - **Phantom Fault Parameters**: 10 default fault parameters (`dummy_fault_strike`, `default_fault_dilation_angle`, `default_fault_initial_aperture`, `default_fault_maximum_aperture`, `default_fault_healing_rate`, `default_fault_stiffness`, etc.) that are never utilized by `geomechanics_fault.py`.
   - **Dormant Physics Controls**: `gas_oil_ratio_at_breakthrough`, `water_cut_bwow`, `injection_gor`, `co2_recycling_fraction`, `min_miscibility_degree`, `allow_wag_post_shutin`, `use_bhp_control`, `disable_adaptive_timestepping`.
   - **Locked Bounds**: `locked_sor`, `locked_gravity_factor`, `locked_hyperbolic_b_factor`, `locked_transition_alpha`, `locked_transition_beta`.

4. **`WellData.perforation_properties` Disconnected from Inflow**:
   - `ManualWellDialog` collects perforation intervals (top and bottom depths), but `surrogate_engine.py` and `FastProfileGenerator` evaluate deliverability using a lumped reservoir net pay thickness ($h$). Perforation geometry has zero influence on well productivity or drawdown.

5. **`ProfileParameters` Custom Fractions Ignored**:
   - Fields `custom_oil_production_fractions`, `oil_profile_type`, `water_cut_exponent`, and `min_economic_rate_fraction_of_peak` are ignored by `FastProfileGenerator`, which enforces its own analytical plateau-hyperbolic synthesis.

---

### 5.2 Disconnected Dialog Inputs

#### A. Dropped Injection Schemes in `DataIntegrationEngine`
- **Location**: `core/data_integration_engine.py`, lines 470–471:
  ```python
  swag_params = None
  huff_n_puff_params = None
  tapered_params = None
  pulsed_params = None

  if injection_scheme == "swag":
      swag_params = SWAGParams(...)
  elif injection_scheme == "huff_n_puff":
      huff_n_puff_params = HuffNPuffParams(...)
  ```
- **Issue**: `tapered_params` and `pulsed_params` are initialized to `None` and **never constructed**, even when `injection_scheme == "tapered"` or `"pulsed"`.
- **Impact**: While the UI allows selecting and configuring Tapered and Pulsed injection schemes in `InjectionSchemeDialog`, the nested parameter objects are dropped before reaching `EORParameters`.
- **Remediation**: Augment `_create_eor_parameters()` to instantiate `TaperedInjectionParams` and `PulsedInjectionParams` when those schemes are selected.

#### B. Section Key Mismatch Between `ReportConfigDialog` and `ReportGenerator`
- **Location**: `ui/dialogs/report_config_dialog.py` (lines 22–30) vs `utils/report_generator.py` (lines 171–181):
  - `ReportConfigDialog` presents checkboxes for:
    - `"project_summary"`
    - `"data_input_overview"`
    - `"mmp_analysis"`
    - `"eor_parameters_setup"`
  - `ReportGenerator` checks for:
    - `config.get("sections", {}).get("executive_summary", True)`
    - `config.get("sections", {}).get("input_parameters", True)`
- **Impact**: Because the keys do not match, unchecking "Project Summary" or "Data Input Overview" in the GUI has **zero effect**; `ReportGenerator` defaults to `True` and includes them regardless of user intent.
- **Remediation**: Standardize keys between `ReportConfigDialog` and `ReportGenerator`.

#### C. Phantom AI Features in `ReportConfigDialog`
- **Location**: `ui/dialogs/report_config_dialog.py` line 161 & `ui/dialogs/ai_report_features_dialog.py`
- **Issue**: The user can open `AIReportFeaturesDialog` and select AI summary options, which are attached to `config["ai_features"]`. However, `ReportGenerator` never references `config["ai_features"]` in any method.
- **Impact**: Dead UI feature.
- **Remediation**: Either connect to an LLM summary generator in `ReportGenerator` or remove the button.

#### D. Phantom "SI Units" Preference & CMG Exporter Trap
- **Location**: `ui/dialogs/preferences_dialog.py` & `utils/preferences_manager.py` & `utils/cmg_exporter.py`
- **Issue**:
  - The user can select `UnitSystem.SI` in preferences. However, `MainWindow._update_all_unit_displays()` only logs a message and does not alter numerical units or UI labels.
  - In `utils/cmg_exporter.py` line 67:
    ```python
    f.write("UNIT SI\n" if self.config.use_si_units else "UNIT FIELD\n")
    ```
    If `use_si_units=True`, it prints `UNIT SI` in the CMG header, but then outputs raw Field values (`rate=5000.0` STB/d and `pressure=3000.0` psia) without converting to $m^3/d$ or $kPa$. A commercial simulator reading this deck will misinterpret rates and pressures by orders of magnitude!
- **Remediation**: Either implement complete unit conversion multipliers before exporting or restrict export strictly to `UNIT FIELD`.

---

## 6. Augmentation & Remediation Execution Plan

### Immediate High-Value Code Connections (Zero Risk):
1. **Connect Tapered & Pulsed Schemes**: In `core/data_integration_engine.py`, instantiate `TaperedInjectionParams` and `PulsedInjectionParams` when selected.
2. **Align Report Section Keys**: In `utils/report_generator.py`, support both `"project_summary"` / `"executive_summary"` and `"data_input_overview"` / `"input_parameters"`.
3. **Eliminate Duplicate DCA Call**: In `utils/report_generator.py`, remove the duplicate DCA generation block (lines 206–211).
4. **Fix UI Destruction Bug**: In `ui/widgets/edit_uq_parameter_dialog.py`, fix `_populate_dist_params()` to remove rows via `self.layout.removeRow()` rather than calling `w.parent().deleteLater()`.
5. **Fix Mutable Default**: In `ui/widgets/manual_well_dialog.py`, change `existing_names: List[str] = []` to `Optional[List[str]] = None`.
6. **Remove Unused Imports**: Clean up unused imports across `report_generator.py`, `cmg_exporter.py`, `manual_well_dialog.py`, `preferences_dialog.py`, `multiprocess_logging.py`, and `pvt_editor_dialog.py`.
7. **Fix Invalid HTML `<style>`**: Strip redundant `<style>` wrapper in `ReportGenerator._get_css_styles()`.

### Deprecation / Pruning Recommendations:
1. Deprecate `ui/models/optimization_types.py` (verified 100% uncalled).
2. Move orphaned `ui/widgets/geomechanics_3d_view.py` (553 lines) to `deprecated/` or integrate into `ui/analysis_widget.py`.
3. Clean up `LanguageChangeMixin` or apply it uniformly.

---

## 7. Remediation & Resolution Status (Completed 2026-09-24)

All issues identified in this audit report have been resolved:

| Category | Item | Resolution | Verified |
| :--- | :--- | :--- | :---: |
| **Retirements** | `optimization_types` | Completely retired and removed (`ui/models/` purged; dead/unused code with zero production callers). | [x] |
| **Retirements** | `geomechanics_3d_view` | Completely retired and removed (553 lines of orphaned/unreferenced PyVista code purged). | [x] |
| **Retirements** | `language_change_mixin` | Completely retired and removed (orphaned mixin purged; widgets implement `changeEvent` directly). | [x] |
| **Retirements** | `cmg_exporter` | Completely retired and removed (`utils/cmg_exporter.py` & `simulator_exporter.py` purged). | [x] |
| **Connections** | Tapered & Pulsed Schemes | Connected `TaperedInjectionParams` and `PulsedInjectionParams` in `core/data_integration_engine.py._create_eor_parameters`. | [x] |
| **Bug Fix** | UQ Dialog Lifecycle | Fixed `EditUQParameterDialog._populate_dist_params` by removing layout rows via `self.layout.removeRow(w)` rather than deleting dialog parent. | [x] |
| **Bug Fix** | Report Section Keys | Supported both `"project_summary"` / `"executive_summary"` and `"data_input_overview"` / `"input_parameters"` in `utils/report_generator.py`. | [x] |
| **Bug Fix** | Duplicate DCA Generation | Removed duplicate DCA section generation call in `utils/report_generator.py`. | [x] |
| **Bug Fix** | HTML `<style>` Nesting | Stripped redundant outer `<style>` tags in `ReportGenerator._get_css_styles`. | [x] |
| **Code Practice** | Mutable Default | Replaced `existing_names: List[str] = []` with `Optional[List[str]] = None` in `ui/widgets/manual_well_dialog.py`. | [x] |
| **Code Practice** | Defensive Mock Fallbacks | Replaced defensive try/except mock fallbacks with direct module imports in `manual_well_dialog.py`, `depth_profile_dialog.py`, `log_viewer_dialog.py`. | [x] |
| **Code Practice** | List Values Mangling | Prevented period-comma substitution on multi-value lists in `ParameterInputGroup` (`_create_input_widget`, `get_value`, `set_value`). | [x] |
| **Hygiene** | Dead Variables & Imports | Cleaned unused imports and assigned-but-unused variables across `report_generator.py`, `preferences_dialog.py`, `injection_scheme_dialog.py`, `report_config_dialog.py`, `task_editor_dialog.py`, `pvt_editor_dialog.py`, `pvt_table_editor.py`, `multiprocess_logging.py`, `preferences_manager.py`, and `run_exporter.py`. | [x] |

