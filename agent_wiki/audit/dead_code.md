# Dead Code, Orphaned Modules & Unused Logic Audit

## 1. Executive Summary

Static analysis, reachability graphs, and ruff linting revealed significant volumes of orphaned code, abandoned engines, unused functions, and calculated variables that are discarded without consumption:
- **3 Entire Engine Subsystems** (`compositional_engine`, `unified_engine`, `engine_simple`) are orphaned from the main simulation pipeline.
- **373 Unused Imports (`F401`)** across active modules.
- **84 Unused Local Variables (`F841`)**, including physical properties calculated but ignored.
- **1 Deprecated Legacy Module (`core/simulation`)** with duplicate implementations.

---

## 2. Orphaned Subsystems & Simulation Engines

| Module / Directory | LOC | Invocation Status | Why It Is Orphaned | Recommendation |
| :--- | :---: | :--- | :--- | :--- |
| `core/compositional_engine/` | ~3,500 | **ORPHANED**. Only called by 3 standalone validation scripts in `validation/` | `EngineFactory.create_engine()` hardwires all calls to `SurrogateEngineWrapper`, bypassing `CompositionalEngine`. | **Retain & Document as Research Sandbox**. Do not delete if CMG benchmark validation scripts rely on it. |
| `core/unified_engine/` (engines, solvers, grid, state) | ~4,200 | **DEAD CODE**. Never instantiated anywhere in the repository. | Abandoned architecture during pivot to fast surrogate models. Only `physics.eos` is imported. | **Retain `physics.eos`**. Deprecate and isolate `engines/`, `solvers/`, and `core/` to reduce maintenance overhead. |
| `core/engine_simple/` | ~2,100 | **DEAD CODE**. Only called by its own internal `examples.py` script. | Superseded by `engine_surrogate`. | **Deprecate**. Mark as legacy reference; do not maintain. |
| `core/simulation/injection_schemes.py` | ~250 | **DEPRECATED**. Methods contain warnings directing to `FastProfileGenerator`. | Logic was refactored into `FastProfileGenerator`. | **Retain for backward compatibility**. Mark methods with `@deprecated`. |
| `core/simulation/profile_generator.py` | ~700 | **DEPRECATED**. Wraps `FastProfileGenerator`. | Redundant wrapper layer. | **Retain**. Keep as adapter; avoid editing internal logic. |
| `analysis/profiler.py` | ~1,200 | **DEAD / SUPERSEDED**. Used only if `self.simulation_engine is None` in fallback. | Replaced by `FastProfileGenerator`. Contains uncalibrated saturation clamps. | **Deprecate**. Remove reliance on profiler fallback in `optimisation_engine.py`. |

---

## 3. Unused Variables & Discarded Calculations (`F841`)

In several physical functions, variables are computed but never passed to consumers:

### 1. `core/engine_surrogate/analytical_models.py` (lines 870–875)
```python
def calculate_gradient(self, **params) -> Dict[str, float]:
    pressure = params.get("pressure", 3000.0)      # UNUSED (F841)
    mmp = params.get("mmp", 2500.0)                # UNUSED (F841)
    c7_plus = params.get("c7_plus_fraction", 0.3) # UNUSED (F841)
    v_dp = params.get("v_dp", 0.5)                 # UNUSED (F841)
    s_wi = params.get("s_wi", S_WI_CONNATE)        # UNUSED (F841)
    hcpvi = params.get("hcpvi", 1.0)
```
- **Audit Finding**: Defaults are unpacked into local variables, but the finite difference loops below them check `if key in params:`. If `"pressure"` is omitted from `params`, `key in params` is `False`, and no gradient is calculated despite the default extraction.
- **Remediation**: Populate `params` dictionary directly with defaults before finite differencing.

### 2. `analysis/profiler.py` (line 390)
```python
recoverable_oil_stb = ooip_stb * recovery_factor # UNUSED (F841)
```
- Calculated but overwritten or unused in profile shaping.

### 3. `core/engine_surrogate/analytical_models.py` (lines 254–255)
```python
soi = params.get("soi", 0.8) # UNUSED (F841)
s_wi = params.get("s_wi", 0.25) # UNUSED (F841)
```
- Unused in immiscible sweep efficiency estimation.

---

## 4. Unused Imports & Dead Dependencies (`F401`)

373 unused import statements exist across active modules.
- Prominent Example: In `core/optimisation_engine.py`:
  - `from core.simulation.simulator_exporter import SimulatorExporter` (line 38, never used)
  - `from core.simulation.recovery_models import EPSILON` (line 76, unused)
  - `from copy import deepcopy` imported multiple times across local scopes.
- In `core/engine_surrogate/analytical_models.py`:
  - Line 30: `from core.simulation.recovery_models import (...)` triggers deprecation warnings upon import even when surrogate models do not call them.

---

## 5. Removed Root Artifacts & Superseded Scripts [RESOLVED]

As part of the codebase cleanup, redundant and orphaned root artifacts were deleted or relocated:
- **`data_processor.py` & `ui/workers/data_processing_worker.py`**: Removed. Superseded by the modernized standalone CLI tool [`scripts/process_las_data.py`](file:///d:/rep/4.6/co2eor_optimizer/scripts/process_las_data.py).
- **`core/engine_surrogate/analytical_models – копія.py`**: Duplicate file removed.
- **`fixed_phd_class.txt`**: Temporary scratch file removed.
- **`test_surrogate.py`**: Ad-hoc root test script removed (all tests integrated into `tests/core/test_surrogate_engine.py`).
- **`nul` & `report.log`**: Stale build and debug artifacts deleted from root.
- **Utility Modules**: Relocated from flat root to `utils/` package (`config_manager.py`, `error_handler.py`, `path_utils.py`, `validation_manager.py`).

---

## 6. Removed Scientific Justification and Help Subsystems [RESOLVED]

The legacy Scientific Justification and Help Panel subsystems were completely decommissioned and removed:
- **Scientific Justification Dialogs & Assets**: Deleted `ui/dialogs/scientific_justification_dialog.py`, subdialogs (`breakthrough_physics_dialog.py`, `eos_modeling_dialog.py`, `profiler_dialog.py`, `recovery_models_dialog.py`, `overview_dialog.py`), `ui/dialogs/mathjax/`, and `ui/assets/docs/*_justification.html`.
- **Help Panel & Documentation Subsystem**: Deleted `ui/dialogs/parameter_help_dialog.py` (`HelpPanel`), `utils/help_manager.py` (`HelpManager`), `config/help_content.yaml`, and the `help/` directory containing markdown guides.
- **UI Integration**: Cleaned up [`ui/main_window.py`](file:///d:/rep/4.6/co2eor_optimizer/ui/main_window.py) and [`main.py`](file:///d:/rep/4.6/co2eor_optimizer/main.py) to remove corresponding menu actions, dialog launchers, side panel splitters, and unused imports.

