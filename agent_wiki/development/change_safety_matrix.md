# Change-Safety Classification Matrix

Before modifying any file in this repository, AI agents and developers must consult this classification matrix to assess the scientific and software risks of the proposed change.

---

## 1. Risk Tier Definitions

| Tier | Definition | Required Pre-Modification Validation | Required Post-Modification Tests |
| :---: | :--- | :--- | :--- |
| **CRITICAL** | Equations, physical models, conversion constants, or parameters directly determining simulation results, recovery factors, or scientific conclusions. | Deep theoretical review; verify equation dimensional consistency; check benchmark impact. | Full pytest suite + CMG benchmark comparison (`validate_against_cmg`) + mass balance check. |
| **HIGH** | Simulation orchestration, profile generation, numerical ODE solvers, PVT interpolation, and constraint penalties. | Inspect call graph and callers; verify unit consistency; review potential solver stiffness. | Full pytest suite (`pytest tests/`) + edge-case tests (zero injection, zero permeability). |
| **MEDIUM** | Data parsing, file handlers, UI background workers, preference management, and reporting. | Trace data flow from input to data models. | Unit tests for relevant package + GUI launch test (`python main.py`). |
| **LOW** | Presentation UI widgets, translations (`i18n`), logging formatters, plotting styles, documentation. | Check widget layout and signal-slot connections. | GUI smoke test; verify no broken Qt signals. |

---

## 2. Component Risk Classification Table

| File / Component Path | Risk Tier | Primary Responsibilities | Danger / Failure Mode If Modified Incorrectly |
| :--- | :---: | :--- | :--- |
| `core/engine_surrogate/analytical_models.py` | **CRITICAL** | Recovery factor calculation (`PhDHybridSurrogate`, Koval, BL) | Aligns or breaks all optimization results; distorts recovery by orders of magnitude; introduces unphysical discontinuities. |
| `core/engine_surrogate/surrogate_engine.py` | **CRITICAL** | 0D pressure ODE, CO₂ mass balance, NPV calculation | Can cause numerical solver divergence, negative pressures, or invalid economic decisions. |
| `core/engine_surrogate/profile_generator_fast.py` | **CRITICAL** | Time series rate synthesis, WAG modulation, breakthrough | Alters breakthrough time and cash flow profiles; can violate mass balance. |
| `core/data_models.py` (`PhysicalConstants`) | **CRITICAL** | Centralized physical constants and unit conversion factors | Altering conversion factors (e.g. $1.062 \times 10^{-14}$) creates million-fold injectivity errors. |
| `evaluation/mmp.py` | **CRITICAL** | Minimum Miscibility Pressure correlations | Dictates whether reservoir operates in miscible or immiscible regime. |
| `core/optimisation_engine.py` | **HIGH** | Algorithm execution, solution evaluation, penalty functions | Modifying parameter packing, penalty thresholds, or reintroducing penalty dilution (`* 0.1`) allows unphysical chromosomes to survive selection. Must strictly enforce `FAILURE_PENALTY` ($-10^{12}$) and NaN pruning. |
| `core/objectives/wrapper.py` | **HIGH** | Multi-objective scoring and metric aggregation | Reintroducing Class E synthetic modifiers (e.g. storage from RF) or magic fallbacks (`1e6`) falsifies results. Missing or unphysical profiles must strictly evaluate to `NaN` to trigger full pruning. |
| `core/unified_engine/physics/eos/` | **HIGH** | Cubic EOS root finding and thermodynamic properties | Iterative solver stagnation or unphysical negative densities. |
| `analysis/material_balance.py` | **MEDIUM** | Post-simulation CO₂ accounting and reporting | Produces misleading verification graphs if mass balance equations are distorted. |
| `utils/las_parser.py` | **MEDIUM** | Petrophysical well log parsing | Ingestion errors or corrupt permeability tracks. |
| `ui/workers/optimization_worker.py` | **MEDIUM** | Multiprocessing/threading for optimization | Thread deadlocks, GUI freezing, unhandled Qt exceptions. |
| `ui/main_window.py`, `ui/widgets/` | **LOW** | Desktop graphical interface presentation | Visual layout glitches, disabled buttons, signal-slot disconnections. |
| `utils/preferences_manager.py` | **LOW** | User settings persistence | Corrupt JSON preferences file (handled gracefully by defaults). |
