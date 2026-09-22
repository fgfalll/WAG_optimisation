# Code Subsystems & Module Index

## 1. Subsystem Architecture Overview

The `co2eor_optimizer` codebase spans 204 Python modules (85,937 lines of code) organized into distinct functional layers.

```
co2eor_optimizer/
├── core/                                   # Primary computation and simulation engine
│   ├── engine_surrogate/                   # [ACTIVE 100%] High-speed semi-analytical simulation proxy
│   ├── optimisation_engine.py              # [ACTIVE] Metaheuristic optimization algorithms (GA, BO, PSO, DE)
│   ├── objectives/                         # [ACTIVE] Multi-objective fitness evaluators and penalties
│   ├── data_models.py                      # [ACTIVE] Typed dataclasses for physical and operational data
│   ├── unified_engine/                     # [DORMANT] 3D compositional simulator (only physics/eos/ active)
│   ├── compositional_engine/               # [LEGACY] 1D finite-difference compositional simulator
│   ├── engine_simple/                      # [LEGACY] Material balance engine
│   └── simulation/                         # [DEPRECATED] Legacy wrappers around surrogate models
├── evaluation/                             # Scientific screening and correlations
│   ├── mmp.py                              # [ACTIVE] Minimum Miscibility Pressure correlations
│   └── economic_analyzer.py                # Economic metrics and cash flow models
├── analysis/                               # Post-processing, diagnostic, and uncertainty tools
│   ├── material_balance.py                 # [ACTIVE] Mass and carbon conservation verification
│   ├── decline_curve_analysis.py           # [ACTIVE] Arps DCA fitting
│   ├── sensitivity_analyzer.py             # Morris and Sobol global sensitivity analysis
│   └── uq_engine.py                        # Monte Carlo and Polynomial Chaos Expansion UQ
├── ui/                                     # PyQt6 Graphical User Interface
│   ├── main_window.py                      # Application window, menu bars, report generation
│   ├── optimization_widget.py              # Parameter bounds, algorithm selection, run controls
│   ├── sensitivity_widget.py               # Interactive sensitivity plotting
│   ├── uq_widget.py                        # Uncertainty quantification layout
│   └── widgets/                            # Reusable UI components, dialogs, and viewers
└── utils/                                  # Common utilities and infrastructure
    ├── run_exporter.py                     # Standardized run export (JSON, CSV, NetCDF)
    ├── cache_manager.py                    # Calculation memoization and serialization
    ├── hardware_detector.py                # CPU, GPU, and thread topology detection
    └── units_converter.py                  # Field to SI unit conversion pipeline
```

---

## 2. Module Classification by Activity & Risk

| Subsystem | Primary Modules | Status | Modification Risk |
| :--- | :--- | :--- | :---: |
| **Active Surrogate Engine** | `core/engine_surrogate/surrogate_engine.py`<br>`core/engine_surrogate/analytical_models.py`<br>`core/engine_surrogate/profile_generator_fast.py`<br>`core/engine_surrogate/surrogate_models.py` | **ACTIVE (100% of runs)** | **CRITICAL** |
| **Optimization Subsystem** | `core/optimisation_engine.py`<br>`core/optimization_analysis.py`<br>`core/objectives/wrapper.py` | **ACTIVE** | **HIGH** |
| **Data & Parameter Models** | `core/data_models.py`<br>`core/state_models.py`<br>`core/well_models.py` | **ACTIVE** | **CRITICAL** |
| **EOS & Fluid Properties** | `core/unified_engine/physics/eos/`<br>`core/unified_engine/physics/co2_properties.py`<br>`evaluation/mmp.py` | **ACTIVE** | **HIGH** |
| **Post-Processing & Analysis** | `analysis/material_balance.py`<br>`analysis/decline_curve_analysis.py`<br>`analysis/sensitivity_analyzer.py`<br>`analysis/uq_engine.py` | **ACTIVE** | **MEDIUM** |
| **User Interface (PyQt6)** | `ui/main_window.py`<br>`ui/optimization_widget.py`<br>`ui/sensitivity_widget.py`<br>`ui/uq_widget.py` | **ACTIVE (UI)** | **LOW** |
| **Dormant 3D Engine** | `core/unified_engine/` (except `physics/eos/`) | **DORMANT** | **MEDIUM (Isolated)** |
| **Legacy Simulators** | `core/compositional_engine/`<br>`core/engine_simple/`<br>`core/simulation/` | **LEGACY / DEPRECATED** | **LOW (Isolated)** |

---

## 3. Subsystem Detail Documents

- [core_engine_surrogate.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/core_engine_surrogate.md): Deep-dive into active simulation modules.
- [core_optimisation.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/core_optimisation.md): Algorithms, parameter unrolling, and candidate evaluation.
- [core_objectives_data.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/core_objectives_data.md): Objectives, penalty wrappers, and typed dataclasses.
- [evaluation_analysis.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/evaluation_analysis.md): MMP, material balance, DCA, and UQ analysis.
- [dormant_engines.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/dormant_engines.md): Dormant 3D and 1D simulation packages.
- [ui_utils.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/modules/ui_utils.md): GUI widgets, workers, and infrastructure utilities.
