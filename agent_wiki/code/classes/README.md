# Class Directory & Architectural Catalog

## 1. Overview

This catalog documents the core classes across all active and authoritative subsystems in the `co2eor_optimizer` codebase.

Every entry outlines the class's purpose, file location, primary attributes, key methods, scientific equations, assumptions, and change safety tier.

---

## 2. Core Class Catalog

| Class Name | Module Location | Subsystem Role | Change Safety Risk |
| :--- | :--- | :--- | :---: |
| **`SurrogateEngine`** | `core/engine_surrogate/surrogate_engine.py` | Production simulation evaluator, coupled IPR pressure solver, NPV calculator. | **CRITICAL** |
| **`SurrogateEngineWrapper`** | `core/engine_surrogate/surrogate_engine.py` | Interface adapter connecting `SurrogateEngine` to `EngineFactory`. | **CRITICAL** |
| **`FastProfileGenerator`** | `core/engine_surrogate/profile_generator_fast.py` | Time series rate generator; Composite Vogel-Darcy IPR; WAG phase buffering. | **HIGH** |
| **`PhDHybridRecoveryModel`** | `core/engine_surrogate/analytical_models.py` | Hybrid miscible/immiscible ultimate recovery factor ($RF$) calculator. | **CRITICAL** |
| **`KovalRecoveryModel`** | `core/engine_surrogate/analytical_models.py` | Analytical Koval miscible displacement model ($H_k, M_e$). | **HIGH** |
| **`BuckleyLeverettRecoveryModel`** | `core/engine_surrogate/analytical_models.py` | Analytical 1D Buckley-Leverett waterflood displacement model. | **MEDIUM** |
| **`CO2StorageSurrogate`** | `core/engine_surrogate/surrogate_models.py` | Analytical structural, residual, and solubility trapping calculator. | **MEDIUM** |
| **`OptimizationEngine`** | `core/optimisation_engine.py` | Metaheuristic search orchestrator (GA, BO, PSO, DE). | **CRITICAL** |
| **`OptimizationConfig`** | `core/optimisation_engine.py` | Dataclass holding algorithm hyperparameters and search bounds. | **MEDIUM** |
| **`OptimizationResults`** | `core/optimisation_engine.py` | Container for optimal parameters, Pareto front, and performance metrics. | **LOW** |
| **`ObjectiveFunctions`** | `core/objectives/wrapper.py` | Multi-objective fitness evaluator and geomechanical penalty calculator. | **HIGH** |
| **`ReservoirData`** | `core/data_models.py` | Dataclass for petrophysical, fluid, and reservoir geometry parameters. | **CRITICAL** |
| **`EORParameters`** | `core/data_models.py` | Dataclass for WAG ratio, cycle length, injection rates, and gas fractions. | **CRITICAL** |
| **`OperationalParameters`** | `core/data_models.py` | Dataclass for well counts, BHP limits, and project lifetime. | **HIGH** |
| **`EconomicParameters`** | `core/data_models.py` | Dataclass for oil price, CO₂ purchase cost, recycling cost, and discount rate. | **HIGH** |
| **`PengRobinsonEOS`** | `core/unified_engine/physics/eos/__init__.py` | Cubic Peng-Robinson equation of state and Rachford-Rice flash solver. | **HIGH** |
| **`ReservoirFluid`** | `core/unified_engine/physics/eos/__init__.py` | Hydrocarbon/CO₂ composition container with component properties ($T_c, P_c, \omega$). | **HIGH** |
| **`CO2Properties`** | `core/unified_engine/physics/co2_properties.py` | Thermodynamic density, viscosity, and solubility calculation for CO₂. | **MEDIUM** |

---

## 3. Detailed Class Documentation

- [surrogate_classes.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/classes/surrogate_classes.md): Complete specifications for simulation surrogate classes.
- [optimization_classes.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/classes/optimization_classes.md): Specifications for optimization engine and objective wrappers.
- [physical_data_classes.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/classes/physical_data_classes.md): Specifications for data models, EOS, and thermodynamic properties.
