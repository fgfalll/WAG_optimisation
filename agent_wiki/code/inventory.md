# Code Inventory: Modules, Classes, and Functions

This document summarizes the core code inventory across active subsystems in the repository.

---

## 1. Top Subsystems and Key Classes

### A. Optimization Engine (`core/optimisation_engine.py`)
- **Key Classes**:
  - `OptimizationEngine`: Central orchestrator for running multi-algorithm optimization (GA, BO, PSO, DE).
  - `OptimizationConfig`: Configuration dataclass for algorithm hyperparameters.
  - `OptimizationResults`: Container for optimal solutions, Pareto front, objective histories, and profiles.
- **Key Methods**:
  - `run_optimization()`: Executes selected algorithm loop.
  - `evaluate_candidate(x)`: Evaluates candidate solution vector via `EngineFactory`.
  - `_unpack_parameters(x)`: Converts 1D float vector into structured `EORParameters` and `OperationalParameters`.
  - `evaluate_scenario(params)`: Runs surrogate engine and computes penalty-adjusted objective fitness.

---

### B. Surrogate Simulation Engine (`core/engine_surrogate/`)
- **`surrogate_engine.py`**:
  - `SurrogateEngine`: High-speed semi-analytical simulation engine.
  - `SurrogateEngineWrapper`: Adapter implementing standard engine interface for `EngineFactory`.
  - Methods: `evaluate_scenario()`, `_calculate_engine_npv()`, `_calculate_co2_purchased_recycled()`, `_solve_pressure_ode()`.
- **`analytical_models.py`**:
  - `AnalyticalSurrogate`: Facade for analytical displacement and recovery models.
  - `PhDHybridRecoveryModel`: Hybrid Koval/Buckley-Leverett miscible/immiscible recovery calculation.
  - `KovalRecoveryModel`: Classical Koval (1963) miscible displacement model.
  - `BuckleyLeverettRecoveryModel`: Classical Buckley-Leverett (1942) immiscible displacement.
- **`profile_generator_fast.py`**:
  - `FastProfileGenerator`: Vectorized generator for oil, water, gas, and pressure time series.
  - Methods: `generate_profiles()`, `_generate_continuous_injection()`, `_generate_wag_injection()`, `_calculate_co2_trapping()`.
- **`surrogate_models.py`**:
  - `CO2StorageSurrogate`: Fast analytical calculation of structural, residual, and solubility trapping.

---

### C. Thermodynamic & Physical Property Models (`core/unified_engine/physics/`)
- **`core/unified_engine/physics/eos/__init__.py`**:
  - `PengRobinsonEOS`: Cubic Peng-Robinson equation of state with Rachford-Rice flash.
  - `CubicEOS`: Base class for cubic equations of state.
  - `ReservoirFluid`: Multi-component hydrocarbon/CO2 fluid composition container.
- **`core/unified_engine/physics/co2_properties.py`**:
  - `CO2Properties`: High-accuracy density, viscosity, and solubility calculation for pure and impure CO2 streams.
- **`evaluation/mmp.py`**:
  - `MMPParameters`: Input parameters for minimum miscibility pressure estimation.
  - `calculate_mmp()`: Multi-method dispatcher (Cronquist, Lee, Glaso, Alston, Yuan).

---

### D. Parameter Models & Constants (`core/data_models.py`)
- **Key Dataclasses**:
  - `ReservoirData`: Geometry, porosity, permeability, initial pressure/temperature, OOIP.
  - `EORParameters`: WAG ratio, cycle length, injection rate, gas fraction.
  - `OperationalParameters`: Well constraints, injector/producer counts, bottom-hole limits.
  - `EconomicParameters`: Oil price, CO2 purchase cost, recycling cost, discount rate, OPEX.
  - `PhysicalConstants`: Standard physical units, conversion factors, and universal gas constant.
  - `EmpiricalFittingParameters`: Calibration multipliers (transverse mixing, sweep tuning).

---

### E. User Interface Layer (`ui/`)
- **`ui/main_window.py`**: Main PyQt6 application window, toolbar, and tab container.
- **`ui/optimization_widget.py`**: Interactive setup for algorithms, objective weighting, and parameter bounds.
- **`ui/data_management_widget.py`**: Petrophysical data loading, PVT properties, and LAS log inspection.
- **`ui/workers/optimization_worker.py`**: Background QThread worker running optimization without blocking the GUI.
