# Codebase Module Map & Inventory

## 1. Inventory Summary

The active codebase consists of **202 Python modules** comprising **~85,530 lines of code** (excluding `.venv`, `.git`, `.idea`, and cache artifacts).

| Directory / Package | Python Files | Lines of Code | Primary Role |
| :--- | :---: | :---: | :--- |
| `core/` | 80 | ~38,500 | Core mathematical, optimization, and simulation engines |
| `ui/` | 52 | ~18,200 | PyQt6 user interface, widgets, custom charts, workers |
| `tests/` | 19 | ~7,500 | Pytest test suite, hypothesis property tests, benchmark tests |
| `analysis/` | 10 | ~5,800 | Material balance, breakthrough physics, DCA, UQ engine |
| `utils/` | 14 | ~5,380 | Logging, preferences, configuration, error handling, run export, LAS log parser |
| `scripts/` | 4 | ~1,100 | Maintenance, LAS log parsing, physics audit scripts |
| `evaluation/` | 1 | ~450 | Minimum Miscibility Pressure (MMP) analytical correlations |
| Root files | 3 | ~850 | Application entry point (`main.py`), package init |

---

## 2. Directory Breakdown & Key Modules

### `core/` Package Details

#### 1. Optimization Orchestration
- [core/optimisation_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py): Main optimizer class `OptimizationEngine` (~4,250 lines). Integrates PyGAD, Bayesian Optimization, PySwarms, and Differential Evolution. Handles candidate vector discretization, constraint checks, and Pareto front generation. Plotting delegations routed to `PlottingManager`.
- [core/plotting_manager.py](file:///d:/rep/4.6/co2eor_optimizer/core/plotting_manager.py): Centralized manager for well schedules, coverage (trend/min/max), Euclidean distance heatmaps, GA distributions, hybrid recovery analysis, and Pareto fronts.

#### 2. Surrogate Engine Subsystem (`core/engine_surrogate/`)
- [surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py): Master wrapper for simulation evaluation (`SurrogateEngine` and `SurrogateEngineWrapper`). Runs analytical recovery prediction, generates dynamic rate profiles via `FastProfileGenerator`, solves deliverability-coupled implicit material balance for pressure, calculates economic NPV, and tracks CO₂ purchased vs recycled.
- [analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py): Core recovery calculations:
  - `MiscibleSurrogate`: Koval (1963) heterogeneity factor + Todd-Longstaff effective mobility ratio.
  - `ImmiscibleSurrogate`: Buckley-Leverett (1942) fractional flow + Craig areal sweep + Johnson vertical sweep.
  - `BuckleyLeverettSurrogate`: Standalone 1D fractional flow solver with Welge shock construction.
  - `HybridSurrogate`: Sigmoidal pressure/MMP interpolation.
  - `KovalSurrogate` (`KovalRecoveryModel` alias): Authentic Koval recovery model.
  - `PhDHybridSurrogate`: Primary research model with dynamic C7+ shifts, HCPVI mass balance, and numerical gradient finite-differencing.
- [profile_generator_fast.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py): Single source of truth for rate profile synthesis (oil, water, gas, CO₂ injection) across continuous, WAG, SWAG, Huff-n-Puff, tapered, and pulsed injection schemes. Implements Composite Vogel-Darcy IPR single-well deliverability and mass-conserving WAG mobility buffering.
- [pvt_state.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/pvt_state.py): `SolventExtendedPVTEngine` evaluating Peng-Robinson EOS density, solvent swelling, and viscosity reduction.
- [geomechanics_fault.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/geomechanics_fault.py): Stress path, caprock seal integrity, and Mohr-Coulomb fault slip calculations.
- [surrogate_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py): Factory `AnalyticalSurrogate`, sweep efficiency calculations (Craig, Johnson), and Corey residual trapping metrics.

#### 3. Geological Modeling (`core/geology/`)
- [geostatistical_modeling.py](file:///d:/rep/4.6/co2eor_optimizer/core/geology/geostatistical_modeling.py): Geostatistical grid generation (`create_geostatistical_grid`) with spatial correlation models (spherical, exponential, gaussian).
- `__init__.py`: Cleanly exports `create_geostatistical_grid` with a lazy deprecated import shim for legacy `GeologyEngine` (relocated to `deprecated/core/geology/`).

#### 4. Simulation Adapters (`core/simulation/`)
- `__init__.py`: Exports active `FastProfileGenerator`.
- `recovery_models.py`: Backward-compatible deprecation shim redirecting to `deprecated.core.simulation.recovery_models`.
- `profile_generator.py`: Backward-compatible deprecation shim redirecting to `deprecated.core.simulation.profile_generator`.
- `simulator_exporter.py`: Backward-compatible deprecation shim redirecting to `utils.cmg_exporter`.

#### 5. Data & Objectives Infrastructure
- [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py): Core typed dataclasses: `ReservoirData`, `EORParameters`, `OperationalParameters`, `EconomicParameters`, `PhysicalConstants`, `PVTProperties`, `FluidProperties`.
- [core/data_integration_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py): Multi-source data ingestion, preprocessing, and quality metrics using `PhysicalConstants`.
- [core/objectives/wrapper.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py): Evaluates multi-objective fitness, geomechanical containment limits (EPA Class VI 90% limit), and environmental leakage penalties.
- [core/objectives/storage.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/storage.py): CO₂ storage efficiency and geomechanical containment scoring.
- [core/objectives/economic.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/economic.py): Standalone cash flow and NPV utilities.

#### 6. Deprecated Subsystems (`deprecated/core/`)
- `deprecated/core/compositional_engine/`: 3D multi-block compositional simulator sandbox.
- `deprecated/core/unified_engine/`: Legacy monolithic simulator and EOS sandbox.
- `deprecated/core/engine_simple/`: 0D material balance prototype.
- `deprecated/core/engine_factory.py`: Legacy multi-engine router.
- `deprecated/core/engine_surrogate/`: Response surface ML prototypes (`response_surfaces.py`, `feature_transformer.py`, `training_data.py`, `model_factory.py`).
- `deprecated/core/geology/geology_engine.py`: Uncalibrated `GeologyEngine`.
- `deprecated/core/simulation/`: Legacy `recovery_models.py` and `profile_generator.py`.
