# Codebase Module Map & Inventory

## 1. Inventory Summary

The active codebase consists of **204 Python modules** comprising **85,716 lines of code** (excluding `.venv`, `.git`, `.idea`, and cache artifacts).

| Directory / Package | Python Files | Lines of Code | Primary Role |
| :--- | :---: | :---: | :--- |
| `core/` | 80 | ~38,500 | Core mathematical, optimization, and simulation engines |
| `ui/` | 52 | ~18,200 | PyQt6 user interface, widgets, custom charts, workers |
| `tests/` | 18 | ~7,400 | Pytest test suite, hypothesis property tests, benchmark tests |
| `analysis/` | 10 | ~5,800 | Material balance, breakthrough physics, DCA, UQ engine |
| `utils/` | 13 | ~5,200 | Logging, preferences, configuration, error handling, run export |
| `scripts/` | 4 | ~1,100 | Maintenance, LAS log parsing, physics audit scripts |
| `evaluation/` | 1 | ~450 | Minimum Miscibility Pressure (MMP) analytical correlations |
| `parsers/` | 3 | ~380 | LAS well log parsers and input file validators |
| Root files | 3 | ~850 | Application entry point (`main.py`), package init |

---

## 2. Directory Breakdown & Key Modules

### `core/` Package Details

#### 1. Optimization Orchestration
- [core/optimisation_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py): Main optimizer class `OptimizationEngine` (3,800+ lines). Integrates PyGAD, Bayesian Optimization, PySwarms, and Differential Evolution. Handles candidate vector discretization, constraint checks, and Pareto front generation.
- [core/engine_factory.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_factory.py): Factory interface `EngineFactory`. Routes 100% of simulation evaluations strictly to `SurrogateEngineWrapper`.
- [core/optimization_analysis.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimization_analysis.py): Convergence analytics, hypervolume metrics, and Pareto front analysis.
- [core/plotting_manager.py](file:///d:/rep/4.6/co2eor_optimizer/core/plotting_manager.py): Centralized manager for well schedules, coverage (trend/min/max), Euclidean distance heatmaps, and Pareto fronts.

#### 2. Surrogate Engine Subsystem (`core/engine_surrogate/`)
- [surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py): Master wrapper for simulation evaluation. Runs analytical recovery prediction, generates dynamic rate profiles via `FastProfileGenerator`, solves deliverability-coupled implicit material balance for pressure, calculates economic NPV, and tracks CO₂ purchased vs recycled.
- [analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py): Core recovery calculations:
  - `MiscibleSurrogate`: Koval (1963) heterogeneity factor + Todd-Longstaff effective mobility ratio.
  - `ImmiscibleSurrogate`: Buckley-Leverett (1942) fractional flow + Craig areal sweep + Johnson vertical sweep.
  - `BuckleyLeverettSurrogate`: Standalone 1D fractional flow solver.
  - `HybridSurrogate`: Sigmoidal pressure/MMP interpolation.
  - `PhDHybridRecoveryModel`: Primary research model with dynamic C7+ shifts, HCPVI mass balance, and numerical gradient finite-differencing.
- [profile_generator_fast.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py): Single source of truth for rate profile synthesis (oil, water, gas, CO₂ injection) across continuous, WAG, SWAG, Huff-n-Puff, tapered, and pulsed injection schemes. Implements Composite Vogel-Darcy IPR single-well deliverability and mass-conserving WAG mobility buffering.
- [surrogate_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py): Factory `AnalyticalSurrogate`, sweep efficiency calculations (Craig, Johnson), and Corey residual trapping metrics.

#### 3. Compositional Engine Subsystem (`core/compositional_engine/`) - Dormant
- `compositional_engine.py`: 1D compositional numerical simulator engine.
- `flow/compositional_solver.py`: Finite-volume flow solver.
- `flow/transmissibility.py`: Inter-block harmonic transmissibility calculations.
- `phase_behavior/flash_calculator.py`: Two-phase vapor-liquid equilibrium (VLE) flash solver using successive substitution and Newton-Raphson with Peng-Robinson EOS.

#### 4. Unified Engine Subsystem (`core/unified_engine/`) - Dormant (except EOS)
- `engines/detailed_engine.py`: 3D multi-phase numerical simulator.
- `physics/eos/__init__.py`: **ACTIVELY IMPORTED**. Houses `CubicEOS`, `PengRobinsonEOS`, and `ReservoirFluid` utilized for fluid characterization.

#### 5. Data & Objectives Infrastructure
- [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py): Core typed dataclasses: `ReservoirData`, `EORParameters`, `OperationalParameters`, `EconomicParameters`, `PhysicalConstants`, `SimulationResults`.
- [core/objectives/wrapper.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py): Evaluates multi-objective fitness, geomechanical containment limits (EPA Class VI 90% limit), and environmental leakage penalties.
- [core/objectives/storage.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/storage.py): CO₂ storage efficiency calculations.
- [core/objectives/economic.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/economic.py): Standalone cash flow models.
