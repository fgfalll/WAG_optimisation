# Architecture Decision Records (ADRs)

This document captures the architectural decisions, trade-offs, and historical pivots in the CO₂ EOR Optimizer repository.

---

## ADR-001: Analytical Surrogate Engine Routing Dominance

### Status
Accepted (De facto implementation reality)

### Context
Running full numerical reservoir simulation (finite-difference compositional simulation, solving multicomponent flow and phase equilibria across thousands of grid blocks) requires minutes to hours per run. The optimization algorithms (GA, PSO, Bayesian Optimization) require $10^2$ to $10^4$ function evaluations.

### Decision
In [core/engine_factory.py:105-116](file:///d:/rep/4.6/co2eor_optimizer/core/engine_factory.py#L105-L116), the engine factory was modified to unconditionally route all simulation requests to `SurrogateEngineWrapper` ([core/engine_surrogate.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate.py)), which internally invokes `FastProfileGenerator` and `PhDHybridSurrogate`.

### Consequences
- **Pros**: Optimization loops execute in milliseconds per evaluation, enabling real-time UI interactivity and interactive GA convergence.
- **Cons**: `CompositionalEngine`, `UnifiedEngine`, and `Phys_engine_full` are effectively orphaned from the main workflow. Numerical multi-phase flow phenomena (viscous fingering, gravity segregation in heterogeneous 3D grids) are approximated by 1D/0D analytical correlations.

---

## ADR-002: Fast Profile Generator vs Explicit Time-Stepping Solvers

### Status
Accepted

### Context
Physics-based ODE solvers (`scipy.integrate.solve_ivp`) occasionally suffer stiffness, convergence failure, or performance bottlenecks when time steps vary during rapid pressure depletion or gas breakthrough.

### Decision
`FastProfileGenerator` in [core/surrogate_engine.py:465-980](file:///d:/rep/4.6/co2eor_optimizer/core/surrogate_engine.py#L465-L980) replaces numerical grid time-stepping with vectorized analytical decline curves (exponential breakthrough, hyperbolic decline) modulated by Buckley-Leverett and Koval fractional flow calculations, coupled with a simple 0D tank material balance ODE.

### Consequences
- **Pros**: Zero convergence failures, deterministic $O(1)$ evaluation speed.
- **Cons**: Loss of spatial resolution. Wellbore pressure-rate interference and complex fault boundaries cannot be resolved. Risk of unit inconsistencies in pressure integration if volumetric conversion factors ($B_g$, $B_o$) are mishandled.

---

## ADR-003: Centralized Error Handling vs Local Silent Fallbacks

### Status
In Progress / Technical Debt

### Context
Legacy code contained extensive bare `except:` clauses returning arbitrary numerical constants (e.g. `recovery_factor = 0.35` or `npv = 0.0`), hiding model divergence or invalid input states.

### Decision
Introduction of [utils/error_handler.py](file:///d:/rep/4.6/co2eor_optimizer/utils/error_handler.py) (`report_caught_error`, `safe_execute`, `ErrorSeverity`, `ErrorCategory`). The project rule mandates no silent error suppression.

### Consequences
- **Pros**: Traceability of calculation failures, structured logging, user-facing error dialogs without unhandled crashes.
- **Cons**: Incomplete migration. 2,459 fallback branches still exist across 224 modules, requiring continuous refactoring to conform to `ERROR_HANDLING_GUIDELINES.md`.

---

## ADR-004: Threading and Worker Architecture for PyQt6 UI

### Status
Accepted

### Context
PyQt6 GUI freezes if scientific calculations or optimization routines execute on the main GUI event loop thread.

### Decision
Background execution using `QThread` and dedicated worker classes (`ui/workers/optimization_worker.py`, `ui/workers/simulation_worker.py`), communicating with the main thread strictly through `pyqtSignal` events (`progress`, `finished`, `error_occurred`).

### Consequences
- **Pros**: UI remains responsive, user can cancel running optimizations, real-time charting updates smoothly.
- **Cons**: State synchronization complexity; worker must catch all thread exceptions, as unhandled exceptions in a Qt worker thread crash the entire Python process without standard traceback output.

---

## ADR-005: Surrogate Engine Identity as Intermediate Physics-Informed Simulator & Decoupled Solvent-Extended PVT / Geomechanics

### Status
Accepted

### Context
A misconception previously existed that "surrogate" implied an empirical black-box machine-learning model or curve-fit. Furthermore, earlier versions suffered from:
1. Conflation of hydrodynamic transport saturation ($S_g$) with thermodynamic composition ($x_{\text{CO2}}$).
2. Omission of in-situ geomechanical stress paths and fault/caprock reactivation modeling.
3. Inverted fractional flow logic (SCI-FLAW-01).
4. Ad-hoc output keys without standardized 4-stream reporting.

### Decision
1. **Engine Identity**: Formally define `core/engine_surrogate` as an **intermediate, physics-informed reduced-order reservoir simulator** positioned between 3D PDE grid solvers and classical 0D tank models. It functions as a general reservoir simulator embedded with specialized CO₂ EOR physics.
2. **Solvent-Extended PVT (`pvt_state.py`)**: Implement `SolventExtendedPVTEngine` decoupling $S_g$ from $x_{\text{CO2}}$, evaluating Peng-Robinson EOS for supercritical CO₂ downhole density and FVF ($B_{\text{CO2}}$), oil swelling $S_F(P, x_{\text{CO2}})$, viscosity thinning $\mu_o(P, x_{\text{CO2}})$, and surface multi-stage separation.
3. **Geomechanics & Containment (`geomechanics_fault.py`)**: Implement `GeomechanicsFaultModel` evaluating the reservoir horizontal stress path ($\Delta \sigma_h = \gamma_h \Delta P_p$), Mohr-Coulomb fault slip tendency ($T_s$), caprock shear/tensile failure envelopes, and dynamic leakage rates.
4. **Authentic Viscous Fingering**: Restore authentic Koval (1963) and Todd-Longstaff (1972) formulation where effective Koval factor $K = H_k \cdot E_{\text{eff}}$ and fractional flow $f_g = K S / (1 + S(K-1))$ strictly satisfy $\partial f_g / \partial M > 0$.
5. **Standardized 4-Stream Delivery**: Guarantee that `evaluate_scenario()` returns standardized streams for (1) Crude oil, (2) Natural gas (sales gas separate from CO₂), (3) Water (brine & water cut), and (4) Injection agent (CO₂ & WAG water with facility compressor throughput limits).

### Consequences
- **Pros**: Retains ultra-fast execution (~10 ms per evaluation) while providing rigorous multi-component thermodynamic and geomechanical fidelity. Eliminates SCI-FLAW-01 and prevents unphysical proxy behavior.
- **Cons**: Introduces additional parameters (caprock strength, fault orientation, compressor capacity) which must be gracefully defaulted when absent in simplified user scenarios.

---

## ADR-006: Deprecation of EngineFactory and Legacy Engines in Favor of Direct Single-Engine Architecture

### Status
Accepted

### Context
The codebase historically retained 5 separate engine directories (`compositional_engine`, `unified_engine`, `engine_simple`, `simulation/`, and `engine_surrogate`), mediated by an `EngineFactory`. However:
1. `EngineFactory` was a deceptive abstraction: all evaluation requests were unconditionally hardwired to `SurrogateEngineWrapper`, while giving callers the false impression of an engine-switching capability.
2. Competing legacy engines were disconnected from optimization runs, unmaintained, and in the case of `core/unified_engine`, afflicted with critical scientific and numerical flaws (SCI-FLAW-04, SCI-FLAW-08).
3. Multiple modules imported `CubicEOS`, `PengRobinsonEOS`, or `ReservoirFluid` from `core/unified_engine/physics/eos` solely for approximate static logs (e.g. `b_gas_rb_per_mscf`), creating unnecessary coupling to dead engine trees. The production surrogate engine already possesses its own self-contained, analytically solved cubic Peng-Robinson EOS in `core/engine_surrogate/pvt_state.py`.

### Decision
1. **Move Legacy Engines to Root `deprecated/`**:
   - `core/engine_factory.py` $\to$ `deprecated/core/engine_factory.py`
   - `core/compositional_engine/` $\to$ `deprecated/core/compositional_engine/`
   - `core/unified_engine/` $\to$ `deprecated/core/unified_engine/`
   - `core/engine_simple/` $\to$ `deprecated/core/engine_simple/`
   - `ui/widgets/engine_selection_widget.py` $\to$ `deprecated/ui/widgets/engine_selection_widget.py`
   - Dead validation scripts $\to$ `deprecated/tests/validation/`
2. **Direct Production Wiring**:
   - Wire `core/optimisation_engine.py` and `core/data_integration_engine.py` directly to `SurrogateEngineWrapper` from `core.engine_surrogate`.
   - Calculate formation volume factor $B_{\text{CO2}}$ directly via `SolventExtendedPVTEngine.calculate_co2_fvf_rb_per_mscf(p_init)`.
   - Update remaining UI and analysis callers to reference `deprecated/` or fall back cleanly.

### Consequences
- **Pros**: Eliminates 100% of architectural ambiguity. Ensures all developers and AI agents target `core/engine_surrogate/` as the single authoritative simulation engine. Cleanses the active production root of dead or scientifically flawed solvers.
- **Cons**: Historical standalone scripts that expected `core.engine_factory` or `core.unified_engine` must be executed with reference to `deprecated/`.

