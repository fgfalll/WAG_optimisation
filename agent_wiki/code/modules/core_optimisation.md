# Optimization Subsystem (`core/optimisation_engine.py`)

## 1. Overview & Architectural Role

The optimization subsystem coordinates metaheuristic search algorithms, parameter unrolling, constraint discretization, parallel candidate evaluation, and multi-objective Pareto front generation.

It interfaces directly between user-defined configuration constraints and the active `core/engine_surrogate/` simulation pipeline.

---

## 2. Module Inventory

| Module | LOC | Primary Classes / Functions | Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `core/optimisation_engine.py` | 4,507 | `OptimizationEngine`, `OptimizationConfig`, `OptimizationResults` | Master orchestrator for GA (PyGAD), BO (scikit-optimize), PSO (PySwarms), and DE (SciPy). | **CRITICAL** |
| `core/optimization_analysis.py` | 886 | `OptimizationAnalyzer` | Convergence history, hypervolume indicator, parallel coordinates, 3D objective scatter. | **LOW** |
| `core/parameter_manager.py` | 320 | `ParameterManager` | Maps continuous algorithm space $[0, 1]^d$ to discrete engineering decision variables. | **HIGH** |
| `core/bayesian_optimizer.py` | 410 | `BayesianOptimizer` | Gaussian Process surrogate with Expected Improvement (EI) and Upper Confidence Bound (UCB). | **MEDIUM** |

---

## 3. Key Classes & Execution Workflow

### `OptimizationEngine`
1. **Instantiation**: Takes `OptimizationConfig`, `ReservoirData`, and target objectives (`["npv", "rf", "co2_stored"]`).
2. **Bounds & Discretization**:
   - `WAG ratio`: typically $[0.5, 3.0]$ in discrete steps (e.g. 0.1).
   - `Cycle length`: typically $[30, 180]$ days in steps of 15 or 30 days.
   - `Injection rate`: $[1000, 15000]$ MSCF/d.
   - `Well counts`: integer grid bounds ($N_{\text{inj}} \in [1, 10], N_{\text{prod}} \in [1, 20]$).
3. **Candidate Evaluation Loop**:
   - `evaluate_candidate(chromosome)` unpacks 1D vector into `OperationalParameters` and `EORParameters`.
   - Calls `EngineFactory.create_engine("surrogate")`.
   - Obtains `SimulationResults` from `SurrogateEngineWrapper.evaluate_scenario()`.
   - Evaluates multi-objective scalar or vector fitness via `core/objectives/wrapper.py`.
4. **Failure Penalty Handling**:
   - Non-viable candidates (e.g. pressure exceeding fracture ceiling, zero wells, negative rates) return `FAILURE_PENALTY = -10^{12}`.
   - Eliminates invalid individuals immediately from the genetic pool.

---

## 4. Invariants for AI Agents

1. **Never scale field recovery by well count**: Adding production wells increases rate acceleration (earlier recovery) and pressure drawdown, but does not multiply ultimate OOIP.
2. **Zero chromosome synthesis**: If a simulation fails or encounters numerical singularity, return `FAILURE_PENALTY = -10^{12}` rather than imputing artificial recovery factors.
3. **PyGAD Integration**: PyGAD fitness function requires maximizing positive fitness; the objective wrapper negates minimization targets automatically.
