# Optimization & Objective Classes

## 1. `OptimizationEngine`

- **File**: `core/optimisation_engine.py`
- **Role**: Coordinates multi-algorithm metaheuristic search (GA, BO, PSO, DE), candidate decoding, parallel workers, and results packaging.
- **Modification Risk**: **CRITICAL**

### Primary Attributes
- `config: OptimizationConfig`: Algorithm hyperparameters, population size, max generations, mutation/crossover rates.
- `reservoir_data: ReservoirData`: Base petrophysical state.
- `objective_weights: Dict[str, float]`: Relative weights for NPV, RF, and CO₂ storage.
- `engine_factory: EngineFactory`: Factory dispensing `SurrogateEngineWrapper`.
- `results: Optional[OptimizationResults]`: Output container populated upon run completion.

### Key Methods
- `run_optimization() -> OptimizationResults`:
  Dispatches to algorithm-specific runner:
  - `_run_genetic_algorithm()` (PyGAD)
  - `_run_bayesian_optimization()` (scikit-optimize)
  - `_run_particle_swarm()` (PySwarms)
  - `_run_differential_evolution()` (SciPy)
- `evaluate_candidate(chromosome: np.ndarray) -> float`:
  Unpacks decision variables vector, instantiates surrogate engine, runs simulation, applies penalties, and returns scalar fitness score to optimizer.
- `_unpack_parameters(chromosome: np.ndarray) -> Tuple[EORParameters, OperationalParameters]`:
  Converts continuous floating-point array into discrete operational variables (e.g. WAG ratio, cycle time, well count).

---

## 2. `OptimizationConfig`

- **File**: `core/optimisation_engine.py`
- **Role**: Strongly typed configuration dataclass for optimization algorithms.
- **Modification Risk**: **MEDIUM**

### Attributes
- `algorithm: str`: Algorithm name (`"genetic_algorithm"`, `"bayesian"`, `"particle_swarm"`, `"differential_evolution"`).
- `population_size: int`: Population / swarm size (e.g. 50).
- `max_iterations: int`: Maximum generations / iterations (e.g. 100).
- `mutation_rate: float`: Mutation probability for GA (default 0.10).
- `crossover_rate: float`: Crossover probability for GA (default 0.80).
- `objectives: List[str]`: Target objectives (e.g. `["npv", "rf", "co2_stored"]`).
- `parallel_evaluations: bool`: Whether to evaluate population in parallel via multiprocessing.

---

## 3. `ObjectiveFunctions`

- **File**: `core/objectives/wrapper.py`
- **Role**: Multi-objective fitness calculator applying environmental, pressure, and operational constraint penalties.
- **Modification Risk**: **HIGH**

### Key Methods
- `calculate_fitness(simulation_results: SimulationResults, weights: Dict[str, float]) -> float`:
  Computes composite fitness:
  $$\text{Fitness} = w_{\text{npv}} \cdot \text{NPV}_{\text{norm}} + w_{\text{rf}} \cdot \text{RF}_{\text{norm}} + w_{\text{co2}} \cdot \text{Storage}_{\text{norm}} - \text{Penalties}$$
- `_calculate_geomechanical_penalty(results: SimulationResults) -> float`:
  Penalizes reservoir pressures exceeding regulatory safety limits:
  $$P_{\text{penalty}} = 10^6 \cdot \left( \frac{\max(0, P_{\text{sandface}} - 0.90 P_{\text{frac}})}{P_{\text{frac}}} \right)^2$$
