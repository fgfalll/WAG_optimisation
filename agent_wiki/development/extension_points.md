# Extension Points Guide

This document outlines how to cleanly and safely extend the CO₂ EOR Optimizer codebase with new physics models, optimization algorithms, objective functions, or fluid models, without breaking the complex wrapper and caching layers.

---

## 1. Adding a New Recovery Model / Analytical Displacement Model

All recovery factor calculations currently run through `PhDHybridSurrogate` in [core/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/analytical_models.py) or `FastProfileGenerator` in [core/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/surrogate_engine.py).

### Step-by-Step Implementation:
1. **Inherit or Implement Base Interface**:
   Define your model in a modular file under `core/models/` or add to `core/analytical_models.py`. Ensure it implements:
   ```python
   def calculate_recovery_factor(
       self,
       pressure_psia: float,
       mmp_psia: float,
       co2_injected_hcpv: float,
       water_injected_hcpv: float = 0.0,
       **kwargs
   ) -> Tuple[float, Dict[str, float]]:
       """
       Returns:
           Tuple of (recovery_factor: float [0, 1], diagnostic_dict: dict)
       """
   ```
2. **Update `PhDHybridSurrogate.calculate_recovery_profile`**:
   In [core/analytical_models.py:756](file:///d:/rep/4.6/co2eor_optimizer/core/analytical_models.py#L756), add a dispatch branch or replace the component calculation.
   *Warning*: Never return values > 1.0 or < 0.0.
3. **Expose in `surrogate_engine.py`**:
   In [core/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/surrogate_engine.py), update `FastProfileGenerator.generate_profile` to accept your model name as an argument.
4. **Expose in GUI**:
   Add the new option to the combobox in [ui/widgets/parameter_input_widget.py](file:///d:/rep/4.6/co2eor_optimizer/ui/widgets/parameter_input_widget.py) and update config schema in `config/default_config.json`.

---

## 2. Adding a New Optimization Algorithm

Optimization algorithms are managed through `OptimizationEngine` in [core/optimisation_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py).

### Step-by-Step Implementation:
1. **Define Algorithm Type**:
   In [core/optimisation_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py), add your algorithm to the `AlgorithmType` enum:
   ```python
   class AlgorithmType(str, Enum):
       ...
       NEW_ALGO = "NEW_ALGO"
   ```
2. **Implement Runner Function**:
   Create a dedicated file `core/optimizers/new_algo_optimizer.py`.
   The runner must accept:
   - `objective_func: Callable[[np.ndarray], float]`
   - `bounds: List[Tuple[float, float]]`
   - `max_iterations: int`
   - `population_size: int`
   - `callback: Optional[Callable[[int, float, np.ndarray], None]]`
   - `cancellation_token: Optional[Event]`
3. **Dispatch in `OptimizationEngine._run_optimization`**:
   Add the dispatcher branch in [core/optimisation_engine.py:1000-1100](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py):
   ```python
   elif self.algorithm_type == AlgorithmType.NEW_ALGO:
       result = run_new_algo(self.objective_function, bounds, ...)
   ```
4. **Standardize Result Format**:
   Ensure results are mapped to `OptimizationResult`:
   ```python
   return OptimizationResult(
       optimal_parameters=best_params_dict,
       objective_value=best_val,
       convergence_history=history,
       success=True,
       message="Optimization converged successfully"
   )
   ```

---

## 3. Adding a New Objective Function or Economic Metric

Objectives are computed in [core/objectives/wrapper.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py) and [core/objective_calculator.py](file:///d:/rep/4.6/co2eor_optimizer/core/objective_calculator.py).

### Step-by-Step Implementation:
1. **Add Objective Key to `ObjectiveCalculator`**:
   In [core/objective_calculator.py](file:///d:/rep/4.6/co2eor_optimizer/core/objective_calculator.py), add calculation method:
   ```python
   def calculate_custom_metric(self, simulation_results: Dict[str, Any], params: EconomicParameters) -> float:
       ...
   ```
2. **Handle in `core/objectives/wrapper.py`**:
   Update `evaluate_objective()` to support multi-objective weighting for the new metric.
   *Critical Invariant*: Verify that profile arrays in `simulation_results` are checked with `len(arr) > 0`, NEVER `if arr:` (which triggers NumPy `ValueError`).
3. **Ensure Safe Default When EconomicParameters is None**:
   Always check `if econ_params is None:` before accessing pricing or discounting fields, to avoid triggering `UnboundLocalError` or `ValueError`.

---

## 4. Registering a Real Simulation Engine with `EngineFactory`

Currently, `EngineFactory.create_engine` routes everything to `SurrogateEngineWrapper`.

To connect a true numerical solver (e.g. `compositional_engine` or `Phys_engine_full`):
1. **Ensure Implementation Conforms to `BaseEngine`**:
   The engine class must implement:
   - `run_simulation(self, params: Dict[str, Any]) -> SimulationResult`
   - `get_profile(self) -> Dict[str, np.ndarray]`
2. **Update `core/engine_factory.py`**:
   Modify `EngineFactory.create_engine` lines 105–116 to conditionally instantiate the requested engine type instead of hardcoding `SurrogateEngineWrapper`:
   ```python
   if engine_type == EngineType.COMPOSITIONAL:
       from core.simulation.compositional_engine import CompositionalEngine
       return CompositionalEngine(config)
   ```
3. **Harmonize Output Profile Dictionary Schema**:
   Ensure your engine returns standard keys:
   `oil_rate`, `water_rate`, `gas_rate`, `co2_rate`, `reservoir_pressure`, `cumulative_oil`, `cumulative_co2_stored`.
   Units must conform to Field units (STB/D, MSCF/D, psia).
