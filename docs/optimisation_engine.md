# Optimization Engine Inner Workings

## 1. Engine Overview

The `OptimizationEngine` is the core orchestration layer responsible for discovering optimal operational parameters for CO2 Enhanced Oil Recovery (EOR) projects. It links reservoir data, physical constraints, and economic models with advanced search algorithms to maximize targeted objectives.

### Purpose

- **Design Discovery**: Autonomously search large, non-linear, mixed-continuous parameter spaces (e.g., injection rates, target pressures, WAG ratios).
- **Multi-Objective Trade-offs**: Resolve competing operational goals (e.g., maximizing NPV vs. maximizing CO2 Storage).
- **Stochastic & Global Search**: Prevent stagnation in local optima using population-based metaheuristics and probabilistic models.

### Architecture

```
OptimizationEngine
├── Optimization Algorithms (pygad, bayes_opt)
│   ├── Genetic Algorithm (Single-Objective)
│   ├── NSGA-II (Multi-Objective Pareto)
│   ├── Bayesian Optimization (Gaussian Process)
│   └── Hybrid Strategies (GA → BO, NSGA-II → BO)
├── ObjectiveFunctions (NPV, RF, Storage Evaluation)
├── SurrogateBreakthrough (Analytical Physical Bounds)
└── multiprocessing.ProcessPoolExecutor (Parallel Evaluation)
```

---

## 2. Optimization Algorithms

The engine employs multiple mathematical strategies, tailored for different dimensionalities and objective topologies.

### 2.1 Genetic Algorithm (Single-Objective)
Uses `pygad` to perform a stochastic global search.
- **Fitness Evaluation**: Evaluates a single combined scalar value (e.g., Net Present Value).
- **Elitism & Selection**: Retains the top performers (`keep_elitism`) and uses tournament selection to breed the next generation.
- **Adaptive Mutation**: Uses dynamically scaling mutation rates to encourage early exploration and late-stage exploitation.
- **Stagnation Handling**: Monitors fitness history; if the population saturates at a local optimum, it triggers a "Stale Restart" by injecting random diverse individuals into the population while preserving elites.

### 2.2 NSGA-II (Multi-Objective)
Non-dominated Sorting Genetic Algorithm II is used to handle conflicting objectives (e.g., Economics vs. Storage) without requiring arbitrary weighting functions. The engine relies on two primary mathematical mechanisms:

1. **Fast Non-Dominated Sorting**:
   The population is partitioned into hierarchical Pareto fronts ($F_1, F_2, \dots, F_k$). A solution $x_1$ dominates $x_2$ ($x_1 \prec x_2$) if it is no worse in any objective and strictly better in at least one. The sorting assigns a $rank$ to each solution based on its front number (lower is better).
2. **Crowding Distance ($d_i$)**:
   To maintain population diversity, the density of solutions surrounding a particular point is estimated. For the $m$-th objective, the distance is calculated as:
   $$d_i = \sum_{m=1}^{M} \frac{f_m(i+1) - f_m(i-1)}{f_m^{max} - f_m^{min}}$$
   Boundary points are assigned $d = \infty$.
3. **Crowded Comparison Operator ($\prec_n$)**:
   During tournament selection, solution $i$ wins against $j$ if $rank(i) < rank(j)$ OR if they share the same rank but $d_i > d_j$ (i.e., solution $i$ resides in a less crowded region).

### 2.3 Bayesian Optimization (Gaussian Process)
Utilizes `bayes_opt` for sample-efficient exploitation of smooth objective manifolds.
- **Surrogate Mapping**: Fits a Gaussian Process (GP) to the evaluated points.
- **Acquisition Function**: Uses Upper Confidence Bound (UCB) parameterized by $\kappa$ (exploration) and $\xi$ (exploitation) to suggest the next evaluation point.
- **Trust Regions**: Enforces dynamic search bounds that shrink upon successive failures and expand upon successes, focusing the GP on promising local neighborhoods.

### 2.4 Hybrid NSGA-II + BO (Механізм передачі на основі просторової різноманітності)
The engine supports a two-phase hybrid approach (`hybrid_nsga2_bo`) that combines the global Pareto exploration of NSGA-II with the local exploitation of Bayesian Optimization. The critical component linking these phases is a **transmission mechanism based on spatial diversity** (механізм передачі на основі просторової різноманітності).

- **Parameter Space Normalization**: To prevent scaling biases, all parameter bounds are normalized to a $[0,1]$ range.
- **Max-Min Diversity Selection**: Instead of blindly seeding the BO with the top fitness scores, the algorithm calculates a cross-population Euclidean distance matrix in the normalized parameter space:
  $$D_{ij} = \sqrt{\sum_{k=1}^{N_{params}} \left( x_{ik}^{(norm)} - x_{jk}^{(norm)} \right)^2}$$
- **Handoff (Передача)**: The algorithm iteratively selects candidates from the NSGA-II Pareto front that maximize the minimum distance ($D_{min} > diversity\_threshold$) to the already selected subset. These geometrically sparse points are injected into the Gaussian Process as initial probes (`n_initial_points`), guaranteeing that the BO phase explores isolated regions of the Pareto manifold simultaneously, drastically accelerating global convergence.

---

## 3. Objective Evaluation and Penalties

The engine delegates the translation of production profiles into fitness scalars to the `ObjectiveFunctions` component.

### 3.1 Core Objectives
- **Net Present Value (NPV)**: Primary economic driver. Evaluated by applying discounting ($r$) to annual cash flows derived from oil revenue minus CAPEX, fixed OPEX, and variable CO2 processing costs.
- **Recovery Factor (RF)**: Physical efficacy metric bounding the cumulative oil produced against the Original Oil In Place (OOIP).
- **CO2 Storage**: Net CO2 permanently sequestered, accounting for recycled gas and solution gas fraction losses.

### 3.2 Breakthrough Physics & Economic Impact
Early CO2 breakthrough severely damages project economics due to high recycling costs and poor sweep efficiency.
- **Physics Model (`SurrogateBreakthrough`)**: Uses the analytical Koval (1963) formula.
  ```
  K_koval = H_factor × E_eff
  t_d_bt = 1.0 / K_koval
  ```
  Converts dimensionless breakthrough time to physical years based on pore volume and injection rates.
- **Impact Scaling**: Applies a Gaussian-like exponential penalty to the objective function if breakthrough occurs earlier than an ideal baseline (e.g., 1/3 of project life).

### 3.3 Constraint Handling
Physical impossibilities and operational violations are strictly penalized to guide the optimizer away from invalid domains.
- **Penalty Method**: Uses $10^{10}$ or $10^{12}$ negative additive penalties to heavily degrade fitness.
- **Dimensional Consistency**: Rejects profiles where volumetric injections exceed voidage space or storage capacities.

---

## 4. Parallel Processing Architecture

To achieve rapid evaluation of large population arrays (e.g., 50-200 individuals per generation), the engine implements a multiprocessing wrapper.

**Batch Dispatching:**
The `_evaluate_solutions_parallel` method intercepts the `pygad` population array.
- **Serialization Safety**: Instantiates a `PickleSafeOptimiser` to strip unpicklable UI elements (like PyQt signals).
- **ProcessPoolExecutor**: Dispatches individual `params_dict` evaluations across available CPU cores.
- **Fault Tolerance**: If an individual simulation fails or diverges, the worker catches the exception, logs the trace, and forcefully returns a heavy penalty (`-1e12`), preventing the optimizer thread from crashing.