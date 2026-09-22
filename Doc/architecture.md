# CO2 EOR Optimization System Architecture

## Research Overview
This framework supports PhD research on "Advanced Computational Methods for CO₂-EOR Parameter Prediction," with a primary focus on a **novel hybrid Genetic Algorithm - Bayesian Optimization (GA-BO) strategy.**

**Key Research Contributions & Focus Areas:**
1.  **Hybrid GA-BO Optimizer Design:**
    *   Development of a robust hybrid optimizer that leverages GA for global search and BO for efficient local refinement.
    *   Investigation of effective information transfer mechanisms from GA to BO (e.g., seeding BO with multiple GA elite solutions).
2.  **Configuration-Driven Optimization Control:**
    *   A system managed by `ConfigManager` and `config.json` allowing detailed, external configuration of:
        *   GA phase parameters (population, generations, operators) specific to the hybrid context.
        *   BO phase parameters (iterations, initial random points, method like 'gp' or 'bayes').
        *   Control over the number of elite solutions transferred from GA to BO.
3.  **Performance Analysis and Benchmarking:**
    *   Systematic comparison of the hybrid GA-BO against standalone GA and BO across various EOR scenarios.
    *   Analysis of the hybrid strategy's parameter sensitivity.
4.  *(Other contributions like refined MMP correlations or GPU-accelerated modeling if they are still central outputs of the research using this optimizer framework)*

**Validation Methodology:**
*   Performance evaluation on synthetic and/or field-inspired EOR case studies using the developed framework.
*   *(Potential: Numerical simulation (ECLIPSE 100) to generate objective function responses or validate optimized parameters derived from the framework in a full-physics environment).*
*   Sensitivity analysis of the hybrid optimizer's own configuration parameters to understand its behavior and robustness.

## Core Components

### 1. Data Layer
-   **Formats Supported**:
    -   LAS (well logs) - Processed by `las_parser.py`.
    -   ECLIPSE 100 (GRID, PROPS, REGIONS, SOLUTION, SCHEDULE, FAULTS, etc.) - Processed by `eclipse_parser.py`.
-   **Key Features**:
    -   Automatic unit conversion (handled within parsers or `WellAnalysis`).
    -   Data validation (within dataclasses and parsers).
    -   Missing value handling.
    -   Orchestrated by `DataProcessor.py` for file ingestion.
-   **Output**: Populated `WellData` and `ReservoirData` dataclasses.

### 2. PVT Modeling & MMP Calculation
-   **PVTProperties Dataclass (`core.py`):** Stores validated PVT data (FVF, viscosity, Rs, type, specific gravity, temperature).
-   **MMP Estimation (`evaluation.mmp.py`):**
    *   Calculates Minimum Miscibility Pressure (MMP) using various correlations (e.g., Cronquist, Yuan, Hybrid GH - *as implemented in `mmp.py`*).
    *   Takes `MMPParameters` (which can be derived from `PVTProperties` or `WellAnalysis`) as input.
    *   Features automatic method selection based on available data.
    *   Includes API gravity estimation from PVT (with appropriate warnings for accuracy).
    *   MMP value is a critical input/constraint for the optimization process and recovery models.

### 3. Core Dataclasses & Configuration Management
-   **Dataclasses (`core.py`):**
    *   `WellData`: Container for well log data.
    *   `ReservoirData`: Container for reservoir simulation input data.
    *   `PVTProperties`: Validated PVT data.
    *   `EORParameters`: Defines CO₂ injection strategy parameters and their bounds.
    *   `GeneticAlgorithmParams`: Defines parameters for the Genetic Algorithm.
-   **Configuration Management (`config_manager.py`, `config.json`):**
    *   Centralized JSON-based configuration (`config.json`) for all major system settings.
    *   `ConfigManager` class loads and provides access to these settings.
    *   Crucially controls EOR defaults, GA/BO/Hybrid optimizer settings, recovery model initialization parameters, and general fallbacks.
    *   **Ensures application requires a valid configuration file to start.**

### 4. Recovery Modeling (`core.py`)
-   **Role:** Serve as the objective function for the `OptimizationEngine` by evaluating the effectiveness (e.g., recovery factor) of a given set of EOR parameters.
-   **Available Models:**
    *   `SimpleRecoveryModel`: Basic empirical model.
    *   `KovalRecoveryModel`: Physics-informed sweep efficiency model.
    *   `MiscibleRecoveryModel`: Accounts for compositional effects, gravity override, viscous fingering.
    *   `ImmiscibleRecoveryModel`: Considers three-phase relative permeability, capillary pressure, residual oil.
    *   `HybridRecoveryModel`:
        *   Combines miscible and immiscible model responses using a `TransitionEngine`.
        *   `TransitionEngine` provides smooth miscibility transitions (e.g., Sigmoid, Cubic functions) based on pressure/MMP ratio.
        *   GPU awareness for `TransitionEngine` calculations (via `cupy` if available).
-   **Parameterization:**
    *   Model `__init__` parameters (e.g., `kv_factor` for `MiscibleRecoveryModel`, `sor` for `ImmiscibleRecoveryModel`) are primarily sourced from the `RecoveryModelKwargsDefaults` section in `config.json`.
    *   Runtime parameters (e.g., `v_dp_coefficient`, `mobility_ratio`, `pressure`, `rate`) are passed by the optimizer to the model's `calculate` method during each evaluation.

### 5. Optimization System (`OptimizationEngine` in `core.py`)
-   **Core of the Research:** Implements and controls the various optimization strategies, with a primary focus on the hybrid approach.
-   **Hybrid GA-BO Approach (Primary Research Focus):**
    1.  **Genetic Algorithm (GA) Phase:**
        *   Performs global exploration of the EOR parameter space defined by `_get_ga_parameter_bounds()`.
        *   Uses dictionary-based individuals for parameter representation, enhancing clarity and flexibility.
        *   Operators: Tournament selection (with elitism), Blend crossover (arithmetic), Gaussian mutation.
        *   Fitness evaluation is parallelized using `ProcessPoolExecutor`.
        *   **Configuration (`config.json` -> `hybrid_optimizer.ga_params_hybrid`):** Population size, number of generations, operator rates, elite count, etc., can be specifically configured for the GA phase when run as part of the hybrid strategy.
    2.  **Information Transfer Mechanism:**
        *   A configurable number (`num_ga_elites_to_bo` in `config.json`) of the top elite solutions (parameter sets and their fitness) from the GA phase are passed to initialize/seed the BO phase.
    3.  **Bayesian Optimization (BO) Phase:**
        *   Performs efficient local refinement of EOR parameters, starting from points informed by GA (and/or its own random initial points).
        *   Supports 'gp' (using `skopt.gp_minimize`) or 'bayes' (using `bayes_opt.BayesianOptimization`) as backend methods.
        *   **Configuration (`config.json` -> `hybrid_optimizer` section):** Number of BO iterations, number of additional random initial points, and the choice of BO backend method are specifically configurable for the hybrid strategy.
-   **Standalone Optimization Modes (for benchmarking & specific tasks):**
    *   `optimize_genetic_algorithm()`: Executes GA as a standalone optimizer.
    *   `optimize_bayesian()`: Executes BO as a standalone optimizer.
    *   `optimize_wag()`: An iterative grid search method tailored for WAG parameters (cycle length, water fraction), followed by pressure optimization.
    *   `optimize_recovery()`: A simple gradient descent optimizer focused on optimizing injection pressure.
-   **Objective Function:** Primarily uses the selected `recovery_factor` model's output.
-   **GPU Awareness:** Conceptually allows for GPU use if underlying models (like `TransitionEngine`) or evaluation steps are GPU-enabled. Checks for `cupy` availability.

```mermaid
graph TD
    subgraph "Configuration Layer"
        direction LR
        CM[config_manager.py] -- Reads --> CFG[config.json]
    end

    subgraph "Data Layer"
        direction LR
        Parsers[Parsers (LAS, ECLIPSE)] --> DataStructs[Dataclasses e.g., ReservoirData, PVTProperties]
    end
    
    subgraph "Modeling & Evaluation Layer"
        direction TB
        MMPCalc[MMP Calculation (evaluation.mmp.py)]
        WellAn[Well Analysis (well_analysis.py)]
        RecModels[Recovery Models (core.py - Koval, Hybrid, etc.)]
        TransEng[TransitionEngine (core.py)]
    end

    subgraph "Optimization Layer"
        direction TB
        OptEng[OptimizationEngine (core.py)]
        GA[Genetic Algorithm]
        BO[Bayesian Optimization]
        OtherOpts[Other Optimizers (WAG, Gradient)]
    end

    CFG -->|Controls All Settings| OptEng
    CM -->|Provides Settings to| OptEng
    
    DataStructs -->|Input to| OptEng
    DataStructs -->|Input to| MMPCalc
    DataStructs -->|Input to| WellAn
    WellAn -->|Derived Params for| MMPCalc
    
    MMPCalc -->|MMP Value| OptEng
    MMPCalc -->|MMP Value| RecModels
    
    OptEng -- Selects & Uses --> RecModels
    RecModels -- Contains --> TransEng
    
    OptEng -- Manages --> GA
    OptEng -- Manages --> BO
    OptEng -- Manages --> OtherOpts
    
    GA -- Candidate Solutions --> RecModels
    BO -- Candidate Solutions --> RecModels
    OtherOpts -- Candidate Solutions --> RecModels

    RecModels -->|Fitness Value| GA
    RecModels -->|Fitness Value| BO
    RecModels -->|Fitness Value| OtherOpts

    GA -- Elite Solutions to Seed --> BO

    OptEng --> Results[Optimization Results / Visualizations]

    style CM fill:#lightgrey,stroke:#333,stroke-width:2px
    style CFG fill:#lightgrey,stroke:#333,stroke-width:2px
    style OptEng fill:#lightblue,stroke:#333,stroke-width:2px
    style GA fill:#add8e6,stroke:#333,stroke-width:1px
    style BO fill:#add8e6,stroke:#333,stroke-width:1px

6. Visualization System (core.py plotting methods)

Purpose: To aid in the analysis of input data, optimization process, and results.

Implemented Engineering Visualizations:

plot_mmp_profile(): MMP vs. depth, optionally with temperature.

plot_optimization_convergence(): Conceptual plot showing outcome of optimization methods (can be enhanced to show history).

plot_parameter_sensitivity(): Shows how recovery factor changes when a single optimized parameter is varied around its optimal value.

Technology: Uses Plotly for interactive plots.

Potential Expansion for Research:

Detailed generation/iteration-wise convergence plots for GA and BO.

Comparative plots for benchmarking results from different optimizer configurations or methods.

System Integration and Workflow

Configuration: Application starts by loading config.json via ConfigManager. Failure to load a valid config is critical.

Data Input: Reservoir and fluid data (.las, .DATA) are parsed into ReservoirData and PVTProperties objects (potentially via DataProcessor). WellData can be used by WellAnalysis.

Engine Initialization: OptimizationEngine is instantiated with the data objects. It configures itself based on settings from ConfigManager (e.g., default recovery model, EOR parameters, GA/BO base configurations). MMP is calculated.

Optimization Execution: A specific optimization method is called on the engine (e.g., hybrid_optimize()).

The chosen method (e.g., hybrid) reads its specific parameters from the hybrid_optimizer section of the config.

The optimizer iteratively proposes EOR parameter sets.

Each set is evaluated using the selected recovery_factor model (which itself might be configured).

The hybrid method manages the GA exploration phase, transfer of elite solutions, and the BO refinement phase.

Results & Analysis: The engine stores and returns the optimization results (best parameters, final recovery). Plotting methods can be used to visualize aspects of the process or sensitivity.

Key Design Principles

Modularity: Separation of concerns (data parsing, PVT/MMP, recovery modeling, optimization, configuration).

Configurability: Extensive use of config.json and ConfigManager allows for flexible control and experimentation without code changes, crucial for research.

Extensibility: Dataclass-based data structures and abstract base classes (e.g., RecoveryModel) facilitate additions.

Robustness: Input validation in dataclasses, error handling in parsers and ConfigManager, and graceful fallbacks (e.g., GPU availability).

Clarity: Use of type hints, descriptive naming, and dictionary-based GA individuals.

Generated code
---

This updated `architecture.md` should better reflect:
*   The central role and detailed control of the **hybrid GA-BO optimizer**.
*   The critical importance of the **`ConfigManager` and `config.json`** in driving the behavior of the entire framework, especially the optimization strategies.
*   The flow of information and control within the `OptimizationEngine` for the hybrid method.
*   The most recent state of how recovery models are parameterized (init vs. runtime).

Let me know if you'd like any section further refined!
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END