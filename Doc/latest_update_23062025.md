# Project CO₂ EOR Optimizer - Recent Updates, Limitations, and Considerations

## Date: October 26, 2023 (Conceptual Date)

This document outlines significant recent enhancements to the CO₂ EOR Optimizer project, focusing on advanced recovery modeling, sophisticated profile generation, sensitivity analysis, and uncertainty quantification. It also highlights current limitations and areas for future consideration.

---

## I. Core Enhancements

### 1. Advanced Recovery Modeling (`core/recovery_models.py`)

*   **New Model: `LayeredRecoveryModel`**
    *   **Functionality:** Allows simulating recovery from a reservoir conceptualized as multiple non-communicating layers, each with distinct properties (Pore Volume fraction, permeability factor, porosity).
    *   It uses a specified `base_model_type` (e.g., 'koval', 'miscible') to calculate recovery within each layer.
    *   Total injection rate is distributed among layers based on their `perm_factor * pv_fraction`.
    *   Overall recovery is the PV-weighted average of individual layer recoveries.
    *   **Configuration:** Layer definitions and base model type can be configured in `config.json` under `RecoveryModelKwargsDefaults.Layered`.
    *   **Integration:** The `recovery_factor` factory function now supports instantiating and using the `LayeredRecoveryModel`.

### 2. Sophisticated Production Profile Generation (`core/optimisation_engine.py`, `core/data_models.py`)

*   **Extended `ProfileParameters` (`core/data_models.py`):**
    *   Added new `oil_profile_type` options: `"plateau_exponential_decline"` and `"plateau_hyperbolic_decline"`.
    *   New parameters: `initial_decline_rate_annual_fraction`, `hyperbolic_b_factor`, `min_economic_rate_fraction_of_peak`.
*   **Enhanced Profile Calculation in `OptimizationEngine._generate_oil_production_profile`:**
    *   Implements logic for exponential and hyperbolic (Arps' equations) decline phases following an optional plateau.
    *   Uses `scipy.optimize.brentq` to solve for the initial/plateau production rate (`q_peak_optimized`) that matches the total target recoverable oil for the given decline parameters and project life.
    *   Respects `min_economic_rate_fraction_of_peak` to truncate production when the annual rate falls below the economic limit.
    *   Includes robust fallback to linear distribution if the solver fails or parameters are invalid.
    *   Ensures final profile sums precisely to `total_oil_to_produce_stb` through normalization.

---

## II. Analysis Capabilities

### 1. Advanced Sensitivity Analysis (`analysis/sensitivity_analyzer.py`)

*   **New Module: `SensitivityAnalyzer`**
    *   A dedicated class for performing comprehensive sensitivity studies.
    *   Initializes with an `OptimizationEngine` instance or data to create one.
    *   Resolves a base case for EOR and economic parameters, either from previous optimization results or engine defaults.
*   **Broader Parameter Handling (Scenario 1 - KPI Sensitivity with Fixed EOR Ops):**
    *   `run_one_way_sensitivity` can now vary parameters beyond just EOR operational variables. It supports dot-notation paths:
        *   `eor.<param>`: EOR operational parameters.
        *   `econ.<param>`: Economic parameters (e.g., `econ.oil_price_usd_per_bbl`).
        *   `reservoir.<param>`: Conceptual reservoir properties (e.g., `reservoir.avg_porosity`).
        *   `fluid.<param>`: Conceptual fluid properties (e.g., `fluid.mmp_override`).
        *   `model.<param>`: Internal parameters of recovery models (e.g., `model.kv_factor`).
    *   `_evaluate_scenario` method temporarily applies these varied parameters to the `OptimizationEngine` or its components for evaluation. **This requires `OptimizationEngine` to respect these temporary overrides (see "Limitations & Considerations for `OptimizationEngine`").**
*   **Scenario 2 - Re-optimization Sensitivity (`run_reoptimization_sensitivity`):**
    *   **Functionality:** Analyzes how the *optimal EOR strategy itself* changes when a primary input parameter (e.g., oil price, average reservoir porosity) is varied.
    *   For each value of the primary uncertain parameter, it re-runs a full EOR optimization (e.g., `hybrid_optimize`).
    *   Collects the resulting new optimal EOR parameters and objective values.
    *   **State Management:** To ensure isolation, this method typically creates new, temporary `OptimizationEngine` instances for each re-optimization, configured with the varied primary parameter.
*   **Two-Way Sensitivity (`run_two_way_sensitivity` & `plot_contour_chart`):**
    *   **Functionality:** Allows varying two input parameters simultaneously and evaluating a target objective (Scenario 1 type - fixed EOR ops).
    *   Results can be visualized as contour plots.
*   **Visualization:**
    *   Retains and enhances `plot_tornado_chart` and `plot_spider_chart`.
    *   Adds `plot_contour_chart` for two-way sensitivity results.

### 2. Uncertainty Quantification Engine (`analysis/uq_engine.py`)

*   **New Module: `UncertaintyQuantificationEngine`**
    *   Dedicated class for UQ studies.
    *   Initializes with a `base_engine_for_config` (an `OptimizationEngine` instance used for its configuration and base, non-varying data) and UQ settings from `config.json`.
*   **Parameter Definition and Sampling (with `chaospy`):**
    *   Parses uncertain parameter definitions from `config.json` (`UncertaintyQuantificationSettings`).
    *   Supports various distributions (Normal, Uniform, Lognormal, Triangular, Beta) via `chaospy`.
    *   Generates Monte Carlo samples using specified rules (e.g., Latin Hypercube).
    *   Includes structure for handling correlated inputs using `chaospy.Nataf` transform (requires correlation matrix in config).
*   **Monte Carlo Simulation - Scenario A (`run_mc_for_fixed_strategy`):**
    *   Propagates uncertainty from input parameters to output KPIs (e.g., NPV, Recovery Factor) for a *fixed* EOR operational strategy.
    *   Applies sampled values to a temporary context for evaluation, leveraging `OptimizationEngine`'s calculation capabilities but with overridden uncertain inputs.
*   **Monte Carlo Simulation - Scenario B (`run_mc_for_optimization_under_uncertainty`):**
    *   Propagates uncertainty to the *optimal EOR strategy itself* and the resulting optimal objective values.
    *   For each Monte Carlo sample of uncertain inputs, it creates a new `OptimizationEngine` instance (configured with those sampled inputs) and runs a full EOR optimization.
    *   Outputs distributions of optimal EOR parameters and objective values.
*   **Polynomial Chaos Expansion (PCE - Foundational) (`run_polynomial_chaos_analysis`):**
    *   Implements the basic workflow for PCE using `chaospy`:
        *   Generation of orthogonal polynomial basis.
        *   Generation of quadrature nodes and weights.
        *   Evaluation of the model (currently assuming fixed EOR strategy for node evaluations) at these nodes.
        *   Fitting the PCE surrogate model.
        *   Estimation of statistics (mean, variance) and Sobol sensitivity indices from the PCE.
    *   This provides a more computationally efficient alternative to MCS for smooth models.
*   **Results Analysis and Plotting:**
    *   `analyze_uq_results`: Calculates mean, std. dev., P10, P50, P90.
    *   `plot_uq_distribution`: Generates histograms (PDFs) and CDFs.

---

## III. Modifications to `OptimizationEngine` (to support advanced analysis)

*   **Initialization Overrides:**
    *   `__init__` now accepts optional `avg_porosity_init_override`, `mmp_init_override`, and `recovery_model_init_kwargs_override`. These are used by `UQEngine` when creating temporary engine instances for Scenario B UQ runs.
*   **Dynamic Properties for Overridable Values:**
    *   `@property def avg_porosity(self)`: Returns average porosity. It first checks for a runtime override (`self._avg_porosity_override` set by analyzers), then an init-time override, then calculates from `self.reservoir.grid`.
    *   `@property def mmp(self)`: Similar logic for MMP, checking `self._mmp_value_override`, then init-time override, then calculates.
*   **Core Evaluation Logic (`_objective_function_wrapper`):**
    *   Now uses `self.avg_porosity` and `self.mmp` properties, ensuring any active overrides are respected.
    *   Ensures `self._recovery_model_init_kwargs` (which can be temporarily modified by analyzers for `model.<param>` sensitivity) are passed to `recovery_factor`.
*   **State Preservation for UQ:**
    *   `__init__` now stores deepcopies of the initial `reservoir`, `pvt`, and parameter dataclass instances (e.g., `self._base_reservoir_data_for_reopt`). This allows `UQEngine` to reliably create fresh, modified engine instances for re-optimization runs without corrupting a shared base state.
*   **`set_recovery_model`:** Consistently updates `self._recovery_model_init_kwargs`.

---

## IV. Limitations and Considerations

### 1. `OptimizationEngine` Override Mechanism:

*   **Current Approach:** The `SensitivityAnalyzer` and `UQEngine` (for Scenario A/PCE) primarily rely on setting "private-convention" attributes (e.g., `_avg_porosity_override`, `_mmp_value_override`) on the `OptimizationEngine` instance or modifying its `_recovery_model_init_kwargs` dictionary temporarily.
*   **Limitation:** This is less explicit than a formal API. While functional, it requires careful coordination between the analysis engines and `OptimizationEngine`.
*   **Consideration:** Refactor `OptimizationEngine` methods (like `_objective_function_wrapper` or helper calculation functions) to accept these overrides as direct keyword arguments for a cleaner interface. This would make the interaction less reliant on modifying engine state directly from outside.

### 2. Handling of Complex Uncertain Parameters:

*   **Reservoir Properties (Grid-based):**
    *   Varying "average porosity" or "average permeability" is a simplification. If uncertainty in spatially distributed properties (e.g., full PORO/PERM fields from `GRDECL`) needs to be propagated, the current `OptimizationEngine` and recovery models (which mostly use bulk/average values) would require significant extension to consume and utilize such detailed uncertain inputs.
    *   The Dykstra-Parsons coefficient (`v_dp_coefficient`) can be treated as an uncertain scalar input to represent heterogeneity.
*   **Fluid Properties (PVT Arrays):**
    *   If PVT properties (Bo, Rs, viscosities as arrays vs. pressure) are uncertain, defining distributions for entire arrays and sampling them is complex. Current UQ typically assumes uncertainty in average values or key scalar PVT parameters.
*   **Relative Permeability:** Similar to PVT arrays, if full rel-perm curves are uncertain, propagating this requires models that use these curves directly and a method to parameterize/sample curve uncertainty.

### 3. State Management in `OptimizationEngine` for Re-optimization (UQ Scenario B):

*   **Current UQ Approach:** The `UQEngine` creates *new* `OptimizationEngine` instances for each Monte Carlo sample in Scenario B. This is robust against state leakage but computationally more expensive due to repeated initializations (including MMP calcs if not overridden).
*   **Alternative (More Complex):** A single `OptimizationEngine` instance could potentially be reused if it had a very reliable `reset_to_base_with_overrides(overrides_dict)` method. This is harder to implement correctly.

### 4. Computational Cost:

*   **Scenario B UQ (Optimization Under Uncertainty):** Re-running full optimization (e.g., `hybrid_optimize`) for hundreds or thousands of Monte Carlo samples is *extremely* computationally intensive. This may only be feasible for simpler optimization methods within the loop or a small number of UQ samples.
*   **PCE for Re-optimization:** Using PCE to create a surrogate for the *optimal EOR parameters themselves* (as a function of uncertain inputs) is also very advanced. The current PCE implementation in `UQEngine` assumes it evaluates a fixed EOR strategy at quadrature nodes.
*   **Two-Way Sensitivity:** While less expensive than UQ Scenario B, running many points for two parameters can still be time-consuming. Re-optimizing for two-way sensitivity would be even more so.

### 5. PCE Implementation Details:

*   **NaN Handling:** The current PCE fitting in `UQEngine` has basic handling for NaNs in model evaluations. Robust PCE fitting with missing data (if evaluations fail at certain nodes) can be challenging and might require specialized `chaospy` techniques or imputation.
*   **Quadrature Order:** Selecting an appropriate polynomial and quadrature order for PCE requires some expertise and experimentation. Too low an order leads to inaccuracy; too high leads to excessive model evaluations.
*   **Model Smoothness:** PCE performs best for models whose outputs are relatively smooth functions of the uncertain inputs. Highly discontinuous or chaotic models can be difficult for PCE to approximate accurately with low polynomial orders.

### 6. Correlation Handling:

*   The `UQEngine` now has structure for `chaospy.Nataf` to handle correlations. This requires users to provide a valid rank correlation matrix and ensure the parameter order aligns. Incorrect correlation setup can lead to invalid sampling.

### 7. User Configuration Complexity:

*   Defining uncertain parameters, their distributions, parameters for those distributions, scopes, internal names, and correlations in `config.json` requires careful attention to detail from the user. Validation of these settings within `UQEngine` is important.

### 8. Scope of Current "Override" Application:

*   The `_apply_sampled_value_to_target` in `UQEngine` and the corresponding logic in `_prepare_evaluation_args` handle a specific set of common overrides (`avg_porosity`, `mmp_value`, `economic` params, `model` init kwargs). If more diverse uncertain parameters need to be introduced (e.g., specific array values in `ReservoirData.grid`), this application logic will need to be extended carefully.