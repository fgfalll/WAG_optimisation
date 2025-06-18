# CO₂-EOR Optimization Framework 🔬

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-XX.XXXXX/XXXX-blue)](https://doi.org)
[![PyPI Version](https://img.shields.io/pypi/v/co2eor-optimizer)](https://pypi.org/project/co2eor-optimizer/)
[![Tests](https://img.shields.io/badge/tests-90%(notalltested)-yellow)](tests/)

### *PhD candidate: Petrenko Taras Sergiyovych*

### *Advisor: Branimir Cvetcovich*

This research develops novel computational methods for CO₂ Enhanced Oil Recovery (EOR) optimization, with key contributions including:
1.  **Advanced Hybrid GA-BO Optimizer:** A primary focus is the development of a configurable hybrid Genetic Algorithm + Bayesian Optimization engine for EOR parameter prediction. This includes mechanisms for transferring multiple elite solutions from GA to BO and fine-grained control over both optimization phases via external configuration.
2.  Hybrid MMP correlation framework with improved accuracy (RMSE < 150 psi) *(Target/Existing)*
3.  Physics-informed genetic algorithm with 20-30% faster convergence *(Target/Related to GA phase)*
4.  GPU-accelerated sweep efficiency modeling *(Conceptual in current framework, future enhancement)*
5.  Field-validated uncertainty quantification *(Future Work)*

*Key Publications:*
* Petrenko, T. (2025). Study of physicochemical and geochemical aspects of enhanced oil recovery and CO₂ storage in oil reservoirs. *Technology Audit and Production Reserves*, *2*(1(82)), 24–29. [https://doi.org/10.15587/2706-5448.2025.325343](https://doi.org/10.15587/2706-5448.2025.325343)
* 

## Research Methodology 📝
1.  **Data Collection & Processing:**
    -   Integration of LAS and ECLIPSE data formats.
    -   Robust data handling and validation using typed dataclasses.
2.  **Model Development (Core of Current Work):**
    -   Development and refinement of the **configurable hybrid GA-BO optimization strategy**.
    -   Implementation of a suite of physics-informed recovery models (Simple, Koval, Miscible, Immiscible, Hybrid with TransitionEngine).
    -   Integration of multiple MMP correlations.
    -   **Centralized Configuration Management** (`config_manager.py` and `config.json`) for all parameters.
3.  **Validation & Analysis:**
    -   Computational experiments to benchmark the hybrid optimizer against standalone GA and BO.
    -   Sensitivity analysis of hybrid strategy parameters (e.g., GA/BO phase lengths, elite transfer).
    -   Validation against numerical simulation (ECLIPSE) and field case studies.

## Key Features ✨

-   **Advanced Physics-Informed Optimization Engine (`OptimizationEngine`):**
    -   **Configurable Hybrid Genetic Algorithm + Bayesian Optimization (`hybrid_optimize`)**:
        -   Tunable GA phase (population, generations, operators) with specific settings for hybrid mode.
        -   Tunable BO phase (iterations, method - `gp_minimize` or `bayes_opt`) for refinement.
        -   Controlled transfer of multiple elite solutions from GA to BO.
    -   Standalone GA (`optimize_genetic_algorithm`) and BO (`optimize_bayesian`) methods.
    -   Gradient-based optimizer (`optimize_recovery`) and WAG-specific optimizer (`optimize_wag`).
    -   **External JSON Configuration (`config.json`)** for all optimizer settings, EOR parameters, and model defaults via `ConfigManager`.
-   **Comprehensive MMP Calculation Module (`evaluation.mmp` & `OptimizationEngine.calculate_mmp`):**
    -   Multiple empirical correlations (Cronquist, Glaso, Yuan) with 'auto' selection.
    -   Temperature and composition dependent, using `PVTProperties` and optionally `WellAnalysis` data.
-   **Sophisticated Recovery Factor Modeling (`core.py`):**
    -   Suite of models: `SimpleRecoveryModel`, `KovalRecoveryModel`, `MiscibleRecoveryModel`, `ImmiscibleRecoveryModel`.
    -   `HybridRecoveryModel` featuring a `TransitionEngine` (Sigmoid, Cubic spline) for smooth miscibility transitions.
    -   Configurable initialization parameters for each model via `config.json`.
-   **Advanced Data Processing (`data_processor.py`, `parsers/`):**
    -   LAS file parsing (`las_parser.py`) with automatic unit conversion.
    -   ECLIPSE simulator data integration (`eclipse_parser.py`).
    -   Robust data validation and structuring via `WellData`, `ReservoirData`, `PVTProperties` dataclasses.
-   **Visualization System (`OptimizationEngine` plotting methods):**
    -   MMP depth profiles (requires `WellAnalysis`).
    -   Conceptual optimization convergence tracking.
    -   Parameter sensitivity analysis plots.
-   **GPU Awareness:** Conceptual support for GPU (e.g., `TransitionEngine`), with `cupy` detection.

## Limitations and Restrictions of Research 🚧

This research, while aiming for robust and applicable results, is subject to the following limitations and restrictions:

1.  **Simplified Recovery Models:** The implemented recovery factor models (Koval, empirical miscible/immiscible, etc.) are simplifications of complex multiphase flow phenomena in porous media. They do not capture all reservoir heterogeneities, detailed fluid phase behavior (e.g., full compositional effects handled by an Equation of State), or intricate wellbore/pattern dynamics that full-physics numerical simulators can. The accuracy of the optimized EOR parameters is therefore dependent on the fidelity of these models to the specific reservoir conditions.
2.  **MMP Correlation Accuracy:** The Minimum Miscibility Pressure (MMP) calculations rely on empirical correlations. While multiple correlations are available, their accuracy can vary depending on the oil composition, gas composition, and reservoir temperature, and they may have inherent error ranges (e.g., +/- 10-20%).
3.  **Optimization Objective:** The current primary optimization objective is the maximization of an estimated recovery factor. Economic considerations (e.g., cost of CO₂, injection/production facilities, NPV) are not yet integrated into the objective function, which might lead to solutions that are technically optimal but not economically viable.
4.  **Computational Demands:** Global optimization techniques like Genetic Algorithms, even when hybridized, can be computationally intensive, especially with a large number of parameters or when coupled with more complex (though still simplified) recovery models. The scale of problems that can be addressed efficiently is a consideration.
5.  **Data Availability and Quality:** The quality of input data (reservoir properties, PVT data, well logs) significantly impacts the reliability of both the MMP calculations and the recovery factor estimations. The research assumes availability of reasonably accurate input data. Validation against diverse, high-quality field datasets is a long-term goal but may be restricted by data accessibility.
6.  **Scope of EOR Processes:** The current framework is primarily focused on CO₂ injection, including continuous and WAG schemes. It does not explicitly model other EOR methods (e.g., chemical EOR, thermal EOR) or complex geochemical interactions between CO₂ and rock/fluids, beyond what's implicitly captured in empirical models.
7.  **Uncertainty Quantification:** While a target, comprehensive uncertainty quantification (UQ) for the optimized parameters (considering uncertainties in input data and model parameters) is a complex extension and not fully implemented in the current core framework.
8.  **Validation:** While computational benchmarking is a core part, extensive validation against multiple, diverse field case studies or detailed full-physics simulations is a significant undertaking that forms part of the longer-term research validation rather than an out-of-the-box feature of the initial framework.

## Installation 🚀

### Prerequisites
- Python 3.9+
- Pip and Venv

### From Source (for development)
```bash
# Clone repository
git clone https://github.com/fgfalll/WAG_optimisation.git
cd WAG_optimisation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate     # Windows

# Install dependencies (ensure requirements.txt or pyproject.toml is present)
pip install -r requirements.txt # Or: pip install .[dev] if using pyproject.toml
```

## Quick Start 🏁

1.  **Prepare Data:** Ensure reservoir data (e.g., ECLIPSE `.DATA`) and PVT information are available.
2.  **Configure:** Create and populate `config.json` with settings for EOR parameters, the hybrid optimizer (GA phase, BO phase, elite transfer), recovery models, etc.
3.  **Initialize & Run:** Use the `OptimizationEngine` to perform hybrid optimization.

## Usage Example 🧑‍💻

```python
import numpy as np
import logging
# Assuming project modules are in PYTHONPATH or structured as a package
from config_manager import config_manager, ConfigNotLoadedError
from core import (
    OptimizationEngine, ReservoirData, PVTProperties,
    EORParameters, GeneticAlgorithmParams # For potential explicit instantiation
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_eor_optimization():
    CONFIG_FILE = "config.json" # Path to your configuration file
    try:
        config_manager.load_config(CONFIG_FILE)
        logging.info(f"Configuration successfully loaded from {CONFIG_FILE}")
    except Exception as e: # Catch FileNotFoundError, JSONDecodeError, etc.
        logging.critical(f"CRITICAL: Failed to load configuration from '{CONFIG_FILE}'. Error: {e}. Exiting.")
        return

    # --- 1. Load/Create Reservoir and PVT Data (Illustrative) ---
    # Replace with actual data loading from your parsers (e.g., using DataProcessor)
    reservoir_data = ReservoirData(
        grid={'PORO': np.array([0.15, 0.18, 0.20])}, # Simplified
        pvt_tables={} # Add PVT table data if used by specific models/MMP
    )
    pvt_properties = PVTProperties(
        oil_fvf=np.array([1.1, 1.2]), oil_viscosity=np.array([1.0, 0.8]),
        gas_fvf=np.array([0.005, 0.004]), gas_viscosity=np.array([0.015, 0.012]),
        rs=np.array(), pvt_type='black_oil', # Added rs example
        gas_specific_gravity=0.7, temperature=190.0
    )
    logging.info("Illustrative Reservoir and PVT data created.")

    # --- 2. Initialize Optimization Engine ---
    # The engine uses the globally loaded config_manager for its settings
    try:
        engine = OptimizationEngine(
            reservoir=reservoir_data,
            pvt=pvt_properties
            # Optionally pass eor_params_instance or ga_params_instance to override config
        )
        logging.info(f"OptimizationEngine initialized. Using recovery model: {engine.recovery_model}")
        logging.info(f"MMP: {engine.mmp:.2f} psi (using method from config or auto)")

        # --- 3. Run Hybrid Optimization ---
        # Behavior is controlled by "OptimizationEngineSettings.hybrid_optimizer" in config.json
        logging.info("Starting Hybrid GA-BO Optimization...")
        hybrid_results = engine.hybrid_optimize()

        # --- 4. Display Results ---
        logging.info("\n--- Hybrid Optimization Results ---")
        logging.info(f"Method: {hybrid_results.get('method')}")
        logging.info(f"Final Estimated Recovery: {hybrid_results.get('final_recovery'):.4f}")
        opt_params = hybrid_results.get('optimized_params', {})
        logging.info(f"  Target Pressure: {opt_params.get('target_pressure_psi'):.2f} psi")
        logging.info(f"  Injection Rate: {opt_params.get('injection_rate'):.2f} bpd (effective if WAG)")
        if engine.eor_params.injection_scheme == 'wag':
            logging.info(f"  Cycle Length: {opt_params.get('cycle_length_days'):.1f} days")
            logging.info(f"  Water Fraction: {opt_params.get('water_fraction'):.2f}")
        logging.info(f"  V_DP Coefficient: {opt_params.get('v_dp_coefficient'):.3f}")
        logging.info(f"  Mobility Ratio: {opt_params.get('mobility_ratio'):.2f}")
        
        ga_info = hybrid_results.get('ga_full_results', {})
        logging.info(f"GA Phase: {ga_info.get('generations')} generations, Pop. {ga_info.get('population_size')}")
        logging.info(f"BO Phase: {hybrid_results.get('iterations_bo_actual')} iterations, "
                     f"{hybrid_results.get('initial_points_from_ga_used')} elites from GA, "
                     f"{hybrid_results.get('initial_points_bo_random')} random BO starts.")

    except ConfigNotLoadedError as e: # Should be caught by the initial config_manager.load_config
        logging.critical(f"Application cannot run: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during the optimization workflow: {e}")

if __name__ == '__main__':
    run_eor_optimization()
```

## Documentation 📖

Comprehensive documentation is available in the `Doc/` directory:
- [Architecture Overview](Doc/architecture.md) - System design and components
- [Development Timeline](Doc/development_timeline.md) - Project milestones
- [Code Audit Review](Doc/audit_review.md) - Quality assurance report
- [Code Audit Review from 16.06](Doc/audit_report_2025-06-16.md) - Quality assurance report

## Contributing 🤝

We welcome contributions! Please see our:
- [Contribution Guidelines](Doc/CONTRIBUTING.md)
- [Code of Conduct](Doc/CODE_OF_CONDUCT.md)

Key areas for contribution:
- Additional MMP correlations and validation against experimental/field data.
- Enhanced visualization features (e.g., detailed GA/BO convergence history).
- Robust unit and integration testing framework.
- Expansion of recovery models (e.g., full compositional effects).

## Roadmap 🗺️

-   [✅] **Configurable Hybrid GA-BO Strategy:** Core framework implemented with elite transfer and phase-specific parameters via `config.json`.
-   [✅] **Suite of Recovery Models:** Simple, Koval, Miscible, Immiscible, and Hybrid models available.
-   [✅] **MMP Calculation Module:** Foundational module with multiple correlations.
-   [✅] **Centralized Configuration Management:** `ConfigManager` for robust parameter handling.
-   [⏳] **Core Research:** Extensive benchmarking of the hybrid GA-BO optimizer against standalone methods and literature.
-   [⏳] **Core Research:** In-depth sensitivity analysis of hybrid optimizer configuration parameters (GA/BO balance, elite transfer impact).
-   [ ] Add support for CMG simulator data (parsing and integration).
-   [ ] Advanced PVT integration (viscosity modeling, EOS support).
-   [ ] Enhanced GPU acceleration (multi-GPU support, memory optimization for core computations).
-   [ ] Implement machine learning-based MMP prediction or surrogate recovery models.
-   [ ] Develop UI (e.g., PyQT6 or web-based).
-   [ ] Field data integration module (e.g., history matching, visualization of ECLIPSE results).

## License 📜

This project is licensed under the MIT License - see the `Doc/LICENSE` file for details.

## Contact 📧

For technical inquiries:
[saynos2011@gmail.com](mailto:saynos2011@gmail.com)

Researcher:
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--1764--5256-a6ce39)](https://orcid.org/0009-0005-1764-5256)
[@fgfalll](https://github.com/fgfalll)