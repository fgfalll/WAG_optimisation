# Project Update - 22.06.2025

## Major Refactoring: Core Module Reorganization

This update introduces a significant refactoring of the `core.py` module to enhance modularity, maintainability, and scalability of the application. The monolithic `core.py` has been broken down into several specialized modules housed within a new `core/` directory.

### Key Changes:

1.  **New `core/` Directory:**
    *   All core application logic previously in `core.py` has been moved into a dedicated `core/` package.
    *   An `__init__.py` file has been added to `core/` to make it a Python package and to provide convenient access to key components.

2.  **New Modules within `core/`:**
    *   **`core/data_models.py`**:
        *   Consolidates all primary data structures (dataclasses) used throughout the application.
        *   Includes: `WellData`, `ReservoirData`, `PVTProperties`, `EORParameters`, `GeneticAlgorithmParams`, and the helper `from_dict_to_dataclass`.
    *   **`core/recovery_models.py`**:
        *   Encapsulates all EOR recovery calculation models and related logic.
        *   Includes:
            *   Abstract Base Class: `RecoveryModel`.
            *   Specific Models: `KovalRecoveryModel`, `SimpleRecoveryModel`, `MiscibleRecoveryModel`, `ImmiscibleRecoveryModel`, `HybridRecoveryModel`.
            *   Transition Logic: `TransitionFunction` (ABC), `SigmoidTransition`, `CubicTransition`, `TransitionEngine`.
            *   Global access function: `recovery_factor`.
            *   Handles `cupy` import for optional GPU acceleration.
    *   **`core/optimization_engine.py`**:
        *   Houses the `OptimizationEngine` class, which is the primary orchestrator for EOR optimization strategies (Gradient Descent, Bayesian, Genetic Algorithm, Hybrid, WAG).
        *   Manages MMP calculation (delegating to `evaluation.mmp` where available).
        *   Includes plotting utilities for MMP profiles, optimization convergence, and parameter sensitivity.

3.  **Removal of `core.py` from Project Root:**
    *   The original `core.py` file in the project root has been removed, as its contents are now distributed within the `core/` package.

4.  **Updated Import Statements:**
    *   All modules that previously imported from the root `core.py` (or intermediate refactored files like a root `data_models.py`) have been updated to import from the new `core` package.
        *   Examples:
            *   `from core.data_models import WellData`
            *   `from core.recovery_models import recovery_factor`
            *   `from core import OptimizationEngine` (utilizing `core/__init__.py`)
    *   This affects files in `analysis/`, `evaluation/`, `parsers/`, `utils/`, and `data_processor.py`.

5.  **PYTHONPATH Considerations:**
    *   The refactoring assumes that the project's root directory is accessible via `PYTHONPATH` or `sys.path` when scripts are executed. This is often handled by:
        *   Running main scripts from the project root.
        *   Setting the `PYTHONPATH` environment variable.
        *   Using `sys.path.insert(0, project_root)` in entry-point scripts (as already practiced in some modules like `analysis/well_analysis.py`).

6.  **`config_manager.py` Location:**
    *   `config_manager.py` remains in the project root. Modules within `core/` now import it using `from config_manager import ...`.

### Benefits of this Refactoring:

*   **Improved Modularity:** Each component has a well-defined responsibility and location.
*   **Enhanced Readability:** Easier to find and understand specific pieces of code.
*   **Better Maintainability:** Changes in one area are less likely to inadvertently affect others.
*   **Increased Scalability:** Simpler to add new features or models without cluttering a single large file.
*   **Reduced Circular Dependencies:** The new structure promotes a more layered and logical flow of dependencies.

### Action Items / Notes:

*   Ensure your development environment and execution scripts correctly handle the `PYTHONPATH` so that the `core` package and `config_manager.py` are discoverable.
*   Thorough testing is recommended to confirm all parts of the application function as expected after these changes.
*   Review `core/__init__.py` for the list of conveniently exposed components.