# AGENTS.md - CO2 EOR Optimizer Development Guide

## Project Overview

CO2 EOR Optimizer is a Python-based scientific application for optimizing CO2 Enhanced Oil Recovery operations. It features a PyQt6 GUI, multiple optimization algorithms (Genetic Algorithm, Bayesian Optimization, Particle Swarm, Differential Evolution), and physics-based reservoir simulation models.

## 🛑 MANDATORY FIRST ACTION: Consult the Agent Wiki (`agent_wiki/`) Gatekeeper

> [!CAUTION]
> **STRICT EXECUTION GATEWAY**:
> Before invoking search tools (`grep_search`), running terminal commands (`run_command`), or proposing/making code edits, the agent's **VERY FIRST tool call in EVERY turn MUST be**:
> `view_file("agent_wiki/README.md")` (or the specific relevant topic document in `agent_wiki/`).
>
> Bypassing this step to immediately search or edit source code is an **invariant violation**. The codebase contains multiple legacy simulation directories, subtle unit conversion traps, and active numerical approximations that are documented in detail within the wiki.

The **Agent Wiki** (`agent_wiki/`) is the definitive, authoritative source of truth for repository architecture, physical models, active vs legacy engines, and safety protocols:

| Topic | Wiki Document | Purpose |
|-------|---------------|---------|
| **Wiki Index & Invariants** | [`agent_wiki/README.md`](agent_wiki/README.md) | Entry point, architectural invariants, reading order |
| **Source of Truth Map** | [`agent_wiki/architecture/source_of_truth_map.md`](agent_wiki/architecture/source_of_truth_map.md) | **Critical**: Identifies which of the 5 engine directories is active vs legacy |
| **Common Pitfalls & Traps** | [`agent_wiki/development/common_pitfalls.md`](agent_wiki/development/common_pitfalls.md) | 28 documented traps (unit conversions, NumPy 2.0, mass balance, Vogel IPR, etc.) |
| **Safe Modification Rules** | [`agent_wiki/development/safe_modification_rules.md`](agent_wiki/development/safe_modification_rules.md) | Invariants that must never be broken during edits |
| **Change Safety Matrix** | [`agent_wiki/development/change_safety_matrix.md`](agent_wiki/development/change_safety_matrix.md) | Risk classification (Critical / High / Medium / Low) |
| **Physics & Equations** | [`agent_wiki/physics/`](agent_wiki/physics/) | Recovery models, CO2 storage, Koval displacement, Composite IPR, PVT |
| **Units & Coordinates** | [`agent_wiki/data/units.md`](agent_wiki/data/units.md) | Field unit definitions (MSCF, STB, psia, ft) and conversion constants |
| **Codebase Audit** | [`agent_wiki/audit/`](agent_wiki/audit/) | Fallbacks, hardcoded values, technical debt, and suspicious logic |
| **Agent Skills Framework** | [`agent_wiki/development/agent_skills.md`](agent_wiki/development/agent_skills.md) | Specialized skills: petroleum-engineer, simulation-orchestrator, physics-simulation |

### Core Architectural Invariants for Agents:
1. **Active Engine**: 100% of simulation evaluations route strictly to `core/engine_surrogate/` (`SurrogateEngineWrapper` + `FastProfileGenerator`). Modifying `core/Phys_engine_full/`, `compositional_engine/`, or `unified_engine/` will **not** affect optimization runs.
2. **Deliverability & Inflow (IPR)**: Well production is governed by Composite Vogel-Darcy IPR clamped to physical reservoir limits. Never scale field recovery by well counts.
3. **Mass Conservation**: Cumulative recycled CO₂ cannot exceed cumulative produced CO₂ ($M_{\text{recycled}} \le M_{\text{produced}} \le M_{\text{injected}}$).
4. **Geomechanical Safety**: Sandface injection pressure is strictly bounded by EPA Class VI UIC standards ($P_{\text{sandface}} \le 0.90 \times P_{\text{frac}}$).
5. **Project Save/Load & State Persistence**: All user inputs, reservoir configurations, PVT models, well coordinates, manual overrides, and optimization results must cleanly serialize and deserialize to/from `.tphd` files via `utils/project_file_handler.py`. Never use recursive `dataclasses.asdict()` (it strips `_dataclass` tags on nested objects). Grid permeability loading must support scalar, 1D flattened, and 3D arrays. Whenever modifying data models or UI widgets, you MUST run `pytest tests/test_project_save_load.py -v`.


## Build, Lint, and Test Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run project save/load verification (MANDATORY after data model or UI changes)
python -m pytest tests/test_project_save_load.py -v

# Run a single test file
python -m pytest tests/test_imports.py -v

# Run a specific test
python -m pytest tests/test_imports.py::test_basic_imports -v

# Run with coverage
python -m pytest tests/ --cov=.

# Run in parallel
python -m pytest tests/ -v --parallel
```

### Application Entry Point

```bash
# Run the application
python main.py
```

## Code Style Guidelines

### Imports

Organize imports in three sections separated by blank lines:
1. Standard library imports
2. Third-party imports
3. Local/relative imports

```python
# Standard library
import logging
from typing import Callable, Dict, List, Optional
from copy import deepcopy

# Third-party
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QWidget

# Local imports
from config_manager import ConfigManager
from utils.preferences_manager import get_preferences_manager
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `OptimizationEngine`, `DataValidator` |
| Functions/variables | snake_case | `calculate_mmp()`, `reservoir_data` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS`, `B_GAS_RB_PER_MSCF` |
| Private methods | _snake_case (leading underscore) | `_validate_inputs()` |
| Type aliases | PascalCase | `ResultList = List[Dict[str, Any]]` |

### Type Hints

Use type hints for all function signatures. Import types from `typing`:

```python
from typing import Callable, Dict, List, Optional, Any, Tuple, Union

def optimize_parameters(
    params: Dict[str, float],
    config: Optional[ConfigManager] = None,
    callback: Optional[Callable[[float], None]] = None
) -> Tuple[bool, float, str]:
    ...
```

### Error Handling

**Critical**: Never suppress errors silently. Follow the guidelines in `ERROR_HANDLING_GUIDELINES.md`.

```python
# GOOD: Proper error handling with context
from error_handler import report_caught_error, ErrorSeverity, ErrorCategory

try:
    result = complex_calculation(data)
except Exception as e:
    report_caught_error(
        operation="calculate recovery factor",
        exception=e,
        context={
            "data_size": len(data),
            "input_params": params_dict
        },
        user_action_suggested="Check input data format and ranges",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.CALCULATION
    )
    raise RuntimeError(f"Calculation failed: {e}") from e

# BAD: Silent error suppression
try:
    result = calculation()
except:
    result = default_value  # Never do this
```

Use the error handler utilities:
- `report_error()` - Report custom errors
- `report_caught_error()` - Handle caught exceptions
- `safe_execute` - Context manager for risky operations
- `safe_function` - Decorator for function-level safety

### Logging

Use the module-level logger pattern:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Starting operation")
    logger.info("Processing data")
    logger.warning("Resource low")
    logger.error("Operation failed", exc_info=True)
```

### Function Design

- Keep functions focused and single-purpose
- Maximum ~50 lines per function when possible
- Use default values for optional parameters
- Document complex parameters

```python
def calculate_recovery_factor(
    porosity: float,
    saturation: float,
    volume_factor: float = 1.0,
    **kwargs
) -> float:
    """
    Calculate the recovery factor for given reservoir parameters.

    Args:
        porosity: Reservoir porosity as decimal (0-1)
        saturation: Oil saturation as decimal (0-1)
        volume_factor: Formation volume factor (default 1.0)
        **kwargs: Additional parameters for extensibility

    Returns:
        Recovery factor as percentage (0-100)

    Raises:
        ValueError: If parameters are out of valid range
    """
    if not 0 <= porosity <= 1:
        raise ValueError(f"Porosity must be 0-1, got {porosity}")
    return porosity * saturation * volume_factor * 100
```

### File Organization

```
project_root/
├── main.py                 # Application entry point
├── config/                 # Configuration files
├── core/                   # Core engine modules
│   ├── optimisation_engine.py
│   ├── simulation/
│   └── Phys_engine_full/
├── ui/                     # PyQt6 GUI components
│   ├── main_window.py
│   ├── widgets/
│   ├── dialogs/
│   └── workers/            # Background workers
├── utils/                  # Utility functions & parsers (las_parser.py)
├── analysis/               # Analysis modules
├── tests/                  # Test suite
├── config_manager.py       # Configuration management
├── error_handler.py        # Centralized error handling
└── requirements.txt        # Dependencies
```

### UI Development (PyQt6)

- Use pyqtSignal for inter-component communication
- Follow the worker pattern for long-running operations
- Handle errors through the central error manager
- Use layout managers (QVBoxLayout, QHBoxLayout) instead of fixed positions

```python
from PyQt6.QtCore import pyqtSignal, QObject

class DataProcessor(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)

    def process(self, data):
        # Long-running operation
        self.finished.emit(result)
```

### Testing Guidelines

- Place tests in `tests/` directory
- Mirror source structure: `tests/core/test_optimisation.py`
- Use pytest fixtures from `conftest.py`
- Test error conditions, not just success paths
- Mock external dependencies

```python
# tests/test_optimization.py
import pytest
from core.optimisation_engine import OptimizationEngine

def test_optimization_runs(optimizer_setup):
    engine = optimizer_setup
    result = engine.run(max_iterations=10)
    assert result.success
    assert result.objective_value > 0

def test_invalid_input_handling():
    engine = OptimizationEngine()
    with pytest.raises(ValueError):
        engine.run(negative_param=-1)
```

### Scientific Computing Conventions

- Use numpy arrays for numerical data
- Validate input ranges explicitly
- Document formulas with references where applicable
- Handle edge cases (zero, NaN, infinity)

```python
import numpy as np

def calculate_pressure(gradient: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if np.any(gradient <= 0):
        raise ValueError("Pressure gradient must be positive")
    if len(gradient) != len(depth):
        raise ValueError("Gradient and depth arrays must have same length")
    return np.cumsum(gradient * np.diff(depth, prepend=0))
```

### Performance Considerations

- Use multiprocessing for CPU-intensive optimization runs
- Cache expensive computations where appropriate
- Use `@functools.lru_cache` for pure functions
- Profile before optimizing (`analysis/profiler_refactored.py`)

### Code Review Checklist

- [ ] Type hints on all function signatures
- [ ] No bare `except:` clauses
- [ ] Error context provided in exception handling
- [ ] Logging at appropriate levels
- [ ] Docstrings on public functions/classes
- [ ] Tests for new functionality
- [ ] No commented-out code in PRs
- [ ] Imports organized correctly

## Key Modules Reference

| Module | Purpose | Status / Notes |
|--------|---------|----------------|
| `agent_wiki/` | Architecture & physics knowledge base | **Authoritative reference** (see `agent_wiki/README.md`) |
| `core/engine_surrogate/` | Active fast surrogate simulation engine | **Active production engine** (`FastProfileGenerator`, `surrogate_engine.py`) |
| `core/optimisation_engine.py` | Optimization algorithms (GA, BO, PSO, DE) | Active optimizer with parameter bounds and discretization |
| `core/objectives/wrapper.py` | Objective function wrapper (NPV, RF, Storage) | Evaluates objectives from surrogate results |
| `evaluation/mmp.py` | Minimum Miscibility Pressure calculations | Active analytical correlations (Cronquist, Yellig, etc.) |
| `error_handler.py` | Centralized error handling interface | Required for exception reporting |
| `ui/central_error_manager.py` | GUI error management | Error display and handling |
| `config_manager.py` | Configuration loading/saving | Application and scenario configs |
| `core/Phys_engine_full/` | Full physics engine with EOS | *Legacy / dormant* (see `agent_wiki/architecture/source_of_truth_map.md`) |
| `compositional_engine/` | 3D compositional simulator | *Legacy / dormant* |

