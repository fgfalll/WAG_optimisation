# GEMINI.md

This file provides guidance to Gemini Code (gemini.ai/code) when working with code in this repository.

## Project Overview

CO2 EOR Optimizer is a PyQt6-based scientific desktop application for petroleum engineers to optimize CO2 Enhanced Oil Recovery operations. It combines physics-based reservoir simulation, multiple optimization algorithms (Genetic Algorithm, Bayesian Optimization, Particle Swarm, Differential Evolution), sensitivity analysis, and uncertainty quantification.

**Version**: 0.8.5 | **Python**: 3.10+ | **License**: Proprietary

## 🛑 MANDATORY FIRST ACTION: Consult the Agent Wiki (`agent_wiki/`) Gatekeeper

> [!CAUTION]
> **STRICT EXECUTION GATEWAY**:
> Before invoking search tools (`grep_search`), running terminal commands (`run_command`), or proposing/making code edits, the agent's **VERY FIRST tool call in EVERY turn MUST be**:
> `view_file("agent_wiki/README.md")` (or the specific relevant topic document in `agent_wiki/`).
>
> Bypassing this step to immediately search or edit source code is an **invariant violation**. The codebase contains multiple legacy simulation directories, subtle unit conversion traps, and active numerical approximations that are documented in detail within the wiki.

Key wiki documents:
- **Index & Architectural Invariants**: [`agent_wiki/README.md`](agent_wiki/README.md)
- **Source of Truth Map**: [`agent_wiki/architecture/source_of_truth_map.md`](agent_wiki/architecture/source_of_truth_map.md) — *Must-read: explains that `core/engine_surrogate/` is the ONLY active simulation engine; `Phys_engine_full/`, `compositional_engine/`, and `simulation/` are legacy/dormant.*
- **Pitfalls & Gotchas**: [`agent_wiki/development/common_pitfalls.md`](agent_wiki/development/common_pitfalls.md) — *28 documented traps (unit mismatches, GOR scaling, Vogel IPR, mass conservation, Todd-Longstaff, NumPy 2.0).*
- **Safe Modification Rules**: [`agent_wiki/development/safe_modification_rules.md`](agent_wiki/development/safe_modification_rules.md)
- **Change Safety Matrix**: [`agent_wiki/development/change_safety_matrix.md`](agent_wiki/development/change_safety_matrix.md)
- **Physics Models**: [`agent_wiki/physics/`](agent_wiki/physics/) — *Koval displacement, Todd-Longstaff, CO2 storage, composite IPR, recovery models.*
- **Field Units**: [`agent_wiki/data/units.md`](agent_wiki/data/units.md) — *Field unit conversions (MSCF, STB, psia, ft).*
- **Agent Skills Reference**: [`agent_wiki/development/agent_skills.md`](agent_wiki/development/agent_skills.md) — *Domain standards, multi-simulation sweeps, and physics validation.*

## Development Commands

```bash
# Environment setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run the application
python main.py

# Run tests
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_imports.py -v     # Single file
python -m pytest tests/test_imports.py::test_basic_imports -v  # Specific test
python -m pytest tests/ --cov=.               # With coverage
python -m pytest tests/ -v --parallel         # Parallel execution

# Code quality (configured in pyproject.toml)
black .           # Format (line-length: 100, target: py310-py312)
ruff .            # Lint
mypy .            # Type check
```

## High-Level Architecture

The application follows a layered architecture with clear separation of concerns:

```
UI Layer (PyQt6)
    ├─ main_window.py          - Main application window with tabbed interface
    ├─ widgets/                - Reusable components (parameter trees, editors, etc.)
    ├─ dialogs/                - Modal dialogs (injection schemes, scenarios, etc.)
    └─ workers/                - QThread workers for background operations

Core Engine Layer
    ├─ engine_surrogate/       - [ACTIVE RUNTIME ENGINE] Fast analytical surrogate engine
    │   ├─ surrogate_engine.py - Primary engine wrapper and NPV/CO2 accounting
    │   ├─ profile_generator_fast.py - Single source of truth for production profiles & Composite Vogel IPR
    │   └─ analytical_models.py - Koval displacement & Todd-Longstaff mixing physics
    ├─ optimisation_engine.py  - Optimization algorithms (GA, Bayesian, PSO, DE)
    ├─ Phys_engine_full/       - [LEGACY/DORMANT] Full physics simulation with EOS models
    ├─ simulation/             - [LEGACY/DORMANT] Deprecated profile generators & recovery models
    ├─ compositional_engine/   - [LEGACY/DORMANT] 3D compositional simulator
    └─ engine_simple/          - [BENCHMARK ONLY] Simplified benchmark engine

Data Processing Layer
    ├─ parsers/                - Industry file format parsers
    │   ├─ eclipse_parser.py   - Eclipse simulation data
    │   └─ las_parser.py       - Well log (LAS) format
    ├─ data_processor.py       - Data transformation and validation
    └─ data_models.py          - Core data structures

Analysis Layer
    ├─ sensitivity_analyzer.py - Parameter sensitivity (SALib integration)
    ├─ uq_engine.py           - Uncertainty quantification (UQpy integration)
    └─ decline_curve_analysis.py - DCA for production forecasting

Utilities
    ├─ error_handler.py       - Centralized error reporting (CRITICAL: use this)
    ├─ config_manager.py      - Configuration loading/saving (JSON-based)
    ├─ validation_manager.py  - Data validation framework
    └─ units_manager.py       - Unit conversion (pint-based)
```

## Key Architecture Patterns

**Factory Pattern**: `core/engine_factory.py` explicitly routes 100% of simulation evaluations strictly to the analytical `SurrogateEngineWrapper` (`core/engine_surrogate/`). The detailed compositional and full physics engines are dormant in production runs (see `agent_wiki/architecture/source_of_truth_map.md`).


**Worker Pattern**: All long-running operations use QThread workers in `ui/workers/` to prevent UI freezing. Workers emit progress signals and handle errors through `error_handler.py`.

**Observer Pattern**: UI updates via pyqtSignal/pyqtSlot for loose coupling between workers and widgets.

**Strategy Pattern**: Optimization algorithms are interchangeable via `core/optimisation_engine.py`.

## Important Configuration

- **Main config**: `config/base_config.json` - Application settings, parameter bounds, defaults
- **Economic scenarios**: `config/economic_scenarios.json` - CO2 price, oil price, discount rates
- **Validation rules**: `config/validation_rules.yaml` - Data quality constraints
- **i18n**: `translations/app_en.ts`, `translations/app_ua.ts` - English/Ukrainian support

## Critical Error Handling

**NEVER suppress errors silently.** The project has a centralized error handling system in `error_handler.py`:

```python
from error_handler import report_caught_error, ErrorSeverity, ErrorCategory

try:
    result = risky_operation()
except Exception as e:
    report_caught_error(
        operation="calculate recovery factor",
        exception=e,
        context={"param": value},
        user_action_suggested="Check input ranges",
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.CALCULATION
    )
    raise
```

See `AGENTS.md` for complete error handling guidelines.

## Project File Format

The application uses a custom `.tphd` format for saving project state. Implementation in `utils/project_file_handler.py`.

## Testing Structure

- **`tests/conftest.py`** - Shared pytest fixtures
- **`tests/test_imports.py`** - Validates module imports
- **`tests/test_*_integration.py`** - Integration tests for major components
- **`tests/test_eos_model.py`** - Equation of State model validation
- Test data in `tests/OPM_data/` (public domain reservoir models: Norne, Sleipner, SPE1/3/5/9/10)

## Domain-Specific Knowledge

**MMP (Minimum Miscibility Pressure)**: Critical parameter for CO2 EOR. Calculated in `evaluation/mmp.py` using correlation methods (Cronquist, Yellig, etc.).

**Recovery Factors**: Calculated in `core/simulation/recovery_models.py` using various methods (material balance, decline curve, etc.).

**Injection Schemes**: Patterns defined in `core/simulation/injection_schemes.py` - include 5-spot, 9-spot, line drive, etc.

## Type Hints Required

All function signatures must have type hints. Use from `typing`:

```python
from typing import Callable, Dict, List, Optional, Any, Tuple, Union

def optimize(params: Dict[str, float], callback: Optional[Callable[[float], None]] = None) -> Tuple[bool, float, str]:
    ...
```

## Import Organization

Three sections separated by blank lines:
1. Standard library
2. Third-party
3. Local imports

## Scientific Computing Notes

- Use numpy arrays for numerical data
- Validate input ranges explicitly (petroleum engineering parameters have physical bounds)
- Handle edge cases: zero porosity, negative pressures, NaN from division
- Document formulas with references where applicable
- Unit conversion via `units_manager.py` - never hard-code conversion factors

## AI Integration

The application integrates OpenAI and Gemini APIs for an AI assistant feature. Configuration in `config/base_config.json` under `ai_assistant` section. Worker implementation in `ui/workers/ai_query_worker.py`.

## Active Agent Skills Framework

When performing tasks in this repository, agents should leverage the four configured skills:

1. **`agent-wiki` Gatekeeper** (`.agents/skills/agent-wiki/SKILL.md`):
   - Mandatory first-tool gateway in every turn: consult `agent_wiki/README.md`.
   - Protects simulation invariants and routes changes exclusively to active modules (`core/engine_surrogate/`).

2. **`petroleum-engineer`** (`theneoai/awesome-skills`):
   - Authoritative domain engineering framework.
   - Enforces the 5-tier Decision Hierarchy: Reserves $\to$ Rate $\to$ Cost $\to$ Risk $\to$ Value (NPV).
   - Governs reservoir risk analysis (early breakthrough, viscous fingering, gravity override, asphaltene precipitation).

3. **`simulation-orchestrator`** (`heshamfs/materials-simulation-skills`):
   - Multi-simulation campaign and sensitivity management:
     - `scripts/sweep_generator.py`: Grid, linspace, and Latin Hypercube Sampling (LHS) with deep dot-notation config overrides.
     - `scripts/campaign_manager.py`: Campaign initialization, lifecycle tracking, and safely quoted commands.
     - `scripts/job_tracker.py`: Real-time and file-based execution status detection.
     - `scripts/result_aggregator.py`: Statistical reduction, IQR outlier detection, and `--maximize`/minimize metric extraction.

4. **`physics-simulation`** (`omer-metin/skills-for-antigravity`):
   - Three-part reference structure:
     - **Creation** (`references/patterns.md`): Numerical ODE integrators (RK4, Velocity Verlet, adaptive RK45), rigid body rotational dynamics, FEM, and particle spatial hashing.
     - **Diagnosis** (`references/sharp_edges.md`): Root cause analysis for timestep instabilities (CFL condition, spring stiffness), energy drift, contact jitter, and float precision loss.
     - **Review** (`references/validations.md`): Automated checks enforcing `float64` states, avoiding bare forward Euler for oscillations, and eliminating in-loop matrix inversions.

For full implementation details, see [`agent_wiki/development/agent_skills.md`](agent_wiki/development/agent_skills.md).
