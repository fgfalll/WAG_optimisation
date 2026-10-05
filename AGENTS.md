# AGENTS.md - CO2 EOR Optimizer Development Guide

> [!CAUTION]
> **The flaw register is generated, not hand-written.**
> `audit/scientific_flaws.md` is machine-validated by `python -m audit --register validate`.
> Adding a finding by editing that file by hand will fail the gate. Use
> `python -m audit --register new ...` instead, and generate the GitHub issue from the record
> with `python -m audit --register issue <ID>`. See
> [`agent_wiki/development/finding_registry.md`](agent_wiki/development/finding_registry.md).

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
| **Common Pitfalls & Traps** | [`agent_wiki/development/common_pitfalls.md`](agent_wiki/development/common_pitfalls.md) | 61 documented traps (unit conversions, NumPy 2.0, mass balance, Vogel IPR, misformatted findings) |
| **Safe Modification Rules** | [`agent_wiki/development/safe_modification_rules.md`](agent_wiki/development/safe_modification_rules.md) | Invariants that must never be broken during edits |
| **Change Safety Matrix** | [`agent_wiki/development/change_safety_matrix.md`](agent_wiki/development/change_safety_matrix.md) | Risk classification (Critical / High / Medium / Low) |
| **Continuity Gate** | [`agent_wiki/development/continuity_gate.md`](agent_wiki/development/continuity_gate.md) | How to re-verify a `RESOLVED` claim; 1-commit-1-issue policy |
| **Finding Registry** | [`agent_wiki/development/finding_registry.md`](agent_wiki/development/finding_registry.md) | **Required**: the CLI for adding findings — never hand-write one |
| **Physics & Equations** | [`agent_wiki/physics/`](agent_wiki/physics/) | Recovery models, CO2 storage, Koval displacement, Composite IPR, PVT |
| **Units & Coordinates** | [`agent_wiki/data/units.md`](agent_wiki/data/units.md) | Field unit definitions (MSCF, STB, psia, ft) and conversion constants |
| **Codebase Audit** | [`agent_wiki/audit/`](agent_wiki/audit/) | Fallbacks, hardcoded values, technical debt, and suspicious logic |
| **Agent Skills Framework** | [`agent_wiki/development/agent_skills.md`](agent_wiki/development/agent_skills.md) | Specialized skills: petroleum-engineer, simulation-orchestrator, physics-simulation |

### Core Architectural Invariants for Agents:
1. **Active Engine**: 100% of simulation evaluations route strictly to `core/engine_surrogate/` (`SurrogateEngineWrapper` + `FastProfileGenerator`). Modifying `core/Phys_engine_full/`, `compositional_engine/`, or `unified_engine/` will **not** affect optimization runs. **None of those directories exist** — verify with `Test-Path` before citing them.
2. **Deliverability & Inflow (IPR)**: Well production is governed by Composite Vogel-Darcy IPR clamped to physical reservoir limits. Never scale field recovery by well counts.
3. **Mass Conservation**: Cumulative recycled CO₂ cannot exceed cumulative produced CO₂ ($M_{\text{recycled}} \le M_{\text{produced}} \le M_{\text{injected}}$). **Holds by construction, not by assertion** — no test enforces it.
4. **Geomechanical Safety**: Sandface injection pressure is bounded by EPA Class VI UIC standards ($P_{\text{sandface}} \le 0.90 \times P_{\text{frac}}$). **Not enforced by physics** — the bound is an artefact of `np.clip` at `surrogate_engine.py:468`, and leakage is identically zero (see HIGH-23).
5. **Project Save/Load & State Persistence**: All user inputs must round-trip via `utils/project_file_handler.py`. Never use recursive `dataclasses.asdict()`. Run `pytest tests/test_project_save_load.py -v` after data-model or UI changes.
6. **Simulation Run Audit Logging**: Every simulation run audit, parameter sweep, or benchmark MUST be recorded in `agent_wiki/audit/simulation_run_audits/DD-MM-YYYY_<run_name>/` with an `audit.md` carrying an explicit **Verdict** (`PASSED` / `ACCEPTABLE WITH CONDITIONS` / `FLAGGED` / `FAILED`), an actionable **Proposal**, and **Relevant Files**, and registered in the [Master Index](agent_wiki/audit/simulation_run_audits/index.md).

## Build, Lint, and Test Commands

### Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest tests/ -v                          # all tests
python -m pytest tests/test_project_save_load.py -v # MANDATORY after data/UI changes
python -m pytest tests/ --cov=. --cov-report=xml
```

### Audit & Continuity Gates (run before claiming a fix works)

```bash
# 1. Re-verify every RESOLVED claim by measurement
python -m audit.continuity check
python -m audit.continuity check CRIT-14            # single finding

# 2. Validate the flaw register schema
python -m audit --register validate

# 3. Enforce 1-commit-1-issue discipline
python -m audit.continuity check-commit "<subject>" "<body>"
python -m audit.continuity selftest                 # proves the gate works
python -m audit --issue-gate                        # staged-commit gate

# 4. Static analysis
python -m ruff check --select F821 .                # F821 is a RELEASE GATE
```

> [!WARNING]
> **`pytest` passing is not evidence a physics fix works.** On 05-10-2026 seventeen findings were
> marked `RESOLVED` while the suite reported `335 passed / 0 failed`; adversarial re-measurement
> found 8 requiring reopening, including 2 regressions. Always cite a measurement in a
> `Status:` line.

### Adding or Updating a Finding — use the CLI

```bash
# Never hand-edit audit/scientific_flaws.md; the schema gate rejects it.
python -m audit --register new CRIT-22 \
    --severity CRITICAL --category MATHEMATICAL \
    --location core/engine_surrogate/pvt_state.py:415 \
    --observed "what the code does" \
    --expected "what the physics requires" \
    --impact "consequence for RF, pressure, economics, containment" \
    --evidence "reproduction command + measured value + literature"

# Then generate the GitHub twin from the record (never write the issue by hand)
python -m audit --register issue CRIT-22
```

`Severity` ∈ `CRITICAL|HIGH|MEDIUM|LOW|INFORMATIONAL` · `Category` ∈
`MATHEMATICAL|PHYSICAL|NUMERICAL|SOFTWARE|PROVENANCE` · `Status` ∈ `NEW|OPEN|CONFIRMED|
PARTIALLY_RESOLVED|CONFIRMED_BUT_INERT|REGRESSED|RECURRED|STILL_OPEN|RESOLVED|STALE|
SUPERSEDED|UNKNOWN_EVIDENCE_REQUIRED`. Full schema and rationale:
[`agent_wiki/development/finding_registry.md`](agent_wiki/development/finding_registry.md).

### Application Entry Point

```bash
python main.py
```

## Code Style Guidelines

### Imports

Three sections, blank-line separated: standard library, third-party, local/relative.

```python
# Standard library
import logging
from typing import Callable, Dict, List, Optional

# Third-party
import numpy as np
from PyQt6.QtWidgets import QWidget

# Local imports
from config_manager import ConfigManager
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `OptimizationEngine` |
| Functions/variables | snake_case | `calculate_mmp()` |
| Constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| Private methods | _snake_case | `_validate_inputs()` |

### Type Hints, Error Handling, Logging

Type-hint all signatures. Never suppress errors silently — follow `ERROR_HANDLING_GUIDELINES.md`
and use `error_handler.report_caught_error()`. Module-level `logger = logging.getLogger(__name__)`
with context at the appropriate level.

```python
def calculate_recovery_factor(porosity: float, saturation: float) -> float:
    """Calculate the recovery factor.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    if not 0 <= porosity <= 1:
        raise ValueError(f"Porosity must be 0-1, got {porosity}")
    return porosity * saturation * 100
```

### Scientific Computing Conventions

numpy for numerical data; validate ranges explicitly; document formulas with references; handle
zero/NaN/infinity.

### Performance Considerations

multiprocessing for CPU-intensive runs, `functools.lru_cache` for pure functions, profile before
optimising (`analysis/profiler_refactored.py`).

### Code Review Checklist

- [ ] Type hints on all signatures
- [ ] No bare `except:`; error context provided
- [ ] Docstrings on public functions/classes
- [ ] Tests for new functionality
- [ ] No commented-out code
- [ ] `ruff check --select F821` clean
- [ ] `python -m audit.continuity check` shows no new regressions
- [ ] Findings added via `python -m audit --register new`, not by hand

## Key Modules Reference

| Module | Purpose | Status / Notes |
|--------|---------|----------------|
| `agent_wiki/` | Architecture & physics knowledge base | **Authoritative reference** |
| `audit/continuity.py` | Autouse gate re-verifying `RESOLVED` claims | `python -m audit.continuity check` |
| `audit/registry.py` | Canonical finding writer/validator + GitHub bridge | `python -m audit --register validate` |
| `core/engine_surrogate/` | Active physics-informed surrogate engine | **Active production engine** |
| `core/optimisation_engine.py` | Optimization algorithms (GA, BO, PSO, DE) | Active optimizer |
| `core/objectives/wrapper.py` | Objective evaluation (NPV, RF, storage) | Consumer of engine profiles |
| `evaluation/mmp.py` | MMP correlations (5 validated literature sources) | Single source of truth for MMP |
| `core/engine_surrogate/pvt_state.py` | EOS, PVT, flash, solvent state | **Known defects: CRIT-03, CRIT-21, HIGH-20/21/26** |
| `core/engine_surrogate/analytical_models.py` | Recovery models (miscible, immiscible, Koval, hybrid) | **Known defects: CRIT-06/13/14/19/20** |
| `core/engine_surrogate/surrogate_engine.py` | Scenario evaluation, pressure ODE, mass balance | **Known defects: CRIT-01/12/17, HIGH-23/24** |
| `error_handler.py` | Centralized error handling | Required for exception reporting |
| `ui/central_error_manager.py` | GUI error management | — |