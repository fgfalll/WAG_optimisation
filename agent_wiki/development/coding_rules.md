# Coding Rules & Error Handling Protocols

## 1. Core Coding Standards

This project adheres to the software development guidelines specified in [AGENTS.md](file:///d:/rep/4.6/co2eor_optimizer/AGENTS.md).

### Summary of Rules:
1. **Type Annotations**: All public functions, methods, and classes must include explicit type annotations from `typing`.
2. **Standard Imports Organization**:
   - Section 1: Standard library (`logging`, `typing`, `copy`, `dataclasses`, `pathlib`).
   - Section 2: Third-party dependencies (`numpy`, `scipy`, `pandas`, `PyQt6`).
   - Section 3: Local/relative modules (`from core.data_models import ...`).
3. **Module Logger Pattern**: Every module must declare `logger = logging.getLogger(__name__)`.

---

## 2. Error Handling Protocol (STRICT)

**Critical Invariant**: Never suppress errors silently. Blind `except:` clauses and silent `pass` blocks in physical calculations are strictly prohibited across the entire repository.

### Approved Pattern: Catching Specific Exceptions with State Logging
```python
import logging
logger = logging.getLogger(__name__)

try:
    phase_split = cubic_eos.solve_flash(pressure, temperature, composition)
except (FloatingPointError, ZeroDivisionError, ValueError, RuntimeError) as e:
    logger.warning(
        f"Flash convergence failed: {e}. "
        f"State variables: P={pressure:.1f} psia, T={temperature:.1f} F, z={composition}"
    )
    # Re-raise or return float('nan') - NEVER pass silently
    raise
```

### Prohibited Anti-Patterns:
```python
# PROHIBITED: Bare except with silent default
try:
    result = calculation()
except:
    result = 0.5  # NEVER DO THIS

# PROHIBITED: Passing silently on error
try:
    parse_data()
except Exception:
    pass  # ABSOLUTELY FORBIDDEN
```

---

## 3. Scientific Invariants & Assertions

When writing or modifying scientific simulation code, developers must validate physical boundaries explicitly:

1. **Saturation Bounds**:
   ```python
   assert 0.0 <= S_w <= 1.0, f"Water saturation out of bounds: {S_w}"
   assert 0.0 <= S_o <= 1.0, f"Oil saturation out of bounds: {S_o}"
   assert 0.0 <= S_g <= 1.0, f"Gas saturation out of bounds: {S_g}"
   assert np.isclose(S_w + S_o + S_g, 1.0, atol=1e-4), "Phase saturations do not sum to 1.0"
   ```
2. **Recovery Factor Bounds**:
   ```python
   assert 0.0 <= recovery_factor <= 1.0, f"Recovery factor unphysical: {recovery_factor}"
   ```
3. **Mass Conservation**:
   Before returning simulation results, check that injected mass equals produced mass plus net in-place storage:
   $$\left| M_{\text{inj}} - (M_{\text{prod}} + \Delta M_{\text{res}} + M_{\text{leak}}) \right| / M_{\text{inj}} < 10^{-3}$$

---

## 4. Optimization Failure & Penalty Protocol

1. **Strict Binary Pruning**:
   - Chromosomes yielding unphysical states, numerical divergence, or missing profile data must evaluate to `float("nan")` or receive the full mathematical penalty:
     $$\text{FAILURE\_PENALTY} = -10^{12}$$
2. **Prohibition of Softened Multipliers**:
   - Never apply penalty dilution multipliers (e.g. `* 0.1` or `* 0.8`). Softening penalties distorts the objective search space and allows unphysical genetic lines to survive selection.
3. **No Class E Metric Synthesis**:
   - Never manufacture synthetic values (e.g. estimating storage efficiency from oil recovery). Infeasible runs must fail cleanly.

---

## 5. Mandatory Visualization Dependencies

1. **Direct Dependency Imports**:
   - `plotly` (with `plotly.graph_objects` and `plotly.subplots`) and `matplotlib` are mandatory hard dependencies.
2. **Prohibition of Mock Swallowing Classes**:
   - Defining dummy mock classes (e.g. `class go: class Figure: pass`) that catch import errors and output console text is strictly forbidden.
3. **Fail-Fast Startup Validation**:
   - Visualization packages are validated during application initialization in `main.py`. Missing dependencies must raise explicit `RuntimeError` instructions rather than silently disabling charts.
