# Dormant & Legacy Engine Subsystems

## 1. Executive Summary

A critical finding of the repository audit is that **multiple simulation engine packages are completely disconnected from the active runtime**.

Future AI agents must be aware that modifying files within these directories will **not** change simulation results in production, GUI runs, or standard optimization routines.

---

## 2. Dormant & Legacy Packages

| Package Directory | LOC | Original Purpose | Current Status | Active Parts |
| :--- | :---: | :--- | :--- | :--- |
| `core/unified_engine/` | 18,450 | Full 3D finite-difference compositional simulator with IMPES/FIM flow solvers. | **DORMANT** (100% disconnected from optimizer loop) | `core/unified_engine/physics/eos/` is imported for PVT calculations. |
| `core/compositional_engine/` | 12,200 | 1D 200-block finite-difference compositional simulator. | **LEGACY / ISOLATED** | Used only in standalone validation tests (`tests/validation/test_compositional_validation.py`). |
| `core/engine_simple/` | 3,100 | Early material balance tank model. | **DORMANT** | Replaced by `surrogate_engine.py`. |
| `core/simulation/` | 4,200 | Earlier generation of surrogate wrappers (`profile_generator.py`, `injection_schemes.py`, `recovery_models.py`). | **DEPRECATED** | Wraps `FastProfileGenerator`; emits runtime deprecation warnings. |

---

## 3. Why These Engines Are Dormant

1. **Computational Speed**:
   - The metaheuristic optimization algorithms (GA, BO, PSO) evaluate **2,000 to 10,000 candidate development plans** per optimization run.
   - `core/unified_engine` requires 15 to 60 seconds per simulation run (taking 8 to 40 hours per optimization).
   - `core/engine_surrogate` evaluates a full 20-year scenario in **1.5 milliseconds** (completing 5,000 runs in under 10 seconds).
2. **Numerical Instability in 3D Grids**:
   - Arbitrary random candidate chromosomes tested by genetic algorithms often produce extreme pressure shocks or unphysical capillary pressure gradients that cause Newton-Raphson solvers in `unified_engine` to diverge.
   - `core/engine_surrogate` guarantees unconditional numerical stability via coupled deliverability damping and physical limits.

---

## 4. Active EOS Package (`core/unified_engine/physics/eos/`)

While the grid and flow solvers of `unified_engine` are dormant, its Equation of State package is **actively maintained and imported**:
- `PengRobinsonEOS`: Cubic equation of state for phase equilibrium, compressibility ($Z$), and fugacity coefficients.
- `ReservoirFluid`: Multi-component hydrocarbon fluid characterization (C1 through C30+ fractions).
- `flash_calculation()`: Accelerated Rachford-Rice isothermal two-phase flash solver.

Agents needing thermodynamic properties or PVT flash routines should import from `core/unified_engine/physics/eos/`.
