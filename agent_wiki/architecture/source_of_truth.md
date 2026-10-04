# Source of Truth Map: Authoritative vs Dormant Implementations

## 1. Executive Summary

Due to historical refactoring phases across multiple engineering iterations, the `co2eor_optimizer` codebase contains multiple competing implementations of simulation engines, PVT models, rate generators, and recovery models.

This document defines the **single authoritative source of truth** for every scientific calculation, simulation pipeline, and architectural component in the repository.

Future AI agents and developers must strictly consult this matrix before making any code modifications.

---

## 2. Master Source-of-Truth Matrix

| Scientific Subsystem | Active Source of Truth | Dormant / Deprecated Equivalents | Notes & Invariants |
| :--- | :--- | :--- | :--- |
| **Simulation Engine** | `core/engine_surrogate/surrogate_engine.py` (`SurrogateEngineWrapper`, `SurrogateEngine`) | `core/unified_engine/`<br>`core/compositional_engine/`<br>`core/engine_simple/` — **all deleted from the tree** | **100% of scenario evaluations route to SurrogateEngine**. The "dormant" engines no longer exist (corrected 2026-10-04, MED-16), so modifying them is impossible rather than merely ineffective. |
| **Profile Generation** | `core/engine_surrogate/profile_generator_fast.py` (`FastProfileGenerator`) | `core/simulation/profile_generator.py` *(exists — re-export shim)*<br>`core/simulation/injection_schemes.py` *(**does not exist** — grep → 0 hits)* | `core/simulation/profile_generator.py` re-exports `FastProfileGenerator`; there is no `injection_schemes.py` to be legacy. |
| **Recovery Factor & Displacement** | `core/engine_surrogate/analytical_models.py` (`PhDHybridRecoveryModel`, `AnalyticalSurrogate`) | `core/simulation/recovery_models.py` *(exists, dormant behind `RECOVERY_MODELS_AVAILABLE = False` — HIGH-11)*<br>`core/unified_engine/physics/multiphase_flow.py` *(deleted)* | Computes Koval $H_k$, Todd-Longstaff $M_e$, Craig $E_A$, and miscibility transition. CRIT-06/CRIT-07 and HIGH-02 live here. |
| **Deliverability & Inflow Performance (IPR)** | `core/engine_surrogate/profile_generator_fast.py` & `surrogate_engine.py` | Well scaling multipliers in legacy engines | Uses composite Vogel-Darcy IPR clamped to physical reservoir limits. Never scale field recovery by well count. |
| **Tank Pressure & Material Balance** | `core/engine_surrogate/surrogate_engine.py` — inline in `SurrogateEngine.evaluate_scenario()` (damped increment $dP = q\Delta t/(V_p c_t + J_{\text{eff}}\Delta t)$) | `scipy.integrate.solve_ivp` ODE solver (removed) | ⚠️ Corrected 2026-10-04: **there is no `_calculate_pressure_profile` or `_solve_pressure_ode` method** (grep → 0 hits). The pressure update is written inline; cite line numbers from `evaluate_scenario()`. |
| **Thermodynamic Equation of State (EOS)** | `core/engine_surrogate/pvt_state.py` (`SolventExtendedPVTEngine`, Peng–Robinson) | `core/unified_engine/physics/eos/`, `core/compositional_engine/pvt/` — **both removed from the tree** | ⚠️ Corrected 2026-10-04: this row previously named `core/unified_engine/physics/eos/` as *active*; that path does not exist, and the claim contradicted `source_of_truth_map.md` in the same wiki. `pvt_state.py` is the only EOS in the repository. Its Z-factor and $B_g$ are flagged **CRIT-05 / CRIT-04**. |
| **CO₂ Physical Properties** | `core/engine_surrogate/pvt_state.py` (`SolventExtendedPVTEngine` — PR density, $B_{\text{CO2}} = 327.362/\rho$) | `core/unified_engine/physics/co2_properties.py` — **removed from the tree** | ⚠️ Corrected 2026-10-04: the previously cited `co2_properties.py` does not exist anywhere (grep = 0). Cite `pvt_state.py` instead; MED-02 records that the Span–Wagner correlation advertised in its docstring is not implemented. |
| **Minimum Miscibility Pressure (MMP)** | `evaluation/mmp.py` (`calculate_mmp`, `MMPParameters`) | Hardcoded MMP constants in test scripts | Evaluates Cronquist, Lee, Glaso, Alston, and Yuan correlations. |
| **CO₂ Storage & Trapping Accounting** | `core/engine_surrogate/surrogate_engine.py` & `surrogate_models.py` (`CO2StorageSurrogate`) | `core/objectives/storage.py` (legacy wrapper) | Structural, residual, and solubility trapping with closed-loop mass balance. |
| **Economic NPV Calculation** | `core/engine_surrogate/surrogate_models.py` — **inline** in `PhDHybridSurrogate.predict()` (`:507-530`), republished as profile key `npv` by `surrogate_engine.py:172,692` | `core/objectives/economic.py`<br>`evaluation/economic_analyzer.py` | ⚠️ Corrected 2026-10-04 (MED-15): **`_calculate_engine_npv` does not exist** (grep → 0 hits). Discounted cash flow (oil revenue minus CO₂ purchase, recycling, water handling, OPEX, and capital) is written inline in `predict()`; `wrapper.py:50-53` consumes the key. |
| **Multi-Objective Evaluation** | `core/objectives/wrapper.py` (`ObjectiveFunctions`) | Standalone fitness scripts in `tests/` | Evaluates NPV, RF, and Storage objectives and applies geomechanical/environmental penalties. |
| **Optimization Orchestration** | `core/optimisation_engine.py` (`OptimizationEngine`) | `scripts/` standalone optimization runners | PyGAD GA, Bayesian Optimization, PSO, Differential Evolution. |
| **Decline Curve Analysis (DCA)** | `analysis/decline_curve_analysis.py` (`DeclineCurveAnalyzer`) | Legacy Arps formulas in `utils/` | Arps hyperbolic, exponential, harmonic decline curve fitting. |
| **Material Balance Audit** | `analysis/material_balance.py` (`MaterialBalanceAuditor`) | In-memory balances in visualization widgets | Evaluates closed-loop conservation of liquid, gas, and carbon. |
| **Data Models & Constants** | `core/data_models.py` (`ReservoirData`, `EORParameters`, `PhysicalConstants`) | Local dictionary mappings in UI scripts | Authoritative typed dataclasses with validation logic. |

---

## 3. Detailed Subsystem Analysis

### 3.1 Simulation Engines

1. **`core/engine_surrogate/` (ACTIVE)**:
   - Contains `surrogate_engine.py`, `profile_generator_fast.py`, `analytical_models.py`, `surrogate_models.py`.
   - Handles 100% of objective function evaluations during GA, BO, PSO, and single-run simulations.
   - Evaluates a full 20-year monthly profile in ~1.5 milliseconds.

2. **`core/unified_engine/` (DORMANT except `physics/eos/`)**:
   - Contains 3D grid managers, IMPES solvers, and full finite-difference flow equations.
   - High computational cost (10-30 seconds per run).
   - Currently isolated from GUI optimization loops. Only its EOS package is actively imported.

3. **`core/compositional_engine/` (LEGACY / ISOLATED)**:
   - 1D 200-block compositional simulator with flash calculations.
   - Used only in standalone validation scripts (`tests/validation/test_compositional_validation.py`).

4. **`core/simulation/` (DEPRECATED WRAPPERS)**:
   - `core/simulation/profile_generator.py` and `injection_schemes.py` are deprecated facades wrapping `FastProfileGenerator`.

---

## 4. Verification & Testing Guardrails

Before modifying any source-of-truth module:
1. Verify behavior against `tests/core/test_surrogate_engine.py` and `tests/core/test_single_simulation.py`.
2. Ensure closed-loop mass conservation:
   $$\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Leakage} + \text{Produced}$$
3. Confirm that no modifications are made to legacy engines under the mistaken belief that they affect optimization runs.
