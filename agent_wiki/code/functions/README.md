# Function Directory & Scientific API Catalog

## 1. Overview

This catalog documents the authoritative functions implementing key scientific calculations, simulation steps, and thermodynamic correlations in the `co2eor_optimizer` repository.

Future AI agents can use this guide to directly identify function signatures, mathematical formulations, required units, edge cases, and callers without scanning multiple modules.

---

## 2. Authoritative Function Index

| Function Name | Defining Module | Mathematical / Physical Purpose | Safety Tier |
| :--- | :--- | :--- | :---: |
| **`evaluate_scenario()`** | `core/engine_surrogate/surrogate_engine.py` | Primary simulation pipeline evaluator producing 20-year profiles and metrics. | **CRITICAL** |
| **`generate_profiles()`** | `core/engine_surrogate/profile_generator_fast.py` | Vectorized time-series synthesis with mass-conserved WAG phase buffering. | **HIGH** |
| **`_calculate_pressure_profile()`**| `core/engine_surrogate/surrogate_engine.py` | Solves deliverability-coupled tank material balance pressure increments $dP(t)$. | **CRITICAL** |
| **`_calculate_co2_purchased_recycled()`** | `core/engine_surrogate/surrogate_engine.py` | Closed-loop accounting of fresh purchased CO₂ vs recycled produced CO₂. | **HIGH** |
| **`_calculate_engine_npv()`** | `core/engine_surrogate/surrogate_engine.py` | Discounted cash flow NPV including oil revenue, OPEX, CAPEX, and CO₂ costs. | **HIGH** |
| **`calculate_mmp()`** | `evaluation/mmp.py` | Computes Minimum Miscibility Pressure across 5 empirical correlations. | **HIGH** |
| **`predict_recovery()`** | `core/engine_surrogate/analytical_models.py` | Hybrid Koval/Buckley-Leverett miscible/immiscible ultimate recovery factor $RF$. | **CRITICAL** |
| **`_calculate_heterogeneity_factor()`** | `core/engine_surrogate/analytical_models.py` | Koval heterogeneity multiplier $H_k = 1 / (1 - 0.80 V_{DP})^2$. | **CRITICAL** |
| **`_calculate_effective_mobility_ratio()`** | `core/engine_surrogate/analytical_models.py` | Todd-Longstaff effective mobility ratio $M_e = [\omega M^{1/4} + (1-\omega)]^4$. | **HIGH** |
| **`_calculate_craig_areal_sweep()`** | `core/engine_surrogate/analytical_models.py` | Craig correlation for breakthrough areal sweep efficiency $E_A(M_e)$. | **MEDIUM** |
| **`calculate_storage()`** | `core/engine_surrogate/surrogate_models.py` | Analytical partitioning of trapped CO₂ into structural, residual, and dissolved. | **MEDIUM** |
| **`flash_calculation()`** | `core/unified_engine/physics/eos/__init__.py` | Rachford-Rice accelerated flash solver for two-phase vapor-liquid equilibrium. | **HIGH** |
| **`calculate_density()`** | `core/unified_engine/physics/co2_properties.py` | High-precision Span-Wagner / Altunin thermophysical CO₂ density. | **MEDIUM** |
| **`corey_rel_perm()`** | `core/unified_engine/physics/relative_permeability.py`| Modified Corey relative permeability equations for gas, oil, and water. | **HIGH** |

---

## 3. Function Documentation Files

- [simulation_functions.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/functions/simulation_functions.md): Simulation orchestration, pressure, deliverability, and economics.
- [recovery_functions.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/functions/recovery_functions.md): MMP, Koval factor, Todd-Longstaff mobility, and sweep.
- [thermo_pvt_functions.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/code/functions/thermo_pvt_functions.md): EOS flash, CO₂ properties, and relative permeability.
