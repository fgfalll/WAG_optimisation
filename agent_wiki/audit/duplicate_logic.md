# Duplicate Implementation & Subsystem Redundancy Audit

## 1. Overview

During the development of the CO₂ EOR Optimizer, several iterations of scientific models were added without fully removing or consolidating prior versions.

This document details all **5 major duplicated subsystems**, highlighting code locations, differences, and why each duplicate exists.

---

## 2. Inventory of Duplicate Implementations

### Duplicate 1: Minimum Miscibility Pressure (MMP) Correlations
- **Implementation A**: [evaluation/mmp.py](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py) (`calculate_mmp`, `calculate_mmp_cronquist`, `calculate_mmp_lee`, `calculate_mmp_glaso`)
  - *Details*: Fully unit-checked, supporting 6 literature correlations, input dataclass `MMPParameters`, validated against SPE publications.
- **Implementation B**: [core/engine_surrogate/analytical_models.py:530-580](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L530-L580) (`_calculate_mmp`, `_cronquist_mmp`)
  - *Details*: Duplicates Cronquist and Yellig-Metcalfe in-line. Uses a modified Cronquist formula: $(55 - \gamma_{API})^{0.279}$.
- **Active Runtime Choice**: `evaluation/mmp.py` is called by `SurrogateEngine.__init__`, but `analytical_models.py` uses its own internal `_calculate_mmp` if called directly.
- **Recommendation**: Standardize entirely on `evaluation/mmp.py`.

---

### Duplicate 2: Profile Generation (WAG & Continuous)
- **Implementation A**: [core/simulation/profile_generator.py](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/profile_generator.py) (`ProfileGenerator`)
  - *Details*: Original implementation. Uses explicit Python time loops, day-by-day stepping, and calls `injection_schemes.py`.
- **Implementation B**: [core/engine_surrogate/profile_generator_fast.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py) (`FastProfileGenerator`)
  - *Details*: Vectorized NumPy replacement. Faster by ~100x. Contains empirical WAG rate modulation bonuses (+8% gas, -4% water).
- **Active Runtime Choice**: **`FastProfileGenerator`** is used for 100% of scenario evaluations in `SurrogateEngine`.
- **Recommendation**: Mark `core/simulation/profile_generator.py` and `injection_schemes.py` as fully deprecated; do not use in new code.

---

### Duplicate 3: Analytical Recovery Models
- **Implementation A**: [core/simulation/recovery_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py) (`KovalModel`, `DykstraParsonsModel`, `BuckleyLeverettModel`)
  - *Details*: Employs `scipy.optimize.fsolve` for producing water-oil ratio (WOR) and vertical sweep balances. Falls back to power law on solver failure.
- **Implementation B**: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py) (`KovalRecoveryModel`, `PhDHybridRecoveryModel`)
  - *Details*: Analytical closed-form Koval fractional flow formulations without nonlinear `fsolve` loops.
- **Active Runtime Choice**: **`core/engine_surrogate/analytical_models.py`** is the active model.
- **Recommendation**: Maintain `analytical_models.py` as source of truth; avoid modifying `core/simulation/recovery_models.py`.

---

### Duplicate 4: Equation of State (EOS) & Phase Behavior
- **Implementation A**: [core/unified_engine/physics/eos/](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/) (`PengRobinsonEOS`, `CubicEOS`, `ReservoirFluid`)
  - *Details*: Full Peng-Robinson EOS with Rachford-Rice flash calculation, binary interaction parameters (BIP), and component critical properties.
- **Implementation B**: [core/compositional_engine/pvt/](file:///d:/rep/4.6/co2eor_optimizer/core/compositional_engine/pvt/) (`CompositionalPVT`, `FlashSolver`)
  - *Details*: 1D finite-volume flash solver tightly coupled to compositional grid.
- **Active Runtime Choice**: `core/unified_engine/physics/eos/` is the active library imported by `optimisation_engine.py`.
- **Recommendation**: `core/unified_engine/physics/eos/` is the authoritative thermodynamic package.

---

### Duplicate 5: CO₂ Trapping & Material Balance
- **Implementation A**: [analysis/material_balance.py](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py) (`MaterialBalanceTracker`)
  - *Details*: Post-run material balance calculation for UI charts. Contains the double-subtraction defect for recycled CO₂.
- **Implementation B**: [core/engine_surrogate/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py) (`_calculate_co2_purchased_recycled`)
  - *Details*: Real-time mass calculation used to drive cash flows and storage metrics during optimization.
- **Active Runtime Choice**: `surrogate_engine.py` is authoritative for optimization; `material_balance.py` is for post-processing.
- **Recommendation**: Correct `analysis/material_balance.py` to match the mass accounting of `surrogate_engine.py`.
