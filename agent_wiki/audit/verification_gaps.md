# Scientific Verification Gaps & Dormant Code Architecture

## 1. Executive Summary of Verification Coverage

The active production simulation pipeline (`core/engine_surrogate/` and `core/optimisation_engine.py`) has been audited and verified across 40 dedicated scientific tests in `tests/scientific/`. However, significant architectural gaps exist in dormant and legacy subsystems:

1. **Active vs Legacy Divergence**: The repository contains five separate engine directories (`engine_surrogate`, `engine_simple`, `Phys_engine_full`, `compositional_engine`, and `unified_engine`). Only `engine_surrogate` is active in production optimization. The others are dormant but contain severe physical errors that would fail immediately if activated.
2. **Dormant Code Flaws**:
   - `core/unified_engine/physics/eos/__init__.py`: Inverted phase labels (SCI-FLAW-08) and corrupted Peng-Robinson fugacity (SCI-FLAW-18).
   - `core/unified_engine/physics/co2_properties.py`: Inverted thermal expansion (SCI-FLAW-04).
   - `core/data_integration_engine.py`: Negative oil compressibility (SCI-FLAW-02) and inverted pressure-viscosity (SCI-FLAW-03).
3. **Surrogate-Physics Decoupling**: The optimizer explores a unconstrained proxy parameter space (e.g., setting target pressure to 4,450 psia) while the underlying 0D tank ODE calculates real pressure at 3,100 psia (SCI-FLAW-06).

---

## 2. Granular Gap Inventory

| Subsystem | Active / Dormant Status | Current Verification Status | Identified Gap / Risk | Recommendation |
|:---|:---|:---|:---|:---|
| **`core/engine_surrogate`** | **Active Production Engine** | **VERIFIED** on mass balance and EPA Class VI; **FLAWED** on Koval fractional flow (SCI-FLAW-01), Craig sweep step (SCI-FLAW-16), and gas trapping (SCI-FLAW-17) | Discontinuous sweep and inverted fractional flow distort optimization landscapes | Fix Koval fractional flow and Craig sweep continuity in post-audit phase |
| **`core/optimisation_engine`** | **Active Production Optimizer** | **VERIFIED** convergence across GA/PSO/BO; **FLAWED** on single-well point drainage (SCI-FLAW-05), decoupled pressure (SCI-FLAW-06), and $B_g$ constant (SCI-FLAW-12) | Allows 1 well to drain entire field reserves without pattern scaling | Enforce pattern well count scaling and unify $B_g$ constant |
| **`core/objectives/wrapper`** | **Active Objective Wrapper** | **VERIFIED** on NPV and RF objectives; **FLAWED** on net utilization denominator (SCI-FLAW-07) | Deflates CO₂ utilization by $10\times$ by including primary oil in denominator | Change denominator to incremental EOR oil: $N_{p,\text{total}} - N_{p,\text{primary}}$ |
| **`core/compositional_engine`** | **Dormant Legacy Engine** | **UNVERIFIED** (0% test coverage) | Contains 3D compositional solver and flash calculator that are never called during optimization | Deprecate or isolate in `legacy/` directory |
| **`core/unified_engine`** | **Dormant Prototype Engine** | **CONTRADICTED BY TEST** (Cubic EOS phase inversion, PR fugacity corruption, inverted thermal expansion) | Multiple critical thermodynamic bugs; cannot be used as an authoritative benchmark | Do not route simulations to `unified_engine` until complete thermodynamic overhaul |
| **`core/Phys_engine_full`** | **Dormant Legacy Engine** | **UNVERIFIED** | Deprecated ODE pressure solver with numerical stiffness issues | Superseded by coupled Darcy/Vogel tank ODE in `surrogate_engine.py` |
