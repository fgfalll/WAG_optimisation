# Comprehensive Scientific Audit & Traceability Report

**Application**: CO₂ EOR Optimizer (`co2eor_optimizer` v0.8.5)
**Role**: Senior Reservoir Engineer, Applied Mathematician & Scientific Auditor
**Standard**: PhD-Level Research Rigor (Zero Hidden Calibration, Strict Mass & Thermodynamic Conservation)

---

## 1. Executive Scientific Audit Summary

- **Total Scanned Python Files**: 238
- **Total Scanned Code Lines**: 88,073
- **Total Documented Scientific Flaws**: 15 (Critical: 5, High: 7, Medium: 3)
- **Total Documented Fallback Exception Handlers**: 459
- **Total Documented Hardcoded Numerical Literals**: 1165

### Primary Scientific Finding:
> **The codebase contains multiple fundamental violations of physical and thermodynamic laws** (inverted Buckley-Leverett/Koval fractional flow, negative compressibility, inverted viscosity-pressure dependence, inverted CO2 thermal expansion, and inverted cubic EOS phase labeling). These defects are currently masked by post-hoc profile scalers, heuristic damping factors, and artificial recovery factor ceilings.

---

## 2. Master Scientific Flaw Summary Table

| ID | Severity | Physical Phenomenon | Code Location | Observed Defect | Status |
|:---|:---|:---|:---|:---|:---|
| **SCI-FLAW-01** | **CRITICAL** | Buckley-Leverett / Koval Solvent Fractional Flow | `core/engine_surrogate/profile_generator_fast.py:944-950` | Favorable piston displacement (M=1.0) yields 70% gas breakthrough, while severe ... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-02** | **CRITICAL** | Isothermal Fluid Compressibility | `core/data_integration_engine.py:370, 456` | Bo(P) = 1.2 + 0.0001 * (P - 4000). Oil expands as reservoir pressure increases.... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-03** | **HIGH** | Viscosity-Pressure Dependence | `core/data_integration_engine.py:372, 375, 459, 465` | Oil and supercritical CO2 viscosities decrease exponentially with pressure.... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-04** | **HIGH** | Isobaric Thermal Expansion | `core/unified_engine/physics/co2_properties.py:140` | CO2 density increases with temperature (drho/dT > 0), predicting density > 1,230... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-05** | **CRITICAL** | Darcy Inflow & Well Interference | `core/optimisation_engine.py:1410, profile_generator_fast.py:501` | Single producer model drains arbitrary reservoir acreage at flat rates without i... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-06** | **CRITICAL** | Thermodynamic State Decoupling | `core/optimisation_engine.py:3215, surrogate_engine.py:407-420` | Profiles generated before pressure profile calculation; oil profile scaled post-... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-07** | **HIGH** | CO2 Utilization Factor | `core/objectives/wrapper.py:212 vs surrogate_engine.py:454` | wrapper.py reports metric tonnes CO2 / STB; surrogate_engine reports MSCF/STB. B... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-08** | **CRITICAL** | Cubic EOS Phase Identification | `core/unified_engine/physics/eos/__init__.py:195` | 'phase': 'V' if Z < 0.8 else 'L'.... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-09** | **HIGH** | Interfacial Tension at Miscibility | `core/simulation/recovery_models.py:197-202` | At P = MMP, IFT equals 20 mN/m (maximum immiscible value), and decays slowly abo... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-10** | **HIGH** | Immiscible Gas Displacement | `core/simulation/recovery_models.py:498-501` | Immiscible displacement efficiency is 0.01-0.03 (predicting ~1% total recovery).... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-11** | **MEDIUM** | Ultimate Recovery Factor Limit | `core/engine_surrogate/analytical_models.py:881, surrogate_engine.py:425` | RF clipped to 1.0 - Swi - Sor (~0.55) instead of (1 - Swi - Sor)/(1 - Swi) (~0.7... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-12** | **HIGH** | Gas Formation Volume Factor (Bg) | `core/optimisation_engine.py:98 vs surrogate_engine.py:201` | optimisation_engine defines B_GAS_RB_PER_MSCF = 5.0; surrogate_engine uses ~0.5 ... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-13** | **MEDIUM** | Minimum Miscibility Pressure Correlation | `evaluation/mmp.py:111` | At API >= 55.0, (55 - API)^0.279 evaluates to 0.0 or produces imaginary/NaN numb... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-14** | **HIGH** | Vapor-Liquid Equilibrium (VLE) | `analysis/material_balance.py:85-108` | Calculates vapor fraction from compressibility factor Z without flash calculatio... | CONFIRMED_AUDIT_OPEN |
| **SCI-FLAW-15** | **MEDIUM** | Reservoir Gas Inventory Dynamics | `core/engine_surrogate/profile_generator_fast.py:983-986` | co2_rate_available = injection_profile[i] * (1.0 - total_trapping). Shut-in drop... | CONFIRMED_AUDIT_OPEN |

---

## 3. Parameter Provenance & Calibration Analysis

Parameters have been classified into Fundamental, Literature, Empirical, Calibrated, Arbitrary, or Contradicted.

| Parameter | Value | Units | Provenance | Code Location | Classification |
|:---|:---|:---|:---|:---|:---|
| `COREY_N_OIL` | 2.0 | dimensionless | Literature (Corey 1954) | `core/engine_surrogate/surrogate_models.py:29` | **VERIFIED** |
| `COREY_N_GAS` | 2.0 | dimensionless | Literature (Corey 1954) | `core/engine_surrogate/surrogate_models.py:30` | **VERIFIED** |
| `S_GC_CRITICAL` | 0.05 | fraction | Empirical Default | `core/engine_surrogate/surrogate_models.py:31` | **EMPIRICAL** |
| `S_OR_BASE` | 0.25 | fraction | Empirical Default | `core/engine_surrogate/surrogate_models.py:32` | **EMPIRICAL** |
| `CO2_DENSITY_TONNE_PER_MSCF` | 0.053 | tonne/MSCF | Fundamental Constant (MW=44.01) | `core/engine_surrogate/surrogate_models.py:26` | **VERIFIED** |
| `TODD_LONGSTAFF_OMEGA` | 0.6 | fraction | Literature (Todd & Longstaff 1972) | `core/engine_surrogate/analytical_models.py:38` | **VERIFIED** |
| `HETEROGENEITY_CALIBRATION_C_TRANS` | 0.8 | dimensionless | Calibrated / Tuning | `core/engine_surrogate/surrogate_engine.py:195` | **CALIBRATED** |
| `NOMINAL_DRAWDOWN` | 500.0 | psi | Arbitrary Fallback | `core/engine_surrogate/surrogate_engine.py:348` | **ARBITRARY** |
| `PRESSURE_INCREMENT_CLAMP` | 450.0 | psi/step | Numerical Safeguard | `core/engine_surrogate/surrogate_engine.py:379` | **NUMERICAL_STABILIZER** |
| `EPA_CLASS_VI_SAFETY_FACTOR` | 0.9 | fraction | Regulatory Standard (EPA Class VI) | `core/engine_surrogate/surrogate_engine.py:343` | **REGULATORY_VERIFIED** |
| `B_GAS_OPTIMISATION` | 5.0 | RB/MSCF | Unknown / Erroneous | `core/optimisation_engine.py:98` | **CONTRADICTED_BY_TEST** |

