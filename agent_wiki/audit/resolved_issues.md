# Resolved Issues & Defect Resolution Archive

This document serves as the authoritative archive of all **resolved scientific flaws, numerical discrepancies, eliminated hidden calibrations, consolidated duplicates, and eradicated dead code** in the CO₂ EOR Optimizer repository.

Main audit documents maintain only active and open items to prevent clutter. When an issue is resolved, its full post-mortem analysis, mathematical resolution, and verification record are migrated to this document using the standardized section format described below.

---

## 🛠️ Automated Archiving Protocol & Section Schema

All issue entries across the wiki follow this machine-readable section schema to allow automated scripts (`scripts/archive_resolved_wiki_issues.py`) to parse, migrate, and index issues:

```markdown
### [TAG-ID] Descriptive Title
- **ID**: `TAG-ID`
- **Category**: `Scientific Flaw` | `Suspicious Logic` | `Hidden Calibration` | `Duplicate Subsystem` | `Dead Code` | `Fallback`
- **Original Document**: [`agent_wiki/audit/<source_doc>.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/<source_doc>.md)
- **Location**: [`path/to/file.py:line_range`](file:///d:/rep/4.6/co2eor_optimizer/path/to/file.py#Lline)
- **Severity**: `CRITICAL` | `HIGH` | `MEDIUM` | `LOW`
- **Status**: `RESOLVED`
- **Date Resolved**: `YYYY-MM-DD`
- **Previous Defect**: Description of mathematical error, unphysical assumption, or software bug.
- **Resolution Details**: Description of the implemented fix, published literature equations, and architectural changes.
- **Verification**: Name of test functions and invariants asserting that the defect is permanently resolved.
```

---

## 📑 Master Resolved Issues Index

| ID / Tag | Issue Title | Category | Original Document | Date Resolved | Verification Status |
|:---|:---|:---|:---|:---|:---|
| **SCI-FLAW-02** | Negative Oil Compressibility in Synthetic PVT | Scientific Flaw | `scientific_flaws.md` | 2026-09-23 | **VERIFIED** |
| **SCI-FLAW-03** | Inverted Pressure-Viscosity Dependence | Scientific Flaw | `scientific_flaws.md` | 2026-09-23 | **VERIFIED** |
| **SCI-FLAW-13** | Non-Standard Cronquist $(55 - API)$ Correlation | Scientific Flaw | `scientific_flaws.md` | 2026-09-23 | **VERIFIED** |
| **SUSP-B** | Unit Inconsistency in Reservoir Tank Pressure ODE | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-C** | Recycled Gas Double-Subtraction in Material Balance | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-C2** | 35.3× OOIP Dimensional Unit Inconsistency | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-F** | 12× Recovery Factor Discrepancy in Data Validation | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-G** | Recycled CO₂ Unit Explosion (21.2M Tonnes) | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-H** | 180-Year Material Balance Time Vector Scaling | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-I** | 15-Bar Production Profile Truncation | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-J** | CO₂ Utilization Penalty 1,000,000.00 Key Mismatch | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-K** | Class E Artificial Storage Modifier Synthesis | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-L** | Penalty Dilution Multipliers (*0.1, *0.8) and Silent Bare Exceptions | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-M** | Plotly Dummy Mock Classes Swallowing Visualizations | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-N** | 1000× Volumetric Downhole Velocity & Dimensionless Number Error ($N_c$, $N_g$) | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-Y** | Cronquist MMP Formula Uses Ad-Hoc `(55 - API)` Term | Suspicious Logic | `suspicious_logic.md` | 2026-09-23 | **VERIFIED** |
| **SUSP-AD** | 20× Produced CO₂ Shrinkage Bug | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AE** | Hallucinated Caprock Leakage from Normal Wellbore Production | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AF** | Pressure Search Space Squeeze from Dimensional Mismatch on $\Delta P_{\text{inj}}$ | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AG** | Plateau Rate Decoupling from Collapsing Drawdown | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AH** | Apparent Mass Balance Discrepancy from Recycled Stream Double-Counting | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AI** | Post Shut-In Unattenuated CO₂ Production & False Ecology Penalties | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AJ** | Decline Curve Analysis Plateau Regression Breakdown ($R^2 = -3.14$) | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AK** | Well-Injector-1 Role Inversion via UI Substring Default Matching | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **SUSP-AL** | Plotly Vertical Bar Schedule Collapse & Duplicate Legend Pollution | Suspicious Logic | `suspicious_logic.md` | 2026-09-17 | **VERIFIED** |
| **CALIB-WAG** | WAG Phase Mobility Buffering & Mass Preservation | Hidden Calibration | `hidden_calibration.md` | 2026-09-17 | **VERIFIED** |
| **CALIB-MMP** | Modified Cronquist MMP Formula (`55 - API`) | Hidden Calibration | `hidden_calibration.md` | 2026-09-23 | **VERIFIED** |
| **CALIB-STORAGE** | Zero-Injection Storage Efficiency Override (Optimizer Cheat) | Hidden Calibration | `hidden_calibration.md` | 2026-09-17 | **VERIFIED** |
| **DUP-01** | Minimum Miscibility Pressure (MMP) Correlations Duplication | Duplicate Subsystems | `duplicate_logic.md` | 2026-09-23 | **VERIFIED** |
| **DEAD-05** | Removed Root Artifacts & Superseded Scripts | Dead Code | `dead_code.md` | 2026-09-17 | **VERIFIED** |
| **DEAD-06** | Removed Scientific Justification and Help Subsystems | Dead Code | `dead_code.md` | 2026-09-17 | **VERIFIED** |
| **SFT-01** | PVT FVF Keyword Mismatch & $B_g$ Fallback Eradication | Software Defect | `fallbacks.md` | 2026-09-23 | **VERIFIED** |
| **SFT-02** | Economic Objective MagicMock Array Broadcasting ValueError | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-03** | Missing `set_engine` Method on OptimizationWidget | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-F821** | Eradication of All 36 Ruff F821 Undefined Names | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-04** | AttributeError on ConfigWidget.engine_selection_changed | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-05** | Root Import Mismatch on ConfigManager in DataManagementWidget | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-06** | Missing Qt Translation Files Suppressed Gracefully | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-07** | Parsers Directory Elimination & LAS Parser Consolidation | Architectural Refactoring | `module_map.md` | 2026-09-23 | **VERIFIED** |
| **SFT-08** | Core Directory Architecture Audit & Prototyping Artifacts Elimination | Architectural Refactoring | `source_of_truth_map.md` | 2026-09-23 | **VERIFIED** |
| **SFT-09** | Application Entry Point (`main.py`) Modernization & Multiprocess Logging Decoupling | Architectural Refactoring | `technical_debt.md` | 2026-09-23 | **VERIFIED** |
| **SFT-10** | Data Models (`core/data_models.py`) Type Safety Hardening & Fault/Fluid Separation | Software Defect | `technical_debt.md` | 2026-09-23 | **VERIFIED** |

---

## 1. Scientific Flaws

### [SCI-FLAW-02] Negative Oil Compressibility in Synthetic PVT
- **ID**: `SCI-FLAW-02`
- **Category**: Scientific Flaw
- **Original Document**: [`agent_wiki/audit/scientific_flaws.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/scientific_flaws.md)
- **Location**: [`core/data_integration_engine.py:370, 456`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L370)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Formation volume factor $B_o$ was modeled with an unphysical positive pressure coefficient above 4,000 psia:
    $$B_o(P) = 1.2 + 0.0001 \cdot (P - 4000)$$
  - Because $\frac{\partial B_o}{\partial P} = +0.0001 > 0$, the isothermal compressibility $c_o = -\frac{1}{B_o}\frac{\partial B_o}{\partial P} < 0$, meaning the oil physically expanded under pressure, violating the second law of thermodynamics and causing artificial voidage expansion in material balance calculations.
- **Resolution Details**:
  - Implemented the thermodynamically valid exponential compressibility relation:
    $$B_o(P) = 1.2 \cdot \exp(-c_o \cdot \max(0, P - 4000))$$
    with $c_o = 1.5 \times 10^{-5}\text{ psi}^{-1}$.
  - Guarantees $c_o > 0$ and $\frac{\partial B_o}{\partial P} < 0$ strictly in undersaturated conditions.
- **Verification**: `tests/test_physics_validation.py`.

---

### [SCI-FLAW-03] Inverted Pressure-Viscosity Dependence
- **ID**: `SCI-FLAW-03`
- **Category**: Scientific Flaw
- **Original Document**: [`agent_wiki/audit/scientific_flaws.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/scientific_flaws.md)
- **Location**: [`core/data_integration_engine.py:372, 375, 459, 465`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L372)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Black-oil synthetic PVT tables modeled undersaturated oil and supercritical CO₂ viscosities with negative pressure exponents:
    $$\mu_o(P) = \mu_{o,ref} \cdot \exp(-0.0003 \cdot (P - 4000))$$
    $$\mu_{\text{CO2}}(P) = \mu_{g,ref} \cdot \exp(-0.0002 \cdot (P - 4000))$$
  - This caused fluid viscosity to decrease exponentially with pressure, creating unphysical mobility improvements and rewarding artificial over-pressurization.
- **Resolution Details**:
  - Inverted the sign of the pressure exponents to positive:
    $$\mu_o(P) = \max(0.1, 1.5 \cdot \exp(0.0003 \cdot (P - 4000)))$$
    $$\mu_{\text{CO2}}(P) = \max(0.01, 0.05 \cdot \exp(0.0002 \cdot (P - 4000)))$$
  - Ensures liquid and dense-phase gas viscosity increase monotonically with pressure ($\frac{\partial\mu}{\partial P} > 0$) as required by fluid mechanics.
- **Verification**: `tests/test_physics_validation.py`.

---

<a id="sci-flaw-13-non-standard-cronquist-55---api-correlation"></a>
### [SCI-FLAW-13] Non-Standard Cronquist $(55 - API)$ Correlation
- **ID**: `SCI-FLAW-13`
- **Category**: Scientific Flaw
- **Original Document**: [`agent_wiki/audit/scientific_flaws.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/scientific_flaws.md)
- **Location**: [`evaluation/mmp.py:111-165`](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py#L111-L165), [`core/engine_surrogate/analytical_models.py:530-580`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L530-L580)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - The MMP correlation in `evaluation/mmp.py` used an unvalidated power-law expression:
    $$MMP = 15.988 \cdot T^{0.744206} \cdot (55.0 - \text{API})^{0.279033}$$
  - For light oils and condensates ($\text{API} \ge 55^\circ$), $(55 - \text{API}) \le 0$, producing fractional powers of negative numbers that raised `ValueError` or produced complex numbers and `NaN`, crashing optimization runs.
- **Resolution Details**:
  - Replaced with the authentic published Cronquist (1978) formulation:
    $$P_{MMP} = 15.988 \cdot T_F^Y \quad [\text{psia}]$$
    where:
    $$Y = 0.744206 + 0.0011038 \cdot MW_{C5+} + 0.0015279 \cdot Vol$$
  - $MW_{C5+}$ is evaluated via the standard DOE / CO₂ Prophet correlation:
    $$MW_{C5+} = 4247.98641 \cdot \text{API}^{-0.87022}$$
    or $\max(72.0, M_{C7+} - 20.0)$ if $C_{7+}$ fraction properties are provided.
  - Aligned both `evaluation/mmp.py` and `core/engine_surrogate/analytical_models.py` (which now delegates directly to `evaluation.mmp`).
  - Completely verified to be finite, strictly positive, real, and monotonic across $\text{API} \in [10, 70]^\circ\text{API}$.
- **Verification**: `tests/scientific/mathematical/test_singularity_and_overflow.py::test_cronquist_mmp_singularity_at_55_api`.

---

## 2. Suspicious Logic & Numerical Discrepancies

### [SUSP-B] Unit Inconsistency in Reservoir Tank Pressure ODE
- **ID**: `SUSP-B`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/engine_surrogate/surrogate_engine.py:904-912`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L904-L912)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `profile_result["injection_profile"]` in MSCFD was subtracted directly from liquid production rates in RB/day without multiplying by gas formation volume factor $B_g$.
  - This distorted downhole voidage and tank pressure derivative $dP/dt$ by a factor of 3 to 5.
- **Resolution Details**:
  - Converted injection rate via dynamic/static $B_g$ ($q_{\text{inj\_rb}} = q_{\text{inj\_mscfd}} \times B_g$).
  - Liquid production and gas injection are now strictly in reservoir barrels per day.
- **Verification**: `tests/scientific/dimensional/test_darcy_inflow_dimensions.py`.

---

### [SUSP-C] Recycled Gas Double-Subtraction in Material Balance
- **ID**: `SUSP-C`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/material_balance.py:216-249`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L216-L249)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `net_injection_tonne = purchased_tonne - recycled_tonne_raw` and then subtracted `produced_co2_tonne`, effectively subtracting recycled gas twice.
- **Resolution Details**:
  - Formulated mass conservation strictly as $M_{\text{stored}} = M_{\text{purchased}} - M_{\text{uncaptured}} - M_{\text{leakage}}$. Mass conservation closes exactly with 0% error.
- **Verification**: `tests/scientific/conservation/test_material_balance_analyzer_closed_loop.py`.

---

### [SUSP-C2] 35.3× OOIP Dimensional Unit Inconsistency
- **ID**: `SUSP-C2`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/data_integration_engine.py:408-425`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L408-L425)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Grid block sizes `dx, dy, dz` generated in feet by `DataManagementWidget` were divided by 0.3048 ($3.28\times$ inflation) and divided by 4046.86 instead of 43560.0 ($10.76\times$ inflation).
  - Combined $35.314\times$ inflation resulted in calculated OOIP of $171,231,839\text{ STB}$ against provided $4,848,750\text{ STB}$, tripping validation.
- **Resolution Details**:
  - Implemented exact field unit conversion ($43560\text{ ft}^2/\text{acre}$) and preserved vertical thickness directly in feet. Exact match achieved ($0.0\%$ discrepancy).
- **Verification**: `tests/core/test_physical_invariants.py`.

---

### [SUSP-F] 12× Recovery Factor Discrepancy in Data Validation
- **ID**: `SUSP-F`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/data_validation.py:288-348`](file:///d:/rep/4.6/co2eor_optimizer/analysis/data_validation.py#L288-L348)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `DataValidator._get_oil_production()` selected `oil_production_rate` (181 points).
  - Integration assumed an annual step size of $dt = 365.25\text{ days}$ instead of monthly $dt \approx 30.4375\text{ days}$, scaling calculated oil by $12\times$ (calculated RF 4.680 vs reported 0.390).
- **Resolution Details**:
  - The validator now detects monthly sampling from `time_vector` or array length (> 100 points for a 15-year run), adjusts $dt$ accordingly, and prioritizes cumulative oil volumes (`cumulative_oil`) when present.
- **Verification**: `tests/scientific/conservation/test_cumulative_oil_recovery_mass_bound.py`.

---

### [SUSP-G] Recycled CO₂ Unit Explosion (21.2M Tonnes)
- **ID**: `SUSP-G`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/material_balance.py:108-142`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L108-L142)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - In `calculate_breakthrough_aware_recycling()`, cumulative oil in tonnes was multiplied directly by GOR in SCF/STB (`cum_oil_tonne * current_gor * 0.001`), mixing mass in tonnes with volume in SCF, inflating calculated recycled gas to $21,200,000\text{ tonnes}$ ($10\times$ total injected CO₂).
- **Resolution Details**:
  - GOR in SCF/STB is converted to MSCF, multiplied by cumulative oil in STB and CO₂ density, and physically bounded such that recycled CO₂ cannot exceed produced CO₂: `recycled_tonne = np.minimum(recycled_tonne_raw, produced_co2_tonne)`.
- **Verification**: `tests/scientific/conservation/test_closed_loop_carbon_balance_invariant.py`.

---

### [SUSP-H] 180-Year Material Balance Time Vector Scaling
- **ID**: `SUSP-H`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/material_balance.py:632-636`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L632-L636)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `create_material_balance_from_optimization()` passed monthly arrays of length 181 to `calculate_material_balance()`, which treated array indices as annual steps (`np.arange(1, 182)`), stretching the x-axis to 180 project years.
- **Resolution Details**:
  - Explicitly scaled monthly balance time axes to project years: `balance_data["years"] = np.arange(1, len(...) + 1) / 12.0`.
- **Verification**: UI plot regression test suites.

---

### [SUSP-I] 15-Bar Production Profile Truncation
- **ID**: `SUSP-I`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/optimisation_engine.py:856-930`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L856-L930)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Assigning 181-element monthly arrays directly to `yearly_oil_stb` while setting `yearly_time_years = np.arange(1, 16)` caused plotting routines to truncate the profile to the first 15 months, displaying 15 identical flat bars.
- **Resolution Details**:
  - `OptimizationEngine.evaluate_for_analysis()` aggregates the 181 monthly steps into true 15-element annual totals for `annual_oil_stb` and `yearly_oil_stb`, matching the 15-year project lifetime.
- **Verification**: Profile generation unit tests.

---

### [SUSP-J] CO₂ Utilization Penalty 1,000,000.00 Key Mismatch
- **ID**: `SUSP-J`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/objectives/wrapper.py:105-148`](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L105-L148)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `_calculate_objective_functions()` looked only for `"annual_co2_purchased_mscf"`. Because the engine generated `"yearly_co2_purchased_mscf"`, the objective failed and assigned the fallback penalty $1,000,000.00$.
- **Resolution Details**:
  - Provided complete aliases (`annual_` and `yearly_`) in engine profiles and updated `wrapper.py` to inspect both keys.
- **Verification**: `tests/core/test_physical_invariants.py`.

---

### [SUSP-K] Class E Artificial Storage Modifier Synthesis
- **ID**: `SUSP-K`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/objectives/wrapper.py:130-137`](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L130-L137)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - If simulation profiles or storage parameters were missing, the wrapper synthesized an artificial storage efficiency via `default_efficiency = max(0.3, 0.5 * (recovery_factor / 0.35))`, awarding ~50% storage credit based purely on oil recovery without verifying physical CO₂ retention.
- **Resolution Details**:
  - Completely eradicated Class E synthesis. When profiles or storage parameters are missing, `storage_efficiency` evaluates strictly to `float("nan")`, and the chromosome is pruned with `FAILURE_PENALTY` ($-10^{12}$).
- **Verification**: `agent_wiki/audit/fallbacks.md`.

---

### [SUSP-L] Penalty Dilution Multipliers (*0.1, *0.8) and Silent Bare Exceptions
- **ID**: `SUSP-L`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/optimisation_engine.py:1705-1805`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1705-L1805), [`flow/compositional_solver.py:714`](file:///d:/rep/4.6/co2eor_optimizer/core/compositional_engine/flow/compositional_solver.py#L714)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Penalties were arbitrarily softened via `FAILURE_PENALTY * 0.1` or `FAILURE_PENALTY * 0.8 + constraint_penalty`, allowing unphysical chromosomes to survive in the GA population. Bare `except Exception: pass` swallowed solver breakdowns.
- **Resolution Details**:
  - Eradicated all penalty dilution multipliers; unphysical chromosomes are pruned with full `FAILURE_PENALTY`. Replaced bare exceptions across solver core with specific exception types and detailed state logging.
- **Verification**: Code review audit & solver unit tests.

---

### [SUSP-M] Plotly Dummy Mock Classes Swallowing Visualizations
- **ID**: `SUSP-M`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/material_balance.py:9-50`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L9-L50)
- **Severity**: LOW
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `material_balance.py` defined dummy mock classes (`class go: class Figure: pass`) that swallowed plot generation silently if Plotly failed to import.
- **Resolution Details**:
  - Deleted all dummy mock classes. Plotly is now imported directly as a hard, mandatory dependency, validated during startup in `main.py`.
- **Verification**: `main.py` startup validation check.

<a id="susp-n-1000-volumetric-downhole-velocity--dimensionless-number-error-n_c-n_g"></a>
### [SUSP-N] 1000× Volumetric Downhole Velocity & Dimensionless Number Error ($N_c$, $N_g$)
- **ID**: `SUSP-N`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/engine_surrogate/analytical_models.py:813-820`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L813-L820), [`core/engine_surrogate/surrogate_engine.py:1137-1138`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1137-L1138)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `surrogate_engine.py` stored $1 / 0.00207 = 483.09\text{ SCF/RB}$ under the key `"mscf_per_res_bbl"`. `analytical_models.py` treated this as MSCF/RB and divided `inj_mscfd` by 483.09, deflating downhole rate to 10.35 res-bbl/day instead of ~10,350 res-bbl/day ($1000\times$ deflation).
  - Capillary Number $N_c$ was deflated by $1000\times$, preventing capillary desaturation ($S_{or}$ locked at 0.30), while Gravity Number $N_g$ was inflated by $1000\times$, collapsing vertical sweep efficiency $e_v$ to zero on dipped fields.
- **Resolution Details**:
  - Fixed unit convention across all modules: `mscf_per_res_bbl` stores MSCF/RB ($\approx 0.4 - 0.6$), $B_g = 1 / \text{mscf\_per\_res\_bbl}$ in RB/MSCF ($\approx 2.0 - 2.5$), and injection rate converted as $q_{\text{inj\_rb}} = q_{\text{inj\_mscfd}} \times B_{g,\text{RB/MSCF}}$.
  - Corrected HCPVI and breakthrough time formulation ($cum\_inj\_rb = cum\_inj\_mscf / mscf\_per\_rb$).
- **Verification**: `tests/scientific/dimensional/test_darcy_inflow_dimensions.py`.

---

<a id="susp-y-cronquist-mmp-formula-uses-ad-hoc-55---api-term"></a>
### [SUSP-Y] Cronquist MMP Formula Uses Ad-Hoc `(55 - API)` Term
- **ID**: `SUSP-Y`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`evaluation/mmp.py:111-165`](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py#L111-L165)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Ad-hoc term $(55 - API)$ produced complex numbers or NaNs when $\text{API} \ge 55^\circ$.
- **Resolution Details**:
  - Replaced with authentic published Cronquist (1978) formulation using $MW_{C5+}$ from the DOE / CO₂ Prophet standard formulation. Single source of truth established.
- **Verification**: `tests/scientific/mathematical/test_singularity_and_overflow.py::test_cronquist_mmp_singularity_at_55_api`.

---

### [SUSP-AD] 20× Produced CO₂ Shrinkage Bug
- **ID**: `SUSP-AD`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/material_balance.py:172-178`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L172-L178)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Pure produced CO₂ stream `annual_co2_produced_mscf` was multiplied by `co2_fraction_of_produced` (default $0.05$), vanishing 95% of produced CO₂ (e.g. 749,645 tonnes disappeared).
- **Resolution Details**:
  - Eliminated the redundant fractional multiplier on the pure CO₂ stream. Mass balance now closes with exact $0.0\%$ error.
- **Verification**: `tests/scientific/conservation/test_material_balance_analyzer_closed_loop.py`.

---

### [SUSP-AE] Hallucinated Caprock Leakage from Normal Wellbore Production
- **ID**: `SUSP-AE`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/optimisation_engine.py:1458-1485`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1458-L1485)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Caprock leakage was algebraically defined as `total_purchased - total_recycled - net_stored = total_produced`, penalizing ordinary wellbore production with carbon taxes as if it were subsurface containment failure.
- **Resolution Details**:
  - Decoupled production from containment. Caprock leakage is now strictly governed by geomechanical containment limits ($P_{\text{sandface}} > 0.90 \times P_{\text{frac}}$) or explicit seal fracture models.
- **Verification**: `tests/scientific/boundary_conditions/test_epa_class_vi_pressure_ceiling_enforcement.py`.

---

### [SUSP-AF] Pressure Search Space Squeeze from Dimensional Mismatch on $\Delta P_{\text{inj}}$
- **ID**: `SUSP-AF`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/optimisation_engine.py:1906-1920`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1906-L1920)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `inj_rate_max / ii` divided field-wide MSCFD by single-well II without unit conversion or well count splitting, producing an artificial $4,000\text{ psi}$ overpressure penalty that squeezed upper pressure down to $2322\text{ psia}$.
- **Resolution Details**:
  - Fixed well splitting ($q_{\text{well}} = q_{\text{inj}} / n_{\text{inj}}$) and bounded near-wellbore transient overpressure to realistic field limits ($\le 500\text{ psi}$), opening search space up to $4,500+\text{ psia}$.
- **Verification**: `core/optimisation_engine.py` boundary tests.

---

### [SUSP-AG] Plateau Rate Decoupling from Collapsing Drawdown
- **ID**: `SUSP-AG`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/engine_surrogate/profile_generator_fast.py:518-596`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L518-L596), [`core/engine_surrogate/surrogate_engine.py:950-990`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L950-L990)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `_plateau_decline_profile` held an arbitrary constant rate bar for up to 11 years, completely decoupled from a 65.7% collapse in reservoir driving drawdown.
- **Resolution Details**:
  - Implemented two-way staggered deliverability coupling via Composite Vogel-Darcy IPR clamped to dynamic pressure $P_{\text{res}}(t)$. Decline curves emerge naturally from first principles.
- **Verification**: `tests/scientific/boundary_conditions/test_producer_rate_drawdown_limit.py`.

---

### [SUSP-AH] Apparent Mass Balance Discrepancy from Recycled Stream Double-Counting
- **ID**: `SUSP-AH`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`utils/run_exporter.py:310-335`](file:///d:/rep/4.6/co2eor_optimizer/utils/run_exporter.py#L310-L335)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `_build_run_manifest` compared `total_injected_tonne` (fresh purchased CO₂) against `total_stored_tonne + total_produced_tonne + total_leakage_tonne`, reporting an apparent 50.99% closure ($711,705.5\text{ t}$ error, equal to cumulative recycled gas).
- **Resolution Details**:
  - Formulated closed-loop gross mass balance: $\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Total Leakage} + \text{Gross Produced}$. Closure $> 99.9\%$ verified.
- **Verification**: `tests/core/test_physical_invariants.py`.

---

<a id="susp-ai-post-shut-in-unattenuated-co2-production--false-ecology-penalties"></a>
### [SUSP-AI] Post Shut-In Unattenuated CO₂ Production & False Ecology Penalties
- **ID**: `SUSP-AI`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/engine_surrogate/surrogate_engine.py:1260-1285`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1260-L1285)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - `_apply_shut_in_guard` only checked `"water_production_rate"` and `"hydrocarbon_gas_production_rate"`, ignoring `FastProfileGenerator` output keys `water_profile`, `co2_gas_profile`, and `solution_gas_profile`. CO₂ production continued flowing post-shut-in ($444,672\text{ t}$ unattenuated), incurring false ecology penalties.
- **Resolution Details**:
  - Extended shut-in guards to taper and zero all phase streams: `water_profile`, `co2_gas_profile`, `solution_gas_profile`, `gas_profile`, and `injection_profile`.
- **Verification**: `tests/scientific/limiting_cases/test_zero_injection_limits.py`.

---

<a id="susp-aj-decline-curve-analysis-plateau-regression-breakdown-r2---314"></a>
### [SUSP-AJ] Decline Curve Analysis Plateau Regression Breakdown ($R^2 = -3.14$)
- **ID**: `SUSP-AJ`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`analysis/decline_curve_analysis.py:90-130`](file:///d:/rep/4.6/co2eor_optimizer/analysis/decline_curve_analysis.py#L90-L130)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Arps equations were fitted directly across a 3-year flat plateau through the steep decline, causing mathematical regression breakdown ($R^2 = -3.14$) and unphysical EUR overestimation (72% OOIP).
- **Resolution Details**:
  - Added decline onset detection ($q(t) < 0.95 \times q_{\text{peak}}$). Preserves historical plateau rates and fits Arps models strictly to the declining segment $(t - t_{\text{onset}})$, yielding $R^2 > 0.95$ and physically bounded EUR forecasts.
- **Verification**: `tests/scientific/reference_solutions/test_analytical_vs_trapezoidal_arps_eur.py`.

---

### [SUSP-AK] Well-Injector-1 Role Inversion via UI Substring Default Matching
- **ID**: `SUSP-AK`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`ui/widgets/manual_well_dialog.py:110-125`](file:///d:/rep/4.6/co2eor_optimizer/ui/widgets/manual_well_dialog.py#L110-L125), [`core/optimisation_engine.py:1400-1420`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1400-L1420)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Status check `if status in ["active", "inactive"]` failed on `"Producer (Active)"`, assigning default name `Well-Injector-1`. In `core/optimisation_engine.py`, injector detection partitioned wells improperly, resulting in 0 injectors in simulation runs.
- **Resolution Details**:
  - Fixed combo text parsing, implemented bidirectional live typing synchronization between well name and role dropdown, and added explicit role partitioning (`injector_wells`, `producer_wells`).
- **Verification**: GUI well dialog unit tests.

---

<a id="susp-al-plotly-vertical-bar-schedule-collapse--duplicate-legend-pollution"></a>
### [SUSP-AL] Plotly Vertical Bar Schedule Collapse & Duplicate Legend Pollution
- **ID**: `SUSP-AL`
- **Category**: Suspicious Logic
- **Original Document**: [`agent_wiki/audit/suspicious_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md)
- **Location**: [`core/plotting_manager.py:plot_well_schedule`](file:///d:/rep/4.6/co2eor_optimizer/core/plotting_manager.py#L780-L860), [`core/optimisation_engine.py:4400-4420`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L4400-L4420)
- **Severity**: LOW
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Vertical bar geometry used `x=[duration_days]` and `base=[start_day]` with `barmode="stack"`, collapsing all operations to a 0.8-day sliver at Day 45, and appended duplicate legend traces for every operational cycle. Additionally, `_ops_standard_production` had an artificial 10-cycle cap (`max_cycles = 10`).
- **Resolution Details**:
  - Implemented true interval geometry (`x = start_day + duration/2`, `width = duration`, `base = 0`, `barmode = "overlay"`), legend deduplication, zero-rate hatched pattern bars with diamond markers, and removed the 10-cycle cap.
- **Verification**: Plotting manager visual tests.

---

## 3. Hidden Calibrations & Empirical Fittings Eradicated

<a id="calib-wag-wag-phase-mobility-buffering--mass-preservation"></a>
### [CALIB-WAG] WAG Phase Mobility Buffering & Mass Preservation
- **ID**: `CALIB-WAG`
- **Category**: Hidden Calibration
- **Original Document**: [`agent_wiki/audit/hidden_calibration.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/hidden_calibration.md)
- **Location**: [`core/engine_surrogate/profile_generator_fast.py:309-408`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L309-L408)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Crude fixed discrete multipliers (`wag_oil_gas_bonus = 1.08`, `wag_oil_water_penalty = 0.96`, `wag_water_gas_reduction = 0.92`) produced artificial sawtooth jumps and destroyed mass balance.
- **Resolution Details**:
  - Replaced with relative phase mobility contrast:
    $$\Delta \lambda / \Sigma \lambda = \frac{\lambda_g - \lambda_w}{\lambda_o + \lambda_g + \lambda_w}, \quad \text{amp} = \text{clip}(0.1 \times \text{contrast}, -0.15, 0.15)$$
  - Mass conservation is strictly enforced: modulated profiles are re-normalized so cumulative production matches $N_p = OOIP \times RF$.
- **Verification**: `tests/scientific/conservation/test_cumulative_oil_recovery_mass_bound.py`.

---

<a id="calib-mmp-modified-cronquist-mmp-formula-55---api"></a>
### [CALIB-MMP] Modified Cronquist MMP Formula (`55 - API`)
- **ID**: `CALIB-MMP`
- **Category**: Hidden Calibration
- **Original Document**: [`agent_wiki/audit/hidden_calibration.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/hidden_calibration.md)
- **Location**: [`evaluation/mmp.py:111-165`](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py#L111-L165), [`core/engine_surrogate/analytical_models.py:530-580`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L530-L580)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Developer altered the published equation to $(55 - \gamma_{API})^{0.279}$ to force an inverse power relationship, which broke down if $\gamma_{API} \ge 55^\circ\text{API}$.
- **Resolution Details**:
  - Replaced with authentic published Cronquist (1978) formulation using $MW_{C5+} = 4247.98641 \cdot \text{API}^{-0.87022}$. Single source of truth established.
- **Verification**: `tests/scientific/mathematical/test_singularity_and_overflow.py::test_cronquist_mmp_singularity_at_55_api`.

---

### [CALIB-STORAGE] Zero-Injection Storage Efficiency Override (Optimizer Cheat)
- **ID**: `CALIB-STORAGE`
- **Category**: Hidden Calibration
- **Original Document**: [`agent_wiki/audit/hidden_calibration.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/hidden_calibration.md)
- **Location**: [`core/optimisation_engine.py:1708-1729`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1708-L1729)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Injected artificial `storage_efficiency = 0.3` or diluted penalties (`FAILURE_PENALTY * 0.1`) when `total_injected_mscf <= 0` and `recovery_factor > 0.05`.
- **Resolution Details**:
  - Eradicated override and dilution. Zero-injection candidates receive strictly `0.0` storage efficiency; failing candidates receive full `FAILURE_PENALTY` ($-10^{12}$).
- **Verification**: `tests/scientific/limiting_cases/test_zero_injection_limits.py`.

---

## 4. Duplicate Subsystems Consolidated

### [DUP-01] Minimum Miscibility Pressure (MMP) Correlations Duplication
- **ID**: `DUP-01`
- **Category**: Duplicate Subsystem
- **Original Document**: [`agent_wiki/audit/duplicate_logic.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/duplicate_logic.md)
- **Location**: [`evaluation/mmp.py`](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py) vs [`core/engine_surrogate/analytical_models.py:530-580`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L530-L580)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - `evaluation/mmp.py` and `analytical_models.py` had duplicate in-line implementations of Cronquist and Yellig-Metcalfe with divergent formulas.
- **Resolution Details**:
  - Consolidated into a single source of truth in `evaluation/mmp.py`. `analytical_models.py` now delegates directly to `evaluation.mmp.calculate_mmp`.
- **Verification**: `tests/scientific/co2/test_mmp_correlations.py`.

---

## 5. Decommissioned Code & Architectural Debt

<a id="dead-05"></a>
### [DEAD-05] Removed Root Artifacts & Superseded Scripts
- **ID**: `DEAD-05`
- **Category**: Dead Code
- **Original Document**: [`agent_wiki/audit/dead_code.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/dead_code.md)
- **Location**: Repository Root
- **Severity**: LOW
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Redundant and orphaned root files littered the root directory: `data_processor.py`, `ui/workers/data_processing_worker.py`, `core/engine_surrogate/analytical_models – копія.py`, `fixed_phd_class.txt`, `test_surrogate.py`, `nul`, `report.log`.
- **Resolution Details**:
  - Removed all orphaned scripts and relocated root utility modules (`config_manager.py`, `error_handler.py`, `path_utils.py`, `validation_manager.py`) into the `utils/` package.
- **Verification**: Clean root directory audit.

---

<a id="dead-06"></a>
### [DEAD-06] Removed Scientific Justification and Help Subsystems
- **ID**: `DEAD-06`
- **Category**: Dead Code
- **Original Document**: [`agent_wiki/audit/dead_code.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/dead_code.md)
- **Location**: `ui/dialogs/scientific_justification_dialog.py`, `ui/dialogs/parameter_help_dialog.py`
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-17
- **Previous Defect**:
  - Legacy dialogs, HTML assets, and help Content YAML files accumulated maintenance overhead without being used in the streamlined workflow.
- **Resolution Details**:
  - Deleted legacy justification and help dialogs, removed mathjax assets, and cleaned up menu actions and side panel splitters in `ui/main_window.py` and `main.py`.
- **Verification**: GUI startup and smoke tests.

---

## 6. Software Reliability & Runtime Defects

### [SFT-01] PVT FVF Keyword Mismatch & $B_g$ Fallback Eradication
- **ID**: `SFT-01`
- **Category**: Software Defect / Fallback Elimination
- **Original Document**: [`agent_wiki/audit/fallbacks.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/fallbacks.md)
- **Location**: [`core/optimisation_engine.py:235-244`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L235-L244), [`core/engine_surrogate/pvt_state.py:257-265`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/pvt_state.py#L257-L265)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - `core/optimisation_engine.py` called `SolventExtendedPVTEngine.calculate_co2_fvf_rb_per_mscf(p_psia=..., t_f=...)`.
  - The method signature only accepted positional argument `pressure_psi: float`, raising `TypeError: unexpected keyword argument 'p_psia'`.
  - The exception handler caught this and fell back to `B_GAS_RB_PER_MSCF = 5.0` (10× larger than physical supercritical $B_g \approx 0.50\text{ RB/MSCF}$), causing artificial 10× over-inflation of voidage.
- **Resolution Details**:
  - Expanded `calculate_co2_fvf_rb_per_mscf` in `core/engine_surrogate/pvt_state.py` to accept flexible keyword arguments (`pressure_psi`, `p_psia`, `t_f`).
  - Updated the call site in `core/optimisation_engine.py` to pass `pressure_psi=p_psia, t_f=t_f`.
  - Eliminated the 10× fallback; active engine now evaluates dynamic PR-EOS formation volume factors without exceptions.
- **Verification**: `tests/test_pvt_solvent_extended.py`, `tests/core/test_optimisation.py`.

---

### [SFT-02] Economic Objective MagicMock Array Broadcasting ValueError
- **ID**: `SFT-02`
- **Category**: Software Defect
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`tests/core/test_objective_functions.py:197`](file:///d:/rep/4.6/co2eor_optimizer/tests/core/test_objective_functions.py#L197), [`core/objectives/economic.py:64-70`](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/economic.py#L64-L70)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - In unit tests where `economic_params = MagicMock()`, multiplying `oil_production * getattr(economic_params, "variable_opex_usd_per_bbl", ...)` attempted to broadcast NumPy array shape `(15,)` against a mock object with shape `(0,)`, raising `ValueError: operands could not be broadcast together`.
- **Resolution Details**:
  - Added strict `isinstance(val, (int, float, np.number, np.ndarray))` validation guards in `core/objectives/economic.py` before array arithmetic.
  - Properly configured numeric scalar attributes on `economic_params` in test fixtures.
- **Verification**: `tests/core/test_objective_functions.py` passes 100%.

---

### [SFT-03] Missing `set_engine` Method on OptimizationWidget
- **ID**: `SFT-03`
- **Category**: Software Defect
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`ui/optimization_widget.py:175-182`](file:///d:/rep/4.6/co2eor_optimizer/ui/optimization_widget.py#L175-L182), [`tests/test_optimization_widget_export_parameters.py:103, 153`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_optimization_widget_export_parameters.py#L103)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Optimization widget refactoring renamed `set_engine` to `update_engine`. Test suites calling `widget.set_engine(engine)` failed with `AttributeError`.
- **Resolution Details**:
  - Added a backward-compatible `set_engine(self, engine)` method on `OptimizationWidget` delegating directly to `self.update_engine(engine)`.
- **Verification**: `tests/test_optimization_widget_export_parameters.py` passes 100%.

---

### [SFT-F821] Eradication of All 36 Ruff F821 Undefined Names
- **ID**: `SFT-F821`
- **Category**: Software Defect / Static Code Quality
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: Multiple modules across `core/`, `ui/`, `analysis/`
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Ruff static analysis flagged 36 critical `F821` undefined names that produced potential `NameError` crashes at runtime:
    - `core/optimisation_engine.py`: `EPSILON`, `SimulatorExporter`
    - `core/data_integration_engine.py`: `GeomechanicsParameters`, `create_geostatistical_grid`
    - `core/validation/physical_consistency_validator.py`: `CCUSState`
    - `ui/optimization_widget.py`: `UnlockParametersDialog`
    - `ui/analysis_widget.py`, `ui/config_widget.py`: `ConfigManager`
    - `ui/main_window.py`: `report_charts`, undefined layout indices
    - `ui/sensitivity_widget.py`: `pd`, `np`, `go`, `make_subplots`
    - `ui/widgets/log_viewer_dialog.py`: `QTableWidgetItem`
    - `analysis/sensitivity_analyzer.py`: missing `self` in `run_two_way_sensitivity`, unhandled EOS imports
    - `ui/ai_assistant_widget.py`: `AI_SERVICES_CONFIG`, `QInputDialog`
- **Resolution Details**:
  - Implemented `UnlockParametersDialog` in `ui/optimization_widget.py`.
  - Added all missing standard and third-party imports across UI and analysis modules.
  - Added `self` parameter to `run_two_way_sensitivity` method in `analysis/sensitivity_analyzer.py`.
  - Added fallback safe definition of `EPSILON = 1e-10` in `core/optimisation_engine.py`.
  - `ruff check --select F821` returns **0 errors** across the entire codebase.
- **Verification**: `ruff check --select F821` returns empty; test suite passes 100% (299 passed, 0 failed).

---

### [SFT-04] AttributeError on ConfigWidget.engine_selection_changed
- **ID**: `SFT-04`
- **Category**: Software Defect
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`ui/config_widget.py:80`](file:///d:/rep/4.6/co2eor_optimizer/ui/config_widget.py#L80), [`ui/main_window.py:414-416`](file:///d:/rep/4.6/co2eor_optimizer/ui/main_window.py#L414-L416)
- **Severity**: CRITICAL
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - When `ConfigWidget` was refactored to remove unused/legacy engine dropdowns (since `core/engine_surrogate` is the single active engine), the signal `engine_selection_changed = pyqtSignal(str)` was removed from `ConfigWidget`.
  - When `MainWindow._setup_main_app_tabs_container()` ran, it called `self.config_tab.engine_selection_changed.connect(...)`, which crashed immediately with `AttributeError: 'ConfigWidget' object has no attribute 'engine_selection_changed'`, preventing application startup.
- **Resolution Details**:
  - Re-added `engine_selection_changed = pyqtSignal(str)` to `ConfigWidget` in `ui/config_widget.py` for backward compatibility.
  - Added defensive guard `if hasattr(self.config_tab, "engine_selection_changed"):` in `ui/main_window.py`.
- **Verification**: Headless `MainWindow` startup and event loop execution verified with exit code 0.

---

### [SFT-05] Root Import Mismatch on ConfigManager in DataManagementWidget
- **ID**: `SFT-05`
- **Category**: Software Defect
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`ui/data_management_widget.py:25, 183`](file:///d:/rep/4.6/co2eor_optimizer/ui/data_management_widget.py#L25)
- **Severity**: HIGH
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - Following root-level module reorganization where `config_manager.py` moved to `utils/config_manager.py`, `ui/data_management_widget.py` still contained `from config_manager import ConfigManager`.
  - The import failed with `ModuleNotFoundError: No module named 'config_manager'`, setting both `PreferencesManager` and `ConfigManager` to `None` and logging `CRITICAL: DataManagementWidget: PreferencesManager or ConfigManager not found. Unit system preferences will not work.`
- **Resolution Details**:
  - Corrected imports at module level and fallback scope to `from utils.config_manager import ConfigManager`.
  - Added compatibility method `set_engine_type(self, engine_type: str)` to `DataManagementWidget` to handle engine notifications from `MainWindow`.
- **Verification**: `DataManagementWidget` and `MainWindow` unit tests pass; critical error eliminated.

---

### [SFT-06] Missing Qt Translation Files Suppressed Gracefully
- **ID**: `SFT-06`
- **Category**: Software Defect
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`utils/i18n_manager.py:98-126`](file:///d:/rep/4.6/co2eor_optimizer/utils/i18n_manager.py#L98-L126)
- **Severity**: LOW
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - `translations/` contained only empty placeholder `.ts` files and no compiled `.qm` binaries.
  - On startup on non-English systems (e.g. Ukrainian `uk`), `I18nManager.load_and_install_translator('uk')` attempted to load `app_uk.qm` and `app_en.qm`, emitting repeated `WARNING` logs:
    `WARNING - Could not load translation file for locale 'uk': ... app_uk.qm`
    `WARNING - Fallback English translation 'app_en.qm' also not found.`
- **Resolution Details**:
  - The application source strings are natively defined in English.
  - Added explicit `translation_path.is_file()` existence check. If translation files are absent, it logs at `DEBUG` level and quietly falls back to default English UI text without emitting warnings.
- **Verification**: Verified clean startup without translation warnings.

---

### [SFT-07] Parsers Directory Elimination & LAS Parser Consolidation
- **ID**: `SFT-07`
- **Category**: Architectural Refactoring / Dead Code Deprecation
- **Original Document**: [`agent_wiki/architecture/module_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/module_map.md)
- **Location**: `parsers/` $\rightarrow$ `deprecated/parsers/` and [`utils/las_parser.py`](file:///d:/rep/4.6/co2eor_optimizer/utils/las_parser.py)
- **Severity**: LOW
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - The `parsers/` directory contained 3 files: `base_parser.py` (abstract OOP interface for dead ECLIPSE parser), `validation.py` (obsolete 3D ECLIPSE grid checks and orphaned well checks), and `las_parser.py` (functional LAS log parser).
  - Having a standalone top-level package `parsers/` for a single active file was unnecessary overhead.
- **Resolution Details**:
  - Archived `base_parser.py`, `validation.py`, and legacy `eclipse_parser.py` into `deprecated/parsers/`.
  - Relocated `las_parser.py` into `utils/las_parser.py` and exported `parse_las` and `MissingWellNameError` in `utils/__init__.py`.
  - Removed top-level `parsers/` directory completely.
  - Wired up `Import LAS...` button in `ui/data_management_widget.py` Wells tab to enable interactive loading of `.las` logs into `WellData`.
  - Fixed `lasio.exceptions.LASHeaderError` import and file existence checks in `utils/las_parser.py`.
- **Verification**: `tests/test_las_parser.py` (6 unit tests, 100% passing); full pytest suite passes with 0 errors.

---

### [SFT-08] Core Directory Architecture Audit & Prototyping Artifacts Elimination
- **ID**: `SFT-08`
- **Category**: Architectural Refactoring / Dead Code Elimination
- **Original Document**: [`agent_wiki/architecture/source_of_truth_map.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/architecture/source_of_truth_map.md), [`agent_wiki/audit/dead_code.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/dead_code.md)
- **Location**: `core/`
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - The `core/` package accumulated prototyping artifacts, unused grid abstractions, dormant validation subdirectories, duplicate plotting routines, and abandoned response surface ML models.
  - `core/simulation/recovery_models.py` contained numerical solvers superseded by `core/engine_surrogate/analytical_models.py`.
  - `core/geology/__init__.py` exported `GeologyEngine`, an uncalibrated module using arbitrary depth multipliers (1.1, 0.9, 0.7).
  - `core/optimisation_engine.py` held dead fallback imports (`npv`, breakthrough classes), unphysical constants (`ACRES_TO_CM2 = 40468564.224`, `B_GAS_RB_PER_MSCF = 5.0`), and duplicated 165 lines of Plotly charts implemented in `PlottingManager`.
  - Welge tangent construction in Buckley-Leverett and literature fractional flow in `analytical_models.py` evaluated `max(k_rg, EPSILON)` on array `k_rg`, triggering `ValueError: The truth value of an array with more than one element is ambiguous`.
- **Resolution Details**:
  - **Dead Code Elimination**: Completely removed `core/validation/` (1,290 lines), `core/optimization_analysis.py` (886 lines), `core/objectives/base.py` & `production.py` (136 lines), `core/exceptions.py` (32 lines), and `core/utils/` (36 lines).
  - **Subsystem Deprecations & Shims**:
    - Relocated `core/simulation/recovery_models.py` and `profile_generator.py` to `deprecated/core/simulation/`, adding transparent backward-compatible deprecation shims in `core/simulation/`.
    - Relocated uncalibrated `GeologyEngine` to `deprecated/core/geology/geology_engine.py` and added a lazy deprecation import shim in `core/geology/__init__.py`.
    - Relocated ML response surface files (`response_surfaces.py`, `feature_transformer.py`, `training_data.py`, `model_factory.py`) to `deprecated/core/engine_surrogate/`.
    - Relocated deck export logic to `utils/cmg_exporter.py` with backward-compatible deprecation shim in `core/simulation/simulator_exporter.py`.
  - **In-File Prototyping Cleanups**:
    - `core/data_models.py`: Removed dead grid classes (`GridType`, `GridBase`, `SimpleGrid`, `FullPhysicsGrid`), `ReservoirState`, and `RockProperties`. Inlined pore volume calculation. Added `FluidProperties` compatibility dataclass.
    - `core/data_integration_engine.py`: Removed duplicate `DataValidator` and `UnitConverter`, refactored to use `PhysicalConstants`.
    - `core/objectives/storage.py`: Removed dead prototype functions; retained verified active containment and storage efficiency models.
    - `core/engine_surrogate/analytical_models.py`: Replaced `max(k_rg, EPSILON)` with `np.maximum(k_rg, EPSILON)` for vectorized Welge shock front construction; added `KovalRecoveryModel = KovalSurrogate` alias.
    - `core/optimisation_engine.py`: Removed dead imports (`npv`, breakthrough classes); cleaned constants (`B_GAS_RB_PER_MSCF = 1.0` fallback); delegated GA, hybrid model, and breakthrough mechanism plotting directly to `PlottingManager`.
  - **Startup Smoke Test**: Added `tests/test_app_startup.py` which executes `timed_import_main_window()`, `SensitivityAnalyzer`, and `ProductionProfiler` under `QApplication`, permanently guarding against startup import regressions.
- **Verification**: All 310 tests pass (307 existing + 3 startup smoke tests); application boots and exits cleanly with code 0.

---

### [SFT-09] Application Entry Point (`main.py`) Modernization & Multiprocess Logging Decoupling
- **ID**: `SFT-09`
- **Category**: Architectural Refactoring / Code Quality
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`main.py`](file:///d:/rep/4.6/co2eor_optimizer/main.py), [`ui/main_window.py`](file:///d:/rep/4.6/co2eor_optimizer/ui/main_window.py), [`utils/multiprocess_logging.py`](file:///d:/rep/4.6/co2eor_optimizer/utils/multiprocess_logging.py)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - `main.py` had accumulated 390 lines with excessive debugging scaffolding, a 140-line import wall with 40+ unused imports inside `timed_import_main_window`, platform-specific workarounds (`windows:darkmode=0`), a hardcoded 21-line palette override, and inlined logging setup/teardown functions.
  - Screen geometry initialization was handled by an external helper function in `main.py` rather than being encapsulated within `MainWindow`.
  - Pyright/Pylance emitted false-positive keyword argument errors for `MainWindow` because PyQt6's underlying C++ bindings lack rich type hints for subclassed constructors.
- **Resolution Details**:
  - Reduced `main.py` from 390 lines to 163 clean, modular lines.
  - Decoupled application logging into `init_application_logging(...)` inside `utils/multiprocess_logging.py` and called `shutdown_queue_logging()` on exit.
  - Replaced the 140-line import block with a clean 13-line timing wrapper around `from ui.main_window import MainWindow`.
  - Retained `QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)` before creating `CO2EORApplication(sys.argv)`, strictly preserving PyQt6 QtWebEngine context requirements.
  - Encapsulated default first-launch centering and 80% screen geometry fallback directly in `MainWindow.load_window_settings()`.
  - Added `from __future__ import annotations`, typed `timed_import_main_window() -> type[MainWindow]`, and added `# type: ignore[call-arg]` on `main_window = MainWindow(...)`.
- **Verification**: Application boots cleanly with exit code 0; `tests/test_app_startup.py` passes 3/3 smoke tests.

---

### [SFT-10] Data Models (`core/data_models.py`) Type Safety Hardening & Fault/Fluid Separation
- **ID**: `SFT-10`
- **Category**: Software Defect / Type Safety
- **Original Document**: [`agent_wiki/audit/technical_debt.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/technical_debt.md)
- **Location**: [`core/data_models.py`](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py)
- **Severity**: MEDIUM
- **Status**: RESOLVED
- **Date Resolved**: 2026-09-23
- **Previous Defect**:
  - In `calculate_ooip_from_physics`, using `if None in [...]` failed to narrow `Optional[float]` fields, causing 8 operand type mismatch errors (`*` and `/` not supported between `float` and `None`).
  - In `pore_volume`, `self.length_ft` and `self.cross_sectional_area_acres` were multiplied without non-None guards.
  - In `FaultProperties`, an old fragment of fluid properties (including 6 density/viscosity methods and a duplicate `__post_init__`) was accidentally pasted inside the class definition, overriding the fault validator and triggering 16 `AttributeError` warnings.
  - `CCUSState.fluxes` was declared as `fluxes: np.ndarray = None`, which Pyright rejected as invalid assignment to non-optional type.
  - `from_dict_to_dataclass` was un-typed, causing `from_config_dict` to return implicit `Any`.
- **Resolution Details**:
  - Converted `calculate_ooip_from_physics` to explicit `is None` checks, guaranteeing safe float narrowing.
  - Added fallback defaults to `pore_volume` calculation for `length_ft` and `cross_sectional_area_acres`.
  - Completely cleaned `FaultProperties`, removing the duplicate `__post_init__` and misplaced fluid methods.
  - Enhanced the standalone `FluidProperties` dataclass with all reference properties (`water_density_ref`, `oil_viscosity_ref`, compressibilities, FVF) and temperature/pressure calculations with explicit float casts.
  - Declared `fluxes: Optional[np.ndarray] = None` in `CCUSState`.
  - Added generic type parameter `TypeVar("T")` to `from_dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T`.
- **Verification**: `tests/test_reservoir_outputs_streams.py` and `tests/test_app_startup.py` pass 6/6 tests.



