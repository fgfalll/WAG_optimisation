# Simulation Run Audits & Historical Performance Registry

This document serves as the authoritative historical registry of all **simulation run audits, benchmark evaluations, campaign sweeps, and surrogate profile audits** in the CO₂ EOR Optimizer repository.

Whenever developers or autonomous AI agents execute, audit, or benchmark a reservoir simulation run, they must log their findings here following the standardized audit protocol. This guarantees long-term visibility into past runs, tracks physical and numerical validity over time, and provides traceable rationale for model proposals.

---

## 📋 Mandatory Simulation Run Audit Protocol & Schema

Every simulation run audit **MUST** follow this machine-readable schema:

```markdown
### [SIM-AUDIT-DD-MM-YYYY-XX] Scenario Title / Description
- **Date**: `DD-MM-YYYY` (e.g. `24-09-2026`)
- **Run ID**: `SIM-AUDIT-DD-MM-YYYY-XX`
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` + `FastProfileGenerator`)
- **Injection Scheme**: `WAG` | `Continuous CO2` | `Waterflooding` | `Gas Cycling`
- **Simulation Duration**: `X years` (Time step: `daily` / `monthly` / `annual`)
- **Verdict**: `PASSED` | `ACCEPTABLE WITH CONDITIONS` | `FLAGGED` | `FAILED`
- **Proposal**: Concrete actionable proposal (e.g., model adjustments, parameter limits, grid updates, operational recommendations).
- **Relevant Files**:
  - Configuration: [`config/recovery_config.json`](file:///d:/rep/4.6/co2eor_optimizer/config/recovery_config.json)
  - Engine Wrapper: [`core/engine_surrogate/surrogate_engine.py`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py)
  - Test / Execution Script: [`tests/test_simulation.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_simulation.py)
  - Export / Output Data: [`output/run_results.json`](file:///d:/rep/4.6/co2eor_optimizer/output/run_results.json)
- **Key Metrics**:
  - STOOIP: `X.XX MMSTB`
  - Ultimate Recovery Factor (RF): `XX.X %`
  - Net CO₂ Stored: `X.XX M tonnes` (`XX.XX BSCF`)
  - Gross CO₂ Mass Balance Closure: `XX.XX %` (Must be > 99.9%)
  - Peak Sandface Pressure: `XXXX psia` (Limit: `XXXX psia`, Margin: `XX %`)
  - Project NPV: `$XX.XX M`
- **Physical Sanity Observations**:
  - Darcy / Vogel IPR drawdown physical check.
  - Viscous fingering & breakthrough check (Koval & Todd-Longstaff).
  - Geomechanical stress path and Class VI UIC bounds check.
```

---

## 📑 Master Simulation Run Audits Index

| Run ID | Date (`DD-MM-YYYY`) | Scenario & Model | Verdict | Proposal Summary | Primary Relevant Files |
|:---|:---|:---|:---|:---|:---|
| **SIM-AUDIT-23-09-2026-01** | `23-09-2026` | SPE 5 Benchmark Comparison (Quarter 5-Spot WAG) | **PASSED** | Adopt dynamic Koval fractional flow as baseline. | [`tests/validation/spe5_benchmark_validation.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/validation/spe5_benchmark_validation.py) |
| **SIM-AUDIT-24-09-2026-01** | `24-09-2026` | 4-Stream Closed-Loop Mass Balance & Deliverability | **PASSED** | Maintain synchronous multi-stream shut-in across all phases. | [`tests/test_closed_loop_and_well_roles.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_closed_loop_and_well_roles.py), [`core/engine_surrogate/profile_generator_fast.py`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py) |
| **SIM-AUDIT-24-09-2026-02** | `24-09-2026` | End-to-End Saved Project Run Restoration (`test.tphd`) | **PASSED** | Require shallow dataclass serialization and `.flat[0]` grid indexing. | [`tests/test_project_save_load.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_project_save_load.py), [`utils/project_file_handler.py`](file:///d:/rep/4.6/co2eor_optimizer/utils/project_file_handler.py) |

---

## 🔬 Historical Simulation Run Audits

### [SIM-AUDIT-23-09-2026-01] SPE 5 Benchmark Comparison (Quarter 5-Spot WAG)
- **Date**: `23-09-2026`
- **Run ID**: `SIM-AUDIT-23-09-2026-01`
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` + `FastProfileGenerator`)
- **Injection Scheme**: `WAG` (Water-Alternating-Gas, 1:1 cycle ratio, 20-year run)
- **Simulation Duration**: `20.0 years` (Monthly time steps)
- **Verdict**: `PASSED`
- **Proposal**:
  1. Standardize on Todd-Longstaff mixing parameter $\omega = 0.67$ for miscible CO₂ displacements to accurately reflect viscous fingering attenuation.
  2. Bounding water injection rates by total mobility contrast $\Delta \lambda / \Sigma \lambda$ to prevent unphysical water over-injection.
- **Relevant Files**:
  - Validation Script: [`tests/validation/spe5_benchmark_validation.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/validation/spe5_benchmark_validation.py)
  - Engine Reference: [`core/engine_surrogate/surrogate_engine.py`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py)
  - Displacement Theory: [`agent_wiki/physics/displacement_model.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/physics/displacement_model.md)
  - Validation Summary: [`agent_wiki/validation/benchmarks.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/validation/benchmarks.md)
- **Key Metrics**:
  - STOOIP: `48.50 MMSTB`
  - Ultimate Recovery Factor (RF): `47.8 %`
  - Net CO₂ Stored: `3.42 M tonnes`
  - Gross CO₂ Mass Balance Closure: `100.0 %`
  - Peak Sandface Pressure: `3,850 psia` (Frac limit: `4,350 psia`, Margin: `11.5 %`)
  - Project NPV: `$142.30 M`
- **Physical Sanity Observations**:
  - Gas breakthrough occurs smoothly at $0.28\text{ PVI}$, matching SPE 5 benchmark envelope within $\pm 4.2\%$.
  - Plateau rate strictly honors Composite Vogel-Darcy drawdown limit.

---

### [SIM-AUDIT-24-09-2026-01] 4-Stream Closed-Loop Mass Balance & Deliverability Audit
- **Date**: `24-09-2026`
- **Run ID**: `SIM-AUDIT-24-09-2026-01`
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` + `FastProfileGenerator`)
- **Injection Scheme**: `WAG` (Dynamic gas recycling, 15-year evaluation)
- **Simulation Duration**: `15.0 years` (Daily time steps aggregated to annual)
- **Verdict**: `PASSED`
- **Proposal**:
  1. Maintain strict synchronization across all 4 fluid streams (`Crude Oil`, `Natural Gas`, `Water`, and `Injection Agent`) when environmental or mechanical shut-in occurs.
  2. Require all automated test suites to assert gross mass balance closure:
     $$\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Leakage} + \text{Produced}$$
- **Relevant Files**:
  - Test Suite: [`tests/test_closed_loop_and_well_roles.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_closed_loop_and_well_roles.py)
  - Output Streams Test: [`tests/test_reservoir_outputs_streams.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_reservoir_outputs_streams.py)
  - Fast Profile Generator: [`core/engine_surrogate/profile_generator_fast.py`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py)
  - Mass Balance Documentation: [`agent_wiki/validation/conservation.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/validation/conservation.md)
- **Key Metrics**:
  - STOOIP: `32.10 MMSTB`
  - Ultimate Recovery Factor (RF): `41.6 %`
  - Net CO₂ Stored: `2.15 M tonnes`
  - Gross CO₂ Mass Balance Closure: `100.00 %`
  - Recycled vs Produced Ratio: `M_recycled <= M_produced` rigorously maintained.
  - Peak Sandface Pressure: `3,420 psia` (Frac limit: `3,900 psia`)
- **Physical Sanity Observations**:
  - Zero post shut-in tail leakage. Produced gas streams taper smoothly to zero when water cut breaches 95%.
  - No negative effective mobilities observed.

---

### [SIM-AUDIT-24-09-2026-02] End-to-End Saved Project Run Restoration (`test.tphd`)
- **Date**: `24-09-2026`
- **Run ID**: `SIM-AUDIT-24-09-2026-02`
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` loaded via `project_file_handler`)
- **Injection Scheme**: `WAG` (Field case loaded from saved `.tphd`)
- **Simulation Duration**: `25.0 years`
- **Verdict**: `PASSED`
- **Proposal**:
  1. Enforce shallow dataclass encoding across all nested objects (`EOSModelParameters`, `LayerDefinition`, `GeostatisticalParams`) to avoid stripping `_dataclass` tags.
  2. Implement array-agnostic indexing using `.flat[0]` across all grid ingestion routines in UI and engine adapters.
  3. Keep `@results.setter` on `OptimizationEngine` to enable seamless result hydration onto UI visualization tabs.
- **Relevant Files**:
  - Test Suite: [`tests/test_project_save_load.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_project_save_load.py)
  - Project File Handler: [`utils/project_file_handler.py`](file:///d:/rep/4.6/co2eor_optimizer/utils/project_file_handler.py)
  - Data Management Widget: [`ui/data_management_widget.py`](file:///d:/rep/4.6/co2eor_optimizer/ui/data_management_widget.py)
  - Optimization Widget: [`ui/optimization_widget.py`](file:///d:/rep/4.6/co2eor_optimizer/ui/optimization_widget.py)
  - Base Test Project: [`test.tphd`](file:///d:/rep/4.6/co2eor_optimizer/test.tphd)
- **Key Metrics**:
  - Restored Parameter Verification: 100% field match across all PVT, reservoir geometry, and well lists.
  - Restored Optimization Objective: Preserved identical best-fit NPV and recovery factor curve upon reload.
  - Gross Closure: `100.00 %`
- **Physical Sanity Observations**:
  - All input parameters loaded intact without fallback overrides.
  - Graphing engine immediately plotted active profiles upon project load.
