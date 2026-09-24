# Simulation Run Audits & Historical Performance Registry

This directory serves as the authoritative historical repository of all **reservoir simulation run audits, benchmark evaluations, campaign sweeps, and surrogate profile audits** in the CO₂ EOR Optimizer repository.

Every simulation run audit is stored in its own dedicated, date-stamped subfolder containing an `audit.md` report alongside key diagnostic artifacts (plots, stream CSVs, execution manifests, and logs).

---

## 🛠️ Mandatory Agent Workflow for Simulation Run Audits

Whenever an autonomous AI agent or developer conducts a simulation run, evaluates an optimization campaign, or benchmarks the reservoir engine, they **MUST** execute the following 5-step workflow:

```
[Simulation Run / Export] 
          │
          ▼
1. Locate Run Directory (in logs/ or output/)
          │
          ▼
2. Create Audit Subfolder: agent_wiki/audit/simulation_run_audits/DD-MM-YYYY_<run_name>/
          │
          ▼
3. Copy Key Run Artifacts (*.png plots, summary_*.csv, run_manifest.json, *.txt)
          │
          ▼
4. Author audit.md (Mandatory Schema: DD-MM-YYYY, Verdict, Proposal, Relevant Files)
          │
          ▼
5. Register in Master Index (agent_wiki/audit/simulation_run_audits/index.md)
```

### Detailed Step-by-Step Instructions:

1. **Step 1: Locate Run Artifacts**:
   Identify the completed simulation output folder in `logs/` (e.g. `logs/Export-single-simulation-YYYYMMDD-HHMMSS/`) or `output/`, along with session log files.
2. **Step 2: Create Date-Stamped Audit Subfolder**:
   Create a new directory inside `agent_wiki/audit/simulation_run_audits/`:
   ```bash
   agent_wiki/audit/simulation_run_audits/DD-MM-YYYY_<descriptive_run_name>/
   ```
   *Requirement*: The folder name **MUST** start with the date in `DD-MM-YYYY` format (e.g. `24-09-2026_single_simulation_baseline`).
3. **Step 3: Copy Diagnostic Artifacts**:
   Copy all essential diagnostic files from the run into the new subfolder:
   - Diagnostic plots: `production_profiles.png`, `material_balance.png`, `cumulative_co2_balance.png`, `storage_efficiency.png`
   - Stream data: `summary_yearly.csv`, `summary_monthly.csv`, `cash_flows_yearly.csv`
   - Metadata & configs: `run_manifest.json`, `run_evaluation_report.md`, `results_summary.txt`
4. **Step 4: Author `audit.md`**:
   Write a comprehensive `audit.md` inside the subfolder following the required schema:
   - **Date**: Formatted strictly as `DD-MM-YYYY`.
   - **Run ID**: Unique identifier (e.g. `SIM-AUDIT-24-09-2026-01`).
   - **Verdict**: One of `PASSED`, `ACCEPTABLE WITH CONDITIONS`, `FLAGGED`, or `FAILED`.
   - **Proposal**: Actionable recommendations (search bound adjustments, physics fixes, operational guidelines).
   - **Relevant Files**: Markdown links to local subfolder artifacts and code sources.
   - **Key Metrics Table**: OOIP, recovery factor (RF), net CO₂ stored, mass balance closure, peak sandface pressure, NPV.
   - **Physical Sanity Checks**: IPR drawdown, viscous fingering, EPA Class VI UIC geomechanical limits.
5. **Step 5: Register in Master Index**:
   Add a new entry to the [Master Simulation Run Audits Index](#master-simulation-run-audits-index) below with a link to the subfolder's `audit.md`.

---

## 📋 Standard Audit Record Schema

```markdown
# Simulation Run Audit: [Run Title]

- **Date**: `DD-MM-YYYY` (e.g. `24-09-2026`)
- **Run ID**: `SIM-AUDIT-DD-MM-YYYY-XX`
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` + `FastProfileGenerator`)
- **Injection Scheme**: `WAG` | `Continuous CO2` | `Waterflooding` | `Gas Cycling`
- **Simulation Duration**: `X years` (Time step: `daily` / `monthly` / `annual`)
- **Verdict**: `PASSED` | `ACCEPTABLE WITH CONDITIONS` | `FLAGGED` | `FAILED`

## Executive Summary & Operational Proposal
- **Verdict Rationale**: Bulleted explanation of why this verdict was assigned.
- **Actionable Proposal**: Specific proposals for tuning bounds, physics adjustments, or operational limits.

## Relevant Files & Run Artifacts
- Links to local plots, CSV tables, manifest JSON, and source logs.

## Key Metrics
- Table of STOOIP, Recovery Factor, Cumulative Streams, Stored CO₂, Mass Balance Closure %, NPV.

## Geomechanical & Containment Integrity
- Sandface injection pressure vs EPA Class VI $0.90 \times P_{\text{frac}}$ ceiling.

## Carbon Mass Balance Reconciliation
- Table asserting Gross Injected = Purchased + Recycled = Net Stored + Leakage + Produced.
```

---

## 📑 Master Simulation Run Audits Index

| Run ID | Date (`DD-MM-YYYY`) | Scenario & Model | Verdict | Proposal Summary | Audit Report Link |
|:---|:---|:---|:---|:---|:---|
| **SIM-AUDIT-24-09-2026-01** | `24-09-2026` | Single Simulation Baseline Evaluation (GA Parameter Profile Export) | **ACCEPTABLE WITH CONDITIONS** | Widen injection rate and well pressure bounds; define explicit well patterns. | [`24-09-2026_single_simulation_baseline/audit.md`](24-09-2026_single_simulation_baseline/audit.md) |
| **SIM-AUDIT-23-09-2026-01** | `23-09-2026` | SPE 5 Benchmark Comparison (Quarter 5-Spot WAG) | **PASSED** | Standardize Todd-Longstaff mixing parameter $\omega = 0.67$. | [`tests/validation/spe5_benchmark_validation.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/validation/spe5_benchmark_validation.py) |
| **SIM-AUDIT-24-09-2026-02** | `24-09-2026` | End-to-End Saved Project Run Restoration (`test.tphd`) | **PASSED** | Require shallow dataclass serialization and array-agnostic `.flat[0]` grid indexing. | [`tests/test_project_save_load.py`](file:///d:/rep/4.6/co2eor_optimizer/tests/test_project_save_load.py) |
