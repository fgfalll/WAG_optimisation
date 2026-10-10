# Simulation Run Audit: Forensic Scientific Audit of the Active `hybrid` Path

- **Date**: `04-10-2026`
- **Run ID**: `SIM-AUDIT-04-10-2026-01`
- **Run Type**: Reproducible numerical verification campaign (18 standalone scripts) + full pytest/coverage
  static-analysis toolchain. **Audit-only: no source code was modified.**
- **Engine**: `core/engine_surrogate` (`SurrogateEngineWrapper` + `FastProfileGenerator`), production default
  `recovery_model_type = "hybrid"`, 10-year run, monthly time base, `T = 180 °F`.
- **Injection Scheme**: `Continuous CO₂` (plus a `WAG` variant used to isolate CRIT-11)
- **Simulation Duration**: `10 years` (monthly)
- **Verdict**: **`FLAGGED`**

> **Post-run refresh (04-10-2026, after commits `a68fc35` / `9d252ce`).** The verdict and every
> physics number above are unchanged — no scientific code was touched. Three *software* facts did
> change and are recorded here rather than silently edited into the tables below:
> 1. **HIGH-10 RESOLVED** — the missing `QIcon` import was added at
>    `ui/widgets/fault_geometry_visualizer_widget.py:19`; `pytest tests/ -q` now returns
>    **`0 failed / 333 passed / 23 skipped`**, so the `AGENTS.md` invariant #5 gate
>    (`tests/test_project_save_load.py`) is green again.
> 2. **HIGH-19 NEW** — re-running ruff found two undefined names in the new workbench code
>    (`pyvista_reservoir_canvas.py:974` `has_active_fault`, `subsurface_data_viewer_widget.py:335`
>    `QToolTip`). One is swallowed by `except Exception` and silently blanks the 3-D caprock view;
>    the other raises in an unguarded Qt slot. Neither is reached by any test, so **0 failures and
>    two reachable `NameError`s coexist** — which is precisely the HIGH-18 point.
> 3. **Register size 52 → 53** (13 CRITICAL / 19 HIGH / 16 MEDIUM / 5 LOW), re-derived by
>    `evidence_scripts/v_recount.py` (53 IDs parsed, 0 missing). Ruff is now `3244 / 191` with
>    `F821 6` (was `3132 / 192`, `F821 5`).

---

## Executive Summary & Operational Proposal

### Verdict Rationale

- **Mass balance closes exactly**, but *by construction rather than by enforcement*: injected
  36 525 000 MSCF = purchased 34 994 496.3 + recycled 1 530 503.7 (closure to the last digit), recycled
  1 530 503.7 ≤ produced 1 611 056.5, `total_leakage_tonne ≡ 0.0`. No code path asserts the inequality —
  recycled is clamped as `min(prod·η_recycle, inj)` at `optimisation_engine.py:871-872`.
- **The reported recovery factor is a post-hoc clipped artifact.** Raw RF for the `hybrid` model came out
  **0.712499** and was reported as **0.500000**; the correct OOIP-normalised cap for this reservoir is
  **0.666667**, so the reported value is **25 % low** because `rf_max_physical` uses a pore-volume
  fraction instead of an OOIP-normalised fraction (**HIGH-01**).
- **The reported NPV has two ledgers.** Same reservoir, same RF 0.500000, same `cumulative_oil`,
  `hybrid` → **$106 601 776** vs `phd_hybrid` → **$60 733 386** — a **45 % divergence** with identical
  reported recovery (**CRIT-02**).
- **Primary throughput is pressure-independent.** Model `hcpvi = 0.750000` at 1500, 2500 **and** 3500 psi
  (textbook: 0.907564 / 0.406907 / 0.278135 → ratios 0.826 / 1.843 / 2.697), so the single most
  important dimensionless group in a CO₂ flood does not respond to pressure (**CRIT-01**).
- **Recovery models are degenerate over their own design space.** Koval sweep returns RF = 0.000000 for
  M = 0.5 … 1.4 with an unexplained spike 0.316738 at exactly M = 1.0 (**CRIT-06**); the immiscible model
  returns min = max = 0.100000 across a 27-point parameter grid (**CRIT-07**); `PhDHybridSurrogate`
  returns RF ≡ 0.628079 for M = 0.98 … 5 (**CRIT-13**).
- **PVT is wrong in sign and magnitude**: `dBo/dP ≈ +1.0e-4 … 1.1e-4 1/psi` with `x_CO2 = 0`
  (**CRIT-03**), `B_g` = 0.032 × textbook at every pressure (**CRIT-04**), Z = 1.0841 … 1.2523 through the
  dense-gas region (**CRIT-05**).
- **The geomechanical safety net does not run**: the wrapper's Class-VI sandface block reads
  `profiles.get("pressure")` while the engine publishes only `yearly_pressure`/`monthly_pressure`
  (**CRIT-09**), and every leakage constraint is structurally zero (**CRIT-10**).
- **Verification tier is not evidence**: of 36 scientific tests, 5 import production symbols they never call
  and 3 assert that a defect exists (**HIGH-18**).

### Actionable Proposal (sequenced)

1. **Fix metric provenance before any physics.** CRIT-01 (HCPVI), CRIT-02 (dual-ledger NPV) and CRIT-12
   (post-hoc profile rescale) must land *before* CRIT-06/CRIT-07: the RF floors are currently the only thing
   keeping sweep output in a plausible range, so removing them first produces a large unexplained regression.
2. **Quarantine reported metrics until then.** Do not quote `recovery_factor` or `npv` from this engine in
   any report; label them `provisional — HIGH-01/CRIT-02 open`.
3. **Re-specify the verification tests against production code** (HIGH-18) *before* touching the sweep models,
   so remediation is detectable and regressions are caught.
4. **PVT campaign** (CRIT-03/04/05): restore a bubble-point model, fix the `B_g` coefficient, replace the
   linear "Hall-Yarborough" Z expression; re-validate against Standing–Katz and a laboratory PVT report.
5. **Re-enable the safety nets**: populate the `pressure` profile key (CRIT-09) and implement or delete the
   leakage constraints (CRIT-10) — then re-run this audit.
6. **Parameter provenance**: 48 of 91 registered constants are `UNKNOWN — EVIDENCE REQUIRED`
   (`audit/parameter_provenance.csv`); assign literature citations or mark them as fitted, before any of
   them is tuned.

---

## Key Metrics

| Metric | Measured value | Status |
|:---|:---|:---|
| Recovery factor (reported) | `0.500000` | **FLAGGED** — post-hoc clip of raw `0.712499`; correct cap `0.666667` (HIGH-01) |
| Recovery factor (`hybrid` vs `phd_hybrid`) | `0.500000` / `0.500000` | Identical RF, **different NPV** (CRIT-02) |
| NPV `hybrid` | **$106 601 776** | **FLAGGED** — ledger A |
| NPV `phd_hybrid` | **$60 733 386** | **FLAGGED** — ledger B, 45 % divergence |
| HCPVI (model) | `0.750000` @ 1500 / 2500 / 3500 psi | **FAILED** — pressure-independent (CRIT-01) |
| HCPVI (textbook) | `0.907564` / `0.406907` / `0.278135` | reference |
| Cumulative oil | identical across both models | consistent |
| CO₂ stored | identical across both models | consistent |
| Koval RF grid (M = 0.5 … 5) | `0.000000` except `0.316738` @ M = 1.0 | **FAILED** (CRIT-06) |
| Immiscible RF grid (27 pts) | min = max = `0.100000` | **FAILED** (CRIT-07) |
| `B_g` model / textbook | `0.032` at all P | **FAILED** — 31.7× low (CRIT-04) |
| Z-factor @ 1500/2500/3500/4500 psi | `1.0841 / 1.1402 / 1.1963 / 1.2523` | **FAILED** (CRIT-05) |
| `dBo/dP` (x_CO₂ = 0) | `+1.01e-4 … +1.13e-4 1/psi` | **FAILED** — wrong sign (CRIT-03) |
| WAG water rate vs SWAG | `25.0` vs `25000.0` bpd | **FAILED** — 1000× (CRIT-11) |
| Containment score floor vs threshold | `0.4400 … 0.7502` vs `0.30` | **FAILED** — never prunes (CRIT-08) |
| pytest (full suite) | *(baseline)* `4 failed / 329 passed / 23 skipped` → **refresh 04-10-2026: `0 failed / 333 passed / 23 skipped`** | all 4 baseline failures = one `QIcon` import (**HIGH-10, RESOLVED**) |
| pytest (`tests/scientific`) | `36 passed` | **not evidence** — 5 tautological, 3 defect-asserting (HIGH-18) |
| Coverage (line) | `37 %` (`13367 / 36062`) | `wrapper.py` and `storage.py` at `0 %` |
| Ruff | *(baseline)* `3132` findings / `192` files (F821 5, F811 5, F401 345, F841 77) → **refresh: `3244` / `191` (F821 6, F811 7, F401 329, F841 85)** | MED-13; the 2 new `F821` are **HIGH-19** |

---

## Carbon Mass Balance Reconciliation (10-year run)

| Term | MSCF | Check |
|:---|---:|:---|
| Gross injected | 36 525 000.0 | — |
| Purchased fresh | 34 994 496.3 | — |
| Recycled produced | 1 530 503.7 | purchased + recycled = injected **exactly** ✓ |
| Cumulative produced (CO₂) | 1 611 056.5 | recycled ≤ produced ✓ |
| Leakage | 0.0 | **structurally zero** — three dead paths (CRIT-10), not a measurement |
| Net stored (`inj − prod`) | 34 913 943.5 | ignores leakage by construction (`surrogate_engine.py:543`) |

**Verdict on the invariant:** $\sum M_{\text{inj}} = \sum M_{\text{purchased}} + \sum M_{\text{recycled}}$ holds
**by construction, not by enforcement**, and the second half of the closed-loop identity
(`= Net Stored + Leakage + Produced`) cannot fail because `Leakage ≡ 0`. The wiki invariant (#3/#10) has been
corrected accordingly.

---

## Geomechanical & Containment Integrity

| Check | Result |
|:---|:---|
| EPA Class VI ceiling $P_{\text{sandface}} \le 0.90\,P_{\text{frac}}$ | Enforcement code exists in the engine and `test_epa_class_vi_pressure_ceiling_enforcement` **passes**, but the wrapper-side Class-VI block is **unreachable** (CRIT-09) and two different sandface models coexist (HIGH-14). **NOT INDEPENDENTLY RE-MEASURED THIS SESSION — treated as UNVERIFIED.** |
| Plume-containment pruning | `S_cont = 0.7502` benign, `0.4400` at $P = P_{\text{frac}}$ — both above the `0.30` threshold ⇒ `prune_possible = False` always (CRIT-08) |
| Caprock / fault leakage | "breached" can be reported with zero leakage; `total_leakage_tonne ≡ 0.0` (CRIT-10, HIGH-05) |
| Injection throttling at ceiling | Present (`geomechanics_fault.py`), constants un-cited (`UNKNOWN — EVIDENCE REQUIRED`) |

---

## Relevant Files & Run Artifacts

| Artifact | Path |
|:---|:---|
| Reproducible evidence scripts (18) | [`evidence_scripts/`](evidence_scripts/) |
| PVT re-verification (`Bo`, `Bg`, Z) | [`evidence_scripts/v_pvt2.py`](evidence_scripts/v_pvt2.py) |
| Sweep/RF grids (CRIT-06, CRIT-07, HIGH-02) | [`evidence_scripts/v_rf.py`](evidence_scripts/v_rf.py) |
| HCPVI + mobile-oil term (CRIT-01, HIGH-01) | [`evidence_scripts/v_hcpvi.py`](evidence_scripts/v_hcpvi.py) |
| Full engine run, dual-ledger NPV (CRIT-02) | [`evidence_scripts/v_engine.py`](evidence_scripts/v_engine.py) |
| Clip / floor evidence (HIGH-01, HIGH-03) | [`evidence_scripts/v_clip.py`](evidence_scripts/v_clip.py) |
| Mass balance & geomechanics (CRIT-08/10) | [`evidence_scripts/v_physics.py`](evidence_scripts/v_physics.py) |
| Inert-gene evidence (CRIT-13) | [`evidence_scripts/v_inert.py`](evidence_scripts/v_inert.py) |
| Test-suite integrity scan (HIGH-18) | [`evidence_scripts/v_tests2.py`](evidence_scripts/v_tests2.py) |
| Register cross-tab derivation (52 findings, superseded) | [`evidence_scripts/v_counts2.py`](evidence_scripts/v_counts2.py) |
| Register self-count, current (**53** findings) | [`evidence_scripts/v_recount.py`](evidence_scripts/v_recount.py) |
| Master flaw register | [`../../../audit/scientific_flaws.md`](../../../audit/scientific_flaws.md) |
| Parameter provenance (91 rows) | [`../../../audit/parameter_provenance.csv`](../../../audit/parameter_provenance.csv) |
| Software/anti-pattern audit | [`../../../res_audit.md`](../../../res_audit.md) |
| Physics & math audit | [`../../../phd_audit.md`](../../../phd_audit.md) |
| Raw pytest output | [`../../../audit/runtime/pytest_output.txt`](../../../audit/runtime/pytest_output.txt) |
| Ruff report (UTF-16) | [`../../../audit/code_quality/ruff_report.json`](../../../audit/code_quality/ruff_report.json) |

> **Note on file-name collision:** `../../../audit/scientific_flaws.md` (this audit's register, **53** findings) is a
> **different file** from `agent_wiki/audit/scientific_flaws.md` (a prior-session register, 18 SCI-FLAW rows).
> Cross-check between the two is in `phd_audit.md` PART E.

### How to reproduce

```bash
.venv\Scripts\python.exe -X utf8 agent_wiki\audit\simulation_run_audits\04-10-2026_forensic_scientific_audit\evidence_scripts\v_hcpvi.py
.venv\Scripts\python.exe -X utf8 ...\evidence_scripts\v_engine.py
.venv\Scripts\python.exe -m pytest tests\scientific -q          # 36 passed
.venv\Scripts\python.exe -m pytest tests\ -q                    # refresh: 0 failed / 333 passed / 23 skipped (baseline: 4 failed / 329 passed / 23 skipped)
.venv\Scripts\python.exe -X utf8 ...\evidence_scripts\v_recount.py   # register self-count: 53 findings, 0 IDs unparsed
.venv\Scripts\ruff.exe check . --output-format json --no-cache       # refresh: 3244 diagnostics, F821 6
.venv\Scripts\pyan3.exe ...                                      # 959 nodes / 2225 edges
```

---

## Categorical Separation (no composite score is issued)

| Category | Verdict this run |
|:---|:---|
| **SOFTWARE CORRECTNESS** | Runs and fails loudly at the boundary; the 4 baseline test failures shared one root cause and are now fixed (**HIGH-10 RESOLVED**, `0 failed / 333 passed`), but two new undefined names in `ui/workbench/` are reachable and untested (**HIGH-19**); data-flow defects (dead constraints, inert genes, wrong profile keys) dominate. |
| **NUMERICAL STABILIZATION** | Pressure step limited to ±450 psi/step with **no error estimator** — stability by clipping, not by adaptivity. Distinct from physics. |
| **PHYSICAL CONSISTENCY** | **FAILED** in PVT (sign + magnitude) and in the sweep/throughput models. |
| **EMPIRICAL CALIBRATION** | **None performed.** 48/91 constants are `UNKNOWN — EVIDENCE REQUIRED`; 16 are labelled `CALIBRATED` with no stated data set. |
| **PREDICTIVE VALIDITY** | **NOT ESTABLISHED** — no experimental or benchmark evidence exists for the active `hybrid` configuration; all benchmark material targets `phd_hybrid` or dormant engines (HIGH-09: 72 vs 6 test references). |

**No single "model accuracy" number is reported**, and benchmark agreement would not be accepted as proof of
correctness in any case.

---

- **Audit-only compliance**: no scientific logic, equation, patch or "cleanup" was applied. Changes this session
  are confined to `audit/`, `res_audit.md`, `phd_audit.md` and `agent_wiki/` documentation.
- **Run ID**: `SIM-AUDIT-04-10-2026-01`
