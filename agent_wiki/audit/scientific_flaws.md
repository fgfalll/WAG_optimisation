# Scientific & Mathematical Flaw Register — CO2 EOR Optimizer

**Audit round:** 04-10-2026 (forensic audit & validation); remediations applied 05-10-2026
**Register refresh:** 05-10-2026 (round 2) — the 17 "RESOLVED" marks written earlier the same day were re-verified by adversarial measurement: **11 CONFIRMED, 4 PARTIALLY RESOLVED, 1 REGRESSED (CRIT-12), 1 CONFIRMED-BUT-INERT (CRIT-06)**. Their `Status:` lines below now state the measured verdict, not the original intention. **11 new findings registered** (CRIT-14..CRIT-21, HIGH-20..26, MED-17..22). Totals: **73 findings — 21 CRITICAL / 25 HIGH / 22 MEDIUM / 5 LOW**. Re-verify with `python -m audit.continuity check`.
**Auditor role:** Senior Reservoir Simulation Engineer / Applied Mathematician
**Register status:** PARTIALLY RESOLVED (17 remediated, 36 open/cataloged)
**Round-2 refresh:** 05-10-2026 — the 17 "RESOLVED" marks were verified by adversarial re-audit: **11 CONFIRMED, 4 PARTIALLY RESOLVED, 1 REGRESSED (CRIT-12), 1 CONFIRMED-but-inert (CRIT-06)**. **11 new findings registered (CRIT-14..CRIT-21, HIGH-20..HIGH-26, MED-17..MED-22).** See [`agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md`](../../agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md).
**Companion deliverables:** `res_audit.md` (software quality + anti-patterns), `phd_audit.md` (scientific reconstruction + CO2-EOR physics), `audit/parameter_provenance.csv`

---

## 0. How to read this register

### 0.1 Required field schema (every finding below)

| Field | Meaning |
|---|---|
| **ID** | `CRIT-nn` / `HIGH-nn` / `MED-nn` / `LOW-nn` |
| **Severity** | CRITICAL / HIGH / MEDIUM / LOW (impact on scientific conclusions, not on code elegance) |
| **Category** | `MATHEMATICAL` \| `PHYSICAL` \| `NUMERICAL` \| `SOFTWARE` \| `PROVENANCE` |
| **Location** | `file_path.py:line_number` (line numbers verified against the working tree on 04-10-2026) |
| **Observed Behavior** | What the code actually does, with measured numbers |
| **Expected Behavior** | What dimensional analysis / thermodynamics / the cited literature requires |
| **Scientific Impact** | Effect on recovery factor, mass balance, containment, economics, or the optimizer's selection pressure |
| **Evidence & Citation** | Reproduction command, measurement, literature reference |
| **Status** | `NEW` (this round) / `CONFIRMED` (previously reported, re-verified) / `OPEN` / `STALE` (previous claim disproved) / `RESOLVED` |

### 0.1a Register totals (recomputed from this file, 05-10-2026)

A parser over this file (heading form `### ID — title` **and** table form `**ID** |`) counts
**73 distinct findings: 21 CRITICAL / 25 HIGH / 22 MEDIUM / 5 LOW**, with no duplicate IDs.

This supersedes the round-1 figure of 53 (13/19/16/5). The increase is entirely the round-2 block
(§2b) plus MED-17…MED-22. The §0.2 category counts below were produced against the round-1 subset
and are **not** restated for the full 73; treat them as historical.

### 0.2 Mandatory categorical separation (no composite score is produced)

This audit deliberately **does not** compute a single "model accuracy" number. Findings are bucketed as follows, because they have different remedies, different owners, and different evidentiary standards:

| Bucket (= Category field) | Definition | Count in this register |
|---|---|---:|
| **MATHEMATICAL CORRECTNESS** (`MATHEMATICAL`) | Wrong formula, wrong normalization, wrong units, non-existent limit, discontinuity | 8 |
| **PHYSICAL CONSISTENCY** (`PHYSICAL`) | Violation of thermodynamics, conservation, or displacement physics in equations that *are* executed | 9 |
| **NUMERICAL STABILIZATION** (`NUMERICAL`) | Clipping, flooring, epsilons, fallbacks used to keep numbers finite — legitimate technique **only** when it does not replace the model | 3 |
| **SOFTWARE CORRECTNESS** (`SOFTWARE`) | Data flow, dead code, key mismatches, masking, inert genes, test gaps — correctness of the program, independent of physics | 25 |
| **PARAMETER / EMPIRICAL PROVENANCE** (`PROVENANCE`) | Constants/coefficients with no cited source, mis-cited correlations, undocumented units | 8 |
| **PREDICTIVE VALIDITY** | Whether outputs can be trusted to forecast a real field (requires benchmark/experimental evidence — **none found in-repo for the active `hybrid` path**) | reported as a **verdict** in `phd_audit.md` §6, not as a score |

Category counts above and in §7 are produced by one parser over this file (`evidence_scripts/v_recount.py`, re-run 04-10-2026): **53 findings — 13 CRITICAL / 19 HIGH / 16 MEDIUM / 5 LOW**. (The pre-refresh copy of this table said PHYSICAL 10 / SOFTWARE 23 / PROVENANCE 7 = 51; that was a transcription error against §7 and has been corrected.)

### 0.3 Hierarchy of scientific truth applied

Findings are graded with this precedence: **Mathematical correctness > Physical consistency > Numerical discretization > Implementation > Parameter provenance > Analytical verification > Experimental validation > Benchmark agreement > Execution speed.**

Corollary explicitly enforced in this audit: **agreement with a benchmark is never credited as proof of correctness**, and a numerically stable output is never credited as a physically correct output.

### 0.4 Numeric evidence

All measured numbers below are reproduced by scripts kept **outside** the repository (audit-only rule) and mirrored into the run-audit folder (invariant #6):

```
.venv\Scripts\python.exe C:\Users\sayno\AppData\Local\Temp\opencode\audit_verify_3.py
.venv\Scripts\python.exe C:\Users\sayno\AppData\Local\Temp\opencode\audit_verify_4.py
.venv\Scripts\python.exe C:\Users\sayno\AppData\Local\Temp\opencode\v_recount.py      # register self-count (§0.2 / §7)
```

Mirror: `agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/evidence_scripts/` (18 scripts).

---

## 1. CRITICAL findings

### CRIT-01 — HCPVI is pressure-independent and dimensionally `MSCF/STB`

- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:491-497` (overwrites `params["hcpvi"]` set at `core/engine_surrogate/surrogate_engine.py:890`)
- **Observed:** 
  ```python
  cum_inj_rb = cum_inj_mscf * b_co2_mean                       # :494
  simulated_hcpvi = cum_inj_rb / (ooip_val * b_co2_mean / (1 - swi_val))   # :497
  ```
  `b_co2_mean` cancels algebraically, leaving
  `simulated_hcpvi = cum_inj_mscf * (1 - swi) / ooip` — a ratio of **MSCF to STB**, not reservoir-pore-volume to hydrocarbon-pore-volume, and containing **no pressure term at all**.
  Measured: 1 MMSCF injected into 1 MMSTB (swi = 0.25) returns **0.750000 at 1500, 2500 and 3500 psi identically**.
- **Expected:** `HCPVI = cum_inj_mscf · Bg(P) · (1 - Swi) / (OOIP · Bo(P))`, i.e. injected reservoir barrels divided by hydrocarbon pore volume, which is strongly pressure-dependent through `Bg/Bo`.
- **Scientific Impact:** Throughput is the independent variable of the Koval/Ekladios sweep and of every miscible RF correlation in the module. Measured error versus the textbook definition for the same case: **0.667× (1500 psi), 1.218× (2500 psi), 1.868× (3500 psi)** — a monotonically growing, pressure-driven bias that silently rewards high-pressure operation. It also discards the correct `params["hcpvi"]` computed at `:890`, so the code computes the right quantity once and then throws it away.
- **Evidence & Citation:** `audit_verify_3.py` section V1. Dimensional check: `[RB]/([STB]·[RB/STB])` in the denominator vs `[RB]` in the numerator ⇒ dimensionless only if the numerator were reservoir barrels of **oil-equivalent**; as written both `b_co2` factors cancel exactly.
- **Status:** PARTIALLY_RESOLVED
- **Note:** units corrected (dimensionless, pressure-dependent) but the DENOMINATOR is total pore volume: `pv_mean_rb = (ooip_val*bo_mean)/(1.0-swi_val)` reduces to `V_p`, not the hydrocarbon `V_p*(1-Swi)`. Measured 05-10-2026: returns 7.6927 where textbook HCPVI is 10.2569 — off by exactly `1-Swi = 0.75`, so the error is Swi-dependent and two runs with different Swi compare different definitions. The formula quoted in the remediation note (`Q*Bg/(V_p*S_oi*B_oi)`) is itself dimensionally wrong: pore volume is already at reservoir conditions and must not carry a `B_o`. Verdict `PARTIAL` by `python -m audit.continuity check CRIT-01`. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/17

---

### CRIT-02 — Reported NPV and reported recovery factor come from two different evaluations

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:165`, `:171-174`, `:490-510`, `:533`, `:685`, `:690`, `:692`; `core/engine_surrogate/surrogate_models.py:465`, `:473`, `:507-530`
- **Observed:** 
  1. `prediction = self.surrogate_model.predict(params)` (`:165`) evaluates RF₁ using **target pressure** (`params["pressure"] = target_pressure_psi`, `:161-162`) and `params["hcpvi"]` from `:890`.
  2. `npv` (`:172`), `cumulative_oil` (`:173`) and the CO₂ ledger inside NPV are all derived from RF₁ (`surrogate_models.py:465`, `:473`, `:493`).
  3. Later, RF₂ is evaluated a **second time** at the whole-profile **mean pressure** with `simulated_hcpvi` (`:490`, `:501`, `:505`) and clipped (`:510`).
  4. The returned object mixes the two: `recovery_factor` = RF₂ (`:690`), `cumulative_oil` = oil profile rescaled to RF₂ (`:519`, `:533`, `:685`), `npv` = f(RF₁) (`:692`), `co2_stored` = injected − produced, whereas NPV's cost term used the breakthrough-aware `co2_stored` of RF₁ (`:174` vs `:544`, `:733`).
- **Expected:** One physical evaluation per simulation: one RF, one oil profile, one CO₂ ledger, one cash flow, all consistent at the same pressure and throughput.
- **Scientific Impact:** The primary objective (`npv`) is not a function of the recovery factor the run reports. Ranking candidates by NPV therefore ranks them by an internally inconsistent quantity; every downstream artifact (cash-flow table, storage efficiency, RF-vs-NPV cross plots) mixes two states. Two independent CO₂ mass ledgers exist simultaneously (breakthrough-aware vs injected−produced) — a direct threat to invariant *"cumulative recycled ≤ produced ≤ injected"* auditing.
- **Evidence & Citation:** Code reading of the two evaluation sites; `audit_verify_4.py` section D shows RF is strongly hcpvi-dependent (0.05 → 0.713 over hcpvi 0 → 3), so the hcpvi substitution between the two calls is not a rounding difference.
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/surrogate_engine.py:598-630: NPV derived directly from annual simulated production and injection profiles consistent with RF.

---

### CRIT-03 — No bubble-point model: `Bo` increases with pressure (apparent negative compressibility)

- **Severity:** CRITICAL
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:232-244` (`R_s`), `:267-307` (swelling / `B_o`); used at `core/engine_surrogate/surrogate_engine.py:362`, `:369`, `:376`, `:453`
- **Observed:** `bubble_point` occurs **0 times** in `core/engine_surrogate/` (grep, 04-10-2026). `R_s = c1·γg·p^c2·exp(c3·API/T_R)` is monotone increasing in `p` for every pressure, so `B_o` is monotone increasing in `p` for every pressure.
  Measured apparent isothermal compressibility `c_o = (1/B_o)·dB_o/dP` (x_CO₂ = 0): **+9.23e-5 (2000 psi), +9.62e-5 (2500), +9.86e-5 (3000), +8.74e-5 (3500), +8.79e-5 (4000) 1/psi**; with x_CO₂ = 0.3: **+1.19e-4 → +9.33e-5 1/psi**. `B_o` rises 1.1594 → 1.4535 rb/STB from 1500 → 4000 psi.
- **Expected:** Above the bubble point `dB_o/dP < 0`, with `c_o ≈ 3e-6 … 1.2e-5 1/psi` for a black oil ( McCain, W.D. (1990), *The Properties of Petroleum Fluids* ). Below the bubble point `R_s` is constant at `R_sb`. A positive `c_o` of order 1e-4 means the oil "expands when compressed".
- **Scientific Impact:** (a) Saturation bookkeeping `S_o = remaining_oil·B_o/V_p` (`surrogate_engine.py:376`) inflates oil saturation as pressure rises, so gas/water saturations are displaced. (b) The material-balance denominator uses a **constant** `c_o = 1e-5` (`surrogate_engine.py:315`) that contradicts the PVT module's own derivative by an order of magnitude — the compressibility in the ODE and the compressibility implied by `B_o(P)` are two different fluids. (c) Voidage `q_prod·B_o` (`:412-414`) grows with pressure, feeding back into the pressure ODE (`:455`).
- **Evidence & Citation:** `audit_verify_3.py` section V4; zero-occurrence grep for `bubble_point`. Prior-audit SCI-FLAW-02 reported the same sign defect in `core/data_integration_engine.py` (a different module); this finding shows the **active** PVT path has the identical class of defect.
- **Status:** STILL_OPEN
- **Note:** (status claim reversed 05-10-2026) — `dB_o/dP` sign IS fixed and measured (-1.20e-05 1/psi above P_b vs +9.73e-05 below), but **there is no Standing bubble-point correlation in the repository**. `pvt_state.py:104` reads `else min(self.p_init, 2800.0)`, a literal. Repo-wide greps for `Rsb`, `pb_standing`, `18.2`, `0.0837` all return 0 hits. Measured: every reservoir at or above 2800 psi initial pressure gets the SAME P_b regardless of API gravity, gas gravity, temperature or R_sb — a 35 API and a 15 API oil are modelled identically. P_b is a measured PVT quantity and must be derived (Standing 1947: `P_b = 18.2[R_sb*sqrt(gamma_g/gamma_o)+1.4]^(1/1.5)`) or flagged `UNKNOWN — EVIDENCE REQUIRED`, never set to a round number. Verdict `FAIL`. See also HIGH-21. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/18

---

### CRIT-04 — Hydrocarbon-gas `B_g` coefficient is 31.73× too small (unit conversion inverted)

- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:372-373` (mixing at `:385`; consumed at `core/engine_surrogate/surrogate_engine.py:366`, `:414`, `:453`)
- **Observed:** 
  ```python
  # Bg in RB/MSCF: Bg = 0.02827 * Z * T_R / P * 5.6146      # comment :372
  bg_hc = 0.1587 * z_hc * self.temp_r / max(p, 1e-4)          # code    :373
  ```
  Correct: `B_g[RB/MSCF] = 0.02827 · Z · T_R / P [ft³/scf] × 1000 scf/MSCF ÷ 5.6146 ft³/bbl = 5.035 · Z · T_R / P`.
  The code **multiplies** by 5.6146 instead of dividing **and** drops the factor 1000 ⇒ `0.1587` vs `5.035` = **31.73× low** (0.02827·5.6146 = 0.15873).
  Measured end-to-end (`calculate_mixture_gas_properties`, 2500 psi, 150 °F) versus textbook with Z = 0.85:
  | y_CO₂ | model B_g (rb/MSCF) | textbook (rb/MSCF) | ratio |
  |---|---|---|---|
  | 0.00 | 0.04439 | 1.0437 | **23.5×** |
  | 0.50 | 0.08223 | 0.9393 | **11.4×** |
  | 1.00 | 0.55743 | 0.8350 | 1.5× (CO₂ path uses PR density, not this formula) |
- **Expected:** `B_g = 5.035·Z·T_R/P` rb/MSCF (Standing 1977 / any field-unit textbook). Pure-CO₂ FVF must come from the same law or from `ρ_CO₂`.
- **Scientific Impact:** Produced-gas voidage `q_prod += gas·B_g` (`surrogate_engine.py:412-414`) is understated 11–24×, so (i) VRR (`:417`) is overstated, (ii) the pressure decline `dp` (`:455`) is far too small, (iii) `c_g` (`:367`, `:453`) and `B_g` enter material balance as if produced gas occupied almost no reservoir volume. Gas-cap/voidage physics is effectively switched off for hydrocarbon gas.
- **Evidence & Citation:** `audit_verify_3.py` sections V2/V2b; arithmetic identity `0.02827×5.6146 = 0.15873`.
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/pvt_state.py:377: Corrected Standing (1977) natural gas FVF conversion constant to 5.035 * Z * T_R / P.

---

### CRIT-05 — "Hall-Yarborough" Z-factor is a linear expression returning Z > 1 in the dense-gas region

- **Severity:** CRITICAL
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:361-367`
- **Observed:** 
  ```python
  z_hc = 1.0 + (0.06422*t_inv - 0.00332*t_inv**2) * Ppr   # :366, clipped to [0.65, 1.4] :367
  ```
  With engine defaults (T_R = 609.67 °R, γg = 0.70, Tpr = 1.581) measured: **Z = 1.0882 (1500 psi, Ppr 2.244), 1.1469 (2500, Ppr 3.740), 1.2057 (3500, Ppr 5.236), 1.2645 (4500, Ppr 6.731)**. The expression is linear and *monotonically increasing* in pressure.
- **Expected:** Hall-Yarborough (1972) is an iterative tangent-compressibility correlation; the Standing–Katz chart at Tpr ≈ 1.58 gives Z ≈ 0.83–0.95 over Ppr = 2–5, i.e. **Z < 1 with a minimum around Ppr ≈ 2–3**, never a monotonic rise above 1. A gas with Z = 1.15 at Ppr = 3.7 does not exist at that reduced state.
- **Scientific Impact:** `ρ_hc` (`:371`) is computed ~20–35% too low and the *sign of dZ/dP* is wrong, so `c_g = 1/P − (1/Z)(dZ/dP)` (`:388`) must be fudged by hand (`:391-394`, see HIGH-18) — the two defects mask each other, which is exactly why numerical agreement in downstream numbers cannot be credited as validation.
- **Evidence & Citation:** `audit_verify_4.py` section A; Standing, M.B. (1977); Hall, K.E. & Yarborough, L. (1972) *J. Pet. Tech.* — the code's own label does not match the code's formula.
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/pvt_state.py:367: Implemented Papay (1968) natural gas Z-factor correlation and analytical dZ/dP with correct dense-gas behavior.

---

### CRIT-06 — Koval sweep returns RF = 0 for 0.5 ≤ M ≤ 1.4 with a spike at exactly M = 1.0, plus `exp` overflow for M < 1

- **Severity:** CRITICAL
- **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:541-559`
- **Observed:** 
  ```python
  c = 1.0/(M-1.0); term1 = (1-exp(1-kv))/(kv-1); term2 = (1-exp(c*(1-kv)))/(c*(kv-1))
  sweep = term1 - (term1 - term2)/(M - 1.0)      # :550-556
  return float(np.clip(sweep, 0.0, 0.75))         # :559
  ```
  Measured (`v_dp = 0.5`, `hcpvi = 1.8`, P = 3000 psi, MMP = 2500 psi):
  | M | 0.5 | 0.9 | **1.0** | 1.1 | 1.3 | 1.4 | 1.5 | 2.0 | 3.0 | 5.0 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | RF | 0.000000 | 0.000000 | **0.316738** | 0.000000 | 0.000000 | 0.000000 | 0.009621 | 0.263024 | 0.324123 | 0.289557 |

  (a) For M < 1, `c < 0` and `kv > 1` ⇒ `exp(c·(1−kv))` = `exp(positive large)` ⇒ overflow → −∞ → clipped to 0. (b) For 1 < M ≤ 1.4 the nested `(term1 − term2)/(M−1)` suffers catastrophic cancellation/overflow → large negative → clipped to 0. (c) The M = 1.0 branch (`:542-547`) is a *different* formula, producing an isolated spike of 0.3167 surrounded by zeros.
- **Expected:** Koval (1963) breakthrough sweep `V_b` is continuous and strictly decreasing in M over (0, ∞), with `V_b → 1` as M → 0. No region of identically zero recovery may exist for finite throughput; the M = 1 limit must be the continuous limit of the general branch.
- **Scientific Impact:** In the M range typical of CO₂–crude systems the model reports **zero recovery**; because the optimizer explores M (it is a relaxable constraint, `optimisation_engine.py:127`), the objective surface contains a large flat zero plateau with a 0.3167 delta-function at M = 1.0 — a guaranteed trap for both GA (selection pressure destroyed) and BO/gradient methods.
- **Evidence & Citation:** `audit_verify_3.py` section V5 (RuntimeWarning on overflow observed); Koval, E.J. (1963) *SPE J.* 3(2), 145–152.
- **Status:** CONFIRMED_BUT_INERT
- **Note:** (status claim qualified 05-10-2026) — the fix is genuine: Koval is now continuous and strictly monotone in M (measured max jump 7.5e-03 across M=1; the old RF=0 plateau for 0.5<=M<=1.4 and the M=1 delta-function spike of 0.3167 are both gone). BUT at the shipped configuration the sweep is saturated at its 0.95 clip: measured sweeps at the default HCPVI=7.6928 are [0.95, 0.95, 0.95, 0.95] for M=1,2,5,10, and the clip is reached at HCPVI~6.0. Mobility ratio therefore cannot influence sweep in practice. Root cause is shared with CRIT-01/CRIT-15: `hcpvi` is built from `injection_rate*365.25*lifetime`, ignoring the WAG schedule, shut-ins, availability and compressor cap, so it is not the throughput actually injected. Verdict `PASS-BUT-INERT`. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/19

---

### CRIT-07 — Immiscible model is the constant RF = 0.10 over the whole parameter space

- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:319-326`
- **Observed:** 
  ```python
  recovery = displacement_eff * areal_eff * vertical_eff    # :321
  rf_max_physical = max(0.0, soi - sor); max_cap = min(0.50, rf_max_physical) if rf_max_physical > 0.10 else 0.50
  return float(np.clip(recovery, 0.10, max_cap))            # :326
  ```
  Measured over a 5 × 3 × 3 grid (M ∈ {0.5, 1, 2, 5, 20}, v_dp ∈ {0, 0.5, 0.9}, S_or ∈ {0.1, 0.4, 0.7}): **the set of distinct returned values is `{0.1}`**.
  Decomposition of the pre-clip product: `E_d = 0.041…0.140`, `E_A ≤ 0.517`, `E_V ≤ 1.0` ⇒ product **0.0140 … 0.0563**, always below the 0.10 floor.
- **Expected:** Immiscible CO₂ flooding recovers 10–40% OOIP depending on `M`, `V_DP`, `S_or` (Buckley & Leverett 1942; Craig 1971). The returned value must be a *function* of the inputs.
- **Scientific Impact:** The immiscible limb carries **zero gradient and zero sensitivity**. Because `HybridSurrogate` blends this constant with the miscible limb (`analytical_models.py:468-472`, weights `w_miscible`), the entire hybrid model's response to `S_or`, `V_DP` and `M` through the immiscible branch is null; and the floor itself fabricates 10% recovery when the physics computes ~1.4–5.6%.
- **Evidence & Citation:** `audit_verify_3.py` section V6; internal decomposition in `audit_verify_4.py` section C.
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/analytical_models.py:296-326: Corrected Buckley-Leverett Welge shock-front tangent construction and removed unphysical 0.10 constant floor.

---

### CRIT-08 — Plume-containment constraint can never prune: floor of the score exceeds the threshold

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/optimisation_engine.py:1645-1666` (prune) with `core/objectives/storage.py:86-102` (score) and `core/data_models.py:1725-1730` (weights/threshold)
- **Observed:** 
  ```python
  s_cont = γ·(w_p·S_press + w_s·S_seal + w_t·S_struct)      # storage.py:100
  ```
  `storage_params` here is `AdvancedEngineParams`, which **has no** `reservoir_seal_integrity_factor` and **no** `structural_trapping_factor` field (fields at `data_models.py:1690-1739`) ⇒ `getattr` silently falls back to the hard-coded 0.9 and 0.85. Therefore
  `min S_cont = 0.3·0.9 + 0.2·0.85 = 0.44` (with `S_press = 0`) while `containment_critical_threshold = 0.3`.
  Measured: **S_cont = 0.4400** when the pressure array saturates the fracture limit, **0.9400** when the array is empty; `prune_possible = False` in both cases.
  Additionally the key requested is `f"{time_res}_pressure"` (`optimisation_engine.py:1652-1653`), which exists only for `yearly` (`:853`) and `monthly` (`:870`); `time_resolution` also legally accepts `weekly` and `quarterly` (`data_models.py:1273`) ⇒ empty array ⇒ `S_press = 1.0` ⇒ score 0.94, i.e. **the score is decided by key naming, not by physics**.
- **Expected:** A critical-threshold constraint must be reachable: `min(S_cont) < threshold` for the configured weights, and the seal/structure factors must come from the user's `CO2StorageParameters` (`data_models.py:1620`, `:1628`), not from unlinked defaults.
- **Scientific Impact:** The Class-VI-style containment guard advertised in `AdvancedEngineParams` is a **no-op**. Optimizer candidates are never rejected for containment reasons; the only visible variation in `S_cont` is an artifact of time-resolution key selection.
- **Evidence & Citation:** `audit_verify_3.py` section V11; `data_models.py:1725-1730`; `storage.py:91-92` (`getattr(..., 0.9/0.85)` fallbacks).
- **Status:** RESOLVED
- **Note:** Remediated in core/data_models.py:1815-1820, core/objectives/storage.py:86-115, and core/optimisation_engine.py:1657-1685: Added containment factors to AdvancedEngineParams and cascading pressure profile retrieval.

---

### CRIT-09 — Wrapper's sandface / Class-VI penalty block is unreachable (wrong profile key)

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/objectives/wrapper.py:61-78`; key producer `core/optimisation_engine.py:841-896`
- **Observed:** `pressure_profile = profiles.get("pressure", profiles.get("reservoir_pressure"))`. The `profiles` dict handed to `_calculate_objective_functions` (`optimisation_engine.py:944-946`) is built at `:841-896` and contains only `yearly_pressure` / `annual_pressure` (`:853-854`) and `monthly_pressure` (`:870`) — never `pressure` or `reservoir_pressure`. Result: `pressure_profile is None` ⇒ the whole `if` block is skipped ⇒ `results["npv"]` never receives the containment penalty and `geomechanical_violation` is never emitted.
  (The engine *does* return `pressure` and `sandface_injection_pressure`, `surrogate_engine.py:663-665`, but those keys are never forwarded into this dict.)
- **Expected:** Either read `yearly_/monthly_pressure`, or forward the engine's `pressure` / `sandface_injection_pressure` arrays.
- **Scientific Impact:** EPA Class-VI 90% fracture-limit enforcement inside the objective (`safe_fracture_limit = 0.90·P_frac`) never runs on the optimization path; the code and the wiki both imply it does.
- **Evidence & Citation:** Full key inventory `optimisation_engine.py:841-896`; single call site `_calculate_objective_functions` at `:944`.
- **Status:** RESOLVED
- **Note:** Remediated in core/objectives/wrapper.py:61-79 and core/optimisation_engine.py:853-860: Forwarded pressure and sandface injection profiles into objective evaluation.

---

### CRIT-10 — Every CO₂ leakage constraint is structurally zero (three dead paths)

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:**
  1. `core/optimisation_engine.py:1367-1381` (profile-constraint "Check 3: Leakage")
  2. `core/objectives/wrapper.py:81-99` (environmental leakage penalty)
  3. `core/engine_surrogate/surrogate_engine.py:678`, `:772-774` (leakage outputs never consumed)
- **Observed:** 
  - `annual_leakage_tonne` is produced **only** by `analysis/material_balance.py:280` (post-processing module), never by the surrogate engine nor by `evaluate_for_analysis` (`grep`, 04-10-2026). `max_sandface_pressure_psi` is produced **nowhere** in the repository (0 hits). Both keys are read at `optimisation_engine.py:1367-1373`.
  - ⇒ `p_sandface = eval_results.get("max_sandface_pressure_psi", 0.0)` = `0.0` ⇒ `0.0 > p_seal` is false ⇒ `leakage_tonne = 0.0` ⇒ `leakage_fraction = 0.0` ⇒ the leakage branch of the constraint can never fire.
  - Wrapper: `profiles` has no `total_leakage_tonne`, no `annual_leakage_tonne`, no `leakage_rate_fraction` ⇒ `leaked_tonnes = 0.0` (`wrapper.py:93`) ⇒ no remediation cost, ever.
  - The engine's own `total_leakage_tonne` (`:678`), `annual_fault_leakage_tonne` and `annual_caprock_leakage_tonne` (`:773-774`) are returned but **never read by any caller** (grep: 0 consumers) — and `annual_caprock_leakage_tonne` / `annual_fault_leakage_tonne` are pre-allocated zero arrays that are never populated (no assignments anywhere), while `cum_stored = inj − prod` (`:543`) ignores leakage entirely.
- **Expected:** Leakage produced by the geomechanical module must flow into (a) the constraint penalty, (b) the NPV remediation cost, (c) the reported storage balance `M_stored = M_inj − M_prod − M_leaked`.
- **Scientific Impact:** CO₂ containment violations have **no economic or selection-pressure consequence** anywhere in the pipeline. Wiki invariant #3/#10 ("mass conservation / leakage accounted") is violated in the implementation, not merely in wording.
- **Evidence & Citation:** Two repo-wide greps for `annual_leakage_tonne` and `max_sandface_pressure_psi`; consumer grep for `total_leakage_tonne`.
- **Status:** RESOLVED
- **Note:** Remediated in core/objectives/wrapper.py:81-99, core/optimisation_engine.py:853-860, and core/engine_surrogate/surrogate_engine.py: Forwarded leakage keys into constraint and objective workflows.

---

### CRIT-11 — WAG water injection is ~1000× too low (missing unit conversion), so WAG degenerates to continuous gas

- **Severity:** CRITICAL
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/profile_generator_fast.py:1109`, `:1117-1118` versus `:1210`; parameter `core/data_models.py:917`
- **Observed:** 
  ```python
  co2_inj_rb_per_day = base_injection_rate * default_b_gas                 # :1117  (WAG, no ×1000)
  enhanced_water_rate_bpd = co2_inj_rb_per_day * enhanced_wag_ratio        # :1118
  water_rate_bpd = base_injection_rate * wgr * params.get("default_gas_fvf", 0.005) * 1000   # :1210 (SWAG)
  ```
  `default_gas_fvf = 0.005` is only meaningful as rb/scf (0.005 rb/scf = 5 rb/MSCF). SWAG applies the `×1000` scf→MSCF conversion; WAG does not.
  Measured consequence for `base = 5000 MSCFD`, `WAG ratio = 1`: WAG water = **25 bpd**, SWAG water = **25 000 bpd** for the same gas rate. For 1000 MSCFD WAG water = 5 bpd.
- **Expected:** WAG water rate must equal `q_gas[MSCFD] · B_g[rb/MSCF] · WAG_ratio`, i.e. tens of thousands of bpd at field rates (and the same `default_gas_fvf` must be used consistently in both schemes).
- **Scientific Impact:** With 5–25 bpd of water against 5000 MSCFD of CO₂, the WAG scheme produces essentially **continuous gas injection**: no mobility control, no mobility banking, no deferred gas. `cum_water_inj_bbl` (`surrogate_engine.py:474`) feeds `S_w` (`:377-378`) and `q_inj_step_rb` (`:406`) ⇒ VRR and saturation paths are computed for a reservoir that is receiving no water. The optimizer's WAG-ratio gene is therefore optimizing a scheme that does not physically exist.
- **Evidence & Citation:** `audit_verify_3.py` reading of `:1117` vs `:1210`; both use the same `default_gas_fvf` default (0.005), so the only difference is the omitted `×1000`.
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/profile_generator_fast.py:1109-1118: Added x1000 unit scaling for WAG gas-to-water injection conversion.

---

### CRIT-12 — Production/saturation/pressure state is computed from an unscaled profile that is rescaled afterwards

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:376-379`, `:412-417`, `:455`, versus post-hoc rescale at `:513-522`
- **Observed:** Inside the time loop the engine integrates `nominal_oil_stb` (`:408`, `:472`) into `cum_oil_stb`, from which `S_o` (`:376`), `q_prod_step_rb` (`:412-414`), `VRR` (`:417`) and `dp` (`:455`) are all computed. **After** the loop, `profile_result["oil_profile"]` is multiplied by `target_cum_oil/max_cum_shape` (`:519`) so that cumulative oil matches RF₂. The saturated/pressured history is never recomputed.
- **Expected:** Either the profile is correct before integration (then no rescale is needed), or the entire state trajectory is recomputed after rescaling. A profile may not be rescaled after the state variables that depend on it have been consumed.
- **Scientific Impact:** Reported `S_o/S_w/S_g` profiles, `B_o/B_g` profiles, VRR and the pressure path are mutually inconsistent with the reported oil rate and cumulative oil. Any material-balance cross-check of the reported streams (injected vs produced vs stored) will fail or, if it passes, passes only because `cum_oil_total` is recomputed from the rescaled array at `:533` while saturations are not.
- **Evidence & Citation:** Prior-audit SCI-FLAW-06 (`CONFIRMED_AUDIT_OPEN`) — this round re-verified the exact ordering and added the specific dependent variables.
- **Status:** REGRESSED
- **Note:** (status claim reversed 05-10-2026) — the remediation for this finding INTRODUCED a new critical defect. `surrogate_engine.py:539` sets `pv_ref = (ooip_val*bo_mean)/(1.0-swi_val)` (hydrocarbon PV) and then uses that same denominator for BOTH `S_o` and `S_w`, while `S_wi` is defined on TOTAL pore volume. Measured on the default 181-step run: `S_o+S_w>1` on **49/181 steps (27.1%)**, `max(S_o+S_w)=1.042829`, and `S_o+S_w+S_g` ranges over [1.000000, 1.042829] — the saturation sum exceeds unity. `S_g=0` on 49 steps only because `np.clip` saturates, so the violation is invisible. See CRIT-17. Verdict `REGRESSED`. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/9

---

### CRIT-13 — Three optimizer genes are inert (`gravity_factor`, `transition_alpha`, `transition_beta`) and a fourth is RF-inert (`mobility_ratio`)

- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/optimisation_engine.py:120-142`, `:1420-1431`; `core/engine_surrogate/analytical_models.py:449`, `:458-459`; `core/engine_surrogate/surrogate_engine.py:865`, `:896`; `core/simulation/recovery_models.py:329`
- **Observed:** 
  - **`gravity_factor`** — 21 occurrences repo-wide, all in bounds/validation/metadata/sensitivity scaffolding (`data_models.py:941-942`, `:974`, `:1032-1033`, `:1073-1074`, `:1699`; `optimisation_engine.py:129`, `:1422-1424`, `:4207`; `sensitivity_analyzer.py:143`, `:639`; tests). **Zero physics consumers.** Optimizing it changes nothing.
  - **`transition_alpha` / `transition_beta`** — sole physics consumer is `core/simulation/recovery_models.py:329`, which is disabled because `RECOVERY_MODELS_AVAILABLE = False` (`analytical_models.py:30`). The active `HybridSurrogate` hard-codes `alpha = 0.95 + 0.05·(c7_plus − 0.3)`, `beta = 20.0` (`analytical_models.py:458-459`). Worse, it reads `params["c7_plus_fraction"]` (`:449`) but the engine writes `params["c7_plus"]` (`surrogate_engine.py:865`) ⇒ `c7_plus ≡ 0.3` ⇒ **`alpha ≡ 0.95` always**.
  - **`mobility_ratio`** — `HybridSurrogate` never reads it; measured RF is **identical 0.628079 for M = 0.98, 1.0, 1.01, 2, 3, 5**. `MiscibleSurrogate` does not read it either. It *does* reach `breakthrough_time` (`surrogate_engine.py:896`), so it perturbs NPV/storage but not recovery. `ImmiscibleSurrogate`/`BuckleyLeverettSurrogate` recompute M internally from `viscosity_oil/viscosity_inj` defaults (`analytical_models.py:270`, `:395`), ignoring the gene.
- **Expected:** Every gene exposed to GA/BO must have a non-zero gradient in at least one reported objective; transition parameters must reach the model that is actually executed; parameter names passed between modules must match.
- **Scientific Impact:** 4 of 9 relaxable constraints (`data_models.py:1692-1704`) cannot influence recovery. The optimizer spends evaluations exploring null directions; sensitivity/tornado reports for these parameters (`sensitivity_analyzer.py:143`, `:639`) will show pure noise, and any conclusion drawn from them is invalid.
- **Evidence & Citation:** Repo-wide greps for `gravity_factor` / `transition_alpha`; `audit_verify_3.py` section V7; name mismatch `c7_plus` vs `c7_plus_fraction` verified by grep.
- **Status:** PARTIALLY_RESOLVED
- **Note:** , and partly a net regression (status claim qualified 05-10-2026) — `transition_alpha`/`transition_beta` are wired, but `alpha_base=0.9750` is always supplied so the `c7_plus` branch is dead and the miscibility weight is now EXACTLY independent of composition (measured omega identical to 6 dp for C7+ = 0.1/0.2/0.3/0.5; legacy C7+-driven alpha varied). See CRIT-20. Worse, `gravity_factor` — dimensionless, no unit, no citation, range [0.5,1.5] — now multiplies recovery in THREE places (`e_v` at :200, `vertical_eff` at :332, `N_g` at :802), twice within the hybrid path. This finding originally recorded the gene as INERT; making it active converts visible dead code into invisible fitted fudge. See CRIT-19. Verdict `FAIL`. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/20

---

## 2. HIGH findings

### HIGH-01 — `rf_max_physical` uses pore-volume fraction instead of OOIP-normalized fraction (25% too restrictive)

- **Severity:** HIGH
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:509`; `core/engine_surrogate/analytical_models.py:814`
- **Observed:** `rf_max_physical = max(0.0, 1 - swi - sor)`. For Swi = 0.25, S_or = 0.30 ⇒ cap = **0.4500**.
- **Expected:** OOIP = V_p·(1−Swi)/B_o, so the recoverable fraction of **OOIP** is `(1−Swi−S_or)/(1−Swi)` = **0.6000** (Craft & Hawkins, *Applied Petroleum Reservoir Engineering*). The code itself uses the correct form 100 lines earlier at `analytical_models.py:801` (`e_d = (soi − sor)/soi`) and in `surrogate_models.py:234`.
- **Scientific Impact:** RF clipped 25% too low (measured under-estimate 25.0%); produces a flat artificial plateau at the cap, hiding all sensitivity above it. Applies on every engine evaluation (`:509-510`).
- **Evidence & Citation:** `audit_verify_3.py` section V9; internal contradiction `:801` vs `:814`. Prior SCI-FLAW-11 content **CONFIRMED**, but its line references (`analytical_models.py:881`, `surrogate_engine.py:425`) are **STALE** (now `:814` / `:509`).
- **Status:** RESOLVED
- **Note:** Remediated in core/engine_surrogate/analytical_models.py and surrogate_engine.py:509: rf_max_physical normalized to OOIP (1 - Swi - Sor)/(1 - Swi).

### HIGH-02 — Craig areal-sweep 48% discontinuity at M = 1.0, and a test that asserts the defect

- **Severity:** HIGH
- **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:304-312`; `core/engine_surrogate/surrogate_models.py:164-182`; test `tests/scientific/mathematical/test_singularity_and_overflow.py:64-73`
- **Observed:** `E_A = 1.0` for `M ≤ 1`, `E_A = 0.517 − 0.072·log10(M)` for `M > 1`. Measured: M = 1.0 → 1.000000; M = 1.0001 → **0.516997**; M = 1.01 → 0.516689. **Step = 0.4833 (48.3% drop).**
- **Expected:** Craig (1971) five-spot correlations are fitted only for M > 1 and must be blended continuously into the favorable-M branch (a C⁰ match, e.g. one correlation normalized to 1.0 at M = 1).
- **Scientific Impact:** Discontinuous objective surface exactly in the physically interesting M ≈ 1 region (CO₂–crude near-miscible conditions). Worse, `test_singularity_and_overflow.py:73` **asserts `step > 0.40`** with the comment *"This test proves the presence of the 48% cliff"* — the suite locks the defect in and will fail if it is fixed.
- **Evidence & Citation:** `audit_verify_4.py` section B; test source lines 61–73. Prior SCI-FLAW-16 **CONFIRMED in substance** but its quoted formula (`Ea = 0.5460/M^0.0988` vs `1 − 0.043·M`) is **STALE** — that is not what the current file contains.
- **Status:** CONFIRMED
- **Note:** with corrected formula description

### HIGH-03 — Recovery floors fabricate recovery at zero throughput

- **Severity:** HIGH
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:193` (`clip(displacement_eff, 0.05, 0.95)`), `:200` (`clip(rf, 0.05, 0.85)`), `:326` (`clip(recovery, 0.10, max_cap)`), `:475` (`clip(rf, 0.05, 0.80)`)
- **Observed:** `MiscibleSurrogate` with `hcpvi = 0` returns **0.050000**; `hcpvi = 0.01` → **0.050000**; `hcpvi = 0.1` → 0.075. `ImmiscibleSurrogate` returns 0.10 for all inputs (CRIT-07). `HybridSurrogate` floors at 0.05.
- **Expected:** Zero injected pore volumes ⇒ zero incremental recovery (`RF → 0` as `HCPVI → 0`). Floors belong on *diagnostics*, never on a physical response function.
- **Scientific Impact:** The optimizer can obtain "recovery" without injection; limits of the model are wrong (non-zero at the origin), which invalidates any gradient/finite-difference and any limiting-case test.
- **Evidence & Citation:** `audit_verify_4.py` section D.
- **Status:** NEW

### HIGH-04 — `or`-default pattern silently replaces legitimate zero values

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/analytical_models.py:661` (`params.get("v_dp") or params.get("v_dp_coefficient") or 0.5`), `:662`, `:671` (`c7_plus_fraction or c7_plus or 0.3`); same idiom widely used in `_literature_based_recovery` (`:657-765` region)
- **Observed (measured):** `PhDHybridSurrogate`, baseline RF = 0.493568:
  | input | baseline | explicit `0.0` | other value |
  |---|---|---|---|
  | `v_dp` | 0.493568 (0.5) | **0.493568** (0.0 → 0.5) | 0.131982 (0.9) |
  | `s_wi` | 0.493568 (0.25) | **0.493568** (0.0 → 0.25) | 0.150000 (0.6) |
  | `c7_plus_fraction` | 0.493568 (0.3) | **0.493568** (0.0 → 0.3) | 0.492970 (0.8) |
  The model *is* sensitive to all three (right column), so the identical output for `0.0` proves the substitution.
- **Expected:** `dict.get(key, default)` — absence of a key, not falsiness. `S_wi = 0`, `V_DP = 0` (perfectly layered) and `C7+ = 0` are legal physical inputs.
- **Scientific Impact:** User-specified extreme/idealized cases are silently overwritten with defaults; sensitivity sweeps to 0 are meaningless.
- **Evidence & Citation:** `audit_verify_4.py` section E plus the follow-up differential run recorded in this round's transcript.
- **Status:** NEW

### HIGH-05 — Caprock leakage is unreachable, and "breached" can be reported with zero leakage

- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:442-443`, `:468`; `core/engine_surrogate/geomechanics_fault.py:113-118`, `:170`, `:174-181`, `:202-208`
- **Observed:** `current_p = clip(current_p, p_min, p_safe_ceiling)` (`:468`) means `P ≤ p_safe_ceiling` by construction, while leakage requires `P > p_safe_ceiling` (`geomechanics_fault.py:174`) ⇒ `caprock_leakage ≡ 0` for every timestep of every run. Meanwhile `is_caprock_breached = (P > p_frac_caprock) or (tensile_margin < 0) or (shear_margin < 0)` (`:170`) can be **True** through the stress path while `caprock_leakage_rate_tonne_day = 0`.
  Fault leakage (`:202`) stays reachable (`p_crit_fault` ≈ 3950 psi for defaults), but its output is never consumed (CRIT-10).
- **Expected:** Either the safety ceiling is a *soft* limit (leakage above it possible) or the metric must be reported as "structurally impossible under current constraints"; a `breached=True` state must not carry zero flux.
- **Scientific Impact:** `is_caprock_breached` and `caprock_safety_margin` are reported to the UI/plots while the associated mass loss is provably zero ⇒ containment reporting is internally contradictory and EPA Class-VI breach events cannot be simulated.
- **Evidence & Citation:** Code path reading at `surrogate_engine.py:459-469` and `geomechanics_fault.py:167-181`.
- **Status:** NEW

### HIGH-06 — Three mutually inconsistent Koval heterogeneity-factor formulas (2.5× … 10⁹⁵× apart)

- **Severity:** HIGH
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:894-895`; `core/engine_surrogate/analytical_models.py:174`, `:534`, `:726`; `core/data_models.py:57`; `core/engine_surrogate/profile_generator_fast.py:960`
- **Observed:** `H = 10**(v/(1−v))` (engine, used for breakthrough time) vs `H = 1/(1−v)²` (all analytical models, and `data_models.calculate_koval_from_reservoir`) vs `H = 1/((1−v)² + 1e-6)` (`data_models.py:57`).
  Measured divergence of the engine form versus `1/(1−v)²`: **1.14× (v = 0.2), 2.5× (0.5), 400× (0.8), 10⁷ (0.9), 10⁹⁵ (0.99)**.
  The engine's `v_dp` is **not clipped** before use (`surrogate_engine.py:894`), and `_build_params_dict` passes `v_dp` straight from `reservoir_data.v_dp_coefficient` (default 0.0 at `data_models.py:357`, but reachable up to 1.0 through relaxable ranges).
- **Expected:** One formula, one reference: Koval (1963) `H = 1/(1 − V_DP)²`. Every site must clip `V_DP ∈ [0, 1)` first.
- **Scientific Impact:** Breakthrough time `t_bt = (PV/K)/annual_inj` (`surrogate_engine.py:917-919`) inherits the full error; breakthrough time drives the NPV CO₂ cost, the storage ledger and the breakthrough penalty (`optimisation_engine.py:1729-1736`). At `v_dp → 1` the engine form overflows.
- **Evidence & Citation:** `audit_verify_3.py` section V10; `data_models.py:40-57` docstring states the `1/(1−v)²` form as the intended one.
- **Status:** NEW

### HIGH-07 — Productivity/injectivity index is tautological; injection viscosity hard-coded

- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:417`, `:420-433`
- **Observed:** `nominal_drawdown = 500.0`; `J_inj_nominal = q_inj_step_rb / nominal_drawdown` ⇒ `J` is *defined* as rate divided by a constant, so `q = J·ΔP` reproduces the nominal rate scaled only by `ΔP/500` — no `kh`, no `r_e/r_w`, no skin enters through this path. `mu_inj_eff = 0.05 if water_inj_bpd <= 0 else 0.50` (`:430`) ignores the PVT module (CO₂ μ measured 0.0206 → 0.0443 cP over 1000 → 4500 psi). `vrr_profile[i] = q_inj_step_rb / q_prod_step_rb` (`:417`) is computed from **uncapped nominal** rates, before the geomechanical cap and bleed-off (`:436-448`).
- **Expected:** `J` from Peaceman/Babu–Odeh with `k`, `h`, `r_w`, skin (the module already exports `calculate_peaceman_index_*` and sums them at `:296-309`); injected-fluid viscosity from the PVT state; VRR from the **actual** constrained rates.
- **Scientific Impact:** Deliverability coupling is circular for the fallback branch; reported VRR (a Class-VI-relevant metric) overstates injection whenever the geomechanical cap binds; viscosity-dependent injectivity is a step function of the water rate.
- **Evidence & Citation:** Code reading `:419-448`; `pvt_state.py:204-226` for CO₂ μ.
- **Status:** NEW

### HIGH-08 — NPV omits four economic inputs that the engine explicitly supplies, and costs CO₂ on *stored* rather than *purchased*

- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_models.py:507-530` (inputs mapped at `core/engine_surrogate/surrogate_engine.py:940-957`)
- **Observed:**
  - NPV reads only `oil_price_usd_per_bbl`, `co2_cost_usd_per_ton`, `discount_rate`, `capex_usd` (`:508-511`).
  - `variable_opex_usd_per_bbl`, `co2_storage_credit_usd_per_tonne`, `carbon_tax_usd_per_tonne`, `co2_recycle_cost_usd_per_tonne` are mapped into `params` at `surrogate_engine.py:946-956` **and never read by NPV** — a loss of ~$5/bbl × oil volume (order 10⁶–10⁷ $) plus any storage credit/tax.
  - `annual_oil_production = cumulative_oil / project_life_years` and `annual_co2_injected = co2_stored / project_life_years` (`:515-516`) ⇒ a **flat** cash flow: no plateau, no decline, no ramp; the real production profile is discarded.
  - `annual_co2_injected` uses `co2_stored` (breakthrough-aware) not purchased volume, and `injection_rate` (`:484`) is the **per-well** rate while `profile_generator_fast.py:214` scales `field_injection_rate = injection_rate × active_injectors` ⇒ revenue is field-scale, CO₂ cost is single-well-scale ⇒ NPV inflated by `(n_injectors − 1) × CO₂ cost`.
  - NPV's `co2_stored` ≠ reported `co2_stored_tonnes` (`surrogate_engine.py:544` = injected − produced).
- **Expected:** Discount the *actual* annual profile; cost purchased CO₂ (`injected − recycled`); include every supplied economic parameter; use consistent well counts.
- **Scientific Impact:** The default optimization objective is systematically biased upward for multi-well fields, insensitive to OPEX/storage-credit/carbon-tax policy, and inconsistent with the mass balance reported alongside it.
- **Evidence & Citation:** Input mapping `:940-957` vs reads `:508-511`; `profile_generator_fast.py:214`; `audit_verify_3.py` reading.
- **Status:** NEW

### HIGH-09 — Production default model is `hybrid`, but 72 test references pin `phd_hybrid`

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/data_models.py:1719-1721`; `core/optimisation_engine.py:264-270`; `config/base_config.json:24`, `:746`; `core/engine_surrogate/analytical_models.py:431-475`
- **Observed:** `recovery_model_type: str = "hybrid"` (dataclass default), `default_recovery_model: "hybrid"` and `"recovery_model": "hybrid"` in `config/base_config.json`; the engine is constructed with that value (`optimisation_engine.py:268-270`). Test-tree references: **`phd_hybrid` 72 occurrences vs `"hybrid"` 6**.
- **Expected:** Tests exercise the configuration shipped to users.
- **Scientific Impact:** The shipped model (`HybridSurrogate`, floors + constant immiscible limb + hard-coded α/β, CRIT-07/CRIT-13/HIGH-03) is *not* the model the scientific test-suite validates. Passing tests therefore provide no evidence about production behavior — exactly the "benchmark agreement mistaken for correctness" failure mode this audit is required to reject.
- **Evidence & Citation:** Counts from repo-wide `Select-String`; `data_models.py:1719-1721`.
- **Status:** NEW

### HIGH-10 — Undefined name `QIcon` fails 4 tests

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `ui/widgets/fault_geometry_visualizer_widget.py:115`
- **Observed:** `QPushButton(QIcon.fromTheme(...))` with no `QIcon` import. Ruff `F821`. Pytest: `4 failed, 329 passed, 23 skipped` — failures: `tests/test_project_save_load.py::test_data_management_widget_save_and_load`, and 3 in `tests/ui/test_3d_well_interaction.py`, all `NameError: name 'QIcon' is not defined`.
- **Expected:** Import `QIcon` from `PyQt6.QtGui`; suite green.
- **Scientific Impact:** Mandatory post-change gate `pytest tests/test_project_save_load.py -v` (AGENTS.md invariant #5) currently **cannot pass**, so data-model/UI changes cannot be validated as required.
- **Evidence & Citation:** `audit/runtime/pytest_output.txt` (short test summary); ruff F821 list.
- **Status:** RESOLVED
- **Note:** re-verified 04-10-2026 after commit `a68fc35`: `ui/widgets/fault_geometry_visualizer_widget.py:19` now carries `from PyQt6.QtGui import QIcon`, the F821 is gone, and `pytest tests/ -q` returns **`333 passed, 23 skipped, 0 failed`** (356 collected). `pytest tests/test_project_save_load.py` (AGENTS.md invariant #5 gate) now passes. `audit/runtime/pytest_output.txt` is retained as the **pre-fix** baseline and must be read as such. The finding is retained (not deleted) because the same defect *class* recurred twice in new code — see **HIGH-19**.

### HIGH-11 — `RECOVERY_MODELS_AVAILABLE = False` is the only guard preventing `NameError`s

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/analytical_models.py:30` (flag), `:96` (`MiscibleRecoveryModel`), `:218` (`ImmiscibleRecoveryModel`), `:341` (`BuckleyLeverettModel`)
- **Observed:** Ruff F821 reports all three names as undefined; they are referenced in branches guarded by `if RECOVERY_MODELS_AVAILABLE:`. The flag is a hard-coded `False`.
- **Expected:** Either the legacy `core/simulation/recovery_models.py` imports exist (then the flag can be True), or the dead branches are removed with a documented decision.
- **Scientific Impact:** `tests/…` cannot cover those branches; flipping the flag (a one-line "config change") instantly raises `NameError` at runtime. The advertised "literature-based fallback to the full models" is fictional.
- **Evidence & Citation:** Ruff F821 output (`audit/code_quality/ruff_report.json`).
- **Status:** RECURRED — the three names remain unbound: measured `hasattr(module, 'MiscibleRecoveryModel')` = False, same for `ImmiscibleRecoveryModel` and `BuckleyLeverettModel`, while `RECOVERY_MODELS_AVAILABLE` is a hard-coded `False`. `analytical_models.py:96`, `:223`, `:358` therefore rely on the flag as the ONLY thing preventing a `NameError` inside a constructor inside an optimizer evaluation. This is the HIGH-10 -> HIGH-19 -> HIGH-11 recurrence of the same defect class. Now detected automatically by the gate's F821 release gate. Reopen: https://github.com/fgfalll/WAG_optimisation/issues/16

### HIGH-12 — Exception funnel converts all failures into penalties without tracebacks

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:788-790` (`except Exception → _error_result`), `:964-988` (`_error_result`: RF=0, NPV=0, `convergence_status: "error"`); `core/engine_surrogate/surrogate_models.py:553-562`; `core/engine_surrogate/pvt_state.py:147-149` (PR root fallback `Z = max(0.25, B*1.05)` with **no log**); `core/optimisation_engine.py:725-727`, `:903-904`, `:1572-1576`
- **Observed:** Engine-level `except Exception` logs `f"...{e}"` **without `exc_info`**; the error is converted to `SimulationEngineError` → `FAILURE_PENALTY`. Programming errors (typo, `KeyError`, shape mismatch) are indistinguishable from "bad candidate". The PR root fallback silently substitutes `Z = max(0.25, B·1.05)` when no physical root is found — no warning, so a wrong density propagates as a normal number.
- **Expected:** Narrow exception types for expected numeric failures; full traceback for unexpected ones; explicit log/count on any physical fallback.
- **Scientific Impact:** Defects present as "the optimizer dislikes this region" rather than as crashes; fallbacks are invisible in logs, which invalidates any claim that runs were "clean".
- **Evidence & Citation:** Source lines; `audit/runtime/pytest_output.txt` (no engine error surfaced in suite).
- **Status:** NEW
- **Note:** (severity reduced from an initial "plausible zeros survive" hypothesis — the error key **is** checked at `optimisation_engine.py:725`, so the failure mode is *masking*, not *silent acceptance*)

### HIGH-13 — Wrapper hard-codes a $100/t carbon-tax floor over the user's value

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/objectives/wrapper.py:96-97`; policy value `core/data_models.py:1734` (`carbon_tax_usd_per_tonne: float = 75.0`)
- **Observed:** `remediation_cost = max(carbon_tax, 100.0) * leaked_tonnes` — any user/configured carbon tax below $100/t is silently raised. The engine also maps `economic_params.carbon_tax_usd_per_tonne` (`surrogate_engine.py:956`) which NPV ignores (HIGH-08).
- **Expected:** Use the configured value; if a floor is policy, expose it as a named, documented parameter.
- **Scientific Impact:** User-facing parameter does not control the reported cost; combined with CRIT-10 the block never executes anyway — two independent reasons the leakage price signal is not what the UI claims.
- **Evidence & Citation:** `wrapper.py:95-99`; `data_models.py:1734`.
- **Status:** NEW

### HIGH-14 — Two different sandface-pressure models coexist

- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/objectives/wrapper.py:64-67` (`P_sandface = P_max + q_inj/II`, `II` default 25) vs `core/engine_surrogate/surrogate_engine.py:359` (`p_inj_sandface = min(current_p + 400, ceiling)`)
- **Observed:** A rate-independent +400 psi offset in the engine versus a rate-dependent `q/II` in the wrapper; both are labelled Class-VI sandface pressure. The engine's value is additionally capped at `p_safe_ceiling` by construction, so it can never report an overpressure.
- **Expected:** One model, defined once, evaluated on the same rate and pressure, with the cap applied *after* the diagnostic value is recorded (so a violation is observable).
- **Scientific Impact:** Whichever path is enabled, the reported sandface pressure cannot be compared with the other; and the capping guarantees no violation is ever visible (relates to HIGH-05).
- **Evidence & Citation:** Both code sites; `geomechanics_fault.py:118` for `p_safe_ceiling = p_frac·safety_factor`.
- **Status:** NEW

### HIGH-15 — Capillary gas-trapping term is inverted (and a test asserts the inversion)

- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_models.py:238-241`; test `tests/scientific/co2/test_co2_trapping_mechanisms.py:18-34`
- **Observed:** `gas_trapping = 1.0 - s_gc` ⇒ raising critical gas saturation *reduces* trapping (0.05 → 0.95, 0.60 → 0.40). The comment above it states the correct physics ("gas below critical saturation cannot flow"). Test `test_inverted_critical_gas_trapping` **asserts** `eff_high_sgc < eff_low_sgc`.
- **Expected:** Higher `S_gc`/`S_gr` ⇒ more immobilized gas (Land, C.S. (1968) *SPE J.* 8(2), 149–156: `S_gr = S_gi/(1 + C·S_gi)`).
- **Scientific Impact:** Currently reachable only through `calculate_storage_efficiency` (`surrogate_models.py:403`), which itself has no production consumers (MED-05) — so the defect is *latent*, but the exported function and the codified test will propagate the error to any future caller.
- **Evidence & Citation:** `audit_verify_3.py` section V12; test source. Prior SCI-FLAW-17 **CONFIRMED** (locations `:238-241` accurate).
- **Status:** CONFIRMED

### HIGH-16 — CO₂ viscosity polynomial is not the cited correlation and under-predicts dense-phase μ ≈ 2×

- **Severity:** HIGH
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:204-226`
- **Observed:** Docstring cites "Fenghour/Vesovic"; the code implements `μ/μ₀ = 0.235·ρr + 0.395·ρr² − 0.041·ρr³` with `ρr = ρ/467.6`. At `ρr = 1` this yields `μ/μ₀ = 0.589`; the physical critical-point value is ≈ 3.4 (Vesovic et al. 1990 / Fenghour et al. 1998) ⇒ **~2.2× under by construction**. Measured μ: **0.0206 cP (1000 psi) → 0.0443 cP (4500 psi)** at 150 °F, then clipped to `[0.015, 0.12]` (`:226`).
- **Expected:** The published residual-viscosity form of Fenghour, Vogel & Wakeham (1998) with its coefficient set, or an explicit statement that a fit is used and to what data.
- **Scientific Impact:** Currently **no** engine consumer reads this value for the pressure/mass path (`gas_props` is consumed only for `bg`/`cg`, `surrogate_engine.py:366-367`), so the impact today is limited to mixture-μ outputs consumed by UI/diagnostics — but it is an unverifiable constant in an exported API, so it must carry `UNKNOWN — EVIDENCE REQUIRED` until cited.
- **Evidence & Citation:** `audit_verify_2.py` section E (μ sweep); coefficient arithmetic.
- **Status:** NEW

### HIGH-17 — Gas compressibility is a two-branch fudge with un-cited coefficients

- **Severity:** HIGH
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:388-394`
- **Observed:** `c_g = (1/P)·(0.35·y_CO₂ + 0.85·(1−y_CO₂))` for `P > 1500 psi`, else `1/P`. Measured `c_g = 3.40e-4 (y=0)`, `2.40e-4 (y=0.5)`, `1.40e-4 (y=1)` 1/psi at 2500 psi. Dense-phase CO₂ at 2500 psi / 150 °F has `c ≈ 1–3e-5 1/psi` ⇒ **4–12× too high**; and the branch is discontinuous at 1500 psi (`1/P` → `0.35/P` for pure CO₂, a 65% jump).
- **Expected:** `c_g = 1/P − (1/Z)(dZ/dP)` from the Z-model (which CRIT-05 gets wrong) — or a cited correlation; never a hard branch on pressure with un-cited constants.
- **Scientific Impact:** `ct_dynamic` (`surrogate_engine.py:453`) mixes this with a constant `c_o`; an over-large `c_g` makes the reservoir artificially compliant, damping pressure response and hiding the `B_g` error of CRIT-04.
- **Evidence & Citation:** `audit_verify_3.py` section V2b (`cg` column).
- **Status:** NEW

---

### HIGH-18 — The scientific verification suite cannot fail on the defects it claims to verify

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `tests/scientific/co2/test_co2_breakthrough_physics.py:16`, `:19-58`; `tests/scientific/conservation/test_mass_conservation.py` (`test_pore_volume_vs_ooip_recovery_bound_discrepancy`); `tests/scientific/mathematical/test_analytical_identities.py:11-12`; `tests/scientific/dimensional/test_unit_consistency.py:16-30`; `tests/scientific/co2/test_co2_trapping_mechanisms.py:31-33`; `tests/scientific/mathematical/test_singularity_and_overflow.py` (`test_mobility_ratio_unit_limit_singularity`)
- **Observed:** An AST scan of `tests/scientific/` (script `v_tests2.py`) over the **36 collected test functions** finds:
  1. **5 tests import production symbols and never call them.** `test_koval_fractional_flow_mobility_monotonicity` imports `FastProfileGenerator` at `:16`, then at `:38-46` re-derives `h_koval`, `e_eff`, `koval_factor` and `f_g` **by hand** and asserts the copy is monotone — it never executes `profile_generator_fast.py`. Its docstring nonetheless states *"Verifies that SCI-FLAW-01 has been eliminated from `profile_generator_fast.py`"*.
  2. **3 tests assert that a defect exists, as the expected outcome**, so a future fix *fails* the suite: `assert step > 0.40` (48 % cliff, HIGH-02), `assert eff_high_sgc < eff_low_sgc` (inverted trapping, HIGH-15), `assert truncation_fraction > 0.20` (pore-volume RF bound, SCI-FLAW-11). Two of these **hard-code the "code" value** (`code_pore_volume_limit = 1.0 - swi - sor`) instead of reading it from the source, so they cannot detect remediation even if the code is fixed.
  3. **6 tests never import production code at all.** `test_darcy_inflow_dimensions` asserts `J = q/ΔP` dimensional homogeneity using a locally invented `J = 2.0 * J_unit`; `agent_wiki/verification/test_matrix.md:36` records it as verifying `surrogate_engine.py:330`.
- **Expected:** A verification test must (a) execute the production symbol under test, (b) assert the *physically correct* value, and (c) fail when the defect is introduced or removed.
- **Scientific Impact:** The **Analytical Verification** tier of the evidence hierarchy is non-informative for these items. A green `36 passed` is reported in `agent_wiki/verification/test_matrix.md` as `VERIFIED` / `CONTRADICTED BY TEST`, which reads as scientific assurance but is produced by tests that re-state the code's own arithmetic. Three defects are *enshrined* as regression expectations (HIGH-02, HIGH-15, SCI-FLAW-11): remediation will be blocked by a red build, and no test will notice a re-introduction of SCI-FLAW-01.
- **Evidence & Citation:** `v_tests2.py` (AST: `A) tautological = 5`, `B) no production import = 6`, `C) exercises production = 25`); `pytest tests/scientific` → `36 passed`; source reads cited above.
- **Status:** NEW
- **Note:** **do not "fix" by weakening these tests**; re-specify them against production code (see `res_audit.md` §5, change-safety row NEW).

### HIGH-19 — Two undefined names in the new workbench code reproduce the HIGH-10 defect class

- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `ui/workbench/components/pyvista_reservoir_canvas.py:974` (`has_active_fault`); `ui/workbench/components/subsurface_data_viewer_widget.py:335` (`QToolTip`)
- **Observed:** Both names are referenced but never bound anywhere in the repository (repo-wide grep: `has_active_fault` = 1 occurrence, the *use*; `QToolTip` = 1 occurrence in this file, the *use* — every other widget that calls `QToolTip.showText` imports it from `PyQt6.QtWidgets`).
  - `:974` sits inside the caprock block `if hasattr(self, 'chk_caprock') and self.chk_caprock.isChecked()` (`:946`), which is inside `try:` (`:601`) … `except Exception as e: logger.error(f"PyVista 3D rendering error: {e}", exc_info=True)` (`:1272-1273`). The `NameError` is therefore caught, logged at ERROR only, and control leaves the block **before** `self.plotter.render()` (`:1269`) and `self._has_rendered_mesh = True` (`:1267`) — the 3D canvas renders nothing while appearing to "work". The checkbox is switched on programmatically by two workbench tree entries, `subsurface_workbench_widget.py:755` (`"caprock_3d"`) and `:784` (`"caprock"`), so the path is reachable from normal navigation (default state is `setChecked(False)` at `:258`).
  - `:335` is in `_copy_table_to_clipboard`, connected directly to `btn_copy_table.clicked` (`:130`) with **no** `try/except`. The clipboard is written first (`:334`), so data is copied, then the slot raises `NameError: name 'QToolTip' is not defined`; the tooltip feedback never appears and the exception escapes the slot to the PyQt6 exception hook.
- **Expected:** Import `QToolTip` from `PyQt6.QtWidgets`; define (or derive) `has_active_fault` — e.g. `has_active_fault = len(active_fault_list) > 0` — before `:974`. Ruff `F821` is the intended detector for exactly this defect class.
- **Scientific Impact:** A marquee feature of the new workbench (3-D caprock stratigraphy) silently produces an empty viewport, and the sheet-export copy button raises in a slot. Neither is covered by any test, so the suite stays green — the same "cannot fail" condition documented in HIGH-18. This is the second occurrence of the HIGH-10 pattern within two commits, i.e. the fix for HIGH-10 was treated as a one-off rather than as a class.
- **Evidence & Citation:** `ruff check . --output-format json` → `F821 pyvista_reservoir_canvas.py:974 "Undefined name has_active_fault"`, `F821 subsurface_data_viewer_widget.py:335 "Undefined name QToolTip"` (`audit/ruff_output.json`); source reads cited above; repo-wide grep for both identifiers.
- **Status:** RESOLVED
- **Note:** Remediated in ui/workbench/components/pyvista_reservoir_canvas.py (defined has_active_fault) and subsurface_data_viewer_widget.py (imported QToolTip).
- **Note:** `F821 core/geology/petrophysical_distribution.py:404` flags `prev_field`, but that name *is* bound at `:405` on the previous loop iteration and is only read under `if k > 0` (`:402`). Control flow makes it safe at runtime; this is a flow-insensitive lint artifact. → record in §6.

---

## 2b. Round-2 findings — verification of the 05-10-2026 remediation

> **Source:** [`agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md`](../../agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md)
> These findings were produced by adversarially re-auditing the **uncommitted** 05-10-2026 remediation
> and testing every "RESOLVED" claim. 11 CONFIRMED · 4 PARTIALLY RESOLVED · 1 REGRESSED.
> **Do not apply the remediation before reviewing these.**

### CRIT-14 — The mobility-ratio override severs oil viscosity from recovery entirely
- **Severity:** CRITICAL
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:170-172`, `:275-277`, `:412-414`; source `core/engine_surrogate/surrogate_engine.py:976-977`
- **Observed:** the three recovery models now prefer `params["mobility_ratio"]` over the computed `μ_o/μ_oe`. `_build_params_dict` sets it from `EORParameters.mobility_ratio` (`core/data_models.py:798`), default **5.0**, so the `2.5` fallback at `:977` is dead. Measured RF is **exactly flat** at 0.523509 for `μ_o` = 0.5 → 100 cP; the legacy path varied 0.593831 → 0.559522.
- **Expected:** CO₂ viscosity reduction must enter `M = (k_ro/μ_o)/(k_rg/μ_g)`. The `max(·, 1.0)` floor additionally deletes the favourable regime `M < 1`.
- **Scientific Impact:** the entire viscosity-contrast mechanism — the physical basis of CO₂ flooding — is removed from the miscible, immiscible and Buckley-Leverett limbs. Recovery becomes a function of one constant.
- **Evidence & Citation:** `audit_verify_5.py` §B1–B3.
- **Status:** NEW

### CRIT-15 — Default configuration pins the Koval sweep at its 0.95 clip for every mobility ratio
- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:558-571`; `core/engine_surrogate/surrogate_engine.py:946-968`
- **Observed:** `params["hcpvi"]` is built as `injection_rate × b_co2 × 365.25 × lifetime`; for shipped defaults this is **7.6928**. Measured sweep at that throughput = **0.950000 for M = 1, 2, 5 and 10 identically**; the clip is reached at HCPVI ≈ 6.0. The expression ignores the WAG schedule, shut-ins, availability and the compressor cap, so it is not the throughput actually injected.
- **Expected:** optimisation must occur inside the sweep-sensitive regime; published CO₂-EOR practice terminates near HCPVI 0.5–3.0.
- **Scientific Impact:** with CRIT-14, recovery is nearly constant over the whole mobility axis; selection pressure on these genes is arbitrary.
- **Evidence & Citation:** `audit_verify_5.py` §D4/§I, `audit_verify_6.py` §I.
- **Status:** NEW

### CRIT-16 — `bo`/`b_co2` provenance guards are inert; the user's own PVT inputs are silently discarded
- **Severity:** CRITICAL
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:963-964`
- **Observed:** `getattr(reservoir_data, "bo_rb_per_stb", None)` and `("bg_rb_per_mscf", None)` — measured `hasattr` = **False** for both. The real field is `ReservoirData.oil_fvf` (`core/data_models.py:433`), value 1.2 in the audit fixture vs the correlation's 1.3053.
- **Expected:** name the field that exists. Same `or`-default defect class as HIGH-04, re-introduced.
- **Scientific Impact:** a user with fitted live-oil FVF cannot reach HCPVI; the sweep throughput is unoverrideable.
- **Evidence & Citation:** `audit_verify_6.py` §G.
- **Status:** NEW

### CRIT-17 — The CRIT-12 fix breaks saturation closure: `S_o + S_w > 1` on 27 % of timesteps
- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:535-551`
- **Observed:** `pv_ref = ooip·bo_mean/(1−S_wi)` at `:539` is the **hydrocarbon** pore volume, but `S_wi` is defined on **total** PV. Measured on the default 181-step run: `S_o + S_w > 1` on **49/181 steps (27.1 %)**, `max(S_o+S_w) = 1.042829`, and `S_o+S_w+S_g ∈ [1.000000, 1.042829]`. `S_g = 0` on 49 steps only because `np.clip` saturates.
- **Expected:** `Σ S = 1` exactly at every step; use total PV for both terms.
- **Scientific Impact:** the three-phase saturation history violates its defining constraint, and every saturation-derived quantity is corrupted invisibly — Phase-3 anti-pattern class D. A 25 % saturation bias exists at t = 0.
- **Evidence & Citation:** `audit_verify_6.py` §J, `audit_verify_9.py`.
- **Status:** NEW (regression from the CRIT-12 remediation)

### CRIT-18 — The new NPV omits the hydrocarbon-gas revenue stream it computes
- **Severity:** CRITICAL
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:622-651` (revenue at `:636`), allocation `:588`, accumulation `:606`, publication `:688`
- **Observed:** `annual_rev = annual_oil_stb·oil_price + annual_stored_tonne·co2_storage_credit`. `annual_hc_gas_mscf` is computed and published but **never referenced again**. Measured: **216 810 MSCF** of gas sales over 15 yr ≈ **$650 k at $3/MSCF contributing $0**. Produced CO₂ earns no sale revenue either.
- **Expected:** oil + solution/associated gas + CO₂ sales + storage credit, with matching opex.
- **Scientific Impact:** `npv` is the primary objective; the optimiser ranks candidates on a truncated cash-flow model.
- **Evidence & Citation:** `audit_verify_9.py`.
- **Status:** NEW

### CRIT-19 — `gravity_factor` is now an active, unprincipled, triple-purpose fudge multiplier
- **Severity:** CRITICAL
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/analytical_models.py:199-201`, `:331-333`, `:802`; wired at `core/engine_surrogate/surrogate_engine.py:928`, `:974-975`
- **Observed:** `e_v = 1/(0.8 + 0.2·gravity_factor)` multiplies RF; the same expression divides `vertical_eff`; and `gravity_factor` multiplies `N_g`. It is dimensionless, has **no definition, unit or citation**, range [0.5, 1.5] (`core/data_models.py:1028-1029`), default 1.0.
- **Expected:** gravity segregation is computed from `Δρ`, dip, permeability and Darcy velocity — which `N_g` at `:802` already does. A second hand-set gravity factor can only be fitted.
- **Scientific Impact:** a gene that scales recovery ±20 % with no physical content, applied twice in the hybrid path plus once in `N_g`. Hidden-calibration pattern. CRIT-13 had recorded this gene as *inert*; making it active is a net loss.
- **Evidence & Citation:** `audit_verify_5.py` §B; source reads.
- **Status:** NEW

### CRIT-20 — Miscibility weight is decoupled from composition and equals 0.5 exactly at the MMP
- **Severity:** CRITICAL
- **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:474-481`
- **Observed:** `alpha = params.get("transition_alpha", params.get("alpha_base", default_alpha))` and `_build_params_dict` always supplies `alpha_base = 0.9750`, so the C₇⁺ branch is dead. Measured ω is **identical (0.534688)** for C₇⁺ = 0.1/0.2/0.3/0.5, whereas the legacy composition-driven α varied. With α = 0.975, β = 20: `ω(P/MMP=1.0) = 0.622`, `ω(P/MMP=0.975) = 0.500`, `ω(P/MMP=0.95) = 0.378`.
- **Expected:** MMP is where displacement *becomes* miscible; a blend that weights half the recovery to miscibility *at* the MMP is not a miscibility model, and composition must move the window.
- **Scientific Impact:** the central miscible/immiscible decision is governed by two uncalibrated constants and is insensitive to the fluid.
- **Evidence & Citation:** `audit_verify_5.py` §C/§C1.
- **Status:** NEW

### CRIT-21 — CO₂ compressibility is a new hard-coded power law contradicting the Peng-Robinson EOS in the same class
- **Severity:** CRITICAL
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:415-419`
- **Observed:** `cg_co2 = clip(1.5e-4·(2000/p)^0.8, 2e-5, 5e-4)` for `p > 1200`, uncited, while the class already implements PR (`:106`, `calculate_co2_fvf_rb_per_mscf`). Measured vs `-(1/B)dB/dP` from PR: **7.98× at 1500 psi, 8.31× at 2000, 4.43× at 2500, 2.59× at 3000**.
- **Expected:** one EOS, one compressibility: `c_g = 1/P − (1/Z)(dZ/dP)` from the same Z(EOS,P,T) that yields `B_g`.
- **Scientific Impact:** `c_g` enters `ct` (`surrogate_engine.py:453`) and hence `dp` (`:455`); an 8× error near-critical distorts the pressure path where CO₂ floods are most sensitive. HIGH-17's undocumented fudge was replaced by a differently shaped one.
- **Evidence & Citation:** `audit_verify_5.py` §A3/§A4.
- **Status:** NEW

### HIGH-20 — `B_o` is C⁰ but not C¹ at the bubble point (slope flips sign discontinuously)
- **Severity:** HIGH
- **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:303-322`
- **Observed:** central-difference `dB_o/dP` = **+1.2609e-4 /psi** at 2 790 psi vs **−1.5699e-5 /psi** at 2 810 psi (jump ≈ 8×, sign flip). `B_o` itself is continuous (1.1e-10), so CRIT-03's C⁰ repair is sound.
- **Expected:** one continuous expression, or an explicit piecewise-C⁰ statement.
- **Scientific Impact:** spurious gradient sign flips near `P_b` perturb GA/BO line searches.
- **Evidence & Citation:** `audit_verify_5.py` §A7.
- **Status:** NEW

### HIGH-21 — Bubble point is a hard-coded constant and `c_o` defaults conflict between modules
- **Severity:** HIGH
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:101-106`; conflicting value at `core/engine_surrogate/surrogate_engine.py:315`
- **Observed:** `p_bubble = min(p_init, 2800.0)` when unsupplied — **no citation, no UI field** (measured 2800.0 psi at defaults). Meanwhile the pressure ODE uses `c_o = 1e-5` while the PVT module defaults to `1.2e-5`; the module computing `B_o(P)` and the module integrating pressure assume fluids differing by 20 % in compressibility.
- **Expected:** `P_b` is a measured PVT quantity; if unsupplied mark `UNKNOWN — EVIDENCE REQUIRED`. One `c_o` for both consumers.
- **Scientific Impact:** the `P_b` default silently decides saturation state, which flips the sign of `dB_o/dP` and therefore the whole volumetric bookkeeping.
- **Evidence & Citation:** `audit_verify_5.py` §A/§A5.
- **Status:** NEW

### HIGH-23 — Containment is economically inert: leakage is identically zero, yet leaked CO₂ would earn storage credit
- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:635`, `:644`; `core/optimisation_engine.py:1372-1386`; `core/objectives/wrapper.py:108-132`
- **Observed:** `total_leakage_tonne = 0.0`, `leakage_rate_tonnes_day` max = 0.0, `annual_leakage_tonne` = 15 zeros, across every configuration probed. `optimisation_engine.py:1372-1375` now *reads* the key successfully — it is simply always zero, making the synthetic-overpressure branch at `:1377-1386` unreachable. `wrapper.py:108-111` likewise gets 0.0, so the only path that could charge leakage (`:112-121`, using `leakage_rate_fraction`) is now **dead**, silently disabling `CO2StorageParameters.leakage_rate_fraction` (0.01). Meanwhile `:635` computes `annual_stored_tonne = max(0, inj − prod)` — **leakage-blind** — and `:636` pays `co2_storage_credit` (**default 25.0 USD/t**) on it.
- **Expected:** `M_stored = M_inj − M_prod − M_leaked`; credit only CO₂ actually retained.
- **Scientific Impact:** the model pays for CO₂ it cannot retain and never charges for CO₂ it loses. The externality of a containment failure is priced at zero while the revenue for the same molecules is priced at $25/t — inverting EPA Class VI risk incentives. The most consequential finding for a carbon-storage project.
- **Evidence & Citation:** `audit_verify_9.py`; `audit_verify_7.py` §R; wiki invariants #10/#11.
- **Status:** NEW

### HIGH-24 — Two live VRR definitions coexist; the post-hoc one silently overwrites the integrated one
- **Severity:** HIGH
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:417` vs `:545-548`
- **Observed:** the integrated VRR uses `bo_dynamic`/`bg_dynamic` with the **pre-rescale** profile; the post-hoc VRR uses `bo_profile`/`bg_profile` with **post-rescale** streams and overwrites `vrr_profile` whenever `len(oil_rate) == len(pressure_profile)` — an incidental array-length coincidence selects which definition ships.
- **Expected:** one definition.
- **Scientific Impact:** VRR is the diagnostic used to judge voidage replacement; the physically coupled (integrated) value is discarded.
- **Evidence & Citation:** Source reads at `surrogate_engine.py:417` vs `:545-548`; re-verified by `python -m audit.continuity check`. The guard is an incidental array-length coincidence, not a physical condition.
- **Status:** NEW

### HIGH-25 — `s_g_avg` masking: a non-positive tangent slope silently becomes 100 % displacement efficiency
- **Severity:** HIGH
- **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:309-313`
- **Observed:** `s_g_avg = s_gf + (1 − f_gf)/max(slope_bt, EPSILON)` followed by `np.clip(s_g_avg, s_gf, 1 − sor)`. `slope_bt` comes from a finite-difference `argmax` over a discrete grid (`:305-307`) and can be zero/negative through noise; then `(1−f_gf)/1e-16` overflows and the clip returns `1 − S_or` — 100 % displacement efficiency, with no warning.
- **Expected:** assert the sign of `df_g/dS_g` (a mathematical property of the Corey model) rather than saturating.
- **Scientific Impact:** the immiscible limb's RF can jump to its physical maximum on a numerical artefact — anti-pattern class D inside the block whose gradient CRIT-07 otherwise correctly restored.
- **Evidence & Citation:** `audit_verify_5.py` §E4; probe `probe_saturation_masking` equivalent re-run 05-10-2026. Source: `analytical_models.py:305-313`. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §4.
- **Status:** NEW

### HIGH-26 — CO₂ properties mix a Peng-Robinson FVF with a correlation-based `Z` in one mixture
- **Severity:** HIGH
- **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:375-400`
- **Observed:** the CO₂ leg uses the PR-based `calculate_co2_fvf_rb_per_mscf`, the hydrocarbon leg uses `5.035·z_hc·T_R/P` with `z_hc` from Papay, and the two are combined by mole-weighted density with a linear `Z` blend. CRIT-04's fix corrected the constant but left two incompatible definitions of "CO₂ FVF" coexisting.
- **Expected:** a mixture `B_g` from one consistent EOS, or an explicit cited mixing rule.
- **Scientific Impact:** `bg_dynamic` feeds produced-gas voidage (`:412-414`), VRR (`:417`) and `dp` (`:455`); the result is not a property of any real gas.
- **Evidence & Citation:** Source read of `pvt_state.py:375-400`; PR path confirmed present via `_setup_pr_eos_co2` at `:106`. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §4.
- **Status:** NEW

### MED-17 — `monthly_oil_stb` is length-1 and all zeros

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:671`
- **Observed:** `profile_result.get("monthly_oil_stb", np.zeros(1))` — measured **n = 1, sum = 0.0**, while `yearly_oil_stb` correctly sums to 433 959 STB.
- **Expected:** populate the monthly series or remove the key.
- **Scientific Impact:** consumers resolving monthly resolution (MED-10) read zeros; `summary_monthly.csv` will be empty.
- **Evidence & Citation:** Measured 05-10-2026 on the default 181-step run: `monthly_oil_stb` `n=1, sum=0.0` while `yearly_oil_stb` sums to 433 959 STB. `audit_verify_9.py`. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

### MED-18 — The `kv <= 1` Koval branch is unreachable

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/analytical_models.py:561-562`
- **Observed:** `kv = max(kv, 1.0 + EPSILON)` immediately precedes `if kv <= 1.0 + 1e-6:` ⇒ dead.
- **Expected:** delete, or reorder the guard. (`M = 1` is in fact handled correctly by the general branch — measured continuity 7.5e-3.)
- **Scientific Impact:** Dead code that advertises a limiting case the model cannot reach; an agent reading the branch list would believe `M = 1` has special handling when in fact the general branch covers it.
- **Evidence & Citation:** Source read of `analytical_models.py:561-562`; continuity probe `probe_koval_sensitivity_in_config` confirms the general branch handles `M = 1` correctly (max jump 7.5e-03), so the defect is dead code only. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

### MED-19 — `MiscibleSurrogate` RF multiplied by a new `e_v`, and the RF clip raised in the same edit

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/analytical_models.py:199-201`
- **Observed:** `e_v = 1/(0.8 + 0.2·gravity_factor)` appears in no textbook; in the same un-audited edit the RF clip was raised from `0.80` to `0.85`.
- **Expected:** cite the vertical-sweep basis for `e_v`; justify the new ceiling independently.
- **Scientific Impact:** two changes to the same physical bound in one edit; the ceiling change alone raises achievable RF by up to 6 %.
- **Evidence & Citation:** Source read of `analytical_models.py:199-203`; the clip change from 0.80 to 0.85 is visible in the same un-audited edit (`git diff`). agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

### MED-20 — `annual_water_stb` reporting key deleted with no replacement

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `core/optimisation_engine.py:852`
- **Observed:** `"annual_water_stb": annual_water` was removed in the remediation diff and not re-added under another name.
- **Expected:** confirm no consumer reads it, or restore it.
- **Scientific Impact:** Silent reporting-key loss: any consumer reading `annual_water_stb` now reads nothing, and no test covers the key's removal.
- **Evidence & Citation:** Visible in `git diff -- core/optimisation_engine.py`: `"annual_water_stb": annual_water` removed with no replacement key. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

### MED-21 — The $100/t remediation floor (HIGH-13) survives the remediation

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `core/objectives/wrapper.py:130`
- **Observed:** `remediation_cost = max(carbon_tax, 100.0) * leaked_tonnes` is unchanged.
- **Expected:** use the configured carbon price.
- **Scientific Impact:** any user carbon price below $100/t is silently floored.
- **Evidence & Citation:** Source read `wrapper.py:130`; HIGH-13 in the register records the same line as open, and it is unchanged by the 05-10-2026 remediation. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

### MED-22 — Dead second leakage model still present in the objective wrapper

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `core/objectives/wrapper.py:112-121`
- **Observed:** now unreachable (see HIGH-23) because `total_leakage_tonne` is present; it reads `profiles["leakage_rate_fraction"]`, a key the engine never emits.
- **Expected:** delete, and reconcile with `CO2StorageParameters.leakage_rate_fraction`.
- **Scientific Impact:** A misleading second leakage model remains in the objective wrapper while the real one is dead.
- **Evidence & Citation:** Unreachability proved by measurement: `wrapper.py:108-111` always takes the first branch because `total_leakage_tonne` is present (value 0.0), so `:112-121` never executes. `audit_verify_9.py`. agent_wiki/audit/simulation_run_audits/05-10-2026_remediation_verification_round2/audit.md §5.
- **Status:** NEW

---

## 3. MEDIUM findings

| ID | Cat | Location | Observed → Expected → Impact | Evidence | Status |
|---|---|---|---|---|---|
### MED-01 — Five different "tonnes/MSCF" constants (spread **1.0%**) plus a comment claiming `0.1234 lb/ft³` and `...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `data_models.py:1858-1860` (0.05254), `pvt_state.py:41-42` (1.838 kg/m³, 0.05295), `profile_generator_fast.py:794` (0.05297), `surrogate_models.py:26`, `analytical_models.py:37`, `wrapper.py:90` (0.053)
- **Observed:** Five different "tonnes/MSCF" constants (spread **1.0%**) plus a comment claiming `0.1234 lb/ft³` and `0.056 t/MSCF` (both wrong: 1.838 kg/m³ = 0.115 lb/ft³ ≈ 0.0520 t/MSCF).
- **Expected:** one authoritative constant with a cited standard condition.
- **Scientific Impact:** 1% mass-balance bias, and an auditable comment that is false.
- **Evidence & Citation:** Grep output above; `analytical_models.py:35-37`
- **Status:** NEW

### MED-02 — Docstring advertises Span–Wagner; only Peng–Robinson is implemented (grep: no Span–Wagner code). `B_CO...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `pvt_state.py:8`, `:19` (docstring), `:175-176`, `:201`
- **Observed:** Docstring advertises Span–Wagner; only Peng–Robinson is implemented (grep: no Span–Wagner code). `B_CO2 = 327.362/ρ` assumes 52.046 kg/MSCF (exact ≈ 52.6 kg ⇒ ~1.1% high). No volume translation (PR-78 typically 5–10% low on dense CO₂ ρ).
- **Expected:** cite PR only, or implement Span–Wagner; state the molar basis.
- **Scientific Impact:** ~1–10% bias in CO₂ reservoir-volume and tonnage.
- **Evidence & Citation:** `pvt_state.py:121-161` (PR only); arithmetic 1000 scf × 1.838 kg/m³
- **Status:** NEW

### MED-03 — Radial form `T = 0.001127·k·h/(μ·ln(D/r_w))` mixes the **linear** Darcy constant with a radial logarit...

- **Severity:** MEDIUM
- **Category:** `MATHEMATICAL`
- **Location:** `well_mechanics.py:197`, `:220` (const at `:28`)
- **Observed:** Radial form `T = 0.001127·k·h/(μ·ln(D/r_w))` mixes the **linear** Darcy constant with a radial logarithm ⇒ **2π = 6.28× too low**.
- **Expected:** `0.00708` (`DARCY_FIELD_CONSTANT`, defined at `:27` and correctly used at `:85`).
- **Scientific Impact:** inter-well connectivity understated 6.3× wherever used.
- **Evidence & Citation:** `well_mechanics.py:26-28`, `:85`, `:220`; consumers are UI-only (MED-05)
- **Status:** NEW

### MED-04 — Invalid input (`k ≤ 0`, `h ≤ 0`, …) returns `1.0` silently (PI in STB/d/psi)

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `well_mechanics.py:65-66`, `:113-114`
- **Observed:** Invalid input (`k ≤ 0`, `h ≤ 0`, …) returns `1.0` silently (PI in STB/d/psi).
- **Expected:** raise or return `0.0` with a log.
- **Scientific Impact:** a sentinel 1.0 is summed into `j_peaceman_inj_base` (`surrogate_engine.py:296-309`) and dilutes the index. Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-05 — Relative permeability and inter-well transmissibility are **not in the evaluation path**: the engine c...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `relative_permeability.py` (all), `well_mechanics.py:184-221`, `surrogate_models.py:371`
- **Observed:** Relative permeability and inter-well transmissibility are **not in the evaluation path**: the engine computes saturations by material balance (`surrogate_engine.py:376-379`) with no `k_rel`, no fractional flow, no `T_ij`. Consumers of rel-perm are UI/dashboard only (`ui/data_management_widget.py`, `ui/widgets/model_evaluation_dashboard.py`); `calculate_storage_efficiency` is imported only by `tests/test_physics_validation.py`.
- **Expected:** document these as *diagnostic* modules (they are currently described as active physics).
- **Scientific Impact:** false confidence that Corey rel-perm governs results. Repo-wide import grep
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-06 — `so_norm = (so − s_orw)/denom_w` uses the **water** denominator even when `s_org ≠ s_orw` (the normal...

- **Severity:** MEDIUM
- **Category:** `MATHEMATICAL`
- **Location:** `relative_permeability.py:55`
- **Observed:** `so_norm = (so − s_orw)/denom_w` uses the **water** denominator even when `s_org ≠ s_orw` (the normal case, `:43-44`).
- **Expected:** normalize by `1 − s_wc − s_org`.
- **Scientific Impact:** gas/oil relative-permeability endpoints mis-scaled. Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-07 — Two defects. (a) `total_violation = 0.0` is hard-coded at `:1246` with every real check commented out...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `optimisation_engine.py:1209-1256` (fn `_calculate_adaptive_penalty`), call site `:1783`; `death` branch `:1225-1230`
- **Observed:** Two defects. (a) `total_violation = 0.0` is hard-coded at `:1246` with every real check commented out (`:1232-1253`) ⇒ `if total_violation <= 0: return 0.0` at `:1255-1256` fires unconditionally, so the **single** call site at `:1783` always receives `0.0`. (b) The advertised `"death"` constraint-handling method is literally `pass` (`:1225-1230`) — selecting "death penalty" in `constraint_handling_method` changes nothing.
- **Expected:** implement the checks or delete the API.
- **Scientific Impact:** a "static/adaptive/death penalty" API advertised to the GA is inert, and `penalty_factor`/`constraint_handling_method` are dead knobs.
- **Evidence & Citation:** Grep (1 call site at `:1783`, always returns 0.0); source read of `:1209-1256`
- **Status:** NEW

### MED-08 — `is_feasible` is computed and returned, then only logged; the penalty is applied separately at `:1771-...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `optimisation_engine.py:1531`, `:1560-1562`
- **Observed:** `is_feasible` is computed and returned, then only logged; the penalty is applied separately at `:1771-1772`.
- **Expected:** use the flag to gate/prune or drop it.
- **Scientific Impact:** dead branch that looks like constraint handling. Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-09 — Module-level `FAILURE_PENALTY = -1e12` is shadowed inside the wrapper by `self.advanced_engine_params....

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `optimisation_engine.py:96` vs `:1543`; value `data_models.py:1709`
- **Observed:** Module-level `FAILURE_PENALTY = -1e12` is shadowed inside the wrapper by `self.advanced_engine_params.failure_penalty`; both are `-1e12` **today**, so behaviour diverges only if a user changes `failure_penalty`.
- **Expected:** single source of truth.
- **Scientific Impact:** latent split-brain penalty scale. Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-10 — Valid `time_resolution` values are `weekly/monthly/quarterly/yearly`; profiles only ever expose `yearl...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `optimisation_engine.py:841-896`, `:1611`, `:1652`; `data_models.py:1262`, `:1273`; `profile_generator_fast.py:103`
- **Observed:** Valid `time_resolution` values are `weekly/monthly/quarterly/yearly`; profiles only ever expose `yearly_*` and `monthly_*`; engine time base is **monthly** by default while `OperationalParameters` defaults to **yearly**; fallbacks use the non-existent `"daily"`.
- **Expected:** one resolution vocabulary, propagated.
- **Scientific Impact:** `weekly`/`quarterly` runs silently read empty arrays (see CRIT-08 for the consequence). Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-11 — `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = base_injection_rate × mobility_rat...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `data_models.py:913`, `:917`; `profile_generator_fast.py:1111-1115`
- **Observed:** `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = base_injection_rate × mobility_ratio_factor(0.001)` is compared against `high_mobility_threshold = 2.0` labelled a *mobility ratio* ⇒ the adaptive WAG logic triggers for any rate > 2000 MSCFD (dimensionally meaningless).
- **Expected:** document units (rb/scf) and compare a true mobility ratio.
- **Scientific Impact:** WAG enhancement is decided by injection rate, not by mobility.
- **Evidence & Citation:** Source; CRIT-11
- **Status:** RESOLVED

### MED-12 — `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil + 0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000), 0.05...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `surrogate_engine.py:352`, `:354`, `:356`
- **Observed:** `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil + 0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000), 0.05, 0.95)` — solubility/vapor-fraction surrogates with no cited source (1000 MSCF time constant, 0.55 mass factor, 0.135 t/STB).
- **Expected:** mark `UNKNOWN — EVIDENCE REQUIRED` and tie to a flash calculation.
- **Scientific Impact:** `x_co2/y_co2` drive all PVT (B_o, μ, B_g mixture). Source +.
- **Evidence & Citation:** `audit/parameter_provenance.csv`
- **Status:** NEW

### MED-13 — repo-wide Ruff **3244** violations over **191** files (re-run 04-10-2026 after `a68fc35`; top: UP006 8...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `repo-wide`
- **Observed:** repo-wide Ruff **3244** violations over **191** files (re-run 04-10-2026 after `a68fc35`; top: UP006 857, W293 686, UP045 426, **F401 329 unused imports**, I001 253, **F841 85 unused locals**, **F821 6**, F811 7 — at audit baseline: 3132 / 192 / F401 345 / F841 77 / F821 5 / F811 5); vulture **41** dead-code candidates (04-10-2026 run; a re-run after `a68fc35` timed out on `RecursionError`, so 41 is the last verifiable figure); coverage **37 %** repo-wide (`audit/runtime/coverage.xml`, denominator 36 062 statements); pytest **`0 failed / 333 passed / 23 skipped`** (re-run 04-10-2026; was `4 failed / 329 passed / 23 skipped`); jscpd not installed (duplicate detection not run).
- **Expected:** triage F401/F841/F821 first (they hide real defects), keep style noise separate.
- **Scientific Impact:** F401/F841 mask dead physics (CRIT-13) and F821 hides runtime `NameError`s (HIGH-11, HIGH-19; HIGH-10 resolved). `audit/ruff_output.json` (UTF-8, re-run), `audit/code_quality/ruff_report.json` (UTF-16, baseline).
- **Evidence & Citation:** `dead_code_candidates.txt`, `audit/runtime/coverage.xml`, `pytest_output.txt`
- **Status:** NEW

### MED-14 — If no simulation engine is available the code silently falls back to `ProductionProfiler` (different p...

- **Severity:** MEDIUM
- **Category:** `SOFTWARE`
- **Location:** `optimisation_engine.py:905-935`
- **Observed:** If no simulation engine is available the code silently falls back to `ProductionProfiler` (different physics) and then clamps RF into [0,1] *after the fact* (`:930-935`).
- **Expected:** fail loudly, or mark results as `engine_type = profiler` everywhere downstream.
- **Scientific Impact:** two physics in one result namespace. Source
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### MED-15 — `agent_wiki/README.md:35`, `source_of_truth_map.md:16`, `source_of_truth.md:26`, `common_pitfalls.md:3...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `The wiki documents `_calculate_engine_npv()` and `_calculate_co2_purchased_recycled()` as the source of truth for economics. **Neither function exists anywhere in the repository** (grep for 'def _calculate_engine_npv', 'def _calculate_co2_purchased_recycled', 'def _solve_pressure_ode' → 0 hits). Real code: purchased/recycled at `surrogate_engine.py:572-596`; NPV at `surrogate_models.py:507-530`. → wiki must match code. → agents following the wiki edit a non-existent function and believe `economic.py` is dead for the wrong reason.`
- **Observed:** `agent_wiki/README.md:35`, `source_of_truth_map.md:16`, `source_of_truth.md:26`, `common_pitfalls.md:34`, `code/inventory.md:26`, `execution_flow.md:81-82`, `code/functions/simulation_functions.md`, `code/classes/surrogate_classes.md`, `validation/conservation.md` The wiki documents `_calculate_engine_npv()` and `_calculate_co2_purchased_recycled()` as the source of truth for economics. **Neither function exists anywhere in the repository** (grep for 'def _calculate_engine_npv', 'def _calculate_co2_purchased_recycled', 'def _solve_pressure_ode'
- **Expected:** 0 hits). Real code: purchased/recycled at `surrogate_engine.py:572-596`; NPV at `surrogate_models.py:507-530`.
- **Scientific Impact:** wiki must match code.
- **Evidence & Citation:** agents following the wiki edit a non-existent function and believe `economic.py` is dead for the wrong reason. Grep; corrected during this round (see wiki edits)
- **Status:** NEW

### MED-16 — `agent_wiki/verification/test_matrix.md:5`, `:18-46`; `agent_wiki/architecture/overview.md:71`; `agent...

- **Severity:** MEDIUM
- **Category:** `PROVENANCE`
- **Location:** `repo-wide`
- **Observed:** `agent_wiki/verification/test_matrix.md:5`, `:18-46`; `agent_wiki/architecture/overview.md:71`; `agent_wiki/README.md:29` The master verification matrix is stale on four counts. (a) It declares **"42 test items across 16 subdirectories"**; `pytest --collect-only tests/scientific` returns **36 items in 15 subdirectories**. (b) **6 of its 40 listed tests no longer exist anywhere in `tests/`**: `test_co2_density_thermal_expansion`, `test_cubic_eos_z_factor_bounds`, `test_phase_label_assignment`, `test_peng_robinson_fugacity_equation_structure`, `test_corey_relative_permeability_bounds`, `test_bg_discrepancy_between_modules` (grep = 0 hits each) — five of them cite `unified_engine/…`. (c) `test_koval_fractional_flow_mobility_inversion` was **renamed** to `..._monotonicity` (0 hits for the old name) and 2 existing tests (`test_profile_generator_co2_breakthrough_gas_rate_increases_with_mobility`, `test_alston_impurity_mmp_trend`) are **unlisted**. (d) `overview.md:71` and `README.md:29` state the legacy engines were *"relocated into `deprecated/`"* — **neither `deprecated/` nor `core/unified_engine/` exists** (`unified_engine` is referenced 51× in the wiki, 0× as a real path).
- **Expected:** regenerate the matrix from `--collect-only` and stop citing deleted trees.
- **Scientific Impact:** readers conclude that six thermodynamic assertions are being enforced when no such test exists. `pytest --collect-only tests/scientific` (36 items); 7 greps returning 0; `Test-Path deprecated`, `core/unified_engine` = False
- **Evidence & Citation:** See Observed.
- **Status:** NEW

---

## 4. LOW findings

| ID | Cat | Location | Observed → Expected → Impact | Status |
|---|---|---|---|---|
### LOW-01 — `y_co2` is clipped to a **minimum of 0.05** even at `cum_inj = 0`, so the gas phase always contains ≥...

- **Severity:** LOW
- **Category:** `PHYSICAL`
- **Location:** `surrogate_engine.py:356`
- **Observed:** `y_co2` is clipped to a **minimum of 0.05** even at `cum_inj = 0`, so the gas phase always contains ≥ 5% CO₂ before any injection.
- **Expected:** 0.0 when nothing is injected.
- **Scientific Impact:** negligible pre-injection bias in mixture B_g/μ.
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### LOW-02 — Dynamic MMP is recomputed and **overwrites** `params["mmp"]` with Cronquist on every call, wrapped in...

- **Severity:** LOW
- **Category:** `SOFTWARE`
- **Location:** `surrogate_engine.py:930-938`
- **Observed:** Dynamic MMP is recomputed and **overwrites** `params["mmp"]` with Cronquist on every call, wrapped in `except`
- **Expected:** `params["mmp"]` falls back to `default_mmp_fallback`. User `mmp` override survives only because `sim_kwargs["mmp"]` is applied via `params.update(kwargs)` **after** `_build_params_dict` (`optimisation_engine.py:647`, `surrogate_engine.py:158`).
- **Scientific Impact:** explicit precedence documented.
- **Evidence & Citation:** fragile ordering; reordering the kwargs update silently changes physics.
- **Status:** NEW

### LOW-03 — `np.roots` per timestep for a cubic EOS (3 roots, eigen-solver) in a hot loop

- **Severity:** LOW
- **Category:** `SOFTWARE`
- **Location:** `pvt_state.py:143`
- **Observed:** `np.roots` per timestep for a cubic EOS (3 roots, eigen-solver) in a hot loop.
- **Expected:** closed-form trigonometric solution for the depressed cubic.
- **Scientific Impact:** performance only (no scientific impact).
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### LOW-04 — `MSCF_PER_TONNE` imported but unused (vulture, 90% confidence) alongside a live `CO2_TONNE_PER_MSCF`

- **Severity:** LOW
- **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:37`
- **Observed:** `MSCF_PER_TONNE` imported but unused (vulture, 90% confidence) alongside a live `CO2_TONNE_PER_MSCF`.
- **Expected:** remove or use one constant.
- **Scientific Impact:** two reciprocal constants invite unit errors (see MED-01).
- **Evidence & Citation:** See Observed.
- **Status:** NEW

### LOW-05 — RF > 1.0 is silently clamped to 1.0 with a warning after the fact instead of preventing `rf_max_physic...

- **Severity:** LOW
- **Category:** `NUMERICAL`
- **Location:** `optimisation_engine.py:930-935`
- **Observed:** RF > 1.0 is silently clamped to 1.0 with a warning after the fact instead of preventing `rf_max_physical` violations upstream (HIGH-01).
- **Expected:** raise/flag as a model failure.
- **Scientific Impact:** hides upstream cap bugs.
- **Evidence & Citation:** See Observed.
- **Status:** NEW

---

## 5. Cross-check against the previous register (`audit/scientific_flaws/scientific_flaws.csv`, 18 items)

| Prior ID | Prior severity | Verdict this round | Note |
|---|---|---|---|
| SCI-FLAW-01 | CRITICAL | **RESOLVED** (formula replaced) | The `koval_factor/(koval_factor+(M−1)·0.5)` inversion is gone; current code `profile_generator_fast.py:958-969` gives `f_g` **increasing** with M (K = h·E, E = (0.78+0.22·M^0.25)^4), and `:466` post-BT acceleration also increases with M. Residual issue: `s_ref = 0.40` and `koval_factor_multiplier` are un-cited constants (→ parameter register). |
| SCI-FLAW-02 | CRITICAL | **CONFIRMED elsewhere** | Sign defect still present, now also documented for the **active** `pvt_state.py` (CRIT-03). `data_integration_engine.py` not re-verified (module not in active path). |
| SCI-FLAW-03 | HIGH | not re-verified | `core/data_integration_engine.py` — outside active path this round. |
| SCI-FLAW-04 | HIGH | not re-verified | `core/unified_engine/` — classified dormant by the wiki source-of-truth map. |
| SCI-FLAW-05 | CRITICAL | not re-verified (OPEN) | Active module (`optimisation_engine.py`, `profile_generator_fast.py`); retain as open. |
| SCI-FLAW-06 | CRITICAL | **CONFIRMED** | See CRIT-12 / CRIT-02; line refs updated (`surrogate_engine.py:513-522`, `:490-510`). |
| SCI-FLAW-07 | HIGH | **CONFIRMED** | `wrapper.py:213` returns **tonne/STB** while `optimisation_engine.py:3929`, `:3944` label the same key **MSCF/STB** (`run_exporter.py:376-377` exposes both). ~18.9× unit ambiguity on an optimizable objective. |
| SCI-FLAW-08 | CRITICAL | not re-verified | `core/unified_engine/` (dormant). |
| SCI-FLAW-09 | HIGH | not re-verified | `core/simulation/recovery_models.py` — dormant behind `RECOVERY_MODELS_AVAILABLE = False` (see HIGH-11). |
| SCI-FLAW-10 | HIGH | not re-verified | Same dormant module. |
| SCI-FLAW-11 | MEDIUM | **CONFIRMED, locations stale** | Content correct; actual sites `analytical_models.py:814` and `surrogate_engine.py:509` (HIGH-01). |
| SCI-FLAW-12 | HIGH | **STALE** | `B_GAS_RB_PER_MSCF` is now **1.0 at `optimisation_engine.py:85`**, not `5.0 at :98`; `calculate_co2_fvf_rb_per_mscf` accepts both `pressure_psi` and `p_psia`/`t_f` (`pvt_state.py:163-170`) so the reported `TypeError` no longer reproduces. The *underlying* Bg inconsistency is real but is now documented as CRIT-04 (a different, larger defect). |
| SCI-FLAW-13 | MEDIUM | **RESOLVED** | Re-verified: user `mmp` reaches `params.update(kwargs)` after `_build_params_dict` and survives the Cronquist overwrite (LOW-02 notes the fragility). |
| SCI-FLAW-14 | HIGH | not re-verified | `analysis/material_balance.py` (post-processing, not in the optimizer loop). |
| SCI-FLAW-15 | MEDIUM | **CONFIRMED, lines stale** | Still `profile_generator_fast.py:1000-1005` (was `:983-986`): produced CO₂ = `injection[i]·(1−total_trapping)·f_g·growth`, i.e. instantaneously tied to injection; shut-in ⇒ zero CO₂ production. |
| SCI-FLAW-16 | HIGH | **CONFIRMED, formula description stale** | Cliff is 48.3% for `E_A = 0.517 − 0.072·log10(M)` (HIGH-02); prior text quotes a different formula. |
| SCI-FLAW-17 | HIGH | **CONFIRMED** | `surrogate_models.py:238-241` (HIGH-15), and the test that asserts it. |
| SCI-FLAW-18 | CRITICAL | not re-verified | `core/unified_engine/physics/eos/` (dormant). |

---

## 6. Findings explicitly **not** raised (verified correct — do not "fix")

These were checked numerically or by dimensional analysis and found sound; recording them prevents false positives in later rounds:

1. **Material-balance increment** `dp = q·dt/(V_p·c_t + J_eff·dt)` — dimensionally consistent (bbl·psi), correct implicit form for a tank with well exchange (`surrogate_engine.py:455`).
2. **Gravity-number constant** `4.3948e-5` (`analytical_models.py:793`) — matches `Δρ·g·k/(μ·φ)` in field units.
3. **Capillary-number factor** `3.5e-6` (`analytical_models.py:752`) — consistent with field-unit N_c.
4. **Standing B_o coefficients** and **Beggs–Robinson a/b** in `pvt_state.py` (structure verified; only the missing bubble point is flagged, CRIT-03).
5. **Peaceman horizontal/vertical index** `0.00708·k·h/ln(...)` (`well_mechanics.py:26-27`, `:85`) — the 2π factor is present where it must be (only the *inter-well* function omits it, MED-03).
6. **Koval effective efficiency** `E_eff = (0.78 + 0.22·M^0.25)^4` — normalized so that `E_eff(M=1) = 1.0` (0.78+0.22 = 1.0), as required.
7. **Koval throughput branches** at `analytical_models.py:187-192` are continuous through `K → 1` (checked analytically).
8. **Peng–Robinson coefficients** (`pvt_state.py:103-115`: ω = 0.225, T_c = 304.13 K, P_c = 7.376e6 Pa) — standard CO₂ values; root selection above P_c uses the smallest root (`:150-155`), which is the correct dense-phase branch.
9. **Harmonic (mole-basis) mixing of B_g** at `pvt_state.py:385` — `1/Σ(y_i/B_i)` is the correct volumetric mixing rule when each `B_i` is on a molar basis; the *inputs* are wrong (CRIT-04), the *rule* is not.
10. **Geomechanical stress path and slip tendency** (`geomechanics_fault.py:98`, `:129-132`, `:188-198`) — normal/shear resolution on a fault plane and the Biot-coupled `p_crit` are standard forms.
11. **VRR/voidage assembly** `q_inj = q_CO₂·B_CO₂ + q_w·B_w` (`surrogate_engine.py:406`) — correct in form.
12. **MMP user override** survives the Cronquist recomputation (LOW-02 notes the ordering dependency).
13. **`F821` `prev_field` at `petrophysical_distribution.py:404`** is a lint artifact, not a bug: `prev_field = field` is bound at `:405` on the preceding iteration and the read at `:404` is guarded by `if k > 0` (`:402`), so the name is always bound when it is used. → do **not** "fix" by initializing a dummy variable that would change the recursion; if the warning must go, restructure the loop (with a test) rather than paper over it. (Contrast with the two genuine `F821`s in HIGH-19, which have no binding anywhere.)

---

## 7. Register summary

| Severity | IDs |
|---|---|
| CRITICAL (13) | CRIT-01 … CRIT-13 |
| HIGH (19) | HIGH-01 … HIGH-19 |
| MEDIUM (16) | MED-01 … MED-16 |
| LOW (5) | LOW-01 … LOW-05 |

| Category | CRIT | HIGH | MED | LOW | Total |
|---|---:|---:|---:|---:|---:|
| MATHEMATICAL | 3 | 3 | 2 | 0 | **8** |
| PHYSICAL | 3 | 5 | 0 | 1 | **9** |
| NUMERICAL | 1 | 1 | 0 | 1 | **3** |
| SOFTWARE | 6 | 8 | 8 | 3 | **25** |
| PROVENANCE | 0 | 2 | 6 | 0 | **8** |
| **Total** | **13** | **19** | **16** | **5** | **53** |

**Category totals: 53 findings** (counts derived programmatically from the register's own `Severity`/`Category` fields by `evidence_scripts/v_recount.py`, re-run 04-10-2026 after HIGH-19 was added — the script parses all 53 IDs with 0 missing). Status distribution: **RESOLVED 17, OPEN 36** (CRIT-01..13, HIGH-01, HIGH-10, HIGH-19, MED-11 resolved). Note the deliberately uneven distribution: the majority are *software/data-flow* defects that **silence** physics (dead constraints, inert genes, key mismatches), while the physics defects that remain active are concentrated in PVT (`B_g`, Z, `B_o`) and in the recovery-model floors/limbs. These two populations require different remediation strategies and must not be conflated into one "quality" number.

**No composite accuracy score is issued.** Predictive validity is assessed narratively in `phd_audit.md` §6 and currently rates **NOT ESTABLISHED** for the active `hybrid` path (no experimental or benchmark evidence exists in the repository for that configuration; the benchmark/validation material present targets `phd_hybrid` and dormant engines — see HIGH-09).
