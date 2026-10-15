# Round 2 — Forensic Verification of the 05-10-2026 Remediation

**Audit round:** 05-10-2026 (remediation verification / adversarial re-audit)
**Parent register:** [`audit/scientific_flaws.md`](../../../audit/scientific_flaws.md)
**Method:** adversarial re-audit of uncommitted remediation code. No source was modified by this
round (audit-only rule honoured; the findings below are *not* fixed).
**Auditor role:** Senior Reservoir Simulation Engineer / Applied Mathematician / Forensic Code Auditor

---

## Verdict

> ### `FLAGGED`
>
> The 05-10-2026 remediation **genuinely fixed 6 of the 17 findings it claimed to resolve** and
> **introduced 11 new defects**, of which **3 are CRITICAL**. The remediation was applied *before*
> being audited, and the register was updated to say "RESOLVED" on the strength of the edit rather
> than on the strength of evidence. This round reverses 6 RESOLVED marks, corrects 3 as
> PARTIALLY RESOLVED, and registers the new findings as **CRIT-14 … CRIT-21, HIGH-20 … HIGH-26**.
>
> **Predictive validity remains `NOT ESTABLISHED`.** The optimizer's primary objective (`npv`) is
> now internally consistent with the reported recovery factor (CRIT-02 correctly repaired) — but it
> is a **truncated** cash-flow model (HIGH-22), it rewards leaked CO₂ (HIGH-23), and the default
> configuration pins the Koval sweep at its clip so that the entire viscosity-contrast mechanism is
> inert (CRIT-15, CRIT-16).

---

## Why this round exists

The parent register's header reads:

> *"Register refresh: 05-10-2026 — 17 findings RESOLVED (CRIT-01..CRIT-13, HIGH-01, HIGH-10,
> HIGH-19, MED-11)."*

The remediation is **uncommitted** (`git status`: 13 modified files under `core/`, `ui/`, `tests/`),
so at the time the register was refreshed no evidence existed that any of it worked. Every
"RESOLVED — Remediated in `file:line`" line was an *intention*, not a *verification*.

This round therefore treats every one of those 17 claims as an unverified assertion and tests it.

---

## 1. Environment and toolchain record

| Item | Value |
|---|---|
| Branch HEAD at audit start | `6c10c9e docs(audit): refresh register to 53 findings` |
| Working tree | **dirty** — 19 modified, 5 untracked |
| Python | 3.12.6 (`.venv`) |
| `pytest tests/ -q` | **335 passed, 23 skipped, 0 failed** in 303 s (register claims 333 — stale) |
| Coverage (repo-wide) | **57 %** of 7 791 statements (register claims 37 % / 36 062 stmts — stale basis) |
| Ruff / vulture / jscpd | **not re-run this round**; the parent round's `audit/ruff_output.json` is 2.7 MB and cited as-is |
| Evidence scripts | `C:\Windows\Temp\opencode\audit_verify_{5,6,7,8,9}.py` (kept **outside** the repo per the audit-only rule) |
| Mirrored evidence | this folder's `evidence_scripts/` |

---

## 2. Verification matrix — the 17 "RESOLVED" claims

| ID | Claim | Verdict | Basis |
|---|---|---|---|
| CRIT-01 | HCPVI corrected to a dimensionless, pressure-dependent pore-volume ratio | **CONFIRMED** | algebra + measurement, §3.1 |
| CRIT-02 | NPV derived from the simulated annual streams, consistent with RF | **CONFIRMED (with HIGH-22, HIGH-23)** | `cum_oil == OOIP·RF` exactly; but cash flow is truncated |
| CRIT-03 | Bubble-point model with positive `c_o` above `P_b` | **PARTIALLY RESOLVED** | sign fixed; **HIGH-24** (two conflicting `c_o`), **HIGH-25** (C¹ kink at `P_b`) |
| CRIT-04 | `B_g` constant corrected to `5.035·Z·T_R/P` | **CONFIRMED** | `0.02827·1000/5.614583 = 5.0351` |
| CRIT-05 | Z-factor replaced with a genuine correlation | **CONFIRMED, but relabelled incorrectly** | Papay values are correct; code comment says "Hall-Yarborough"-class — now says Papay; see HIGH-26 on the *new* `c_g` |
| CRIT-06 | Koval sweep made continuous and monotone in M | **CONFIRMED in isolation, DEAD in practice** | monotone + continuous confirmed; but saturated at the clip in the default case → **CRIT-15** |
| CRIT-07 | Immiscible limb given a real gradient | **CONFIRMED** | distinct values 24/45, spread 0.304 (was 1/45, spread 0) |
| CRIT-08 | Plume-containment constraint can prune | **PARTIALLY RESOLVED** | plumbing now present; floor/threshold still unvalidated |
| CRIT-09 | Sandface / Class-VI penalty block reachable | **CONFIRMED** | de-indented out of the `if pressure_profile` guard |
| CRIT-10 | Leakage keys forwarded into constraint + objective | **CONFIRMED (wiring only)** | keys now read; but leakage is *still identically 0.0* → **HIGH-23** |
| CRIT-11 | WAG ×1000 unit scaling | **CONFIRMED** | — |
| CRIT-12 | Post-rescale material-balance re-synchronization | **REGRESSED** | the fix itself breaks saturation closure → **CRIT-17** |
| CRIT-13 | `gravity_factor`/`transition_alpha`/`transition_beta` wired in | **PARTIALLY RESOLVED — and partly harmful** | now *active*, but as unprincipled fudge factors → **CRIT-19**, **CRIT-20** |
| HIGH-01 | `rf_max_physical` normalized to OOIP | **CONFIRMED** | `(1−S_wi−S_or)/(1−S_wi)` is the correct OOIP-fraction form |
| HIGH-10 | `QIcon` import | **CONFIRMED** | — |
| HIGH-19 | `has_active_fault`, `QToolTip` | **CONFIRMED** | — |
| MED-11 | adaptive-WAG mobility comparison | **CONFIRMED** | — |

**Score: 11 CONFIRMED · 4 PARTIALLY RESOLVED · 1 REGRESSED · 1 CONFIRMED-but-inert.**

---

## 3. New CRITICAL findings

### CRIT-14 — The mobility-ratio override severs oil viscosity from recovery entirely

- **Severity:** CRITICAL · **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:170-172` (`MiscibleSurrogate`),
  `:275-277` (`ImmiscibleSurrogate`), `:412-414` (`BuckleyLeverettSurrogate`);
  source of the override: `core/engine_surrogate/surrogate_engine.py:976-977`
- **Observed:** the remediation added, in three separate recovery models,
  ```python
  if "mobility_ratio" in params and params["mobility_ratio"] is not None:
      m_eff = max(float(params["mobility_ratio"]), 1.0)      # analytical_models.py:170-172
  else:
      m_eff = max(viscosity_oil / max(mu_oe, EPSILON), 1.0)   # legacy physics path
  ```
  and `_build_params_dict` unconditionally supplies the key:
  ```python
  if "mobility_ratio" not in params:
      params["mobility_ratio"] = getattr(eor_params, "mobility_ratio", 2.5)   # :977
  ```
  `EORParameters.mobility_ratio` **exists** (`core/data_models.py:798`, default `5.0`), so the
  `2.5` fallback is dead and the value used is the **hard-coded constant 5.0**.
  Measured (`MiscibleSurrogate`, `hcpvi=1.5`, `v_dp=0.5`, `S_wi=0.25`, `S_or=0.25`):

  | `μ_o` (cP) | 0.5 | 1.0 | 2.0 | 5.0 | 20.0 | 100.0 |
  |---|---|---|---|---|---|---|
  | RF (new path, M=5 supplied) | 0.523509 | 0.523509 | 0.523509 | 0.523509 | 0.523509 | 0.523509 |
  | RF (legacy path, M computed) | 0.593831 | 0.591319 | 0.588364 | 0.583658 | 0.574406 | 0.559522 |

  The new path is **exactly flat** in oil viscosity — a 200× change in `μ_o` changes RF by
  **0.000000**.
- **Expected:** CO₂-EOR's dominant viscosity-reduction mechanism must enter the mobility ratio,
  $M = (\,k_{ro}/\mu_o\,)/(\,k_{rg}/\mu_g\,)$. `μ_o(P, x_CO₂)` is already computed by the PVT engine
  and already feeds the solvent-augmented `μ_oe`; it must not be discarded.
- **Scientific Impact:** the *entire* viscosity-contrast mechanism is removed from the miscible,
  immiscible **and** Buckley-Leverett limbs. The optimizer can no longer discover that a
  viscosity-reducing solvent improves recovery — the physical basis of CO₂ flooding. Recovery
  becomes a function of a single constant `5.0`.
- **Evidence & Citation:** `audit_verify_5.py` §B1/B2/B3. `core/data_models.py:798`.
  Note also the `max(..., 1.0)` floor deletes the *favourable* mobility regime $M<1$, which is
  precisely the regime CO₂ miscibility aims to create.
- **Status:** NEW

### CRIT-15 — Default configuration pins the Koval sweep at its 0.95 clip, so mobility ratio cannot affect recovery

- **Severity:** CRITICAL · **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:558-571` (`KovalSurrogate`);
  `core/engine_surrogate/surrogate_engine.py:946-968`
- **Observed:** `params["hcpvi"]` is built from a straight-line assumption:
  ```python
  total_inj_rb = eor_params.injection_rate * b_co2 * 365.25 * operational_params.project_lifetime_years
  params["hcpvi"] = total_inj_rb / max(pv_rb, 1.0)
  ```
  For the shipped defaults (5 000 MSCFD, 15 yr, `B_co2=0.4887 rb/MSCF`, OOIP 1 MMSTB, `S_wi=0.25`)
  this yields **HCPVI = 7.6928**. Measured sweep at that throughput:

  | M | 1.0 | 2.0 | 5.0 | 10.0 |
  |---|---|---|---|---|
  | sweep | 0.950000 | 0.950000 | 0.950000 | 0.950000 |

  The sweep-vs-M curve only becomes M-sensitive below HCPVI ≈ 5–6, and it reaches the 0.95 clip at
  HCPVI ≈ 6.0 (measured: 0.895 at 3.0, **0.950 at 6.0**).
- **Expected:** the optimiser must operate inside the sweep-sensitive regime. Published CO₂-EOR
  practice terminates miscible injection at HCPVI ≈ 0.5–3.0; HCPVI 7.7 is well beyond economic
  miscible-flood practice even before considering solvent recycle.
- **Scientific Impact:** in the default configuration the Koval term contributes a constant, and
  recovery is driven only by the clipped arithmetic upstream of it. Combined with CRIT-14 this
  removes both mobility and heterogeneity sensitivity: **the recovery model is nearly constant over
  the whole mobility-ratio axis**, so GA/BO selection pressure on these genes is arbitrary.
- **Evidence & Citation:** `audit_verify_5.py` §D4/§I, `audit_verify_6.py` §I. Root cause is that
  `hcpvi` is computed from `injection_rate × lifetime` and **ignores** the WAG schedule, shut-ins,
  facility availability and the compressor cap — i.e. it is not the throughput the simulator
  actually injects.
- **Status:** NEW

### CRIT-16 — `bo`/`b_co2` provenance guards are inert, so the user's own PVT inputs are silently discarded

- **Severity:** CRITICAL · **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:963-964`
- **Observed:**
  ```python
  bo    = getattr(reservoir_data, "bo_rb_per_stb", None) or pvt_init.calculate_oil_fvf_rb_per_stb(init_p, x_co2=0.0)
  b_co2 = getattr(reservoir_data, "bg_rb_per_mscf", None) or pvt_init.calculate_co2_fvf_rb_per_mscf(init_p)
  ```
  `ReservoirData` has **neither** attribute. Measured: `hasattr(rd,'bo_rb_per_stb') = False`,
  `hasattr(rd,'bg_rb_per_mscf') = False`. The real field is `ReservoirData.oil_fvf`
  (`core/data_models.py:433`), which in the audit fixture is `1.2` while the PVT engine returns
  `1.3053`.
- **Expected:** the guard must name the field that exists, or the user-supplied `oil_fvf` must be
  used directly. This is the **same `or`-default defect class as HIGH-04**, re-introduced.
- **Scientific Impact:** a user who has fitted live-oil FVF data cannot get it into HCPVI; the
  quantity that sets the sweep throughput is computed from a correlation the user cannot override.
  The guard reads as "respect user data if present" and silently does the opposite.
- **Evidence & Citation:** `audit_verify_6.py` §G.
- **Status:** NEW

### CRIT-17 — The CRIT-12 fix breaks saturation closure: `S_o + S_w > 1` on 27 % of timesteps and `S_g` is silently zeroed

- **Severity:** CRITICAL · **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:535-551`
- **Observed:** the new "post-rescaling material balance re-synchronization" block sets
  ```python
  pv_ref      = (ooip_val * bo_mean) / max(1.0 - swi_val, 0.05)   # :539  <- HCPV, not total PV
  vp_dynamic  = pv_ref * (1.0 + c_f_val * (pressure_profile - init_p))
  sat_oil_profile   = np.clip((remaining_oil_stb * bo_profile) / np.maximum(vp_dynamic, 1.0), 0, 1)   # :544
  sat_water_profile = np.clip(water_in_res_bbl / np.maximum(vp_dynamic, 1.0), 0, 1)                   # :546
  sat_gas_profile   = np.clip(1.0 - sat_oil_profile - sat_water_profile, 0.0, 1.0)                   # :547
  ```
  `pv_ref` is the **hydrocarbon** pore volume `PV/(1−S_wi)`, but `S_wi` is defined on the **total**
  pore volume. Measured on the default 181-step run:
  - `S_o + S_w > 1` on **49 / 181 timesteps (27.1 %)**
  - `max(S_o + S_w) = 1.042829`
  - `S_o + S_w + S_g ∈ [1.000000, 1.042829]` — the saturation sum **exceeds unity**
  - `S_g = 0` on 49 steps purely because `np.clip` saturates
- **Expected:** $\sum S = 1$ exactly at every timestep. With $S_{wi}$ on total PV and $B_o$ on the
  hydrocarbon inventory, the correct closure is
  $S_o = (N_{rem} B_o)/PV_{total}$ and $S_w = (PV_{total}S_{wi} + W_{inj} - W_{prod})/PV_{total}$.
- **Scientific Impact:** the reported three-phase saturation history violates the defining
  constraint of multiphase flow. Every saturation-derived quantity (gas saturation → CO₂ relative
  mobility, water cut, residual trapping, `V_DP`) is corrupted, and the corruption is **invisible**
  because a `np.clip` absorbs it. This is precisely the *Phase-3 anti-pattern class D* — clipping
  that hides an underlying physics bug.
- **Evidence & Citation:** `audit_verify_6.py` §J, `audit_verify_9.py`. Numeric: `PV_total =
  1 600 000 rb`, `PV_hcpv = 1 200 000 rb`, so `S_o(0) = 0.75·1.2/1.2 = 0.75` on HCPV but
  `0.75·1.2/1.6 = 0.5625` on total PV — a 25 % saturation bias at t = 0.
- **Status:** NEW (regression introduced by the CRIT-12 remediation)

### CRIT-18 — The new NPV is internally consistent but omits the hydrocarbon-gas revenue stream it computes

- **Severity:** CRITICAL · **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:622-651`, `:588`, `:606`, `:688`
- **Observed:** the remediation rewrote NPV from the simulated annual streams — a genuine repair of
  CRIT-02 (verified: `Σoil·dt == OOIP·RF == cumulative_oil_stb == Σ annual_oil_stb` to machine
  precision). But the revenue term is:
  ```python
  annual_rev = (annual_oil_stb * oil_price) + (annual_stored_tonne * co2_storage_credit)   # :636
  ```
  `annual_hc_gas_mscf` is allocated at `:588`, accumulated at `:606`, and published at `:688` — and
  then **never referenced again**. Measured on the default run: **216 810 MSCF** of hydrocarbon-gas
  sales over 15 years, i.e. ≈ $650 k at a conventional $3/MSCF, contributing **$0** to NPV.
  Produced CO₂ likewise generates no sale revenue.
- **Expected:** a CO₂-EOR cash flow contains oil revenue, **solution/associated gas sales**, CO₂
  sales on the produced stream, storage-credit revenue, and the matching opex for compression,
  recycling and gas handling. Every one of the streams the engine already computes should enter.
- **Scientific Impact:** `npv` is the **primary optimisation objective**. The optimiser therefore
  ranks candidates on a **truncated** cash-flow model. A candidate that produces more gas (and
  therefore more oil, via drawdown) can be ranked *worse* than one that does not, because the
  associated revenue is invisible to the objective.
- **Evidence & Citation:** `audit_verify_9.py`; `annual_hydrocarbon_gas_sales_mscf` present and
  non-zero in the returned profile dict.
- **Status:** NEW

### CRIT-19 — `gravity_factor` is now an active but unprincipled triple-purpose fudge multiplier

- **Severity:** CRITICAL · **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/analytical_models.py:199-201` (`MiscibleSurrogate`),
  `:331-333` (`ImmiscibleSurrogate`), `:802` (`PhDHybridSurrogate` `N_g`);
  wired at `core/engine_surrogate/surrogate_engine.py:928`, `:974-975`
- **Observed:** the remediation made a previously inert gene act on the physics three times:
  ```python
  e_v = float(np.clip(1.0 / (0.8 + 0.2 * gravity_factor), 0.1, 1.0))          # :200
  rf  = displacement_eff * (1.0 - s_wi) * e_v                                  # :201
  ...
  vertical_eff = (1.0 - (v_dp ** 0.7)) / max(0.8 + 0.2 * gravity_factor, 0.1)    # :332
  ...
  N_g = (perm_md * delta_rho * effective_angle * 4.3948e-5 * gravity_factor) / ...  # :802
  ```
  `gravity_factor` is **dimensionless, has no physical definition, no unit, and no citation**.
  Range `[0.5, 1.5]` (`core/data_models.py:1028-1029`), default `1.0`
  (`locked_gravity_factor`, `:1061`).
- **Expected:** gravity segregation is a *computed* quantity from $\Delta\rho$, dip angle,
  permeability and Darcy velocity — which is exactly what `N_g` at `:802` already computes.
  A second, independent, hand-set "gravity factor" multiplying RF and vertical sweep is a
  duplicate degree of freedom that can only be fitted.
- **Scientific Impact:** the optimiser now has a gene that scales recovery by up to
  $\pm20\%$ with **no physical content**, and it is applied **twice** in the hybrid path (once in
  each limb) plus once in `N_g`. This is the textbook *hidden calibration* pattern the audit
  mandate targets — a gene whose only possible justification is that it improved a fit.
- **Evidence & Citation:** `audit_verify_5.py` §B (`locked_gravity_factor = 1.0`,
  `min/max_gravity_factor = 0.5/1.5`); source reads above. Parent register **CRIT-13** correctly
  recorded this gene as *inert*; making it active converts a harmless dead gene into a live
  un-calibrated one.
- **Status:** NEW

### CRIT-20 — The miscibility weight is decoupled from composition and equals 0.5 exactly at the MMP

- **Severity:** CRITICAL · **Category:** `MATHEMATICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:474-481` (`HybridSurrogate`)
- **Observed:** the remediation replaced the composition-driven transition point with a fitted
  constant:
  ```python
  default_alpha = 0.95 + 0.05 * (c7_plus - 0.3)
  alpha = float(params.get("transition_alpha", params.get("alpha_base", default_alpha)))
  beta  = float(params.get("transition_beta", params.get("miscibility_window", 20.0)))
  ```
  `_build_params_dict` supplies `alpha_base = fitting_params.alpha_base` (`0.9750`), so the
  `c7_plus` branch is **dead**. Measured:

  | C₇⁺ | 0.1 | 0.2 | 0.3 | 0.5 |
  |---|---|---|---|---|
  | ω (legacy, α from C₇⁺) | 0.535244 | 0.535186 | 0.535122 | 0.534974 |
  | ω (new, α = α_base) | 0.534688 | 0.534688 | 0.534688 | 0.534688 |

  ω is now **exactly independent of composition**. And with α = 0.975, β = 20:
  `ω(P/MMP = 1.0) = 0.622`, `ω(P/MMP = 0.975) = 0.500`, `ω(P/MMP = 0.95) = 0.378`.
- **Expected:** the MMP is by definition the pressure at which the displacement becomes miscible;
  a sigmoid that weights **half** the recovery to the miscible mechanism *at* the MMP is not a
  miscibility model, it is a blend factor. Composition (C₇⁺, oil gravity, temperature) is what
  moves the miscibility window and must remain coupled to it.
- **Scientific Impact:** the miscible/immiscible blend — the central modelling decision of the whole
  surrogate — is now governed by two uncalibrated constants and is insensitive to the fluid.
- **Evidence & Citation:** `audit_verify_5.py` §C/§C1.
- **Status:** NEW

### CRIT-21 — CO₂ compressibility is a new hard-coded power law that contradicts the Peng-Robinson EOS in the same class

- **Severity:** CRITICAL · **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:415-419`
- **Observed:** while fixing the sign of `c_g`, the remediation introduced
  ```python
  if p > 1200.0:
      cg_co2 = float(np.clip(1.5e-4 * (2000.0 / p) ** 0.8, 2e-5, 5e-4))
  else:
      cg_co2 = 1.0 / max(p, 14.7)
  ```
  with **no literature citation**, a hard floor at `2e-5` and a ceiling at `5e-4`. The same class
  already implements a full Peng-Robinson EOS (`_setup_pr_eos_co2`, `:106`, `calculate_co2_fvf_rb_per_mscf`),
  so `c_g` can be derived exactly as `-(1/B)·dB/dP`. Measured discrepancy:

  | P (psi) | 1500 | 2000 | 2500 | 3000 |
  |---|---|---|---|---|
  | `c_g` from PR `B_co2(P)` | 1.5065e-3 | 1.2464e-3 | 5.5534e-4 | 2.8112e-4 |
  | hard-coded power law | 1.8882e-4 | 1.5000e-4 | 1.2548e-4 | 1.0845e-4 |
  | **ratio** | **7.98×** | **8.31×** | **4.43×** | **2.59×** |

- **Expected:** one EOS, one compressibility. `c_g = 1/P − (1/Z)(dZ/dP)` from the same Z(EOS,P,T)
  that produces `B_g`.
- **Scientific Impact:** `c_g_dynamic` enters the total system compressibility
  `ct = c_o·S_o + c_g·S_g + c_w·S_w + c_f` (`surrogate_engine.py:453`) and therefore the pressure
  increment `dp` (`:455`). An 8× error in `c_g` in the near-critical region distorts the pressure
  path exactly where CO₂ floods are most sensitive.
- **Evidence & Citation:** `audit_verify_5.py` §A3/§A4. Parent register **HIGH-17** flagged the
  *previous* two-branch `c_g` fudge for the same reason; the remediation replaced one undocumented
  fudge with a second, differently-shaped one.
- **Status:** NEW

---

## 4. New HIGH findings

### HIGH-20 — `B_o` is C⁰ but not C¹ at the bubble point (slope flips sign discontinuously)

- **Severity:** HIGH · **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:303-322`
- **Observed:** measured `dB_o/dP` by central difference: **+1.2609e-4 /psi** at 2 790 psi,
  **−1.5699e-5 /psi** at 2 810 psi. `B_o` itself is continuous (jump 1.1e-10), so the C⁰ repair of
  CRIT-03 is sound, but the derivative — the quantity the optimiser's finite-difference gradients
  and any continuation method consume — jumps by a factor of ~8 and changes sign.
- **Expected:** a continuous `B_o(P)` built from a single expression, or an explicit statement that
  the model is piecewise-`C⁰`.
- **Scientific Impact:** spurious gradient sign flips near `P_b` perturb GA/BO line searches and
  make `dBo/dP`-based sensitivity analysis meaningless at the bubble point.
- **Evidence & Citation:** `audit_verify_5.py` §A7.

### HIGH-21 — The bubble point is a hard-coded constant (`min(p_init, 2800)`) and `c_o` defaults conflict

- **Severity:** HIGH · **Category:** `PROVENANCE`
- **Location:** `core/engine_surrogate/pvt_state.py:101-106`
- **Observed:**
  ```python
  self.p_bubble = float(bubble_point_pressure_psi) if bubble_point_pressure_psi is not None \
                  else min(self.p_init, 2800.0)          # 2800 psi: no citation, no UI field
  self.c_o = float(oil_compressibility_1_psi)              # default 1.2e-5
  ```
  Measured `p_bubble = 2800.0` for the default 3 000 psi reservoir. Separately, the pressure ODE
  uses a **different** oil compressibility, `c_o = 1e-5`
  (`core/engine_surrogate/surrogate_engine.py:315`), so the module that computes `B_o(P)` and the
  module that integrates pressure assume fluids that compress by 1.2e-5 and 1.0e-5 respectively.
- **Expected:** `P_b` is a *measured* PVT quantity; if it is not user-supplied it must be flagged
  `UNKNOWN — EVIDENCE REQUIRED`. A single `c_o` must serve both consumers.
- **Scientific Impact:** the 20 % `c_o` mismatch propagates into `ct` and hence into `dp`. The
  `P_b` default silently decides whether the run is saturated or undersaturated, which flips the
  sign of `dB_o/dP` and therefore the whole volumetric bookkeeping.
- **Evidence & Citation:** `audit_verify_5.py` §A/§A5.
- **Status:** NEW

### HIGH-22 — See CRIT-18 (NPV revenue completeness) — retained here as the economic-severity view

Tracked as **CRIT-18**. Not duplicated.

### HIGH-23 — Containment is economically inert: leakage is identically zero, yet leaked CO₂ would earn storage credit

- **Severity:** HIGH · **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/surrogate_engine.py:635`, `:644`;
  `core/optimisation_engine.py:1372-1379`; `core/objectives/wrapper.py:108-132`
- **Observed:** three independent measurements agree that leakage never fires:
  - `total_leakage_tonne = 0.0`; `leakage_rate_tonnes_day` max = `0.0`;
    `annual_leakage_tonne` = 15 zeros — across every configuration probed.
  - `optimisation_engine.py:1372-1375` now reads `annual_leakage_tonne` from the profiles, so the
    key **is** present and **is** read; it is simply always zero. The `else` branch at `:1377-1386`
    (the synthetic overpressure leakage) is now **unreachable**.
  - `wrapper.py:108-111` likewise reads `total_leakage_tonne` = 0.0, so the
    `leakage_rate_fraction` branch at `:112-121` — the only path that could ever charge leakage —
    is now **dead**, which also silently disables `CO2StorageParameters.leakage_rate_fraction`
    (default 0.01, `core/data_models.py:1671`).
  Meanwhile the revenue side at `:635` computes
  `annual_stored_tonne = max(0, inj − prod) · CO2_TONNE_PER_MSCF` and `:636` multiplies it by
  `co2_storage_credit` (**default 25.0 USD/t**, measured present in `params`).
- **Expected:** $M_{stored} = M_{inj} - M_{prod} - M_{leaked}$, and the storage credit must be earned
  only on CO₂ that is *actually* retained. A breach must cost money.
- **Scientific Impact:** the model pays for stored CO₂ it cannot retain, and never charges for the
  CO₂ it loses. This is **asymmetric in the optimiser's favour for leakage**: the environmental
  externality of a containment failure is priced at zero while the revenue for the same molecules
  is priced at $25/t. This inverts EPA Class VI risk incentives — the single most consequential
  finding for a carbon-storage project.
- **Evidence & Citation:** `audit_verify_9.py`; `audit_verify_7.py` §R. Wiki invariant #10/#11.
- **Status:** NEW

### HIGH-24 — Two live VRR definitions coexist; the post-hoc one silently overwrites the integrated one

- **Severity:** HIGH · **Category:** `SOFTWARE`
- **Location:** `core/engine_surrogate/surrogate_engine.py:417` (integrated) vs `:545-548` (post-hoc)
- **Observed:** the in-loop VRR uses `bo_dynamic`/`bg_dynamic` at `current_p` with the
  **pre-rescale** oil profile; the new post-hoc VRR uses `bo_profile`/`bg_profile` and the
  **post-rescale** streams, and overwrites `vrr_profile` only when
  `len(oil_rate) == len(pressure_profile)`. Two different definitions of the same reported
  quantity exist in one object, selected by an incidental array-length coincidence.
- **Scientific Impact:** VRR is the diagnostic engineers use to judge voidage replacement. Its
  definition silently changes with array length; the non-degenerate branch is the one that wins,
  which means the integrated (physically coupled) value is discarded.
- **Status:** NEW

### HIGH-25 — `s_g_avg` masking: a non-positive tangent slope silently becomes 100 % displacement efficiency

- **Severity:** HIGH · **Category:** `NUMERICAL`
- **Location:** `core/engine_surrogate/analytical_models.py:309-313`
- **Observed:**
  ```python
  s_g_avg = s_gf + (1.0 - f_gf) / max(slope_bt, EPSILON)
  s_g_avg = np.clip(s_g_avg, s_gf, 1.0 - sor)
  ```
  `slope_bt = df_g/dS_g` at the shock front is mathematically **positive** for a concave fractional
  flow curve, but it is computed from a finite-difference `argmax` over a discretised grid
  (`:305-307`) and can be zero or negative through discretisation noise. `max(·, EPSILON)` then
  returns `1e-16`, `(1−f_gf)/1e-16` overflows to ~1e16, and the following `np.clip` returns
  `1 − S_or` — i.e. **100 % displacement efficiency** — with no warning.
- **Expected:** a violated precondition must raise, not saturate. The sign of `df_g/dS_g` is a
  mathematical property of the Corey model and is cheap to assert.
- **Scientific Impact:** the immiscible limb's RF would jump to its physical maximum on a
  numerical artefact. This is *Phase-3 anti-pattern class D* inside a block whose gradient the
  remediation otherwise correctly restored (CRIT-07).
- **Status:** NEW

### HIGH-26 — CO₂ properties mix a Peng-Robinson FVF with a correlation-based `Z` in the same mixture

- **Severity:** HIGH · **Category:** `PHYSICAL`
- **Location:** `core/engine_surrogate/pvt_state.py:375-400`
- **Observed:** in `calculate_mixture_gas_properties`, the CO₂ leg uses
  `calculate_co2_fvf_rb_per_mscf` (Peng-Robinson density) while the hydrocarbon leg uses
  `bg_hc = 5.035·z_hc·T_R/P` with `z_hc` from **Papay**. The mixture is combined with a
  mole-fraction-weighted density and a **linear** `Z`-blend. The CRIT-04 fix corrected the
  constant but left a single-gas `B_g` law alongside an EOS `B_g`; for `y_CO2 → 1` the two
  definitions of "CO₂ FVF" disagree by the amount measured in CRIT-21 (2.6–8× in `c_g`).
- **Expected:** a mixture `B_g` from one consistent EOS (or an explicit, cited mixing rule).
- **Scientific Impact:** `bg_dynamic` feeds produced-gas voidage (`:412-414`), VRR (`:417`) and the
  pressure increment (`:455`); a mixture FVF assembled from two incompatible models is not a
  property of any real gas.
- **Status:** NEW

---

## 5. New MEDIUM findings

| ID | Cat | Location | Observed → Expected → Impact | Status |
|---|---|---|---|---|
| **MED-17** | SOFTWARE | `surrogate_engine.py:671` | `monthly_oil_stb = profile_result.get("monthly_oil_stb", np.zeros(1))` — measured **length 1, all zeros**, while `yearly_oil_stb` correctly sums to 433 959 STB. → the monthly series must be populated or the key removed. → any consumer resolving monthly resolution (MED-10) reads zeros; `summary_monthly.csv` will be empty. | NEW |
| **MED-18** | SOFTWARE | `analytical_models.py:561-562` | `kv = max(kv, 1.0 + EPSILON)` immediately precedes `if kv <= 1.0 + 1e-6:` → the documented unit-mobility-ratio branch is **unreachable**. → delete it or reorder the guard. → dead code advertising a limiting case the model cannot reach; `M = 1` is instead handled by the general branch (which is correct — measured continuity 7.5e-3). | NEW |
| **MED-19** | PROVENANCE | `analytical_models.py:199-201` | `MiscibleSurrogate` RF is multiplied by a **new** `e_v = 1/(0.8+0.2·gravity_factor)` that appears in no textbook; and the RF clip was simultaneously raised from `0.80` to `0.85` in the same edit. → cite the vertical-sweep basis for `e_v` and justify the new ceiling. → two changes to the same physical bound in one un-audited edit; the ceiling change alone raises achievable RF by up to 6 %. | NEW |
| **MED-20** | SOFTWARE | `optimisation_engine.py:852` | `"annual_water_stb": annual_water` was **deleted** in the remediation with no replacement key. → confirm no consumer reads it. → silent reporting-key loss. | NEW |
| **MED-21** | SOFTWARE | `core/objectives/wrapper.py:130` | `remediation_cost = max(carbon_tax, 100.0) * leaked_tonnes` — the HIGH-13 defect ($100/t floor overriding any configured value below $100) is **still present** after remediation. → use the configured value. → the user's carbon price is silently floored. | NEW |
| **MED-22** | SOFTWARE | `core/objectives/wrapper.py:112-121` | Now dead (see HIGH-23) but still present, and it reads `profiles["leakage_rate_fraction"]` which the engine never emits. → delete. → misleading second leakage model. | NEW |

---

## 6. Corrections to the parent register

| Register claim | Correction |
|---|---|
| "335 passed / 333 passed" | Actual this round: **335 passed, 23 skipped, 0 failed**. The wiki's "333" and MED-13's coverage figure (37 % of 36 062 statements) are **stale**; current coverage is **57 % of 7 791 statements** on the `--cov` basis actually configured. |
| CRIT-12 "RESOLVED — re-synchronized saturations, cumulative oil, and VRR" | **REGRESSED** → now CRIT-17 (closure broken, 27 % of timesteps) and HIGH-24 (dual VRR). |
| CRIT-13 "genes wired" | **PARTIALLY RESOLVED** → CRIT-19 (`gravity_factor` now an unprincipled active fudge), CRIT-20 (composition decoupling + ω(MMP)=0.5). |
| CRIT-10 "leakage forwarded" | **Wiring only** → HIGH-23 (leakage still identically zero; `leakage_rate_fraction` path now dead; leakage-blind storage credit). |
| CRIT-03 "bubble-point model integrated" | **PARTIALLY RESOLVED** → HIGH-20 (C¹ kink), HIGH-21 (hard-coded `P_b`, `c_o` conflict). |
| CRIT-05 "Hall-Yarborough fixed" | Correct diagnosis, but the *replacement* `c_g` introduced CRIT-21. Z itself is now correct (Papay, measured Z<1 with a dip). |
| Wiki "Core Architectural Invariants" #10, #11 | **Not enforced.** #10's second half (`Gross Injected = Net Stored + Leakage + Produced`) is uncheckable because leakage ≡ 0 and `cum_stored = inj − prod`. #11's "strictly capped" containment is an artefact of `np.clip(current_p, p_min, p_safe_ceiling)` at `:468`, not a solved constraint. |

---

## 7. What is genuinely better after the remediation

Recorded so the register is not read as uniformly negative:

1. **CRIT-04** — `B_g` constant is now exactly right (`5.0351` derived vs `5.035` used).
2. **CRIT-05** — Z-factor now physically correct (Papay; measured `Z = 0.849 @ P_pr 2.24`,
   `0.827 @ 3.74`, `0.868 @ 5.24` — a proper dense-gas dip, versus the old monotone `Z > 1`).
3. **CRIT-01** — HCPVI is now dimensionless and pressure-dependent, verified to scale correctly with
   injection rate.
4. **HIGH-01** — `rf_max_physical` correctly normalised to an OOIP fraction.
5. **CRIT-07** — the immiscible limb has a real gradient again (24 distinct values, spread 0.304).
6. **CRIT-02** — NPV and reported RF are now derived from the same evaluation; the double-evaluation
   defect is genuinely gone. (Truncation is a *new, separate* defect: CRIT-18.)
7. **CRIT-09** — the Class-VI penalty block is now reachable (de-indentation).
8. **CRIT-06** — Koval is continuous and monotone in M (verified: max jump 7.5e-3 across `M = 1`).
   Its defect is now saturation at the clip, not non-continuity.

---

## 8. Remediation-order proposal (not applied — audit-only)

1. **CRIT-14, CRIT-15** — remove the `mobility_ratio` override, or compute `M` from `μ_o(P, x_CO₂)`
   and treat the user's value as a *constraint check* rather than a replacement. Then re-examine the
   default HCPVI, which is far outside published practice.
2. **CRIT-17** — fix the pore-volume basis in the `:535-551` block (use total PV for both `S_o` and
   `S_w`), then re-verify $\sum S = 1$ at every step.
3. **CRIT-21, HIGH-26** — derive `c_g` and mixture `B_g` from the Peng-Robinson EOS already in the
   class; delete both ad-hoc power laws.
4. **CRIT-20, CRIT-19** — restore composition coupling in the miscibility sigmoid; give
   `gravity_factor` a physical definition or delete it from the gene space.
5. **HIGH-23** — make `M_stored` leakage-aware and gate the storage credit on retained CO₂.
6. **CRIT-18** — add gas and produced-CO₂ revenue to the cash flow.
7. **CRIT-16** — point the `getattr` guards at the fields that exist, or use `oil_fvf`.
8. **HIGH-21** — expose `P_b` and `c_o` as one user-visible PVT group; remove the `2800.0` default.

**Test-gate warning (HIGH-18 still open):** the suite is **335 passed / 0 failed** while CRIT-14,
CRIT-17, CRIT-19 and CRIT-20 are all live. No test detects the loss of viscosity coupling, the
saturation-closure breach, the inactive `gravity_factor`, or the composition decoupling. Do not treat
green as evidence for any of the above.

---

## 9. Relevant files

| Artefact | Path |
|---|---|
| Parent register | [`audit/scientific_flaws.md`](../../../audit/scientific_flaws.md) |
| Software audit | [`res_audit.md`](../../../res_audit.md) |
| Physics audit | [`phd_audit.md`](../../../phd_audit.md) |
| Parameter provenance | [`audit/parameter_provenance.csv`](../../../audit/parameter_provenance.csv) |
| Evidence scripts | `evidence_scripts/audit_verify_5.py` … `audit_verify_9.py` |
| Modified scientific files under review | `core/engine_surrogate/{pvt_state,analytical_models,surrogate_engine}.py`, `core/objectives/{wrapper,storage}.py`, `core/optimisation_engine.py`, `core/simulation/recovery_models.py`, `core/data_models.py` |