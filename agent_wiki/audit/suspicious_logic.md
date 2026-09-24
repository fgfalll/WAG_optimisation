# Suspicious Logic & Scientific Discrepancies Audit

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

This document catalogs logic, equations, and code paths that are mathematically or physically questionable, internally contradictory, or scientifically unvalidated.

---

## 1. Top Critical Scientific & Numerical Discrepancies

### A. Discrepancy Between Claimed Miscibility Physics and Executed Code
- **Location**: [core/engine_surrogate/analytical_models.py:737-740](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L737-L740), [782-791](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L782-L791), [957-980](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L957-L980).
- **Claim**: In `validation/VALIDATION_STATUS_REPORT.md` (lines 37–38), `core/engine_surrogate/LITERATURE_REFERENCES.md`, and module docstrings, the PhD hybrid model claims:
  `"Smooth, differentiable transition at MMP (no miscibility cliff) via modified hyperbolic tangent: ω = 0.5 * (1 + tanh(β·(P/P_MMP - α_eff)))"` and recovery interpolation `RF = ω · RF_mis + (1 - ω) · RF_limit`.
- **Actual Code**: In [core/engine_surrogate/analytical_models.py:737-740](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L737-L740):
  ```python
  if pressure >= mmp:
      omega = 1.0 - np.exp(-(pressure - mmp) / max(mmp, EPSILON))
  else:
      omega = 0.0
  ```
  And in lines 782-791:
  ```python
  mu_g_eff = (mu_mix**omega_tl) * (viscosity_inj ** (1.0 - omega_tl)) if omega > 0.01 else viscosity_inj
  mu_o_eff = (mu_mix**omega_tl) * (viscosity_oil ** (1.0 - omega_tl)) if omega > 0.01 else viscosity_oil
  ```
- **Consequence**:
  1. The code executes a piecewise non-differentiable step/kink function.
  2. `get_miscibility_weight()` (which implements the $\tanh$ equation) is completely orphaned and never called.
  3. `omega` is **never used to interpolate recovery factors** ($RF = \omega RF_{mis} + (1-\omega) RF_{imm}$ was deleted).
  4. Instead, `omega` is only used in `if omega > 0.01:` to toggle Todd-Longstaff mixed viscosity, causing a **7.5× instantaneous step drop in mobility ratio** (from ~30 down to ~4) at $P \approx 1.01 \times MMP$, creating a severe jump discontinuity in the objective space.
  5. For all $P < MMP$, $\omega = 0.0$, producing zero pressure gradient sensitivity $\partial\omega/\partial P = 0$.
- **Severity**: Critical (Violates core PhD model physics claim; introduces artificial cliffs and non-differentiability in optimization landscape).
- **Status**: Open.
- **Recommended fix**: Reconnect `get_miscibility_weight()` into `PhDHybridRecoveryModel.calculate_recovery()`, replace the piecewise exponential and `if omega > 0.01` with smooth Todd-Longstaff mixing weighted by $\omega$, and interpolate between miscible and immiscible asymptotic displacement states.

---

### D. Hard Recovery Factor Cap of 0.80
- **Location**: [core/engine_surrogate/analytical_models.py:205](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L205).
- **Code**: `return float(np.clip(rf, 0.05, 0.80))`
- **Physical Defect**:
  - Imposes an arbitrary ceiling of 80% recovery factor on miscible floods, regardless of how many pore volumes of CO₂ are injected, how low connate water is ($S_{wi} \le 0.15$), or how homogeneous the reservoir is.
  - In ultra-clean, high-perm laboratory corefloods or 1D slim tube experiments, miscible recovery can reach 90–95%.
  - When Koval/Welge calculates $RF > 0.80$, clipping occurs silently with zero warning or log message, and gradients $\nabla RF$ vanish to zero, artificially flattening the search space.
- **Severity**: High (Flattens optimization objective space and traps optimizers on artificial boundaries).
- **Status**: Open.
- **Recommended fix**: Make the upper cap configurable via `EORParameters` (defaulting to physical displacement limit $1 - S_{or\_min} - S_{wi}$), and log a warning whenever clipping binds during optimization.

---

### E. `calculate_gradient` Parameter Default Dead-End and Omission
- **Location**: [core/engine_surrogate/analytical_models.py:905-942](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L905-L942).
- **Code**:
  ```python
  pressure = params.get("pressure", 3000.0) # Unpacked into local variable
  c7_plus = params.get("c7_plus_fraction", 0.3)
  ...
  for key in ["pressure", "mmp"]:
      if key in params: # Checks original dict!
  ```
- **Defect**:
  1. If `params` lacks `"pressure"`, the method unpacks a default of `3000.0` into a local variable, but then the finite-difference loop checks `if "pressure" in params:`, which evaluates to `False`. Gradients are silently omitted rather than evaluated at the default value.
  2. `c7_plus` is unpacked at line 907 but completely omitted from all perturbation loops (its gradient is never computed).
  3. However, neither GA, BO, PSO, nor DE in `core/optimisation_engine.py` calls `calculate_gradient()`; only validation scripts call it.
- **Severity**: Low (Dead code in active optimization paths; only called in validation scripts).
- **Status**: Open.
- **Recommended fix**: Perturb against the merged dictionary `params_with_defaults`, include `c7_plus_fraction` in the loop, or deprecate/remove the method if gradient-based optimization is not supported.

---

### O. Silent Profile Constraint Erasure in `_objective_function_wrapper`
- **Location**: [core/optimisation_engine.py:1777-1788](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1777-L1788) vs [1820-1835](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1820-L1835).
- **Claim / Expected Math**: `_check_profile_constraints()` calculates proportional penalties for environmental guardrails (injection period shortfall, storage efficiency shortfall, excessive annual leakage, and carbon tax). These penalties must be deducted from the candidate solution's objective fitness.
- **Actual Code**:
  ```python
  # Lines 1782-1784:
  profile_penalty = profile_constraint_result["penalty"]
  if profile_penalty > 0:
      result -= profile_penalty

  # Lines 1820-1834:
  if self.chosen_objective == "co2_utilization":
      result = -objective_value
  ...
  else:
      result = objective_value  # OVERWRITES result!
  ```
- **Consequence**: `result` is unconditionally re-assigned at line 1834, completely discarding `profile_penalty`. Every single environmental guardrail in `_check_profile_constraints()` is 100% inoperative during optimization. Candidates violating minimum injection periods or exceeding leakage limits suffer zero penalty.
- **Severity**: Critical (All profile-based environmental constraints and carbon penalties are completely erased).
- **Status**: Open.
- **Recommended fix**: Delete the redundant second objective-handling block (lines 1820–1834) or ensure `result` initialization occurs once at line 1756 and penalties are deducted sequentially after all base objective assignments.

---

### P. Optimization Objective Conflates Breakthrough Impact Penalty with Economic DCF NPV (~2× Discrepancy)
- **Location**: [core/optimisation_engine.py:1241-1275](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1241-L1275), [1837-1842](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1837-L1842), [3020-3025](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L3020-L3025), [ui/optimization_widget.py:1959](file:///d:/rep/4.6/co2eor_optimizer/ui/optimization_widget.py#L1959), [utils/report_generator.py:508](file:///d:/rep/4.6/co2eor_optimizer/utils/report_generator.py#L508).
- **Claim / Expected Math**: "Final Optimized NPV" must report the financial Net Present Value calculated via discounted cash flow. Internal heuristic penalty scores must not be labeled as NPV.
- **Actual Code**:
  Inside `_objective_function_wrapper`:
  ```python
  result = objective_value  # eval_results["npv"] = ~$6.30e8
  result *= breakthrough_impact  # breakthrough_impact in [0.5, 1.2], e.g. ~0.511
  ```
  Inside GA completion:
  ```python
  self._results = {
      "objective_function_value": fitness,  # ~$3.22e8 (penalized heuristic)
      "final_metrics": final_eval,          # final_eval["npv"] = ~$6.30e8 (true DCF NPV)
  }
  ```
  UI summary line 1959:
  `f"Final Optimized {self.objective_combo.currentText()}: {self.current_results.get('objective_function_value'):.4g}"` displays **$3.22e8**, while the KPI table displays **$630,000,000**.
- **Consequence**: The header value reported to the user as "Final Optimized NPV" is roughly half ($0.51\times$) of the true project NPV because `_calculate_breakthrough_economic_impact()` applies an ad-hoc multiplier $\in [0.5, 1.2]$ to the financial NPV and the result is mislabeled as NPV rather than "Objective Fitness Score".
- **Severity**: Critical (Causes a 50% discrepancy in reported project financial value; violates standard financial math by scaling discounted cash flows with an arbitrary scalar).
- **Status**: Open.
- **Recommended fix**: Model breakthrough economic consequences directly through actual fluid cash flows (water/gas processing costs and separator upgrade capex in `_calculate_engine_npv`), do not multiply cash-flow NPV by arbitrary factors, and clearly label `objective_function_value` as "Optimizer Fitness" rather than "NPV".

---

### Q. Dual Conflicting Definitions of CO₂ Storage Efficiency (0.4525 vs 0.8378)
- **Location**: [core/engine_surrogate/surrogate_models.py:364-366](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L364-L366) vs [analysis/material_balance.py:228-230, 522](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L228-L230).
- **Claim / Expected Physics**: Storage efficiency must have a clear, documented physical definition.
- **Actual Code**:
  1. `surrogate_models.py`:
     ```python
     storage_efficiency = net_stored_tonne / max(co2_injected_total, EPSILON)
     ```
     Denominator is **gross cumulative injection** (fresh purchased + recycled gas). For typical projects with recycling, $E_{\text{gross}} \approx 0.4525$ (45.25%).
  2. `analysis/material_balance.py`:
     ```python
     storage_efficiency[i] = net_stored_tonne[i] / purchased_tonne[i]
     ...
     "avg_storage_efficiency": np.mean(material_balance_data["storage_efficiency"])
     ```
     Denominator is **fresh purchased CO₂** (OSTI-1204577 standard). Because recycled gas is retained or re-injected, purchased retention is high ($E_{\text{purchased}} \approx 0.8378$ or 83.78%).
- **Consequence**: Two distinct physical metrics (~1.85× apart) are displayed in the same report under identical labels ("Storage Efficiency"), leading users to suspect numerical errors.
- **Severity**: High (Dual contradictory definitions presented under the same terminology).
- **Status**: Open.
- **Recommended fix**: Formally separate the metrics into `co2_retention_efficiency_purchased` ($M_{\text{stored}} / M_{\text{purchased}}$, ~80-95%) and `co2_storage_factor_gross` ($M_{\text{stored}} / M_{\text{gross\_injected}}$, ~40-60%), update documentation and UI labels accordingly.

---

### R. CO₂ Utilization Factor Deflation and Triple Unit Mismatch
- **Location**: [core/objectives/wrapper.py:188-212](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L188-L212), [ui/optimization_widget.py:1297](file:///d:/rep/4.6/co2eor_optimizer/ui/optimization_widget.py#L1297), [utils/report_generator.py:500](file:///d:/rep/4.6/co2eor_optimizer/utils/report_generator.py#L500).
- **Claim / Expected Physics**: Net CO2 utilization factor in EOR literature is defined as:
  $$\text{Utilization} = \frac{\text{Cumulative Purchased CO}_2}{\text{Incremental Oil Attributable to CO}_2}$$
  with typical field values ranging between 0.3–0.6 tonne/bbl (~6–10 MSCF/STB).
- **Actual Code**:
  ```python
  total_oil = np.sum(oil_produced_calc) # Cumulative TOTAL field oil (primary + secondary + EOR)
  results["co2_utilization"] = total_co2_purchased_tonne / total_oil
  ```
  UI display: `f"{co2_util:.2f} MSCF/STB"` (raw number displayed with MSCF/STB label).
  HTML report display: `<tr><td>CO2 Utilization</td><td>...</td><td>bbl/ton</td></tr>` (inverted unit label).
- **Consequence**:
  1. `total_oil` includes the entire reservoir primary and waterflood production (3–4× incremental EOR oil), deflating calculated utilization to ~0.04 tonne/bbl (~0.77 MSCF/bbl), far below physical reality.
  2. The value is calculated in `tonnes / STB`, displayed in the UI as `MSCF/STB` without multiplying by 18.9 MSCF/tonne, and labeled in HTML reports as `bbl/ton`.
- **Severity**: High (Metric deflated by 4×; unit labels across GUI and reports are mathematically and physically wrong).
- **Status**: Open.
- **Recommended fix**: Calculate incremental oil over a primary decline baseline ($N_{p\_eor} = N_{p\_total} - N_{p\_primary}$), convert consistently to MSCF/STB ($1\text{ tonne} = 18.9\text{ MSCF}$), and correct UI and HTML table labels.

---

### S. Hard-Coded Gaseous $B_g = 2.07$ RB/MSCF Overestimates Supercritical Downhole Volume by 5×
- **Location**: [core/engine_surrogate/surrogate_engine.py:928-939, 1137-1138](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L928-L939).
- **Claim / Expected Physics**: Gas formation volume factor $B_g$ must be computed from reservoir pressure and temperature ($B_g = 0.02827 \frac{z T}{P}\text{ res-bbl/SCF}$). Dense supercritical CO₂ at 2500–3500 psi and 120–180°F has $B_g \approx 0.00035 - 0.00045\text{ RB/SCF}$ ($0.35 - 0.45\text{ RB/MSCF}$).
- **Actual Code**:
  ```python
  "mscf_per_res_bbl": 1.0 / max(getattr(reservoir_data, "bg", 0.00207), 1e-6)
  ...
  q_inj_rb = profile_result["injection_profile"] * 1000.0 * bg # bg = 0.00207 RB/SCF
  ```
- **Consequence**: $B_g = 0.00207\text{ RB/SCF}$ ($2.07\text{ RB/MSCF}$) represents low-pressure gas at ~800 psi. Applying it to dense-phase CO₂ overestimates the downhole volumetric injection rate by **$4.5\times$ to $6\times$**, driving artificial over-pressurization in the 0D tank pressure ODE.
- **Severity**: High (Physical PVT property is hardcoded to a depleted-gas value rather than dense-phase CO₂).
- **Status**: Open.
- **Recommended fix**: Compute $B_g$ dynamically from reservoir $P$ and $T$ using Peng-Robinson EOS or NIST CO₂ density tables ($B_g = \rho_{sc} / \rho_{res}(P, T)$).

---

### T. GA-to-BO Hybrid Handoff Defects: Final-Generation Clumping, Bound-Sticking Acquisition, Parameter Dropping, and Negated Targets
- **Location**: [core/optimisation_engine.py:2660-2675, 2988, 3095-3105, 3453, 3536, 3544](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L2660-L2675).
- **Claim / Expected Math**: Hybrid GA $\to$ BO handoff should initialize Bayesian Optimization with diverse solutions across the design space, maintain balanced acquisition exploration, pass acquisition hyperparameters correctly, and evaluate consistent objective definitions.
- **Actual Code**:
  1. `hybrid_optimize()` selects points only from `ga_instance.population` (final converged generation). When points are clustered ($d \le \text{threshold}$), it falls back to picking the highest-fitness remaining points, passing near-duplicates.
  2. `acq_kappa` is scaled by `exploration_factor = 2.5`, yielding $\kappa \approx 6.44$. At $\kappa = 6.44$, GP variance dominates predictive mean, forcing UCB to sample hypercube boundaries and corners.
  3. `maximize_with_logging` defines `(init_points, n_iter, acq="ucb", kappa=2.576, xi=0.01, **kwargs)` but calls `original_maximize(..., **kwargs)`, which **drops `acq`, `kappa`, and `xi`** because they were captured as explicit arguments.
  4. In GA results, points are recorded with `"target": -fit` (negated).
  5. GA evaluates `_objective_function_wrapper` with increasing `current_gen` (strengthening adaptive penalties), while BO always passes `current_gen=None` (generation 0, weakest penalties).
- **Severity**: High (Degrades BO initialization into clustered points, strips acquisition tuning, and evaluates inconsistent penalty strengths).
- **Status**: Open.
- **Recommended fix**: Archive diverse solutions across all GA generations, fix parameter forwarding in `maximize_with_logging`, set $\kappa \in [1.5, 2.5]$, store positive targets, and standardize penalty evaluation.

---

### U. Produced CO₂ Artificially Halved (by 50%) in `core/objectives/storage.py`
- **Location**: [core/objectives/storage.py:116](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/storage.py#L116).
- **Claim / Expected Physics**: Produced CO₂ mass is $M_{\text{co2\_prod}} = \sum q_{\text{co2\_prod}} \times \rho_{\text{co2}}$.
- **Actual Code**:
  ```python
  total_produced = float(np.sum(co2_produced)) * produced_co2_frac_val * co2_density_val
  ```
  where `produced_co2_frac_val = 0.50`.
- **Consequence**: `co2_produced` is extracted from `f"{time_resolution}_co2_produced_mscf"`, which is ALREADY only the CO₂ stream from `FastProfileGenerator`. Multiplying it by 0.50 cuts the produced CO₂ in half, artificially inflating calculated net CO₂ storage.
- **Severity**: High (Fabricates 50% storage retention credit by double-discounting hydrocarbon gas fraction).
- **Status**: Open.
- **Recommended fix**: Only apply `produced_co2_frac` if the source array is `total_gas_produced_mscf`. If the array is already `co2_produced_mscf`, do not multiply by `produced_co2_fraction`.

---

### V. WAG Cycle Modulation Corrupts Mass Balance Integral Without Re-Normalization
- **Location**: [core/engine_surrogate/profile_generator_fast.py:330-348](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L330-L348).
- **Claim / Expected Math**: WAG cycle modulations must preserve total cumulative recovery: $\int q_o(t) dt = OOIP \times RF$.
- **Actual Code**: After `oil_profile` is normalized to `ultimate_recovery`, `_apply_wag_oil_modulation` multiplies gas phases by 1.08 and water phases by 0.96 without re-normalizing the integral.
- **Consequence**: Actual cumulative oil production diverges from the analytical recovery factor by several percent depending on the ratio of gas to water cycles.
- **Severity**: Medium (Cumulative production violates analytical model recovery factor).
- **Status**: Open.
- **Recommended fix**: Re-scale `oil_profile = oil_profile * (ultimate_recovery / np.sum(oil_profile * dt))` after WAG modulation.

---

### W. Beginning-of-Period Discounting in Surrogate Engine Inflates Project NPV
- **Location**: [core/engine_surrogate/surrogate_engine.py:1472-1473](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1472-L1473).
- **Claim / Expected Math**: Annual production cash flows during year $t$ should be discounted using end-of-period ($1 / (1+r)^t$) or mid-period ($1 / (1+r)^{t - 0.5}$) discounting.
- **Actual Code**:
  ```python
  discount_factors = 1.0 / (1.0 + discount_rate) ** np.arange(total_years)
  ```
- **Consequence**: Year 1 cash flow uses index 0 ($(1+r)^0 = 1.0$), treating all year 1 revenue as received at day 0. This inflates project NPV by approximately 10% (at 10% discount rate) compared to end-of-period discounting.
- **Severity**: Medium (Systematic financial discounting distortion).
- **Status**: Open.
- **Recommended fix**: Use mid-period discounting (`np.arange(total_years) + 0.5`) or end-of-period discounting (`np.arange(1, total_years + 1)`).

---

### X. Magic 100,000 STB/year Threshold in DataValidator Causes 365× RF Error for Small/Pilot Fields
- **Location**: [analysis/data_validation.py:310-340](file:///d:/rep/4.6/co2eor_optimizer/analysis/data_validation.py#L310-L340).
- **Claim / Expected Math**: Rate vs. volume classification must inspect profile metadata or key prefixes, not raw array magnitudes.
- **Actual Code**:
  ```python
  if max_val > 100000:
      total_oil_produced = np.sum(oil_production)
  else:
      dt = 365.25 # for annual arrays
      total_oil_produced = np.sum(oil_production) * dt
  ```
- **Consequence**: For pilot or marginal fields ($OOIP \le 1,000,000$ STB) where annual oil production is $< 100,000$ STB/year, annual volumes are mistaken for daily rates and multiplied by 365.25, inflating cumulative oil by **365×** and causing DataValidator to erroneously reject valid simulation runs.
- **Severity**: Medium (Fails simulation validation for small/pilot reservoirs).
- **Status**: Open.
- **Recommended fix**: Use profile key naming prefixes (`yearly_`, `monthly_`, `daily_`) or time vector delta to determine integration behavior, rather than magnitude thresholds.

---

### Z. Unconstrained Recycling Volume Can Exceed Injected Volume in `surrogate_engine.py`
- **Location**: [core/engine_surrogate/surrogate_engine.py:1350-1353](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1350-L1353).
- **Claim / Expected Physics**: Recycled CO₂ reinjected cannot exceed total CO₂ injection rate ($q_{\text{recycled}} \le q_{\text{injected}}$).
- **Actual Code**:
  ```python
  recycled = annual_co2_prod[year] * recycling_efficiency
  purchased = max(0.0, annual_co2_inj[year] - recycled)
  annual_co2_purchased.append(purchased)
  annual_co2_recycled.append(recycled)
  ```
- **Consequence**: When produced gas is high (post-breakthrough), `recycled` can exceed `annual_co2_inj[year]`. The operator is charged recycling OPEX on gas volumes that cannot physically be reinjected.
- **Severity**: Medium (Inconsistent mass balance and OPEX distortion in high-GOR periods).
- **Status**: Open.
### AA. $2.4 \times 10^{11}$ Missing $g_c$ Unit Factor in Dimensionless Gravity Number ($N_g$) and Viscous-Gravity Ratio ($R_{v/g}$)
- **Location**: [analytical_models.py:871](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L871), [breakthrough_physics.py:314, 361](file:///d:/rep/4.6/co2eor_optimizer/analysis/breakthrough_physics.py#L314)
- **Claim / Expected Physics or Math**:
  The dimensionless Gravity Number in oilfield units represents the ratio of gravity to viscous pressure gradients:
  $$N_g = \frac{\Delta\rho g \sin(\theta)}{\mu u / k}$$
  In oilfield units ($k$ in $\text{ft}^2$, $\Delta\rho$ in $\text{lb}_m/\text{ft}^3$, $\mu$ in $\text{lb}_f\cdot\text{day}/\text{ft}^2$, $u$ in $\text{ft/day}$), converting mass to force requires dividing by Newton's constant $g_c = 32.174 \times (86,400)^2 = 2.40 \times 10^{11}\text{ lb}_m\cdot\text{ft}/(\text{lb}_f\cdot\text{day}^2)$. Because standard gravity $g = 32.174\text{ ft/s}^2 = 2.40 \times 10^{11}\text{ ft/day}^2$, the ratio is identically unity:
  $$\frac{g}{g_c} = 1.0\text{ lb}_f / \text{lb}_m$$
- **Actual Code**:
  In [analytical_models.py:871](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L871):
  ```python
  N_g = (perm_md * 1.0623e-14 * delta_rho * 2.4e11 * sin_theta) / (
      viscosity_inj * 2.417e-10 * u_ft_day + EPSILON
  )
  e_v = 1.0 / (1.0 + beta_gravity * N_g * override_severity)
  rf = e_sweep * e_v * e_d
  ```
  And in [breakthrough_physics.py:314, 361](file:///d:/rep/4.6/co2eor_optimizer/analysis/breakthrough_physics.py#L314):
  ```python
  g_ft_day2 = 2.4e11
  ng = (k_ft2 * delta_rho_lb_ft3 * g_ft_day2 * abs(sin_theta)) / (
      mu_g_lb_day_ft2 * u_ft_day + EPSILON
  )
  phi_ng_m = 1.0 / (1.0 + np.sqrt(ng * max(m_ratio, EPSILON)))
  ```
- **Consequence**:
  $2.4 \times 10^{11}$ was multiplied in the numerator without dividing by $g_c$, inflating $N_g$ by **$2.4 \times 10^{11}\times$** (evaluating to $N_g \sim 10^{10}$ instead of $N_g \sim 0.04$):
  1. In `analytical_models.py`, whenever reservoir dip angle $\theta > 0$, vertical sweep efficiency $e_v$ collapses from $0.98$ to $1.5 \times 10^{-8}$, causing field recovery factor $RF$ to drop from $39.3\%$ to **$5.9 \times 10^{-9}$ (essentially zero oil)**!
  2. In `breakthrough_physics.py`, $\Phi(N_g, M)$ collapses from $0.60$ to $1.8 \times 10^{-6}$, causing breakthrough time to drop from $2$ years to **$3.4 \times 10^{-7}$ years (10 milliseconds)**!
- **Severity**: **Critical** (A catastrophic 11-order-of-magnitude dimensional error completely zeroes out recovery and breakthrough time for any dipped reservoir).
- **Status**: Open / Confirmed Defect.
- **Recommended fix**:
  Divide by $g_c$ (or eliminate the `2.4e11` factor since $g/g_c = 1.0$ in $\text{lb}_f/\text{lb}_m$):
  ```python
  N_g = (perm_md * 1.0623e-14 * delta_rho * 1.0 * sin_theta) / (
      viscosity_inj * 2.417e-10 * u_ft_day + EPSILON
  )
  ```

---

### AB. Inverted CO₂ Fractional Flow in `profile_generator_fast.py`
- **Location**: [profile_generator_fast.py:895-901](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L895-L901)
- **Claim / Expected Physics or Math**:
  In Buckley-Leverett and Koval fractional flow theory, higher mobility ratio $M = \mu_o / \mu_{co2} \gg 1$ creates severe viscous fingering, causing rapid gas breakthrough and high gas fractional flow ($F_{co2} \to 1.0$).
- **Actual Code**:
  ```python
  E = (0.78 + 0.22 * (mobility_ratio ** 0.25)) ** 4
  koval_factor = (E / mobility_ratio) * koval_mult
  koval_factor = float(np.clip(koval_factor, 0.5, 10.0))

  frac_flow_co2 = koval_factor / (koval_factor + (mobility_ratio - 1) * 0.5)
  frac_flow_co2 = float(np.clip(frac_flow_co2, 0.1, 0.7))
  ```
- **Consequence**:
  Dividing $E$ by `mobility_ratio` causes `koval_factor` to decrease as mobility ratio worsens ($M=1 \implies K=1.0$; $M=10 \implies K=0.5$).
  Then $(M - 1) \times 0.5$ in the denominator causes `frac_flow_co2` to evaluate to:
  - At $M = 1.0$ (favorable): `frac_flow_co2 = 0.70` (maximum).
  - At $M = 10.0$ (unfavorable fingering): `frac_flow_co2 = 0.10` (minimum).
  The gas fractional flow is **physically inverted**: light oils with favorable mobility produce $70\%$ gas, while heavy oils with severe fingering produce only $10\%$ gas!
- **Severity**: **Critical** (Inverts the fundamental physics of gas breakthrough and GOR development).
- **Status**: Open / Confirmed Defect.
- **Recommended fix**:
  Implement true Koval (1963) fractional flow: $K = H_k \times E \ge 1.0$, and $F_{co2} = \frac{K S_{co2}}{1 + S_{co2}(K - 1)}$.

---

### AM. Decoupled Optimizer Pressure vs. Dynamic Tank ODE Pressure (~1,350 psi Disconnect)
- **Location**: [core/optimisation_engine.py:3215-3222](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L3215-L3222), [core/engine_surrogate/surrogate_engine.py:948-965](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L948-L965).
- **Observed Defect**:
  - The optimizer selects `pressure` as an unconstrained decision variable (pinned to the upper search bound $4,450\text{ psia}$ in run `Export-hybrid-ga-bo-20260917-115129`).
  - This $4,450\text{ psia}$ is fed directly to `AnalyticalSurrogate.predict()` and `PhDHybridSurrogate.calculate_recovery()`, maximizing ultimate recovery ($RF = 41.47\%$) and driving miscibility weight $\omega \approx 1.0$.
  - However, the 0D material balance tank ODE (`_solve_pressure_ode_stiff`) calculates in-situ reservoir pressure independently based on voidage replacement and fluid compressibility, simulating dynamic pressures between $3,080\text{ psia}$ and $3,335\text{ psia}$ across the entire 15-year life.
  - At no point during the 15-year project does the simulated reservoir ever reach $4,450\text{ psia}$ (the actual maximum is $3,478\text{ psia}$ post-shut-in).
- **Consequence**: Recovery factor and displacement efficiency are evaluated under a phantom high-pressure regime that is completely decoupled by $\sim 1,350\text{ psi}$ from the actual reservoir depletion state.
- **Status**: Open.
- **Remediation**: Eliminate `pressure` as an independent decision variable or enforce that the surrogate recovery factor is recalculated using the simulated time-averaged dynamic reservoir pressure.

---

### AN. Unphysical CO₂ Net Utilization Factor (0.665 MSCF/STB vs. DOE/NETL 5.0–12.0 Benchmark)
- **Location**: [core/engine_surrogate/surrogate_models.py:465-495](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L465-L495), [core/engine_surrogate/profile_generator_fast.py:207-215](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L207-L215).
- **Observed Defect**:
  - In `AnalyticalSurrogate.predict()`, recovery factor is evaluated independently of injection rate.
  - The model reported a net utilization of **$0.665\text{ MSCF/STB}$ ($0.0352\text{ tonne/STB}$)** and gross utilization of **$1.212\text{ MSCF/STB}$ ($0.0639\text{ tonne/STB}$)**.
  - Published benchmarks from the **U.S. DOE/NETL** and empirical Permian Basin data establish typical net utilization at **$5.0 \text{ to } 12.0\text{ MSCF/STB}$ ($0.25 \text{ to } 0.60\text{ tonne/STB}$)** and gross utilization at **$10.0 \text{ to } 30.0\text{ MSCF/STB}$**.
  - At reservoir conditions ($B_g \approx 2.07\text{ RB/MSCF}$), injecting $665\text{ scf}$ of $\text{CO}_2$ provides only $1.38\text{ RB}$ of fluid displacement—physically insufficient to displace $1.0\text{ STB}$ ($1.30\text{ RB}$) accounting for residual oil, connate water, and phase dissolution.
- **Consequence**: The optimizer exploits this decoupling by driving injection rate to the minimum search bound ($5,000\text{ MSCFD}$) to minimize gas purchase OPEX while harvesting the full oil recovery of the field ($20.3\text{ MMSTB}$), generating a fictitious $\$878.8\text{M}$ NPV.
- **Status**: Open.
- **Remediation**: Couple recovery factor directly to cumulative Hydrocarbon Pore Volume Injected (HCPVI) via fractional flow displacement curves, and enforce a minimum net utilization penalty floor ($\ge 3.0\text{ MSCF/STB}$).

---

### AO. Single-Well Pattern Fallback Point-Drainage Anomaly (1 Producer Draining 1,354 Acres at 7,270 BOPD)
- **Location**: [core/engine_surrogate/profile_generator_fast.py:501-516](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L501-L516), [core/optimisation_engine.py:1410-1425](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1410-L1425).
- **Observed Defect**:
  - When users do not define explicit pattern wells, the engine falls back to `n_injectors = 1` and `n_producers = 1` while keeping the full field OOIP ($48,487,500\text{ STB}$, representing $\sim 1,354\text{ acres}$ in a $50\text{ ft}$, $20\%$ porosity sandstone).
  - The entire field recovery ($20,331,161\text{ STB}$) is allocated directly to `Well-Producer-1`, sustaining a flat plateau of **$7,270\text{ STB/day}$ ($2.655\text{ MMbbl/year}$)** for 6.4 consecutive years in a $100\text{ mD}$ formation.
- **Consequence**: Bypasses well pattern interference, well spacing economics, and localized Darcy drawdown limits. Field-scale recovery is achieved with the CAPEX and OPEX of a single wellbore.
- **Status**: Open.
- **Remediation**: Implement automated pattern partitioning (e.g. 5-spot 40-acre patterns: $N_{\text{wells}} = \text{Area} / 40 \approx 34\text{ wells}$) and clamp single-well rates to realistic inflow limits ($\le 500\text{ BOPD}$).

---

### AP. Catastrophic Breakthrough Wall Penalty (-1.0e12) Inducing Rate Floor Collapse
- **Location**: [core/optimisation_engine.py:1650-1675](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1650-L1675).
- **Observed Defect**:
  - The objective evaluation applies a catastrophic penalty (`-1.0e12` / `FAILURE_PENALTY`) whenever solvent breakthrough occurs earlier than 1.0 year (`breakthrough_time < 1.0`).
  - In a single-injector / single-producer pattern draining 1,354 acres, any realistic field injection rate causes early solvent breakthrough.
  - In run `Export-hybrid-ga-bo-20260917-115129`, evaluations 2, 6, 7, 9, 10, 14, 15, 21, 46, 47, 48, 78, 79, 80 hit this wall and received `-1.0e12`.
- **Consequence**: The optimizer encounters a severe artificial numerical cliff, driving injection rate straight into the lower search bound ($5,000\text{ MSCFD}$) to push breakthrough to $2.45\text{ years}$.
- **Status**: Open.
- **Remediation**: Replace the hard step failure penalty with a smooth, continuous penalty function or proper multi-well streamline transit time calculation.

---

## 2. Verified Correct Modules and Physical Formulations

The following modules, functions, and mathematical derivations were rigorously reviewed and found to be scientifically, dimensionally, and mathematically sound:

1. **Composite Vogel-Darcy Inflow Deliverability** (`FastProfileGenerator.calculate_composite_ipr_deliverability`):
   - Mathematically continuous $C^0$ and $C^1$ across all three operating regimes ($P_{wf} \ge MMP$, $P_{wf} < MMP \le P_{res}$, and $P_{res} < MMP$).
   - Derivative matching at boundary $P_{wf} = MMP$ verified: $\lim_{P_{wf} \to MMP^-} dq/dP_{wf} = \lim_{P_{wf} \to MMP^+} dq/dP_{wf} = -J$.
2. **Geomechanical Sandface Injection Pressure & Containment Protocol**:
   - Strictly enforces EPA Class VI UIC standards ($P_{\text{sandface}} = P_{\text{res}} + \frac{q_{\text{inj}}}{II} \le 0.90 \times P_{\text{frac}}$).
   - Quadratic penalty scaling and chromosome pruning verified.
3. **Koval Heterogeneity & Viscosity Ratio Derivations**:
   - Heterogeneity factor $H = 1/(1-V_{DP})^2$ and effective viscosity ratio $E_{\text{eff}} = (0.78 + 0.22 M^{0.25})^4$ accurately reproduce Koval (1963) SPE-145-PA equations.
4. **Stiff 0D Tank Pressure ODE Integration**:
   - `_solve_pressure_ode_stiff()` using Backward Differentiation Formulas (BDF) successfully eliminates explicit Euler numerical oscillations.
5. **Analytical MMP Published Correlations**:
   - Yellig & Metcalfe (1980) pure CO₂ polynomial coefficients verified against SPE-7477-PA.
   - Yuan et al. (2005) multi-component MMP coefficients verified against SPE-89359-PA with corrected intercept ($1.356$).
6. **Material Balance Mass Conservation Closure**:
   - Formulation $M_{\text{stored}} = M_{\text{purchased}} - M_{\text{uncaptured}} - M_{\text{leakage}}$ in `analysis/material_balance.py` verified closed ($0.0\%$ residual error).
   - Closed-loop gross balance $\text{Gross Injected} = \text{Purchased} + \text{Recycled} = \text{Net Stored} + \text{Total Leakage} + \text{Gross Produced}$ verified in `utils/run_exporter.py`.
7. **Reservoir Unit Conversions & Grid Dimensions**:
   - Permeability factor $1.0623 \times 10^{-14}\text{ ft}^2/\text{mD}$ verified against primary physical constants.
   - Acre-to-sqft conversion ($43,560\text{ ft}^2/\text{acre}$) verified in `core/data_integration_engine.py`.
8. **Optimization Diagnostic Analytics & Well Scheduling**:
   - Multi-generation coverage tracking (trend, min, max) with shaded envelope in `core/plotting_manager.py:plot_coverage`.
   - Symmetric Euclidean distance matrix heatmap in normalized parameter space $[0, 1]^d$ in `core/plotting_manager.py:plot_euclidean_distance_matrix`.
   - Interval-accurate multi-well operational schedule plotting with zero-rate event tracking.

