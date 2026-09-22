# Hidden Calibration & Artificial Fitting Audit

## 1. Executive Summary

A critical scientific finding of this audit is the presence of **unstated calibrations, heuristic tuning multipliers, and empirical fittings** embedded in the physics and profile generation layers.

While some parameters are documented in `docs/surrogate_engine_fitting_parameters.md` as calibration levers, several other adjustments previously operated silently without explicit notice. Recent refactorings have begun replacing crude heuristic steps with mass-conserving physical formulations.

---

## 2. Inventory of Calibrations & Empirical Fittings

### A. Transverse Mixing Multiplier (`transverse_mixing_calibration`)
- **Location**: [core/engine_surrogate/surrogate_engine.py:192](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L192) and [data_models.py:2243](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L2243).
- **Equation**:
  $$H_k = \frac{1}{(1 - V_{DP} \cdot C_{trans})^2}$$
- **What is Being Fitted**: The effective reservoir heterogeneity factor $H_k$ in the Koval displacement formula.
- **Why It Exists**: To prevent premature CO₂ breakthrough in layered models. In the standard Koval formulation, $H_k = 1 / (1 - V_{DP})^2$. At $V_{DP} = 0.8$, $H_k = 25$, predicting immediate breakthrough ($t_{D,bt} = 1/25 = 0.04\text{ PVI}$). Multiplying $V_{DP}$ by $C_{trans} = 0.80$ lowers effective $V_{DP}$ to 0.64, yielding $H_k = 7.7$, artificially delaying breakthrough by a factor of 3 to match CMG GEM benchmarks.
- **Physical Meaning**: Represents vertical and transverse dispersion / crossflow mitigating severe channeling.
- **Scientific Audit Verdict**: Physically plausible mechanism, but acts as a manual tuning knob to force-match CMG or field breakthrough times without geologic crossflow verification.

### B. WAG Phase Mobility Buffering & Mass Preservation [RESOLVED]
- **Location**: [core/engine_surrogate/profile_generator_fast.py:309-408](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L309-L408).
- **Previous Defect**:
  - Crude fixed discrete multipliers: `wag_oil_gas_bonus = 1.08` (+8% oil), `wag_oil_water_penalty = 0.96` (-4% oil), and `wag_water_gas_reduction = 0.92` (-8% water).
  - Production curves exhibited artificial sawtooth jumps and destroyed mass balance.
- **Current Physics Formulation**:
  - Replaced with relative phase mobility contrast:
    $$\Delta \lambda / \Sigma \lambda = \frac{\lambda_g - \lambda_w}{\lambda_o + \lambda_g + \lambda_w}$$
    $$\text{amp} = \text{clip}(0.1 \times \text{contrast}, -0.15, 0.15)$$
  - Mass conservation is strictly enforced: modulated profiles are re-normalized so that cumulative production matches $N_p = OOIP \times RF$.
- **Scientific Audit Verdict**: Now grounded in multi-phase mobility buffering (Stalkup 1983; Lake 1989) with exact volume conservation.

### C. Modified Cronquist MMP Formula (`55 - API`)
- **Location**: [evaluation/mmp.py:111](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py#L111) and [analytical_models.py:554](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L554).
- **Equation**:
  $$P_{MMP} = 15.988 \cdot T^{0.744206} \cdot (55.0 - \gamma_{API})^{0.279033}$$
- **What is Being Fitted**: The effect of crude oil gravity on minimum miscibility pressure.
- **Origin**: The published Cronquist (1978) correlation has different variables. The developer modified the term to $(55 - \gamma_{API})$ to force an inverse power relationship so that higher API yields lower MMP.
- **Scientific Audit Verdict**: Unvalidated manual alteration of a published empirical equation. It reproduces reasonable trends for medium-to-light oils ($\sim 30-45^\circ\text{API}$), but breaks down if $\gamma_{API} \ge 55^\circ\text{API}$ (producing complex or negative numbers).

### D. Zero-Injection Storage Efficiency Override (Optimizer Cheat) [RESOLVED]
- **Location**: [core/optimisation_engine.py:1708-1729](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1708-L1729).
- **Previous Defect**:
  - When `total_injected_mscf <= 0` and `recovery_factor > 0.05`, the optimizer injected an artificial `eval_results["storage_efficiency"] = 0.3` or returned diluted penalties (`FAILURE_PENALTY * 0.1`).
  - This prevented the GA penalty function from killing zero-injection candidates in storage optimization runs.
- **Scientific Audit Verdict**: **Completely unphysical**. Injected an artificial 30% storage efficiency into a scenario where zero CO₂ was injected.
- **Resolution**:
  - The artificial override and dilution multiplier were completely eradicated.
  - When zero CO₂ is injected, true physical storage efficiency is evaluated strictly as `0.0`.
  - If a zero-injection candidate fails operational requirements ($RF \le 0.05$), or if an injected candidate achieves negligible storage, it is immediately pruned with full `FAILURE_PENALTY` ($-10^{12}$).
