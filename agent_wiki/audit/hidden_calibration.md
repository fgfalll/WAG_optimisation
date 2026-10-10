# Hidden Calibration & Artificial Fitting Audit

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

## 1. Executive Summary

A critical scientific finding of this audit is the presence of **unstated calibrations, heuristic tuning multipliers, and empirical fittings** embedded in the physics and profile generation layers.

While some parameters are documented in [`agent_wiki/data/parameters.md`](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/data/parameters.md#e-empiricalfittingparameters-line-2241) as calibration levers, several other adjustments previously operated silently without explicit notice. Recent refactorings have begun replacing crude heuristic steps with mass-conserving physical formulations.

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

---

