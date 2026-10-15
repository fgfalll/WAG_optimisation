# Scientific Decisions & Formulation Rationale

This document documents the physical assumptions, scientific formulations, trade-offs, and critical critiques of the scientific modeling choices in this codebase.

---

## 1. Miscibility Representation: Sharp Piecewise vs Continuous Formulation

### Formulation Reality
In [core/analytical_models.py:709-712](file:///d:/rep/4.6/co2eor_optimizer/core/analytical_models.py#L709-L712):
```python
if pressure >= mmp:
    miscibility_factor = 1.0 - np.exp(-1.5 * (pressure - mmp) / max(mmp, 1.0))
else:
    miscibility_factor = 0.0
```

### Scientific Critique & Trade-offs
- **Intended Design**: Documentation and validation reports claim a smooth, continuous $\tanh$ transition:
  $$f_{\text{misc}} = \frac{1}{2}\left[1 + \tanh\left(\frac{P - P_{\text{MMP}}}{0.1 P_{\text{MMP}}}\right)\right]$$
  implemented in `PhDHybridSurrogate.get_miscibility_weight()`.
- **Actual Execution**: The code executes a piecewise branch with a sharp derivative discontinuity at $P = P_{\text{MMP}}$ (first derivative jumps from $0$ to $1.5/P_{\text{MMP}}$).
- **Optimization Impact**: Gradient-free optimizers (PSO, GA) can cross this boundary, but gradient-based methods or sensitivity analyzers will encounter numerical stiffness or false local minima at the MMP cusp.
- **Scientific Reality**: In reservoir porous media, multicomponent vaporizing-condensing drive mechanisms produce near-miscible extraction across an envelope around MMP ($0.85 P_{\text{MMP}} < P < 1.15 P_{\text{MMP}}$). Setting recovery boost to exactly $0.0$ below MMP ignores significant extraction, swelling, and viscosity reduction in the immiscible/near-miscible regime.

---

## 2. 1D Koval Model vs 3D Multi-Phase Todd-Longstaff

### Formulation in Code
The codebase incorporates two viscous fingering models:
1. **Koval (1963)** in [core/analytical_models.py:270-360](file:///d:/rep/4.6/co2eor_optimizer/core/analytical_models.py#L270-L360):
   $$K_{\text{eff}} = H \cdot \left[0.78 + 0.22 \left(\frac{\mu_o}{\mu_s}\right)^{1/4}\right]^4$$
   Where $H$ is the heterogeneity factor derived from the Dykstra-Parsons coefficient:
   $$\log(H) = \frac{V_{\text{DP}}}{(1 - V_{\text{DP}})^{0.2}}$$
2. **Todd-Longstaff (1972)** in [core/Phys_engine_full/displacement.py:284-315](file:///d:/rep/4.6/co2eor_optimizer/core/Phys_engine_full/displacement.py#L284-L315):
   Mixing parameter $\omega \in [0, 1]$ (default 0.67) blending pure solvent and pure oil into effective dispersed and bypass phases:
   $$\mu_{se} = \mu_s^{1-\omega} \mu_m^\omega, \quad \mu_{oe} = \mu_o^{1-\omega} \mu_m^\omega$$

### Scientific Assessment
- Koval's model is valid for 1D stable/unstable miscible solvent displacement in heterogeneous layered sands. It cannot capture gravity tongue formation in thick dipping reservoirs.
- Todd-Longstaff is standard in commercial field simulators (ECLIPSE, CMG GEM) for coarse-grid pseudo-miscibility.
- **Divergence**: The surrogate pipeline (`PhDHybridSurrogate`) uses Koval's $K$ factor, while the unused numerical engine uses Todd-Longstaff. Results between the two will diverge if gas overrides oil due to high density differences ($\Delta \rho > 0.3 \text{ g/cm}^3$).

---

## 3. 0D Tank Pressure Integration vs Compressibility

### Formulation Reality
In [core/surrogate_engine.py:904-920](file:///d:/rep/4.6/co2eor_optimizer/core/surrogate_engine.py#L904-L920):
$$\frac{dP}{dt} = \frac{q_{\text{inj}} - q_{\text{prod}}}{V_p \cdot c_t}$$

### Unit Defect & Physical Warning
- $q_{\text{inj}}$ is passed from `FastProfileGenerator` in MSCFD (surface gas volume) or STB/D (water).
- $q_{\text{prod}}$ is computed in STB/D (oil, water) and MSCFD (solution + free gas).
- In the surrogate ODE, injection is directly subtracted from production without applying formation volume factors ($B_g, B_o, B_w$):
  ```python
  net_rate_rb_d = q_inj_rb - q_prod_rb  # q_inj_rb has NOT been converted via B_g!
  dp_dt = net_rate_rb_d / (pore_volume_rb * total_compressibility_psi_inv)
  ```
- **Consequence**: Because $B_g \approx 0.003 - 0.007 \text{ res-bbl/scf}$ (or $0.5 - 1.0 \text{ RB/MSCF}$ at 2000–3000 psi), treating MSCFD directly as res-bbl overstates the pressure support of injected gas by 200% to 500%, leading to an artificially rapid pressure increase in the surrogate engine.

---

## 4. Empirical WAG Performance Modulation

### Code Logic
In [core/analytical_models.py:840-880](file:///d:/rep/4.6/co2eor_optimizer/core/analytical_models.py#L840-L880):
- Peak recovery enhancement is hardcoded at $WAG_{\text{ratio}} = 1.0$:
  $$F_{\text{WAG}} = 1.0 + 0.15 \cdot \exp\left(-0.5 \cdot \left(\frac{R_{\text{WAG}} - 1.0}{0.5}\right)^2\right)$$
- If $WAG_{\text{ratio}} > 3.0$, penalty decay is applied.

### Scientific Assessment
- In real reservoirs, the optimal WAG ratio is governed by fluid mobility ratio ($M$), relative permeability hysteresis, water salinity, and reservoir dip/gravity number ($N_g$).
- In high-viscosity oils ($\mu_o > 15 \text{ cP}$), optimal WAG ratios typically shift towards higher water ratios ($1.5:1$ to $2.5:1$) to suppress severe gas fingering.
- In light-oil low-dip reservoirs with high vertical permeability ($k_v/k_h > 0.1$), gas channelling may dominate unless WAG cycles are short.
- The hardcoded Gaussian peak at $1.0$ is an idealized empirical assumption that artificially biases any optimization algorithm towards selecting $WAG = 1.0 \pm 0.2$.

---

## 5. CO₂ Trapping Mechanisms & Mass Balance

### Formulations
1. **Residual Trapping**: Land (1968) correlation ($C = 1.27$):
   $$S_{gr} = \frac{S_{gi}}{1 + C \cdot S_{gi}}$$
2. **Dissolution Trapping**: Henry's Law / Chang et al. (1998) brine solubility:
   $$R_{s,\text{CO2}} = f(P, T, \text{Salinity})$$
3. **Structural / Hydrodynamic Trapping**: Remaining mobile gas cap.
4. **Mineral Trapping**: Ignored or simplified to a zero-rate/first-order reaction.

### Material Balance Flaw
As audited in [agent_wiki/audit/suspicious_logic.md](file:///d:/rep/4.6/co2eor_optimizer/agent_wiki/audit/suspicious_logic.md), `analysis/material_balance.py` double-deducts recycled gas volumes, occasionally producing negative stored masses. Future agents modifying carbon accounting must ensure that recycled streams are properly partitioned between closed-loop reinjection and external makeup.

---

## 6. Elimination of Artificial Result Modifiers & Penalty Dilution Hacks

### Scientific Assessment of Artificial Modifiers
- **Class E Modifiers (Synthesizing Missing Physical Data)**:
  In previous iterations, `core/objectives/wrapper.py` computed an artificial storage efficiency via $\max(0.3, 0.5 \times (RF / 0.35))$ when simulation profiles lacked explicit CO₂ retention data.
  *Scientific Critique*: This violates the core physical principle that CO₂ storage is a mass conservation quantity, not a recovery factor byproduct. In a primary depletion or waterflood scenario with high $RF$ and zero CO₂ injection, this formula credited the operator with ~50% CO₂ storage efficiency, completely falsifying CCUS accounting.
  *Resolution*: Eradicated. Missing profile data evaluates strictly to `float("nan")`, and candidate solutions receive the full mathematical failure penalty.

### Scientific Assessment of Penalty Dilution
- **Softened Penalties (`FAILURE_PENALTY * 0.1` and `* 0.8`)**:
  In multi-objective optimization, when candidate solutions violate geomechanical constraints, break mass balance, or produce unphysical metrics, assigning a softened penalty allows unphysical parameter combinations to remain competitive against marginally performing physical solutions.
  *Scientific Critique*: Metaheuristics (GA, PSO, DE) exploit gradient gradients in softened penalties to drift toward boundary violations if the objective landscape offers higher apparent returns.
  *Resolution*: Strict binary pruning. Infeasible or unphysical solutions receive an un-diluted `FAILURE_PENALTY` ($-10^{12}$), creating an impenetrable barrier that forces the solver population to remain strictly within the physical feasibility envelope.
