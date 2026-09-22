# Conservation & Material Balance Audit

## 1. Overview of Physical Conservation Principles

In a rigorous reservoir simulation of Carbon Capture, Utilization, and Storage (CCUS) and CO₂ Enhanced Oil Recovery (EOR), four fundamental conservation laws must hold across every time step and boundary:

1. **Overall Mass Conservation**:
   $$\text{Mass Injected} = \text{Mass Produced} + \Delta \text{Mass Stored} + \text{Mass Leaked}$$
2. **Component Mass Conservation ($CO_2$, Hydrocarbon fractions, Brine)**:
   $$M_{i,\text{inj}} = M_{i,\text{prod}} + \Delta M_{i,\text{res}} \quad \forall i \in \{CO_2, C_1\dots C_{7+}, H_2O\}$$
3. **Volume Balance (Pore Volume Saturation Constraint)**:
   $$S_o(\mathbf{x}, t) + S_w(\mathbf{x}, t) + S_g(\mathbf{x}, t) = 1.0 \quad \forall \mathbf{x}, t$$
   $$V_{\text{fluid}}(P, T) \le V_{\text{pore}}(P) = V_{p0} [1 + c_f(P - P_0)]$$
4. **Energy / Thermodynamic Consistency**:
   Phase equilibrium flash calculations must minimize Gibbs free energy ($\sum z_i \ln f_i$).

---

## 2. Realized Conservation Behavior in Active Codebase

### A. Mass Balance in Primary Surrogate Engine (`core/engine_surrogate`)
- **Status**: **Semi-Analytical Material Balance**.
- **Implementation**: [SurrogateEngine._calculate_co2_purchased_recycled()](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L950)
- **Equation**:
  $$\text{Purchased CO}_2 = \text{Total Injected CO}_2 - \text{Recycled CO}_2$$
  where recycled CO₂ begins only after CO₂ breakthrough:
  $$t > t_{bt} \implies q_{\text{recycled}}(t) = q_{\text{gas,prod}}(t) \cdot f_{\text{capture}} \cdot y_{CO2}$$
- **Findings**:
  1. The calculation correctly tracks that purchased fresh CO₂ plus recycled produced CO₂ equals total injection rate.
  2. However, dissolved CO₂ in produced oil and brine is neglected; all produced gas is assumed to carry the entire produced CO₂ mass.

---

### B. Double-Subtraction Defect in `analysis/material_balance.py` (RESOLVED)
- **Status**: **RESOLVED / VERIFIED EXACT**.
- **Location**: [analysis/material_balance.py:213-234](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L213-L234).
- **Historical Error**: Code computed `net_injection_tonne = purchased_tonne - recycled_tonne_raw` and then subtracted produced CO₂ again, double-subtracting recycled CO₂ and driving calculated net storage artificially negative.
- **Resolution**: Aligned with OSTI-1204577 Equation 6:
  - Fresh CO₂ entering the reservoir system boundary is $I_{\text{purchased}}$.
  - Produced CO₂ consists of recycled gas plus uncaptured gas: $P_{CO2} = I_{\text{recycled}} + P_{\text{uncaptured}}$.
  - Net CO₂ stored before leakage is:
    $$\Delta M = I_{\text{purchased}} - P_{\text{uncaptured}}$$
  - Recycled CO₂ correctly cycles internally without being subtracted twice. Net mass balance closes to machine precision.

---

### C. Reservoir Tank ODE Rate Balance Unit Mismatch (RESOLVED)
- **Status**: **RESOLVED / VERIFIED CONSISTENT**.
- **Location**: [core/engine_surrogate/surrogate_engine.py:918-925](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L918-L925).
- **Historical Error**: `q_inj_rb` was set directly to `profile_result["injection_profile"]` (in MSCFD) while `q_prod_rb` was computed in reservoir barrels per day (RB/d), violating dimensional homogeneity in $dP/dt$.
- **Resolution**: Gas injection is explicitly converted to reservoir barrels per day:
  ```python
  q_inj_rb = profile_result["injection_profile"] * 1000.0 * bg
  q_prod_rb = (
      profile_result["oil_profile"] * bo
      + profile_result["water_profile"] * 1.0
      + profile_result["gas_profile"] * 1000.0 * bg
  )  # RB/day
  net_rate = q_inj_rb - q_prod_rb
  ```
  Both injection and production rate terms are strictly in RB/d, ensuring proper dimensional consistency for the stiff BDF pressure solver.

---

### D. Saturation Bounds and Normalization
- In `FastProfileGenerator`, phase rates are derived from fractional flow curves and empirical recovery profiles rather than cell-by-cell numerical saturation solving.
- Saturation sum $S_o + S_w + S_g = 1.0$ is not continuously enforced on a spatial grid in the surrogate engine because the surrogate is 0D/1D semi-analytical.
- In `core/compositional_engine`, saturation sum is explicitly enforced at every time step:
  $$\sum_{\alpha} S_\alpha = 1.0 \pm 10^{-7}$$
  but this engine is orphaned and inactive during production optimization runs.

---

## 3. Conservation Verification Checklist for Agents

Before modifying pressure or storage equations, verify:
- [ ] Mass of injected CO₂ equals mass of stored CO₂ plus produced CO₂ plus emissions.
- [ ] Injection rates are converted to reservoir volume units ($RB/d$) using accurate $B_g(P, T)$ before performing pressure material balance calculations.
- [ ] Fresh purchased CO₂ is not conflated with gross injected CO₂.
- [ ] Recycled gas streams are tracked with a single mass balance node.
