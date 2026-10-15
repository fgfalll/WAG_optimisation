# Conservation Tests: Mass & Closed-Loop Carbon Balance Verification

## 1. Scope & Physical Conservation Laws

Conservation of mass is the inviolable bedrock of all continuum mechanics and reservoir engineering. In CO₂ EOR with carbon sequestration, two simultaneous conservation invariants must hold:
1. **Total Hydrocarbon Mass Conservation**: Cumulative oil produced cannot exceed the mobile original oil in place:
   $$N_p(t) \le N_{\text{mobile}} = \text{OOIP} \cdot \frac{1 - S_{wi} - S_{or}}{1 - S_{wi}}$$
2. **Closed-Loop Carbon Mass Conservation**: Injected carbon mass must exactly balance net subsurface storage, produced carbon, and fugitive leakage:
   $$\text{Gross Injected} = \text{Purchased CO2} + \text{Recycled CO2} = \text{Net Stored} + \text{Leakage} + \text{Cumulative Produced}$$

---

## 2. Mass Balance Audit & Verification Results

### 2.1 Ultimate Oil Recovery Bound vs Mobile OOIP

- **Physical Equation**:
  $$RF_{\text{max}} = \frac{1 - S_{wi} - S_{or}}{1 - S_{wi}}$$
- **Code Audit**: [`core/engine_surrogate/analytical_models.py:881`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L881)
  The code erroneously bounded recovery factor by:
  $$RF \le 1 - S_{wi} - S_{or}$$
  This expresses mobile oil as a fraction of total pore volume ($V_p$) rather than a fraction of initial oil in place ($V_p(1 - S_{wi})$), artificially underestimating maximum recoverable oil by a factor of $(1 - S_{wi}) \approx 20\% - 30\%$.
- **Verification Test**: [`tests/scientific/conservation/test_mass_conservation.py::test_pore_volume_vs_ooip_recovery_bound_discrepancy`](file:///d:/rep/4.6/co2eor_optimizer/tests/scientific/conservation/test_mass_conservation.py) demonstrates that this bound truncates realistic recovery.
- **Audit Status**: **SCIENTIFIC FLAW (SCI-FLAW-11)**.

---

### 2.2 Closed-Loop Carbon Balance & Recycle Constraints

In CO₂-EOR operations, produced associated gas is processed in surface separation facilities, and CO₂ is re-compressed and recycled into injection wells.

**Invariants**:
1. **Recycle Ceiling**: Cumulative recycled CO₂ cannot exceed cumulative produced CO₂:
   $$\sum M_{\text{recycled}} \le \sum M_{\text{produced}}$$
2. **Purchased Gas Balance**:
   $$\sum M_{\text{injected}} = \sum M_{\text{purchased}} + \sum M_{\text{recycled}}$$
3. **Net Storage Definition**:
   $$M_{\text{net stored}} = \sum M_{\text{injected}} - \sum M_{\text{produced}} = \sum M_{\text{purchased}} - (1 - \eta_{\text{recycle}})\sum M_{\text{produced}}$$

**Verification Tests**:
- [`tests/scientific/conservation/test_carbon_accounting.py::test_closed_loop_carbon_balance_invariant`](file:///d:/rep/4.6/co2eor_optimizer/tests/scientific/conservation/test_carbon_accounting.py):
  Evaluates `SurrogateEngine.evaluate_scenario()` over a 10-year injection schedule.
  - Invariant 1: Injected = Purchased + Recycled ($\Delta < 10^{-6}\text{ MSCF}$).
  - Invariant 2: Recycled $\le$ Produced.
  - **Status**: **VERIFIED** in `SurrogateEngine`.
- [`tests/scientific/conservation/test_carbon_accounting.py::test_material_balance_analyzer_closed_loop`](file:///d:/rep/4.6/co2eor_optimizer/tests/scientific/conservation/test_carbon_accounting.py):
  Evaluates `analysis/material_balance.py::MaterialBalanceAnalyzer`.
  - Net storage matches $\sum q_{\text{inj}} - \sum q_{\text{prod}}$.
  - **Status**: **VERIFIED**.

---

### 2.3 Double-Subtraction Defect Resolution

Previous versions of `analysis/material_balance.py` exhibited a double-subtraction defect where produced CO₂ was subtracted twice (once as recycled gas and once as fugitive loss), creating artificial negative storage inventories. This bug has been verified as eliminated in the active codebase.
