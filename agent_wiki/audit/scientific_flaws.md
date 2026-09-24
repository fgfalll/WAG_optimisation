# Scientific Flaw Register

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

This document catalogs confirmed scientific flaws, unphysical equations, and thermodynamic contradictions in the CO₂ EOR Optimizer codebase.

---

## Master Scientific Flaw Summary

| ID | Severity | Model / Location | Physical Principle Violated | Scientific Impact | Status |
|:---|:---|:---|:---|:---|:---|
| **SCI-FLAW-01** | **CRITICAL** | `profile_generator_fast.py:895-901` | Buckley-Leverett / Koval Fractional Flow | Inverts gas breakthrough: unfavorable mobility yields 10% gas, favorable yields 70% | OPEN |
| **SCI-FLAW-04** | **HIGH** | `unified_engine/co2_properties.py:140` | Isobaric Thermal Expansion | Supercritical CO₂ density increases with temperature ($\partial\rho/\partial T > 0$) | OPEN |
| **SCI-FLAW-05** | **CRITICAL** | `optimisation_engine.py:1410`, `profile_generator_fast.py:501` | Darcy Inflow & Pattern Interference | Single producer drains 1,354 acres at 7,270 BOPD flat plateau for 6.4 years | OPEN |
| **SCI-FLAW-06** | **CRITICAL** | `optimisation_engine.py:3215`, `surrogate_engine.py:948` | Thermodynamic State Consistency | Optimizer surrogate evaluated at 4,450 psia, while reservoir tank ODE is at 3,100 psia | OPEN |
| **SCI-FLAW-07** | **HIGH** | `core/objectives/wrapper.py:212`, `surrogate_models.py:465` | EOR Fluid Displacement Accounting | Net CO₂ utilization reported as 0.665 MSCF/STB (deflated by 10× vs DOE benchmark) | OPEN |
| **SCI-FLAW-08** | **CRITICAL** | `unified_engine/physics/eos/__init__.py:195` | Cubic EOS Phase Identification | Inverted phase label: $Z < 0.8$ labeled Vapor, $Z \ge 0.8$ labeled Liquid | OPEN |
| **SCI-FLAW-09** | **HIGH** | `simulation/recovery_models.py:197-202` | Miscibility Thermodynamics ($\sigma \to 0$) | IFT at MMP remains at maximum immiscible value ($20\text{ mN/m}$) | OPEN |
| **SCI-FLAW-10** | **HIGH** | `simulation/recovery_models.py:498-501` | Fractional Flow Displacement | Immiscible recovery models only capillary desaturation $\Delta S_{or}$, predicting 1-2% RF | OPEN |
| **SCI-FLAW-11** | **MEDIUM** | `analytical_models.py:205` | Conservation & Ultimate Recovery | Silent 80% recovery factor ceiling erases optimization gradients | OPEN |
| **SCI-FLAW-12** | **HIGH** | `optimisation_engine.py:98`, `surrogate_engine.py:201` | Gas Volumetric Factor ($B_g$) | 10× discrepancy in $B_g$ ($0.50$ vs $5.0\text{ RB/MSCF}$) between engine and optimizer | OPEN |
| **SCI-FLAW-14** | **HIGH** | `analysis/material_balance.py:85-108` | Thermodynamic Vapor-Liquid Equilibrium | Heuristic vapor fraction formula $V = 1 - Z + 0.2$ has zero physical basis | OPEN |
| **SCI-FLAW-15** | **MEDIUM** | `profile_generator_fast.py:983-986` | Reservoir Material Balance | Produced CO₂ tied directly to instantaneous injection; shut-in zeroes production | OPEN |
| **SCI-FLAW-16** | **HIGH** | `core/engine_surrogate/surrogate_models.py:164-182` | Areal Sweep Continuity | Discontinuous 48% cliff in Craig areal sweep correlation at $M = 1.0$ | OPEN |
| **SCI-FLAW-17** | **HIGH** | `core/engine_surrogate/surrogate_models.py:238-241` | Capillary Trapping Inversion | Trapping calculated as $1.0 - S_{gc}$; higher critical gas reduces trapping | OPEN |
| **SCI-FLAW-18** | **CRITICAL** | `core/unified_engine/physics/eos/__init__.py:206, 257-270` | Peng-Robinson Fugacity Formulation | Corrupted PR fugacity equation omits $2\sqrt{2}B$ denominator and partial derivatives | OPEN |

---

## Detailed Scientific Flaw Records

### SCI-FLAW-01: Inverted CO₂ Fractional Flow in Profile Generator
- **ID**: `SCI-FLAW-01`
- **Severity**: **CRITICAL**
- **Location**: [`core/engine_surrogate/profile_generator_fast.py:895-901`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L895-L901)
- **Model**: Koval (1963) / Buckley-Leverett Gas Fractional Flow
- **Observation**: In calculating post-breakthrough CO₂ fractional flow, the effective viscosity ratio $E$ is divided by the mobility ratio $M$, and $(M - 1) \times 0.5$ is added to the denominator.
- **Equation**:
  ```python
  E = (0.78 + 0.22 * (mobility_ratio ** 0.25)) ** 4
  koval_factor = (E / mobility_ratio) * koval_mult
  koval_factor = float(np.clip(koval_factor, 0.5, 10.0))
  frac_flow_co2 = koval_factor / (koval_factor + (mobility_ratio - 1) * 0.5)
  frac_flow_co2 = float(np.clip(frac_flow_co2, 0.1, 0.7))
  ```
- **Expected Behavior**: As mobility ratio worsens ($M = \mu_o / \mu_g \gg 1$), viscous fingering intensifies, causing higher gas fractional flow ($F_{\text{CO2}} \to 1.0$).
- **Actual Behavior**:
  - At $M = 1.0$ (favorable piston displacement): `frac_flow_co2 = 0.70` (maximum gas breakthrough).
  - At $M = 10.0$ (severe viscous fingering): `frac_flow_co2 = 0.10` (minimum gas breakthrough).
- **Evidence**: Mathematical limits of the code expression above.
- **Scientific Consequence**: Inverts the core physics of gas breakthrough and GOR development. Heavy oils with severe fingering produce negligible gas, while light oils with stable displacement produce maximum gas.
- **Numerical Consequence**: Distorts objective function scoring, misdirects WAG ratio optimization, and reverses genetic algorithm selection pressure.
- **Affected Outputs**: `gas_profile`, `co2_gas_profile`, `annual_co2_produced_mscf`, `annual_co2_recycled_mscf`, `co2_utilization`.
- **Confidence**: 100% (Direct analytical proof).
- **Recommended Investigation**: Replace with authentic Koval (1963) fractional flow: $K = H_k \cdot E \ge 1.0$, and $F_{\text{CO2}}(S) = \frac{K S}{1 + S(K - 1)}$.
- **Status**: OPEN.

---

### SCI-FLAW-04: Inverted Thermal Expansion in Empirical CO₂ Density
- **ID**: `SCI-FLAW-04`
- **Severity**: **HIGH**
- **Location**: [`core/unified_engine/physics/co2_properties.py:140`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/co2_properties.py#L140)
- **Model**: Empirical CO₂ Density Correlation
- **Equation**:
  $$\rho[\text{g/cm}^3] = 1.01 + 0.0109 \cdot P_{\text{MPa}} - 1.25\times 10^{-5} \cdot P_{\text{MPa}}^2 + 0.0023 \cdot T_{^\circ\text{C}}$$
- **Expected Behavior**: Thermal expansion causes density to decrease as temperature rises: $\frac{\partial\rho}{\partial T} < 0$.
- **Actual Behavior**: The temperature coefficient is $+0.0023 > 0$, causing hot CO₂ to be denser than cold CO₂. Furthermore, at $10\text{ MPa}$ ($1,450\text{ psi}$) and $50^\circ\text{C}$, the formula yields $1,232\text{ kg/m}^3$ (NIST reference is $384\text{ kg/m}^3$, an error of $+220\%$).
- **Scientific Consequence**: Massive overestimation of downhole CO₂ density; fluid is modeled as denser than water.
- **Affected Outputs**: `CO2Properties.density(method="empirical")`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-05: Single-Well Point Drainage Anomaly
- **ID**: `SCI-FLAW-05`
- **Severity**: **CRITICAL**
- **Location**: [`core/optimisation_engine.py:1410-1425`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1410-L1425), [`core/engine_surrogate/profile_generator_fast.py:501-516`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L501-L516)
- **Model**: Well Schedule and Field Rate Synthesis
- **Observation**: When optimization scenarios do not define explicit pattern wells, the system assigns the entire recovery of a multi-million-barrel field ($48.5\text{ MMSTB}$ OOIP, $1,354\text{ acres}$) to a single producer (`Well-Producer-1`).
- **Equation / Implementation**: `peak_rate_estimate = ultimate_recovery / plateau_time / 365.25`, resulting in $q_o = 7,270\text{ STB/day}$ sustained for 6.4 years from one vertical wellbore.
- **Expected Behavior**: Single-well inflow is constrained by Darcy drawdown ($q = J \cdot (P_{res} - P_{wf}) \le 300 - 800\text{ BOPD}$). Draining $1,354\text{ acres}$ requires 30–35 pattern wells.
- **Scientific Consequence**: Complete circumvention of well spacing physics, pattern interference, and capital expenditure scaling. Optimizers harvest field-scale reserves with single-well CAPEX/OPEX.
- **Affected Outputs**: Well schedules, field cash flows, capital expenditure, NPV.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-06: Decoupled Decision Variable Pressure vs Tank ODE Pressure
- **ID**: `SCI-FLAW-06`
- **Severity**: **CRITICAL**
- **Location**: [`core/optimisation_engine.py:3215-3222`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L3215-L3222), [`core/engine_surrogate/surrogate_engine.py:948-965`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L948-L965)
- **Model**: Surrogate Evaluation vs Dynamic Tank Material Balance
- **Observation**: The optimizer treats `pressure` as an unconstrained decision variable, pegging it to the upper search bound ($4,450\text{ psia}$). This phantom pressure is passed directly to `AnalyticalSurrogate.predict()`, maximizing ultimate recovery ($RF = 41.5\%$) and driving miscibility weight $\omega \to 1.0$.
- **Actual Behavior**: The coupled 0D material balance tank ODE (`_solve_pressure_ode_stiff`) calculates in-situ reservoir pressure independently, simulating dynamic pressures between $3,080\text{ psia}$ and $3,335\text{ psia}$ across the 15-year life. At no point does the reservoir reach $4,450\text{ psia}$.
- **Scientific Consequence**: Simulation outputs and recovery factors are calculated under a fictitious high-pressure thermodynamic regime that is physically disconnected by $\sim 1,350\text{ psi}$ from the reservoir depletion state.
- **Affected Outputs**: `recovery_factor`, `npv`, `miscibility_degree`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-07: Unphysical CO₂ Net Utilization Factor
- **ID**: `SCI-FLAW-07`
- **Severity**: **HIGH**
- **Location**: [`core/objectives/wrapper.py:212`](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L212), [`core/engine_surrogate/surrogate_models.py:465-495`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L465-L495)
- **Model**: EOR Project Carbon Utilization Metric
- **Equation**:
  $$\text{Utilization} = \frac{\sum M_{\text{CO2,purchased}}}{\sum N_{p,\text{total}}}$$
- **Expected Behavior**: Net utilization in petroleum engineering is defined relative to *incremental EOR oil* attributable to CO₂ injection ($N_{p,\text{EOR}} = N_{p,\text{total}} - N_{p,\text{primary}}$), with typical field values of $5.0 - 12.0\text{ MSCF/STB}$ ($0.25 - 0.60\text{ tonne/STB}$).
- **Actual Behavior**: Total field cumulative oil (including primary depletion and secondary waterflood) is placed in the denominator. This deflates reported utilization to $0.665\text{ MSCF/STB}$ ($0.0352\text{ tonne/STB}$).
- **Scientific Consequence**: A net utilization of $0.665\text{ MSCF/STB}$ implies that injecting only $1.38\text{ RB}$ of CO₂ produces $1.0\text{ STB}$ ($1.30\text{ RB}$) of oil, which is physically impossible given connate water, residual oil, and phase dissolution. The optimizer exploits this by driving injection rates to the lower boundary to minimize gas purchase costs while harvesting full field recovery.
- **Affected Outputs**: `co2_utilization`, `net_utilization_mscf_per_stb`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-08: Inverted Phase State Classification in Cubic EOS
- **ID**: `SCI-FLAW-08`
- **Severity**: **CRITICAL**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:195`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L195)
- **Model**: Cubic Equation of State Phase Identification
- **Equation**:
  ```python
  "phase": "V" if Z < 0.8 else "L"
  ```
- **Expected Behavior**: Liquid compressibility factor $Z_L$ is small ($Z_L \approx 0.05 - 0.30$). Vapor compressibility factor $Z_V$ is near unity ($Z_V \approx 0.80 - 1.05$). Therefore, $Z < 0.8$ represents Liquid and $Z \ge 0.8$ represents Vapor.
- **Actual Behavior**: The code inverts the logic: $Z < 0.8$ is classified as Vapor ('V'), and $Z \ge 0.8$ is classified as Liquid ('L').
- **Scientific Consequence**: Dense liquid oil/condensate is labeled as gas vapor, while low-pressure gas is labeled as liquid. Flash calculations and phase routing across the compositional engine are completely compromised.
- **Affected Outputs**: `get_properties_si()["phase"]`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-09: Interfacial Tension Constant at Maximum Immiscible Value at MMP
- **ID**: `SCI-FLAW-09`
- **Severity**: **HIGH**
- **Location**: [`core/simulation/recovery_models.py:197-202`](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L197-L202)
- **Model**: IFT vs Pressure Model
- **Equation**:
  ```python
  if pressure_psi >= mmp_psi:
      sigma = sigma_0 * np.exp(-lambda_ift * (pressure_psi - mmp_psi))
  else:
      sigma = sigma_0
  ```
- **Expected Behavior**: In CO₂-oil systems, interfacial tension approaches zero as pressure approaches MMP ($\lim_{P \to MMP} \sigma(P) = 0$).
- **Actual Behavior**: For all $P < MMP$, $\sigma = \sigma_0 = 20\text{ mN/m}$. At $P = MMP$, $\sigma$ is still $20\text{ mN/m}$ (maximum immiscible value), only decaying exponentially at pressures *above* MMP.
- **Scientific Consequence**: Near-miscible fluids are treated as fully immiscible with high IFT, suppressing capillary desaturation.
- **Affected Outputs**: `interfacial_tension`, `capillary_number`, `residual_oil_saturation`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-10: Immiscible Recovery Model Vanishes Without Capillary Desaturation
- **ID**: `SCI-FLAW-10`
- **Severity**: **HIGH**
- **Location**: [`core/simulation/recovery_models.py:498-501`](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L498-L501)
- **Model**: Immiscible CO₂ Recovery Factor
- **Equation**:
  $$E_d = \frac{S_{or,\text{base}} - S_{or}^*}{1 - S_{wi}}$$
  $$RF_{\text{im}} = E_d \cdot E_A \cdot E_V$$
- **Expected Behavior**: An immiscible gas drive recovers mobile oil by viscous displacement ($S_{oi} \to S_{or}$). A typical immiscible water/gas flood yields $20\% - 45\%$ recovery.
- **Actual Behavior**: The code defines displacement efficiency solely as the capillary-induced reduction from $S_{or,\text{base}}$ to $S_{or}^*$. In the absence of extreme capillary numbers ($N_c \le 10^{-5}$), $S_{or}^* \approx S_{or,\text{base}}$, resulting in $E_d \approx 0.03 - 0.05$ and $RF_{\text{im}} \approx 1\% - 2\%$.
- **Scientific Consequence**: Massive discrepancy between `simulation/recovery_models.py` ($RF_{\text{im}} \approx 2\%$) and `analytical_models.py` ($RF_{\text{im}} \approx 35\%$).
- **Affected Outputs**: `ImmiscibleRecoveryModel.calculate()`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-11: Arbitrary Recovery Factor Cap of 0.80
- **ID**: `SCI-FLAW-11`
- **Severity**: **MEDIUM**
- **Location**: [`core/engine_surrogate/analytical_models.py:205`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L205)
- **Model**: Miscible Displacement Surrogate
- **Equation**: `return float(np.clip(rf, 0.05, 0.80))`
- **Expected Behavior**: The upper physical bound on recovery is determined by connate water and minimum residual oil: $RF_{\max} = 1 - S_{wi} - S_{or,\min}$. In clean homogeneous sands, recovery can exceed 85%.
- **Actual Behavior**: Recovery is hard-capped at 80% with no warning or logger notification.
- **Scientific Consequence**: Artificially flattens objective fitness landscapes and destroys gradient information when $RF \ge 0.80$.
- **Affected Outputs**: `recovery_factor`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-12: Supercritical $B_g$ Discrepancy Across Active vs Legacy Modules
- **ID**: `SCI-FLAW-12`
- **Severity**: **HIGH**
- **Location**: [`core/optimisation_engine.py:98`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L98), [`core/engine_surrogate/surrogate_engine.py:201`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L201)
- **Model**: PVT Formation Volume Factor Conversion
- **Observation**:
  - `surrogate_engine.py:201`: $B_g \approx 0.50\text{ RB/MSCF}$ ($0.0005\text{ RB/SCF}$).
  - `optimisation_engine.py:98`: `B_GAS_RB_PER_MSCF = 5.0` ($0.005\text{ RB/SCF}$).
- **Scientific Consequence**: A factor of $10\times$ discrepancy exists between modules regarding the reservoir volume occupied by injected gas.
- **Affected Outputs**: Downhole voidage replacement, reservoir pressure calculations, rate conversions.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-14: Heuristic Vapor Fraction in Material Balance
- **ID**: `SCI-FLAW-14`
- **Severity**: **HIGH**
- **Location**: [`analysis/material_balance.py:85-108`](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py#L85-L108)
- **Model**: Material Balance Reservoir CO₂ Partitioning
- **Equation**:
  $$V = \min(1.0, \max(0.0, 1.0 - Z_{\text{avg}} + 0.2))$$
  $$y_{\text{CO2}} = x_{\text{CO2}} \cdot V + x_{\text{CO2}} \cdot (1 - V) \cdot (1 - V)$$
- **Expected Behavior**: Vapor fraction $V$ and phase compositions $x, y$ are governed by thermodynamic equilibrium: $f_{i}^L = f_{i}^V$ (Rachford-Rice equation).
- **Actual Behavior**: Vapor fraction is estimated from compressibility factor $Z$ via an arbitrary linear expression ($1 - Z + 0.2$), and phase concentration is multiplied by $(1 - V)^2$.
- **Scientific Consequence**: Fabricated pseudo-thermodynamics with zero theoretical foundation.
- **Affected Outputs**: `_get_co2_fraction_at_reservoir()`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-15: Instantaneous CO₂ Production Decoupled from In-Situ Mobile Storage
- **ID**: `SCI-FLAW-15`
- **Severity**: **MEDIUM**
- **Location**: [`core/engine_surrogate/profile_generator_fast.py:983-986`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L983-L986)
- **Model**: Gas Production Profile Synthesis
- **Equation**:
  $$q_{\text{CO2,prod}}(t) = q_{\text{inj}}(t) \cdot (1 - \text{trapping}) \cdot F_{\text{CO2}} \cdot (1 - e^{-c(t - t_{bt})})$$
- **Expected Behavior**: Produced gas rate post-breakthrough is driven by in-situ free gas mobility and reservoir pressure gradient. If injection is shut in, gas continues to produce as the reservoir depressurizes.
- **Actual Behavior**: Produced CO₂ rate is computed as a direct product of instantaneous injection rate $q_{\text{inj}}(t)$. If $q_{\text{inj}}(t) = 0$, produced CO₂ immediately collapses to zero.
- **Scientific Consequence**: Shutting in injection immediately halts gas production, violating mass transfer and pressure dissipation physics.
- **Affected Outputs**: `co2_gas_profile`, `gas_profile`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-16: Discontinuous Cliff in Craig Areal Sweep at Unit Mobility
- **ID**: `SCI-FLAW-16`
- **Severity**: **HIGH**
- **Location**: [`core/engine_surrogate/surrogate_models.py:164-182`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L164-L182)
- **Model**: Craig-Geffen-Morse Areal Sweep Efficiency Correlation
- **Observation**: Areal sweep efficiency $E_A$ jumps discontinuously by 48% across $M = 1.0$.
- **Equation**:
  ```python
  if mobility_ratio <= 1.0:
      return 1.0
  else:
      return 0.5460 - 0.0195 * np.log(mobility_ratio) + ...
  ```
- **Expected Behavior**: Areal sweep efficiency is a continuous, smooth function of mobility ratio ($\lim_{M \to 1^+} E_A(M) = E_A(1.0)$).
- **Actual Behavior**:
  - For $M \le 1.0$: $E_A = 1.000$ (100% sweep).
  - For $M = 1.0001$: $E_A = 0.517$ (51.7% sweep).
  - A step discontinuity of $\Delta E_A = 0.483$ (48.3%) occurs at $M = 1.0$.
- **Scientific Consequence**: Severe non-physical step change causes gradient-based optimizers to fail and causes genetic algorithms to falsely lock into $M \le 1.0$.
- **Affected Outputs**: `areal_sweep_efficiency`, `recovery_factor`, `cumulative_oil`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-17: Inverted Critical Gas Saturation Trapping Formulation
- **ID**: `SCI-FLAW-17`
- **Severity**: **HIGH**
- **Location**: [`core/engine_surrogate/surrogate_models.py:238-241`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L238-L241)
- **Model**: Residual / Capillary Gas Trapping Model
- **Observation**: Trapping efficiency is computed as $1.0 - S_{gc}$.
- **Equation**:
  ```python
  trapping_efficiency = 1.0 - critical_gas_saturation
  ```
- **Expected Behavior**: Critical gas saturation $S_{gc}$ represents the minimum saturation required for the gas phase to become mobile. Higher $S_{gc}$ means more gas is trapped in pore throats by capillary forces, so trapping efficiency must increase monotonically with $S_{gc}$ ($\frac{\partial \eta_{\text{trap}}}{\partial S_{gc}} > 0$).
- **Actual Behavior**: Because trapping is defined as $1.0 - S_{gc}$, increasing $S_{gc}$ from $0.05$ to $0.25$ causes calculated trapping efficiency to *decrease* from $95\%$ to $75\%$.
- **Scientific Consequence**: Inverts the fundamental physical relationship between capillary snap-off/hysteresis and residual gas retention.
- **Affected Outputs**: `trapping_efficiency`, `co2_stored`, `storage_efficiency`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### SCI-FLAW-18: Corrupted Peng-Robinson Mixture Fugacity Formulation
- **ID**: `SCI-FLAW-18`
- **Severity**: **CRITICAL**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:206, 257-270`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L206)
- **Model**: Peng-Robinson (1978) Mixture Phase Equilibrium
- **Observation**: The analytical expression for component fugacity coefficient $\ln \phi_i$ omits the critical $2\sqrt{2}B$ denominator in the logarithmic term, and omits the cross-interaction partial derivative summation $\sum_j x_j a_{ij}$.
- **Expected Equation** (Standard Peng-Robinson):
  $$\ln\phi_i = \frac{B_i}{B}(Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B}\left[\frac{2\sum_j z_j A_{ij}}{A} - \frac{B_i}{B}\right]\ln\left(\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}\right)$$
- **Actual Behavior in Code**:
  Omits the $2\sqrt{2}B$ scaling and substitutes $A/B$, distorting the chemical potential balance by $\approx 2.83\times$.
- **Scientific Consequence**: Equilibrium K-values ($K_i = y_i / x_i = \phi_i^L / \phi_i^V$) computed from this formulation do not satisfy Gibbs phase rule or thermodynamic equilibrium, corrupting compositional phase boundaries.
- **Affected Outputs**: `calculate_fugacity()`, `flash_calculation()`, `compositional_solver`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

