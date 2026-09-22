# CO₂ Storage & Material Balance Accounting

## 1. Overview and Trapping Mechanisms

In Carbon Capture, Utilization, and Storage (CCUS) coupled with EOR, injected CO₂ is permanently trapped in the subsurface via four distinct physical and geochemical mechanisms:

1. **Structural / Stratigraphic Trapping**: Mobile CO₂ trapped beneath impermeable caprock or sealing faults.
2. **Residual / Capillary Trapping**: Disconnected CO₂ ganglia immobilized in pore throats by capillary forces.
3. **Solubility / Dissolution Trapping**: CO₂ dissolved into formation brine and remaining unproduced crude oil.
4. **Mineral Trapping**: Geochemical precipitation where dissolved CO₂ reacts with silicate/carbonate minerals to form solid carbonates (e.g., calcite, siderite).

---

## 2. Mathematical Breakdown of Trapping Efficiencies

Implemented in [core/engine_surrogate/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1323-L1360):

$$\eta_{\text{total}} = \eta_{\text{structural}} + \eta_{\text{residual}} + \eta_{\text{solubility}} + \eta_{\text{mineral}}$$

### Empirical Formulations in Code
```python
# Residual Trapping (from Corey critical gas saturation S_gc)
eta_residual = S_gc / (1.0 - S_wi) * (1.0 - V_DP * 0.3)

# Structural Trapping
eta_structural = (1.0 - eta_residual) * (1.0 - leakage_rate_fraction) * 0.4

# Solubility Trapping (proportional to pressure and water saturation)
eta_solubility = min(0.25, 0.05 * (pressure / 1000.0) * S_wi)

# Mineral Trapping (long-term asymptotic kinetic rate)
eta_mineral = min(0.10, 0.002 * project_lifetime_years)
```

---

## 3. Dynamic Mass Balance Accounting

In `surrogate_engine.py`, mass balance tracks CO₂ volumes across the project lifecycle:
$$\text{CO}_{2,\text{injected}}(t) = \text{CO}_{2,\text{purchased}}(t) + \text{CO}_{2,\text{recycled}}(t)$$

### Breakthrough-Aware Recycling
- **Pre-Breakthrough ($t < t_{bt}$)**:
  $$\text{CO}_{2,\text{purchased}}(t) = \text{CO}_{2,\text{injected}}(t), \quad \text{CO}_{2,\text{recycled}}(t) = 0$$
- **Post-Breakthrough ($t \ge t_{bt}$)**:
  Produced gas contains breakthrough CO₂. An operator captures and reinjects a fraction $\eta_{\text{recycle}}$:
  $$\text{CO}_{2,\text{recycled}}(t) = \eta_{\text{recycle}} \cdot \text{CO}_{2,\text{produced}}(t)$$
  $$\text{CO}_{2,\text{purchased}}(t) = \max\left( 0.0, \, \text{CO}_{2,\text{injected}}(t) - \text{CO}_{2,\text{recycled}}(t) \right)$$
- Implementation: [core/engine_surrogate/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L1267-L1320).

---

## 4. Material Balance Implementation in `analysis/material_balance.py`

### Correct Mass Conservation Formulation & Stream Typing
In [analysis/material_balance.py](file:///d:/rep/4.6/co2eor_optimizer/analysis/material_balance.py), mass conservation uses the OSTI-1204577 formulation:
$$\text{CO}_{2,\text{stored}} = \text{CO}_{2,\text{purchased}} - \text{CO}_{2,\text{uncaptured produced}} - \text{CO}_{2,\text{leakage}}$$

1. **Purchased CO₂ as Net Fresh Injection**:
   Only fresh purchased CO₂ enters the system boundary ($I_{\text{purchased}}$). Recycled CO₂ returns to the reservoir and cancels out in net storage accounting.
2. **Produced Stream Identity (Elimination of 20× Shrinkage Bug)**:
   When `profiles["yearly_co2_produced_mscf"]` or `annual_co2_produced_mscf` is passed, it represents the **pure separated $\text{CO}_2$ stream**. It must NOT be multiplied by `co2_fraction_of_produced` (which was intended for raw total solution gas, causing a 95% / 20× undercounting error). Pure $\text{CO}_2$ converts directly via density:
   $$M_{\text{produced,CO2}} = V_{\text{produced,CO2 (MSCF)}} \times \rho_{\text{CO2,surface}}$$
3. **Breakthrough GOR Unit Conversion**:
   In `calculate_breakthrough_aware_recycling()`, GOR in SCF/STB is converted to MSCF by dividing by 1,000 before computing gas mass via `co2_density_tonne_per_mscf`.
4. **Physical Recycling Cap**:
   Recycled CO₂ mass cannot physically exceed produced CO₂ mass:
   $$\text{CO}_{2,\text{recycled}} = \min(\text{CO}_{2,\text{recycled,raw}}, \, \text{CO}_{2,\text{produced}})$$
   $$\text{CO}_{2,\text{uncaptured}} = \max(0.0, \, \text{CO}_{2,\text{produced}} - \text{CO}_{2,\text{recycled}})$$
5. **Time Vector Scaling**:
   When monthly resolution profiles (e.g. 181 steps for 15 years) are supplied to `create_material_balance_from_optimization()`, time is scaled to fractional years ($\text{months} / 12.0$) rather than treating each monthly point as a full year.

---

## 5. Geomechanical Containment, Mohr-Coulomb Fault Slip & Caprock Integrity (`geomechanics_fault.py`)

Subsurface containment risk is evaluated using coupled geomechanics rather than heuristic loss factors:

### A. In-Situ Stress Path Formulation
As pore pressure $P_p$ increases due to injection, the total minimum horizontal stress $\sigma_h$ increases elastically via the reservoir stress path parameter $\gamma_h$:
$$\Delta \sigma_h = \gamma_h \Delta P_p = \frac{1 - 2\nu}{1 - \nu} \alpha_{\text{Biot}} \Delta P_p$$
Where Poisson's ratio $\nu \approx 0.25$ and Biot coefficient $\alpha_{\text{Biot}} \approx 1.0$ yield $\gamma_h \approx 0.67$.
Effective normal stress on a plane oriented at angle $\theta$ to horizontal is:
$$\sigma_n' = \sigma_n(P_p) - \alpha_{\text{Biot}} P_p$$

### B. Mohr-Coulomb Fault Reactivation & Slip Tendency
For a pre-existing critically oriented fault dipping at angle $\theta$:
1. **Normal and Shear Stresses**:
   $$\sigma_n = \sigma_v \cos^2\theta + \sigma_h \sin^2\theta$$
   $$\tau = (\sigma_v - \sigma_h) \sin\theta \cos\theta$$
2. **Effective Normal Stress**:
   $$\sigma_n' = \max(1.0, \, \sigma_n - \alpha_{\text{Biot}} P_p)$$
3. **Mohr-Coulomb Slip Tendency ($T_s$)**:
   $$T_s = \frac{\tau}{\sigma_n'}$$
   A fault is stable when $T_s \le \mu_f$ (where fault friction $\mu_f \approx 0.60$). If $T_s > \mu_f$, shear slip reactivation occurs.
4. **Critical Reactivation Pressure**:
   $$P_{\text{crit,fault}} = \frac{\sigma_n - (\tau / \mu_f)}{\alpha_{\text{Biot}}}$$

### C. Caprock Tensile & Shear Failure Margins
Caprock seal integrity is evaluated at the bottomhole sandface injection pressure $P_{\text{sandface}} = P_{\text{res}} + \Delta P_{\text{skin}}$:
1. **Tensile Margin**:
   $$M_{\text{tensile}} = \sigma_{h,\text{caprock}} - P_{\text{sandface}} + T_0$$
   Where $T_0$ is the tensile strength of the caprock shale (psi).
2. **Shear Margin**:
   $$M_{\text{shear}} = (S_0 + \mu_f \sigma_{n,\text{caprock}}') - \tau_{\text{caprock}}$$
Failure occurs if either safety margin becomes negative ($M_{\text{tensile}} < 0$ or $M_{\text{shear}} < 0$).

### D. Dynamic Subsurface Leakage Flux
When geomechanical thresholds are breached, dynamic CO₂ leakage occurs:
$$q_{\text{leak,caprock}} = C_{\text{leak,caprock}} \cdot \max(0, \, P_{\text{sandface}} - P_{\text{frac,caprock}}) \quad [\text{tonnes/day}]$$
$$q_{\text{leak,fault}} = C_{\text{leak,fault}} \cdot \max(0, \, T_s - \mu_f) \quad [\text{tonnes/day}]$$
$$M_{\text{leakage,annual}} = \sum_{t} (q_{\text{leak,caprock}} + q_{\text{leak,fault}}) \cdot \Delta t$$

### E. Surface Facility Recycle Compressor Bottleneck
Surface capture and reinjection are governed by real plant constraints:
1. **Compressor Capacity Limit**:
   $$q_{\text{recycle}}(t) = \min\left( \eta_{\text{recycle}} \cdot q_{\text{produced,CO2}}(t), \, Q_{\text{recycle,max}} \cdot A_{\text{facility}} \right)$$
   Where $Q_{\text{recycle,max}}$ is the rated compressor throughput (MSCFD) and $A_{\text{facility}} \approx 0.95$ is operational uptime availability.
2. **Closed-Loop Balance**:
   $$q_{\text{purchased}}(t) = \max\left( 0.0, \, q_{\text{injected}}(t) - q_{\text{recycle}}(t) \right)$$
   Excess produced $\text{CO}_2$ above compressor capacity cannot be reinjected into the reservoir.

### F. EPA Class VI UIC 90% Formation Fracture Limit
Under 40 CFR § 146.88, sandface injection pressure is strictly capped at $0.90 \times P_{\text{fracture}}$. If reservoir pressure reaches this ceiling, injection is throttled to zero (Class VI UIC shut-in). Overpressure violations incur quadratic economic penalties ($10^6 \cdot (\Delta P / P_{\text{limit}})^2$).


---

## 6. Strict Numerical Accounting & Elimination of Class E Storage Modifiers [RESOLVED]

To protect the physical and scientific integrity of the optimization process, all artificial storage synthesis mechanisms have been eradicated:

1. **Eradication of Class E Artificial Modifier**:
   - **Previous Defect**: In [core/objectives/wrapper.py](file:///d:/rep/4.6/co2eor_optimizer/core/objectives/wrapper.py#L130-L137), when simulation profiles or storage parameters were missing, the wrapper synthesized an artificial storage efficiency via:
     $$\eta_{\text{default}} = \max\left(0.3, \, 0.5 \times \frac{RF}{0.35}\right)$$
     This awarded ~50% storage credit to primary depletion runs without verifying any physical CO₂ retention.
   - **Resolution**: This fallback was completely removed. Missing profiles or storage metrics now set `storage_efficiency = float("nan")` and flag `method="unphysical_or_missing_data"`.
2. **Zero-Injection Accounting**:
   - In [core/optimisation_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1708-L1729), when total injected CO₂ is $\le 0$, true storage efficiency is strictly evaluated as $0.0$. If the scenario fails primary depletion requirements ($RF \le 0.05$), the candidate is immediately pruned with `FAILURE_PENALTY` ($-10^{12}$).
3. **Full Mathematical Penalty Enforcement**:
   - Penalty dilution multipliers (`* 0.1` and `* 0.8`) have been eliminated. Any unphysical chromosome or missing storage dataset is assigned full `FAILURE_PENALTY` ($-10^{12}$) so the genetic algorithm naturally kills off unviable genetic lines.

