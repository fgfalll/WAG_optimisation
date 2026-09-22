# Scientific Unit System & Dimensional Audit

This document presents the rigorous dimensional and unit-system audit of the entire CO₂ Enhanced Oil Recovery Optimizer codebase, conducted in accordance with petroleum engineering, fluid mechanics, and reservoir simulation standards.

---

## 1. System-Wide Unit Convention Matrix

The repository contains three competing unit systems used across different modules, often without explicit conversion boundaries:

| Subsystem / Module | Primary Unit System | Pressure | Permeability | Viscosity | Liquid Rate / Vol | Gas Rate / Vol | Density |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Active Surrogate Engine** (`core/engine_surrogate/`) | Oilfield (Mixed) | $\text{psia}$ | $\text{mD}$ | $\text{cP}$ | $\text{STB/d}$, $\text{RB/d}$ | $\text{MSCFD}$, $\text{MSCF}$ | $\text{lb/ft}^3$, $\text{tonne/MSCF}$ |
| **Unified Physics & EOS** (`core/unified_engine/`) | SI (Metric) | $\text{Pa}$, $\text{MPa}$ | $\text{m}^2$, $\text{mD}$ | $\text{Pa}\cdot\text{s}$, $\text{cP}$ | $\text{m}^3/\text{s}$, $\text{m}^3/\text{d}$ | $\text{Sm}^3/\text{d}$, $\text{kg/s}$ | $\text{kg/m}^3$ |
| **Simple Grid Engine** (`core/engine_simple/`) | SI (Metric) | $\text{Pa}$ | $\text{m}^2$ | $\text{Pa}\cdot\text{s}$ | $\text{m}^3/\text{s}$ | $\text{m}^3/\text{s}$ | $\text{kg/m}^3$ |
| **Material Balance** (`analysis/material_balance.py`) | Mixed (Field/Metric) | $\text{psia}$ | N/A | N/A | $\text{STB}$ | $\text{MSCF}$ | $\text{metric tonne/MSCF}$ |
| **Breakthrough Physics** (`analysis/breakthrough_physics.py`) | Oilfield Hybrid | $\text{psia}$ | $\text{mD}$, $\text{ft}^2$ | $\text{cP}$, $\text{lb}_f\cdot\text{day/ft}^2$ | $\text{ft}^3/\text{day}$ | $\text{MSCFD}$ | $\text{lb}_m/\text{ft}^3$ |
| **Data Integration** (`core/data_integration_engine.py`) | Oilfield / Custom | $\text{psia}$ | $\text{mD}$ | $\text{cP}$ | $\text{STB}$ | $\text{MSCF}$ | $\text{API}$, SG |

---

## 2. Dimensional Consistency Checks ($[LHS] = [RHS]$)

### Equation 1: 0D Tank Pressure Material Balance Derivative
- **Location**: [`core/engine_surrogate/surrogate_engine.py:376`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L376)
- **Code Expression**:
  $$dP = \frac{q_{\text{net,IPR}} \cdot \Delta t}{V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t}$$
- **Dimensional Verification**:
  - $q_{\text{net,IPR}} = q_{\text{inj,RB/d}} - q_{\text{prod,RB/d}} \implies [\text{RB/day}] = [\text{bbl/T}]$
  - $\Delta t \implies [\text{day}] = [T]$
  - Numerator: $[\text{bbl/T}] \times [T] = [\text{bbl}]$ (Reservoir volume of voidage deficit)
  - Denominator term 1: $V_p [\text{RB}] \times c_t [\text{psi}^{-1}] = [\text{bbl} / \text{psi}]$
  - Denominator term 2: $J_{\text{eff}} [\text{RB/day/psi}] \times \Delta t [\text{day}] = [\text{bbl} / \text{psi}]$
  - Division: $[\text{bbl}] / [\text{bbl} / \text{psi}] = [\text{psi}]$
- **Dimensional Status**: **CONSISTENT** ($[LHS] = \text{psi} = [RHS]$).

---

### Equation 2: Dimensionless Gravity Number ($N_g$) in `analytical_models.py`
- **Location**: [`core/engine_surrogate/analytical_models.py:860`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L860)
- **Code Expression**:
  $$N_g = \frac{k_{\text{mD}} \cdot \Delta\rho \cdot \sin(\theta) \cdot 4.3948 \times 10^{-5}}{\mu_{\text{inj}} \cdot u_{\text{ft/day}}}$$
- **Dimensional Verification**:
  - Definition: $N_g = \frac{k \Delta\rho (g/g_c) \sin(\theta)}{\mu u}$
  - $k [\text{mD}] \times 1.0623 \times 10^{-14} \implies [\text{ft}^2]$
  - $\Delta\rho [\text{lb}_m/\text{ft}^3] \times (g/g_c) [1.0\text{ lb}_f/\text{lb}_m] \implies [\text{lb}_f/\text{ft}^3]$
  - $\mu [\text{cP}] \times 2.4172 \times 10^{-10} \implies [\text{lb}_f\cdot\text{day/ft}^2]$
  - $u \implies [\text{ft/day}]$
  - Numerator/Denominator conversion constant: $\frac{1.0623 \times 10^{-14}}{2.4172 \times 10^{-10}} = 4.3948 \times 10^{-5}$
  - Dimensions: $\frac{[\text{ft}^2] \cdot [\text{lb}_f/\text{ft}^3]}{[\text{lb}_f\cdot\text{day/ft}^2] \cdot [\text{ft/day}]} = \frac{[\text{lb}_f/\text{ft}]}{[\text{lb}_f/\text{ft}]} = [-]$
- **Dimensional Status**: **CONSISTENT** (Corrected from previous $2.4 \times 10^{11}$ defect).

---

### Equation 3: Gravity Number ($N_g$) & Viscous-Gravity Ratio ($R_{v/g}$) in `breakthrough_physics.py`
- **Location**: [`analysis/breakthrough_physics.py:318, 364`](file:///d:/rep/4.6/co2eor_optimizer/analysis/breakthrough_physics.py#L318)
- **Code Expression**:
  $$R_{v/g} = \frac{u_{\text{ft/day}} \cdot \mu_{o,\text{lb-day/ft}^2} \cdot L}{k_{\text{ft}^2} \cdot \Delta\rho_{\text{lb/ft}^3} \cdot g_{\text{ft/day}^2} \cdot h}$$
  $$N_g = \frac{k_{\text{ft}^2} \cdot \Delta\rho_{\text{lb/ft}^3} \cdot g_{\text{ft/day}^2} \cdot |\sin(\theta)|}{\mu_{g,\text{lb-day/ft}^2} \cdot u_{\text{ft/day}}}$$
  where $g_{\text{ft/day}^2} = 2.4 \times 10^{11}$.
- **Dimensional Verification**:
  - Numerator has $k [\text{ft}^2] \cdot \Delta\rho [\text{lb}_m/\text{ft}^3] \cdot g [\text{ft/day}^2] = [\text{lb}_m / \text{day}^2]$.
  - Denominator has $\mu [\text{lb}_f\cdot\text{day/ft}^2] \cdot u [\text{ft/day}] = [\text{lb}_f / \text{ft}]$.
  - Ratio:
    $$\frac{[\text{Numerator}]}{[\text{Denominator}]} = \frac{\text{lb}_m / \text{day}^2}{\text{lb}_f / \text{ft}} = \frac{\text{lb}_m\cdot\text{ft}}{\text{lb}_f\cdot\text{day}^2} = [g_c]$$
  - The ratio is NOT dimensionless; it has the dimensional units of Newton's constant $g_c = 2.40 \times 10^{11}\text{ lb}_m\cdot\text{ft}/(\text{lb}_f\cdot\text{day}^2)$.
  - Because the author multiplied by $g = 2.4 \times 10^{11}\text{ ft/day}^2$ without dividing by $g_c = 2.4 \times 10^{11}$, the value is numerically inflated by $2.4 \times 10^{11}\times$.
- **Dimensional Status**: **INCONSISTENT / CRITICAL DIMENSIONAL FLAW** ($[LHS] = [-]$, $[RHS] = [g_c] \approx 2.4 \times 10^{11}$).

---

### Equation 4: Interfacial Tension Capillary Number ($N_c$)
- **Location**: [`core/engine_surrogate/analytical_models.py:819`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L819)
- **Code Expression**:
  $$N_c = \frac{\mu_{\text{inj}}[\text{cP}] \cdot u[\text{ft/day}]}{\sigma[\text{mN/m}]} \times 3.5 \times 10^{-6}$$
- **Dimensional Verification**:
  - $1\text{ cP} = 10^{-3}\text{ Pa}\cdot\text{s} = 10^{-3}\text{ N}\cdot\text{s/m}^2$
  - $1\text{ ft/day} = \frac{0.3048}{86400}\text{ m/s} = 3.5278 \times 10^{-6}\text{ m/s}$
  - $1\text{ mN/m} = 10^{-3}\text{ N/m}$
  - Ratio in SI:
    $$\frac{(10^{-3}\text{ N}\cdot\text{s/m}^2) \times (3.5278 \times 10^{-6}\text{ m/s})}{10^{-3}\text{ N/m}} = 3.5278 \times 10^{-6}\ [-]$$
- **Dimensional Status**: **CONSISTENT** ($[LHS] = [-] = [RHS]$).

---

### Equation 5: Gas Formation Volume Factor ($B_g$) Unit Inversion & Multiplier Confusion
- **Locations**:
  - `core/engine_surrogate/surrogate_engine.py:200-208`
  - `core/optimisation_engine.py:98, 229, 4148`
  - `core/data_integration_engine.py:373, 462`
- **Code Expressions & Discrepancies**:
  1. `surrogate_engine.py:201`:
     `bg_rb_per_mscf = 1.0 / max(mscf_per_rb, 1e-6)` (where `mscf_per_rb = 2.0` $\implies B_g = 0.50\text{ RB/MSCF}$).
  2. `optimisation_engine.py:98`:
     `B_GAS_RB_PER_MSCF = 5.0` (A $10\times$ higher fallback assuming $0.005\text{ RB/SCF}$).
  3. `data_integration_engine.py:373`:
     `GAS_FVF = 0.005 * (4000 / pressure_points)` (produces $B_g = 0.005\text{ RB/SCF} = 5.0\text{ RB/MSCF}$).
- **Physical Reality**:
  - For supercritical CO₂ at reservoir conditions ($P = 2000 - 3500\text{ psia}$, $T = 120 - 180^\circ\text{F}$):
    $$\rho_{\text{CO2}} \approx 0.65 - 0.82\text{ g/cm}^3 = 40.5 - 51.2\text{ lb/ft}^3$$
  - At standard surface conditions ($60^\circ\text{F}$, $14.7\text{ psia}$): $\rho_{sc} \approx 0.1234\text{ lb/ft}^3$.
  - Downhole volume per SCF:
    $$B_g = \frac{\rho_{sc}}{\rho_{res}} = \frac{0.1234}{45.0} \times \frac{1\text{ bbl}}{5.61458\text{ ft}^3} \approx 0.000488\text{ RB/SCF} = 0.488\text{ RB/MSCF}$$
  - The ratio $1 / B_g$ is $\approx 2.05\text{ MSCF/RB}$.
  - Storing $B_g = 5.0\text{ RB/MSCF}$ represents low-pressure methane gas at $\sim 400\text{ psia}$, overestimating the in-situ downhole expansion of supercritical CO₂ by **$10\times$**.
- **Dimensional Status**: **DIMENSIONALLY VALID BUT PHYSICALLY DISCORDANT** ($10\times$ property spread across active vs legacy modules).

---

### Equation 6: CO₂ Utilization Metric Unit Discordance
- **Locations**:
  - `core/objectives/wrapper.py:212`
  - `ui/optimization_widget.py:1297`
  - `utils/report_generator.py:500`
- **Code Expression**:
  $$\text{Utilization} = \frac{\sum M_{\text{CO2,purchased}}[\text{tonnes}]}{\sum N_p[\text{STB}]}$$
- **Dimensional Verification**:
  - Computed unit: $\text{tonne / STB}$
  - GUI Display string: `f"{co2_util:.2f} MSCF/STB"` (Raw numerical value in tonnes/STB displayed with MSCF/STB label without applying conversion $1\text{ tonne} \approx 18.9\text{ MSCF}$).
  - HTML Report table: `<td>bbl/ton</td>` (Inverted unit label).
- **Consequence**:
  - Numerical value: $0.035\text{ tonne/STB}$.
  - Displayed in GUI as: $0.035\text{ MSCF/STB}$ (Physically absurd: typical EOR utilization is $6 - 10\text{ MSCF/STB}$).
  - True value in MSCF/STB: $0.035 \times 18.924 = 0.662\text{ MSCF/STB}$ (Still deflated due to primary oil denominator error).
- **Dimensional Status**: **CRITICAL UNIT CONVERSION & LABELING DEFECT**.

---

### Equation 7: PVT Negative Compressibility ($B_o$ Expansion)
- **Location**: [`core/data_integration_engine.py:370, 456`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L370)
- **Code Expression**:
  $$B_o(P) = 1.2 + 0.0001 \cdot (P - 4000)$$
- **Dimensional Verification**:
  - $[\text{LHS}] = [\text{RB/STB}] = [-]$
  - $[\text{RHS}] = 1.2 + [0.0001\text{ psi}^{-1}] \times [P - 4000\text{ psi}] = [-]$
  - Dimensionally homogeneous.
  - **Thermodynamic Direction**:
    $$\frac{d B_o}{d P} = +0.0001\text{ psi}^{-1} > 0 \implies c_o = -\frac{1}{B_o} \frac{d B_o}{d P} = -8.33 \times 10^{-5}\text{ psi}^{-1} < 0$$
- **Thermodynamic Status**: **PHYSICALLY IMPOSSIBLE (NEGATIVE COMPRESSIBILITY)**. Undersaturated liquid oil expands when pressurized.

---

## 3. Verified Conversion Constants

The following conversion constants from [`core/data_models.py:PhysicalConstants`](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L1735) have been verified against NIST and API standards:

| Constant | Code Value | Exact Scientific Value | Relative Error |
|:---|:---|:---|:---|
| $1\text{ psi} \to \text{Pa}$ | `6894.76` | $6894.757293\text{ Pa}$ | $< 0.0001\%$ |
| $1\text{ mD} \to \text{m}^2$ | `9.869233e-16` | $9.869233 \times 10^{-16}\text{ m}^2$ | $0.0\%$ |
| $1\text{ mD} \to \text{ft}^2$ | `1.0623e-14` | $1.062319 \times 10^{-14}\text{ ft}^2$ | $< 0.002\%$ |
| $1\text{ acre} \to \text{ft}^2$ | `43560.0` | $43,560.0\text{ ft}^2$ | $0.0\%$ |
| $1\text{ STB} \to \text{m}^3$ | `0.158987` | $0.158987295\text{ m}^3$ | $< 0.0002\%$ |
| $1\text{ MSCF} \to \text{Sm}^3$ | `28.3168` | $28.3168466\text{ Sm}^3$ | $< 0.0002\%$ |
| $\text{CO}_2$ Surface Density | `0.05254 t/MSCF` | $0.052538\text{ tonne/MSCF}$ | $< 0.004\%$ |
| Standard Gravity $g$ | `32.174 ft/s²` | $9.80665\text{ m/s}^2 \times 3.28084 = 32.17405\text{ ft/s}^2$ | $< 0.0002\%$ |

---

## 4. Priority Remediation List

1. **Remove $g = 2.4 \times 10^{11}$ from `breakthrough_physics.py`**: Divide by $g_c$ or eliminate $g$ where pressure gradient is already in $\text{lb}_f/\text{ft}^3$.
2. **Harmonize $B_g$ Across Modules**: Unify `B_GAS_RB_PER_MSCF` to $0.50\text{ RB/MSCF}$ ($0.0005\text{ RB/SCF}$) for supercritical CO₂ across `optimisation_engine.py` and `data_integration_engine.py`.
3. **Correct Negative Compressibility in Synthetic PVT Tables**: Invert the sign of $B_o(P)$ slope in `data_integration_engine.py`: $B_o(P) = 1.2 - 1.5\times 10^{-5} (P - 4000)$.
4. **Standardize CO₂ Utilization**: Convert metric strictly to MSCF/STB via $18.924\text{ MSCF/tonne}$ before displaying in GUI and reporting.
