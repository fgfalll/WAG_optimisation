# CO₂ Thermodynamic & Physical Model

## 1. Physical State of Injected CO₂

In reservoir EOR operations, CO₂ is typically injected at pressures and temperatures well above its critical point:
- **Critical Temperature ($T_c$)**: $31.04^\circ\text{C}$ ($304.19\text{ K}$, $87.87^\circ\text{F}$)
- **Critical Pressure ($P_c$)**: $7.382\text{ MPa}$ ($73.82\text{ bar}$, $1,071\text{ psia}$)

Under typical reservoir conditions ($P > 1,500\text{ psi}$, $T > 100^\circ\text{F}$), CO₂ exists as a **supercritical dense-phase fluid**, exhibiting gas-like viscosity and liquid-like density.

---

## 2. Fundamental Constants & Conversions

Implemented in [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L1714-L1760):

| Property | Symbol | Value in Code | Units | Source / Reference |
| :--- | :---: | :---: | :---: | :--- |
| Molecular Weight | $M_{CO2}$ | 44.01 | g/mol | IUPAC / NIST |
| Critical Temperature | $T_c$ | 304.13 | K | NIST Reference |
| Critical Pressure | $P_c$ | $7.376 \times 10^6$ | Pa | NIST Reference |
| Acentric Factor | $\omega$ | 0.225 | dimensionless | Poling et al. |
| Standard Surface Density | $\rho_{sc}$ | 0.05254 (0.053 in formulas) | tonnes / MSCF | NIST standard conditions |

### Density Inconsistency Note:
- `PhysicalConstants.CO2_DENSITY_TONNE_PER_MSCF` defines `0.05254` tonnes/MSCF (based on $1.857\text{ kg/m}^3$ at $60^\circ\text{F}, 14.696\text{ psi}$).
- Throughout `analytical_models.py`, `surrogate_models.py`, `surrogate_engine.py`, and `EORParameters`, the constant `0.053` is hardcoded directly.
- Discrepancy: ~0.87% difference between centralized constants and hardcoded calculations.

---

## 3. CO₂ Dissolution & Swelling

When CO₂ contacts crude oil in the reservoir, it dissolves, causing two primary recovery mechanisms:
1. **Oil Swelling (Swelling Factor $SF$)**: Increases oil saturation, expelling oil from dead-end pores into flowing channels.
2. **Viscosity Reduction**: Drastically reduces crude oil viscosity, improving mobility ratio $M$.

In the surrogate engine, CO₂ solubility is parameterized via:
- `co2_solubility_scm_per_bbl`: Defaulted or populated from PVT tables.
- In `surrogate_engine.py:180-186`, oil viscosity reduction under dissolved CO₂ is modeled via Todd-Longstaff partial miscibility power-law mixing rather than an explicit swelling factor curve.

---

## 4. Minimum Miscibility Pressure (MMP) Formulations

MMP defines the pressure threshold at which multi-contact miscibility (MCM) develops, eliminating the interfacial tension between oil and CO₂.

All MMP correlations are centralized in [evaluation/mmp.py](file:///d:/rep/4.6/co2eor_optimizer/evaluation/mmp.py):

### A. Cronquist Correlation (1978) - Modified
$$P_{MMP} = 15.988 \cdot T^{0.744206} \cdot (55.0 - \gamma_{API})^{0.279033} \quad [\text{psia}]$$
- Note: Uses $(55 - \gamma_{API})$ to force an inverse relationship where lighter oil has lower MMP.
- Applicability: Pure CO₂ injection into light-to-medium oils ($T \in [70, 300]^\circ\text{F}$, $\gamma_{API} \in [20, 50]^\circ\text{API}$).

### B. Yellig & Metcalfe (1980)
$$P_{MMP} = 1016 + 4.773 \cdot T - 0.00946 \cdot T^2 + 2.1 \times 10^{-5} \cdot T^3 \quad [\text{psia}]$$
- Valid for pure CO₂ ($>98\%$). If $T < 95^\circ\text{F}$, MMP is capped at the bubble-point/critical pressure.

### C. Alston et al. (1985) - Impure Gas Streams
$$P_{MMP} = P_{MMP,\text{pure}} \cdot \left( \frac{T_{pc,\text{gas}}}{T_{pc,CO2}} \right)^A$$
Where:
- $T_{pc,\text{gas}} = \sum y_i T_{pc,i}$ (Kay's pseudo-critical rule with $T_{pc,CO2}=304.1\text{K}, T_{pc,CH4}=190.6\text{K}, T_{pc,N2}=126.2\text{K}$).
- Exponent $A = 2.41 - 0.00284 \cdot M_{C7+}$.

### D. Yuan et al. (2005)
$$P_{MMP} = 145.038 \cdot a \cdot b^{x_{CO2}} \cdot c \quad [\text{psia}]$$
- Analytical correlation based on multi-component gas flooding theory, explicitly adjusting for CH₄ contamination.
