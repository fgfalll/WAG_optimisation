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

### A. Cronquist Correlation (1978) - Published & Validated
$$P_{MMP} = 15.988 \cdot T^{Y} \quad [\text{psia}]$$
Where:
- $Y = 0.744206 + 0.0011038 \cdot MW_{C5+} + 0.0015279 \cdot Vol$
- $MW_{C5+} = 4247.98641 \cdot \gamma_{API}^{-0.87022}$ (DOE / CO₂ Prophet standard formulation) or $\max(72.0, M_{C7+} - 20.0)$ if $C_{7+}$ molecular weight is known.
- $Vol$: mole percent of volatiles ($C_1 + N_2$) in the oil phase.
- SCI-FLAW-13 eliminated: No negative power singularities at $\gamma_{API} \ge 55^\circ\text{API}$. Strictly positive, finite, monotonically decreasing with API gravity.

### B. Yellig & Metcalfe (1980)
$$P_{MMP} = 1833.7217 + 2.2518055 \cdot T + 0.01800674 \cdot T^2 - \frac{103949.93}{T} \quad [\text{psia}]$$
- Valid for pure CO₂ ($>98\%$). If $T < 95^\circ\text{F}$, MMP is capped at the CO₂ bubble-point/critical pressure ($1070$ psia).

### C. Alston et al. (1985) - Impure Gas Streams
$$P_{MMP} = P_{MMP,\text{pure}} \cdot \left( \frac{T_{pc,CO2}}{T_{pc,\text{gas}}} \right)^A$$
Where:
- $T_{pc,\text{gas}} = \sum y_i T_{pc,i}$ (Kay's pseudo-critical rule with $T_{pc,CO2}=304.1\text{K}, T_{pc,CH4}=190.6\text{K}, T_{pc,N2}=126.2\text{K}, T_{pc,H2S}=373.2\text{K}$).
- Exponent $A = \max(0.2, 2.41 - 0.00284 \cdot M_{C7+})$.
- Impurity ratio ensures adding volatile contaminants ($\text{CH}_4, \text{N}_2$) properly increases MMP.

### D. Yuan et al. (2005)
$$P_{MMP} = 145.038 \cdot a \cdot b^{x_{CO2}} \cdot c \quad [\text{psia}]$$
- Analytical correlation based on multi-component gas flooding theory.
- Impurity term $c = 1.0 + 1.25 \cdot (1 - x_{CO2})^{0.8}$ strictly enforces $MMP_{\text{impure}} \ge MMP_{\text{pure}}$.
