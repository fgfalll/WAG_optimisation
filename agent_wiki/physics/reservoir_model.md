# Reservoir Model & Petrophysics

## 1. Overview and Purpose

The reservoir model defines the geological container, petrophysical rock properties, fluid in-place volumes, and vertical/areal heterogeneity that govern fluid flow, displacement efficiency, and pressure propagation during CO₂ injection.

---

## 2. Core Dataclass and State Attributes

The primary data structure is `ReservoirData` in [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L224-L350).

| Attribute | Type | Units | Typical Range | Physical Description | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `initial_pressure` | `float` | psia | 1,000 – 6,000 | Initial reservoir pore pressure | Field gauge / User input |
| `temperature` | `float` | °F | 80 – 300 | Reservoir isothermal temperature | Well log / User input |
| `ooip_stb` | `float` | STB | $10^5 - 10^9$ | Original Oil In Place | Volumetric calculation |
| `average_porosity` | `float` | fraction (0–1) | 0.05 – 0.35 | Bulk average porosity $\phi$ | Core / Petrophysical log |
| `permeability` | `float` | mD | 0.1 – 2,000 | Absolute permeability $k_h$ | Well test / Core analysis |
| `kv_kh_ratio` | `float` | fraction | 0.01 – 1.0 | Vertical to horizontal permeability ratio | Core analysis / default |
| `connate_water_saturation` (`s_wi`) | `float` | fraction (0–1) | 0.10 – 0.40 | Irreducible water saturation $S_{wc}$ | Core analysis / default |
| `rock_compressibility` | `float` | $\text{psi}^{-1}$ | $3\times 10^{-6} - 10^{-5}$ | Pore volume compressibility $c_f$ | Hall correlation / default |
| `thickness_ft` | `float` | ft | 10 – 500 | Net pay thickness $h$ | Log interpretation |
| `area_acres` | `float` | acres | 40 – 5,000 | Reservoir drainage area $A$ | Seismic / Well spacing |

---

## 3. Physical Formulations & Equations

### A. Pore Volume Calculation
Pore volume $V_p$ (in reservoir barrels, RB) is derived from volumetric dimensions or back-calculated from OOIP:
$$V_p = \frac{\text{OOIP} \cdot B_{oi}}{1 - S_{wi}}$$
- Implementation: [core/engine_surrogate/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L897).
- Assumptions: Constant initial oil formation volume factor $B_{oi}$ (defaulted to 1.2 if missing).

### B. Heterogeneity: Dykstra-Parsons Coefficient ($V_{DP}$)
Vertical permeability variation across reservoir layers is represented by the Dykstra-Parsons coefficient:
$$V_{DP} = 1 - \exp(-\sigma_{\ln k})$$
Where:
- $\sigma_{\ln k} = \text{std}(\ln k)$ across all permeability layers.
- Implementation: [core/simulation/recovery_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L588-L591).
- Range: $V_{DP} \in [0.0, 0.999]$ ($0 = \text{homogeneous}$, $>0.7 = \text{severely layered/channelized}$).

### C. Koval Heterogeneity Factor ($H$)
Koval (1963) maps the Dykstra-Parsons coefficient into an effective heterogeneity factor:
$$H = \frac{1}{(1 - V_{DP} \cdot C_{trans})^2}$$
Where $C_{trans}$ is the `transverse_mixing_calibration` empirical factor:
- Implementation: [core/engine_surrogate/surrogate_engine.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L193).
- Origin: Koval (1963) SPE-450-PA, modified with transverse mixing calibration.

---

## 4. Known Simplifications & Audit Flags

1. **Zero-Dimensional (Tank) Model**:
   In `engine_surrogate`, the entire 3D reservoir is modeled as an isothermal 0D tank with lumped pore volume $V_p$. Spatial pressure gradients between injector and producer are approximated via steady-state Darcy drawdown formulas rather than spatial PDE discretization.
2. **Porosity-Permeability Leverett Scaling**:
   In `PhDHybridSurrogate` (lines 720–735), porosity $\phi$ and permeability $k$ modify Corey exponents and endpoints via empirical power laws:
   $$n_{adj} = n \cdot (1.0 + 0.03 \cdot \ln(k / 100))$$
   $$k_{end} = k_0 \cdot (\phi / 0.15)^{0.15}$$
   *Scientific Origin*: Empirical PhD heuristic; not an SPE standard correlation.
