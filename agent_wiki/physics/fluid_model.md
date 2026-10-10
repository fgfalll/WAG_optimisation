# Fluid Model & Phase Behavior

## 1. Overview and Purpose

The fluid model defines the thermodynamic and physical properties of reservoir fluids: crude oil, formation brine, associated hydrocarbon gas, and pure/impure injected CO₂.

---

## 2. Core Dataclass and Attributes

Fluid properties are parameterized in `PVTProperties` and `FluidProperties` in [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L482-L570, L1104-L1158).

| Parameter | Symbol | Units | Typical Range | Physical Description |
| :--- | :---: | :---: | :---: | :--- |
| `oil_api` | $\gamma_{API}$ | °API | 15 – 50 | Stock tank oil gravity |
| `viscosity_oil` | $\mu_o$ | cP | 0.5 – 50.0 | In-situ live oil viscosity at reservoir $P, T$ |
| `co2_viscosity` | $\mu_{CO2}$ | cP | 0.02 – 0.08 | Supercritical CO₂ viscosity at reservoir $P, T$ |
| `water_viscosity`| $\mu_w$ | cP | 0.3 – 1.0 | Formation brine viscosity |
| `oil_fvf` | $B_o$ | RB/STB | 1.05 – 1.80 | Oil formation volume factor |
| `gas_fvf` | $B_g$ | RB/SCF | 0.0005 – 0.005 | Gas formation volume factor |
| `mscf_per_res_bbl` | $1/B_g$ | MSCF/RB | 0.15 – 0.45 | Reciprocal gas FVF |
| `c7_plus_fraction` | $z_{C7+}$ | fraction (0–1)| 0.10 – 0.60 | Heavy fraction mole/mass fraction |
| `c7_plus_mw` | $M_{C7+}$ | g/mol | 120 – 350 | Molecular weight of heavy C7+ fraction |

---

## 3. Thermodynamic Formulations & Correlations

### A. Solvent-Extended 4-Component Representation (`pvt_state.py`)
In enhanced oil recovery, the system comprises 4 effective pseudo-components:
1. **Stock Tank Oil** ($o$): Resident liquid hydrocarbons.
2. **Solution Hydrocarbon Gas** ($g$): Native methane/ethane associated gas dissolved in oil.
3. **Formation / Injected Water** ($w$): Brine phase.
4. **Solvent Injection Agent** ($s$): Pure or enriched supercritical $\text{CO}_2$.

**Thermodynamic Invariant: Decoupling Transport ($S_g$) from Composition ($x_{\text{CO2}}$)**:
Gas saturation $S_g$ is a hydrodynamic transport variable governed by relative permeabilities, capillary pressure, and multiphase Darcy flow. It must **never** be used as a proxy for dissolved solvent concentration. The thermodynamic state is indexed strictly by:
- $x_{\text{CO2}}$: Dissolved $\text{CO}_2$ concentration in the oleic phase (liquid mole/mass fraction).
- $y_{\text{CO2}}$: $\text{CO}_2$ concentration in the vapor phase (mole fraction).

### B. Pure Supercritical $\text{CO}_2$ Density & FVF via Peng-Robinson EOS
Supercritical $\text{CO}_2$ density is computed directly from the Peng-Robinson (1976) cubic equation of state:
$$Z^3 - (1 - B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0$$
Where pure $\text{CO}_2$ critical constants are $T_c = 304.13\text{ K}$, $P_c = 7.3773\text{ MPa}$ ($1,070\text{ psia}$), $\omega = 0.2239$, and molar mass $M = 44.01\text{ g/mol}$.
- Downhole density $\rho_{\text{CO2}}(P, T)$ ranges from $400$ to $950\text{ kg/m}^3$ under typical reservoir conditions ($1,500-5,000\text{ psia}$, $100-250^\circ\text{F}$).
- Downhole gas formation volume factor ($B_{\text{CO2}}$ in RB/MSCF):
  $$B_{\text{CO2}}(P, T) = \frac{\rho_{\text{sc}}}{\rho_{\text{res}}} \times \frac{5.615 \text{ ft}^3/\text{bbl}}{1000 \text{ scf}/\text{mscf}} \approx \frac{327.36}{\rho_{\text{res}} (\text{kg/m}^3)} \quad [\text{RB/MSCF}]$$
- Invariant: Supercritical $\text{CO}_2$ is highly compressible near its critical point ($1,070-1,500\text{ psia}$); using an ideal gas law ($B_g \propto 1/P$) severely miscalculates downhole voidage and displacement fronts.

### C. Oil Swelling and Viscosity Thinning
Dissolution of supercritical $\text{CO}_2$ into crude oil induces volumetric swelling and sharp viscosity reduction:
1. **Oil Swelling Factor ($S_F$)**:
   $$S_F(P, x_{\text{CO2}}) = 1.0 + c_s \cdot x_{\text{CO2}} \cdot \min\left(1.0, \, \frac{P}{P_{MMP}}\right)$$
   Where $c_s \approx 0.35$ is the empirical swelling coefficient.
2. **Swelled Oil Formation Volume Factor ($B_o$)**:
   $$B_o(P, x_{\text{CO2}}) = B_{o,\text{base}} \cdot S_F(P, x_{\text{CO2}}) \cdot \left[ 1 - c_o (P - P_{\text{bubble}}) \right]$$
3. **Viscosity Thinning ($\mu_o$)**:
   $$\ln \mu_o(P, x_{\text{CO2}}) = (1 - x_{\text{CO2}}) \ln \mu_{o,\text{base}} + x_{\text{CO2}} \ln \mu_{\text{CO2}}$$
   $$\mu_o(P, x_{\text{CO2}}) = \exp\left[ (1 - x_{\text{CO2}}) \ln \mu_{o,\text{base}} + x_{\text{CO2}} \ln \mu_{\text{CO2}} \right]$$

### D. Todd-Longstaff Partial Miscibility Viscosity Mixing Rule
In the dispersed displacement zone, effective phase viscosities for Koval flow are determined from Todd & Longstaff (1972) using solvent concentration $x_{\text{CO2}}$:
$$\mu_m = \left[ x_{\text{CO2}} \cdot \mu_{\text{CO2}}^{-1/4} + (1 - x_{\text{CO2}}) \cdot \mu_o^{-1/4} \right]^{-4}$$
The effective viscosities of oil and gas phases are calculated using the mixing parameter $\omega \in [0, 1]$:
$$\mu_{oe} = \mu_m^\omega \cdot \mu_o^{1 - \omega}$$
$$\mu_{ge} = \mu_m^\omega \cdot \mu_{\text{CO2}}^{1 - \omega}$$
- Implementation: [core/engine_surrogate/pvt_state.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/pvt_state.py) and [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py).

### E. Surface Multi-Stage Flash Separation
Produced wellstream fluid is separated at surface conditions ($P_{\text{sep}}, T_{\text{sep}}$):
1. **Oil Shrinkage**: Downhole live oil shrinks to stock tank oil (STB) via $B_o(P, x_{\text{CO2}})$.
2. **Solution Hydrocarbon Gas Degassing**: Native dissolved methane/ethane gas is liberated at GOR $R_{s,\text{base}}$ (MSCF/STB).
3. **Dissolved $\text{CO}_2$ Flashing**: Dissolved $\text{CO}_2$ vaporizes completely at surface conditions, contributing:
   $$V_{\text{CO2,degassed}} = \frac{x_{\text{CO2}} \cdot \rho_o}{\rho_{\text{CO2,sc}} \cdot (1 - x_{\text{CO2}})} \quad [\text{MSCF/STB}]$$
   This guarantees that produced $\text{CO}_2$ includes both breakthrough free gas and degassed dissolved solvent.

