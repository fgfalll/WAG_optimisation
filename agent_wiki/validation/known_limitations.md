# Known Limitations & Boundary Conditions

## 1. Physical Limitations of the Active Surrogate Engine

1. **Zero-Dimensional (Lumped Tank) Approximation**:
   - The surrogate engine does not solve 3D partial differential equations (PDEs) for fluid flow.
   - It cannot resolve localized pressure cones near wellbores, pressure sinks, or spatial saturation plumes.
   - Injectivity is approximated by steady-state 1D Darcy radial drawdown rather than dynamic pressure transient analysis.
2. **First-Contact Miscibility (FCM) Proxy**:
   - Multi-contact miscibility (vaporizing and condensing gas drives) is modeled using Koval heterogeneity factors and Todd-Longstaff mixing rather than multi-component compositional phase ties.
   - It cannot track compositional grading or gravity drainage in thick vertical columns.
3. **No 3D Gravity Segregation (Override)**:
   - Supercritical CO₂ is significantly less dense than reservoir brine and crude oil ($\rho_{CO2} \sim 0.5-0.7\text{ g/cm}^3$ vs $\rho_{oil} \sim 0.8-0.9\text{ g/cm}^3$ and $\rho_{water} \sim 1.0-1.1\text{ g/cm}^3$).
   - In 3D reservoirs, CO₂ naturally overrides the oil column and channels along the formation top. The surrogate engine approximates this using Craig's 2D areal sweep correlations and Johnson's vertical sweep without explicit vertical crossflow simulation.
4. **Isothermal Reservoir Assumption**:
   - Joule-Thomson cooling near the wellbore during high-pressure CO₂ expansion is neglected. Reservoir temperature is assumed constant.

---

## 2. Numerical Limitations

1. **Pressure ODE Stiff Solver Limits**:
   - The 0D tank material balance ODE is solved using `scipy.integrate.solve_ivp(method="BDF")`.
   - If cumulative production volume exceeds injection and initial compressible expansion, the pressure equation can attempt to cross below zero. The ODE system clips pressure to $\max(P, 100.0\text{ psi})$ to prevent negative values and singular gas compressibilities ($c_g = 1/P$).
2. **Fixed Time Resolution**:
   - Dynamic profiles are synthesized at discrete daily, monthly, or annual time points.
   - Rapid WAG cycles ($< 15\text{ days}$) evaluated with `"annual"` time resolution will suffer numerical aliasing, as multiple WAG cycles collapse into a single time point.

---

## 3. Valid Operating Envelope

| Parameter | Minimum Valid Bound | Maximum Valid Bound | Consequence of Exceeding |
| :--- | :---: | :---: | :--- |
| **Reservoir Pressure** | 500 psia | 10,000 psia | Below 500 psia, gas compressibility singularity; above 10,000 psia, caprock fracture risk. |
| **Reservoir Temperature**| 70 °F | 300 °F | Outside this range, Cronquist and Yellig-Metcalfe MMP empirical correlations lose validity. |
| **Oil API Gravity** | 20 °API | 50 °API | Heavy oils ($<20^\circ\text{API}$) violate miscibility assumptions; oils $\ge 55^\circ\text{API}$ crash Cronquist formula. |
| **Dykstra-Parsons ($V_{DP}$)**| 0.0 | 0.95 | $V_{DP} \ge 1.0$ causes division-by-zero in Koval formula $1/(1-V_{DP})^2$. |
| **Porosity** | 0.03 | 0.40 | Porosities $< 0.03$ cause singular pore volume calculations. |

---

## 4. Known Surrogate-Optimizer Coupling Pathologies (Post-Run Audit Findings)

1. **Uncoupled Parameter Pressure Optimization**:
   - The optimizer treats `pressure` as an unconstrained decision variable fed to `AnalyticalSurrogate.predict()`, maximizing analytical recovery factor at the search upper bound (e.g. $4,450\text{ psia}$).
   - However, the dynamic tank material balance ODE (`_solve_pressure_ode_stiff`) calculates in-situ reservoir pressure independently ($\sim 3,080 - 3,335\text{ psia}$), creating an unphysical $\sim 1,350\text{ psi}$ discrepancy between optimization assumptions and reservoir reality.
2. **Sub-PVI Throughput Displacement Overprediction**:
   - Analytical Koval displacement curves can predict $>40\%$ OOIP incremental tertiary recovery at cumulative throughputs $< 0.5\text{ HCPVI}$ ($0.27\text{ Net PVI}$) under continuous gas injection.
   - U.S. DOE/NETL and SPE field data demonstrate that continuous adverse-mobility gas floods ($M=50$) achieve at most $7\% - 15\%$ OOIP incremental recovery at such low throughputs due to severe viscous fingering and gravity override.
3. **Single-Well Field-Scale Drain Abstraction**:
   - When users specify no pattern injectors, the engine defaults to $N_{\text{prod}} = 1$ and $N_{\text{inj}} = 1$ for multi-thousand-acre fields (e.g., 1,354 acres, 48.5 MMbbl OOIP).
   - This routes the entire field recovery through a single wellbore ($7,270\text{ BOPD}$ for 6.4 years in a $100\text{ mD}$ formation) without accounting for multi-well interference, pattern geometry, or Darcy drawdown limits.
4. **Static CAPEX Economic Distortions**:
   - Fixed static CAPEX (\$5.0M) does not scale with well count or gas processing capacity ($Q_{\text{peak}}^{0.65}$).
   - On large fields requiring 20–34 pattern wells and a $10\text{ MMSCFD}$ recycling plant, true CAPEX is $\$60\text{M} \text{ to } \$120\text{M}+$. A flat \$5M CAPEX yields fictitious economics (11-day payback, $\$878.8\text{M}$ NPV).

