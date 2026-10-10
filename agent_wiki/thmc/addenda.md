# Addenda — D3 (Addendum 1) and D4 (Addendum 2)

Sources:
- **D3** — `3D_THMC_docs/Доповнення 1.md` (443 lines) — *Produced Water, Asphaltenes and Hydrates*
- **D4** — `3D_THMC_docs/Доповнення 2.md` (314 lines) — *Remediation Works, Formation-Damage Removal and Emergency Control*

**Design intent only.** Both DTO listings are verbatim, including the mangled `# [derive(...)]`.

---

## 1. What the addenda extend

D3 supplies the mathematical / thermodynamic / hydrodynamic foundation for **three filtration-capacity
property (ФЄВ) degradation mechanisms** plus the wellbore flow path:
1. **Produced-Water Re-Injection (PWRI)** — mechanical plugging by Total Suspended Solids, external
   filter-cake growth, emulsified oil droplet trapping (OiW) under Jamin capillary forces.
2. **Asphaltenes** — Asphaltene Onset Pressure (AOP) via three-parameter PR-EOS, convective-aggregative
   deposition via Verma–Pruess percolation, radial loss of tubing ID.
3. **Gas hydrates** — van der Waals–Platteeuw statistical thermodynamics coupled to a `feos-ad` flash
   solver, Bischoff–Englezos kinetics, Thomas rheology with sigmoidal regularisation for tubing and
   borehole plugging.

D4 modifies the classical global PDE/ODE system to handle **shock-like, discrete boundary-condition
changes** caused by acid treatments, solvent injection, well-kill during gas-oil-water blowouts (ГНВП),
and emergency SSSV actuation.

---

## 2. D3 — architecture integration

| Core component | D3 integration module | Mechanism |
|---|---|---|
| Data preparation (`petekIO`) | PWRI / Asphaltene / Hydrate Parsers | Deserialize extended DTOs; physical-bound checks via `validate()`; validation of PSD bins, gas composition, kinetic constants |
| Tubing hydraulics (Drift-Flux Solver) | Wellbore Clogging & Rheology Engine | Dynamic recomputation of `D(z,t)`, Darcy–Weisbach `f_D` with deposit roughness `ε_asph`, phase holdup and effective emulsion viscosity `μ_eff(φ_H)` in a 1D discretised momentum system |
| Thermodynamics (`feos-ad` Flash) | PR-EOS Phase Equilibrium & AOP Solver | Fugacities `f_j`, liquid molar volume `V_m`, oil solubility parameter `δ_oil`; dual numbers (`Dual64`) → derivatives straight into the global Newton Jacobian |
| Porous medium | Verma–Pruess & Jamin Degradation Engine | Non-linear recomputation of local $\phi(x,t)$, $k(x,t)$, and $k_{rw}(S_w, \sigma_{OiW})$ under trapped oil and TSS colmatation |

### 2.1 Discretised momentum equation — D3 §1.3.2

1D two-phase tubing flow, FVM control-volume on a **staggered grid**:

$$\frac{\partial}{\partial t}(\rho_m u_m) + \frac{\partial}{\partial z}(\rho_m u_m^2 + \gamma_m) = -\frac{\partial P}{\partial z} - \frac{f_D\,\rho_m\,|u_m|u_m}{2\,D_{tubing}(z,t)} - \rho_m g\sin\theta$$

- $\rho_m = \alpha_L\rho_L + \alpha_G\rho_G$; $\gamma_m$ = extra momentum-transfer from phase slip (drift velocity $u_{dg}$)
- $D_{tubing}(z,t) = D_0 - 2h_{asph}(z,t)$; $f_D$ from deposit roughness $\varepsilon_{asph}$ and relative diameter

### 2.2 AD hand-off contract — D3 §1.3.1

$$\mathbf{f}_j^{AD} = f_j + \frac{\partial f_j}{\partial P}dP + \frac{\partial f_j}{\partial T}dT + \sum_{k=1}^{N_c-1}\frac{\partial f_j}{\partial x_k}dx_k$$

Claim: feeds `J_THMC` **without numerical finite differences**, guaranteeing **quadratic** Newton
convergence under asphaltene phase transition and hydrate crystallisation.

---

## 3. D3 §2 — Produced-Water Re-Injection (PWRI)

### 3.1 Deep-bed filtration (convection–dispersion with retention source)

$$\frac{\partial(\phi C_{TSS})}{\partial t} + \nabla\cdot(\mathbf{u}C_{TSS}) - \nabla\cdot(\mathbf{D}\nabla C_{TSS}) = -\frac{\partial \sigma_p}{\partial t}$$

### 3.2 Iwasaki capture kinetics

$$\frac{\partial \sigma_p}{\partial t} = \lambda(\sigma_p)\,|\mathbf{u}|\,C_{TSS}$$

$$\lambda(\sigma_p) = \lambda_0\left(1 + \frac{a\,\sigma_p}{\phi_0}\right)\left(1 - \frac{\sigma_p}{\sigma_{p,\max}}\right)^{b}$$

### 3.3 Matrix porosity loss and external cake growth

$$\phi(t) = \phi_0 - \frac{\sigma_p(t)}{1 - \phi_{cake}}$$

$$\frac{d h_{cake}}{dt} = \frac{|\mathbf{u}|\,C_{TSS}}{\rho_{cake}(1-\phi_{cake})} - \beta_{eros}\max\left(0,\ |\mathbf{u}| - u_{crit}\right)\,h_{cake}(t)$$

Cake skin factor: $S_{cake}(t) = \dfrac{k_{res}}{r_w\,k_{cake}}\,h_{cake}(t)$

### 3.4 OiW droplets and Jamin capillary back-pressure

$$\frac{\partial \sigma_{OiW}}{\partial t} = \lambda_{OiW}\,|\mathbf{u}|\,C_{OiW}\qquad \Delta P_{Jamin} = \frac{2\gamma}{r_{throat}}\left(\cos\theta_r - \cos\theta_a\right)$$

$$k_{rw}(S_w, \sigma_{OiW}) = k^0_{rw}(S_w)\exp\left(-\alpha_{OiW}\,\sigma_{OiW}(t)\right)$$

---

## 4. D3 §3 — Asphaltenes

**PR-EOS:** $P = \dfrac{RT}{V_m - b_{PR}} - \dfrac{a_{PR}(T)}{V_m^2 + 2b_{PR}V_m - b_{PR}^2}$

**Oil solubility parameter:** $\delta_{oil} = \sqrt{\Delta U_{vap}/V_m}$, with $V_m$ from PR-EOS.

**Flory–Huggins extended lattice theory:**

$$\ln\phi_a + 1 - \frac{V_a}{V_m} + \chi = 0,\qquad \chi = \frac{V_a}{RT}\left(\delta_{oil} - \delta_a\right)^2$$

**Deposition kinetics, dimensionally stated:**

$$\frac{\partial S_a}{\partial t} = \frac{\alpha}{\phi\,\rho_a}\left(C_a - C_a^*\right)^{m}|\mathbf{u}|$$

with $\alpha$ in $\text{m}^{3m-1}/(\text{kg}^m\!\cdot\!\text{s})$, $C_a^* = C_a^*(P,T)$ from PR-EOS.

**Verma–Pruess percolation collapse:** $k/k_0 = \left(\dfrac{\phi-\phi_c}{\phi_0-\phi_c}\right)^{n}$, $n\in[2.0, 3.5]$

**Tubing deposition:** $\partial h_{asph}/\partial t = \dfrac{k_{dep}(C_a - C_a^*)^{m}\rho_{fluid}}{\rho_a}$, $D_{tubing}(z,t)=D_0-2h_{asph}$

**Colebrook/Chen friction factor:** $\dfrac{1}{\sqrt{f_D}} = -2\log_{10}\left(\dfrac{\varepsilon_{asph}}{3.7\,D_{tubing}(z,t)} + \dfrac{5.74}{Re^{0.9}}\right)$

Substituting `D_tubing` and `f_D` back into §2.1 closes **two-way feedback** between deposition and
pressure drop.

> ⚠️ **CONF-44.** The friction-factor form pairs the **Fanning-style** Reynolds term $5.74/Re^{0.9}$
> with a Darcy–Weisbach $f_D$, which uses $1/f \sim 2.51/(Re\sqrt f)$ in smooth pipes and the
> $\varepsilon/(3.7D)$ roughness term **without** the corresponding factor. The two conventions are
> mixed. Not reconciled anywhere in D3.

---

## 5. D3 §4 — Gas hydrates

**van der Waals–Platteeuw equilibrium:**

$$\Delta\mu_w^H(T,P) = RT\sum_i \nu_i \ln\left(1 + \sum_j C_{ij} f_j\right)$$

$\nu_i$ = number of cavities of type $i$ per water molecule (**sI**, **sII**, **sH**);
$C_{ij}(T)$ = Langmuir constant; $f_j = \varphi_j\,y_j\,P$ from the `feos-ad` flash via PR-EOS.

**Equilibrium point** found numerically from $\Delta\mu_w^H(T,P_{eq}) = \mu_w^L(T,P_{eq}) - \mu_w^\alpha(T,P_{eq})$.

**Salinity and inhibitor shift:** $P_{eq}(T,S,X_{inh}) = P_{eq,0}(T)\exp\left(\alpha_S S + \beta_{inh}X_{inh}\right)$

**Bischoff–Englezos kinetics:** $\dfrac{dn_H}{dt} = K_H A_s (f_g - f_{eq})$

**Thomas rheology:** $\mu_{eff}(\phi_H) = \mu_w\left(1 + 2.5\,\phi_H + A\,\phi_H^2\right)$, $A\approx 10.05$

**Sigmoidal smooth plug:** $k_{rel,tubing}(\phi_H) = \dfrac{1}{1+\exp\left(\gamma_H(\phi_H - \phi_{H,crit})\right)}$, $\phi_{H,crit}\approx0.30\text{–}0.35$, $\gamma_H\in[50,100]$

---

## 6. D3 — Rust DTOs (verbatim)

### 6.1 `ParticleSizeBin`

`# [derive(Debug, Clone, PartialEq, Serialize, Deserialize)]`

| Field | Type | Unit |
|---|---|---|
| `min_diameter_m` | `f64` | [m] |
| `max_diameter_m` | `f64` | [m] |
| `mass_fraction` | `f64` | dimensionless, 0.0–1.0 |

### 6.2 `PwriModuleInputDTO`

| Field | Type | Unit |
|---|---|---|
| `tss_concentration` | `f64` | [g/m³] |
| `particle_size_distribution` | `Vec<ParticleSizeBin>` | — |
| `oiw_concentration` | `f64` | [ppm] / [mg/l] |
| `filtration_coefficient_lambda_0` | `f64` | [1/m] |
| `iwasaki_a` | `f64` | — |
| `iwasaki_b` | `f64` | — |
| `sigma_p_max` | `f64` | [m³/m³] |
| `phi_0` | `f64` | dimensionless |
| `cake_permeability_md` | `f64` | [mD] |
| `cake_density_kg_m3` | `f64` | [kg/m³] |
| `cake_porosity` | `f64` | dimensionless |
| `erosion_rate_beta` | `f64` | [1/m] |
| `critical_velocity` | `f64` | [m/s] |
| `oil_entrapment_lambda` | `f64` | [1/m] |
| `jamin_blocking_factor` | `f64` | dimensionless |

**`pub fn validate(&self) -> Result<(), String>`** — checks and exact error strings (translated):

| Condition | Message |
|---|---|
| `tss_concentration < 0.0` | `tss_concentration cannot be negative` |
| `oiw_concentration < 0.0` | `oiw_concentration cannot be negative` |
| `phi_0 <= 0.0 \|\| phi_0 >= 1.0` | `phi_0 must be in (0.0, 1.0)` |
| `cake_porosity <= 0.0 \|\| cake_porosity >= 1.0` | `cake_porosity must be in (0.0, 1.0)` |
| `sigma_p_max <= 0.0 \|\| sigma_p_max >= self.phi_0` | `sigma_p_max must be in (0, phi_0)` |
| `(Σ mass_fraction - 1.0).abs() > 1e-4 && !psd.is_empty()` | `Sum of PSD mass fractions must equal 1.0` |

### 6.3 `AopPoint`

`pressure_pa: f64` [Pa] · `temperature_k: f64` [K] · `equilibrium_solubility_kg_m3: f64` [kg/m³]

### 6.4 `AsphalteneModuleInputDTO`

`asphaltene_content_weight_fraction` [0,1] · `aop_curve: Vec<AopPoint>` · `solubility_parameter_delta_a` [Pa^0.5] ·
`asphaltene_molar_volume_v_a` [m³/mol] · `asphaltene_density_kg_m3` [kg/m³] ·
`deposition_rate_constant_alpha` · `reaction_order_m` · `critical_porosity_phi_c` ·
`percolation_exponent_n` · `tubing_deposition_rate_k_dep` [m/s] · `tubing_asphalt_roughness_m` [m]

**`validate() -> Result<(), String>`:** `weight_fraction ∉ [0,1]`; `density <= 0.0`; `phi_c ∉ (0, 0.5)`; `n < 1.0`.

### 6.5 `InhibitorTypeEnum`

`None` · `Methanol` · `MonoEthyleneGlycol` (MEG) · `DiEthyleneGlycol` (DEG) · `KineticInhibitor` (KHI)

### 6.6 `GasComponentFraction`

`component_name: String` (e.g. "CH4", "C2H6", "CO2", "H2S") · `mole_fraction: f64` [0,1]

### 6.7 `HydrateModuleInputDTO`

`gas_composition: Vec<GasComponentFraction>` · `water_salinity_ppm` · `alpha_s` · `inhibitor_type: InhibitorTypeEnum` ·
`inhibitor_concentration` [0,1] · `beta_inh` · `kinetic_rate_constant_kh` [mol/(m²·Pa·s)] ·
`specific_surface_area_as` [m²/m³] · `thomas_coefficient_a` · `critical_hydrate_fraction_plug` ·
`plug_smoothing_gamma`

**`validate()`:** `salinity < 0`; `inhibitor_concentration ∉ [0,1]`; `plug < 0 or >= 1`; `gamma <= 0`;
gas mole-fraction sum deviation `> 1e-4` → `Sum of gas mole fractions must equal 1.0`.

### 6.8 D3 constants

| Quantity | Value |
|---|---|
| Verma–Pruess exponent `n` | `[2.0, 3.5]` |
| Thomas coefficient `A` | `≈ 10.05` |
| Critical hydrate fraction `φ_H,crit` | `≈ 0.30–0.35` |
| Sigmoid steepness `γ_H` | `[50, 100]` |
| PSD / mole-fraction closure tolerance | `1e-4` |
| `critical_porosity_phi_c` bound | `(0.0, 0.5)` |
| `percolation_exponent_n` bound | `>= 1.0` |
| AD numeric type | `num_dual::Dual64` |

> ⚠️ **CONF-45.** D3 specifies **no** test matrix. Its only declared gate is input `validate()`.
> The solver-level requirements are qualitative ("quadratic convergence", "C¹ derivatives").
> Additional open items: no algorithm, tolerance or bracketing for the numerical $P_{eq}$ solve;
> $A_s$ has **no closure relation** (pure input); the mapping from PR-EOS outputs to a `[kg/m³]`
> asphaltene solubility $C_a^*$ is unspecified.

---

## 7. D4 — integration architecture

| Simulator component | Functionality in main documents | D4 extension |
|---|---|---|
| THMC core and ODE/PDE solvers | multiphase flow, mass/energy balance, component transport on a **fixed** grid | **localised** time- and space-dependent source terms `R_dissolution`, `R_kill`; pointwise Jacobian modification; dynamic re-assembly of local boundaries |
| Geomechanics and phase transitions (D3) | 3D stress–strain, thermodynamic equilibrium, stress tensor, fracture opening | nonlinear changes in kill-fluid rheology; hydrate breakdown under inhibitors/depressurisation; colmatant removal on flow reversal |
| Emergency Control Module (D4) | well as **static** sink/source with constant skin factor `S` | time-varying dynamic well boundary conditions, SSSV `ShutIn` algorithms, filter-cake flushing, squeeze cementing |

**Stated computational mechanism:** localised source terms integrated directly into the nonlinear
Newton–Raphson loop; **targeted re-assembly of local Jacobian elements** without full invalidation of the
global mesh; **dynamic spatial adaptive mesh re-discretisation** in the active influence zone (acid front,
shear-wash channel) allowing discontinuous boundary problems without loss of the convergence coefficient.

**Rust justification for HPC:** zero-cost abstractions for asynchronous emergency boundary handling;
ownership/borrowing memory safety; data-race-free parallel recomputation of local conductivity matrices;
`Send`/`Sync` marker traits to distribute subdomains across CPU threads with cache locality, no GC pauses.

**Damage context:** the damaged near-wellbore zone (ПЗП) is caused by pore-channel clogging with
fine-dispersed sludge, deposition of heavy oil fractions, or oil filter-cake formation. Dynamic
recomputation of the skin factor $S(t)$ and effective permeability tensor $k_{eff}(t)$ during remedial
treatment is stated as a **critical condition for accurate rate modelling**.

---

## 8. D4 — physics

### 8.1 Matrix / hydrochloric acidising (§2.1)

$$R_{dissolution} = k_{acid}\,A_{sp}\,(C_{acid} - C_{eq})^{n}$$
$$h_{cake}(t) = h_{cake,0}\exp\left(-\lambda_{acid}\,C_{acid}\,t\right)$$
$$S_{perf}(t) = S_0 - \alpha_{acid}\ln\left(\frac{r_{w,eff}(t)}{r_w}\right)$$

Reduction of $S_{perf}$ below zero models wormhole creation.

### 8.2 Aromatic solvents (§2.2)

$$\delta_{oil}(t) = x_{solv}\,\delta_{solv} + (1-x_{solv})\,\delta_{oil,0}$$
$$\frac{dS_a}{dt} = -k_{solv}\,S_a\max\left(0,\ \delta_{oil} - \delta_{crit}\right)$$
$$r_{pore}(t) = r_{pore,0}\sqrt{1-S_a}\qquad D_{tubing}(z,t) = D_{tubing,0} - 2h_{dep}(z,t)$$

### 8.3 Backwashing (§2.3) — explicit 5-step algorithm

Trigger: reversal of filtration direction ($u_r < 0$).

1. **Velocity-vector update** — local Darcy velocity $\mathbf{u}_r$ per near-wellbore cell.
2. **Flow-direction check** — if $u_r < 0$, initiate shear computation at the boundary.
3. **Shear-force computation** — $\tau_{shear} = \mu\,|u_r|\big/\sqrt{k\phi}$
4. **Cake degradation** — if $\tau_{shear}>\tau_{crit}$: $\Delta h_{cake} = -\kappa_{wash}(\tau_{shear}-\tau_{crit})\Delta t$
5. **State-vector modification** — $k_r(t) = k_0\left(1 - h_{cake}(t)/r_{pore}\right)^{-4}$ and $k_{ij}(t) = k_{ij,0}\,f(\phi(t))/f(\phi_0)$

> ⚠️ **CONF-46.** In step 5, the function $f(\phi)$ is **never defined**, and $\Delta\phi_{strip}$ is
> introduced with **no closure** linking it to $\Delta h_{cake}$. The permeability update is therefore
> not evaluable.

### 8.4 Hydrate dissolution via THI and heating (§3.1)

$$\Delta T_{eq} = \frac{R\,T_0^2}{\Delta H_{hy}}\ln(a_{water})\qquad a_{water} = f(w_{THI})$$

$$P = \frac{RT}{v-b_{mix}} - \frac{a_{mix}(T)}{v^2 + 2b_{mix}v - b_{mix}^2}\quad(\text{PR or SRK})$$

$$\frac{dn_H}{dt} = -k_{decomp}\,A_{hy}\left(f_{g,eq}(T, w_{THI}) - f_g\right) < 0$$

### 8.5 Depressurisation (§3.2)

Trigger $P_{tubing} < P_{eq}(T)$. Latent heat absorbed and integrated into the THMC energy equation:

$$q_{thermal} = \Delta H_{hy}\,\frac{dn_H}{dt}$$

### 8.6 Conformance control (§3.3)

Crosslinked polymer gels and foams fill thief zones then polymerise, driving channel permeability to zero.
RF ceiling and post-breakthrough $f_g(t)$ repeat D1/D6/D7 verbatim. → **CONF-01, CONF-02**.

Reagent table:

| Reservoir type | Complication | Reagent | Isolation mechanism | Coverage change `ΔE_v` |
|---|---|---|---|---|
| High-permeability sandstone | aquifer water breakthrough | polyacrylamide crosslinked gel (PPG) | pore filling, $k_{water}\to 0$ | **+15 % … +25 %** |
| Fractured carbonate | gas coning | dispersed silicate foam | selective fracture blocking | **+10 % … +18 %** |
| Heterogeneous layered | early CO₂ breakthrough | thermo-reversible polymer (PAG) | blocking of washed-out streaks | **+12 % … +20 %** |

### 8.7 Well kill (§4.1)

$$u_g = C_0 u_m + u_d\qquad C_0 \approx 1.05\text{–}1.2$$
$$P_{hydro}(z,t) = \int_0^z \rho_m(z',t)\,g\,dz',\qquad \rho_m = \alpha_g\rho_g + (1-\alpha_g)\rho_{mud}$$

**Kill-success condition:** $P_{hydro}(z_{bottom},t) + \Delta P_{fric}(Q_{kill}) > P_{res}$

### 8.8 SSSV and water hammer (§4.2)

```
IF P_tubing < P_sssv_trigger OR Q_gas > Q_critical THEN
    Set BoundaryCondition = ShutIn
    Set SSSV_Valve_State = Closed
END IF
```

$$\frac{\partial P}{\partial t} + \rho a^2 \frac{\partial u}{\partial z} + \gamma_{damp}P = 0$$
$$\frac{\partial u}{\partial t} + \frac{1}{\rho}\frac{\partial P}{\partial z} + \frac{f_w\,u|u|}{2D_{tubing}} = 0$$
$$u(z_{sssv}, t) = 0,\quad \forall t \ge t_{close}$$

### 8.9 Squeeze cementing (§5.1)

$$w_{micro}(t)\to 0,\qquad k_{micro}(t) = \frac{w_{micro}(t)^2}{12}\to 0$$

$$\mu_{squeeze}(t) = \mu_0\left(1-\frac{t}{t_{setting}}\right)^{-m},\qquad 0\le t < t_{setting}$$

The explicit domain constraint removes the singularity in the numerical solver.

> [!CAUTION]
> **CONF-47.** $k_{micro}=w_{micro}^2/12$ is **dimensionally wrong**. The cubic law for a planar
> fracture is $k = w^3/12$. An exponent of 2 is an error, not a convention.
>
> Other open items: $Q_{critical}$ in the SSSV logic has **no definition, unit or validation range**
> (unlike `sssv_trigger_pressure`, bounded at $1.0\text{e}8$ Pa); no time-step or co-location prescription
> for the water-hammer PDE system; `$R_kill` is named in the comparison table as a source term and
> **never given a functional form anywhere**.

---

## 9. D4 — Rust DTOs (verbatim)

### 9.1 `RemediationValidationError`

`# [derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]`

| Variant | `Display` format string |
|---|---|
| `InvalidSolventInjectionRate(String)` | `"Solvent Rate Error: {}"` |
| `InvalidAcidVolumeConcentration(String)` | `"Acid Concentration Error: {}"` |
| `InvalidInhibitorMassFraction(String)` | `"Inhibitor Fraction Error: {}"` |
| `InvalidKillMudDensity(String)` | `"Kill Mud Density Error: {}"` |
| `InvalidKillPumpRate(String)` | `"Kill Pump Rate Error: {}"` |
| `InvalidSssvTriggerPressure(String)` | `"SSSV Pressure Error: {}"` |
| `InvalidSqueezeCementViscosity(String)` | `"Cement Viscosity Error: {}"` |

Plus `impl std::error::Error for RemediationValidationError {}`.

### 9.2 `RemediationInputDTO`

| Field | Type | SI dimension | Documented range | `validate()` guard |
|---|---|---|---|---|
| `solvent_injection_rate` | `f64` | [m³/s] | `[0.0, 0.1]` | `!(0.0..=0.1).contains(...)` |
| `acid_volume_concentration` | `f64` | — | `[0.0, 1.0]` | `!(0.0..=1.0).contains(...)` |
| `inhibitor_mass_fraction` | `f64` | — | `[0.0, 1.0]` | `!(0.0..=1.0).contains(...)` |
| `kill_mud_density` | `f64` | [kg/m³] | `[800.0, 3000.0]` | `!(800.0..=3000.0).contains(...)` |
| `kill_pump_rate` | `f64` | [m³/s] | `[0.0, 0.2]` | `!(0.0..=0.2).contains(...)` |
| `sssv_trigger_pressure` | `f64` | [Pa] | `(0.0, 1.0e8]` | `<= 0.0 \|\| > 1.0e8` |
| `squeeze_cement_viscosity` | `f64` | [Pa·s] | `(0.001, 100.0]` | `< 0.001 \|\| > 100.0` |

> ⚠️ **CONF-48.** For `squeeze_cement_viscosity` the **doc comment and summary table state the open
> interval** `(0.001, 100.0]` (rejecting `0.001`), while the **code guard `< 0.001` accepts it**.
> Doc/code disagreement on a boundary value.

### 9.3 The architectural delta between D3 and D4

| Aspect | D3 | D4 |
|---|---|---|
| Error return | `Result<(), String>` (raw string) | `Result<(), RemediationValidationError>` (typed, `Display` + `std::error::Error`) |
| Thread safety | **never stated** | explicitly `Send + Sync`, immutable thread-safe DTOs |
| EOS | PR-EOS only | **PR-EOS or SRK-EOS** for hydrate-equilibrium fugacity |
| DTO derives | `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` | same, plus `Eq` on the error enum |

This is the clearest intentional design evolution in the set: **D4 supersedes D3's error convention.**
Any adoption should implement D4's pattern and back-port it to D3's DTOs.

---

## 10. Cross-addendum symbol collisions

| Symbol | D3 | D4 | Status |
|---|---|---|---|
| `D_tubing(z,t)` | $D_0 - 2h_{asph}(z,t)$ (asphaltene deposition) | $D_{tubing,0} - 2h_{dep}(z,t)$ (wax/asphaltene) restoring toward $D_{tubing,0}$ | Same form, different restoring semantics; not reconciled |
| `h_cake(t)` | ODE growth + erosion threshold | exponential decay $h_{cake,0}e^{-\lambda_{acid}C_{acid}t}$ | **Directly incompatible** — one model, two laws |
| `S_a` | full deposition kinetic closure | appears with an undeclared source equation outside the solvent term | D4's $S_a$ is under-determined |
| `φ` | porosity | porosity | D4's `k_micro` exponent error (CONF-47) |
| Flash kernel | `feos-ad` PR-EOS | `feos-ad` PR **or** SRK | D4 strictly superset |

Neither addendum contains a `TODO`/`TBD` marker, and neither cross-references D5 (the V&V framework).