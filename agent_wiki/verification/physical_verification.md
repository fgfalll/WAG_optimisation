# Physical & Thermodynamic Verification: Conservation & State Invariants

## 1. Scope & Physical Principles

Level 2 Physical Verification checks whether the computational models obey the fundamental laws of thermodynamics, continuum mechanics, and reservoir physics:
1. **First Law of Thermodynamics (Conservation of Mass & Energy)**: No mass is created or destroyed. Cumulative fluid production and in-situ accumulation must exactly equal cumulative injection and initial in-place volume.
2. **Second Law of Thermodynamics (Compressibility & Entropy)**: Isothermal compressibility must be strictly positive ($c = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T > 0$). Fluid volume decreases under pressure.
3. **Viscosity & Shear Rheology**: Liquid and dense supercritical fluid viscosities must increase under pressure ($\frac{\partial\mu}{\partial P} > 0$).
4. **Thermal Expansion**: Isobaric thermal expansion must cause density to decrease with increasing temperature ($\left(\frac{\partial\rho}{\partial T}\right)_P < 0$).
5. **Phase Equilibrium & EOS Consistency**: Vapor compressibility factor $Z_V \ge 0.80$, Liquid compressibility factor $Z_L < 0.80$. Equal fugacities at equilibrium ($f_i^L = f_i^V$).
6. **Geomechanical Containment Safety**: Bottom-hole injection pressure must never breach the safe caprock fracture gradient ($P_{\text{sandface}} \le 0.90 P_{\text{frac}}$).

---

## 2. Documented Physical Invariants & Code Audit

### 2.1 Positivity of Oil Compressibility
- **Physics**: In undersaturated oil reservoirs ($P > P_b$), increasing pore pressure compresses the fluid:
  $$c_o = -\frac{1}{B_o}\frac{dB_o}{dP} > 0 \implies \frac{dB_o}{dP} < 0$$
- **Code Audit**: [`core/data_integration_engine.py:370, 456`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L370)
  $$B_o(P) = 1.2 + 0.0001 \cdot (P - 4000) \implies \frac{\partial B_o}{\partial P} = +10^{-4} > 0$$
- **Verification Status**: **CONTRADICTED BY PHYSICAL TEST (SCI-FLAW-02)**. Oil expands upon pressurization, violating the second law of thermodynamics.

---

### 2.2 Pressure Dependence of Fluid Viscosities
- **Physics**: Liquids and dense supercritical fluids experience increased molecular interaction and packing density under compression, causing dynamic viscosity to rise:
  $$\left(\frac{\partial\mu_o}{\partial P}\right)_T > 0, \quad \left(\frac{\partial\mu_{\text{CO2}}}{\partial P}\right)_T > 0$$
- **Code Audit**: [`core/data_integration_engine.py:372, 375`](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L372)
  $$\mu_o(P) = \mu_{o,\text{ref}} \cdot \exp(-0.0003 \cdot \Delta P)$$
  $$\mu_{\text{CO2}}(P) = \mu_{g,\text{ref}} \cdot \exp(-0.0002 \cdot \Delta P)$$
- **Verification Status**: **CONTRADICTED BY PHYSICAL TEST (SCI-FLAW-03)**. At 6,000 psia, fluid viscosities drop by 45%, providing artificial incentives for over-injection.

---

### 2.3 Isobaric Thermal Expansion of Supercritical CO₂
- **Physics**: Fluid density decreases with temperature due to thermal expansion:
  $$\beta = -\frac{1}{\rho}\left(\frac{\partial\rho}{\partial T}\right)_P > 0 \implies \left(\frac{\partial\rho}{\partial T}\right)_P < 0$$
- **Code Audit**: [`core/unified_engine/physics/co2_properties.py:140`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/co2_properties.py#L140)
  $$\rho = 1.01 + 0.0109 P_{\text{MPa}} - 1.25\times 10^{-5} P_{\text{MPa}}^2 + 0.0023 T_{^\circ\text{C}}$$
- **Verification Status**: **CONTRADICTED BY PHYSICAL TEST (SCI-FLAW-04)**. Hot CO₂ is denser than cold CO₂, predicting unphysical densities up to $1,232\text{ kg/m}^3$.

---

### 2.4 Cubic Equation of State Phase Identification
- **Physics**: In a cubic EOS (Peng-Robinson or Soave-Redlich-Kwong), the largest root of $Z$ corresponds to the vapor phase ($Z_V \approx 0.8 - 1.05$), while the smallest positive real root corresponds to the liquid phase ($Z_L \approx 0.05 - 0.35$).
- **Code Audit**: [`core/unified_engine/physics/eos/__init__.py:195`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L195)
  ```python
  "phase": "V" if Z < 0.8 else "L"
  ```
- **Verification Status**: **CONTRADICTED BY PHYSICAL TEST (SCI-FLAW-08)**. Completely inverts phase assignment: dense liquid oil is treated as vapor gas, and supercritical gas is treated as liquid.

---

### 2.5 Geomechanical Containment & Regulatory Pressure Cap
- **Physics & Regulation**: EPA Class VI Underground Injection Control (UIC) guidelines require injection pressure to remain strictly below the caprock fracturing threshold to prevent induced seismicity and caprock breach:
  $$P_{\text{sandface}} \le 0.90 \cdot P_{\text{frac}}$$
- **Code Audit**: [`core/engine_surrogate/surrogate_engine.py:343-346`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L343)
  Enforces $P \le 0.90 P_{\text{frac}}$ by throttling net injection $q_{\text{inj}} \to 0$ when reservoir pressure reaches the safe caprock ceiling.
- **Verification Status**: **VERIFIED (Passes EPA Class VI Standard)** in `tests/scientific/boundary_conditions/test_geomechanical_ceiling.py`.
