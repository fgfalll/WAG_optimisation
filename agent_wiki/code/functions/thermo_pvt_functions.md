# Thermodynamic & PVT Functions

## 1. `flash_calculation()`

- **File**: `core/unified_engine/physics/eos/__init__.py`
- **Signature**: `def flash_calculation(p: float, t: float, z_feed: np.ndarray) -> FlashResult`
- **Modification Risk**: **HIGH**

### Description
Performs isothermal two-phase vapor-liquid equilibrium (VLE) flash using the cubic Peng-Robinson equation of state.

### Solution Algorithm
1. Initializes equilibrium ratios $K_i = y_i / x_i$ using Wilson's empirical correlation:
   $$K_i = \frac{P_{c,i}}{P} \exp\left[ 5.373 (1 + \omega_i) \left( 1 - \frac{T_{c,i}}{T} \right) \right]$$
2. Solves the 1D Rachford-Rice equation for vapor fraction $\Psi \in [0, 1]$:
   $$f(\Psi) = \sum_{i=1}^{N_c} \frac{z_i (K_i - 1)}{1 + \Psi (K_i - 1)} = 0$$
   using bounded Newton-Raphson iteration with bisection fallback.
3. Computes phase compositions $x_i = z_i / [1 + \Psi(K_i - 1)]$ and $y_i = K_i x_i$.
4. Evaluates fugacity coefficients $\phi_{i,L}$ and $\phi_{i,V}$ via cubic PR EOS.
5. Updates $K_i^{(k+1)} = K_i^{(k)} \cdot (\phi_{i,L} / \phi_{i,V})$ until convergence $|\ln(\phi_{i,L}/\phi_{i,V})| < 10^{-6}$.

---

## 2. `calculate_density()`

- **File**: `core/unified_engine/physics/co2_properties.py`
- **Signature**: `def calculate_density(p_psia: float, t_f: float) -> float`
- **Modification Risk**: **MEDIUM**

### Description
Calculates the thermophysical density of pure CO₂ (lb/cu ft and kg/m³) across subcritical, critical, and supercritical regimes using the Span-Wagner / Altunin formulation.
- Converts pressure: $P_{\text{bar}} = P_{\text{psia}} \times 0.0689476$.
- Converts temperature: $T_{\text{K}} = (T_{\text{°F}} - 32) \times 5/9 + 273.15$.
- Enforces physical density range: $\rho \in [0.1, 75.0]\text{ lb/ft}^3$.

---

## 3. `corey_rel_perm()`

- **File**: `core/unified_engine/physics/relative_permeability.py`
- **Signature**: `def corey_rel_perm(sw: float, sg: float, params: RelativePermeabilityParams) -> Tuple[float, float, float]`
- **Modification Risk**: **HIGH**

### Formulation
Computes 3-phase relative permeabilities using modified Corey power-law models:
- **Normalized Water Saturation**:
  $$S_{wn} = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{orw}}$$
  $$k_{rw} = k_{rw}^0 \cdot (S_{wn})^{n_w}$$
- **Normalized Gas Saturation**:
  $$S_{gn} = \frac{S_g - S_{gc}}{1 - S_{wc} - S_{org} - S_{gc}}$$
  $$k_{rg} = k_{rg}^0 \cdot (S_{gn})^{n_g}$$
- **Oil Relative Permeability (Stone's Method II / Corey)**:
  $$S_{on} = \frac{1 - S_w - S_g - S_{or}}{1 - S_{wc} - S_{or}}$$
  $$k_{ro} = k_{ro}^0 \cdot (S_{on})^{n_o}$$
All saturations are strictly clamped to $[0, 1]$.
