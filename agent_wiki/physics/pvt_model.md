# PVT Model & Flash Calculations

## 1. Overview and Purpose

The Pressure-Volume-Temperature (PVT) model provides phase equilibria and volumetric behavior of reservoir fluids as pressure depletes or builds during injection.

Two distinct PVT modalities exist in the repository:
1. **Black-Oil PVT**: Tabulated values of $B_o, B_g, R_s, \mu_o, \mu_g$ versus pressure.
2. **Compositional EOS**: Detailed thermodynamic phase split via the Peng-Robinson Equation of State.

---

## 2. Tabulated Black-Oil PVT Structure

The `PVTProperties` class in [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py#L1104-L1158) supports tabular lookups:

| Table / Property | Symbol | Units | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| `pressure_points` | $P$ | psia | Grid of pressure evaluation points |
| `bo_table` | $B_o$ | RB/STB | Oil formation volume factor |
| `bg_table` | $B_g$ | RB/SCF | Gas formation volume factor |
| `rs_table` | $R_s$ | SCF/STB | Solution gas-oil ratio |
| `oil_viscosity_table` | $\mu_o$ | cP | Live oil viscosity |
| `gas_viscosity_table` | $\mu_g$ | cP | In-situ gas viscosity |

Interpolation is performed using 1D linear splines (`scipy.interpolate.interp1d`). If evaluated outside tabulated ranges, values are clamped to endpoint limits.

---

## 3. Compositional Two-Phase Flash Calculation

Implemented in [core/compositional_engine/phase_behavior/flash_calculator.py](file:///d:/rep/4.6/co2eor_optimizer/core/compositional_engine/phase_behavior/flash_calculator.py):

### 1. Rachford-Rice Equation
For an overall mixture composition $z_i$ at specified $P, T$, the vapor mole fraction $V = n_V / n_{total}$ is found by solving the Rachford-Rice objective:
$$f(V) = \sum_{i=1}^{N_c} \frac{z_i (K_i - 1)}{1 + V (K_i - 1)} = 0$$
Where equilibrium ratios (K-values) are defined as $K_i = y_i / x_i$.

### 2. K-Value Initialization: Wilson Correlation
Initial estimates of K-values are generated using the Wilson (1969) empirical equation:
$$K_i^{(0)} = \frac{P_{c,i}}{P} \exp\left[ 5.373 (1 + \omega_i) \left( 1 - \frac{T_{c,i}}{T} \right) \right]$$

### 3. Thermodynamic Equilibrium: Equal Fugacities
Equilibrium requires that the chemical potential (fugacity) of each component be identical in the vapor and liquid phases:
$$f_i^V(P, T, \vec{y}) = f_i^L(P, T, \vec{x}) \implies \ln \phi_i^V + \ln y_i = \ln \phi_i^L + \ln x_i$$
Fugacity coefficients $\phi_i^V$ and $\phi_i^L$ are computed via analytical Peng-Robinson EOS root-finding:
$$\ln \phi_i = \frac{b_i}{b}(Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2}B} \left( \frac{2\sum_j z_j a_{ij}}{a} - \frac{b_i}{b} \right) \ln\left( \frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B} \right)$$
Successive substitution is iterated until $\|K_i^{(k+1)} - K_i^{(k)}\| < 10^{-6}$, accelerating to Newton-Raphson near the critical point.

---

## 4. Execution Modes: Baseline Propagation vs. Per-Step Flash (`enforce_step_flash`)

To achieve high optimization throughput without sacrificing thermodynamic accuracy, the engine supports two PVT execution paradigms:

| Mode | Flag | Default | Description | Performance |
| :--- | :--- | :---: | :--- | :--- |
| **Project PVT Baseline Propagation** | `enforce_step_flash=False` | **Yes** | Performs 1 baseline PVT evaluation at initial reservoir conditions $(P_{\text{init}}, T_{\text{res}})$, establishing reference $B_{o,\text{ref}}, B_{g,\text{ref}}, \mu_{o,\text{ref}}, \mu_{g,\text{ref}}$. Properties across timesteps are propagated analytically via compressibility and volume factor relations. | $\sim 0.1\text{ ms}$ / eval |
| **Rigorous Per-Step Flash** | `enforce_step_flash=True` | No | Evaluates full Peng-Robinson EOS two-phase flash at every discrete pressure step in the simulation profile. Recommended for highly volatile oils near the critical point. | $\sim 10-50\text{ ms}$ / eval |

### User Interface Control
Configured in `DataManagementWidget` (PVT tab) via the **"Enforce Per-Step Flash Calculations"** checkbox. It populates `EORParameters.enforce_step_flash` and `PVTProperties.enforce_step_flash`.

---

## 5. Solvent-Extended Evolution Without Per-Step Flash Overhead (`SolventExtendedPVTEngine`)

To capture full phase behavior (oil swelling, viscosity thinning, supercritical $\text{CO}_2$ compressibility, solution gas liberation) during optimization without paying the steep computational cost of iterative Rachford-Rice flash equations at every single timestep, the engine employs a physics-informed **Solvent-Extended State formulation** (`core/engine_surrogate/pvt_state.py`):

1. **State Tracking Variables**:
   - Liquid solvent concentration: $x_{\text{CO2}}(t) \in [0, x_{\text{max}}]$ where $x_{\text{max}} = \min(0.85, 0.20 + 0.65 \cdot (P / MMP))$.
   - Vapor solvent concentration: $y_{\text{CO2}}(t) \in [y_{\text{min}}, 1.0]$.
   - Reservoir pore pressure: $P(t)$.
2. **Coupled Volumetric & Transport Dynamics**:
   - Pure $\text{CO}_2$ density and formation volume factor $B_{\text{CO2}}(P, T)$ are solved analytically using the Peng-Robinson cubic equation of state.
   - Oil swelling $S_F(P, x_{\text{CO2}})$ and live oil formation volume factor $B_o(P, x_{\text{CO2}})$ adjust in-situ liquid volume.
   - Live oil viscosity $\mu_o(P, x_{\text{CO2}})$ thins logarithmically with dissolved solvent.
   - Gas phase properties (density, viscosity, $Z$-factor) are evaluated as real-gas binary mixtures of $(\text{CO}_2 + \text{Hydrocarbon Solution Gas})$.
3. **Decoupled Hydrodynamic Evolution**:
   - Free gas transport saturation $S_g$ changes dynamically via the Koval fractional flow formulation.
   - Thermodynamic properties depend strictly on $(P, x_{\text{CO2}})$, completely eliminating the fatal flaw of indexing PVT properties against hydrodynamic saturation $S_g$.
4. **Surface Multi-Stage Flash Separation**:
   - Produced oil shrinks to stock tank oil by $1 / B_o(P, x_{\text{CO2}})$.
   - Dissolved $\text{CO}_2$ flashes into the produced $\text{CO}_2$ stream.
   - Solution hydrocarbon gas degasses at $R_{s,\text{base}}$ into the sales gas stream.

