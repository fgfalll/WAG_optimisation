# Surrogate Engine: Empirical Fitting Parameters Guide

By default, the PhD Surrogate Engine operates in a strictly physics-based mode, utilizing uncalibrated analytical correlations (such as Koval and Buckley-Leverett fractional flow) driven purely by dynamic material balance. 

However, to achieve precise alignment with field history data or fine-grid compositional simulations, you can enable and supply **Empirical Fitting Parameters**. This allows the surrogate engine to act as a highly tuned proxy model for your specific reservoir.

## Enabling Fitting Parameters

To use these parameters, you must construct an `EmpiricalFittingParameters` object and pass it to the engine evaluation function. These parameters override the default, uncalibrated physical limits.

```python
from core.data_models import EmpiricalFittingParameters

fitting_params = EmpiricalFittingParameters(
    omega_tl=0.66,
    transverse_mixing_calibration=0.8,
    miscibility_window=0.015,
    c7_plus_fraction=0.45,
    k_ro_0=0.85,
    k_rg_0=0.95,
    n_o=2.5,
    n_g=2.0
)

# Pass the fitting parameters to the engine wrapper
res = wrapper.evaluate_scenario(
    reservoir_data, 
    eor_params, 
    operational_params, 
    fitting_params=fitting_params,
    use_dynamic_fractional_flow=True # Recommended to keep true for physical bounds
)
```

---

## Key Parameters and Calculation Methods

### 1. Todd-Longstaff Mixing Parameter (`omega_tl`)

The `omega_tl` parameter ($\omega$) dictates the degree of mixing between the injected CO2 and the resident oil within a representative block. It controls the effective viscosity ($\mu_{eff}$) and density of the fluids.

* **$\omega = 0.0$**: No mixing (immiscible, stark viscosity contrast).
* **$\omega = 1.0$**: Complete mixing (perfectly miscible, single-phase fluid).

**Equations:**
The effective viscosity is calculated dynamically in the engine based on the quarter-power mixing rule:
$$ \mu_{mix} = \left[ 0.5 \mu_g^{-1/4} + 0.5 \mu_o^{-1/4} \right]^{-4} $$
$$ \mu_{g,eff} = \mu_{mix}^{\omega} \cdot \mu_g^{(1-\omega)} $$
$$ \mu_{o,eff} = \mu_{mix}^{\omega} \cdot \mu_o^{(1-\omega)} $$

**How to Estimate from Data:**
* **Laboratory / Coreflood scale:** Set `omega_tl = 0.66` (matches Blackwell's experimental data).
* **Field scale (Heterogeneous):** Set `omega_tl = 0.33` to account for bypassing and incomplete grid-block mixing.
* **Secondary SWAG:** Set `omega_tl = 1.0`.
* **Calculation from Fine-Grid:** If you have a fine-grid simulation, run a coarse model and tune $\omega$ until the coarse model's breakthrough time matches the fine-grid breakthrough time.

### 2. Transverse Mixing Calibration (`transverse_mixing_calibration`)

This parameter scales the impact of the Dykstra-Parsons coefficient ($V_{DP}$) on the Koval heterogeneity factor ($H_k$). It represents how much vertical and transverse crossflow mitigates severe channeling.

**Equation:**
$$ H_k = \frac{1}{(1 - V_{DP} \cdot C_{trans})^2} $$
*(Where $C_{trans}$ is the `transverse_mixing_calibration`)*

**How to Estimate from Data:**
* **Default Physics:** $C_{trans} = 0.5$.
* **Highly Stratified (No Crossflow):** Set $C_{trans} \approx 1.0$. The layers are isolated, maximizing channeling.
* **High Vertical Permeability (Strong Crossflow):** Set $C_{trans} \approx 0.1 - 0.3$. Gravity and transverse dispersion smear the displacement front, reducing the effective heterogeneity.

### 3. Miscibility Window (`miscibility_window`) & Alpha Base (`alpha_base`)

These parameters define the shape of the physical logistic curve (sigmoid function) that dictates how the displacement transitions from immiscible to miscible as reservoir pressure ($P$) approaches and exceeds the Minimum Miscibility Pressure (MMP).

**Equations:**
The dynamic miscibility fraction ($\omega_P$) is computed using a continuous hyperbolic tangent (equivalent to a logistic sigmoid) to prevent unnatural "miscibility cliffs":
$$ \omega_P = \frac{1}{1 + \exp\left( -\beta \cdot \left(\frac{P}{MMP} - \alpha_{eff}\right) \right)} $$

Where:
* **$\beta$ (Steepness):** Determined by the `miscibility_window` ($\Delta P_r$). The engine maps this window to the 10%-to-90% logit transition interval using $\ln(9) \times 2 \approx 4.394$:
  $$ \beta = \frac{4.394}{\text{miscibility\_window}} $$
* **$\alpha_{eff}$ (Effective Midpoint):** The midpoint is shifted from the default `alpha_base` based on the heavy component fraction ($C_{7+}$):
  $$ \alpha_{eff} = \alpha_{base} + \lambda_{c7} \cdot (C_{7+} - 0.3) $$
  *(Where $\lambda_{c7} = 0.1$ by default)*

**How to Estimate from Laboratory Data (Slim-Tube / Coreflood):**

You can explicitly calculate `alpha_base` and `miscibility_window` if you have laboratory data showing recovery factor ($RF$) at various pressures.

**Step 1: Identify Key Transition Pressures**
Plot your recovery factor vs. pressure. Identify the following three points on the curve where recovery drops from its maximum miscible plateau down to its immiscible baseline:
1. **$P_{90}$**: The pressure where the miscibility benefit is **90%** retained (just below the MMP plateau).
2. **$P_{50}$**: The pressure where the miscibility benefit is exactly **50%** degraded (the midpoint of the cliff).
3. **$P_{10}$**: The pressure where the miscibility benefit is only **10%** retained (almost completely immiscible).

**Step 2: Calculate `miscibility_window`**
The `miscibility_window` defines the normalized pressure width of the 10% to 90% transition zone.
$$ \text{miscibility\_window} = \frac{P_{90} - P_{10}}{MMP} $$
*Example:* If MMP is 2000 psi, and the recovery cliff drops from 90% at 1950 psi down to 10% at 1850 psi:
`miscibility_window` = $(1950 - 1850) / 2000 = 100 / 2000 = 0.05$.

**Step 3: Calculate `alpha_base`**
The `alpha_base` defines the normalized midpoint of the transition, correcting for the $C_{7+}$ composition shift.
$$ \alpha_{base} = \left( \frac{P_{50}}{MMP} \right) - 0.1 \cdot (C_{7+} - 0.3) $$
*Example:* If the cliff midpoint ($P_{50}$) is 1900 psi, MMP is 2000 psi, and $C_{7+}$ is 0.3 (30%):
$\alpha_{base} = (1900 / 2000) - 0.1 \cdot (0.3 - 0.3) = 0.95 - 0 = 0.95$.

*(Note: If you do not have fine-grained data, the standard defaults are `alpha_base = 1.0` and `miscibility_window = 0.10`.)*

### 4. Relative Permeability Endpoints & Exponents

These parameters define the Corey-style relative permeability curves used when calculating the effective mobility ratio ($M_e$).

* **`k_ro_0`**: End-point relative permeability to oil at connate water saturation.
* **`k_rg_0`**: End-point relative permeability to gas at residual oil saturation.
* **`n_o`**: Corey exponent for oil (typically 2.0 - 4.0).
* **`n_g`**: Corey exponent for gas (typically 1.5 - 3.0).

**Equation:**
$$ M_e = \frac{k_{rg,0} / \mu_{g,eff}}{k_{ro,0} / \mu_{o,eff}} $$

**How to Estimate from Data:**
These should be extracted directly from your laboratory Special Core Analysis (SCAL) reports. 
1. Look at the Gas-Oil relative permeability tables.
2. Read the maximum $k_{rg}$ and $k_{ro}$ values.
3. Fit the normalized saturations to a power-law curve to derive $n_o$ and $n_g$.

### 5. C7+ Fraction (`c7_plus_fraction`)

This parameter fundamentally governs the heavy, non-volatile components in the oil that resist vaporization into the CO2 phase. As shown in the miscibility equations above, it directly shifts the effective miscibility midpoint ($\alpha_{eff}$).

**Equation / Relationship:**
$$ \alpha_{eff} = \alpha_{base} + 0.1 \cdot (C_{7+} - 0.3) $$
If $C_{7+}$ is higher than the 30% baseline ($0.3$), the $\alpha_{eff}$ midpoint shifts higher, requiring the reservoir to be *over-pressured* relative to the baseline MMP to achieve the same degree of miscibility.

**How to Estimate from Data:**
1. **Direct PVT Extraction:** Extract from standard PVT compositional reports. Sum the mole fractions ($z_i$) of all components from Heptane ($C_7$) and heavier:
   $$ C_{7+} \text{ Fraction} = \sum_{i=C7}^{C_{max}} z_i $$
2. **Correlation Estimation (Katz-Firoozabadi method):** If you only have API gravity ($\gamma_{API}$), you can estimate the $C_{7+}$ fraction using empirical compositional heuristics:
   $$ C_{7+} \text{ Mole Fraction} \approx 1.0 - 0.015 \cdot \gamma_{API} $$
   *(Applicable roughly for black oils with API $< 45^\circ$. For volatile oils or gas condensates, the Ovalle correlation based on GOR is preferred: $z_{C7+} \approx 0.3157 \cdot GOR^{-0.9205}$ where GOR is in Mscf/STB).*

---

## Recommended Workflow for Calibration

If you have CMG, ECLIPSE, or Intersect history data, follow this sequence to calibrate the surrogate engine:

1. **Match Breakthrough Time:** Tune the `transverse_mixing_calibration` ($C_{trans}$). If breakthrough happens too early in the surrogate, decrease $C_{trans}$. If it happens too late, increase $C_{trans}$.
2. **Match Plateau Rate / Mobility:** Tune the Todd-Longstaff parameter (`omega_tl`). If the surrogate produces oil too fast after breakthrough, decrease `omega_tl` (makes the CO2 more mobile/viscous fingering worse).
3. **Match Pressure Depletion Response:** Tune the `miscibility_window`. Run a scenario where pressure drops below MMP and adjust the window until the surrogate's drop in recovery matches the fine-grid simulator's penalty.