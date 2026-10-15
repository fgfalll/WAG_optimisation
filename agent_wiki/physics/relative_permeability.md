# Relative Permeability & Capillary Desaturation

## 1. Overview and Purpose

Relative permeability functions characterize the simultaneous flow of multiple immiscible fluids through the porous rock matrix. In the CO₂ EOR Optimizer, relative permeability directly dictates the fractional flow of oil, gas, and water, as well as capillary residual trapping.

---

## 2. Corey Power-Law Correlations

The standard model used throughout `analytical_models.py`, `surrogate_models.py`, and `profile_generator_fast.py` is the Corey (1954) empirical relationship.

### A. Normalized Saturation ($S^*$)
For the gas-oil system:
$$S^* = \frac{S_g - S_{gc}}{1 - S_{or} - S_{wi} - S_{gc}}$$
Where:
- $S_g$: Gas (CO₂) saturation.
- $S_{gc}$: Critical gas saturation (typically 0.05).
- $S_{or}$: Residual oil saturation (typically 0.20–0.30).
- $S_{wi}$: Connate water saturation (typically 0.20–0.25).
- $S^*$ is bounded to $[0.0, 1.0]$.

### B. Relative Permeability Functions
$$k_{ro}(S^*) = k_{ro,0} \cdot (1 - S^*)^{n_o}$$
$$k_{rg}(S^*) = k_{rg,0} \cdot (S^*)^{n_g}$$
Where:
- $k_{ro,0}$: Endpoint relative permeability to oil at $S_{wi}$ (default = 0.80).
- $k_{rg,0}$: Endpoint relative permeability to gas at $S_{or}$ (default = 1.00).
- $n_o$: Corey exponent for oil (default = 2.0).
- $n_g$: Corey exponent for gas (default = 2.0).
- Implementation: [core/engine_surrogate/analytical_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L725-L740).

---

## 3. Capillary Desaturation & Interfacial Tension (IFT)

Near and above MMP, interfacial tension ($\sigma$) between CO₂ and crude oil drops by several orders of magnitude (from ~25 mN/m to < 0.01 mN/m). This increases the capillary number $N_c$ and mobilizes residual oil.

### A. Interfacial Tension Formulation
In [core/simulation/recovery_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L176-L210):
$$\sigma(P) = \sigma_0 \cdot \exp\left( -20.0 \cdot \frac{P}{P_{MMP}} \right)$$
Bounded to $\sigma \ge 0.001\text{ mN/m}$.

### B. Capillary Number ($N_c$)
$$N_c = \frac{v \cdot \mu_{CO2}}{\sigma}$$
Where $v$ is interstitial Darcy velocity ($q / (A \cdot \phi)$).

### C. Reduced Residual Oil Saturation ($S_{or}^*$)
Following the capillary desaturation curve:
$$S_{or}^* = S_{or} \cdot \left[ 1 + \left( \frac{N_c}{N_{c,\text{crit}}} \right)^{-0.5} \right]^{-1}$$
Where critical capillary number $N_{c,\text{crit}} = 10^{-5}$.
- Implementation: [core/simulation/recovery_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/simulation/recovery_models.py#L237-L265).
- Scientific Origin: Lake (1989), *Enhanced Oil Recovery*.
