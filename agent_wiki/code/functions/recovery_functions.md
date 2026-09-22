# Recovery & Displacement Functions

## 1. `calculate_mmp()`

- **File**: `evaluation/mmp.py`
- **Signature**: `def calculate_mmp(params: MMPParameters, method: str = "cronquist") -> float`
- **Modification Risk**: **HIGH**

### Description
Calculates the Minimum Miscibility Pressure (psia) required for multi-contact miscibility between injected pure/impure CO₂ and reservoir crude.

### Available Methods
1. `"cronquist"` (Default):
   $$MMP = 15.988 \cdot T^{0.7442} \cdot \left( \frac{C_1}{C_2 - C_5} \right)^{0.2111} \cdot (55 - \gamma_{API})^{0.279}$$
   - $T$: Temperature in °F.
   - $C_1$: Mole fraction of methane in crude.
   - $C_2 - C_5$: Mole fraction of intermediate hydrocarbons.
   - $\gamma_{API}$: Crude stock tank oil gravity.
   - *Singularity guard*: If $\gamma_{API} \ge 55$, clamps API to 54.9 to avoid complex numbers.
2. `"lee"`: Temperature and heptanes-plus molecular weight polynomial.
3. `"glaso"`: Volatile oil correlation.
4. `"alston"`: C2-C4 intermediate enrichment correlation.
5. `"yuan"`: Impure CO₂ stream correlation (N₂ and CH₄ contamination adjustments).

---

## 2. `_calculate_heterogeneity_factor()`

- **File**: `core/engine_surrogate/analytical_models.py`
- **Signature**: `def _calculate_heterogeneity_factor(self, v_dp: float) -> float`
- **Modification Risk**: **CRITICAL**

### Formulation
Computes Koval heterogeneity factor $H_k$ with calibrated transverse dispersion coefficient:
$$H_k = \frac{1}{(1 - 0.80 \cdot V_{DP})^2}$$
Where $V_{DP}$ is the Dykstra-Parsons permeability variation coefficient ($0 \le V_{DP} < 1$).
- For homogeneous reservoirs ($V_{DP} = 0$), $H_k = 1.0$.
- For typical reservoirs ($V_{DP} = 0.6$), $H_k = 1 / (1 - 0.48)^2 = 3.698$.

---

## 3. `_calculate_effective_mobility_ratio()`

- **File**: `core/engine_surrogate/analytical_models.py`
- **Signature**: `def _calculate_effective_mobility_ratio(self, m_endpoint: float, omega: float = 0.67) -> float`
- **Modification Risk**: **HIGH**

### Formulation
Implements Todd-Longstaff effective mobility ratio $M_e$:
$$M_e = \left[ \omega \cdot M^{1/4} + (1 - \omega) \right]^4$$
- $M$: Endpoint mobility ratio $M = (\mu_o / \mu_g) \cdot (k_{rg}^0 / k_{ro}^0)$.
- $\omega$: Todd-Longstaff mixing parameter (default 0.67 for field-scale viscous fingering).
- Accounts for partial transverse dispersion within fingers.

---

## 4. `_calculate_craig_areal_sweep()`

- **File**: `core/engine_surrogate/analytical_models.py`
- **Signature**: `def _calculate_craig_areal_sweep(self, m_e: float) -> float`
- **Modification Risk**: **MEDIUM**

### Formulation
Evaluates breakthrough areal sweep efficiency $E_{A,\text{bt}}$ for 5-spot patterns:
$$E_A(M_e) = 0.546 + \frac{0.059}{M_e + 0.1} + 0.05 \cdot \log_{10}(M_e)$$
Clamped to $E_A \in [0.40, 0.95]$.
