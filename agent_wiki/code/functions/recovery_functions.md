# Recovery & Displacement Functions

## 1. `calculate_mmp()`

- **File**: `evaluation/mmp.py`
- **Signature**: `def calculate_mmp(params: MMPParameters, method: str = "cronquist") -> float`
- **Modification Risk**: **HIGH**

### Description
Calculates the Minimum Miscibility Pressure (psia) required for multi-contact miscibility between injected pure/impure CO₂ and reservoir crude.

### Available Methods
1. `"cronquist"` (Default):
   $$P_{MMP} = 15.988 \cdot T_F^Y \quad [\text{psia}]$$
   - $T_F$: Temperature in °F.
   - $Y = 0.744206 + 0.0011038 \cdot MW_{C5+} + 0.0015279 \cdot Vol$.
   - $MW_{C5+} = 4247.98641 \cdot \text{API}^{-0.87022}$ (DOE / CO₂ Prophet standard formulation) or $\max(72.0, M_{C7+} - 20.0)$.
   - $Vol$: Mole percent of volatiles ($C_1 + N_2$) in crude oil.
   - *Robustness*: Monotonically decreasing with API gravity, non-singular for all $\text{API} \ge 55^\circ$ (SCI-FLAW-13 resolved).
2. `"yellig_metcalfe"`: SPE 7477 pure-CO₂ correlation with $1070\text{ psia}$ lower floor for $T < 95^\circ\text{F}$.
3. `"alston"`: C2-C4 intermediate enrichment and impure gas streams via Kay's pseudo-critical temperature rules $(T_{pc,\text{CO2}}/T_{pc,\text{gas}})^A$.
4. `"yuan"`: Compositional correlation with non-decreasing impurity factor $c = 1.0 + 1.25 \cdot (1 - x_{\text{CO2}})^{0.8}$.
5. `"lee"` & `"glaso"`: Volatile and heavy crude correlations.

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
