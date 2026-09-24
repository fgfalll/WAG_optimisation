# Simulation Pipeline Functions

## 1. `evaluate_scenario()`

- **File**: `core/engine_surrogate/surrogate_engine.py`
- **Signature**: `def evaluate_scenario(self, params: Dict[str, Any]) -> SimulationResults`
- **Modification Risk**: **CRITICAL**

### Description
Coordinates the entire surrogate simulation pipeline. Evaluates candidate operational decisions and reservoir properties, returning a complete `SimulationResults` container containing rate time-series, pressure profile, NPV, and trapping breakdowns.

### Algorithm
1. Unpacks input dictionaries into `EORParameters` and `OperationalParameters`.
2. Invokes `PhDHybridRecoveryModel.predict_recovery()` to calculate ultimate $RF$.
3. Calls `FastProfileGenerator.generate_profiles()` to generate monthly time-series for oil, water, and gas.
4. Computes reservoir pressure history $P(t)$ using coupled deliverability material balance (`_calculate_pressure_profile()`).
5. Evaluates CO₂ recycling vs make-up purchase (`_calculate_co2_purchased_recycled()`).
6. Computes field discounted cash flow (`_calculate_engine_npv()`).
7. Evaluates carbon storage trapping mechanisms (`CO2StorageSurrogate.calculate_storage()`).
8. Verifies mass balance and returns structured results.

---

## 2. `_calculate_pressure_profile()`

- **File**: `core/engine_surrogate/surrogate_engine.py`
- **Signature**: `def _calculate_pressure_profile(self, profiles: Dict[str, np.ndarray]) -> np.ndarray`
- **Modification Risk**: **CRITICAL**

### Physical Formulation
Solves the discrete material balance equation with effective deliverability damping:
$$dP_t = \frac{(q_{\text{inj,res}} - q_{\text{prod,res}}) \cdot \Delta t}{V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t}$$
Where:
- $q_{\text{inj,res}} = q_{\text{inj,actual}} \times B_{g,\text{dynamic}}$ (converting MSCF/d to RB/d).
- $q_{\text{prod,res}} = q_{o} B_o + q_w B_w + (q_g - q_o R_s) B_g$ (RB/d).
- $J_{\text{eff}} = J_{\text{prod}} + J_{\text{inj}}$ (psi$^{-1}$ damping deliverability factor).
- Sandface injection ceiling: $P(t) \le 0.90 \times P_{\text{frac}}$ (EPA Class VI limit).
- Numerical step limiter: $|dP_t| \le 450\text{ psi/step}$.

---

## 3. `_calculate_co2_purchased_recycled()`

- **File**: `core/engine_surrogate/surrogate_engine.py`
- **Signature**: `def _calculate_co2_purchased_recycled(self, profiles: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]`
- **Modification Risk**: **HIGH**

### Mass Balance Rule
- Injected gas is satisfied first by recompressing produced CO₂ up to available compressor capacity.
- Any remaining injection requirement is purchased as fresh make-up CO₂:
  $$q_{\text{purchased}}(t) = \max\left(0, q_{\text{inj,gas}}(t) - \min(q_{\text{prod,gas}}(t) \cdot \eta_{\text{capture}}, C_{\text{compressor}})\right)$$
  $$q_{\text{recycled}}(t) = q_{\text{inj,gas}}(t) - q_{\text{purchased}}(t)$$
- Guarantees: $\sum q_{\text{recycled}} \le \sum q_{\text{prod,gas}}$ (no recycled gas created out of nothing).
