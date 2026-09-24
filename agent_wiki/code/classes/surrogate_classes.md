# Surrogate Simulation Classes

## 1. `SurrogateEngine`

- **File**: `core/engine_surrogate/surrogate_engine.py`
- **Role**: Active master simulation engine evaluated during every optimization iteration.
- **Inherits**: `BaseEngine` (duck-typed or explicit adapter).
- **Modification Risk**: **CRITICAL**

### Primary Attributes
- `reservoir_data: ReservoirData`: Reservoir geometry, pore volume ($V_p$), compressibility, initial state.
- `eor_params: EORParameters`: WAG ratio, cycle time, injection rate, gas fraction.
- `operational_params: OperationalParameters`: Well counts, bottom-hole drawdown limits, fracture caprock bounds.
- `economic_params: EconomicParameters`: Oil price, CO₂ purchase/recycle costs, discount rate.
- `analytical_surrogate: AnalyticalSurrogate`: Facade instance computing $RF$ and fractional flow.
- `profile_generator: FastProfileGenerator`: Vectorized rate profile generator.

### Key Methods
- `evaluate_scenario(params: Dict[str, Any]) -> SimulationResults`:
  Coordinates the full 20-year monthly simulation run:
  1. Computes ultimate recovery factor $RF$ via `analytical_surrogate.predict_recovery()`.
  2. Synthesizes oil, water, and gas profiles via `profile_generator.generate_profiles()`.
  3. Evaluates reservoir pressure history $P(t)$ via `_calculate_pressure_profile()`.
  4. Evaluates fresh vs recycled CO₂ streams via `_calculate_co2_purchased_recycled()`.
  5. Computes net discounted cash flow via `_calculate_engine_npv()`.
  6. Evaluates trapped carbon via `CO2StorageSurrogate.calculate_storage()`.
  7. Packages results into typed `SimulationResults`.
- `_calculate_pressure_profile(profiles: Dict[str, np.ndarray]) -> np.ndarray`:
  Solves explicit damped material balance pressure increments:
  $$dP = \frac{(q_{\text{inj,actual}} - q_{\text{prod,actual}}) \cdot \Delta t}{V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t}$$
  Enforces sandface injection ceiling ($P \le 0.90 \times P_{\text{frac}}$) and producer minimum drawdown ($P_{wf} \ge P_{\text{min}}$).
- `_calculate_co2_purchased_recycled(profiles: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]`:
  Tracks fresh make-up CO₂ and recycled produced CO₂, ensuring cumulative recycled gas never exceeds cumulative produced gas.

---

## 2. `FastProfileGenerator`

- **File**: `core/engine_surrogate/profile_generator_fast.py`
- **Role**: Generates monthly oil, gas, water, and injection time-series profiles.
- **Modification Risk**: **HIGH**

### Key Methods
- `generate_profiles(recovery_factor: float, eor_params: EORParameters) -> Dict[str, np.ndarray]`:
  Vectorized profile generator supporting `CONTINUOUS_CO2`, `WAG`, `SWAG`, and `HUFF_N_PUFF`.
- `_calculate_composite_ipr(p_res: float, p_wf: float, p_bubble: float, j_linear: float) -> float`:
  Implements Composite Vogel-Darcy IPR:
  - If $P_{wf} \ge P_b$: Darcy linear deliverability $q = J \cdot (P_{\text{res}} - P_{wf})$.
  - If $P_{wf} < P_b$: Quadratic Vogel deliverability below bubble point.
- `_apply_wag_buffering(oil_rate: np.ndarray, water_rate: np.ndarray, gas_rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`:
  Models phase mobility contrast ($\Delta \lambda / \Sigma \lambda$) during gas/water cycles and strictly re-normalizes cumulative oil to $OOIP \times RF$.

---

## 3. `PhDHybridRecoveryModel`

- **File**: `core/engine_surrogate/analytical_models.py`
- **Role**: Evaluates the ultimate recovery factor ($RF$) combining Koval displacement, Craig areal sweep, and miscibility weighting.
- **Modification Risk**: **CRITICAL**

### Mathematical Formulations
- **Heterogeneity Multiplier**:
  $$H_k = \frac{1}{(1 - 0.80 \cdot V_{DP})^2}$$
- **Effective Mobility Ratio**:
  $$M_e = \left[ 0.67 \cdot M^{1/4} + (1 - 0.67) \right]^4$$
- **Craig Areal Sweep**:
  $$E_A = 0.546 + \frac{0.059}{M_e + 0.1} + 0.05 \cdot \log_{10}(M_e)$$
- **Miscibility Transition Weight ($\omega$)**:
  $$\omega = 1 - e^{-(P - MMP)/MMP} \quad \text{for } P \ge MMP, \quad \omega = 0 \quad \text{for } P < MMP$$
- **Final Clamping**:
  $$RF = \text{clip}(E_A \cdot E_V \cdot E_D \cdot [\omega RF_{\text{misc}} + (1 - \omega) RF_{\text{immisc}}], 0.05, 0.80)$$
