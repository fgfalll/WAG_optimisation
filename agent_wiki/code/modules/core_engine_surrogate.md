# Core Engine Surrogate Subsystem (`core/engine_surrogate/`)

## 1. Overview & Architectural Role

The `core/engine_surrogate/` package is the **sole active production simulation engine** in the `co2eor_optimizer` codebase. Every optimization iteration launched from the GUI or CLI routes through this package.

Its primary design goal is high-speed simulation: evaluating a 20-year field development scenario (with production rates, injection rates, bottom-hole pressures, average reservoir pressure, CO₂ recycling, and net storage) in **1.0 to 2.5 milliseconds**.

---

## 2. Module Inventory

| Module | LOC | Primary Classes / Functions | Scientific Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `surrogate_engine.py` | 512 | `SurrogateEngine`, `SurrogateEngineWrapper` | Master simulation coordinator; tank material balance pressure ODE; NPV and CO₂ recycling. | **CRITICAL** |
| `profile_generator_fast.py` | 420 | `FastProfileGenerator` | Synthesizes rate profiles; Composite Vogel-Darcy IPR; mass-conserved WAG phase buffering. | **HIGH** |
| `analytical_models.py` | 785 | `PhDHybridRecoveryModel`, `AnalyticalSurrogate`, `KovalRecoveryModel` | Ultimate recovery factor ($RF$); Koval heterogeneity $H_k$; Craig sweep; Todd-Longstaff $M_e$. | **CRITICAL** |
| `surrogate_models.py` | 310 | `CO2StorageSurrogate` | Structural, residual, and solubility trapping fractions; storage efficiency. | **MEDIUM** |
| `feature_transformer.py` | 165 | `FeatureTransformer` | Feature scaling and dimensional transformation for ML surrogates (experimental). | **LOW** |
| `model_factory.py` | 140 | `SurrogateModelFactory` | Factory for selecting between analytical, polynomial, and RBF surrogate models. | **LOW** |
| `training_data.py` | 321 | `TrainingDataGenerator` | Generates Latin Hypercube samples for surrogate surface fitting. | **LOW** |

---

## 3. Deep-Dive: `surrogate_engine.py`

### Key Responsibilities
1. **Scenario Evaluation**: Entry point `evaluate_scenario(params: Dict[str, Any]) -> SimulationResults`.
2. **Coupled IPR Deliverability & Material Balance**:
   Computes time-dependent reservoir pressure $P(t)$ through an explicit material balance equation damped by effective well deliverability:
   $$dP = \frac{(q_{\text{inj,actual}} - q_{\text{prod,actual}}) \cdot \Delta t}{V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t}$$
   - Converts gas injection rate from MSCF/d to RB/d using dynamic $B_g(P, T)$.
   - Bounds production by sandface drawdown: $q_{\text{prod,actual}} = \min(q_{\text{target}}, J_{\text{prod}} \cdot (P - P_{\text{min}}))$.
   - Bounds injection by EPA Class VI caprock fracture limit: $q_{\text{inj,actual}} = \min(q_{\text{target}}, J_{\text{inj}} \cdot (0.90 P_{\text{frac}} - P))$.
3. **Closed-Loop CO₂ Accounting**:
   - `_calculate_co2_purchased_recycled()` separates gross injection into purchased (make-up) fresh CO₂ and recycled produced CO₂:
     $$\text{Gross Injected} = \text{Purchased Fresh} + \text{Recycled Produced}$$
   - Verifies that recycled gas never exceeds cumulative produced gas.
4. **Engine-Owned NPV**:
   - `_calculate_engine_npv()` computes discounted cash flow over the field lifetime:
     $$\text{NPV} = \sum_{t=1}^N \frac{R_{\text{oil}}(t) - C_{\text{CO2,fresh}}(t) - C_{\text{CO2,recycle}}(t) - C_{\text{water}}(t) - \text{OPEX}(t)}{(1 + r)^t} - \text{CAPEX}$$

### Invariants & Traps
- **Do not reinstate `solve_ivp` without damping**: Uncoupled stiff ODE solvers cause catastrophic step failure during rapid WAG valve switching.
- **EPA Class VI Shut-in**: If reservoir pressure reaches $0.90 \times P_{\text{frac}}$, injection must throttle to zero.

---

## 4. Deep-Dive: `analytical_models.py`

### Key Responsibilities
1. **Heterogeneity Factor ($H_k$)**:
   Modifies Dykstra-Parsons coefficient $V_{DP}$ with calibrated transverse mixing multiplier $C_{\text{trans}} = 0.80$:
   $$H_k = \frac{1}{(1 - 0.80 \cdot V_{DP})^2}$$
2. **Effective Mobility Ratio ($M_e$)**:
   Evaluates Todd-Longstaff effective mobility with mixing parameter $\omega_{\text{TL}} = 0.67$:
   $$M_e = \left[ \omega_{\text{TL}} \cdot M^{1/4} + (1 - \omega_{\text{TL}}) \right]^4$$
3. **Craig Areal Sweep Efficiency ($E_A$)**:
   Computes breakthrough areal sweep as a function of $M_e$:
   $$E_A(M_e) = 0.546 + \frac{0.059}{M_e + 0.1} + 0.05 \cdot \log_{10}(M_e)$$
4. **Ultimate Recovery Factor ($RF$)**:
   $$RF = E_A \cdot E_V \cdot E_D \cdot \left[ \omega \cdot RF_{\text{miscible}} + (1 - \omega) \cdot RF_{\text{immiscible}} \right]$$
   Clamped to physical range $[0.05, 0.80]$.

---

## 5. Deep-Dive: `profile_generator_fast.py`

### Key Responsibilities
1. **Total Recoverable Oil**: Computes $N_p = OOIP \times RF$.
2. **Mass-Conserving WAG Mobility Buffering**:
   Instead of crude empirical boosts (+8% / -4%), models phase mobility contrast:
   $$\Delta \lambda / \Sigma \lambda = \frac{\lambda_g - \lambda_w}{\lambda_g + \lambda_w}$$
   Profiles are normalized over the run duration so that cumulative oil exactly equals $N_p$.
3. **Composite Vogel-Darcy IPR**:
   Evaluates deliverability with Darcy linear flow for $P_{wf} \ge P_b$ and Vogel quadratic flow for $P_{wf} < P_b$:
   $$q_o = q_{\text{bubble}} + (q_{o,\text{max}} - q_{\text{bubble}}) \left[ 1 - 0.2 \left(\frac{P_{wf}}{P_b}\right) - 0.8 \left(\frac{P_{wf}}{P_b}\right)^2 \right]$$
