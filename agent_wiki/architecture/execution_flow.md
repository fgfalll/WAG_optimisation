# Runtime Execution Flow

## 1. Application Startup Flow

The execution begins at `main.py`:

```mermaid
graph TD
    M[main.py: main()] --> LOG[utils/multiprocess_logging.py: configure_logging()]
    LOG --> CFG[utils/config_manager.py: ConfigManager.get_instance()]
    CFG --> PREF[utils/preferences_manager.py: PreferencesManager]
    PREF --> I18N[utils/i18n_manager.py: I18nManager]
    I18N --> APP[QApplication instance]
    APP --> WIN[ui/main_window.py: MainWindow.__init__()]
    WIN --> DATA[Initialize Reservoir, EOR, Operational Datamodels]
    WIN --> WIDG[Load OptimizationWidget, DataManagementWidget, etc.]
    WIN --> SHOW[window.show()]
    SHOW --> LOOP[app.exec()]
```

---

## 2. Interactive Scenario & Optimization Evaluation Flow

When a user clicks **"Run Simulation"** or when the metaheuristic optimizer evaluates a candidate vector:

### Step 1: Input Collection and Validation
1. `ui/optimization_widget.py` / `ui/data_management_widget.py` collects values from UI input widgets.
2. Constructs typed dataclass instances:
   - `ReservoirData` (dimensions, OOIP, porosity, permeability, thickness, initial pressure, temperature)
   - `EORParameters` (injection scheme, target pressure, injection rate, WAG ratio, cycle times, MMP)
   - `OperationalParameters` (simulation lifetime, time resolution, well count)
   - `EconomicParameters` (oil price, fresh CO₂ cost, recycling cost, OPEX, discount rate)
3. Validates dimensional ranges using `ReservoirData.validate(physics_based_model=True)`.

### Step 2: Handoff to Optimization Engine
1. Method: `OptimizationEngine.evaluate_candidate(x)` or `evaluate_for_analysis(params_dict, **overrides)` in `core/optimisation_engine.py`.
2. Computes active well counts: counts injectors ($N_{\text{inj}}$) and producers ($N_{\text{prod}}$) from `well_data_list`.
   - If well list is empty, assigns 1 field-wide pattern injector and 1 field-wide pattern producer.
3. Sets simulation mode:
   - $N_{\text{inj}} > 0 \land N_{\text{prod}} > 0 \implies \text{"co2\_eor"}$
   - $N_{\text{inj}} == 0 \land N_{\text{prod}} > 0 \implies \text{"primary\_production"}$
   - $N_{\text{inj}} > 0 \land N_{\text{prod}} == 0 \implies \text{"injection\_storage"}$

### Step 3: Engine Factory Dispatch
1. Calls `EngineFactory.create_engine(EngineType.SURROGATE, recovery_model_type=...)` in `core/engine_factory.py`.
2. Engine factory unconditionally instantiates `SurrogateEngineWrapper`, logging:
   `"EngineFactory: Routing all simulation requests to Surrogate Engine."`

### Step 4: Analytical Recovery Factor Calculation
1. Calls `SurrogateEngine.evaluate_scenario()` in `core/engine_surrogate/surrogate_engine.py`.
2. MMP Estimation: `evaluation/mmp.py:calculate_mmp()` computes minimum miscibility pressure using Cronquist correlation.
3. Delegates to `PhDHybridRecoveryModel.calculate_recovery()` in `core/engine_surrogate/analytical_models.py`:
   - Koval heterogeneity factor: $H_k = 1 / (1 - V_{DP})^2$.
   - Todd-Longstaff effective mobility ratio: $M_e = (k_{rg}^0 / \mu_{g,\text{eff}}) / (k_{ro}^0 / \mu_{o,\text{eff}})$.
   - Craig areal sweep efficiency: $E_A = f(M_e, V_{\text{inj}})$.
   - Smooth miscibility weight: $\omega = \tanh((P_{\text{avg}} - MMP) / (0.15 \cdot MMP))$.
   - Returns scalar `recovery_factor` ($RF \in [0.05, 0.80]$).

### Step 5: Dynamic Profile Synthesis
1. Calls `FastProfileGenerator.generate_profile()` in `core/engine_surrogate/profile_generator_fast.py`.
2. Computes total recoverable oil: $N_p = OOIP \times RF$.
3. Singles-well deliverability: `calculate_composite_ipr_deliverability` evaluates Composite Vogel-Darcy IPR across pressure regimes.
4. Shapes production into plateau + exponential/hyperbolic decline.
5. Applies mass-conserving WAG phase mobility buffering:
   $$\text{amp} = \text{clip}\left(0.1 \times \frac{\lambda_g - \lambda_w}{\lambda_o + \lambda_g + \lambda_w}, -0.15, 0.15\right)$$
   Strictly re-normalizes profile arrays so cumulative volume matches $N_p$.

### Step 6: Pressure Evolution via Coupled IPR & Material Balance
1. Inside `SurrogateEngine._calculate_pressure_profile()`:
   - Gas injection converted via dynamic $B_{g,\text{dynamic}}$ to RB/day.
   - Dynamic Koval fractional flow $f_g(t_D)$ tracks voidage replacement.
   - Producer deliverability: Darcy/Vogel drawdown ($J_{\text{prod}} \times (P - P_{\text{min}})$).
   - Injector deliverability: sandface injection capped at EPA Class VI UIC ceiling ($0.90 \times P_{\text{frac}}$).
   - Damped continuous material balance derivative:
     $$dP = \frac{(q_{\text{inj,actual}} - q_{\text{prod,actual}}) \cdot \Delta t}{V_p \cdot c_t + J_{\text{eff}} \cdot \Delta t}$$
   - Single-step pressure derivative bounded to $\pm 450\text{ psi/step}$.

### Step 7: Engine-Owned NPV & Carbon Balance Accounting
1. Inside `SurrogateEngine`:
   - `_calculate_co2_purchased_recycled()` computes fresh vs recycled gas streams.
   - `_calculate_engine_npv()` discounts project cash flows (oil revenues minus CAPEX, OPEX, fresh gas purchases, recycling compression, and water disposal).
   - Returns complete `SimulationResults` dictionary to `SurrogateEngineWrapper`.

### Step 8: Multi-Objective Fitness & Penalty Evaluation
1. `core/objectives/wrapper.py:ObjectiveFunctions._calculate_objective_functions()` consumes engine results.
2. Checks for geomechanical overpressure against EPA Class VI 90% limit:
   - Overpressure penalty: $10^6 \times (\Delta P / P_{\text{safe limit}})^2$ deducted from NPV.
3. Computes environmental leakage remediation penalty ($100/tonne if CO₂ leaks).
4. If candidate violates physical constraints, assigns `FAILURE_PENALTY` ($-10^{12}$) or `NaN`.
5. Returns evaluated fitness score to the optimizer.
