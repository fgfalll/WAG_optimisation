# Simulation Outputs & Objective Metrics

## 1. Engine Results Dictionary

When `SurrogateEngine.evaluate_scenario()` completes, it returns a structured Python dictionary containing scalar metrics and dynamic profile arrays:

| Output Key | Type | Units | Physical Interpretation |
| :--- | :--- | :--- | :--- |
| `recovery_factor` | `float` | fraction (0–1) | Ultimate fraction of OOIP recovered by the end of the project |
| `cumulative_oil` | `float` | STB | Cumulative oil produced over project lifetime ($N_p = \text{OOIP} \cdot RF$) |
| `npv` | `float` | USD | Discounted net present value of cash flows |
| `co2_stored` | `float` | tonnes | Total CO₂ mass trapped in reservoir at end of simulation |
| `storage_efficiency` | `float` | fraction (0–1) | Ratio of stored CO₂ mass to total purchased CO₂ mass |
| `annual_co2_purchased_mscf`| `np.ndarray` | MSCF/yr | Annual time series of fresh CO₂ purchased |
| `annual_co2_recycled_mscf` | `np.ndarray` | MSCF/yr | Annual time series of breakthrough CO₂ captured and reinjected |
| `structural_trapping_efficiency` | `float` | fraction | Portion of storage from caprock/structural containment |
| `residual_trapping_efficiency`   | `float` | fraction | Portion of storage from capillary immobilization |
| `solubility_trapping_efficiency` | `float` | fraction | Portion of storage from dissolution into brine/oil |
| `mineral_trapping_efficiency`    | `float` | fraction | Portion of storage from geochemical mineral precipitation |
| `profiles` | `dict` | mixed | Nested dictionary containing time series curves |

---

## 2. Dynamic Profiles Dictionary Schema (`results["profiles"]` / `results["optimized_profiles"]`)

Simulation profiles are standardized into **4 primary fluid streams** and coupled geomechanical/thermodynamic state tracks:

### Stream 1: Crude Oil (Surface & Reservoir Volumetrics)
| Profile Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `oil_profile` / `oil_production_rate` | `np.ndarray` | STB/day | Stock tank oil production rate |
| `cumulative_oil_bbl` | `np.ndarray` | STB | Cumulative oil production time series |
| `annual_oil_stb` / `yearly_oil_stb` | `np.ndarray` | STB/yr | Annual calendar year oil production |
| `monthly_oil_stb` | `np.ndarray` | STB/month | Monthly aggregated oil production |
| `oil_fvf_profile` | `np.ndarray` | RB/STB | In-situ swollen oil formation volume factor $B_o(P, x_{\text{CO2}})$ |

### Stream 2: Natural Gas (Hydrocarbon Sales Gas)
| Profile Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `hydrocarbon_gas_sales_mscfd` | `np.ndarray` | MSCFD | Separated hydrocarbon sales gas (methane/ethane, pure from $\text{CO}_2$) |
| `solution_gas_profile` | `np.ndarray` | MSCFD | Degassed native solution gas from live oil |
| `annual_hydrocarbon_gas_sales_mscf` | `np.ndarray` | MSCF/yr | Annual hydrocarbon sales gas volume |
| `cumulative_hydrocarbon_gas_sales_mscf` | `np.ndarray` | MSCF | Cumulative sales gas production |
| `gas_profile` / `total_gas_production_rate` | `np.ndarray` | MSCFD | Total raw wet surface gas rate ($\text{CO}_2 + \text{Hydrocarbon}$) |

### Stream 3: Water (Formation Brine & Injected Water)
| Profile Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `water_profile` / `water_production_rate` | `np.ndarray` | bbl/day | Surface brine water production rate |
| `water_cut_profile` / `water_cut` | `np.ndarray` | fraction | Instantaneous produced water cut $q_w / (q_o + q_w)$ |
| `cumulative_water_bbl` | `np.ndarray` | bbl | Cumulative produced brine water volume |
| `annual_water_bbl` | `np.ndarray` | bbl/yr | Annual produced water volume |

### Stream 4: Injection Agent ($\text{CO}_2$ & WAG Water)
| Profile Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `injection_profile` / `co2_injection` | `np.ndarray` | MSCFD | Gross downhole $\text{CO}_2$ injection rate |
| `co2_purchased_mscfd` | `np.ndarray` | MSCFD | Fresh purchased commercial $\text{CO}_2$ injection rate |
| `co2_recycled_mscfd` | `np.ndarray` | MSCFD | Captured and reinjected recycle $\text{CO}_2$ rate |
| `co2_gas_profile` / `co2_production_rate` | `np.ndarray` | MSCFD | Total surface $\text{CO}_2$ production (breakthrough + degassed) |
| `water_injection_profile` | `np.ndarray` | bbl/day | Injected water rate for WAG/SWAG schemes |
| `annual_co2_purchased_mscf` | `np.ndarray` | MSCF/yr | Annual fresh $\text{CO}_2$ purchase volume |
| `annual_co2_recycled_mscf` | `np.ndarray` | MSCF/yr | Annual reinjected $\text{CO}_2$ volume |
| `annual_co2_injected_mscf` | `np.ndarray` | MSCF/yr | Annual gross $\text{CO}_2$ injection volume |

### State & Integrity Tracks (Daily Resolution)
| Profile Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `pressure` / `reservoir_pressure` | `np.ndarray` | psia | Average reservoir pore pressure |
| `sandface_injection_pressure` | `np.ndarray` | psia | Sandface bottomhole injection pressure ($P_{\text{res}} + \Delta P_{\text{skin}}$) |
| `vrr_local` | `np.ndarray` | ratio | Dual-pressure Voidage Replacement Ratio ($q_{\text{inj,res}} / q_{\text{prod,res}}$) |
| `fault_slip_tendency` | `np.ndarray` | ratio | Mohr-Coulomb slip tendency $T_s = \tau / \sigma_n'$ |
| `caprock_tensile_margin` | `np.ndarray` | psi | Safety margin to caprock tensile fracturing |
| `caprock_shear_margin` | `np.ndarray` | psi | Safety margin to caprock shear slip |
| `leakage_rate_tonnes_day` | `np.ndarray` | tonnes/d | Subsurface geological leakage rate through compromised caprock/fault |
| `x_co2_liquid` | `np.ndarray` | mole frac | Liquid-phase dissolved $\text{CO}_2$ concentration |
| `y_co2_vapor` | `np.ndarray` | mole frac | Vapor-phase $\text{CO}_2$ purity |
| `saturation_oil`, `saturation_water`, `saturation_gas` | `np.ndarray` | frac | Dynamic reservoir phase saturations ($S_o + S_w + S_g = 1$) |



---

## 3. Objective Function Metrics (`core/objectives/wrapper.py`)

Optimization algorithms evaluate candidate solutions using the following normalized objectives:

1. **Recovery Factor Objective**: Maximized ($f_1 = -RF$).
2. **Economic Objective**: Maximized ($f_2 = -NPV$).
3. **CO₂ Storage Objective**: Maximized ($f_3 = -\text{Storage Efficiency}$). True physical storage is evaluated; missing profile data evaluates to `float("nan")` without artificial Class E synthesis.
4. **CO₂ Utilization Objective**: Minimized ($f_4 = \text{Tonnes CO}_2 \text{ purchased} / \text{BBL oil produced}$). Missing or empty profiles evaluate to `float("nan")` rather than arbitrary magic numbers.
5. **Containment Safety & Constraint Enforcement**: Geomechanical fracture margin or sanity check failures trigger immediate candidate pruning via `FAILURE_PENALTY` ($-10^{12}$) without dilution multipliers (`* 0.1`, `* 0.8`), ensuring that unviable genetic lines are eliminated by the solver.
