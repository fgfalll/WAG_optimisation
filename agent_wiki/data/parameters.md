# Parameters & Dataclasses Reference

## 1. Overview of Core Parameter Dataclasses

All data schemas are defined as `@dataclass` structures in [core/data_models.py](file:///d:/rep/4.6/co2eor_optimizer/core/data_models.py).

---

## 2. Parameter Class Definitions

### A. `ReservoirData` (Line 224)
Represents the static and dynamic reservoir properties, geometry, and PVT table bindings:

| Attribute | Type | Default | Valid Range | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `ooip_stb` | `float` | 1,000,000.0 | $> 0$ | Original Oil in Place (Stock Tank Barrels) |
| `initial_pressure` | `float` | 4,000.0 | $> 0$ | Initial average reservoir pressure (psia) |
| `rock_compressibility` | `float` | $3 \times 10^{-6}$ | $> 0$ | Formation rock compressibility ($\text{psi}^{-1}$) |
| `temperature` | `float` | 150.0 | 50 – 400 | Reservoir temperature (°F) |
| `area_acres` | `Optional[float]`| `None` | $> 0$ | Drainage area in acres ($43,560\text{ ft}^2/\text{acre}$) |
| `thickness_ft` | `Optional[float]`| `None` | $> 0$ | Net pay reservoir thickness (ft) |
| `length_ft` | `Optional[float]`| 2,000.0 | $> 0$ | Reservoir inter-well displacement distance (ft) |
| `cross_sectional_area_acres` | `Optional[float]`| 10.0 | $> 0$ | Pattern cross-sectional area (acres) |
| `density_contrast` | `Optional[float]`| 0.3 | 0.05 – 0.8 | Oil-gas density contrast ($\text{g/cm}^3$) for gravity segregation |
| `interfacial_tension` | `Optional[float]`| 5.0 | 0.01 – 50.0 | Reference oil-gas IFT (dynes/cm) |
| `average_porosity` | `Optional[float]`| `None` | 0.01 – 0.45 | Bulk average reservoir porosity (fraction) |
| `average_permeability` | `Optional[float]`| `None` | $> 0$ | Bulk average absolute permeability (mD) |
| `initial_water_saturation` | `Optional[float]`| `None` | 0.05 – 0.9 | Initial connate water saturation $S_{wi}$ (fraction) |
| `oil_fvf` | `Optional[float]`| `None` | 1.0 – 2.5 | Oil formation volume factor $B_o$ (RB/STB) |

### B. `EORParameters` (Line 690)
Controls injection schemes, rates, pressures, and fluid displacement settings:

| Attribute | Type | Default | Valid Range | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `target_pressure_psi` | `float` | 3,000.0 | 500 – 10,000 | Injection operating pressure |
| `max_pressure_psi` | `float` | 4,500.0 | 1,000 – 15,000 | Maximum pressure constraint (fracture limit) |
| `injection_rate` | `float` | 1,000.0 | 10 – 100,000 | Gas injection rate (MSCFD) |
| `injection_scheme` | `str` | `"continuous"` | `continuous, wag, huff_n_puff, swag, tapered, pulsed, storage` | Active injection strategy |
| `mmp` | `float` | 2,500.0 | 800 – 6,000 | Minimum miscibility pressure (psia) |
| `wag_ratio` | `float` | 1.0 | 0.1 – 10.0 | Water-to-Gas volume ratio for WAG cycles |
| `wag_cycle_days` | `float` | 90.0 | 10 – 720 | Duration of one full WAG injection cycle |
| `co2_density_tonne_per_mscf` | `float` | 0.053 | 0.045 – 0.060 | Conversion factor from MSCF to metric tonnes |
| `enforce_step_flash` | `bool` | `False` | `True, False` | If True, evaluates EOS flash at every timestep; if False, propagates project PVT baseline |

### C. `OperationalParameters` (Line 1161)
Defines project operational horizon, time resolution, and analytical recovery model selection:

| Attribute | Type | Default | Valid Choices / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `project_lifetime_years` | `int` | 15 | 1 – 100 | Total project operational timeline |
| `time_resolution` | `str` | `"yearly"` | `"weekly"`, `"monthly"`, `"quarterly"`, `"yearly"` | Time step resolution for profiles |
| `recovery_model_selection` | `str` | `"hybrid"` | `"simple"`, `"miscible"`, `"immiscible"`, `"hybrid"`, `"phd_hybrid"`, `"koval"`, `"layered"`, `"co2_specific"` | Analytical displacement physics model (note: `"phd_hybrid"` alias normalizes to `"hybrid"`) |
| `co2_breakthrough_year_fraction` | `float` | 0.25 | 0.01 – 1.0 | Target breakthrough timeline constraint fraction |
| `target_objective_name` | `Optional[str]` | `None` | `"npv"`, `"recovery_factor"`, `"co2_utilization"`, etc. | Target optimization metric |
| `target_objective_value` | `Optional[float]` | `None` | `float` | Bound or target objective value |
| `target_tolerance` | `float` | 0.05 | $> 0$ | Convergence or constraint tolerance |

*(Note: Well counts are configured dynamically via `WellSetup` and project well dictionaries, not in `OperationalParameters`.)*

### D. `EconomicParameters` (Line 572)
Defines capital, operating, commodity, and environmental costs for NPV calculation:

| Attribute | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `oil_price_usd_per_bbl` | `float` | 70.0 | USD/bbl | Crude oil sale price |
| `co2_purchase_cost_usd_per_tonne` | `float` | 40.0 | USD/tonne | Cost to purchase fresh capture CO₂ |
| `co2_recycle_cost_usd_per_tonne` | `float` | 15.0 | USD/tonne | Cost to separate, compress, and reinject CO₂ |
| `water_disposal_cost_usd_per_bbl` | `float` | 2.0 | USD/bbl | Water handling and disposal OPEX |
| `discount_rate` | `float` | 0.10 | fraction | Annual discount rate for cash flow |
| `carbon_credit_usd_per_ton` | `float` | 0.0 | USD/tonne | Tax credit (e.g. 45Q) for sequestered CO₂ |

### E. `EmpiricalFittingParameters` (Line 2241)
Calibration factors for the surrogate engine. Note that `breakthrough_time_years` was **removed** because breakthrough is calculated directly from Koval (1963) physics in `surrogate_engine._build_params_dict()`:

| Attribute | Type | Default | Tuned Range | Physical Role |
| :--- | :--- | :--- | :--- | :--- |
| `c7_plus_fraction` | `float` | 0.57 | 0.10 – 0.80 | Heavy hydrocarbon fraction shifting miscibility transition |
| `alpha_base` | `float` | 1.0 | 0.5 – 2.0 | Miscibility transition midpoint parameter |
| `miscibility_window` | `float` | 0.011 | 0.005 – 0.05 | Beta value controlling transition sharpness around MMP |
| `trapping_efficiency` | `float` | 0.40 | 0.05 – 0.60 | Capillary trapping fraction for injected CO₂ |
| `recycle_growth_rate` | `float` | 1.5 | 0.5 – 5.0 | Rate of recycled gas ramp-up post-breakthrough ($1/\text{year}$) |
| `initial_gor_scf_per_stb` | `float` | 500.0 | 50 – 5,000 | Initial produced Gas-Oil Ratio |
| `transverse_mixing_calibration` | `float` | 0.5 | 0.1 – 1.0 | Scaling factor for $V_{DP}$ heterogeneity in Koval formula |
| `omega_tl` | `float` | 0.6 | 0.33 – 1.0 | Todd-Longstaff partial miscibility parameter |
| `k_ro_0` | `float` | 0.8 | 0.60 – 1.0 | Endpoint relative permeability to oil |
| `k_rg_0` | `float` | 1.0 | 0.60 – 1.0 | Endpoint relative permeability to gas |
| `n_o` | `float` | 2.0 | 1.5 – 4.0 | Corey exponent for oil |
| `n_g` | `float` | 2.0 | 1.5 – 3.5 | Corey exponent for gas |

