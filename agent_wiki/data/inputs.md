# Data Inputs & Scenario Specifications

## 1. Input Modalities

The application accepts simulation and optimization inputs through four primary channels:
1. **JSON Configuration Files**: Base parameters in `config/base_config.json`, demo cases in `config/demo_data.json`.
2. **PyQt6 GUI Data Management Widget**: Direct user entry via `ui/data_management_widget.py`.
3. **LAS Well Log Files**: Parsed by `utils/las_parser.py` to extract depth, porosity, and permeability tracks.
4. **Project Files (`.co2eor`)**: Serialized JSON/zipped project archives loaded via `utils/project_file_handler.py`.

---

## 2. Configuration File Schemas

### `config/base_config.json`
Central default configuration containing standard reservoir properties, default economic parameters, and optimization bounds:

```json
{
  "reservoir": {
    "initial_pressure": 3000.0,
    "temperature": 160.0,
    "permeability": 100.0,
    "porosity": 0.20,
    "thickness": 50.0,
    "area": 640.0,
    "connate_water_saturation": 0.25,
    "residual_oil_saturation": 0.25,
    "rock_compressibility": 4.0e-6
  },
  "eor": {
    "target_pressure_psi": 3200.0,
    "max_pressure_psi": 4500.0,
    "injection_rate": 2000.0,
    "injection_scheme": "continuous",
    "mmp": 2500.0
  },
  "operational": {
    "project_lifetime_years": 20,
    "time_resolution": "monthly"
  },
  "economic": {
    "oil_price_usd_per_bbl": 70.0,
    "co2_purchase_cost_usd_per_tonne": 40.0,
    "co2_recycle_cost_usd_per_tonne": 15.0,
    "water_disposal_cost_usd_per_bbl": 2.0,
    "discount_rate": 0.10
  }
}
```

---

## 3. Well Schedule & Operations Schema

Wells are parameterized as a list of `WellData` objects:
- `name`: String identifier (e.g., `"INJ-01"`, `"PROD-01"`).
- `metadata["type"]`: `"injector"` or `"producer"`.
- `operational_schedule`: List of `WellScheduleEntry` objects specifying rates, pressures, and status changes by day.

### Simulation Mode Detection Logic
In `core/optimisation_engine.py:902`:
- If `n_injectors > 0` and `n_producers > 0` $\implies$ `"co2_eor"`.
- If `n_injectors == 0` and `n_producers > 0` $\implies$ `"primary_production"`.
- If `n_injectors > 0` and `n_producers == 0` $\implies$ `"injection_storage"`.
- If `n_injectors == 0` and `n_producers == 0` $\implies$ Raises `OptimizationError("No wells configured")`.
