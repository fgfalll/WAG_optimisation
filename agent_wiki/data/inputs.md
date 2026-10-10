# Data Inputs & Scenario Specifications

## 1. Input Modalities

The application accepts simulation and optimization inputs through four primary channels:
1. **JSON Configuration Files**: Base parameters in `config/base_config.json`, demo cases in `config/demo_data.json`.
2. **PyQt6 GUI Data Management Widget**: Direct user entry via `ui/data_management_widget.py`.
3. **LAS Well Log Files**: Parsed by `utils/las_parser.py` to extract depth, porosity, and permeability tracks.
4. **Project Files (`.tphd`)**: Serialized JSON project archives loaded and saved via `utils/project_file_handler.py`.

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
 
---
 
## 4. Project File Schema (`.tphd`) & State Persistence
 
The primary format for saving and exchanging full application state is the `.tphd` file (JSON serialized via `utils/project_file_handler.py`).
 
### Top-Level Schema
```json
{
  "schema_version": "1.1",
  "application_version": "0.8.5-alpha",
  "project_name": "Project-Name",
  "reservoir_data": { ... },
  "pvt_properties": { ... },
  "well_data_list": [ ... ],
  "mmp_value": 2053.65,
  "economic_parameters": { ... },
  "eor_parameters": { ... },
  "operational_parameters": { ... },
  "profile_parameters": { ... },
  "ga_parameters": { ... },
  "bo_parameters": { ... },
  "manual_inputs": { ... },
  "uq_parameters": [ ... ],
  "uq_results": { ... },
  "optimization_results": { ... },
  "ui_state": { ... }
}
```
 
### Serialization & Ingestion Protocols
1. **Shallow Dataclass Fields**: `ProjectEncoder` iterates over `dataclasses.fields(o)` and extracts fields without recursively calling `asdict()`, ensuring nested dataclasses retain their `_dataclass` type tag.
2. **Backwards-Compatible Decoding**: `project_decoder` automatically detects legacy untyped dictionary forms of nested structures (`EOSModelParameters`, `LayerDefinition`, `GeostatisticalParams`) in older projects and converts them into proper dataclass instances.
3. **Resilient Grid Loading**: `DataManagementWidget.load_project_data()` accommodates 1D flattened grids from `DataIntegrationEngine` as well as scalar and 3D arrays using `.flat[0]`.
4. **Active UI Flush**: `MainWindow._perform_project_save()` queries `DataManagementWidget.get_current_project_data()` to ensure manual overrides and newly configured inputs are flushed from UI widgets into serialized models.
5. **Validation Test Suite**: Full roundtrip serialization and backwards compatibility are verified by `tests/test_project_save_load.py`.

