# Objectives & Data Models (`core/objectives/` & `core/data_models.py`)

## 1. Objectives Subsystem (`core/objectives/`)

The objectives subsystem evaluates economic return, resource recovery, and environmental safety metrics, applying mathematical constraint penalties before returning fitness values to the optimization engine.

### Module Inventory
| Module | LOC | Primary Classes | Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `wrapper.py` | 215 | `ObjectiveFunctions` | Central dispatcher for multi-objective optimization; computes weighted sum or Pareto vector; applies geomechanical penalties. | **HIGH** |
| `economic.py` | 149 | `EconomicObjective` | Evaluates project Net Present Value (NPV), internal rate of return (IRR), and payout time. | **MEDIUM** |
| `production.py` | 96 | `ProductionObjective` | Evaluates ultimate oil recovery factor (RF) and cumulative hydrocarbon production. | **MEDIUM** |
| `storage.py` | 241 | `StorageObjective` | Evaluates net CO₂ storage volume, storage efficiency, and trapping mechanism breakdown. | **MEDIUM** |
| `base.py` | 35 | `BaseObjective` | Abstract base class enforcing interface for objective calculators. | **LOW** |

### Penalty Enforcement Formulation
The objective wrapper enforces regulatory and physical limits using continuous quadratic barrier penalties:
$$\text{Fitness} = w_1 \cdot \text{NPV} + w_2 \cdot \text{RF} + w_3 \cdot \text{Storage} - P_{\text{frac\_penalty}} - P_{\text{env\_penalty}}$$
Where:
- **Geomechanical Penalty**:
  If peak sandface pressure $P_{\text{sandface}} > 0.90 \times P_{\text{frac}}$:
  $$P_{\text{frac\_penalty}} = 10^6 \cdot \left( \frac{P_{\text{sandface}} - 0.90 P_{\text{frac}}}{P_{\text{frac}}} \right)^2$$
- **Environmental Leakage Penalty**:
  If surface casing pressure or fault reactivation risk exceeds threshold:
  $$P_{\text{env\_penalty}} = 10^5 \cdot \text{RiskFactor}$$

---

## 2. Core Data Models (`core/data_models.py`)

`core/data_models.py` defines the canonical dataclasses used across the entire application.

### Key Dataclasses

1. **`ReservoirData`**:
   - `area`: Reservoir area (acres)
   - `thickness`: Net pay thickness (ft)
   - `porosity`: Average porosity (fraction, $0 < \phi < 1$)
   - `permeability`: Average permeability (mD)
   - `initial_pressure`: Discovery pressure (psia)
   - `temperature`: Reservoir temperature (°F)
   - `initial_oil_saturation`: $S_{oi}$ (fraction)
   - `initial_water_saturation`: $S_{wi}$ (fraction)
   - `ooip`: Original Oil in Place (STB, calculated or override)
   - `fracture_pressure`: Formation parting pressure (psia)

2. **`EORParameters`**:
   - `wag_ratio`: Water-alternating-gas volume ratio ($V_w / V_g$)
   - `cycle_time`: Total duration of one WAG cycle (days)
   - `injection_rate`: Total field injection rate (MSCF/d)
   - `gas_fraction`: Volumetric gas fraction during WAG injection
   - `miscibility_type`: Enum (`MISCIBLE`, `NEAR_MISCIBLE`, `IMMISCIBLE`)

3. **`OperationalParameters`**:
   - `num_injectors`: Integer active injection well count
   - `num_producers`: Integer active production well count
   - `min_bottomhole_pressure`: Producer drawdown limit (psia)
   - `max_injection_pressure`: Injector sandface caprock limit (psia)
   - `project_lifetime`: Field simulation duration (years, default 20)

4. **`EconomicParameters`**:
   - `oil_price`: $/STB (default $75.00)
   - `co2_purchase_cost`: $/MSCF fresh CO₂ (default $2.50)
   - `co2_recycle_cost`: $/MSCF recycled CO₂ (default $0.75)
   - `water_handling_cost`: $/bbl produced water (default $0.50)
   - `discount_rate`: Annual discount fraction (default 0.10)
   - `capex_per_well`: Capital drilling cost per well (default $5,000,000)
