# Data Model & Value Pool Gap

**The owner's assessment, restated 07-10-2026:** current data models will not be sufficient for
full-simulation tasks, and the value pool needs updating. This page substantiates both with file/line
evidence, so the gap is scoped before P2 begins rather than discovered during it.

**Measured 07-10-2026.** All citations verified against the working tree.

---

## 1. The two artefacts

| Artefact | Location | Size | Role |
|---|---|---|---|
| **Data models** | `core/data_models.py` | ~2 300 lines, **33 dataclasses** | All typed inputs for the surrogate |
| **Value pool** | `config/base_config.json` | 26 206 bytes | Defaults, ranges and metadata consumed by GUI + engine |

Supporting: `config/fluid_composition.json` (259 B), `config/fault_properties.json`,
`config/recovery_config.json`, `config/economic_scenarios.json`, `config/uq_and_sensitivity.json`.

> ⚠️ **Stale wiki reference.** `agent_wiki/development/extension_points.md:34` instructs agents to
> "update config schema in **`config/default_config.json`**". That file **does not exist**
> (`Test-Path` → `False`). The real value pool is `config/base_config.json`. Same class of defect as
> **MED-16**. See §6.

---

## 2. The grid is the biggest single gap

`core/data_models.py:415-447` — `ReservoirData`:

```python
@dataclasses.dataclass
class ReservoirData:
    grid: Dict[str, np.ndarray]              # :417
    pvt_tables: Dict[str, np.ndarray]        # :418
    regions: Optional[Dict[str, np.ndarray]] # :419
    runspec: Optional[Dict[str, Any]]         # :420
    faults: Optional[Dict[str, Any]]         # :421
    ooip_stb: float = 1_000_000.0            # :422
    initial_pressure: float = 4000.0         # :423
    rock_compressibility: float = 3e-6       # :424
    temperature: float = 150.0               # :425
    length_ft: Optional[float] = 2000.0      # :426
    cross_sectional_area_acres: Optional[float] = 10.0   # :427
    area_acres: Optional[float] = None       # :428
    thickness_ft: Optional[float] = None     # :429
    average_permeability: Optional[float] = None          # :432
    average_porosity: Optional[float] = None              # :431
    initial_water_saturation: Optional[float] = None     # :430
```

**`grid` is a `Dict[str, np.ndarray]` whose only content is dimensions.** From the live test fixture at
`tests/scientific/conftest.py:19`:

```python
grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])}
```

### 2.1 What exists vs. what a full simulation requires

| Requirement | Present? | Evidence |
|---|---|---|
| Grid dimensions $N_x, N_y, N_z | ✅ | `grid["NX"/"NY"/"NZ"]` |
| Cell sizes $\Delta x, \Delta y, \Delta z$ as arrays | ❌ | no such key |
| **Per-cell porosity / permeability arrays** | ❌ | `average_porosity`, `average_permeability` are **scalars** (`:431-432`) |
| **NTG (net-to-gross)** | ❌ | absent repo-wide — see `../thmc/satellite_toolkit.md` **CONF-42** |
| Cell volumes, face areas | ❌ | absent |
| **Transmissibility array** | ❌ | absent |
| **Corner-point geometry** (8 vertices/cell) | ❌ | absent |
| `ActiveMask` / inactive-cell handling | ❌ | absent |
| **Full 6-component permeability tensor** | ❌ | `permeability_multiplier` is scalar |
| Layer structure | ⚠️ partial | `LayerDefinition:408-412` = `thickness`, `porosity`, `permeability_multiplier`, `param_overrides` — a **multiplier stack**, not a real grid |
| Fault cell-pair transmissibility table | ❌ | `FaultData` (:220-242) is **descriptive**, not transmissibility-connected |

**Consequence.** A compositional engine cannot be fed this. The grid must become a first-class typed
object. Everything downstream — geology generation, initialisation, IO, the engine itself — depends on
that decision, so it is the **first P2 data-model task** and should be **specified during P1**.

### 2.2 Geology is 2D only

```python
geostatistical_grid: Optional[np.ndarray] = None    # data_models.py:443
```
with `GeostatisticalParams.grid_resolution: Tuple[int, int] = (100, 100)` (`:370`).

`grid_resolution` is a **2-tuple**. There is no third dimension, so no 3D facies realisation. The design
set's facies generator produces 3D (`../thmc/satellite_toolkit.md` §6.2), and there is **no PGS path in
the code at all**.

---

## 3. The fluid description is 2-component

`config/fluid_composition.json` — the **entire** shipped composition:

```json
{
  "eos_composition": {
    "eos_type": "PR",
    "component_properties": [
      [0.8, 44.0, 304.2, 1071.0, 0.225],
      [0.2, 16.0, 190.6, 464.0, 0.011]
    ],
    "binary_interaction_coeffs": [[0.0, 0.1], [0.1, 0.0]]
  }
}
```

**Two components** — CO₂ and C1. The `binary_interaction_coeffs` is $2\times2$, i.e. **one** unique
$k_{ij}$.

| Requirement | Present? |
|---|---|
| $N_c \ge 4$: C1, C2–C3, C4–C6, C7+, CO₂ | ❌ **2** |
| Full symmetric BIP set (**21** unique $k_{ij}$ for 6 components) | ❌ **1** |
| **Volume-translation parameters** | ❌ absent repo-wide (**CONF-10**) |
| Pseudocomponent construction (Joback-Reid / PPR78) | ❌ |
| Component molar volume, density, MW per component | partial — MW present, density derived |
| $T_c$, $P_c$, $\omega$ per component | ✅ (in those 2 rows) |

This is the direct cause of **CONF-29** (`../thmc/conflict_and_gap_register.md`): the design set's own
`PVT_EOS_DTO` example is malformed the same way.

---

## 4. The PVT tables are degenerate

`config/base_config.json` → `PVTPropertiesDefaults` — **ten keys, six of which are single-element
lists**:

| Key | Value shape |
|---|---|
| `pressure_points` | `[1]` |
| `oil_viscosity` | `[1]` |
| `co2_viscosity` | `[1]` |
| `oil_fvf` | `[1]` |
| `gas_fvf` | `[1]` |
| `rs` | `[1]` |
| `pvt_type`, `gas_specific_gravity`, `temperature`, `c7_plus_fraction` | scalars |

These are **scalar placeholders in list clothing**. A compositional engine needs real tabulated or
correlated $B_o$, $B_g$, $R_s$, $\mu_o$, $\mu_g$ versus pressure and temperature, plus a **PVT-region**
concept for variable-composition tracking. None of that exists.

`PVTProperties` exists as a dataclass (`data_models.py:1221`) but the **value pool feeding it is
degenerate**.

---

## 5. The value pool is surrogate-shaped

`config/base_config.json` — measured structure:

| Section | Keys | Assessment |
|---|---|---|
| `EORParametersDefaults` | **132** | Dominant section. Almost all **surrogate-specific**: profile damping, smoothing, `spatial_smoothing_iterations/weight`, `cfl_safety_factor`, `saturation_damping_factor`, `pressure_damping_factor`, `rate_smoothing_rate`, `flux_limiter_type`, log/fudge factors |
| `RecoveryModelParamsDefaults` | 5 models (Koval, Miscible, Immiscible, Hybrid, Layered) | Recovery-model tuning, incl. `locked_gravity_factor`, `locked_productivity_index`, `locked_transition_alpha/beta` |
| `EconomicParametersDefaults` | 11 | See §7 |
| `SurrogateEngine` | 8 subsections | `response_surface`, `profile_generator`, `performance`, `parameter_ranges` (8) |
| `EmpiricalFittingParameters` | 13 × `{default,min,max,description,unit}` | Well-formed metadata pattern |
| `PVTPropertiesDefaults`, `MMPCalculation`, `FaultMechanics`, `Parsers.LASParser`, `UISettings`, `Logging`, `GeneralFallbacks`, `ReservoirDataDefaults`, `OperationalParametersDefaults`, `GeneticAlgorithmParamsDefaults`, `AdvancedEngineParamsDefaults`, `ProfileParametersDefaults`, `ui_config` | — | |

### 5.1 The metadata pattern is good and should be reused

`EmpiricalFittingParameters` uses `{default, min, max, description, unit}` for all 13 entries, and
`ui_config.optimization.parameter_metadata` carries **32** documented parameters. **That is the right
shape for the compositional value pool** — the new sections should follow it rather than the flat
`EORParametersDefaults` style.

### 5.2 Saturation endpoints are global scalars

`EORParametersDefaults` has `connate_water_saturation`, `residual_oil_saturation`,
`critical_gas_saturation`, `residual_gas_saturation_trapping`, `minimum_relative_permeability`,
`maximum_relative_permeability_oil/gas/water`, `endpoint_oil/gas/water_relative_permeability`.

**All single values.** Full physics needs these **per rock type / per PVT region / per facies**, driven by
`FZI` (`../thmc/satellite_toolkit.md` §3.3). `CoreyParameters` exists as a dataclass
(`data_models.py:2115`) and `EmpiricalFittingParameters` has `k_ro_0`, `k_rg_0`, `n_o`, `n_g` — but
**there is no `k_rw`/`n_w`, no capillary-pressure curve, and no hysteresis parameter** (Killough/Carlson
are absent repo-wide).

---

## 6. Wells

```python
@dataclasses.dataclass                       # data_models.py:63
class WellData:
    name: str; depths: np.ndarray; properties: Dict[str, np.ndarray]
    units: Dict[str, str]; metadata: Dict[str, Any]
    perforation_properties: List[Dict[str, float]]
    well_path: Optional[np.ndarray]; skin_factor: float = 0.0
    wellbore_radius_ft: float = 0.354; perforations: List[List[float]]
    well_index: Optional[float] = None
```

| Requirement | Present? |
|---|---|
| Name, depths, properties, well path, skin, wellbore radius, well index | ✅ |
| Perforations (flat list of lists) | ⚠️ partial — **no interval/depth structure, no phasing, no cluster/stage grouping** |
| **Completion type** (vertical/horizontal/multilateral, lower/upper) | ❌ |
| **Cement sheath** geometry and properties | ❌ |
| **ICD / AICV / ICV** | ❌ |
| Stage isolation | ❌ |
| Multi-segment wellbore | ❌ |
| `WellOperationalSchedule` (:210) with `max_operations: int = 100` | ⚠️ a **hard cap of 100 schedule entries** |

### 6.1 Two stale paths in the extension guide

`agent_wiki/development/extension_points.md` gives step-by-step instructions that **cannot be followed**:

| Line | Instruction | Reality |
|---|---|---|
| `:34` | "update config schema in `config/default_config.json`" | **does not exist** → `config/base_config.json` |
| `:108` | "Modify `core/engine_factory.py` lines 105–116" | **does not exist** |
| `:99-117` | Entire section "Registering a Real Simulation Engine with `EngineFactory`" | **No `EngineFactory` exists.** This is the exact task P2 needs |
| `:9,29,32` | `core/analytical_models.py`, `core/surrogate_engine.py` | **do not exist** → under `core/engine_surrogate/` |
| `:13` | `core/models/` | **does not exist** |

> **This is the most actionable finding in this page.** The wiki's own extension guide is unbuildable,
> and it is unbuildable *in exactly the places P2 needs*. `core/architecture/source_of_truth_map.md`
> already corrected the equivalent claim for retired engines; `extension_points.md` was not updated.

---

## 7. Economics

`EconomicParametersDefaults` — **11 keys**: `oil_price_usd_per_bbl`, `co2_purchase_cost_usd_per_tonne`,
`co2_recycle_cost_usd_per_tonne`, `water_injection_cost_usd_per_bbl`,
`water_disposal_cost_usd_per_bbl`, `discount_rate_fraction`, `capex_usd`, `fixed_opex_usd_per_year`,
`variable_opex_usd_per_bbl`, `npv_time_steps_per_year`, `carbon_tax_usd_per_tonne`.

| Requirement | Present? |
|---|---|
| Oil price, CO₂ purchase, CO₂ recycle, water inj/disposal, fixed+variable OPEX, discount rate, capex, carbon tax | ✅ |
| **Gas price** | ❌ |
| **CO₂ storage credit** | ❌ — read in code at `surrogate_engine.py:626` with a hardcoded default of `0.0`, **not in the value pool** |
| Price escalation | ❌ |
| Terminal value | ❌ |
| IRR / payback | ❌ |

This substantiates **CONF-37** and **CONF-11** with hard evidence: the **storage credit defaults to
zero** and **gas price does not exist**, which is why 216 810 MSCF of gas sales over 15 years
contribute $0 (**CRIT-18**).

> Per the **CONF-04** ruling the economic terms are **off-core** in the new engine too. So the
> compositional engine's economic module defines its own value pool — and per **CONF-58** it must carry
> **CO₂ purchase, CO₂ recycle, storage credit and carbon tax on leakage**, all four of which
> `surrogate_engine.py:624-646` already computes inline.

---

## 8. Missing entirely

| Item | Why full simulation needs it |
|---|---|
| **Component tracking** | CO₂-EOR is about component fate. There is no component-level state object anywhere |
| **PVT regions** | Variable composition requires multiple regions with distinct properties |
| **Numerical control surface** | `EORParametersDefaults` has `pressure_tolerance`, `max_pressure_iterations`, `mass_balance_tolerance` — **nothing** for Newton iterations, Jacobian convergence, line search, **preconditioner** (**CONF-14**), adaptive-timestep growth/shrink, flash tolerance, Rachford-Rice bracket, component-balance tolerance, Krylov restart |
| **IO format + version** | **CONF-30b** — `petekIO` is named 3× and defined 0× |
| **Initial-state array** | Only `initial_water_saturation` (scalar, `:430`). No $S_o$, no $S_g$, no per-cell field, no FWL, no capillary–gravity equilibration |
| **`runspec`** | `Optional[Dict[str, Any]]` (`:420`) — untyped |
| **`faults`** | `Optional[Dict[str, Any]]` (`:421`) — untyped, while `FaultData`/`FaultGeometry`/`FaultProperties` exist as typed dataclasses but are **not referenced here** |
| **Tracer / surveillance** | Required to validate component allocation |

---

## 9. P2 rebuild sketch

Ordered by dependency. **Specify during P1, implement after M6** — the shape is stable even if the
engine's internals are not.

| # | Artefact | Change | Closes |
|---|---|---|---|
| **D1** | `CompositionalGrid` | **New.** $N_x\times N_y\times N_z$, $\Delta x/\Delta y/\Delta z$, cell volumes, face areas, transmissibilities, `ActiveMask`, NTG, corner-point vertices, per-cell $\phi$, full-tensor $K$ | §2, §2.1 |
| **D2** | `FluidComposition` | **New.** $N_c$ components, per-component $T_c, P_c, \omega, MW$, volume translation, **full symmetric BIP matrix**, C7+ definition | §3, **CONF-29**, **CONF-10** |
| **D3** | `CompositionalPVT` | **New.** correlated or tabulated $B_o, B_g, R_s, \mu_o, \mu_g$ vs. $P,T$; PVT regions | §4 |
| **D4** | `SCALTables` | **New.** per-rock-type $k_{ro}, k_{rw}, k_{rg}$ with Corey exponents **including water**, $P_c$ curves, hysteresis (Killough/Carlson), endpoints | §5.2 |
| **D5** | `WellCompletion` | **New.** completion type, MD/TVD intervals, perforation clusters and phasing, cement sheath, ICD/AICD, stage isolation | §6 |
| **D6** | `InitialState` | **New.** per-cell $S_o, S_w, S_g$, composition, pressure, temperature, FWL | §8 |
| **D7** | `NumericalControls` | **New.** Newton limits, Jacobian tolerance, line search, **preconditioner**, timestep growth/shrink, flash tolerance, component-balance tolerance | §8, **CONF-14** |
| **D8** | `CompositionalEconomics` | **New.** adds **gas price**, **storage credit**, escalation, terminal value; retains the four CO₂ terms | §7, **CONF-11**, **CONF-37**, **CONF-58** |
| **D9** | IO format | **New.** versioned, round-trippable, with a checkpoint/restart form | **CONF-30b**, **CONF-28** |
| **D10** | `value_pool` section | **New** section in `config/base_config.json` for all of the above, using the `EmpiricalFittingParameters` `{default,min,max,description,unit}` pattern | §5.1 |
| **D11** | Engine abstraction | **New.** replaces the hardcoded construction at `core/optimisation_engine.py:268` and `core/data_integration_engine.py:48` | [`integration_plan.md`](integration_plan.md) §2 |
| **D12** | Fix `extension_points.md` | Correct or delete the five stale paths | §6.1 |

### 9.1 Backward compatibility is not optional

`agent_wiki/README.md` invariant 13 requires `.tphd` projects to round-trip, with **shallow** dataclass
encoding — never recursive `dataclasses.asdict()` (`utils/project_file_handler.py`). D1–D9 are new
top-level dataclasses; `ProjectEncoder` and `project_decoder` must be extended for them, and
`pytest tests/test_project_save_load.py -v` becomes a **mandatory gate for every D-change**.

`ReservoirData.schema_version = "2.0"` (`:447`) is the existing version marker. D1–D9 need their own
versions with an explicit migration path — a `.tphd` written by the old app must still load.

---

## 10. Priority

| Priority | Items |
|---|---|
| **Specify in P1, implement in P2** | **D1** (grid — everything depends on it), **D2** (composition), **D4** (SCAL) |
| **P2 early** | D3, D6, D7, D11 |
| **P2** | D5, D8, D9, D10 |
| **P2 / immediate** | **D12** — the extension guide is unbuildable and P2 depends on it |