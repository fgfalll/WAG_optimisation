# Data Architecture — PostgreSQL + pgvector, with a layered file substrate

**Decisions:**
- **08-10-2026 — project data: PostgreSQL + `pgvector`.** To make data usable by **both** engines,
  transition from file-based configuration to a database.
- **08-10-2026 — simulation output: a layered file substrate** (S4) — `petekIO` in memory; **HDF5 /
  VTK-HDF**, **RESQML 2.0.1 / GRDECL** for spatial; **DuckDB / SQLite (libSQL)** and **Parquet / Arrow**
  for tabular. Engine emits **all raw values**; satellite tools interpret and format.

⚠️ **These are complementary, not competing.** Simulation output is **open, tool-neutral files**; project
definition is a **transactional database**. Full detail: [`output_schema.md`](output_schema.md) §5.

> ✅ **This dissolves the P1 separation tension cleanly.** The engine writes *standard open formats*. The
> Python satellite reads them with `h5py` — **already a project dependency** (`requirements.txt`,
> `pyproject.toml`, and used by `validation/sr3_reader.py`). **No FFI, no shared library, no Rust in the
> Python process.** The two engines share *data*, never *code*.

---

## 0. Store map

| Concern | Store | Status |
|---|---|---|
| In-memory state during a run | `petekIO` GeoData slabs (`f64`, zero-allocation) | Spec L1 |
| 3D/4D spatial output fields | **HDF5 / VTK-HDF** (`.h5`, `.vtu`) | Spec L2 |
| Grid interchange | **RESQML 2.0.1 / GRDECL**, `.pproj` | Spec L2 |
| Well + field time series | **Apache Parquet / Arrow** | Spec L3 |
| Solver diagnostics, MBE, run logs | **DuckDB / SQLite (libSQL)** | Spec L3 |
| V&V reference results | SPE 1/3/5/9/10/11 suite | Spec L4 |
| **Project / reservoir definition** | **PostgreSQL** | Decided 07-10 |
| **Training pairs, realisation ensembles** | **`pgvector`** | Decided 08-10 |

---

## 0b. ⚠️ Python satellite gaps — measured 08-10-2026

Checked in the project `.venv` (`.venv\Scripts\python.exe`):

| Package | Status | Needed for |
|---|---|---|
| `h5py` **3.16.0** | ✅ installed | reading Spec L2 — **already a dependency** |
| `numpy` **2.5.3**, `pandas` **3.0.5**, `scipy` **1.18.1**, `sklearn` **1.9.1** | ✅ installed | analysis, ML |
| **`pyarrow`** | ❌ **MISSING** | reading Spec L3 Parquet |
| **`duckdb`** | ❌ **MISSING** | Spec L3 SQL |
| `torch` | ❌ absent | expected — the neural surrogate is *yet to be developed* |

**Both missing packages are cheap and on the critical path.** Without them the satellite cannot read a
single Parquet or DuckDB artifact the engine produces. Tracked as correction **C-27** in
[`spec_corrections_log.md`](spec_corrections_log.md).

---

## 0c. ⚠️ Volume reality check — measured 08-10-2026

Per-cell float fields per timestep across all nine domains, at $N_c = 6$: **116**.

| Case | Cells | Per timestep | 100 timesteps |
|---|---|---|---|
| SPE 5 (7×7×3) | 147 | 0.07 MB | 0.01 GB |
| 200 k | 200 000 | 92.8 MB | 9.3 GB |
| **SPE 10 (D5 Level 5)** | 1 100 000 | **510.4 MB** | **51.0 GB** |

| # | Implication |
|---|---|
| 1 | ⚠️ **HDF5 chunking + compression is mandatory.** 51 GB uncompressed for a 100-step SPE-10 run is not viable |
| 2 | ⚠️ **1.1 M cells × 1 field `f32` = 4.4 MB.** Reading one field per 16.6 ms frame is **not viable from disk** — the caching strategy in [`output_schema.md`](output_schema.md) §3.1 must be implemented, not just documented |
| 3 | ✅ **Daily/monthly aggregation before writing** cuts volume ~30× versus micro-steps |
| 4 | ⚠️ **Not every run populates every domain** — see the capability mechanism (**INV-6**) in [`engine_invariants.md`](engine_invariants.md) §6 |
| 5 | ⚠️ **Open branch B-1:** a PCA / Karhunen–Loève reduced basis over 116 correlated fields would cut storage **and** shrink the RL observation space. Worth deciding before the dataset grows |

---

## 1. Why a database, not more JSON

| Problem today | Evidence |
|---|---|
| Grid is dimensions-only | `ReservoirData.grid` = `{"NX":[50],"NY":[50],"NZ":[10]}` (`tests/scientific/conftest.py:19`) |
| Porosity/permeability are **scalars** | `data_models.py:431-432` |
| Geology is **2D** | `grid_resolution: Tuple[int, int]` (`:370`) |
| Fluid is **2 components, 1 $k_{ij}$** | `config/fluid_composition.json` |
| PVT tables are single-element lists | `config/base_config.json` → `PVTPropertiesDefaults` |
| Economics incomplete | No gas price; storage credit defaults to `0.0` in code, absent from the pool |
| Everything is process-local state | `EORParameters` / `ReservoirData` passed through call chains |

**A single file cannot serve two engines with different needs.** The surrogate needs a small, fast,
coarse summary; the compositional engine needs cell-level arrays, component vectors and run provenance.
A database lets both read the same source with different projections, and gives versioning, transactions
and concurrent access for free — which a JSON file cannot.

---

## 2. Store split

| Store | Holds | Rationale |
|---|---|---|
| **PostgreSQL** | Reservoir definition, grid, geology, fluid composition, PVT, SCAL, wells, schedules, numeric controls, economic parameters, run manifests, findings | Relational, transactional, versioned. Correct choice for structured reservoir data |
| **Vector store** | Per-cell property fields; compositional output fields; **training datasets for the neural surrogate** | High-dimensional arrays and nearest-neighbour retrieval over realisation ensembles |
| **Rust engine** | Pure computation. **Holds no persistent state.** Loads a run specification, writes results | Keeps the engine testable and reproducible |

> ⚠️ **The vector store is not a replacement for PostgreSQL.** It is a *projection* for array-shaped and
> similarity-search workloads. The system of record is PostgreSQL; the vector store is derived and
> **rebuildable**.

---

## 3. Why this is better for the engine than more JSON

| Property | JSON today | PostgreSQL |
|---|---|---|
| Per-cell arrays (1.1M cells, D5 Level 5) | Awkward; memory-bound | Native large objects / partitioned rows |
| Transactional multi-table update | No | **Yes** |
| Schema migration with rollback | Manual | Versioned migrations |
| Concurrent read during a write | No | Yes — MVCC |
| Audit trail of a reservoir's history | None | Native |
| Reproducibility | Depends on file provenance | Row version + transaction id |
| Two engines reading different projections | Duplicate files | **Views / projections** |

### 3.1 Grid storage — the design decision

`ReservoirData.grid` must become real grid data. Options:

| Option | Assessment |
|---|---|
| Row per cell | Correct and queryable, but 1.1M rows × many properties. Needs **partitioning by region/layer** |
| **Array column** (`float8[]`) per property | Simple; whole-field load is one read. Poor for per-cell queries |
| **Hybrid** — array column per property + a region/layer index table | ✅ **Recommended.** Whole-field load for the engine (its access pattern is sweep, not query), plus the index table for region queries, validation and UI |

```sql
-- recommended shape (illustrative)
CREATE TABLE grid_property (
    reservoir_id   uuid NOT NULL REFERENCES reservoir(id),
    property       text NOT NULL,   -- 'porosity' | 'permeability' | 'ntg' | 'transmissibility'
    layer_id       int  NOT NULL,
    values         float8[] NOT NULL,
    PRIMARY KEY (reservoir_id, property, layer_id)
);
```

> **Rust integration:** arrays come back contiguous, which maps directly onto the SoA flat-slab layout the
> design set requires (`../thmc/architecture_design.md` §2.1). **A DB array column and an SoA slab are the
> same memory layout.** This is the natural fit — and it is a reason to prefer PostgreSQL over, say, an
> object store.

---

## 4. Schema outline

Designed at the **domain** level. Table names are provisional.

### 4.1 Core reservoir definition

| Table | Key columns | Replaces |
|---|---|---|
| `reservoir` | `id`, `name`, `schema_version`, `created_at` | `ReservoirData` scalars (`:422-435`) |
| `grid` | `reservoir_id`, `nx`, `ny`, `nz`, `dx`, `dy`, `dz`, `active_mask`, `ntg`, `corner_point_uri` | **`ReservoirData.grid`** (`:417`) |
| `grid_property` | see §3.1 | `average_porosity`, `average_permeability` **scalars** (`:431-432`) |
| `region` | `reservoir_id`, `region_id`, `name`, `perm_multiplier`, `poro_multiplier` | `regions` (`:419`), `LayerDefinition` (`:408`) |
| `layer` | `reservoir_id`, `layer_id`, `thickness`, `n_top`, `n_bottom` | `LayerDefinition.thickness` only |
| `fault` | `reservoir_id`, `strike`, `dip`, `dip_direction`, `throw`, `heave`, `length`, `z_top`, `z_base`, `transmissibility_multiplier`, `damage_zone_width`, `friction_coefficient`, `cohesion`, `shale_gouge_ratio`, `is_active` | `FaultData` (`:220-242`) — already well specified, needs persistence |
| `fault_transmissibility` | `fault_id`, `cell_i`, `cell_j`, `multiplier` | **absent today** |

### 4.2 Fluid and thermodynamics

| Table | Key columns | Replaces |
|---|---|---|
| `fluid` | `reservoir_id`, `eos_type`, `n_components`, `volume_translation_model` | `eos_composition` (`config/fluid_composition.json`) |
| `component` | `fluid_id`, `name`, `mole_fraction`, `molecular_weight`, `boiling_point`, `critical_temperature`, `critical_pressure`, `acentric_factor`, `c7_plus_flag` | `EOSModelParameters.component_properties` (`:311`) — **2 rows today** |
| `binary_interaction` | `fluid_id`, `component_i`, `component_j`, `kij`, `source` | `binary_interaction_coeffs` (`:312`) — **2×2 today** |
| `pvt_table` | `reservoir_id`, `property`, `pressure`, `temperature`, `value` | `PVTPropertiesDefaults` **single-element lists** |
| `pvt_region` | `reservoir_id`, `region_id`, `composition_spec` | **absent** |

> **`source` on `binary_interaction` is mandatory.** Per the M1 references
> ([`spec_defects.md`](spec_defects.md) §2), each $k_{ij}$ must record whether it came from experimental
> regression, the Abudour et al. (2014) QSPR fallback, or a default. **A $k_{ij}$ with no provenance is
> not usable** — that is exactly **CONF-31**.

### 4.3 Petrophysics and initial state

| Table | Key columns | Replaces |
|---|---|---|
| `rock_type` | `reservoir_id`, `name`, `fzi`, `rqi`, `swc`, `sor`, `sgc`, `sgr` | **global scalars only** today |
| `relative_permeability` | `rock_type_id`, `phase`, `corey_exponent`, `endpoint_kr`, `table_uri` | `CoreyParameters` (`:2115`) — **no water term, no Pc, no hysteresis** |
| `capillary_pressure` | `rock_type_id`, `sw`, `pc`, `leverett_j` | **absent** |
| `hysteresis` | `rock_type_id`, `model` (killough/carlson), `scan_table_uri` | **absent** |
| `initial_state` | `reservoir_id`, `cell_index`, `pressure`, `temperature`, `so`, `sw`, `sg`, `z[]` | `initial_water_saturation` **scalar** (`:430`) |
| `free_water_level` | `reservoir_id`, `depth_ft` | **absent** |

### 4.4 Wells and operations

| Table | Key columns | Replaces |
|---|---|---|
| `well` | `reservoir_id`, `name`, `well_index`, `skin_factor`, `wellbore_radius_ft` | `WellData` (`:64-75`) |
| `well_trajectory` | `well_id`, `md`, `tvd`, `x`, `y` | `well_path: Optional[np.ndarray]` — flat |
| `well_completion` | `well_id`, `completion_type`, `md_top`, `md_bottom` | **absent** |
| `perforation_cluster` | `well_id`, `stage`, `cluster`, `md_top`, `md_bottom`, `phasing_deg`, `hole_diameter` | `perforations: List[List[float]]` — flat, no clustering |
| `cement_sheath` | `well_id`, `md_top`, `md_bottom`, `thickness`, `bond_strength` | **absent** |
| `icd` | `well_id`, `md_top`, `md_bottom`, `device_type`, `characteristic` | **absent** |
| `well_schedule` | `well_id`, `day`, `action`, `rate`, `duration_days`, `bhp_target`, `trigger_condition`, `trigger_value` | `WellScheduleEntry` (`:198-207`) — note `max_operations = 100` hard cap |
| `fault_mechanics` | `reservoir_id`, … | `GeomechanicsParameters` (`:1576`) |

### 4.5 Runs and provenance — required by INV-1 and INV-4

| Table | Key columns | Why |
|---|---|---|
| `run` | `id`, `reservoir_id`, `engine`, **`engine_version`**, **`build_hash`**, `status`, `started_at`, `finished_at`, `manifest_id` | **INV-1**: terminal status. **INV-4**: provenance per sample |
| `run_control_vector` | `run_id`, `parameter_name`, `value` | **INV-4**: the $x$ half of the training pair |
| `run_result` | `run_id`, `timestep`, `variable`, `component`, `value` | **INV-4**: the $y$ half |
| `run_error` | `run_id`, `error_variant`, `message`, `cell_id`, `context` | **INV-1**: typed error taxonomy, persisted |
| `schedule_stage` | `run_id`, `from`, `to`, `action`, `target` | `utils/run_exporter.py` already produces a manifest — **extend, don't replace** |

> ⚠️ **INV-1 consequence.** `run.status` must have a terminal state that is **not** "success with
> degraded output". A failed run writes `run_error` and produces **no usable artifact**.

### 4.6 Economics — a **separate engine**, per INV-3

| Table | Key columns |
|---|---|
| `economic_scenario` | `reservoir_id`, `name`, `discount_rate`, `capex`, `base_date` |
| `price_deck` | `scenario_id`, `commodity`, `year`, `price`, `escalation` |
| `cost_item` | `scenario_id`, `category`, `unit_cost`, `basis` |
| `carbon_terms` | `scenario_id`, `storage_credit_usd_per_tonne`, `carbon_tax_usd_per_tonne` |

> **Today:** 11 keys in `EconomicParametersDefaults`, **no gas price**, **no storage credit**, no
> escalation, no terminal value. See [`data_model_gap.md`](data_model_gap.md) §7. This is the future
> economic engine's scope, **not** the compositional engine's.

### 4.7 Findings

| Table | Key columns | Note |
|---|---|---|
| `finding` | `id` (**`COMP-nn`**), `severity`, `category`, `location`, `observed`, `expected`, `impact`, `evidence`, **`verification`**, `status` | **Mirrors `audit_comp` register** (INV-5). Keep the markdown register as the git-tracked source of record; mirror to DB for querying |

---

## 5. Vector store

| Collection | Contents | Origin |
|---|---|---|
| `cell_fields` | Per-cell property realisations, per region, per geological realisation | Generated from `grid_property` |
| `realisation_ensemble` | Geo-statistical realisations for history matching / UQ | Variogram parameters + seed |
| **`training_pairs`** | **$(x,y)$ for the neural surrogate** | **Compositional engine runs (INV-4)** |

> **`training_pairs` is the reason a vector store is in scope.** Per-reservoir surrogate training
> (INV-4) needs large labelled sets with nearest-neighbour retrieval over input vectors for
> curriculum sampling and out-of-distribution detection.

### 5.1 A train/test leakage warning

Compositional runs are **expensive**. If the same well configuration appears in both train and test
splits, reported surrogate accuracy is meaningless — the network memorises the configuration, not the
physics.

**Required:** split by **geological realisation** and by **development pattern**, not randomly over
samples. This must be stated in the dataset schema from the start; retrofitting it is painful.

---

## 6. Consequences for the plan

### 6.1 P1 needs a **run-specification** concept, not just a schema

The engine must be loadable from the DB while still testable standalone. Define:

| Requirement | Detail |
|---|---|
| **Run specification** | A self-contained, versioned object the engine consumes: reservoir + grid + fluid + PVT + SCAL + wells + schedule + numeric controls |
| **Single source** | Built from PostgreSQL, serialised into the engine's own binary form (**CONF-30b** — `petekIO` is named 3× and defined 0×) |
| **Deterministic** | Same spec ⇒ same output. Required by INV-4 and by every M-gate |
| **Hashable** | `engine_version` + `build_hash` + `spec_hash` together identify a result completely |

> **This resolves an architectural tension.** [`separation_doctrine.md`](separation_doctrine.md) forbids
> the Rust crate importing Python modules during P1. A shared database schema satisfies that: the DB is
> **data**, not code. Both engines read the same tables without either importing the other.

### 6.2 What moves out of the file-based model

| Retired / superseded | Replacement | Note |
|---|---|---|
| `config/base_config.json` (26 KB, 132-key flat section) | `price_deck`, `cost_item`, `numeric_control` tables | Metadata pattern reused from `EmpiricalFittingParameters` |
| `config/fluid_composition.json` (2 components) | `component` + `binary_interaction` tables | With mandatory `source` provenance |
| `config/fault_properties.json` | `fault` table | Already well specified (`FaultData`) |
| `ReservoirData.grid` (dimensions only) | `grid` + `grid_property` | **The critical change** |
| `.tphd` project files | `reservoir` + tables | ⚠️ **Backward compatibility is not optional** — see §7 |
| `core/data_models.py` dataclasses | Typed Rust structs + DB schema | **Python dataclasses become read models / validation shims** |

### 6.3 Numeric controls — currently absent entirely

Needs its own table, because there is **nothing** in the value pool for it:

| Group | Parameters |
|---|---|
| Nonlinear solve | Newton max iterations, Jacobian convergence tolerance, line-search max steps, damping factor min |
| **Linear algebra** | Preconditioner, drop tolerance, fill-reducing ordering, Krylov restart — **CONF-14** |
| Timestep | initial $\Delta t$, min, max, growth factor, cut factor |
| Thermodynamics | flash tolerance, Rachford–Rice bracket policy, stability-test sample count |
| Conservation | per-component mass-balance tolerance — **the M4 gate** |
| Parallelism | thread count, subdomain decomposition |

---

## 7. Backward compatibility — non-negotiable

`agent_wiki/README.md` invariant 13 requires `.tphd` projects to round-trip, and
`utils/project_file_handler.py` uses **shallow** `ProjectEncoder` encoding — **never** recursive
`dataclasses.asdict()`.

Moving to a database **must not orphan existing projects**:

| Requirement | Detail |
|---|---|
| **Import path** | `.tphd` → PostgreSQL, a one-time migration with a **recorded source provenance** per reservoir |
| **Export path** | PostgreSQL → `.tphd`, so file-based workflows survive |
| **Versioning** | `reservoir.schema_version` — `ReservoirData.schema_version = "2.0"` exists today (`:447`) and is the current marker |
| **Round-trip test** | ⚠️ `pytest tests/test_project_save_load.py -v` remains a **mandatory gate for every schema change**, database or not |

---

## 8. Sequencing

| Step | Work | Notes |
|---|---|---|
| 1 | Schema design + migrations | Start with `grid`, `grid_property`, `fluid`, `component`, `binary_interaction` |
| 2 | Run-specification format — **resolves CONF-30b** | ⚠️ **Do this in P1.** The engine needs a loadable spec before any gate can run |
| 3 | PostgreSQL instance, connection layer, test fixtures | |
| 4 | Rust read path: spec → in-memory | |
| 5 | Rust write path: results → `run`, `run_result`, `run_error` | **INV-1**, **INV-4** |
| 6 | Vector store: `cell_fields`, `training_pairs` | After the engine emits data |
| 7 | `.tphd` import/export | **Invariant 13 gate** |
| 8 | Retire the flat `EORParametersDefaults` section | Late — nothing depends on it early |

> ⚠️ **Step 2 is P1 work, not P2.** Every M-gate needs a run specification. If the DB migration is
> deferred to P2, the engine must first carry a second, throwaway file format — which is precisely the
> **CONF-30b** mistake the design set already makes.

---

## 9. Open decisions

| # | Question |
|---|---|
| 1 | **Managed or self-hosted PostgreSQL?** Affects how run data leaves the machine |
| 2 | **Which vector store** — pgvector (Postgres extension) or a dedicated engine? ⚠️ `pgvector` keeps one system of record; a dedicated engine gives more capability. Recommendation: **pgvector** until a measured need appears |
| 3 | **How is `training_pairs` split** to prevent well-configuration leakage between train and test? (§5.1) |
| 4 | **Does the surrogate engine read PostgreSQL directly, or does a read model feed it?** Performance differs sharply — the surrogate must stay fast |
| 5 | **Multi-tenancy / reservoir sharing** — is one reservoir visible to multiple users? |
| 6 | **Retention** — run outputs are large. How long are they kept, and does `training_pairs` outlive the run? |
| 7 | **Does `audit_comp` mirror into the DB, or stay markdown-only?** (INV-5; §4.7) |