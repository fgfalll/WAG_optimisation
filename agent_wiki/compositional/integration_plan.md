# P2 Integration Plan

**When:** after the compositional engine passes every gate in [`build_plan.md`](build_plan.md) M0 → M6.
**Not now.** This page exists so P2 is scoped during P1 rather than discovered during it.

**Owner condition:** *"after compositional engine is verified working on all features and fronts we will
add him to app fully."*

> ⚠️ **UI/UX is OUT OF SCOPE** (owner, 07-10-2026): *"before rust engine is fully build and tested ui
> and ux isnt in scope."* §5 has been reduced to a **deferred requirements register** — the questions
> P2 will need answered, recorded now so they are not invented late. **No UI design work is scheduled
> here.**
>
> 🔴 **Economic scope has moved out of the engine.** Per **INV-3** a **separate economic engine** owns all
> field development calculations. **CONF-11 / 37 / 58 / 59 and CRIT-18 belong there, not here.**

Companion: [`data_model_gap.md`](data_model_gap.md) (evidence), [`data_architecture.md`](data_architecture.md)
(the PostgreSQL + vector store replacement).

---

## 1. P2 entry criteria

| # | Criterion | Source |
|---|---|---|
| 1 | Every M0–M6 gate passed with a **recorded measured value** | [`build_plan.md`](build_plan.md) |
| 2 | Component mass balance `< 1e-12`, BL ≤ 0.1 %, SPE 5 ≤ 1.5 %, density MAPD ≤ 2 % | D5 §3.1/§2.1/§6.1, **CONF-49** |
| 3 | `audit_comp` register validates; no open `CRITICAL` | [`register_spec.md`](register_spec.md) |
| 4 | A P2 ADR exists, retiring the Phase-1 separation gates | [`separation_doctrine.md`](separation_doctrine.md) §2.1 |
| 5 | Data-model shape D1–D10 **specified** (see [`vision_and_phases.md`](vision_and_phases.md) §5 Q1) | [`data_model_gap.md`](data_model_gap.md) §9 |

---

## 2. Engine routing — workstream 1

### 2.1 Current state: construction is hardcoded in two places

| Site | Line | Code |
|---|---|---|
| `core/optimisation_engine.py` | `:268` | `self.simulation_engine = SurrogateEngineWrapper(` |
| `core/data_integration_engine.py` | `:48` | `self.surrogate_engine = SurrogateEngineWrapper()` |

`core/engine_surrogate/__init__.py:13` contains a name-based dispatch branch
(`elif name == 'SurrogateEngineWrapper':`), which is a partial prototype of the idea.

### 2.2 There is no engine abstraction to extend

**`core/engine_factory.py` does not exist** (`Test-Path` → `False`). And
`agent_wiki/development/extension_points.md:98-117` — a whole section titled *"Registering a Real
Simulation Engine with `EngineFactory`"*, instructing the reader to *"Modify `core/engine_factory.py`
lines 105–116"* — documents a file that does not exist.

> **The wiki already documents this task and gets it wrong.** Anyone following that section in P2 would
> fail immediately. Fix it as part of P2 (**D12**).

### 2.3 What P2 needs

| Item | Decision |
|---|---|
| **Interface** | Common protocol both engines satisfy. Minimum: `evaluate_scenario(params) -> profiles` |
| **Profile contract** | The surrogate publishes a documented 4-stream schema (`agent_wiki/README.md` invariant 8). **The compositional engine must publish the same keys**, or every consumer must be rewritten |
| **Selection policy** | Explicit, per-call. **Never silent** — see §2.4 |
| **Where the switch lives** | `core/optimisation_engine.py` and `core/data_integration_engine.py` construction sites, plus a config/UI selector |
| **Capability declaration** | The compositional engine is ~10³–10⁵× slower. It must declare cost so a caller cannot put it in an optimisation loop by accident |

### 2.4 Two consumers that will break if the contract is not honoured

| Consumer | Line | Dependency |
|---|---|---|
| `core/objectives/wrapper.py` | `:50-53` | Reads `profiles["npv"]`, then applies containment/remediation penalties |
| `core/optimisation_engine.py` | `:268` | Constructs the engine directly |

Plus the UI: profile plots, Pareto fronts, and the run exporter all read the surrogate's profile keys.

### 2.5 🔴 INV-1 and INV-2 govern this workstream — no fallback, no automatic invocation

`agent_wiki/compositional/engine_invariants.md` INV-1 and INV-2 are **mandatory**. Concretely, for P2:

| Condition | Required behaviour |
|---|---|
| Compositional run fails to converge | **Fail loudly.** Return a typed error. Never substitute surrogate output |
| Compositional run fails for any reason | **Stop.** No retry with a looser tolerance, no partial result |
| User requests compositional, surrogate is faster | **Refuse or warn with the cost.** Never swap silently |
| Any background or batch path | **Must not** invoke the compositional engine (INV-2) |
| Engine identity | Recorded in **every** artifact, via the run manifest |

> ⚠️ **The temptation to write here, explicitly refused.** A developer integrating this will be tempted
> to write `if compositional.run() failed: return surrogate.run()`. **Do not.** It ships invalid results
> as valid, and with **INV-4** it also poisons the neural surrogate's training set — permanently and
> invisibly. There is no safe fallback, which is exactly why the owner calls it impossible by design.
>
> `utils/run_exporter.py` already produces a run manifest. **Engine identity must be a manifest field**,
> alongside a terminal `status` that has no "success with degraded output" value.

---

## 3. Data models — workstream 2

Full analysis and D1–D12 list in [`data_model_gap.md`](data_model_gap.md). Summary of the headline:

> **`ReservoirData.grid` contains only dimensions.** Its real content is
> `{"NX": [50], "NY": [50], "NZ": [10]}` (`tests/scientific/conftest.py:19`). Porosity and permeability
> are **scalars** (`data_models.py:431-432`). There is no NTG, no transmissibility array, no corner-point
> geometry, no `ActiveMask`, no full permeability tensor, and no 3D facies realisation
> (`grid_resolution` is a **2-tuple**, `:370`).

**Most important P2 decision:** `CompositionalGrid` (D1). Everything downstream depends on its shape.

---

## 4. Value pool — **superseded by the database decision**

🔵 **The value pool moves to PostgreSQL** — [`data_architecture.md`](data_architecture.md). The flat
`config/base_config.json` structure (132 keys in one section) is retired in favour of typed tables.

| Retire | Replacement table |
|---|---|
| `EORParametersDefaults` (132 flat keys) | `numeric_control`, plus engine-specific tables |
| `config/fluid_composition.json` (2 components, 2×2 BIP) | `fluid`, `component`, `binary_interaction` — with mandatory **`source`** provenance per value |
| `PVTPropertiesDefaults` (single-element lists) | `pvt_table`, `pvt_region` |
| `config/fault_properties.json` | `fault` |
| `EconomicParametersDefaults` (11 keys) | 🔴 **the economic engine's tables** — not this engine's |

The `EmpiricalFittingParameters` `{default, min, max, description, unit}` metadata pattern (13 entries)
remains the right model for DB column metadata and any residual JSON config.

> ⚠️ **Gas price and CO₂ storage credit are absent today** — storage credit defaults to `0.0` in code at
> `surrogate_engine.py:626` and is not in the pool. That is **CONF-11 / CONF-37 / CRIT-18**, and it now
> belongs to the **separate economic engine**.

---

## 5. UI/UX — DEFERRED, requirements register only

⚠️ **No UI/UX work is in scope** until the Rust engine is fully built and tested (owner, 07-10-2026).
This section records the questions P2 will need answered so they are **not invented at the last minute**.
**It is not a design and schedules nothing.**

### 5.1 Questions P2 must answer

| # | Question | Why it matters |
|---|---|---|
| 1 | Where does the engine selector live? | The app currently has **no** engine switch |
| 2 | Per-run or global? | A compositional run is expensive — per-run is likely right |
| 3 | **May the optimiser ever invoke it?** | **INV-2 says no.** The UI must therefore keep them visibly separate, not offer a mode that violates the invariant |
| 4 | How is the **cost** disclosed before the user commits? | INV-2 is only meaningful if the user understands the cost |
| 5 | Where do field visualisations go? | Compositional output is **fields**; the surrogate produces **profiles**. A map/field view is a **new capability class**, not a plot extension |
| 6 | Does invariant 8 (4 standardised streams) extend or get scoped? | ⚠️ Compositional output is **per component**, including CO₂ fate **by mechanism**. Per-mechanism trapping (Free / Trapped / Dissolved / Adsorbed / Mineralised — `../thmc/architecture_design.md` §6.5) has **no equivalent** in the current schema |
| 7 | Which existing plots are surrogate-only? | Response surfaces, profile-shape plots, Pareto fronts — all surrogate artefacts |

### 5.2 New input surfaces P2 will eventually need

Recorded for completeness only — **none is scheduled**.

| New data | Plausible placement | Note |
|---|---|---|
| Grid / geometry (D1) | Data Management → geometry | Needs a **new** editor; current geometry is scalar fields |
| Fluid composition (D2) | Data Management → fluid | Component editor with per-component properties, a **symmetric BIP matrix**, and ⚠️ a **`source`** field per value |
| PVT (D3) | Data Management → PVT | Replaces single-value placeholders |
| SCAL tables (D4) | Data Management → rel-perm | Per rock type; Corey exponents **including water**; $P_c$; hysteresis |
| Completion geometry (D5) | Data Management → wells | Completion type, intervals, clusters, phasing, cement, ICD |
| Numerical controls (D7) | An **advanced** panel | Default-hidden; most users should never touch it |
| **Economic parameters** | Economics panel | 🔴 **Belongs to the separate economic engine**, not this one |

> ⚠️ **Invariant 8 should be decided during P1**, not P2. It shapes the engine's output schema, and with
> **INV-4** the output schema determines what the neural surrogate can be trained on. Deciding it late
> means re-running historical simulations.

---

## 6. P2 sequencing

| Step | Work | Depends on |
|---|---|---|
| 1 | ADR: retire P1 separation gates; record engine identity in manifests | P2 entry criteria |
| 2 | Engine abstraction + interface + capability declaration | — |
| 3 | `CompositionalGrid` (D1) — **the critical path** | spec decided in P1 |
| 4 | Fluid composition + PVT (D2, D3) | D1 |
| 5 | SCAL (D4) | D1 |
| 6 | Initial state (D6), wells/completions (D5), numerics (D7) | D1 |
| 7 | Wire the engine at both construction sites (§2.1) | 2–6 |
| 8 | Value pool section (D10) | 3–6 |
| 9 | `.tphd` round-trip for every new dataclass | 3–6 — **`pytest tests/test_project_save_load.py -v` is a mandatory gate** |
| 10 | IO format (D9) | 3–6 |
| 11 | UI: engine selector first, then new input surfaces | 2, 8 |
| 12 | Result/field visualisation | 11 |
| 13 | Fix `extension_points.md` stale paths (D12) | — can be done any time; **do it early** |

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Grid shape decided twice** — once in P1 implicitly, once in P2 explicitly | 🔴 High | Specify D1 during P1; it is the critical path |
| **Silent engine fallback ships invalid results as valid** | 🔴 High | §2.5 — fail loudly, record engine identity in every manifest |
| `.tphd` backward compatibility broken by D1–D9 | 🟠 Medium | Versioned migration; mandatory project-save-load gate |
| UI scope discovered late | 🟠 Medium | Specify input surfaces during P1 even if built in P2 |
| Invariant 8 revised late, forcing a schema migration | 🟠 Medium | Decide the compositional output schema in P1 |
| Optimiser accidentally routed to the 10³–10⁵× slower engine | 🟠 Medium | Capability declaration + cost disclosure (§2.3) |
| Two engines diverge physically with no cross-check | 🟠 Medium | **P3** — but keep shared analytic/CMG reference values as the arbiter from day one |
| `extension_points.md` misleads a P2 implementer | 🟡 Low | D12 early |

---

## 8. P2 open decisions

| # | Question |
|---|---|
| 1 | Does the compositional engine **replace** the surrogate or **complement** it? Determines whether P3 exists |
| 2 | Is the surrogate kept as a fast path permanently, or retired once the compositional engine is fast enough? |
| 3 | Where does engine identity live in the run manifest, and is it a mandatory field? |
| 4 | Does invariant 8 (4 standardised streams) extend to per-component output, or is it scoped to the surrogate? |
| 5 | Is the UI engine selector per-run or global? |
| 6 | Does the compositional engine ever run inside an optimisation loop, or strictly for final evaluation? |
| 7 | Does the `audit_comp` register merge into `audit/scientific_flaws.md` at P2, or stay separate permanently? |