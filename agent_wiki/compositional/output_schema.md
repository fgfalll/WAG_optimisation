# Output Schema & Visualisation Specification

> [!CAUTION]
> **This is a TARGET specification, not a description of a current state.**
> It describes **what the compositional engine is intended to emit** once built. **No engine exists.**
> Verified 08-10-2026: `*.rs` → 0 files, `Cargo.toml` → 0, CI → 0. Nothing here has been implemented.
> Every table below states **intent**; the **current state is M0, not started**.
> **Current state** is recorded in [`build_plan.md`](build_plan.md) and [`vision_and_phases.md`](vision_and_phases.md).

**Status: approved as the target output contract, 08-10-2026** (reservoir engineer + project owner),
**as adjudicated** in [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md)
§6–§13. It supersedes any output description in `3D_THMC_docs/`.

**Sources, normalised here:**
- **S1** — `3D THMC Reservoir Simulator Full Output Data Schema & UI Visualization Specification.md` (285 lines, attached 08-10-2026)
- **S2** — the *3D THMC Compositionally-Coupled Simulation Engine — Master Map* (7 layers)
- **S3** — the *Full Output Persistence Pipeline* (4 schema groups)
- **S4** — the *Data Persistence Stack* (4 layers)

**Governing decisions:**
- The engine outputs **ALL raw values**. Satellite tools interpret and format them. No presentation
  logic in the engine.
- **Economics are NOT the engine's job** — ruled 08-10-2026. The core emits **strictly physical**
  time-series: `Q_o, Q_w, Q_g, Q_inj, W_p, G_p, N_p, ||R_m||, E_v, E_a, E_m, RF`. All monetary terms and
  Python filenames are **purged from S1** (**CONF-64 / CONF-65**, ruled).
- **The economic engine reads DuckDB / Parquet only, never HDF5** — ruled 08-10-2026. §8.1.

---

## 1. Three temporal resolutions

| Resolution | Contents | Consumer |
|---|---|---|
| **Micro-steps** (Newton iterations) | Internal solver `dt^n`, line-search damping, residual monitoring | convergence diagnostics |
| **Daily** | High-resolution well dynamics, rates, phase pressures, component skin factors | well/time-series analytics |
| **Monthly / yearly** | Aggregated material balance, final RF, **DCF** — see §8, economics is out | reservoir + economics |

> **Monthly/yearly must not carry economics** (**INV-3**). DCF belongs to the economic engine.

---

## 2. Spatial indexing

Single global cell index, matching the SoA flat-slab layout
([`../thmc/architecture_design.md`](../thmc/architecture_design.md) §2.1):

```
I_global = i + (j-1)*N_x + (k-1)*N_x*N_y
```

**Non-Neighbour Connection (NNC) index arrays are stored alongside** — required for exact EDFM fracture
and fault intersection rendering.

> This is a good decision: the DB array column, the `Vec<f64>` slab and the renderer all share one
> indexing scheme. No translation layer.

---

## 3. Persistence & UI mapping (verbatim from S1)

| Format | Path | Data types | UI component | Buffer strategy |
|---|---|---|---|---|
| **HDF5 / VTK-HDF** | `/mesh/geometry/vertices`<br>`/mesh/fields/P_res`<br>`/mesh/fields/S_gas` | `float32[N_nodes][3]`<br>`float32[N_cells]`<br>`float32[N_cells]` | 3D/4D viewports (volume rendering, cross-sections, isosurfaces) | **Interleaved Array Buffers** — vertex coords + scalar fields in one VBO to cut draw calls at 60 FPS |
| **HDF5 / VTK-HDF** | `/mesh/tensors/K_tensor`<br>`/mesh/vectors/disp_u` | `float32[N_cells][6]`<br>`float32[N_cells][3]` | 3D tensor ellipsoids, displacement arrows | **Instanced Rendering Buffers** — one base mesh glyph x N, transforms via Storage Buffers |
| **Apache Parquet** | `/timeseries/wells/{well_id}`<br>`/timeseries/field_totals` | `int64` (Timestamp)<br>`float32` (rates, P_bhp)<br>`Enum` (well status) | 2D time-series charts, 1D log diagrams | **TypedArray Streaming** — Parquet/Arrow vectors straight into Plotly.js / Canvas2D |
| **DuckDB / SQLite** | `main.material_balance`<br>`main.solver_diagnostics`<br>`main.cash_flows_yearly` | `VARCHAR`, `DOUBLE`, `INT32` | tabular analytics dashboards, ternary P-T-x diagrams, DCF NPV | **In-Memory WASM Worker** — local SQL in a background browser thread, no UI-thread block |

```
[ THMC Solver ]
   |
   +---> [HDF5 / VTK-HDF] --> Interleaved & Instanced Buffers --> WebGPU 3D Viewport (60 FPS)
   +---> [Apache Parquet]  --> Arrow Columnar Memory -----------> 2D Plotly / D3 Charts
   +---> [DuckDB WASM DB]  --> SQL Engine in Worker Thread ----> Financial & Convergence Dashboards
```

### 3.1 Frame-budget rationale

60 FPS means **≤ 16.6 ms per frame**. Loading unstructured 3D fields of millions of cells from disk
*every frame* causes I/O stalls and drops below **10 FPS** when rotating a 3D model. Hence:

- Direct VBO load during timestep initialisation.
- Residual norms and convergence accumulate in DuckDB, so solver diagnostics run via fast SQL
  **without re-reading heavy HDF5 arrays**.
- A new HDF5 frame is read only when the simulation timestep changes.

> This justifies the whole architecture. Because the engine writes *standard open formats*, the
> Python satellite can read them with **zero coupling** — no FFI, no shared library, no Rust in the
> Python process. `h5py` is **already a project dependency** (`requirements.txt`, `pyproject.toml`)
> and `validation/sr3_reader.py` already uses it to read CMG results.

---

## 4. Output domains — the nine schemas

### Domain 1 — Hydrodynamics and multiphase flow

Per cell, per timestep: `P`, `T`; saturations `S_o, S_w, S_g, S_scl` (**`S_o + S_w + S_g = 1.0`**
asserted by S1); densities `rho_o, rho_w, rho_g, rho_scl`; viscosities `mu_o, mu_w, mu_g`; formation
volume factors `B_o, B_w, B_g`; compressibilities `c_o, c_w, c_g`. Units: Pa/psi, K/°F, kg/m³,
Pa·s/cP, m³/m³ or bbl/STB, Pa⁻¹/psi⁻¹.

> **Two embedded physics formulas are carried over unrepaired** — see §7.

**Visualisation:** Viridis/Jet continuous colour scales on 3D cells; `S_g` by **isosurface thresholding**
to highlight gas coning; **effective-oil-thickness maps** by vertical integration of
`(1 - S_w - S_g) * h_cell`; time-series of `f_g(t)`, `HCPVI`, `RF(t)` with the displacement-efficiency
threshold marked.

### Domain 2 — Compositional thermodynamics and phase stability

Per node: overall `z_i`, liquid `x_i`, vapour `y_i`, fugacity coefficients `phi_i^L`, `phi_i^V`, K-values
`K_i = y_i / x_i`, phase moles `L/F` and `V/F` (**sum to 1.0**).
EOS: three-parameter SRK and PR with **VT-SRK / VT-PR** and `k_ij` by van der Waals or **QSPR /
Huron-Vidal**.

Spatio-temporal stability variables: **`MMP(x,y,z,t)`**, **`AOP(x,y,z,t)`**, asphaltene deposit
saturation `S_a`, gas hydrate concentration `C_hydrate`.

**Visualisation:** ternary P-T-x diagrams for pseudocomponents **(C1+N2, C2-C6, C7+)** with axes 0.0–1.0;
**flash trajectory plots**; **3D isosurface of `(P_cell - MMP) = 0`** separating miscible from
immiscible zones.

> The `(P - MMP) = 0` isosurface is the highest-value single visualization in the spec — it turns
> "miscibility degraded" from a scalar into a spatially locatable zone, and it is actionable (raise
> pressure to restore miscibility before irreversible loss).

### Domain 3 — Dynamic petrophysics and porous media

Porosity split three ways: matrix `phi_m`, fracture `phi_f`, total `phi_tot` — this resolves an
ambiguity the design set never resolved.
Full **6-component symmetric** anisotropy tensor **K** — resolves **CONF-41**.
Relative permeabilities `k_ro, k_rw, k_rg` with **Land hysteresis** branch tracking. Capillary
pressures `P_cow(S_w)`, `P_cog(S_g)`.
Flow-zone metrics: **FZI**, **RQI**, mean pore-throat radius `r_throat` (µm). Pore-structure: percolation
threshold `phi_c`, **Verma-Pruess throat-blocking index**.

> 🔴 **The `phi_c` guard is physics, not an artificial floor** (**C-63**). `k = k0[((phi_acc − phi_c)/(phi0 − phi_c))]^n`
> **must** be evaluated as `k = 0` for `phi_acc <= phi_c`. Measured: without it, $n{=}2$ gives a fully
> plugged cell (`phi_acc = 0`) $k/k_0 = \mathbf{0.0076}$ — still flowing — and $n{=}1.5$ returns **NaN**,
> poisoning the **entire global Jacobian** (an INV-1 `Unsolvable`, so the researcher gets *nothing*).
> A separate **denominator** singularity needs handling too: $k/k_0 = 6.4{\times}10^{5}$ at
> `phi0 = 0.0201`, divide-by-zero at `phi0 = phi_c`. See `engine_invariants.md` §7c.2 for the three-branch
> ruled implementation.
Facies by **Plurigaussian Simulation**: truncation flags and thresholds for geological contacts across
**Sand, ShSand, Sh, Lst, argLst**; allowed and **forbidden contacts** by truncating ≥2 Gaussian random
fields with a Gibbs sampler; **Hermite polynomials** for indicator variogram integration.

**Visualisation:** **3D tensor ellipsoid glyphs** — anisotropy principal axes via instanced rendering;
1D/2D `k_r(S)` and `P_c(S)` hysteresis curves with numeric bounds.

> The tensor-ellipsoid glyph view directly answers a real question: geomechanics and drilling can align
> horizontal wells with `K_max` to avoid premature water breakthrough.

### Domain 4 — Reactive geochemistry and mineralogy

Mineral volume fractions `phi_m(t)` for **7 minerals: calcite, dolomite, quartz, anhydrite, kaolinite,
smectite, illite**. Aqueous species: **7 species — `Ca2+, Mg2+, Fe2+, H+, HCO3-, SO4(2-), Cl-`**
(mol/L or ppm). Reaction rates `r_m` (mol/m³·s). Dimensionless indicators `Da` (reaction/convection)
and `Pe` (convection/diffusion). Near-wellbore contamination: filter-cake thickness `h_cake` (mm),
suspended solids `sigma_p` (kg/m³), emulsified oil-in-water (ppm).
**Hydrate rate `r_hyd`** (mol/m³·s) — dissociation kinetics, **not** the vdW-P equilibrium condition
(**C-68**); hydrate saturation `phi_h`; **latent heat** in the energy equation.

> 🔴 **Wormholing does not end at `k -> infinity`.** Kozeny-Carman diverges as $\phi\to1$
> ($k/k_0 = 3.6\times10^{7}$ at $\phi=0.999$), which is **mathematically correct and physically
> meaningless** — past some $\phi_*$ the **continuum porous-media description stops being valid**,
> because there is no pore network left to describe (**C-64**).
>
> ✅ **The unconstrained answer is a representation switch, not a clamp.** A dissolved channel whose
> aperture exceeds a cell dimension **is a fracture**: it leaves Domain 4's porous cells and enters
> Domain 7 as an **EDFM** fracture with its own $w_f$, $k_f$ and proppant state. This is the one place
> where INV-7 correctly **changes what the physics is**, rather than bounding it — and it is the first
> real link between Domain 4 and Domain 7 that the design set never provided.

> **Do not also store `pH`** — `pH = -log10(H+)` determines one from the other. Store `H+` activity as
> the state variable (speciation kinetics need it); derive `pH` for reporting.

**Visualisation:** 3D mineral dissolution/deposition maps via `D phi_m` gradient; **wormhole isosurfaces**
thresholded on `Da`/`Pe`; **Stiff / Piper diagrams** for produced-water ionic balance.

### Domain 5 — Geomechanics, stress and strain

Full 6-component total stress `(s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)`.
Biot effective stress `s'_ij = s_ij - alpha_B * P * delta_ij`.
Displacement vector `u = (u_x, u_y, u_z)` (m). Strain components: volumetric `eps_v`, deviatoric
`eps_d`; **`eps_p` (plastic, tensor-valued — RESTORED 08-10-2026)**. Dynamic moduli: `E(t)`, `nu(t)`,
`C_0(t)`, `alpha_B(t)`. Yield: `F(sigma', eps_p)`; **dilatancy `psi`**; accumulated plastic work.

> ✅ **`eps_p` RESTORED — CONF-66 moved `RESOLVED` → `PARTIALLY_RESOLVED`** (ruling 5, **C-62**).
> `eps_p` was removed because the design set named Mohr-Coulomb / Drucker-Prager **without a yield
> surface, flow rule or hardening**. The owner has now supplied all three: a yield surface
> `F(sigma', eps_p) = 0` and an explicitly **non-associated** plastic flow rule with dilatancy.
>
> 🔴 **Three things still block it** (all pre-M7b):
> 1. **Return-mapping algorithm** — radial-return vs. closest-point.
> 2. **Consistent tangent.** A non-associated Drucker-Prager return map has a **non-symmetric** algorithmic
>    derivative. Handing Newton the symmetric approximation silently costs quadratic convergence — a
>    **C-35**-class defect.
> 3. **Hardening law** — elastic–perfectly-plastic, or strain-hardening? Decides whether shear
>    localises into a band or spreads.
>
> ⚠️ **Mohr-Coulomb requires an explicit dilation angle `psi != phi`** to be non-associated.
> **Drucker-Prager is inherently non-associated** and is the better default for frictional rock.

**Visualisation:** 3D displacement vector glyphs; cross-sections of `s'_zz` or deviatoric stress `q`;
**subsidence/heave maps** from top-boundary `u_z`, scale **−0.05 m to +0.02 m**; `eps_p` magnitude and
plastic-strain-rate isosurfaces marking the **yielded volume**; caprock `s'_3 <= -s_t` (tensile) and
`DCFS > 0` (shear) zones highlighted as **containment-loss indicators** — emitted, never suppressed
(**INV-7** §7c).

> S1 names **Mohr-Coulomb *and* Drucker-Prager** failure envelopes. **Drucker-Prager is new** — not in
> the design set. It is the better choice for frictional rock (no unbounded shear strength at high
> confinement), so this is a genuine improvement worth keeping.

### Domain 6 — 5-state CO2/gas trapping inventory

Total gas mass `M_total` per cell and reservoir, distributed over five states, in metric tonnes or kg:

| # | State | Basis |
|---|---|---|
| 1 | `m_mobile` | pore gas able to flow under pressure gradient or gravity |
| 2 | `m_residual` | capillary-trapped in micropores (**Land hysteresis**) |
| 3 | `m_dissolved` | in formation water and oil (**Setschenov + Henry**) |
| 4 | `m_adsorbed` | sorbed on coal or clay surfaces (**Langmuir**) |
| 5 | `m_mineral` | carbonate-bound (calcite, **siderite**, **ankerite**) |

> Siderite and ankerite are new — the design set mentioned only CaCO3. Correct for CO2
> mineralisation, and it is why `Fe2+` is required in the aqueous set (**CONF-67**).

**Visualisation:** stacked-area charts of the five states over **0–100+ years** with mass annotations.

### Domain 7 — Discontinuities, EDFM fractures and fault mechanics

Fractures: `L_f`, `H_f`, dynamic aperture `w_f(t)` from **Barton-Bandis**
`w_f = w_f0 / (1 + s_n' / (K_ni · w_f0))`; joint conductivity `k_f * w_f`.
Proppant: concentration `C_prop` (kg/m²), pack height `h_pack`, embedment depth `d_embed`, crush
fraction, **Forchheimer beta-factor** (m⁻¹).
NNC transmissibilities: `T_mf` (matrix-fracture), `T_ff` (fracture-fracture).
Faults: **`CoulombFS(t) = tau - mu(s_n - P) - C0`** (absolute), **`DCFS(t) = (s_n-P)(mu1-mu0) - (s_tau,0 - tau)`**
(increment, the reported reactivation metric), shear displacement `u_s(t)` (mm), **SGR**, and a
**seal-vs-conduit state flag**.

> 🔴 **Three corrections govern this domain** (ruling 5, **C-65** / **C-66**) — the design set's version
> is not buildable as written:
>
> 1. **$k_f(w_f)$ is MISSING.** Barton-Bandis supplies the **aperture**; DFN transmissibility needs
>    `T = k_f · w_f`. Without a relation the `Seal`↔`Conduit` switch — whose whole purpose is to drive
>    `T_ff` — has no magnitude. Required: $k_f\propto w_f^{1/3}$, or $w_f^2/12$ — see **CONF-47**,
>    which argues for $w_f^{\mathbf{3}}/12$ and is **still open**.
> 2. **Barton-Bandis is singular in tension** at `s_n' = -K_ni·w_f0` (verified: $w_{f0}=10\ \mu m$,
>    $K_{ni}=10^{12}$ ⇒ $-10$ MPa) and returns a **negative aperture** beyond it. ✅ The unconstrained
>    behaviour there is **fracture initiation → open conduit**, i.e. the feature **converts
>    representation** rather than clipping (**INV-7** §7c.4). Same pattern as **C-64**.
> 3. **$u_s(t)$ is underdetermined.** Coulomb gives a *criterion*, not a *magnitude* — $u_s = 0$ while
>    `DCFS > 0` is equally "correct", so `T_ff` is unbounded. **Required: a slip law** — rate-and-state
>    (Dieterich healing $r = \dot u_s / A_0 \cdot s_n' e^{-B u_s}$) for diagenetic seals; rate-and-state
>    with state evolution for reactivating faults.

**Visualisation:** planar fracture meshes semi-transparent inside the matrix volume; `CoulombFS` and
`DCFS` colour maps along fault planes with `DCFS > 0` zones highlighted; `w_f(t)` evolution through the
tensile singularity; the `Seal`→`Conduit` transition timestamp.

### Domain 8 — Complex wellbore, 1D drift-flux and Schwarz sub-domains

**Time series:** `P_bhp`, `P_whp`, `T_thp`; `Q_o, Q_w, Q_g, Q_inj`; `WCUT`, `GOR`, `GCUT`.
**Pattern sizing:** `N_pat = Area / 40` acres; `N_inj = N_prod = N_pat`. ⚠️ **These are scenario
inputs, not constraints** — never imposed silently (**C-60**). 🔴 **`q_well <= 1000` BOPD is REMOVED**
under **INV-7** (**CONF-07 / C-56**): measured **11× to 1101× below** the Darcy limit of the stated
pattern, so it is a round-number cap, not a physics limit. Rate is an input; the consequence is the
engine's job to compute.
> The 20 000–60 000 MSCFD field rate is **REMOVED** per **CONF-54** — D5 §1.1 cites the same range as
> the *anti-pattern* signature of a clamped rate.
**Breakthrough:** `t_bt = V_p,pattern*(1 - S_wi) / (K_koval * q_inj,pattern)`.
> **CONF-15 persists** — `(1 - S_wi)` ignores `S_or` and gas saturation; no unit basis for `q_inj` or
> `K_koval`.
**CO2 GOR** grows to **5 000–25 000 SCF/STB** at mature flood.
**1D drift-flux segments:** `P_tubing(z,t)`; phase velocities `u_alpha(z)`; gas drift `u_dg`; holdup
`alpha_g(z,t)`; **ICV/ICD opening `theta(t)` in [0,1]**.
**Skin decomposition** — additive, with a geometry-dependent floor (see §8.5):

```
S_tot = S_mech + S_perf + S_cake + S_asph
```

**Schwarz 3D sub-domain:** cement-rock micro-annulus gap `w_micro(z,t)` (µm); cement sheath stress tensor
`s_sheath(r,theta,z,t)`.

**Visualisation:** 1D `P_tubing(z)` and `alpha_g(z)` from bottomhole to surface; 2D/3D cement-ring stress
cross-sections; stacked skin-component bar chart (S1's worked example: `S_tot = 8.5` = Mech 2.1 + Perf 1.2
+ Cake 0.7 + Asph 4.5).

### Domain 9 — Global diagnostics and convergence

Global cumulatives: `N_p, W_p, G_p` and injection `W_i, G_i`.
Mass-balance residual norms `||R_m||_2`, `||R_m||_inf`.
Sweep efficiencies `E_v, E_a, E_m`; **`RF = E_v * E_a * E_m`**.
Convergence history: `dt^n`, `N_Newton^n`, line-search damping `lambda_k`.
**Run validity:** `ValidityClass` ∈ `{Validated, ConvergedOutsideEnvelope, Unsolvable}` plus the
`ValidityWarning` list — required by **INV-7** (§7b.2, **C-59**).

> 🔴 **`ValidityClass` is what makes an unconstrained engine usable for research.** Tier 2 runs complete
> and emit the absurd answer the researcher asked for, *plus* the knowledge that it lies outside the
> V&V-verified envelope. Tier 3 emits **nothing** — a run that cannot converge produces no artifact
> (**INV-1**), not a plausible-looking file.

**Visualisation:** Havlena-Odeh plots; Newton residual-norm vs iteration per timestep.

> **The economics half of Domain 9 is PURGED** — see §8.

---

## 5. Data persistence stack (S4)

| Layer | Technology | Role |
|---|---|---|
| **1. In-memory substrate** | `petekIO` GeoData slabs | Contiguous `f64` for `P(x,t)`, `S_alpha(x,t)`, `z_i(x,t)`, `s_ij(x,t)` — zero heap allocation |
| **2. Spatial grid and 3D time-series** | **HDF5 / VTK-HDF** (`.h5`, `.vtu`), **RESQML 2.0.1 / GRDECL**, `.pproj` | 3D meshes, 4D output fields, visualisation |
| **3. Embedded tabular and metadata** | **DuckDB / SQLite (libSQL)**, **Apache Parquet / Arrow** | Well schedules, rate time series, RTA/DCA history, run logs |
| **4. Validation benchmarks** | SPE series 1, 3, 5, 9, 10, 11 | Solver accuracy vs. industry reference solutions |

### 5.1 Complementary to PostgreSQL + pgvector, not a replacement

| Concern | Store | Why |
|---|---|---|
| **Simulation output** (S4 layers 1–3) | HDF5/VTK-HDF + Parquet + DuckDB | **Open, standard, tool-neutral.** The Python satellite reads them with existing deps. No coupling |
| **Project / reservoir definition** | **PostgreSQL + pgvector** | Transactions, versioning, concurrency, vector search |
| **Training pairs** | **separate pgvector / Parquet store** — ruled 08-10-2026, see §8.1 | Must be decoupled from failed runs |

> This is the right split and it dissolves the P1 separation tension cleanly. The engine writes *files*,
> the project lives in *PostgreSQL*, and neither engine imports the other. Satellite tools read HDF5 with
> `h5py` — **already a dependency**.

### 5.2 Python satellite dependencies — measured 08-10-2026

Checked in the project `.venv`:

| Package | Status | Needed for |
|---|---|---|
| `h5py` **3.16.0** | installed | reading Layer 2 — **already a dependency** |
| `numpy` **2.5.3**, `pandas` **3.0.5**, `scipy` **1.18.1**, `sklearn` **1.9.1** | installed | analysis, ML |
| `pyarrow` | **MISSING** | reading Layer 3 Parquet |
| `duckdb` | **MISSING** | Layer 3 SQL |
| `torch` | absent | expected — the neural surrogate is *yet to be developed* |

> **Owner ruling 08-10-2026: these are leftovers, not gaps.** Most satellites are developed *after* the
> compositional engine, so the dependencies are added then. **Not a P1 blocker.** Recorded, not removed.

---

## 6. Volume reality check — measured 08-10-2026

Per-cell float fields per timestep, from the nine domains at `N_c = 6`: **116**.

| Case | Cells | Per timestep | 100 timesteps |
|---|---|---|---|
| SPE 5 (7x7x3) | 147 | 0.07 MB | 0.01 GB |
| 200 k | 200 000 | 92.8 MB | 9.3 GB |
| **SPE 10 (D5 Level 5)** | 1 100 000 | **510.4 MB** | **51.0 GB** |

**Ruled remedies (08-10-2026):**

| # | Remedy |
|---|---|
| 1 | **Chunked HDF5 with `zstd` compression** |
| 2 | **Active-frame RAM cache** — the UI data layer loads and retains active 3D timestep slabs in contiguous RAM rather than streaming from disk during interactive scrubbing |
| 3 | ✅ **KL/PCA reduced basis** on intermediate micro-step snapshots; full fields retained at monthly/yearly checkpoints. **5–10× disk reduction** |
| 4 | ✅ **Master grid + sparse deltas**: geometry once in `/Geometry`; temporal deltas only where `abs(dP) > epsilon`; static caprock/aquifer cells never written. **Up to 70 % smaller** |

**Verified 08-10-2026:**

| Active fraction | Per timestep (SPE-10, all 116 fields) |
|---|---|
| 100 % of cells | 510.4 MB |
| 30 % active | 153.1 MB |
| 5 % front-tracking | 25.5 MB |

> 1.1 M cells x 1 field `f32` = 4.4 MB. Reading one field per 16.6 ms frame is **not viable from disk** —
> the caching strategy in §3.1 must be implemented, not just documented.
>
> **The KL rationale is stronger than stated.** `S_o + S_w + S_g = 1.0` is an **exact** rank deficiency:
> three saturation fields span a two-dimensional subspace, so KL captures it in one mode with **zero
> residual** — not an approximation. Further derived-not-stored quantities among the 116:
> `RF = E_v*E_a*E_m`; `Pbar_res_eff`; `K_i = y_i/x_i`; `L/F = 1 - V/F`;
> `s'_ij = s_ij - alpha_B*P*delta_ij`; `phi_tot = phi_m + phi_f`; `K = H*E_eff`.
> **The 116 are not 116 independent signals.**

---

## 7. Inherited conflicts — measured in S1

**S1 was written as an *output* specification and did not re-audit the physics it embeds.**

### 7.1 CONF-01 closure defect — PURGED

S1 line 67 reprinted:

```
f_g(t) = 1 / (1 + ((1 - S_w - S_g)/(S_g - S_gc)) * (mu_g/mu_o))
```

**Ruling 08-10-2026: purge this line.** Enforce `f_g = K*S_g/(1 + S_g*(K-1))` or Corey phase mobility
ratios.

> **The action is right; the stated reason is not.** The sign was retracted by measurement — the two forms
> agree to 4 dp across the whole viscosity-ratio range. The real defect is the **closure**: no `S_or`,
> linear instead of Corey exponents, no water term, unguarded at `S_g -> S_gc`. Tracked as **C-37** and
> [`spec_defects.md`](spec_defects.md) §1. **CONF-01**.

### 7.2 CLOSED

| Conflict | S1 line | Ruling |
|---|---|---|
| **CONF-02** — inert HCPVI cap, `tau = 1.5` | 77 | ✅ **REMOVED** |
| **CONF-35** — net-utilisation floor | 83 | ✅ **CLOSED** by unit conversion — §7.3 |
| **CONF-54** — 20 000–60 000 MSCFD | 329 | ✅ **REMOVED** as a design target |
| **CONF-63** — Koval form | 71, 335 | ✅ **STANDARDISED** — §7.4 |
| **CONF-64 / 65** — economics + Python filenames | 390–414 | ✅ **PURGED** — §8 |

### 7.3 CONF-35 — closed with the basis stated

Ruling: *standard density at **60 °F and 14.7 psia** yields 1 MSCF CO2 ≈ 0.0519 metric tonnes.*

| Quantity | MSCF/STB | t/STB |
|---|---|---|
| **Floor** | 2.5 | **0.13** |
| **Benchmark** | 5 | **0.26** |
| **Benchmark** | 10 | **0.52** |

> The floor and the benchmark range are **separate quantities**, both correct under one conversion. My
> earlier framing of CONF-35 as a "2–4× contradiction about the same hard floor" was **wrong** — I compared
> a floor against a benchmark. **Withdrawn** (correction **C-38**).

### 7.4 CONF-63 — standardised, with the magnitude recorded correctly

**Ruled form:**

```
K_Koval = H * E_eff,   E_eff = (0.78 + 0.22 * M^0.25)^4
```

**Measured severity of the rejected linear form** (`E_disp = (3K² − 3K + 1)/K³`, `RF = E_disp*(1 − S_wi)`):

| M | K ratio (linear / Koval) | RF impact |
|---|---|---|
| 10 | **5.3x** | **3.3x** |
| 100 | 21x | 17x |
| 1000 | 60x | 57x |

> "Orders of magnitude" holds only above roughly **M ≈ 50**. At the realistic adverse ratio M = 10 the
> error is 5.3x in K and 3.3x in RF. Note `E_disp -> 3/K` saturates, so RF impact grows *sub*-linearly
> with the K error. The fix is clearly right; the magnitude language was overstated.

### 7.5 Still open

| Conflict | S1 line | Status |
|---|---|---|
| **CONF-07** — 1 000 BOPD clamp | 329 | ✅ **REMOVED under INV-7** — C-56. Measured 11×–1101× below the Darcy limit |
| **CONF-15** — `t_bt` uses `(1 − S_wi)` | 333 | Ignores `S_or` and gas saturation |

---

## 8. Economics PURGED from the engine (ruled 08-10-2026)

**The ruling is unambiguous:** *"Violates separation of concerns. The C++/Rust PDE core computes physical
conservation laws (`P, S_alpha, z_i, s_ij, Q_o, Q_w, Q_g`). Monetary evaluations belong strictly in the
Satellite Post-Suite. Fix: purge all monetary variables and Python filenames from S1."*

**The core's complete output contract is now strictly physical:**

```
Q_o, Q_w, Q_g, Q_inj, W_p, G_p, N_p, ||R_m||, E_v, E_a, E_m, RF
```

### 8.1 The clean two-output contract

```
                      IN-CORE PDE SIMULATION ENGINE
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
   3D/4D Volumetric Fields                     Well & Zone Time-Series
   [ HDF5 / VTK-HDF (.h5/.vtu) ]               [ Apache Parquet / DuckDB ]
               │                                           │
               ▼                                           ▼
   3D Viewport / ParaView                    SATELLITE ECONOMIC ENGINE
   (Spatial Rendering & Slices)              (NPV, NOCF, DCF, Cash Flows)
```

| Consumer | Reads | Never reads |
|---|---|---|
| 3D viewport / ParaView / geometry | **HDF5, VTK-HDF** | DuckDB, Parquet |
| **Satellite economic engine** | **DuckDB / Parquet only** | **never HDF5** |
| Well analytics, DCA/RTA | Parquet / DuckDB | HDF5 |

> **Why this is the only sound split.** Economics needs `Q_o(t), Q_w(t), Q_g(t), Q_inj(t), P_bhp(t),
> P_whp(t)` — **KB per timestep** in columnar form. HDF5 holds **510 MB per timestep** of 3D spatial mesh.
> Forcing the economic engine to parse gigabytes of spatial data to sum well rates is a guaranteed I/O
> failure. DuckDB executes zero-copy OLAP across thousands of timesteps in **sub-milliseconds**.

### 8.2 `training_pairs` — separate store, NOT the run HDF5

Ruling 08-10-2026:

> *"If a simulation run fails halfway due to physical divergence, the run HDF5 is marked `FAILED`. If
> training vectors are embedded inside the run file, partial non-converged states risk polluting ML
> surrogate training sets. Decoupling training vectors into `pgvector` ensures that **only validated
> states from `IMPLEMENTED` or `DEGRADED` runs are committed**."*

> This closes a real hole between **INV-1** and **INV-4**. If training data lived inside the run
> artifact, a run failing **mid-write** could leave partial non-converged states that a later harvesting
> step treats as samples.
>
> **Requirement (C-55):** the commit must be **atomic** with respect to the capability declaration. A
> sample cannot become visible before its run's `ModuleCapabilityState` is known to be `Implemented` or
> `Degraded`. If that ordering is not enforced, the hole reopens by a different route. Enforced in the
> **write path** (M5), not the UI.

### 8.3 Mid-year discounting — CONF-62, still open

S1 mandated `DF_y = (1+d)^-(y-0.5)`; the live engine uses **end-of-year**
(`surrogate_engine.py:649-650`). **Measured +4.88 % NPV** at `r = 0.10` over 15 yr. Adopting mid-year
**re-baselines every published result**. Now belongs to the economic engine and must be a **recorded
decision**, not a silent inheritance.

### 8.4 The economic engine is on the critical path for simulation learning

Full output enables **reinforcement / simulation learning** to tune the surrogate on full physics. The RL
loop needs a **reward** — naturally NPV. **Therefore the economic engine is required to close the learning
loop**, even though it is not part of the simulation engine.

> Economics is out of the *simulation* engine by **INV-3**, and in the *learning loop* by necessity.

### 8.5 CONF-68 — CLOSED: geometry-dependent skin floor with a stated margin

**Governing relation, Peaceman's well model:**

```
q = 2*pi*k*h / ( mu * [ ln(r_e/r_w) + S ] ) * (P_grid - P_bhp)
```

so the singular skin is `S_sing = -ln(r_e / r_w)`. **Ruled form:**

```
S_min = -ln(r_e/r_w) + dS_margin,    dS_margin ~ +0.50 to +1.00
```

**Verified 08-10-2026.** Substituting `S = S_sing + dS_margin` makes the Peaceman denominator exactly
`dS_margin`:

| Property | Result |
|---|---|
| Denominator | `dS >= 0.5 > 0` at **every** radius, so `J` **cannot** go negative — inflow cannot become outflow |
| `r_wa / r_e` | `= exp(-dS) = 0.607` — **constant across all radii**, so the guard is scale-invariant |
| Stimulation cap | `J / J_unskinned <= abs(S_sing) / dS` — at `r_e = 52.5` ft: ≤ 10x; at 1000 ft: ≤ 15.9x |
| Cost of the margin | **0.5 skin units** of wormhole stimulation benefit |

> **Why the margin matters.** A fixed floor of `-5.0` is safe only for `r_e > 52.5` ft. With
> `dS >= 0.5` the singularity is unreachable at **any** radius, and the margin puts a **principled
> ceiling** on the stimulation amplification instead of a round number.
>
> **Second consideration:** wormholing legitimately produces large negative skin — that is the point.
> **The floor and the stimulation cap are the same parameter.** Choose it against a measured stimulation
> target.

---

## 9. Scope consequence — stated plainly

**The master map (S2) is the full 3D THMC simulator**, not a compositional engine:

| Layer | Content |
|---|---|
| 1 | Discretisation — `petekIO`, corner-point GRDECL, **MPFA-O**, TVD (Superbee, van Leer), WENO3/WENO5 |
| 2 | **Fully coupled THMC** — thermal, hydrodynamics, **Biot FEM Taylor-Hood (P2-P1) + fixed-stress split**, **Lasaga geochemistry**, wormholing, clay swelling, chemo-mechanical softening, 5-state trapping |
| 3 | Thermodynamics — PR, SRK, **PC-SAFT, CPA**; hyper-dual AD; AOP; **van der Waals-Platteeuw** hydrates; **GPU flash** (wgpu/CUDA) |
| 4 | Wells and fractures — P3D/Planar-3D/DFN, proppant, Forchheimer beta, EDFM NNC, drift-flux wellbore, **Schwarz 3D sub-domains**, **remediation and well kill** |
| 5 | Core — Rayon zero-alloc assembly, **adjoint gradients**, faer / russell_sparse / cuDSS, L-stable + soft-start |
| 6 | Satellite — EOS regressor, SCAL fitter, Amaefule rock typing, upscaler, procedural geology, Havlena-Odeh, DCA/RTA, **ES-MDA + LHS** |
| 7 | V&V, output schema, **4D visualisation** |

> **This is not 8–11 person-months.** The earlier `build_plan.md` estimate covered the compositional
> subset with geochemistry, THMC, fractures, faults, GPU and adjoint deferred. **That deferral is now
> void.**
>
> **The tractable path** — recorded in [`build_plan.md`](build_plan.md):
>
> **Define the output schema completely now, because it is an interface.** Interfaces are cheap to define
> in full and expensive to change later. Then implement physics **incrementally**, with each run
> **declaring which domains it actually populated** (**INV-6**). A hydrodynamics-only M3 emits Domains 1,
> 2, 3 and 9 (physical part) and declares Domains 4–8 absent — not because they failed, but because they
> are not yet implemented.

---

## 10. Good decisions in S1, recorded so they are not lost

| Decision | Why it is right |
|---|---|
| Engine emits **all raw values**; satellite interprets and formats | No presentation logic in the numerical core; the Zero-Bloat Core principle holds. Reformatting never needs a core change |
| **Standard open formats** (HDF5/VTK-HDF, Parquet, RESQML/GRDECL) | The Python satellite reads them with `h5py`, which is **already a dependency**. Zero coupling in P1 |
| Complete schema up front, physics incrementally | Interfaces are cheap to define and expensive to change. Avoids re-running history |
| `float32` for fields | Half the I/O of `f64`; adequate for display and most analytics |
| Dual saturation split `phi_m/phi_f/phi_tot` | The design set never distinguished them; its own DP/DP section assumed the distinction |
| **Full 6-component tensor** in output | Fixes the design set's diagonal-only payload (**CONF-41**) at the schema level |
| Siderite + ankerite added | Real CO2 sinks missing from the design set — and why `Fe2+` is needed (**CONF-67**) |
| **Drucker-Prager** alongside Mohr-Coulomb | Better for frictional rock; avoids unbounded shear strength at high confinement |
| Additive **skin decomposition** | Distinguishes mechanical damage from asphaltene/wax deposition — S1's example correctly shows a `S_asph = 4.5` skin that calls for **solvent wash, not re-frac** |
| `float32[N_cells][6]` tensor layout | Directly renderable via instanced ellipsoid glyphs |
| Three temporal resolutions | Separates convergence diagnostics from engineering series from accounting |
| **Two-output contract** (ruled) | Spatial and time-series consumers never fight over the same file |
| **`training_pairs` externalised** (ruled) | Makes the dataset commit boundary explicit and enforceable against **INV-1** |
| **KL/PCA + master grid + sparse deltas** (ruled) | 5–10x storage reduction and a compressed RL observation space |

---

## 11. Corrections log

Per the owner decision — `3D_THMC_docs` are references; the **wiki** carries the corrected
specification — every deviation is logged in [`spec_corrections_log.md`](spec_corrections_log.md).
S1's contributions are recorded there as corrections **C-39 … C-54**, including **two of my own framings
withdrawn by measurement** (**C-37**, **C-38**).