# Satellite Toolkit and Synthetic Geology — D6 + D7

Sources:
- **D6** — `Специфікація автономного сателітного інструментарію та калькуляторів (Pre- & Post-Simulation Utilities Suite).md` (399 lines)
- **D7** — `Специфікація генерації синтетичних геологічних даних та інтеграції з сателітними калькуляторами.md` (177 lines)

**Design intent only.** D6 and D7 carry a duplicated title suffix — see `doc_inventory.md` F-07.

---

## 1. D6 — integration stack

Five satellite components:

| Component | Input | Output | Purpose |
|---|---|---|---|
| **PVT & EOS Regression Fitter** | Lab experiments (CCE, DL, Swelling) + component composition | `PVT_EOS_DTO` (black-oil tables, PR/SRK parameters) | Thermodynamic calibration; tables transferred into the computational grid before the run |
| **Plurigaussian & SCAL Module** | Quality-index values (FZI, RQI), indicator variograms, truncation rules | `PGS_Grid_DTO` (3D facies grid, $k_r(S)$, $P_c(S)$, hysteresis coefficients) | Stochastic geo-domains, capillary-curve scaling, rel-perm modelling |
| **Grid Validation & Translator** | `GRDECL`, `RESCUE`, `petekIO` | `Grid_Validation_DTO` (corner-point / unstructured FVM grid) | Grid conversion and validation; fracture-aperture computation; initial-state translation |
| **Coupled Material Balance & DCA** | DTO of time series (rates, reservoir pressures, displaced volumes) | Diagnostic Summary DTO, dynamic $\bar{P}_{res,eff}$, decomposed RF indicators | Post-sim reservoir-energy recomputation; DCA; production-weighted pressure for back-correcting miscibility |
| **CO₂ Storage & Economics Suite** | Dynamic production/injection profiles, gas-plant parameters | `Economic_DCF_DTO` (NPV, NCF, CAPEX/OPEX, CO₂ partitioning by mechanism) | DCF NPV; gas-breakthrough financial-risk analysis; CO₂-capture efficiency |

### 1.1 Execution modes

1. **Asynchronous batch — "Pre-Simulation Dynamic Batching".** PVT, SCAL, PGS and Grid Validation run
   as isolated preliminary procedures *before* invoking the core; results serialised into FVM input manifests.
2. **Synchronous reverse — "Mid-Simulation Dynamic Callbacks".** During history-matching closed loops
   or GEP surrogate generation, post-simulation modules are invoked through the solver's internal
   **IPC / gRPC interface after each integration stage**, allowing real-time recomputation of
   $\bar{P}_{res,eff}$ and correction of RF **without restarting**.

> ⚠️ **CONF-27.** Mode 2 is **architecturally incompatible** with the Zero-Bloat Core premise and with
> D1 §10.1's zero-allocation requirement: an in-solver IPC call inside the integration stage is exactly
> the "auxiliary services inside the hot loops" that D1 §1.1 declares the primary cause of cache
> thrashing. D1 and D6 also disagree on whether economics/miscibility is on-core or off-core
> (see CONF-04).

### 1.2 Serialisation and the DTO bus

Data transferred via a standardised **DTO bus**, in **JSON / Protobuf**, giving full module decoupling
and cross-version compatibility.

> ⚠️ **CONF-28.** **No `.proto` file, no field numbers, no service or method names, no versioning or
> negotiation scheme, no error/status codes, no checkpoint or restart format** — "checkpointing" and
> "restart" appear nowhere in D6. The hand-off contract is not implementable as written.

### 1.3 DTO schemas (as printed)

```yaml
PVT_EOS_DTO:
  fluid_id: "ST_FLUID_001"
  eos_type: "PR_1978_VT"
  components: ["C1","C2-C3","C4-C6","C7+","CO2","N2"]
  critical_properties:
    Tc_K:             [190.56, 305.32, 412.65, 540.20, 304.13, 126.20]
    Pc_MPa:           [4.599, 4.872, 3.796, 2.740, 7.377, 3.390]
    acentric_factor:  [0.011, 0.099, 0.195, 0.385, 0.225, 0.037]
    volume_translation_c: [0.0, -0.0024, 0.0051, 0.0182, 0.0012, 0.0]
  bip_matrix: [[0.0, 0.02, 0.03],
               [0.02, 0.0, 0.01]]
  black_oil_tables:
    pressure_grid_bar: [100.0, 150.0, 200.0, 250.0]
    Rs_m3m3:           [45.2, 72.1, 98.4, 115.0]
    Bo_m3m3:           [1.12, 1.21, 1.30, 1.35]
    mu_o_cP:           [1.85, 1.42, 1.15, 0.98]
```

```yaml
Grid_Validation_DTO:
  grid_id: "MESH_1000ACRES_25PATTERNS"
  dimensions: {nx: 100, ny: 100, nz: 20}
  active_cells_count: 184500
  validation_status: { non_orthogonality_max_deg: 38.4,
                      has_negative_volumes: false,
                      min_porosity: 0.02, max_porosity: 0.31 }
  geomechanical_parameters: { barton_bandis_b0_mm: 0.25,
                              joint_closure_Vm_mm: 0.18 }
```

```yaml
Economic_DCF_DTO:
  project_id: "EOR_CO2_FLOOD_01" ; pattern_count: 25
  wells: {n_inj: 25, n_prod: 25}
  capex_breakdown_usd: { drilling_conversion: 62500000.0,
                         gathering_network:    9250000.0,
                         gas_plant_recycle:   14200000.0,
                         surface_facilities:   5000000.0,
                         total_capex_year0:  90950000.0 }
  discount_rate: 0.10
  yearly_profiles:
    year: [1,2,3]
    net_operating_cash_flow_usd: [12400000.0, 28500000.0, 31200000.0]
    dcf_usd:                    [11822830.0, 24698500.0, 24584200.0]
    cum_npv_usd:                [-79127170.0, -54428670.0, -29844470.0]
```

> [!CAUTION]
> **CONF-29 — `bip_matrix` is dimensionally wrong.** It is **2 × 3** for a **6-component** system.
> A valid symmetric 6-component BIP set requires **21** unique $k_{ij}$; **15 are absent**, and the
> trailing entry has no symmetric counterpart. Any PR/SRK mixture built from this DTO is underdetermined.
>
> **CONF-30 — `Economic_DCF_DTO` contradicts its own CAPEX band and its own DCF arithmetic.**
> - Line items sum correctly: $62.5 + 9.25 + 14.2 + 5.0 = 90.95$ M ✓
> - The compressor term 14.2 − 5 = 9.2 M implies $Q_{recycle,peak} \approx 17\,600$ MSCFD from
>   $10\text{M}\times(Q/20\,000)^{0.65}$ ✓ (consistent)
> - **Year 2 DCF is wrong** (measured 07-10-2026): $28.5\text{e}6 / 1.1^{1.5} = 24\,703\,349$,
>   stated $24\,698\,500$ — a **$4 849 (0.020 %) discrepancy**, roughly **485×** the document's own
>   required **$0.01** agreement (§6).
> - Year 1 and Year 3 match to rounding; `cum_npv` ladder is exactly consistent with the *stated* DCFs.
> - **$90.95$M exceeds D6's own stated "realistic $40M–$75M" band for 1 000 acres / 25 patterns**,
>   and contradicts D2 §6.2 criterion 4 (same band). See **CONF-20**.

---

## 2. `petekIO` — what is actually specified

The **complete** set of `petekIO` mentions across all 8 documents:

1. D1 §1.1: named as the **"Flat Binary Substrate"** interface between satellite and core.
2. D1 §2.1: named as the container of **Contiguous Flat Memory Slabs**.
3. D6 §1.1 / §3.1: one of **three** accepted geometric input formats (`GRDECL`, `RESCUE`, `petekIO`) for
   the Grid Validation & Translator.

The only property attributed to it anywhere: the unified parser "parses text and binary sections",
recovering grid topology, fault geometry and cell-to-cell transmissibility multipliers.

> [!CAUTION]
> **CONF-30b.** **Never specified**: magic bytes, header struct, byte order, slab typing/dtype, chunk or
> offset table, alignment contract, schema or format version, extension. The format is named in three
> places and defined in none. The satellite doc's own parser pipeline
> (`[petekIO / GRDECL / RESCUE] → Unified Parser → FVM Input Grid DTO`) cannot be implemented.

---

## 3. D6 — pre-simulation tools

### 3.1 Format converters

Exactly **three**: `petekIO`, `GRDECL`, `RESCUE` → `FVM Input Grid & Properties Structure`.
**ECLIPSE, VIP, OPM, GOCAD are not mentioned anywhere** in the set.

### 3.2 PVT recovery and fluid manipulation (§2.1)

**Black-oil correlations:** bubble-point $P_{sb}$ and solution GOR $R_s$ from **Standing** and **Glaso**;
viscosities from **Beggs–Robinson** (dead + saturated, with high-pressure correction) plus gas-viscosity
correlations; $B_o$, $B_g$ from computed $R_s$, gas density, reservoir temperature.

**EOS Regression Fitter** fits cubic-EOS parameters to three lab experiment types:
1. **CCE — Constant Composition Expansion** → $c_o$, $P_{sb}$, total fluid volume
2. **DL — Differential Liberation** → gas-release rate, $R_s$, $B_o$ dynamics, residual-oil density
3. **Swelling Test** → oil volume and $P_{sb}$ change under hydrocarbon or CO₂ solvent injection

**Volume Translation (VT):** temperature-dependent volume correction; basis **17 pure hydrocarbons**
from `CH4` to `n-C40H82` including cycloalkane and aromatic classes; source **Baled et al., 2012**;
HTHP envelope **7–276 MPa**, **278–533 K**.

**BIPs $k_{ij}$:** regressed from experimental VLE with classical **van der Waals** or **Huron–Vidal**
mixing rules. Where lab data are missing, a **QSPR** approach models $k_{ij}$ from component structural
descriptors against a database of **> 900 binary systems**.

**Group contribution & MMP/IFT:** **Joback-Reid** and **PPR78** for $T_c$, $P_c$, $\omega$ of heavy
pseudocomponents; automated **MMP** and **IFT** calculators for CO₂–hydrocarbon systems.

**Equations of state (verbatim):**

$$\text{PR:}\quad P = \frac{RT}{v-b} - \frac{a(T)}{v(v+b)+b(v-b)}\qquad \text{SRK:}\quad P = \frac{RT}{v-b} - \frac{a(T)}{v(v+b)}$$

Both systematically **overestimate liquid-phase molar volume**; correction
$v_{corr} = v_{EOS} - c(T)$ with $c(T)$ correlated against **$(M\omega)^{-1}$**.
Achieved **MAPD** on liquid density: **1–2 % VT-SRK**, **1–4 % VT-PR**.

> ⚠️ **CONF-31.** **No coefficient is ever given** for Standing, Glaso, Beggs–Robinson, Joback-Reid,
> PPR78, Huron–Vidal or QSPR — all are named only. `M` (molar mass) and `ω` (acentric) appear in
> $(M\omega)^{-1}$ but are never defined as symbols.

### 3.3 Petrophysics, geostatistics, SCAL (§2.2)

**Rock typing (Amaefule):** $RQI = 0.0314\sqrt{K/\phi}$, $\Phi_z = \phi/(1-\phi)$, $FZI = RQI/\Phi_z$.

**SCAL relative permeability — Corey / Brooks-Corey:**

$$S_w^* = \frac{S_w - S_{wi}}{1 - S_{wi} - S_{or}},\qquad k_{rw} = k^0_{rw}(S_w^*)^{n_w},\qquad k_{ro} = k^0_{ro}(1-S_w^*)^{n_o}$$

**Leverett J-function:** $J(S_w) = \dfrac{P_c(S_w)}{\sigma\cos\theta}\sqrt{\dfrac{K}{\phi}}$

**Hysteresis:** **Killough** and **Carlson** C¹-smooth drainage/imbibition switching (WAG, CO₂).
$S_{gr} = S_{gr,max}\cdot S_{gi}/(1 + C\,S_{gi})$ with $C = 1/S_{gr,max} - 1/S_{g,max}$.

**Plurigaussian Simulations (PGS) — 5 steps:**

```
Step 1 Truncation Rule (Flag)
Step 2 Truncation Thresholds          <- Facies proportions P_i(x)
Step 3 Gaussian-field variograms      <- Indicator variograms γ_ij(h)
Step 4 Gibbs Sampler                  <- sample points u_α
Step 5 Grid simulation & mapping      -> 3D DTO of geo-domains
```

1. **Flags.** Geometric-topological rule in the space of two or more latent Gaussian fields
   $Z=\{Z_1(x),Z_2(x)\}$, fixing allowed and **forbidden** geological contacts (e.g. sandstone may
   contact siltstone but **never** shale).
2. **Thresholds.** $P_i(x) = \int_{D_i} g(z_1,\dots,z_N)\,dz_1\dots dz_N$ from spatial proportions of
   each of $M$ facies.
3. **Variograms.** Experimental indicator cross-variograms
   $\gamma_{ij}(h) = C_{ij}(0) - \tfrac{1}{2}[C_{ij}(h)+C_{ij}(-h)]$ from well facies, with
   $C_{ij}(h) = \text{prob}\{G(x)\in D_i,\ G(x+h)\in D_j\}$; **Hermite polynomial expansion** converts
   indicator covariances into latent Gaussian-field covariances.
4. **Gibbs sampler.** Generates field values at $u_\alpha$ satisfying both the covariance structure and
   membership in $D_{i(\alpha)}$; iterative point selection, **Simple Kriging**, acceptance of $Z_\alpha\in D_{i(\alpha)}$.
5. **Grid simulation.** **Spectral simulation**, **SGS**, or **Turning Bands** at grid nodes, then the
   Flag rule converts continuous values into categorical facies.

### 3.4 Geomechanics, wells, geometry (§2.3)

- **Dynamic → static elastic moduli** from acoustic logs: $E_{stat}=a E_{dyn}^b$, $\nu_{stat}=c\nu_{dyn}$.
- **Barton–Bandis dynamic fracture aperture:** $b_f = b_0/(1+\sigma_n'/V_m)$.
- **Karakas–Tarik perforation skin:** $S_{dp} = S_h + S_v + S_{wb}$.
- **VFP tables** for tubing: pressure-loss functions $P_{wf}(q_o, WCR, GOR, P_{tf})$.
- **GRDECL validation:** non-orthogonality **$\theta > 45^\circ$ ⇒ warning**; skewness coefficient and
  cell volume computed, **$V_{cell}\le 0$ blocks execution**.

### 3.5 Upscaling (§2.4)

**Standalone Flow-Based Tensor Upscaler:** solve local single-phase flow $\nabla\cdot(K(x)\nabla P)=0$
inside each upscaled block with **periodicity** or **isolated-face** boundary conditions;
$K_{eff,ij} = \langle q_i\rangle/\langle\nabla P_j\rangle$.

Pseudo relative permeability $k_r^*(S)$ via **Kyte–Berry** or **Stone** dynamic scaling.

### 3.6 DTO validation — four-level cascade (§3.2)

`[DTO input data]` → **1. porosity/permeability ranges** → **2. monotonicity of $P_c(S)$ and $k_r(S)$**
→ **3. boundary conditions and reservoir limits** → **4. well synchronisation ($N_{inj}=N_{prod}=N_{pat}$)`
→ `[Validated FVM DTO]`

| Stage | Check |
|---|---|
| 1 | Reject non-physical/negative porosity $0<\phi<0.6$; permeability $K>0$ mD |
| 2 | First-derivative signs $dP_c/dS_w\le 0$, $dk_{rw}/dS_w\ge 0$, $dk_{ro}/dS_w\le 0$; oscillations or flat segments corrected by **C¹-smoothing** |
| 3 | Closed outer boundaries; correct **ACTNUM** (active/inactive) assignment; edge **aquifers** present |
| 4 | For areas **> 40 acres**: auto-patterning $N_{pat}=\max(1,\text{round}(\text{Area}/\text{spacing}))$; validator **enforces** $N_{inj}=N_{prod}=N_{pat}$ |

> ⚠️ **CONF-32.** Stages 2 and 4 **silently mutate** the input — C¹-smoothing rel-perm curves, and
> auto-inserting patterns and well counts. This is *remediation presented as validation*, and it means
> a "validated" DTO may not be the DTO the user supplied. No audit trail or diff is specified.

### 3.7 Initialisation (§3.3)

1. **FWL** — the point where $P_c(S_w)=0$.
2. **Capillary–gravitational distribution** above FWL: $P_c(S_w(z)) = (\rho_w-\rho_o)\,g\,(z_{FWL}-z)$.
3. **Connate-water closure** — cells high above FWL fixed at $S_{wi}$.

---

## 4. D6 — post-simulation tools

### 4.1 Sweep efficiency and recovery (§4.1)

$$RF = E_v\cdot E_a\cdot E_m$$

**Koval fractional-flow hard cap:** $RF(t)\le RF_{ult}(\bar{P}_{res,eff})(1-e^{-\text{HCPVI}/\tau})$,
$\tau = 1.5$. → **CONF-01, CONF-02**.

**Documented failure case prevented:** declaring **> 41.8 % OOIP after only 0.18 HCPVI**; with
insufficient injection, recovery is physically bounded, e.g. **≤ 15 % OOIP**.

**Breakthrough time:** $t_{bt} = V_{p,pattern}(1-S_{wi})/(K_{koval}\,q_{inj,pattern})$ → **CONF-15**.

### 4.2 DCA / RTA (§4.2)

**Arps:** $q(t) = q_i/(1+b D_i t)^{1/b}$ — exponential $b=0$, hyperbolic $0<b<1$, harmonic $b=1$.
**Duong** for ultra-low-permeability reservoirs. **SEPD:** $q(t) = q_i e^{-(t/\tau)^n}$ — conservative,
physically grounded EUR forecasts "without unjustified late-life rate inflation".

> ⚠️ **CONF-33.** D6 explicitly rejects unphysical late-life **rate inflation**, then adopts
> **SEPD**, whose defining property is rate *inflation* relative to Arps late time (the stretched
> exponential decays more slowly than exponential at long times). The stated rationale contradicts the
> stated model. This is the exact shape of the live **CRIT-19 `gravity_factor` ±20 % fudge** defect —
> a decay-shape fudge.

### 4.3 Material balance diagnostics (§4.3)

**Havlena–Odeh** linearised MBE: $F = N(E_o + m E_g + E_{fw}) + W_e$ — $F$ cumulative withdrawal,
$N$ OOIP, $m$ gas-cap/variable-drive ratio (undefined in the text), $W_e$ aquifer water influx.

**Dynamic production-weighted pressure:** $\bar{P}_{res,eff} = \sum_t P_{res}(t)q_o(t)\Delta t \big/ \sum_t q_o(t)\Delta t$
— fed **back** into the pre-simulation block to recompute the miscibility stage and $\omega$, removing
recovery mis-estimation when pressure falls below MMP.

**Mass-conservation check:** $|M_{init} + M_{inj} - M_{prod} - M_{rem}| \le 0.001\,M_{init}$; deviation
above **0.1 %** signals numerical dispersion or boundary-condition error.

### 4.4 History matching and GEP (§4.4)

**Metrics:** RMSE and a composite index

$$\text{HM Mismatch Index} = \sum_i w_i\sqrt{\frac{1}{N}\sum_{t=1}^{N}\left(Y^{obs}_i(t) - Y^{sim}_i(t)\right)^2}$$

with $Y$ = oil rate, gas rate, water rate, reservoir pressure, and weights $w_i$.

> [!CAUTION]
> **CONF-34.** **No ensemble method is named in D6.** ES-MDA appears only in D1 §11.3 as a one-line
> assertion; D6 gives only RMSE / Mismatch Index. No $\alpha_i$ inflation schedule, no covariance update,
> no assimilation iteration count, no convergence criterion.

**GEP pipeline:**

```
[GEP Chromosome DTO] (linear genes: Head + Tail + Domain Dc)
   -> translation / K-expressions -> [Expression Trees (ETs)]
   -> (mutation, transposition, recombination) -> [Surrogate NPV / RF evaluation]
```

Chromosomes are fixed-length linear symbol strings; each gene splits into **head `h`**, **tail `t`**
and constant domain `D_c`; tail length `t = h(n-1)+1` for max arity `n`; `D_c` sits immediately behind
the tail with length equal to `t`, holding `?` symbols indexing an array of **Ephemeral Random Constants**.

Expression: read left-to-right per **K-expressions (Karva)**; `?` symbols in ETs replaced top-down,
left-to-right by `D_c` values; `D_c` indices decoded into array `A`.

Genetic operators:
1. **Mutation** — any head symbol → function or terminal; any tail symbol → **terminals only**; any `D_c` symbol → a new constant index
2. **Transposition** — **IS**, **RIS**, gene transposition
3. **Crossover** — 1-point, 2-point, gene recombination

Multigene chromosomes with linking functions: **addition, multiplication, IF, OR**.

### 4.5 Production, recycle and CO₂ partitioning (§4.5)

**Dynamic gas fraction:** $f_g(t) = \dfrac{1}{1+\left(\dfrac{1-S_w-S_g}{S_g-S_{gc}}\right)\dfrac{\mu_g}{\mu_o}}$ → **CONF-01**.

Stated post-breakthrough ranges: produced-gas share rises to **35–60 %** of injected volume;
**CO₂ recycle reaches 30–55 %** of gross injection. **$GOR_{co2}(t) = q_{co2,prod}(t)/q_o(t)$ grows to
5 000–25 000 SCF/STB.**

**Four trapping mechanisms:** structural/stratigraphic · residual/capillary (**Killough**) · dissolved
(EOS phase equilibria) · mineralised (geochemical reaction with rock).

**Net Utilisation Floor:** miscible tertiary displacement requires **Net Utilisation ≥ 2.5 MSCF/STB
(≥ 0.12 t/STB)**; the stated industrial benchmark is **0.25–0.50 t/STB (5–10 MSCF/STB)**.

> ⚠️ **CONF-35.** The floor is stated as **≥ 0.12 t/STB** in D6 §4.5 but **≥ 0.25–0.50 t/STB** in
> D1 §6.3, D7 §4.4 and D8 §2.4. A 2–4× disagreement about the same hard floor, across the set.

### 4.6 Economics (§4.6) — see §5 below.

**Reporting:** the only export artefact named anywhere is **`cash_flows_yearly.csv`**, with required
agreement of **$0.01** across surrogate engine results, final manifest reports and CSV tables.

> ⚠️ **CONF-36.** **No plotting, charting, report generation, or figure specification exists** in D6 —
> no Excel/PDF/HTML/Dashboard tooling. Yet D1 §1.1 lists "export of plots" as an explicit satellite duty.

---

## 5. D6 — economics, with arithmetic audit

### 5.1 Dynamic CAPEX

$$\text{CAPEX} = N_{inj}\times\$1.5\text{M} + N_{prod}\times\$1.0\text{M} + (N_{pat}\times\$250\text{k}+\$3\text{M}) + \$10\text{M}\times\left(\frac{Q_{recycle,peak}}{20\,000}\right)^{0.65} + \$5\text{M}$$

$Q_{recycle,peak}$ = peak gas-recycle capacity in **MSCFD**. Stated scaling: 1 000-acre field
(25 patterns) → realistic **$40M–$75M**.

**Audit:** the `Economic_DCF_DTO` example totals **$90.95M** — **$15.95M above its own band**, and
$15.95M above D2 §6.2 criterion 4's identical band. → **CONF-20 / CONF-30**.

### 5.2 Unified DCF NPV

1. $\text{NCF}_t = \text{Rev}_{oil}(t) + \text{Credit}_{storage}(t) - \text{Cost}_{purchased}(t) - \text{Cost}_{recycled}(t) - \text{OPEX}_{fixed/var}(t)$
2. Mid-year discounting: $\text{DCF}_t = \text{NCF}_t/(1+r)^{t-0.5}$
3. $\text{Cum\_NPV} = -\text{CAPEX}_0 + \sum_{t=1}^N \text{DCF}_t$

> ⚠️ **CONF-11.** **No hydrocarbon-gas revenue term.** Same structural gap as the live **CRIT-18**.
> Compare `core/engine_surrogate/surrogate_engine.py:636`:
> `annual_rev = (annual_oil_stb * oil_price) + (annual_stored_tonne * co2_storage_credit)` — with
> `annual_hc_gas_mscf` accumulated and published but never referenced in revenue.

> ⚠️ **CONF-37.** **No price deck.** No oil price, no escalation, no storage-credit price ($/tCO₂e),
> no OPEX split, no tax/depreciation/depletion, no terminal value, no IRR, no payback.
> `r = 0.10` appears only in the DTO example.

---

## 6. D7 — synthetic geological data generation

### 6.1 Use cases (§1.2)

1. **Algorithm testing and verification** — edge cases for stability, accuracy, convergence.
2. **Scientific research experiments** — sensitivity of CO₂-EOR algorithms to heterogeneity, barriers, anisotropy.
3. **Benchmark model creation** — standardised reference models for cross-simulator comparison.
4. **Data imputation** — plausible petrophysical fields in sparse-well zones.

### 6.2 Generator catalogue

| Generator | Purpose | Specified parameters |
|---|---|---|
| **Perlin / Simplex / Ridge noise + fBm** | continuous top/base surfaces, anticlines, domes, troughs | $Z(x,y)=\sum_{i=0}^{N-1}A_i\,\text{Noise}(f_i x, f_i y)$; $A_i = A_0 p^i$ (persistence $p$); $f_i = f_0 l^i$ (lacunarity $l$); ridge transform $1-|\text{Noise}|$ |
| **Voronoi tessellation + DFN** | tectonic blockiness, fault planes, karst cavities, DFN | $V_k = \{x\in D \mid \|x-s_k\| \le \|x-s_j\|,\ \forall j\ne k\}$; boundaries define fault planes; blocks displaced along boundary vectors for throw |
| **Wave Function Collapse (WFC)** | meandering channel systems, deltaic cones | tiles/voxels with admissible-adjacency constraint matrices; sequential entropy reduction guarantees continuity and no isolated fragments |
| **L-Systems** | fractal channel deltas, promontories, karst | grammar $G=(V,\omega,P)$, parallel rewriting |
| **Marching Cubes** | isosurfaces over voxel arrays | **256** standard topological configurations |
| **Dual Contouring** | sharp-edged surfaces (salt domes, cavities, unconformities) | Hermite data via **QEM**; preserves sharp tectonic edges without smoothing |
| **PGS** | categorical facies with forbidden contacts | 5 steps, §3.3 above |

> ⚠️ **CONF-38.** Only fBm and PGS carry any equations. **No defaults or ranges** for $N$, $A_0$, $p$,
> $l$, $f_0$; **no** seed count $m$; **no** voxel resolution or isovalue $c$; **no** tile catalogue,
> constraint matrices, axiom string, production rules, or backtracking policy; **no** fracture-segment
> statistics (length, aperture, orientation distribution) or stress-tensor constitutive relation.

### 6.3 Physically-coupled property cascade

**Porosity with compaction** (§3.1): $\phi(z) = \phi_0 e^{-c_c z}$ — $\phi_0$ = **surface** porosity,
$c_c$ = compaction coefficient, $z$ = burial depth.

**Rock typing / FZU (§3.2):**

$$RQI = 0.0314\sqrt{\frac{K}{\phi}},\qquad \phi_z = \frac{\phi}{1-\phi},\qquad FZI = \frac{RQI}{\phi_z}$$

**Inverse cascade** (FZI fixed per rock type):

$$K = 1014\,(FZI)^2\frac{\phi^3}{(1-\phi)^2}$$

> [!CAUTION]
> **CONF-39.** D7 states the derivation as "$1/0.0314^2 \approx 1012.7 \approx 1014$".
> $0.0314^2 = 9.8596\times10^{-4}$, so $1/0.0314^2 = 1014.2$, **not 1012.7**.
> The final constant **1014 is correct**; the intermediate value **1012.7 is arithmetically wrong**.
> Do not cite the derivation. Note also D6 attributes this to **Amaefule**; D7 names no author.

**Relative permeability and capillary (§3.3):** Leverett J-function

$$P_c(S_w) = \sigma\cos\theta\sqrt{\frac{\phi}{K}}\,J(S_w)$$

with **modified Corey** models for $k_{ro}(S)$, $k_{rg}(S)$, $k_{rw}(S)$, saturation normalisation anchored
to **$(S_{wi}, S_{or}, S_{gc})$** parameterised per FZI type.

> ⚠️ **CONF-40.** D7 names **three** permeability functions ($k_{ro},k_{rg},k_{rw}$) but supplies
> $(S_{wi}, S_{or}, S_{gc})$ — two residual/connate anchors and one **critical** anchor. There is **no
> $S_{gr}$** in D7 (it exists only in D6, as Killough/Carlson). No Corey exponents $n_{ro},n_{rw},n_{rg}$
> are given, no endpoint permeabilities, no $J(S_w)$ form or tabulated curve, no $\sigma$, no $\theta$.

**PR vs SRK comparison (§3.4):**

| Criterion | PR-EOS | SRK-EOS |
|---|---|---|
| $P_{sat}$ accuracy | high for heavy/medium hydrocarbon mixtures; computes phase boundary | high for light hydrocarbons, methane fractions, nonpolar gases |
| Liquid density | **underestimates by 5–15 %**, better phase boundaries | **substantially underestimates, up to 8–20 %** |
| Scope | oil reservoirs, CO₂ EOR, PVT analysis, **ORC** cycles | light gas, cryogenic separation, **BOG** |

**VT correction** at HTHP **up to 276 MPa and 533 K**; shift correlates with **$(M\omega)^{-1}$**.
BIPs $k_{ij}$ regressed for impurities **`CO₂`, `H₂S`, `N₂`** via van der Waals or Huron–Vidal.

**Archie (§3.6):** $F = a/\phi^m$, $RI = b/S_w^n$ — $a$, $b$, $m$, $n$ all unspecified.

**Gassmann (§3.6):**

$$\frac{K_{sat}}{K_m - K_{sat}} = \frac{K_{dry}}{K_m - K_{dry}} + \frac{K_f}{\phi(K_m - K_f)}$$

**Somerton (§3.6):** $\lambda_{eff} = \lambda_{dry} + \sqrt{S_w}\,(\lambda_{sat}-\lambda_{dry})$

### 6.4 GEP calibration (§3.5)

Same symbology as D2 §1.2 (see [`solver_and_numerics.md`](solver_and_numerics.md) §12): fixed-length
chromosomes, head `h` / tail `t` with $t = h(n-1)+1$, ORF/K-expressions (Karva), `D_c` domain of
length `t` immediately after the tail. Operators: point mutation (tail → **terminals only**), IS /
RIS / gene transposition, 1-point / 2-point / inter-gene recombination.

### 6.5 DTO hand-off and integration (§4)

**Satellite pre-sim utilities (§4.1)** — four named, none defined: Standalone Tensor Upscaler
(anisotropic tensor averaging $K_{ij}$ from the **fine geological grid** to the **coarser computational
grid**, preserving diagonal **and off-diagonal** components) · EOS Fitter · SCAL Fitter ·
Karakas–Tarik Skin Calculator.

**FVM input payload (§4.2)** — three groups, on **flat contiguous 1D arrays**, **zero-copy** via
C-API C++/Python binding:
1. **3D geometric grids:** vertex coordinates, cell volumes, face areas
2. **Void-filling arrays:** $\phi$, anisotropic permeability $(K_x, K_y, K_z)$, $S_{wi}$, $P_c$
3. **Vector fault fields:** tectonic displacement vectors, inter-cell-face transmissibility multipliers

> [!CAUTION]
> **CONF-41 — direct internal contradiction in D7.** §4.1 mandates the upscaler preserve **off-diagonal**
> $K_{ij}$; §4.2 transmits **only three diagonal components** $(K_x,K_y,K_z)$. **The payload cannot carry
> what the upscaler is mandated to produce.** Either MPFA-O's off-diagonal correction is silently dropped
> on hand-off — which is precisely the error MPFA-O exists to avoid — or D7 §4.1 is wrong.
>
> **CONF-42.** Porosity is handed over as a **single scalar field**; **no NTG field exists anywhere in
> D7** (0 hits for `NTG`), despite NTG being required for grid-block pore volume. Fault data is reduced
> to displacement vectors + transmissibility multipliers, with **no fault-plane geometry**, no per-segment
> `MULT`, no cell-pair transmissibility table.

**Pattern partitioning (§4.3):** trigger `Area > 40` acres with no explicit well grid;
$N_{pat} = \max(1, \text{round}(\text{Area}/\text{pattern\_spacing}))$ with **40 acres hardcoded**;
$N_{inj}=N_{prod}=N_{pat}$ (5-spot implied, **never named**); per-producer IPR limit **≤ 1 000 BOPD**.

**Post-sim diagnostics (§4.4) and economics (§4.5):** identical to D6 §4.1/§4.6 — same RF cap,
same $\bar{P}_{res,eff}$, same $f_g(t)$, same CAPEX, same mid-year discounting.

### 6.6 What D7 does not specify at all

| Absent | Verified |
|---|---|
| Grid dimensions `NI/NJ/NK`, cell counts | full read |
| Cell sizes `DX/DY/DZ` | full read |
| **Layering** — layer count, thickness scheme; the word "layer" does not appear | full read |
| **Corner-point construction** — 0 occurrences of "corner-point" in D7 (concept lives in D1 §2.2 and D6 §2.3) | full read |
| Tensor anisotropy **generation** algorithm, ratio, angle | full read |
| **NTG** | 0 hits, case-insensitive |
| Variogram kernel/model family, nugget, sill, range | full read |
| **Random seeds, `random_state`, determinism clause** | 0 hits for `seed` |
| Any file format (`GRDECL`/`ECLIPSE`/`INTERSECT`/`RESCUE`/`petekIO`) — the hand-off is **in-memory / zero-copy** | 0 hits |
| Any QA/validation of the *generated data* (histogram, QQ, variogram re-validation, range checks, well tie-back) | full read |
| Corey exponents, endpoint permeabilities, $J(S_w)$ functional form, $\sigma$, $\theta$ | full read |

> ⚠️ **CONF-43.** **The absence of any seed or determinism clause directly defeats two of D7's own four
> stated use cases** — "benchmark model creation" (cross-simulator comparison requires bit-reproducible
> models) and "data imputation" (requires re-derivable fields). This is the highest-impact single gap in
> D7.

---

## 7. D7 — mapping to the live repository

D7's missing grid/statistics parameters **already exist** in this repository:

`core/data_models.py:358-404` — `@dataclasses.dataclass class GeostatisticalParams`:

| Field | Default | Notes |
|---|---|---|
| `variogram_type` | `"spherical"` | validated against `["spherical","exponential","gaussian","matern","cubic"]` |
| `range` | `0.0` | `0` = auto-calculate from grid size |
| `sill` | `0.1` | — |
| `nugget` | `0.0` | — |
| `anisotropy_ratio` | `1.0` | — |
| `anisotropy_angle` | `0.0` | must be $0\le\theta<360$ |
| `trend_type` | `"none"` | `["none","linear","quadratic"]` |
| `trend_parameters` | `[0.0, 0.0]` | — |
| `simulation_method` | `"sequential_gaussian"` | `["sequential_gaussian","turning_bands","fft"]` |
| `random_seed` | `42` | — |
| `grid_resolution` | `(100, 100)` | — |

`__post_init__` validation: `range >= 0`, `sill > 0`, `nugget >= 0`, `anisotropy_ratio > 0`,
`0 <= anisotropy_angle < 360`, 2-element positive `grid_resolution`.

> **Two divergences worth noting:**
> 1. The code supports **`fft`** as a third simulation method — absent from D7.
> 2. The code has **no PGS / plurigaussian path at all** — D7's central facies algorithm has no
>    dataclass counterpart.
>
> Also: `agent_wiki/audit/dialogs_models_utils_widgets_audit.md:187` reports `GeostatisticalParams`
> at **27.3 % parameter usage** — i.e. a mostly-unused capability that a THMC adoption would
> immediately need.