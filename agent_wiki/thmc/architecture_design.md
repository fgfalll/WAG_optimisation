# Proposed Architecture — D1 (Full Architecture Design)

Normalised transcription of `3D THMC_docs/3D THMC Reservoir Simulator Full Architecture Design Document.md`
(490 lines). **Design intent only — no code exists.** See [`README.md`](README.md) for the no-code warning.

---

## 1. Architectural paradigm: Zero-Bloat Core

**Thesis (D1 §1.1).** In EOR and underground storage of hydrocarbons, CO₂ and H₂, RAM overrun by
object-overhead structures — or calling auxiliary services *inside the solver's hot loops* — causes
catastrophic degradation through cache thrashing and uncontrolled memory bloat. The single
architectural decision is therefore to relocate everything not directly tied to local differential
discretisation and iterative equation solving into a **satellite environment**.

**Zero-Bloat Core definition.** The numerical core is responsible *exclusively* for solving the
discretised transport equations, geomechanics, phase equilibrium and chemical kinetics **directly in
dedicated flat memory arrays**. Exiled to the satellite suite:

- IO format conversion
- generation / procedural synthesis of geological grids
- pre-processing of rock parameter inputs
- post-simulation analytics
- **economic evaluation (NPV / CAPEX)**
- plot export

**Diagram (verbatim):**

```
+-----------------------------------------------------------------------------------+
|                            SATELLITE TOOLKIT (OFF-CORE)                           |
|  [IO-Конвертори & Exporters]  [Синтез Сіток & PGS/GEP]  [Post-Analytics & ES-MDA] |
+----------------------------------------+------------------------------------------+
                                         |  (Flat Binary Substrate / petekIO)
                                         v
+-----------------------------------------------------------------------------------+
|                              IN-CORE SOLVER (RUST CORE)                           |
|  - Дискретизація FVM/MPFA-O & FEM      - Термодинамічний Flash (PR/SRK/PC-SAFT) |
|  - Поропружність Біо & Barton-Bandis   - Реактивна кінетика & 5-State Trapping     |
|  - Parallel Matrix Assembly (Rayon)    - Linear Solvers (faer / cuDSS Stack)       |
+-----------------------------------------------------------------------------------+
```

### 1.1 Responsibility boundary

| In-Core Solver (4 duties) | Satellite Suite (4 duties) |
|---|---|
| 1. Parallel assembly of residual vector + Jacobian in contiguous arrays | 1. Bidirectional conversion of external geological formats → `petekIO` |
| 2. Isothermal flash + fugacity coefficients | 2. Procedural facies modelling (PGS) + evolutionary relation synthesis (GEP) |
| 3. Interphase convective–diffusive fluxes (FVM/MPFA-O), FEM deformations, geochemical kinetics | 3. Dynamic DCF NPV + CAPEX/OPEX + summary tables |
| 4. Iterative SLE solve via modular linear-solver stack + automatic adaptive timestep control | 4. Assisted History Matching via **ES-MDA** |

> ⚠️ **Cross-cutting contradiction inside D1 itself.** Economics (duty 3) is *excluded* from the core,
> yet D1 §10.2 puts the **NPV objective's adjoint gradient inside the core**. Gradient of an
> off-core objective computed in-core implies the objective is in-core. Tracked **CONF-04**.

### 1.2 Physical regimes and complexity

| Regime | Key state variables | Per-cell cost |
|---|---|---|
| Black-Oil | `P_res, S_w, S_g, R_s` | `O(N_cells · 3²)` |
| Compositional EOR | `P_res, S_w, S_g, z_i, x_i, y_i` | `O(N_cells · (N_c+1)²) + Flash` |
| Thermal THMC | `P_res, S_w, S_g, z_i, T, u, C_m` | `O(N_cells · (N_c+7)²) + FEM` |
| Poroelasticity (Biot) | `u`, `p` | `O(N_nodes · 4²)` |
| Reactive Geochemistry | `C_m, φ, Da, Pe` | `O(N_cells · N_minerals)` |

Extensibility is via **Rust traits** — static dispatch and type composition at compile time instead of
dynamic VTABLE inheritance, "guaranteeing Zero-Cost Abstractions".

> ⚠️ Static dispatch means the composition is fixed at compile time. Regime switching per run then
> requires either monomorphisation of all combinations or dynamic dispatch somewhere. Not addressed.

---

## 2. Data substrate — `petekIO`

### 2.1 Contiguous flat memory slabs (SoA)

Rejected: pointer-chasing `Vec<Cell>` — random access causes constant L1/L2/L3 misses and
"completely precludes" SIMD vectorisation (AVX-512 / ARM Neon). Adopted: all physical variables as
continuous `Vec<f64>`.

```
[ P_res:  f64, f64, f64, f64, ... | Cell 0..N ]
[ S_w:    f64, f64, f64, f64, ... | Cell 0..N ]
[ S_g:    f64, f64, f64, f64, ... | Cell 0..N ]
[ z_0:    f64, f64, f64, f64, ... | Cell 0..N ]
[ z_1:    f64, f64, f64, f64, ... | Cell 0..N ]
```

AoS interleaving forces gather/scatter during parallel matrix assembly. In SoA, each component forms a
separate contiguous aligned block, letting **512-bit AVX-512 registers load 8 consecutive `f64` per cycle**.

### 2.2 3D corner-point IJK (GRDECL) and geometric defects

Each cell = **8 spatial vertices**; geometry via vertical or inclined **pillar geometry** coordinate
lines plus face-cover depths.

| Defect | Handling |
|---|---|
| **Pinch-outs** — top and bottom faces merge, $V_{\text{cell}} \le 10^{-12}\,\text{m}^3$ | Deactivated **or** algebraically collapsed with neighbours, **without removal from global vectors** — preserves contiguous canonical $I\times J\times K$ indexing |
| **Inactive / zero-volume cells** | Retained inside the flat slabs, flagged in a bitwise **`ActiveMask`**; loops filter by bitmasking without breaking cache lines |
| **Explicit fault step-outs** | Adjacency `I±1, J±1, K±1` lost; a separate **list array of Non-Neighbour Connections (NNC)** processed vectorised after the regular stencil |

> ⚠️ Pinch-out handling is stated as a disjunction ("deactivated **or** collapsed") — the
> implementation is undecided. Tracked **CONF-05**.

### 2.3 Schwarz 3D sub-domains (full-physics 3D zooming)

Two-scale modelling: large regional grid + local high-detail 3D grid containing **explicit cement
sheath, perforations and micro-annulus**.

1. **Global step** on the regional grid; average `P_res` and interphase fluxes on the local zone's outer faces.
2. **Schwarz boundary conditions** by spatial interpolation onto $\partial\Omega_{\text{sub}}$:
   $P_{\text{sub}}(\mathbf{x}) = P_{\text{global}}(\mathbf{x})$ (Dirichlet) or flux (Neumann).
3. **Local THMC solve** with micro-timestep $\delta t$.
4. **Reverse transfer + nonlinear smoothing**; repeat under the Schwarz alternating (variable-operator)
   scheme until convergence in pressure and mass balance on the boundary.

---

## 3. Spatial discretisation

### 3.1 FVM + MPFA-O

Motivation: non-orthogonality of skewed faces in corner-point grids plus high anisotropy make **TPFA**
unsuitable — artificial numerical diffraction and large flux errors.

Full symmetric 6-component permeability tensor:

$$K = \begin{pmatrix} K_{xx} & K_{xy} & K_{xz}\\ K_{xy} & K_{yy} & K_{yz}\\ K_{xz} & K_{yz} & K_{zz}\end{pmatrix}$$

Around each cell vertex an **Interaction Region** forms; flux through face $f$:

$$\Phi_f = \sum_{j\in\Omega_v} T_{f,j}\,P_j$$

with $T_{f,j}$ accounting for face orientation and the off-diagonal components.

Rust type named: **`AnisotropicTensor3D`**.

### 3.2 High-order convective schemes

TVD with **Superbee, van Leer, minmod** limiters, plus **WENO3** and **WENO5**. WENO builds the local
flux from a convex combination of stencils with smoothness-dependent weights — high order in smooth
regions, no numerical dispersion on sharp saturation fronts.

**Gas fractional flow (⚠️ see CONF-01):**

$$f_g(t) = \frac{1}{1 + \left(\frac{1-S_w(t)-S_g(t)}{S_g(t)-S_{gc}}\right)\cdot\frac{\mu_g}{\mu_o}}$$

**Stated behaviour (D1 line 126):** after breakthrough $f_g(t)$ "dynamically grows to **0.60–0.85**"
as the oil bank is displaced, and CO₂ GOR reaches "**5,000–25,000 SCF/STB**".

> ⚠️ **CONF-01 — corrected 07-10-2026.** The formula has the **correct sign**: it agrees with the
> classical $f_g = \frac{M}{1+M}$ to 4 dp across the whole viscosity-ratio range (measured). An earlier
> version of this page claimed it was an inverted fractional flow; that was a substitution error and is
> withdrawn. The range 0.60–0.85 **is** reachable. The real defect is that $\frac{S_o}{S_g-S_{gc}}$
> is an **ad-hoc proxy for $k_{ro}/k_{rg}$** — no $S_{or}$, linear rather than Corey, no water term,
> unguarded at $S_g\to S_{gc}$. See [`conflict_and_gap_register.md`](conflict_and_gap_register.md) §1.2.

### 3.3 FEM for Biot poroelasticity

Via internal modules **`fenris` / `RustFEA`**. Effective stress:

$$\sigma'_{ij} = \sigma_{ij} - \alpha_B\,p\,\delta_{ij}$$

To avoid instability and grid **locking** at low fluid compressibility, discretisation uses
**Taylor–Hood quadratic–linear elements (P2–P1 / Q2–Q1)** — displacement with quadratic basis,
pressure linear — which *strictly satisfies* the **LBB (Ladyzhenskaya–Babushka–Brezzi)** condition.

### 3.4 Fixed-stress split and stabilisation

Coupled THMC solved by sequential fixed-stress split. Stabilisation term added to the diagonal of the
filtration mass matrix:

$$S_{\text{stab}} = \frac{\alpha_B^2}{K_{\text{dry}}}$$

Production-weighted effective average reservoir pressure, computed to remove pressure mismatch and
ensure closure of 0D/3D basin connections:

$$\bar{P}_{\text{res,eff}} = \frac{\sum_t P_{\text{res}}(t)\,q_o(t)\,\Delta t}{\sum_t q_o(t)\,\Delta t}$$

**Claim:** $S_{\text{stab}}$ plus the $\bar{P}_{\text{res,eff}}$ correction "guarantees monotone decay of
the splitting error and absolute stability of timesteps even at significant pressure depressions".

---

## 4. Extended thermodynamics and GPU flash

| § | Content |
|---|---|
| 4.1 | **Zero-hardcoded EOS**: cubic **PR**, **SRK**, plus **PC-SAFT** and **CPA**, with **volume translations VT-PR / VT-SRK**. HTHP range **7–276 MPa**, **278–533 K**. BIPs `k_ij` from van der Waals or Huron–Vidal mixing rules. |
| 4.2 | **Asphaltenes (AOP)**: onset pressure via Flory–Higgins or PC-SAFT. Kinetics $\dot S_{asph} = k_{precip}\max(0, P_{AOP}-P_{res}) - k_{entrain}\,v_f\,S_{asph}$. Precipitate narrows pores → reduced $\phi$, $k$. Viscosity from **f-theory** or **Free Volume**. |
| 4.3 | **GPU flash**: isothermal flash (Rachford-Rice + fugacity $\varphi^L_i,\varphi^V_i$) is "**up to 70 %** of total compositional runtime". Parallelised via **`wgpu`** shaders or **CUDA**. |
| 4.4 | **Hyper-dual AD**: exact Jacobian needs $\partial\varphi_i/\partial P$, $\partial\varphi_i/\partial x_j$, $\partial^2\varphi_i/(\partial x_j\partial x_k)$. Libraries **`feos-ad`**, **`num-dual`**; hyper-duals give analytic derivatives to **2nd order in a single pass at $10^{-16}$**. |

### 4.1 Flash benchmark (D1, for $10^6$ cells, 8 components)

| Platform | Parallelisation | Flash time (ms) | Speed-up |
|---|---|---|---|
| CPU 1 core | sequential loop | **4 250** | 1.0× (baseline) |
| CPU 64 cores | Rayon | **85** | 50.0× |
| GPU `wgpu` shaders | massively parallel compute | **4.2** | 1 011.9× |
| GPU CUDA kernel | tensor cores / warps | **3.1** | 1 370.9× |

Arithmetic is internally consistent (4250/85 = 50.0; 4250/4.2 = 1011.9; 4250/3.1 = 1371.0).

> ⚠️ **No hardware is identified** — no CPU model, GPU model, core count, interconnect, precision
> caveat, or statement of whether flash tolerances differ between CPU and GPU paths. The $1371\times$
> figure is unanchored. Tracked **CONF-06**.

---

## 5. Reactive geochemistry, leaching and swelling

### 5.1 Lasaga kinetic law

$$r_m = A_m\,k_m\left(1-\frac{Q}{K_{eq}}\right)^{\eta}$$

`A_m` specific reactive surface area, `k_m` rate constant, `Q` ionic activity product, `K_eq` equilibrium
constant, `η` empirical exponent. Reaction balancing uses **LA-GE** (Gaussian elimination) or
**LP-2P** (two-phase linear programming).

### 5.2 Dimensionless numbers

$$Da = \frac{k_m A_m L}{v_f},\qquad Pe = \frac{v_f L}{D_m}$$

### 5.3 Wormholing and permeability–porosity coupling

Wormholes form at **$Da \approx 1$** with high $Pe$. Two closures given, **no selection rule**:

- Modified Kozeny–Carman: $k(\phi) = k_0\left(\frac{\phi}{\phi_0}\right)^n\left(\frac{1-\phi_0}{1-\phi}\right)^2$
- Verma–Prusa: $k/k_0 = \left(\frac{\phi-\phi_c}{\phi_0-\phi_c}\right)^t$

### 5.4 Swelling and chemo-mechanical softening

Clay porosity $\phi_{clay}(C_{salinity})$ varies under low-salinity injection; sorption swelling on CO₂/H₂
adsorption narrows the flow path and drops $k(t)$. Impurities **`SO₂, H₂S, CH₄, N₂`** correct solubility
and ionic balance. Softening: $E(\phi) = E_0[1 - d_m(\phi-\phi_0)]$, $C_0(\phi) = C_{0,a}e^{-a_m\phi}$.

---

## 6. Five-phase conservation inventory and trapping

### 6.1 Stone I / Stone II and the Koval recovery cap

Three-phase relative permeability via **Stone I** and **Stone II**.

**Koval-based recovery bound (verbatim):**

$$RF = RF_{\text{ultimate}}(\bar{P}_{\text{res,eff}})\cdot\left(1 - \exp\left(-\frac{HCPVI}{1.5}\right)\right)$$

> ⚠️ **CONF-02.** At the shipped default HCPVI 7.69 this cap is 0.9939 — inactive where the optimiser
> operates. `1.5` has no units, provenance or sensitivity study.

### 6.2 Capillary trapping

**Land:** $S_{gr} = \dfrac{S_{gi}}{1 + C_{Land}\,S_{gi}}$ — $S_{gi}$ = max gas saturation on drainage,
$C_{Land}$ = Land constant. Hysteresis smoothed with **C¹-continuous Killough / Carlson** models to remove
discontinuities on flow reversal.

### 6.3 Dissolved phase and Setschenov

Henry's law plus salting-out: $\ln(H_{g,brine}/H_{g,water}) = K_{Setschenov}\,C_{salinity}$. Soluble
species: `CO₂`, `CH₄`, `H₂S`.

**Hard floor (D1 line 264):** for secondary and tertiary CO₂ regimes a strict **Net CO₂ Utilisation
floor of ≥ 2.5 MSCF/STB (0.25–0.50 tonne/STB)** is imposed.

### 6.4 Langmuir competitive isotherm

$$V_i = \frac{V_{L,i}\,b_i\,P_i}{1 + \sum_j b_j P_j}$$

### 6.5 The five trapping states

| # | State | Transport / equilibrium | Trigger |
|---|---|---|---|
| 1 | **Free** | FVM/Darcy, MPFA-O, Stone I/II | Convective transport from $\nabla P$ + gravity |
| 2 | **Trapped** | Land $S_{gr}$, Killough hysteresis | Capillary trapping on imbibition ($S_g \to S_{gr}$) |
| 3 | **Dissolved** | Henry + Setschenov | Mass exchange vs $P, T, C_{salinity}$ |
| 4 | **Adsorbed** | Langmuir isotherm | Sorption as partial pressure changes |
| 5 | **Mineralised** | Lasaga $r_m$ | Carbonate precipitation $\text{CO}_3^{2-} + \text{Ca}^{2+} \to \text{CaCO}_3 \downarrow$ |

---

## 7. Hydraulic fracturing and proppant

**Models (D1 §7.1):** **P3D**, **Planar-3D**, **DFN**. Explicit target named: **the Petrykiv horizon of
the Prypiat Trough**, tight reservoirs.

**Barton–Bandis nonlinear aperture:**

$$w_f(\sigma'_n) = \frac{w_0}{1 + \sigma'_n/V_m} + w_{res}$$

**Proppant:** convection + gravitational settling (Stokes / Gauer) + dense pack formation.
Conductivity $k_f w_f$ degrades by **embedment**, **crushing** (fine sludge clogging pack pores) and the
**Forchheimer factor $\beta$** (non-Darcy resistance at high velocity).

**EDFM coupling** — fractures integrate into the matrix FVM grid without rebuilding the mesh:

```
+-------------------+
|  Matrix Cell      |
|      *--------/---|---> NNC T_mf (Matrix-Fracture)
|     /        /    |
+----/--------/-----+
    / Fracture
   v
  NNC T_ff (Fracture-Fracture)
```

Three NNC transmissibility types: `T_mf` (matrix cell ↔ fracture segment), `T_ff` (adjacent segments),
`T_{f1f2}` (two-fracture intersection). All computed **geometrically, without rebuilding the primary
matrix-cell framework**.

---

## 8. Discontinuities and fault mechanics

**DP/DP and MINC.** Dual Porosity / Dual Permeability for naturally fractured reservoirs; **MINC**
splits matrix blocks into nested sub-continua by distance from fractures for strongly nonlinear
matrix–fracture transfer and thermal fronts.

**Shale Gouge Ratio (sealing capacity):**

$$SGR = \frac{\sum V_{shale}\,\Delta z}{\text{Fault Throw}}$$

**Coulomb Failure Stress (slip activation):**

$$\Delta CFS = \tau - \mu(\sigma_n - p)$$

When **$\Delta CFS > 0$** → slip activation, i.e. the **"Fault Seal to Conduit Transition"**, with a
jump-like increase in conductivity.

---

## 9. Well scheduling, signals and sub-stepping

### 9.1 Unconstrained scheduling and PID

Arbitrary boundary-condition time series — $q_{inj}(t)$, $P_{bhp}(t)$, $T_{inj}(t)$, $z_i(t)$ — approximated
with **C²-smooth splines**. Closed-loop **PID controllers** drive downhole **ICV / ICD** valves in real time,
converting artificial throttling into process-constraint-respecting control.

**Pattern sizing:** for fields **> 40 acres**,

$$N_{pat} = \max\left(1,\ \text{round}\left(\frac{\text{Area}}{\text{pattern\_spacing\_acres}}\right)\right),\qquad N_{inj} = N_{prod} = N_{pat}$$

with **per-well rate hard-limited to ≤ 1 000 BOPD** by Darcy + Vogel laws.

> ⚠️ A hard numerical clamp sits between physics and scheduling. D1 does not say whether it is applied
> before or after the drift-flux model, nor how the clamped rate propagates to mass balance.
> Tracked **CONF-07**.

### 9.2 Multi-segment drift-flux hydraulics

Phase slip and regime transitions between gas, slug and annular flow. Total acceptance computed as
`N_pat × q_well`. Longitudinal flow through the cement **micro-annulus** $w_{micro}$ from cement
decarbonation or geomechanical debonding is additionally accounted for.

### 9.3 Sub-adaptive sub-stepping

Global reservoir step $\Delta T$ split into adaptive well micro-steps $\delta t_k$ (illustrated with **8**
micro-steps):

```
Глобальний крок пласта Delta T: |---------------------------------------------------|
Мікро-кроки свердловини delta t: |-dt1-|-dt2-|-dt3-|-dt4-|-dt5-|-dt6-|-dt7-|-dt8-|
```

Mass accumulation, $Q^{cum}_{j,\text{perf}} = \sum_k \int_{\delta t_k} q_j(t)\,dt$; stated goal
**"100 % mass balance (mass preservation > 99.9 %)"**, "completely excludes mass imbalance when
splitting time scales".

> ⚠️ `δt` (Schwarz sub-domain micro-step, D1 §2.3) and `δt_k` (wellbore sub-step, D1 §9.3) reuse the
> same symbol for two different sub-grids with no stated coupling or tolerance. Tracked **CONF-08**.

---

## 10. Solver, adjoint gradients and GPU solvers

See [`solver_and_numerics.md`](solver_and_numerics.md) for the full treatment (Newton, timestep,
AD, adjoint, Rust types).

Headline claims:

- **Zero-allocation parallel assembly (Rayon)** — all heap allocations excluded from hot loops during
  Newton iterations; per-worker buffers pre-allocated in contiguous arrays.
- **Adjoint gradients in-core.** $(\partial R/\partial x)^\top\lambda = \partial J/\partial x$,
  $\nabla_u J = \partial J/\partial u - \lambda^\top(\partial R/\partial u)$ — objective $J$ explicitly
  **NPV**, exact gradient in **one backward pass independent of control count**.
- **Solver stack:** `faer` (Rust-native SIMD), `russell_sparse` (C-bind to MUMPS + UMFPACK), `cuDSS`
  (NVIDIA, in-GPU-memory). Line search and backtracking for Newton stability.
- **D1 §10.3 change note (verbatim):** *"artificial artificial penalties are removed in favour of a
  natural NPV reduction through the cost of gas recycling"*. → **CONF-03**.

---

## 11. Satellite tooling and synthetic ecosystem

### 11.1 Pre-simulation calculators (4)

1. **Black-Oil PVT generator** — PVT tables from cubic EOS (PR, SRK) or empirical correlations.
2. **SCAL / rock-typing calculator** — capillary and relative-permeability curves via Amaefule **FZI / RQI**.
3. **Standalone tensor upscaler** — effective anisotropic `K_eff` from **MPFA-O**.
4. **Karakas–Tarik skin calculator** — mechanical and geometric skin for deviated/perforated wells.

### 11.2 Procedural generator

**GameDev layer:** Perlin / Simplex / Ridge noise + Voronoi; **Wave Function Collapse** synthesising
facies from conceptual templates (training images); hydraulic erosion for river channels and levees.

**PGS suite** — four stages, with a `Plurigaussian Truncation Flag` ASCII schematic (`G1`, `G2` thresholds;
`T1`, `T2`; facies A/B/C/D):

```
   G2 ^
      |  [ Facies C ]  |  [ Facies D ]
   T2 +----------------+----------------
      |  [ Facies A ]  |  [ Facies B ]
      +----------------+------------------> G1
                   T1
```

1. Flags & thresholds — geo-domains by truncating $N$ continuous Gaussian fields $Z = (Z_1,\dots,Z_N)$; $P_i(x) = \int_{D_i} g(z_1,\dots,z_N)\,dz_1\dots dz_N$.
2. Variogram transformation + **Gibbs sampler** — indicator covariances $C_{ij}(h) \to$ Gaussian-field covariances by **Hermite polynomial expansion**; values at sample points by **MCMC Gibbs sampler**.
3. Grid simulation — **Turning Bands** or spectral simulation, then back-truncation to facies.
4. Non-stationarity / cyclicity — variograms with **dampened hole-effect**.

**GEP engine** — chromosome example, positions 0–15, `[ + * Q / a b a | a b a a b b a a ]`, `<Head h>` covering 0–6, `<Tail t>` covering 7–15:

$$t = h\,(n-1) + 1$$

ORFs and **K-expressions**; multi-gene chromosomes with linking functions **Addition, Multiplication,
IF, OR**. Variation operators with probabilities: mutation `p_m`, IS transposition `p_is`, RIS
transposition `p_ris`, gene transposition `p_gt`, one-point recombination `p_1r`, two-point `p_2r`,
gene recombination `p_gr`. Ephemeral random-constant domain `D_c`. Fitness:

$$f_i = f_{max} - \sum_{j=1}^{C_t}\left|C(i,j) - T_j\right|,\qquad f_{max} = C_t\,M$$

Selection by **roulette-wheel with elitism**. Worked example: `f_max = 1000` for $C_t=10$, $M=100$.

> ⚠️ The example is internally inconsistent with the tail-length formula: $h=7$, $t=9$ requires
> $n = 2.143$, not an integer arity. Tracked **CONF-09**.

### 11.3 Post-simulation analytics, AHM, ES-MDA

- **Material balance Havlena–Odeh** — drained volume and displacement-mechanism check.
- **Coverage coefficients** — $E_v$ volumetric, $E_a$ areal, $E_m$ microscopic.
- **RTA / DCA & EUR** — decline-curve analysis.

**ES-MDA** calibrates the reservoir parameter vector (permeability, porosity, fault conductivity)
against declared well pressures and rates, without exceeding physical bounds.

**Dynamic CAPEX:**

$$CAPEX = N_{inj}\times \$1.5\mathrm{M} + N_{prod}\times \$1.0\mathrm{M} + (N_{pat}\times \$250\mathrm{k} + \$3\mathrm{M}) + \$10\mathrm{M}\times\left(\frac{Q_{recycle,peak}}{20\,000}\right)^{0.65} + \$5\mathrm{M}$$

**Unified DCF NPV:**

$$\text{NOCF}_t = \text{Oil\_rev} + \text{Storage\_credit} - \text{Purch\_cost} - \text{Rec\_cost} - \text{Operating\_cost}$$
$$NPV = -CAPEX_0 + \sum_{t=1}^{N}\frac{\text{NOCF}_t}{(1+r)^{t-0.5}}$$

> ⚠️ **CONF-10.** The `Q_recycle,peak / 20 000` normaliser has **no declared units** in D1.
> ⚠️ **CONF-11.** The NOCF expression **omits hydrocarbon-gas revenue** entirely — structurally
> identical to the shipped **CRIT-18**. Also see CONF-04.
> ⚠️ D1 **ends here** (line 490, mid-sentence). No conclusion, no acceptance criteria, no
> references, no section 12.

---

## 12. Explicitness matrix — what D1 actually specifies

| Specified with numbers | Named but unparameterised | Absent |
|---|---|---|
| Pinch-out threshold `1e-12 m³`; AVX-512 8×f64; HTHP 7–276 MPa / 278–533 K; flash 70 % / 4 250→3.1 ms; flash benchmark grid 1e6 cells × 8 comp; wormhole `Da≈1`; CO₂ floor 2.5 MSCF/STB; pattern 40 ac; `≤1000 BOPD`; 8 micro-steps; mass preservation >99.9 %; CO₂ GOR 5 000–25 000 SCF/STB; CAPEX coefficients | WENO3/WENO5 (no weights), Stone I/II (no formulation), Killough/Carlson (no constants), HCPVI divisor 1.5, Moser/TPD seeds, ES-MDA (no $\alpha_i$ schedule), pre-simulation calculators, PGS stage details | Preconditioners; sparse storage format; Krylov methods; Newton iteration counts; residual tolerances; damping parameters; timestep growth rules; `petekIO` schema; convergence criteria per mode; split-iteration limits; parallel efficiency; any benchmark case (no SPE, no Brugge, no Norne, no PUNQ) |