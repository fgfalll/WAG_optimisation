# Hydraulic Fracturing and Complex Wellbore — D8

Source: `3D_THMC_docs/Hydraulic Fracturing and Complex Wellbore Architecture Specification.md`
(282 lines, 38 772 bytes). **Design intent only.**

---

## 1. Scope correction

D8 is **not** a fracture-model catalogue. It names **three** fracture geometry formulations
(**P3D**, **Planar-3D**, **DFN**) and is otherwise an **EDFM coupling + wellbore + NPV-unification**
document. Zero occurrences of PKN, KGD, DPM, cohesive zone, phase-field, peridynamics, PFC or RBSM.
No fracture-evolution, proppant or wellbore Rust types exist — those sections are prose and equations
only. See [`doc_inventory.md`](doc_inventory.md) §5.

---

## 2. Section map

| § | Title |
|---|---|
| 1.1 | Module architecture and FVM/FEM integration via **EDFM** |
| 1.2 | Thermodynamic phase-equilibrium core (EoS and Delumping) |
| 1.3 | Integration with the surrogate engine and GEP algorithms |
| 2.1 | HF fracture geometry dynamics (`L_f`, `H_f`, `w_f`) |
| 2.2 | Proppant transport, settling, pack formation |
| 2.3 | Fracture conductivity dynamics and permeability degradation |
| 2.4 | Fluid breakthrough hydraulics and the Koval fractional-flow model |
| 3.1 | Wellbore and casing geometry |
| 3.2 | Cement sheath integrity |
| 3.3 | Perforation-zone dynamics and productivity (IPR) limiting |
| 3.4 | Multi-segment drift-flux wellbore hydraulics |
| 4.1 | Dynamic CAPEX model and infrastructure sizing |
| 4.2 | Unified DCF NPV model + **3 numbered NPV-unification acceptance criteria** |

---

## 3. §1.1 — EDFM integration (the strongest part of D8)

> Integration of discrete fracture geometry into the 3D FVM/FEM grid is performed on the basis of the
> **Embedded Discrete Fracture Model (EDFM)**. With EDFM, fractures geometrically intersect the matrix
> FVM/FEM grid **without the need to rebuild the mesh**. Mass and energy transfer between matrix and
> fracture is computed by geometric transmissibility factors on the geometric-intersection face.

This is architecturally consistent with D1 §2.2's NNC list-array approach and with D2 §5.1's NNC
transmissibility. **EDFM, NNC, DP/DP and MINC form the one coherent idea across D1, D2 and D8**:
fracture–matrix coupling without mesh rebuild.

---

## 4. §1.2 — EoS and Delumping

**Thermodynamic kernel:** isothermal flash over PR, SRK, VT-PR, VT-SRK, PC-SAFT; **Michelsen stability
test** then **Rachford–Rice** root-finding by Newton–Raphson with step limiting:

$$f(V) = \sum_{i=1}^{N_c}\frac{z_i\,(K_i - 1)}{1 + V\,(K_i - 1)} = 0$$

with $K_i = y_i/x_i$ corrected each step via fugacity coefficients $\varphi_i^V$, $\varphi_i^L$.

**Surface-facility composition** via **delumping ("Delumping Mix-Method")** consistent with detailed PVT
data and **Nichita's analytical approach**.

### 4.1 Accuracy table (D8 §1.2 — the only quantitative validation artefact in D8)

| EoS | Applicability | MAPD density error | f-theory viscosity | FVT viscosity |
|---|---|---|---|---|
| PR-EOS | `P < 50 MPa` | **3.0 – 8.0 %** | **~ 5.0 %** | 3.0 – 4.0 % |
| SRK-EOS | `P < 50 MPa` | **4.0 – 9.0 %** | **~ 5.0 %** | 3.0 – 4.0 % |
| HTHP VT-SRK | 7–276 MPa, 278–533 K, shift ~ `(Mω)^{-1}` | **1.0 – 2.0 % (1.5 %)** | **~ 5.0 %** | **< 3.0 %** |
| HTHP VT-PR | 7–276 MPa, 278–533 K | **1.0 – 4.0 % (2.0 %)** | **~ 5.0 %** | **< 3.0 %** |
| PC-SAFT | 7–276 MPa, 278–533 K | **~ 1.0 %** | **N/A** | **< 2.0 %** |

> ⚠️ **CONF-49 — a documented accuracy ceiling of 8–9 %.** PR and SRK are declared to have **8–9 %**
> mean-absolute-relative-percent density error**. Any adoption must state which EoS governs the
> optimisation objective, because at that error level recovery-factor differences between candidate
> designs are inside the fluid-model noise band. **PC-SAFT is the only row at ~1 %** and is the only one
> in the set with a defensible density claim.

---

## 5. §1.3 — Surrogate engine and GEP integration

**GEP surrogates** are compiled to bytecode `Vec` instructions or **JIT** functions consuming
`&[MatrixCell]` and returning production/pressure surrogates, bypassing Python runtime overhead.
Also **`C-ABI / FFI`** (`unsafe extern "C"`) bindings.

**GEP constraints:** fixed-length chromosome strings; gene = head `h` (functions `F` + terminals `T`)
and tail `t` (terminals only) with $t = h\cdot(n_{max}-1)+1$; ORF read left-to-right, top-to-bottom;
remainder outside the active ORF is non-coding; sub-ETs joined by addition, multiplication or logical
IF; mutation replaces head symbols with function/terminal but tail symbols **only with terminals**.

> ⚠️ **CONF-50.** `MatrixCell` (see §8.2) carries `cell_id`, `volume`, `pressure`,
> `permeability: [f64;3]`, `porosity` — **no saturations, no composition, no temperature, no
> transmissibility**. A GEP surrogate consuming `&[MatrixCell]` therefore cannot represent any
> multiphase state. The type and the stated use are incompatible. The same type also cannot carry the
> §2.1 fracture geometry, which has no home in any struct.

---

## 6. §2 — Module 1: hydraulic fracturing and proppant transport

### 6.1 Fracture geometry (§2.1) — 3 models, one relation

**Models:** `Pseudo-3D (P3D)`, `Planar-3D`, `Discrete Fracture Network (DFN)`.

**Net pressure:** $p_{net}(x,z) = p_{fluid}(x,z) - \sigma_h(z)$, $\sigma_h$ = minimum horizontal stress.

**Opening:** $w_f(x,z) = C_w\cdot\dfrac{1-\nu^2}{E}\cdot p_{net}(x,z)\cdot H_f$
($E$ = Young's, $\nu$ = Poisson's, $C_w$ = shape/geometric coefficient)

**Length/height growth:** front propagation set by local fluid mass balance against viscous-critical
fracture toughness $K_{IC}$ [Pa·m^0.5]. In P3D, vertical growth $H_f$ across barrier layers is limited by
a stress jump $\Delta\sigma_h$.

> ⚠️ **CONF-51.** No per-model equations, degrees of freedom, or applicability ranges are given — one
> geometry relation is applied to all three models. $w_f \propto H_f$ is dimensionally consistent but is
> **not** the standard PKN/DPM relation. $C_w$ has **no value**, $K_{IC}$ has **no value**, and the
> propagation criterion is qualitative. D8 is not implementable as a fracture model.

### 6.2 Proppant transport (§2.2)

$$\frac{\partial(w_f C_p)}{\partial t} + \nabla\cdot(w_f C_p\,\mathbf{u}_p) - \frac{\partial}{\partial z}(w_f C_p\,v_{settling}) = 0$$

**Discretisation:** upwind **TVD**, high order, with flux limiters **Superbee** or **Van Leer**:

$$F_{i+1/2} = F^{Low}_{i+1/2} + \varphi(r_i)\left(F^{High}_{i+1/2} - F^{Low}_{i+1/2}\right)$$

Time integration: explicit-implicit **TVD Runge–Kutta**.

**Settling** — Stokes corrected for confined walls and steric hindrance ("Power-Gower" model as printed):

$$v_{settling} = \frac{g\,d_p^2(\rho_p - \rho_f)}{18\,\mu_f}\,(1-C_p)^{n}\,f_{wall}(w_f/d_p)$$

**Pack formation:** at local concentration $C_{p,max}\approx 0.58\text{–}0.63$ a static proppant pack
forms in the lower fracture; packed-fracture $w_f$ under closure set by granular-medium equilibrium.

### 6.3 Conductivity dynamics (§2.3)

Closure stress $\sigma_{closure} = p_{confining} - p_{pore}$. Modified Kozeny–Carman with **StimLab**
degradation curves:

$$k_f(\sigma_{closure}) = k_{f,0}\cdot\Phi_{crush}(\sigma_{closure})\cdot\Phi_{embed}(\sigma_{closure})$$

**Inputs:** `w_f_initial`, `d_prop`, `sigma_closure`, `rock_youngs_modulus`, `proppant_strength`,
`concentration_area` [kg/m²], `gel_residue_damage` (0..1).
**Outputs:** `w_f_effective`, `k_f_effective`, `fracture_conductivity` = $k_f\cdot w_f$.

### 6.4 Breakthrough and Koval (§2.4)

Per-pattern injection rate $q_{inj,pattern}$ [m³/s / MSCFD] drives breakthrough.
RF cap and $f_g(t)$ are **verbatim identical** to D1 §6.1 / D6 §4.5. → **CONF-01, CONF-02**.

---

## 7. §3 — Module 2: wellbore and completion

### 7.1 Casing string (§3.1) — 5 elements

1. **Conductor** 2. **Intermediate casings** 3. **Production casing** 4. **Liners / tail sections**
5. **Tubing**

**Mandatory per-segment parameters:** `D_in` [m/in], `D_out` [m/in], wall thickness `t_w` [mm/in],
steel surface mean-absolute-roughness `ε` [mm/in], plus `MD` and `TVD` [m/ft].

> ⚠️ **CONF-52.** No completion-type taxonomy at all: no vertical/horizontal builder distinction, no
> multilateral, no lower/upper completion, no gravel pack, no screen, no fish-mouthed vs. slotted liner,
> no packers, no plugs, no stage-isolation tooling, no inflow-profiling model. ICD/AICD appear only as a
> phrase in §3.4; **AICD is never mentioned**, and no device characteristic curve or parameter is given.

### 7.2 Cement sheath (§3.2)

Parameters: inter-annular cement thickness [mm/in]; `E_cem`; `ν_cem`; **bond strength to casing** and
**to formation**.

The module computes cement stress state under changing inner-column pressure and temperature;
**exceeding the material strength limit models micro-annulus formation and longitudinal annular leak
paths causing inter-reservoir crossflow**.

> ⚠️ **CONF-53.** No damage model, no tension criterion, no cyclic-count provision, no
> thermal-expansion coefficient. "Exceeding the strength limit" is not an evaluable condition without a
> damage law.

### 7.3 Perforations (§3.3)

Parameters: shot density [holes/m, shots/ft]; hole **diameter** [mm/in]; channel depth `L_p` [mm/in];
**phasing angle 60°, 90°, 120°, 180°**; skin factor `S_perf` of the damaged zone.

**IPR clamp:** $q_{o,well} \le 1\,000$ BOPD per well via Darcy/Vogel IPR. Field injection
$q_{inj,field} = q_{inj,pattern}\times N_{inj}$, range **20 000–60 000 MSCFD** field-wide or
**1 000–2 500 MSCFD** per pattern.

> ⚠️ **CONF-54.** Only a **single perforated interval** is defined — no perforation-cluster or stage-tower
> geometry, so no stage-isolation or diversion capability exists in the model. Also note the 20 000–60 000
> MSCFD field-wide range against 1 000–2 500 MSCFD per pattern is only consistent for specific pattern
> counts (8–60 patterns at 40-acre spacing ⇒ 320–2 400 acres); the reconciliation is not stated.
> This range **directly contradicts D5 §1.1's anti-pattern text**, which cites
> "rate clamped to 5 000 MSCFD instead of the 20 000–60 000 MSCFD range" as a *defect signature*.

### 7.4 Drift-flux wellbore hydraulics (§3.4)

$$\frac{dp}{dz} = \rho_m g\sin\theta + \frac{f_m\,\rho_m\,v_m^2}{2D} + \rho_m v_m\frac{dv_m}{dz},\qquad \rho_m = \alpha_g\rho_g + \alpha_l\rho_l$$

Gas slip: $v_g = C_0 v_m + V_{gj}$ — distribution coefficient $C_0$ + drift velocity.

Non-isothermal heat transfer (conduction + convection) through casings and cement sheath.
Annular flow and **ICD / ICV** operation are listed as modelled.

---

## 8. §1.1 — Rust types (verbatim)

### 8.1 `SimulatorError`

```rust
/// Системний перелік помилок обчислювального рушія симулятора
# [derive(Debug, Clone, PartialEq)]
pub enum SimulatorError {
    FlashConvergenceError(String),
    MatrixInversionError(String),
    EDFMGeometryError(String),
    ProppantTransportError(String),
    WellboreHydraulicsError(String),
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::FlashConvergenceError(msg) => write!(f, "Flash Convergence Error: {}", msg),
            SimulatorError::MatrixInversionError(msg) => write!(f, "Matrix Inversion Error: {}", msg),
            SimulatorError::EDFMGeometryError(msg) => write!(f, "EDFM Geometry Error: {}", msg),
            SimulatorError::ProppantTransportError(msg) => write!(f, "Proppant Transport Error: {}", msg),
            SimulatorError::WellboreHydraulicsError(msg) => write!(f, "Wellbore Hydraulics Error: {}", msg),
        }
    }
}

impl std::error::Error for SimulatorError {}
```

Line 15 of D8 requires: **no numerical non-convergence may trigger `panic!`; all compute methods return
`Result<T, SimulatorError>`.**

### 8.2 `FractureGeometry`

```rust
/// Геометричні та фільтраційні параметри елемента тріщини
# [derive(Debug, Clone)]
pub struct FractureGeometry {
    pub fracture_id: usize,
    pub length: f64,          // Довжина тріщини [m / ft]
    pub height: f64,          // Висота тріщини [m / ft]
    pub aperture: f64,        // Розкриття wf [m / ft]
    pub permeability: f64,    // Проникність kf [mD / D]
    pub conductivity: f64,    // Провідність kf * wf [mD·m / D·ft]
}
```

Carries **no** $\sigma_h$, $E$, $\nu$, $K_{IC}$, coordinates, or per-step geometry.

### 8.3 `MatrixCell`

```rust
/// Властивості матричного осередку сітки FVM/FEM
# [derive(Debug, Clone)]
pub struct MatrixCell {
    pub cell_id: usize,
    pub volume: f64,            // Об'єм осередку [m³ / ft³]
    pub pressure: f64,          // Поточний тиск [Pa / psi]
    pub permeability: [f64; 3], // Проникність по X, Y, Z [mD / D]
    pub porosity: f64,          // Пористість [безрозмірна]
}
```

### 8.4 `EDFMIntersection`

```rust
/// Результат геометричного перетину тріщини з осередком матриці за моделлю EDFM
# [derive(Debug, Clone)]
pub struct EDFMIntersection {
    pub intersection_id: usize,
    pub cell_id: usize,
    pub fracture_id: usize,
    pub intersection_area: f64, // Площа перетину [m² / ft²]
    pub average_distance: f64,  // Середня серединна відстань від матриці до тріщини [m / ft]
    pub transmissibility: f64,  // Коефіцієнт провідності EDFM [m³/(Pa·s) / bbl/(psi·day)]
}
```

### 8.5 `trait FractureMatrixFlux`

```rust
/// Інтерфейс для потокобезпечного обчислення міжфазних та міжпросторових потоків (Send + Sync)
pub trait FractureMatrixFlux: Send + Sync {
    /// Чисте обчислення масового потоку компонента між матрицею та тріщиною без внутрішньої мутації стану
    fn calculate_mass_flux(
        &self,
        matrix: &MatrixCell,
        fracture: &FractureGeometry,
        intersection: &EDFMIntersection,
        fluid_viscosity: f64,
        fluid_density: f64,
    ) -> Result<f64, SimulatorError>;

    /// Обчислення оновленого коефіцієнта провідності EDFM для батч-стрибка тиску
    fn compute_transmissibility(
        &self,
        fracture: &FractureGeometry,
        matrix: &MatrixCell,
        intersection_area: f64,
        average_distance: f64,
    ) -> Result<f64, SimulatorError>;
}
```

> ⚠️ **CONF-50 (continued).** `calculate_mass_flux` takes only **viscosity and density** — no pressure
> difference, no composition, no capillary term, no phase saturations — and returns a **single `f64`**
> rather than a component vector. With D2 §4.1's actual transfer function
> $q_{m-f,\alpha} = \sigma V_{block}\frac{K_m k_{r\alpha}}{\mu_\alpha}\Delta\Phi_{\alpha,m-f}$,
> the trait signature **cannot express** the physics D2 specifies: no $\sigma$, no $V_{block}$, no
> $k_{r\alpha}$, no $\Delta\Phi$, no phase index $\alpha$, and no component index $i$.

---

## 9. §4 — Economics and the NPV-unification criteria

**Dynamic CAPEX (identical to D1/D6/D7):**

$$\text{CAPEX} = N_{inj}\times\$1.5\text{M} + N_{prod}\times\$1.0\text{M} + N_{patterns}\times\$250\text{k} + \$3\text{M} + \$10\text{M}\left(\frac{Q_{recycle,peak}}{20\,000}\right)^{0.65} + \$5\text{M}$$

**Unified DCF:** `r = 0.10` example, mid-year factor $DF(t)=(1+r)^{-(t-0.5)}$.

**Required cross-artefact agreement:** `economic_npv_usd` (manifest) = `results["npv"]` (engine) =
`Cumulative_NPV_USD` (`cash_flows_yearly.csv`) to within **$0.01**.

### 9.1 The three numbered NPV-unification acceptance criteria

1. **Single computational trajectory.** NPV calculation paths in `surrogate_engine.py`,
   `surrogate_models.py` and `run_exporter.py` must be fully unified. The simplified averaging method
   of cash flows (of the type `N_p / 15`) is **strictly forbidden**.
2. **Absolute agreement accuracy (0.00 discrepancy).** The three artefacts above must agree within **0.01**.
3. **Complete removal of heuristic penalties.** Non-physical multipliers and algorithmic penalty
   barriers (`result *= breakthrough_impact`, `-1.0×10^{12}`) fully removed; gas-breakthrough dynamics
   regulated **exclusively** by rising recycle/compression operating costs in the DCF.

> [!CAUTION]
> **CONF-55 — D8 names live Python files as if the fix were already made.** Criterion 1 requires changes
> to `surrogate_engine.py`, `surrogate_models.py` and `run_exporter.py` — all of which exist. Criterion 3
> asserts the penalties are "fully removed" — but `FAILURE_PENALTY = -1e12` remains a **documented
> invariant** (`agent_wiki/README.md` invariant 12), and `core/objectives/wrapper.py` is documented as
> applying "containment/remediation penalties" to `profiles["npv"]`. **D8 §4.2 criteria 1–3 are, at
> present, unfulfilled requirements written as accomplishments.**
>
> Note also criterion 1 forbids `N_p / 15`-style averaging — compare the **live** implementation at
> `core/engine_surrogate/surrogate_engine.py:646-651`, which sums per-timestep annual cash flows
> directly. Criterion 1's second half appears already satisfied in the active engine; its first half
> (unifying three files) has not been measured.

---

## 10. Hard architectural constraints D8 declares as de-facto invariants

1. No heap allocation per cell in computational loops.
2. State in contiguous `Vec<T>` / typed arena.
3. Zero-copy `&[T]` / `&mut [T]`.
4. No `panic!`; all methods return `Result`.
5. Traits are `Send + Sync` with pure functions.

> These five are compatible in principle with the live Python engine only by analogy (NumPy arrays are
> contiguous, and `agent_wiki/README.md` invariant 12 already mandates zero silent swallowing and
> explicit mathematical penalties). **Items 1–3 have no Python analogue** — NumPy allocates freely.
> Item 4 and 5 map onto existing conventions; items 1–3 do not. See
> [`evaluation_plan.md`](evaluation_plan.md) §4.

---

## 11. D8 gaps

| Gap |
|---|
| No fracture-network model catalogue (PKN/KGD/DPM/cohesive/phase-field/Peridynamics/PFC/RBSM — 0 hits) |
| No per-model fracture equations, dofs or applicability ranges |
| No completion-type taxonomy, gravel pack, ICD/AICD specs, micro-annulus mechanics |
| No HF workflow: stage count, cluster spacing, perforation strategy per stage, fluid rheology, proppant schedule, shut-in schedule |
| No geomechanics section: no Mohr–Coulomb envelope, no slip tendency $T_s$, no fault reactivation, no poroelastic stress path, no geomechanical injection ceiling |
| **No test or validation section**: no `#[test]`, no benchmark dataset, no acceptance list, no property-based tests, no V&V hierarchy, no convergence requirement |
| §2.2 and §2.4 both use duplicated `1.` bullet numbering |
| `τ = 1.5` treated as dimensionless against a pore-volume multiple |