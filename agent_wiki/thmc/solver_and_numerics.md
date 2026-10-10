# Solver, Numerics and Rust Type Definitions — D1 §10 + D2

Sources: `3D THMC_docs/Технічна специфікація розробки ядра draft.md` (452 lines) and D1 §10.
**Design intent only — no Rust crate exists in this repository.**

> [!WARNING]
> **The listings below are verbatim from the source documents, including their bugs.**
> They are reproduced so an agent can see exactly what is (and is not) specified.
> The intended source form is `#[derive(...)]`; the documents write `# [derive(...)]`
> (space after `#`) and have no code fences.

---

## 1. Error taxonomy — D2

### 1.1 `ComputationalError`

```rust
# [derive(Debug, Clone, PartialEq)]
pub enum ComputationalError {
    NonFiniteValueInterrupted {
        variable_name: &'static str,
        cell_id: usize,
        value: f64,
    },
    NewtonRaphsonDiverged {
        iterations_taken: usize,
        last_residual_norm: f64,
    },
    ThermodynamicFlashFailed {
        pressure: f64,
        temperature: f64,
        reason: String,
    },
    SingularJacobianMatrix {
        row_index: usize,
    },
    PhysicalConstraintViolated {
        description: String },
}
```

### 1.2 `ConvergenceControl` and `StateValidator`

```rust
# [derive(Debug, Clone, Copy)]
pub struct ConvergenceControl {
    pub max_newton_iterations: usize,
    pub tolerance_residual: f64,
    pub line_search_max_steps: usize,
    pub damping_factor_min: f64,
}

pub trait StateValidator {
    fn validate_state_variables(&self) -> Result<(), ComputationalError>;
}
```

> ⚠️ **CONF-12.** All four `ConvergenceControl` fields have **no numeric defaults anywhere** in D1–D8,
> and in the listing below **none of them is read** except `max_newton_iterations` and
> `tolerance_residual`. `line_search_max_steps` and `damping_factor_min` are declared and never used.

---

## 2. Newton–Raphson — D2 (verbatim listing)

```rust
pub struct NewtonRaphsonSolver {
    pub control: ConvergenceControl,
}

impl NewtonRaphsonSolver {
    pub fn step_time(
        &self,
        state: &mut [f64],
        residuals: &mut [f64],
        dt: &mut f64,
    ) -> Result<f64, ComputationalError> {
        let mut lambda = 1.0;
        let mut iteration = 0;

        while iteration < self.control.max_newton_iterations {
            if let Err(err) = self.compute_residuals_and_jacobian(state, residuals) {
                *dt *= 0.5;   // Адаптивне скорочення кроку
                return Err(err);
            }

            let norm = self.calculate_l2_norm(residuals);
            if !norm.is_finite() {
                *dt *= 0.25;
                return Err(ComputationalError::NonFiniteValueInterrupted {
                    variable_name: "L2_Norm_Residual",
                    cell_id: 0,
                    value: norm,
                });
            }

            if norm < self.control.tolerance_residual {
                return Ok(*dt);
            }

            // Докремент ітерації та адаптація кроку через Line Search
            iteration += 1;
            lambda *= 0.85;
        }

        *dt *= 0.5;
        Err(ComputationalError::NewtonRaphsonDiverged {
            iterations_taken: iteration,
            last_residual_norm: self.calculate_l2_norm(residuals),
        })
    }

    fn compute_residuals_and_jacobian(
        &self,
        state: &[f64],
        residuals: &mut [f64],
    ) -> Result<(), ComputationalError> {
        for (i, val) in state.iter().enumerate() {
            if !val.is_finite() {
                return Err(ComputationalError::NonFiniteValueInterrupted {
                    variable_name: "StateVariable",
                    cell_id: i,
                    value: *val,
                });
            }
        }
        Ok(())
    }

    fn calculate_l2_norm(&self, vec: &[f64]) -> f64 {
        vec.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}
```

> [!CAUTION]
> **The listing contradicts its own prose and is not a solver.** Measured against the listing:
>
> | Claimed | Actual in listing |
> |---|---|
> | "backtracking line search" | `lambda *= 0.85` computes a scalar that is **never applied to any step** — there is no `Δx` in the function |
> | "compute_residuals_and_jacobian" | computes **no Jacobian** and **fills no residuals**; it is a finiteness scan returning `Ok(())` |
> | "linear SLE solve" | **no linear solver exists** in the listing |
> | Armijo–Goldstein residual reduction `‖R(x+λΔx)‖ < ‖R(x)‖` | never evaluated |
> | SIMD auto-vectorisation AVX-512 / FMA3 | listing is a scalar `.map().sum().sqrt()` |
> | `damping_factor_min` bounds `λ` | `λ` starts at `1.0`, above any plausible minimum, and is unbounded below only by exhaustion |
>
> The residual vector `residuals` is written by the caller and never written by the solver, so the
> `norm < tolerance_residual` test can never legitimately succeed. Tracked **CONF-13**.

---

## 3. Adaptive time-stepping — D2

**Prose:** on thermal / hydrodynamic / geochemical divergence the controller **cancels** the current
step and recomputes $\Delta t_{new} = \eta\,\Delta t_{old}$ with **$\eta \in [0.1,\,0.5]$**.

**In-code cut factors:**

| Trigger | Factor |
|---|---|
| residual / Jacobian error | `*dt *= 0.5` |
| non-finite L2 residual norm | `*dt *= 0.25` |
| divergence exit | `*dt *= 0.5` |
| iteration damping | `lambda *= 0.85` |

**Absent:** step-**growth** rule, minimum/maximum $\Delta t$, reporting-interval logic, operator-split
iteration limits, per-mode convergence criteria.

> ⚠️ Two `δt` symbols with different meanings: Schwarz sub-domain micro-step (D1 §2.3) and wellbore
> sub-step (D1 §9.3). See **CONF-08**.

---

## 4. Line search — D2

Newton with backtracking, residual-reduction condition:

$$\left\|\mathbf{R}\!\left(\mathbf{x}^{(k)} + \lambda\,\Delta\mathbf{x}\right)\right\|_2 < \left\|\mathbf{R}\!\left(\mathbf{x}^{(k)}\right)\right\|_2,\qquad \lambda \in [\lambda_{min}, 1)$$

Armijo–Goldstein named; only the simple residual-reduction form is given. Stopping test
`norm < tolerance_residual` with `norm = sqrt(Σ v²)`.

---

## 5. Linear algebra layer — D1 §10.3

| Component | Role | Status in the docs |
|---|---|---|
| **`faer`** | Rust-native SIMD solver, dense and sparse subsystems | Named |
| **`russell_sparse`** | C-bind interface to **MUMPS** and **UMFPACK** | Named |
| **`cuDSS`** | NVIDIA direct sparse solver, computes in GPU memory | Named |
| Rayon | Parallel Jacobian + residual assembly | Named |
| Hyper-dual AD (`feos-ad`, `num-dual`) | Analytic 1st/2nd derivatives of fugacity at $10^{-16}$ | Named |

> [!CAUTION]
> **The conditioning layer is entirely unspecified.** No preconditioner (no ILU/ILUT, no AMG, no block
> preconditioning), no sparse storage format (no CSR/CSC/CSF/ELL, no fill-reducing ordering), no
> symbolic/numeric phase separation, **no Krylov / iterative method of any kind**. Only direct solvers.
> Tracked **CONF-14**.

**Zero-allocation parallel assembly** is the stated architecture: all heap allocation excluded from
hot loops during Newton; per-worker element matrices and residual buffers pre-allocated in contiguous
arrays.

---

## 6. Hyper-dual automatic differentiation — D1 §4.4 + D3 §1.3.1

Exact Jacobian assembly needs $\partial\varphi_i/\partial P$, $\partial\varphi_i/\partial x_j$,
$\partial^2\varphi_i/(\partial x_j\,\partial x_k)$ **without finite differences**.

Dual numbers flow from the flash directly into the global Newton Jacobian. D3 §1.3.1:

$$\mathbf{f}_j^{AD} = f_j + \frac{\partial f_j}{\partial P}dP + \frac{\partial f_j}{\partial T}dT + \sum_{k=1}^{N_c-1}\frac{\partial f_j}{\partial x_k}dx_k$$

Claim: derivatives feed `J_THMC` **without numerical finite differences**, guaranteeing **quadratic**
Newton convergence under asphaltene phase transition and hydrate crystallisation.

AD numeric type named: **`num_dual::Dual64`**.

---

## 7. Adjoint gradient engine — D1 §10.2

$$(\partial R/\partial x)^\top \lambda = \partial J/\partial x$$
$$\nabla_u J = \frac{\partial J}{\partial u} - \lambda^\top\frac{\partial R}{\partial u}$$

Objective $J$ is explicitly **NPV**. Claim: exact gradient in **one backward pass, independent of the
number of control variables**, avoiding thousands of forward-model reruns.

> ⚠️ **CONF-04.** This places NPV inside the core while D1 §1.1 explicitly exiles economics from the
> core. Also: no preconditioner is specified for the adjoint linear solve (CONF-14), and no
> gradient-verification procedure is specified anywhere in D1–D8.

---

## 8. Fractured-continuum data structures — D2 §4.1

```rust
# [derive(Debug, Clone)]
pub struct DualPorositySystem {
    pub matrix_phi: f64,
    pub fracture_phi: f64,
    pub matrix_perm: f64,
    pub fracture_perm: f64,
    pub shape_factor_sigma: f64,
}

# [derive(Debug, Clone)]
pub struct DualPermeabilitySystem {
    pub dp_base: DualPorositySystem,
    pub matrix_to_matrix_transmissibility: f64,
}

# [derive(Debug, Clone)]
pub struct MincSubvolume {
    pub subvolume_index: usize,
    pub volume_fraction: f64,
    pub distance_from_fracture: f64,
    pub internal_porosity: f64,
    pub internal_permeability: f64,
}

# [derive(Debug, Clone)]
pub enum FracturedContinuumModel {
    DualPorosity(DualPorositySystem),
    DualPermeability(DualPermeabilitySystem),
    MINC {
        global_fracture: DualPorositySystem,
        nested_subvolumes: Vec<MincSubvolume>,
    },
}
```

Note: **only** `ComputationalError` derives `PartialEq`; these four carry only `Debug, Clone`.

### 8.1 Model comparison (D2 §4.1)

| Criterion | Dual Porosity | Dual Permeability | MINC + EDP |
|---|---|---|---|
| Transfer matrix→fracture | pseudo-stationary or non-stationary (capillary, gravitational, viscous) | same, in every block | intra-matrix diffusive transfer between nested sub-zones and fractures |
| Fracture→fracture transfer | present | present | present between fractured-space blocks |
| Complexity | low — **2× equations per block** | medium — 2× equations + extra inter-block links | high — **`N_sub` × equations per block** |
| Scope | intensely fractured, impermeable matrix | fractured with permeable matrix + gravitational cross-flow | thermal/chemical EOR with slow matrix diffusion |

### 8.2 Matrix↔fracture transfer function

$$q_{m-f,\alpha} = \sigma\,V_{block}\,\frac{K_m\,k_{r\alpha}}{\mu_\alpha}\,\Delta\Phi_{\alpha,m-f}$$

$$\Delta\Phi_{\alpha,m-f} = (P_{m,\alpha} - P_{f,\alpha}) - \rho_\alpha g\,(z_m - z_f) + \Delta P_{c,\alpha}$$

### 8.3 Pattern breakthrough time

$$t_{bt} = \frac{V_{p,pattern}\,(1 - S_{wi})}{K_{koval}\,q_{inj,pattern}}$$

> ⚠️ $1 - S_{wi}$ is used as the mobile hydrocarbon fraction, ignoring $S_{or}$ and any gas
> saturation. No unit system, condition basis, or $K_{koval}$ value is given. Tracked **CONF-15**.

---

## 9. Fault and fracture relations — D2

**NNC transmissibility:**

$$T_{NNC} = \frac{A_{NNC}}{\dfrac{d_i}{K_i} + \dfrac{d_j}{K_j} + R_{fault}},\qquad R_{fault} = \frac{t_{fault}}{K_{fault}}$$

**Coulomb failure stress:** $\Delta\text{CFS} = \Delta\tau - \mu(\Delta\sigma_n - \Delta P)$;
**$\Delta\text{CFS} \ge 0$ ⇒ shear reactivation**.

**Shale Gouge Ratio:** $\text{SGR} = \dfrac{\sum_k V_{shale,k}\,\Delta z_k}{H}\times 100\%$ over throw $H$.

**Seal → conduit transition:**
1. $\Delta\text{CFS} < 0$ **and** $\text{SGR} \ge 20\%$ ⇒ closed seal with $K_{fault} = K_{base}\cdot 10^{-4\,\text{SGR}}$.
2. At $\Delta\text{CFS} \ge 0$ ⇒ gouge dilation and failure; $K_{fault,active} = K_{fault,base} + \dfrac{e_{shear}^3}{12\,t_{fault}}$.

**Barton–Bandis hydraulic aperture:**

$$e = e_0 - \frac{\sigma_n'\,v_{max}}{\sigma_n' + K_{j0}\,v_{max}}$$

**Anisotropic permeability tensor from a fracture set (Poiseuille → tensor):**

$$K_{ij}(\sigma'_{eff}) = \sum_{k=1}^{N_{frac}} \frac{e_k^3(\sigma'_{eff})}{12\,d_k}\left(\delta_{ij} - n_i^{(k)} n_j^{(k)}\right)$$

---

## 10. Thermo-chemo-mechanical coupling — D2 §3.1

**Stress equilibrium:** $\nabla\cdot\boldsymbol{\sigma}' - \alpha\nabla P + \mathbf{f} = 0$ with
Biot coefficient $\alpha = 1 - K_{dry}/K_s$.

**Constitutive law:**

$$\boldsymbol{\sigma}' = \mathbf{C} : \left(\boldsymbol{\varepsilon} - \alpha_T (T - T_0)\mathbf{I} - \boldsymbol{\varepsilon}_{chem}\right)$$

$$\boldsymbol{\varepsilon}_{chem} = \tfrac{1}{3}\Delta V_{m,tot}\,\mathbf{I}$$

> ⚠️ **CONF-16.** $\boldsymbol{\varepsilon}_{chem}$ is an **unnormalised** strain — a strain requires
> division by a reference (or bulk) volume. As written, $\Delta V_{m,tot}$ carries volume units and the
> tensor components are not dimensionless. This is a dimensional error, not a modelling choice.

**Pressure-coupling channel** (production-weighted):

$$\bar{P}_{res,eff} = \frac{\sum_t P_{res}(t)\,q_o(t)\,\Delta t}{\sum_t q_o(t)\,\Delta t}$$

D2 §6.1 requires RF and the miscibility parameter $\omega$ be recomputed **exclusively** via
$\bar{P}_{res,eff}$.

> ⚠️ **CONF-17.** This contradicts D2 §3.1's fully-coupled Biot formulation in the same document: a
> fully-coupled solve has cell-local pressure, not one production-weighted scalar.

---

## 11. Porosity / permeability evolution — D2 §2.2

$$\Delta V_m = r_m\,V_{m,molar}\,A_m\,\Delta t$$
$$\phi^{t+\Delta t} = \phi^{t} - \sum_m \Delta V_m$$
$$K^{t+\Delta t} = K_0\left(\frac{\phi^{t+\Delta t}}{\phi_0}\right)^3\frac{1-\phi_0}{(1-\phi^{t+\Delta t})^2}$$

with an alternative power law $K = K_0(\phi^{t+\Delta t}/\phi_0)^{\gamma}$, $\gamma$ unspecified.

> ⚠️ **CONF-18.** Porosity is updated by **direct subtraction** of reacted-phase volume change with
> **no solid-volume (Bethel) correction** — porosity is a ratio of volumes, so a bare subtraction of
> a volume is dimensionally inconsistent unless implicitly normalised.

---

## 12. GEP API — D2 §1.2 (identifiers verbatim)

| Component | Signature / structure |
|---|---|
| Chromosome | `struct Chromosome { genes: Vec<Gene> }`, fixed length |
| Gene | `struct Gene { head: Vec<Symbol>, tail: Vec<Symbol> }`, tail length `t = h(n-1)+1`, `n` = max arity |
| ORF / K-expressions | `Vec<Symbol>`, read from 0 to phenotype terminator, **Karva** language |
| Random-constant domain | `struct DcDomain { data: Vec<Symbol> }`, length `t` |
| Mutation | `fn mutate(&mut self, p_m: f64)` |
| IS transposition | `fn is_transposition(&mut self, p_is: f64)` |
| RIS transposition | `fn ris_transposition(&mut self, p_ris: f64)` |
| Gene transposition | `fn gene_transposition(&mut self, p_gt: f64)` |
| 1-point recombination | `fn one_point_recombination(p1: &Self, p2: &Self) -> (Self, Self)` |
| 2-point recombination | `fn two_point_recombination(p1: &Self, p2: &Self) -> (Self, Self)` |
| Gene recombination | `fn gene_recombination(p1: &Self, p2: &Self) -> (Self, Self)` |

**Never specified:** `h`, function set `F`, terminal set, fitness function, population size, generation
count, ERC array size, selection scheme, and the binding between GEP-discovered constants and the
deterministic solver inputs.

---

## 13. External (Python-side) contract identifiers cited by D2

`results["npv"]`, `economic_npv_usd`, `Cumulative_NPV_USD`, `cash_flows_yearly.csv`,
`test_surrogate_engine.py`.

| Identifier | Exists in this repo? |
|---|---|
| `results["npv"]` | Yes — `core/engine_surrogate/surrogate_engine.py:172` |
| `economic_npv_usd` / `Cumulative_NPV_USD` / `cash_flows_yearly.csv` | Yes — `tests/test_physical_invariants.py:8,159,181,184,211` |
| `test_surrogate_engine.py` | **No** — `Test-Path` → `False` |

D2 §6.2 claims "**48 module tests** in `test_surrogate_engine.py`" pass. **That file does not exist.**
Tracked **CONF-03**.

---

## 14. Acceptance-criteria checklist — D2 §6.2 (all boxes `[ ]`)

| # | Criterion | Exact requirement |
|---|---|---|
| 1 | Material balance | Mass conservation / material balance of CO₂ + hydrocarbons in a closed volume, convergence **> 99.9 %** |
| 2 | NPV accuracy | **Zero** divergence between `economic_npv_usd` in the manifest, final `results["npv"]`, and last `Cumulative_NPV_USD` in `cash_flows_yearly.csv` |
| 3 | Zero-panic robustness | **48** module tests in `test_surrogate_engine.py` at pressure gradients to **276 MPa** with extreme concentration jumps, **without NaN, Inf or `panic!`** |
| 4 | Pattern sizing | For **1 000 acres** → `N_patterns = 25`, rates limited **≤ 1000 BOPD**, total CAPEX dynamically within **$40M–$75M** |
| 5 | HCPVI limit | At **0.18 HCPVI** throughput, tertiary RF strictly limited to **≤ 15 % OOIP** |
| 6 | EOR indicators | (a) net CO₂ utilisation **0.25–0.50 t/STB (5–10 MSCF/STB)**; (b) recycle share **30 %–55 %** of gross injection; (c) CO₂ GOR in **year 15** within **5 000–20 000 SCF/STB** |

> ⚠️ **CONF-19.** Criterion 3 references a non-existent test file; the whole checklist is unchecked;
> §6.1 is written as a **remediation to-do list** ("following strict limits and formulas are
> implemented/implemented into the numerical core"), i.e. requirements, not observations. Internal
> conflicts: recycle share **30–60 %** (§6.1) vs **30–55 %** (§6.2); GOR **5 000–25 000** (§6.1) vs
> **5 000–20 000 SCF/STB at year 15** (§6.2); net utilisation "**≥ 2.5 MSCF/STB**" vs "**5–10 MSCF/STB**".
> Criterion 4's $40M–$75M band is **contradicted by D6's own worked example, which totals $90.95M**
> (see [`satellite_toolkit.md`](satellite_toolkit.md) §5). Tracked **CONF-20**.