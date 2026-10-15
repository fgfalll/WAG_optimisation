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

### 1.3 Resolved Primary Variable Vector & Convergence Control (Approved 15-10-2026)

Under the 15-10-2026 forensic engineering audit, the primary variables and convergence control are resolved and locked:

#### Primary Variable Vector $\mathbf{X}$ (Overall Composition Formulation — Flaw 3 / CONF-14)
For an $N_c$-component hydrocarbon + water + thermal + geomechanical system, the primary state vector per cell is:
$$\mathbf{X} = \begin{bmatrix} P \\ z_1 \\ z_2 \\ \vdots \\ z_{N_c-1} \\ z_w \\ T \\ u_x \\ u_y \\ u_z \end{bmatrix} \in \mathbb{R}^{N_c + 5}$$
where:
- $P$: Reference pore pressure [Pa].
- $z_1 \dots z_{N_c-1}$: Overall hydrocarbon mole fractions (with $z_{N_c} = 1 - z_w - \sum_{i=1}^{N_c-1} z_i$).
- $z_w$: **Total water mole fraction** (must be explicitly tracked in primary variables; closing Flaw 3).
- $T$: Temperature [K].
- $\mathbf{u} = [u_x, u_y, u_z]^T$: Solid displacement vector [m].

#### Resolved `ConvergenceControl`
```rust
#[derive(Debug, Clone, Copy)]
pub struct ConvergenceControl {
    /// Maximum nonlinear Newton iterations before time-step cutback (default: 25)
    pub max_newton_iterations: usize,
    /// L2 norm residual convergence tolerance (default: 1.0e-6)
    pub tolerance_residual: f64,
    /// Maximum backtracking line search trials (default: 10)
    pub line_search_max_steps: usize,
    /// Minimum damping parameter lambda_min (default: 1.0e-4)
    pub damping_factor_min: f64,
    /// Armijo-Goldstein sufficient decrease parameter c1 (default: 1.0e-4)
    pub armijo_c1: f64,
    /// Adaptive time-step cutback factor eta_cut (default: 0.5)
    pub dt_cutback_factor: f64,
    /// Adaptive time-step growth factor eta_grow (default: 1.25)
    pub dt_growth_factor: f64,
    /// Maximum permissible time-step dt_max [s]
    pub dt_max: f64,
    /// Minimum permissible time-step dt_min [s]
    pub dt_min: f64,
}

impl Default for ConvergenceControl {
    fn default() -> Self {
        Self {
            max_newton_iterations: 25,
            tolerance_residual: 1.0e-6,
            line_search_max_steps: 10,
            damping_factor_min: 1.0e-4,
            armijo_c1: 1.0e-4,
            dt_cutback_factor: 0.5,
            dt_growth_factor: 1.25,
            dt_max: 86400.0 * 30.0, // 30 days
            dt_min: 1.0,           // 1 second
        }
    }
}
```

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

### 4.1 Resolved True Newton-Raphson Solver Specification (Closing CONF-13)

Under the approved implementation plan, the Newton-Raphson loop executes genuine nonlinear solves with proper residual updates, linear solution, Armijo line search, and exponential soft-start:

```rust
pub struct ProductionNewtonSolver {
    pub control: ConvergenceControl,
    pub linear_solver: Box<dyn BlockLinearSolver>,
}

impl ProductionNewtonSolver {
    /// Solve single time-step with true Newton-Raphson, Armijo line search, and adaptive cutback
    pub fn step_time<M: ReservoirModel>(
        &self,
        model: &mut M,
        state: &mut PrimaryStateVector,
        dt: &mut f64,
    ) -> Result<f64, ComputationalError> {
        let mut iteration = 0;
        let mut residuals = model.allocate_residual_buffer();
        let mut jacobian = model.allocate_jacobian_matrix();

        while iteration < self.control.max_newton_iterations {
            // 1. Evaluate residuals R(x) and Jacobian J(x)
            model.assemble_residuals_and_jacobian(state, &mut residuals, &mut jacobian)?;

            // 2. Check nonlinear convergence
            let norm = residuals.l2_norm();
            if !norm.is_finite() {
                *dt = (*dt * self.control.dt_cutback_factor).max(self.control.dt_min);
                return Err(ComputationalError::NonFiniteValueInterrupted {
                    variable_name: "L2_Norm_Residual",
                    cell_id: 0,
                    value: norm,
                });
            }
            if norm < self.control.tolerance_residual {
                // Adaptive time-step growth on clean convergence
                if iteration < 8 {
                    *dt = (*dt * self.control.dt_growth_factor).min(self.control.dt_max);
                }
                return Ok(*dt);
            }

            // 3. Solve linear system J(x) * dx = -R(x) using Block-CPR preconditioned FGMRES
            let delta_x = self.linear_solver.solve(&jacobian, &residuals)?;

            // 4. Backtracking Armijo-Goldstein line search: ||R(x + lambda * dx)|| <= (1 - c1 * lambda) * ||R(x)||
            let mut lambda = 1.0;
            let mut step_accepted = false;
            for _ in 0..self.control.line_search_max_steps {
                let trial_state = state.step_update(&delta_x, lambda);
                let trial_res = model.evaluate_residuals_only(&trial_state)?;
                let trial_norm = trial_res.l2_norm();

                if trial_norm <= (1.0 - self.control.armijo_c1 * lambda) * norm {
                    *state = trial_state;
                    step_accepted = true;
                    break;
                }
                lambda *= 0.5;
                if lambda < self.control.damping_factor_min {
                    break;
                }
            }

            if !step_accepted {
                *dt = (*dt * self.control.dt_cutback_factor).max(self.control.dt_min);
                return Err(ComputationalError::NewtonRaphsonDiverged {
                    iterations_taken: iteration,
                    last_residual_norm: norm,
                });
            }

            iteration += 1;
        }

        *dt = (*dt * self.control.dt_cutback_factor).max(self.control.dt_min);
        Err(ComputationalError::NewtonRaphsonDiverged {
            iterations_taken: iteration,
            last_residual_norm: residuals.l2_norm(),
        })
    }
}
```

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
> **The conditioning layer was entirely unspecified in legacy drafts.** No preconditioner (no ILU/ILUT, no AMG, no block
> preconditioning), no sparse storage format, only direct solvers. Tracked **CONF-14**.

### 5.1 Resolved Linear Solver Architecture: Block-CPR Preconditioned FGMRES (Approved 15-10-2026 / Closing CONF-14)

Standard CPR (Constrained Pressure Residual) degrades when applied directly to coupled THMC systems because of vector elasticity $[u_x, u_y, u_z]$ degrees of freedom. The approved 3D THMC engine uses **Block-CPR (Schur Splitting)**:

```
                      Global THMC Jacobian [J]
           [ A_pp   A_pz   A_pu ] [ dp ]   [ -R_p ]
           [ A_zp   A_zz   A_zu ] [ dz ] = [ -R_z ]
           [ A_up   A_uz   A_uu ] [ du ]   [ -R_u ]
                                |
             +------------------+------------------+
             |                                     |
   Stage 1a (Pressure)                   Stage 1b (Displacement)
   Unsymmetric AMG on A_pp               Vector AMG on A_uu
   (BoomerAMG / SAMG)                    (Smoothed Aggregation)
             |                                     |
             +------------------+------------------+
                                |
                             Stage 2
                Global Smoothing: Block-ILU(1)
             Across all degrees of freedom [p, z, T, u]
                                |
                             Outer
                     FGMRES Acceleration
                 (Flexible GMRES with right precond)
```

1. **Stage 1a (Pressure Sub-Block Preconditioning)**:
   - Extract the elliptic pressure component $\mathbf{A}_{pp}$ via quasi-IMPES decoupling or true-IMPES transformation.
   - Solve with **Unsymmetric Algebraic Multigrid (AMG)** (e.g. BoomerAMG / classical Ruge-Stüben or classical aggregation).
2. **Stage 1b (Geomechanical Displacement Sub-Block Preconditioning)**:
   - Extract the vector elasticity block $\mathbf{A}_{uu}$.
   - Solve with **Vector AMG** (rigid body modes in nullspace, smoothed aggregation AMG).
3. **Stage 2 (Global Smoothing)**:
   - Apply 1 or 2 iterations of **Block-ILU(1)** (or parallel domain-decomposed restricted additive Schwarz ILU) across the entire coupled $[P, \mathbf{z}, z_w, T, \mathbf{u}]$ system to damp high-frequency composition and saturation residuals.
4. **Outer Krylov Acceleration**:
   - **Flexible GMRES (FGMRES)** with right preconditioning, restarted at 30 iterations.
   - Zero-allocation memory design: workspace vectors pre-allocated per thread/worker during milestone M0/M1 setup.

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

### 7.1 Resolved Core-Satellite Adjoint Boundary (Closing CONF-04 & CONF-58/59)

The reservoir engineer ruling and forensic engineering audit establish a clean physical separation:
1. **In-Core PDE Adjoint Layer**:
   - The simulation core solves the adjoint system with respect to physical well production/injection streams:
     $$\mathbf{J}^T \boldsymbol{\lambda}_i = \frac{\partial q_i}{\partial \mathbf{X}}$$
   - Exposes exact production adjoint sensitivities:
     $$\frac{\partial q_i(t)}{\partial \mathbf{u}} = \boldsymbol{\lambda}_i^T \frac{\partial \mathbf{R}}{\partial \mathbf{u}}$$
     for each physical phase $i \in \{\text{oil}, \text{gas}, \text{water}\}$ and component stream (pure CO₂, hydrocarbon sales gas).
   - Core linear solver reuses the **Block-CPR preconditioner** transposed for the adjoint equation.
2. **Off-Core Satellite Economic Gradient**:
   - Satellite toolkit applies commodity prices, Mid-Year Discrete Discounting $(1+r)^{-(t-0.5)}$, OPEX/CAPEX, carbon credits, and leakage penalties:
     $$\nabla_{\mathbf{u}} J_{NPV} = \sum_{t=1}^N \sum_i \frac{\partial J_{NPV}}{\partial q_i(t)} \frac{\partial q_i(t)}{\partial \mathbf{u}} + \frac{\partial J_{NPV}}{\partial \mathbf{u}}\Big|_{\text{direct}}$$
   - Guarantees PDE solver contains zero economic constants (INV-3).

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

> ⚠️ **CONF-16.** $\boldsymbol{\varepsilon}_{chem}$ was **unnormalised** in the legacy draft.

### 10.1 Resolved Normalised Chemical Strain & Local Coupling (Closing CONF-16 & CONF-17)

1. **Dimensionally Normalised Chemical Strain Tensor (CONF-16)**:
   $$\boldsymbol{\varepsilon}_{chem} = \frac{1}{3} \frac{\sum_m \Delta V_m}{V_{bulk}} \mathbf{I}$$
   where $\sum_m \Delta V_m$ is the total mineral volume alteration and $V_{bulk}$ is the cell bulk volume, ensuring strict dimensionless strain $[-]$.

2. **Cell-Local Poroelastic Coupling (CONF-17)**:
   The fully coupled THMC solver evaluates local Biot effective stress per cell:
   $$\boldsymbol{\sigma}'_{cell} = \boldsymbol{\sigma}_{cell} - \alpha_{cell} P_{cell} \mathbf{I}$$
   The global scalar production-weighted pressure $\bar{P}_{res,eff}$ is retired from in-core solves; local phase equilibria and rock-fluid mechanics respond strictly to cell-local $P_{cell}$ and $T_{cell}$.

3. **Resolved Elastoplasticity: Mohr-Coulomb / Drucker-Prager & Lode Angle (Flaw 4 / CONF-66)**:
   - **Convention**: Compression-positive ($\sigma > 0$ in compression).
   - **Meridian Identification**: Under compression-positive convention, triaxial compression yields $\det(\mathbf{s}) > 0 \implies \theta_L = -\pi/6$ (**Compression Meridian**). $\theta_L = +\pi/6$ is the **Extension Meridian**.
   - **Convexity Preservation**: Curvature numerator $N_\kappa = g(\theta_L)^2 + g(\theta_L) g''(\theta_L) \ge \kappa_{\min} > 0$ strictly enforced to guarantee convexity and prevent corner/apex divergence.
   - **Algorithmic Tangent**: Closest-Point Projection Method (CPPM) return mapping with mathematically consistent algorithmic tangent $\mathbf{D}^{alg} = \frac{\partial \boldsymbol{\sigma}_{n+1}}{\partial \boldsymbol{\varepsilon}_{n+1}}$ preserving quadratic Newton convergence.

---

## 11. Porosity / permeability evolution — D2 §2.2

$$\Delta V_m = r_m\,V_{m,molar}\,A_m\,\Delta t$$
$$\phi^{t+\Delta t} = \phi^{t} - \sum_m \Delta V_m$$
$$K^{t+\Delta t} = K_0\left(\frac{\phi^{t+\Delta t}}{\phi_0}\right)^3\frac{1-\phi_0}{(1-\phi^{t+\Delta t})^2}$$

> ⚠️ **CONF-18.** Porosity was updated by direct subtraction without solid-volume correction.

### 11.1 Resolved Solid-Volume Corrected Porosity Evolution (Closing CONF-18)

Porosity is a ratio of pore volume to bulk volume. Under concurrent mechanical deformation and geochemical dissolution/precipitation, the dimensionally consistent and volume-conserving update is:
$$\phi^{t+\Delta t} = \frac{\phi^t - \frac{\sum_m \Delta V_m}{V_{bulk}}}{1 + \varepsilon_v^{mech}}$$
where:
- $\Delta V_m = r_m V_{m,molar} A_m \Delta t$ is the net mineral reaction volume $[m^3]$.
- $\varepsilon_v^{mech} = \text{tr}(\boldsymbol{\varepsilon})$ is the mechanical volumetric strain $[-]$.
Permeability updates via the Carman-Kozeny relation anchored to the corrected $\phi^{t+\Delta t}$.

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

---

## 13. External (Python-side) contract identifiers cited by D2

`results["npv"]`, `economic_npv_usd`, `Cumulative_NPV_USD`, `cash_flows_yearly.csv`,
`test_surrogate_engine.py`.

| Identifier | Exists in this repo? | Status |
|---|---|---|
| `results["npv"]` | Yes — `core/engine_surrogate/surrogate_engine.py:172` | Python surrogate output key |
| `economic_npv_usd` / `Cumulative_NPV_USD` / `cash_flows_yearly.csv` | Yes — `tests/test_physical_invariants.py` | Export artefacts |
| `test_surrogate_engine.py` | **No** — `Test-Path` → `False` | Replaced by regression suite in `tests/` |

---

## 14. Resolved Acceptance-Criteria Specification (Closing CONF-19 & CONF-20)

| # | Criterion | Resolved Engine Requirement | Authority & Verification Reference |
|---|---|---|---|
| 1 | **Material Balance** | Component mass conservation residual $< 10^{-12}$ per time-step in solved Rust THMC core; surrogate diagnostic check $\ge 99.9\%$ | INV-1, INV-4, SPE 5 deck (`D:\RAG`) |
| 2 | **Economic Separation** | Zero economics inside PDE Jacobian (INV-3); off-core DCF matching within $\$0.01$ across exported profiles | Architecture Decision, Flaw 5 |
| 3 | **Zero-Panic Invariant** | Clean error handling with `Result<T, ComputationalError>`; zero `panic!`, NaN, or Inf up to 276 MPa | INV-1, INV-6 |
| 4 | **Physical Deliverability** | Arbitrary $\le 1000$ BOPD clamp removed (INV-7); deliverability governed by Peaceman well index $WI$ and skin floor $S_{min} = -\ln(r_e/r_w) + \Delta S_{margin}$ | Flaw 4, Karakas-Tariq (1991) |
| 5 | **Compositional Recovery** | Inert $1 - e^{-\text{HCPVI}/1.5}$ cap purged; recovery computed via exact component mass integral $M_i(t)$ | Flaw 2 / CONF-02 |
| 6 | **EOR Industrial Benchmarks** | Validated against SPE 5 reference (`D:\RAG\Data files\SPE5-ProbForecasting-BaseCase.txt`): GOR, WAG breakthrough, and recovery match published solution | Milestone M2 / Level 5 V&V |