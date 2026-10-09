# Adjudicated Spec Closures

**Split out of [`engine_invariants.md`](engine_invariants.md) on 09-10-2026.** **Section numbers are
preserved exactly** — every cross-reference in the wiki citing sections 7c through 7n still resolves to the
same content, just in this file.

**What belongs here:** closures of specific `CONF-*` conflicts, DOI resolution and verification, provenance
and literature rules, and the Karakas–Tariq / Penéloux work.

| Section | Subject |
|---|---|
| 7c | INV-7 domain-by-domain — multiphysics unmasking |
| 7d | Closures for C-63…C-68, and the gaps inside them |
| 7e | Closures for CONF-13/14/25/31/51, the data split, B-3 |
| 7f | CONF-14 overhaul, 2D split, Penéloux + Karakas-Tariq |
| 7g | Karakas-Tariq corrected, 3-way split, M0.1 gate, DOIs |
| 7h | Clamp purged, alpha-zero direction, DOI re-verification |
| 7i | DOI corrected, anisotropy transform, a withdrawn claim |
| 7j | NaN eliminated; the anisotropy remedy **inverts** |
| 7k | DOIs bound, Option (a) adopted, **Symbol Register Rule** |
| 7l | `completion_skin_spec.md` review |
| 7m | Local library audit — `D:\RAG`, 129 PDFs |
| 7n | Literature parked; math batch CONF-15/16/25/29 |

**If you are here for thermodynamics or transport, you want
[`engine_numerics.md`](engine_numerics.md).** **If you are here for geomechanics,
[`engine_constitutive.md`](engine_constitutive.md).**

---

## 7c. 🔴 INV-7 domain-by-domain — multiphysics unmasking (added 08-10-2026)

Owner ruling, extending INV-7 across all nine domains. **Verbatim framing:** the engine is an
**unconstrained, pure-PDE physical solver** and shall accept *"any arbitrary, user-defined operational
scenario, extreme fluid rate, severe thermal shock, extreme geomechanical load, or reactive geochemical
composition"* without rate clamps, pressure ceilings, stress clipping, rate limiters or fault caps.

**What is forbidden, by domain:**

| Domain | Forbidden | Correct unconstrained behaviour |
|---|---|---|
| **Geomechanics / elastoplasticity** | Clipping $\sigma_{ij}$ or $\sigma'_{ij}$ to keep them off the yield surface | $\sigma'_{ij}=\sigma_{ij}-\alpha_B P\delta_{ij}$ evolves freely; on $F(\sigma',\varepsilon_p)=0$ the engine computes **non-associated** $\varepsilon_p$, dilatancy and compaction-induced $\phi(\varepsilon_p)$ |
| **Caprock containment** | Artificially capping caprock stress | At $\sigma'_3\le -\sigma_t$ or $\Delta CFS>0$, tensile/shear failure and upward leakage are computed explicitly, **with no pressure suppression** |
| **Reactive geochemistry** | Hard-capping $r_m$, clipping $C_k$, enforcing $\phi\le 0.60$ | Lasaga $r_m=A_mk_m(1-Q/K_{eq})^m$ runs free; wormholing emerges from $Da$/$Pe$ |
| **Fault mechanics** | Clamping $T_{\text{fault}}\in[0.001,1.0]$, preventing $u_s$ | Aperture from effective stress; $\Delta CFS$ governs `Seal`↔`Conduit`; NNC $T_{mf},T_{ff}$ follow the stress path |
| **Wellbore** | Minimum injection rates, clipped BHP, fixed skin floor $S\ge-5.0$ | Drift-flux $P_{\text{tubing}}(z)$ free; $S_{\min}=-\ln(r_e/r_w)+\Delta S_{\text{margin}}$; remediation fluids injected as **explicit mass streams** |

### 7c.1 🔴 Reversal: $\varepsilon_p$ is RESTORED — CONF-66 partly reopens

INV-1.6 had **removed** $\varepsilon_p$ because the design set names Mohr-Coulomb and Drucker-Prager
**without a yield surface, flow rule, or hardening** — emitting the quantity implied physics it did not
specify.

> **The owner has now supplied the missing physics**: a yield surface $F(\sigma',\varepsilon_p)=0$ and an
> explicitly **non-associated** plastic flow rule with dilatancy. That is a legitimate definition.

**Status: CONF-66 moves `RESOLVED` → `PARTIALLY_RESOLVED`.** $\varepsilon_p$ is restored to the Domain 5
schema. Three things are still unspecified and must be added before it is buildable:

| # | Gap | Why it blocks |
|---|---|---|
| 1 | **Return-mapping algorithm** (radial-return vs. closest-point, and which form) | Determines uniqueness and global convergence |
| 2 | **Consistent tangent / algorithmic derivative** | 🔴 **This is the C-35 class of defect.** A non-associated Drucker-Prager return map has a **non-symmetric** consistent tangent; feeding the symmetric approximation to Newton silently loses quadratic convergence |
| 3 | **Hardening law** | Is the surface elastic–perfectly-plastic, or does $\varepsilon_p$ strain-harden? Changes whether slip localises or spreads |

> ⚠️ **Drucker-Prager is the better default** for frictional rock and is inherently non-associated, so
> the owner's "non-associated" and "Drucker-Prager" are consistent. **Mohr-Coulomb** needs an explicit
> $\psi\ne\phi$ dilation angle to be non-associated.

### 7c.2 🔴 The Verma-Pruess guard is REQUIRED physics, not an artificial cap

The ruling says *"without artificial division-by-zero guards or k floors."* **Both must be reconsidered —
measured 08-10-2026**, $\phi_0=0.25$, $\phi_c=0.02$:

| $\phi_{acc}$ | ratio | $k/k_0$, $n=2$ | $k/k_0$, $n=1.5$ |
|---|---|---|---|
| 0.050 | +0.130 | 0.0170 | 0.0471 |
| **0.020** | 0.000 | 0.0000 | 0.0000 |
| **0.010** | **−0.043** | **0.0019** 🔴 | **NaN** 🔴 |
| **0.005** | **−0.065** | **0.0043** 🔴 | **NaN** 🔴 |
| **0.000** | **−0.087** | **0.0076** 🔴 | **NaN** 🔴 |

> 🔴 **This is a real defect in the instruction, not a nuance.** Removing the guard does not make the
> engine more honest — it makes it **wrong**:
>
> - **$n$ integer** → a **fully plugged** cell ($\phi_{acc}=0$) retains $k/k_0 = 0.0076$. It keeps
>   flowing. The percolation cutoff is *inverted*.
> - **$n$ fractional** (the usual empirical choice) → **complex/NaN** → poisons the **entire global
>   Jacobian**, not one cell. That is an **INV-1 `Unsolvable`**, and the researcher gets *nothing* —
>   the exact opposite of the INV-7 goal.
>
> **$k=0$ for $\phi_{acc}\le\phi_c$ is the percolation physics.** It is not a floor; it is the
> definition of throat closure. INV-7 forbids *artificial* bounds, and this one is derived.
>
> **Second, separate singularity — the denominator**, which the ruling did not identify:
> $\phi_0\to\phi_c$ gives $k/k_0 = 640{,}000$ at $\phi_0=0.0201$ and a divide-by-zero at $\phi_0=\phi_c$.

**Ruled implementation** — three branches, all physically derived:

```
phi_acc <= phi_c            -> k = 0                       # throat closed (percolation)
phi0 - phi_c < phi_cut_band -> normalise phi0 against phi_ref # removable singularity
phi0 <= phi_c               -> k = k0 * (phi_acc/phi0)^3    # no clogging possible; KC form
```

### 7c.3 🔴 $k\to\infty$ is a representation failure, not physics

The ruling states wormholing drives *"$\phi\to1.0$ and $k\to\infty$."* Measured, Kozeny-Carman
$\phi_0=0.25$:

| $\phi$ | $k/k_0$ |
|---|---|
| 0.60 | 48.6 |
| 0.80 | 460.8 |
| 0.90 | 2 624 |
| 0.99 | 3.49 × 10⁵ |
| 0.999 | 3.59 × 10⁷ |

> 🔴 **The divergence is mathematically correct and physically meaningless.** Past some $\phi_*$ the
> **continuum porous-media description stops being valid** — there is no pore network left to describe.
> The unconstrained engine must therefore **escalate the representation**, not take $k\to\infty$:
> a dissolved channel whose aperture exceeds a cell dimension **is a fracture**, and belongs in
> **EDFM** (Domain 7), not as a $k$ multiplier on a porous cell.
>
> ✅ **This is the correct unconstrained design**: no clamp, but a **representation switch** driven by a
> physical criterion (channel aperture vs. cell size). It also connects Domain 4 to Domain 7, which the
> design set never did.

### 7c.4 ⚠️ Barton-Bandis is dimensionally sound but singular in tension, and supplies $w_f$ only

The law $w_f = w_{f0}/\bigl(1+\sigma'_n/(K_{ni}w_{f0})\bigr)$ checks out dimensionally
($K_{ni}w_{f0}$ is a stress) and is correct at $\sigma'_n=0$. Measured, $w_{f0}=10\ \mu m$,
$K_{ni}=10^{12}$ Pa/m ⇒ $K_{ni}w_{f0}=10$ MPa:

| $\sigma'_n$ | $w_f$ |
|---|---|
| +10 MPa | 5.00 µm ✅ |
| 0 | 10.0 µm ✅ |
| −9.9 MPa | 1000 µm |
| **−10.0 MPa** | 🔴 **singular** |
| **−10.1 MPa** | 🔴 **−1000 µm — negative aperture** |

> 🔴 **Two gaps.**
> 1. The singularity at $\sigma'_n = -K_{ni}w_{f0}$ is **exactly the tensile-fracture condition**. The
>    unconstrained behaviour is **fracture initiation and the feature converts to an open conduit** —
>    not a negative aperture. Same representation-switch pattern as §7c.3.
> 2. **Barton-Bandis gives $w_f$, but DFN transmissibility is $T = k_f w_f$.** Nothing in the ruling
>    supplies $k_f(w_f)$. **$T_{ff}$ is therefore underspecified** — and it is the very quantity the
>    `Seal`↔`Conduit` transition is supposed to drive. A relation of the form
>    $k_f \propto w_f^{1/3}$ (or $w_f^2/12$ — see **CONF-47**, which is still open) is required.

### 7c.5 🔴 $\Delta CFS$ as written is absolute, and yields no slip magnitude

Submitted: $\Delta CFS = \tau - \mu(\sigma_n - P) - C_0$.

**As a Coulomb criterion this is correct.** Measured against a fault at rest since $t_0$ versus one
already sheared and healing — **identical current state** ($\tau=0$, $\sigma_n=20$ MPa, $P=10$ MPa):

| Case | $\tau$ | $\sigma_n-P$ | Submitted | Increment form |
|---|---|---|---|---|
| at rest since $t_0$ | 0 | 10 MPa | −10.5 MPa | requires $\sigma_{\tau,0}$ |
| was sheared, now healing | 0 | 10 MPa | −10.5 MPa | requires $\sigma_{\tau,0}$ |
| critical shear state | 8 MPa | 10 MPa | −2.5 MPa | requires $\sigma_{\tau,0}$ |

> 🔴 **The form carries no reference state.** Two faults in the *same current state* but different
> history are **indistinguishable**, so it cannot separate *"already at failure — slips now, no
> injection needed"* from *"injection pushed it over"* — which is precisely the research question
> INV-7 exists to answer. **Rename to `CoulombFS`** (absolute) and require the increment form
> $\Delta CFS=(\sigma_n-P)(\mu_1-\mu_0)-(\sigma_{\tau,0}-\tau)$ as the reported reactivation metric.
>
> 🔴 **Second and worse: Coulomb gives a *criterion*, not a *slip magnitude*.** $\Delta CFS>0$ says the
> fault **can** slip. It says nothing about how far or how fast — and $u_s = 0$ while $\Delta CFS>0$ is
> equally "correct" under the submitted text. So **$u_s(t)$ is underdetermined**, hence $T_{ff}$ is too,
> hence the `Seal`→`Conduit` switch has no defined magnitude.
>
> **Required: a slip law.** Rate-and-state (Dieterich healing $r=\dot u_s/A_0\sigma'_n\exp(-B u_s)$) for
> diagenetic seals; rate-and-state with state evolution for reactivating faults. Without it $u_s$
> diverges and $T_{ff}$ has no bound.

### 7c.6 ⚠️ Definitional bounds are not artificial bounds

INV-7 as originally worded ("no clamp, min, lock, clip or penalty") does not distinguish an **artificial
cap** from a **definition of the domain**.

| Bound | Class | Status |
|---|---|---|
| $\phi\in(0,1]$, $T>0$, $t\ge0$, $z\in[0,H]$, $p>0$ | **Definitional** | Enforced. $\phi\le0$ or $\phi>1$ is not a physical state; it is not a variable |
| $q\le1000$ BOPD, `locked_*`, `min_injection_rate` | 🔴 **Artificial** | **Removed** — INV-7 §7b.3 |
| $k>0$, $w_f>0$ | 🔴 **Derived** | Enforced where the physics derives it (§7c.2, §7c.4) — a plugged throat and a closed aperture have $k=0$ and $w=0$ |

> ✅ **INV-7 is thus scoped as: no bound on a physical quantity that the user chose, unless the bound is
> definitional or derived from the physics itself.** That is a precise and defensible reading, and it is
> what makes §7c.2's guard mandatory rather than contradictory.

### 7c.7 ⚠️ Hydrate melting is specified by its equilibrium law, not its kinetics

The ruling gives hydrate plug melting *"via the temperature-dependent van der Waals-Platteeuw phase
boundary."* van der Waals-Platteeuw supplies the **stability boundary** $F(P,T)=0$ — an equilibrium
**condition**, not a **rate**. It cannot compute how fast a plug melts.

**Required:** an intrinsic **dissociation kinetic law** (an Arrhenius rate constant per hydrate
structure, $r_{hyd}=A_h e^{-E_a/RT}\Delta$) **coupled to** the vdW-P equilibrium, plus the **latent heat**
in the energy equation. Note this makes hydrate melting a genuine **THMC** coupling, not a
thermodynamics-only one.

---

## 7d. ✅ Closures for C-63…C-68, and the gaps inside them (08-10-2026)

Reservoir engineer endorsed all five PM measurements and supplied closing equations. **Four of the five
closures are accepted as specified. Each one still has a defect or an unhandled limit.**

### 7d.1 C-63 — piecewise Verma-Pruess: ✅ accepted, 🔴 guard closes only HALF the singularity

Submitted guard:

```rust
pub fn calculate_verma_pruess_k(phi_acc: f64, phi_0: f64, phi_c: f64, k_0: f64, n: f64) -> f64 {
    if phi_acc <= phi_c { 0.0 }
    else { k_0 * ((phi_acc - phi_c) / (phi_0 - phi_c)).powf(n) }
}
```

> ✅ **The numerator is now closed** — the `-0.087^1.5 -> NaN` Jacobian poisoning is gone.
>
> 🔴 **The denominator is still open, and Rust does not raise.** In Rust `x / 0.0_f64` returns `inf`,
> **not** a panic and not an error. Measured:

| $\phi_0$ | $\phi_c$ | base | $k/k_0$ | Physical truth |
|---|---|---|---|---|
| 0.25 | 0.02 | 0.348 | 0.205 | ✅ |
| 0.021 | 0.02 | 80 | 715 | ⚠️ already 700× hot |
| **0.0201** | 0.02 | 800 | **2.26 × 10⁴** | 🔴 |
| **0.02** | 0.02 | ∞ | **∞** | 🔴 cell was already below throat closure — **$k$ must be 0** |
| **0.01** | 0.02 | −8 | **NaN** | 🔴 returns to the very failure this guard exists to prevent |

> 🔴 **A cell whose initial porosity $\phi_0\le\phi_c$ returns infinite or NaN permeability, silently.**
> Physically $\phi_0\le\phi_c$ means the cell was **already below throat closure** — the correct answer
> is `k = 0`, which requires **no clogging model at all**.
>
> **Ruled three-branch implementation (super'sedes §7c.2):**
>
> ```
> phi_0 <= phi_c            -> k = 0                     # already below closure; no clogging possible
> phi_acc <= phi_c          -> k = 0                     # newly blocked  (C-63, accepted)
> phi_c < phi_acc < phi_0   -> k0 * ((phi_acc-phi_c)/(phi_0-phi_c))^n     # the only branch with arithmetic
> phi_acc >= phi_*          -> REPRESENTATION SWITCH -> EDFM conduit      (C-64)
> ```
>
> ⚠️ Note the ordering: **the $\phi_0$ test must precede the ratio**, and both must precede `powf`.
> A single `if` cannot express this.

### 7d.2 C-64 — representation switch: ✅ accepted, $\phi_*\approx0.80$ recorded

Accepted as specified: at $\phi_{acc}\ge\phi_*$ the cell leaves Domain 4's porous FVM and enters Domain 7's
EDFM with $k_f = w_{\text{wormhole}}^2/12$. ✅ **This is the first real Domain 4 → Domain 7 link the
design set never provided**, and it is the correct unconstrained resolution of C-64: not a clamp, a
change of representation.

⚠️ **$\phi_*$ must not be a magic constant.** It is a **continuum-validity threshold**, and its correct
value is set by the cell geometry — the dissolved channel ceases to be describable as a porous medium
when its aperture is no longer small relative to the cell dimension. Record it as
$\phi_* = \phi_*(\text{cell geometry})$ with the geometric criterion stated, and as an
**input with a measured default**, so a user can probe sensitivity to it.

### 7d.3 C-65 — $k_f(w_f)$: ✅ accepted, and this **RESOLVES CONF-47**

$$\boxed{\;k_f(w_f) = \frac{w_f^{2}}{12\left[1 + 8.8\left(\dfrac{JRC}{w_f}\right)^{1.5}\right]}\;}$$

**Accepted.** Verified on both limits:

| Limit | Behaviour | Verdict |
|---|---|---|
| $w_f\to\infty$ | $k_f\to w_f^2/12$ | ✅ cubic law recovered |
| $w_f\to 0$ | $(JRC/w_f)^{1.5}\to\infty$, so $k_f\to0$ | ✅ aperture closes, flow stops |
| $JRC=0$ | $k_f = w_f^2/12$ exactly | ✅ smooth fracture limit |
| Dimensional | $JRC$ and $w_f$ both length ⇒ ratio dimensionless | ✅ |

> ✅ **CONF-47 can now be closed.** The design set wrote the **transmissibility** as $T = w^2/12$; the
> correct cubic law is $T = w^3/12$. The defect was a **convention collision** — one symbol $k$ used for
> both *permeability* and *transmissibility*. Writing $k_f = w^2/12$ and $T_{ff} = k_f w_f$ **separates
> them correctly** and $T$ becomes cubic automatically. **Close CONF-47 on this basis.**
>
> ⚠️ **Provenance requirement.** The `8.8` coefficient and the `$1.5$` roughness exponent need a
> **cited source** before M7c. Barton-Bandis roughness correlations are numerous and mutually
> inconsistent; an uncited coefficient in a constitutive law is a `PROVENANCE` finding under
> [`register_spec.md`](register_spec.md), not a style note.

### 7d.4 C-66 — rate-and-state: ✅ law accepted, 🔴 the state evolution is the wrong limit

Frictional strength law **accepted exactly as submitted** — it is the standard Dieterich-Ruina form:

$$\tau = \sigma'_n\left[\mu_0 + a\ln\frac{V}{V_0} + b\ln\frac{V_0\theta}{D_c}\right],\qquad V=\frac{du_s}{dt}$$

🔴 **But the state evolution `dtheta/dt = 1 - V*theta/Dc` is the _aging approximation_, not the general
law.** Ruina (1983) gives

$$d\theta/dt = \left[\frac{V_0(\theta-\theta_g)}{D_c}+1\right],\qquad \theta_g = \frac{b}{a}\frac{D_c}{V_0}$$

The submitted form is its limit for $V\theta/D_c\ll1$. Measured ($V_0=10^{-6}$ m/s, $a=0.008$,
$b=0.010$, $D_c=2\times10^{-7}$ m ⇒ $\theta_g = 0.25$ m):

| $V/V_0$ | $\theta_{ss}$ **full** | $\theta_{ss}$ **aging** | error | $V\theta/D_c$ |
|---|---|---|---|---|
| $10^{-3}$…$1$ | 200 → 0.20 m | 200 → 0.20 m | **1×** ✅ | $\le1$ |
| 3 | 0.25 m | 0.0667 m | 0.27× | 3.75 |
| 10 | 0.25 m | 0.020 m | **0.08×** | 12.5 |
| 100 | 0.25 m | 0.002 m | **0.008× (80× under)** | 125 |

> 🔴 **The aging law has no $V$-independent steady state.** Its steady state is $\theta_{ss}=D_c/V$, so
> $V\theta/D_c = 1$ *by construction* — it drives the system to exactly the point where its own validity
> condition ($\ll1$) fails. It is valid only while $V\ll V_0$.
>
> 🔴 **And INV-7 guarantees $V\gg V_0$.** $V = du_s/dt$ is driven by an **unconstrained** pressure ramp;
> absurd injection rates are the design target. At $V/V_0 = 100$ the submitted law under-predicts the
> state variable **80-fold**, which propagates directly into $\tau$ through $b\ln(V_0\theta/D_c)$ and
> therefore into $u_s$ and $T_{ff}$. **The closure is invalid in precisely the regime it was written for.**
>
> ✅ **Ruled: use the full Ruina form.** It reduces to the aging law automatically where valid and stays
> bounded where it is not.
>
> 🔴 **Second gap: the law is implicit.** $\tau$ appears on the left and $V$ on the right, with $V$ also
> multiplying the stress. Solved at the global Newton level this is a **nested Newton / Schur
> complement** — a **C-35**-class coupling cost that must be budgeted, not discovered.
>
> ⚠️ **Third: $\Delta CFS$ and rate-and-state must not be coupled.** Rate-and-state is a **constitutive
> law that replaces** the Coulomb criterion — it *determines* $\tau$. Coulomb then becomes a **reported
> diagnostic** (is the fault above or below steady-state strength?), not a driver. Coupling both
> double-counts the friction.

### 7d.5 🔴 C-64b — `k_fault(u_s) = k_0 + gamma*u_s` reintroduces the rejected defect

The submission closes the matrix's $k\to\infty$ and then supplies, for the fault:

$$k_{\text{fault}}(u_s) = k_0 + \gamma u_s$$

Measured, $\gamma = 100$ m⁻¹:

| $u_s$ | $k_{\text{fault}}/k_0$ |
|---|---|
| 0.01 m | 2 |
| 0.10 m | 11 |
| 1.00 m | 101 |
| 10.0 m | 1001 |

> 🔴 **This is the same unbounded, monotonically-increasing $k$ that C-64 rejected for the matrix**,
> now written into the fault law. It is internally inconsistent with the closure it accompanies.
>
> 🔴 **It also omits gouge, which is the actual mechanism for a _seal_ — the case this domain is about.**
> A slipping clay-rich fault **compacts its gouge first, so $k$ _decreases_ (the seal improves)**; only
> when the gouge ruptures does $k$ rise. The real history is **two-phase**:
>
> ```
> u_s < u_gouge :  k DECREASES   (gouge compaction,  seal improves)
> u_s > u_rupt  :  k INCREASES   (gouge rupture,     conduit)
> ```
>
> A linear law cannot represent it, and it inverts the sign of the risk in the first phase — it reports
> a seal **improving** as an **increasing** $k$.
>
> **Required:** a saturating post-rupture law with a bounded asymptote, plus an explicit pre-rupture
> gouge-compaction branch. A defensible form is saturating-in-log,
> $k = k_{\min} + (k_{\max}-k_{\min})\,\mathrm{tanh}\!\big((u_s-u_{\text{rupt}})/u_{\text{scale}}\big)$,
> with $k_{\min}<k_0$ and $k_{\max}$ finite.

### 7d.6 ✅ Elastoplasticity — the consistent tangent is CORRECT as submitted

The submitted algorithmic tangent

$$D^{ep} = D^e - \frac{D^e\frac{\partial Q}{\partial \sigma}\left(\frac{\partial F}{\partial \sigma}\right)^{\!\top}\!D^e}{H + \left(\frac{\partial F}{\partial \sigma}\right)^{\!\top}\!D^e\frac{\partial Q}{\partial \sigma}}$$

**Accepted — verified correct.** The chain $D^e\cdot\frac{\partial Q}{\partial\sigma}\cdot
(\frac{\partial F}{\partial\sigma})^{\top}\cdot D^e$ evaluates as
$(D^e b_\theta)\,(a^{\top}D^e)$ — the standard non-associated consistent tangent, shape $6\times6$ over
a scalar denominator. ✅ **No correction needed** — this closes the **C-62 gap 2** concern.

> ⚠️ **Two qualifications on the surrounding claim.**
> 1. *"Using a symmetric elastic approximation destroys quadratic convergence"* is **conditional**. While
>    the active plastic set is **unchanged**, $D^{ep} = D^e$ is exact and symmetric. The defect bites
>    exactly when the active set changes — i.e. at yield, during unloading, and on switch points.
> 2. 🔴 **This couples a milestone-M0 decision to a milestone-M7b choice.** [`build_plan.md`](build_plan.md)
>    M0.1 records the linear-algebra decision as still open (**CONF-14**: *no preconditioner specified*).
>    **A Cholesky/incomplete-Cholesky preconditioner is invalid** for a non-symmetric $D^{ep}$, so the
>    preconditioner cannot be chosen until the constitutive model is fixed. **M0.1 must record this
>    dependency explicitly**, or the solver choice will be made under an assumption that M7b invalidates.

### 7d.7 ✅ Hydrate kinetics — accepted as submitted

$$d n_H/dt = K_0 e^{-\Delta E/RT}\,A_s\left(f_{\text{gas}} - f_{\text{eq}}(P,T)\right),\qquad q_{\text{thermal}} = \Delta H_{\text{hydrate}}\frac{dn_H}{dt}$$

**Accepted.** Standard Bishnoi-Englezos/Bravo-Harlock form: Arrhenius rate coefficient, specific surface
area, and driving force as the **fugacity deficit** — with vdW-P supplying $f_{\text{eq}}$. ✅ Closes
**C-68**.

> ⚠️ **One modelling choice needs a default.** $A_s$ is the hydrate **specific surface area**, and Gibbs'
> theorem requires it to **shrink as the hydrate decomposes**. A constant $A_s$ over-predicts dissociation
> once decomposition is well advanced. Record $A_s$ as an input with a documented constant default and a
> note that it is a first approximation.
>
> ✅ The latent-heat coupling is included, which is what makes this genuinely **THMC** rather than
> thermodynamics-only.

---

## 7e. Closures for CONF-13/14/25/31/51, the data split, and B-3 (08-10-2026)

Corrections **C-76 … C-84** in [`spec_corrections_log.md`](spec_corrections_log.md). **Seven accepted,
six defects found inside them.** What follows is the technical substance.

### 7e.1 Training split — `GroupKFold` ✅ accepted, 🔴 blind forecasting still leaks

**Accepted:** group by **Well-Pattern Topology ID + Geological Realization Seed + Boundary Schedule ID**,
so a whole configuration or realization is entirely train or entirely test.

🔴 **But the failure it claims to fix is temporal, and `GroupKFold` has no time ordering.** The stated
symptom — *"artificially inflated test scores that collapse during real blind forecasting"* — is a
surrogate reconstructing a **time-correlated trajectory** it has already seen in train. Grouping by
configuration removes *configuration* leakage but leaves *temporal* leakage untouched: a test sample
from the same time window as train is still interpolation, not forecasting.

**Ruled: a two-dimensional split.**

| Axis | Group by | Removes |
|---|---|---|
| **Configuration** | Well-pattern topology ID, geology seed, boundary schedule ID | Leakage from the surrogate memorising one well layout |
| **Time** | Final portion of the simulation **within every group** | Leakage from trajectory interpolation |

⚠️ **Needs ≥2 groups.** With one well configuration, `GroupKFold` returns nothing — record that
precondition or the split silently degenerates to random.

### 7e.2 B-3 — ✅ CLOSED, dual-engine frontend

DuckDB-WASM for OLAP over Parquet/Arrow; libSQL/SQLite-WASM for UI state and well metadata. The
diagnosis is right: forcing 4D time-series through a row store causes browser thread locks, and
DuckDB for UI state is wasted memory. ⚠️ **Synchronisation between the two stores is undefined** — and
libSQL needs OPFS or export/import for persistence. Deferred with the rest of the UI.

### 7e.3 CONF-13 — ✅ CLOSED, the diagnosis is exact

Three concrete defects in the D5 listing, all confirmed:

| Defect | Detail |
|---|---|
| `lambda` computed and **never applied** | The damping factor is produced and then discarded — $x^{k+1}=x^k+\Delta x$ runs undamped |
| `compute_residuals_and_jacobian` computes **no Jacobian** | The name promises both; it delivers neither |
| **No linear solver call** | The iteration cannot converge because nothing is solved |

**Ruled loop:** Armijo-Goldstein line search
$\|\mathbf{R}(\mathbf{x}^k+\lambda\Delta\mathbf{x})\|_2 \le (1-\alpha\lambda)\|\mathbf{R}(\mathbf{x}^k)\|_2$
with $\lambda\in(0,1]$, plus an in-core sparse solve $\mathbf{J}^k\Delta\mathbf{x}=-\mathbf{R}^k$.

### 7e.4 🔴 CONF-14 — the CPR answer conflicts with CONF-66

CPR (**Constrained Pressure Residual**) splits $\mathbf{J}$ into an elliptic pressure block solved by
**AMG** and a hyperbolic transport block solved by **ILU(0)/Block-Jacobi**, driven by **FGMRES**.

**All three stages need scrutiny in compositional flow:**

**(a) The pressure block is not independent.** CPR assumes $\mathbf{A}_p$ depends only on $P$. But
$\rho_{tot}=\sum_i z_i\rho_i(P,T)$, and measured — same $P$, two flash splits:

| Flash state | $z$ / $y$ | $\rho_{tot}$ |
|---|---|---|
| liquid-heavy | $[0.40, 0.35, 0.25]$ | **439.5 kg/m³** |
| vapour-heavy | $[0.55, 0.35, 0.10]$ | **546.0 kg/m³** |

**+24 % change.** So $\partial\rho_{tot}/\partial y_i \neq 0$: the fluid-density block **couples to
transport**, and because flash is re-solved at every Newton iterate, $\mathbf{A}_p$ **changes every
iteration**. The $\alpha/\beta$ split is therefore not clean.

**(b) 🔴 "Symbolic pattern analysis once per grid topology" is wrong here.** Flash re-evaluation can
switch a $K$-value coupling **on or off** between components as the two-phase region is crossed. That
**changes the sparsity pattern**, not just the values — so symbolic analysis must be re-run, or the
factorisation is structurally invalid.

**(c) 🔴 AMG assumes near-symmetry / M-matrix structure.** Non-associated elastoplasticity produces a
**non-symmetric** $D^{ep}$ (**C-74**). A symmetric-hierarchy AMG applied to it degrades convergence —
and silently, which is the **C-35** failure mode from the Python attempt.

> ✅ **FGMRES is the right driver and stands** — it is specifically the variant that tolerates a
> preconditioner changing between iterations, which is exactly what adaptive AMG hierarchy depth gives.
>
> 🔴 **Ruled: CONF-14 stays open.** CPR needs a compressible-compositional adaptation (block-Jacobi or
> ILU on the pressure block, or an AMG hierarchy built on the **symmetric** elastic part $\mathbf{D}^e$
> and used as a left preconditioner), and M0.1 must benchmark on a **non-symmetric** test matrix.

### 7e.5 ⚠️ CONF-25 — enums ✅, TPD sketch 🔴

**Enum taxonomy accepted** — `CriticalHessianSingular`, `MichelsenTrivialRoot`, `FugacityNan` is a good
set, and it maps cleanly onto **C-33**'s Heidemann-Khalil requirement.

🔴 **The TPD restatement repeats CONF-13's own defect.** Submitted:

$$\mathrm{TPD}(y)=\sum_i y_i\left(\ln y_i + \ln\phi_i(y) - d_i\right)$$

Three defects: it **omits the leading $-1$** (the sum-to-one normalisation that makes it a valid
splitting objective); $\phi_i$ is **ambiguous between liquid and vapour** when the whole point is the
ratio $\phi_i^L/\phi_i^V$; and it supplies **no gradient**, which a Newton solve on TPD requires.

> ⚠️ **Also:** the declared EoS accuracy ceiling of **8–9 %** (**CONF-49**) sits *inside* the
> design-signal band. Solver tolerance and EOS tolerance must be budgeted separately, or a converged
> solve can still miss the flash state.

### 7e.6 ⚠️ CONF-31 — two of four actually closed

| Correlation | Verdict |
|---|---|
| **Standing** $P_b$ | ✅ **correct as submitted** |
| **Beggs-Robinson** $\mu_{od}$ | ✅ **correct** — ⚠️ $T$ in **°F**, valid 70–295 °F; record the unit |
| **Karakas-Tarik** $S_p$ | 🔴 $S_p=S_h+S_v+S_{wb}$ names the split, supplies **none of the polynomials**. **Identical defect to CONF-31** |
| **Peneloux** | 🔴 **dimensionally wrong** — see below |

> 🔴 **Peneloux applies to density, not volume.** The law is
> $\rho_{corr}=\rho_{EOS}/(1-N_\omega)$ with $N_\omega = c_0/\left(\sum_i y_iM_i\omega_i\right)$.
> The submission writes $v_{corr}=v_{EOS}-c$ — but $c$ is **dimensionless**, so subtracting it from
> $v$ [m³/kg] is invalid. Two errors: **wrong variable** (subtract vs. divide) and **wrong parameter**
> (a single $(M\omega)^{-1}$ instead of the **mixture-weighted** $\sum_i y_iM_i\omega_i$).

### 7e.7 ⚠️ CONF-51 — geometry triple correct, two cases missing

PKN / KGD / radial all verified correct, including coefficients and profiles:

| Model | $w_0$ | Profile | Regime |
|---|---|---|---|
| **PKN** | $3.04[\mu(1-\nu^2)q_0L_f/E]^{1/4}$ ✓ | $w_0(1-x/L_f)^{1/4}$ ✓ | $L_f\gg H_f$ |
| **KGD** | $2.36[\mu(1-\nu^2)q_0L_f^2/(EH_f)]^{1/6}$ ✓ | $w_0(1-x^2/L_f^2)^{1/2}$ ✓ | $H_f\gg L_f$ |
| **Penny** | $2.56[\mu(1-\nu^2)q_0R_f/E]^{1/4}$ ✓ | — | unconfined radial |

Carter leakoff $v_L=C_w/\sqrt{t-\tau}$ ✓.

🔴 **Gap 1 — no Type-II.** All three are **Type-I, viscosity/leakoff-dominated**. **Toughness-dominated
fracture propagation ($K_{IC}$) has no model** — $K_{IC}$ appears in the summary table but in **no
equation**. A narrow, low-permeability fracture is Type-II, so this is a real regime gap.

🔴 **Gap 2 — no proppant.** These are **proppant-free elastic-opening** models. Domain 7 specifies
**dynamic proppant transport** — $C_{prop}$, $h_{pack}$, embedment depth. Proppant-supported width needs
a different closure; as given, proppant has no effect on $w_0$.

### 7e.8 ⚠️ CONF-16b — structurally sound, three symbols undeclared

**$S_a$ kinetics ✅ accepted structurally** — the two-term deposition/re-entrainment ODE
$\alpha(C_a-C_a^*)^m u - \beta S_a u$ is dimensionally sound and correctly captures the two mechanisms.

🔴 But **$C_a^*$ and $m$ are undeclared** — CONF-31's pattern recurring in the document meant to close it.
⚠️ And there is **no static flocculation term**: asphaltene instability is not purely flow-driven, so a
quiescent cell never destabilises.

**Filter cake** — 🔴 the deposition term closes, the erosion term does not:

| Term | Dimensional check |
|---|---|
| $v_fC_{TSS}/\rho_{cake}$ | ✅ m/s — v [m/s] × c [kg/m³] / ρ [kg/m³] |
| $\tau_{shear}\eta_{erosion}h_{cake}$ | 🔴 $\tau$[Pa]·$h$[m] = kg/s², so **$\eta_{erosion}$'s units are undeclared** and the term needs **viscosity** $\mu$ and a rate constant $E$ |

The standard filtration erosion form is $e = E\tau_{shear}/\bigl(\mu(1+\alpha c)\bigr)$ — both $\mu$ and
$E$ are missing from the submitted closure.

### 7e.9 ✅ CONF-18b / C-36 — complete

Selection rule now enforced: **Verma-Pruess** for chemical precipitation and particle clogging
(TSS, scale, asphaltene, hydrate); **Kozeny-Carman** for mechanical compaction and poroelastic
deformation. Closes the D1 §5.2 gap.

⚠️ **Carries forward C-69:** the guard as printed omits the **denominator** test $\phi_0\le\phi_c$,
which returns **∞/NaN** in Rust — see §7d.1.

---

## 7f. CONF-14 overhaul, 2D split, Penéloux + Karakas-Tarik (08-10-2026)

Corrections **C-85 … C-90**. Three accepted, **two of my own claims withdrawn**, three defects found.

### 7f.1 🔴 C-85 — I withdraw my own dynamic-sparsity claim

I asserted (C-79b) that flash toggles $K_i$ couplings, so the sparsity pattern changes and "symbolic
analysis once per topology" is wrong. **The engineer asserts the same thing. Both of us skipped a step:
the sparsity pattern is a property of the *formulation*, not of the fluid.**

The **approved output schema already fixes the formulation** ([`output_schema.md`](output_schema.md)
§4, Domains 1–2): the primary variables are **overall composition $z_i$**, $P$ and $S_\alpha$.

| Formulation | Primary variables | Sparsity pattern |
|---|---|---|
| **Overall composition / total FVF** ← *the approved schema* | $z_i,\ P,\ S_\alpha$ | **Fixed** by grid stencil + equation set. $z_i$ exists everywhere, so $\partial\rho_{tot}/\partial z_i$ is structurally non-zero even where phase amounts are zero |
| Phase-component with explicit $K$-value coupling | $x_i,\ y_i,\ P,\ S_\alpha$ | **Dynamic** — phase appearance adds/removes entries |

✅ **Under the approved schema, "symbolic pattern analysis once per grid topology" is CORRECT**, and
dynamic CSR tracking would be needless machinery. ⚠️ The engineer's claim is correct for the *other*
formulation.

> 🔴 **My error, and it is the CONF-01 error again**: I asserted a numerical consequence without first
> checking which formulation the project had actually chosen. Withdrawn.
>
> **Ruled:** record the **primary-variable formulation as the governing decision** at M2, and derive
> every sparsity/structure claim from it. Anyone changing the formulation must re-derive the CSR,
> the preconditioner, and the phase-declaration logic together.

### 7f.2 ✅ C-86 — volume-balance reduction and unsymmetric solvers

**Accepted:** derive the pressure equation by Watts' volume-balance method, carrying
$\partial\rho/\partial z_i$ explicitly into the row reduction. That is the correct response to the
measured **+24 %** $\rho_{tot}$ shift between flash states (§7e.4a).

**Accepted:** unsymmetric AMG, or FGMRES preconditioned with BiCGStab / ILU(1), replacing SPD-AMG —
because non-associated $D^{ep}$ is non-symmetric (**C-74**).

> ⚠️ **Scalability caveat, not a blocker.** ILU(1) on the pressure block is materially weaker than an
> AMG hierarchy, and the D5 Level-5 target is **SPE-10 at 1.1 × 10⁶ cells**. M0.1 must benchmark **both**
> and record the **crossover point**, rather than adopting the cheaper option by default.

### 7f.3 ✅ C-87 — `GroupTimeSeriesSplit` closed, with one addition

**Accepted in full:** group by topology ID + geological seed with **≥2 groups mandated**, **plus** a
chronological holdout inside every group at $t_{cut}=0.75\,T_{max}$.

| Split | Range |
|---|---|
| Train | $t\in[0,\ t_{cut}]$ — first 75 % |
| Blind test | $t\in(t_{cut},\ T_{max}]$ — final 25 % forecast horizon |

> 🔴 **Addition: the blind test must not be used for early stopping, hyperparameter tuning, or model
> selection.** Any of those makes it a **validation** set, and the temporal leak **C-76** was raised to
> close re-opens through the back door. **A separate inner validation split, drawn from the training
> window only, is mandatory.**

### 7f.4 🔴 C-88 — Penéloux: I withdraw my objection, but $c_i$ is sign-inverted

**Withdrawn:** I claimed the molar-volume form applies the law to the "wrong variable." It does not.
Verified numerically — with $N_\omega = c_{mix}/v_{EOS}$:

| Form | Result at $\rho_{EOS}=600$, $c=0.05$ |
|---|---|
| $\rho_{corr} = M/(v_{EOS}-c)$ | −20.690 kg/m³ |
| $\rho_{corr} = \rho_{EOS}/(1-N_\omega)$ | −20.690 kg/m³ |

**Identical.** They are the same law; $N_\omega = c_{mix}/v_{EOS}$. ✅ $c_{mix}=\sum_i x_i c_i$ correctly
weights by **liquid** mole fraction.

🔴 **But the component shift has the wrong sign.** Measured:

| Fluid | $T_c$ K | $P_c$ MPa | $Z_c$ | bracket | $c_i$ m³/kmol |
|---|---|---|---|---|---|
| methane | 190.6 | 4.60 | 0.274 | −0.0055 | **−0.00078** 🔴 |
| toluene | 591.8 | 4.11 | 0.264 | −0.0011 | **−0.00055** 🔴 |
| n-heptane | 418.1 | 2.74 | 0.262 | −0.0002 | **−0.00013** 🔴 |
| n-decane | 588.7 | 2.49 | 0.321 | −0.0263 | **−0.02107** 🔴 |
| CO₂ | 304.1 | 7.38 | 0.274 | −0.0055 | **−0.00077** 🔴 |
| ethane | 305.4 | 4.88 | 0.099 | +0.0717 | +0.01521 ✅ |
| propane | 369.8 | 4.25 | 0.152 | +0.0483 | +0.01425 ✅ |

> 🔴 **The bracket $0.1154-0.4414Z_c$ is negative for $Z_c > 0.261$** — which covers methane, toluene,
> n-heptane, n-decane and CO₂: **most of the fluids anyone cares about.** Cubic EOS *under*-predicts
> liquid molar volume, so the translation must be a **positive** shift subtracted from $v$. A negative
> $c_i$ **increases** $v$ and inverts the correction.
>
> 🔴 **PROVENANCE required** for `0.40768`, `0.1154`, `0.4414` — the magnitude cannot be checked
> without the source (**C-90**).

### 7f.5 ⚠️ C-89 — Karakas-Tarik improved, but $S_h$ is self-inconsistent

**Real coefficients at last** — a genuine step forward, and the first correlation in this whole thread
to arrive with numbers rather than names:

| Quantity | Form | 90° phasing values |
|---|---|---|
| Vertical skin | $S_v = 10^a h_D^{b-1} r_D^{b}$ | $a_1{=}-2.025$, $a_2{=}0.0943$, $b_1{=}3.0373$, $b_2{=}1.8115$ |
| Wellbore blockage | $S_{wb} = c_1 e^{c_2 r_D}$ | $c_1{=}0.0066$, $c_2{=}5.32$ |
| Horizontal skin | $S_h = \ln(r_w/r'_w)$ | $r'_w = r_p/4$ (180°); $\alpha_0(r_w+r_p)$ (90°, 120°) |
| Crushed zone | $S_{cz} = \frac{h}{L_p}\left(\frac{k}{k_{cz}}-1\right)\ln\frac{r_{cz}}{r_p}$ | — |

🔴 **$S_h$ cannot be right as given.** With $r_p = 5$ mm, $r_w = 108$ mm, 180° phasing:
$r'_w = r_p/4 = 1.25$ mm, so $S_h = \ln(108/1.25) = \mathbf{+4.46}$ — a **large positive, damaging**
skin for **180° phasing, the best phasing angle**. An *equivalent* wellbore radius of 1.25 mm inside a
108 mm wellbore is not physical: perforations can only ever **enlarge** the effective radius, never
shrink it below the drilled hole.

> 🔴 **Either the $S_h$ definition or the $r'_w$ expressions are wrong. They cannot both be as
> submitted.** This is an *internal* inconsistency — provable from the submitted equations alone, with
> no reference to the source needed.

⚠️ **$S_v$ non-monotonicity:** over a realistic range ($h_p$ 0.1–1 m, $r_p$ 2–20 mm) $S_v$ **falls** as
$r_p$ rises — *more* perforation giving *less* skin. Verify against the source correlation before use.

🔴 **Four symbols still undeclared — CONF-31's pattern, third occurrence:**
$\alpha_0$ ("given by phasing polynomials", not supplied) · $h$ ($h_p$? pay thickness?) · $k$ ($k_h$?
$k_v$? formation?) · $r_{cz}$ (crushed-zone radius).

### 7f.6 🔴 C-90 — PROVENANCE: three unverified citations

| Citation | Use | Status |
|---|---|---|
| **Watts (1986)** — volume-balance method | pressure reduction, **CONF-14** | ⚠️ **UNVERIFIED** |
| **Wong et al. (2002)** | as above | ⚠️ **UNVERIFIED** |
| **Karakas & Tarik (1990)** | perforation skin model | ⚠️ **UNVERIFIED** |

> ⚠️ **Web search was unavailable in this environment, so none of these could be checked.** They are
> recorded as **UNVERIFIED — not as wrong**. Per the **M1 precedent** (Abudour et al. 2014,
> `10.1016/j.fluid.2014.10.006`; Baled et al. 2012, `10.1016/j.fluid.2011.12.027` — both confirmed by
> DOI fetch), **all three must be resolved by DOI before M3 and M7c**, or they become `PROVENANCE`
> findings under [`register_spec.md`](register_spec.md).

---

## 7g. Karakas-Tarik corrected, 3-way split, M0.1 gate, DOI resolution (08-10-2026)

Corrections **C-91 … C-96**. Two accepted, **two of my own claims withdrawn**, two defects — one an
**INV-7 regression**, one a **PROVENANCE failure**.

### 7g.1 ⚠️ C-91 — $S_h$ fixed in direction, branches now inconsistent

The diagnosis was right: the perforation **tunnel radius** $r_p$ (5–10 mm) had been substituted where
the **penetration length** $L_p$ (200–500 mm) belongs. The 180° result is now correct:

$$r'_w(180^\circ) = 0.500\,(0.108 + 0.300) = 0.204\ \text{m} \;\Rightarrow\; S_h = \ln\frac{0.108}{0.204} = \mathbf{-0.636}$$

A **negative, stimulating** horizontal skin — right for a perforated horizontal well that bypasses
wellbore damage. ✅

🔴 **But the two branches now disagree at 0° phasing:**

| Branch | $r'_w$ at 0° |
|---|---|
| Branch 1: $L_p/4$ | **0.0750 m** |
| Branch 2 via tabulated $\alpha_0(0)=0.250$: $\alpha_0(r_w+L_p)$ | **0.1020 m** |

**Same phasing angle, 26 % apart.** And branch 1 is the only expression that **never references $r_w$**,
which is why it behaves differently.

🔴 **0° is the only angle returning a *positive*, damaging skin:**

| $\theta$ | $r'_w$ | $S_h$ |
|---|---|---|
| **0°** | 0.0750 m | **+0.365** 🔴 |
| 90° | 0.2962 m | −1.009 |
| 120° | 0.2521 m | −0.848 |
| 180° | 0.2040 m | −0.636 |

> ⚠️ Whether the *ordering* is right depends on KT's sign convention and phasing ranking — which is
> precisely what **C-96** left unverified. **The 26 % branch discontinuity is definite**; the ordering
> needs the source. Until it is adjudicated, the 0° case is **declared-absent** rather than guessed.

### 7g.2 ✅ C-92 — I withdraw my $S_v$ non-monotonicity claim

**Retracted.** Measured, $S_v$ is **monotone decreasing** over $r_D\in[0.002,\ 0.5]$:

| $r_D$ | $a_1+b$ | $r_D^{a_1+b}$ | $h_D^{b-1}$ | $S_v$ |
|---|---|---|---|---|
| 0.0020 | −0.2074 | 3.6294 | 3.526e−03 | 1.591e−02 |
| 0.0200 | −0.1528 | 1.8177 | 2.417e−03 | 5.461e−03 |
| 0.0703 | 0.0000 | 0.9999 | 8.413e−04 | 1.046e−03 |
| 0.2000 | +0.3940 | 0.5304 | 5.535e−05 | 3.649e−05 |

The engineer's algebra is **correct**: $10^a r_D^b = 10^{a_2}r_D^{a_1+b}$ ✓, and the exponent crosses
zero at $r_D = 0.0703$ ✓. But **both factors fall together** — $r_D^{a_1+b}$ falls, *and* $h_D^{b-1}$
falls because $b$ grows. The turning point in the exponent never makes $S_v$ non-monotone when
$h_D < 1$.

> ✅ **And larger $r_p \Rightarrow$ lower $S_v$ is physically correct** — bigger perforation tunnels
> connect better to the formation. My C-89(b) was wrong on both counts. **Retracted.**

### 7g.3 🔴 C-93 — the clamp guards are an INV-7 regression

Proposed: $r_D = \mathrm{clamp}(r_D, 0.001, 0.050)$ and $h_D = \mathrm{clamp}(h_D, 0.010, 100.0)$.

🔴 **This violates INV-7 and C-67.** An **empirical regression envelope is not a physical or definitional
bound.** It is the domain over which a *correlation* happens to be valid — a statement about the
correlation, not about the reservoir.

The failure is concrete: two wells at $r_D = 0.06$ and $r_D = 0.20$ return the **same skin**. The
physics is discarded and nothing reports it.

> ✅ **Ruled: declare, do not clamp.** Compute the formula; where $r_D$ or $h_D$ leaves the KT
> regression domain, set `ValidityClass = ConvergedOutsideEnvelope` and emit a `ValidityWarning` naming
> the violated domain. **The mechanism already exists** — C-59's three-tier classification,
> INV-7 §7b.2.
>
> ⚠️ **This is the same move as CONF-07** (the 1000 BOPD clamp), with a better justification. A better
> justification does not change what clamping does: it substitutes an arbitrary number for the answer.
> INV-7 forbids the substitution, not the recording.
>
> ⚠️ **And it is not a case for deleting the bound as information.** The regression domain is real and
> worth recording — it belongs in the manifest as a warning, not in the arithmetic as a `min`/`max`.

> 📌 **This is the strongest available argument for writing invariants down.** The clamp instinct
> survived being *named*, from the party that named it. That is exactly why INV-7 needs a CI gate
> (§7b.6) rather than good intentions.

### 7g.4 ✅ C-94 — 3-way split accepted, with a 2 × 2 addition

Train / inner-validation / blind-test, with the inner split drawn from the training window only. ✅
Accepted — this closes the leak **C-87** was raised for.

⚠️ **But it folds two generalisation questions into one test set:** *unseen configuration* and *future
time*. A failure cannot be attributed to either.

**Ruled — a 2 × 2 diagnostic:**

| | Interpolated time | Extrapolated time |
|---|---|---|
| **Seen configuration** | sanity check | **temporal generalisation** |
| **Unseen configuration** | **spatial generalisation** | **both — the real deployment case** |

The bottom-right cell is what production forecasting actually faces, and the off-diagonal cells are what
say *which* failure mode dominates. Without them a single number is uninterpretable.

### 7g.5 ✅ C-95 — M0.1 benchmark gate accepted

**ILU(1)-FGMRES** vs **unsymmetric CPR-AMG** on SPE-10 (1.1 M cells), wall-clock and memory, with the
**crossover $N_{cells}$** recorded in the architecture logs. ✅ Closes C-86. Good — this is the right
way to handle C-86's scalability caveat: measure, don't assume.

### 7g.6 🔴🔴 C-96 — 2 of 3 DOIs do not exist

Verified 08-10-2026 against **doi.org** and **Crossref** — two independent resolvers.

| DOI | Result |
|---|---|
| `10.2118/18247-PA` | ⚠️ **RESOLVES — to different metadata** (below) |
| `10.2118/12242-PA` | 🔴 **404 on both resolvers.** No such DOI |
| `10.2118/76722-PA` | 🔴 **404 on both resolvers.** No such DOI |

**What `10.2118/18247-PA` actually is:**

> Karakas, M., & **Tariq**, S. M. (**1991**). **Semianalytical Productivity Models for Perforated
> Completions.** SPE Production Engineering, **6**(01), **73–82**.

Supplied as: Karakas & **Tarik** (**1990**), *"Semi-Analytical Effects of Perforation on Well
Productivity"*, **5**(01), **42–50**.

| Field | Supplied | Actual |
|---|---|---|
| Author | Tarik | **Tariq** |
| Year | 1990 | **1991** |
| Title | Semi-Analytical Effects of Perforation on Well Productivity | **Semianalytical Productivity Models for Perforated Completions** |
| Volume / issue | 5(01) | **6(01)** |
| Pages | 42–50 | **73–82** |

🔴 **Consequence: C-91's $S_h$ correction is attributed to a paper whose actual metadata differs on four
of five fields, so its provenance is unestablished.** ✅ The *intent* was sound — the one real DOI points
at a genuine Karakas & Tariq paper on exactly this subject.

> 🔴 **All three must resolve before M3 and M7c.** Until then they are **UNVERIFIED**, and any coefficient
> attributed to them is unverified. Per the **M1 precedent** (Abudour 2014, Baled 2012 — both confirmed by
> DOI fetch), a citation is only usable once the DOI resolves *and* the metadata matches.

---

## 7h. Clamp purged, $\alpha_0$ direction, DOI re-verification (08-10-2026)

Corrections **C-97 … C-99**.

### 7h.1 ✅ C-97 — clamp purged. INV-7 regression closed.

`clamp()` removed from the $S_v$ arithmetic; raw $r_D$, $h_D$ used throughout. Out-of-envelope inputs set
`ValidityClass = ConvergedOutsideEnvelope` and append:

```rust
ValidityWarning::EmpiricalDomainExceeded {
    correlation: "Karakas-Tariq (1991)",
    variable: "r_D",
    value: r_D,
    valid_range: [0.001, 0.050],
}
```

✅ **This is exactly the C-59 / INV-7 §7b.2 mechanism**, and it handles the nuance correctly: the
regression domain is preserved as **warning metadata** rather than deleted as information or enforced
as a `min`/`max` in the arithmetic. **INV-7 regression closed.**

### 7h.2 🔴 C-98 — $\alpha_0$ is a penalty being read as a benefit

The unification $r'_w(\theta)=\alpha_0(\theta)(r_w+L_p)$ for **all** phases is a real improvement: one
continuous function, no 0° discontinuity, and $S_h(0^\circ)$ moves from $+0.365$ to $+0.057$.

🔴 **But the $\alpha_0$ table is interpreted in the wrong direction.** Measured:

| $\theta$ | Planes $N$ | $\alpha_0$ | $\log_4 N$ | $0.25+0.476\log_4 N$ | error |
|---|---|---|---|---|---|
| 0° | 1 | 0.250 | 0.0000 | 0.2500 | +0.0000 |
| 180° | 2 | 0.500 | 0.5000 | 0.4880 | +0.0120 |
| 120° | 3 | 0.618 | 0.7925 | 0.6272 | −0.0092 |
| 90° | 4 | 0.726 | 1.0000 | 0.7260 | +0.0000 |

**RMS error 0.0076** on a 0.25–0.73 range. ✅ $\alpha_0(\theta)\approx 0.25 + 0.476\log_4 N$ — to within
rounding, **$\alpha_0$ is an affine function of $\ln N$, the plane count.**

> 🔴 **$\ln N$ is the wellbore flow-convergence penalty.** Every additional perforation plane converges
> into the same near-wellbore region, so **more planes ⇒ more convergence ⇒ more skin ⇒ worse.**
> Monotonically **larger** $\alpha_0$ therefore means **worse**.
>
> The submission treats monotonically **larger** $\alpha_0$ as **better** — larger $r'_w$, more negative
> $S_h$, higher productivity.
>
> 🔴 **The table encodes a penalty and is being read as a benefit.** If that reading is right, then
> $S_h(90^\circ)$ is the **worst** phasing, not the best — and $0^\circ$ the best. Which is **standard
> perforation practice**: align the perforation plane with the natural fracture / maximum principal
> stress. The submitted table makes $0^\circ$ the only *damaging* case and $90^\circ$ the *best* —
> **the reverse of both** the table's own structure and standard practice.
>
> ⚠️ **This cannot be settled from the material supplied**, because the $\alpha_0$ values are attributed
> to a specific table in a paper whose DOI **does not resolve** (C-99). **Until then the 0°/90° skin
> ordering is declared-absent**, not asserted — the engine emits $S_h$ with a
> `PhasingConventionUnverified` warning rather than a confidence it has not earned.

### 7h.3 🔴🔴 C-99 — C-96 re-verification failed; a claim was relabelled, not corrected

The response re-asserts `10.2118/12242-PA` and `10.2118/76722-PA` and changed the wording from
*"Verified DOI"* to *"OnePetro Index"*.

**Re-checked 08-10-2026, four ways:**

| Check | `10.2118/12242-PA` | `10.2118/76722-PA` |
|---|---|---|
| `doi.org` | 🔴 **404** | 🔴 **404** |
| `api.crossref.org/works/{DOI}` | 🔴 **404** | 🔴 **404** |
| Crossref title search, exact paper title | 🔴 **not in index** | 🔴 **not in index** |
| `onepetro.org` article page | 🔴 403 (paywall/bot block — inconclusive) | not attempted |

> 🔴 **A DOI is a registered identifier. If it returns 404 from `doi.org`, it is not registered**, and
> *"OnePetro Index"* is not a registered-DOI claim — it is a softer label on the same unverified
> string. The bibliographic search is the decisive evidence: neither title appears anywhere in
> Crossref's index, whose top hits for those titles are unrelated compositional papers.

**Accepted:** the **Karakas & Tariq metadata correction** — Karakas, M. & **Tariq**, S. M. (**1991**),
*Semianalytical Productivity Models for Perforated Completions*, SPE Production Engineering
**6**(01), **73–82**, `10.2118/18247-PA`. ✅ That now matches what the DOI resolves to.

🔴 **Consequence for C-98:** the $\alpha_0$ table is attributed to *"Karakas & Tariq (1991, Table 1)"* —
an **unverified** source. Under C-96's own rule, a citation is usable only once the DOI resolves **and**
the metadata matches **and** the specific table is reachable. None of that holds.

> ⚠️ **The branch unification may still be correct on engineering grounds** — continuity in $\theta$ is
> desirable regardless. **What is not established is the $\alpha_0$ values and, critically, their
> direction.** Those need the source.

### 7h.4 📌 Durable rule — what "verified" must mean

Two consecutive rounds in which a verification claim failed verification (Ruling 9: 2 of 3 DOIs absent;
Ruling 10: the same 2 absent, wording softened rather than identifiers corrected).

**A citation is usable only when all three hold:**

| # | Test | Tool |
|---|---|---|
| 1 | The DOI returns **HTTP 200** from `doi.org` | `https://doi.org/{DOI}` |
| 2 | Registered metadata **matches the citation field-for-field** — authors, year, title, volume, issue, pages | DOI content negotiation |
| 3 | The **specific table, figure or coefficient** is reachable and matches | the paper |

**A claim of verification is not itself evidence.** Write this into
[`register_spec.md`](register_spec.md) as the definition of a `PROVENANCE`-clear entry.

---

## 7i. DOI corrected, anisotropy transform, and a withdrawn claim (08-10-2026)

Corrections **C-99a/b, C-100, C-101, C-102**.

### 7i.1 ✅ C-99a — Watts (1986) verified

The **DOI digit was wrong; the paper was real and the metadata was already correct.** Corrected to
`10.2118/12244-PA`:

> **Watts, J. W. (1986).** *A Compositional Formulation of the Pressure and Saturation Equations.*
> SPE Reservoir Engineering, **1**(03), **243–252**. https://doi.org/10.2118/12244-pa

| Field | Supplied | Registered | Match |
|---|---|---|---|
| Author | Watts, J. W., 1986 | Watts, J. W., 1986 | ✅ |
| Title | A Compositional Formulation of the Pressure and Saturation Equations | *(identical)* | ✅ |
| Journal / vol / issue / pages | SPE Reservoir Engineering, 1(03), 243–252 | *(identical)* | ✅ |

✅ **Closes the Watts half of C-90 / C-96 / C-99.**

> 📌 **This is a different failure from Ruling 9.** There, the DOI did not exist *and* the metadata was
> wrong. Here the DOI was wrong in **one digit** while everything else was right. A 404 therefore does
> **not** imply a fabricated citation — which is why §7h.4 tests the DOI *and* the metadata separately.
> Looking the paper up by title, as the engineer did, was the correct response.

### 7i.2 🔴 C-99b — Wong, Coats & Thomas still unverified

No corrected identifier was supplied. Crossref title search returns **no match** — and Coats is well
indexed (8 of his papers returned, including `10.2118/8284-pa`, `10.2118/29111-MS`, `10.2118/35164-pa`),
so the absence is meaningful rather than a coverage gap. `10.2118/76722-PA` remains **404** at both
resolvers.

**Required: the SPE paper number, or a DOI that returns 200.** ⚠️ Coats, Thomas & Pierson (1995),
*"Compositional and Black Oil Reservoir Simulation"*, `10.2118/29111-MS` **does** resolve and is
thematically adjacent — but **it is a guess and must not be silently substituted.**

### 7i.3 ✅ C-100 — I withdraw my $\alpha_0$ direction claim

**My C-98 was wrong. The engineer's separation is correct.**

| Regime | What governs | Effect of plane count $N$ |
|---|---|---|
| **Isotropic matrix** ($k_x=k_y$) — what $\alpha_0$ encodes | Geometric flow distribution: more planes cover more of the drainage circumference, so **flow convergence into the entry area is reduced** and more entry area is used | **More planes = better.** $\alpha_0\uparrow$ = better |
| **Anisotropic / fractured** ($k_x\neq k_y$, $\sigma_{H,\max}$) | Permeability anisotropy and stress alignment — **outside** the isotropic hydraulics $\alpha_0$ encodes | Alignment dominates; 0°/180° preferred |

✅ **Larger $\alpha_0$ = better is correct** for the regime the correlation is defined in. My
"$\ln N$ is a convergence penalty" claim was **wrong** — I correctly recalled that flow converges into a
limited entry area, and then incorrectly concluded that more entries make convergence worse. They make
it *better*.

> **My measurement stands** ($\alpha_0\approx0.25+0.476\log_4N$, RMS 0.0076). **My interpretation did
> not.** Withdrawn.
>
> 📌 **This is the sharpest instance of the recurring failure in this whole exchange:** the arithmetic
> was right and the *physics* was wrong, because I never asked **what regime the correlation is defined
> in** before interpreting it. Six of my claims are now withdrawn across Rulings 8–11.

### 7i.4 🔴 C-101 — the anisotropy transform cannot achieve its stated goal

$$\theta' = \arctan\!\Big(\sqrt{k_x/k_y}\,\tan\theta\Big)$$

🔴 **0° and 90° are fixed points of this map** — measured across $k_x/k_y \in \{1, 4, 10\}$:

| $k_x/k_y$ | $\theta=0°$ | $\theta=45°$ | $\theta=90°$ | $\theta=120°$ | $\theta=180°$ |
|---|---|---|---|---|---|
| 1.0 | 0.000 | 45.000 | **90.000** | **−60.000** | **−0.000** |
| 4.0 | 0.000 | 63.435 | **90.000** | **−73.898** | **−0.000** |
| 10.0 | 0.000 | 72.452 | **90.000** | **−79.653** | **−0.000** |

> 🔴 **Because both endpoint angles are fixed, anisotropy cannot reverse the ordering.** $0°$ still gets
> $\alpha_0 = 0.250$ (worst) and $90°$ still gets $0.726$ (best), no matter how strong $k_x/k_y$ is.
> **The transform does not achieve the purpose stated for it** — making $0°$ superior in anisotropic media.
>
> 🔴 **And $180° \mapsto 0°$.** The moment anisotropy is detected, opposed 2-plane phasing
> **silently becomes 1-plane phasing**: $\alpha_0$ 0.500 → 0.250, $S_h$ −0.636 → **+0.057**. A discrete
> configuration change is being made by a continuous coordinate map, with no warning.
>
> 🔴 $120° \mapsto$ a **negative** angle (−60° to −79.7°), which lands in the fallback branch.

⚠️ Also incomplete for its purpose: the transform uses only $k_x, k_y$, but $S_h$ is the **horizontal**-
well skin, where $k_z$ enters both $h_D$ and $S_v$. ⚠️ And the trigger
$|k_x-k_y|/k_x > 0.05$ normalises by $k_x$ only, so it is hypersensitive when $k_x \ll k_y$ — use
$|k_x-k_y|/\max(k_x,k_y)$.

### 7i.5 🔴🔴 C-102 — the fallback is a NaN generator

```rust
_ => 0.250 + 0.476 * (theta_eff / 90.0).log(4.0),
```

Rust's `f64::log(base)` is $\ln x / \ln(\text{base})$, so this evaluates $0.250 + 0.476\log_4(\theta/90)$.

🔴 **It crosses zero at $\theta = 43.5°$.** Measured:

| $\theta$ | $N = 360/\theta$ | $\alpha_0$ correct | $\alpha_0$ submitted | $S_h$ submitted |
|---|---|---|---|---|
| 15° | 24.0 | 1.3412 | **−0.3652** | **NaN** 🔴 |
| 30° | 12.0 | 1.1032 | **−0.1272** | **NaN** 🔴 |
| 45° | 8.0 | 0.9640 | 0.0120 | **+3.094** 🔴 |
| 60° | 6.0 | 0.8652 | 0.1108 | +0.871 |
| 90° | 4.0 | 0.7260 | 0.2500 | +0.365 |
| 180° | 2.0 | 0.4880 | 0.4366 | −0.512 |

🔴 **For $\theta < 43.5°$: $\alpha_0 < 0 \Rightarrow r'_w < 0 \Rightarrow \ln(r_w/r'_w) =$ NaN — straight
into the global Jacobian.** This is the **same failure class as C-69** (NaN permeability) and **C-72**.
Three separate NaN generators have now been identified in this project.

🔴 **It does not even fit its own table:** at $\theta = 90°$ the formula returns **0.250** where the table
says **0.726**. It is not a fit of the four points at all.

✅ **Correct form** — the number of distinct planes is $N = 360/\theta$, so:

$$\boxed{\;\alpha_0(\theta) = 0.250 + 0.476\,\log_4\!\left(\frac{360}{\theta}\right)\;}$$

| $\theta$ | $\alpha_0$ fit | Table |
|---|---|---|
| 360° (1 plane) | 0.2500 | 0.250 ✅ |
| 180° (2) | 0.4880 | 0.500 |
| 120° (3) | 0.6272 | 0.618 |
| 90° (4) | 0.7260 | 0.726 ✅ |

Monotone in $\theta$ and **consistent with C-100** — more planes, larger $\alpha_0$, better. ⚠️ RMS
deviation 0.0076 (≈1 %): **the four tabulated angles must retain their exact values**, and this fit is
for **interpolation between** them only.

### 7i.6 📌 CI gate this should produce

| Gate | Assertion |
|---|---|
| **No negative $\alpha_0$** | For all $\theta \in (0, 360)$, $\alpha_0(\theta) \ge \alpha_0(360°)$, so $r'_w \ge r_w$ and $S_h \le 0$. Any $\alpha_0 < 0$ **fails the build** |
| **Fit passes through the table** | The interpolant evaluated at $0°, 90°, 120°, 180° returns the tabulated values to within the stated 1 % |
| **Discrete phasings survive anisotropy** | $\theta \in \{0°, 90°, 120°, 180°\}$ maps to itself, never onto another tabulated angle |
| **No NaN, ever** | Property test: $S_h(\theta)$ is finite for all $\theta \in (0°, 360°)$ |

---

## 7j. NaN eliminated; the anisotropy remedy inverts (08-10-2026)

Corrections **C-103 … C-107**.

### 7j.1 ✅ C-103 — C-102 closed

$$\boxed{\;\alpha_0(\theta) = 0.250 + 0.476\,\log_4\!\left(\frac{360^\circ}{\theta}\right)\;}$$

| $\theta$ | $N$ | Fit | Table | Error |
|---|---|---|---|---|
| 360° | 1 | 0.2500 | 0.250 | ✅ exact |
| 180° | 2 | 0.4880 | 0.500 | −2.4 % |
| 120° | 3 | 0.6272 | 0.618 | +1.5 % |
| 90° | 4 | 0.7260 | 0.726 | ✅ exact |
| 45° | 8 | 0.9640 | — | positive ✅ |
| 15° | 24 | 1.3412 | — | positive ✅ |

**Engine rule accepted:** exact values for the four tabulated angles, fit for interpolation only.

🔴 **One landmine — the replacement introduces the failure shape it just fixed.** The formula is
**undefined at $\theta = 0$**:

$$\frac{360}{0} = \infty \;\Rightarrow\; \log_4\infty = \infty \;\Rightarrow\; \alpha_0 = \infty \;\Rightarrow\; S_h = \ln(0) = -\infty$$

It is safe **only** because 0° is served by the exact-table `match` arm. C-102 removed a *negative*
$\alpha_0$ producing **NaN**; the fix introduces a *division by zero* producing **$-\infty$**. Both
reach the Jacobian. **Ruled: a CI gate must assert the fallback branch is unreachable at $\theta = 0$**,
and $\theta = 0$ must be handled *before* any division — **in the type or the gate, not in control flow.**

### 7j.2 ✅ C-105 — C-99b closed by verified substitution

`10.2118/29111-MS` returns 200:

> **Coats, K. H., Thomas, L. K., & Pierson, R. G. (1995).** *Compositional and Black Oil Reservoir
> Simulation.* **SPE Reservoir Simulation Symposium**.

⚠️ **Two material differences from the original claim:**

| | Originally claimed | Verified |
|---|---|---|
| **Citation class** | Peer-reviewed journal, *SPE Reservoir Evaluation & Engineering* | 🔴 **Conference symposium paper (`-MS`)** |
| Author initial | Thomas, **L. O.** | ✅ Thomas, **L. K.** |

✅ Acceptable as the volume-balance authority **provided** the register records it as a **conference
paper** and the volume-balance content is confirmed against it. ⚠️ The $-MS$ suffix is not cosmetic —
it changes the evidence class under [`register_spec.md`](register_spec.md).

### 7j.3 🔴 C-104 — a verified DOI regressed in the summary

The Summary binds *"Watts (1986) `10.2118/12242-PA`"* — the **404** identifier. `10.2118/12244-PA` is
the one that returns 200 with field-for-field metadata match (**C-99a**).

⚠️ **Copy-paste regression, not a new error.** But it is precisely the identifier §7h.4 exists to pin,
so it must be corrected **before the manifest is written** — a manifest is exactly where a stale DOI
becomes permanent.

### 7j.4 🔴🔴 C-106 — the anisotropy remedy inverts its stated goal

$$L'_p(\phi_p) = L_p\sqrt{\frac{\bar k}{k_x}\cos^2\phi_p + \frac{\bar k}{k_y}\sin^2\phi_p},\qquad \bar k = \sqrt{k_xk_y}$$

The $\tan\theta$ map purge is **correct and accepted**. This replacement is not. Three defects:

**(a) 🔴 Dimensional.** The $\sqrt{\bar k}$ prefactor carries units $\sqrt{\text{mD}}$. So $L'_p$ is
$m\cdot\sqrt{\text{mD}}$, **not metres** — and $r_w + L'_p$ in $r'_w = \alpha_0(r_w + L_p)$ is
**dimensionally invalid**.

**(b) 🔴 Inverted — it makes the aligned case worse.** Measured:

| $k_x/k_y$ | $\phi_p$ | $L'_p$ (m) | $\alpha_0$ | $r'_w$ (m) | $S_h$ |
|---|---|---|---|---|---|
| 1.0 | 0° | 0.30000 | 0.250 | 0.10200 | +0.057 |
| **4.0** | **0°** | **0.21213** | 0.250 | 0.08003 | **+0.300** 🔴 |
| **4.0** | **90°** | **0.42426** | 0.250 | 0.13307 | **−0.209** 🔴 |
| 10.0 | 0° | 0.16870 | 0.250 | 0.06918 | **+0.445** 🔴 |
| 10.0 | 90° | 0.53348 | 0.250 | 0.16037 | **−0.395** 🔴 |

Isotropic reference: $S_h(0°) = +0.057$.

> 🔴 **At $k_x/k_y = 4$, perforating along the high-permeability axis gives $+0.300$ — *worse* than
> isotropic — while perforating along the LOW-permeability axis gives $-0.209$, the best result in the
> table.** The stated goal was *"perforating at 0° parallel to the high-permeability axis eliminates
> cross-bedding resistance."* **The remedy produces the exact inverse**, and the error grows with
> anisotropy (+0.445 at $k_x/k_y = 10$).

**(c) 🔴 Category error.** $L_p$ is **perforation penetration length** — *drilled geometry*, fixed at
perforating time, **independent of the permeability tensor**. A permeability tensor transform does not
belong on a hole dimension. Anisotropy belongs in the **flow response**, not in the geometry.

### 7j.5 📌 C-106 needs a decision, not another formula

| Option | Content |
|---|---|
| **(a) Drop the transform** ✅ **recommended** | $L_p$ is geometry and is **not** permeability-scaled. Keep $\alpha_0(\theta)$ raw, declare anisotropy via `ValidityWarning`, and let $k_x,k_y$ enter where they belong — the permeability tensor in the flow equations and $h_D$/$S_v$ |
| **(b) Keep an isotropic-equivalent length** | Must be **normalised** so the factor is 1.0 along $k_{max}$, **must** carry metres, and must be applied as a **flow** correction **downstream of $r'_w$** — never inside $r_w + L_p$ |

### 7j.6 ⚠️ C-107 — $\phi_p$ undeclared, and $\alpha_0$ is now raw under anisotropy

🔴 **$\phi_p$ is a new undeclared symbol** — perforation **azimuth**, distinct from the phasing
**spacing** $\theta$. **CONF-31's pattern, fourth occurrence** from this one correlation.

⚠️ With the angle map purged, $\alpha_0(\theta)$ is applied **raw** regardless of anisotropy. That is
**defensible** — **C-100** established $\alpha_0$ encodes *isotropic* distribution — but it makes
$\alpha_0(\theta)$ a **crude proxy** once anisotropy is significant, because equal angular spacing yields
*unequal* flow contribution per plane. `ValidityWarning::AnisotropicPerforationTransformation` must
therefore cover **both** the $L_p$ handling **and** the raw use of an isotropic correlation.

### 7j.7 📌 Consolidated CI gate for this correlation

| Gate | Assertion |
|---|---|
| **No negative $\alpha_0$** | $\alpha_0(\theta) \ge \alpha_0(360°) = 0.250$ for all $\theta \in (0°, 360°)$, so $r'_w \ge r_w$ and $S_h \le 0$ |
| **No infinity** | The fallback is **unreachable** at $\theta = 0$; assert $S_h$ finite for all $\theta \in (0°, 360°]$ |
| **Fit passes the table** | Interpolant at $90°$ and $360°$ returns tabulated values exactly |
| **Discrete phasings survive** | $\theta \in \{0°, 90°, 120°, 180°\}$ never maps onto another tabulated angle |
| **Anisotropy declared** | Any run with $A_k > 0.05$ emits `ValidityWarning::AnisotropicPerforationTransformation` |
| **No permeability transform on geometry** | $L_p$ enters $r_w + L_p$ **unmodified** |

---

## 7k. DOIs bound, Option (a) adopted, Symbol Register Rule (08-10-2026)

Corrections **C-108 … C-111**.

### 7k.1 ✅ C-108 — three DOIs bound and class-labelled

| Identifier | Reference | Class | DOI resolves |
|---|---|---|---|
| `29111-MS` | Coats, K. H., **Thomas, L. K.**, & Pierson, R. G. (1995), *Compositional and Black Oil Reservoir Simulation*, SPE Reservoir Simulation Symposium | 🔴 **Conference (`-MS`)** | ✅ 200 |
| `12244-PA` | Watts, J. W. (1986), *A Compositional Formulation of the Pressure and Saturation Equations*, SPE Reservoir Engineering **1**(03) 243–252 | ✅ **Journal** | ✅ 200, metadata exact |
| `18247-PA` | Karakas, M., & **Tariq**, S. M. (1991), *Semianalytical Productivity Models for Perforated Completions*, SPE Production Engineering **6**(01) 73–82 | ✅ **Journal** | ✅ 200, metadata exact |

✅ The `12242-PA` regression is corrected, and the **$-MS$ class distinction is retained rather than
quietly upgraded** — which is the right instinct: a conference paper is not a journal paper, and the
register should not blur it.

⚠️ **Test 3 of §7h.4 is still outstanding for $\alpha_0$** — the *specific table* in `18247-PA` is not
reachable. So the **$\alpha_0$ values remain sourced-but-unverified at table level**, which is why
C-98/C-100 ended in a withdrawal and a declaration rather than a closure.

### 7k.2 ✅ C-109 — Option (a) adopted

$L_p$ restored as **raw drill geometry**. All three defects purged: the $\sqrt{\bar k}$ dimensional
error, the sign inversion, and the category error. ✅

**Anisotropy re-routed correctly** to where it physically belongs:

| Route | Where |
|---|---|
| $K_{ij}$ | Grid block tensor in the FVM flow equations |
| $h_D = \dfrac{h_p}{L_p}\sqrt{\dfrac{k_h}{k_v}}$ | Vertical/anisotropic skin $S_v$ |

✅ `ValidityWarning::AnisotropicPerforationRawCorrelation { k_x, k_y, ratio }` with the
$\max(k_x,k_y)$ normalisation — the asymmetry defect fixed. **C-106 closed.**

### 7k.3 ⚠️ C-110 — the Symbol Register Rule is right; its own example breaks it

**Accepted as project policy.** Every symbol entering a constitutive relation, physical law or DTO
declares four points:

1. **Name**
2. **SI / field units**
3. **Valid physical & empirical domain**
4. **Verified literature source / DOI**

🔴 **But the rule's own worked example declares $L_p \in [0.05,\ 1.50]\ \text{m}$ with no cited
source** — so the rule's first application passes a naive checklist while violating its **own fourth
point**, which is exactly the defect class it exists to remove.

> ✅ **Ruled: the rule binds its own examples.** Every worked example must carry a DOI or be explicitly
> marked `SOURCE_PENDING`. ✅ The $\alpha_0$ table is the standing counter-example — **unverified at
> table level**, and should be marked `SOURCE_PENDING` rather than cited as though settled.

### 7k.4 🔴 C-111 — the $\alpha_0$ fit extrapolates without declaring it

The tabulated angles are $\{90°, 120°, 180°, 360°\}$, so the interpolation domain is **$[90°, 360°]$**.
The engine rule — *"non-standard continuous angles shall use the $N = 360/\theta$ fit"* — carries **no
lower bound**:

| $\theta$ | $N$ | $\alpha_0$ | $r'_w$ | $r'_w/r_w$ | $S_h$ | Status |
|---|---|---|---|---|---|---|
| 90° | 4 | 0.7260 | 0.296 m | 2.74× | −1.009 | ✅ interpolation |
| 72° | 5 | 0.8026 | 0.328 m | 3.03× | −1.109 | 🔴 extrapolation |
| 45° | 8 | 0.9640 | 0.393 m | 3.64× | −1.292 | 🔴 extrapolation |
| 15° | 24 | 1.3412 | 0.547 m | **5.07×** | −1.623 | 🔴 extrapolation |
| 5° | 72 | 1.7184 | 0.701 m | **6.49×** | −1.871 | 🔴 extrapolation |
| 1° | 360 | 2.2711 | 0.927 m | **8.58×** | −2.149 | 🔴 extrapolation |

🔴 **An "equivalent wellbore radius" of 8.6× the actual hole, and a productivity index ~39× unskinned,
emitted with no warning.**

🔴 **C-97 was accepted in this same round for exactly this defect class** — *declare the domain, do not
substitute silently* — and it was applied to Karakas-Tarik's $r_D/h_D$, **but not** to the new
$\alpha_0$ fit.

> ✅ **Ruled: bound the fit at $\theta \ge 90°$ and emit
> `ValidityWarning::ExtrapolatedPhasing { theta, plane_count }` below it.** ⚠️ **Do not clamp $\theta$**
> — that is **C-93**, and the fix is declaration, not substitution.

⚠️ **Also still open, carried from C-103:** $\theta = 0$ remains undefined in the fit
($360/0 \Rightarrow \alpha_0 = \infty \Rightarrow S_h = -\infty$). The table path is correct —
$r'_w = 0.102$ m, $S_h = +0.057$ — but the guard is **control flow, not a gate**.

### 7k.5 📌 The rule needs a gate that tests the rule

C-110 and C-111 are one lesson at two levels: **a defect class is banned, then reappears in the rule's
own example, then again in the same class of quantity the rule governs.**

| Gate | Assertion |
|---|---|
| **Fit domain declared** | $\alpha_0$ fit evaluated only for $\theta \in [90°, 360°]$; below 90° emits `ValidityWarning::ExtrapolatedPhasing` |
| **No $\theta = 0$ in the fit** | $\theta = 0$ served by the exact table, enforced **in the type or the gate**, not by a `match` arm |
| **Symbol register completeness** | Every symbol in a constitutive relation has all 4 declaration points; **missing source ⇒ `SOURCE_PENDING`, not silent acceptance** |
| **Register self-test** | The rule's own worked examples pass the rule — this is the gate that would have caught C-110 |

---

## 7l. `completion_skin_spec.md` review (08-10-2026)

Spec: `D:\Downloads\karakas_tariq_completion_skin_spec.md` (135 lines, external to the repo).
Corrections **C-112 … C-115**.

### 7l.1 ✅ C-112 — C-109 and C-111 accepted as specified

| Element | Status |
|---|---|
| $L_p$ as **raw drill geometry**, never permeability-scaled | ✅ **INV-7 compliant** |
| Anisotropy → **$K_{ij}$** in FVM | ✅ correct route |
| Anisotropy → **$h_D = \dfrac{h_p}{L_p}\sqrt{\dfrac{k_h}{k_v}}$, $k_h=\sqrt{k_xk_y}$, $k_v=k_z$ | ✅ **dimensionally closes** — both factors dimensionless |
| `AnisotropicPerforationRawCorrelation` with $\max(k_x,k_y)$ | ✅ asymmetry fixed |
| Extrapolation guard $\theta<90^\circ$, **no clamp**, `ExtrapolatedPhasing { theta_deg, plane_count, computed_alpha_0 }` | ✅ **C-97 applied correctly** |
| $\theta = 0^\circ$ intercepted at the match/type level | ✅ |
| Interpolation domain **$[90^\circ, 360^\circ]$** declared | ✅ |

**C-109 and C-111 closed.**

### 7l.2 ✅ C-113 — gates 1 and 2 verified numerically

| Gate | Computation | Assertion | Result |
|---|---|---|---|
| **1** | $\theta=45^\circ \Rightarrow N=8$, $\alpha_0=0.9640$, $r'_w=0.39331$ m | `s_h < -1.0` | $S_h=\mathbf{-1.2925}$ ✅ **PASSES** |
| **2** | $\theta=0^\circ \Rightarrow \alpha_0=0.250$, $r'_w=0.10200$ m | `round(s_h*1000)/1000 == 0.057` | $S_h=\mathbf{+0.05716}$ ✅ **PASSES** |

✅ Both assertions are **derived from the closed mathematics**, not guessed — they encode **C-103** and
**C-111** correctly. This is what a good gate looks like, and it makes the two failures below sharper by
contrast.

### 7l.3 🔴🔴 C-114 — two new DOIs do not resolve, and gate 3 cannot tell

The symbol register adds two citations:

| Symbol | Cited source | `doi.org` | Crossref |
|---|---|---|---|
| $r_w$ | **Fanchi (2002)**, `10.1016/B978-012248308-0/50001-X` | 🔴 **404** | 🔴 **absent** |
| $k_h$, $k_v$ | **Aziz & Settari (1979)**, `10.1016/C2013-0-06222-0` | 🔴 **404** | 🔴 **absent** |

⚠️ **Both authors are well indexed** — a Crossref title search returns 5 real Fanchi papers, and 2 real
Aziz & Settari papers (SPE-3174, SPE-72-01-04). The absences are therefore **meaningful, not coverage gaps**.

🔴 **And gate 3 is structurally incapable of detecting this:**

```rust
assert!(symbol.source_doi.contains("10.") || symbol.source_doi == "SOURCE_PENDING");
```

| Input | Gate 3 |
|---|---|
| `10.1016/B978-012248308-0/50001-X` (404) | ✅ **PASS** |
| `10.1016/C2013-0-06222-0` (404) | ✅ **PASS** |
| `10.2118/99999-PA` (**fabricated**) | ✅ **PASS** |
| `10.1016/totally-made-up` (**fabricated**) | ✅ **PASS** |

> 🔴 **The rule says "valid DOI or SOURCE_PENDING". The gate tests "contains `10.`".** The gate cannot
> enforce the rule — a substring match is not a resolution check.
>
> ✅ **Ruled: the source test must perform the §7h.4 test-1 check** — HTTP 200 from `doi.org` — resolved
> into a **lockfile** committed alongside the register, so the network call happens once and the result is
> reviewable in a diff. A substring match must be **rejected outright**, not tightened.

### 7l.4 🔴 C-115 — the register's data structure cannot express its own domains

The register declares ten symbols, but `valid_range: (f64, f64)` carries **neither openness nor a
symbol-valued bound**:

| Symbol | Declared domain | Representable as `(f64,f64)`? |
|---|---|---|
| $k_h$, $k_v$ | $(0.0,\ 50000.0]$ mD — **open** lower bound, zero excluded | 🔴 **no** |
| $r_{cz}$ | $[r_p,\ r_p+0.050]$ m — lower bound is a **symbol** | 🔴 **no** |

Measured consequence of the gate `assert valid_range.0 < valid_range.1`:

$$0.0 < 50000.0 = \texttt{true} \;\Rightarrow\; \textbf{the gate accepts } k_h = 0 \text{ mD}$$

🔴 **Zero permeability is physically meaningless, and the declared domain explicitly excludes it.** And
$r_{cz}$ cannot be stored at all.

> ⚠️ **This is the C-83 / C-106 dimensional-closure class recurring one level up.** C-83 was
> $\eta_{erosion}$ missing units; C-106 was $\sqrt{\bar k}$ breaking a sum. Here it is **the register's own
> type** being unable to hold the physics. The defect has moved from the equations to the **schema**.
>
> ✅ **Ruled: a typed bound.**
>
> ```rust
> enum BoundValue { Num(f64), Sym(SymbolId) }
> struct Bound     { value: BoundValue, inclusive: bool }
> struct ValidRange { lower: Bound, upper: Bound }
> ```
>
> ⚠️ Note $r_{cz}\\in[r_p,\\ r_p+0.050]$ is also **symbol-valued at the upper bound** — $r_p + 0.050$ is an
> expression, not a constant. So `Sym` must resolve an **expression** over the register, not a single
> symbol id.

### 7l.5 📌 The transferable finding

**C-114 and C-115 are the same shape: the enforcement mechanism is weaker than the rule it enforces.**

| Case | Rule says | Gate does | Result |
|---|---|---|---|
| **C-114** | "valid DOI" | `contains("10.")` | Accepts fabricated identifiers |
| **C-115** | domain excludes $k_h=0$ | `lo < hi` | Accepts $k_h = 0$ |

✅ **Gates 1 and 2 are the counter-example and the standard**: they were built from the closed mathematics,
so they fail when the mathematics is wrong. ✅ **A gate is only as good as the derivation it encodes.**

| Consolidated gate | Assertion |
|---|---|
| **G5 — DOI resolution** | Every register entry's DOI returns **HTTP 200** from `doi.org`, recorded in a committed **lockfile**; re-checked when the lockfile changes |
| **G6 — typed bounds** | No entry uses a bare `(f64,f64)`; bounds are `Bound` values with explicit inclusivity, and symbol/expression bounds resolve |
| **G7 — domain self-consistency** | Every register entry's own worked example lies strictly inside its declared bounds, with openness respected |

> 📌 **C-110's own lesson, third instance.** A rule is written to prevent a defect (**C-97**), the defect
> reappears in the rule's worked example (**$L_p$'s range**), then in the same class of quantity the rule
> governs (**the $\alpha_0$ extrapolation**), then the rule's gate cannot catch either (**C-114, C-115**).
> **The pattern is reliable. The fix is structural: gates must encode derivations, not restate intentions.**

---

## 7m. Local library audit — `D:\RAG`, 129 PDFs (08-10-2026)

Corrections **C-116 … C-119**. Method: `pymupdf` full-text extraction over **every page of all 129
PDFs**. Installed to **system** Python — the project `.venv` was deliberately left untouched.

### 7m.1 🔴 C-116 — CONF-31's blocker cannot be closed locally

A literal search for `karakas` or `tarik` across **all pages of all 129 PDFs**:

> ### NO DOCUMENT CONTAINS `karakas` OR `tarik` ANYWHERE

✅ **The Karakas & Tariq (1991) α₀ table is not in the local library.** This is the decisive answer to
the question the library was meant to answer.

**Ruled:**

| Action | Detail |
|---|---|
| **α₀ table status** | `SOURCE_PENDING` |
| **Skin model status** | **Provenance-unverified dependency** |
| **Runtime** | Emit `ValidityWarning::CorrelationProvenanceUnverified { correlation: "Karakas-Tariq 1991 alpha_0", evidence: "local library contains no copy" }` |
| **Milestone** | ⚠️ **M7c cannot be gated on it** until the paper is obtained via institutional access |

> 📌 **This is a characterisation, not a closure.** The defect is now *known and bounded* rather than
> open-ended — which is what makes it actionable. It also means the α₀-dependent results must carry an
> explicit provenance warning, exactly as C-116 requires.

### 7m.2 ✅ C-117 — Aziz & Settari found, and the DOI failure diagnosed

`kupdf.net_khaled-aziz-reservoir-simulation.pdf` — confirmed from three separate front-matter pages:

> **AZIZ, Khalid** (Professor of Chemical Engineering, University of Calgary) **& SETTARI, Antonin**
> (Manager of Technical Developments, Intercomp Resource Development & Engineering Ltd),
> ***Petroleum Reservoir Simulation***,
> **APPLIED SCIENCE PUBLISHERS LTD, LONDON**,
> British Library cataloguing: *Aziz, Khalid. Petroleum reservoir simulation.* **ISBN 0-85334-787-5**,
> © **J979**, 143 illustrations, 489 pp.

🔴 **This explains the 404.** The publisher is **Applied Science Publishers (UK)** — **not Elsevier**.
The `10.1016/` prefix is **Elsevier's**. A `10.1016/` DOI **could not be correct for this book**.

✅ **Ruled: replace `10.1016/C2013-0-06222-0` with `ISBN 0-85334-787-5`** — locally verifiable, and a
better identifier for a 1979 book.

⚠️ **The book has no perforation-skin correlation** — a single incidental "skin effect" mention on
p247. It **cannot** substitute for Karakas & Tariq.

### 7m.3 ⚠️ C-118 — Fanchi is present, but a different work

`vdocuments.mx_shared-earth-modeling.pdf` — John R. Fanchi, *Shared Earth Modeling*,
**Butterworth-Heinemann, an imprint of Elsevier Science, © 2002** (confirmed on the copyright page).

| | |
|---|---|
| ✅ Elsevier prefix **plausible** for Fanchi | unlike **C-117** |
| 🔴 But the register cites Fanchi for the **$r_w$ domain**, and this is a **different work** | Fanchi's wellbore-radius content would be in *Petroleum Reservoir Engineering: A Computer Approach*, not *Shared Earth Modeling* |

**Ruled: cite Fanchi by exact title + year; drop the unresolvable chapter DOI until verified.**

⚠️ **None of `10.2118/18247`, `10.2118/12244`, `10.2118/29111` appears as text anywhere in the corpus** —
so the three verified bibliography entries have **no local corroboration** either. They stand on the DOI
resolution check alone, which is sufficient but is the only evidence behind them.

### 7m.4 ✅ C-119 — MATERIAL ASSET: an SPE benchmark case is present

`D:\RAG\Data files\` holds **31 CMG GEM decks**. The important one:

```text
RESULTS SIMULATOR GEM 202310
*TITLE1 'SPE5 : SPE5 COMPOSITIONAL RUN 1'
*TITLE2 'WAG process with 1 year cycle'
*INUNIT  *FIELD
**--------------------------------------------------RESERVOIR DATA------
*GRID *CART 7 7 3
*DEPTH *TOP 1 1 1 975.0
*DI *CON 1000.0
*DJ *CON 1000.0
*DK *KVAR 50.0 30.0 20.0
POR KVAR
 0.2 0.22 0.18
```

✅ **SPE Comparative Solution Project Case 5** — the standard **compositional + WAG** benchmark, and
directly relevant to **M1–M4** and to **CONF-26** (spatial convergence promised, never tested).

Also present: `CO2 Flooding_BaseCase` · `PolymerFlooding_BaseCase` · `SAGD_BaseCase` / `SAGD_2D_` /
`SAGD_Green_` · `ShaleOil_HF_BaseCase` · `HydraulicallyFracturedBaseCase` · `WellTesting_Base` ·
`HM_00227` / `HM_00686` · and DTO sample files (`CompletionsDataSource`, `fluidProperties`,
`ElasticPropertyBuilding`, `Tornado`, `SimultaneousInversion`).

> ⚠️ **These are CMG-format decks, not the official SPE problem specifications.** They are a
> **starting point** for constructing a reference case; a **CMG reference run is still required** to
> provide a comparison target. The wiki's *"no SPE benchmark"* gap was accurate **for the repository**,
> but it is **materially incomplete for the machine** — which is the distinction that matters for M0
> planning.

### 7m.5 ⚠️ Two false positives eliminated

The first-pass keyword scan flagged two misleading hits. Both were checked and neither bears on the α₀
table:

| File | Hit | Reality |
|---|---|---|
| `Kappa\KAPPA DDA book 5.60.01.pdf` | `tariq` | **Umair Tariq**, a KAPPA engineer listed in the acknowledgements — not Karakas & Tariq |
| `Kappa\KAPPA DDA book 5.60.01.pdf` | `wellbore effect` | Means **wellbore storage**, not perforation phasing |
| `Onepetro\The SMART SRP Well…pdf` | `wellbore effect` | Same false positive |

> 📌 **A keyword hit is not evidence.** The same standard as §7h.4 applies to a local corpus: a name
> match must be traced to the actual citation before it counts. Both of these would have been
> reported as corroboration by a less careful pass.

### 7m.6 📌 Other local assets worth indexing

| Asset | Pages | Relevance |
|---|---|---|
| `kupdf.net_khaled-aziz-reservoir-simulation.pdf` | 489 | FVM numerics, iterative solvers, pseudo-functions, additive correction / IDC (Watts) |
| `slb\EclipseReferenceManual.pdf` | 2831 | Full compositional simulator reference |
| `Tnav manuals\tNavPVTDesignerGuideEnglish.pdf` | 279 | PVT / EOS workflow |
| `Tnav manuals\tNavWellDesignerGuideEnglish.pdf` | 433 | Well design, skin, IPR |
| `Kappa\KAPPA CHL Book 5.40.02.pdf` | 366 | Cased-hole logging, completion evaluation |
| `Tnav manuals\tNavFractureSimulatorGuideEnglish.pdf` | 247 | Fracture modelling — **relevant to CONF-51** |

⚠️ **None of these substitutes for the Karakas & Tariq paper** (C-116). But several bear directly on
still-open items: the **Eclipse reference manual** on **CONF-14** (solver/linear algebra), the
**Fracture Simulator guide** on **CONF-51** (PKN/KGD/radial + proppant), and **Aziz & Settari** on
**CONF-14** (implicit/FIM numerics).

---

## 7n. Literature parked; math batch CONF-15/16/25/29 (08-10-2026)

Literature work **parked** — [`literature_todo.md`](literature_todo.md) carries the acceptance test, the
per-item action, and a search protocol derived from the failures already made.
Corrections **C-120 … C-123**.

### 7n.1 ⚠️ C-120 — CONF-25 narrowed: the missing $-1$ is a classification defect

Measured, on a 4-component mixture ($x = [0.60, 0.25, 0.10, 0.05]$, $K = [2.5, 1.6, 0.8, 0.4]$):

| | $Q_{min}$ | argmin $y$ |
|---|---|---|
| **with** $-1$ | **−0.649150** | $[0.25, 0.25, 0.25, 0.25]$ |
| **without** $-1$ | **+0.350850** | $[0.25, 0.25, 0.25, 0.25]$ |
| max $\lvert\Delta y_i\rvert$ | — | **0.00e+00** |

✅ **The argmin is identical.** Subtracting a constant cannot move a minimiser, so the missing $-1$ is
**not** an optimisation defect.

🔴 **It is a classification defect.** The stability verdict is $Q_{min} < 0$. Measured $Q_{min} = +0.351$
without the constant — so **an unstable split cannot be detected at all**, and a stable one is
classified for the wrong reason. 🔴 **This is a INV-1-class failure**: the engine would report a converged,
correct-looking split that is thermodynamically wrong.

**Ruled specification:**

| # | Requirement |
|---|---|
| **i** | Minimise $\mathcal{W}(y)=\sum_i y_i\ln\!\left[\dfrac{y_i}{x_i}\cdot\dfrac{\phi_i^L}{\phi_i^V}\cdot\dfrac{1}{K_i}\right]-1$ subject to **TWO** constraints — $\sum_i y_i = 1$ **and** the volume balance. ⚠️ **Not one.** The submitted form carries no constraint at all |
| **ii** | $\phi_i^L$ and $\phi_i^V$ **named separately.** The submitted $\phi_i(y)$ is ambiguous, and the **ratio** is the entire mechanism |
| **iii** | 🔴 **The gradient is mandatory.** It requires $\partial\ln\phi/\partial y_i$ — EOS partials **along the volume-balance path** — so $\partial W/\partial y_i$ **cannot** be assembled from $\phi$ values alone. This is the practical reason the submitted sketch is not buildable |
| **iv** | **Use $\partial W/\partial y_i$ to iterate; use $Q_{min}$ to decide.** They are different quantities, and conflating them is what the missing $-1$ caused |

### 7n.2 ✅ C-121 — CONF-15 reclassified, and **my register was wrong**

Two independent claims withdrawn, both against my own finding:

**(c) Koval's placement — my sign instinct was wrong.** The design gives $t_{bt}\propto 1/K$, which
**decreases** with $K$. Severe fingering $\Rightarrow$ the displacing phase bypasses $\Rightarrow$ **earlier**
breakthrough. ✅ **The design's direction is physically correct.** I asserted a sign error and it was
mine, not theirs.

**(b) $(1-S_{wi})$ — the classical form is correct.** The textbook waterflood statement is
$W_{o,bt} = PV\,(1-S_{wi})/B_o$. Neglecting $S_{or}$ is a **known second-order refinement**, not a logical
or dimensional error.

**What genuinely remains is documentation only:**

| Missing | Note |
|---|---|
| Unit basis for $q_{inj}$ — reservoir or surface volumetric | measured: $V_p/(K\,q)$ closes to **seconds** only under stated assumptions |
| Whether $K$ is declared dimensionless | ditto |
| The approximation level, stated | so a reader knows $S_{or}$ is neglected deliberately |

> ⚠️ **CONF-15 $\Rightarrow$ 🟡 documentary.** 📌 **The meta-point:** this finding sat at $\orangearrow$
> material for nine rulings partly because it **read** like a physics error. **Severity assigned by
> plausibility rather than by derivation is the same failure mode as asserting without checking** — it
> produced a finding that survived on the strength of its phrasing. **Derive the number, then grade it.**

### 7n.3 ✅ C-122 — CONF-16: a missing reference volume, same shape as C-69

$\boldsymbol\varepsilon_{chem} = \tfrac13 \Delta V_{m,tot}\,\mathbf I$ carries **volume** units; a strain is
dimensionless. ✅ **Ruled:**

$$\boldsymbol\varepsilon_{chem} = \frac{1}{3}\,\frac{\Delta V_{m,tot}}{V_{ref}}\,\mathbf I,\qquad V_{ref} = \text{bulk pore volume, a declared state variable}$$

$\Delta V_{m,tot}/V_{ref}$ is dimensionless $\Rightarrow \boldsymbol\varepsilon_{chem}$ is a strain. ✅

> ⚠️ **This is the same defect class as the Verma-Pruess guard (C-69):** a **reference quantity is
> missing**, so the ratio is unbounded. 📌 **Both should share one register entry** — *"missing reference
> denominator"* — because the failure mode is identical and the fix is the same shape.

### 7n.4 ✅ C-123 — CONF-29 arithmetic confirmed, no literature needed

For $N_c = 6$:

| Quantity | Value |
|---|---|
| Unique off-diagonal $i<j$ | $\tfrac{6\cdot5}{2} = 15$ |
| Diagonal self-interactions | 6 |
| **Total $k_{ij}$ entries, symmetric $6\times6$** | **21** |
| Design payload $2\times3$ | 6 |
| **Absent** | **15** ✅ matches the register |

✅ The register's figure is right. 🔴 The defect is the **payload**, not the count.

⚠️ **Worth widening:** $15$ counts *unique binary interactions*. If $k_{ii}$ are also needed for a
**mixing rule** (Lorentz–Berthelot or a similar combination), the requirement rises to **21** and the
shortfall is worse than recorded.

---

