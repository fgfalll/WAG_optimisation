# Python Compositional Attempt — Forensic Post-Mortem

**Supplied by the reservoir engineer, 08-10-2026.** This is the answer to a question I had flagged as
"the highest-value unknown" — the knowledge **existed but was not written down anywhere**.

> **Why it is preserved here.** The Python engine and its `EngineFactory` were built and **deleted**.
> Without this record, the *reasons* for deletion were lost with them, and the Rust attempt would repeat
> them. The previous wiki only recorded *that* the attempt was removed, not *why* it failed.

**Headline finding:** the failure was **not** Python's execution speed. It was **fundamental numerical and
thermodynamic formulation defects**. Those transfer to any implementation language, which is exactly why
this document must be read before M2.

---

## 1. Root causes

| # | Area | Failure |
|---|---|---|
| **1** | Thermodynamics / flash | Finite-difference derivative cancellation; trivial-root trap ($K_i\to1$); inverted $f_g$ |
| **2** | Hydrodynamic startup | Dirac-like impulse at $t=0^+$; non-$L$-stable integration; un-damped constraint switching; phase-appearance $C^1$ shock |
| **3** | Porous media / geochemistry | Kozeny–Carman requires $\phi\to0$ for $k\to0$; real clogging occurs at $\phi_c>0$ |

---

## 2. Thermodynamics and near-critical flash

### 2.1 Finite-difference derivative cancellation

Fugacity derivatives $\partial\ln\varphi_i/\partial x_j$ were computed by central/forward differences
with $\Delta x \approx 10^{-8}$. Near the critical point the Gibbs free-energy curvature approaches zero,
so $\partial G/\partial x \to 0$ — and **subtracting two nearly-identical `f64` values causes catastrophic
cancellation**. The resulting Jacobian was noisy, throwing Newton steps out of the physical domain
($x_i < 0$, or $V/F \notin [0,1]$).

**Rust fix:** analytical hyper-dual AD (`feos-ad` / `num-dual`) — derivatives to machine precision
($\sim10^{-16}$) with **zero numerical noise**. Already specified in D1 §4.4 and D3 §1.3.1.

> ✅ **This validates the hyper-dual requirement as load-bearing, not optional.** Without analytic
> derivatives the new engine fails near-critical for exactly the same reason. M2's Jacobian gate
> (analytic vs. central difference, rel. error `< 1e-6`) now has a **second purpose**: it demonstrates
> the cancellation problem is gone.

### 2.2 The trivial-solution trap ($K_i \to 1$)

Standard successive substitution, or un-damped Newton-Raphson, collapses to $x_i = y_i = z_i$ near
critical conditions. The solver oscillates between a single-phase liquid, a single-phase vapour, and a
non-physical two-phase split.

**Rust fix:** multi-seeded **Michelsen Tangent Plane Distance (TPD)** stability test combined with
**Heidemann–Khalil 2D critical solvers**.

> ⚠️ **NEW SPECIFICATION REQUIREMENT.** The design set specifies only Michelsen TPD + Rachford–Rice
> (D5 §4.2). **Heidemann–Khalil is not in any of the eight documents** — it is a negative-flash method for
> reliable two-phase root location. It must be added to the M2 specification. Tracked as **C-33**.

### 2.3 ⚠️ The $f_g$ claim is stated as fact here and it is **wrong** — must be corrected in this document

The post-mortem asserts:

> *"SCI-FLAW-01 Inverted Fractional Flow: $f_g$ formula predicted heavier oil **REDUCED** gas cut
> ($\partial f_g/\partial M < 0$)"*

**Measured 07-10-2026 — this is false, and was already retracted:**

| $\mu_g/\mu_o$ | $M=\lambda_g/\lambda_o$ | design $f_g$ | classical $M/(1+M)$ |
|---|---|---|---|
| 0.10 | 10.0 | 0.9091 | 0.9091 |
| 1.00 | 1.0 | 0.5000 | 0.5000 |
| 10.00 | 0.1 | 0.0909 | 0.0909 |

Identical to 4 dp. **The sign is correct.** The original error was substituting a viscosity ratio into a
slot labelled with a mobility ratio. Full arithmetic:
[`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) §2.

> 🔴 **Action and reason must be separated here.**
> - **The action is right:** purge S1 line 67, use $f_g = \frac{K S_g}{1+S_g(K-1)}$ or Corey phase
>   mobilities. ✅ Adopted.
> - **The stated reason is wrong.** If this document teaches the next implementer that the defect was a
>   sign error, someone will eventually "restore" the original formula believing the sign was the problem
>   — and the *real* defect (closure: no $S_{or}$, linear not Corey, no water term, unguarded at
>   $S_g\to S_{gc}$) will go unfixed.

The genuine defect is recorded at [`spec_defects.md`](spec_defects.md) §1 and **CONF-01**.

---

## 3. Hydrodynamic startup and impulse shocks

### 3.1 Dirac-like well impulse at $t=0^+$

Instantly applying full injection $q_{inj}$ at $t=0^+$ produced a spatial pressure shock,
$\partial P/\partial t \to \infty$. Using **$A$-stable but non-$L$-stable** integration — Crank–Nicolson
or IMPES — meant high-frequency spatial oscillations were **not damped**, propagating non-physical
pressure spikes through surrounding cells.

> ✅ **The engineer's stability classification is correct.** Crank–Nicolson is $A$-stable but **not
> $L$-stable**: its amplification factor tends to $-1$ on the imaginary axis, so undamped oscillation
> persists. Backward Euler and Radau IIA are genuinely $L$-stable ($|R(z)|\to0$ as $z\to-\infty$).

**Rust fix:** $L$-stable integration (Backward Euler / Radau IIA) with **exponential soft-start ramping**

$$q_{inj}(t) = q_{target}\left(1 - e^{-t/\tau_{well}}\right)$$

> ⚠️ **NEW SPECIFICATION DETAIL.** The soft-start *formula* and $\tau_{well}$ do not appear in any of the
> eight documents — the master map names "L-Stable Integration & Soft-Start" but gives no expression.
> Tracked as **C-34**.

### 3.2 Un-damped well constraint switching

When a well hit $P_{bhp} > P_{max}$, the Python scripts **flipped the control mode outside the primary
Newton iteration**. On step $k$ it switched to BHP control; $P_{bhp}$ dropped; on step $k+1$ it flipped
back to rate control. **Infinite ping-pong loop.**

**Rust fix:** fully implicit wellbore–reservoir coupling with **well equations embedded directly in the
global Jacobian**.

> ⚠️ **This is a new hard requirement, and it contradicts D4 §4.2.** D4 specifies SSSV logic as
> *external pseudocode* — `IF P_tubing < P_sssv_trigger … Set BoundaryCondition = ShutIn`. A mode switch
> outside the Jacobian is exactly the failure mode above. **D4's shutdown logic must be inside the
> nonlinear system, not a script branch.** Tracked as **C-35**.

### 3.3 Phase-appearance boundary shock

$S_g = 0 \to S_g > 0$ lacked $C^1$-continuity, causing Newton–Raphson to bounce endlessly across phase
boundaries.

**Fix:** primary-variable substitution with $C^1$ Jacobian continuity — already specified in D5 §7.2 and
in my M4 gate. ✅ Consistent with the design set.

---

## 4. Porous media and geochemical clogging

### 4.1 The Kozeny–Carman fallacy

Kozeny–Carman, $k \propto \phi^3/(1-\phi)^2$, reaches $k = 0$ only when **total** porosity reaches zero.
In reality suspended solids (TSS) or precipitated asphaltene/scale block the narrow **pore throats** long
before total pore volume is filled. With $k$ still non-zero at partial clogging, fluid kept entering,
pressure diverged, and the linear solver crashed.

**Rust fix:** **Verma–Pruess percolation cutoff**

$$k = k_0\left(\frac{\phi - \phi_c}{\phi_0 - \phi_c}\right)^{n}$$

driving $k \to 0$ strictly when accessible porosity hits the throat cutoff $\phi_c > 0$, combined with
adaptive sub-stepping.

> ✅ **This closes a real open question.** D1 §5.2 presents Kozeny–Carman **and** Verma–Pruess with **no
> selection rule** (my earlier note). The post-mortem supplies the rule:
> **Verma–Pruess for clogging/percolation-limited permeability collapse; Kozeny–Carman for
> compaction-driven porosity change**, which is a different mechanism. Tracked as **C-36**.

---

## 5. What transfers, and what does not

| Finding | Transfers to Rust? |
|---|---|
| Finite-difference derivative cancellation near critical | ✅ **Yes** — language-independent. Requires analytic AD |
| Trivial-root trap $K_i\to1$ | ✅ **Yes** — requires Heidemann–Khalil |
| Non-$L$-stable startup integration | ✅ **Yes** — requires $L$-stable + soft-start |
| Un-damped constraint switching outside the Jacobian | ✅ **Yes** — requires well equations in the Jacobian |
| Phase-appearance $C^1$ discontinuity | ✅ **Yes** — requires primary-variable substitution |
| Kozeny–Carman $\phi\to0$ requirement | ✅ **Yes** — requires Verma–Pruess for clogging |
| The $f_g$ "sign error" | ❌ **No** — the sign was never wrong. Closure was |
| Python execution speed | ❌ **No** — not the cause |

> **The decisive conclusion.** Six of seven root causes are **numerical-method failures, not language
> failures**. Rust's contribution is memory safety, zero-cost abstractions and SIMD — **not** correctness.
> Correctness comes from the methods in the right-hand column. Writing this down is the difference between
> the second attempt and a first attempt.

---

## 6. New specification requirements arising from this post-mortem

| ID | Requirement | Milestone |
|---|---|---|
| **C-33** | **Heidemann–Khalil 2D critical solver** — absent from all eight documents | M2 |
| **C-34** | **Exponential soft-start** $q_{inj}(t)=q_{target}(1-e^{-t/\tau_{well}})$ with a stated $\tau_{well}$ | M3 |
| **C-35** | **Well control modes embedded in the global Jacobian** — D4 §4.2's external pseudocode is the failure mode | M5 |
| **C-36** | **Permeability-collapse selection rule**: Verma–Pruess for clogging, Kozeny–Carman for compaction | M3 |
| **C-37** | ⚠️ **Correct the $f_g$ reason** in this post-mortem — action right, stated reason wrong | now |