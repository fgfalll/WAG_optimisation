# Conflict & Gap Register — `3D_THMC_docs` vs. the live codebase

This is the **most important page in the section.** It records every place where the 8 THMC design
documents contradict the live Python implementation, contradict the existing finding register, or
contradict **each other**.

> [!CAUTION]
> **This is NOT `audit/scientific_flaws.md`.** It is a wiki-side triage of the *documents*, not a
> register of defects in shipped code. Nothing here is a registered finding.
>
> **If any entry below turns out to describe a live code defect**, it must be registered via
> `python -m audit --register new <ID> …` and validated with `python -m audit --register validate`.
> Never hand-write into `audit/scientific_flaws.md`. See
> [`../development/finding_registry.md`](../development/finding_registry.md).

**Severity key:** 🔴 **BLOCKING** — must be resolved before any adoption decision ·
🟠 **MATERIAL** — changes the numbers or the architecture materially ·
🟡 **DOCUMENTARY** — must be corrected in the documents, does not change the model.

> [!NOTE]
> **Ruling history.** A reservoir engineer reviewed this register on **07-10-2026**. Two rulings
> accepted, one accepted with corrections, one **rejected as itself in error** — which also revealed
> that **CONF-01 as originally written was wrong**. Read
> [`reservoir_engineer_ruling.md`](reservoir_engineer_ruling.md) before quoting any conflict here.

---

## 1. 🔴 BLOCKING conflicts

### CONF-01 — RETRACTED 07-10-2026, then REPLACED: the defect is the *closure*, not the sign

> [!CAUTION]
> **The original CONF-01 was WRONG and is withdrawn.** It claimed the design set's gas fractional flow
> has $\partial f_g/\partial M < 0$ and that it "re-proposes the inverted fractional flow rejected by
> SCI-FLAW-01". **Measurement disproves this.** See §1.1 for the arithmetic. A reservoir engineer
> reviewed CONF-01 and **endorsed it while making the identical substitution error**, so the
> correction below supersedes both.

#### 1.1 The withdrawal, with the arithmetic

The design formula (D1 §6.1, D4 §3.3, D5 §1.1, D6 §4.1, D6 §4.5, D7 §4.4, D8 §2.4 — **seven documents**):

$$f_g(t) = \frac{1}{1 + \left(\frac{1 - S_w - S_g}{S_g - S_{gc}}\right)\cdot\frac{\mu_g}{\mu_o}}$$

Classical two-phase gas fractional flow:

$$f_g = \frac{\lambda_g}{\lambda_g+\lambda_o} = \frac{1}{1+\dfrac{k_{ro}}{k_{rg}}\,\dfrac{\mu_g}{\mu_o}} = \frac{M}{1+M},\qquad M \equiv \frac{\lambda_g}{\lambda_o}$$

**These have the same structure and the same sign.** $f_g$ is increasing in $\mu_g/\mu_o$, equivalently
increasing in $M$, and $f_g \to 1$ as $M \to \infty$ — gas dominates the stream when gas is the more
mobile phase. Measured, design formula vs. classical, sweeping the viscosity ratio:

| $\mu_g/\mu_o$ | $M=\lambda_g/\lambda_o$ | design $f_g$ | classical $M/(1+M)$ |
|---|---|---|---|
| 0.10 | 10.0 | 0.9091 | 0.9091 |
| 0.50 | 2.0 | 0.6667 | 0.6667 |
| 1.00 | 1.0 | 0.5000 | 0.5000 |
| 2.00 | 0.5 | 0.3333 | 0.3333 |
| 10.00 | 0.1 | 0.0909 | 0.0909 |

**Identical to 4 dp at every point.** The two formulas agree wherever the saturation ratio happens to
equal 1 (which is what the original test point $S_w{=}0.2, S_g{=}0.5, S_{gc}{=}0.2$ forces).

**The error was mine:** I wrote "$M=10 \Rightarrow f_g = 0.0909$" while substituting
$\mu_g/\mu_o = 10$. Under the mobility-ratio convention, $M=10$ means $\mu_g/\mu_o = 0.1$, which gives
$f_g = 0.909$ — **rising, not falling**. I labelled a viscosity ratio as a mobility ratio.

**Consequence for the finding register:** SCI-FLAW-01 is **not** re-introduced by the design set. The
shipped form (`core/engine_surrogate/analytical_models.py:176-193`) and the design form **agree on sign**.
Nothing here reopens SCI-FLAW-01.

#### 1.2 The real defect: an ad-hoc saturation ratio standing in for relative permeability

The genuine problem is that $\left(\dfrac{1-S_w-S_g}{S_g-S_{gc}}\right) = \dfrac{S_o}{S_g-S_{gc}}$
is being used as a **proxy for $k_{ro}/k_{rg}$**, and it is not one.

| # | Defect | Consequence |
|---|---|---|
| a | **$S_{or}$ is absent from the numerator.** $k_{ro} = 0$ at $S_o = S_{or}$; here $f_g = 0.800$ at $S_o = S_{or}=0.20$ (measured) and only reaches 1.0 at $S_o = 0$ | The model cannot represent the state where all mobile oil is gone. **$S_{or}$ is not a small correction — it sets the terminal gas cut.** |
| b | **Linear, not Corey/Brooks-Corey.** $k_{ro}/k_{rg} \propto \dfrac{(S_o-S_{or})^{n_o}}{(S_g-S_{gc})^{n_g}}$, i.e. **different exponents on different saturations** | Wrong curvature, wrong transition width, wrong sensitivity to endpoint saturations |
| c | **No water term.** Three-phase $f_g = \lambda_g/(\lambda_g+\lambda_o+\lambda_w)$ | Directly contradicts **D1 §6.1's own claim** that Stone I/II three-phase relative permeability is used, and interacts with live **CRIT-17** ($\sum S \neq 1$ on 27 % of steps) |
| d | **Singularity at $S_g \to S_{gc}$.** $1/k_{rg} \propto (S_g-S_{gc})^{-n_g}$; the design uses $1/(S_g-S_{gc})$ | Right *direction* ($f_g\to1$, since no gas mobility ⇒ all flow is gas) but wrong *rate* of divergence, and **no numerical guard** in any of the 7 occurrences |

> **Correction to the reservoir engineer's input:** they also claimed the linear form "cannot reach the
> observed post-breakthrough gas cuts (0.60–0.85)". **This is false.** Measured: $S_o{=}0.10$,
> $S_g{=}0.70$, $S_{gc}{=}0.10$, $\mu_g/\mu_o{=}0.88$ gives $f_g = 0.872$; $S_o{=}0.20$,
> $S_g{=}0.70$, $S_{gc}{=}0.10$, $\mu_g/\mu_o{=}1.00$ gives $f_g = 0.750$. The range **is** reachable.
> The defect is the closure's *shape*, not its range.

**Engineer's proposed replacement** (accepted in principle, caveated):

$$f_g = \frac{1}{1 + \left(\frac{1-S_w-S_g-S_{or}}{S_g-S_{gc}}\right)^{n_o/n_g}\cdot\frac{\mu_g}{\mu_o}}$$

> ⚠️ **Caveat:** this applies a *single* combined exponent $n_o/n_g$ to a *ratio*, whereas Corey applies
> **separate** exponents to **separate** saturations. It is a defensible engineering approximation and
> it fixes (a), but it is **not** Corey and must be documented as an approximation, not as a
> "Corey-based" formulation. It also still omits the water term and the $S_g\to S_{gc}$ guard.

**Action required:** replace the proxy with a declared rel-perm closure (Corey/Brooks-Corey or the
shipped Koval $f_g = \tfrac{KS}{1+S(K-1)}$), add $S_{or}$, add the water phase, add the guard, at all
7 sites. **Severity downgraded 🔴 → 🟠.** The formula is not *wrong-signed*; it is a crude closure.

---

### CONF-02 — The proposed recovery cap is inert at the shipped default HCPVI

**Design claim** (D1 §6.1, D2 §6.1, D4 §3.3, D6 §4.1, D7 §4.4, D8 §2.4 — **six documents**):

$$RF(t) \le RF_{ult}(\bar{P}_{res,eff})\left(1 - e^{-\text{HCPVI}/\tau}\right),\qquad \tau = 1.5$$

**Measurement at the repository default HCPVI = 7.69:**

$$1 - e^{-7.69/1.5} = 1 - e^{-5.127} = \mathbf{0.9939}$$

The cap removes **0.61 %** of $RF_{ult}$ at the default operating point. It is **inactive exactly where
the optimiser operates**, and only bites at HCPVI $\lesssim 0.05$.

**Finding-register conflict.** Structurally identical to **CRIT-15** (`CONFIRMED`): the default HCPVI
of 7.69 pins the Koval sweep at its `0.95` clip **for every mobility ratio**. The THMC cap replaces one
HCPVI-saturated clip with another HCPVI-saturated clip.

**Secondary problems:**
- `τ = 1.5` has **no units, no provenance and no sensitivity study** in any of the 8 documents.
- D2 §6.2 criterion 5 requires "at **0.18 HCPVI**, tertiary RF strictly limited to **≤ 15 % OOIP**",
  while D6 §4.1 states the *failure signature* as "**> 41.8 % OOIP** at 0.18 HCPVI". With $\tau=1.5$,
  $1-e^{-0.12} = 0.113$; if $RF_{ult}\approx 0.55$ the cap yields ≈ 6.2 % OOIP — inside the ≤ 15 %
  bound but **not equal to it**, so the two stated numbers are not reconciled by the formula.
- The cap is dimensionally a *saturation* function of HCPVI only. It adds **no** dependence on mobility
  ratio, $k_v/k_h$, heterogeneity or pattern geometry — i.e. it cannot fix the mechanism it is aimed at.

**Action required:** the cap must be re-derived with an auditable provenance and calibrated against
D2 §6.2 criterion 5 and the live HCPVI range, or dropped.

---

### CONF-03 — The documents report remediation of the Python engine that has not happened

**Design claim** (D1 §10.3, D2 §6.1, D4, D6 §1.2, D8 §2.4 and §4.2): artificial penalty barriers
`result *= breakthrough_impact` and `-1.0×10^{12}` are **"fully removed"** from the simulator code and
replaced by natural NPV reduction through gas-recycle cost.

**Live-code status:**

| Claim | Live state |
|---|---|
| Penalties removed | ❌ `FAILURE_PENALTY = -10^{12}` is a **documented invariant** (`agent_wiki/README.md` invariant 12) |
| Objective has no penalty wrapper | ❌ `core/objectives/wrapper.py:_calculate_objective_functions()` is documented as a **consumer that applies containment/remediation penalties** to `profiles["npv"]` |
| NPV computed in one place | ❌ NPV computed inline in `surrogate_engine.py:622-651`, with a **shadowed** older inline NPV still at `surrogate_models.py:507-530` |
| `test_surrogate_engine.py` passes 48 tests | ❌ **`Test-Path` → `False`. The file does not exist.** |
| Cash flows not averaged | ✅ satisfied: `surrogate_engine.py:646-651` sums per-timestep annual cash flows, no `N_p/15` |
| Three artefacts agree to $0.01 | ⚠️ unmeasured — must be run via `tests/test_physical_invariants.py` (which references `cash_flows_yearly.csv` at lines 8, 159, 181, 184, 211) |

**Action required:** reclassify every such statement in D1–D8 from *accomplished* to *requirement*.
This is the single most important correction: an agent reading D1 §10.3 or D8 §4.2 today would
conclude the penalty defects are closed.

> ⚠️ **Caveats on the engineer's CONF-03 remediation.** They propose (a) "replace step-function
> penalties with continuous barrier/penalty functions", and (b) "author and execute
> `tests/test_surrogate_engine.py` to validate all 48 surrogate test paths". Both need narrowing:
>
> - **CONF-60 (new) — a smooth barrier conflicts with the adjoint-engine design.** D1 §10.2's entire
>   premise is $\nabla_u J$ via the adjoint method, which **requires $J$ to be smooth in $u$**. A
>   barrier/barrier-squared penalty on constraint violation is $C^1$ but its gradient blows up at the
>   constraint boundary, degrading the adjoint solve exactly where the optimiser needs resolution.
>   Worse, it blurs a distinction the live engine gets right: **non-convergence is a hard failure with
>   no meaningful value** (a step penalty of $-10^{12}$ is correct there), whereas **physical
>   constraint violation lives on a bounded region** (a smooth barrier is better there, because it
>   gives the optimiser a usable gradient). **Split the invariant**, do not merge it: keep the step
>   penalty for solver failure, use barriers only for bounded-physics constraints.
> - **CONF-61 (new) — "48 tests in `test_surrogate_engine.py`" is not a spec.** The number 48 appears in
>   D2 §6.2 criterion 3, a checklist box for a **Rust** core that does not exist, and it names a Python
>   file that does not exist. Manufacturing a Python test file to reach the count 48 is cargo-culting a
>   target from a phantom document. The correct derivation is **one regression test per registered
>   finding** — which is a different and non-arbitrary number, and it is auditable.
> - **CONF-62 (new) — mid-year vs. end-of-year discounting is an unreconciled live-vs-design gap.**
>   All design documents mandate mid-year discounting $d_t = (1+r)^{-(t-0.5)}$ (D1 §11.3, D6 §6.4,
>   D7 §4.5, D8 §4.2, with the rationale that production occurs continuously). The **live engine uses
>   end-of-year**: `core/engine_surrogate/surrogate_engine.py:649-650`
>   `discount_factors = 1.0 / ((1.0 + discount_rate) ** years_arr)` with `years_arr = arange(1, n_years+1)`.
>   At $r=0.10$ over 15 yr with level annual cash flow the two conventions differ by **+4.88 % in NPV**
>   (measured 07-10-2026: mid-year sum 7.9773 vs end-year 7.6061). Neither the live wiki nor
>   `phd_audit.md` records this. It must be decided deliberately — the mid-year convention is the
>   better argument, but switching it **re-baselines every published result**.

---

### CONF-04 — Economics is simultaneously off-core and in-core — **ADJUDICATED 07-10-2026**

- **D1 §1.1**: NPV/CAPEX economic evaluation is **exiled** to the satellite suite (duty 3).
- **D1 §10.2**: the **adjoint gradient of the NPV objective** is computed **inside the simulator core**.
- **D2 §6.1**: economics is listed as an in-core module (`economic_npv_usd`, `cash_flows_yearly.csv`).

**Ruling (reservoir engineer, adopted): the engineer is correct and this supersedes the
recommendation previously recorded here.**

| Layer | Owns |
|---|---|
| **Simulation core (PDE solver)** | Conservation equations; physical state $P(x,t)$, $S_\alpha(x,t)$, $z_i(x,t)$, $q_o(t), q_g(t), q_w(t)$; **production adjoints** $\partial q_i(t)/\partial\mathbf{x}$ |
| **Satellite / economic wrapper** | Economic vector $(P_o, P_g, C_w, \text{CAPEX}, r)$; $NPV = \int[q_oP_o + q_gP_g - q_wC_w]e^{-rt}dt - \text{CAPEX}$; $\partial NPV/\partial\mathbf{x} = \int\left[(\partial q_o/\partial\mathbf{x})P_o + (\partial q_g/\partial\mathbf{x})P_g\right]e^{-rt}dt$ |

Putting prices ($P_o$, $P_g$, $r$) inside the Jacobian solver violates separation of concerns and
couples unit-testable conservation physics to economic assumptions. **This is the right boundary**, and
it is the *only* one of the four options that makes D1 §1.1 and D1 §10.2 simultaneously true.

> ⚠️ **Two corrections to the engineer's ruling — adopting it verbatim would REGRESS the economics.**
>
> **CONF-58 (new) — the ruling's NPV expression drops the terms that dominate a CO₂-EOR project.**
> The engineer's $NPV = \int[q_oP_o + q_gP_g - q_wC_w]e^{-rt}dt - \text{CAPEX}$ omits
> **CO₂ purchase cost**, **CO₂ recycle cost**, **storage credit**, and **carbon tax on leakage**.
> The live engine already has all four at `core/engine_surrogate/surrogate_engine.py:624-646`:
> `co2_purch_cost`, `co2_recyc_cost`, `co2_storage_credit`, and `carbon_tax` applied to
> `annual_caprock_leakage_tonne + annual_fault_leakage_tonne`. In a CO₂-EOR project the CO₂ purchase
> term is typically a dominant cost; dropping it makes the ruling's expression **less** complete than
> the code it would replace. The correct ruling is *"prices off-core"*, not *"fewer prices off-core"*.
>
> **CONF-59 (new) — applying $P_g$ to the whole gas stream would sell recycled CO₂.**
> The engineer's $q_gP_g$ is unqualified. In this project produced gas splits into **sales gas**
> (`hc_gas_rate`) and **CO₂** (`co2_prod_rate`, largely recycled). The live engine already keeps these
> separate at `surrogate_engine.py:531`. The satellite must receive them as **separate streams**, or
> recycle gas gets booked at gas price. Note this is adjacent to **CONF-11**: the live engine has the
> opposite bug — it omits gas revenue entirely (**CRIT-18**). The fix must satisfy **both**: add
> $q_{g,\text{sales}}P_g$, exclude $q_{\text{CO2,prod}}$ from sales, and keep CO₂ purchase/recycle cost.

**Consequence for the live engine:** `surrogate_engine.py:622-651` computes NPV **inline inside the core**,
which violates the ruling. Under the ruling, the economic terms at `:624-634` and the discounting at
`:650` move to a satellite module; the core publishes $q_o, q_g, q_{w,\text{inj}}, q_{w,\text{prod}}$
and, if H3 (adjoints) is adopted, $\partial q_i/\partial\mathbf{x}$.

**Action required:** adopt the ruling. Record as an ADR. Do **not** adopt the ruling's NPV expression
verbatim — carry the live engine's CO₂ cost/credit/carbon-tax terms and split the gas stream.

---

### CONF-05 — Pinch-out handling is undecided

D1 §2.2 states pinch-out cells are "**deactivated or** algebraically collapsed with neighbours without
removal from the global vectors" — a disjunction, not a decision. D1 §2.2 also gives the threshold
$V_{cell}\le 10^{-12}\,\text{m}^3$ but not the resulting transmissibility treatment in either branch.

**Severity:** 🟠 MATERIAL — the two branches give different transmissibility matrices, hence different
recovery.

---

### CONF-06 — The GPU flash benchmark is unanchored

D1 §4.3 claims **1 011.9×** (wgpu) and **1 370.9×** (CUDA) speed-up over single-core CPU for a
$10^6$-cell, 8-component flash, 4 250 ms → 4.2 / 3.1 ms. Arithmetic is internally consistent.

Missing: **any hardware identification** — no CPU model, no GPU model, no core count, no interconnect,
no precision caveat, and **no statement of whether flash tolerances differ between the CPU and GPU
paths**. This matters: a GPU flash at $10^{-16}$ vs. a CPU flash at $10^{-12}$ will not agree to the
Level-2 mass-balance tolerance.

**Severity:** 🟠 MATERIAL — the 1371× figure cannot be used for planning without hardware.

---

## 2. 🟠 MATERIAL conflicts

### ✅ CONF-07 — CLOSED 08-10-2026: the 1 000 BOPD clamp must not exist

~~D1 §9.1, D6 §3.6 stage 4, D7 §4.3, D8 §3.3 all impose $q_{o,well}\le 1\,000$ BOPD "hardly limited by
Darcy and Vogel". Not stated: whether the clamp is applied before or after drift-flux; how a clamped
rate propagates to mass balance; whether the clamp is reported or silently truncates.~~

**Resolved under INV-7** ([`../compositional/engine_invariants.md`](../compositional/engine_invariants.md)
§7b). The owner requires the engine to be **unconstrained by construction** — absurd input must still
receive full physical evaluation. A hard clamp is therefore not merely misplaced, it must be **removed**.

**Measured 08-10-2026 — the clamp is not a physics limit.** Steady-state vertical-well Darcy throughput at
the design set's own 40-acre pattern ($r_e = 227$ m, $\ln(r_e/r_w) = 7.65$; $\Delta P = 50$ MPa, $\mu=1$ cP):

| $k$, $h$ | $J$ (Darcy limit) | vs. the clamp |
|---|---|---|
| 50 mD, 10 m | 11 011 STB/d | **11×** above |
| 200 mD, 30 m | 132 127 STB/d | **132×** above |
| 1000 mD, 50 m | 1 101 055 STB/d | **1101×** above |

The clamp binds **11× to 1101× below** the Darcy limit of the stated pattern. It is a round-number cap.
**CONF-54** already recorded D5 naming `20 000–60 000 MSCFD` the *anti-pattern signature of a clamped rate*.

**Corrections: C-56 (this) · C-57** (`locked_*` / `min_*`) · **C-58** (pressure limits modelled, not
clipped) · **C-59** (three-tier `ValidityClass`). **Supersedes C-22**, whose question — *where does the
clamp sit?* — is moot.

> ⚠️ **Related, and still relevant:** **CONF-15** (`$t_{bt}$) remains a genuine spec defect, independent of
> the clamp. It ignores $S_{or}$ and gas saturation and has no unit basis for $q_{inj,pattern}$ or
> $K_{koval}$. The **failure-penalty convention** in the live Python engine is removed under
> **CONF-37**, so only CONF-15 survives here.

---

### CONF-08 — `δt` denotes two different sub-grids

D1 §2.3 Schwarz sub-domain micro-step and D1 §9.3 wellbore sub-step $\delta t_k$ use the same symbol
with no stated coupling or tolerance for the two-way interaction.

---

### CONF-09 — The GEP chromosome example contradicts its own tail-length formula

D1 §11.2 example: $h = 7$, $t = 9$. Formula $t = h(n-1)+1$ ⇒ $n = 7(9-1)+1 \ldots$ ⇒ solving
$9 = 7(n-1)+1$ gives $n = 2.143$ — **not an integer function arity**.

---

### CONF-10 — `Q_recycle,peak` normaliser has no declared units

D1 §11.3's CAPEX formula uses `Q_recycle,peak / 20,000` with the normaliser's unit never stated.
D6 §6.3 **does** declare MSCFD. Inconsistency between the two statements of the same constant.

---

### CONF-11 — NOCF omits hydrocarbon-gas revenue (structurally CRIT-18)

D1 §11.3, D6 §6.4, D7 §4.5, D8 §4.2 all define:

$$\text{NOCF}_t = \text{Rev}_{oil} + \text{Credit}_{storage} - \text{Cost}_{purch} - \text{Cost}_{recycled} - \text{OPEX}$$

**No hydrocarbon-gas revenue line.** Live code at `core/engine_surrogate/surrogate_engine.py:636`:

```python
annual_rev = (annual_oil_stb * oil_price) + (annual_stored_tonne * co2_storage_credit)
```

with `annual_hc_gas_mscf` accumulated at `:588-606` and published at `:688-689` but **never referenced in
revenue**. This is **CRIT-18** exactly — 216 810 MSCF over 15 yr contributing $0.

**The design document set inherits the defect.**

---

### CONF-12 — `ConvergenceControl` has no defaults and two unused fields

`max_newton_iterations`, `tolerance_residual`, `line_search_max_steps`, `damping_factor_min` — **no
numeric default appears anywhere in D1–D8**. In the only listing, `line_search_max_steps` and
`damping_factor_min` are **never read**.

---

### CONF-13 — The Newton listing is not a solver

`step_time` as printed: `lambda` is multiplied by 0.85 and **never applied** (there is no `Δx` in the
function); `compute_residuals_and_jacobian` computes **no Jacobian** and **fills no residuals** —
it is a finiteness scan returning `Ok(())`; there is **no linear solver**; `residuals` is written by the
caller and never written by the solver, so `norm < tolerance_residual` can never legitimately succeed.

Against the document's own prose: backtracking line search with Armijo–Goldstein condition — not present;
SIMD auto-vectorisation AVX-512 / FMA3 — listing is scalar `.map().sum().sqrt()`.

**Severity:** 🟠 — any agent treating this listing as a reference implementation will reproduce a
non-solver.

---

### CONF-14 — The linear-algebra conditioning layer is entirely unspecified

No preconditioner (no ILU/ILUT, no AMG, no block preconditioning), no sparse storage format (no
CSR/CSC/CSF/ELL, no fill-reducing ordering), no symbolic/numeric phase separation, and **no Krylov /
iterative method at all** — only direct solvers (`faer`, MUMPS, UMFPACK, `cuDSS`).

This directly contradicts D5 §3.1, which describes the target as "Newton-Raphson **+ Krylov solver**".

---

### CONF-15 — `$t_{bt}$` uses `$1 - S_{wi}$` as the mobile hydrocarbon fraction

D1 §11.x, D2 §4.1, D6 §4.1, D7 §4.3:

$$t_{bt} = \frac{V_{p,pattern}(1-S_{wi})}{K_{koval}\,q_{inj,pattern}}$$

Ignores $S_{or}$ and any gas saturation. No unit system, condition basis (reservoir vs surface volume
for $q_{inj}$), or $K_{koval}$ value is given.

---

### CONF-16 — Chemical strain is unnormalised

D2 §3.1: $\boldsymbol{\varepsilon}_{chem} = \tfrac{1}{3}\Delta V_{m,tot}\mathbf{I}$. A strain is
dimensionless; $\Delta V_{m,tot}$ carries volume units. Requires division by a reference or bulk volume.
**Dimensional error, not a modelling choice.**

---

### CONF-17 — `P̄_res,eff` coupling contradicts the fully-coupled Biot formulation

D2 §3.1 specifies a fully-coupled Biot poroelasticity solve with cell-local pressure. D2 §6.1 requires
RF and miscibility $\omega$ be recomputed **exclusively** via the single production-weighted scalar
$\bar{P}_{res,eff}$. These are different models. Same document.

---

### CONF-18 — Porosity updated by bare volume subtraction

D2 §2.2: $\phi^{t+\Delta t} = \phi^t - \sum_m \Delta V_m$. Porosity is a **ratio of volumes**; subtracting
a volume requires an implicit normalisation. **No solid-volume (Bethel) correction term is given.**

---

### CONF-19 — "Remove the artificial penalties" is written as done, not required

See **CONF-03**. Kept as a separate entry because it recurs in **six** documents and is stated in the
past tense in all of them.

---

### CONF-20 — CAPEX band is contradicted by the worked example

| Statement | Value |
|---|---|
| D6 §6.3: "1 000-acre field (25 patterns) scales to a realistic **$40M–$75M**" | band |
| D2 §6.2 criterion 4: total CAPEX within **$40M – $75M** | band |
| D6 `Economic_DCF_DTO` example, `total_capex_year0` | **$90,950,000** |

The worked example for exactly the stated project size exceeds its own band by **$15.95M**.
Line items do sum correctly ($62.5 + 9.25 + 14.2 + 5.0 = 90.95$ M), and the compressor term
$14.2 - 5 = 9.2$ M correctly implies $Q_{recycle,peak}\approx 17\,600$ MSCFD from
$10\text{M}\times(Q/20\,000)^{0.65}$.

---

### CONF-21 — Coverage rejected as a gate, no replacement proposed

D5 §1.1 rejects 95–100 % coverage as "a false criterion creating an illusion of correctness" and then
proposes **no quantitative replacement**. The CI has time budgets and physics tolerances but no
coverage, mutation-score, or property-based-testing requirement.

For comparison, the live repository runs `pytest` and `ruff --select F821` as release gates and has
**335 passing tests while 20+ findings are live** (HIGH-18). D5's critique of coverage is correct but
its conclusion is worse than what is already failing.

---

### CONF-22 — D5's benchmark tolerance table has shifted columns

The Level-1 summary table's 4th column ("за довжиною" / over length) belongs to the **Buckley–Leverett**
tolerance, not to a fourth metric column. Transcribing the table verbatim yields wrong tolerances.

---

### CONF-23 — Three documents disagree about which mass-balance tolerance applies to which tier

| Source | Claim |
|---|---|
| **D5 §3.1** | `< 10^{-12}` for the full 3D FVM nonlinear solver; **`≥ 99.9 %` is "categorically inadmissible inside the main numerical core"** |
| **D2 §6.2 criterion 1** | convergence **`> 99.9 %`** |
| **D6 §4.3** | `≤ 0.001·M_initial`, i.e. **99.9 %** |
| **Live repo** | `agent_wiki/README.md` invariant 14; `utils/run_exporter.py` evaluate **> 99.9 %** |

**The live engine is a surrogate, i.e. D5's "coarse high-level surrogate diagnostic model" tier**, for
which D5 mandates the `10^{-3}`-class tolerance — but D2 §6.2 and D6 apply that same tolerance to what
they describe as the numerical core. The tier boundary is never defined by interface or component.

> ⚠️ **Correction to the reservoir engineer's input.** They audited this as *"conflicting tolerances
> `10^-12` vs `10^-14` vs `10^-6`"* and proposed *"strict FVM `10^-12`, surrogate `10^-6`"*.
> Measured against the documents (pattern search across all 8 files, verified 07-10-2026):
>
> | Value | Where it actually occurs | What it governs |
> |---|---|---|
> | `10^-12` | **D5 §3.1 only** (6 lines) | component-wise mass balance, full 3D FVM core |
> | `10^-14` | **D5 §5.3 only** (1 line) | **streamline↔FVM remap** mass preservation — a *different quantity*, not a mass-balance tier |
> | `10^-3` / `99.9 %` | **D2 §6.2, D6 §4.3** | material balance, "the core" as they describe it |
> | `10^-6` | **appears nowhere in the set** — 0 hits | **unsourced** |
>
> The real conflict is **two** tiers (`10^-12` vs `10^-3`) applied to the **same object**, plus an
> unrelated remap tolerance. Their `10^-6` proposal is a **third** number with no source, and it would
> silently *tighten* the live `10^-3` policy by three orders of magnitude without evidence.
>
> **Action required:** keep the engineer's *intent* (one tolerance per tier) but use only sourced
> numbers — `10^-12` for an exact-conservative FVM core, and a **justified** figure for the surrogate
> tier derived from the surrogate's own arithmetic (`evaluation_plan.md` §3.1), not from a round number.
> Define "the core" by interface or component, since that is what makes the two tiers distinguishable.

---

### CONF-24 — Clamping saturations is forbidden *and* required

D5 §3.2: clamping $S_\alpha$ to $[0,1]$ after the linear solve is **"categorically forbidden"**;
membership must be guaranteed by the discretisation and an exact Jacobian.

But the same document requires line-search backtracking (D2 §4.1), adaptive step cuts (D2 §1.1a), and
primary-variable substitution on phase appearance (D5 §7.2). None of those mechanisms can preserve
$[0,1]$ membership unconditionally. Either the discretisation must be proven positivity-preserving, or
bounds handling must be specified. Not addressed.

---

### CONF-25 — D5's SIMD TPD listing claims vectorisation it does not contain

`evaluate_tpd_simd` claims `std::simd` with **AVX-512** and zero heap allocation; the shown loop is
scalar. `Kelvin`, `NumericalDivergenceError`, `fugacity_coeff_gas` and `fugacity_coeff_mixture` are
referenced and **never declared**; `NumericalDivergenceError` has one variant named and no `enum`
definition anywhere in D5.

---

### CONF-26 — Spatial convergence is promised and never tested

D5 §1.2 and line 79 promise adherence to declared **"temporal and spatial"** convergence order. Level 4
is **temporal only** ($\mathcal{O}(\Delta t)$ throughout). There is **no** spatial grid-refinement
study, **no** order-of-accuracy target in $h$, and **no** truncation-error quantification.

MPFA-O + WENO + Taylor–Hood spatial verification is therefore **entirely absent** — despite Level 3
requiring M-matrix, symmetry and positive-definiteness properties of the MPFA-O operator that only a
spatial test would establish.

---

### CONF-27 — Mid-simulation IPC contradicts the Zero-Bloat Core

D6 §1.2 mode 2 invokes post-simulation modules through the solver's internal **IPC / gRPC interface
after each integration stage**. D1 §1.1 declares "calling auxiliary services inside the solver's hot
loops" to be **the primary cause of cache thrashing and memory bloat**. D1 §10.1 requires **zero heap
allocation** in the Newton loop. These cannot all hold.

---

### CONF-28 — The DTO bus contract is not implementable

No `.proto` file, no field numbers, no service/method names, no versioning or negotiation scheme, no
error/status codes, **no checkpoint or restart format** ("checkpointing" and "restart" appear nowhere
in D6).

---

### CONF-29 — `bip_matrix` is dimensionally wrong in `PVT_EOS_DTO`

Declared **2 × 3** for a **6-component** system. A valid symmetric 6-component BIP set requires **21**
unique $k_{ij}$; **15 are absent**, and the trailing entry has no symmetric counterpart. Any PR/SRK
mixture built from this DTO is **underdetermined**.

---

### CONF-30 — `Economic_DCF_DTO` contains a DCF arithmetic error

Year 2 (measured 07-10-2026): $28.5\times10^6 / 1.1^{1.5} = 24\,703\,349$; stated $24\,698\,500$ — a
**$4\,849 (0.020 %)** gap, which is **~485× the document's own required $0.01** cross-artefact
agreement (D6 §1.2, D8 §4.2).

Years 1 and 3 match to rounding; the `cum_npv` ladder is exactly consistent with the *stated* DCFs, so
the error is in `dcf_usd[2]`, not the ladder.

---

### CONF-30b — `petekIO` is named three times and defined zero times

No magic bytes, header struct, byte order, slab typing/dtype, chunk/offset table, alignment contract,
schema or format version, or extension. D6's own pipeline
(`[petekIO / GRDECL / RESCUE] → Unified Parser → FVM Input Grid DTO`) cannot be implemented.

---

### CONF-31 — No coefficient is ever given for any named correlation

**Standing**, **Glaso**, **Beggs–Robinson**, **Joback-Reid**, **PPR78**, **Huron–Vidal**, **QSPR**,
**Karakas–Tarik** are all named without a single coefficient or equation. `(M\omega)^{-1}` appears in
the VT shift with $M$ and $\omega$ undefined as symbols.

---

### CONF-32 — "Validation" silently mutates the input

D6 §3.2 stage 2 **C¹-smooths** rel-perm and capillary curves; stage 4 **auto-inserts patterns and well
counts** to enforce $N_{inj}=N_{prod}=N_{pat}$. A "validated" DTO may therefore not be the DTO the user
supplied, and no diff or audit trail is specified. This is remediation presented as validation — the same
shape as the live **CRIT-19 `gravity_factor` ±20 % fudge** defect.

---

### CONF-33 — SEPD is adopted for the property it was chosen to avoid

D6 §4.2 rejects "unjustified **late-life rate inflation**" and then adopts **SEPD**,
$q(t)=q_i e^{-(t/\tau)^n}$, whose defining property *is* slower-than-exponential late-time decay. The
stated rationale contradicts the stated model. This is the exact shape of **CRIT-19**.

---

### CONF-35 — The Net CO₂ Utilisation floor is stated four ways

| Source | Floor | Benchmark |
|---|---|---|
| D1 §6.3 | **≥ 2.5 MSCF/STB = 0.25–0.50 t/STB** | — |
| D2 §6.1 | **≥ 2.5 MSCF/STB** | — |
| D6 §4.5 | **≥ 2.5 MSCF/STB = ≥ 0.12 t/STB** | 0.25–0.50 t/STB (5–10 MSCF/STB) |
| D7 §4.4 | **≥ 2.5 MSCF/STB = 0.25–0.50 t/STB** | — |
| D8 §2.4 | **≥ 2.5 MSCF/STB = ≥ 0.12 t/STB** | 0.25–0.50 t/STB (5–10 MSCF/STB) |
| D2 §6.2 criterion 6 | — | **0.25–0.50 t/STB (5–10 MSCF/STB)** |

$0.12$ vs $0.25$ t/STB is a **2.1×** disagreement about the same hard floor.

---

### CONF-36 — Plot/report export is an assigned duty with no specification

D1 §1.1 lists "export of plots" as an explicit satellite duty. D6 — the satellite specification —
contains **no** plotting, charting, report generation or figure specification, and names
`cash_flows_yearly.csv` as its only export artefact.

---

### CONF-37 — No price deck anywhere

No oil price, no escalation, no storage-credit price ($/tCO₂e), no OPEX split, no
tax/depreciation/depletion, no terminal value, no IRR, no payback. `r = 0.10` appears only in the D6
DTO example. Since NPV is the **primary optimisation objective**, this is the largest single
specification hole in the economics chain.

---

### CONF-39 — The FZI constant's stated derivation is arithmetically wrong

D7 §3.2 states "$1/0.0314^2 \approx 1012.7 \approx 1014$". $0.0314^2 = 9.8596\times10^{-4}$, so
$1/0.0314^2 = 1014.2$. The **final constant 1014 is correct**; the intermediate **1012.7 is wrong**.
Do not cite the derivation.

---

### CONF-40 — D7's rel-perm specification is under-parameterised

Names **three** permeability functions ($k_{ro}, k_{rg}, k_{rw}$) but supplies $(S_{wi}, S_{or}, S_{gc})$
— two residual/connate anchors and one **critical** anchor. No $S_{gr}$ in D7 (it exists only in D6 as
Killough/Carlson). No Corey exponents $n_{ro},n_{rw},n_{rg}$; no endpoint permeabilities; no $J(S_w)$
functional form or tabulated curve; no $\sigma$; no $\theta$.

---

### CONF-41 — D7's upscaler output cannot pass through D7's own payload

§4.1 mandates the Standalone Tensor Upscaler preserve **off-diagonal** $K_{ij}$ components; §4.2
transmits **only three diagonal components** $(K_x,K_y,K_z)$. **The payload cannot carry what the
upscaler is mandated to produce** — i.e. the off-diagonal MPFA-O correction is silently dropped on
hand-off, which is exactly the error MPFA-O exists to prevent.

---

### CONF-42 — NTG does not exist in the D7 hand-off contract

Porosity is handed over as a **single scalar field**; **no NTG field exists anywhere in D7** (0 hits,
case-insensitive), and there is no grid specification at all (no `NI/NJ/NK`, no `DX/DY/DZ`, no layering,
no corner-point construction). Fault data is reduced to displacement vectors + transmissibility
multipliers, with no fault-plane geometry and no per-segment `MULT`.

---

### CONF-43 — D7 has no seed or determinism clause at all

0 hits for `seed` / `random_state`. This **directly defeats two of D7's own four stated use cases**:
"benchmark model creation" (cross-simulator comparison requires bit-reproducible models) and "data
imputation" (requires re-derivable fields).

**This repository already has the answer:** `core/data_models.py:358-404`
`GeostatisticalParams.random_seed = 42`, plus `variogram_type`, `range`, `sill`, `nugget`,
`anisotropy_ratio`, `anisotropy_angle`, `trend_type`, `trend_parameters`, `simulation_method`,
`grid_resolution`, with `__post_init__` validation. D7 supplies none of these; the code supplies all
of them plus an `fft` method D7 does not mention. Conversely the code has **no PGS path at all**,
which is D7's central algorithm.

---

### CONF-44 — Friction-factor conventions are mixed (D3 §3.3)

$$1/\sqrt{f_D} = -2\log_{10}\left(\frac{\varepsilon_{asph}}{3.7 D_{tubing}} + \frac{5.74}{Re^{0.9}}\right)$$

The $\varepsilon/(3.7D)$ term is Darcy–Weisbach/Swamee–Jain; the $5.74/Re^{0.9}$ term is
**Fanning-style** and does not pair with it. Not reconciled with the Darcy–Weisbach $f_D$ used in D3 §1.3.2.

---

### CONF-46 — D4's backwashing permeability update is not evaluable (D4 §2.3 step 5)

$k_r(t) = k_0\left(1 - h_{cake}(t)/r_{pore}\right)^{-4}$ and $k_{ij}(t)=k_{ij,0}\,f(\phi(t))/f(\phi_0)$ —
the function **$f(\phi)$ is never defined**, and $\Delta\phi_{strip}$ is introduced with **no closure**
linking it to $\Delta h_{cake}$.

---

### ✅ CONF-47 — CLOSED 08-10-2026: a convention collision, fixed by the $k_f/w_f$ split

~~The cubic law for a planar fracture is $k = w^3/12$. The exponent 2 is an error.~~

**Root cause identified — and it is not a simple exponent slip.** The design set used **one symbol $k$**
for two different quantities:

| Quantity | Meaning | Correct form |
|---|---|---|
| $k_f$ | **permeability** of the fracture | $k_f = w^2/12$ |
| $T_{ff}$ | **transmissibility** (what NNC actually needs) | $T_{ff} = k_f w_f = w^{\mathbf{3}}/12$ — **cubic** |

Writing $T = w^2/12$ is the error; writing $k_f = w^2/12$ is **correct**. The two are one symbol apart.

**Ruled closure** (ruling 6, **C-71**):

$$\boxed{\;k_f(w_f) = \frac{w_f^{2}}{12\left[1 + 8.8\left(\frac{JRC}{w_f}\right)^{1.5}\right]}\;},\qquad T_{ff} = k_f\,w_f$$

**Verified on every limit:** $w\to\infty$ recovers the cubic law; $w\to0 \Rightarrow k_f\to0$ (aperture
closes, flow stops); $JRC=0 \Rightarrow k_f = w_f^2/12$ exactly; $JRC/w_f$ is dimensionless. ✅

> ⚠️ **Provenance still required before M7c.** The `8.8` coefficient and the `$1.5$` roughness exponent
> need a **cited source** — Barton-Bandis roughness correlations are numerous and mutually inconsistent.
> An uncited coefficient in a constitutive law is a **PROVENANCE** finding, not a style note.

---

### CONF-48 — D4's doc comment and code disagree on a boundary value

`squeeze_cement_viscosity`: doc comment and summary table state the open interval `(0.001, 100.0]`
(rejecting exactly `0.001`); the code guard is `< 0.001`, which **accepts** `0.001`.

---

### CONF-49 — The declared EoS accuracy ceiling is 8–9 %

D8 §1.2 accuracy table: PR-EOS **3.0–8.0 %** MAPD density error, SRK-EOS **4.0–9.0 %**. At that level,
**differences in recovery factor between candidate EOR designs can sit inside the fluid-model noise
band.** PC-SAFT (~1 %) and VT-PR/VT-SRK (1–4 %) are the only defensible rows. D8 also states
`TPD(y) < 0` as the instability criterion and a `n_max`-driven GEP tail.

---

### CONF-50 — `MatrixCell` cannot carry the physics the D8 interface computes

`MatrixCell` = `{ cell_id, volume, pressure, permeability: [f64;3], porosity }` — **no saturations, no
composition, no temperature, no transmissibility**. Consequences:
1. The GEP surrogate consuming `&[MatrixCell]` (§1.3) cannot represent any multiphase state.
2. `FractureMatrixFlux::calculate_mass_flux` takes **only viscosity and density**, returns a **single
   `f64`**, and has **no** pressure difference, composition, capillary term, phase index $\alpha$, or
   component index $i$ — so it **cannot express** D2 §4.1's transfer function
   $q_{m-f,\alpha}=\sigma V_{block}\frac{K_m k_{r\alpha}}{\mu_\alpha}\Delta\Phi_{\alpha,m-f}$.

---

### CONF-51 — D8's fracture model is one relation applied to three geometries

One equation set, no per-model formulation, no degrees of freedom, no applicability ranges, $C_w$ and
$K_{IC}$ unvalued, propagation criterion qualitative. $w_f\propto H_f$ is dimensionally consistent but is
not the standard PKN/DPM relation.

---

### CONF-52 — No completion taxonomy in D8

No vertical/horizontal builder distinction, multilateral, lower/upper completion, gravel pack, screen,
fish-mouthed vs. slotted liner, packers, plugs, stage-isolation tooling, inflow-profiling model.
**AICD is never mentioned.** No device characteristic curve or parameter.

---

### CONF-53 — Cement "exceeding the strength limit" is not an evaluable condition

No damage model, no tension criterion, no cyclic-count provision, no thermal-expansion coefficient.

---

### CONF-54 — D8's rate range contradicts D5's anti-pattern text

D8 §3.3 gives field-wide **20 000–60 000 MSCFD** and per-pattern **1 000–2 500 MSCFD**.
D5 §1.1 cites "rate clamped to **5 000 MSCFD** instead of the **20 000–60 000 MSCFD** range" as a
*defect signature to be avoided* — i.e. D5 presents D8's number as the anti-pattern, while D8 presents
it as the design target. Also, 20 000–60 000 field-wide against 1 000–2 500 per pattern is consistent
only for 8–60 patterns (320–2 400 acres at 40-acre spacing); the reconciliation is unstated.

---

### CONF-55 — D8 §4.2 names live Python files as if the fix were already made

Criterion 1 requires changes to `surrogate_engine.py`, `surrogate_models.py`, `run_exporter.py` — all of
which exist. Criterion 3 asserts penalties "fully removed" — see **CONF-03**. At present, D8 §4.2
criteria 1–3 are **unfulfilled requirements written as accomplishments**. Criterion 1's second half
(forbidding `N_p/15`-style averaging) is already satisfied at
`core/engine_surrogate/surrogate_engine.py:646-651`; its first half is unmeasured.

---

## 3. 🟡 DOCUMENTARY conflicts

| ID | Conflict |
|---|---|
| CONF-06b | D1 §3.2 states `f_g` grows to **0.60–0.85** after breakthrough; the formula cannot produce that. D6 §4.5 says produced-gas share rises to **35–60 %**. |
| CONF-09b | GEP tail-length example $h=7,t=9$ requires non-integer arity $n=2.143$. |
| CONF-14b | D5 §3.1 describes the target as "Newton-Raphson **+ Krylov solver**" — no Krylov method is named anywhere else, and none exists in D1 or D2. |
| CONF-16b | `S_a` in D4 appears with an undeclared source equation outside the solvent term; `h_cake` is exponential decay in D4 but an ODE with erosion in D3 — **the same symbol, two incompatible laws**. |
| CONF-18b | D1 §5.2 gives **both** Kozeny–Carman and Verma–Prusa with no selection rule or switching criterion. |
| CONF-21b | D1 §4.1 claims "Zero-Hardcoded EOS … configured dynamically by compilation of Rust generic structures" with **no configuration interface and no stability-region check** for PR/SRK. |
| CONF-25b | D5's TPD test uses **10 000 randomly generated points with no seed** → not reproducible. |
| CONF-31b | D1 §1.2 claims static dispatch via traits "guarantees Zero-Cost Abstractions" while regime switching per run is unaddressed (compile-time monomorphisation of all combinations, or dynamic dispatch somewhere). |
| CONF-38 | D7 generators: only fBm and PGS carry equations. No defaults for $N, A_0, p, l, f_0$; no seed count $m$; no voxel resolution or isovalue; no WFC tile catalogue or constraint matrices; no L-system axiom or production rules; no fracture-segment statistics. |
| CONF-45 | D3 specifies **no** test matrix; its only gate is input `validate()`. No algorithm/tolerance for the numerical $P_{eq}$ solve; $A_s$ has no closure relation; PR-EOS → `[kg/m³]` $C_a^*$ mapping unspecified. |
| CONF-45b | D4: `$Q_{critical}` in the SSSV logic has no definition, unit or validation range; no time-step prescription for the water-hammer PDE system; `$R_kill` is named in the comparison table and **never given a functional form**. |
| CONF-56 | D1 §4.3 claims `f_g(t)` "grows to 0.60–0.85" — see CONF-06b. |
| CONF-57 | D7 documents an MPI/OpenMP/OpenHPC context (OpenMPI domain decomposition, OpenMP thread-safety, RAM fragmentation) that **does not describe the live engine** — `core/engine_surrogate` is a 0D sequential surrogate with no MPI. |

---

## 4. What the design set gets right (worth carrying forward)

Recorded so a future implementation does not discard these.

| Item | Value |
|---|---|
| **No penalty-function fudging for physical constraints** | Correct principle. The live engine should adopt it — with the correction that a "physical" constraint must itself be physical, not a fitted clip (CONF-02 is precisely such a clip). |
| **Component-wise mass balance per component per step** | Correct principle and a real gate. Live engine does not enforce it per component. |
| **No artificial saturation clamping** | Correct principle. Live engine already breaks $\sum S=1$ on 27 % of timesteps (CRIT-17). |
| **EDFM / NNC without mesh rebuild** | The one coherent cross-document idea (D1 §2.2, §7.3; D2 §5.1; D8 §1.1). |
| **Adjoint gradients for NPV** | Would be a large win for the optimiser — the live engine does finite-difference-free but *forward-model-per-chromosome* evaluation. Requires a preconditioner (CONF-14) and a gradient-verification procedure (absent). |
| **Typed DTO error returns, D4's `RemediationValidationError` pattern** | Correct and clearly supersedes D3's `Result<(), String>`. |
| **`panic = "abort"` + `Result<T, E>` in the compute kernel** | Sound for a numerical core. |
| **Level-6 adversarial tests** (`t=0⁺`, phase appearance, percolation clogging, hydrate blockage) | The right level of thinking, and this level is **entirely absent** from the live repository. |
| **D5 §1.1 coverage critique** | The critique is correct; only the conclusion is wrong (CONF-21). |
| **`t_bt`, `RF = E_v E_a E_m`, Arps/Duong/SEPD, Havlena–Odeh, RQI/FZI, Corey, Leverett J** | Standard, correct, and worth having as named references — with the caveats in CONF-33, CONF-40. |

---

## 5. Assets the repository already holds that the design set lacks

| Asset | Location | What it fills |
|---|---|---|
| **SPE 5 comparative solution project** | `validation/spe5_config.py`; `agent_wiki/validation/benchmarks.md` §3 | D5 requires SPE 5 at ≤ 1.5 % — the data already exists |
| **CMG GEM reference runs** `gmflu001`–`gmflu003` | `validation/cmg/flu/`, read via `h5py` in `validation/sr3_reader.py` | D5 names **no** comparison simulator; CMG is one |
| **Five validated MMP correlations** | `evaluation/mmp.py` (Cronquist 1978, Yellig & Metcalfe 1980, Alston 1985, Yuan 2005, Lee 1979) | D6 names an "MMP calculator" with no correlation |
| **Geostatistical parameters** incl. `random_seed`, variogram family, sill, nugget, anisotropy | `core/data_models.py:358-404` `GeostatisticalParams` | CONF-43 — D7 supplies none |
| **Test suite + release gates** | `pytest` (335 passing) and `ruff --select F821` | CONF-21 |
| **Typed-unit precedent** | — | D5's `Pascal`/`MoleFraction` newtypes have no Python analogue; consider `pint`-style units or typed dataclasses |
| **Documented remediation contract** | D2 §6.2, D8 §4.2, D6 §1.2 | These requirements can be turned into live acceptance criteria — see [`evaluation_plan.md`](evaluation_plan.md) §3 |

### 5.1 ✅ External assets on this machine — `D:\RAG` (discovered 08-10-2026, **C-119**)

> ⚠️ **These are NOT in the repository.** They were located on the same workstation and are recorded here
> because the \"no SPE benchmark\" gap above is accurate *for the repository* but **materially incomplete
> for the machine**. 129 PDFs, full-text scanned.

| Asset | Detail | Unblocks |
|---|---|---|
| ✅ **`D:\RAG\Data files\SPE5-ProbForecasting-BaseCase.txt`** | **CMG GEM** deck, `*TITLE1 'SPE5 : SPE5 COMPOSITIONAL RUN 1'`, `*TITLE2 'WAG process with 1 year cycle'`, `*GRID *CART 7 7 3`, `DI/DJ = 1000 ft`, `*DK *KVAR 50/30/20`, `POR KVAR 0.2/0.22/0.18`, 208 lines | ✅ **SPE Comparative Solution Project Case 5** — the standard compositional + WAG benchmark. **CONF-26** (spatial convergence never tested), and **M1–M4** |
| 30 further CMG GEM decks | `CO2 Flooding_BaseCase` · `PolymerFlooding_BaseCase` · `SAGD_BaseCase` / `SAGD_2D_` / `SAGD_Green_` · `ShaleOil_HF_BaseCase` · `HydraulicallyFracturedBaseCase` · `WellTesting_Base` · `HM_00227` / `HM_00686` | Additional comparison cases; DTO sample files (`CompletionsDataSource`, `fluidProperties`, `ElasticPropertyBuilding`, `Tornado`, `SimultaneousInversion`) |
| `slb\EclipseReferenceManual.pdf` (2831 pp) | Full compositional simulator reference | **CONF-14** — solver and linear algebra |
| `kupdf.net_khaled-aziz-reservoir-simulation.pdf` (489 pp) | **Aziz & Settari (1979)**, *Petroleum Reservoir Simulation*, **Applied Science Publishers Ltd, London**, **ISBN 0-85334-787-5** | **CONF-14** — FVM numerics, implicit/FIM, pseudo-functions, additive correction (Watts IDC). 🔴 **Diagnoses the C-114 DOI failure**: the publisher is **not Elsevier**, so a `10.1016/` DOI is impossible for it (**C-117**) |
| `Tnav manuals\tNavFractureSimulatorGuideEnglish.pdf` (247 pp) | Fracture modelling | **CONF-51** — PKN/KGD/radial, proppant |
| `Tnav manuals\tNavPVTDesignerGuideEnglish.pdf` (279 pp) | PVT / EOS | **CONF-25**, **CONF-31** |
| `Tnav manuals\tNavWellDesignerGuideEnglish.pdf` (433 pp) | Well design, skin, IPR | **CONF-68**, **CONF-15** |
| `vdocuments.mx_shared-earth-modeling.pdf` (319 pp) | **Fanchi (2002)**, *Shared Earth Modeling*, **Butterworth-Heinemann / Elsevier Science** | ⚠️ A **different Fanchi work** than the one cited for the $r_w$ domain (**C-118**) |

> 🔴 **The one thing the library could NOT supply, and the thing that matters most:** a literal search for
> `karakas` / `tarik` across **all pages of all 129 PDFs** returns **zero hits**. The **Karakas & Tariq
> (1991) α₀ table is not obtainable locally**, so **CONF-31's blocker stands** (**C-116**). The α₀ table
> is `SOURCE_PENDING` and any α₀-dependent result must carry a provenance warning.
>
> ⚠️ **Two false positives were eliminated rather than reported as support:** `KAPPA DDA book`'s \"tariq\"
> is **Umair Tariq**, a KAPPA engineer in the acknowledgements; its \"wellbore effect\" means **wellbore
> storage**, not perforation phasing. **A keyword hit is not evidence.**

---

## 6. Conflict index

| ID | Sev | One-line |
|---|---|---|
| CONF-01 | ✅ | **RETRACTED 07-10-2026 & REPLACED** — the original "$\partial f_g/\partial M<0$" claim was a **substitution error** (viscosity ratio used as mobility ratio); measured, the design and classical forms agree to 4 dp. ⚠️ **The replacement defect is still open:** the *closure* — no $S_{or}$, linear not Corey, no water term, unguarded at $S_g\!\to\!S_{gc}$. Tracked as **C-37**. See §1.1 |
| CONF-02 | ✅ | **REMOVED 08-10** — the HCPVI cap was 99.4 % inert at the shipped default HCPVI 7.69; **deleted**, not merely flagged |
| CONF-03 | ⚪ | **DOWNGRADED 08-10 — documentation + Python-audit debt, not an engine blocker.** Reports penalty removal as done while `FAILURE_PENALTY` is still invariant and `test_surrogate_engine.py` does not exist. **INV-7** removes the need entirely (**CONF-19** closed); Python is left as-is per owner decision 1, so this stays in the **Python audit** |
| CONF-04 | ✅ **ADJUDICATED** | Core publishes $\partial q_i/\partial x$; prices + discounting off-core. **Adopted** — but the ruling's NPV expression must not be adopted verbatim (CONF-58, CONF-59) |
| CONF-58 | ⚪ | **RECLASSIFIED 08-10 — belongs to the SATELLITE ECONOMIC ENGINE, not this engine.** The NPV expression drops CO₂ purchase, recycle, storage credit and carbon tax. **INV-3** purges economics from the core (**CONF-64** closed), so this no longer blocks the compositional engine. ⚠️ It is a **live `CRIT-18`-class defect in the Python engine** and must be fixed in the satellite |
| CONF-59 | ⚪ | **RECLASSIFIED 08-10 — satellite economic engine.** Unqualified $q_gP_g$ books recycled CO₂ as gas revenue. Same reasoning as **CONF-58**. ⚠️ The live Python engine has the **opposite** defect; both must be fixed together, **in the satellite** |
| CONF-60 | 🟠 | **NEW** — a smooth barrier penalty conflicts with the adjoint engine's smoothness requirement; non-convergence is a hard failure a step penalty gets right |
| CONF-61 | 🟠 | **NEW** — "48 tests in `test_surrogate_engine.py`" is a phantom target; derive tests from the finding register |
| CONF-62 | 🟠 | **NEW** — live engine discounts end-of-year, all design docs mandate mid-year: **+4.88 % NPV** |
| CONF-05 | 🟠 | Pinch-out handling undecided ("deactivated **or** collapsed") |
| CONF-06 | 🟠 | GPU flash 1371× unanchored; no hardware; CPU/GPU tolerance parity unstated |
| CONF-07 | ✅ | **CLOSED 08-10** — the 1 000 BOPD clamp is a round number **11×–1101× below the Darcy limit**; removed under **INV-7** (C-56) |
| CONF-08 | 🟠 | `δt` reused for two different sub-grids |
| CONF-09 | 🟠 | GEP example violates its own tail-length formula ($n=2.143$) |
| CONF-10 | 🟠 | `Q_recycle,peak/20 000` normaliser unit undeclared (D1) / declared (D6) |
| CONF-11 | ⚪ | **RECLASSIFIED 08-10 — satellite economic engine.** NOCF omits hydrocarbon-gas revenue (inherits live **CRIT-18**). **INV-3** puts economics outside the core, so this does not block the compositional engine; it is a satellite defect |
| CONF-12 | 🟠 | `ConvergenceControl` has no defaults; 2 of 4 fields unused |
| CONF-13 | ✅ | **CLOSED 08-10** — diagnosis exact (`lambda` computed but never applied; no Jacobian; no linear-solve call). Ruled: **Armijo-Goldstein line search + in-core sparse solve** (**C-78**) |
| CONF-14 | 🟠 | **STILL OPEN** — overhaul partially accepted (**C-85**, **C-86**): ✅ **Watts volume-balance reduction** carrying $\partial\rho/\partial z_i$, and ✅ **unsymmetric AMG / FGMRES+BiCGStab / ILU(1)** replacing SPD-AMG (non-associated $D^{ep}$ is non-symmetric, **C-74**). ⚠️ **My dynamic-sparsity claim WITHDRAWN (C-85)** — under the **approved overall-composition formulation** the sparsity pattern is **fixed** by stencil + equation set, so *"symbolic once per topology"* is **correct**; the dynamic claim holds only for a phase-component formulation. **Ruled: fix the primary-variable formulation at M2 and re-derive from it.** ⚠️ ILU(1) vs AMG crossover must be benchmarked at 1.1 M cells |
| CONF-15 | ✅→🟡 | **RECLASSIFIED 08-10 → DOCUMENTATION gap, not a physics defect** (**C-121**). ✅ **Both of my claims withdrawn:** Koval's placement is **correct** ($t_{bt}\propto1/K$ *decreases* with K, and severe fingering ⇒ **earlier** breakthrough — the sign error was mine); $(1-S_{wi})$ is the **classical** mobile-oil fraction $W_{o,bt}=PV(1-S_{wi})/B_o$, $S_{or}$ neglect being a known second-order refinement. 🔴 **Remaining:** declare the unit basis for $q_{inj}$, declare $K$ dimensionless, state the approximation level |
| CONF-16 | ⚠️ | **PARTIALLY CLOSED** — the $S_a$ deposition/re-entrainment ODE accepted (**C-83**), but 🔴 **$C_a^*$ and $m$ undeclared** (CONF-31's pattern recurring) and there is **no static flocculation** term; 🔴 the filter-cake erosion term $\tau_{shear}\eta_{erosion}h_{cake}$ **does not dimensionally close** — the standard form $e=E\tau/[\mu(1+\alpha c)]$ needs **$\mu$ and $E$**, both missing. ✅ **Chemical-strain half FIXED SPECIFIED** (**C-122**): $\boldsymbol\varepsilon_{chem}=\tfrac13\tfrac{\Delta V_{m,tot}}{V_{ref}}\mathbf I$ with $V_{ref}$ = **bulk pore volume as a declared state variable** — 📌 the **same defect class as C-69** (missing reference denominator), so both belong under one register entry |
| CONF-17 | ⚪→✅ | **CLOSED 08-10** — ✅ **cell-local $P$ and $z_i$ as the *sole* coupling inputs; `P̄_res,eff` demoted to a post-processed diagnostic** (**C-158**). ✅ **RF is domain-integrated** (3-D volume integral, never a single-cell scalar). ✅ **RF balance is now mass-consistent** — RF = (N_p - N_inj)/(N_0 + N_influx,o), with the double-subtraction of injected oil removed (**C-167**) and **water** influx removed from the **oil** denominator (**C-168**, 8th dimensional failure) |
| CONF-18 | ✅ | **CLOSED 08-10** — C-36 selection rule enforced: Verma-Pruess (chemical precipitation / particle clogging) vs Kozeny-Carman (mechanical compaction / poroelastic deformation) (**C-84**). ⚠️ The guard still omits the $\phi_0\le\phi_c$ **denominator** test (**C-69**) |
| CONF-19 | ✅ | **CLOSED 08-10 as an engine concern — penalties are now removed by construction.** **INV-7** forbids clamps, penalties and score-shaping on physical quantities (**C-56**, **C-58**), so there is nothing left to remove. ⚠️ **Documentary residue remains:** 6 source documents still assert removal in the **past tense**, and `FAILURE_PENALTY` is still live in the Python engine (**CONF-03**). Python is left as-is per owner decision 1, so this is **documentation debt, not a blocker** |
| CONF-20 | 🟠 | `$90.95M` example exceeds the `$40M–$75M` band, twice |
| CONF-21 | 🟠 | Coverage rejected with no replacement gate |
| CONF-22 | 🟠 | D5 tolerance table column shift |
| CONF-23 | ✅ | **CLOSED 08-10 — three-tier tolerance architecture** (**C-140**). **Tier A** in-core FVM: absolute per-component $\\|\mathbf{R}_{m,i}\\|_\\infty<10^{-12}$ (D5 §3.1) · **Tier B** remap operator: $\\|\\sum M_{FVM}-\\sum M_{SL}\\|<10^{-14}$, renamed **`RemapConservationPrecision`** (D5 §5.3) · **Tier C** satellite/surrogate: relative global $\\text{MB}_{closure}\\ge99.9\\%$ (D2 §6.2, D6 §4.3). ✅ A and C **hold simultaneously** — different quantities. ⚠️ **Two refinements:** 🔴 **$10^{-12}$ kg/s is not $\\Delta t$-invariant** — measured **4 orders looser** at $\\Delta t=10^{-4}$, fatal with C-34 adaptive stepping; normalise by local mass rate or use kg/step. ⚠️ **the index $i$ is overloaded** (component in $M$, well in $W$/$G$). ⚠️ **Exporter ruled**: standardise 99.9 %, warn on [99.0,99.9), quarantine <99.0 (**C-141**) — but **Tier 3 from the exporter ≠ Tier 3 from the solver**; carry the source |
| CONF-24 | 🟠 | Clamping forbidden yet required by the same framework |
| CONF-25 | ⚠️ | **NARROWED 08-10** (**C-120**) — enum taxonomy accepted (**C-80**). ✅ **Measured: the argmin is bit-identical with and without the $-1$** (max $\lvert\Delta y\rvert=0$), because subtracting a constant cannot move a minimiser — so this is **not** an optimisation defect. 🔴 **It IS a classification defect:** the verdict is $Q_{min}<0$, and measured $Q_{min}=+0.351$ without the constant, so **an unstable split cannot be detected at all** (an INV-1-class failure). ✅ **Spec now:** two constraints (sum-to-one **and** volume balance), $\phi_i^L/\phi_i^V$ **named separately**, and 🔴 **the gradient is mandatory** — it needs $\partial\ln\phi/\partial y_i$ along the volume-balance path, so it cannot be built from $\phi$ values alone. **Iterate on $\partial W/\partial y_i$, decide on $Q_{min}$** |
| CONF-26 | 🟠 | Spatial convergence promised, never tested |
| CONF-27 | ✅ | **CLOSED 08-10 — DTO/IPC bus DELETED from the compute core** (**C-126**). The approved two-output contract (HDF5 + Parquet, open formats) already replaces it, so deleting it **removes** the IPC-vs-Zero-Bloat contradiction rather than resolving it. ⚠️ **Checkpoint must be a separate, independently-committed file** — it cannot live in the run artifact, because restarting a FAILED run would otherwise destroy the restart point under INV-1/C-51 |
| CONF-28 | ✅ | **CLOSED 08-10 — by deletion, with CONF-27** (**C-126**). No `.proto`, no field numbers, no versioning and no restart format were needed because the bus is gone. ⚠️ Restart is defined **on top of HDF5** in a **separate, independently-committed** checkpoint file, and the restart vector is **driven by the INV-6 capability declaration** — an M3 hydrodynamics run has no $\sigma_{ij}$ or $\mathbf u$ |
| CONF-29 | ⚠️ | **ARITHMETIC VERIFIED CORRECT — no literature needed** (**C-123**). $N_c{=}6$: unique off-diagonal $i<j = 15$, diagonal $= 6$, total symmetric $6{\times}6$ entries $= 21$; the $2{\times}3 = 6$ payload leaves **exactly 15 absent** ✅. 🔴 The defect is the **payload**, not the count. ⚠️ If $k_{ii}$ are also needed for a **mixing rule** the requirement rises to **21** |
| CONF-30 | 🟠 | DCF year-2 arithmetic error, ~400× the document's own $0.01 requirement |
| CONF-30b | ✅ | **CLOSED 08-10 — by deletion** (**C-127**). `petekIO` is declared an **in-memory substrate only**; the parser boundary accepts **GRDECL / RESQML / RESCUE** alone. 🔴 Consequently the seven unstated items (magic bytes, header, byte order, slab typing, chunk table, alignment, version) **stop being requirements** — there is no file to specify |
| CONF-31 | ⚠️ | **PARTIALLY CLOSED** — **Standing $P_b$ and Beggs-Robinson $\mu_{od}$ VERIFIED CORRECT** (⚠️ BR needs $T$ in **°F**, 70–295 °F). ✅ **Branch discontinuity eliminated**; ✅ **4 symbols declared**; ✅ **two of my claims RETRACTED** ($S_v$ non-monotonicity **C-92**; $\alpha_0$ direction **C-100** — $\alpha_0$'s $\ln N$ scaling encodes **isotropic** flow distribution, so larger $\alpha_0$ = better in that regime, and field preference for 0°/180° is an **anisotropy/stress-alignment** effect outside it). ✅ **Branch discontinuity eliminated**; ✅ **Symbol Register Rule adopted** (**C-110**, 4-point declaration). ✅ **two of my claims RETRACTED** ($S_v$ non-monotonicity **C-92**; $\alpha_0$ direction **C-100** — $\alpha_0$'s $\ln N$ scaling encodes **isotropic** flow distribution, so larger $\alpha_0$ = better there; field preference for 0°/180° is an **anisotropy/stress-alignment** effect outside it). ✅ **NaN generator CLOSED** (**C-102/C-103**): $\alpha_0 = 0.250+0.476\log_4(360/\theta)$ reproduces the table (360°→0.250 ✅, 90°→0.726 ✅); ⚠️ **undefined at $\theta=0$** — needs a gate, not a `match` arm. ✅ **Option (a) adopted** (**C-109**): $L_p$ restored as **raw drill geometry**, anisotropy correctly re-routed to $K_{ij}$ and $h_D$, `max(k_x,k_y)` normalisation fixed. 🔴 **Remaining: the $\alpha_0$ fit has no lower domain bound** (**C-111**) — tabulated domain is $[90°, 360°]$, so $\theta<90°$ **silently extrapolates** to $r'_w/r_w = 5.1\times$ at 15° and **8.6×** at 1° (~$39\times$ unskinned PI) with no warning — **the same defect C-97 banned, applied to our own new fit**. ✅ **3 DOIs bound and class-labelled** (**C-108**): `29111-MS` **Conference**, `12244-PA` **Journal**, `18247-PA` **Journal** — ⚠️ but the **$\alpha_0$ table in `18247-PA` remains unverified at table level** (`SOURCE_PENDING`). 🔴 **Penéloux $c_i$ still sign-inverted** (**C-88**). 🔴 **THE α₀ TABLE IS PROVENANCE-UNVERIFIABLE LOCALLY** (**C-116**): a search for `karakas`/`tarik` across **all pages of all 129 PDFs** in `D:\RAG` returns **zero hits**, so the table is `SOURCE_PENDING` and must carry `ValidityWarning::CorrelationProvenanceUnverified` at runtime; ⚠️ **M7c cannot be gated on it.** ✅ **2 of 3 disputed citations now bibliographically resolved from the local library** (**C-117**, **C-118**): Aziz & Settari (1979) is present as *Petroleum Reservoir Simulation*, **Applied Science Publishers Ltd London, ISBN 0-85334-787-5** — which **diagnoses** the `10.1016/C2013-0-06222-0` failure, because the publisher is **not Elsevier**; and Fanchi is present but as a **different work** (*Shared Earth Modeling*, Elsevier 2002). ✅ **SPE5 compositional-WAG benchmark deck discovered** (**C-119**) |
| CONF-32 | 🟠 | "Validation" silently mutates the input (C¹-smoothing, auto-patterning) |
| CONF-33 | 🟠 | SEPD adopted for the property it was chosen to avoid (CRIT-19 shape) |
| CONF-35 | ✅ | **CLOSED 08-10** — basis recorded: 60 °F / 14.7 psia ⇒ 1 MSCF CO₂ ≈ 0.0519 t. Floor `2.5` MSCF/STB = 0.13 t/STB and benchmark `5–10` = 0.26–0.52 t/STB are **separate quantities** (**C-49**) |
| CONF-36 | 🟠 | Plot export is an assigned duty with no specification |
| CONF-37 | 🟠 | No price deck at all, for the primary objective |
| CONF-39 | 🟡 | FZI derivation states `1012.7` where `1014.2` is correct |
| CONF-40 | 🟠 | D7 rel-perm: 3 functions, wrong saturation anchors, no exponents |
| CONF-41 | 🟠 | D7 upscaler must emit off-diagonal $K_{ij}$; payload carries only $(K_x,K_y,K_z)$ |
| CONF-42 | 🟠 | No NTG field; no grid specification at all in D7 |
| CONF-43 | ✅ | **CLOSED 08-10** (**C-129**). `random_seed: u64` required in every stochastic DTO (Perlin/Simplex, Voronoi, sequential Gaussian simulation). 🔴 **But seed alone does not give bit-reproducibility** — a **fixed PRNG algorithm + version**, **deterministic reduction order** (Rayon reorders `f64` summation, so output depends on thread count), and **per-cell streams keyed `hash(seed, cell_index)`** are all required. The per-cell stream is what makes **SPE5 cross-simulator comparison (C-119)** bit-stable and **INV-4** samples re-derivable |
| CONF-44 | 🟡 | Friction-factor conventions mixed in D3 |
| CONF-45 | 🟡 | D3 specifies no test matrix; $P_{eq}$, $A_s$, $C_a^*$ closures absent |
| CONF-46 | 🟠 | D4 backwashing `f(φ)` undefined; `Δφ_strip` has no closure |
| CONF-47 | ✅ | **CLOSED 08-10** — convention collision, not an exponent slip: $k_f=w^2/12 \Rightarrow T_{ff}=w^3/12$ (**C-71**) |
| CONF-48 | 🟡 | D4 doc/code boundary disagreement at `0.001 Pa·s` |
| CONF-49 | 🟠 | Declared EoS density error up to 9 % — inside the design-signal band |
| CONF-50 | 🟠 | `MatrixCell` / `calculate_mass_flux` cannot express D2's transfer function |
| CONF-51 | ⚠️ | **PARTIALLY CLOSED** — the PKN/KGD/radial triple is **verified correct** including all coefficients, both profiles and Carter leakoff (**C-82**). ✅ **Type-II toughness-dominated model CLOSED 08-10** — $\mathcal{C}_K=\sqrt2$ locked and derived (**C-159**); $C^1$ smoothstep blend over $\mathcal{K}\in[0.8,1.25]$ verified (**C-160**). ✅ $K_{IC}$ now appears in equations. 🔴 **STILL OPEN — the proppant case**: all three are **proppant-free elastic-opening** models, yet Domain 7 specifies **dynamic proppant transport** ($C_{prop}$, $h_{pack}$, embedment), so **proppant has no effect on $w_0$**. ⚠️ $w_0$ is not a closed function of the state until this is fixed |
| CONF-52 | 🟡 | No completion taxonomy; AICD never mentioned |
| CONF-53 | 🟡 | Cement failure condition not evaluable without a damage law |
| CONF-54 | ✅ | **REMOVED 08-10** — D8's `20 000–60 000 MSCFD` target was D5's own cited anti-pattern signature of a clamped rate; deleted as a design target |
| CONF-55 | ✅ | **CLOSED 08-10** (**C-142**). ✅ **`N_p/15$ declared a Ghost Finding and WITHDRAWN** — my line reference `surrogate_engine.py:646-651` pointed at the **end-of-year NPV block (CONF-62)**, and the only `15` hits in the file are unrelated day/cost constants. ✅ **D8 §4.2 reclassified as a Remediation Specification**, all criteria subjunctive, with **Gate 1** (raw physical state vectors, no hardcoded penalty/cost overrides) and **Gate 2** (unified 99.9 % validation, matching assertion bounds). ✅ **Dual-track `FAILURE_PENALTY`**: removed by construction in Rust (INV-7), retained as a legacy artifact in the Python audit. 🔴 **Corrected:** the ruling's *"decommissioned in Milestone M6"* reference is **struck** — **M6 is CO₂-EOR specifics (Domain 6)**, "decommission" appears **nowhere** in the compositional section, and owner decision 1 is **"Leave Python as is"** + engines complement at P3 |
| CONF-57 | 🟡 | D7 assumes MPI/OpenMP — does not describe the live engine |
| **CONF-63** | ✅ | **CLOSED 08-10** — standardised on Koval's **sublinear** law $E_{eff}=(0.78+0.22M^{0.25})^4$. S1's linear form over-predicts $K$ by **5.3×** at $M{=}10$, 21× at 100, 60× at 1000 (**C-48**) |
| **CONF-64** | ✅ | **CLOSED 08-10** — economics **purged** from the engine output per **INV-3**. Core emits strictly physical `Q_o,Q_w,Q_g,Q_inj,W_p,G_p,N_p,‖R_m‖,E_v,E_a,E_m,RF` |
| **CONF-65** | ✅ | **CLOSED 08-10** — the Python-file agreement requirement **deleted**. No Rust schema references `economic_npv_usd`, `results["npv"]` or `cash_flows_yearly.csv` |
| **CONF-66** | ⚠️→✅ | **CLOSED 09-10 (Ruling 38)** — yield surface + **non-associated** flow rule (**C-62**); ✅ consistent tangent (**C-74**); ✅ return mapping **CPPM**; ✅ solver **FGMRES+ILU(1)** latched per timestep; ✅ Hermite regularisation (**C-186**); ✅ **Drucker-Prager sign confirmed twice** (**C-176** criterion, **C-184** apex identity) and coefficients locked (**C-170**); ✅ **Rankine cap deleted** (**C-185**); ✅ **mu = 1 exactly**, verified in two parameterisations (**C-217**, **C-221**) — withdraws C-211/C-215; ✅ **all four M7b blockers closed** (**C-219a** dimensionless floor, **C-220** re-entrant locus, **C-221** mu, **C-222** honest meridian naming). ⚠️ Cosmetic: `VonMisesExtension` → `J2EquivalentCylinder` (**C-224**) |
| **CONF-67** | ✅ | **CLOSED 08-10** — `Fe²⁺` **added** to the aqueous species set (**7 species**, store `H⁺` not `pH`), which closes the siderite/ankerite mineralisation balance |
| **CONF-68** | ✅ | **CLOSED 08-10** — $S_{min}=-\ln(r_e/r_w)+\Delta S_{margin}$, $\Delta S\approx+0.50$–$+1.00$. Verified: denominator $=\Delta S>0$ at every radius ⇒ $J$ can never go negative; the floor and the stimulation cap are the same parameter (**C-47**) |

### Persisting in the approved output specification S1 (08-10-2026)

S1 was written as an **output** specification and did **not** re-audit the physics it embeds. These
already-registered conflicts are reprinted verbatim:

| Conflict | S1 line | Status |
|---|---|---|
| **CONF-01** (closure) | 67 | 🔴 Formula reprinted. Sign was retracted by measurement; the **closure defect stands** — do not transcribe |
| **CONF-02** | 77 | 🔴 Inert HCPVI cap, $\tau = 1.5$, reprinted |
| **CONF-35** | 83 | ⚠️ **Self-contradictory in one sentence**: "≥ 2.5 MSCF/STB (**≥ 0.12 tonne/STB**)" vs reference "**0.25–0.50 tonne/STB**" |
| **CONF-54** | 329 | ⚠️ S1 presents **20 000–60 000 MSCFD** as the design target; D5 §1.1 cites the same range as the **anti-pattern signature** of a clamped rate |
| **CONF-07** | 329 | ✅ **REMOVED under INV-7** — not a physics limit (C-56) |
| **CONF-15** | 333 | ⚠️ $t_{bt}$ uses $(1-S_{wi})$; ignores $S_{or}$ and gas saturation |
| **CONF-62** | 406–412 | ⚠️ Mid-year discounting mandated. Live engine uses **end-of-year** (`surrogate_engine.py:649-650`). **Measured +4.88 % NPV** |
---

### CONF-66 — geomechanical constitutive gaps: return mapping, hardening, DP convention, solver stack

🔴 **Body entry added 08-10-2026** (the index row existed from Ruling 24; the narrative did not).

The geomechanics addendum (D6/D7) specifies an elastoplastic constitutive block that is **incomplete in four
independent places**, and where corrections have been offered, **two of them introduced new errors of the same
class**. Blocks **CONF-14** (the M0.1 benchmark) because the non-symmetric algorithmic tangent invalidates the
Cholesky/IC path the benchmark was designed to verify.

| # | Gap | State |
|---|---|---|
| **1** | **Return-mapping algorithm unspecified** | ✅ CPPM adopted; ⚠️ citations need DOIs (**L-6**) |
| **2** | **Hardening law: peak and post-peak undefined** | ⚠️ CPPM adopted; 🔴 $H$ jumps $+3.2\times10^5\to-4.455\times10^8$ Pa at $\bar\varepsilon_p^{\text{peak}}$ — **$C^0$ in $c$, discontinuous in $H$**. ✅ cubic-Hermite regularisation over $[\bar\varepsilon_p^{\text{peak}}\pm\delta]$ adopted — but **3 of 4 Hermite conditions given** (**C-169**) |
| **3** | **Drucker-Prager convention** | 🔴 **Unresolved after two rounds.** Ruling 26 found the $\sqrt3$ scaling **inverted** (**C-161**); Ruling 27's attempted adoption of Formulation A left the coefficients **3× wrong** (**C-164**) and put the apex in **tension** (**C-165**) |
| **4** | **Solver / preconditioner stack** | ✅ non-associated $\Rightarrow$ non-symmetric $\mathbf{D}^{\text{alg}}$ ⇒ **FGMRES + ILU(1)**; ✅ **latched per timestep**; ✅ Watts' is *residual assembly*, CPR-AMG is the *preconditioner* (supersedes **C-126**). ⚠️ **CPR-AMG banned from $\mathbf{A}_{pp}$** — 3rd resurrection of **C-79** |

#### The $\sqrt3$ hazard — read before touching item 3

📌 **Three attempts, two inversions.** The correct result ($\alpha_B=\sqrt3\alpha_A$, $k_B=\sqrt3k_A$; Formulation A
with an **extra factor 3** in both denominators) was stated on round one and then discarded twice.

✅ **Ruled: the scaling is carried as a single derived constant, never re-typed per convention**, and locked by
**Gate 3** (limit-case suite). ⚠️ **Neither the dimensional gate nor a pure convention-lock gate would catch a
correct-$\sqrt3$-reasoning / wrong-limit-case error** — the mandatory assertion is $\phi\to0\Rightarrow$ **Tresca**.

**Locked target:** $F=\sqrt{J_2/3}+\alpha I_1-k(\bar\varepsilon_p)$, compression-positive $\sigma$,
$k=\frac{2c\cos\phi}{3\mp\sin\phi}$, $\alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)}$. Detail:
[`engine_invariants.md`](../compositional/engine_invariants.md) §7y.

#### CONF-66 update — 08-10-2026, Ruling 28

| # | Gap | State |
|---|---|---|
| **1** | Return-mapping algorithm | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **quintic Hermite, 6 conditions — $C^1$ verified to $10^{-20}$ (C-172)**; 🔴 **overshoots $c_{peak}$** — assert $\max c\le c_{peak}(1+\varepsilon)$ and build in a local coordinate |
| **3** | **Drucker-Prager convention** | ✅ **coefficients LOCKED**: $\alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)}$, $k=\frac{2c\cos\phi}{3\mp\sin\phi}$ (**C-170**) · 🔴 **sign pairing re-inverted — 3rd round** (**C-171**) · 🔴 **the "+15.47 % extension-meridian over-prediction is inherent, not a constant bug** (**C-175**) — needs a tension cap or a declared limitation |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **And Gate 3 — the suite meant to lock all of the above — shipped with a test that fails and a test that
cannot fail (C-173, C-174).**

📌 **Convention hazard, four inversions in three rounds** (C-161 → C-164 → C-171). ✅ **Final ruling: no scalar
convention factor is typed anywhere; named closures return a validated $(\alpha,k)$ pair.**
Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7z**.

#### CONF-66 update — 09-10-2026, Ruling 29

🔴 **The Drucker-Prager sign is now CLOSED — against my own ruling.** ✅ The submitted form
$F=\sqrt{J_2/3}-\alpha I_1-k$, compression-positive, is the **exact** Mohr-Coulomb correspondence
(**0.0000%** error at every confining pressure tested; **C-176**). ⚠️ My C-165 and C-171 rulings on
this sign were **both wrong** and are withdrawn; **§7y.2** and **§7z.2** of
[`engine_invariants.md`](../compositional/engine_invariants.md) are superseded.

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ $C^1$ verified to $10^{-20}$; 🔴 the **"monotone" Hermite is 4 DOF with 6 conditions** and the Fritsch–Carlson clamp is inapplicable to a band with an interior maximum (**C-181**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED** (**C-176**) ✅ **coefficients CLOSED** (**C-170**) · 🔴 the factory's **`Extension` branch is von Mises, not MC** (**C-177**) · 🔴 the **Rankine cap adds a discontinuity of one full cohesion at $I_1=0$ and its warning machinery targets a phantom failure** (**C-178**) · ⚠️ `k`'s denominator is a mutation of $\alpha$'s (**C-179**) · ⚠️ `MeridianType` is **inert at $\phi=0$**, so Gate 3 covers nothing (**C-180**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

⚠️ **Remaining MC fidelity gap is Lode-angle only** — the meridians are exact (**C-176**), and the
$+15.47\%$ at $I_1=0$ is the $\pi$-plane gap ✅ **validated**, with the Lode-Angle-Dependent Modified DP
$F=\sqrt{J_2/3}\,g(\theta_L)-\alpha I_1-k$ the correct remedy.

📌 **Gate 3 found the sign error that two of my rulings missed** — ✅ the limit-case suite is the artefact;
the factory is optional. Detail: **§7aa**.

#### CONF-66 update — 09-10-2026, Ruling 30

✅ **The constitutive block is substantially closed.**

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED — two-interval PCHIP.** $C^1$ residual $0.00$ at **all three nodes** across the declared $\delta$ range; $\max c = c_{peak}$ **exactly**; zero overshoot (**C-186**). ⚠️ the monotonicity clamp is **provably inactive for a linear hardening law** — if it ever activates (law flatter than $3H$) it **trades interior overshoot for an edge tangent jump**, so it must emit `ValidityWarning::HardeningMonotonicityClampActive` (**C-187**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED, confirmed twice** — by criterion comparison (**C-176**, $0.0000\%$ error at every confining pressure) and by the apex identity $I_1^{\text{apex}}=-3c\cot\phi$ (**C-184**) · ✅ **coefficients CLOSED** (**C-170**) · ✅ **Rankine cap deleted** (**C-185**) · 🔴 factory **`Extension` branch is von Mises, not MC** (**C-177**) · ⚠️ `k`'s denominator is a mutation of $\alpha$'s (**C-179**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **Optional Lode-angle extension ($g(\theta_L)$) carries two declared costs** — it **trades away** the exact
meridian match (**C-188**, up to $\approx5.71\%$) and **$\theta_L$ is undefined at $J_2=0$**, including the apex
itself (**C-189**). ⚠️ It is **not** the fix for C-177 — that is a coefficient error, not a cap-shape error, and
merging the two is the **third** time this register has done so.

📌 **Gate 3 found the sign error that two of my rulings missed** ✅ — the limit-case suite is the artefact; the
factory is optional. Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7aa**, **§7bb**.

#### CONF-66 update — 09-10-2026, Ruling 31

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** — two-interval PCHIP, $C^1$ verified to $0.00$ at all three nodes, $\max c = c_{peak}$ exactly (**C-186**). ✅ clamp warning + quadratic-claim suspension adopted (**C-190**, **C-187 CLOSED**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED, confirmed twice** (**C-176** criterion, **C-184** apex identity) · ✅ **coefficients CLOSED** (**C-170**) · ✅ **Rankine cap deleted** (**C-185**) · 🔴 factory **`Extension` branch is von Mises, not MC** (**C-177**) · ⚠️ `k`'s denominator is a mutation of $\alpha$'s (**C-179**) · 🔴 **Lode primacy anchored to the _extension_ meridian** (**C-191**) · 🔴 **$J_2$ guard threshold is dimensionless and unreachable — 9th dimensional failure** (**C-192**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **New standing CI requirement (C-194):** every $\pm$ constant in a constitutive expression must have a
`#[cfg(test)]` case evaluating it at a state whose value is known independently. 📌 **Third instance of the same
mechanism** — **C-161** ($\sqrt3$), **C-176** ($I_1$ sign), **C-191** ($\theta_L$) — and the third to be caught by
the limit-case suite rather than by review.

⚠️ **L-7: Ménétrey & Willam verified (`10.14359/1132`); Bardet 1990's submitted record was wrong in title, issue
and DOI** (`10.1115/1.2892023` → **404**; correct **`10.1115/1.2897051`**, *Lode Dependences for Isotropic
Pressure-Sensitive Elastoplastic Materials*, J. Appl. Mech. **57(3)**, 498–506).

Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7cc**.

#### CONF-66 update — 09-10-2026, Ruling 32

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** — two-interval PCHIP, $C^1$ verified to $0.00$ at all three nodes, $\max c=c_{peak}$ exactly (**C-186**); clamp warning + quadratic-claim suspension adopted (**C-187**, **C-190**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED, confirmed three times** (**C-176** criterion · **C-184** apex identity · **C-195** analytic $\theta_L$) · ✅ **coefficients CLOSED** (**C-170**) · ✅ **Rankine cap deleted** (**C-185**) · ✅ **$\theta_L=-\pi/6$ anchor LOCKED and test-pinned** (**C-195**) · ✅ **dimensional failure #9 fixed** — relative guard (**C-196**) · 🔴 `sigma_scale` division **unguarded** (**C-196a**) · 🔴 free anchors permit a **re-entrant cap** (**C-197**) · 🔴 `ValidityWarning::LodeAngleUndefined` **omitted three rounds** (**C-199**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **New rule §7s.6 — a singularity guard and its `ValidityWarning` are one indivisible unit, and edits to a
guarded expression are cumulative.** ⚠️ The `LodeAngleUndefined` warning has now been dropped from three
consecutive revisions of the same block, because each rewrite answered a different finding and replaced the text
wholesale. 📌 **The residual defects in this component are now in _how corrections are carried forward_, not in the
physics.**

✅ **Both standing CI gates endorsed** (**C-198**): constitutive-sign verification, and safety-code dimensional
linting — ⚠️ the latter largely subsumed by the newtype discipline, so it must target raw-`f64` leakage.

✅ **L-7 CLOSED.** Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7dd**.

#### CONF-66 update — 09-10-2026, Ruling 33

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** — PCHIP, $C^1$ verified, $\max c=c_{peak}$ exactly (**C-186**); clamp warning adopted (**C-187**, **C-190**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED** (three independent confirmations) · ✅ **coefficients CLOSED** · ✅ Rankine cap deleted · ✅ $\theta_L=-\pi/6$ locked (**C-195**) · ✅ relative guard, all validation ordered, warning paired (**C-196**, **C-199**) · 🔴 **dissipation inequality sign-inverted** (**C-202**) · 🔴 **convexity condition wrong** ($g+g''$ vs $g^2+2(g')^2+2gg''$) (**C-203**) · 🔴 **ratio clamp saturates silently** (**C-204**) · ⚠️ garbled $J_2$ formula (**C-200**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

📌 **Pattern now four deep: every convexity/monotonicity criterion in this component has been written in the
simplest plausible form and has been wrong** — **C-93**, **C-181**, **C-187**, **C-203**. ✅ All four were caught
by **deriving from the defining geometry**; all four **recalled** versions were confidently wrong.

🔴 **§7s.6 generalised — every _saturating transform_ is paired with its `ValidityWarning`**, not just
singularity guards. ⚠️ Silent saturation is the shared mechanism of **C-93**, **C-151**, **C-204**.

📌 **§7cc.6 generalised:** *evaluate every $\pm$ constant at an independently-known state* (C-161, C-176, C-191) —
now also *derive every inequality from its defining geometry* (C-93, C-181, C-187, C-203).

✅ **Both standing CI gates endorsed**; Gate 2 requires the newtype layer to carry **squared/exponent** types.
Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7ee**.

#### CONF-66 update — 09-10-2026, Ruling 34

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** (**C-186**, **C-190**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED** (three independent confirmations) · ✅ **coefficients CLOSED** · ✅ Rankine cap deleted · ✅ $\theta_L=-\pi/6$ locked (**C-195**) · ✅ relative guard + validation order + paired warning (**C-196**, **C-199**, **C-201**) · ✅ **convexity criterion $g^2+gg''\ge0$ CONFIRMED — my C-203 withdrawn** (**C-205**) · ⚠️ strict-convexity refinement (**C-206**) · 🔴 **dissipation check is right-signed but off-target; misses the apex entirely** (**C-207**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **C-205 is the register's clearest structural self-correction:** ⚠️ **I did not make an arithmetic error — I chose
the _polar-dual_ object instead of the locus and then derived faithfully**, so every step was internally consistent
and nothing signalled a problem. 🔴 **My counterexample was also sampled at a _single_ point** ($\theta=0$), which
happens to be exactly where the curvature numerator vanishes.

✅ **Two new rules: (a) name the object whose convexity is claimed, _then_ derive; (b) sweep any counterexample
over the interval it is claimed on.**

📌 **C-205 and C-207 are the same failure from opposite sides** — a correct criterion replaced by a wrong one after
one point, and a criterion right in sign but wrong in what it measures. 🔴 **Both are syntactically fine and
semantically off-target.**

✅ **§7s.6 generalised to every saturating transform** (silent saturation = the shared mechanism of **C-93**,
**C-151**, **C-204**). Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7ff**.

#### CONF-66 update — 09-10-2026, Ruling 35

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** (**C-186**, **C-190**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED** · ✅ **coefficients CLOSED** · ✅ Rankine cap deleted · ✅ $\theta_L=-\pi/6$ locked (**C-195**) · ✅ guard + validation + paired warnings (**C-196**, **C-199**, **C-201**, **C-208**) · ✅ **convexity $N_\kappa=g^2+gg''\ge0$ confirmed** (**C-205**) · ✅ strict floor adopted (**C-206**) · 🔴 **floor written on $N_\kappa$ but named $\kappa$ — 10th dimensional failure, stress-dependent check** (**C-210**) · 🔴 **stored-energy term uses $\dot\gamma$ instead of $\dot{\bar\varepsilon}_p$; $\mu$ factor of 2–4 flips the apex verdict** (**C-211**) · ⚠️ two sign/derivative slips (**C-212**) · ⚠️ $\kappa_{\min}$ provenance + ungated CPPM claim (**C-213**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

🔴 **Ninth compression/tension sign slip** in this component (**C-212b**), and ⚠️ the last three survived into
submitted text — 📌 the **C-161 / C-176 / C-191** family.

📌 **Fourth instance of prose/code divergence in the same message** (after **C-164**, **C-167**): ✅ **Ruled — a
`#[cfg(test)]` case per submitted _code block_, not per formula.**

✅ **L-9 added** — provenance for the curvature floor, the third tolerance constant in the register
(**L-1**, **L-5**, **L-9**) whose value nobody can cite.

Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7gg**.

#### CONF-66 update — 09-10-2026, Ruling 36

| # | Gap | State |
|---|---|---|
| **1** | Return mapping | ✅ CPPM adopted; ⚠️ DOIs pending (**L-6**) |
| **2** | Hardening law at peak | ✅ **CLOSED** (**C-186**, **C-190**) |
| **3** | **Drucker-Prager** | ✅ **sign CLOSED** · ✅ **coefficients CLOSED** · ✅ Rankine cap deleted · ✅ $\theta_L=-\pi/6$ locked (**C-195**) · ✅ guard + validation + paired warnings (**C-196**, **C-199**, **C-201**, **C-208**) · ✅ **convexity $N_\kappa\ge0$ confirmed, strict floor on `MIN_CURVATURE_NUMERATOR`** (**C-205**, **C-206**, **C-209**, **C-210**) · ✅ **dissipation form correct incl. stored energy** (**C-207**, **C-211**, **C-212**) · 🔴 **$\mu$ formula wrong and state-dependent — three orders of magnitude along one path** (**C-215**) · 🔴 **submitted code-block test hardcodes $\mu$ — fifth check that cannot fail** (**C-216**) · ⚠️ $\kappa_{\min}$ provenance (**C-213** / **L-9**) |
| **4** | Solver / preconditioner stack | ✅ FGMRES + ILU(1), latched per timestep; ✅ Watts' = assembly, CPR-AMG = preconditioner; ✅ CPR-AMG banned from $\mathbf{A}_{pp}$ |

⚠️ **C-214: my $\kappa$ scaling exponent was wrong** ($\kappa\propto1/C$, not $C^2$; measured $C\kappa\equiv1.000000$).
📌 **C-210's conclusion stands** — ⚠️ **a right conclusion reached by a wrong argument is indistinguishable by
reading it**, which is why C-214 records the measurement rather than the verdict.

📌 **CONF-66's residual list is now two items, both narrow**: the $\mu$ formula and its manifest treatment, and a
test that hardcodes the factor it exists to verify. ✅ The **physics** of this component — sign, coefficients,
Lode anchor, convexity criterion, dissipation form, non-finite gating — is settled and test-pinned.

✅ **Code-block test pinning policy adopted** (**C-216**), ⚠️ subject to the two-sided independence rule (**C-174**).

Detail: [`engine_invariants.md`](../compositional/engine_invariants.md) **§7hh**.
