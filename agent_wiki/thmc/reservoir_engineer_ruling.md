# Reservoir Engineer Rulings

Two rulings recorded:

| # | Date | Scope | Outcome |
|---|---|---|---|
| **§1–§5** | **07-10-2026** | CONF-01 … CONF-04 + secondary-flag audit | 2 accepted · 1 accepted with corrections · 1 **rejected as itself in error** |
| **§6–§9** | **08-10-2026** | CONF-63 … CONF-68 + persistent discrepancies + INV-6 + data volume + economic-engine data source | 6 adjudicated · 1 **resolved** · 2 claims corrected by measurement |

A reservoir engineer reviewed [`conflict_and_gap_register.md`](conflict_and_gap_register.md) and issued
rulings on the conflict set plus secondary audits. This page records both adjudications.

**Net effect across both rulings: 8 accepted · 1 accepted with corrections · 1 rejected as itself in
error · 1 resolved by unit conversion · 5 new conflicts opened from the remediation text · 2 of my own
framings corrected by measurement.**

> [!CAUTION]
> **The review confirmed a defect in the reviewer, not only in the documents.** CONF-01 as I originally
> wrote it was **wrong**. The engineer endorsed it while repeating the identical arithmetic error. Both
> of us substituted a viscosity ratio into a slot labelled with a mobility ratio. Measured proof is in
> [`conflict_and_gap_register.md`](conflict_and_gap_register.md) §1.1. **CONF-01 is retracted and
> replaced.** This is recorded rather than quietly edited because the repository's own discipline
> (`agent_wiki/README.md` invariant 15) requires that a changed conclusion carry its measurement.

---

## 1. Ruling summary

| ID | Engineer's position | Adjudication |
|---|---|---|
| **CONF-01** | Agrees: "$df_g/dM < 0$ CRITICAL PHYSICAL ERROR"; "violates conservation of momentum and Darcy's Law"; "cannot reach 0.60–0.85" | ❌ **REJECTED as stated.** The sign claim is wrong (measurement, §1.1). The 0.60–0.85 claim is wrong (measurement). A real defect exists in the *closure*. Severity 🔴 → 🟠. **CONF-01 replaced.** |
| **CONF-02** | Agrees: inert cap, "no physical provenance", "missing mobility coupling" | ✅ **ACCEPTED** unchanged. |
| **CONF-03** | Agrees on all four status facts | ✅ **ACCEPTED** with two corrections to the *remediation* → **CONF-60**, **CONF-61**; plus **CONF-62** surfaced. |
| **CONF-04** | Core solves conservation + publishes $\partial q_i/\partial x$; satellite applies prices and discounting | ✅ **ACCEPTED** — supersedes my earlier recommendation. Two corrections to the *expression* → **CONF-58**, **CONF-59**. |
| CONF-11 | Agrees; gives `$NOCF = Q_oP_o + Q_gP_g - Q_wC_w - Q_{inj}C_{inj}$` | ✅ agree on the defect. ⚠️ The proposed fix drops CO₂ recycle cost, storage credit and carbon tax — see **CONF-58**. |
| CONF-43 | Agrees; confirms `core/data_models.py:359` | ✅ **ACCEPTED** unchanged. |
| CONF-41 | Agrees; sharpens to full 6-component `AnisotropicTensor3D` | ✅ **ACCEPTED**, wording improved. |
| CONF-23 | Audits as "`10^-12` vs `10^-14` vs `10^-6`"; proposes FVM `10^-12`, surrogate `10^-6` | ⚠️ **PARTIALLY REJECTED.** `10^-6` has **0 occurrences** in the set. Intent kept, number rejected. |
| CONF-26 | Agrees; proposes $L_2$-norm Richardson extrapolation across mesh levels | ✅ **ACCEPTED** — upgrades a gap to a specified remediation. |
| Arithmetic | `$1/0.0314^2 = 1014.24$ ✓; year-2 DCF off by $4 849 ✓` | ✅ **ACCEPTED** — both confirmed. Their remedy ("re-run generators in `f64`") is partly inapplicable: these documents are hand-written LaTeX-in-Markdown, not generated. |

---

## 2. CONF-01 — the withdrawal, stated plainly

### 2.1 What I claimed

> "$\partial f_g/\partial M < 0$. An adverse mobility ratio reduces the produced-gas fractional flow…
> The design set re-proposes, as its reference implementation, the formula a completed audit rejected
> (**SCI-FLAW-01**)."

### 2.2 Why it is wrong

The design formula is $f_g = \dfrac{1}{1 + \left(\frac{S_o}{S_g-S_{gc}}\right)\frac{\mu_g}{\mu_o}}$.
The classical form is $f_g = \dfrac{1}{1 + \frac{k_{ro}}{k_{rg}}\frac{\mu_g}{\mu_o}} = \dfrac{M}{1+M}$.

**Same structure. Same sign.** Sweeping the viscosity ratio (measured 07-10-2026):

| $\mu_g/\mu_o$ | $M=\lambda_g/\lambda_o$ | design $f_g$ | classical $M/(1+M)$ |
|---|---|---|---|
| 0.10 | 10.0 | 0.9091 | 0.9091 |
| 1.00 | 1.0 | 0.5000 | 0.5000 |
| 10.00 | 0.1 | 0.0909 | 0.0909 |

Identical to 4 dp. My test point forced the saturation ratio to exactly 1.0, which is why the two
formulas coincided at that point — and why the error hid.

**My specific error:** I wrote "$M = 10 \Rightarrow f_g = 0.0909$" while substituting
$\mu_g/\mu_o = 10$. Under the mobility-ratio convention $M=10$ means $\mu_g/\mu_o = 0.1 \Rightarrow f_g = 0.909$,
**rising**. I had swapped the two ratio conventions.

**The engineer's identical error:** their write-up states "At $M = 10.0$ ($\mu_g/\mu_o = 0.1$):
$f_g = \frac{1}{1+1.0\times 10.0} = \mathbf{0.0909}$" — they name the viscosity ratio as 0.1 and then
multiply by 10.0 in the same line. Their own preceding paragraph defines $M = \mu_o/\mu_g$, which is a
third, non-standard convention; under it, "adverse" means $\mu_g/\mu_o \to 0$ and $f_g \to 1$.

> **Three conventions of $M$ are in play** — $\lambda_g/\lambda_o$ (mobility ratio, standard),
> $\mu_g/\mu_o$ (gas/oil viscosity ratio), and $\mu_o/\mu_g$ (engineer's definition). Two of the three
> were used interchangeably in this exchange. **Convention discipline is now a required practice for
> every fractional-flow or sweep claim in this repository.** Fixed notation is recorded in §4 below.

### 2.3 What survives

The defect is real but different: $\frac{S_o}{S_g-S_{gc}}$ is an **ad-hoc proxy for $k_{ro}/k_{rg}$**.
It omits $S_{or}$, is linear where Corey is power-law, has no water term, and is unguarded at
$S_g \to S_{gc}$. Detail and the corrected remediation are in
[`conflict_and_gap_register.md`](conflict_and_gap_register.md) §1.2.

**What does not survive:** the claim that the design set re-introduces SCI-FLAW-01, and the claim that
the formula cannot reach $f_g = 0.60$–$0.85$ (measured: it reaches $0.75$–$0.87$ at
$S_o \in [0.10, 0.20]$, $S_g \in [0.65, 0.70]$).

**Consequence for SCI-FLAW-01:** nothing. The shipped form
(`core/engine_surrogate/analytical_models.py:176-193`, `f_g = KS/(1+S(K-1))`, `K = H·E_eff`,
`E_eff = (0.78+0.22M^{0.25})^4`) and the design form **agree on sign**. SCI-FLAW-01 stays closed.

---

## 3. CONF-04 — the ruling is adopted, with two corrections that must travel with it

### 3.1 Adopted boundary

| Layer | Owns |
|---|---|
| **Simulation core** | Conservation equations; $P(x,t)$, $S_\alpha(x,t)$, $z_i(x,t)$, $q_o, q_g, q_w$; **production adjoints** $\partial q_i(t)/\partial\mathbf{x}$ |
| **Satellite / economic wrapper** | Economic vector $(P_o, P_g, C_w, \text{CAPEX}, r)$; $NPV$; $\partial NPV/\partial\mathbf{x}$ by chaining prices onto $\partial q_i/\partial\mathbf{x}$ |

This supersedes the recommendation previously recorded in this section ("put NPV in-core"). **The
engineer's boundary is better**: it is the only option under which D1 §1.1 and D1 §10.2 are both true,
and it keeps conservation physics unit-testable without economic assumptions.

### 3.2 CONF-58 — the ruling's NPV expression is less complete than the code it would replace

The engineer's expression is $NPV = \int[q_oP_o + q_gP_g - q_wC_w]e^{-rt}dt - \text{CAPEX}$.

It omits four terms the **live engine already has** at `core/engine_surrogate/surrogate_engine.py:624-646`:

| Omitted term | Live symbol | Why it matters |
|---|---|---|
| CO₂ purchase cost | `co2_purch_cost` (`$50/t` default) | Typically the **dominant** operating cost in a CO₂-EOR project. Dropping it makes every candidate look profitable. |
| CO₂ recycle cost | `co2_recyc_cost` (`$15/t` default) | Directly penalises high recycle share — the mechanism D8 §4.2-3 relies on to regulate breakthrough "naturally" |
| CO₂ storage credit | `co2_storage_credit` | A revenue term; in the live engine it is the **second** revenue line |
| Carbon tax on leakage | `carbon_tax` on `annual_caprock_leakage_tonne + annual_fault_leakage_tonne` | The only term in the model that prices containment failure. Removing it removes the economic consequence of **HIGH-23** |

**Ruling: adopt "prices off-core"; reject "fewer prices off-core."** The satellite must carry the full
economic vector that the live engine currently has, not the engineer's reduced set.

### 3.3 CONF-59 — $q_gP_g$ unqualified would sell recycled CO₂ at gas price

In this project the produced gas stream is **not** all sales gas. It splits into hydrocarbon sales gas
(`hc_gas_rate`) and CO₂ (`co2_prod_rate`, largely recycled). The live engine already keeps them separate
at `surrogate_engine.py:531`.

- The engineer's $q_gP_g$ treats the combined stream as sales gas → **books recycled CO₂ as revenue**.
- The **live engine has the opposite bug**: it omits gas revenue entirely (**CRIT-18**, 216 810 MSCF
  over 15 yr contributing $0).

**The fix must satisfy both simultaneously:**
$$NPV = \int\Big[q_o P_o + q_{g,\text{sales}}P_g + q_{\text{CO}_2,\text{stored}} C_{\text{credit}} - \big(q_{\text{CO}_2,\text{purch}}C_{\text{purch}} + q_{\text{CO}_2,\text{recycled}}C_{\text{recycl}} + q_w C_w + q_{\text{leak}} C_{\text{tax}}\big)\Big]e^{-rt}dt - \text{CAPEX}$$

---

## 4. Notation discipline fixed by this exchange

Adopted for all future fractional-flow, sweep and mobility claims in this repository:

| Symbol | Definition | Never used for |
|---|---|---|
| $M$ | mobility ratio, $\lambda_g/\lambda_o$ (gas over oil); $M>1$ adverse | viscosity ratios |
| $\mu_g/\mu_o$ | gas/oil **viscosity** ratio | called $M$ |
| $E_v$ | effective viscosity ratio, $(0.78 + 0.22M^{0.25})^4$ | — |
| $K$ | Koval factor, $K = H\cdot E_v$ | — |
| $V_{DP}$ | Dykstra–Parsons heterogeneity | — |
| $S_{or}$ | residual oil saturation | — |
| $S_{gc}$ | critical gas saturation | — |

Rule: **every claim of the form "$f$ rises/falls with $M$" must state the definition of $M$ on the
same line and must be verified by a sweep, not by a single point.** The single-point test is what let
this error survive a full audit cycle.

---

## 5. Openings from the remediation text

| ID | Sev | Origin | Finding |
|---|---|---|---|
| **CONF-58** | 🔴 | Engineer CONF-04 / CONF-11 | The ruling's NPV expression drops CO₂ purchase, CO₂ recycle, storage credit and carbon tax — terms the live engine already computes |
| **CONF-59** | 🔴 | Engineer CONF-04 | Unqualified $q_gP_g$ would book recycled CO₂ as gas revenue; the live engine has the opposite defect (**CRIT-18**) |
| **CONF-60** | 🟠 | Engineer CONF-03 | A smooth barrier penalty conflicts with the adjoint engine's smoothness requirement; and non-convergence is a *hard* failure that a step penalty gets right |
| **CONF-61** | 🟠 | Engineer CONF-03 | "48 tests in `test_surrogate_engine.py`" is a phantom target from a nonexistent Rust core — derive tests from the finding register instead |
| **CONF-62** | 🟠 | surfaced by CONF-11 audit | Live engine discounts **end-of-year** (`surrogate_engine.py:649-650`); all design docs mandate **mid-year**. **+4.88 % NPV difference** at $r=0.10$ over 15 yr |

---

## 6. Effect on [`evaluation_plan.md`](evaluation_plan.md)

| Item | Change |
|---|---|
| **CONF-01** severity | 🔴 → 🟠. It no longer blocks adoption on physics-sign grounds. Stage 0 still must fix the closure at 7 sites. |
| **CONF-04** | **No longer an open decision** — adjudicated. Recorded as an ADR in [`../decisions/architecture_decisions.md`](../decisions/architecture_decisions.md). Removes one of the four Stage-0 blockers, leaving **CONF-02** and **CONF-03** as the genuine physics/correctness blockers. |
| **CONF-02 remediation** | The engineer's proposal (Koval $K_K = H\cdot E_v$ + Dykstra–Parsons) **collapses to fixing CRIT-14 and CRIT-15 in the existing engine** — `analytical_models.py:176-193` already implements exactly that structure. Nothing needs importing from the design set. |
| **New Stage 0 items** | CONF-58, CONF-59 must be resolved *before* the CONF-04 refactor moves economics off-core, or the refactor will delete working cost terms. |
| **CONF-62** | Add "mid-year discounting decision" to Stage 1 — it is a one-line measurement that re-baselines every published NPV. |
---

# PART 2 — Ruling of 08-10-2026

Issued against **S1** (*3D THMC Reservoir Simulator Full Output Data Schema & UI Visualization
Specification*, approved 08-10-2026) and the **master map**. Full normalised S1:
[`../compositional/output_schema.md`](../compositional/output_schema.md).

## 6. Adjudication of CONF-63 … CONF-68

| ID | Ruling | Verdict |
|---|---|---|
| **CONF-63** | Standardise on `K_Koval = H · E_eff`, `E_eff = (0.78 + 0.22·M^0.25)^4` **before M3** | ✅ **ACCEPTED**, with a magnitude correction — §6.1 |
| **CONF-64 / 65** | **Purge all monetary variables and Python filenames from S1.** Core emits strictly physical time-series: `Q_o, Q_w, Q_g, Q_inj, W_p, G_p, N_p, ‖R_m‖, E_v, E_a, E_m, RF` | ✅ **ACCEPTED** — confirms INV-3 and my resolution |
| **CONF-66** | Either define yield criterion + flow rule, **or** remove `ε_p` from the baseline elastic schema and mark `NotImplemented` | ✅ **ACCEPTED** — take the second option |
| **CONF-67** | Add `Fe²⁺` to the aqueous species vector: `Ca²⁺, Mg²⁺, Fe²⁺, H⁺, HCO₃⁻, SO₄²⁻, Cl⁻` | ✅ **ACCEPTED**, with one caveat — §6.2 |
| **CONF-68** | `S_tot = max(S_min, Σ S_i)` with `S_min = -5.0`, because `r_wa = r_w·e^(−S)` drives `J → ∞` | ⚠️ **ACCEPTED IN PRINCIPLE, floor value corrected** — §6.3 |

### 6.1 CONF-63 — fix correct, magnitude overstated

The ruling states the linear form *"over-predicts viscous fingering severity by **orders of magnitude** at
high mobility ratios ($M > 10$)"*. **Measured 08-10-2026:**

| $M$ | Koval $E_{eff}$ | linear | $K$ ratio | $E_{disp}$ Koval | $E_{disp}$ linear | **RF ratio** |
|---|---|---|---|---|---|---|
| 2 | 1.177 | 2.0 | 1.70× | 0.9966 | 0.8750 | 1.14× |
| 5 | 1.512 | 5.0 | 3.31× | 0.9611 | 0.4880 | 1.97× |
| **10** | 1.882 | 10.0 | **5.31×** | 0.8971 | 0.2710 | **3.31×** |
| 20 | 2.404 | 20.0 | 8.32× | 0.8007 | 0.1426 | 5.61× |
| 100 | 4.742 | 100.0 | 21.1× | 0.5086 | 0.0297 | 17.1× |
| 1000 | 16.556 | 1000.0 | 60.4× | 0.1705 | 0.0030 | 56.9× |

where $E_{disp} = (3K^2 - 3K + 1)/K^3$ and $RF = E_{disp}(1 - S_{wi})$.

> **"Orders of magnitude" holds only above roughly $M \approx 50$.** At the realistic adverse ratio
> $M = 10$ the $K$ error is **5.3×** and the **RF** impact is **3.3×**. Note also that $E_{disp}$ saturates —
> $E_{disp}\to 3/K$ — so the RF impact is bounded and grows *sub*-linearly with the $K$ error.
>
> **The fix is still clearly right** and the error is still large. Only the magnitude language was
> overstated. Recorded so the number quoted in future discussions is correct.

### 6.2 CONF-67 — one caveat: `pH` and `H⁺` are not independent

The ruling's species vector replaces S1's `pH` with `H⁺`. ✅ Correct — and note that **storing both `pH`
and `H⁺` invites inconsistency**, since $pH = -\log_{10}H^+$ determines one from the other.

**Requirement:** store **one** — `H⁺` (activity) as the state variable, since speciation kinetics need
it; derive `pH` for reporting. S1 listed `pH` among the *state* variables, which is the weaker choice.

### 6.3 CONF-68 — the floor `-5.0` is only safe for a large drainage radius

The ruling's reasoning is correct: $r_{wa} = r_w e^{-S}$ grows without bound as $S\to-\infty$, and
$J \propto 1/\ln(r_e/r_{wa})$ **diverges** when $r_{wa}\to r_e$.

**But `-5.0` is a constant, and the singular value is geometry-dependent:**

$$S_{singular} = -\ln\!\left(\frac{r_e}{r_w}\right)$$

With $r_w = 0.354$ ft (the repo default, `core/data_models.py:73`):

| Drainage radius $r_e$ | $S_{singular}$ | Is the floor `-5.0` safe? |
|---|---|---|
| 300 ft | −6.74 | ✅ safe, margin 1.74 |
| 1000 ft | −7.95 | ✅ safe, margin 2.95 |
| 3000 ft | −9.04 | ✅ safe, margin 4.04 |
| **52.5 ft** | **−5.00** | ⚠️ **exactly at the singularity** |
| < 52.5 ft | > −5.00 | 🔴 **floor is past the singularity — `r_wa > r_e`, `ln` goes negative, `J` becomes negative** |

> **Requirement:** the floor must be **geometry-dependent**, not universal:
> $$S_{\min} = -\ln\!\left(\frac{r_e}{r_w}\right) + \text{margin}$$
> equivalently $r_{wa}\le r_e\,e^{-\text{margin}}$.
>
> **A universal `-5.0` is safe only for $r_e > 52.5$ ft.** For a tight vertical, a short horizontal, or a
> fractured interval — all realistic in this project — it is **unsafe**.
>
> **Second consideration:** acidising legitimately produces large negative skin; that is the *point* of
> wormholing. Clamping at −5 caps the stimulation benefit. The floor and the stimulation cap are the same
> parameter, and should be chosen deliberately against a measured stimulation target.

## 7. Persistent discrepancies — CONF-01, CONF-02, CONF-35, CONF-54

| ID | Ruling | Verdict |
|---|---|---|
| **CONF-01** | *"Completely purge line 67 in `S1`. Enforce $f_g = \frac{K S_g}{1+S_g(K-1)}$ or Corey phase mobility ratios."* | ✅ **ACCEPTED** — action right, **stated reason wrong** — §7.1 |
| **CONF-02** | *"Remove $1-e^{-\text{HCPVI}/\tau}$ from `S1` line 77."* | ✅ **ACCEPTED** |
| **CONF-35** | Standardise: $1\ \text{MSCF}\ \text{CO}_2 \approx 0.0519$ tonne, so $2.5\ \text{MSCF/STB} \approx 0.13$ tonne/STB | ✅ **RESOLVED — and my framing was wrong** — §7.2 |
| **CONF-54** | *"Remove 20,000–60,000 MSCFD from `S1` line 329 so it is not cited as a design target."* | ✅ **ACCEPTED** |

### 7.1 CONF-01 — action right, reason wrong

The ruling's action is exactly right and matches my recommendation: purge the formula, use the Koval
form or Corey.

> ⚠️ **But the reason must not propagate.** The ruling and the accompanying post-mortem both describe the
> defect as *"$f_g$ predicted heavier oil REDUCED gas cut ($\partial f_g/\partial M < 0$)"*. That is the
> claim **already retracted by measurement** — the sign is correct; the two forms agree to 4 dp across
> the whole viscosity-ratio range.
>
> **The real defect is the closure:** no $S_{or}$, linear instead of Corey exponents, no water term,
> unguarded at $S_g\to S_{gc}$. If the reason is recorded as "sign error", the closure stays broken and a
> later implementer may restore the formula. Tracked as **C-37**.

### 7.2 CONF-35 — resolved by unit conversion, and I mis-framed it

**I recorded CONF-35 as a 2–4× contradiction about "the same hard floor". That was wrong.** Measured
08-10-2026 with the ruling's conversion ($1$ MSCF = 28.3168 m³, CO₂ at 1.833 kg/m³ ⇒ 0.0519 t):

| Quantity | MSCF/STB | t/STB |
|---|---|---|
| **Floor** | 2.5 | **0.130** |
| **Benchmark** | 5 | **0.260** |
| **Benchmark** | 10 | **0.519** |

**The floor and the benchmark range are different quantities.** $2.5\ \text{MSCF/STB} = 0.13$ t/STB and
$5$–$10\ \text{MSCF/STB} = 0.26$–$0.52$ t/STB are both correct under one consistent conversion. I compared a
**floor** against a **benchmark** and called it a contradiction.

> **CONF-35 is RESOLVED.** The fix is to standardise on **0.0519 t/MSCF** and state the floor and the
> benchmark as the **separate quantities they are**. ✅ The ruling's arithmetic verifies.

## 8. INV-6 — endorsed, with the enum

The ruling endorses the four-state capability model and supplies the Rust declaration:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleCapabilityState {
    Implemented,   // Executed, valid physics -> Populated
    NotImplemented,// Module deferred -> Absent, explicitly declared
    Degraded,      // Reduced fidelity/fallback -> Populated + fidelity log
    Failed,        // Solver/module crashed -> Run outcome = FAILED
}
```

> ✅ Adopted verbatim. The ruling's stated reason is the right one and is worth preserving:
> *"If an unassigned field is simply zeroed out, an ML model or researcher cannot distinguish between a
> zero physical value (e.g. zero plastic strain) and an unimplemented module."*

## 9. Data volume and the economic engine

**Volume — confirmed.** The ruling endorses the measurement (116 fields × 1.1 M cells = 510.4 MB per
timestep, ~51 GB per 100 steps) and prescribes:

| # | Prescription |
|---|---|
| 1 | **Chunked HDF5 with `zstd` compression** |
| 2 | **Active-frame RAM cache** — the UI data layer loads and retains active 3D timestep slabs in contiguous RAM rather than streaming from disk during interactive scrubbing |

**Economic engine reads DuckDB / Parquet — NOT HDF5.** ✅ **ACCEPTED**, and the reasoning is sound:
economics needs well-head and bottom-hole time series ($Q_o, Q_w, Q_g, Q_{inj}, P_{bhp}, P_{whp}$), which are
KB-per-timestep in columnar form — not 510 MB spatial meshes. DuckDB executes zero-copy OLAP across
thousands of timesteps in sub-milliseconds.

> ✅ This resolves open branch **B-4** and gives the engine a **clean two-output contract**:
> **spatial → HDF5/VTK-HDF** (rendering, ParaView), **time-series → Parquet/DuckDB** (economics,
> analytics). The economic engine never opens an HDF5 file.

## 10. Python-attempt post-mortem

A complete forensic autopsy was supplied. It answers the question I had flagged as the highest-value
unknown, and it is preserved in full at
[`../compositional/python_attempt_postmortem.md`](../compositional/python_attempt_postmortem.md).

**Headline:** the failure was **not** execution speed — it was six **numerical-method** defects, all of
which transfer to Rust. It also yields five new specification requirements (**C-33 … C-37**), including
a **Heidemann–Khalil** critical solver and an **exponential soft-start** formula that appear in **none** of
the eight design documents.

---

# PART 3 — Ruling of 08-10-2026 (verification and branch closure)

The engineer verified all three of my measurement corrections and answered all four open branches.
**Every PM technical correction was accepted.**

## 10. Verification of the PM's corrections — all accepted

### 10.1 CONF-68 — geometry-dependent skin floor ACCEPTED, with a margin specified

The engineer adopted the Peaceman formulation as the governing relation:

$$q = \frac{2\pi k h}{\mu\left[\ln\!\left(\frac{r_e}{r_w}\right)+S\right]}(P_{grid}-P_{bhp})$$

and the critical singular skin $S_{sing} = -\ln(r_e/r_w)$ — which matches my derivation exactly.

**Ruled form, now with a stated margin:**

$$S_{\min} = -\ln\!\left(\frac{r_e}{r_w}\right) + \Delta S_{margin},\qquad \Delta S_{margin} \approx +0.50\ \text{to}\ +1.00$$

**Verified 08-10-2026.** Substituting $S = S_{sing}+\Delta S$ into Peaceman gives a denominator of exactly
$\Delta S$, so:

| Property | Result |
|---|---|
| Denominator | $\Delta S \ge 0.5 > 0$ at **every** radius ⇒ $J$ **cannot** go negative. Sign flip impossible |
| $r_{wa}/r_e$ | $= e^{-\Delta S} = 0.607$ — **constant across all radii**, so the guard is scale-invariant |
| Stimulation cap | $J/J_{unskinned} \le \lvert S_{sing}\rvert/\Delta S$. At $r_e = 52.5$ ft: $\le 10\times$; at $1000$ ft: $\le 15.9\times$ |
| Cost of the margin | **0.5 skin units** of wormhole stimulation benefit |

> ✅ **This closes CONF-68.** The margin is not cosmetic — it is what makes the guard safe at *every* radius
> rather than only for large $r_e$, and it puts a **principled ceiling** on the stimulation amplification
> rather than a round number. The engineer's point that *"inflow unphysically turns into outflow"* for
> $r_e < 52.5$ ft is correct and is now recorded as the reason.

### 10.2 CONF-63 — Koval sublinear scaling ACCEPTED

Accepted with my measured figures: **5.3×** over-prediction in $K$ at $M=10$, **21×** at $M=100$.
Standardise on $K_{Koval}=H\cdot E_{eff}$, $E_{eff}=(0.78+0.22M^{0.25})^4$. ✅ Closed.

### 10.3 CONF-35 — CLOSED with the conversion basis stated

> *"Standard density at $60^\circ$F and $14.7$ psia yields $1$ MSCF CO₂ $\approx 0.0519$ metric tonnes.
> All target ranges align under this conversion."*

✅ **Closed, and the basis is now recorded** — which is what was missing. Floor $2.5$ MSCF/STB $= 0.13$
t/STB; benchmark $5$–$10$ MSCF/STB $= 0.26$–$0.52$ t/STB. **Separate quantities, both correct.**

## 11. C-33 … C-37 — all validated as physically mandatory

| ID | Milestone | Engineer's validation |
|---|---|---|
| **C-33** Heidemann–Khalil | M2 | *"Michelsen TPD and Rachford–Rice fail near the critical point because the Hessian becomes ill-conditioned. A 2D HK solver directly calculates $T_c, P_c$ stability boundaries."* ✅ |
| **C-34** Exponential soft-start | M3 | *"Eliminates $t=0^+$ Dirac impulse shocks in $L$-stable time integrators."* ✅ |
| **C-35** Well control modes in the global Jacobian | M5 | ⚠️ *"**The single most critical numerical fix.** Embedding well boundary conditions directly into the global Jacobian guarantees monotonic Newton convergence."* ✅ |
| **C-36** Dual permeability-collapse rule | M3 | ✅ Split made explicit — see §11.1 |
| **C-37** Document the correct $f_g$ reason | now | *"record that $f_g$ was purged due to **lack of relative mobility closure, not sign artifacts**"* ✅ |

### 11.1 C-36 — the selection rule, now explicit

| Model | Applies to | Trigger |
|---|---|---|
| **Verma–Pruess** $k=k_0\left[\frac{\phi-\phi_c}{\phi_0-\phi_c}\right]^n$ | **Chemical / particle clogging** — TSS, scale, asphaltenes, hydrates | Throat shutoff at accessible porosity cutoff $\phi_c > 0$ |
| **Kozeny–Carman** $k=k_0\left[\frac{\phi}{\phi_0}\right]^3\left[\frac{1-\phi_0}{1-\phi}\right]^2$ | **Mechanical stress compaction** — poroelastic / plastic pore-volume reduction | Continuous porosity reduction |

✅ **Closes D1 §5.2**, which presented both with no selection rule.

## 12. Open branches — all four closed

### B-1 — ✅ ADOPT Karhunen–Loève / PCA compression

| Aspect | Ruling |
|---|---|
| **Scope** | KL/PCA over intermediate **micro-step** snapshots; **full uncompressed fields retained at major reporting checkpoints** (monthly/yearly) |
| **Rationale** | The 116 fields are heavily correlated — saturations satisfy $\sum S = 1.0$; total stress tracks $P$ and $\epsilon_{ij}$ |
| **Benefit** | **5–10× disk reduction**, and a compressed state vector for RL agents |

> ✅ **Verified 08-10-2026 — the rationale is stronger than stated.** $S_o + S_w + S_g = 1.0$ is an **exact**
> rank deficiency: three saturation fields span a two-dimensional subspace**, so KL captures it in one mode
> with **zero residual** — not an approximation. Further derived-not-stored quantities among the 116:
> $RF = E_vE_aE_m$; $\bar{P}_{res,eff}$; $K_i = y_i/x_i$; $L/F = 1 - V/F$;
> $\sigma'_{ij}=\sigma_{ij}-\alpha_BP\delta_{ij}$; $\phi_{tot}=\phi_m+\phi_f$; $K = H\cdot E_{eff}$.
>
> **The 116 are not 116 independent signals.** Both the storage saving and the RL observation-space
> reduction are well-founded.

### B-2 — ✅ ADOPT master grid topology + sparse delta state compression

| Aspect | Ruling |
|---|---|
| **Static** | Grid geometry — corner points, block volumes, initial $K_{ij}$ — written **once** in the HDF5 root group `/Geometry` |
| **Temporal** | Baseline states for active cells, then sparse deltas $\Delta P(x,t) = P^{n+1}-P^n$ only where $\lvert\Delta P\rvert > \epsilon_{threshold}$ |
| **Never written** | Static caprock and far-field aquifer cells |
| **Benefit** | Up to **70 %** reduction in 4D VTK-HDF size |

> ✅ **Verified — the volume arithmetic holds.** At SPE-10 scale the 116 fields are 510.4 MB/step for all
> cells; **30 % active → 153.1 MB/step**; **5 % front-tracking → 25.5 MB/step**.

### B-3 — ✅ `training_pairs` go to a **separate `pgvector` / Parquet store**, NOT the run HDF5

> *"If a simulation run fails halfway due to physical divergence, the run HDF5 is marked `FAILED`. If
> training vectors are embedded inside the run file, partial non-converged states risk polluting ML
> surrogate training sets. Decoupling training vectors into `pgvector` ensures that **only validated
> states from `IMPLEMENTED` or `DEGRADED` runs are committed**."*

> 🔴 **This is the strongest of the four answers, because it closes a real hole between INV-1 and INV-4.**
> If training data lived inside the run artifact, a run that failed **mid-write** could leave partial,
> non-converged states that a later harvesting step would treat as samples. Externalising the dataset
> makes the **commit** boundary explicit and enforceable: a run contributes samples **only after** its
> capability declaration is `Implemented`/`Degraded`.
>
> ⚠️ **Requirement this creates:** the commit must be **atomic** with respect to the capability declaration.
> A sample cannot become visible before its run's status is known good. If that ordering is not enforced,
> the hole reopens.

### B-4 — ✅ `audit_comp` mirroring into PostgreSQL **APPROVED**

> *"Mirroring convergence metrics ($\|R_m\|_2$, mass balance residual, Newton iteration counts, well
> constraint flips) to PostgreSQL gives QA and project management automated dashboards to track solver
> health across large benchmark suites without parsing heavy binary HDF5 files."*

> ✅ Confirms [`data_architecture.md`](../compositional/data_architecture.md) §4.7. Note the metric set is
> now explicit: $\|R_m\|_2$, mass-balance residual, Newton iteration counts, **and well constraint flips** —
> the last being a direct read-out of the **C-35** failure mode.

### B-3 (browser) — still open, correctly

`duckdb` vs `libsql` **in the browser** remains unanswered — and correctly so, since **UI/UX is out of scope**
until the engine is verified.

## 13. Net effect of Ruling 3

| | |
|---|---|
| Corrections to my work | **3 accepted** (CONF-68 floor, CONF-63 magnitude, CONF-35) — all now closed |
| New requirements validated | **5 of 5** (C-33 … C-37) |
| Open branches closed | **4 of 4** (B-1, B-2, B-5, B-7) |
| Remaining open | **1** — `duckdb` vs `libsql` in the browser (UI scope) |
| Outstanding on the register | CONF-13, CONF-14, CONF-16, CONF-18, CONF-25, CONF-47, CONF-51, CONF-08, CONF-62 |
