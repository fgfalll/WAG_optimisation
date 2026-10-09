# Constitutive Model Rulings

**Split out of [`engine_invariants.md`](engine_invariants.md) on 09-10-2026.** **Section numbers are
preserved exactly** — every cross-reference in the wiki citing sections 7x through 7hh still resolves to the
same content, just in this file.

**What belongs here:** fracture propagation scaling, the Drucker–Prager hardening law, the Lode-angle
anchor, the convexity criterion, plastic dissipation, and the resulting gate architecture.

**This is the M7b content.** All 34 rulings and `C-1…C-216` landed on the M7b constitutive model, which had
no milestone body until 09-10-2026 — see [`build_plan.md`](build_plan.md) M7b.

| Section | Subject |
|---|---|
| 7x | C_K locked, blending verified, DP scaling inverted |
| 7y | Drucker-Prager Formulation A, RF balance, Hermite, Gate 3 |
| 7z | DP coefficients locked, sign re-inverted, quintic, gate tests |
| **7aa** | **C-171 WITHDRAWN** — the correct form, and what the gate caught |
| 7bb | Tensile apex confirmed, PCHIP verified, Lode-angle cost |
| 7cc | Lode primacy on the wrong meridian; guard unreachable |
| 7dd | Lode angle locked at minus pi over 6; the **7s.6 rule** |
| 7ee | Guard accepted; the two new requirements are wrong |
| **7ff** | **C-203 withdrawn**; dissipation off-target at the apex |
| 7gg | mu state-dependent; code-block test cannot fail |
| 7hh | curvature = 1/C; mu range; the checks-cannot-fail table |

**If you are here for fluid properties, flash, transport or the status lattice, you want
[`engine_numerics.md`](engine_numerics.md) or [`engine_spec_closures.md`](engine_spec_closures.md).**

---

## 7x. C_K locked, blending verified, DP scaling inverted (08-10-2026)

Corrections **C-159 … C-163**.

### 7x.1 ✅ C-159 — $\mathcal{C}_K = \sqrt{2}$ confirmed

Volume conservation $q_{2D}t = 2L_fw_0$ into the time-volume form gives
$w_0^2 = 2K_{IC}^2(1-\nu^2)^2L_f/E^2$, hence $\mathcal{C}_K = \sqrt2 \approx 1.41421$ — **uniquely fixed by
mass conservation, not a free parameter.** ✅ **CONF-51(a) closed.**

### 7x.2 ✅ C-160 — blending weight verified correct

$W = 3s^2 - 2s^3$, $s = (\mathcal{K}-0.8)/0.45$ — the **smoothstep**:

| | |
|---|---|
| $W(0.8)$ | $0$ ✅ |
| $W(1.25)$ | $3-2 = 1$ ✅ |
| $W'(s) = 6s(1-s)$ | ✅ **$0$ at both ends** $\Rightarrow$ $C^1$ |

✅ Verified on $w_0$ itself:
$\mathrm{d}w_0/\mathrm{d}\mathcal{K} = (1-W)w_I' + W'w_{II} + Ww_{II}'$ evaluates to $w_I'$ at $\mathcal{K}=0.8$
and $w_{II}'$ at $1.25$ — **continuous with both branches**. ✅

⚠️ Both $w_I$ and $w_{II}$ are $\mathcal{K}$-**independent**, so $w_I' = w_{II}' = 0$ and the whole
transition shape is carried by $W'$ — which is the intended behaviour. ✅ **CONF-51(b) closed.**

### 7x.3 🔴🔴 C-161 — the Drucker-Prager "resolution" is INVERTED

🔴 **Decisive, convention-free algebra.** Since $\sqrt{J_2} = \sqrt3\,\sqrt{J_2/3}$:

$$\sqrt3\,F_A = \sqrt{J_2} + \sqrt3\,\alpha_A I_1 - \sqrt3\,k_A \;\Longrightarrow\; \boxed{\alpha_B = \sqrt3\,\alpha_A,\quad k_B = \sqrt3\,k_A}$$

**The formulations are related by MULTIPLYING by $\sqrt3$.** The submission defines Formulation B with $k$
and $\alpha$ **divided** by $\sqrt3$.

Measured at $c=1$ MPa, $\phi=30°$:

| | $\alpha_B$ | $k_B$ |
|---|---|---|
| **required** | $\sqrt3\,\alpha_A = 0.69282$ | $\sqrt3\,k_A = 3.600\times10^6$ |
| **submitted** | $\alpha_A/\sqrt3 = 0.23094$ | $k_A/\sqrt3 = 1.200\times10^6$ |
| ratio | **1/3** | **1/3** |

> 🔴 **The "fix" is worse than the mismatch it claims to resolve.**
>
> ✅ **Ruled — either:** adopt **Formulation A**, $F = \sqrt{J_2/3} + \alpha I_1 - k$ with
> $k = \dfrac{6c\cos\phi}{3\mp\sin\phi}$; **or** keep $F = \sqrt{J_2}$ and use
> $k = \dfrac{6\sqrt3\,c\cos\phi}{3\mp\sin\phi}$, $\alpha = \dfrac{2\sqrt3\,\sin\phi}{3\mp\sin\phi}$.
>
> ⚠️ **Recommend Formulation A** — it is what the classic DP↔MC tangency derivation produces, and it
> removes the $\sqrt3$ from the denominators entirely.
>
> ⚠️ *An MC-apex tangency cross-check was attempted and came out **inconclusive** — the $I_1$ sign
> convention under compression is not fixed in the spec. The algebra above is the reliable evidence.*

### 7x.4 ⚠️ C-162 — Hermite regularisation accepted; $\delta$ undeclared

✅ Specifying $c$ **and** $H$ at both ends of $[\bar\varepsilon_p^{peak}\!\pm\delta]$ is a valid cubic
Hermite specification and removes the tangent jump ✅.

🔴 **But $\delta$ is UNDECLARED** — no value, no basis. 📌 **Fifth occurrence of the CONF-31 pattern.**
✅ **Ruled: declare $\delta$ as a fraction of $\bar\varepsilon_p^{peak}$ with a sourced default**, recorded
in the symbol register (**C-110**'s 4 points).

⚠️ The Hermite must match the **actual** pre-peak value $c(\bar\varepsilon_p^{peak}-\delta) < c_{peak}$ —
matching $c_{peak}$ at the left end would **shift the peak**.

### 7x.5 🔴 C-163 — RF omits injection; plus four smaller items

$$\mathrm{RF} = \frac{\iiint_\Omega\big[\rho_o S_o|_0 - \rho_o S_o|_t\big]\,d\Omega}{\iiint_\Omega \rho_o S_o|_0\,d\Omega}$$

✅ Closes dimensionally (kg/kg) ✅ and is correct **for a no-oil-injection scheme** (WAG ✅).

🔴 **But solvent and polymer schemes inject oil**, and the balance then omits $\int_\Omega\rho_{o,\text{inj}}$:
$$\mathrm{RF} = \frac{\int_\Omega \rho_oS_o|_0 - \int_\Omega \rho_oS_o|_t + \int_\Omega \rho_{o,\text{inj}}}{\int_\Omega \rho_oS_o|_0}$$
— **or** restrict to the no-oil-injection case and **declare the restriction as an input-domain warning**.
⚠️ $\mathrm{RF} = N_p/N_{initial}$ holds only with **no aquifer influx**; with an aquifer the denominator
must include aquifer oil. **Both conditions belong in the manifest.**

| Item | Verdict |
|---|---|
| ✅ Non-proportional hardening declared, `proportional_hardening_only` flag | ✅ correct — and a modelling switch **must be a declared input**, which it now is |
| 🔴 Dynamic solver switch | ✅ **must be LATCHED per timestep.** Plasticity **deactivates on unloading**, so `is_plastic_active` can toggle and alternate CG-Cholesky with FGMRES-ILU. Latch on first activation for the remainder of the timestep |
| 🔴 "Watts' Volume-Balance Decoupling" listed as a **preconditioner** | ⚠️ It is a **residual formulation**, not a preconditioner. Listing it as one conflates assembly with linear algebra — the same category error as **C-126**. ✅ $\mathbf{A}_{pp}$ takes **Block-ILU(1) or unsymmetric AMG**; the volume-balance reduction belongs to residual assembly |

### 7x.6 📌 When corrections are reliable — and when they are not

| Correction type | Outcome |
|---|---|
| ✅ **Dimensional / algebraic** | **Held first time** — $W(\mathcal{K})$ smoothstep (**C-160**), $\mathcal{C}_K$ (**C-159**), volume balance, gates 1–2 (**C-113**), AMG determinism (**C-135**) |
| 🔴 **Convention-dependent / physical intuition** | **Needed correction** — DP $\sqrt3$ scaling (**C-161**), Tier-A normaliser (**C-143**), elastic-capability reasoning (**C-121**, withdrawn) |

> 📌 **C-161 is the most consequential finding of the round: a "resolution" that introduced a factor-3
> error where it claimed to remove an ambiguity.** It is the mirror image of **C-155**, which *was*
> correct — **same mechanism, opposite outcome.**
>
> ⚠️ **Which is why the dimensional and algebraic gates must be _executable_, not advisory:**
> **neither this scaling inversion nor the seven earlier dimensional failures would survive a
> `#[cfg(test)]` unit check.** But note the boundary — **C-161 would NOT be caught by a dimensional
> gate either.** Both sides are dimensionless; the error is a *convention* error. ✅ **So a third gate is
> warranted: a _convention-lock_ test that pins each yield/correlation family to a reference
> implementation with a known outer-bound check** (e.g. the MC tangency test I attempted, once the $I_1$
> sign convention is declared).

---

## 7y. Drucker-Prager locked to Formulation A (7y.1–7y.3), RF balance corrected (7y.4), Hermite completed (7y.5)

Corrections **C-164 … C-169**. ⚠️ **Two adopted items were still wrong on arrival.**

### 7y.1 🔴🔴 C-164 — Formulation A coefficients are STILL 3× wrong

🔴 **The $\sqrt3$ was removed from $F$, but the coefficients were not rescaled with it.** The submission
adopted $F=\sqrt{J_2/3}+\alpha I_1-k$ ✅ and cited Chen & Han tangency ✅ — then kept
$k=\frac{6c\cos\phi}{3\mp\sin\phi}$, $\alpha=\frac{2\sin\phi}{3\mp\sin\phi}$, which is **Formulation B's
coefficients minus $\sqrt3$**: a hybrid belonging to **neither** convention.

**✅ Decisive limit-case test — $\phi=0$ must reduce Drucker-Prager to Tresca, $\tau=c$.**

| | |
|---|---|
| state | pure shear at $\tau=c$: $s_2=s_1+2c$, $s_3=s_1$ |
| invariants | $I_1=2c$, $J_2=\frac{4c^2}{3}$, $\sqrt{J_2/3}=\frac{2c}{3}$ |
| $F$ with **submitted** $k=2.000\times10^6$ | **$-1.333\times10^6$** ❌ (must be $0$) |
| $F$ with **correct** $k=\frac{2c}{3}=6.667\times10^5$ | **$0.000$** ✅ |

$$\boxed{\;k=\frac{2c\cos\phi}{3\mp\sin\phi},\qquad \alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)}\;}$$

🔴 **Both coefficients carry an extra factor $3$ in the denominator.** Internal consistency check passed:
$\alpha$ is wrong by the *same* factor as $k$ ($0.1333$ vs $0.0444$ at $\phi=30°$) — required, since the
two must scale identically.

### 7y.2 🔴 C-165 — the apex lands in TENSION

⚠️ *"$+\alpha I_1$ reduces $F$ under confinement, increasing shear resistance"* is right in **direction** but
**inconsistent in sign** — it holds only for $\alpha<0$.

With $\alpha>0$ and tension-positive $\sigma$: hydrostatic apex at $I_1=k/\alpha=+5.196\times10^6$ ⇒ **tension**.
Measured: hydrostatic compression $10^7$ Pa ⇒ $F=-4.69\times10^6$ **elastic, never yields**; hydrostatic
tension ⇒ $F=+3.31\times10^6$ **yields** ❌. A frictional material whose cone opens in tension.

✅ **Ruled — the $I_1$ term and the stress convention must be paired, never chosen independently:**

| | Convention | Yield function |
|---|---|---|
| **(i)** | tension-positive $\sigma$ | $F=\sqrt{J_2/3}-\alpha I_1-k$, $\alpha>0$; apex $I_1=-k/\alpha=-5.196\times10^6$ ✅ **compression** |
| **(ii) ✅ recommend** | compression-positive $\sigma$ | $F=\sqrt{J_2/3}+\alpha I_1-k$, $\alpha>0$ |

⚠️ **Recommend (ii)** — matches the geomechanics stress convention already used in the design set and keeps
the submitted sign of the $I_1$ term.

🔴 **Mandatory `#[cfg(test)]` case: the apex is in compression.** First mandatory case of **Gate 3**.

### 7y.3 ✅ C-166 — the scaling algebra is confirmed

$\alpha_B=\sqrt3\alpha_A$, $k_B=\sqrt3k_A$ ✅, and eliminating $\sqrt3$ from $F$ is a real benefit ✅.

### 7y.4 🔴 C-167 / C-168 — RF balance

🔴 **The domain form double-subtracts injected oil.** From
$\frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega\rho_oS_o\,\mathrm{d}\Omega=-Q_{o,\text{prod}}+Q_{o,\text{inj}}$:

$$\int_\Omega\rho_oS_o\big|_0-\int_\Omega\rho_oS_o\big|_t = N_p-N_{\text{inj}}$$

— **the domain integral already equals $N_p-N_{\text{inj}}$**, so the extra $-\int_0^tQ_{o,\text{inj}}\mathrm{d}t$
subtracts injection twice. Measured $N_p=100,\ N_{\text{inj}}=10$: removal $=90$, submitted $=80$, **error $=-10$**.

$$\boxed{\;\mathrm{RF}(t)=\frac{\iiint_\Omega\rho_o^0S_o^0\,\mathrm{d}\Omega-\iiint_\Omega\rho_ot^S_ot\,\mathrm{d}\Omega}{\iiint_\Omega\rho_o^0S_o^0\,\mathrm{d}\Omega+\iiint_\Omega\rho_o^{\text{influx}}\mathrm{d}\Omega}\;}$$

⚠️ The sign/placement of $\rho_o^{\text{influx}}$ is a **convention that must be declared**: external influx
schedule ⇒ add cumulative oil volume; aquifer modelled **inside** $\Omega$ ⇒ numerator already contains it and
**only** the denominator term may be added.

🔴 **C-168 — $W_e$ is WATER influx inside an OIL inventory. 8th dimensional failure.** The denominator is
$[\text{kg oil}]$; the flag is `active water influx $W_e>0` ⇒ $[\text{kg water}]$, which cannot be added to it.
✅ **Ruled: $W_{e,o}$ (oil influx), flagged `ValidityWarning::AquiferOilInfluxPresent`.** A water aquifer still
affects RF — but through **water cut**, as a separate declared diagnostic, not inside the balance.

📌 *The well form $\frac{N_p-N_{\text{inj}}}{N_0+N_{\text{influx}}}$ was correct; only the domain form was wrong.
Two forms of one quantity must be asserted equal to each other in `#[cfg(test)]`.*

### 7y.5 ⚠️ C-169 — Hermite completed

A cubic Hermite needs **two values + two slopes**. Three were given; the missing one is the $C^1$ condition:

$$\boxed{\;c'(\bar\varepsilon_p^{\text{peak}}+\delta)=-\eta\,(c_{\text{peak}}-c_{\text{res}})e^{-\eta\delta}\;}$$

— the post-peak law's slope at the band's right edge. Without it $H$ still steps at
$\bar\varepsilon_p^{\text{peak}}+\delta$ and **$C^1$ does not hold where it matters.** ✅ The other three
conditions are correct and the peak-shift defect from **C-162** is genuinely fixed.

### 7y.6 ✅ Accepted unchanged

✅ $\delta$ declared — dimensionless $[\,]$, $\delta\in[10^{-4},10^{-3}]$, `SOURCE_PENDING` (Abbo & Sloan 1995).
**Sixth CONF-31 occurrence, now closed**: symbol, units, range, provenance status all recorded. ⚠️ The range's
`SOURCE_PENDING` becomes literature item **L-5**; the symbol is no longer undeclared.
✅ Per-timestep solver latching ✅ · ✅ Watts' = residual assembly / CPR-AMG = preconditioner (supersedes **C-126**)
· ✅ Hermite peak BCs ✅ · ✅ Gate 3 adopted.

### 7y.7 🔴📌 The durable gate is a LIMIT-CASE suite, not a dimensional gate

| Claim | Type | Outcome |
|---|---|---|
| **C-164** | **algebraic** — correct $\sqrt3$ reasoning | ❌ **still wrong** — wrong *limit case* |
| C-159 · C-160 | dimensionless identities, checked symbolically | ✅ held |
| C-161 | algebraic, convention-confused | ❌ inverted |
| C-167 · C-168 | balance algebra | ❌ double-count · ❌ units |

📌 **The discriminator is not dimensional-vs-algebraic. It is: was the claim tested against an independent
limit, degenerate case, or equivalence assertion?** ✅ **C-159/C-160 held only because they were checkable
symbolically.**

✅ **Ruled — every constitutive family must reproduce its degenerate limit:**

| Family | Limit case | Assertion |
|---|---|---|
| Drucker-Prager | $\phi\to0$ | reduces to **Tresca**, $\tau=c$ |
| Drucker-Prager | — | apex lies in **compression** |
| Drucker-Prager | $A\leftrightarrow B$ | $\sqrt3F_A\equiv F_B$ |
| Relative permeability | $S_w\equiv1$ | single-phase |
| Viscoplastic / creep | $t\to\infty$ | steady state |

🔴 **Named hazard — the $\sqrt3$ DP scaling.** 📌 **Three attempts, two inversions** (Rulings 26 and 27), with
the correct answer stated on round one and discarded twice. ✅ **Ruled: carry it as a single derived constant,
never re-typed per convention.**

---

## 7z. DP coefficients LOCKED (7z.1), sign pairing re-inverted (7z.2), quintic overshoot (7z.3), gate tests defective (7z.4)

Corrections **C-170 … C-175**.

### 7z.1 ✅ C-170 — coefficients LOCKED

$$\boxed{\;\alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)},\qquad k=\frac{2c\cos\phi}{3\mp\sin\phi}\;}$$

✅ The submitted derivation is **correct**, and **better than mine** — it evaluates **two** states at
$\tau_{\max}=c$ and reports **two different** $\sqrt{J_2/3}$, which is the distinction I had blurred
(**C-175**). ✅ **CONF-66 item 3 coefficients CLOSED.**

🔴 **But the "single derived constant" policy is half-implemented:**

| | |
|---|---|
| `DP_ALPHA_SCALE` | exists ✅ |
| `DP_K_SCALE` | ❌ **absent** — $k$ keeps its own hand-typed factor |
| `const DP_ALPHA_SCALE: f64 = 2.0/3.0` | 🔴 a **hardcoded literal** — exactly what a convention-lock gate forbids |

✅ **Ruled: both constants derived from the Tresca limit case inside `#[cfg(test)]`; neither re-typed.**

### 7z.2 🔴🔴 C-171 — C-165 has REGRESSED

Three pairings, all with $\alpha>0$, $k>0$:

| Pairing | $F$ | apex $I_1$ | region |
|---|---|---|---|
| (i) tension-positive | $\sqrt{J_2/3}-\alpha I_1-k$ | $-5.196\times10^6$ | compression ✅ |
| (ii) **compression-positive (RULED)** | $\sqrt{J_2/3}+\alpha I_1-k$ | $+5.196\times10^6$ | compression ✅ |
| **(iii) §2 SUBMITTED** | $\sqrt{J_2/3}-\alpha I_1-k$, **compression-positive** | $-5.196\times10^6$ | **TENSION** ❌ |

🔴 **(iii) pairs "compression-positive" with "$-\alpha I_1$" — a combination that appears in neither ruled
option.** And the submission reports *"yields in tension … apex lands strictly in the tensile regime"* **as the
desired outcome**. ⚠️ Under §2's form the material has **zero shear strength at tensile mean stress**
$-k/\alpha$: cohesive-frictional rock cannot be that weak in extension.

⚠️ The inference *"compression increases shear strength $\Rightarrow$ MUST write $-\alpha I_1$"* is sound. 🔴 **The
conclusion drawn from it is not.**

✅ **Ruled — §2 reverts to (ii): $F=\sqrt{J_2/3}+\alpha I_1-k$, compression-positive.**

🔴 **The pathology is not fixable by a sign choice, and must be recorded as such.** Linear DP matches MC on
**one meridian only** and is **too strong on the extension meridian** — measured **+15.47 %** at $I_1=0$
(**C-175**). ✅ **Ruled: add a tension-side cap surface, or declare the over-prediction as a model limitation
with a `ValidityWarning`.** A correction that makes the defect *disappear* is forbidden (**C-93**, **C-151**).

### 7z.3 🔴 C-172 — the quintic violates the invariant it was built for

✅ 6 conditions = quintic, **correct count**. ✅ **$C^1$ verified**: slope residuals at all three nodes
$\le 2.7\times10^{-20}$.

🔴 **But $\max_{\text{band}} c > c_{peak}$** — the previous round's BC set was adopted *specifically* to guarantee
"$c\le c_{peak}$ everywhere", and the quintic breaks it. Error controlled by $H_{peak}\delta/c_{peak}$:

| $H/c_{peak}$ | $H\delta/c_{peak}$ | overshoot |
|---|---|---|
| 0.06 | $6\times10^{-5}$ | bounded ✅ |
| 1 | $10^{-3}$ | $+5.7\times10^{-5}$ ❌ |
| 100 | 0.1 | $+1.07\times10^{-2}$ ❌ |
| 10⁴ | 10 | $+1.078$ ❌ |

⚠️ Present for essentially **any** non-zero $H$; small for rock ($H/c=0.32$, $\delta=10^{-3}\Rightarrow\sim3\times10^{-5}$
relative — physically negligible, invariant-breaking).

✅ **Ruled: (a) assert $\max_{\text{band}}c\le c_{peak}(1+\varepsilon)$, $\varepsilon$ declared; (b) build the
interpolant in a LOCAL coordinate $u=(\bar\varepsilon_p-\bar\varepsilon_p^{\text{peak}})/\delta$.**

> ⚠️ **The local coordinate is not cosmetic.** Evaluated on the raw $\bar\varepsilon_p$ axis at $\delta=10^{-3}$ the
> Vandermonde returned $1.6\times10^{14}$ — **catastrophic ill-conditioning**; the coefficients are meaningless
> without it.
>
> ⚠️ $\mathrm{d}^2c/\mathrm{d}\bar\varepsilon_p^2$ is **unbounded across the band** by the spec. A bounded-curvature
> assertion is missing.

### 7z.4 🔴🔴 Gate 3's two shipped tests are both defective

**Test 1 — fails.** `sqrt_j2_over_3 = c/3.0.sqrt()` with comment `J2 = c^2` is the $(c,0,-c)$ state. With the correct
$k=2c/3$:

$$f_{yield}=\frac{c}{\sqrt3}-\frac{2c}{3}=\mathbf{-0.0893c}\qquad(-13.40\%)$$

🔴 **It asserts DP yields at $I_1=0$ — the very state §1A itself shows is $\sqrt{J_2/3}=c/\sqrt3\ne k$.**
✅ **Ruled: evaluate the Tresca limit on the triaxial-compression meridian** (matching the meridian $k$ was
derived on), **and** assert all three: compression meridian, extension meridian, $I_1=0$ plane — with the
**$15.47\%$ gap recorded, not asserted away**.
⚠️ `DP_ALPHA_SCALE`, `calculate_dp_k`, `calculate_dp_alpha` are referenced but **appear nowhere in the test body**.

**Test 2 — vacuous.** `delta_m_domain = n_p - n_inj`, then assert `(n_p-n_inj)/n_initial == delta_m_domain/n_initial`.
🔴 **A quantity compared with itself; no integral is evaluated.** It passes against an engine whose RF is
arbitrarily wrong — **including the exact C-167 double-subtraction it was written to catch.**
✅ **Ruled: the two forms must be built from _independent_ inputs** — wellhead from $\int Q_{o,\text{prod}}$,
$\int Q_{o,\text{inj}}$; domain from an actual 3-D mesh reduction of $\rho_oS_o$ — then compared.
⚠️ `assert_eq!` on `f64` $\Rightarrow$ `abs_diff_eq!`.

> 📌 **C-174 is the register's fourth "check that cannot fail"** — after **C-134** (Kahan), **C-152**
> ($\Delta t$-cancellation) and **C-143** (Tier-A normaliser). 📌 **All four share one mechanism: the assertion's
> inputs were derived from the quantity being asserted.** ✅ **Gate 3 must additionally require that each
> equivalence test draw its two sides from independent sources.**

### 7z.5 🔴 C-175 — 🔴 MY OWN C-164 PREMISE WAS WRONG

I wrote *"$\phi=0$ must reduce DP to Tresca ($\tau=c$)."* 🔴 **At $\phi=0$, $\alpha=0$, so $F=\sqrt{J_2/3}-k$ is
independent of $I_1$ — DP becomes a von Mises criterion, which does _not_ reduce to Tresca.**

Four states, all with $\tau_{\max}=c$:

| state | $I_1$ | $\sqrt{J_2/3}$ | $F$ |
|---|---|---|---|
| pure shear $(0,2c,0)$ | $+2c$ | $2c/3$ | $0$ ✅ |
| **$(c,0,-c)$** | $0$ | $c/\sqrt3$ | $\mathbf{-0.0893c}$ ❌ |
| triax. compression $(-2c,0,0)$ | $-2c$ | $2c/3$ | $0$ ✅ |
| triax. extension $(2c,0,0)$ | $+2c$ | $2c/3$ | $0$ ✅ |

📌 **$\sqrt{J_2/3}$ is not a function of $\tau_{\max}$; no single $k$ can match Tresca everywhere.**
$k=2c/3$ matches **only** the meridians.

✅ **The constant is correct. Its justification must be restated:** $k$ is fixed by the **compression meridian**,
and the residual **$+15.47\%$ at $I_1=0$** is **inherent to the correspondence**, not a defect in the constant.
⚠️ If *exact* Tresca at $\phi=0$ on **all** paths is required, the fix is a **sectoral / three-invariant
criterion**, **not a different $k$**.

### 7z.6 🔴📌 The convention hazard has now inverted in *both* directions

| Round | Error | Direction |
|---|---|---|
| 26 | **C-161** $\sqrt3$ scaling | too small (÷3) |
| 27 | **C-164** Formulation A coefficients | too large (×3) |
| 28 | **C-171** sign of $\alpha I_1$ | **flipped while correcting the previous flip** |

📌 **Four inversions across three rounds** — and each was introduced *by the act of fixing the last*.

✅ **This settles the open question from Ruling 27: the constants must not be re-typed _per convention_ — they
must not be typed _at all_.**

✅ **Ruled — the yield function is assembled from _named_ $\phi$-dependent closures returning a validated
$(\alpha, k)$ pair. No scalar factor is written by hand anywhere in the codebase; Gate 3 asserts the limit
cases.**

---

## 7aa. 🔴 C-171 WITHDRAWN — the correct form, and what the gate caught (09-10-2026)

Corrections **C-176 … C-183**. 🔴 **§7y.2 and §7z.2 are WITHDRAWN.**

### 7aa.1 🔴 C-176 — RETRACTION. The submitted sign is the exact MC correspondence

$$\boxed{\;F=\sqrt{J_2/3}-\alpha I_1-k(\bar\varepsilon_p),\qquad \alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)},\quad k=\frac{2c\cos\phi}{3\mp\sin\phi}\;}$$

Compression-positive $\sigma$. Measured against the MC triaxial-compression line, $c=10^6$ Pa, $\phi=30°$:

| $P/P_{\text{apex}}$ | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|
| **$-\alpha I_1$ (submitted)** | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** | **0.0000%** |
| $+\alpha I_1$ (**my C-171**) | −57.14% | −73.47% | −80.00% | −83.52% | −85.71% |

**And the apex — my justification was wrong too.** MC in compression-positive: $\tau=c+P_{\text{comp}}\tan\phi$.

- pure hydrostatic **compression**: required $\tau>0$, actual $\tau=0$ ⇒ **always elastic, no compression apex**
- pure hydrostatic **tension** $T$: required $c-T\tan\phi\le0$ at $T=c\cot\phi=1.7321\times10^6$ Pa ⇒ **MC's apex is a TENSILE cutoff**

DP with $-\alpha I_1$: apex at $I_1/3=-1.7321\times10^6$ — **exactly MC's tensile apex**, matching to $10^{-16}$.

🔴 **Two compounding errors on my side: I wrote the MC principal line with the wrong sign on the $P$ term, *and* I reasoned the apex from the cone's geometry instead of the criterion.** 📌 *Same mechanism as C-134 and C-152: a physical claim asserted from a shape and never evaluated.*

✅ **A yield-criterion sign cannot be reasoned from geometry — only settled by comparing against the parent criterion over a range of states.**

### 7aa.2 🔴 C-177 — the factory's `Extension` branch is not Mohr-Coulomb

| $P$ | MC $T$ | factory `Extension` | error |
|---|---|---|---|
| 0 | $3.4641\times10^6$ | $1.1547\times10^6$ | **−66.67%** |
| $1.5P_{\text{apex}}$ | $8.6603\times10^5$ | $2.0207\times10^6$ | +133.33% |
| $2P_{\text{apex}}$ | $4.7\times10^{-10}$ | $2.3094\times10^6$ | diverges |

⚠️ $1.1547\times10^6 = 2c/\sqrt3$ — it returns the **von Mises** uniaxial-tension value. 🔴 **The $3+\sin\phi$ branch is a von Mises extension correspondence mislabelled as MC.**
✅ **Ruled: rename `MeridianType::VonMisesExtension` and document it as *not* MC — or delete it and keep one MC compression branch.**

### 7aa.3 🔴 C-178 — the Rankine cap repairs nothing and adds a discontinuity

⚠️ With the correct sign, **DP already reproduces MC's tensile cutoff exactly.** So $\sigma_t$-driven Rankine cap **overrides a correct feature**:

- if user $\sigma_t\ne$ MC's $2c\cos\phi/(1+\sin\phi)$, the composite surface **kinks**
- 🔴 at $I_1=0$: Rankine $\tau=0$ vs DP $\tau=k=6.928\times10^5$ Pa — **discontinuity of one full cohesion, independent of $\sigma_t$**

✅ **Ruled: delete the cap, and delete `ValidityWarning::UncappedDruckerPragerTensileRegime` + `ValidityClass::ConvergedOutsideEnvelope`.**

> ⚠️ **A warning for a failure that does not exist is itself a defect.** It trains a reader to treat `ConvergedOutsideEnvelope` as noise — the reporting-side twin of the C-93 / C-151 prohibition on making failure *disappear* rather than be *reported*.

### 7aa.4 ⚠️ C-179 / C-180 — factory wiring, and an inert branch

🔴 **`k`'s denominator must not be a mutation of $\alpha$'s.** `k = 2c\cos\phi/(\text{denom}/3.0)` means **editing one line moves both coefficients together** — the C-161 hazard re-seated one abstraction level up.
✅ **Ruled: two independently-stated denominators + a test asserting the $\alpha/k$ ratio.** ⚠️ No validation on `c_pa` or $\phi_{\text{rad}}\in[0,\pi/2)$ — INV-7 requires both.

🔴 **`MeridianType` is INERT at $\phi=0$** ($\sin\phi=0$ collapses both denominators), so Gate-3 Test 1 — run at $\phi=0$ — **never exercises the branch it exists to test.**
✅ **Ruled: run the meridian test at $\phi>0$ and assert the two branches DIFFER.**

### 7aa.5 🔴 C-181 — the "monotone Hermite" has 4 DOF and 6 conditions

The submitted basis is the standard **cubic** Hermite on $s\in[0,1]$ — **4** degrees of freedom. $c(0.5)=c_{peak}$ and $c'(0.5)=0$ **are not free parameters.**

| required | measured |
|---|---|
| $c(0.5)=1.0$ | $0.99994$ (error $-6.0\times10^{-5}$) |
| $c'(0.5)=0$ | $-1.5$ |

🔴 **And the Fritsch–Carlson clamp is inapplicable:** it is a theorem for **monotone** node data; this band has an interior maximum by construction. $\Delta=c_1-c_0=-1.19973\times10^{-4}\Rightarrow3\Delta=-3.599\times10^{-4}$, while $d_0=+1.200\times10^{-4}$ — 🔴 **clamping $d_0\le3\Delta$ negates the hardening slope outright.**

✅ **Ruled: (a) two monotone segments $[0,0.5]$, $[0.5,1]$, each clamped; or (b) keep the quintic** ($C^1$ verified to $10^{-20}$) **and assert $\max_{\text{band}}c\le c_{peak}(1+\varepsilon)$ in a local coordinate** ✅ — the local-coordinate half of C-172's fix stands.

### 7aa.6 ⚠️ C-182 — the $10^{-12}$ tolerance is not scale-free

✅ Test 2's **independence fix is accepted** — mesh reduction vs. drift-flux well integrals are genuinely independent.

Measured at $N=1.1\times10^6$ cells: naive f64 rel. err. $1.99\times10^{-14}$; random-walk floor $\sqrt N\epsilon=2.33\times10^{-13}$; worst case $N\epsilon=2.44\times10^{-10}$.

🔴 **$10^{-12}$ leaves $\sim4\times$ headroom over pure summation**, and does not scale with $N_t$ or Newton depth — **the same constant cannot serve a 500-step implicit run.**
✅ **Ruled: tolerance $=\kappa\sqrt N\,\epsilon$, $\kappa$ declared, re-derived per accumulation depth** — consistent with the Tier-A bound already verified for this engine (**C-143**).

### 7aa.7 🔴📌 The gate is the artefact; the factory is optional

⚠️ **Ruling 27 asked whether the factory would prevent a repeat. This round answers it: no.**

| | |
|---|---|
| My proposed factory | wired wrong (**C-179**) and inert where tested (**C-180**) |
| What actually caught the error | **a numerical comparison against the parent criterion across a sweep** (C-176) |

🔴 **Ruled: the `#[cfg(test)]` limit-case suite is the artefact. The factory is optional convenience.**

⚠️ **Honest count on this item:** my Ruling 26 and Ruling 27 sign rulings were **both net-negative** — the coefficient chain survived *by luck of the Tresca check*, while **the sign was wrong twice**, both times asserted confidently. ⚠️ **Confidence was uncorrelated with correctness here** — the same trap as **C-134**, **C-121**, **C-152**.

---

## 7bb. Tensile apex confirmed, PCHIP verified, Lode-angle cost declared (09-10-2026)

Corrections **C-184 … C-189**.

### 7bb.1 ✅ C-184 — tensile apex identity

$$I_1^{\text{apex}}=-\frac{k}{\alpha}=-\frac{2c\cos\phi/(3-\sin\phi)}{2\sin\phi/[3(3-\sin\phi)]}=-3c\cot\phi,\qquad P_{\text{apex}}=\frac{I_1}{3}=-c\cot\phi$$

✅ **Exact**, and identical to Mohr-Coulomb's tensile cutoff $T=c\cot\phi$. **C-176 stands; [engine_constitutive.md](engine_constitutive.md) 7aa is the authority.**

### 7bb.2 ✅ C-185 — Rankine cap and its warnings deleted

The $k=6.928\times10^5$ Pa jump at $I_1=0$, and a warning that reported a **phantom failure**, are both gone. **C-178 closed.**

### 7bb.3 ✅ C-186 — two-interval PCHIP VERIFIED

**C-181's overdetermination is resolved** — each interval is an independent cubic (4 DOF, 4 conditions):

| interval | conditions |
|---|---|
| $[0,\,0.5]$ | $c(0)=c_{hard}$, $c(0.5)=c_{peak}$, $c'(0.5)=0$, $d_0=2\delta H_{hard}$ |
| $[0.5,\,1]$ | $c(0.5)=c_{peak}$, $c'(0.5)=0$, $c(1)=c_{soft}$, $d_1=2\delta H_{soft}$ |

| $\delta$ | $C^1$ resid @$s=0$ | @$s=0.5$ | @$s=1$ | $\max c/c_{peak}$ |
|---|---|---|---|---|
| $10^{-4}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ |
| $3\times10^{-4}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ |
| $10^{-3}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ |

✅ **$C^1$ at all three nodes; $\max c = c_{peak}$ exactly; zero overshoot.**

✅ **And the simplified $d\le3\Delta$ clamp IS sufficient** — I suspected the sharper $\sqrt3\Delta$ (which applies when $d_{\text{mid}}\ne0$), searched for a counterexample, **found none, and withdraw that suspicion.** **CONF-66 item 2 CLOSED.**

### 7bb.4 ⚠️ C-187 — the clamp is inactive-by-construction, and that is the risk

Activation condition: $\delta H>3(c_{peak}-c_{hard})$.

For a **linear** hardening law $c_{peak}-c_{hard}=H\delta$ ⇒ $H\delta>3H\delta$ is **false** ⇒ 🔴 **the clamp is
provably inactive and $C^1$ is exact for _any_ $\delta$.** It is a pure safety net.

It bites only when the law **flattens toward the peak** so the local **secant** slope exceeds $3\times$ the
**tangent** modulus:

| hardening law | $d_0/\Delta_1$ | clamp | $C^1$ residual @$s=0$ |
|---|---|---|---|
| linear | 1.000 | no | $0.0$ |
| flattening ×3 | 3.000 | **YES** | $2.3\times10^{-10}$ |
| flattening ×10 | 10.000 | **YES** | $\mathbf{4.48\times10^{2}}$ |
| flattening ×30 | 30.000 | **YES** | $\mathbf{5.76\times10^{2}}$ |

🔴 **The clamp trades the _interior overshoot_ for a _tangent jump at the band edge_** — structurally identical
to **C-151** (Δt cutback / fallback solver) and the **C-93** clamp.

✅ **Ruled: on either activation, emit `ValidityWarning::HardeningMonotonicityClampActive` carrying $\delta$,
$d_0/\Delta_1$ and the $C^1$ residual — the clamp may not be silent — and suspend the quadratic-Newton claim
for that timestep.**

> 📌 **An inactive-by-construction safeguard cannot be validated by observation, and the day it activates is the
> day its assumption has already failed.** Same rule as C-93 and C-151: *can a reader tell what happened?*

### 7bb.5 🔴 C-188 / C-189 — the Lode-angle extension is a **trade**, not a fix

⚠️ **$g(\theta_L)$ does _not_ repair the $-66.67\%$.** That figure is the **mislabelled `Extension` branch**
(**C-177**), which returns the **von Mises** uniaxial value $2c/\sqrt3$. 🔴 **This is the third time these two
distinct gaps have been merged** (after **C-183** and §1B of the previous round). $g$ is a *cap-shape*
correction; `MeridianType::Extension` is a *coefficient* error. **They need different fixes.**

🔴 **C-188 — $g$ destroys the exact meridian match.** $F=\sqrt{J_2/3}\,g(\theta_L)-\alpha I_1-k$: on the
compression meridian $\theta_L$ is meridian-specific, so unless $g\equiv1$ **exactly there**, C-176's measured
**$0.0000\%$** agreement is lost (deviation to $\approx5.71\%$ at the $g/\Delta_1=3$ extremum).
✅ **Ruled: this is a _trade_. The spec must declare which exactness is required — meridians (current) or
$\pi$-plane (with $g$) — and record the other as a declared limitation.**

🔴 **C-189 — $\theta_L$ is UNDEFINED at $J_2=0$.** Every hydrostatic state and the apex lie there.
✅ **Ruled: $g$ carries a declared value at $J_2=0$, and any run whose cells land there emits
`ValidityWarning::LodeAngleUndefined`** — otherwise $g$'s sign at the apex is arbitrary and the apex identity of
**§7bb.1 is not reproducible.**

---

## 7cc. Lode-angle primacy on the wrong meridian; guard unreachable; citation corrected (09-10-2026)

Corrections **C-190 … C-194**.

### 7cc.1 ⚠️ C-190 — C-187 closed; and my own 5.71% was borrowed

✅ `ValidityWarning::HardeningMonotonicityClampActive { delta, ratio, c1_residual }` ✅ plus suspension of the
$\mathcal{O}(\|R\|^2)$ claim ✅ — **C-187 CLOSED**.

🔴 **Withdrawn: the 5.71%.** It came from the **PCHIP clamp sweep** (§7bb.4, $d_0/\Delta_1=3$), and I then
re-used it in **C-188** as a *Lode-angle* deviation, writing "at the $g/\Delta_1=3$ extremum" — which conflates
$g$ with $\Delta_1$ and means nothing. ✅ **The Lode deviation must be measured against MC's $\pi$-plane hexagon**
(inscribed or circumscribed). **C-188's magnitude is unquantified until then.**

### 7cc.2 🔴🔴 C-191 — primacy is anchored to the EXTENSION meridian

Submitted: compression at $\theta_L=+\pi/6$, with $g(+\pi/6)\equiv1.0$.
Measured under the submission's **own** formula $\sin3\theta_L=-\frac{3\sqrt3}{2}\frac{J_3}{J_2^{3/2}}$, $J_3=\det(s)$:

| state (compression-positive) | $J_2$ | $J_3$ | $\theta_L$ |
|---|---|---|---|
| triaxial **compression** $(2,\ 0.5,\ 0.5)$ | 0.750 | +0.250 | $\mathbf{-30.0000°}=-\pi/6$ |
| triaxial **extension** $(0.2,\ 1,\ 1)$ | 0.213 | −0.038 | $\mathbf{+30.0000°}=+\pi/6$ |
| pure shear $(1,\ 0,\ -1)$ | 1.000 | 0.000 | $0°$ |

🔴 **$g(+\pi/6)=1$ anchors primacy to extension** — C-176's $0.0000\%$ match **silently relocates off compression.**
Nothing crashes; the locus stays plausible; the number is simply wrong for the loading that matters.

$$\boxed{\;g(\theta_L)\equiv 1\quad\text{at}\quad \theta_L=-\pi/6\;}$$

✅ **Ruled: verify the anchor with a `#[cfg(test)]` case computing $\theta_L$ at a triaxial-compression state.**
⚠️ The sign flips with the definition of $J_3$ **and** with the ordering convention ($s_1\ge s_2\ge s_3$ vs $\le$) —
📌 **it cannot be asserted, only measured.**

### 7cc.3 🔴🔴 C-192 — the guard threshold is dimensionless. **9th dimensional failure**

`if j2 <= 1.0e-12 { return 1.0; }` — a **dimensionless literal** against $J_2$ **in $\mathrm{Pa}^2$**.

| state | deviatoric $J_2$ | guard fires? |
|---|---|---|
| weak sediment $p\sim10^4$ Pa | $6.667\times10^{7}\ \mathrm{Pa}^2$ | **NO** |
| reservoir rock $p\sim10^7$ Pa | $6.667\times10^{13}\ \mathrm{Pa}^2$ | **NO** |
| deep rock $p\sim10^8$ Pa | $6.667\times10^{15}\ \mathrm{Pa}^2$ | **NO** |

🔴 **Unreachable — so the $0/0$ it exists to prevent is not prevented.**

$$\text{guard on a _relative_ scale:}\qquad J_2\le\frac{\varepsilon\,\sigma_{\text{scale}}^{\,2}}{3},\qquad \sigma_{\text{scale}}=\text{declared}$$

⚠️ The already-`.clamp(-1,1)`'d ratio is dimensionless and is the **better** guard: indeterminate **iff** $J_2=0$
exactly, so a relative test is both correct and sufficient.

> 📌 **Safety code is written with fewer dimensional checks than physics code**, because it is treated as
> plumbing. ✅ **The dimensional gate therefore covers guard/threshold constants too — a bare numeric literal in
> any comparison against a physical quantity is a gate failure, whether or not it sits on the yield function.**

### 7cc.4 ⚠️ C-193 — only one bound of $g$ is constrained

⚠️ $g(-\pi/6)$ is **unconstrained**. A single $g$ matching MC's hexagon matches **one** bounding meridian and
carries the **full** deviation at the other. ✅ **Ruled: both $g(\pm\pi/6)$ and the inscribed/circumscribed choice
are declared inputs**, with the unmatched-meridian deviation measured and recorded.

### 7cc.5 🔴 C-194 — the "verified" Bardet citation fails the L-2 test

| field | submitted | verified 09-10-2026 (Crossref) |
|---|---|---|
| **DOI** | `10.1115/1.2892023` | ❌ **404** → **`10.1115/1.2897051`** |
| **Title** | *Lode angle function for the yield surfaces of soils and rocks* | ❌ no such record → ***Lode Dependences for Isotropic Pressure-Sensitive Elastoplastic Materials*** |
| Journal | J. Appl. Mech. | ✅ |
| Volume | 57 | ✅ **57** |
| **Issue** | 2 | ❌ **3** |
| Pages | 498–506 | ✅ **498–506** |
| Year | 1990 | ✅ |
| Author | Bardet, J. P. | ✅ |

🔴 **Two of eight fields wrong, and the DOI does not resolve.** ⚠️ *Six of eight were right — including the exact
page range — so any completeness or plausibility review **passes it**.* This is **L-2's failure mode repeating**
(`12242 → 12244`), now with a title corruption on top.

✅ **Ménétrey & Willam 1995 VERIFIED** — `10.14359/1132` **resolves 200**, *Triaxial Failure Criterion for Concrete
and its Generalization*, **ACI Structural Journal 92(3)**, 1995 ✅. ⚠️ Page range 311–318 unconfirmed from
Crossref — check in the retrieved copy. ✅ **L-7 narrowed to Bardet.**

### 7cc.6 📌 Standing CI requirement

🔴 **Third instance of the same mechanism** — a sign/index convention asserted in prose and never computed:
**C-161** ($\sqrt3$), **C-176** ($I_1$ sign), **C-191** ($\theta_L$).

$$\boxed{\text{Every }\pm\text{ constant in a constitutive expression gets a }#[cfg(test)]\text{ case that evaluates it at a state whose value is known independently.}}$$

✅ This is now a **standing requirement**, not a per-item fix. It is the general form of the limit-case suite, and
it is what caught all three.

---

## 7dd. theta_L locked at -pi/6; guard sound but unguarded; the §7s.6 rule (09-10-2026)

Corrections **C-195 … C-199**.

### 7dd.1 ✅ C-195 — $\theta_L=-\pi/6$, proven. C-191 CLOSED

Compression-positive triaxial compression $\sigma_1>\sigma_2=\sigma_3$:

$$s=\left(\tfrac23,-\tfrac13,-\tfrac13\right)\Delta\sigma,\qquad J_2=\tfrac13\Delta\sigma^2,\qquad J_3=+\tfrac{2}{27}\Delta\sigma^3$$

$$\sin3\theta_L=-\tfrac{3\sqrt3}{2}\cdot\frac{2/27}{(1/3)^{3/2}}=-1\;\Longrightarrow\;\boxed{\theta_L=-\tfrac{\pi}{6}}$$

✅ Verified analytically **and** numerically to full double precision. ✅ Extension side confirmed: $J_3<0\Rightarrow\theta_L=+\pi/6$.

$$\boxed{\;g(\theta_L)\equiv1\quad\text{at}\quad\theta_L=-\tfrac{\pi}{6}\;}$$

✅ **Pinned by a `#[cfg(test)]` case computing $\theta_L$ at a triaxial-compression state.**

> 📌 **$\pm$-convention trilogy complete:** **C-161** ($\sqrt3$), **C-176** ($I_1$ sign), **C-191** ($\theta_L$).
> Each was a sign/index asserted in prose; each was caught by evaluating at an independently-known state; each
> now has a test that keeps it caught. ✅ **Ruling 32 is the first round in which the engineer's derivation and my
> independent check agreed, with neither as the other's source.**

### 7dd.2 ✅🔴 C-196 — dimensional failure #9 fixed, but a new silent-wrong-answer path

$j_{2,\text{norm}}=J_2/\sigma_{\text{scale}}^2$ is **dimensionless** ✅ — the $10^{-12}$ literal is now meaningful.

| $\sigma_{\text{scale}}$ | $J_2$ at threshold | $\Delta\sigma$ at threshold | relative deviator |
|---|---|---|---|
| $10^{4}$ Pa | $10^{-4}\ \mathrm{Pa}^2$ | $1.73\times10^{-2}$ Pa | $1.73\times10^{-6}$ |
| $10^{7}$ Pa | $10^{2}\ \mathrm{Pa}^2$ | $17.3$ Pa | $1.73\times10^{-6}$ |
| $10^{8}$ Pa | $10^{4}\ \mathrm{Pa}^2$ | $173$ Pa | $1.73\times10^{-6}$ |

✅ **Direction of safety is correct**: f64 round-off leaves a "hydrostatic" state with $j_{2,\text{norm}}\sim10^{-32}$,
so the guard fires **before** the $0/0$, not after.

🔴 **(a) The fix guards the wrong thing and leaves its own denominator unguarded.** $\sigma_{\text{scale}}=0$ gives
`1.0/0.0 = inf` in Rust `f64` ⇒ `inf <= 1e-12` is **false** ⇒ **guard not taken** ⇒
`ratio = -1.5*sqrt(3)*j3/inf = 0` ⇒ $\theta_L=0$ ⇒ **a silently wrong mid-plane value, with no error raised.**
✅ **Ruled: validate `sigma_scale_pa > 0` (and finite) _before_ the division, and fail loudly per INV-1.**

⚠️ **(b)** A negative $J_2$ (f64 cancellation) is caught only **incidentally**, because the inequality happens to
accept negatives. ✅ Add an explicit $j_2\ge0$ test rather than relying on that.

🔴 **(c) `ValidityWarning::LodeAngleUndefined` is STILL MISSING** — see §7dd.4.

### 7dd.3 🔴 C-197 — free anchors can produce a re-entrant cap

⚠️ $g(-\pi/6)=0.4,\ g(+\pi/6)=1.6$ gives a **radially inverted, hourglass cap**: non-convex, and capable of
**negative plastic dissipation**, at which point the **CPPM return mapping is not well posed**.

$$\boxed{\text{Validate convexity and positive dissipation of the configured cap, then FAIL LOUDLY (INV-1).}}$$

🔴 **It may _not_ clamp, reorder, or silently substitute $g\equiv1$** — under INV-7 the input is unconstrained,
so the engine's response is a loud failure, not a repair.

⚠️ And because the anchors are now **inputs**, C-191's primacy is no longer a constant:
✅ **the _default_ is the measured $-\pi/6$ anchor, written to the manifest and pinned by a test**, so a run cannot
silently make a different meridian exact.

### 7dd.4 ✅ C-198 — L-7 CLOSED; both gates endorsed

✅ Bardet 1990 now matches the verified record field for field: *Lode Dependences for Isotropic
Pressure-Sensitive Elastoplastic Materials*, **ASME J. Appl. Mech. 57(3)**, 498–506, **`10.1115/1.2897051`**.
✅ Ménétrey & Willam 1995 verified (`10.14359/1132`). ⚠️ Only residual: pp. 311–318 unconfirmed → **L-8**.

⚠️ **Gate 2 is largely subsumed by the C-124 newtype discipline**: with `#[repr(transparent)]` newtypes the
*compiler* already rejects a dimensional comparison against a bare literal. ✅ **Its real value is the case where
the newtype discipline is broken** (raw `f64` leaking in), so it must also catch `f64 × f64` mixing — not only
literal comparisons.

### 7dd.5 🔴 §7s.6 — a singularity guard and its warning are ONE unit

🔴 `ValidityWarning::LodeAngleUndefined` was ruled in **C-189**, re-stated in **C-188**, and is **still absent**
after three revisions of the same block. The guard returns $1.0$ **silently**.

$$\boxed{\textbf{§7s.6\ \ \text{Any singularity guard that substitutes a value MUST emit its paired }\texttt{ValidityWarning}\text{.}}$$

⚠️ **This is error class B** — *making the failure disappear rather than be reported* — but occurring in the
**specification** rather than in code, with the same structure as **C-93** (clamp) and **C-151** ($\Delta t$
cutback / fallback solver).

📌 **And the omission mechanism is predictable, which is why it must be ruled against:** each rewrite was
responding to a *different* finding and **replaced the block wholesale**, discarding the previous round's
requirement. 🔴 **This is invisible to any check that only verifies "is the newest finding fixed?"**

$$\boxed{\text{Edits to a guarded expression are \textbf{cumulative}; its §7s checklist is re-verified on every touch.}}$$

> 📌 **Note what has changed in kind.** The physics of this component is now settled enough that the remaining
> defects are in **how corrections are carried forward** — a different failure class from the one that produced
> C-161 … C-197, and arguably the one more likely to recur.

---

## 7ee. Guard accepted; the two new requirements are wrong; §7s.6 generalised (09-10-2026)

Corrections **C-200 … C-204**.

### 7ee.1 🔴 C-200 — the $J_2$ formula is garbled, and its failure mechanism is unreachable

⚠️ Submitted: $\tfrac16[(\sigma_{11}-\sigma_{22})^2+(\sigma_{23})^2+(\sigma_{31})^2]+\sigma_{12}^2+\sigma_{23}^2+\sigma_{31}^2$
— 🔴 **the three shear terms appear twice.**

$$\boxed{J_2=\tfrac12\,s_{ij}s_{ij}\;\equiv\;\tfrac16\sum_{\text{cyc}}(\sigma_i-\sigma_j)^2}$$

🔴 **$J_2<0$ cannot occur.** Both routes are sums of squares; summing non-negative terms cannot go negative,
and overflow gives $+\infty$. Measured: **0 negatives in 200 000 pseudo-random states.**
✅ **Keep the check** — it is real insurance against a *badly implemented* $J_2$ — 🔴 **but do not record the
f64-truncation mechanism as a risk**, or the register grows a phantom entry (**C-178**'s lesson).

### 7ee.2 🔴 C-201 — the leak is silent **NaN**, not a bounded wrong angle

Rust's `f64::clamp` is `if self<min{min} else if self>max{max} else{self}` — with `self=NaN$, **both
comparisons are false** $\Rightarrow$ returns **`self`** = **NaN**.

$$0/0\Rightarrow\theta_L=\mathrm{NaN}\Rightarrow g=\mathrm{NaN}\Rightarrow f=\mathrm{NaN}\Rightarrow\mathbf{D}^{alg}=\mathrm{NaN}\Rightarrow R=\mathrm{NaN}$$

🔴 **And NaN compares `false` against _every_ guard**, so each downstream check **silently passes**.

$$\boxed{\text{The test must be a NaN-propagation assertion (}\texttt{is\_finite()}\text{ on }f\text{ and }\mathbf{D}^{alg}\text{), NOT a range check.}}$$

### 7ee.3 🔴🔴 C-202 — the dissipation inequality is SIGN-INVERTED

Submitted: $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p\,\mathbf{<}\,0$.
✅ **Correct: $\ge0$.** With $\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,\partial f/\partial\boldsymbol\sigma$,
$\boldsymbol\sigma:s=2J_2$, $\partial J_2/\partial\boldsymbol\sigma=s$:

$$\boldsymbol\sigma:\frac{\partial f}{\partial\boldsymbol\sigma}=\frac{J_2}{\sqrt{J_2/3}}-\alpha I_1=\sqrt{J_2/3}+f+k\;\Longrightarrow\;\text{on }f=0:\;\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,k\ge0$$

Measured at $\dot\gamma=10^{-6},10^{-3},1$ $\Rightarrow$ $+0.693,\ +692.8,\ +6.928\times10^5$ Pa — **all
non-negative**.

🔴 **A check written "$<0\Rightarrow$ error" fires on every plastic increment.** ⚠️ **And it matters more here
than in an associated model, because this spec uses _non-associated_ flow**, where $\dot\gamma k$ is not
guaranteed:

$$\boxed{\text{Bound }\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p-\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^{p,\text{elastic}}\ \text{(the dissipated part), and require it }\ge0.}$$

### 7ee.4 🔴🔴 C-203 — the convexity condition is wrong (misses the square AND $(g')^2$)

At fixed $I_1$: $r(\theta)=\frac{\alpha I_1+k}{g(\theta)}=\frac{C}{g}$. A polar curve is convex iff
$r^2-2rr''+6r'^2\ge0$. With $r=C/g$, $r'=-Cg'/g^2$, $r''=C(2g'^2-gg'')/g^3$:

$$\left(r^2-2rr''+6r'^2\right)\frac{g^4}{C^2}=\boxed{g^2+2(g')^2+2g\,g''\ \ge\ 0}$$

🔴 **Counterexample $g=1-\tfrac12\theta^2$:** submitted $g+g''=1-1=\mathbf{0}\ge0$ (**passes**) · correct
$=1+0-2=\mathbf{-1}<0$ (**fails**) — and $r=C/(1-\tfrac12\theta^2)$ widens with $|\theta|$: a **non-convex
peanut**. Second counterexample $g=\cos\theta$: submitted $\equiv0$ (passes), correct $=-1$ (fails), and
$r=C\sec\theta$ degenerates to the line $x=C$.

> 📌 **Fourth consecutive wrong convexity/monotonicity criterion in this component:** **C-93**, **C-181**,
> **C-187**, **C-203**. ✅ **All four were caught by _deriving from the geometry_ rather than recalling** — and
> all four recalled versions were confidently wrong, like **C-134**, **C-121**, **C-152**, **C-191**.
>
> $$\boxed{\text{Do not write an inequality from memory. Derive it from the defining geometry and commit the derivation.}}$$

### 7ee.5 🔴 C-204 — the ratio clamp is itself a silent clamp. §7s.6 generalised

⚠️ $\left|\tfrac{3\sqrt3J_3}{2J_2^{3/2}}\right|>1$ violates $|J_3|\le\tfrac{2}{3\sqrt3}J_2^{3/2}$ — the state is
**physically impossible** — yet the clamp saturates and the run proceeds on a **fabricated** Lode angle, with
**no measurement able to distinguish it afterwards**.

$$\boxed{\text{§7s.6 generalised: EVERY saturating transform is paired with its }\texttt{ValidityWarning}\text{.}}$$

⚠️ **The class — silent saturation — is precisely what produced C-93, C-151 and C-204.**

### 7ee.6 ✅ Closed this round

✅ **Validation order** (finite $\to\sigma_{\text{scale}}>0\to j_2\ge0\to$ relative guard $\to$ ratio) —
**C-196 CLOSED**. ✅ **`Result<(f64, Option<ValidityWarning>), TypedError>`** — the warning travels **with** the
value, so §7s.6 is structurally enforced — **C-199 CLOSED**. ✅ **INV-7 reading** (accept, never silently
substitute $g\equiv1$) ✅. ✅ **Compression-meridian primacy as default** ✅ (matches C-195). ✅ **Rule §7s.6
formalised with a CI build failure**, including the cumulative-edit clause 🔴 **which was exactly C-199's missing
second half** ✅. ✅ **Gate 2 expansion** — ⚠️ requires the C-124 layer to define **squared/exponent** newtypes
(`PascalsSquared`), or the second rule is unenforceable.

---

## 7ff. C-203 withdrawn; the dissipation check is right-signed but off-target (09-10-2026)

Corrections **C-205 … C-208**. 🔴 **§7ee.4 is superseded.**

### 7ff.1 🔴 C-205 — RETRACTION. $g^2+gg''\ge0$ is correct

The polar-curvature formula $\kappa=\dfrac{r^2+2(r')^2-rr''}{[r^2+(r')^2]^{3/2}}$ is the **standard** one, and with
$r=C/g$ the numerator reduces to $\frac{C^2}{g^4}[g^2+gg'']$ — **every step of the submission is right.**

🔴 **My $r^2-2rr''+6(r')^2\ge0$ is the criterion for convexity of $1/r^2$** — of the **polar-dual** body — a
different object. And **my "counterexample 1" was evaluated at $\theta=0$, where the curvature numerator is
exactly $0$**:

| $\theta$ | 0° | 10° | 20° | 30° |
|---|---|---|---|---|
| $\kappa_{num}$, $g=1-\tfrac12\theta^2$ | $-8.9\times10^{-5}$ | $-1.60\times10^{-2}$ | $-7.35\times10^{-2}$ | $\mathbf{-2.13\times10^{-1}}$ |

✅ **Criterion A catches it.** And $g=\cos\theta$ gives $r=C\sec\theta\Rightarrow x=C$ — a straight line, which
**is** a convex set; my criterion rejected a legitimate locus.

> 📌 **I did not make a sign or coefficient error. I chose the _wrong object_ and then derived faithfully** —
> every step internally consistent, no signal. Harder to catch than C-176, because a self-consistent derivation
> produces nothing to notice.
>
> $$\boxed{\text{(a) Name the object whose convexity is claimed, _then_ derive. \quad (b) A counterexample claimed over an interval must be SWEPT over it.}}$$

### 7ff.2 ⚠️ C-206 — the correct criterion admits degenerate loci

$g^2+gg''\ge0\Leftrightarrow\kappa\ge0$ means _the curve does not reverse curvature_. It therefore **accepts a flat
sub-interval** ($\kappa\equiv0$) — convex as a set, but **unphysical for a pressure-sensitive rock**, since
Mohr-Coulomb's $\pi$-plane is a **hexagon** with non-zero curvature except at its vertices.

✅ **Ruled: require $\kappa>0$ strictly, or $\kappa\ge\kappa_{\min}$ with $\kappa_{\min}$ declared.**
⚠️ **A refinement of a correct criterion — not a fix to a wrong one — and recorded as such so it is not re-litigated.**

### 7ff.3 🔴🔴 C-207 — C-202 stays OPEN: the check measures the wrong quantity

✅ §1A's diagnosis and the sign fix are correct. 🔴 **But $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p\ge0$ is
_necessary_ only.** Under non-associated flow the **dissipated** part is the mechanical work **minus the
hardening stored energy**:

$$\boxed{\dot D^p=\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p-\dot E^{p,\text{stored}}\ \ge0,\qquad \dot E^{p,\text{stored}}=H(\bar\varepsilon^p)^n\ \text{(declared)}}$$

🔴 **Measured — the failure is worst exactly at C-184's apex.** $\psi=0$ $\Rightarrow g=\sqrt{J_2/3}$ $\Rightarrow$
$\boldsymbol\sigma:\partial g/\partial\boldsymbol\sigma=\sqrt{J_2/3}$; linear hardening $H=3.2\times10^5$ Pa,
$\bar\varepsilon_p^{peak}=0.05$, $\dot\gamma=1$:

| deviator | $\sqrt{J_2/3}$ | $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p$ | submitted $\ge0$ | true $\dot D^p\ge0$ |
|---|---|---|---|---|
| $2.0$ | $6.67\times10^{-1}$ | $6.67\times10^{-1}$ | **PASS** | **FAIL** |
| $2.0\times10^{-4}$ | $6.67\times10^{-5}$ | $6.67\times10^{-5}$ | **PASS** | **FAIL** |
| $0$ (**apex**) | $0$ | $\mathbf{0}$ | **PASS** | **FAIL** |

📌 **At the apex the mechanical work vanishes, so the submitted check passes _with equality_ while the true
dissipation is $-\dot E^{p,\text{stored}}=-1.6\times10^4$ Pa — negative across the entire near-apex band.**
🔴 **The stored-energy term cannot be omitted: at the apex it is the _only_ term left.** Magnitude $2.3\%$ of
cohesion — ⚠️ **not a rounding effect, a sign error in the energy budget.**

✅ **Ruled: `TypedError::NegativePlasticDissipation` fires on $\dot D^p<0$, not on $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p<0$.**

### 7ff.4 ✅ C-208 — C-200 narrowed, C-201 CLOSED, §7s.6 generalised

✅ The §4A tensor formula is **correct** — 🔴 **the "garbled formula" half of my C-200 is withdrawn**; it had
already been corrected. ✅ **The unreachability half stands**: $J_2<0$ cannot arise (**0/200 000**), check retained
as defensive code, **no phantom physical risk recorded**.
✅ **`is_finite()` gating accepted verbatim** — **C-201 CLOSED**; ⚠️ extend to $\dot\gamma$ and $\boldsymbol\sigma$.
✅ **§7s.6 generalised to every saturating transform**, including
`DeviatoricStateSaturated { raw_value, clamped_value }` and the static CI build failure.

> 📌 **C-205 and C-207 are the same failure from opposite directions:** C-205 replaced a **correct** criterion with
> a wrong one after **one** evaluation point; C-207 is a submitted criterion right in **sign** but wrong in
> **what it measures**. 🔴 **Both are syntactically fine and semantically off-target** — which is precisely what a
> test written from the **derivation** (forcing the object to be named) catches, and what a test written from the
> **recalled formula** does not.

---

## 7gg. Convexity confirmed; the stored-energy term uses the wrong rate variable (09-10-2026)

Corrections **C-209 … C-213**.

### 7gg.1 ⚠️ C-209 — $N_\kappa(0)=0$, and it is the flat spot

$g=1-\tfrac12\theta^2$: $g(0)=1$, $g''(0)=-1$ $\Rightarrow N_\kappa(0)=1+1(-1)=\mathbf{0}$ — **not $1.0$**.

| $\theta$ | 0° | 15° | 30° |
|---|---|---|---|
| $N_\kappa$ | $+0.000000$ | $-0.033095$ | $-0.118288$ |

📌 **This strengthens C-206 rather than weakening it:** the floor $N_\kappa\ge\kappa_{\min}>0$ rejects this $g$
at **$\theta_L=0$** — 🔴 **and $\theta_L=0$ is the compression meridian that C-195 pins as primary.** ⚠️ An error
message quoting only $\pi/6$ would misdirect the diagnosis.
✅ **Ruled: report the failing $\theta_L$, not just the verdict.**

### 7gg.2 🔴🔴 C-210 — the floor is on $N_\kappa$ but named $\kappa$. 10th dimensional failure

$$\kappa=\frac{(C^2/g^4)\,N_\kappa}{[r^2+(r')^2]^{3/2}},\qquad C=\alpha I_1+k\ \ [\mathrm{Pa}]$$

📌 **$N_\kappa$ is dimensionless and stress-independent. $\kappa$ is $\mathrm{Pa}^{-1}$ and scales as
$C^2\sim(\text{stress})^2$**, and $C=\alpha I_1+k$ spans $[0,2k]$ across the locus.

🔴 **A floor stated on $\kappa$ is a _different check at every $I_1$_** — it rejects weak states and passes strong
ones for the **same** $g$.

$$\boxed{\text{Floor on }N_\kappa\text{, named }\texttt{MIN\_CURVATURE\_NUMERATOR}\text{; a true curvature floor must be normalised, }\kappa/\kappa_{MC}\text{.}}$$

📌 **The most consequential dimensional failure: not a missing unit, but _two quantities sharing one symbol_.**
**C-161 / C-176 / C-191 family** — and it survived three rounds because each occurrence read as stylistic.

### 7gg.3 🔴🔴 C-211 — the stored term uses $\dot\gamma$, but $H$ is defined against $\dot{\bar\varepsilon}_p$

`stored_energy = H * eps_p * delta_gamma` 🔴 **is off by** $\mu=\dot{\bar\varepsilon}_p/\dot\gamma$.

$$\boxed{\dot E^{p,\text{stored}}=H(\bar\varepsilon_p)\,\bar\varepsilon_p\,\dot{\bar\varepsilon}_p\qquad\text{(linear }H\text{)}}$$

| $\|\partial g/\partial\boldsymbol\sigma\|$ | $\mu$ | $H\bar\varepsilon_p\mu$ |
|---|---|---|
| 0.5 | 2.0 | $3.2\times10^{4}$ Pa |
| 1 | 1.0 | $1.6\times10^{4}$ Pa |
| 4 | 0.25 | $4.0\times10^{3}$ Pa |

📌 **At the apex the mechanical work is $0$, so the stored term _alone_ decides the sign** — and it is off by
$1/\mu$, a factor of 2–4. $\dot D^p$ at the apex: $-4.0\times10^{3}$ … $\mathbf{-1.6\times10^{4}}$ … $-3.2\times10^{4}$ Pa
= **0.58 %–4.62 % of cohesion.** ✅ **Ruled: $\mu$ declared, written to the manifest.**

### 7gg.4 ⚠️ C-212 — two slips in §2A

🔴 **(a)** $\rho\frac{\partial\psi^p}{\partial\bar\varepsilon_p}=H$ (**not** $H\bar\varepsilon_p$ — that is
$\psi^p$ itself). ✅ The final boxed form is correct for linear $H$; only the intermediate equality is wrong.
🔴 **(b)** *"$\boldsymbol\sigma\to-p\mathbf I$"* — under **compression-positive** (**C-176/C-184**) it is
$\boldsymbol\sigma\to\mathbf+p\mathbf I$. ⚠️ **Ninth compression/tension sign slip in this component**, and the last
three survived into submitted text.

### 7gg.5 ⚠️ C-213 — provenance and an ungated claim

⚠️ $\kappa_{\min}=10^{-4}$ is a bare literal: ✅ **keep it**, but ✅ **declare it, and record it
`SOURCE_PENDING`** — 📌 the same class as **$\delta$ (L-5)** and the **$\alpha_0$ table (L-1)**: a number that
works and whose origin nobody can cite. → **L-9**.
⚠️ The CPPM flat-facet flip-flop claim is **plausible and unrefuted, but recalled**: ✅ **Ruled: gate it** — a
synthetic flat-facet $g$ run through CPPM with sub-iteration count recorded.

### 7gg.6 📌 Four rounds of prose/code divergence

**C-164** (coefficients), **C-167** (RF balance), and now **C-211** — ⚠️ **the prose is right and the code in the
same message contradicts it.** 📌 **Ruled: submitted code blocks are checked against the submitted prose
explicitly; a `#[cfg(test)]` case per code block, not per formula.**

---

## 7hh. mu is state-dependent; the code-block test cannot fail; kappa = 1/C (09-10-2026)

Corrections **C-214 … C-216**.

### 7hh.1 ⚠️ C-214 — my $\kappa$ scaling exponent was wrong

Measured with $g\equiv1$ ($r=C$):

| $C$ [Pa] | $\kappa$ [Pa⁻¹] | $C\kappa$ |
|---|---|---|
| $10^{5}$ | $10^{-5}$ | $1.000000$ |
| $10^{7}$ | $10^{-7}$ | $1.000000$ |

✅ **$\kappa\propto1/C$ — the submission is right.** 🔴 **I compared the _numerator_'s $C^2$ without dividing out
the denominator, which itself scales as $C^3$.** ⚠️ **C-210's _conclusion_ stands** (the floor must be on the
dimensionless $N_\kappa$); **only the exponent was wrong** — in the direction that flattered my own argument.

> 📌 **A conclusion can be right while the argument for it is wrong** — the mirror of **C-176**. 🔴 Reading a
> conclusion cannot distinguish the two; only a direct measurement can.

### 7hh.2 🔴🔴 C-215 — $\mu$ is state-dependent, and the declared formula is wrong

With $\partial\bar\varepsilon_p/\partial\boldsymbol\sigma=s/2\bar\varepsilon_p$ and
$\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,\partial g/\partial\boldsymbol\sigma$:

$$\boxed{\mu\equiv\frac{\dot{\bar\varepsilon}_p}{\dot\gamma}=\frac{J_2}{\bar\varepsilon_p\,\left\lVert\partial g/\partial\boldsymbol\sigma\right\rVert}}\qquad\text{(dimensionless ✓)}$$

🔴 **Not $\lVert\partial g/\partial\boldsymbol\sigma\rVert$ alone.** Measured along one pure-shear path at
yield, $\psi=0$ ($\lVert\partial g/\partial\boldsymbol\sigma\rVert$ pinned at $1.2247$):

| deviator | $\bar\varepsilon_p$ | $J_2$ | $\lVert\partial g/\partial\boldsymbol\sigma\rVert$ | $\mu$ |
|---|---|---|---|---|
| $2.0$ | $1.633$ | $1.3333$ | $1.2247$ | $0.6667$ |
| $6\times10^{-2}$ | $4.899\times10^{-2}$ | $1.2\times10^{-3}$ | $1.2247$ | $0.0200$ |
| $2\times10^{-3}$ | $1.633\times10^{-3}$ | $1.2\times10^{-6}$ | $1.2247$ | $\mathbf{0.000667}$ |

📌 **Three orders of magnitude along one path with the same flow rule.**
🔴 **So "write $\mu$ to the manifest" is not implementable as one number** — ✅ **Ruled: computed per increment and
recorded per increment, or its bounds recorded.**
🔴 **And the quoted range $\mu\in[0.25,2.0]$ excludes measured values by two orders** — **a bound stated without
its scale**, i.e. the dimensional-failure class again.

### 7hh.3 🔴 C-216 — the fifth check that cannot fail

✅ The **policy** (a `#[cfg(test)]` suite per submitted code block) is exactly right and is adopted.
🔴 **Test 2 hardcodes the factor it exists to verify:** `let flow_norm_mu = 0.5;` — 📌 it asserts that
$H\bar\varepsilon_p(0.5\dot\gamma)$ reproduces $8000$, and **never computes $\mu$ from a stress state**, so it
**cannot detect the wrong $\mu$ formula that C-215 just found.**
⚠️ Test 1's `assert_eq!(n_kappa, 0.0)` is **exact float equality** — passes in f64 ✅, breaks under FMA
reassociation. ✅ **Use `abs_diff_eq!`.**

$$\boxed{\text{Test 2 must compute }\mu=\frac{J_2}{\bar\varepsilon_p\lVert\partial g/\partial\boldsymbol\sigma\rVert}\text{ from a real state, compute }\dot{\bar\varepsilon}_p\text{ independently, and compare.}}$$

📌 **The two sides from independent routes** — the **C-174** rule.

### 7hh.4 📌 Five checks that cannot fail — one mechanism

| | finding |
|---|---|
| **C-134** | Kahan summation |
| **C-152** | $\Delta t$-cancellation |
| **C-143** | Tier-A normaliser |
| **C-174** | RF equivalence |
| **C-216** | $\mu$ factor |

✅ **All five: the assertion's inputs were derived from, or chosen to satisfy, the quantity being asserted.**
✅ **The §5B code-block test policy is the right countermeasure — provided each test draws its two sides from
independent routes.**

---

## 7ii. B-7b.1–4 resolved against the submitted answers (09-10-2026)

Corrections **C-217 … C-220**. ✅ **B-7b.2 and B-7b.4 close.** 🔴 **B-7b.1 and B-7b.3 stay open** — and
in both cases the submitted fix is **off by a constant factor that changes the sign of a verdict**.

### 7ii.1 🔴🔴 C-217 — the $\mu$ derivation has a factor-3 error in the gradient; $\mu\equiv1$, exactly

⚠️ **The submitted derivation is structurally right and arithmetically wrong.**

**Step 1 — the gradient is wrong by a factor of 3.** Claimed
$\dfrac{\partial g}{\partial\boldsymbol\sigma}=\dfrac{\mathbf{s}}{2\sqrt{3J_2}}-\alpha_\psi\mathbf I$. ✅ Measured:

$$\frac{\mathrm{d}}{\mathrm{d}(J_2/3)}\sqrt{J_2/3}=\frac{1}{2\sqrt{J_2/3}}=\frac{\sqrt3}{2\sqrt{J_2}}\qquad\text{vs claimed}\qquad\frac{1}{2\sqrt3\,\sqrt{J_2}}$$

📌 **Ratio exactly $3.000000$** — the same factor-3 class as **C-164**.

**Step 2 — the contraction inherits it.** $\mathbf{s}:\mathbf{I}=0$ so the cross terms vanish, and the
squared term is $\frac{2J_2}{4J_2/3}=\mathbf{\tfrac32}$, not $\tfrac16$. Measured side by side:

| | $J_2=1.0$ | $J_2=0.25$ | $J_2=0.01$ |
|---|---|---|---|
| claimed $\partial g:\partial g$ | $0.166667$ | $0.166667$ | $0.166667$ |
| **correct** | **$1.500000$** | **$1.500000$ | **$1.500000$** |

**Step 3 — and the submitted $\mu$ is not merely mis-scaled, it is the wrong quantity.** 📌
$\bar\varepsilon_p=\sqrt{\tfrac23\varepsilon_p^{\mathrm{dev}}:\varepsilon_p^{\mathrm{dev}}}$, so
$\dot{\bar\varepsilon}_p$ sees only the **deviatoric** part of $\partial g/\partial\boldsymbol\sigma$ — and
$-\alpha_\psi\mathbf I$ is **purely volumetric**, so dilatancy **cannot enter $\mu$ at all**:

$$\boxed{\;\mu\equiv\frac{\dot{\bar\varepsilon}_p}{\dot\gamma}=\sqrt{\tfrac23\cdot\frac{2J_2}{4J_2/3}}=\sqrt{\tfrac23\cdot\tfrac32}=1\;}\qquad\text{for all }J_2,\ \text{for all }\alpha_\psi$$

Measured, $\mu_{\text{true}}=1.0000000000$ at $J_2\in\{1,\ 0.25,\ 0.04,\ 10^{-3}\}$ and
$\alpha_\psi\in\{0,\ 0.15,\ 0.40\}$ — **every combination.** ⚠️ Submitted values: $0.333$, $0.395$, $0.657$.

✅ **Ruled:**
- **$\mu=1$ exactly. It is not a computed quantity and must not be computed** — it is a definition.
- **$\dot{\bar\varepsilon}_p=\dot\gamma$**, so ✅ **the stored-energy term $\dot E^{p,\text{stored}}=H(\bar\varepsilon_p)\bar\varepsilon_p\dot\gamma$ uses $\dot\gamma$ directly** — which is what the submitted code already did.
- 🔴 **This WITHDRAWS my own C-211.** I ruled that $\dot\gamma$ was "a factor $2$–$4$ error" and that $\mu$ "spans three orders of magnitude along one path". ⚠️ **That was wrong twice**: I put $\bar\varepsilon_p$ in a denominator where it has no business being (making $\mu$ state-dependent by construction), and I never checked the definition that makes it identically $1$. ✅ **C-211 retracted; C-215's state-dependence claim retracted with it.**
- 📌 *Independent confirmation:* for any potential whose **deviatoric part is a function of $J_2$ alone**, $\dot{\bar\varepsilon}_p=\dot\gamma$ — the classical von Mises result. DP's deviatoric part is $\sqrt{J_2/3}$, so it applies.

### 7ii.2 🔴 C-218 — the apex is NOT automatically a dissipation failure (C-207 was $\alpha_\psi=0$-only)

My **C-207** measured $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p\to0$ at the apex. ✅ **That is correct
only when $\alpha_\psi=0$.** With dilatancy:

$$\boldsymbol\sigma:\frac{\partial g}{\partial\boldsymbol\sigma}=\sqrt{J_2/3}-\alpha_\psi I_1\ \xrightarrow[\text{apex}]{}\ -\alpha_\psi\!\left(-3c\cot\phi\right)=\mathbf{3\alpha_\psi c\cot\phi}$$

Measured at $\phi=30°$, $c=10^6$, apex $I_1=-5.1962\times10^6$: $\alpha_\psi=0\to0$; $0.05\to+2.60\times10^5$;
$0.15\to+7.79\times10^5$; $0.40\to+2.08\times10^6$ Pa — **all positive and non-zero.**

$$\boxed{\;\dot D^p=\dot\gamma\left(3\alpha_\psi c\cot\phi-H\bar\varepsilon_p\right)\quad\text{— a _comparison_, not an automatic failure}\;}$$

⚠️ **The gate is still required** — it is precisely what evaluates that comparison. But ⚠️ **its failure
condition is a window, not a rule:** negative $\dot D^p$ requires $\alpha_\psi$ *below* a threshold that
**falls as $\phi$ and $H$ rise** — measured $\alpha_\psi^{*}=0.00143/\ 0.00308/\ 0.00533$ at
$\phi=15°/30°/45°$ for $H\bar\varepsilon_p=1.6\times10^4$ Pa. ✅ **The test must be run at $\alpha_\psi=0$
to hit the failure branch; at realistic dilatancy it passes.** 📌 *This is why the earlier statement "the apex
is a guaranteed dissipation failure" was too strong — right conclusion, over-general scope.*

### 7ii.3 ✅ C-219a — B-7b.2 CLOSED, and its physical-curvature argument is confirmed

✅ Enforce the floor on the **dimensionless, stress-independent** $N_\kappa=g^2+gg''$; rename the constant
`MIN_CURVATURE_NUMERATOR`; log $\kappa/\kappa_{MC}$ if a normalised curvature is reported. ✅ **Exactly the
ruling, and the $1/C$ stress-scaling rationale is correct** (measured `7hh`: $C\kappa\equiv1.000000$).
✅ **B-7b.2 CLOSED.**

⚠️ **One implementation note on the submitted sweep:**

```rust
let step = std::f64::consts::FRAC_PI_3 / (num_samples as f64);
let theta = -FRAC_PI_6 + (i as f64) * step;   // i in 0..=num_samples
```

✅ The span is correct ($-\pi/6\to+\pi/6$) — 📌 but `step` divides by `num_samples` while the loop runs to
`num_samples` **inclusive**, giving $181$ samples over $180$ intervals ⚠️ harmless, yet ✅ **write it as
`step = (2.0*FRAC_PI_6) / num_samples as f64` so the endpoints are exact and the count is obvious.**

### 7ii.4 🔴 C-219 — B-7b.3 STAYS OPEN: renaming does not fix a wrong coefficient

⚠️ **The submitted `MohrCoulombExtension` uses $\alpha=\frac{2\sin\phi}{3(3+\sin\phi)}$,
$k=\frac{2c\cos\phi}{3+\sin\phi}$ — the same coefficients previously shown to be von Mises, now under a
name asserting they are Mohr-Coulomb.**

Measured: $k=4.948717\times10^5$ Pa, and the true MC triaxial extension at $P=0$ is $4.732051\times10^6$ Pa —
🔴 **$-89.5\%$**, worse than the $-66.67\%$ originally reported, because the comparison is now against a
correctly-derived MC line rather than the earlier mistaken one. Solving MC exactly
($T/2=c+\tfrac{T-P}{2}\tan\phi\Rightarrow T=\frac{2c-P\tan\phi}{1-\tan\phi}$) and comparing the DP branch
directly:

| $P$ | MC $T$ | DP $T$ | error |
|---|---|---|---|
| 0 | $4.732051\times10^6$ | $2.797032\times10^6$ | **$-40.89\%$** |
| $5\times10^5$ | $4.049038\times10^6$ | $2.566176\times10^6$ | **$-36.62\%$** |
| $10^6$ | $3.366025\times10^6$ | $2.335321\times10^6$ | **$-30.62\%$** |

📌 **Root cause: $3+\sin\phi$ is the _apex-matched_ form, not the MC _tangency_ fit.** A DP cone can match MC
on **one meridian** and not both — the **C-188** trade, unchanged. 🔴 **A name asserting Mohr-Coulomb
correspondence on a coefficient that does not achieve it is worse than an honest mislabel.**

✅ **Ruled — three options, and the spec must *choose*, not imply:**
1. **Delete the branch.** ⚠️ **Recommended** — one MC-compression branch, no extension claim.
2. **Derive the true MC-extension tangency** and use it — 📌 requires matching MC's *slope* as well as its value, which a single $(\alpha,k)$ pair with the $3+\sin\phi$ form cannot do. ⚠️ **Not derivable by relabelling.**
3. **Keep it as `VonMisesExtension`**, documented as **not** Mohr-Coulomb ✅ — **acceptable only with option 2's naming discipline applied to every branch.**

### 7ii.5 ✅ C-220 — B-7b.4 CLOSED

✅ Pre-run sweep of $N_\kappa$ over $[-\pi/6,+\pi/6]$; $\texttt{Err(NonConvexYieldSurface \{ theta\_lode, n\_kappa \})}$ at
initialisation under **INV-1**; ✅ **no silent clamping** under **INV-7** — the engine never adjusts $g$ to
force a non-convex locus to run. ✅ `ValidityWarning::HardeningMonotonicityClampActive { delta, ratio, c1_residual }` ✅ and
suspension of the $\mathcal{O}(\|R\|^2)$ claim with damped line search ✅. **B-7b.4 CLOSED.**

### Blockers after this round

| # | State |
|---|---|
| **B-7b.1** | ⚠️ **RESOLVED, differently.** $\mu\equiv1$ (**C-217**) — so the stored term uses $\dot\gamma$ and **C-211 is withdrawn**. ⚠️ **New residual:** the apex dissipation gate must be exercised at $\alpha_\psi=0$ to reach its failure branch (**C-218**) |
| **B-7b.2** | ✅ **CLOSED** (**C-219a**) |
| **B-7b.3** | 🔴 **STILL OPEN** (**C-219**) — rename does not fix the coefficient |
| **B-7b.4** | ✅ **CLOSED** (**C-220**) |
| **L-9** | ⚪ confirmed non-blocking — $N_\kappa\equiv1\gg10^{-4}$ for a circular cone ✅ |

> 📌 **Two of my own findings are retracted in this round, and both were the same error: I asserted a property
> of a quantity without first asking what makes it constant.** $\mu$ I treated as state-dependent because
> I had put $\bar\varepsilon_p$ in a denominator myself; the apex I called a guaranteed failure because I had
> measured it at $\alpha_\psi=0$ and generalised. ✅ **The correction in both cases came from deriving the
> quantity's _definition_ before measuring it** — the same discipline that produced C-176 and C-205, and the
> same one I keep failing to apply to my own claims first.
>
> ⚠️ **And C-219 is the fifth factor-3 in this component** — **C-164**, **C-205** (reversed), **C-211**
> (reversed), **C-214**, **C-217**. 📌 **The $\sqrt3$ and the factor-3 in $J_2$-derivatives are the same
> recurring attractor**, and ✅ **the remedy is unchanged and unenforced: no scalar factor in a constitutive
> expression is typed by hand; the limit-case suite asserts it (`7cc.6`).**

---

## 7jj. B-7b.3 CLOSED; the μ proof verified in a second parameterisation (09-10-2026)

Corrections **C-221 … C-224**. ✅ **All four blockers are now closed.** ⚠️ Two naming and sign details found.

### 7jj.1 ✅ C-221 — the μ proof verified independently, and it is CORRECT

Re-derived in the $\bar\sigma=\sqrt{3J_2}$ parameterisation, which is where the previous answer went wrong.

✅ **Gradient confirmed numerically** — $g=\sqrt{3J_2}-3\alpha_\psi P-k$ at $\boldsymbol\sigma=(2,-0.5,-1.5)$, $\alpha_\psi=0.15$:

| | $\partial_1$ | $\partial_2$ | $\partial_3$ |
|---|---|---|---|
| numerical difference | $+0.810768923$ | $-0.390192231$ | $-0.870576692$ |
| analytic $\tfrac{3}{2\bar\sigma}\mathbf s-\alpha_\psi\mathbf I$ | $+0.810768923$ | $-0.390192231$ | $-0.870576692$ |

✅ **Identity $2J_2=\tfrac23\bar\sigma^2$ confirmed** at $J_2=1,\,0.25,\,4$ to $10^{-14}$. ✅ **Deviatoric contraction** $\tfrac{9}{4\bar\sigma^2}\cdot\tfrac23\bar\sigma^2=\tfrac32$ ✓, giving $\mu=\sqrt{\tfrac23\cdot\tfrac32}=\mathbf{1.0000000000}$.

📌 **So $\mu\equiv1$ by two independent parameterisations, one of them the one that previously produced a
factor-3 error.** ✅ **B-7b.1 fully closed**; ✅ `dot(E)^p_stored = H(eps_bar) * eps_bar * dot(gamma)` confirmed exact;
✅ **C-211 and C-215 stay withdrawn.**

⚠️ **And this names the real C-217 defect precisely.** Measured: the $\sqrt{J_2/3}$ and $\sqrt{3J_2}$ forms give
**identical** gradients (ratio $1.000000000$ per component). 🔴 **The earlier submission took the
$\sqrt{J_2/3}$ potential but supplied the gradient of $\sqrt{3J_2}$ — two different stress measures in one
expression.** 📌 *That is why the factor 3 appeared, and it is a cleaner statement than "the gradient is off by
3": the potential and its derivative came from different definitions.*

### 7jj.2 ✅ C-222 — `ApexMatchedExtension` is an HONEST name. **B-7b.3 CLOSED**

✅ **Verified: the name describes exactly what the coefficients do.** Setting $F=0$ at the apex gives
$k=3\alpha c\cot\phi$, and the submitted $k=\frac{2c\cos\phi}{3+\sin\phi}$ satisfies it to
**$1.0000000000$** at $\phi=15°,\,30°,\,45°$:

| $\phi$ | $\alpha$ | $k$ required | $k$ formula | ratio |
|---|---|---|---|---|
| 15° | 0.05294741 | $5.928073\times10^5$ | $5.928073\times10^5$ | $1.0000000000$ |
| 30° | 0.09523810 | $4.948717\times10^5$ | $4.948717\times10^5$ | $1.0000000000$ |
| 45° | 0.12716238 | $3.814871\times10^5$ | $3.814871\times10^5$ | $1.0000000000$ |

📌 **Both branches are apex-matched** — measured the compression branch likewise at $1.0000000000$.
⚠️ **So the real distinction is not "MC vs apex-matched" but "apex-matched _and additionally tangent to MC_".** ✅ **Only the compression branch is both**, and the three-way naming now says exactly that. 🔴 **This closes B-7b.3 — and it closes it by making the spec honest, not by changing a coefficient**, which was the only option that did not require inventing an MC-extension tangency.

### 7jj.3 🔴 C-223 — a sign error in the apex stress tensor that cancels in the product

⚠️ Submitted: *"the stress tensor ... is $\boldsymbol\sigma=+p\mathbf I=+(c\cot\phi)\mathbf I$"*.

🔴 **Under the locked compression-positive convention (`7aa`), $p=I_1/3=-c\cot\phi=-1.732051\times10^6$ Pa** —
the apex is hydrostatic **tension** — so the expression is $\boldsymbol\sigma=\mathbf{-}c\cot\phi\,\mathbf I$.
Measured $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p/\dot\gamma=3pa$: $0$, $-7.794229\times10^5$,
$-2.078461\times10^6$ Pa at $\alpha_\psi=0,\,0.15,\,0.40$.

✅ **But the final formula $\dot D^p=\dot\gamma\left(3\alpha_\psi c\cot\phi-H\bar\varepsilon_p\right)$ is CORRECT** and matches my **C-218** exactly — ⚠️ **the sign error cancels in the product, so the right answer survives a wrong intermediate.** 📌 *This is the **tenth** compression/tension sign slip in this component.*

✅ **Ruled: $\boldsymbol\sigma_{\text{apex}}=\mathbf{-}c\cot\phi\,\mathbf I$ under compression-positive.**
⚠️ Note the resulting $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p$ is **negative** at the apex with
dilatancy — ✅ **which strengthens C-218**: the dissipation there is a *three-way* competition between a
**negative** mechanical-work term and a positive stored term, not the simple offset previously stated.

### 7jj.4 ⚠️ C-224 — `VonMisesExtension` still asserts a correspondence it does not have

$\alpha=0,\ k=2c/\sqrt3\Rightarrow F=\sqrt{J_2/3}-k$, a cylinder. Measured pure-shear yield:
$\tau=k\sqrt3=2c=2.000000\times10^6$ Pa, against $\tau=c=1.000000\times10^6$ Pa for Tresca at $\phi=0$.

🔴 **So it is not the von Mises match to Tresca either.** It is self-consistent — $\sqrt{J_2/3}=k$ — but the
name asserts a correspondence that does not hold. ✅ **Ruled: rename `J2EquivalentCylinder`**, or document
$\tau_{yield}=2c$ in the branch's doc comment.

📌 **The naming discipline that worked here is worth stating as a rule:** ✅ *every `MeridianType` variant must
name the property its coefficients actually have, verified by a test that evaluates that property.* C-222 is
the proof it works — one honest name replaced an incorrect one without touching a number.

### Blockers — ALL CLOSED

| # | State |
|---|---|
| **B-7b.1** | ✅ **CLOSED** — $\mu\equiv1$ verified twice (**C-221**). ⚠️ apex sign corrected (**C-223**) |
| **B-7b.2** | ✅ **CLOSED** (**C-219a**), endpoint step aligned |
| **B-7b.3** | ✅ **CLOSED** (**C-222**) — resolved by naming, not by changing a coefficient |
| **B-7b.4** | ✅ **CLOSED** (**C-220**) |
| **C-224** | ⚠️ **new, non-blocking** — `VonMisesExtension` → `J2EquivalentCylinder` |
| **L-9** | ⚪ non-blocking, confirmed |

🔴 **M7b now has: a locked constitutive model, four closed blockers, a 13-metric gate, and 8 tasks.**
⚠️ **Remaining open in M7b's dependency set is `CONF-18`** (Bethel correction, task 7b.7) and the absence of a
cited $k_{th}$ (task 7b.2) — 📌 **neither is a constitutive question.**

---

## 7kk. C-222/223/224 verified against the submitted answers (09-10-2026)

Corrections **C-225**. ✅ **C-222 and C-224 confirmed and accepted.** ⚠️ **C-223's own resolution contains a
sign slip — and correcting it strengthens the gate.**

### 7kk.1 ✅ C-222 — apex matching verified for **both** branches

Submitted algebra re-checked. $-k/\alpha$ for both branches reduces to $-3c\cot\phi$:

| branch | $\alpha$ | $k$ | $-k/\alpha$ | target |
|---|---|---|---|---|
| compression | $\frac{2\sin\phi}{3(3-\sin\phi)}$ | $\frac{2c\cos\phi}{3-\sin\phi}$ | $-3c\cot\phi$ | ✅ $1.0000000000$ |
| extension | $\frac{2\sin\phi}{3(3+\sin\phi)}$ | $\frac{2c\cos\phi}{3+\sin\phi}$ | $-3c\cot\phi$ | ✅ $1.0000000000$ |

✅ **Both are exact apex-matched cones terminating at the MC tensile vertex** $p_{\text{apex}}=-c\cot\phi$.
✅ **B-7b.3 CLOSED — and correctly, by naming rather than by altering a coefficient.**

### 7kk.2 ⚠️ C-225 — the cancellation is valid, but 🔴 **the sign in the resolution is wrong**

⚠️ The submitted resolution is careful and explicit: it derives
$\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p=(p\mathbf I):(-\dot\gamma\alpha_\psi\mathbf I)
=-3p\alpha_\psi\dot\gamma=+3\alpha_\psi c\cot\phi\,\dot\gamma$. ✅ **That chain is correct.**

🔴 **But my own C-223 stated the apex mechanical work is _negative_** ($-3\alpha_\psi c\cot\phi$), and I
carried that into the M7b gate table. ⚠️ **That was my error, not the submission's.**

Measured — the sign depends entirely on the volumetric term of the flow potential, which resolves the
ambiguity:

| flow potential at the apex | $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p/\dot\gamma$ |
|---|---|
| $+\alpha_\psi\mathbf I$ | $-7.794229\times10^5$ Pa |
| $\mathbf{-}\alpha_\psi\mathbf I$ | **$+7.794229\times10^5$ Pa** ✅ |

✅ The locked potential ($`7aa`$) carries $-\alpha_\psi I_1$, so $\partial g/\partial\boldsymbol\sigma\to
\mathbf{-}\alpha_\psi\mathbf I$ and the work is **positive**. Swept:

| $\alpha_\psi$ | 0.00 | 0.05 | 0.15 | 0.40 | 1.00 |
|---|---|---|---|---|---|
| work [Pa] | $+0.000000$ | $+2.598076\times10^5$ | $+7.794229\times10^5$ | $+2.078461\times10^6$ | $+5.196152\times10^6$ |

📌 **Monotone in $\alpha_\psi$ and vanishing linearly at $0$ — so the two submitted statements are the _same
formula at two values of $\alpha_\psi$, not two separate cases.** ✅ *The resolution's physical reading is
therefore internally consistent, and my C-223 contradicted it.*

⚠️ **One thing the spec must now declare:** 📌 **the sign of the volumetric term of the flow potential.** The
$+3\alpha_\psi c\cot\phi$ term is only correct because the potential carries $-\alpha_\psi I_1$ — 🔴 **that sign
appears nowhere in the specification**, and writing the potential isochorically ($g=\sqrt{J_2/3}-k$) would
make the term exactly zero. ✅ **Ruled: the potential's volumetric term and its sign are declared inputs.**

✅ **Corrected gate semantics:** the apex dissipation is
$$\dot D^p=\dot\gamma\left(3\alpha_\psi c\cot\phi-H\bar\varepsilon_p\right)$$
with **positive** work offsetting positive storage — ⚠️ so $\dot D^p<0$ requires **both** small
$\alpha_\psi$ **and** large $H\bar\varepsilon_p$. 📌 *This is the "three-way competition" I described in C-223,
with the work term on the positive side.*

### 7kk.3 ✅ C-224 — $J_2=4c^2$ confirmed; one terminology note

✅ Verified exactly: $\sqrt{J_2/3}=2c/\sqrt3\Rightarrow J_2=4c^2=4.0000\times10^{12}$; pure shear $J_2=\tau^2$
$\Rightarrow\tau_{\text{yield}}=2c=2.0000\times10^6$ Pa — **against Tresca's $c$**, i.e. a **$2\times$ shear
envelope** ✅ as stated.

⚠️ **Terminology:** the doc line *"Unconfined $J_2$ cylinder with pure-shear yield $\tau=2c$"* is **accurate**,
but the *identifier* `J2EquivalentCylinder` still implies equivalence to something. ✅ Either rename to
`PlainJ2Cylinder`, or ✅ **keep the name and keep the doc line** — 📌 *the doc line is what satisfies the naming
rule (`7jj.4`), so the two together are compliant.*

### Status

🔴 **All four M7b blockers closed. CONF-66 closed.** ✅ New, non-blocking: declare the flow potential's
volumetric-term sign (**C-225**); optional identifier rename (**C-224**).
