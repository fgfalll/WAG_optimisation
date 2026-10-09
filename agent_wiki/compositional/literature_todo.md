# Literature Action List — NEEDS SOURCE

> [!CAUTION]
> **Status: INCOMPLETE. Every item below is unverified or unresolvable from local material.**
> This page exists so the remaining bibliography work is **actionable and bounded** rather than open-ended.
> **No item here may be cited in a constitutive relation until its acceptance test passes.**

**Compiled 08-10-2026 · updated 09-10-2026** (L-7 closed, L-8 and L-9 added; L-1 confirmed
unobtainable). Supersedes nothing; the authority for provenance remains
[`register_spec.md`](register_spec.md) and the acceptance standard is
[`engine_spec_closures.md`](engine_spec_closures.md) §7h.4.

> 📌 **This is the research backlog for the 🦀 Rust 3D THMC engine.** None of these items gate a
> milestone — ✅ **M7c must not be gated on L-1.**

---

## 1. Acceptance test — three conditions, all required

| # | Test | Tool |
|---|---|---|
| 1 | The DOI returns **HTTP 200** from `doi.org` | `https://doi.org/{DOI}` |
| 2 | Registered metadata **matches field-for-field** — authors, year, title, volume, issue, pages | DOI content negotiation |
| 3 | The **specific table / figure / coefficient** is reachable and matches the value used | the paper |

> ⚠️ **A claim of verification is not evidence.** A DOI string that merely *contains* `10.` passes a
> substring test and fails this one — that is exactly the gate defect recorded as **C-114**.

**Verified items are listed in §4** so they are not re-worked.

---

## 2. 🔴 BLOCKING — the highest-value item

### L-1 · Karakas & Tariq (1991) — the α₀ table

| Field | Value |
|---|---|
| **Status** | ✅ **CLOSED 09-10-2026 (C-226)** — Table 1 obtained and transcribed. 🔴 **It exposed a transcription error in my own §7h.2 and invalidated the fitted coefficient.** See §7h.2a |
| **DOI (verified, §4)** | `10.2118/18247-PA` — ✅ **re-verified against `api.crossref.org` 09-10-2026**: HTTP 200, title, journal, **6**(01), 73–82, 1991-02-01 all match. 91 citing references. |
| **Confirmed metadata** | Karakas, M. & **Tariq**, S. M. (1991), *Semianalytical Productivity Models for Perforated Completions*, SPE Production Engineering **6**(01), 73–82 |
| **What was needed** | Test 3 only — **Table 1**, *"DEPENDENCY OF $r_{we}$ ON PHASING"* |
| **Values obtained** | $\alpha_0=r_{we}/(r_w+L_p)$, **all six rows** — 0.250 · 0.500 · **0.648** · 0.726 · 0.813 · 0.860. ✅ Table in [`engine_spec_closures.md`](engine_spec_closures.md) §7h.2a.1 |
| **🔴 Was wrong** | My §7h.2 recorded $\alpha_0(120^\circ)=0.618$. 🔴 **The source says $0.648$.** Two rows ($60^\circ$/6 and $45^\circ$/8 planes) were **missing entirely** |
| **Direction** | 🔴 **The phasing optimum is PARAMETER-DEPENDENT (C-232).** $S_H$ and $S_{wb}$ both fall as phasing goes $0^\circ\to45^\circ$ — *more* phasing is better — while $S_V$ **rises** — *less* is better. Measured net optimum: **45° / 60° / 90°** depending on $h_D$, $r_{pD}$, $r_{wD}$. ✅ **Both earlier readings (C-98, C-100) were half right**; see §7h.2b.3 |
| **Also needed** | ✅ **OBTAINED (C-231).** Tables 2, 3, 4 and 5 — all six phasings for each. ✅ See §7h.2b.2 |
| **Closed defect** | ✅ **CONF-31's coefficient defect is closed** — every named correlation has published coefficients. ⚠️ **CONF-31 stays open** on: 🔴 **C-102** no closed form for $\alpha_\theta$, ⚠️ **C-233** the $0^\circ$ ambiguity, ⚠️ **C-234** the $r_{wD}$ gap outside Eq. 9's domain |

> 📌 **How this closed.** The owner supplied the table image after locating the paper. The lesson is the one
> recorded at §7h.2a: 🔴 **a table transcribed without the source in hand is a table of remembered values.**
> My four-row version had three rows right and one digit wrong — 🔴 **and the error was in the digit that a
> reader is least able to check**, since $0.618$ looks exactly as plausible as $0.648$.

> ✅ **Nothing about Karakas & Tariq remains unobtained.** 🔴 **What remains is a set of _decisions_, not
> sources** — see §7h.2b: the $0^\circ$ ambiguity (C-233), the $r_{wD}$ domain gap (C-234), whether
> $\alpha_\theta$ gets a closed form or stays tabulated (C-102), and whether $S_V$ comes from Eq. 12 or
> Eq. 15 (C-237). ⚠️ **M7c must not be gated on any of them.**

---

## 3. 🟠 PENDING — replace the fabricated DOI

### L-2 · Aziz & Settari (1979) — ✅ source found, DOI must be replaced

| Field | Value |
|---|---|---|
| **Status** | 🟠 **Source is local and confirmed** (**C-117**) |
| **Local copy** | `D:\RAG\kupdf.net_khaled-aziz-reservoir-simulation.pdf` (489 pp) |
| **Confirmed metadata** | Aziz, Khalid & Settari, Antonin, *Petroleum Reservoir Simulation*, **Applied Science Publishers Ltd, London**, **ISBN 0-85334-787-5**, © **J979** |
| **Rejected identifier** | 🔴 `10.1016/C2013-0-06222-0` — **404**, and **structurally impossible**: `10.1016/` is Elsevier's prefix and this book is **Applied Science Publishers (UK)** |
| **Action** | ✅ **Replace with `ISBN 0-85334-787-5`** — locally verifiable, and a better identifier |
| **Used for** | $k_h$, $k_v$ domains; also **CONF-14** (FVM numerics, implicit/FIM, additive correction / Watts IDC) |
| **Note** | Contains **no** perforation-skin correlation — it cannot substitute for L-1 |

### L-3 · Fanchi — ✅ publisher verified, wrong work cited

| Field | Value |
|---|---|
| **Status** | 🟠 **A Fanchi title is local, but a different one** (**C-118**) |
| **Local copy** | `D:\RAG\vdocuments.mx_shared-earth-modeling.pdf` (319 pp) — Fanchi, *Shared Earth Modeling*, **Butterworth-Heinemann / Elsevier Science, © 2002** |
| **Rejected identifier** | 🔴 `10.1016/B978-012248308-0/50001-X` — **404** at doi.org and absent from Crossref. ⚠️ Unlike L-2, the Elsevier prefix **is** plausible here |
| **Problem** | The register cites Fanchi for the $r_w$ domain, but that content would be in *Petroleum Reservoir Engineering: A Computer Approach*, **not** *Shared Earth Modeling* |
| **Action** | Identify the **exact work** (title + year + publisher), then obtain its DOI. Until then cite by ISBN only |
| **Used for** | $r_w$ valid domain in the symbol register |

---

## 4. ✅ VERIFIED — do not re-work

| Identifier | Reference | Class | Used for |
|---|---|---|---|
| `10.2118/12244-PA` | Watts, J. W. (1986), *A Compositional Formulation of the Pressure and Saturation Equations*, SPE Reservoir Engineering **1**(03), 243–252 | ✅ Journal | Sequential-implicit compositional formulation (**C-99a**) |
| `10.2118/29111-MS` | Coats, K. H., **Thomas, L. K.**, & Pierson, R. G. (1995), *Compositional and Black Oil Reservoir Simulation*, SPE Reservoir Simulation Symposium | 🔴 **Conference (`-MS`)** | Volume-balance compositional formulation (**C-105**) |
| `10.2118/18247-PA` | Karakas, M. & **Tariq**, S. M. (1991), *Semianalytical Productivity Models for Perforated Completions*, SPE Production Engineering **6**(01), 73–82 | ✅ Journal | Perforation skin — ✅ tests 1–2, 🔴 **test 3 outstanding → L-1** |

⚠️ **Test 3 is outstanding for all three.** None of these DOI strings appears anywhere in the `D:\RAG`
corpus, so there is **no local corroboration** — they stand on DOI resolution alone. That is sufficient
for the citation's *existence*, not for the *specific numbers* drawn from them.

### L-4 · Official SPE benchmark input files

| Field | Value |
|---|---|
| **Status** | 🔴 **ACQUISITION REQUIRED** — opened by **C-139** |
| **What is needed** | The **official** SPE Comparative Solution Project files: `SPE5_PERM.GRDECL` (or `.HDF5`), plus the PVT / fluid-property tables for Case 5 |
| **Why** | **C-135 / C-139:** SPE5/SPE10 tests must **import** the official grid and permeability, never regenerate them. Regeneration is a *comparability* failure — a different PRNG yields a different realisation (**C-135**) |
| ⚠️ **What we have is NOT this** | `D:\RAG\Data files\SPE5-ProbForecasting-BaseCase.txt` (**C-119**) is a **CMG GEM transcription** of SPE5 — a valid *starting point*, but **not** the official problem specification. Its dimensions ($7\times7\times3$, $1000$ ft spacing) are consistent with SPE5, which makes it useful for **parsing** work, not for **comparison** |
| **Also needed** | A **CMG reference run** to provide the comparison target (**C-119**) — the deck alone is an input, not a result |
| **Search terms** | `SPE Comparative Solution Project Case 5 SPE5_PERM` · `SPE5 compositional WAG benchmark data files` · `SPE10 Carmona grid` |

---

## 5. 🟠 Coefficients awaiting a source — `SOURCE_PENDING`

Each is a **named correlation with no coefficient**, the original CONF-31 pattern (**C-31**).

| Symbol / model | Where | Note |
|---|---|---|
| **Penéloux $c_i$** | `0.40768`, `0.1154`, `0.4414` | 🔴 **sign-inverted** — the bracket is negative for $Z_c > 0.261$, covering methane, toluene, n-heptane, $n$-decane, CO₂ (**C-88**) |
| **Barton–Bandis roughness** | `8.8` coefficient, `1.5` exponent in $k_f(w_f)$ | Dimensionally sound and limit-correct, but **uncited** |
| **$C_a^{\*}$, $m$** | Asphaltene kinetics | 🔴 undeclared (**C-83**) |
| **$\eta_{erosion}$, $E$** | Filter-cake erosion | 🔴 does not dimensionally close without viscosity and a rate constant (**C-83**) |
| **$K_{IC}$** | Fracture propagation | 🔴 named in a summary, **absent from every equation** (**C-82**) |
| **$A_s$** | Hydrate specific surface area | Gibbs' theorem requires it to **shrink** as hydrate decomposes (**C-75**) |
| **$K$, $q_{inj}$** | CONF-15 $t_{bt}$ | See [`../thmc/conflict_and_gap_register.md`](../thmc/conflict_and_gap_register.md) |
| **Glaso, Joback–Reid, PPR78, Huron–Vidal, QSPR** | **CONF-31** | Named by the design set, **no coefficient given anywhere** |
| Karakas–Tarik ✅ **resolved** | — | ✅ **RESOLVED 09-10-2026 (C-231).** ✅ Tables 1–5 obtained: $c_1,c_2$ (Tab. 2), $L_{p\min}/r_w$ (Tab. 3), $a_1,a_2,b_1,b_2$ (Tab. 4), $s_x$ (Tab. 5). 🔴 **These are finite-element tables, not closed forms** — see §7h.2b.2 |

---

## 6. Search protocol — to avoid the failures already made

| Rule | Reason |
|---|---|
| **Search by title when the DOI 404s** | This is how **L-2** was resolved: the DOI was one digit wrong (**12242 → 12244**) while the paper was real. Guessing the digit would have produced a wrong "correction" |
| **Check the publisher prefix against the title page** | This is how the **L-2** failure was *diagnosed*: Applied Science Publishers ≠ Elsevier, so `10.1016/` could never have been right |
| **A title search is stronger evidence than a DOI lookup** | Crossref's bibliographic index showed neither disputed title existed, while both authors were well indexed — so absence was meaningful, not a coverage gap |
| **A keyword hit in a local corpus is not evidence** | `KAPPA DDA`'s "tariq" is **Umair Tariq**, an engineer in the acknowledgements; its "wellbore effect" is **wellbore storage**, not perforation phasing (**§7m.5**) |
| **Record the class** | `-PA` is a journal paper; `-MS` is a conference paper. Not a cosmetic distinction (**C-105**) |

---

## 7. What is *not* needed

✅ **CONF-29's arithmetic is already correct** and needs no literature — verified (**C-123**).
✅ **CONF-25's TPD objective and gradient are derivable from first principles** and need no literature
— specified (**C-120**).
✅ **CONF-16's chemical-strain fix is a dimensional correction** — no literature (**C-122**).
---

## 9. Items added 08-10-2026 (Ruling 27)

| ID | What is needed | Why it is needed | Blocker? |
|---|---|---|---|
| **L-5** | Sourced basis for the Hermite half-bandwidth $\delta\in[10^{-4},10^{-3}]$ (dimensionless, in $\bar\varepsilon_p$) | **C-162/C-169.** The symbol, units and range are now declared ✅ — only the *provenance* of the range is `SOURCE_PENDING` (Abbo & Sloan 1995 cited as a numerical-regularisation reference) | ⚪ **No** — ✅ **M7c must not be gated on it**; runtime emits `ValidityWarning::CorrelationProvenanceUnverified` |
| **L-6** | DOIs for the **CPPM** return-mapping algorithm and the Abbo & Sloan elastic-plastic work | **CONF-66 items 1–2.** The algorithm is adopted ✅; only the citations are missing | ⚪ **No** |

### 9.1 📌 A new search class — "is this coefficient a *convention* or a *property*?"

📌 **L-5 and L-6 are the first items whose need is a *citation-hygiene* problem, not a *missing-data* problem.**

⚠️ In both cases the **physics is already settled by decision** — the regularisation form, the solver stack and the
convention were all locked by ruling. What is missing is only a citable origin.

🔴 **And the L-5 search has a trap worth recording in advance:** the range $[10^{-4},10^{-3}]$ is a **numerical
tolerance class**, so a hit is almost certain in *any* regularisation paper and will usually be a **value quoted
for a different quantity** (a relative error tolerance, not a strain half-band). ✅ **Apply the L-2 test
strictly** — the candidate must state the quantity, the units, **and** that it is a transition half-bandwidth;
a number without those three is not evidence. ⚠️ This is the same class as the `"tariq"` false positive in
**§7m.5** — *a keyword hit in a local corpus is not evidence.*

---

## 10. Items added 09-10-2026 (Ruling 30)

| ID | What is needed | Why | Blocker? |
|---|---|---|---|
| **L-7** | DOIs + full bibliographic records for **Bardet (1990)** and **Ménétrey & Willam** | **C-188/C-189.** Both are cited in the Lode-angle correction with **no year and no DOI** — the same provenance gap that blocked CONF-31 (**$\alpha_0$**) and L-5 (**$\delta$) | ⚪ **No** — $g(\theta_L)$ is an **optional** extension; the base surface is exact on the meridians (C-176) |

⚠️ **Search-protocol warning, pre-registered.** 🔴 **These two are cited as a bare author pair with no year, and
the register has already produced three *plausible-looking but wrong* citations in this family** (C-31's $\alpha_0$,
the `"tariq"` false positive in §7m.5, and **C-175's wrong MC principal-stress line, which was mine**).
✅ **L-7 must satisfy the full L-2 test** — DOI returns 200, metadata matches **field for field** including the
**year**, and the specific equation or figure carrying $g(\theta_L)$ is reachable in the retrieved copy. ⚠️ A
citation that verifies as *"a Lode-angle-dependent yield function exists by Bardet/Ménétrey-Willam"* is
**insufficient** — the specific $g$ form adopted must be attributable.

---

## 11. L-7 updated — 09-10-2026 (Ruling 31)

### 11.1 ✅ Ménétrey & Willam 1995 — VERIFIED

| field | value | status |
|---|---|---|
| DOI | [`10.14359/1132`](https://doi.org/10.14359/1132) | ✅ **resolves 200** |
| Title | *Triaxial Failure Criterion for Concrete and its Generalization* | ✅ exact match |
| Journal | ACI Structural Journal | ✅ |
| Volume / Issue | 92 (3) | ✅ |
| Year | 1995 | ✅ |
| Pages | 311–318 | ⚠️ **not in Crossref — confirm in the retrieved copy** |

### 11.2 🔴 Bardet 1990 — THE SUBMITTED RECORD IS WRONG

| field | submitted | **verified** |
|---|---|---|
| **DOI** | `10.1115/1.2892023` | ❌ **404** → ✅ **`10.1115/1.2897051`** |
| **Title** | *Lode angle function for the yield surfaces of soils and rocks* | ❌ no such record → ✅ ***Lode Dependences for Isotropic Pressure-Sensitive Elastoplastic Materials*** |
| Journal | Journal of Applied Mechanics | ✅ |
| Volume | 57 | ✅ **57** |
| **Issue** | 2 | ❌ → ✅ **3** |
| **Pages** | 498–506 | ✅ **498–506** |
| Year | 1990 | ✅ |
| Author | Bardet, J. P. | ✅ |

🔴 **Six of eight fields were correct — including the exact page range — so any completeness or plausibility
review passes it. Only the L-2 test catches it.**

### 11.3 🔴 404-DOI list — EXTENDED (do not reuse)

| DOI | Why it appeared | Correct form |
|---|---|---|
| `10.2118/12242-PA` | Watts 1986 — one digit wrong | ✅ `10.2118/12244-PA` |
| **`10.1115/1.2892023`** | **Bardet 1990 — digits transposed** | ✅ **`10.1115/1.2897051`** |
| `10.2118/76722-PA` | unresolvable | — |
| `10.1016/C2013-0-06222-0` | unresolvable | — |
| `10.1016/B978-012248308-0/50001-X` | unresolvable | — |

📌 **New failure class: _real paper, corrupted metadata._** 🔴 This is distinct from **L-2** (mis-typed DOI on a
correct record) and from a fabricated reference — here **the title was also wrong**, and *the plausible-looking
fields were the correct ones*. ✅ **Standing rule: a citation is verified when the DOI resolves _and_ the metadata
matches field-for-field. Completeness is not evidence.**

⚠️ **And note the structural hazard: `2892023` vs `2897051` are both plausible ASME JAM identifiers.** 📌 **DOI
correctness cannot be inferred from plausibility — only from a live resolution.**

---

## 12. Literature status after Ruling 32 (09-10-2026)

| ID | Item | State |
|---|---|---|
| **L-1** | Karakas & Tariq 1991 Table 1 ($\alpha_0$) | 🔴 **unobtainable** — zero hits across 129 local PDFs. `SOURCE_PENDING`; runtime emits `ValidityWarning::CorrelationProvenanceUnverified`. ⚠️ **M7c must not be gated on it** |
| **L-2** | Watts 1986 | ✅ **CLOSED** — `10.2118/12244-PA` |
| **L-3** | Fanchi exact work | ⚪ open |
| **L-4** | Official `SPE5_PERM.GRDECL` | ⚪ open |
| **L-5** | Hermite $\delta$ range provenance | ⚪ open · non-blocking |
| **L-6** | CPPM + Abbo & Sloan DOIs | ⚪ open · non-blocking |
| **L-7** | Lode-shaping citations | ✅ **CLOSED (Ruling 32)** — Bardet 1990 = **`10.1115/1.2897051`**, J. Appl. Mech. **57(3)**, 498–506; Ménétrey & Willam 1995 = **`10.14359/1132`**, ACI Struct. J. **92(3)**, 1995 ✅ resolves 200 |
| **L-8** | Ménétrey & Willam **page range 311–318** | ⚪ open — Crossref carries no page field for this record; confirm in the retrieved copy |

⚠️ **A pattern worth keeping visible across all of L-1…L-8:**

📌 **Four of the eight items are open for reasons that are *not* about difficulty of access.** **L-5** and **L-6**
need only a citation, not data. **L-8** needs one field. **L-1** is open because the table does not exist in any
obtainable source. ✅ **None of the four is a physics blocker, and M7c is gated on none of them** — a fact that
should be re-read whenever someone proposes making a milestone depend on a literature item.

---

## 13. Items added 09-10-2026 (Ruling 35)

| ID | What is needed | Why | Blocker? |
|---|---|---|---|
| **L-9** | Provenance for the strict curvature floor $\kappa_{\min}=10^{-4}$ on $N_\kappa=g^2+gg''$ | **C-213.** ✅ The value is retained and declared as `MIN_CURVATURE_NUMERATOR`, but ⚠️ no source is citable for it — the same gap as **L-5** ($\delta$) and **L-1** ($\alpha_0$) | ⚪ **No** — the default is safe ($N_\kappa\equiv1$ for a circular cone) and fails loudly on flattening $g$ |

⚠️ **A pattern across L-1, L-5 and L-9 worth naming:** 📌 **all three are _numerical regularisation or tolerance
constants_**, not physical properties. 🔴 **They are routinely chosen by whoever writes the code, never cited, and
never revisited** — because a tolerance that produces a plausible answer does not attract scrutiny.

✅ **The standing response is unchanged and deliberately weak:** declare, default, record `SOURCE_PENDING`, and
**never gate a milestone on one.** ⚠️ **What has changed is that they are now at least _visible_ in one list** —
which is the most that can be done without the literature the owner has chosen to search personally.
