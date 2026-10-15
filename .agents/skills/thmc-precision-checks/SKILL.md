---
name: thmc-precision-checks
description: >-
  Precision discipline for the Rust 3D THMC compositional reservoir simulator.
  Use when writing, reviewing or testing any constitutive relation, correlation,
  unit conversion, solver gate or benchmark in agent_wiki/compositional/. Supplies
  the SI/field unit table computed by measurement, the dimensional-failure patterns
  already logged in the spec register, and the evidence standard that a novel
  (non-published) relation must satisfy before it may enter a solve path.
  Also states the governing rule that a third-party skill is procedural guidance and
  never a citable data source.
---

# THMC Precision Checks

**Read this before writing any constitutive relation or any gate tolerance.**

This skill exists because the specification register has accumulated **ten dimensional
failures** and **five checks that cannot fail**. Both classes are invisible to review and
loud only in hindsight. This file is the pre-flight for that class of error.

---

## 1. 🔴 The governing rule — a skill is not a source

> **A third-party skill is *procedural guidance*. It is never *data*.**
>
> No skill may close a `CONF-*` item, satisfy an `L-*` literature item, or supply a
> coefficient that enters a constitutive relation.

**Why this rule exists — the measured cost of breaking it:**

| Finding | What happened |
|---|---|
| **C-116** | Searched 129 PDFs, found nothing, then **quoted a four-row table from the submission text** instead of marking it absent |
| **C-227** | With the real table in hand, transcribed **0.618** where the source reads **0.648**, and dropped two rows — the fit was good enough (RMS 0.0076) that the bad digit read as scatter |
| **C-229** | Misread the subscript $r_{we}$ as $r_{wo}$, then **invented an inversion** no source contains, and produced a "measured 5.25× divergence" that was pure artefact |

**None of those was fixed by better tooling.** All three were fixed by obtaining the
paper and reading the equations. ✅ **The tables ($\alpha_\theta$, $c_1/c_2$,
$a_1..b_2$) came from the source; no skill supplied them.**

**Provenance for anything numeric, in order of strength:**

1. 🔴 **The primary source**, transcribed row by row, with row count recorded
2. ⚠️ **A secondary source**, labelled as such, with its own citation
3. 🔴 **Never** a remembered value, a submission's transcription, or a skill

---

## 2. Unit table — computed by measurement, not copied

Verified 10-10-2026 by direct computation (`psi = 6894.75729316836`, etc.). These are
**exact by definition** and safe to encode as `#[repr(transparent)]` newtypes.

| Quantity | Exact value | Note |
|---|---|---|
| 1 psi | `6894.75729316836` Pa | exact |
| 1 ft | `0.3048` m | exact |
| 1 in | `0.0254` m | exact |
| 1 lbm | `0.45359237` kg | exact |
| 1 US gal | `3.785411784e-3` m³ | exact |
| 1 bbl (42 gal) | `0.158987294928` m³ | exact |
| 1 atm | `101325` Pa | exact |
| R | `8.31446261815324` J/(mol·K) | CODATA 2018 |
| 1 darcy | `9.869233e-13` m² | **rounded** — carry ≥4 significant figures |
| 1 cP | `1e-3` Pa·s | exact |
| 1 poise | `0.1` Pa·s | exact |
| 1 scf | `0.0283168466` m³ | = ft³, exact |
| 1 STB | `0.158987294928` m³ | exact |

🔴 **Mscf is NOT an exact unit.** 1 Mscf is 1000 ft³ *at 60 °F and 14.696 psia*, i.e.
`28.3168` m³ **at those conditions** — it is a *standard-condition volume*, not a length
cube. ⚠️ Converting Mscf to m³ without carrying $T$ and $p$ is a dimensional error even
when the number looks plausible.

---

## 3. 🔴 Temperature — the failure that already landed

**Four relations, and getting the wrong one is silent:**

$$T_K=T_C+273.15 \qquad T_K=T_R\cdot\tfrac59 \qquad T_R=T_F+459.67 \qquad T_F=T_R-459.67$$

| °C | K | °F |
|---|---|---|
| 20.0 | 293.150 | 68.000 |
| 60.0 | 333.150 | 140.000 |
| 90.0 | 363.150 | 194.000 |

**Beggs & Robinson dead-oil viscosity requires T in °F** — this is already logged as
CONF-31's *"⚠️ BR needs T in **°F**, 70–295 °F"*.

**Measured consequence of handing it K instead:** API dead-oil $\mu$ goes as $10^z$ with
$z=3.0324-0.02023\,T_F$. 🔴 **A 100 °F error is a factor of $10^{2.02}=105$ in
viscosity** — and the output still looks like a plausible number.

📌 **Rule: every correlation declares its required temperature unit in its own docstring,
and a `#[cfg(test)]` case asserts the conversion at a fixed state.**

---

## 4. 🔴 Overburden — the cheapest possible dimensional self-test

$\Delta p = \rho\,g\,\Delta z$ with $g=9.80665$ m/s².

| Overburden | Δp | ψ |
|---|---|---|
| 100 m @ 2200 kg/m³ | 2.157 MPa | 312.9 |
| 1000 m @ 2200 kg/m³ | 21.575 MPa | 3129.1 |

✅ **Sanity band: 10.0–10.5 MPa/km.** Anything outside means $g$, $\rho$, or the psi
factor is wrong. 📌 **Run this before trusting any pressure-dependent correlation** — it
catches a whole class of unit errors in one line.

---

## 5. Dimensional failure patterns already in the register

Ten logged. The recurring shapes:

1. **Wrong temperature unit in a correlation argument** — §3 above
2. **A symbol carrying two dimensions** (e.g. $r_w$ used for both a radius and a
   dimensionless ratio) — **C-106**, **C-122**
3. **A ratio inverted into its reciprocal** — **C-151**, and the DP $\sqrt3$ family
4. **A coefficient typed by hand where the limit cases constrain it** — C-161, C-164,
   C-205, C-214, C-217 (five factor-3/$\sqrt3$ inversions)
5. **Adding terms whose dimensions disagree** — **C-88** (Penéloux bracket),
   **C-140**, **C-192**

📌 **Rule: derive the dimension before writing the expression. If the expression does not
close dimensionally, the *model* is wrong, not the units.**

---

## 6. The oracle ladder — for anything whose answer is not obvious

A test must assert a property computable **two independent ways**.

> 🔴 **The cardinal sin is an expected value obtained by running the code under test.**
> That is a snapshot. Snapshots detect change; they never detect error.

Use the strongest available:

1. **Closed form** — an exact limit or special-parameter solution
2. **Independent implementation (the slow twin)** — dense, brute-force, obviously-correct,
   agreeing to machine precision on randomized inputs at small size. 📌 **If the project
   has no slow twin, building one comes before anything else.**
3. **Invariants** — mass balance, $\sum_i z_i=1$, $0\le S_\alpha\le1$, trace
   preservation, symmetry, degeneracy limits
4. **Limiting cases** — turn a coupling to zero, recover known simpler physics; each
   limit is a separate test wiring a different part of the code
5. **Convergence order** — assert the **rate**, not that the error is small. Error
   falling is weak evidence; error falling as $h^2$ under a second-order scheme is strong.
   Use Richardson extrapolation and the method of manufactured solutions.

📌 **Record which oracle each test uses, in the test's own docstring.**

---

## 7. 🔴 Evidence standard for a NEW relation

The project intends to go beyond reproducing published correlations. 📌 **A novel relation
is a *claim* like any other and owes the same evidence.** Required before it enters a
solve path:

| # | Requirement | Test |
|---|---|---|
| 1 | **Dimensional closure** written out explicitly | §4-style overburden self-test |
| 2 | **Limiting cases** derived, not assumed | each is a `#[cfg(test)]` case |
| 3 | **No hand-typed scalar** where the limits constrain it | assert the factor via limits |
| 4 | **Derive-then-assert** for every inequality | §6 oracle 2 (two independent routes) |
| 5 | **Independent validation set** not used in fitting | held-out cases |
| 6 | **Provenance class recorded** | published / derived / assumed |
| 7 | **Failure mode characterised** | what happens outside its validity region |

> 📌 **Criterion 3 is not optional.** Five factor-3/$\sqrt3$ inversions were typed by hand
> and only caught by limit-case tests — never by review. **The remedy already ruled: no
> scalar factor is typed by hand; the `#[cfg(test)]` limit-case suite asserts it.**

🔴 **An assumed-but-unvalidated coefficient gets `SOURCE_PENDING` and a runtime
`ValidityWarning`.** It is never silently used.

---

## 8. Standing CI requirements

- Constitutive **sign gate** (`engine_constitutive.md` §7cc.6)
- **Derive-then-assert** for every inequality
- Every **saturating transform** and singularity guard paired with a `ValidityWarning`
  (`engine_numerics.md` §7s.6)
- **Two-sided independence** for equivalence tests
- **Code-block test pinning** — every submitted code block gets a test
- Cumulative edits to guarded expressions
- Every `MeridianType` names the property its coefficients have, **verified by a test**
- No silent degradation; no clamp that makes a failure disappear (C-93, C-188)

---

## 9. Third-party skills installed here, and what each is good for

| Skill | Use for | Limitation |
|---|---|---|
| `numerical-verification` | The oracle ladder (§6) | none found; general |
| `nist-refprop` | PVT/phase-equilibrium **validation oracle** | needs a commercial licence + `ctREFPROP`; **not available in CI** |
| `dimensional-analysis` | Unit/precision bug patterns | 🔴 **written for Solidity/DeFi** — examples are token decimals and price oracles. **Use the patterns, ignore the examples; this file supplies the domain table** |
| `neqsim-capability-map` | Architecture reference for a real compositional simulator | Java; map only |
| `neqsim-eos-regression` | $k_{ij}$ fitting, saturation pressure, C7+ characterisation | NeqSim workflow |
| `neqsim-phase-envelope` | PT envelope generation and branch identity | NeqSim-specific API |
| `design-flash-benchmark` | Flash benchmark **matrix design** | 🔴 benchmark **design only — supplies no expected values** |
| `analyze-gibbs-convergence` | Gibbs minimisation convergence, Jacobian conditioning | chemical equilibrium framing |
| `neqsim-input-validation` | Input validation patterns | NeqSim-specific |

> 🔴 **Registry gap — reported, not papered over.** The public registry returned **zero**
> results for: *geomechanics plasticity*, *Drucker-Prager*, *Mohr-Coulomb*, *solid
> mechanics plasticity*, *reactive transport*, *species transport*, *finite volume
> method*, *conservation law discretization*, *flow in porous media*, *porous media*,
> *method of manufactured solutions*, *upscaling permeability*, *CO₂ geological storage*,
> *phase equilibrium flash*, *well logging / petrophysics*.
>
> **Those are M7b, the geochemistry milestone, and the core spatial discretization.** ⚠️
> **No skill covers them, so they must be sourced from literature (L-1…L-9) or derived.**
> 🔴 **A milestone must not be gated on a skill that does not exist.**

---

## 10. Rejected skills — and why

Recorded so the search is not repeated and the reasoning is auditable.

| Skill | Reason for rejection |
|---|---|
| `tnav-reservoir-sim` | 🔴 **Self-declared "educational emulation… results are approximate."** For a precision engine this is an *anti-skill*: it is a source of correlations applied without provenance. |
| `multi-phase-flows` | 🔴 Wrong domain — combustion/propulsion CFD (spray, cavitation, VOF), not porous media. |
| `solver-numerics` | 🔴 Wrong domain — SPICE circuit simulation (MNA systems, trapezoidal for circuits). |
| `fenics-fem` | 🔴 Python FEniCS; this engine is Rust. |
| `*@arbor` (15 mirrors, 1.1K installs) | 🔴 Dendrite morphology. **High install count is not relevance.** |