# Scientific & Mathematical Audit — CO₂ EOR Optimizer (`phd_audit.md`)

**Role:** Senior Reservoir Simulation Engineer / Applied Mathematician
**Date:** 04-10-2026
**Scope:** Phase 2 (Scientific/Mathematical Reconstruction) and Phase 4 (CO₂-EOR Physics) of the
four-phase forensic audit.

**Companion deliverables**

| File | Phases |
|---|---|
| `audit/scientific_flaws.md` | Master register — **52 findings**, each with Severity / Category / Location (`file:line`) / Observed / Expected / Impact / Evidence, plus a *"Verified correct — do not flag"* section |
| `audit/parameter_provenance.csv` | 91 constants → LITERATURE / EMPIRICAL / CALIBRATED / **UNKNOWN — EVIDENCE REQUIRED** |
| `res_audit.md` | Phase 1 (software quality) + Phase 3 (anti-patterns) + toolchain + change-safety |
| **`phd_audit.md` (this file)** | Phase 2 + Phase 4 + categorical separation + predictive-validity verdict |

**Audit-only rule observed.** No equation, patch, or "cleanup" was applied. Every number below was
measured this session against the working tree by scripts kept **outside** the repository
(`%TEMP%/opencode/v_*.py`, `audit_verify_*.py`), so nothing in the repo was modified to obtain them.

---

## 0. Method: hierarchy of scientific truth applied

Findings are graded by this precedence, and a defect in a higher row overrides any agreement in a
row below it:

1. **Mathematical correctness** — is the formula the formula it claims to be? Are the dimensions right?
2. **Physical consistency** — does it conserve mass, obey monotonicity, and respect thermodynamics?
3. **Numerical discretization** — are step sizes, roots, and integrations handled honestly?
4. **Implementation** — does the code compute what the equation says?
5. **Parameter provenance** — can each constant be traced to a source?
6. **Analytical verification** — does it reproduce a closed-form solution?
7. **Experimental validation** — has anything been compared to lab/field data?
8. **Benchmark agreement** — does it match another simulator?
9. **Execution speed.**

**Two explicit refusals**, applied throughout:

- **Benchmark agreement is never credited as proof of correctness.** The repository's
  CMG/SPE-style validation material targets `phd_hybrid` and dormant engines; the *default* active
  path is `hybrid` (**HIGH-09**). Even had it matched, it would not validate the shipped default.
- **No composite score is issued.** Software correctness, numerical stabilization, physical
  consistency, empirical calibration, and predictive validity are reported as five separate
  verdicts (§7). A single "model accuracy" number would destroy the distinction between
  *the code works*, *the equations are right*, and *the predictions are trustworthy* — which in this
  codebase are three different answers.

---

## PART A — PHASE 2: MATHEMATICAL RECONSTRUCTION OF THE ACTIVE PATH

### A.0 What the active path actually computes

Per `agent_wiki/architecture/source_of_truth_map.md`, 100 % of optimization evaluations route to
`core/engine_surrogate/`. The executed chain, verified by reading + call graph, is:

```
OptimizationEngine._evaluate_fitness (optimisation_engine.py:841-896 assembles `profiles`)
  └─ SurrogateEngine.evaluate_scenario (surrogate_engine.py:150-…)
       ├─ _build_params_dict → params                    (:940-957 economic mapping)
       ├─ AnalyticalSurrogate.predict (surrogate_models.py:455-560)
       │    ├─ recovery_model.calculate_recovery → RF₁     (:465)   ← at target_pressure_psi
       │    ├─ cumulative_oil = RF₁ · OOIP                (:473)
       │    ├─ co2_stored = breakthrough-aware helper     (:493-505)
       │    └─ NPV(RF₁, flat cashflow)                    (:507-530)
       ├─ FastProfileGenerator.generate_profile           (:188)
       ├─ mean_p = mean(pressure_profile)                 (:490)
       ├─ simulated_hcpvi                                 (:491-497)  ← OVERWRITES :890
       ├─ recovery_model.calculate_recovery → RF₂         (:504-507)  ← at mean_p, hcpvi
       ├─ clip(RF₂, 0, 1-swi-sor)                         (:509-510)
       ├─ rescale oil_profile to RF₂·OOIP                 (:513-522)
       └─ returns: recovery_factor=RF₂, cumulative_oil∝RF₂, npv=f(RF₁)   (:685-692)
  └─ ObjectiveFunctions._calculate_objective_functions (wrapper.py:28-…)
       ├─ npv = profiles["npv"]  (= f(RF₁))               (:52)
       ├─ sandface/Class-VI penalty  ← KEY NEVER PRESENT  (:61-78)   [CRIT-09]
       └─ leakage remediation cost   ← KEY NEVER > 0      (:81-99)   [CRIT-10]
```

The reconstruction's central structural finding: **the objective function is a hybrid of two
different evaluations.** Everything downstream of `:504` (reported RF, reported oil, profiles,
storage efficiency, VRR, saturations) describes one reservoir state; everything downstream of
`:465`/`:473` (NPV) describes a *different* one.

---

### A.1 CRIT-01 — Throughput (HCPVI) is dimensionally wrong and pressure-independent

**Category:** MATHEMATICAL · **Location:** `core/engine_surrogate/surrogate_engine.py:491-497`
(overwriting `params["hcpvi"]` set at `:890`)

**Equation as written**

```python
b_co2_mean   = pvt_engine.calculate_co2_fvf_rb_per_mscf(mean_p)   # :493
cum_inj_rb   = cum_inj_mscf * b_co2_mean                          # :494
simulated_hcpvi = cum_inj_rb / (ooip_val * b_co2_mean/(1-swi_val))  # :497
```

Algebraically `b_co2_mean` cancels identically:

$$\text{hcpvi}_{\text{model}}=\frac{V_{\text{inj}}\,B_{\text{CO}_2}}{\text{OOIP}\,B_{\text{CO}_2}/(1-S_{wi})}
=\frac{\text{cum\_inj}_{\text{MSCF}}\,(1-S_{wi})}{\text{OOIP}_{\text{STB}}}$$

— a ratio of **MSCF to STB** (dimensionally `Mscf/STB`, not `res bbl / hydrocarbon bbl`) containing
**no pressure term**.

**Measured** (1 MMSCF into 1 MMSTB, `swi = 0.25`, this session, `v_hcpvi.py`):

| P (psi) | 1500 | 2500 | 3500 |
|---|---:|---:|---:|
| model hcpvi | **0.750000** | **0.750000** | **0.750000** |
| textbook `V_inj·Bg·(1−swi)/(OOIP·Bo)` | 0.907564 | 0.406907 | 0.278135 |
| ratio model/true | **0.826** | **1.843** | **2.697** |

(The register's independent run at a different temperature gave 0.667 / 1.218 / 1.868; both runs
agree on the two facts that matter: the model value is *constant* in P, and the error grows
monotonically with P.)

**Expected**
$$\text{HCPVI}=\frac{V_{\text{inj,rb}}}{\text{OOIP}\cdot B_o}=\frac{\text{cum\_inj}_{\text{MSCF}}\cdot B_g(P)\,(1-S_{wi})}{\text{OOIP}\cdot B_o(P)}$$

**Scientific impact.** HCPVI is the independent variable of the Koval/Ekladios sweep and of every
miscible RF correlation in `analytical_models.py`. Because the model value is *constant in P* while
the truth falls by 3.2× from 1500→3500 psi, the engine **systematically rewards high-pressure
operation** on throughput alone. This is not noise: it is a monotone, pressure-driven bias in the
optimizer's landscape. At 3500 psi the engine believes throughput is 2.7× higher than it is.

**Category:** MATHEMATICAL (dimensional error), propagating to PHYSICAL (unphysical pressure bias).

---

### A.2 CRIT-02 — Reported NPV and reported RF come from two different evaluations

**Category:** MATHEMATICAL/PHYSICAL · **Location:** `surrogate_engine.py:165,171-174` (RF₁ + NPV)
vs `:490-510,513-522,533,685,690,692` (RF₂); `surrogate_models.py:465,473,507-530`

**Mechanism**
1. `predict()` computes `recovery_factor = RF₁` at `target_pressure_psi` (`:465`), then
   `cumulative_oil = RF₁·OOIP` (`:473`) and **NPV from RF₁** (`:507-530`).
2. Later, RF₂ is computed at the whole-profile **mean pressure** with the overwritten `hcpvi`
   (`:490`,`:501`,`:505`), clipped (`:510`).
3. The returned object mixes them: `recovery_factor = RF₂` (`:690`), `cumulative_oil` rescaled to
   RF₂ (`:519`,`:533`,`:685`), but `npv = f(RF₁)` (`:692`).

**Measured — the two ledgers disagree by construction** (`v_engine.py`, same reservoir, 10 yr,
monthly, `T = 160 °F`, `P_i = 4000 psi`, `MMP = 1800 psi`):

| | `hybrid` | `phd_hybrid` |
|---|---:|---:|
| reported `recovery_factor` | 0.500000 | 0.500000 |
| reported `cumulative_oil` | 2 499 999.9999999977 | 2 500 000.0000000014 |
| reported `co2_stored` | 1 848 693.306 | 1 848 693.306 |
| reported **`npv`** | **$106 601 776** | **$60 733 386** |

Identical reported oil, identical reported CO₂ — **NPV differs by 45 % (75 % higher)**. Two
objective components that are supposed to describe the same scenario do not. And the clip that
forces both RF₂ to 0.5 is *why* RF₂ can agree while NPV does not: **the optimizer can move NPV
without moving reported recovery**, i.e. the objective surface contains a direction along which
reported physics is constant but reported economics changes by tens of millions of dollars.

**Further, two CO₂ ledgers exist simultaneously:**

- NPV's cost term uses `co2_stored` from `calculate_co2_stored_breakthrough_aware`
  (`surrogate_models.py:493-505`) — breakthrough-aware, trapping-limited.
- The reported `co2_stored` uses `cum_inj − cum_prod` (`surrogate_engine.py:543`), ignoring
  leakage entirely.

**Expected:** one evaluation, one state vector; `npv`, `recovery_factor`, and `cumulative_oil`
must all be derived from the same `(P, hcpvi)` pair, or the coupling must be stated explicitly.

**Scientific impact.** Violates the hierarchy at rows 1 and 2 simultaneously: the reported NPV is
not a function of the reported production. Any multi-objective weighting of NPV vs RF is
mathematically incoherent as shipped.

---

### A.3 CRIT-03 — No bubble-point model: `Bo` increases with pressure

**Category:** PHYSICAL · **Location:** `core/engine_surrogate/pvt_state.py:232-244, 267-307`

There is **no `bubble_point` field anywhere in `pvt_state.py`** (grep: 0 hits), so the undersaturated
branch is never entered correctly. The swelling/shrinkage logic is driven by `x_co2` and a
`tanh` pressure term only.

**Measured** (`v_pvt2.py`, `x_co2 = 0`, i.e. *no CO₂ dissolved at all*, `T = 180 °F`):

| P (psi) | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---:|---:|---:|---:|---:|---:|
| `Bo` (RB/STB) | 1.1698 | 1.2203 | 1.2752 | 1.3341 | 1.3882 | 1.4449 |
| apparent `dBo/dP` (1/psi) | — | **+1.01e-4** | **+1.10e-4** | **+1.18e-4** | **+1.08e-4** | **+1.13e-4** |

Above the bubble point, real crude **shrinks** with pressure: `dBo/dP = −c_o < 0`, with
`c_o ≈ 5–10 × 10⁻⁶ 1/psi`. The engine returns a **positive** derivative an order of magnitude larger
than any physical oil compressibility. Oil `Bo` grows 23.5 % from 1500→4000 psi with no dissolved
gas whatsoever.

**Compounding contradiction:** the engine's own total-compressibility term uses a **constant**
`c_o = 1.0e-5` (`surrogate_engine.py:315`) while the PVT module implies `+8.7e-5 … +1.19e-4`.
Two mutually exclusive oil compressibilities coexist in the same evaluation.

**Impact.** `Bo` enters `dp/dt` (via `J·dt` and pore volume), the RF normalization, and the
material balance. A monotone-incorrect `Bo(P)` makes pressure maintenance look cheaper than it is.

---

### A.4 CRIT-04 — Hydrocarbon-gas `Bg` is wrong by ~31× (constant `0.1587`)

**Category:** MATHEMATICAL · **Location:** `pvt_state.py:372-373`, consumed at `:366,414,453`

```python
# Bg in RB/MSCF: Bg = 0.02827 * Z * T_R / P * 5.6146
bg_hc = 0.1587 * z_hc * self.temp_r / max(p, 1e-4)
```

The comment states the correct relation; the code uses a constant **31.73× too small**. The comment
`0.02827·Z·T_R/P·5.6146` is itself wrong twice over: for RB/**MSCF** the conversion is
`0.02827 · Z · T_R/P · 1000/5.6146 = 5.0349·Z·T_R/P`, and `0.1587 ≠ 5.0349 · 0.02827 · 5.6146`
under any association (the author appears to have inverted the `5.6146` division *and* dropped the
×1000).

**Measured** (`v_pvt2.py`, `y_co2 = 0` → pure HC gas, `T = 180 °F`, `γ_g = 0.7`):

| P (psi) | 1500 | 2500 | 3500 | 4500 |
|---|---:|---:|---:|---:|
| Ppr | 2.24 | 3.74 | 5.24 | 6.73 |
| model `Z` (linear, see CRIT-05) | 1.0841 | 1.1402 | 1.1963 | 1.2523 |
| model `Bg` (RB/MSCF) | 0.07337 | 0.04630 | 0.03470 | 0.02825 |
| textbook `0.02827·Z·T_R/P·1000/5.6146` | 2.32901 | 1.46969 | 1.10141 | 0.89681 |
| **ratio** | **0.032** | **0.032** | **0.032** | **0.032** |

i.e. **31.7× too small**, uniformly. (The register's mixture-level measurement at `y = 0.5`,
2500 psi, giving 11.4×, uses the harmonic mixing rule at `:385`, which dilutes but does not remove
the error.)

**Note the pure-CO₂ path is *not* affected:** `:358` routes pure CO₂ through the PR-EOS density
(`327.362/ρ`, `pvt_state.py:201`), which is only ~1.5× off versus a `Z = 0.85` assumption. The
defect is specific to the hydrocarbon gas branch — which is exactly the branch used when `y_co2 < 1`
(i.e. always, early in the project).

**Impact.** `Bg_hc` feeds `c_g` and the mixture `bg_mix` (`:385`) → gas-phase mobility, `dp`, and
throughput. 31× understated gas FVF means the HC gas phase is treated as ~31× less voluminous than
it is.

---

### A.5 CRIT-05 — The "Hall-Yarborough" Z-factor is a linear function

**Category:** MATHEMATICAL · **Location:** `pvt_state.py:361-367`

```python
Ppr = p / (709.6 - 58.7 * gamma_g)          # Standing correlation (correct form)
Tpr = self.temp_r / (170.5 + 307.3 * gamma_g)
t_inv = 1.0 / max(Tpr, 0.1)
z_hc = 1.0 + (0.06422*t_inv - 0.00332*t_inv**2) * Ppr   # "Hall-Yarborough"
z_hc = float(np.clip(z_hc, 0.65, 1.4))
```

Hall-Yarborough (1973) is the **Standing–Katz** virial-type equation of state solved iteratively for
`y`:
$$Z = 0.06125\,p_{pr}\,t^{-1}e^{-1.2(1-t)^2}+\frac{1-y}{1+y\,y\,b_1}+\frac{b_2}{y^2}-b_3y^2$$
with a rational-expansion virial coefficient series. What is implemented is a **truncated linear
expansion in Ppr** — a tangent line at `Ppr = 0`.

**Measured** (`v_pvt2.py`, `Tpr = 1.660`):

| P (psi) | 1500 | 2500 | 3500 | 4500 |
|---|---:|---:|---:|---:|
| Ppr | 2.24 | 3.74 | 5.24 | 6.73 |
| model `Z` | 1.0841 | 1.1402 | 1.1963 | 1.2523 |
| Standing–Katz `Z` (chart) | ≈0.84 | ≈0.83 | ≈0.86 | ≈0.91 |
| model error | +29 % | +37 % | +39 % | +38 % |

Real hydrocarbon gas at `Tpr ≈ 1.6` has `Z < 1` throughout this Ppr range (the
compressibility-factor minimum lies near `Ppr ≈ 4–8`); the model returns **monotonically rising
`Z > 1`**, the signature of an ideal-ish expansion. It also never dips, so it reproduces neither the
minimum nor the shape.

**Impact.** Directly multiplies `Bg` (CRIT-04, partially offsetting — that is *why* the register's
mixture-level error reads 11–23× rather than 31×) and gas density, `c_g`, and `μ_hc`.

---

### A.6 CRIT-06 — Koval sweep returns 0 for favorable mobility, spikes at M = 1

**Category:** NUMERICAL (branch discontinuity) → PHYSICAL · **Location:** `analytical_models.py:541-559`

**Measured** (`v_rf.py`, `KovalSurrogate`, `v_dp = 0.5`):

| M | 0.5 | 0.9 | 0.999999 | **1.0** | 1.000001 | 1.1 | 1.3 | 1.4 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | **0** | **0** | **0** | **0.316738** | **0** | **0** | **0** | **0** | 0.009621 | 0.263024 | 0.324123 | 0.289557 |

Three simultaneous defects:

1. **`exp` overflow → 0 for `M < 1`.** In the general branch, `c = 1/(M−1)` is *negative*;
   `np.exp(c·(1−kv))` with `c < 0` and `kv ≫ 1` underflows, and the
   `term1 − (term1−term2)/(M−1)` cancellation produces garbage clipped to `0.0` by `:559`.
   A **favorable** mobility ratio (`M < 1`), the *ideal* case for a miscible flood, yields **zero
   recovery**.
2. **A 0.317 spike exactly at `M = 1`**, 10⁻⁶ away from 0 — a 31 674 % discontinuity.
3. `np.clip(sweep, 0.0, 0.75)` at `:559` hides the underflow (returns a legitimate-looking 0.0)
   rather than surfacing it. This is stabilization masking a mathematical error.

**Expected:** the Koval (1963) fractional-flow integral is continuous and *increasing* as `M → 1`
from below, and smooth through `M = 1` (the `M = 1` branch at `:542-547` exists precisely to
handle the removable singularity — it is correct; the *neighbouring* branch is not).

---

### A.7 CRIT-07 — Immiscible RF is a constant 0.10 over an entire parameter grid

**Category:** MATHEMATICAL/PHYSICAL · **Location:** `analytical_models.py:319-326`

```python
recovery = displacement_eff * areal_eff * vertical_eff   # pre-clip product
rf_max_physical = max(0.0, soi - sor)
max_cap = min(0.50, rf_max_physical) if rf_max_physical > 0.10 else 0.50
return float(np.clip(recovery, 0.10, max_cap))           # <-- lower floor 0.10
```

**Measured** (`v_rf.py`, 27-point grid over `swi ∈ {0.15,0.25,0.35}`, `sor ∈ {0.20,0.30,0.40}`,
`v_dp ∈ {0.3,0.6,0.9}`):

```
n=27  min=0.100000  max=0.100000  all_equal_0.10=True
pre-clip product spans 0.0140 … 0.0563  (measured separately)
```

The pre-clip product `E_d·E_A·E_V` never reaches 0.10 on this grid (max 0.0563), so the **floor is
the answer**. The model reports **exactly 10 % recovery regardless of mobility ratio,
heterogeneity, residual oil, or connate water** — the entire physics of the Buckley-Leverett +
Craig + Johnson chain it documents is inert.

**Impact.** Any optimization over immiscible-regime parameters (below MMP) optimizes against a
constant. Gradient = 0. Also, the `max_cap` expression at `:325` is malformed: when
`rf_max_physical ≤ 0.10` the cap is set to **0.50**, i.e. the cap is *loosened* precisely when the
mobile-oil ceiling is lowest — a `min` whose fallback is larger than its own argument.

---

### A.8 HIGH-01 — Mobile-oil term missing its normalization (25 % low)

**Category:** MATHEMATICAL · **Location:** `surrogate_engine.py:509`, `analytical_models.py:814`
vs correct form at `analytical_models.py:801`, `surrogate_models.py:234`

```python
rf_max_physical = max(0.0, 1.0 - swi_val - sor)          # :509  (fraction of PV)
rf_max_physical = max(0.0, 1.0 - s_wi - sor)             # :814
# correct (fraction of OOIP / hydrocarbon pore volume):
e_d = max(soi - sor, 0.0) / max(soi, EPSILON)            # :801
max_displacement = (1.0 - s_wi - s_or) / (1.0 - s_wi)    # surrogate_models.py:234
```

**Measured** (`v_hcpvi.py`, `swi = 0.25`, `sor = 0.30`): model `0.4500`, correct `0.6000`,
ratio `0.7500` — **25.0 % low**.

The clip at `:509-510` divides a *hydrocarbon-pore-volume* fraction by an *OOIP* denominator, so it
caps RF at 75 % of the physically reachable value.

**This clip was observed binding in a full engine run** (`v_clip.py`):

| model | raw RF from `calculate_recovery` | cap | reported RF | clip binds? |
|---|---:|---:|---:|:--|
| `hybrid` | **0.712499** | 0.500000 | 0.500000 | **YES** |
| `phd_hybrid` | 0.500000 | 0.500000 | 0.500000 | marginal |
| *correct* cap | — | 0.666667 | — | — |

For `hybrid`, the model's own answer (71.25 %) is **discarded** and replaced by the (25 %-too-low)
cap. Combined with A.2 this is the mechanism by which CRIT-02 becomes visible: RF₂ saturates at the
cap for both models while RF₁ (uncapped, at a different pressure) keeps moving NPV.

---

### A.9 HIGH-02 — `E_A` has a 48.3 % cliff at M = 1, and a test enshrines it

**Category:** NUMERICAL/PHYSICAL · **Location:** `analytical_models.py:304-312`,
`surrogate_models.py:164-182`; test `tests/scientific/mathematical/test_singularity_and_overflow.py:73`

```python
if mobility_ratio <= 1.0:
    areal_eff = 1.0                       # Craig's favorable-mobility branch
else:
    areal_eff = 0.517 - 0.072*np.log10(mobility_ratio)
```

**Measured** (`v_rf.py`): `M = 1.0 → E_A = 1.000000`; `M = 1.0001 → 0.516997`. **Step = 0.4833
(48.3 % instantaneous drop).**

The Craig (1971) fit `0.517 − 0.072·log10(M)` is a regression calibrated for `M > 1`; extrapolated
to `M = 1` it gives `0.517`, not `1.0`. The discontinuity is in the *branching logic*, not the
correlation.

**Aggravating:** `test_singularity_and_overflow.py:73` **asserts `step > 0.40`** — the test suite
*requires* the cliff. This is the clearest instance in the repository of the category distinction:
software-correct (test passes), physically wrong (no such cliff exists), and the test actively
blocks the fix.

---

### A.10 HIGH-03 — RF floors return a fixed value at zero throughput

**Category:** NUMERICAL (stabilization presented as physics) · **Locations:**
`analytical_models.py:193` (`clip(displacement_eff, 0.05, 0.95)`), `:200` (`clip(rf, 0.05, 0.85)`),
`:326` (`clip(recovery, 0.10, …)`), `:475` (`clip(rf, 0.05, 0.80)`)

**Measured:** miscible branch at `hcpvi = 0` returns **0.050000**; at `hcpvi = 0.1`, 0.075.
The floor converts "nothing injected" into "5 % recovered".

This is category 2 (numerical stabilization) masquerading as category 3 (physics): no comment marks
it as a numerical guard, and downstream code treats the value as a recovery factor.

---

### A.11 HIGH-04 — `or`-defaults silently replace legitimate zeros

**Category:** SOFTWARE → PHYSICAL · **Location:** `analytical_models.py:661, 671, 678-679, 687-694`

```python
v_dp = np.clip(params.get("v_dp") or params.get("v_dp_coefficient") or 0.5, 0.0, 0.95)
s_wi = np.clip(params.get("s_wi") or params.get("connate_water_saturation") or S_WI_CONNATE, 0, 0.8)
c7_plus = params.get("c7_plus_fraction") or params.get("c7_plus") or 0.3
```

Python's `or` treats `0.0` as absent. **Measured** (`v_inert.py` + `audit_verify_4.py`,
`PhDHybridSurrogate`, `base = {mobility_ratio:3.0, pressure:3000, mmp:2500, hcpvi:1.8, sor:0.25,
s_wi:0.25, v_dp:0.5}`):

| input | baseline | explicit `0.0` (**legal physical value**) | other value |
|---|---:|---:|---:|
| `v_dp` | 0.493568 (0.5) | **0.493568** (0.0 → 0.5) | 0.131982 (0.9) |
| `s_wi` | 0.493568 (0.25) | **0.493568** (0.0 → 0.25) | 0.150000 (0.6) |
| `c7_plus_fraction` | 0.493568 (0.3) | **0.493568** (0.0 → 0.3) | 0.492970 (0.8) |

The model *is* demonstrably sensitive to all three (right column), so the **identical output for
`0.0` proves the substitution**. A perfectly homogeneous reservoir (`v_dp = 0`), a dry reservoir
(`s_wi = 0`), and a base oil with no heavies (`c7⁺ = 0`) are all silently replaced by the default.
Second instance, `HybridSurrogate` (`analytical_models.py:657-671`): same idiom for `pressure`,
`mmp`, `hcpvi`, `v_dp`, `s_wi`, `viscosity_oil`, `viscosity_inj`, `perm_md`, `porosity`, `n_o`,
`n_g`, `k_ro_0`, `k_rg_0` — `params.get(k) or default` throughout.

**The correct idiom is `params.get(key, default)`** (absence of a key, not falsiness), with `None`
handled explicitly where `None` is a valid sentinel.

**Impact.** The optimizer's search includes `v_dp = 0.0` and `s_wi = 0.0` boundaries; those
evaluations are indistinguishable from defaults, flattening the objective surface exactly where it
should be most sensitive, and any sensitivity/tornado sweep that extends to 0 returns pure noise.

---

### A.12 MED-tier mathematical findings (summary)

| ID | Finding | Location | Measured / note |
|---|---|---|---|
| MED-03 | Radial WI form mixes the **linear** Darcy constant `0.001127` with a radial `ln(D/r_w)` ⇒ missing `2π` | `well_mechanics.py:197,220` (constant defined at `:28`; correct field form `0.00708 = 2π·0.001127` used at `:50`,`:108`) | **6.2832× low** |
| MED-07 | `_calculate_adaptive_penalty` hard-codes `total_violation = 0.0` (real checks commented out), so its only call site always receives 0.0; **and** the advertised `"death"` method is a bare `pass` | `optimisation_engine.py:1209-1256`, call site `:1783`, `death` branch `:1225-1230` | inert guard; `penalty_factor` / `constraint_handling_method` are dead knobs |
| CRIT-11 | `default_gas_fvf` used with and without ×1000 → **1000×** WAG/SWAG water asymmetry | `profile_generator_fast.py:1117-1118` vs `:1210` | **25 bpd vs 25 000 bpd** at 5000 MSCFD (`v_physics.py`) |
| MED-11 | `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = rate × 0.001` compared against threshold `2.0` — not a mobility ratio | `data_models.py:913,917`; `profile_generator_fast.py:1103-1115` | dimensionally invalid |
| MED-12 | Heuristic fudge factors: `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil+0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000), 0.05, 0.95)` | `surrogate_engine.py:352,354,356` | no derivation — `UNKNOWN — EVIDENCE REQUIRED` |
| MED-05 | Relative permeability and inter-well transmissibility are **not in the evaluation path** | `relative_permeability.py` (all), `well_mechanics.py:184-221`, `surrogate_models.py:371` | saturations come from material balance only (`surrogate_engine.py:376-379`) |

---

## PART B — PHASE 4: CO₂-EOR PHYSICS AUDIT

### B.1 Miscibility & MMP

- **MMP correlations are the healthiest part of the codebase.** `evaluation/mmp.py:96-128`
  implements Cronquist (1978) with the published `Y` exponent; `_calculate_mmp_cronquist` is
  registered at `:305`, auto-selected at `:470-471`. Yellig–Metcalfe exists at
  `data_models.py:2196`. Tests in `tests/scientific/co2/test_mmp_correlations.py` compare Cronquist,
  Yuan, Alston. **LITERATURE-provenance, verified structure.**
- **User override survives** (`LOW-02`): `params["mmp"] = calculate_mmp(..., method='cronquist')` at
  `surrogate_engine.py:936` runs after the user's value is placed in `params` — but the code path
  at `:935-936` is guarded by "if no other data is available", so override *does* currently win.
  Flagged only for **ordering fragility**: a future refactor that moves `:936` earlier silently
  overrides the user.
- **Regime selection is where miscibility physics breaks down**, because it is driven by A.1
  (pressure-independent HCPVI) and A.9 (`E_A` cliff at `M = 1`). The miscibility *switch* (sigmoid at
  `analytical_models.py:423-440`) is well-formed; the two *branches* it selects between are not.

### B.2 Displacement & sweep (Koval / Buckley-Leverett / Craig / Johnson)

Documented lineage in the module docstrings is genuine and correctly attributed:
Koval (1963), Buckley & Leverett (1942), Corey (1954), Johnson (1956), Craig (1971). The
implementations that *exist* are structurally right:

- `E_eff = (0.78 + 0.22·M^0.25)^4` — **verified correct** (Koval's effective fractional-flow term).
- `H = 1/(1−v)²` at `analytical_models.py:174,534,726` — **verified correct**, consistent with
  `data_models.py:57`.
- `E_V = 1 − V_DP^0.7` at `:316` — standard Johnson asymptote, correct.
- `E_d = (soi − sor)/soi` at `:801` — correct.

**But the assembled product is not correct**, for three independent reasons:

1. `E_A` jumps 48.3 % at `M = 1` (A.9).
2. The immiscible product is floored to a constant 0.10 (A.7).
3. The **engine-side** Koval `H` uses `10^(v/(1−v))` instead of `1/(1−v)²`
   (**HIGH-06**, `surrogate_engine.py:894-895`):

   | v | 0.2 | 0.5 | 0.8 | 0.9 | 0.99 |
   |---|---:|---:|---:|---:|---:|
   | `1/(1−v)²` | 1.562 | 4 | 25 | 100 | 1e4 |
   | `10^(v/(1−v))` | 1.778 | 10 | 1e4 | 1e9 | **1e99** |
   | ratio | 1.14× | 2.5× | **400×** | **10⁷×** | **10⁹⁵×** |

   Two formulas for the same quantity in the same evaluation path, diverging to `10⁹⁵`.

**Verdict (B.2):** the *named* correlations are literature-correct; the *composition* is not.
Correct components assembled incorrectly is a more dangerous failure than a wrong component, because
the docstrings and citations survive review.

### B.3 Injectivity, VRR, and WAG

| Aspect | Location | Status |
|---|---|---|
| Peaceman WI `0.00708` | `well_mechanics.py:50,108`; `surrogate_engine.py:306` | **verified correct** |
| Peaceman WI at `:197,:220` | `well_mechanics.py:197,220` | **6.28× low** — radial `ln()` with linear Darcy constant (MED-03) |
| `J = q / nominal_drawdown` with `nominal_drawdown = 500` | `surrogate_engine.py:420-422` | **tautological**: J is defined from the rate it is supposed to predict. Injectivity carries **no information** about `kh`, skin, or drainage radius — HIGH-07 |
| `mu_inj_eff = 0.05` (dry) / `0.50` (WAG) | `surrogate_engine.py:430` | 10× jump at any nonzero water rate; no T/salinity dependence — HIGH-07 |
| VRR assembly | `surrogate_engine.py:406-417` | structurally correct (injected − produced over time), but computed from **uncapped** rates |
| Pressure update `dp = q·dt/(V_p·c_t + J·dt)` | `surrogate_engine.py:455` | **verified correct** — correct implicit-Euler form for a tank with storage + well terms |
| WAG water `rate·bg` vs `rate·wgr·bg·1000` | `profile_generator_fast.py:1117-1118` vs `:1210` | **1000×**: 25 bpd vs 25 000 bpd (`v_physics.py`) — CRIT-11 / MED-11 |
| `default_gas_fvf = 0.005` (units unstated) | `data_models.py:917` | `UNKNOWN — EVIDENCE REQUIRED` |

**Physics impact of the WAG bug:** a WAG flood (the standard CO₂-EOR injection scheme) receives
essentially **no water** (25 bpd against a 5000 MSCFD gas rate ⇒ WAG ratio ≈ 0.005 : 1 instead of
1 : 1), while a SWAG scheme receives 1000× too much. Viscous-fingering control, which is the entire
*point* of WAG, is absent in the WAG configuration. The optimizer cannot discover this: the
injection scheme is a categorical input, not a searched parameter.

### B.4 CO₂ mass balance & storage

**Verified correct (do not flag):** on a full 10-year run (`v_physics.py`, `hybrid`):

```
injected = 36 525 000 MSCF
purchased = 34 994 496.3  +  recycled = 1 530 503.7  =  36 525 000.0   (exact, err 0)
recycled 1 530 504 ≤ produced 1 611 057                                  ✓
```

`M_recycled ≤ M_produced ≤ M_injected` holds **by construction** — recycled is capped as
`min(prod·η_recycle, inj)` at `optimisation_engine.py:871-872`. The wiki's invariant #3 is
satisfied but **not asserted anywhere** (see `res_audit.md` §2.4 for the wording correction).

**Where storage physics fails:**

| ID | Defect | Location | Evidence |
|---|---|---|---|
| **CRIT-10** | Leakage outputs have **no consumers**; `annual_leakage_tonne` is only *produced* by `analysis/material_balance.py:280` and never by the active engine; `max_sandface_pressure_psi` — **0 hits repo-wide** | `wrapper.py:81-99`, `optimisation_engine.py:1367-1381` | measured `total_leakage_tonne = 0.0000` on the 10-yr run while 1 933 999 t were injected |
| **CRIT-09** | Class-VI sandface penalty reads `profiles["pressure"]` / `["reservoir_pressure"]`; the assembled `profiles` dict contains only `yearly_pressure`, `annual_pressure`, `monthly_pressure` | `wrapper.py:61-78` vs `optimisation_engine.py:841-896` | key-absence verified by direct read + run (`pressure_keys: ['pressure','pressure_profile',...]` exist **in the engine's raw output** but are **not copied into `profiles`**) |
| **HIGH-05** | Leak law triggers only *above* `p_safe_ceiling`, while breach flag triggers on `P > p_frac_caprock` **or** tensile/shear failure ⇒ `is_caprock_breached = True` with `caprock_leakage = 0` | `geomechanics_fault.py:170,174-181` | `caprock_leakage` is gated on `:174` alone |
| **HIGH-15** | `gas_trapping = 1 − s_gc` — **inverted**; `s_gc = 0.05` ⇒ 95 % trapped | `surrogate_models.py:238` | measured 0.95 / 0.90 / 0.80 for `s_gc` 0.05/0.10/0.20; `tests/scientific/co2/test_co2_trapping_mechanisms.py:18-34` **asserts the inversion** |
| `cum_stored = inj − prod` | ignores leakage entirely | `surrogate_engine.py:543` | consistent with `total_leakage ≡ 0`, so mass *appears* closed |

**Net effect:** the CO₂ ledger is internally exact (B.4 first block) but **closed on the wrong
system** — leakage, which the geomechanics module computes, never enters it, so
"permanent storage" is reported as `injected − produced` with no loss term at all.

### B.5 Trapping mechanisms

- **Structural/stratigraphic:** `structural_trapping_factor` is defined on `Co2StorageParams`
  (`data_models.py:1596`, default 0.2) but `storage.py:92` reads it via
  `getattr(storage_params, "structural_trapping_factor", 0.85)` from **`AdvancedEngineParams`**,
  which has no such field ⇒ fallback **0.85**. Measured consequence (**CRIT-08**, `v_misc.py`):

  | scenario | `S_cont` | threshold | prune possible? |
  |---|---:|---:|:--|
  | avg P well below frac | 0.7502 | 0.30 | **no** |
  | avg P **= fracture pressure** (worst case) | **0.4400** | 0.30 | **no** |

  The **floor of the containment score (0.44) is above the pruning threshold (0.30)** ⇒ the
  containment prune is **unreachable for any input**. A plume at fracture pressure is never pruned.
- **Residual (capillary) trapping:** inverted — HIGH-15 above.
- **Solubility trapping:** `calculate_co2_solubility_scf_per_stb` (`pvt_state.py:246-265`) uses
  `650·(API/35)·(P/2500)^0.85·(560/T_R)·x^1.1`, attributed to "Emera-Sarma / Simon-Graue" — but
  neither paper contains that functional form. **UNKNOWN — EVIDENCE REQUIRED** (MED).
- **Mineral trapping:** not modeled (appropriate for a screening tool; noted as an explicit scope
  limit, not a defect).

### B.6 Geomechanics & Class VI conformance

**Verified correct — do not flag:**

- Fracture-pressure limit `λ = 0.9` (`storage.py:90`) is consistent with EPA Class VI practice and
  with `AGENTS.md` invariant #4 (`P_sandface ≤ 0.90 × P_frac`).
- Stress-path and slip-tendency algebra (`geomechanics_fault.py:98,129-132,188-198`) —
  `τ_max = ½|σ_v−σ_h|`, `σ_m = ½(σ_v+σ_h)`, `τ_crit = c₀ + σ_m·tanφ`, `τ/σ_n'` slip tendency:
  standard Mohr-Coulomb, dimensionally consistent, verified.
- Effective-stress sign convention `σ_n' = σ_n − α·P` (`:191`) — correct (Biot `α`, default 1).

**Where it fails:** the safety *enforcement*, not the safety *math*:

- **CRIT-09** — the Class-VI sandface penalty never runs (key absence).
- **CRIT-10** — leakage never reaches the objective.
- **CRIT-08** — containment pruning unreachable (fallback floor above threshold).
- **HIGH-05** — breach flag and leak gate disagree.
- **HIGH-14** — two sandface models coexist: `wrapper.py:64-67` (`P_sf = P_max + q/II`) vs
  `surrogate_engine.py:359` (`+400 psi`, capped). Which one a result reflects depends on which
  code path produced it.
- **HIGH-13** — remediation cost uses `max(carbon_tax, 100.0)` (`wrapper.py:97`), silently
  overriding any configured value below 100 (default is 75 at `data_models.py:1734` ⇒ the *default*
  is overridden).

**Categorical statement:** the geomechanical *equations* are the most defensible in the repository;
the geomechanical *constraints* are inert. A reviewer reading `storage.py` and
`geomechanics_fault.py` would conclude the project enforces Class VI compliance. It does not.

### B.7 PVT property suite — full status

| Property | Location | Status |
|---|---|---|
| PR-EOS constants `0.45724`, `0.07780`, `α`-correlation `0.37464 + 1.54226ω − 0.26992ω²` | `pvt_state.py:111-115` | **verified correct** (standard PR) |
| Dense-root selection (`min` above `Pc`, `max` below) | `pvt_state.py:150-155` | **verified correct** |
| CO₂ density clip `[2, 1100] kg/m³` | `pvt_state.py:161` | legitimate stabilization, documented |
| `B_CO₂ = 327.362/ρ` | `pvt_state.py:201` | **verified correct** (`28.3168 m³ × 6.2898 bbl/m³ = 178.1`… note: `1 MSCF = 28.3168 m³ × 1.838 kg/m³ = 52.046 kg`; `52.046/ρ × 6.2898 = 327.36/ρ` ✓) |
| `Bo(P)` no bubble point | `pvt_state.py:267-307` | **CRIT-03** — `dBo/dP > 0` |
| HC-gas `Bg` | `pvt_state.py:373` | **CRIT-04** — 31.7× |
| HC-gas `Z` "Hall-Yarborough" | `pvt_state.py:366` | **CRIT-05** — linear, `Z > 1` where `Z < 1` is physical |
| CO₂ viscosity polynomial | `pvt_state.py:218-226` | **HIGH-16** — `μ/μ₀ = 0.589` at `ρ_r = 1` vs ≈3.4 expected; measured 0.0206→0.0443 cP. **UNKNOWN provenance** |
| `c_g = (1/P)(0.35y + 0.85(1−y))` | `pvt_state.py:388-394` | **HIGH-17** — 4–12× too high for dense CO₂, discontinuous at 1500 psi |
| Harmonic `B_g` mixing `1/(y/B₁ + (1−y)/B₂)` | `pvt_state.py:385` | **verified correct** (volume-weighted harmonic mean is the right mixing rule for FVF) |
| Vasquez–Beggs `R_s` constants | `pvt_state.py:239-241` | **verified correct** (`0.0178/1.1870/23.931` for API > 30) |
| Lee–Gonzalez–Eakin `μ_hc` | `pvt_state.py:376-380` | structure correct (`k,x,y` form); clip `[0.01, 0.06]` is tight but plausible |

---

## PART C — CATEGORIZED VERDICTS (no composite score)

### C.1 SOFTWARE CORRECTNESS

**Verdict: PASS WITH SERIOUS DEFECTS.** The engine runs, returns 92 keys, and does not propagate
NaN into selection (`FAILURE_PENALTY`, `optimisation_engine.py:725-727,1543,1576`). But 5 undefined
names (1 failing a mandatory gate), 77 discarded computations, and a family of *produced-but-never-
consumed* outputs (CRIT-09, CRIT-10, MED-06) mean parts of the safety machinery are decorative.
Detail: `res_audit.md` §6.

### C.2 NUMERICAL STABILIZATION

**Verdict: OVER-APPLIED AND DISGUISED.** Legitimate, documented guards exist
(`Z ≥ 0.1`, `ρ ∈ [2,1100]`, `B_{CO₂} ∈ [0.2,15]`, `μ ∈ [0.015,0.12]`) and are fine. But four
clips **return stabilization output as a physical answer** with no indication:

| Clip | Effect |
|---|---|
| `clip(rf, 0.10, cap)` (`analytical_models.py:326`) | CRIT-07 — constant 0.10 over a 27-point grid |
| `clip(rf, 0.05, …)` (`:200,:475`) | HIGH-03 — 0.05 at zero throughput |
| `clip(sweep, 0, 0.75)` (`:559`) | CRIT-06 — hides `exp` underflow as a plausible 0 |
| `clip(rf, 0, 1−swi−sor)` (`surrogate_engine.py:510`) | HIGH-01 — binding; discards raw 0.7125 for 0.5000 |

A clip that hides a math error is not stabilization; it is defect suppression. Category 2 and
category 1 must not be credited together, which is precisely why no single score is issued.

### C.3 PHYSICAL CONSISTENCY

**Verdict: FAIL.** Independent of any calibration question:

- `Bo` increases with pressure at positive `dBo/dP ≈ +1.1e-4 1/psi` with **no dissolved gas**
  (CRIT-03) — violates thermodynamics above the bubble point.
- HCPVI has **no pressure dependence at all** (CRIT-01) — injects a monotone pressure bias into
  every sweep correlation.
- `Bg_hc` wrong by 31.7× (CRIT-04); `Z > 1` and monotone where physics requires `Z < 1` and a
  minimum (CRIT-05).
- Favorable mobility ratio `M < 1` ⇒ **zero** recovery (CRIT-06).
- Immiscible RF constant regardless of reservoir properties (CRIT-07).
- WAG receives 1000× too little water (CRIT-11) — the displacement-control mechanism is absent.
- Trapping efficiency **inverted** (`1 − s_gc`) and asserted by its own test (HIGH-15).
- Containment score floor 0.44 > pruning threshold 0.30 ⇒ **no input can ever be pruned** (CRIT-08).
- Reported NPV is not a function of reported production (CRIT-02).

**Mass conservation is the one strong invariant** (B.4: exact to the digit), *but* it is enforced by
construction rather than asserted, and it closes over a system from which leakage is excluded.

### C.4 EMPIRICAL CALIBRATION

**Verdict: NOT ESTABLISHED — 53 % of constants unverifiable.**
From `audit/parameter_provenance.csv` (91 rows): **LITERATURE 21**, **EMPIRICAL 6**,
**CALIBRATED 16**, **UNKNOWN — EVIDENCE REQUIRED 48**.

The LITERATURE set is real and verified by this audit (PR coefficients, Corey, Craig, Johnson,
Vasquez–Beggs, Darcy `0.00708`, IUPAC CO₂ critical density, `λ = 0.9`). The UNKNOWN set is not
"wrong" — it is **undefendable**: leakage base rate `0.05 t/d/psi`, breach multiplier `×5`,
overpressure exponent `1.5`, `nominal_drawdown = 500 psi`, `mu_inj_eff` 0.05/0.50, `s_ref = 0.40`,
`x_co2` factor `0.135`, the CO₂ μ-polynomial coefficients, `c_g` weights 0.35/0.85, the
"Hall-Yarborough" coefficients, `0.1587`, all five CO₂ density constants, the containment weights
0.5/0.3/0.2, and both `FAILURE_PENALTY` copies.

Under the truth hierarchy, provenance (row 5) outranks benchmark agreement (row 8). A number that
cannot be traced cannot be defended even when it happens to reproduce a curve.

### C.5 PREDICTIVE VALIDITY

**Verdict: NOT ESTABLISHED.**

Reasoning, stated categorically and without reference to any score:

1. **No experimental evidence exists in the repository for the active configuration.** The
   validation material (`tests/validation/spe5_config.py`, CMG comparison paths) targets
   `phd_hybrid` and dormant engines; the shipped default is `"hybrid"`
   (`data_models.py:1725-1730`, `config/base_config.json:24,746`,
   `optimisation_engine.py:264-270`), while tests use `phd_hybrid` **72 times vs 6** for
   `"hybrid"` (**HIGH-09**). Validation of a configuration that is not the default does not
   transfer.
2. **Benchmark agreement would not rescue it anyway.** Per the truth hierarchy, benchmark agreement
   sits at row 8. With CRIT-01…CRIT-07 unresolved — defects in rows 1–3 — agreement with another
   simulator could only indicate shared assumptions, not correctness. **No benchmark credit is
   taken anywhere in this audit.**
3. **The optimizer's objective surface contains directions unmoored from physics**: NPV moves 45 %
   while reported RF and oil are identical (A.2); `v_dp = 0.0` is indistinguishable from default
   (A.11); 2 of the searched gene pairs are inert (CRIT-13). A search on such a surface converges
   to whatever the artifacts favor.
4. **Coverage does not evidence validity**: 329 passing tests coexist with all of the above because
   one test *asserts* the 48.3 % cliff (A.9) and another *asserts* inverted trapping (B.4).

**What would be required to change this verdict** (documented, not implemented — audit-only rule):

1. Resolve CRIT-01…CRIT-05 (dimensional + PVT) — these are row-1/row-2 defects.
2. Single-evaluate NPV and RF (CRIT-02); delete or reconcile the RF cap (HIGH-01).
3. Fix the sweep branches (CRIT-06, CRIT-07, HIGH-02) **before** removing any RF floor, in the
   order specified in `res_audit.md` §5.
4. Wire or remove the inert constraints (CRIT-08, CRIT-09, CRIT-10, CRIT-13).
5. *Then* obtain at least one independent validation for the **default** `hybrid` configuration —
   a published CO₂-EOR case with lab-measured `B_o(P)`, `B_g(P)`, `Z(P)`, MMP, and a field or
   slim-tube recovery curve. Only after steps 1–5 does benchmark comparison mean anything.

---

## PART D — VERIFIED CORRECT (do not flag)

Recorded to prevent false positives in future sessions. All re-verified numerically this session
unless noted.

1. **Pressure-update ODE** `dp = q·dt/(V_p·c_t + J·dt)` — `surrogate_engine.py:455` — correct
   implicit form.
2. **Gravity number** `N_g = k·Δρ·cosθ·4.3948e-5 / (…)` — `analytical_models.py:793` — constant
   correct for field units.
3. **Capillary number** `N_c` with `3.5e-6` — `analytical_models.py:752` — correct unit factor.
4. **Standing solution-Gor structure** and Vasquez–Beggs constants — `pvt_state.py:239-243`.
5. **Beggs–Robinson-style oil viscosity structure** — `pvt_state.py` (documented form).
6. **Peaceman `0.00708`** — `well_mechanics.py:27,50,85,108`; `surrogate_engine.py:306`.
7. **`E_eff = (0.78 + 0.22·M^0.25)^4`** — Koval effective fractional flow, normalized correctly.
8. **Koval `M = 1` branch continuity** — `analytical_models.py:542-547` (the *singular* branch is
   right; the *neighbouring* branch is CRIT-06).
9. **Peng–Robinson constants and dense-root selection** — `pvt_state.py:103-115, 147-155`.
10. **Harmonic `B_g` mixing rule** — `pvt_state.py:385`.
11. **Mohr–Coulomb stress path / slip tendency** — `geomechanics_fault.py:98,129-132,188-198`.
12. **VRR assembly** — `surrogate_engine.py:406-417`.
13. **CO₂ ledger arithmetic** — `injected = purchased + recycled` exact; `recycled ≤ produced`
    holds by construction (`optimisation_engine.py:871-872`).
14. **MMP Cronquist implementation** — `evaluation/mmp.py:96-128,305,470-471`.
15. **`E_d = (soi − sor)/soi`** and **`E_V = 1 − V_DP^0.7`** — `analytical_models.py:801,316`.
16. **`B_CO₂ = 327.362/ρ`** derivation — `pvt_state.py:175-176,201`.
17. **`λ_limit = 0.9 × P_frac`** — `storage.py:90`, consistent with invariant #4.
18. **Lee–Gonzalez–Eakin `μ_hc` functional form** — `pvt_state.py:376-380`.

---

## PART E — Cross-check of the prior 18-SCI-FLAW register

`audit/scientific_flaws/scientific_flaws.csv` (prior session) was re-verified line-by-line against
the working tree; every claim was re-measured rather than inherited.

| Prior ID | Status this session |
|---|---|
| SCI-FLAW-01 | **RESOLVED** — formula replaced; now `profile_generator_fast.py:958-969,466`. The wiki's test of record (`test_koval_fractional_flow_mobility_inversion`) was **renamed** to `…_monotonicity` and that test re-derives the formula inline instead of calling the module (HIGH-18), so the resolution rests on the source read, not on the test. |
| SCI-FLAW-02 | **RESOLVED in `data_integration_engine.py`** — `:431` is now `oil_fvf = 1.2 − 0.000015·(P−4000)`, i.e. `dBo/dP = −1.5e-5 < 0` (the cited `1.2 + 0.0001·(P−4000)` is gone). The **same defect class is still live in the active engine** as CRIT-03 (`pvt_state.py`, `dBo/dP ≈ +1.0e-4 1/psi`). |
| SCI-FLAW-03 | **RESOLVED in `data_integration_engine.py`** — `:435` `exp(+0.00005·(P−4000))` and `:441` `exp(+0.0002·(P−4000))`; both signs are now positive (`dμ/dP > 0`). |
| SCI-FLAW-04, -08, -18 | **CANNOT BE RE-VERIFIED — cited tree deleted.** All three cite `core/unified_engine/…`; neither `core/unified_engine/` nor `deprecated/` exists in the working tree. `agent_wiki/architecture/overview.md:71` and `README.md:29` claim the engines were *"relocated into `deprecated/`"*, which never happened (MED-16). |
| SCI-FLAW-09, -10 | in `core/simulation/recovery_models.py` — exists but dormant behind `RECOVERY_MODELS_AVAILABLE = False` (HIGH-11); **not re-verified**. |
| SCI-FLAW-05 | **OPEN, not re-verified this session** |
| SCI-FLAW-06 | **CONFIRMED** → CRIT-12 / CRIT-02 (line refs updated to `surrogate_engine.py:490-510,513-522`) |
| SCI-FLAW-07 | **CONFIRMED** |
| SCI-FLAW-11 | **CONFIRMED, line refs stale** → see CRIT/… in register; the test of record asserts the truncation instead of the correct bound (HIGH-18) |
| SCI-FLAW-12 | **STALE — superseded**: `B_GAS_RB_PER_MSCF = 1.0` at `optimisation_engine.py:85`, and `p_psia`/`t_f` are now accepted (`pvt_state.py:163-201`) |
| SCI-FLAW-13 | **RESOLVED** |
| SCI-FLAW-15 | **CONFIRMED, line refs stale** — still `profile_generator_fast.py:1000-1005` (was `:983-986`) |
| SCI-FLAW-16 | **CONFIRMED, formula text stale** — cliff is 48.3 % for `E_A = 0.517 − 0.072·log10(M)` (HIGH-02) |
| SCI-FLAW-17 | **CONFIRMED** (HIGH-15) |

**Verification-tier caveat (HIGH-18).** Of the 36 collected tests in `tests/scientific/`, 5 import production symbols
they never call and 6 never import production code at all; three of them assert that a defect *exists* as the expected
outcome. `agent_wiki/verification/test_matrix.md` further declares 42 items across 16 subdirectories and lists **6 tests
that do not exist anywhere in `tests/`**. A green suite therefore cannot be read as analytical verification for the
items above — see `res_audit.md` §1.4 and `audit/scientific_flaws.md` HIGH-18 / MED-16.

Full detail with line-level evidence: `audit/scientific_flaws.md` §5.

---

## PART F — Evidence index

| Item | Path |
|---|---|
| Master register (52 findings) | `audit/scientific_flaws.md` |
| Parameter provenance (91 rows) | `audit/parameter_provenance.csv` |
| Software/anti-pattern audit | `res_audit.md` |
| PVT re-verification (`Bo`, `Bg`, `Z`) | `agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/evidence_scripts/v_pvt2.py` |
| Sweep/RF grid (CRIT-06, CRIT-07, HIGH-02) | `…/evidence_scripts/v_rf.py` |
| HCPVI + mobile-oil term (CRIT-01, HIGH-01) | `…/evidence_scripts/v_hcpvi.py` |
| Full engine run, dual-ledger NPV (CRIT-02) | `…/evidence_scripts/v_engine.py` |
| Clip-binding proof (HIGH-01) | `…/evidence_scripts/v_clip.py` |
| WAG ratio + CO₂ ledger (CRIT-11, invariant #3) | `…/evidence_scripts/v_physics.py` |
| Koval H, containment floor, `or`-defaults | `…/evidence_scripts/v_misc.py` |
| Inert genes / key mismatches (CRIT-13) | `…/evidence_scripts/v_inert.py` |
| Test-suite integrity AST scan (HIGH-18) | `…/evidence_scripts/v_tests2.py` |
| Register cross-tab derivation (52 findings) | `…/evidence_scripts/v_counts2.py` |
| Original scripts (unmodified copies) | `%TEMP%/opencode/v_*.py`, `audit_verify_*.py` |
| Prior register (18 `SCI-FLAW-*` rows) | `audit/scientific_flaws/scientific_flaws.csv` |
| Toolchain raw output | `audit/code_quality/`, `audit/runtime/` |
| **Run audit record + Verdict/Proposal** | `agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/audit.md` (indexed in `…/index.md`) |
| Wiki corrections applied (MED-15 / MED-16) | 23 files under `agent_wiki/` — see `res_audit.md` §2.4 |
