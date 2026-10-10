# Specification Corrections Log

**Owner decision, 08-10-2026:**

> *"`3D_THMC_docs` are references but inside it should be documents that is correcting original specs with
> references. All approved updates, changes and branches are recorded to wiki. Wiki serves as a guideline
> for development and evaluation."*

**What this means operationally:**

| Rule | Detail |
|---|---|
| `3D_THMC_docs/` | **Immutable reference originals.** Never edited |
| `agent_wiki/` | **The authoritative corrected specification.** Where the two differ, **the wiki wins** |
| Every deviation | Logged here with: source, original text, correction, and rationale |
| Approved branches | Recorded here — a spec variant that was considered and rejected is as valuable as one that was adopted |
| Provenance | Every correction cites the file and line it corrects |

This file is the audit trail. It exists because a corrected spec without a correction history is
indistinguishable from a spec that was never wrong — and because the 05-10-2026 audit showed what happens
when documentation and code drift apart without a record.

---

## Corrections from S1 — Output Data Schema & Visualisation Specification (08-10-2026)

Source: `3D THMC Reservoir Simulator Full Output Data Schema & UI Visualization Specification.md`,
285 lines, attached 08-10-2026.

### ✅ Adopted — improvements that supersede the design set

| # | Correction | Supersedes | Rationale |
|---|---|---|---|
| 1 | Full **6-component symmetric** anisotropy tensor in output | D7 payload carried only $(K_x,K_y,K_z)$ — **CONF-41** | Fixes the schema-level cause: MPFA-O's off-diagonal correction was being silently dropped on hand-off |
| 2 | Porosity split into **matrix $\phi_m$, fracture $\phi_f$, total $\phi_{tot}$** | D1/D2 implied the distinction but never stated it | Without it the DP/DP sections of the design set are unrepresentable |
| 3 | **Land hysteresis** selected for $k_r$ branch tracking | D1 §6.2 offered Land *or* Killough *or* Carlson | Removes ambiguity |
| 4 | **Drucker–Prager** added alongside Mohr–Coulomb | D1 §3.3 | Correct for frictional rock; avoids Mohr–Coulomb's unbounded shear strength at high confinement |
| 5 | **Siderite + ankerite** added as CO₂ sinks | D1 §6.5 listed only $\text{CaCO}_3$ | $\text{FeCO}_3$ and $\text{CaMg(CO}_3)_2$ are real mineralisation sinks. ⚠️ Requires Fe²⁺ — see **CONF-67** |
| 6 | Concrete **5-facies** set: Sand, ShSand, Sh, Lst, argLst | D7 used generic A/B/C/D | Implementation-ready |
| 7 | Concrete **7-mineral** and **6-species** sets | D1 §5, D2 §2.1 named minerals inconsistently | Implementation-ready |
| 8 | **Additive skin decomposition** $S_{tot}=S_{mech}+S_{perf}+S_{cake}+S_{asph}$ | — | Distinguishes mechanical damage from chemical deposition. S1's own example shows $S_{asph}=4.5$ calling for solvent wash, **not re-frac** — directly actionable. ⚠️ Negative-component policy missing — **CONF-68** |
| 9 | **`float32` field precision** | design set silent | Halves I/O; adequate for display and analytics |
| 10 | **Three temporal resolutions**: micro-step / daily / monthly-yearly | design set silent | Separates convergence diagnostics from engineering series from accounting |
| 11 | `(P_cell − MMP) = 0` **isosurface** | — | Highest-value single visualisation: turns "miscibility degraded" into a locatable zone |
| 12 | **Tensor-ellipsoid glyph** view | — | Answers a real question — align wells with $K_{max}$ to avoid premature water breakthrough |
| 13 | `float32[N_cells][6]` tensor layout | — | Directly renderable by instanced rendering; no conversion |

### 🔴 Adopted with correction — physics conflicts S1 inherited

| # | Correction | Conflict | Detail |
|---|---|---|---|
| **C-14** | 🔴 **Economics removed from Domain 9** | **CONF-64** | S1 §Domain 9 (lines 390–414) specifies CAPEX, NOCF, discounting and NPV unification **as engine output**. Contradicts **INV-3** and the owner's decision that a separate economic engine does field development. **KEEP** the physical part (cumulatives, $\|R_m\|$ norms, $E_v/E_a/E_m$, $RF$, $\Delta t^n$, $N_{Newton}^n$, $\lambda_k$, Havlena–Odeh). **MOVE** CAPEX, NOCF, discounting, DCF, NPV rule → economic engine |
| **C-15** | 🔴 **Python artefacts removed** | **CONF-65** | S1 line 414 requires agreement between `economic_npv_usd`, `results["npv"]` and `cash_flows_yearly.csv`. All three are **current Python-engine symbols** (`surrogate_engine.py:172`; `tests/test_physical_invariants.py:8,159,181,184,211`). None belongs in a Rust engine's output schema. Re-scope as: economic engine's own artefacts, with its own naming |
| **C-16** | 🔴 **Koval factor form to be unified** | **CONF-63** | S1 lines 71 and 335 give $K_{koval}=H\cdot E\cdot M$. Shipped code: $K=H\cdot E_{eff}$, $E_{eff}=(0.78+0.22M^{0.25})^4$. Design set D2 §6.1: $K_{Koval}=H\cdot E_{eff}$. **Three forms.** S1's form multiplies $M$ **linearly**, discarding Koval's empirical **sublinear** $M^{0.25}$ law → over-predicts fingering. **Unify before M3** |
| **C-17** | 🔴 **Do not transcribe the $f_g$ formula** | **CONF-01** closure | S1 line 67 reprints the formula whose *sign* was retracted by measurement and whose *closure* remains defective — no $S_{or}$, linear not Corey, no water term, unguarded at $S_g\to S_{gc}$. Corrected closure required: [`spec_defects.md`](spec_defects.md) §1 |
| **C-18** | 🔴 **Do not adopt the HCPVI cap** | **CONF-02** | S1 line 77. Inert at operating point, no mobility dependence — would hide exactly the defect M6 must prove absent. Use a mobility-sensitive sweep ceiling |
| **C-19** | ⚠️ **Resolve the self-contradictory utilisation floor** | **CONF-35** | S1 line 83 states "≥ 2.5 MSCF/STB (**≥ 0.12 tonne/STB**)" against reference "**0.25–0.50 tonne/STB** (5–10 MSCF/STB)". A 2.1× contradiction **inside one sentence** |
| **C-20** | ⚠️ **Unstate the 20 000–60 000 MSCFD range** | **CONF-54** | S1 line 329 presents it as the design target; D5 §1.1 cites the same range as the **anti-pattern signature** of a clamped rate. One of the two is wrong |
| **C-21** | ⚠️ **Fix $t_{bt}$** | **CONF-15** | S1 line 333. $(1-S_{wi})$ ignores $S_{or}$ and gas saturation; no unit basis for $q_{inj,\text{pattern}}$ or $K_{koval}$ |
| **C-22** | ⚠️ **Record the 1 000 BOPD clamp's position** | **CONF-07** | S1 line 329. Still unstated: before or after drift-flux, and how a clamped rate propagates to mass balance |
| **C-23** | ⚠️ **`ε_p` needs a plasticity model** | **CONF-66** | S1 line 223 emits plastic compaction strain. D1 §3.3 / D2 §3.1 specify **linear Biot elasticity** only — no yield surface, hardening or flow rule. Either specify elastoplasticity or drop $\epsilon_p$ |
| **C-24** | ⚠️ **Add Fe²⁺ to the aqueous species set** | **CONF-67** | S1 line 258 adds siderite/ankerite; line 190's species set has Ca²⁺, Mg²⁺, no Fe²⁺. Mineralisation terms cannot close |
| **C-25** | ⚠️ **Define the negative-skin policy** | **CONF-68** | S1 line 344 sums skins additively; D4 §2.3 makes $S_{perf}$ negative on acidising. No policy for a negative summand |
| **C-26** | ⚠️ **Decide mid-year discounting deliberately** | **CONF-62** | S1 lines 406–412 mandate $DF_y=(1+d)^{-(y-0.5)}$. Live engine uses end-of-year (`surrogate_engine.py:649-650`). **Measured +4.88 % NPV** at $r=0.10$ over 15 yr. Adopting it re-baselines every published result — must be a recorded decision |

### 🔵 Additions required by the new architecture

| # | Addition | Detail |
|---|---|---|
| **C-27** | `pyarrow` and `duckdb` added to Python dependencies | Measured 08-10-2026 in `.venv`: **both MISSING**. `h5py` 3.16.0 ✅ present. Without these the satellite cannot read Layer 3 (Parquet / DuckDB) at all |
| **C-28** | HDF5 chunking + compression is **mandatory** | Measured: **116 float fields per cell per timestep**. At SPE-10 scale (1.1 M cells) that is **510.4 MB per timestep**, **51.0 GB for 100 steps**. Uncompressed, unsplittable field output is not viable |
| **C-29** | Capability declaration mechanism | The full schema spans domains the engine will not populate early. A run must declare **which domains it populated**. Without this, "absent field" is indistinguishable from "failed module" — and that ambiguity violates **INV-1**. See [`engine_invariants.md`](engine_invariants.md) §7 |
| **C-30** | `pyarrow`/`duckdb` reading is the satellite's only coupling point | Because the engine writes **standard open formats**, the Python satellite reads them with existing tooling. **No FFI, no shared library, no Rust in the Python process.** This is the cleanest available resolution of the P1 separation tension |
| **C-31** | Volume-translation made mandatory | Baled et al. (2012) gives density MAPD **1–2 %** (VT-SRK) against the design set's declared **3–9 %** for bare PR/SRK (**CONF-49**). Bare cubic EOS will fail the M1 gate |
| **C-32** | `source` provenance mandatory per fluid value | **CONF-31**. Abudour et al. (2014) is a **factor-of-two** fallback on $k_{ij}$, not a primary source. An un-sourced $k_{ij}$ is a defect |

---

## Open branches — considered, not yet decided

Recorded so they are not re-litigated or silently lost.

| # | Branch | Status |
|---|---|---|
| **B-1** | **PCA / Karhunen–Loève basis for 116 fields** | ⚠️ **Unresolved and worth deciding.** 116 correlated fields per cell is highly redundant. A reduced basis would cut storage and — more importantly — cut the dimensionality of the **RL observation space**. Consider before the dataset grows |
| **B-2** | Store master grid + deltas, or every field every step? | Open. Delta storage is smaller; master-grid storage is simpler to query. Interacts with **C-28** |
| **B-3** | `duckdb` vs `libsql` in the browser | S4 lists both. DuckDB-WASM is more capable; libSQL is lighter. ⚠️ **Note this is the *UI* choice and UI is out of scope until the engine is verified** |
| **B-4** | Should the economic engine read HDF5, or the engine's DuckDB output? | Open. Determines whether economics can run without loading 3D fields |
| **B-5** | Is `training_pairs` in the same HDF5 run, or a separate pgvector store? | Open. Affects whether a failed run leaves partial training data — interacts with **INV-1** and **INV-4** |
| **B-6** | Retain all micro-steps, or only those of failed/CI runs? | ⚠️ Open, and a **volume** question — micro-steps are the highest-resolution tier |
| **B-7** | Does `audit_comp` mirror into PostgreSQL, or stay markdown-only? | Open. `INV-5` favours keeping the markdown register as the git-tracked source of record |

---

## Verification discipline for this log

A correction is only **closed** when:

| # | Requirement |
|---|---|
| 1 | It cites the source file **and line** it corrects |
| 2 | The correction is stated as a **rule**, not a preference — an implementer can act on it |
| 3 | Where it rests on a measurement, the measurement is cited with its command |
| 4 | Where it supersedes an earlier wiki claim, that claim is **struck through, not deleted** |

> ⚠️ **Rule 4 is why this file exists.** On 08-10-2026 I recorded CONF-01 as a blocking finding; the
> reservoir engineer's review showed **that claim was wrong** — a viscosity ratio had been substituted for
> a mobility ratio. The original claim is preserved and struck through rather than removed, because
> "someone once believed this, and here is why it was wrong" is the record that prevents the same error
> recurring. See [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) §2.
---

## Corrections from R2 — Reservoir Engineer Ruling, 08-10-2026

Adjudication: [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) §6–§9.

### ✅ Resolved — and my framing was wrong

| # | Correction | Against |
|---|---|---|
| **C-38** | 🔵 **CONF-35 RESOLVED** by unit conversion: $1\ \text{MSCF}\ \text{CO}_2 \approx 0.0519$ tonne. Floor $2.5\ \text{MSCF/STB} = 0.13$ t/STB; benchmark $5$–$10\ \text{MSCF/STB} = 0.26$–$0.52$ t/STB. ⚠️ **I recorded CONF-35 as a "2–4× contradiction about the same hard floor". That was wrong — I compared a FLOOR against a BENCHMARK.** Both are correct under one conversion | My own framing — **withdrawn** |

### ✅ Accepted rulings applied

| # | Correction | Against |
|---|---|---|
| **C-39** | 🔴 **CONF-63**: standardise $K_{Koval}=H\cdot E_{eff}$, $E_{eff}=(0.78+0.22M^{0.25})^4$ before M3 | S1 lines 71, 335 |
| **C-40** | 🔴 **CONF-64/65**: **purge all monetary variables and Python filenames.** Core emits strictly physical $Q_o, Q_w, Q_g, Q_{inj}, W_p, G_p, N_p, \|R_m\|, E_v, E_a, E_m, RF$ | S1 §Domain 9 |
| **C-41** | ✅ **CONF-66**: remove $\epsilon_p$ from the elastic baseline; mark `NotImplemented` until a yield criterion + non-associated flow rule exists | S1 line 223 |
| **C-42** | ✅ **CONF-67**: aqueous species → $\text{Ca}^{2+}, \text{Mg}^{2+}, \text{Fe}^{2+}, \text{H}^+, \text{HCO}_3^-, \text{SO}_4^{2-}, \text{Cl}^-$ (7). ⚠️ **do not also store `pH`** — derive it from $H^+$ | S1 line 190 |
| **C-43** | ⚠️ **CONF-68 floor corrected**: $S_{min}=-\ln(r_e/r_w)+\text{margin}$, **not** a universal `-5.0`. Measured: `-5.0` is safe only for $r_e > 52.5$ ft | Ruling §6.3 |
| **C-44** | 🔴 **CONF-02 / CONF-54**: remove the inert HCPVI cap and the 20 000–60 000 MSCFD range from S1 | S1 lines 77, 329 |
| **C-45** | ✅ **Two-output contract**: spatial → HDF5/VTK-HDF; time-series → Parquet/DuckDB. **The economic engine never reads HDF5** | Closes branch **B-4** |
| **C-46** | ✅ **Chunked HDF5 with `zstd` compression** + **active-frame RAM cache** for interactive scrubbing | Volume finding §6 |

### ⚠️ New requirements from the Python post-mortem

Full record: [`python_attempt_postmortem.md`](python_attempt_postmortem.md).

| # | Requirement | Milestone |
|---|---|---|
| **C-33** | ⚠️ **Heidemann–Khalil 2D critical solver** — a negative-flash method for reliable two-phase root location. **Absent from all eight documents**; the design set specifies only Michelsen TPD + Rachford-Rice | **M2** |
| **C-34** | ⚠️ **Exponential soft-start** $q_{inj}(t)=q_{target}(1-e^{-t/\tau_{well}})$, with a stated $\tau_{well}$. The master map names "soft-start" but gives no expression | **M3** |
| **C-35** | 🔴 **Well control modes embedded in the global Jacobian.** D4 §4.2 specifies SSSV as *external pseudocode* — flipping the control mode outside the Jacobian **is** the infinite ping-pong failure mode | **M5** |
| **C-36** | ✅ **Permeability-collapse selection rule**: **Verma–Pruess** for clogging/percolation ($k\to0$ at $\phi_c>0$); **Kozeny–Carman** for compaction-driven porosity change. Closes D1 §5.2's missing rule | **M3** |
| **C-37** | 🔴 **Correct the stated reason for the $f_g$ purge.** Action right; *"the sign was inverted"* is **false**. The defect is the **closure** | now |

### Owner rulings recorded

| # | Ruling |
|---|---|
| 1 | ⚠️ **CONF-64…68 attribution to be confirmed.** The owner notes these "refer to our surrogate engine, not 3D THMC". ⚠️ **Recorded with a caveat** — see below |
| 2 | ✅ **`sr3_reader.py` will be updated and expanded, with a proper Rust rewrite** |
| 3 | ✅ **Long load time for complex maps and fields is acceptable.** Geometry loads fine |
| 4 | 🔵 **`pyarrow` / `duckdb` are not gaps — they are leftovers.** Most satellites are developed *after* the compositional engine, so the dependencies are added then. Not a P1 blocker |
| 5 | 🔴 **This wiki states CURRENT state.** Forward-looking specifications are targets, not present reality |
| 6 | ✅ **All micro-steps retained** ("we can update it later") — closes branch **B-6** |
| 7 | ✅ **Economic engine reads DuckDB / Parquet** — closes branch **B-4** |

### ⚠️ Caveat on item 1 — recorded, not silently accepted

The owner states CONF-64…68 refer to the **surrogate** engine. On review:

| ID | Ruling's fix applies to | Applies to the surrogate, or to S1? |
|---|---|---|
| **CONF-64/65** | purge monetary terms + Python filenames **from S1** | **S1** — the fix is explicitly about the new spec |
| **CONF-66** | remove $\epsilon_p$ **from the baseline elastic schema** | **S1** Domain 5 geomechanics — a new-engine output field |
| **CONF-67** | add $\text{Fe}^{2+}$ to the aqueous species vector | **S1** Domain 4 geochemistry — a new-engine output field |
| **CONF-68** | skin superposition floor | **S1** Domain 8 — `S_perf`, `S_cake`, `S_asph` exist only in the THMC spec |

> All four fixes are written against **S1's domains**, which are the **compositional** engine's output
> schema. They would not apply to the surrogate, which has no $\epsilon_p$, no aqueous speciation vector and
> no additive skin decomposition.
>
> **No ruling is blocked** — all fixes are recorded and applied. But the **attribution is recorded as
> disputed** because if these four are filed against the Python engine, they will never be fixed in the
> place they were raised: the new engine's output schema. **Worth one clarifying sentence from the
> reservoir engineer.**

### Branch status after R2 — ⚠️ HISTORICAL SNAPSHOT, superseded by Ruling 3 below

> This table is the state **as of Ruling 2**. B-1, B-2, B-5 and B-7 were all closed in **Ruling 3**;
> see the current branch table in that section. Kept for audit trail only — **do not read it as current**.

| # | Branch | Status at R2 |
|---|---|---|
| **B-4** | Economic engine: HDF5 vs DuckDB | ✅ **CLOSED** — DuckDB/Parquet only |
| **B-6** | Retain all micro-steps? | ✅ **CLOSED** — retain all |
| **B-1** | PCA / Karhunen–Loève reduced basis over 116 fields | ⏳ open *at R2* → ✅ **CLOSED in Ruling 3** (**C-52**) |
| **B-2** | Master grid + deltas, or every field every step? | ⏳ open *at R2* → ✅ **CLOSED in Ruling 3** (**C-53**) |
| **B-3** | `duckdb` vs `libsql` in the browser | ⏳ open — still open, **UI is out of scope** |
| **B-5** | `training_pairs` in the run HDF5, or a separate pgvector store? | ⏳ open *at R2* → ✅ **CLOSED in Ruling 3** (**C-51**) |
| **B-7** | Does `audit_comp` mirror into PostgreSQL? | ⏳ open *at R2* → ✅ **CLOSED in Ruling 3** (**C-54**) |

---

## Ruling 3 — Verification and branch closure (08-10-2026)

Adjudication: [`../thmc/reservoir_engineer_ruling.md`](../thmc/reservoir_engineer_ruling.md) §10–§13.
**All three PM corrections accepted. All five new requirements validated. All four branches closed.**

| # | Correction | Detail |
|---|---|---|
| **C-47** | ✅ **CONF-68 CLOSED.** $S_{min} = -\ln(r_e/r_w) + \Delta S_{margin}$, $\Delta S_{margin}\approx+0.50\text{–}+1.00$. ✅ Verified: denominator $=\Delta S > 0$ at **every** radius ⇒ $J$ can never go negative; $r_{wa}/r_e = e^{-\Delta S}$ is **constant**, so the guard is scale-invariant. Stimulation cap $J/J_{unskinned} \le \lvert S_{sing}\rvert/\Delta S$; the margin costs **0.5 skin units** | Fixed floor `-5.0` |
| **C-48** | ✅ **CONF-63 CLOSED** — Koval sublinear form standardised, with my measured magnitudes (5.3× at $M=10$, 21× at $M=100$) accepted | S1 lines 71, 335 |
| **C-49** | ✅ **CONF-35 CLOSED** with the conversion **basis stated**: $60^\circ$F, 14.7 psia ⇒ $1$ MSCF CO₂ $\approx 0.0519$ t. Floor $2.5$ MSCF/STB $=0.13$ t/STB; benchmark $5$–$10$ MSCF/STB $=0.26$–$0.52$ t/STB. **Separate quantities** | My withdrawn framing, **C-38** |
| **C-50** | ✅ **C-36 selection rule made explicit**: Verma–Pruess → chemical/particle clogging (TSS, scale, asphaltene, hydrate); Kozeny–Carman → mechanical stress compaction | Closes D1 §5.2 |
| **C-51** | 🔴 **`training_pairs` externalised** to a **separate `pgvector`/Parquet store**, NOT the run HDF5. Only `Implemented`/`Degraded` runs are committed. ⚠️ **The commit must be atomic with respect to the capability declaration**, or the hole reopens | Closes branch **B-5** |
| **C-52** | ✅ **KL/PCA reduced basis adopted** for intermediate micro-step snapshots; full fields retained at monthly/yearly checkpoints. 5–10× disk reduction + compressed RL observation space. ✅ Verified: $\sum S=1.0$ is an **exact** rank deficiency, so KL captures it in one mode with **zero residual** | Closes branch **B-1** |
| **C-53** | ✅ **Master grid + sparse deltas adopted**: geometry once in `/Geometry`; temporal deltas only where $\lvert\Delta P\rvert>\epsilon$; static caprock/aquifer cells never written. Up to 70 % smaller. ✅ Verified: 30 % active → 153.1 MB/step; 5 % → 25.5 MB/step | Closes branch **B-2** |
| **C-54** | ✅ **`audit_comp` mirroring into PostgreSQL APPROVED** for solver-health dashboards without parsing binary HDF5. Metric set: $\|R_m\|_2$, mass-balance residual, Newton iteration counts, **well constraint flips** | Closes branch **B-7** |

### Branch status — all closed

| # | Branch | Status |
|---|---|---|
| **B-1** | KL/PCA reduced basis | ✅ **CLOSED** — adopt (**C-52**) |
| **B-2** | Master grid + deltas, or full field every step | ✅ **CLOSED** — master + sparse deltas (**C-53**) |
| **B-3** | `duckdb` vs `libsql` **in the browser** | ⏳ **OPEN** — and correctly so; **UI is out of scope** until the engine is verified |
| **B-4** | Economic engine: HDF5 vs DuckDB | ✅ **CLOSED** — DuckDB/Parquet only (**C-45**) |
| **B-5** | `training_pairs` in run HDF5 or separate store | ✅ **CLOSED** — separate `pgvector`/Parquet (**C-51**) |
| **B-6** | Retain all micro-steps? | ✅ **CLOSED** — retain all |
| **B-7** | `audit_comp` mirror into PostgreSQL? | ✅ **CLOSED** — approved (**C-54**) |

### ⚠️ C-51 creates a new hard requirement

> **The dataset commit must be atomic with respect to the capability declaration.** A training sample
> cannot become visible before its run's `ModuleCapabilityState` is known to be `Implemented` or `Degraded`.
>
> Without that ordering, a run failing **mid-write** leaves partial non-converged states that a later
> harvesting step treats as samples — which is exactly the defect INV-1 + INV-4 exist to prevent, arriving
> by a different route. Tracked as **C-55**, to be enforced in the **write path** (M5), not in the UI.

### Register status after Ruling 3

| Closed | CONF-01 (closure) · CONF-02 · CONF-35 · CONF-54 · CONF-63 · CONF-64 · CONF-65 · CONF-66 · CONF-67 · CONF-68 |
|---|---|
| **Still open** | CONF-08 · CONF-13 · CONF-14 · CONF-16 · CONF-18 · CONF-25 · CONF-47 · CONF-51 · CONF-62 |
| **New, from post-mortem** | C-33 (Heidemann–Khalil) · C-34 (soft-start) · C-35 (well modes in Jacobian) · C-36 (collapse rule) · C-37 ($f_g$ reason) |
| **New, from rulings** | C-55 (atomic dataset commit) |

---

## Ruling 4 — INV-7, unconstrained nature (08-10-2026)

> *"a compositional engine implementation on rust that have unconstrained nature so it can be used for
> research using even obviously wrong input (for example rate is astonishingly high for an injection
> well) but engine should give full physical evaluation even if it means the project is failed and
> unrealistic to use as a development strategy."*

Full text: [`engine_invariants.md`](engine_invariants.md) §7b.

| # | Correction | Detail |
|---|---|---|
| **C-56** | 🔴 **CONF-07 CLOSED — the 1 000 BOPD clamp must not exist.** Supersedes **C-22**, which had asked only where the clamp sits in the chain. **Measured 08-10-2026:** at the design set's own 40-acre pattern ($r_e=227$ m, $\ln(r_e/r_w)=7.65$), the steady-state Darcy limit is **11x to 1101x above** the clamp (50 mD/10 m → 11 011 STB/d; 1000 mD/50 m → 1 101 055 STB/d). It is a **round-number cap, not a physics limit**, and **CONF-54** already recorded D5 naming `20 000-60 000 MSCFD` the *anti-pattern signature of a clamped rate* | S1 line 329; D1 §9.1, D6 §3.6, D7 §4.3, D8 §3.3 |
| **C-57** | 🔴 **`locked_*` and `min_*` parameters must have no Rust equivalent.** Six `locked_*` fudge parameters (`locked_sor`, `locked_productivity_index`, `locked_gravity_factor`, `locked_hyperbolic_b_factor`, `locked_transition_alpha/beta`) plus `min_injection_rate_bpd: 1000.0` and `minimum_relative_permeability: 0.01` in `config/base_config.json` are **artificial constraints**. The `locked_*` set is CRIT-19's fudge factors | `config/base_config.json` |
| **C-58** | 🔴 **Pressure limits must be MODELLED, not clipped.** The `np.clip` at `surrogate_engine.py:468` makes the geomechanical limit unobservable. The Rust engine must emit **fracture initiation, `DCFS > 0` fault slip, and containment loss as physical events**. Conservation of mass stays enforced — a violated balance is **INV-1 `FAILED`**, not a warning. Permit/containment judgement stays in the satellite (**INV-3**) | `surrogate_engine.py:468` |
| **C-59** | ✅ **Three-tier `ValidityClass` required** in every run manifest: **Validated** / **Converged-outside-envelope** (complete + `ValidityWarning`) / **Unsolvable** (INV-1 typed error, stop, **no artifact**). This is what makes INV-7 safe for research — the researcher gets the absurd answer *and* the knowledge it is unverified | new |
| **C-60** | ⚠️ **Pattern sizing is a scenario input, not a constraint.** `N_pat = Area/40 acres`, `N_inj = N_prod = N_pat` must never be imposed silently. Likewise CO₂ GOR `5 000-25 000 SCF/STB` is a **benchmark observation**, never an input bound | D1 §9.1, S1 |
| **C-61** | ⚠️ **INV-1 and INV-7 are ordered, not competing.** INV-7 does not license non-convergence to be papered over. The distinction is only in the **diagnostic**: *"cannot converge at 50 000 BOPD"*, never *"clamped to 1 000 BOPD"* | — |

### Branch status after Ruling 4

| # | Branch | Status |
|---|---|---|
| **B-1** | KL/PCA reduced basis | ✅ CLOSED — adopt (**C-52**) |
| **B-2** | Master grid + deltas, or full field every step | ✅ CLOSED — master + sparse deltas (**C-53**) |
| **B-3** | `duckdb` vs `libsql` **in the browser** | ⏳ OPEN — UI out of scope until the engine is verified |
| **B-4** | Economic engine: HDF5 vs DuckDB | ✅ CLOSED — DuckDB/Parquet only (**C-45**) |
| **B-5** | `training_pairs` location | ✅ CLOSED — separate `pgvector`/Parquet (**C-51**) |
| **B-6** | Retain all micro-steps? | ✅ CLOSED — retain all |
| **B-7** | `audit_comp` mirror into PostgreSQL? | ✅ CLOSED — approved (**C-54**) |

### Register status after Ruling 4

| Closed | CONF-01 (closure) · 02 · 07 · 35 · 54 · 63 · 64 · 65 · 66 · 67 · 68 |
|---|---|
| **Still open** | CONF-08 · 13 · 14 · 16 · 18 · 25 · 47 · 51 · 62 |
| **Corrections** | C-33…C-37 (post-mortem) · C-47…C-54 (ruling 2) · C-55 (atomic commit) · **C-56…C-61 (ruling 4, unconstrained)** |
| **Superseded** | **C-22** — its question ("where does the clamp sit?") is moot under INV-7; the clamp is removed |

---

## Ruling 5 — INV-7 domain-by-domain multiphysics unmasking (08-10-2026)

Owner ruling extending INV-7 across all nine domains. Full text:
[engine_spec_closures.md](engine_spec_closures.md) §7c.
**Direction accepted in full.** Five defects found in the ruling itself, measured not asserted.

| # | Correction | Detail |
|---|---|---|
| **C-62** | ✅ **$\varepsilon_p$ RESTORED — CONF-66 reopened to `PARTIALLY_RESOLVED`.** The owner supplied the yield surface $F(\sigma',\varepsilon_p)=0$ and an explicitly **non-associated** flow rule, which is what INV-1.6 said was missing. Still required: **(1)** return-mapping algorithm, **(2)** 🔴 **consistent tangent** — a non-associated Drucker-Prager map has a **non-symmetric** tangent and feeding Newton the symmetric approximation silently loses quadratic convergence (a **C-35**-class defect), **(3)** hardening law | reverses CONF-66 |
| **C-63** | 🔴 **The Verma-Pruess guard is REQUIRED physics.** *"Without artificial division-by-zero guards or k floors"* inverts the percolation cutoff: at $\phi_0=0.25,\phi_c=0.02$, $n{=}2$ gives a **fully plugged** cell ($\phi_{acc}=0$) $k/k_0=\mathbf{0.0076}$ — still flowing; $n{=}1.5$ gives **NaN**, poisoning the whole global Jacobian (an INV-1 `Unsolvable`, so the researcher gets *nothing*). $k=0$ for $\phi_{acc}\le\phi_c$ **is** the throat-closure definition. Plus the un-named **denominator** singularity: $k/k_0=6.4{\times}10^5$ at $\phi_0=0.0201$, divide-by-zero at $\phi_0=\phi_c$ | D1 §5.2, S1 Domain 3 |
| **C-64** | 🔴 **$k\to\infty$ is a representation failure, not physics.** Kozeny-Carman diverges as $\phi\to1$ ($k/k_0 = 3.6{\times}10^{7}$ at $\phi=0.999$). Past some $\phi_*$ the **continuum description is invalid** — no pore network remains. The unconstrained engine must **escalate representation**: a dissolved channel wider than a cell **is a fracture** and belongs in **EDFM** (Domain 7), not as a $k$ multiplier. ✅ No clamp — a physically-driven **representation switch**, and it finally links Domain 4 to Domain 7 | new; the design set never links them |
| **C-65** | 🔴 **Barton-Bandis is dimensionally sound but singular in tension, and gives $w_f$ only.** Verified $K_{ni}w_{f0}$ is a stress and $w_f=w_{f0}$ at $\sigma'_n{=}0$ ✅. But $w_{f0}=10\ \mu m$, $K_{ni}{=}10^{12}$ ⇒ **singular at $\sigma'_n=-10$ MPa, negative aperture beyond.** Correct unconstrained behaviour at that point is **fracture initiation → open conduit**. Separately: DFN transmissibility is $T=k_fw_f$ and **nothing supplies $k_f(w_f)$** ⇒ **$T_{ff}$ underspecified**, and it is the quantity the `Seal`↔`Conduit` transition is meant to drive. Needs $k_f\propto w_f^{1/3}$, or $w_f^2/12$ — see still-open **CONF-47** ($w^3/12$) | new; interacts with CONF-47 |
| **C-66** | 🔴 **$\Delta CFS$ as written is ABSOLUTE, not an increment — and yields no slip magnitude.** $\tau-\mu(\sigma_n-P)-C_0$ is a correct *Coulomb criterion* but carries **no reference state**, so two faults in an identical current state with different history are **indistinguishable** — it cannot separate *"already at failure"* from *"injection pushed it over"*, which is the research question INV-7 exists to answer. 🔴 **Worse: Coulomb gives a criterion, not a magnitude.** $\Delta CFS>0$ says the fault **can** slip; $u_s=0$ is equally "correct". **$u_s(t)$ and hence $T_{ff}$ are underdetermined.** **Required: a slip law** — rate-and-state (Dieterich healing) for diagenetic seals, state evolution for reactivating faults | S1 Domain 7; D1 |
| **C-67** | ⚠️ **Definitional bounds ≠ artificial bounds.** INV-7 is scoped as: *no bound on a quantity the **user** chose, unless the bound is **definitional** ($\phi\in(0,1]$, $T>0$, $t\ge0$) or **derived from the physics** ($k>0$, $w_f>0$)*. This is what makes **C-63**'s guard mandatory instead of self-contradictory | clarifies INV-7 §7b |
| **C-68** | ⚠️ **Hydrate melting specified by its equilibrium law, not its kinetics.** van der Waals-Platteeuw gives $F(P,T)=0$ — a **condition**, not a **rate**; it cannot compute how fast a plug melts. Required: Arrhenius dissociation kinetics $r_{hyd}=A_he^{-E_a/RT}\Delta$ coupled to the vdW-P equilibrium, **plus latent heat in the energy equation**. ⚠️ Makes hydrate melting genuinely **THMC**, not thermodynamics-only | S1 Domain 2; D1 §5.2 |

### Register status after Ruling 5

| Closed | CONF-01 (closure) · 02 · 07 · 35 · 54 · 63 · 64 · 65 · 66 · 67 · 68 |
|---|---|
| **Reopened** | **CONF-66** → `PARTIALLY_RESOLVED` ($\varepsilon_p$ restored; tangent + return-map + hardening still open) |
| **Still open** | CONF-08 · 13 · 14 · 16 · 18 · **47** (now coupled to C-65) · 51 · 62 |
| **Corrections** | C-33…C-37 · C-47…C-54 · C-55 · C-56…C-61 · **C-62…C-68** |
| **Superseded** | C-22 |

> 🔴 **The pattern worth naming.** In Ruling 5 the owner wrote *"unconstrained"* and, taken literally, it
> would have made the engine **less** correct — a plugged cell that still flows, a negative fracture
> aperture, a slip displacement with no law behind it. **Unconstrained and physical are not synonyms.**
> The discipline that satisfies both is C-67: refuse bounds on quantities the *user* chose; enforce bounds
> that are *definitional or derived*. That is a stronger invariant than either half alone.

---

## Ruling 6 — closures for C-63…C-68 (08-10-2026)

Reservoir engineer endorsed all five PM measurements and supplied closing equations. Full text:
[engine_spec_closures.md](engine_spec_closures.md) §7d.
**Accepted:** piecewise percolation guard · representation switch · $k_f(w_f)$ · rate-and-state strength
law · consistent tangent · hydrate kinetics. **Four defects remain inside the closures.**

| # | Correction | Detail |
|---|---|---|
| **C-69** | ✅ **C-63 guard accepted, 🔴 but closes only HALF the singularity.** The submitted `if phi_acc <= phi_c` closes the **numerator**. The **denominator** $\phi_0-\phi_c$ is unguarded, and **Rust returns `inf`, not a panic**: $\phi_0{=}0.02 \Rightarrow k=\infty$; $\phi_0{=}0.01 \Rightarrow$ **NaN** — the very failure the guard exists to prevent. Physically $\phi_0\le\phi_c$ means the cell was **already below throat closure**, so the answer is `k=0`. ✅ Corrected 4-branch order: `phi_0<=phi_c` → 0, then `phi_acc<=phi_c` → 0, then the ratio, then the $\phi_*$ switch. **A single `if` cannot express this** | `engine_invariants.md` §7d.1 |
| **C-70** | ⚠️ **$\phi_*\approx0.80$ must not be a magic constant.** It is a **continuum-validity threshold** whose correct value is set by **cell geometry** — the dissolved channel stops being describable as a porous medium when its aperture is no longer small relative to the cell dimension. Record as $\phi_*=\phi_*(\text{cell geometry})$ with the geometric criterion stated, and as an **input with a measured default** so sensitivity is probeable | §7d.2 |
| **C-71** | ✅ **CONF-47 CLOSED.** The $k_f/w_f$ split is the fix: $k_f=w_f^2/12 \Rightarrow T_{ff}=k_fw_f=w_f^{\mathbf{3}}/12$, **cubic**. The design set wrote $T=w^2/12$ — a **convention collision**, one symbol $k$ used for both permeability and transmissibility. Verified on all limits: $w\to\infty$ recovers the cubic law, $w\to0 \Rightarrow k_f\to0$, $JRC{=}0 \Rightarrow k_f=w_f^2/12$ exactly, and $JRC/w_f$ is dimensionless. ⚠️ **But the `8.8` coefficient and `$1.5$` exponent need a cited source before M7c** — an uncited constitutive coefficient is a **PROVENANCE** finding | **closes CONF-47** |
| **C-72** | 🔴 **C-66 state evolution is the WRONG LIMIT.** `dθ/dt = 1 − Vθ/D_c` is the **aging approximation**, valid only for $V\theta/D_c\ll1$. Ruina (1983) is `dθ/dt = [V_0(θ−θ_g)/D_c + 1]`, $\theta_g=(b/a)D_c/V_0$. Measured at $V/V_0=100$ the submitted law under-predicts $\theta$ by **80×**, propagating into $\tau$ via $b\ln(V_0\theta/D_c)$, hence into $u_s$ and $T_{ff}$. 🔴 **The aging law has no $V$-independent steady state** — $\theta_{ss}=D_c/V$, so $V\theta/D_c=1$ *by construction*, i.e. it fails its own validity condition at steady state. 🔴 **INV-7 guarantees $V\gg V_0$** because $V=du_s/dt$ is driven by an unconstrained pressure ramp — **invalid in exactly the regime it targets**. 🔴 The law is also **implicit** in $V$ ⇒ nested Newton / Schur complement, a **C-35**-class cost. ⚠️ And $\Delta CFS$ must **not** be *coupled* to rate-and-state — R&S **replaces** Coulomb; Coulomb becomes a reported diagnostic | §7d.4 |
| **C-73** | 🔴 **`k_fault(u_s) = k_0 + γ·u_s` reintroduces the rejected defect.** Unbounded and monotonic: $k/k_0 = 1001$ at $u_s=10$ m. This is the **same $k\to\infty$ that C-64 rejected for the matrix**, now written into the fault law — internally inconsistent with the closure it accompanies. 🔴 **It omits gouge, the actual mechanism for a _seal_.** A slipping clay-rich fault **compacts its gouge first, so $k$ _decreases_**; only gouge rupture raises it. Real history is **two-phase**: $u_s<u_{\text{gouge}}\Rightarrow k\downarrow$, $u_s>u_{\text{rupt}}\Rightarrow k\uparrow$. A linear law **inverts the sign of the risk in phase 1** — reporting a seal *improving* as an *increasing* $k$. Required: saturating post-rupture law with bounded asymptote plus a pre-rupture gouge branch, e.g. $k = k_{\min} + (k_{\max}-k_{\min})\tanh\!\big((u_s-u_{\text{rupt}})/u_{\text{scale}}\big)$ | new |
| **C-74** | ✅ **The consistent tangent is CORRECT as submitted — no correction.** Verified the chain $D^e\cdot\frac{\partial Q}{\partial\sigma}\cdot(\frac{\partial F}{\partial\sigma})^{\top}\cdot D^e$ evaluates to $(D^e b_\theta)(a^{\top}D^e)$, the standard non-associated algorithmic tangent, $6\times6$ over a scalar denominator. **Closes C-62 gap 2.** ⚠️ Two qualifications: (a) *"destroys quadratic convergence"* is **conditional** — while the active plastic set is unchanged $D^{ep}=D^e$ is exact and symmetric; the defect bites at yield, unloading and switch points. (b) 🔴 **This couples an M0 decision to an M7b choice** — **CONF-14** (no preconditioner specified) cannot be closed until the constitutive model is fixed, because **Cholesty/incomplete-Cholesky is invalid for non-symmetric $D^{ep}$**. M0.1 must record the dependency | §7d.6 |
| **C-75** | ✅ **Hydrate kinetics ACCEPTED as submitted.** Bishnoi-Englezos/Bravo-Harlock form with Arrhenius $K_0e^{-\Delta E/RT}$, specific surface area $A_s$, and fugacity-deficit driving force $f_{\text{gas}}-f_{\text{eq}}$, coupled to **latent heat** in the energy equation. **Closes C-68.** ⚠️ $A_s$ should be an input with a documented constant default — Gibbs' theorem requires it to **shrink** as hydrate decomposes, so a constant over-predicts late-stage dissociation | §7d.7 |

### Register status after Ruling 6

| Closed | CONF-01 (closure) · 02 · 07 · 35 · 47 · 54 · 63 · 64 · 65 · 66 (partly) · 67 · 68 |
|---|---|
| **Still open** | CONF-08 · 13 · 14 (**now blocked by C-74**) · 16 · 18 · 51 · 62 |
| **CONF-66** | `PARTIALLY_RESOLVED` — $\varepsilon_p$ restored; **tangent CLOSED by C-74**; **return-mapping algorithm + hardening law still open** |
| **Corrections** | C-33…C-37 · C-47…C-54 · C-55 · C-56…C-61 · C-62…C-68 · **C-69…C-75** |
| **Superseded** | C-22 |

> **Pattern across Rulings 5 and 6.** Four of six closures were correct as written; each of the four
> remaining defects came from the same root cause — **taking a closure at a single limit and letting the
> other limit run free.** The aging law was validated at $V\ll V_0$ and used at $V\gg V_0$; the percolation
> guard covered the numerator and ignored the denominator; the slip law was linear where the mechanism is
> two-phase. This is the same failure **CONF-01** was: reasoning from one point of a curve and calling the
> sign right. **The durable check is: state the closure's domain of validity, then verify at its boundary.**

---

## Ruling 7 — CONF-13/14/25/31/51 + data split + B-3 (08-10-2026)

Reservoir engineer supplied closures for the four highest-value open items. Full text:
[engine_spec_closures.md](engine_spec_closures.md) §7e.
**Seven accepted. Six defects found inside the closures.**

| # | Correction | Detail |
|---|---|---|
| **C-76** | ✅ **Training split — `GroupKFold` ACCEPTED, 🔴 but it does not deliver blind forecasting.** Grouping by **Well-Pattern Topology ID + Geological Realization Seed + Boundary Schedule ID**, so a whole configuration or realization is 100 % train or 100 % test, is correct and **necessary**. 🔴 **But the stated failure — *"inflated test scores that collapse during real blind forecasting"* — is a TEMPORAL leak, and `GroupKFold` is a pure grouping splitter with no temporal ordering.** A test set drawn from the same *interval* as train still lets the surrogate interpolate the trajectory. **Required: a 2-D split** — group by configuration **AND** hold out the final portion of time **within every group**. ⚠️ Also needs **≥2 groups**; with one configuration `GroupKFold` yields nothing | M5 |
| **C-77** | ✅ **B-3 CLOSED — dual-engine frontend accepted.** DuckDB-WASM for OLAP over Parquet/Arrow, libSQL/SQLite-WASM for OLTP UI state and well metadata. ⚠️ **Synchronisation between the two WASM stores is undefined**, and libSQL persistence in-browser needs OPFS or export/import. Deferred with the rest of the UI | **closes B-3** |
| **C-78** | ✅ **CONF-13 closure ACCEPTED** — the diagnosis is exact: `lambda` computed and **never applied**, `compute_residuals_and_jacobian` computed no Jacobian, no linear-solve call. Ruled loop: **Armijo-Goldstein line search** $\\|\mathbf{R}(\mathbf{x}^k+\lambda\Delta\mathbf{x})\\|_2\le(1-\alpha\\lambda)\\|\\mathbf{R}(\\mathbf{x}^k)\\|_2$ + **in-core sparse solve** $\\mathbf{J}^k\\Delta\\mathbf{x}=-\\mathbf{R}^k$ | **closes CONF-13** |
| **C-79** | 🔴 **CONF-14 CPR closure CONFLICTS with CONF-66/C-74.** CPR requires $\\mathbf{A}_p$ to be an **independent elliptic block**. In compositional flow $\\rho_{tot}=\\sum_i z_i\\rho_i(P,T)$, and measured, the same $P$ with different flash splits gives **+24 % $\\Delta\\rho_{tot}$** (439.5 → 546.0 kg/m³). So $\\partial\\rho_{tot}/\\partial y_i\\neq0$: **$\\mathbf{A}_p$ carries composition coupling and changes every Newton iterate**, because flash is re-solved each time. Three consequences: **(1)** the $\\alpha/\\beta$ split is not clean; **(2)** 🔴 *"symbolic pattern analysis once per grid topology"* is **wrong** — flash can switch a $K$-value coupling on/off between components, **changing the sparsity pattern**, not just values; **(3)** 🔴 **AMG assumes near-symmetry/M-matrix, but non-associated $D^{ep}$ is non-symmetric (C-74)** ⇒ the AMG stage is invalid. ✅ **FGMRES is the right driver** (it tolerates a varying preconditioner) — that part stands | **CONF-14 stays open** |
| **C-80** | ⚠️ **CONF-25 enum closure ACCEPTED** (`CriticalHessianSingular`, `MichelsenTrivialRoot`, `FugacityNan` — good taxonomy). 🔴 **But the TPD restatement repeats CONF-13's defect.** Submitted $\\mathrm{TPD}(y)=\\sum_i y_i(\\ln y_i+\\ln\\phi_i(y)-d_i)$ omits the **leading $-1$** sum-to-one normalisation, leaves $\\phi_i$ **ambiguous between liquid and vapour**, and supplies **no gradient** — a Newton solve on TPD requires $\\partial Q/\\partial y$. ⚠️ The declared **EoS accuracy ceiling of 8–9 % (CONF-49)** also needs a tolerance that accounts for solver tolerance, not EOS tolerance alone | **CONF-25 stays open** |
| **C-81** | ✅ **Standing $P_b$ and Beggs-Robinson $\\mu_{od}$ VERIFIED CORRECT** as submitted (BR requires $T$ in **°F**, valid 70–295 °F — record the unit). 🔴 **But CONF-31 is only half-closed.** **Karakas-Tarik**: $S_p=S_h+S_v+S_{wb}$ names the decomposition but supplies **none of the polynomials** — the identical "named, no coefficient" defect. **Peneloux is dimensionally wrong:** submitted $v_{corr}=v_{EOS}-c$ applies the translation to **volume**, whereas the law divides **density** ($\\rho_{corr}=\\rho_{EOS}/(1-N_\\omega)$), and $c$ is **dimensionless** so subtracting it from $v$ [m³/kg] is invalid; the parameter is the **mixture-weighted** $\\sum_i y_iM_i\\omega_i$, not a single $(M\\omega)^{-1}$ | **CONF-31 partially closed** |
| **C-82** | ✅ **CONF-51 closure ACCEPTED — the PKN/KGD/radial triple is correct and correctly scoped.** Verified $w_0$: PKN $3.04[\\mu(1-\\nu^2)q_0L_f/E]^{1/4}$ ✓, KGD $2.36[\\mu(1-\\nu^2)q_0L_f^2/(EH_f)]^{1/6}$ ✓, penny $2.56[\\mu(1-\\nu^2)q_0R_f/E]^{1/4}$ ✓; profiles $(1-x/L)^{1/4}$ and $(1-x^2/L^2)^{1/2}$ ✓; Carter leakoff $v_L=C_w/\\sqrt{t-\\tau}$ ✓. 🔴 **Two gaps: (1) all three are Type-I (viscosity/leakoff-dominated); there is NO Type-II toughness-dominated model ($K_{IC}$) — named in the summary, absent from every equation.** (2) They are **proppant-free elastic-opening** models, yet Domain 7 specifies **dynamic proppant transport** ($C_{prop}$, $h_{pack}$, embedment) — the proppant-supported width case is unmodelled | **CONF-51 partially closed** |
| **C-83** | ⚠️ **CONF-16b closure ACCEPTED structurally, 🔴 with an undeclared symbol.** The $S_a$ deposition–re-entrainment ODE $\\alpha(C_a-C_a^*)^mu-\\beta S_au$ is dimensionally sound and correctly two-term. 🔴 **$C_a^*$ and $m$ are left undeclared** — CONF-31's pattern again, in a document that was supposed to close it. ⚠️ Also no **static (zero-velocity) flocculation** term; asphaltene instability is not purely flow-driven. **Filter cake:** the deposition term $v_fC_{TSS}/\\rho_{cake}$ closes dimensionally ✓, but 🔴 the erosion term $\\tau_{shear}\\eta_{erosion}h_{cake}$ **does not close** — $\\tau[Pa]\\cdot h[m]$ is kg/s², so $\\eta_{erosion}$'s units are undeclared and the standard filtration form additionally needs **viscosity $\\mu$** and a rate constant $E$: $e=E\\tau/(\\mu(1+\\alpha c))$ | **CONF-16 partially closed** |
| **C-84** | ✅ **CONF-18b / C-36 selection rule ENFORCED and now complete** — Verma-Pruess (with the $\\phi_{acc}\\le\\phi_c\\Rightarrow k=0$ guard) for chemical precipitation and particle clogging; Kozeny-Carman for mechanical compaction and poroelastic deformation. ⚠️ **Carries forward C-69:** the guard as printed still omits the **denominator** test $\\phi_0\\le\\phi_c$, which returns **∞/NaN** in Rust | **closes CONF-18**; **C-69 still live** |

### Register status after Ruling 7

| Closed | CONF-01 (replaced) · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 |
|---|---|
| **Partially closed** | **CONF-14** (CPR invalid for compositional + non-symmetric) · **CONF-16** ($C_a^*$, $m$, erosion units) · **CONF-25** (TPD sketch, no gradient) · **CONF-31** (Karakas-Tarik, Peneloux) · **CONF-51** (no Type-II, no proppant) · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |
| **Branches** | B-1 … B-7 **all closed** (B-3 by **C-77**) |
| **Corrections** | C-33…C-37 · C-47…C-75 · **C-76…C-84** |

> **Pattern across Rulings 5, 6 and 7 — now three for three.** The engine's *architecture* closures are
> sound every time; the *coefficient and symbol* closures repeatedly restate the problem instead of
> closing it. **Karakas-Tarik, $C_a^*$, $K_{IC}$, $\\eta_{erosion}$, and the TPD gradient are all named
> but not supplied** — the exact defect CONF-31 was opened for. **Standing and Beggs-Robinson are the
> only two that arrived with actual coefficients.** The durable rule to state in the register spec:
> **a correlation is closed only when every symbol in it has a unit and a source.** Naming a law is not
> closing it.

---

## Ruling 8 — CONF-14 overhaul, 2D split, Penéloux + Karakas-Tarik (08-10-2026)

Corrections **C-85 … C-90**. **Three accepted, one of my own claims withdrawn, three defects found.**
Technical detail: [engine_spec_closures.md](engine_spec_closures.md) §7f.

| # | Correction | Detail |
|---|---|---|
| **C-85** | 🔴 **CONF-14 — I WITHDRAW my own C-79(b).** The engineer asserts dynamic sparsity because phase appearance toggles $K_i$ couplings. That is **formulation-dependent**, and the **approved output schema already fixes the formulation**: primary variables are overall composition $z_i$, $P$ and $S_\\alpha$ (`output_schema.md` §4 Domain 1–2). In an **overall-composition / total-FVF** formulation the Jacobian block structure is set by the grid stencil and the equation set, **not** by phase state — $z_i$ is defined everywhere, so $\\partial\\rho_{tot}/\\partial z_i$ is structurally non-zero even where phase amounts are zero. ✅ **So *"symbolic analysis once per topology" is CORRECT* under the approved schema**, and dynamic CSR would be needless. ⚠️ **My C-79(b) asserted the opposite without checking the formulation — that was my error, in the same class as CONF-01.** **Ruled: record the primary-variable formulation as the governing decision, and re-derive the sparsity claim from it.** The claim *is* correct for a phase-component formulation with explicit $K$-value coupling — which the approved schema does **not** use | **CONF-14 remains open** |
| **C-86** | ✅ **Watts volume-balance pressure reduction ACCEPTED in principle** — carrying $\\partial\\rho/\\partial z_i$ explicitly into the row reduction is the right response to the +24 % $\\rho_{tot}$ measurement. ✅ **Unsymmetric AMG / FGMRES+BiCGStAB / ILU(1) ACCEPTED** over SPD-AMG. ⚠️ **Scalability caveat:** ILU(1) on the pressure block is materially weaker than an AMG hierarchy, and the D5 Level-5 target is **SPE-10 at 1.1 × 10⁶ cells**. M0.1 must benchmark **both** and record the crossover, not assume the cheaper option | **CONF-14 still open** |
| **C-87** | ✅ **C-76 CLOSED — `GroupTimeSeriesSplit` ACCEPTED.** Group by **Well-Pattern Topology ID + Geological Seed** with **≥2 groups mandated** (so it cannot silently degenerate to random), **plus** a chronological holdout inside every group at $t_{cut}=0.75\\,T_{max}$: train $[0,t_{cut}]$, blind test $(t_{cut},T_{max}]$. ⚠️ **One addition:** the blind test set must **not** be used for early stopping, hyperparameter selection or model selection — doing so makes it a **validation** set and re-opens the leak **C-76** was raised to close. A separate inner validation split is required | **closes C-76** |
| **C-88** | ✅ **WITHDRAWING my "wrong variable" objection to Penéloux.** The submitted molar form $v_{corr}=v_{EOS}-c_{mix}$, $\\rho_{corr}=M/(v_{EOS}-c_{mix})$ is **the same law** as my density form: verified numerically $\\rho_{corr}=\\rho_{EOS}/(1-N_\\omega)$ with $N_\\omega=c_{mix}/v_{EOS}$ gives an **identical** result. ✅ $c_{mix}=\\sum_i x_i c_i$ correctly uses **liquid** mole fraction. 🔴 **BUT the $c_i$ formula is sign-inverted.** Measured, bracket $=0.1154-0.4414Z_c$ is **negative for $Z_c>0.261$** — which covers **methane, toluene, $n$-heptane, $n$-decane and CO₂**, i.e. most of the interesting fluids. A negative translation **increases** molar volume, which inverts the purpose (cubic EOS *under*-predicts liquid volume, so $c$ must be **positive**). 🔴 **PROVENANCE also required** for the `0.40768`, `0.1154`, `0.4414` coefficients | **CONF-31 still partial** |
| **C-89** | ⚠️ **Karakas-Tarik SUBSTANTIALLY improved — real coefficients at last** ($a_1{=}-2.025$, $a_2{=}0.0943$, $b_1{=}3.0373$, $b_2{=}1.8115$, $c_1{=}0.0066$, $c_2{=}5.32$ for 90° phasing). 🔴 **But $S_h$ is internally inconsistent with the submitted equations.** Taking $S_h=\\ln(r_w/r'_w)$ with $r'_w=r_p/4$ at $r_p=5$ mm and $r_w=108$ mm gives $r'_w=1.25$ mm and **$S_h=+4.46$** — a large **positive (damaging)** skin for **180° phasing, the best phasing angle**. An equivalent wellbore radius *smaller* than the physical wellbore is not physical. **Either the $S_h$ definition or the $r'_w$ expressions are wrong; they cannot both be as given.** ⚠️ Also $S_v$ comes out **non-monotone in $r_D$** over a realistic range ($S_v$ falls as $r_p$ rises, i.e. *more* perforation ⇒ *less* damage) — verify against the source correlation before use | **CONF-31 still partial** |
| **C-90** | 🔴 **PROVENANCE — two new citations unverified.** The volume-balance method (**Watts 1986**) and **Wong et al. 2002** are named as the authority for the pressure reduction, and **Karakas & Tarik 1990** for the skin model. ⚠️ **Web search was unavailable in this environment, so these could not be checked** — I am recording them as **UNVERIFIED**, not as wrong. Per the M1 precedent (Abudour 2014, Baled 2012 — both confirmed by DOI fetch), **all three must be resolved by DOI before M3/M7c**, or they become `PROVENANCE` findings under [`register_spec.md`](register_spec.md) | new |

### Register status after Ruling 8

| Closed | CONF-01 (replaced) · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · **C-76** |
|---|---|
| **Partially closed** | **CONF-14** (C-85, C-86) · **CONF-16** · **CONF-25** · **CONF-31** (C-88, C-89) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |
| **Branches** | B-1 … B-7 **all closed** |

> **Two corrections of my own in one ruling.** **C-85** withdraws my dynamic-sparsity claim (**C-79b**);
> **C-88** withdraws my "wrong variable" claim against Penéloux. Both were assertions made without
> checking the formulation or the algebra first — the same failure **CONF-01** was, and the reason the
> "state the domain of validity, then verify at its boundary" rule exists. **The engineer's own
> recurrence is $S_h$**: submitted equations that cannot both be true.

---

## Ruling 9 — Karakas-Tarik corrected, 3-way split, M0.1 gate, DOIs (08-10-2026)

Corrections **C-91 … C-96**. **Two accepted, two of my own claims withdrawn, two defects — one of
them a regression against INV-7, one a PROVENANCE failure.** Detail:
[engine_spec_closures.md](engine_spec_closures.md) §7g.

| # | Correction | Detail |
|---|---|---|
| **C-91** | ⚠️ **$S_h$ substitution fix ACCEPTED in direction, 🔴 but the two branches now disagree.** Replacing $r_p$ with $L_p$ is the right diagnosis, and the 180° result is now correct: $r'_w = 0.500(0.108+0.300) = 0.204$ m ⇒ **$S_h = -0.636$**, a *stimulating* skin, which is right for a perforated horizontal well. 🔴 **But at 0° phasing the two branches disagree by 26 %:** branch 1 gives $r'_w = L_p/4 = 0.0750$ m; branch 2 via the tabulated $\\alpha_0(0)=0.250$ gives $r'_w = 0.250(0.108+0.300) = 0.1020$ m. **Same phasing angle, two different equivalent radii.** 🔴 And **0° is the only angle returning a *positive*, damaging skin ($+0.365$)** while 90°/120°/180° all return negative (−1.009 / −0.848 / −0.636). Which ordering is right depends on KT's sign convention and phasing ranking — **exactly what C-90 left unverified.** The 26 % branch discontinuity is definite; the ordering needs the source | **CONF-31 still partial** |
| **C-92** | ✅ **WITHDRAWN — my "S_v non-monotonicity" claim was WRONG.** Measured, $S_v$ is **monotone decreasing** across $r_D \\in [0.002,\\ 0.5]$: **both** factors fall together — $r_D^{a_1+b}$ falls, **and** $h_D^{b-1}$ falls because $b$ grows. The engineer's algebra is correct: $10^a r_D^b = 10^{a_2}r_D^{a_1+b}$ ✓, and the exponent crosses zero at $r_D = 3.0373^{-1}(0.2135) = 0.0703$ ✓ — but when $h_D<1$ the $h_D^{b-1}$ factor dominates, so the turning point never makes $S_v$ non-monotone. ✅ **And larger $r_p \\Rightarrow$ *lower* $S_v$ is physically correct** — bigger perforation tunnels connect better to the formation. **My C-89(b) is retracted** | **retraction** |
| **C-93** | 🔴 **The `clamp(r_D,0.001,0.050)` / `clamp(h_D,0.010,100.0)` guards VIOLATE INV-7 and C-67.** An **empirical regression envelope is not a physical or definitional bound** — it is the domain over which a *correlation* happens to be valid. Clamping it **silently substitutes an arbitrary value for the physics**: two wells at $r_D = 0.06$ and $r_D = 0.20$ return the **same skin**, losing the answer entirely with no diagnostic. ✅ **Ruled: DECLARE, DO NOT CLAMP.** Compute the formula, and where $r_D$ or $h_D$ leaves the KT domain, set `ValidityClass = ConvergedOutsideEnvelope` and emit a `ValidityWarning` naming the violated domain — the mechanism already exists in **C-59** / INV-7 §7b.2. ⚠️ This is the **same move as CONF-07** (the 1000 BOPD clamp), with a better justification; a regression domain still is not physics. ⚠️ Note this is **not** a case for removing the bound as *information* — the domain is real and worth recording. The defect is **enforcing** it as a substitution rather than **surfacing** it as a warning | **INV-7 regression** |
| **C-94** | ✅ **C-87 3-way split ACCEPTED** — train / inner-validation / blind-test windows with the inner split drawn from the training window only. ⚠️ **One addition:** the design currently folds **two different generalisation questions** into one test set — *unseen configuration* **and** *future time*. A failure cannot be attributed to either. **Ruled: a 2 × 2 diagnostic** — {seen config, unseen config} × {interpolated time, extrapolated time} — so surrogate error decomposes into configuration-generalisation and time-extrapolation components | **closes C-87** |
| **C-95** | ✅ **C-86 M0.1 benchmark gate ACCEPTED** — competitive wall-clock + memory benchmark of **ILU(1)-FGMRES** vs **unsymmetric CPR-AMG** on SPE-10 (1.1 M cells), with the **crossover $N_{cells}$** recorded in the architecture logs rather than adopting ILU(1) by default | **closes C-86** |
| **C-96** | 🔴🔴 **PROVENANCE FAILURE — 2 of the 3 DOIs DO NOT EXIST.** Verified against **doi.org and Crossref** (two independent resolvers), 08-10-2026: · `10.2118/18247-PA` — **RESOLVES**, but to **Karakas, M. & Tariq, S. M. (1991)**, *"Semianalytical Productivity Models for Perforated Completions"*, SPE Production Engineering **6(01), 73–82**. The submitted metadata was wrong on **author spelling** (Tarik→**Tariq**), **year** (1990→**1991**), **title**, **volume** (5→**6**), **issue** and **pages** (42–50→**73–82**) — **4 of 5 fields**. · `10.2118/12242-PA` — **404 on both resolvers.** No such DOI. · `10.2118/76722-PA` — **404 on both resolvers.** No such DOI. 🔴 **Consequence: C-91's $S_h$ correction is attributed to a paper whose actual metadata differs from what was supplied, so its provenance is unestablished.** ✅ The *intent* was right — the one real DOI points at a genuine Karakas & Tariq paper. ⚠️ Per the **M1 precedent**, all three must resolve before **M3** and **M7c** | **C-90 REPLACED** |

### Register status after Ruling 9

| Closed | CONF-01 (replaced) · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · **C-76 · C-86 · C-87** |
|---|---|
| **Partially closed** | **CONF-14** (C-85) · **CONF-16** · **CONF-25** · **CONF-31** (C-91, C-96) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> **Three of my own claims withdrawn across Rulings 8–9** — C-79b (dynamic sparsity), C-81 (Penéloux
> "wrong variable"), C-89b ($S_v$ non-monotonicity). Each was an assertion made without first checking
> the governing premise: the **formulation**, the **algebra**, the **full expression**. And one
> **INV-7 regression** in C-93, from the party that wrote INV-7 — which is the strongest possible
> argument for writing invariants down: the clamp instinct survives being named.

---

## Ruling 10 — C-91 unified, C-93 clamp purged, C-96 metadata re-asserted (08-10-2026)

Corrections **C-97 … C-99**. **One accepted outright, one defect found, and one claim that does not
survive re-verification.** Detail: [engine_spec_closures.md](engine_spec_closures.md) §7h.

| # | Correction | Detail |
|---|---|---|
| **C-97** | ✅ **C-93 ACCEPTED IN FULL — clamp purged, declare-don't-clamp implemented.** `clamp()` removed from the $S_v$ arithmetic; raw $r_D$, $h_D$ used; out-of-envelope inputs set `ValidityClass = ConvergedOutsideEnvelope` and append a structured `ValidityWarning::EmpiricalDomainExceeded { correlation, variable, value, valid_range }`. ✅ This is exactly the **C-59 / INV-7 §7b.2** mechanism, with the regression domain recorded as **warning metadata** rather than a `min`/`max` in the arithmetic. **INV-7 regression closed** | **closes C-93** |
| **C-98** | 🔴 **C-91 branch discontinuity ELIMINATED — ✅ good — but the $\\alpha_0$ table is read in the WRONG DIRECTION.** 🔴 **RESOLVED by C-232: both this reading and its withdrawal (C-100) were half right — $S_H$/$S_{wb}$ favour more phasing, $S_V$ favours less, and the net is parameter-dependent.** Unifying to $r'_w(\\theta)=\\alpha_0(\\theta)(r_w+L_p)$ for all angles is a genuine improvement: one continuous function, no 0° step, and $S_h(0^\\circ)$ moves from $+0.365$ to $+0.057$. 🔴 **But $\\alpha_0$ is, to RMS error 0.0076, an affine function of the plane count:** $\\alpha_0 \\approx 0.25 + 0.476\\log_4 N$ for $N=1,2,3,4$ planes (0°/180°/120°/90°). $\\ln N$ is the wellbore **flow-convergence penalty** — each extra plane converges into the same near-wellbore region, so more planes means **more** skin, i.e. monotonically **larger $\\alpha_0$ = worse**. The submission treats larger $\\alpha_0$ as **better**. 🔴 **The table encodes a penalty and is being read as a benefit.** If so, $S_h(90^\\circ)$ is the **worst** case, not the best — and $0^\\circ$ the best, which is also **standard perforation practice** (align with the fracture / max principal stress). The submitted table makes $0^\\circ$ the only damaging case and $90^\\circ$ the best: **the reverse of both** | **CONF-31 still partial** |
| **C-99** | 🔴🔴 **C-96 RE-VERIFICATION FAILED — the two DOIs still do not exist, and the claim was relabelled rather than corrected.** The response re-asserts `10.2118/12242-PA` and `10.2118/76722-PA` and changed the wording from *"Verified DOI"* to *"OnePetro Index"*. Re-checked 08-10-2026 against **doi.org** (404), **Crossref** (404), and a **Crossref bibliographic title search for both titles — neither appears anywhere in the index** (top hits are unrelated compositional papers). 🔴 A DOI is a registered identifier; if it 404s at `doi.org` it is not registered, and "OnePetro Index" is not a registered-DOI claim. ⚠️ The **Karakas & Tariq metadata correction IS accepted** (C-96a) — that part now matches what the DOI resolves to. 🔴 **But C-98 rests on an $\\alpha_0$ table attributed to "Karakas & Tariq (1991, Table 1)", which is unverified** — under C-96's own rule (*usable only once the DOI resolves **and** the metadata matches*), an unverified table cannot be the basis for a physics claim. **The branch unification may still be correct on engineering grounds; the $\\alpha_0$ values and their direction are not established** | **C-90/C-96 still open** |

### Register status after Ruling 10

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · **C-93** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-98, C-99) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> **Two consecutive rounds in which a "verified" claim failed verification.** Ruling 9: 2 of 3 DOIs did
> not exist. Ruling 10: the same 2 do not exist, and the wording was softened from *"Verified DOI"* to
> *"OnePetro Index"* rather than the identifiers being corrected.
>
> ✅ **The durable rule this produces — write it into `register_spec.md`:** a citation is usable only
> when **(1)** the DOI returns HTTP 200 from `doi.org`, **(2)** the registered metadata matches the
> cited metadata field-for-field, and **(3)** the specific table/figure is reachable. **A claim of
> verification is not itself evidence.** Until the $\\alpha_0$ direction is settled, the 0°/90° skin
> ordering must be **declared-absent**, not asserted.

---

## Ruling 11 — DOI corrected, anisotropy transform, C-98 withdrawn (08-10-2026)

Corrections **C-99a/b, C-100, C-101, C-102**. Detail: [`engine_invariants.md`](engine_invariants.md) §7i.

| # | Correction | Detail |
|---|---|---|
| **C-99a** | ✅ **Watts (1986) NOW VERIFIED.** The DOI digit was wrong, not the paper. Corrected to **`10.2118/12244-PA`**, which returns HTTP 200 and whose registered metadata matches the supplied citation on **all three checked fields** — author (Watts, J. W., 1986), title (*A Compositional Formulation of the Pressure and Saturation Equations*), journal/vol/issue/pages (SPE Reservoir Engineering, **1**(03), **243–252**). ✅ **Closes the Watts half of C-90/C-96/C-99** | **closes C-99a** |
| **C-99b** | 🔴 **Wong, Coats & Thomas STILL UNVERIFIED — no corrected identifier supplied.** Crossref title search returns **no match** (Coats is well indexed — 8 of his papers returned — and this title is not among them), and `10.2118/76722-PA` remains **404** at doi.org and Crossref. A content summary is not a citation. **Required: the SPE paper number or a DOI that returns 200.** ⚠️ Related and real: Coats, Thomas & Pierson (1995) *"Compositional and Black Oil Reservoir Simulation"*, `10.2118/29111-MS` **does** resolve — possibly the intended paper, but that is a guess and must not be substituted silently | **C-99b OPEN** |
| **C-100** | ⚠️ **PARTIALLY RETRACTED by C-232** — correct that C-98 was wrong; 🔴 **but "larger $\\alpha_0$ = better" is not a net rule** ($S_V$ opposes) — ⚠️ **and the measurement it preserved, $\\alpha_0\\approx0.250+0.476\\log_4N$, is itself wrong (C-228).** ✅ **WITHDRAWN — my C-98 "$\\alpha_0$ is a penalty read as a benefit" was WRONG.** The engineer's separation is correct and the physics checks out: **$\\alpha_0$'s $\\ln N$ scaling encodes geometric flow distribution in an ISOTROPIC matrix** — more planes cover more of the drainage circumference, so flow convergence into the entry area is *reduced* and more entry area is used. **Larger $\\alpha_0$ = better** in that regime. **Why field practice prefers 0°/180° is a different mechanism entirely** — permeability anisotropy and $\\sigma_{H,\\max}$ alignment, both *outside* the isotropic hydraulics $\\alpha_0$ encodes. My measurement ($\\alpha_0\\approx0.25+0.476\\log_4N$) stands; **my interpretation did not.** Withdrawn | **retraction** |
| **C-101** | 🔴 **The anisotropy transform cannot achieve its stated goal.** $\\theta'=\\arctan\\!\\big(\\sqrt{k_x/k_y}\\tan\\theta\\big)$ has **0° and 90° as fixed points**, so anisotropy leaves $0^\\circ\\!\\to\\!\\alpha_0{=}0.250$ (worst) and $90^\\circ\\!\\to\\!0.726$ (best) **unchanged** — it cannot make 0° superior in anisotropic media, which was its stated purpose. 🔴 **And $180^\\circ$ maps to $0^\\circ$**: the moment anisotropy is detected, opposed 2-plane phasing **silently becomes 1-plane phasing** ($\\alpha_0$ 0.500 → 0.250, $S_h$ −0.636 → +0.057). $120^\\circ$ maps to a **negative** angle (−60°, −73.9°), falling into the fallback | **CONF-31 still partial** |
| **C-102** | 🔴🔴 **The continuous fallback is a NaN generator.** `0.250 + 0.476 * (theta/90.0).log(4.0)` — Rust's `log(base)` is $\\ln x/\\ln(\\text{base})$, so this is $0.250+0.476\\log_4(\\theta/90)$. 🔴 **It crosses zero at $\\theta = 43.5^\\circ$**; for $\\theta<43.5^\\circ$, $\\alpha_0<0 \\Rightarrow r'_w<0 \\Rightarrow \\ln(r_w/r'_w)=$ NaN — **straight into the Jacobian.** Same failure class as **C-69** (NaN permeability) and **C-72**. 🔴 **It also does not fit its own table**: at $\\theta=90^\\circ$ it returns **0.250**, where the table says **0.726**. ✅ **Correct form**, since $N = 360/\\theta$ planes: $\\alpha_0 = 0.250 + 0.476\\log_4\\!\\left(360/\\theta\\right)$, which reproduces all four tabulated values and is monotone in $\\theta$, consistent with **C-100**. ⚠️ It still approximates — RMS 0.0076 (≈1 %) — so the four tabulated angles must keep exact values and the fit is for **interpolation only** | **CONF-31 still partial** |

### Register status after Ruling 11

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · C-93 · **C-99a** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-101, C-102, C-99b) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **The DOI lesson, applied correctly this time.** The Watts identifier was wrong in **one digit**
> (12242 → 12244) while the **paper was real and the metadata was already correct** — which is a very
> different failure from Ruling 9, where the DOI did not exist *and* the metadata was wrong. Looking the
> paper up by title, as the engineer did, was the right response to a 404; guessing the digit, which was
> my prior position, would have been wrong. **The rule from §7h.4 stands and now has a worked example
> in both directions.**
>
> **Six of my own claims withdrawn across Rulings 8–11:** C-79b (dynamic sparsity), C-81 (Penéloux
> "wrong variable"), C-89b ($S_v$ non-monotonicity), C-98 ($\alpha_0$ direction), plus C-79a/C-81b
> reformulations. **Each failed on the same mechanism — asserting a consequence before establishing
> whether the governing premise applied.** C-98 is the sharpest case: the arithmetic was right and the
> physics was wrong, because I never checked what regime $\alpha_0$'s correlation is defined in.

---

## Ruling 12 — NaN eliminated, but the anisotropy remedy inverts (08-10-2026)

Corrections **C-103 … C-107**. Detail: [`engine_invariants.md`](engine_invariants.md) §7j.

| # | Correction | Detail |
|---|---|---|
| **C-103** | ✅ **C-102 CLOSED.** $\alpha_0(\theta)=0.250+0.476\log_4(360/\theta)$ matches the correction exactly, reproduces the table (360°→0.250 ✅, 90°→0.726 ✅; 180° −2.4 %, 120° +1.5 %), and the engine rule is right: **exact values for the four tabulated angles, fit for interpolation only.** ⚠️ **One landmine:** the formula is **undefined at $\theta=0$** — $360/0=\infty \Rightarrow \log_4\infty=\infty \Rightarrow \alpha_0=\infty \Rightarrow S_h=\ln(0)=-\infty$. Safe **only** because 0° is served by the exact-table branch. **Ruled: a CI gate must assert the fallback branch is unreachable at $\theta=0$**, and $\theta=0$ must be rejected or handled before any division | **closes C-102** |
| **C-104** | 🔴 **DOI REGRESSION — the summary reverted a verified identifier.** The Summary section binds *"Watts (1986) `10.2118/12242-PA`"* — the **404** DOI. `10.2118/12244-PA` is the one that returns 200 with matching metadata (**C-99a**). ⚠️ This is a **copy-paste regression, not a new error** — but it is exactly the identifier §7h.4 exists to pin, so it must be corrected before the manifest is written | **C-99a at risk** |
| **C-105** | ✅ **C-99b CLOSED BY SUBSTITUTION — DOI verified real.** `10.2118/29111-MS` returns 200: **Coats, K. H., Thomas, L. K., & Pierson, R. G. (1995)**, *"Compositional and Black Oil Reservoir Simulation"*, **SPE Reservoir Simulation Symposium**. ⚠️ **Two differences from the original claim, both material:** **(a) citation class changed** — this is a **conference symposium paper ($-MS$)**, not the peer-reviewed *SPE Reservoir Evaluation & Engineering* journal article originally asserted; **(b) author initial** — **Thomas, L. K.**, not the originally cited *L. O.* ✅ Acceptable as the volume-balance authority **if** the register records it as a conference paper and the content claim is confirmed against it | **closes C-99b** |
| **C-106** | 🔴🔴 **The C-101 remedy inverts its own stated goal.** $L'_p=L_p\sqrt{(k_\\bar/k_x)\\cos^2\\phi_p+(k_\\bar/k_y)\\sin^2\\phi_p}$ has three defects: **(a) dimensional** — the $\\sqrt{k_\\bar}$ prefactor carries units $\\sqrt{\\text{mD}}$, so $L'_p$ is m·$\\sqrt{\\text{mD}}$, **not metres**, and $r_w+L'_p$ is **dimensionally invalid**; **(b) inverted** — it **shortens** $L_p$ along the **high**-$k$ axis, shrinking $r'_w$ and making the aligned case **worse**: at $k_x/k_y=4$ it gives $S_h(\\phi_p{=}0^\\circ)=+0.300$ against the isotropic $+0.057$, while $\\phi_p{=}90^\\circ$ gives $-0.209$ — i.e. **perforating along the LOW-permeability axis scores best**, the exact inverse of the rationale given; **(c) category error** — $L_p$ is **drilled geometry**, fixed at perforating time and independent of the permeability tensor. Anisotropy belongs in the **flow response**, not in the hole geometry | **CONF-31 still partial** |
| **C-107** | ⚠️ **$\\phi_p$ is a new undeclared symbol** — perforation **azimuth**, distinct from the phasing **spacing** $\\theta$. CONF-31's pattern, **fourth occurrence**. ⚠️ Also: with the angle map purged, $\\alpha_0(\\theta)$ is now applied **raw** in anisotropic media. That is defensible — **C-100** established $\\alpha_0$ encodes *isotropic* distribution — but it makes $\\alpha_0(\\theta)$ a crude proxy under anisotropy, and `ValidityWarning::AnisotropicPerforationTransformation` must cover **that**, not only $L_p$ | **CONF-31 still partial** |

### Register status after Ruling 12

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · C-93 · **C-99a · C-99b · C-102** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-106, C-107) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **The $\\theta=0$ infinity path is the same shape as the NaN path it just fixed.** C-102 removed a
> *negative* $\\alpha_0$ producing NaN; the replacement introduces a *division by zero* producing
> $-\\infty$ at $\theta=0$. It is defended only by a `match` arm the caller must hit. **That is a
> fragile invariant — it belongs in the type or the CI gate, not in control flow.**
>
> ⚠️ **And the anisotropy remedy needs a decision, not another formula.** Two coherent options:
> **(a)** $L_p$ is geometry and must **not** be permeability-transformed — drop the transform, keep
> $\\alpha_0(\\theta)$ raw, and declare anisotropy via `ValidityWarning`; **(b)** if an
> isotropic-equivalent length is genuinely wanted, it must be **normalised** (factor = 1 along $k_{max}$)
> and is then a **flow** correction applied downstream of $r'_w$, never inside $r_w + L_p$.
>
> ⚠️ **Watch the pattern, not just the instance.** This is the **fourth** undeclared symbol from this
> same correlation. **The rule stands: no symbol enters a constitutive relation without a unit, a range,
> and a source** — and it is cheaper to enforce at the register than to find in review.

---

## Ruling 13 — DOIs bound, Option (a) adopted, Symbol Register Rule (08-10-2026)

Corrections **C-108 … C-111**. Detail: [`engine_invariants.md`](engine_invariants.md) §7k.

| # | Correction | Detail |
|---|---|---|
| **C-108** | ✅ **C-104 and C-105 CLOSED.** Three DOIs bound and class-labelled: `10.2118/29111-MS` (**Coats, Thomas L.K. & Pierson 1995**, **Conference Proceeding**) · `10.2118/12244-PA` (**Watts 1986**, *SPE Reservoir Engineering* **1**(03) 243–252, **Journal**) · `10.2118/18247-PA` (**Karakas & Tariq 1991**, *SPE Production Engineering* **6**(01) 73–82, **Journal**). ✅ The `12242-PA` regression is corrected to the verified `12244-PA`, and the **$-MS$ class distinction is retained** rather than quietly upgraded. ✅ **All three pass §7h.4 test 1 (DOI returns 200); tests 2 and 3 are met for the two journal papers, and test 3 (the specific table) remains outstanding for $\\alpha_0$** — see **C-98/C-100** | **closes C-104, C-105** |
| **C-109** | ✅ **C-106 CLOSED — Option (a) adopted, as recommended.** $L_p$ is restored as **raw drill geometry**; the $\\sqrt{\\bar k}$ dimensional defect, the sign inversion, and the category error are all purged. ✅ **Anisotropy correctly re-routed** to where it physically belongs: the **grid block tensor $K_{ij}$** in the FVM equations, and **$h_D = (h_p/L_p)\\sqrt{k_h/k_v}$** for $S_v$. ✅ `ValidityWarning::AnisotropicPerforationRawCorrelation { k_x, k_y, ratio }` with the $\\max(k_x,k_y)$ normalisation — the corrected asymmetry fix. **C-106 closed** | **closes C-106** |
| **C-110** | ⚠️ **Symbol Register Rule ACCEPTED — it is the right rule. ⚠️ But its own worked example violates it.** The mandate requires 4 points per symbol: name, units, valid domain, **verified source**. Its illustration declares **$L_p \\in [0.05,\\ 1.50]\\ \\text{m}$** as the valid domain with **no cited source**. 🔴 So the rule's first application would pass a naive checklist while failing its **own fourth point** — which is precisely the defect class it exists to remove. ✅ **Ruled: the rule applies to its own worked examples**, and the $\\alpha_0$ table (no source) is the standing counter-example. Every worked example in the rule must carry a DOI or be explicitly marked `SOURCE_PENDING` | **accepted with condition** |
| **C-111** | 🔴 **The $\\alpha_0$ fit has no lower domain bound — and that is the C-97 defect one level up.** The tabulated angles are $\\{90°, 120°, 180°, 360°\\}$, so the interpolation domain is **$[90°, 360°]$**. The engine rule says *"non-standard continuous angles shall use the $N = 360/\\theta$ fit"* with **no lower bound**, so $\\theta < 90°$ silently extrapolates: \| $\\theta$ \| $N$ \| $\\alpha_0$ \| $r'_w$ \| $r'_w/r_w$ \| $S_h$ \| \| 90° \| 4 \| 0.726 \| 0.296 m \| **2.74×** \| −1.009 (**interp**) \| \| 45° \| 8 \| 0.964 \| 0.393 m \| **3.64×** \| −1.292 (**EXTRAP**) \| \| 15° \| 24 \| 1.341 \| 0.547 m \| **5.07×** \| −1.623 (**EXTRAP**) \| \| 5° \| 72 \| 1.718 \| 0.701 m \| **6.49×** │ −1.871 (**EXTRAP**) \| \| 1° \| 360 \| 2.271 \| 0.927 m \| **8.58×** │ −2.149 (**EXTRAP**) \| 🔴 **An \"equivalent wellbore radius\" of $8.6\\times$ the actual hole, and a productivity index $\\sim 39\\times$ unskinned, emitted with no warning.** 🔴 **C-97 was accepted this same round for exactly this defect class** — declare the domain, do not substitute silently — and it was applied to Karakas-Tarik's $r_D/h_D$ but **not** to our own new fit. ✅ **Ruled: bound the fit at $\\theta \\ge 90°$ and emit `ValidityWarning::ExtrapolatedPhasing { theta, plane_count }` below it.** Do **not** clamp $\\theta$ (**C-93**) | **CONF-31 still partial** |

### Register status after Ruling 13

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · 93 · 99a · 99b · 102 · **104 · 105 · 106** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-111) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **C-110 and C-111 are the same lesson at two levels, and it is worth naming because it recurs
> reliably.** A rule is written to prevent a defect (**C-97**); the defect reappears one level up in the
> rule's own worked example (**$L_p$'s range, no source**) and again in the same class of quantity the
> rule governs (**the $\\alpha_0$ fit's unbounded extrapolation**). **Every rule needs a CI gate that
> tests the rule against its own examples**, or it protects only the code it was written for.
>
> ⚠️ **Also still open, carried from C-103:** $\\theta = 0$ remains undefined in the fit
> ($360/0 \\Rightarrow S_h = -\\infty$), defended only by the exact-table `match` arm. The table path is
> correct — $r'_w = 0.102$ m, $S_h = +0.057$ — but the guard is **control flow, not a gate**.

---

## Ruling 14 — completion_skin_spec.md review (08-10-2026)

Spec: `D:\Downloads\karakas_tariq_completion_skin_spec.md` (135 lines, external to the repo).
Corrections **C-112 … C-115**. Detail: [`engine_invariants.md`](engine_invariants.md) §7l.

| # | Correction | Detail |
|---|---|---|
| **C-112** | ✅ **C-109 and C-111 ACCEPTED as specified.** $L_p$ preserved as raw drill geometry; anisotropy routed to **$K_{ij}$** (FVM) and **$h_D = (h_p/L_p)\\sqrt{k_h/k_v}$** with $k_h=\\sqrt{k_xk_y}$, $k_v=k_z$ — ✅ that closes dimensionally. Extrapolation guard for $\\theta<90^\\circ$ emits `ValidityWarning::ExtrapolatedPhasing { theta_deg, plane_count, computed_alpha_0 }` **without clamping** ✅. $\\theta=0^\\circ$ intercepted at the match/type level ✅. Interpolation domain **$[90^\\circ, 360^\\circ]$** stated ✅ | **closes C-109, C-111** |
| **C-113** | ✅ **Gates 1 and 2 are NUMERICALLY CORRECT — verified, both pass.** Gate 1: $\\theta=45^\\circ \\Rightarrow N=8$, $\\alpha_0=0.9640$, $r'_w=0.39331$ m, $S_h=\\mathbf{-1.2925}$, and `-1.2925 < -1.0` ✅. Gate 2: $\\theta=0^\\circ \\Rightarrow \\alpha_0=0.250$, $r'_w=0.10200$ m, $S_h=\\mathbf{+0.05716}$, and `round(S_h*1000)/1000 == 0.057` ✅. ✅ Both assertions are *derived*, not guessed — they encode C-103/C-111 correctly | **gates 1, 2 verified** |
| **C-114** | 🔴🔴 **The two NEW DOIs in the symbol register DO NOT RESOLVE — and the document's own gate cannot detect it.** Verified 08-10-2026: · `10.1016/B978-012248308-0/50001-X` (**Fanchi 2002**, cited for $r_w$) — **404 at doi.org**, **absent from Crossref** · `10.1016/C2013-0-06222-0` (**Aziz & Settari 1979**, cited for $k_h$, $k_v$) — **404 at doi.org AND Crossref**, **absent from Crossref title search**. ⚠️ Both authors are **well indexed** (Fanchi returns 5 real papers; Aziz & Settari return SPE-3174, SPE-72-01-04), so the absences are meaningful, not coverage gaps. 🔴 **Gate 3 is a string test — `source_doi.contains("10.")` — and every fabricated DOI passes it:** `10.2118/99999-PA` ✅ PASS, `10.1016/totally-made-up` ✅ PASS. **The rule as written says "valid DOI or SOURCE_PENDING"; the gate as written tests "contains `10.`". The gate cannot enforce the rule.** ✅ **Ruled: the source test must perform the §7h.4 test-1 resolution check** (HTTP 200 from `doi.org`), cached in a lockfile, not a substring match | **C-108 REOPENED** |
| **C-115** | 🔴 **`valid_range: (f64, f64)` cannot express two of the ten declared domains — and its own gate then admits invalid input.** The register declares **$k_h \\in (0.0,\\ 50000.0]$ mD** — an **open** lower bound excluding zero — and **$r_{cz} \\in [r_p,\\ r_p+0.050]$ m**, whose lower bound is a **symbol**, not a number. 🔴 A `(f64,f64)` pair carries **neither** openness nor a symbol-valued bound. Consequence, measured: `assert valid_range.0 < valid_range.1` evaluates `0.0 < 50000.0 = true`, so **the gate accepts $k_h = 0$ mD** — a physically meaningless permeability that the declared domain explicitly excludes. And $r_{cz}$ cannot be stored at all. ⚠️ **This is the C-83 / C-106 dimensional-closure class recurring one level up:** the *data structure* cannot express the domain the physics needs. ✅ **Ruled: the register needs a typed bound** — `{ value: BoundValue, inclusive: bool }` where `BoundValue` is `Num(f64)` or `Sym(symbol_id)` | **CONF-31 still partial** |

### Register status after Ruling 14

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · **109 · 111** |
|---|---|
| **Reopened** | **C-108** — the bibliography manifest gained two unverifiable DOIs (**C-114**) |
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-114, C-115) · **CONF-51** · **CONF-66** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **The document that closes the provenance finding introduces two more unverifiable DOIs, and its
> own gate is structurally incapable of noticing.** C-114 is the sharpest illustration yet of why §7h.4
> exists: *"a claim of verification is not itself evidence."* Gate 3 is a claim of verification encoded as
> a substring match. It would pass `10.2118/99999-PA`.
>
> ⚠️ **And the failure is symmetric with C-115.** The register is a **data structure**, and its gate is a
> **test of that structure**. Where the structure cannot represent the physics ($k_h\\in(0,\\dots]$,
> $r_{cz}\\in[r_p,\\dots]$), the test passes vacuously. **Both C-114 and C-115 are the same shape:
> the enforcement mechanism is weaker than the rule it enforces.** That is the single most transferable
> finding in this exchange, and it is why the consolidated gate in §7l.4 tests **resolution**, not
> **syntax**.
>
> ✅ **To be explicit about what is now solid:** gates 1 and 2 are correct and verified, $L_p$ is
> geometry, anisotropy is routed properly, the extrapolation guard is right, and the $\\alpha_0$ domain
> is declared. **CONF-31 now has a working enforcement mechanism with two specific, small holes.**

---

## Ruling 15 — local library audit, `D:\RAG` (129 PDFs scanned) (08-10-2026)

Corrections **C-116 … C-119**. Method: `pymupdf` full-text extraction over **every page of all 129 PDFs**
in `D:\RAG` (installed to **system** Python; the project `.venv` was left untouched). Detail:
[`engine_invariants.md`](engine_invariants.md) §7m.

| # | Correction | Detail |
|---|---|---|
| **C-116** | 🔴 **CONF-31's blocker is now DEFINITIVELY characterised — and it cannot be closed locally.** A literal search for **`karakas`** or **`tarik`** across **all pages of all 129 PDFs** returns **zero hits**. ✅ The **Karakas & Tariq (1991) α₀ table is not in the local library**, so it cannot be corroborated offline. ✅ **Ruled: the α₀ table is `SOURCE_PENDING`**, the perforation-skin model is treated as **provenance-unverified**, and the engine must emit `ValidityWarning::CorrelationProvenanceUnverified { correlation: "Karakas-Tariq 1991 alpha_0", evidence: "local library contains no copy" }`. ⚠️ **M7c cannot be gated on it** — the skin model is a **declared-absent / unverified** dependency until the paper is obtained through institutional access | **CONF-31 partially resolved by characterisation, not closure** |
| **C-117** | ✅ **Aziz & Settari (1979) IS in the library, and the DOI failure is now DIAGNOSED.** `kupdf.net_khaled-aziz-reservoir-simulation.pdf` = **Aziz, Khalid & Settari, Antonin**, *Petroleum Reservoir Simulation*, **APPLIED SCIENCE PUBLISHERS LTD, LONDON**, **ISBN 0-85334-787-5**, © **J979** (confirmed from the title page, author page and British Library cataloguing page). 🔴 **The publisher is Applied Science Publishers (UK), not Elsevier** — and `10.1016/` is **Elsevier's** prefix. **A `10.1016/` DOI could not possibly be correct for this book**, which is why `10.1016/C2013-0-06222-0` returns 404. ✅ **Ruled: replace the fabricated DOI with `ISBN 0-85334-787-5`**, which is locally verifiable and a *better* identifier for a 1979 book. ⚠️ **The book contains no perforation-skin correlation** (one incidental "skin effect" mention on p247), so it **cannot** substitute for Karakas & Tariq | **C-114 half-closed for $k_h,k_v$** |
| **C-118** | ⚠️ **Fanchi IS in the library — but a DIFFERENT work than the one cited.** `vdocuments.mx_shared-earth-modeling.pdf` = John R. Fanchi, *Shared Earth Modeling*, **Butterworth-Heinemann / Elsevier Science, © 2002** (imprint confirmed on the copyright page). ✅ So an Elsevier `10.1016/` prefix is **structurally plausible** for Fanchi — unlike C-117. 🔴 **But the symbol register cites Fanchi for the $r_w$ domain, and the cited work is not this one**; Fanchi's wellbore-radius content would be in *Petroleum Reservoir Engineering: A Computer Approach*, not *Shared Earth Modeling*. **Ruled: cite Fanchi by exact title + year, and drop the unresolvable chapter DOI until verified.** ⚠️ **Neither bound DOI string (`10.2118/18247`, `10.2118/12244`, `10.2118/29111`) appears as text anywhere in the corpus** — no local corroboration of the three-verified bibliography entries either | **C-114 half-closed, `$r_w$ still open** |
| **C-119** | ✅ **MATERIAL ASSET DISCOVERED — an SPE benchmark case is present.** `D:\RAG\Data files\` contains **31 CMG GEM decks**. `SPE5-ProbForecasting-BaseCase.txt` is a genuine **RESULTS Simulator GEM 202310** deck: `*TITLE1 'SPE5 : SPE5 COMPOSITIONAL RUN 1'`, `*TITLE2 'WAG process with 1 year cycle'`, `*GRID *CART 7 7 3`, `DI/DJ = 1000 ft`, `*DK *KVAR 50/30/20`, `POR KVAR 0.2/0.22/0.18`, 208 lines. ✅ **SPE Comparative Solution Project Case 5** is the standard **compositional + WAG** benchmark — directly relevant to M1–M4 and to **CONF-26** (spatial convergence promised, never tested). Also present: `CO2 Flooding_BaseCase`, `PolymerFlooding_BaseCase`, `SAGD_BaseCase` / `SAGD_2D_` / `SAGD_Green_`, `ShaleOil_HF_BaseCase`, `HydraulicallyFracturedBaseCase`, `WellTesting_Base`, `HM_00227` / `HM_00686`, plus DTO sample files (`CompletionsDataSource`, `fluidProperties`, `ElasticPropertyBuilding`, `Tornado`, `SimultaneousInversion`). ⚠️ **These are CMG-format decks, not the official SPE problem specifications** — they are a *starting point* for a reference case, and a CMG reference run is still required for a comparison target. The wiki's \"no SPE benchmark\" gap was accurate **for the repository**; it is **materially incomplete for the machine** | **asset gap closed** |

### Register status after Ruling 15

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · **C-117 · C-119** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-16** · **CONF-25** · **CONF-31** (C-116 characterisation) · **CONF-51** · **CONF-66** |
| **Provenance-unverified** | Karakas & Tariq (1991) **α₀ table** — **not obtainable locally** (C-116) · Fanchi **$r_w$ domain** (C-118) |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 15 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 29 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **The honest answer to the question the library was meant to answer.** C-116 is the result that
> matters: **the one source blocking CONF-31 is not on this machine.** Everything else the library could
> do, it did — it confirmed two of the three disputed citations' *identities* (C-117, C-118), diagnosed a
> DOI failure down to its **publisher prefix**, and produced an SPE benchmark the wiki did not know existed
> (C-119).
>
> ⚠️ **And two false positives were eliminated rather than reported as support.** `KAPPA DDA book`'s
> "tariq" is **Umair Tariq**, a KAPPA engineer in the acknowledgements; its "wellbore effect" means
> **wellbore storage**, not perforation phasing. Neither bears on the α₀ table. **A keyword hit is not
> evidence** — the same standard §7h.4 applies to a local corpus as to a DOI.

---

## Ruling 16 — literature parked; math batch CONF-15/16/25/29 (08-10-2026)

Literature work **parked** in [`literature_todo.md`](literature_todo.md) with an acceptance test and
search protocol. Detail: [`engine_invariants.md`](engine_invariants.md) §7n.
Corrections **C-120 … C-123**.

| # | Correction | Detail |
|---|---|---|
| **C-120** | ⚠️ **CONF-25 narrowed and specified — the missing $-1$ is a CLASSIFICATION defect, not an optimisation defect.** Measured: the argmin of $Q$ is **bit-identical with and without the $-1$** (max $\\lvert\\Delta y_i\\rvert = 0.00\\mathrm{e}{+00}$) — because subtracting a constant cannot move a minimiser. 🔴 **But the stability test is $Q_{min}<0$, which is meaningless without it**: measured $Q_{min}=+0.351$ without the $-1$ vs $-0.649$ with it, so **an unstable split cannot be detected at all** and a stable one is classified for the wrong reason. ✅ **Ruled specification:** (i) minimise $\\mathcal{W}(y)=\\sum_i y_i\\ln\\!\\left[\\dfrac{y_i}{x_i}\\dfrac{\\phi_i^L}{\\phi_i^V}\\dfrac{1}{K_i}\\right]-1$ subject to **two** constraints ($\\sum y_i=1$ and the volume balance), **not one**; (ii) $\\phi_i^L$ and $\\phi_i^V$ must be **named separately** — the submitted $\\phi_i(y)$ is ambiguous and the ratio is the whole point; (iii) the **gradient is mandatory** — it needs $\\partial\\ln\\phi/\\partial y_i$, i.e. EOS partials along the volume-balance path, so the Newton matrix cannot be assembled from $\\phi$ alone; (iv) **use $\\partial W/\\partial y_i$ for the iteration and $Q_{min}$ for the verdict** — they are different quantities and conflating them is what the missing $-1$ caused | **CONF-25 narrowed** |
| **C-121** | ✅ **CONF-15 RECLASSIFIED $\\orangearrow\\yellow$: my register OVERSTATED it, and my own first instinct was wrong.** ✅ **(c) withdrawn** — I initially read Koval's placement as a sign error, but the design's $t_{bt}\\propto 1/K$ **decreases** with $K$, and that is the **physically correct direction**: severe fingering $\\Rightarrow$ the displacing phase bypasses $\\Rightarrow$ **earlier** breakthrough. The register is right about the sign. ✅ **(b) withdrawn** — $(1-S_{wi})$ is the **classical textbook** mobile-oil fraction in $W_{o,bt}=PV(1-S_{wi})/B_o$; neglecting $S_{or}$ is a known second-order refinement, not a logical or dimensional error. 🔴 **What genuinely remains is documentation only:** the unit basis for $q_{inj}$ (reservoir vs surface), whether $K$ is declared dimensionless, and the approximation level. Measured: $V_p/(K\\,q)$ closes dimensionally **only** under those two unstated assumptions | **CONF-15 $\\to$ 🟡 documentary** |
| **C-122** | ✅ **CONF-16 fix specified — a missing reference volume, and the same shape as C-69.** $\\boldsymbol\\varepsilon_{chem}=\\tfrac13\\Delta V_{m,tot}\\mathbf{I}$ carries **volume** units; a strain is dimensionless. ✅ **Ruled:** $\\boldsymbol\\varepsilon_{chem}=\\tfrac13\\dfrac{\\Delta V_{m,tot}}{V_{ref}}\\mathbf{I}$ with $V_{ref}$ = **bulk pore volume**, declared as a state variable, not a constant. ⚠️ **This is the same defect class as the Verma-Pruess guard (C-69)** — a **reference quantity is missing**, so the ratio is unbounded. 📌 Both should share one register entry for \"missing reference denominator\" | **CONF-16 fix specified** |
| **C-123** | ✅ **CONF-29's arithmetic VERIFIED CORRECT — no literature needed.** For $N_c = 6$: unique off-diagonal $i<j = 15$, diagonal self-interactions $= 6$, total entries in a symmetric $6\\times6$ $= 21$. The design's $2\\times3 = 6$ entries leave **exactly 15 absent**. ✅ The register's figure is right; 🔴 the defect is the **payload**, not the count. ⚠️ Note $15$ is the count of *unique binary interactions* — if $k_{ii}$ are also needed for a **mixing rule** (e.g. a Lorentz-Berthelot combination), the count rises to **21** and the shortfall is worse | **CONF-29 confirmed** |

### Register status after Ruling 16

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 |
|---|---|
| **Reclassified** | **CONF-15** $\\to$ 🟡 documentary (**C-121**) |
| **Partially closed** | **CONF-14** · **CONF-25** narrowed (**C-120**) · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-16** (**C-122**) · **CONF-29** confirmed (**C-123**) |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 27 · 28 · 30 · 30b · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 43 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **C-121 is the seventh withdrawal of my own claim in this exchange, and the first one against my
> own register rather than against a proposal.** Two independent instincts — the Koval sign and the
> $(1-S_{wi})$ fraction — were both wrong, and in both cases the **design text was right and the
> critique was not**. The measurement that settled it was re-deriving the direction from the physics
> rather than from the shape of the formula.
>
> ⚠️ **And the meta-point that matters for the project:** CONF-15 was graded $\\orangearrow$ material
> partly because it *read* like a physics error. **Severity assigned by plausibility rather than by
> derivation is the same failure mode as asserting without checking** — it produced a finding that
> survived nine rulings on the strength of its phrasing. **Derive the number, then grade it.**

---

## Ruling 17 — architecture: all seven items ruled (08-10-2026)

Corrections **C-124 … C-130**. **Five accepted outright; five defects found inside the rulings; one
infrastructure reference I was wrong to doubt is verified real.** Detail:
[`engine_invariants.md`](engine_invariants.md) §7o.

| # | Correction | Detail |
|---|---|---|
| **C-124** | ✅ **Compile-time zero-cost newtypes ACCEPTED over `uom`** — correct, because `uom` checks at runtime and the Newton hot loop cannot afford it. 🔴 **But the code sketch defeats its own purpose:** `#[repr(transparent)] struct Pascals(pub f64)` exposes `.0`, returning a bare `f64` freely mixable with any other `f64`. ✅ That keeps the **zero-cost** property and **loses the type-safety** property. **The field must be private** with a getter/`Deref`, or the dimensional checking is theatre. ⚠️ And **dimensional analysis is not a compile-time check** with plain newtypes — it runs in `#[cfg(test)]` at **test time**. State it as a **test gate**, not a build gate, or nobody will expect a compile error (**C-115**) | accepted with correction |
| **C-125** | ✅ **M2 FORMULATION RESOLVED — Option A, overall composition $(P, S_\\alpha, z_i)$.** ✅ **This closes C-85**, which was mine and had been the highest-leverage open decision. ✅ State-vector size $1 + (N_p-1) + (N_c-1)$ **verified correct** (two constraints: $\\sum S_\\alpha = 1$, $\\sum z_i = 1$). ✅ The $C^1$ continuity argument for $z_i$ is the standard one and is sound; it removes variable switching and the $C^0$ residual breaks that cause oscillation near the critical point. ✅ **`feos-ad` VERIFIED REAL** — v0.2.3, 2025-05-28, 4 346 downloads, same author/org as `feos`, MIT/Apache-2.0, keywords `autodiff`/`equations_of_state`. ⚠️ **I initially assumed it was fabricated and was wrong** — the CONF-31 lesson caught in the act; the check was correct and the instinct was not. ⚠️ **Pin `default-features = false` and allowlist features explicitly.** `feos` 0.8.0 shipped a `python` feature (pyo3) that links libpython; 0.10.1's feature list no longer shows it, but an explicit allowlist is what *guarantees* the P1 separation doctrine rather than trusting a version bump. ⚠️ **$C^1$ on the primary variable $\\neq$ $C^1$ on the residual** — the flash map still changes character where the active phase set changes, so variable switching is gone but near-boundary convergence still needs **C-33/C-72** | **closes C-85** |
| **C-126** | ✅ **DTO bus DELETED from the compute core — closes CONF-27 and CONF-28 together.** ✅ This is the right call: the approved two-output contract (HDF5 + Parquet, open formats) already replaces a DTO bus, and deleting it removes the IPC-vs-Zero-Bloat contradiction outright rather than resolving it. 🔴 **But restart state cannot live inside a run artifact whose failure invalidates it.** You restart **failed** runs — so if `/Snapshots/t_n` sits in the run HDF5 and **INV-1/C-51** quarantine failed runs, **the restart point is destroyed by the rule that protects the artifact**. ✅ **Checkpoint must be a separate, independently-committed file** with its own atomic write and its own lifecycle. ⚠️ The restart vector $[P,T,S_\\alpha,z_i,\\sigma_{ij},\\mathbf u]^T$ is only valid when geomechanics is active — it must be **driven by the INV-6 capability declaration**, not fixed | **closes CONF-27, CONF-28** |
| **C-127** | ✅ **`petekIO` declared in-memory substrate ONLY — closes CONF-30b by deletion.** ✅ The parser boundary becomes **GRDECL / RESQML / RESCUE** only. 🔴 Elegant: the seven unstated items (magic bytes, header struct, byte order, slab typing, chunk table, alignment, version) **stop being requirements** because there is no file to specify. This is the second *deletion* ruling and the second time one has closed two conflicts at once | **closes CONF-30b** |
| **C-128** | 🔴 **The status lattice is NOT a partition — there is a hole.** Counter-case: **all active modules = `Degraded`, all inputs inside the V&V envelope.** · Tier 3 requires `Failed` or a typed error → **fails** · Tier 1 requires **all `Implemented`** → **fails** (`Degraded` $\\neq$ `Implemented`) · Tier 2 requires (outside-envelope **OR** $\\exists$ `NotImplemented`) → **fails** (neither holds) ⇒ **no tier matches; the run is unclassified.** ✅ **Ruled: strict priority cascade** — Tier 3 first, then Tier 1, then **Tier 2 = everything else that completed**. 🔴 Second defect: Tier 2 **conflates `ModuleAbsent` with envelope violation**. A declared-absent Domain 5 is *normal* for an M3 hydrodynamics run (**INV-6**) and says nothing about whether the input was verified. ✅ **Tier 2 = "completed with $\\ge1$ warning"**, and the warning list must distinguish `EnvelopeExceeded` / `ModuleAbsent` / `CorrelationProvenanceUnverified` | corrected |
| **C-129** | ✅ **`random_seed: u64` required in all stochastic DTOs — closes CONF-43**, and the cheapest close on the register. 🔴 **But "100 % bit reproducibility" does not follow from a seed alone.** Three further requirements: **(a)** a **fixed PRNG algorithm *and version*** — PCG64, ChaCha and xoshiro yield different streams from the same seed; **(b)** **deterministic reduction order** — Rayon's parallel `f64` reduction reorders summation, so output depends on thread count; **(c)** `f64` behaviour pinned (Rust forbids fast-math, which helps). ✅ **Ruled: per-cell independent streams keyed by `hash(seed, cell_index)`** — this makes the realisation independent of thread count *and* traversal order, which is the only way the **SPE5 cross-simulator comparison (C-119)** can be bit-stable | accepted with correction |
| **C-130** | ✅ **Single crate for M0–M6, Cargo Workspace refactor deferred to M7.** ✅ Sensible, and consistent with `separation_doctrine.md`. ⚠️ Note the `deny(unsafe_code)` vs `russell_sparse` FFI conflict is **already recorded as breaking the lint** — that needs a scoped `#[allow]` on the FFI module or a different solver, and it is an **M0** decision **independent of crate layout** | accepted |

### Register status after Ruling 17

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 |
|---|---|
| **Partially closed** | **CONF-14** (now unblocked on the formulation half) · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **Two deletion rulings closed four conflicts between them (C-126, C-127).** Both worked the same
> way: a requirement looked unimplementable because it was **underspecified**, and the fix was to notice
> the requirement **did not need to exist**. CONF-28 had no checkpoint format because a DTO bus was
> redundant against an approved file contract; CONF-30b had no magic bytes because `petekIO` was never a
> file. **Specification work that ends in a deletion is cheaper than specification work that ends in a
> document** — and this register has spent many rulings on documents.
>
> ⚠️ **And a note on my own process.** I read `feos-ad` as fabricated before checking, on the strength of
> **six prior rulings** in which the engineer had supplied unverifiable identifiers. **The prior was
> right; the generalisation was wrong.** Had I not checked, I would have filed a false `PROVENANCE`
> finding against a **real** crate — and the register would have carried an error I introduced by pattern
> matching instead of measurement. **C-125 records the verification; §7h.4 test 1 is what caught it.**

---

## Ruling 18 — private newtypes, cascade lattice, .ckpt.h5, determinism triad (08-10-2026)

Corrections **C-131 … C-135**. **Four accepted; four refinements.** Detail:
[`engine_invariants.md`](engine_invariants.md) §7p.

| # | Correction | Detail |
|---|---|---|
| **C-131** | ✅ **C-124 ACCEPTED — private-field newtype is the correct form.** `pub struct Pascals(f64)` + `new()` + `as_f64()` ✅ correct: cross-dimension arithmetic is **blocked at compile time**, and conversion requires an explicit `From`/`Into` call. ✅ `as_f64()` as the single named escape hatch is the right shape — it is **explicit and greppable**, so boundary conversions can be audited by `rg`. ⚠️ Two refinements: **(a)** `as_f64()` is a real escape hatch, so the gate must **count and bound its uses inside constitutive relations** (it must not appear in the solver hot path). **(b)** `#[derive(PartialOrd)]` is physically meaningful only for **scalar-ordered** quantities; deriving it on a tensor-valued or direction-ambiguous type invites meaningless comparisons — derive it per-type, not blanket | **closes C-124** |
| **C-132** | ✅ **C-128 hole CLOSED — the cascade is now a partition over _completed_ runs.** Verified: all-`Degraded` + in-envelope now fires **Tier 2** (the added "$\\lor\\ \\exists$`Degraded`" disjunct), and all-`Implemented` + in-envelope fires **Tier 1**. ✅ **INV-6 decoupling accepted and correct** — `ModuleAbsent` does not participate in active-module evaluation and does **not** downgrade `Validated` for domains that *are* active. 🔴 **But a FOURTH state is still missing: incomplete-but-not-failed.** A run stopped by the user (**INV-2** — *"the engine runs only when the user starts it"*) produces **no** `Failed` module and **no** `TypedError`, yet did **not** complete. · Tier 3 requires a failure → **no** · Tier 2 requires "run completed" → **no** · Tier 1 requires "run completed" → **no** ⇒ **unclassified, exactly as before.** ✅ **Ruled: add `Aborted`** (user-initiated stop / graceful termination). It is **not** `Validated`, **not** `Unsolvable`, and — per **INV-1** — a partial artifact is **never** committed to the training store. ✅ **CI gate: the four classes must be provably exhaustive over $\{$complete, incomplete$\\}\\times\\{$failure, no-failure$\\}$** | **C-128 still open, one state short** |
| **C-133** | ⚠️ **C-126 `.ckpt.h5` decoupling ACCEPTED**, and the INV-6-driven state vector is exactly right — M2/M3 $[P,T,S_\\alpha,z_i,\\text{WellStates}]$, M7+ $[\\dots,\\sigma_{ij},\\mathbf u,w_f,C_k,\\text{WellStates}]$. 🔴 **But the checkpoint's OWN durability has no rule** — and that is the whole point of the file. A torn write (crash mid-`flush`, full disk, killed process) would leave a `.ckpt.h5` that **loads without error and yields a corrupted restart** — strictly worse than no restart, because it fails silently. ✅ **Ruled:** write to `run_name.ckpt.h5.tmp`, then **atomic rename**; write a `commit_token` dataset **last**; a checkpoint is loadable **iff** `commit_token` matches the expected run/step identity. Partial files are unlinkable by construction | **closes C-126, with durability clause** |
| **C-134** | ✅ **C-129 determinism triad ACCEPTED, with the floating-point claim now MEASURED.** Engineer's per-op scale is consistent: measured naive summation over $N=10^{6}$ gives **relative error $1.29\\times10^{-14}$**, matching a $\\sqrt{N}\\,\\varepsilon$ walk. 🔴 **Correction to my own expectation — I expected Kahan to buy accuracy but not reproducibility, and the measurement contradicts that.** Measured, on the same data: · **naive summation is order-dependent** — shuffling the input changes the bit pattern ($500161.97345979\\!54$ → $500161.97345980\\!775$, $\\lvert\\Delta\\rvert = 1.23\\times10^{-8}$) at the *same* accuracy · **Kahan was bit-identical across both orders** and matched `math.fsum` exactly (error **0.0**). ✅ So Kahan is **not** redundant — it delivers accuracy **and** order-independence. ⚠️ **But the guarantee lives in the ORDER clause, not the Kahan clause.** Kahan is not *provably* order-independent for all inputs; the fixed strict-cell-index tree merge is what turns reproducibility from a *likelihood* into a **guarantee**. ✅ **Ruled: keep both, and document the fixed-order clause as the load-bearing element** | **closes C-129** |
| **C-135** | 🔴 **Determinism $\\ne$ cross-simulator comparability — and the protocol only delivers the first.** ✅ Per-cell ChaCha8 streams keyed $\\text{Blake3}(\\text{MasterSeed}\\,\\Vert\\,\\text{CellIndex}\\,\\Vert\\,\\text{DomainID})$ give **perfect reproducibility within Rust**. 🔴 **But they cannot reproduce a CMG or SPE reference realisation** — those used different PRNGs entirely. ⚠️ So for **C-119**'s SPE5 work there are **two distinct goals** and the protocol conflates them: | (a) **Reproducibility** — same input $\\Rightarrow$ same output, every run. ✅ Delivered. **(b) **Comparability** — our geological realisation matches the reference. ❌ Not achievable by PRNG choice. ✅ **Ruled: SPE5 comparison must use the SPE-supplied realisation, not a regenerated one**, and the two goals must be named separately in the V&V framework — otherwise a legitimate PRNG change will be misdiagnosed as solver error. ⚠️ Also: **tolerance-based stopping makes the iteration count data-dependent**, so reproducibility depends on the reduction order being **strictly index-derived and never history-derived** — including inside the solver, where the adaptive AMG hierarchy of **C-95** is history-dependent by construction | **new, C-119** |

### Register status after Ruling 18

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · **124 · 126 · 129** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **C-134 is the second time this round that measurement corrected my expectation rather than
> confirming it** (the first being `feos-ad`). I predicted Kahan would fix accuracy but not
> reproducibility; measured, it fixed **both**. 📌 **Recording the prediction, not just the result, is
> what makes the next one cheaper** — and the pattern across Rulings 5–18 is that my priors are useful for
> *where to look* and unreliable for *what the answer is*.
>
> ⚠️ **C-132 shows the partition defect was not a one-off arithmetic slip but a structural gap.** The
> fix closed the case I had found and left an adjacent one — which is the signature of a **missing
> exhaustive-case argument** rather than a typo. ✅ **Ruled: the gate must prove exhaustiveness over the
> full cross-product** $\\{$complete, incomplete$\\}\\times\\{$failure, no-failure$\\}$, not spot-check
> the case that was reported.

---

## Ruling 19 — 2×2 status matrix, atomic checkpoint, two-tier reduction, goal split (08-10-2026)

Corrections **C-136 … C-139**. **Three closed; one closed with two refinements; one action item
created.** Detail: [`engine_invariants.md`](engine_invariants.md) §7q.

| # | Correction | Detail |
|---|---|---|
| **C-136** | ✅ **C-132 CLOSED — the 2×2 matrix IS a partition.** Verified all four cells against the four classes, no overlap and no gap: · **Complete + No-Failure** $\\Rightarrow$ Tier 1 (all `Implemented` + in-envelope) **or** Tier 2 (`Degraded` **or** out-of-envelope) — exhaustive, since either $\\forall$`Implemented` $\\wedge$ in-envelope, or $\\neg$ that, which is $\\exists$`Degraded` $\\vee$ out-of-envelope · **Complete + Failure** $\\Rightarrow$ Tier 3 · **Incomplete + No-Failure** $\\Rightarrow$ **Tier 4 `Aborted`** (new) · **Incomplete + Failure** $\\Rightarrow$ Tier 3. ✅ **Artifact rule correct and consistent** with **C-133**: on `Aborted`, `run_name.out.h5` is **never committed** (no partial-data leakage) while the autonomous `run_name.ckpt.h5` **is** retained for recovery. ✅ Exactly the decoupling the round before required | **closes C-132** |
| **C-137** | ⚠️ **C-133 ACCEPTED with two refinements.** The protocol — `.tmp` write, BLAKE3 `CommitToken` last, `fsync()`, atomic POSIX `rename()`, verify-on-load — is sound. 🔴 **Gap 1: no directory `fsync()` after the rename.** POSIX durability requires fsyncing the **parent directory** after `rename()`, or the rename itself may not survive a crash (classic ext4/XFS rule). `fsync()` on the temp file makes the *contents* durable but not the *directory entry*. ✅ **Ruled:** `fsync(tmp)` → `rename()` → **`fsync(dirfd)`**. ⚠️ **Gap 2: `unlink()` on token mismatch destroys the evidence.** Silent deletion makes post-mortem impossible, which contradicts this project's own audit discipline. ✅ **Ruled: quarantine to `run_name.ckpt.h5.corrupt`**, never `unlink`. ⚠️ Also: *"token written last"* and *"rename only after a complete write"* are **redundant** as atomicity guarantees — `rename` is what provides atomicity; the token detects later bit-rot. ✅ **Ruled: name which clause is load-bearing** so the protocol is not "simplified" into a hole later | **closes C-133** |
| **C-138** | ✅ **C-134 CLOSED — the two-tier designation is accepted verbatim** and is the right documentation: *"Порядок редукції за статичними індексами комірок є несучим елементом детермінізму; підсумовування за Каханом є несучим елементом числової точності."* ✅ Fixed index order = **determinism guarantee**; Kahan = **accuracy defence**. Both retained, roles distinguished, matching the measurement (**naive order-dependent**, Kahan bit-identical, `fsum` error **0.0**) | **closes C-134** |
| **C-139** | ✅ **C-135 ACCEPTED**, and the AMG clause is the part that matters most: coarse-grid construction must use a **static adjacency graph with deterministic index-based tie-breaking**, excluding **Rayon traversal order and memory addresses**. ✅ That is the classic hidden nondeterminism in adaptive AMG, and naming memory addresses as an exclusion is exactly right. ✅ FGMRES inner products and $\\lVert R\\rVert_2$ routed through the fixed tree ✅ — so the tolerance-based stopping criterion of **C-134** is now deterministic. 🔴 **New ACTION ITEM:** SPE5/SPE10 tests must import the **official** grid and permeability files (`SPE5_PERM.GRDECL` / `.HDF5`), not regenerate them. ⚠️ **The CMG deck found in `D:\\RAG` (C-119) is NOT the official specification** — it is a CMG-format transcription, explicitly a *starting point*. **Acquisition required** — see [`literature_todo.md`](literature_todo.md) | **closes C-135; opens L-4** |

### Register status after Ruling 19

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · **124 · 126 · 129 · 132 · 133 · 134 · 135** |
|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 23 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 55 · 57 · 60 · 61 · 62 |

> 📌 **C-136 is the first closure in fourteen rulings where the fix was verified against the *whole*
> partition rather than the reported case** — and it held. That is the direct dividend of ruling **C-132**'s
> gate requirement: *"prove exhaustiveness over the full cross-product, do not spot-check."* The
> requirement produced a correct answer on the first attempt.
>
> ⚠️ **And C-137 is the same shape one level down.** The protocol was verified against the *reported*
> failure (torn write) and was correct — but not against **crash-during-rename**, which is a different
> moment. 📌 **A protocol verified against one hazard has been verified against one hazard.** The durable
> check is to enumerate the failure *instants*, not the failure *modes*.

---

## Ruling 20 — CONF-23 three-tier tolerances, CONF-55 remediation plan (08-10-2026)

Corrections **C-140 … C-142**. **Both reds closed**; three refinements. Detail:
[`engine_invariants.md`](engine_invariants.md) §7r.

| # | Correction | Detail |
|---|---|---|
| **C-140** | ✅ **CONF-23 CLOSED — the 3-tier architecture is correct and is the right resolution.** Tier A absolute per-component residual $\|\\mathbf{R}_{m,i}\\|_\\infty < 10^{-12}$ (D5 §3.1) · Tier B remap $\\|\\sum M_{FVM} - \\sum M_{SL}\\| < 10^{-14}$ (D5 §5.3) · Tier C relative global closure $\\ge 99.9\\%$ (D2 §6.2, D6 §4.3) ✅ **three genuinely different quantities**, correctly separated. ✅ Renaming $10^{-14}$ to `RemapConservationPrecision` is exactly right — it is operator precision, not a convergence tolerance. ⚠️ **Two refinements on the closure expression:** **(a)** 🔴 **the index $i$ is overloaded** — $M_{i,\\text{remaining}}$ and $M_{i,\\text{initial}}$ index **component**, while $W_i$, $G_i$ index **well**, and $W_p,G_p,N_p$ carry **no component index at all**. The balance is component-total vs component-total only if the production/injection terms are component-resolved. **Use $c$ for component and $w$ for well**, and state the summation explicitly — this is the register's recurring *notation collision* class. **(b)** 🔴 **an absolute $10^{-12}$ **kg/s** tolerance is NOT invariant under timestep refinement.** Measured: the same net flux gives a mass-per-step residual $R = q\\,dt$, so a fixed $10^{-12}$ kg/s bound is a $10^{-12}$ kg tolerance at $dt=1$ and a **$10^{-8}$ kg** tolerance at $dt=10^{-4}$ — **four orders looser**. With **C-34** soft-start and adaptive $dt$, one tolerance is a different strictness at every step. **Ruled: normalise by the local mass rate (dimensionless relative residual), or express in kg per timestep** | **closes CONF-23** |
| **C-141** | ⚠️ **Exporter ruling accepted — with a status-semantics collision.** ✅ Standardise on **99.9 %**; **[99.0 %, 99.9 %)** $\\Rightarrow$ `ValidityWarning::MaterialBalanceSuboptimal`; **< 99.0 %** $\\Rightarrow$ quarantine from ML training ✅ — the quarantine point is exactly right and consistent with **C-51/C-55**'s atomic-commit rule. 🔴 **But Tier 3 from the exporter is not Tier 3 from the solver.** The solver's Tier 3 means *"the numerics could not produce an answer"* (**INV-1**); the exporter's means *"the run solved, and the post-hoc audit found the closure inadequate."* **Two different events sharing one label**, and the manifest cannot distinguish them — which weakens the **C-136** lattice just closed, because a reader of `ValidityClass` cannot tell a solver failure from an audit failure. ✅ **Ruled: distinguish the source** — e.g. `Unsolvable{solver}` vs `Unsolvable{post_hoc_audit}` (or an `audit_disposition` field), and keep the training-set quarantine in **both** cases | **CONF-23 closed, semantics amended** |
| **C-142** | ✅ **CONF-55 CLOSED — all three parts accepted.** **(A)** ✅ **`N_p/15$ Ghost Finding formally withdrawn** — the cleanest close in the register: a finding of mine was audited, found to have cited the wrong lines, and **withdrawn on evidence** rather than defended. **(B)** ✅ **D8 §4.2 reclassified as a Remediation Specification**, all criteria in the subjunctive, with explicit **Gate 1** (raw physical state vectors, no hardcoded cost/penalty overrides) and **Gate 2** (unified 99.9 % validation, matching assertion bounds). **(C)** ✅ **Dual-track `FAILURE_PENALTY` accepted** — removed by construction in Rust (**INV-7**), retained as a legacy artifact in the Python audit scope. 🔴 **One factual error:** *"...until the Python codebase is fully decommissioned in Milestone M6."* 🔴 **M6 is CO₂-EOR specifics (Domain 6, 5-state trapping)** — verified against [`build_plan.md`](build_plan.md) — and the string **"decommission" appears nowhere in the compositional section.** Owner decision 1 is **"Leave Python as is"** and the engines **complement** at P3. ✅ **Ruled: strike the M6 decommission reference**; `FAILURE_PENALTY` stays in the **Python audit** with no scheduled end date | **closes CONF-55** |

### Register status after Ruling 20

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 |
|---|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **All eight blocking (🔴) conflicts are now closed** — CONF-03 · 11 · 19 · 23 · 43 · 55 · 58 · 59.
> Every remaining item is 🟠 material or 🟡 documentary, and none is a red.
>
> ✅ **Two of this round's closures came from things I had to check first.** CONF-23's live-repo claim
> was **wrong in two places** (README cites no tolerance at all; the enforced number is **99.5 %** in the
> exporter, not 99.9 %, and the engine core enforces **neither**). CONF-55's line reference pointed at the
> **NPV block**, not an averaging divisor. 📌 **Neither would have been found by re-reading the docs** —
> only by measuring the repo, which is what the register's `Status:` convention has been driving at all.
>
> ⚠️ **And the engineer's response to (A) is worth recording as a process note.** Given evidence that a
> finding of mine was wrong, the correct move was to **withdraw it immediately and reclassify it as a Ghost
> Finding** — not to reinterpret it, hedge it, or keep it "provisionally open". That is the behaviour that
> makes a review loop worth running.

---

## Ruling 21 — Tier A normalization, component-resolved closure, typed Unsolvable (08-10-2026)

Corrections **C-143 … C-146**. **All four accepted; three carry required conditions.**
Detail: [`engine_invariants.md`](engine_invariants.md) §7s.

| # | Correction | Detail |
|---|---|---|
| **C-143** | ⚠️ **$\\Delta t$-invariance fix ACCEPTED in principle — 🔴 but the local-throughput normalizer makes convergence UNREACHABLE in exactly the cells that matter.** ✅ Normalising is right and kills the $\\Delta t$ dependence (**C-140b**). 🔴 **But the denominator $\\sum_f\\lvert\\mathbf{F}\\rvert + M_{c,k}/\\Delta t$ vanishes in depleted cells and dead zones.** Measured, at $R = 10^{-15}$ kg/s: · active producer cell (denom 5.0e1) $\\Rightarrow$ ratio **2.0e−17** ✅ · nearly-depleted (denom 5.0e−3) $\\Rightarrow$ ratio **2.0e−13** ✅ · **depleted / dead zone (denom 1.0e−9)** $\\Rightarrow$ ratio **1.0e−6** 🔴 — **six orders above the $10^{-12}$ threshold**, and at a true dead zone the denominator is **0**, so the ratio is **undefined**. Meeting $10^{-12}$ relative in a depleted cell demands an absolute residual of **$10^{-21}$ kg/s**. 🔴 **Under INV-7 (unconstrained) depleted cells and dead zones are COMMON, not exceptional** — absurd rates create them deliberately. **This is the same defect class as C-69 (the $\\phi_0$ denominator) and C-102 (the $\\alpha_0$ zero crossing): a normalisation whose denominator can vanish.** ✅ **Ruled — a mixed criterion, never a bare ratio:** accept convergence when `R_abs < tol_abs` **OR** `R_rel < 1e-12`, with `tol_abs` a **declared global floor** (e.g. $10^{-14}$ kg/s per cell-component) **and the dead-zone case routed to the absolute branch by construction** (`denom <= 0` $\\Rightarrow$ absolute test only). ⚠️ **And the residual form must be DECLARED as a RATE** ($\\mathrm{kg/s}$): the ratio is dimensionless **only** if $\\mathbf{R} = \\sum_f\\mathbf{F} - \\mathrm{d}M/\\mathrm{d}t$. With a mass-form residual the ratio has units of seconds | **C-140b closed with condition** |
| **C-144** | ✅ **Component/well index disambiguation ACCEPTED** — $\\mathrm{MB}_{closure,c}$ with $c\\in[1,N_c]$, $w\\in[1,N_w]$, and $\\sum_w Q^{cum}_{p,c,w}$, $\\sum_w Q^{cum}_{i,c,w}$ ✅ **removes the overload cleanly** and is strictly *better* than a total-only check, because a dominant component can no longer mask a trace component's error. ⚠️ **Two conditions:** **(a) the condition basis must be declared** — $M_{c,\\text{total}}$ is a **reservoir** quantity and $Q^{cum}_{p,c,w}$ is naturally a **surface** quantity; mixing them without an explicit $B$-factor / density conversion manufactures a closure error that looks like a mass-balance failure. This is the **same gap CONF-15 retains** (unit basis for $q$). **(b)** ⚠️ **per-component $99.9\\,\\%$ may be unattainable for trace components** — a component at $10^{-3}$ mole fraction carries so little mass that its closure sits near solver tolerance, so the test fails on arithmetic rather than physics. **Ruled: apply the strict per-component test to components above a declared mass fraction, and a mass-weighted aggregate to the trace remainder**, with the threshold and the fraction both in the manifest | **C-140c closed with condition** |
| **C-145** | ✅ **C-141 typed-source split ACCEPTED exactly as ruled** — `UnsolvableSource::{SolverFailure{error}, PostHocAuditFailure{metric,value,threshold}}` and `ValidityClass::{Validated, ConvergedOutsideEnvelope(Vec<ValidityWarning>), Unsolvable(UnsolvableSource), Aborted}`. ✅ Correctly preserves provenance; ✅ **both variants quarantine**, as required; ✅ `Vec<ValidityWarning>` is consistent with **C-128**'s requirement that warnings be *distinguishable kinds* rather than a string list | **closes C-141** |
| **C-146** | ✅ **CONF-55 fully confirmed and locked** — withdrawal locked, **M6 decommission struck**, `FAILURE_PENALTY` retained **indefinitely** in the Python audit scope with no sunset date, Python remaining active for high-level proxy experiments, and the Rust core staying strictly unconstrained under **INV-7** with zero artificial penalties. ✅ Consistent with owner decision 1 (*"Leave Python as is"*) and with the engines **complementing** at P3 | **CONF-55 locked** |

### Register status after Ruling 21

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 |
|---|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 (unit basis — now also gates **C-144**) · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **C-143 is the FIFTH instance of one defect class, and that is the finding.** · **C-69** Verma-Pruess $\\phi_0 - \\phi_c$ denominator · **C-102** $\\alpha_0$ fit crossing zero at $\\theta = 43.5^\\circ$ · **C-115** the register's `(f64,f64)` range type · **C-134** history-dependent AMG coarsening · **C-143** the Tier A throughput normalizer. 📌 **In every case a quantity that can approach zero was used as a divisor or a comparator with nothing else governing the limit.**
>
> ✅ **The durable rule this produces:** *no convergence criterion, correlation or normalised ratio may rely on a denominator without an **explicit branch for its zero**.* Recorded once, and cited by every future occurrence. It is the single highest-transfer item in twenty-one rulings — because it has now predicted five defects *after the fact*, in four different documents, with no shared author.
>
> ⚠️ **And a second pattern, inverse:** each fix has been *directionally correct and locally incomplete*. Normalising **is** right; petekIO **was** memory; the overall-composition formulation **is** better. The recurring gap is not wrong intuition — it is **the limit case nobody wrote down**.

---

## Ruling 22 — rate-form residual, dual-branch Tier A gate (08-10-2026)

Corrections **C-147 … C-149**. **The gate is accepted and verified as sound**, with three precision
additions. Detail: [`engine_invariants.md`](engine_invariants.md) §7t.

| # | Correction | Detail |
|---|---|---|
| **C-147** | ✅ **C-143 rate-form residual ACCEPTED — and the dimensional anomaly is closed.** $\\mathbf{R}_{c,k} = \\sum_f \\mathbf{F}_{c,k,f} + \\dfrac{M^{n+1}_{c,k}-M^n_{c,k}}{\\Delta t} - Q_{c,k}$ [kg/s] ✅ sign convention self-consistent ($\\mathbf{R}=0 \\Rightarrow \\mathrm{d}M/\\mathrm{d}t = Q - \\sum_f \\mathbf{F}$, i.e. $\\mathbf{F}$ net **outflow**, $Q$ net **injection**) ✅ and using $\\lvert\\mathbf{F}\\rvert$ in the denominator while $\\mathbf{F}$ stays **signed** in the residual is correct — magnitude for scale, sign for error. ✅ Ratio now strictly dimensionless | **C-143(a) closed** |
| **C-148** | ✅ **C-143 dual-branch gate ACCEPTED and VERIFIED.** Measured: the `OR` yields an effective tolerance $\\mathbf{R} < \\max(tol_{abs},\\ 10^{-12}\\cdot denom)$, and the branch switch lands exactly where intended — **relative governs above ~$10^{-4}$ kg/s**, **absolute below**. ✅ The Rust short-circuit is correct: `denom > denom_min` is tested **before** the division, so **no division by zero is possible**. ✅ It resolves the exact case I measured — at `denom = 1e-9`, `R = 1e-15` now **passes via the absolute branch** where the bare ratio gave 1.0e−6 and failed. ⚠️ **Three precision additions:** **(a)** `denom_min = 1e-11` is a **safety guard, not an accuracy knob** — measured, $10^{-11}$ divides fine in `f64`; it exists **only** to stop `denom = 0`. ✅ **Record it as such so nobody "tunes" it.** **(b)** The routed-off zone is **loose by design**: at `denom = 1e-12` the gate implies a **1.0 %** local relative error. ✅ **And it is bounded** — measured, if **every** one of 1.1 M cells sat at its worst, the aggregate is 1.1e−8 kg/s $\\Rightarrow$ **0.35 kg over a year** against a $10^{9}$ kg reservoir, i.e. **3.5e−10 relative**. **Tier A's local looseness is covered by Tier C's global closure — the two-tier design does its job.** **(c)** ⚠️ **No `is_finite()` guard.** Measured: with `R = NaN` the gate returns `false`, so the solver iterates to max-iterations and then errors — **safe but confusing**. Given three NaN generators already found (**C-69**, **C-102**, **C-143**), ✅ **add `if !r_abs.is_finite() \\|\\| !denom.is_finite() { return Err(TypedError::NonFiniteResidual) }`** | **C-143(b) closed** |
| **C-149** | ⚠️ **$\\Delta t$-invariance is PARTIAL, and the claim should be narrowed.** ✅ The **flux** term $\\sum_f\\lvert\\mathbf{F}\\rvert$ is $\\Delta t$-independent ✅ — the original defect is fixed for throughput-dominated cells. 🔴 **But the storage term $M^{n+1}/\\Delta t$ is inversely proportional to $\\Delta t$**, so the denominator **grows as $\\Delta t$ shrinks** and the criterion becomes **looser** at small steps in **storage-dominated** cells (shut-in, depletion). ✅ **Ruled: narrow the claim to what is true** — *"$\\Delta t$-invariant in throughput-dominated cells; storage-dominated cells carry an explicit $1/\\Delta t$ scaling."* ⚠️ This is a **design choice, not a defect** — loosening at small $\\Delta t$ on a storage-dominated cell is physically reasonable — but it must be **stated**, or a later reader will re-open it as an unfixed **C-140b** | **C-143 closed with narrowing** |

### Register status after Ruling 22

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 · **143** |
|---|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 (unit basis — also gates **C-144**) · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **The zero-denominator rule (§7s.5) predicted this one.** C-143 **was** that rule firing — the
> finding was raised against the proposal before the code was written, the fix was built with an explicit
> zero branch (`denom_min`), and the short-circuit was verified to make division by zero **impossible**.
> 📌 **Two consecutive findings now derived from a single recorded rule, in different subsystems
> (clogging/perforation → convergence gating), with no shared author.**
>
> ✅ **And the two-tier coverage result is the substantive one.** The dual-branch gate is loose by design
> in dead cells — up to **1 % local** — and measured, the worst case across 1.1 M cells contributes
> **3.5 × 10⁻¹⁰** of reservoir mass per year. **Tier A's local looseness is bounded by Tier C's global
> closure.** That is the architecture working as designed, and it is worth stating explicitly because the
> gate would otherwise look like a weakening.

---

## Ruling 23 — precision additions locked; one INV-1 boundary found (08-10-2026)

Corrections **C-150 … C-152**. **Two accepted and locked; one INV-1 boundary found; one of my own
caveats refined as too broad.** Detail: [`engine_invariants.md`](engine_invariants.md) §7u.

| # | Correction | Detail |
|---|---|---|
| **C-150** | ✅ **C-148(a) ACCEPTED and LOCKED** — `denom_min = 1e-11` documented as a safety guard with `#[doc = "SAFETY GUARD against div-by-zero; DO NOT TUNE"]` ✅ correct call: tuning it would move the active/dead-zone boundary and silently change which branch governs. ⚠️ **One factual overreach to correct:** the justification *"or numbers approaching $f64$ subnormal limits"* does **not** apply. $f64$ min-normal is $\\approx 2.2\\times10^{-308}$, so $10^{-11}$ sits **297 orders of magnitude above** it — there is no underflow or subnormal risk at this value. ✅ **Div-by-zero is the only real reason the guard exists.** Harmless to the design, but recorded accurately so nobody later "generalises" the reasoning to a value where it would matter | **C-148(a) locked** |
| **C-151** | 🔴 **🔴 `NonFiniteResidual` + "adaptive $\\Delta t$ cutback" / "fallback solver routines" CONFLICTS WITH INV-1 — must be split.** A `NaN` originating in a **failed flash** is a **thermodynamic failure**, and **INV-1** is explicit: *"Flash failure $\\Rightarrow$ return `ThermodynamicFlashFailed` and **stop**"*, with *"substitute a single-phase guess"* listed as **forbidden**. The ruling's stated impact — *"triggering immediate adaptive time-step cutback ($\\Delta t \\to \\Delta t/2$) **or fallback solver routines**"* — is **precisely the silent-retry pattern INV-1 prohibits**. ✅ **Ruled — two distinct error variants, and the boundary is the *kind* of failure, not its size:** · **`TypedError::NonFiniteResidual`** (NaN/Inf residual or denominator) $\\Rightarrow$ **hard stop, INV-1.** A NaN means the thermodynamics or the algebra is broken, and halving $\\Delta t$ does not repair a broken flash. · **`NewtonDivergence`** (residuals **finite**, merely above tolerance) $\\Rightarrow$ $\\Delta t$ cutback **is** legitimate — it is standard adaptive stepping, not a fallback. 🔴 **But it must not be silent:** the manifest records **`cutback_count`** and **`total_cutback_time`**, and a run that converged **only after** cutbacks is surfaced. ✅ **And "fallback solver routines" is struck outright** — there is no fallback solver under INV-1, in any circumstance | **new boundary** |
| **C-152** | ✅⚠️ **C-149 REFINED — my earlier caveat was TOO BROAD, and the precise statement is narrower.** I wrote that the storage term $\\propto 1/\\Delta t$ breaks invariance in storage-dominated cells. 🔴 **Measured, that is wrong in the common case.** · **Pure storage** (shut-in, zero flux, zero $Q$): $R = \\mathrm{d}M/\\mathrm{d}t$ and $\\mathrm{denom} = M^{n+1}/\\Delta t$ — the $1/\\Delta t$ appears in **both**, so the ratio is $(M_1-M_0)/M_1$ and **cancels exactly**. Verified invariant across $\\Delta t \\in [10^0,\\ 10^{-8}]$ (ratio constant at 1.000000e−09). · **Non-invariance occurs only in the MIXED case**, where $R$ and $\\mathrm{denom}$ are dominated by **different** terms — measured, flux-dominated $R$ with storage-dominated $\\mathrm{denom}$ improves **4 orders** as $\\Delta t$ shrinks 4 orders. ✅ **Corrected specification:** *"the criterion is $\\Delta t$-invariant whenever $\\mathbf{R}$ and the denominator share a dominant term; it is scale-dependent only in the mixed regime."* ✅ **And a useful corollary:** a shut-in cell at convergence has $\\mathrm{d}M \\to 0 \\Rightarrow R \\to 0$, so it fires the **absolute branch at iteration 1** and **never reaches the relative branch** — no cutback loop, no stalling | **C-149 refined** |

### Register status after Ruling 23

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 · 143 |
|---|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 (unit basis — also gates **C-144**) · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 🔴 **C-151 is the second time a precision addition has collided with INV-1.** The first was **C-93**,
> the `clamp(r_D, …)` guard on the Karakas-Tariq regression domain — an *arithmetic* substitution where
> **C-151** is a *recovery* substitution. 📌 **But the underlying rule is one rule:** a correction that
> makes a failure *disappear* rather than *reported* is forbidden, whether it hides behind a `clamp`, a
> halving $\\Delta t$, or an alternate solver. ✅ **The distinguishing question is always the same: *after
> this, can a reader tell what happened?*** A $\\Delta t$ cutback that increments a recorded counter and
> appears in the manifest **passes**; one that quietly retries until it succeeds **fails**.
>
> ⚠️ **And C-152 is the second time this round a measurement corrected my own caveat** (the first being
> the Kahan result in **C-134**). I predicted the storage term broke invariance; measured, it cancels in
> the case I had in mind. 📌 **The failure mode is consistent: I generalise from a mechanism to a rule
> without checking whether the mechanism actually propagates.** That is the same shape as the very error
> this whole register was opened to document — which is the argument for measuring rather than reasoning.

---

## Ruling 24 — C-150/151/152 accepted; cutback trigger re-opens the C-136 partition (08-10-2026)

Corrections **C-153**. **Three accepted exactly as ruled; one consequence found.**
Detail: [`engine_invariants.md`](engine_invariants.md) §7v.

| # | Correction | Detail |
|---|---|---|
| **C-153** | ✅ **C-150, C-151, C-152 all ACCEPTED as restated — correct in every detail.** ✅ **Fallback solvers struck from the architecture entirely** ✅ — there is exactly one physical PDE solver, and that is the strongest possible form of the ruling. ✅ The $10^{-11}$ rationale now records div-by-zero only, with the $2.22\\times10^{-308}$ min-normal cited ✅. ✅ The $\\Delta t$-cancellation proof and its confinement to the mixed regime are correctly stated ✅. 🔴 **But one consequence was not carried through: "a run completing after cutbacks is surfaced as `ConvergedOutsideEnvelope`" RE-OPENS the C-136 partition.** Measured: `completed = TRUE`, all modules `Implemented`, all params **inside** envelope, `cutback_count = 3` $\\Rightarrow$ **Tier 1 matches AND Tier 2 matches.** A partition requires exactly one. Adding a third disjunct to Tier 2 without excluding it from Tier 1 creates an **overlap**, and it is the *same* class of defect as **C-128** — a hole closed by one fix, reintroduced by the next. ✅ **Ruled — two changes:** **(a)** 🔴 **`cutback` is a WARNING, not a Tier condition** — `ValidityWarning::NewtonCutback { count: u32, total_cutback_time: f64 }`. It belongs in the same category as `EnvelopeExceeded`, not in the Tier predicate, because a cutback is **not** an envelope violation — it is the solver handling a physical non-linearity. **(b)** ✅ **Tier 2 := completed $\\wedge$ $\\neg$Tier 1** — the **complement** rule from **C-128**, which is what closed the original hole. ✅ Verified: Tier 1 fires, Tier 2 does not, exactly one match. **Fix A** (also excluding `cutback_count == 0` from Tier 1) restores exclusivity but leaves two predicates that must be kept in sync by hand — **Fix B is the same rule that already solved this problem once, so reuse it** | **partition re-secured** |

### Register status after Ruling 24

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 · 143 |
|---|---|---|
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) · **CONF-51** · **CONF-66** |
| **Fix specified** | **CONF-15** 🟡 (unit basis — also gates **C-144**) · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 17 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **C-153 is the second time the C-128/C-136 partition has been re-broken by an adjacent change** —
> first by `Aborted` (**C-132**), now by `cutback`. 📌 **That recurrence is itself the finding:** a
> partition maintained by *listing conditions* degrades every time a new condition arrives. A partition
> maintained by *one predicate plus its complement* cannot. ✅ **The durable change is therefore
> structural, not another clause:** **`ValidityClass` must derive from a single `Tier1Predicate` plus
> `NOT`, never from an independently-maintained condition list.**
>
> ⚠️ **And the category point, which is the substantive half.** A cutback is **not** an envelope
> violation — it is the solver **handling** a physical non-linearity. Putting it in the Tier predicate
> conflates *"your input is unverified"* with *"the solver worked hard"*, which is precisely the
> conflation **C-128** was raised to remove (`ModuleAbsent` vs `EnvelopeExceeded`) — and the same one
> **C-141** was raised to remove (solver failure vs audit failure). 📌 **Three rulings, one recurring
> error: putting two different *reasons* into one *status*.** The `ValidityWarning` enum is the correct
> home for reasons; `ValidityClass` should carry only the verdict.

---

## Ruling 25 — CONF-51 dimensional repairs, CONF-66 two-phase hardening (08-10-2026)

Corrections **C-154 … C-158**. **Both dimensional repairs are correct; CONF-17 closed; four residual
defects.** Detail: [`engine_invariants.md`](engine_invariants.md) §7w.

| # | Correction | Detail |
|---|---|---|
| **C-154** | ✅ **CONF-51 dimensional repair VERIFIED CORRECT.** The $q_{2D}=Q_{inj}/H_f$ [m²/s] substitution is the right diagnosis, and the fix closes: $[E'^3\\mu'q_{2D}]^{1/4} = [\\text{Pa}^3\\cdot\\text{Pa·s}\\cdot\\text{m}^2/\\text{s}]^{1/4} = [\\text{Pa}^4\\text{m}^2]^{1/4} = \\text{Pa·m}^{1/2}$, matching $[K_{IC}] = \\text{Pa·m}^{1/2}$ $\\Rightarrow$ $\\mathcal{K}$ **strictly dimensionless** ✅. ✅ **And the $(1-\\nu^2)^{3/4}$ placement is self-consistent** — verified by expanding $E'^3 = E^3/(1-\\nu^2)^3$ into the denominator, which reproduces exactly the submitted numerator. ✅ Both $w_0$ forms **close in metres**: Form 1 $\\Rightarrow$ m ✅, Form 2 $\\Rightarrow[\\text{Pa}^2\\text{m}\\cdot\\text{m}^2]^{1/3} = \\text{m}$ ✅. ✅ **Model-specific viscosity routing** ($\\mu^{1/4}$ PKN/penny, $\\mu^{1/6}$ KGD) **correct** ✅. ✅ **Krieger-Dougherty** $\\mu_s=\\mu_f(1-C_{prop}/\\phi_m)^{-[\\eta]\\phi_m}$, $[\\eta]=2.5$, $\\phi_m=0.64 \\Rightarrow \\beta = \\mathbf{-1.60}$ ✅ **correct**. ✅ **Bandis-Lumsden-Barton (1983)** DOI independently verified last round. 🔴 **Two residuals:** **(a)** 🔴 **the two $w_0$ forms are not independent** — under the volume balance $q_{2D}t = 2L_fw_0$, Form 2 gives $w_0^2 = 2K_{IC}^2(1-\\nu^2)^2L_f/E^2$ while Form 1 gives $\\mathcal{C}_K^2\\times$(the same). **They agree only if $\\mathcal{C}_K = \\sqrt{2} = 1.4142$**, so $\\mathcal{C}_K$ **is not a free constant — Form 2 fixes it.** **(b)** ⚠️ **the Type-I/Type-II switch is discontinuous** — $H_f$ grows during propagation, so $\\mathcal{K}$ can cross 1 mid-growth, and the two $w_0$ formulas do **not** agree at $\\mathcal{K}=1$. ✅ **Ruled: lock $\\mathcal{C}_K=\\sqrt2$ and use a **blended** transition over $\\mathcal{K}\\in[0.8,1.25]$**, not a hard switch | **CONF-51 residuals** |
| **C-155** | 🔴 **CONF-66 B — two-phase law fixes the hardening gap ✅ BUT it is $C^0$, not $C^1$, at the peak.** Pre-peak parabola verified: $c(0)=c_{yield}$ ✅, $c(\\bar\\varepsilon_p^{peak})=c_{peak}$ ✅, $H>0$ strictly for $x<1$ ✅ — matches the stated modulus exactly. Post-peak exponential $\\to c_{res}$ ✅. 🔴 **Measured at the switch:** $H$ jumps from $\\approx 0$ (pre-peak) to $-\\eta(c_{peak}-c_{res}) = -4.5\\times10^8$ (post-peak). **The consistent tangent therefore jumps exactly at the hardening→softening transition**, costing the asymptotic quadratic rate at that instant. ⚠️ **Declare it**, and average the tangent over the active set at the switch (standard practice, **C-145**'s `Vector<ValidityWarning>` pattern applies: the switch is an event, not a silent kink). 🔴 **Separately, the Drucker-Prager parameterisation has a convention mismatch:** the submitted $F = \\sqrt{J_2} + \\alpha I_1 - k$ uses $\\sqrt{J_2}$, while the standard $k = 6c\\cos\\phi/\\big[\\sqrt3(3\\mp\\sin\\phi)\\big]$, $\\alpha = 2\\sin\\phi/\\big[\\sqrt3(3\\mp\\sin\\phi)\\big]$ pair **assumes** $F = \\sqrt{J_2/3} + \\alpha I_1 - k$. Measured at $c=1$ MPa, $\\phi=30°$: $k = 1.200\\times10^6$ with the $\\sqrt3$, $2.078\\times10^6$ without — **ratio exactly $\\sqrt3 = 1.7321$**. ✅ **Ruled: either write $F = \\sqrt{J_2/3} + \\alpha I_1 - k$, or drop the $\\sqrt3$ from $k$ and $\\alpha$.** ⚠️ **Also declare** that hardening drives **both** $k$ and $\\alpha$ from one scalar $\\bar\\varepsilon_p$ — standard isotropic DP hardening scales $k$ **alone** with $\\alpha$ fixed; varying both is a non-proportional law and should be named as such | **CONF-66 B conditional** |
| **C-156** | ⚠️ **CONF-66 C — Schur condensation ELIMINATED ✅, and the ill-conditioning argument is correct** ($E/(1-\\nu^2)$ vs $c_t$ contrast propagating into $\\mathbf{S}_p$). ✅ **Block ILU(1) + FGMRES on the 2×2 THMC block is the right answer.** ⚠️ **Two residuals:** **(a)** 🔴 **Cholesky on $\\mathbf{K}_{uu}$ is valid only while the displacement block is ELASTIC.** With active plasticity $\\mathbf{K}_{uu}$ contains $D^{ep}$ and is **non-symmetric**, so the preconditioner must switch to LU. ✅ **Ruled: preconditioner selection is conditioned on whether any plastic set is active.** **(b)** 🔴 **§C prohibits splitting $\\mathbf{A}_{pp}$ by pressure/transport, then prescribes *CPR-AMG* on that same block — which IS that split.** This is the **third** resurrection of **C-79**. ✅ **Ruled: $\\mathbf{A}_{pp}$ takes a single coupled preconditioner (Block ILU(1) or unsymmetric AMG); no CPR inside it** | **CONF-66 C conditional** |
| **C-157** | ⚠️ **CPPM citations improved and plausible.** *Computational Inelasticity* (Simo & Hughes, Springer 1998) ✅ is real and canonical; Simo & Taylor (1985/86) and Abbo & Sloan (1995) ✅ plausible. ✅ **Terminology corrected** — $\\mathbf{D}^{alg} = \\partial\\boldsymbol\\sigma_{n+1}/\\partial\\boldsymbol\\varepsilon_{n+1}$ distinguished from the continuum $\\mathbf{D}^{ep}$ ✅ **is the right distinction** and matters for Newton. ⚠️ **All three need DOIs before M7b** per the M1 precedent (**literature_todo.md** §1 test 1) | **book refs, unverified DOI** |
| **C-158** | ✅ **CONF-17 CLOSED.** Cell-local $P$ and $z_i$ as **sole** coupling inputs; $\\bar{P}_{res,\\text{eff}}$ demoted to a post-processed diagnostic; **zero-feedback constraint** stated explicitly — the strongest form, since it forbids reuse rather than merely demoting. ⚠️ **One addition carried from the briefing:** **RF cannot be cell-local.** D2 §6.1 drove **both** miscibility *and* RF from the scalar. Miscibility correctly becomes a field; **RF's definition must change** to a volume integral of local quantities, $\\mathrm{RF} = \\int_\\Omega\\Phi_{prod}/\\int_\\Omega\\Phi_{init}$ — otherwise RF is accidentally made local too | **closes CONF-17** |

### Register status after Ruling 25

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · **23** · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · **17** · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 · 143 |
|---|---|
| **Conditional** | **CONF-51** (C-154: $\\mathcal{C}_K$, blended transition) · **CONF-66** (C-155: $C^0$ switch, $\\sqrt3$ convention; C-156: Cholesky scope, CPR contradiction) |
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **The dimensional gate is now formally adopted (C-154).** Dimensional closure has been the most-failed
> check in this entire exchange — **seven instances** (C-83 $\\eta_{erosion}$, C-88 Penéloux sign, C-106
> $\\sqrt{\\bar k}$, C-122 $\\varepsilon_{chem}$, C-140 $\\Delta t$ units, C-151 Type-II $\\mathcal{K}$,
> and now the DP $\\sqrt3$ convention). ✅ **A `#[cfg(test)]` unit gate over the typed newtypes
> (C-124/C-131) would have caught all seven before review.** 📌 **It is the single highest-value
> engineering artefact this register has produced**, because it converts the most-repeated defect class
> from a review finding into a build failure.
>
> 🔴 **And C-79 has now been resurrected three times** — CPR proposed (Ruling 8), corrected and rejected
> (Ruling 9), prohibited in §C, and re-prescribed on the same block in §B. 📌 **A conflict that keeps
> returning after closure is not a documentation problem — it is a structural one.** The fix is to name
> **CPR inside $\\mathbf{A}_{pp}$ as a banned construct** in the CI gate, exactly as §7s.5's
> zero-denominator rule was promoted.

---

## Ruling 26 — C_K locked, blending verified, DP scaling inverted (08-10-2026)

Corrections **C-159 … C-163**. **Three accepted; the Drucker-Prager "resolution" is wrong by a factor of 3;
the RF definition omits injection.** Detail: [`engine_invariants.md`](engine_invariants.md) §7x.

| # | Correction | Detail |
|---|---|---|
| **C-159** | ✅ **$\\mathcal{C}_K = \\sqrt{2}$ CONFIRMED and the derivation is correct** ✅ — volume conservation $q_{2D}t = 2L_fw_0$ substituted into the time-volume form gives $w_0^2 = 2K_{IC}^2(1-\\nu^2)^2L_f/E^2$, hence $\\mathcal{C}_K = \\sqrt2 \\approx 1.41421$ ✅ **uniquely fixed by mass conservation, not a free parameter** | **CONF-51(a) closed** |
| **C-160** | ✅ **Blending weight VERIFIED CORRECT.** $W = 3s^2 - 2s^3$ with $s = (\\mathcal{K}-0.8)/0.45$ is the **smoothstep**: $W(0.8)=0$ ✅, $W(1.25)=1$ ✅, and $W'(s) = 6s(1-s) so **$W' = 0$ at both ends** $\\Rightarrow$ the blend is **$C^1$** ✅. ✅ Verified on $w_0$: $\\mathrm{d}w_0/\\mathrm{d}\\mathcal{K} = (1-W)w_I' + W' w_{II} + Ww_{II}'$ evaluates to $w_I'$ at $\\mathcal{K}=0.8$ and $w_{II}'$ at $1.25$ — **continuous with both branches** ✅. ⚠️ Note both $w_I$ and $w_{II}$ are $\\mathcal{K}$-**independent**, so $w_I' = w_{II}' = 0$ and the entire transition shape is carried by $W'$ — which is exactly the intended behaviour | **CONF-51(b) closed** |
| **C-161** | 🔴🔴 **THE DRUCKER-PRAGER "RESOLUTION" IS WRONG — the $\\sqrt3$ SCALING IS INVERTED, A FACTOR-3 ERROR.** 🔴 **Decisive, convention-free algebra:** since $\\sqrt{J_2} = \\sqrt3\\,\\sqrt{J_2/3}$, multiplying $F_A$ by $\\sqrt3$ gives $\\sqrt3 F_A = \\sqrt{J_2} + \\sqrt3\\,\\alpha_A I_1 - \\sqrt3 k_A$, hence **$\\alpha_B = \\sqrt3\\,\\alpha_A$ and $k_B = \\sqrt3\\,k_A$ — the formulations are related by MULTIPLYING by $\\sqrt3$.** The submission defines Formulation B with $k$ and $\\alpha$ **divided** by $\\sqrt3$. Measured at $c=1$ MPa, $\\phi=30°$: required $\\alpha_B = \\sqrt3\\alpha_A$, submitted $\\alpha_B = \\alpha_A/\\sqrt3$ — **ratio exactly $1/3$**, identically for $k$. 🔴 **The "fix" is worse than the mismatch it claims to resolve.** ✅ **Ruled:** if $F = \\sqrt{J_2} + \\alpha I_1 - k$, then $k = \\dfrac{6c\\cos\\phi}{\\sqrt3(3\\mp\\sin\\phi)}$ is **wrong**; either adopt Formulation A ($F = \\sqrt{J_2/3} + \\alpha I_1 - k$) with $k = \\dfrac{6c\\cos\\phi}{3\\mp\\sin\\phi}$, **or** keep $F = \\sqrt{J_2}$ and use $k = \\dfrac{6\\sqrt3\\,c\\cos\\phi}{3\\mp\\sin\\phi}$, $\\alpha = \\dfrac{2\\sqrt3\\,\\sin\\phi}{3\\mp\\sin\\phi}$. ⚠️ **Recommend Formulation A** — it is the one the classic DP↔MC tangency derivation produces, and it avoids the $\\sqrt3$ in the denominators entirely. ⚠️ *(I also attempted an MC-apex tangency cross-check; it came out inconclusive because the $I_1$ sign convention under compression is not fixed in the spec. The algebraic derivation above is the reliable evidence.)* | **CONF-66 B reopens** |
| **C-162** | ⚠️ **Hermite regularisation ACCEPTED** ✅ — specifying $c$ **and** $H$ at both ends of $[\\bar\\varepsilon_p^{peak}\\!\\pm\\delta]$ is a valid cubic Hermite specification and removes the tangent jump ✅. 🔴 **But $\\delta$ is UNDECLARED** — a new symbol with no value and no basis. 📌 **Fifth occurrence of the CONF-31 pattern** in this register. ✅ **Ruled: $\\delta$ declared as a fraction of $\\bar\\varepsilon_p^{peak}$ with a sourced default**, and recorded in the symbol register (**C-110**'s 4 points). ⚠️ Also: the Hermite must match the **actual** pre-peak value $c(\\bar\\varepsilon_p^{peak}-\\delta) < c_{peak}$, **not** $c_{peak}$ — otherwise the regularisation shifts the peak | **1 gap** |
| **C-163** | 🔴 **The RF definition OMITS OIL INJECTION.** $\\mathrm{RF} = \\dfrac{\\iiint_\\Omega[\\rho_o S_o|_0 - \\rho_o S_o|_t]d\\Omega}{\\iiint_\\Omega \\rho_o S_o|_0 d\\Omega}$ ✅ closes dimensionally (kg/kg) ✅ and is correct **for a no-oil-injection scheme** (WAG ✅). 🔴 **But the design set covers broader EOR — solvent and polymer schemes inject oil**, and the balance then omits $\\int_\\Omega \\rho_{o,\\text{inj}}$. ✅ **Ruled:** write the full balance $\\mathrm{RF} = \\dfrac{\\int_\\Omega \\rho_oS_o|_0 - \\int_\\Omega \\rho_oS_o|_t + \\int_\\Omega \\rho_{o,\\text{inj}}}{\\int_\\Omega \\rho_oS_o|_0}$, **or** restrict to the no-oil-injection case and **declare that restriction as an input-domain warning**. ⚠️ Also, $\\mathrm{RF} = N_p/N_{initial}$ holds only with **no aquifer influx**; with an aquifer the denominator must include aquifer oil. Both conditions belong in the manifest. · ⚠️ **Two smaller items:** **(a)** ✅ non-proportional hardening declared with a `proportional_hardening_only` flag ✅ — correct, and a modelling switch must be a **declared input**, which it now is. **(b)** 🔴 **the dynamic solver switch should be LATCHED per timestep** — plasticity **deactivates on unloading**, so `is_plastic_active` can toggle within a run and alternate CG-Cholesky with FGMRES-ILU. ✅ Latch on first activation for the remainder of the timestep (FGMRES tolerates the varying preconditioner, which is why it was chosen). ⚠️ Also ⚠️ **"Watts' Volume-Balance Decoupling" is a RESIDUAL FORMULATION, not a preconditioner** — listing it as a preconditioning option for $\\mathbf{A}_{pp}$ conflates assembly with linear algebra, the same category error as **C-126**. ✅ **Ruled: $\\mathbf{A}_{pp}$ preconditioned by Block-ILU(1) or unsymmetric AMG; the volume-balance reduction belongs to residual assembly.** | **CONF-17 conditional** |

### Register status after Ruling 26

| Closed | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 · C-76 · C-85 · C-86 · C-87 · 93 · 99a · 99b · 102 · 104 · 105 · 106 · 109 · 111 · 117 · 119 · 124 · 126 · 129 · 132 · 133 · 134 · 135 · 141 · 143 · **159 · 160** |
|---|---|
| **Conditional** | **CONF-17** (C-163: RF omits injection, aquifer caveat, solver latch, preconditioner category) · **CONF-66** (**C-161 reopens the DP scaling**; C-162: $\\delta$ undeclared) · **CONF-51** ✅ dimensional + blending now closed |
| **Partially closed** | **CONF-14** · **CONF-25** · **CONF-31** (literature parked) |
| **Fix specified** | **CONF-15** 🟡 · **CONF-16** · **CONF-29** |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **C-159 + C-160 close CONF-51's dimensional work completely** — and both arrived *verified on the first pass*. 📌 That is the third time a fix has needed no correction (**C-113** gates 1–2, **C-135** AMO determinism clause, now these). ⚠️ **The pattern: corrections are reliable when they are **dimensional or algebraic**, and unreliable when they are **convention-dependent or physical-intuition claims.** The $W(\\mathcal{K})$ smoothstep, the $\\mathcal{C}_K$ derivation and the volume balance are all convention-free algebra — and all three held. The DP scaling, the Tier-A normaliser and the elastic-capability reasoning are all convention/intuition — and all three needed correction.
>
> 🔴 **And C-161 is the most consequential finding of the round: a "resolution" that introduced a factor-3 error where it claimed to remove an ambiguity.** 📌 It is the mirror image of **C-155**'s fix, which *was* correct. ⚠️ **Same mechanism, opposite outcome** — which is why the dimensional and algebraic gates must be *executable*, not advisory: **neither this scaling inversion nor the seven earlier dimensional failures would survive a `#[cfg(test)]` unit check.**

---

## Ruling 27 — C-164…C-169: Formulation A still 3x wrong, apex sign inverted, RF double-counts injection (08-10-2026)

Corrections **C-164 … C-169**. **Four accepted without change** (algebra, latching, categorisation, $\delta$).
**Two of the newly "adopted" items are still wrong** — and both are caught by a *limit-case* test,
not by the dimensional gate.

| # | Correction | Detail |
|---|---|---|
| **C-164** | 🔴🔴 **ADOPTING "FORMULATION A" DID NOT FIX IT — the coefficients are STILL wrong by a factor of 3.** 🔴 The submission correctly adopts $F=\sqrt{J_2/3}+\alpha I_1-k$ ✅ and correctly cites Chen & Han tangency ✅, but then pairs it with $k=\frac{6c\cos\phi}{3\mp\sin\phi}$, $\alpha=\frac{2\sin\phi}{3\mp\sin\phi}$ — which is **Formulation B's coefficients with the $\sqrt3$ stripped**, a hybrid belonging to **neither** convention. ✅ **Decisive test — $\phi=0$ must reduce Drucker-Prager to Tresca ($\tau=c$).** Pure shear at $\tau=c$: $s_2=s_1+2c$, $s_3=s_1$ ⇒ $I_1=2c$, $J_2=\frac{4c^2}{3}$, $\sqrt{J_2/3}=\frac{2c}{3}$. Measured at $c=10^6$ Pa: $F(\tau=c)$ with the submitted $k=2.000\times10^6$ gives **$-1.333\times10^6$** ❌ (must be $0$); with $k=\frac{2c}{3}=6.667\times10^5$ gives **$0.000$** ✅. 🔴 **Correct Formulation A: $k=\dfrac{2c\cos\phi}{3\mp\sin\phi}$, $\alpha=\dfrac{2\sin\phi}{3(3\mp\sin\phi)}$ — both carry an extra $3$ in the denominator.** ✅ **Both coefficients scale identically**, which is the internal-consistency check: $\alpha$ was wrong by the same factor $3$ as $k$ ($0.1333$ vs $0.0444$ at $\phi=30°$). 📌 *Root cause: the scaling direction was reasoned about in prose three times and inverted twice. Executable limit-case tests would have caught it twice.* | **CONF-66 B reopens again** |
| **C-165** | 🔴 **THE DECLARED CONVENTION PUTS THE YIELD APEX ON THE TENSILE SIDE.** ⚠️ The claim *"$+\alpha I_1$ under $I_1<0$ reduces $F$, increasing confinement shear resistance as physically required"* is **directionally right but sign-inconsistent**: it is true only for $\alpha<0$. With $\alpha>0$ and tension-positive $\sigma$, the hydrostatic apex sits at $I_1 = k/\alpha = +5.196\times10^6$ — **tension**. Measured directly: hydrostatic compression at $10^7$ Pa gives $F=-4.69\times10^6$ (**elastic, never yields**), hydrostatic tension gives $F=+3.31\times10^6$ (**yields**) ❌ — a frictional material whose yield surface opens in tension. ✅ **Ruled — the term and the convention must be paired, not chosen independently:** **(i)** tension-positive $\sigma$: $F=\sqrt{J_2/3}-\alpha I_1-k$, $\alpha>0$, apex $I_1=-k/\alpha=-5.196\times10^6$ ✅ **compression**; **or (ii)** compression-positive $\sigma$: $F=\sqrt{J_2/3}+\alpha I_1-k$, $\alpha>0$. ⚠️ **Recommend (ii)** — it matches the geomechanics stress convention already used elsewhere in the design set and keeps the submitted sign of the $I_1$ term. 🔴 **Whichever is chosen must be pinned by a `#[cfg(test)]` case asserting the apex is in compression** — this is **Gate 3**'s first mandatory case. | **CONF-66 B reopens** |
| **C-166** | ✅ **Formulation-A algebra confirmed** — $\alpha_B=\sqrt3\alpha_A$, $k_B=\sqrt3k_A$ ✅ and the $\sqrt3$ elimination is a real benefit ✅ | **C-161(a) closed** |
| **C-167** | 🔴 **THE RF DOMAIN FORM DOUBLE-SUBTRACTS INJECTED OIL (error $=-N_{\text{inj}}$).** ✅ Reservoir oil balance $\frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega\rho_oS_o\,\mathrm{d}\Omega = -Q_{o,\text{prod}}+Q_{o,\text{inj}}$ integrates to $\int_\Omega\rho_oS_o|_0-\int_\Omega\rho_oS_o|_t = N_p-N_{\text{inj}}$ — so **the domain integral already equals $N_p-N_{\text{inj}}$, and the extra $-\int_0^tQ_{o,\text{inj}}\,\mathrm{d}t$ subtracts injection a second time.** Measured: $N_p=100$, $N_{\text{inj}}=10$ ⇒ removal $=90$, submitted numerator $=80$, **error $=-10$**. ✅ **Ruled:** $$\mathrm{RF}(t)=\frac{\iiint_\Omega\rho_o^0S_o^0\,\mathrm{d}\Omega-\iiint_\Omega\rho_ot^S_ot\,\mathrm{d}\Omega}{\iiint_\Omega\rho_o^0S_o^0\,\mathrm{d}\Omega+\iiint_\Omega\rho_o^{\text{influx}}\mathrm{d}\Omega}$$ ⚠️ Note the sign of $\rho_o^{\text{influx}}$ in the denominator is a **convention question that must be declared** — if the aquifer is modelled as an external source with an influx schedule, add its cumulative oil volume; if it is modelled **inside** $\Omega$, the numerator already contains it and **only** the denominator term may be added, or **double-counted**. 📌 *The well form $\frac{N_p-N_{\text{inj}}}{N_0+N_{\text{influx}}}$ was correct — only the domain form was wrong. Two forms of one quantity must be asserted equal to each other in `#[cfg(test)]`; that assertion would have caught this.* | **CONF-17 conditional** |
| **C-168** | 🔴 **$W_e$ IS WATER INFLUX INSIDE AN OIL INVENTORY — A UNIT ERROR (8th dimensional failure).** ⚠️ The denominator is an **oil** mass $[\text{kg oil}]$; the manifest flag is `active water influx $W_e>0$`. ⚠️ $[\text{kg water}]\ne[\text{kg oil}]$ — they cannot be added. ✅ **Ruled: use $W_{e,o}$ (oil influx)**, flagged `ValidityWarning::AquiferOilInfluxPresent`. ⚠️ A water aquifer still matters to RF, but **through the water-cut term, not the oil denominator** — so it belongs as a separate declared diagnostic, not inside the balance. 📌 **This is the 8th dimensional failure in this register — the unit gate would have caught it.** | **CONF-17 conditional** |
| **C-169** | ⚠️ **HERMITE UNDER-SPECIFIED — 3 of 4 conditions given, and the missing one is exactly the $C^1$ condition.** A cubic Hermite needs **two values + two slopes**. Submitted: $c(\bar\varepsilon_p^{\text{peak}}-\delta)$ ✅, $c(\bar\varepsilon_p^{\text{peak}})$ ✅, $c'(\bar\varepsilon_p^{\text{peak}})=0$ ✅ — but **not** $c'(\bar\varepsilon_p^{\text{peak}}+\delta)$. ✅ **Ruled: add $c'(\bar\varepsilon_p^{\text{peak}}+\delta) = -\eta\,(c_{\text{peak}}-c_{\text{res}})e^{-\eta\delta}$,** i.e. the slope of the post-peak law evaluated at the band's right edge — otherwise $H$ still steps at $\bar\varepsilon_p^{\text{peak}}+\delta$ and **the $C^1$ claim does not hold where it is actually needed.** ⚠️ The other three conditions are correct and the peak-shift problem is properly solved ✅ | **1 gap** |

### Accepted unchanged

| Item | Verdict |
|---|---|
| ✅ **$\delta$ declared** — dimensionless $[\,]$, $\delta\in[10^{-4},10^{-3}]$, `SOURCE_PENDING` (Abbo & Sloan 1995) | ✅ **Sixth CONF-31 pattern, now closed** — symbol, units, range and provenance status all recorded. ⚠️ The `SOURCE_PENDING` on the *range* becomes a literature item (**L-5**); the *symbol* is no longer undeclared |
| ✅ **Solver latching per timestep**, reset only at $t^{n+1}$ if all cells elastic | ✅ Correct, and it is the right granularity: it prevents CG-Cholesky / FGMRES-ILU flip-flop **within** a Newton solve while still allowing the cheaper elastic solver next timestep |
| ✅ **Watts' = residual assembly; CPR-AMG = preconditioner** | ✅ Correct split. ⚠️ Record that this supersedes **C-126**'s conflation |
| ✅ **Hermite peak BCs** $c(\bar\varepsilon_p^{\text{peak}}-\delta)=c_{\text{hardening}}(\cdot)$, $c(\bar\varepsilon_p^{\text{peak}})=c_{\text{peak}}$, $c'=0$ | ✅ The peak-shift defect I raised in **C-162** is genuinely fixed by these — ⚠️ subject to adding the 4th condition (C-169) |
| ✅ **Gate 3 — convention-lock suite** (DP vs canonical MC states, $\phi=30°$, $c=1$ MPa) | ✅ Adopted. ⚠️ **Strengthened by measurement:** the exact case it must contain is now specified — **(a)** $\phi=0$ ⇒ **Tresca**, $F=0$; **(b)** apex is in **compression**; **(c)** the $A\leftrightarrow B$ round trip, $\sqrt3\,F_A\equiv F_B$ |

### Register status after Ruling 27

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-17** (C-167 double-count, C-168 unit error) · **CONF-66** (**C-164 coefficients still 3x wrong; C-165 apex on tensile side**) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** (Karakas & Tariq $\alpha_0$ unobtainable; now also L-5 for the $\delta$ range) · **CONF-51** (Type-II ✅ **C-159/C-160**; 🔴 proppant coupling still open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **The pattern in this round is sharper than "dimensional vs algebraic".** **C-164 was *algebraic* — correct reasoning about the $\sqrt3$ scaling — and it was still wrong**, because the error was in *which* limit case the result was checked against. ⚠️ **The reliable discriminator is different: a claim survives if it was tested against an independent limit, degenerate case, or an equivalence assertion** (Tresca at $\phi=0$; $A\leftrightarrow B$ round trip; domain form $\equiv$ well form). ⚠️ C-159, C-160 survived *without* such a test only because they were dimensionless identities checked symbolically.
>
> 🔴 **Consequence for the gates: a dimensional gate is not enough, and a convention-lock gate is not enough either — both were proposed *after* the failures they would prevent.** ✅ **The durable gate is a _limit-case suite_: every constitutive family must reproduce its degenerate limit** ($\phi\to0$ ⇒ Tresca; $\kappa\to\infty$ ⇒ rigid; $k\to0,\alpha\to0$ ⇒ Tresca again; $S_{w}\equiv1$ ⇒ single-phase).
>
> ⚠️ **And C-161 → C-164 is the third consecutive inversion of the same $\sqrt3$ direction.** 📌 Three attempts, two inversions — with the *correct* answer ($k_B=\sqrt3k_A$, Formulation A with the extra $3$) stated on round one and then discarded twice. ✅ **Recorded as a named hazard: the $\sqrt3$ scaling must be carried as a single derived constant, never re-typed per convention.**

---

## Ruling 28 — C-170…C-175: coefficients locked, apex regression, overshoot found, both gate tests defective (08-10-2026)

Corrections **C-170 … C-175**. ✅ **The two DP coefficients are now correct and locked.** 🔴 **Three items regressed or were wrong on arrival** — including a *second* inversion of the sign convention, in the opposite direction from the one it was correcting.

| # | Correction | Detail |
|---|---|---|
| **C-170** | ✅ **COEFFICIENTS CONFIRMED AND LOCKED.** ✅ The §1A derivation is **correct** and better than my own: it evaluates **two different states** at $\tau_{\max}=c$ and reports **two different** $\sqrt{J_2/3}$ — $(c,0,-c)\Rightarrow c/\sqrt3$ and triaxial compression $\Rightarrow 2c/3$ ✅ — which is exactly the distinction I had blurred. ✅ **Locked target:** $\alpha=\dfrac{2\sin\phi}{3(3\mp\sin\phi)}$, $k=\dfrac{2c\cos\phi}{3\mp\sin\phi}$. ✅ **$\phi=0\Rightarrow k=2c/3$ verified exact** (triaxial-compression meridian). ⚠️ But the **"single derived constant" policy is half-implemented**: `DP_ALPHA_SCALE` exists, there is **no `DP_K_SCALE`**, and the value is a **hardcoded literal `2.0/3.0`** — precisely what a convention-lock gate forbids. ✅ **Ruled: both constants derived from the Tresca limit case in `#[cfg(test)]`, neither re-typed as a literal** | **CONF-66 item 3 coefficients CLOSED** |
| **C-171** | 🔴🔴 **C-165 HAS REGRESSED — §2 ADOPTS A *THIRD* SIGN PAIRING AND PRESENTS THE UNPHYSICAL OUTCOME AS THE RESOLUTION.** ✅ I ruled two self-consistent options: (i) tension-positive with $-\alpha I_1$; (ii) **compression-positive with $+\alpha I_1$** (recommended). 🔴 **§2 pairs "compression-positive" with "$-\alpha I_1$" — a combination that appears in neither.** Measured apex $I_1=-k/\alpha=-5.196\times10^6$: **(i)** compression ✅ · **(ii)** compression ✅ · **§2 SUBMITTED → TENSION** ❌. ⚠️ The chain *"compression increases shear strength $\Rightarrow$ MUST write $-\alpha I_1$"* is sound; ⚠️ but the submission then reports *"**yields in tension** (apex lands strictly in the tensile regime at $I_1=-k/\alpha$)"* **as the desired outcome** — it is not. Under §2's form the material has **zero shear strength at tensile mean stress** $-k/\alpha$: cohesive-frictional rock cannot be that weak in extension. 🔴 **This is the *same class* of defect as C-161/C-164 — a sign/convention inversion — now in the other direction, i.e. the item that was raised *to fix* one.** ✅ **Ruled — §2 reverts to option (ii): $F=\sqrt{J_2/3}+\alpha I_1-k$, compression-positive.** 🔴 **And the underlying pathology must be stated, because it is not fixable by a sign choice:** linear DP matches MC on **one meridian only** and is **too strong on the extension meridian** (measured: at $I_1=0$ pure shear it predicts $\tau=1.1547c$ against Tresca's $c$, **+15.47 %** — the classic von-Mises↔Tresca gap, see **C-175**). ✅ **Ruled: add a tension-side cap surface, or record the $15.47\%$ over-prediction as a declared model limitation with a `ValidityWarning`.** | **CONF-66 item 3 REOPENS — 3rd round** |
| **C-172** | 🔴 **THE QUINTIC HERMITE VIOLATES THE BOUNDEDNESS INVARIANT IT WAS BUILT TO GUARANTEE.** ✅ 6 conditions = quintic ✓ **correct count**; ✅ **$C^1$ verified numerically** — slope residuals at all three nodes $\le 2.7\times10^{-20}$ ✓. 🔴 **But it overshoots $c_{peak}$.** The previous round's BC set was adopted *specifically* to guarantee "$c\le c_{peak}$ everywhere"; the quintic **breaks that**, and the error is controlled by the dimensionless group $H_{\text{peak}}\delta/c_{peak}$. Measured over the declared $\delta\in[10^{-4},10^{-3}]$: | $H/c_{peak}$ | $H\delta/c_{peak}$ | overshoot | |---|---|---| | 0.06 | $6\times10^{-5}$ | bounded ✅ | | 1 | $10^{-3}$ | $+5.7\times10^{-5}$ ❌ | | 100 | 0.1 | $+1.07\times10^{-2}$ ❌ | | 10⁴ | 10 | $+1.078$ ❌ | ⚠️ **overshoot appears for essentially any non-zero hardening modulus**, though it is small for rock ($H/c=0.32$, $\delta=10^{-3}\Rightarrow$ ~$3\times10^{-5}$ relative — physically negligible, invariant-breaking). ✅ **Ruled: (a) assert $\max_{\text{band}}c\le c_{peak}(1+\varepsilon)$ with $\varepsilon$ declared, and (b) build the interpolant in a LOCAL coordinate $u=(\bar\varepsilon_p-\bar\varepsilon_p^{\text{peak}})/\delta$** — ⚠️ my first evaluation on the raw axis returned $1.6\times10^{14}$, i.e. **the Vandermonde is catastrophically ill-conditioned at $\delta=10^{-3}$; the local coordinate is not cosmetic, it is required for the coefficients to mean anything.** ⚠️ $\mathrm{d}^2c/\mathrm{d}\bar\varepsilon_p^2$ is **unbounded across the band** by the spec; a bounded-curvature assertion is missing | **CONF-66 item 2** |
| **C-173** | 🔴 **GATE-3 TEST 1 FAILS AS WRITTEN.** `test_drucker_prager_tresca_limit` sets `sqrt_j2_over_3 = c/3.0.sqrt()` with the comment `J2 = c^2` — that is the $(c,0,-c)$ state, $\sqrt{J_2/3}=c/\sqrt3$. With the now-correct $k=2c/3$: $f_{yield}=\dfrac{c}{\sqrt3}-\dfrac{2c}{3}=\mathbf{-0.0893c}$, **$-13.40\%$** ❌. 🔴 **The test asserts DP yields at the $I_1=0$ state — the very state where §1A itself shows $\sqrt{J_2/3}=c/\sqrt3\ne k$.** ✅ **Ruled: the Tresca limit case must be evaluated on the **triaxial-compression meridian** ($s_1-s_3=2c$), matching the meridian $k$ was derived on** — and **both** meridians plus the $I_1=0$ plane must be asserted, with the $15.47\%$ gap recorded rather than asserted away (C-175). ⚠️ Also `DP_ALPHA_SCALE`, `calculate_dp_k`, `calculate_dp_alpha` are referenced but **none appear in the test body** — the constant is declared and never exercised | **Gate 3 defective** |
| **C-174** | 🔴 **GATE-3 TEST 2 IS VACUOUS.** `test_recovery_factor_equivalence` computes `delta_m_domain = n_p - n_inj` and then asserts `(n_p - n_inj)/n_initial == delta_m_domain/n_initial`. 🔴 **It compares a quantity with itself; no integral is ever evaluated.** It would pass against an engine whose RF is arbitrarily wrong — including the exact double-subtraction bug (**C-167**) it was written to catch. ✅ **Ruled: the two forms must be built from *independent inputs* — the wellhead form from rate integrals $\int Q_{o,\text{prod}},\int Q_{o,\text{inj}}$; the domain form from an actual 3-D field reduction of $\rho_oS_o$ over a mesh — then compared.** ⚠️ And `assert_eq!` on `f64` must be `abs_diff_eq!`. 📌 *A self-referential equivalence test is worse than no test: it manufactures false assurance.* — **new instance of the same class as C-134 (Kahan) and C-152 ($\Delta t$-cancellation): a check that cannot fail.** | **Gate 3 defective** |
| **C-175** | 🔴 **MY OWN C-164 PREMISE WAS IMPRECISE — the constant is right, my stated reason was wrong.** I wrote *"$\phi\to0$ must reduce DP to Tresca ($\tau=c$)"*. 🔴 **At $\phi=0$, $\alpha=0$, so $F=\sqrt{J_2/3}-k$ is independent of $I_1$ — DP becomes a von Mises criterion, which does *not* reduce to Tresca.** Measured, four states all with $\tau_{\max}=c$: | state | $I_1$ | $\sqrt{J_2/3}$ | $F$ | |---|---|---|---| | pure shear $(0,2c,0)$ | $+2c$ | $2c/3$ | $0$ ✅ | | **$(c,0,-c)$** | $0$ | $c/\sqrt3$ | $\mathbf{-0.0893c}$ ❌ | | triax. compression $(-2c,0,0)$ | $-2c$ | $2c/3$ | $0$ ✅ | | triax. extension $(2c,0,0)$ | $+2c$ | $2c/3$ | $0$ ✅ | 📌 **$\sqrt{J_2/3}$ is not a function of $\tau_{\max}$, so *no single* $k$ can match Tresca everywhere.** $k=2c/3$ matches **only** the triaxial-compression/extension meridians. ✅ **The constant is therefore correct, but its justification must be restated:** $k$ is fixed by the **compression meridian**, and the residual **$+15.47\%$ at $I_1=0$** is **inherent to the correspondence**, not a defect in the constant. ⚠️ If the design set wants *exact* Tresca at $\phi=0$ on **all** paths, the fix is a **sectoral/three-invariant criterion**, **not a different $k$** | **C-164 premise refined** |

### Accepted unchanged

| Item | Verdict |
|---|---|
| ✅ **§1A Tresca derivation** — and its **two-state** treatment of $\sqrt{J_2/3}$ | ✅ Correct, and **better than my own** — see C-175 |
| ✅ **§3 RF, both forms** — wellhead $(N_p-N_{\text{inj}})/(N_0+N_{\text{influx},o})$ and domain $\frac{\iint(\rho_o^0S_o^0-\rho_ot^S_ot)}{\int\rho_o^0S_o^0+N_{\text{influx},o}}$ | ✅ Exactly the ruled form. ✅ **C-167 CLOSED** |
| ✅ **§3B water influx removed**, `ValidityWarning::AquiferOilInfluxPresent` | ✅ Correct — **C-168 CLOSED** (8th dimensional failure closed) |
| ✅ **§4 six Hermite conditions** at three nodes | ✅ Correct count for $C^1$; ✅ **$C^1$ verified to $10^{-20}$** — **C-169 CLOSED**, ⚠️ subject to C-172's boundedness assertion |
| ✅ **§5 Gate 3 promoted to `#[cfg(test)]` CI** | ✅ Adopted. ⚠️ **its two shipped tests are defective — C-173, C-174** |

### Register status after Ruling 28

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-171 sign regression; C-172 overshoot**) · **CONF-14** · **CONF-25** |
| **Closed this round** | **CONF-17** — ✅ C-158 (`P̄_res,eff` demoted to a diagnostic) + ✅ C-167 (RF mass balance) + ✅ C-168 (influx units) |
| **Partially closed** | **CONF-31** (literature parked; now also **L-5**) · **CONF-51** (Type-II ✅; 🔴 proppant coupling open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **This round produced the register's clearest instance of my own failure mode — and it was caught by the submission, not by me.**
> **C-175: I asserted "$k=2c/3$ because $\phi\to0$ reduces DP to Tresca." The constant is right; the reason was wrong** — at $\alpha=0$ DP is a **von Mises cylinder**, not Tresca, and it over-predicts pure-shear strength by **$15.47\%$**. 📌 **The engineer's §1A caught it by simply evaluating *two* states instead of one.** ⚠️ **I generalised a single state's behaviour to a criterion identity — the same move as C-134 and C-152.**
>
> 🔴 **And the $\sqrt3$/sign hazard has now inverted in *both* directions**: too small (C-161), too large (C-164), and now the *sign* flipped while attempting the correction (C-171). 📌 **Four inversions across three rounds.** ✅ **This settles the earlier question about whether the constants should be typed per convention: they must not be typed *at all*.** ✅ **Ruled — the yield function is assembled from *named* $\phi$-dependent closures returning a validated `(alpha, k)` pair, and Gate 3 asserts the limit cases; no scalar factor is written by hand anywhere in the codebase.**
>
> ⚠️ **And C-174 is the register's fourth "check that cannot fail"** (after C-134, C-152, and the Tier-A normaliser of C-143). 📌 **Every one of them was an assertion whose inputs were derived from the thing being asserted.** ✅ **Gate 3 must additionally require that each equivalence test draw its two sides from *independent* sources** — mesh reduction vs. well integrals, never one from the other.

---

## Ruling 29 — C-176…C-183: 🔴🔴 **C-171 RETRACTED** — the submitted sign is the exact MC correspondence (09-10-2026)

Corrections **C-176 … C-183**. 🔴 **My own C-171 was wrong and is withdrawn.** The engineer's §2 sign is **correct**, and is the **exact** Mohr-Coulomb correspondence.

| # | Correction | Detail |
|---|---|---|
| **C-176** | 🔴🔴 **C-171 RETRACTED. The submitted sign is the exact Mohr-Coulomb correspondence; my ruling was $57$–$86\%$ wrong.** ✅ Measured, compression-positive, $\alpha=\frac{2\sin\phi}{3(3\mp\sin\phi)}$, $k=\frac{2c\cos\phi}{3\mp\sin\phi}$, $c=10^6$, $\phi=30°$, against the MC triaxial-compression line: | sign | $P=0$ | $P=c\cot\phi/2$ | $P=c\cot\phi$ | $P=1.5c\cot\phi$ | $P=2c\cot\phi$ | |---|---|---|---|---| | **$-\alpha I_1$ (submitted)** | **$0.0000\%$** | **$0.0000\%$** | **$0.0000\%$** | **$0.0000\%$** | **$0.0000\%$** | | $+\alpha I_1$ (**my C-171**) | $-57.14\%$ | $-73.47\%$ | $-80.00\%$ | $-83.52\%$ | $-85.71\%$ | 🔴 **My justification was also wrong.** I asserted *"the apex must be in compression."* It must not: MC's own apex is a **tensile cutoff**, because friction *increases* strength with confinement. Measured: MC plane criterion in compression-positive $\tau=c+P_{\text{comp}}\tan\phi$ ⇒ pure hydrostatic **compression** is **always elastic** (no compression apex), while pure hydrostatic **tension** fails at $T=c\cot\phi=1.7321\times10^6$ Pa. DP with $-\alpha I_1$ puts its apex at $I_1/3=-1.7321\times10^6$ — **exactly MC's tensile apex**, matching to $10^{-16}$. ⚠️ *Root cause: I reasoned about the apex by analogy ("compression should close the cone") without evaluating the MC plane criterion, and I twice wrote the MC principal-stress line with the wrong sign on the $P$ term. Both errors pointed the same way.* 📌 **This is my third self-correction on this sign — and the first two were confident.** ✅ **C-176 supersedes C-171; §7y.2 and §7z.2 are withdrawn.** | **CONF-66 item 3 sign CLOSED — round 3** |
| **C-177** | 🔴 **THE FACTORY'S `Extension` BRANCH IS NOT MOHR-COULOMB.** ✅ `factory(Compression)` reproduces the locked coefficients exactly ✅ — ⚠️ but `factory(Extension)` at $P=0$ gives $T=1.1547\times10^6$ against MC's $3.4641\times10^6$: **$-66.67\%$**. ⚠️ And $1.1547\times10^6 = 2c/\sqrt3$ — it returns the **von Mises** uniaxial-tension value, **not** MC's. Error grows to $+133\%$ at $P=1.5c\cot\phi$ and diverges as $P\to$ apex ($P=2c\cot\phi$ where MC itself returns $4.7\times10^{-10}$). 🔴 **The $3+\sin\phi$ branch is a von Mises extension correspondence mislabelled as Mohr-Coulomb.** ✅ **Ruled: either rename it `MeridianType::VonMisesExtension` and document that it is *not* MC, or delete it and keep the single MC compression branch.** ⚠️ A reader cannot currently tell which surfaces the engine will actually produce. | **CONF-66 item 3** |
| **C-178** | 🔴 **THE RANKINE CAP AND ITS WARNING MACHINERY ADDRESS A FAILURE THAT DOES NOT EXIST.** ⚠️ With the **correct** sign, DP's apex already sits **exactly** at MC's tensile apex ($-c\cot\phi$, matching to $10^{-16}$) — i.e. **DP already reproduces the tensile cutoff.** ✅ Adding a separate $\sigma_t$-driven Rankine cap therefore **overrides an already-correct feature**. Two measured consequences: **(a)** if user $\sigma_t$ differs from MC's implied $2c\cos\phi/(1+\sin\phi)$, the composite surface **kinks**; **(b)** 🔴 at $I_1=0$ the Rankine branch gives $\tau=0$ while DP gives $\tau=k=6.928\times10^5$ Pa — **a discontinuity of one full cohesion**, and **no value of $\sigma_t$ changes it**. ✅ **Ruled: delete the cap, and delete `ValidityWarning::UncappedDruckerPragerTensileRegime` + `ValidityClass::ConvergedOutsideEnvelope`.** ⚠️ Under the C-93 / C-151 rule — *a correction that makes failure **disappear** rather than be **reported** is forbidden* — **a warning for a failure that does not exist is itself a defect**: it trains a reader to treat `ConvergedOutsideEnvelope` as noise. ✅ If a tension cutoff is wanted as a **declared input override**, it must be a separately named surface with a stated precedence and an **asserted continuity** at the junction | **CONF-66 item 3** |
| **C-179** | ⚠️ **`k`'s DENOMINATOR MUST NOT BE A MUTATION OF `\alpha`'s.** 🔴 `k: (2.0*c_pa*cos_phi)/(denom/3.0)` — $k$'s denominator is produced by **dividing $\alpha$'s by 3**. Numerically correct today ✅, but **editing the one `denom` line silently moves both coefficients together** — which is precisely the C-161/C-164 hazard, re-seated one abstraction level up. 📌 The factory is the right *shape*; this is the wrong *wiring*. ✅ **Ruled: two independently-stated denominators, plus a `#[cfg(test)]` asserting the $\alpha/k$ ratio.** ⚠️ Also: no validation on `c_pa` or $\phi_{\text{rad}}\in[0,\pi/2)$ — INV-7 requires every input validated and declared | **CONF-66 item 3** |
| **C-180** | 🔴 **`MeridianType` IS INERT AT $\phi=0$ — Gate-3 Test 1 COVERS NOTHING.** $\sin\phi=0$ collapses both denominators, so `Compression` and `Extension` return **identical** coefficients at the frictionless limit. ⚠️ Gate-3 Test 1 as remediated runs **at $\phi=0$** ⇒ **it never exercises the meridian branch it exists to test.** ✅ **Ruled: the meridian test must run at $\phi>0$ and assert that the two branches **differ**, alongside the $\phi=0$ Tresca limit** | **Gate 3** |
| **C-181** | 🔴 **THE "MONOTONE HERMITE" IS OVERDETERMINED — 4 DOF, 6 CONDITIONS.** The submitted basis is the standard **cubic** Hermite on $s\in[0,1]$: **4** degrees of freedom. The submission imposes $c(0),c'(0),c(1),c'(1)$ **plus** $c(0.5)=c_{peak}$ and $c'(0.5)=0$ — 🔴 **$c(0.5)$ and $c'(0.5)$ are not free parameters.** Measured: $c(0.5)=0.99994$ against required $1.0$ (**error $-6.0\times10^{-5}$**); $c'(0.5)=-1.5$ against required $0$. 🔴 **And Fritsch–Carlson's $d\le3\Delta$ clamp cannot be used here at all:** it is a theorem for **monotone** node data, and this band has an interior maximum by construction. Measured $\Delta=c_1-c_0=-1.19973\times10^{-4}$ ⇒ $3\Delta=-3.599\times10^{-4}$, while $d_0=+1.200\times10^{-4}$ — 🔴 **clamping $d_0\le3\Delta$ negates the hardening slope outright.** ✅ **Ruled — one of: (a) split into two monotone segments $[0,0.5]$ and $[0.5,1]$, clamp each; or (b) keep the **quintic** (C-172: $C^1$ verified to $10^{-20}$) and assert $\max_{\text{band}}c\le c_{peak}(1+\varepsilon)$ in a local coordinate.** ⚠️ The $10^{14}$ condition number **is** fixed by the local coordinate ✅ — that half of the fix stands | **CONF-66 item 2** |
| **C-182** | ⚠️ **THE $10^{-12}$ MASS-CLOSURE TOLERANCE IS NOT SCALE-FREE.** ✅ Gate-3 Test 2's **independence fix is correct and accepted** — a 3-D FVM mesh reduction against drift-flux well integrals are genuinely independent sources ✅. 🔴 **But the constant is wrong.** Measured at $N=1.1\times10^6$ cells: naive f64 accumulation rel. err. $1.99\times10^{-14}$; random-walk floor $\sqrt N\,\epsilon=2.33\times10^{-13}$; worst-case $N\epsilon=2.44\times10^{-10}$. So $10^{-12}$ leaves only **$\sim4\times$** headroom over pure summation, and the constant **does not scale with $N_t$ or Newton depth** — 🔴 **the same constant cannot be reused for a 500-step implicit run.** ✅ **Ruled: tolerance $=\kappa\sqrt{N}\,\epsilon$ with $\kappa$ declared, re-derived per accumulation depth**, consistent with the Tier-A bound already verified for this engine (**C-143**: 1.1M cells $\Rightarrow3.5\times10^{-10}$ of reservoir mass/yr) | **Gate 3** |
| **C-183** | ✅ **GATE-3 TEST 2 IS GENUINELY FIXED** — ✅ the two sides now come from *independent subsystems* (mesh reduction vs. well integrals), which is exactly what C-174 required. ⚠️ subject only to C-182's tolerance. ✅ **Test 1's _state_ is now correct** — the compression meridian, where $F\equiv0$ exactly at $\phi=0$ ✅ — ⚠️ but ⚠️ it remains a **single-point** check at $\phi=0$ (C-180). ✅ **§1B von Mises analysis accepted** and now **independently validated**: the $+15.47\%$ at $I_1=0$ is the $\pi$-plane (Lode-angle) gap, and the **Lode-Angle-Dependent Modified DP** $\sqrt{J_2/3}\,g(\theta_L)-\alpha I_1-k$ is the correct remedy — ✅ because the meridians *are* exact (C-176), so the only defect left is Lode-dependence | **Gate 3 largely closed** |

### Accepted unchanged

| Item | Verdict |
|---|---|
| ✅ **§1A factory shape** — a pure functional closure returning $(\alpha,k)$ | ✅ Right shape. ⚠️ wrong wiring (C-179) and an unexercised branch (C-180) |
| ✅ **§1B von Mises cylinder at $\phi=0$**; extension-meridian hexagon vs circumscribed circle | ✅ Correct, and **validated by measurement** |
| ✅ **§4A Test 1 state** — compression meridian, $\sigma_1=\sigma_2=0,\ \sigma_3=2c$ | ✅ $F\equiv0$ exactly |
| ✅ **§4B Test 2 independence** | ✅ Real fix |
| ✅ **§2's inference that the sign must be reconciled with compression-positive** | ✅ Necessary check — and §2's answer survived it |

### Register status after Ruling 29

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-177** extension branch is von Mises not MC · **C-178** Rankine cap · **C-179** wiring · **C-181** Hermite DOF) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** (literature parked) · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 🔴 **C-176 is the register's most important self-correction.** I asserted a sign ruling in **C-165**, "corrected" it in **C-171**, and **both were wrong** — the engineer's original form was right, and it is the *exact* Mohr-Coulomb correspondence to $0.0000\%$.
>
> 📌 **The mechanism is now unambiguous, and it is the same one as C-134 and C-152:** **I asserted a physical claim from a shape ("a compression cone should close on the compression side") and never evaluated the governing criterion.** ⚠️ The MC plane criterion $\tau=c+P\tan\phi$ settles it in one line: hydrostatic compression is *always* elastic, and **MC's apex is a tensile cutoff.** 📌 **The generalisable form: a yield-criterion sign cannot be reasoned from the geometry of the cone; it can only be settled by comparing against the parent criterion over a range of states.**
>
> ✅ **And this finally answers the question Ruling 27 left open.** After four inversions I had proposed the factory as the remedy. 📌 **The factory is not the remedy — the *test* is.** ⚠️ A factory can be wired wrong (**C-179**) and can be inert where it is tested (**C-180**), exactly as mine was. ✅ **What actually caught the error was a numerical comparison against the parent criterion across a parameter sweep (C-176) — which is the `#[cfg(test)]` limit-case suite, working as intended.** 🔴 **Ruled: the gate is the artefact; the factory is optional.**
>
> ⚠️ **And the honest count for this item: my Ruling 26 and Ruling 27 rulings on the Drucker-Prager sign were both net-negative.** The coefficient chain ($2/3$, $2/[3(3\mp\sin\phi)]$) was right by luck of the Tresca check; **the sign was wrong twice.** ⚠️ **Confidence was not correlated with correctness here** — the same trap that produced C-134, C-121 and C-152, and the reason the limit-case suite must be executable rather than advisory.

---

## Ruling 30 — C-184…C-187: sign confirmed, PCHIP accepted, Lode-angle trades one exactness for another (09-10-2026)

Corrections **C-184 … C-187**. ✅ **The tensile-apex identity and the two-interval PCHIP both verified exactly.** ⚠️ **Two residuals, both about silent trades.**

| # | Correction | Detail |
|---|---|---|
| **C-184** | ✅ **TENSILE APEX IDENTITY CONFIRMED.** $I_1^{\text{apex}}=-k/\alpha=-3c\cot\phi$, $P_{\text{apex}}=I_1/3=-c\cot\phi$ ✅ — the substitution $k=\frac{2c\cos\phi}{3-\sin\phi}$, $\alpha=\frac{2\sin\phi}{3(3-\sin\phi)}$ gives $-\frac{2c\cos\phi/(3-\sin\phi)}{2\sin\phi/[3(3-\sin\phi)]}=-3c\cot\phi$ ✅ **exactly**, and it coincides with the Mohr-Coulomb tensile cutoff $T=c\cot\phi$ ✅. **C-176 independently re-derived from the criterion and stands; §7aa is the current authority** | **CONF-66 item 3 sign — CLOSED, twice-confirmed** |
| **C-185** | ✅ **RANKINE CAP AND ITS WARNING MACHINERY DELETED** ✅ — the cap created a $k=6.928\times10^5$ Pa surface jump at $I_1=0$, and `UncappedDruckerPragerTensileRegime` → `ConvergedOutsideEnvelope` reported a **phantom failure**. **C-178 CLOSED** | **CONF-66 item 3** |
| **C-186** | ✅ **TWO-INTERVAL PCHIP ACCEPTED — VERIFIED CORRECT.** ✅ **The DOF overdetermination of C-181 is resolved**: each interval is an independent cubic (4 DOF) with 4 conditions — interval 1: $c(0)=c_hard$, $c(0.5)=c_{peak}$, $c'(0.5)=0$, $d_0=2\delta H_hard$; interval 2: $c(0.5)=c_{peak}$, $c'(0.5)=0$, $c(1)=c_{soft}$, $d_1=2\delta H_{soft}$ ✅. Measured across the declared $\delta$ range: | $\delta$ | $C^1$ resid @$s=0$ | @$s=0.5$ | @$s=1$ | $\max c/c_{peak}$ | |---|---|---|---|---| | $10^{-4}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ | | $3\times10^{-4}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ | | $10^{-3}$ | $0.00$ | $0.00$ | $0.00$ | $1.0000000000$ | ✅ **$C^1$ at all three nodes, $\max c = c_{peak}$ exactly, zero overshoot.** 🔴 **Also confirmed: the simplified $d\le3\Delta$ clamp IS sufficient here** — I suspected the sharper $\sqrt3\,\Delta$ condition (which applies when $d_{\text{mid}}\ne0$) and **found no counterexample**; that suspicion was mine and is withdrawn ✅. ⚠️ **One residual — C-187** | **CONF-66 item 2 CLOSED** |
| **C-187** | ⚠️ **THE CLAMP IS INACTIVE FOR A LINEAR HARDENING LAW — BUT WHEN IT DOES BITE, IT COSTS $C^1$ AT THE BAND EDGE, SILENTLY.** ✅ Activation condition derived: clamp active iff $\delta H>3(c_{peak}-c_{hard})$. For a **linear** law $c_{peak}-c_{hard}=H\delta\Rightarrow H\delta>3H\delta$ is **false** ⇒ 🔴 **the clamp is provably inactive and $C^1$ is exact for _any_ $\delta$** ✅ — it is a pure safety net. ⚠️ It bites only when the hardening law **flattens toward the peak** so that the local **secant** slope exceeds $3\times$ the **tangent** modulus $H$. Measured: | hardening law | $d_0/\Delta_1$ | clamp | $C^1$ residual @$s=0$ | |---|---|---|---| | linear | 1.000 | no | $0.0$ | | flattening $\times3$ | 3.000 | **YES** | $2.3\times10^{-10}$ | | flattening $\times10$ | 10.000 | **YES** | $\mathbf{4.48\times10^{2}}$ | | flattening $\times30$ | 30.000 | **YES** | $\mathbf{5.76\times10^{2}}$ | 🔴 **The clamp trades the _interior overshoot_ for a _tangent jump at the band edge_.** ⚠️ That is structurally identical to **C-151** (a $\Delta t$ cutback / fallback solver that makes a failure *disappear* rather than be *reported*) and to the **C-93** clamp. ✅ **Ruled: when either clamp activates, emit `ValidityWarning::HardeningMonotonicityClampActive` carrying $\delta$, $d_0/\Delta_1$ and the resulting $C^1$ residual — the clamp may not be silent**, and the quadratic-Newton claim must be suspended for that timestep | **1 gap** |

### §3 — accepted, with one conflation and two residuals

| Item | Verdict |
|---|---|
| ✅ **Lode-angle $g(\theta_L)$ accepted as the remedy** — Bardet 1990 / Ménétrey–Willam ✅ correct citations for the family. ⚠️ **But it does _not_ repair the $-66.67\%$** — that figure is the **mislabelled `Extension` branch of the factory** (C-177), which returns the **von Mises** uniaxial value $2c/\sqrt3$. 🔴 **Conflating them is the third time this round's Lode-angle gap and its factory-branch gap have been merged** (after **C-183** and the §1B of the previous round). $g(\theta_L)$ is a *cap-shape* correction; `MeridianType::Extension` is a *coefficient* error. **They need different fixes.** |
| 🔴 **C-188 — $g(\theta_L)$ DESTROYS the exact meridian match that C-176 verified.** $F=\sqrt{J_2/3}\,g(\theta_L)-\alpha I_1-k$: on the compression meridian $\theta_L$ is meridian-specific, so unless $g\equiv1$ **exactly there**, the measured **$0.0000\%$** meridian agreement is **lost** (deviation up to $\approx5.71\%$ at the $g/\Delta_1=3$ extremum). ✅ **Ruled: this is a _trade_, not a strict improvement.** The spec must declare **which exactness is required** — meridians (current) or $\pi$-plane (with $g$) — and record the other as a **declared limitation**. |
| 🔴 **C-189 — $\theta_L$ IS UNDEFINED AT $J_2=0$.** ⚠️ Every hydrostatic state and the apex itself lie on $J_2=0$, where the Lode angle has no value. ✅ **Ruled: $g$ carries a declared value at $J_2=0$, and any run whose cells land there emits `ValidityWarning::LodeAngleUndefined`** — otherwise the sign of $g$ at the apex is arbitrary and the apex identity of **C-184** is not reproducible. |
| ✅ **C-183 mass-balance test accepted** — mesh reduction vs. drift-flux well integrals are genuinely independent ✅. ⚠️ tolerance still governed by **C-182** ($\kappa\sqrt N\,\epsilon$, $\kappa$ declared) |
| ⚠️ **L-7 added** — DOIs for Bardet (1990) and Ménétrey & Willam; and **Bardet 1990 and Menétrey-Willam are given without years/DOIs**, which is the same provenance gap as **C-31**'s $\alpha_0$ and **L-5**'s $\delta$ |

### Register status after Ruling 30

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-177** factory `Extension` branch is von Mises not MC · **C-179** factory wiring · **C-188** $g(\theta_L)$ trades meridian exactness · **C-189** $\theta_L$ undefined at $J_2=0$) — ⚠️ **items 2 and 3 of the constitutive block are now CLOSED; only the factory and the optional Lode extension remain** · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** (literature parked) · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **CONF-66's constitutive block is now substantially closed.** Items **2** (hardening law — two-interval PCHIP, $C^1$ verified to $0.00\times10^{0}$ at all three nodes, zero overshoot) and **3** (Drucker-Prager sign + coefficients — confirmed twice, from the criterion and from the apex identity) are **done**. ⚠️ What remains is **not physics**: a mislabelled factory branch (**C-177**), its wiring (**C-179**), and the **optional** Lode-angle extension whose cost must be declared (**C-188/C-189**).
>
> 📌 **And the strongest evidence in this item is a *negative* result I should keep visible:** I suspected the submitted $d\le3\Delta$ clamp was too loose (expected $\sqrt3\Delta$), built a counterexample search, **found none, and withdrew the suspicion.** ⚠️ That is the fifth time measurement has withdrawn one of my claims — **C-79b, C-81, C-89b, C-98, C-121×2, C-149, and now this.** ✅ **The asymmetry is durable: measurement withdraws from me roughly as often as it confirms, so the prior worth acting on is "this claim needs a test", never "this claim is probably fine".**
>
> ⚠️ **C-187 is the pattern to watch for in all remaining "safe" mechanisms.** The clamp is **provably inactive** for the declared parameter range ✅ — which is exactly why it is dangerous: a mechanism that never fires cannot be validated by observation, and if it *does* fire (hardening law flatter than $3H$), it silently degrades $C^1$ and the quadratic-Newton claim. ✅ **Rule: an inactive-by-construction safeguard must still carry a warning, because the day it activates is the day its assumption has already failed.**

---

## Ruling 31 — C-190…C-194: clamp warning accepted, Bardet citation corrected, Lode primacy on the WRONG meridian (09-10-2026)

Corrections **C-190 … C-194**. 🔴 **The verified-Bardet citation fails the L-2 test** — wrong title *and* a 404 DOI. 🔴 **And "Compression Meridian Primacy" is anchored to the extension meridian**, under the submission's own Lode formula.

| # | Correction | Detail |
|---|---|---|
| **C-190** | ✅ **C-187 REMEDIATION ACCEPTED, AND MY OWN $5.71\%$ WAS BORROWED FROM THE WRONG MECHANISM.** ✅ Manifest warning `HardeningMonotonicityClampActive { delta, ratio, c1_residual }` ✅ and suspension of the $\mathcal{O}(\|R\|^2)$ claim ✅ — both exactly as ruled. ✅ Activation criterion restated correctly: $\dfrac{d_0}{\Delta_1}=\dfrac{\delta H}{c_{peak}-c_0}>3$. ⚠️ **But: the $5.71$% figure is mine, and it is wrong.** I derived it from the **PCHIP clamp sweep** (§7bb.4, the $d_0/\Delta_1=3$ case) and then re-used it in **C-188** as a *Lode-angle* deviation, writing it "at the $g/\Delta_1=3$ extremum" — which conflates $g$ with $\Delta_1$ and means nothing. 🔴 **Withdrawn.** The actual Lode deviation depends on the chosen $g$ and on whether MC's hexagon is inscribed or circumscribed; it **must be measured against MC's $\pi$-plane hexagon**, not borrowed. **C-188's magnitude is unquantified until that is done** | **C-187 CLOSED · C-188 magnitude withdrawn** |
| **C-191** | 🔴🔴 **"COMPRESSION MERIDIAN PRIMACY" IS ANCHORED TO THE **EXTENSION** MERIDIAN.** The submission declares compression at $\theta_L=+\pi/6$ and sets $g(+\pi/6)\equiv1.0$. 🔴 **Measured under the submission's _own_ formula** $\sin 3\theta_L=-\frac{3\sqrt3}{2}\frac{J_3}{J_2^{3/2}}$, $J_3=\det(s)$: | state (compression-positive) | $J_2$ | $J_3$ | $\theta_L$ | |---|---|---|---| | triaxial **compression** $(2,\ 0.5,\ 0.5)$ | 0.750 | +0.250 | $\mathbf{-30.0000°}=-\pi/6$ | | triaxial **extension** $(0.2,\ 1,\ 1)$ | 0.213 | −0.038 | $\mathbf{+30.0000°}=+\pi/6$ | | pure shear $(1,\ 0,\ -1)$ | 1.000 | 0.000 | $0°$ | 🔴 **So $g(+\pi/6)=1$ anchors primacy to the extension meridian, and the exact Mohr-Coulomb match of C-176 ($0.0000\%$) moves _off_ compression** — silently, because the surface still looks like a sensible yield locus. ✅ **Ruled: `$g(\theta_L)\equiv1$` must be anchored at $\theta_L=-\pi/6$, and the anchoring meridian must be verified by a `#[cfg(test)]` case computing $\theta_L$ at a triaxial-compression state.** ⚠️ The sign of $\theta_L$ flips with the definition of $J_3$ and with the principal-stress ordering convention ($s_1\ge s_2\ge s_3$ vs $\le$) — 📌 **it cannot be asserted, only measured. This is the C-161 / C-176 lesson arriving for the third time, now on a *new* symbol** | **CONF-66 C-188 — C-191 supersedes** |
| **C-192** | 🔴🔴 **THE $J_2$ GUARD THRESHOLD IS DIMENSIONLESS — 9TH DIMENSIONAL FAILURE.** `if j2 <= 1.0e-12 { return 1.0; }` compares a **dimensionless literal** against $J_2$ **in $\mathrm{Pa}^2$**. Measured — the guard **never fires** at any plausible stress scale: | state | deviatoric $J_2$ | guard fires at $10^{-12}$? | |---|---|---| | weak sediment, $p\sim10^4$ Pa | $6.667\times10^{7}\ \mathrm{Pa}^2$ | **NO** | | reservoir rock, $p\sim10^7$ Pa | $6.667\times10^{13}\ \mathrm{Pa}^2$ | **NO** | | deep rock, $p\sim10^8$ Pa | $6.667\times10^{15}\ \mathrm{Pa}^2$ | **NO** | 🔴 **The guard is unreachable, so the $0/0$ it exists to prevent is _not_ prevented.** ⚠️ It would only fire at a $10^{-12}$ Pa deviatoric stress, which is physically meaningless. ✅ **Ruled — guard on a _relative_ stress scale, not an absolute literal:** test $J_2\le\varepsilon\,\sigma_{\text{scale}}^2/3$ with $\sigma_{\text{scale}}$ a declared characteristic stress (mean stress, $\|I_1\|$, or the yield-function's own $k$), and carry $\varepsilon$ as a declared newtype. ⚠️ The already-`.clamp(-1,1)`'d ratio is dimensionless and is the **better** guard: indeterminate **iff** $J_2=0$ exactly, so a relative test is both correct and sufficient | **1 gap** |
| **C-193** | ⚠️ **ONLY ONE BOUND OF $g$ IS CONSTRAINED.** ⚠️ $g(+\pi/6)=1$ fixes one meridian; 🔴 $g(-\pi/6)$ is **unconstrained** — yet a single $g$ chosen to match MC's hexagon necessarily matches **one** bounding meridian and carries the **full** Lode deviation at the other. Which one is the matched bound **is the modelling decision**. ✅ **Ruled: both $g(\pm\pi/6)$ values and the inscribed/circumscribed choice are declared inputs**, and the resulting deviation at the unmatched meridian is **measured and recorded** (pending C-190's measurement) | **1 gap** |
| **C-194** | 🔴 **THE VERIFIED BARDET CITATION FAILS THE L-2 TEST — WRONG TITLE _AND_ A 404 DOI.** ✅ The correct record, retrieved and verified 09-10-2026 via Crossref: | field | submitted | verified | |---|---|---|---| | **DOI** | `10.1115/1.2892023` | ❌ **404** | **`10.1115/1.2897051`** | | **Title** | *Lode angle function for the yield surfaces of soils and rocks* | ❌ **no such record** | ***Lode Dependences for Isotropic Pressure-Sensitive Elastoplastic Materials*** | | Journal | J. Appl. Mech. | ✅ | ✅ | | Volume | 57 | ✅ | **57** | | **Issue** | 2 | ❌ | **3** | | Pages | 498–506 | ✅ | **498–506** | | Year | 1990 | ✅ | 1990 | | Author | Bardet, J. P. | ✅ | ✅ | 🔴 **Two of eight fields wrong, and the DOI does not resolve.** ⚠️ *Note the shape of the failure: volume and pages were correct, so it was not a fabricated reference — it was a **real paper with a corrupted title and a mistyped DOI**, the same class as **L-2**'s `12242 → 12244`.* ✅ **Ménétrey & Willam 1995 verified** — `10.14359/1132` **resolves 200** ✅, *Triaxial Failure Criterion for Concrete and its Generalization*, **ACI Structural Journal 92(3)** ✅, 1995 ✅, title matches ✅. ⚠️ **Page range 311–318 not confirmed from Crossref** — must be checked in the retrieved copy. ✅ **L-7 remains open on Bardet only** | **L-7 narrowed** |

### Accepted unchanged

| Item | Verdict |
|---|---|
| ✅ **§1A clamp analysis** — $d_0/\Delta_1=1.0<3.0$ for linear hardening; the clamp is **inactive-by-construction** ✅ | Matches my §7bb.4 derivation exactly |
| ✅ **§1B warning + solver-claim suspension** | Correct, and it satisfies the C-93 / C-151 rule: the safeguard **may not be silent** |
| ✅ **§2A coefficient-vs-cap disambiguation** — `MeridianType` sets $\alpha,k$; $g$ shapes the cap | ✅ This is the separation I asked for in C-183 — ✅ **and it is stated correctly here, so C-183's third conflation is now resolved** |
| ✅ **§2B Lode-angle formula** $\sin3\theta_L=-\frac{3\sqrt3}{2}\frac{J_3}{J_2^{3/2}}$ with `.clamp(-1,1)` | ✅ Formula and clamping are correct. 🔴 Only the **guard** and the **primacy anchor** are wrong |
| ✅ **§3 Ménétrey & Willam** | ✅ DOI verified live |

### Register status after Ruling 31

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-191** primacy on the wrong meridian · **C-192** unreachable guard, 9th dimensional failure · **C-179** factory wiring · **C-177** factory `Extension` branch) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** (literature parked) · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |

> 📌 **C-194 is a clean specimen of a citation failure that a "looks complete" check would pass.** Eight bibliographic fields were supplied; **six were right**, including the exact page range — so any plausibility or completeness review accepts it. 🔴 **Only the L-2 test catches it: the DOI returned 404.** 📌 And this is **L-2's failure mode repeating verbatim** — *"the DOI was one digit wrong while the paper was real"* (`12242 → 12244`), now `2892023 → 2897051`, with a **title corruption** on top. ✅ **The 404-DOI list in [`literature_todo.md`](literature_todo.md) is extended; and the standing rule is restated: a citation is verified when the DOI resolves and the metadata matches field-for-field — never when it looks complete.**
>
> 🔴 **C-191 is more serious.** "Compression Meridian Primacy" reads like a physics guarantee, and it is implemented — but under the submission's own Lode formula it is anchored to the **extension** meridian, so C-176's exact Mohr-Coulomb match **silently relocates**. ⚠️ Nothing crashes; the yield surface remains a plausible closed locus; the number is simply wrong for the loading that matters. 📌 **Third time this register has been caught by the same mechanism** (C-161 $\sqrt3$, C-176 sign, now $\theta_L$) — **a sign/index convention asserted in prose and never computed.** ✅ **Ruled: every $\pm$ constant in a constitutive expression gets a `#[cfg(test)]` case that evaluates it at a state whose value is known independently.** That is now a **standing CI requirement**, not a per-item fix.
>
> 🔴 **C-192 makes it nine dimensional failures** — and this one is in the *guard clause*, i.e. in the code written specifically to be safe. 📌 **Pattern: safety code is written with fewer dimensional checks than physics code**, because it is treated as plumbing. ✅ **Ruled: the `#[cfg(test)]` dimensional gate covers guard/threshold constants too — a bare numeric literal in any comparison against a physical quantity is a gate failure, whether or not it sits on the yield function.**

---

## Ruling 32 — C-195…C-198: theta_L = -pi/6 proven, guard is dimensionally sound but unguarded (09-10-2026)

Corrections **C-195 … C-198**. ✅ **C-191's derivation is exact and closes it.** ✅ **Dimensional failure #9 is genuinely fixed.** 🔴 **But the fix introduces a new silent-wrong-answer path, and a required warning has now been omitted three rounds running.**

| # | Correction | Detail |
|---|---|---|
| **C-195** | ✅ **$\theta_L=-\pi/6$ PROVEN — C-191 CLOSED.** The submitted derivation is **complete and correct**: compression-positive triaxial compression $\sigma_1>\sigma_2=\sigma_3$ gives $s=(\tfrac23,-\tfrac13,-\tfrac13)\Delta\sigma$, $J_2=\tfrac13\Delta\sigma^2$, $J_3=+\tfrac{2}{27}\Delta\sigma^3$, hence $\sin3\theta_L=-\tfrac{3\sqrt3}{2}\cdot\frac{2/27}{(1/3)^{3/2}}=-1$ and $\theta_L=-\pi/6$ ✓. ✅ **Independently reproduced numerically** to full double precision, and it agrees with my C-191 measurement exactly. ✅ **The extension side is also correct as stated**: compression-positive extension gives $J_3<0\Rightarrow\theta_L=+\pi/6$ ✓. ✅ **Locked:** $g(\theta_L)\equiv1$ **at $\theta_L=-\pi/6$**, with a `#[cfg(test)]` case computing $\theta_L$ at a triaxial-compression state | **CONF-66 C-188 anchor — CLOSED** |
| **C-196** | ✅ **DIMENSIONAL FAILURE #9 IS FIXED** ✅ — $j_{2,\text{norm}}=J_2/\sigma_{\text{scale}}^2$ is **dimensionless**, so the $10^{-12}$ literal is now meaningful. ✅ Measured firing behaviour — the guard triggers when the deviatoric stress drops below $\sim1.7\times10^{-6}$ of the scale, at **every** stress scale from $10^4$ to $10^8$ Pa ✓: | $\sigma_{\text{scale}}$ | $J_2$ at threshold | $\Delta\sigma$ at threshold | relative deviator | |---|---|---|---| | $10^4$ Pa | $10^{-4}\ \mathrm{Pa}^2$ | $1.73\times10^{-2}$ Pa | $1.73\times10^{-6}$ | | $10^7$ Pa | $10^{2}\ \mathrm{Pa}^2$ | $17.3$ Pa | $1.73\times10^{-6}$ | | $10^8$ Pa | $10^{4}\ \mathrm{Pa}^2$ | $173$ Pa | $1.73\times10^{-6}$ | ✅ **And the direction of safety is correct**: pure f64 round-off leaves a "hydrostatic" state with $j_{2,\text{norm}}\sim10^{-32}$, far below the trigger — so the guard fires **before** the 0/0, not after. 🔴 **But it introduces a new silent-wrong-answer path (a), and the C-189 warning is still missing (c)** | **1 gap + 1 process gap** |
| **C-197** | 🔴 **FREE ANCHORS ALLOW A RE-ENTRANT CAP — INV-7 FORBIDS CLAMPING, SO IT MUST FAIL LOUDLY.** ✅ §3 correctly exposes $g(\pm\pi/6)$ and the inscribed/circumscribed mode as **declared DTO inputs** — and ✅ that finally satisfies C-193. 🔴 **But unconstrained anchors can produce a non-convex, re-entrant cap:** with $g(-\pi/6)=0.4$, $g(+\pi/6)=1.6$ the cap is radially inverted (hourglass), which (i) is **non-convex** and (ii) can make the **plastic dissipation non-negative-violating**, so the **CPPM return mapping stops being well posed**. ✅ **Ruled: the engine validates that the configured anchors yield a convex, closed, positive-dissipating cap and calls `error_handler`-level failure (INV-1: fail loudly, never fall back) — it may _not_ clamp, reorder, or silently substitute $g\equiv1$.** ⚠️ And since the anchors are now inputs, C-191's primacy is **no longer a constant**: ✅ **Ruled: the _default_ is the measured $-\pi/6$ anchor, recorded in the manifest and pinned by a `#[cfg(test)]` case**, so a run cannot silently choose a different meridian to be exact on | **1 gap** |
| **C-198** | ✅ **BARDET 1990 CORRECTED — L-7 CLOSED.** ✅ The submitted record now matches my verified Crossref record **field for field**: *Lode Dependences for Isotropic Pressure-Sensitive Elastoplastic Materials*, **ASME J. Appl. Mech. 57(3)**, 498–506, 1990, Bardet J. P., **`10.1115/1.2897051`** ✓. ✅ **Ménétrey & Willam 1995** verified separately (`10.14359/1132`, resolves 200). ⚠️ **Only residual:** pp. 311–318 for Ménétrey–Willam unconfirmed in Crossref — confirm in the retrieved copy. ✅ **Both §5 gates endorsed** — ⚠️ with the note that **Gate 2 is largely subsumed by the C-124 newtype discipline**: with `#[repr(transparent)]` newtypes the *compiler* already rejects a dimensional comparison against a bare literal. ✅ **Gate 2's real value is the case where the newtype discipline is broken** (raw `f64` sneaking in), so it must also catch `f64 × f64` mixing, not just literal comparisons | **L-7 CLOSED** |

### 🔴 C-199 — a required warning has been omitted three rounds running

🔴 `ValidityWarning::LodeAngleUndefined` was ruled in **C-189** (Ruling 30), re-stated as part of **C-188**, and is **still absent** from the submitted guard in Ruling 32. ⚠️ The guard returns $1.0$ **silently** at $J_2=0$.

📌 **This is error class B — _making the failure disappear rather than be reported_ — but occurring in the _specification_ rather than in code**, and it is the same structure as **C-93** (clamp) and **C-151** ($\Delta t$ cutback / fallback solver).

✅ **Ruled: any singularity guard that substitutes a value must emit its paired `ValidityWarning`. The two travel together as a single rule, recorded in [`engine_invariants.md`](engine_invariants.md) §7s.6.**

⚠️ **And the mechanism of the omission is worth naming, because it is not carelessness.** 📌 In each round the guard was rewritten *in response to a different finding* (#9 dimensional → now a division hazard; then a missing warning), and each rewrite **replaced the previous round's text wholesale**. ✅ **Ruled: corrections to a guarded expression are _cumulative_ — a rewrite must carry forward every prior requirement on that expression, and §7s.6 is the checklist that is re-verified whenever that block is touched.**

### Register status after Ruling 32

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-196** unguarded `sigma_scale` division · **C-197** anchor convexity · **C-199** warning omitted ×3) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** (literature parked) · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | L-1 … L-6 open · **L-7 CLOSED** ✅ · L-8 (Ménétrey–Willam pages, confirm in copy) |

> 📌 **C-195 completes the $\pm$-convention trilogy.** **C-161** ($\sqrt3$), **C-176** ($I_1$ sign), **C-191** ($\theta_L$) — each was a sign or index asserted in prose, each caught by evaluating at a state whose value is known independently, and **each now has a `#[cfg(test)]` case that will keep it caught**. ✅ **The standing requirement has paid for itself twice: the engineer derived $-\pi/6$ from first principles in Ruling 32 and it matched my measurement exactly, which is the first time a correction and its check have agreed without either being the other's source.**
>
> 🔴 **C-199 is the more important finding, and it is a defect in _my_ process.** A warning I ruled has been dropped from three consecutive revisions of the same block. 📌 **Each revision was responding to a _different_ finding and replaced the text wholesale, discarding the previous round's requirement.** ⚠️ This is a predictable failure of wholesale block replacement, and it is invisible to any check that only verifies "is the newest finding fixed?".
>
> ✅ **Two durable rules adopted: (a) §7s.6 — a singularity guard and its `ValidityWarning` are one indivisible unit; (b) edits to a guarded expression are _cumulative_, and its checklist is re-verified on every touch.**
>
> ⚠️ **Both are process rules, not physics.** 📌 That is the signal: the physics of this component is now settled enough that the remaining defects are in **how corrections are carried forward** — which is a different failure class from the one that produced C-161 through C-197, and arguably the one more likely to recur.

---

## Ruling 33 — C-200…C-204: guard and validation accepted; two of the NEW requirements are wrong (09-10-2026)

Corrections **C-200 … C-204**. ✅ **C-196, C-197, C-199 and C-198 all close.** 🔴 **But the convexity test and the dissipation inequality are both incorrect as written** — and the first would accept a non-convex surface, the second would reject all normal plastic flow.

| # | Correction | Detail |
|---|---|---|
| **C-200** | 🔴 **THE $J_2$ FORMULA IS GARBLED, AND ITS STATED FAILURE MECHANISM IS UNREACHABLE.** ⚠️ Submitted: $\tfrac16[(\sigma_{11}-\sigma_{22})^2+(\sigma_{23})^2+(\sigma_{31})^2]+\sigma_{12}^2+\sigma_{23}^2+\sigma_{31}^2$ — 🔴 **the three shear terms appear twice**, once inside the bracket and once outside. ✅ **Correct:** $J_2=\tfrac12 s_{ij}s_{ij}$, equivalently $\tfrac16\sum_{\text{cyc}}(\sigma_i-\sigma_j)^2$ in principal form. 🔴 **And $J_2<0$ cannot occur.** Both routes are **sums of squares**, so every term is $\ge0$; summing non-negative terms cannot go negative, and overflow yields $+\infty$, never $-\infty$. Measured: **0 negative values in 200 000 pseudo-random states.** ✅ **Ruled: keep the `j2_pa2 < 0.0` check — it is cheap insurance against a *badly implemented* $J_2$, which is a real risk — but record that the f64-truncation mechanism is not one, so it does not become a phantom entry in the risk register** (the **C-178** lesson) | **1 correction** |
| **C-201** | 🔴 **THE NaN LEAK IS WORSE THAN DESCRIBED — IT IS SILENT NaN, NOT A BOUNDED WRONG ANGLE.** ⚠️ The submission states `.clamp(-1,1)` *"returns $-1.0$ or $1.0$ depending on NaN propagation rules."* 🔴 **Wrong.** Rust's `f64::clamp` is `if self<min{min} else if self>max{max} else{self}` — with `self=NaN$, **both comparisons are false**, so it returns **`self`** = **NaN**. 📌 So $0/0$ yields **NaN** $\Rightarrow\theta_L=$ NaN $\Rightarrow g=$ NaN $\Rightarrow f=$ NaN $\Rightarrow\mathbf{D}^{alg}=$ NaN $\Rightarrow$ **Newton residual = NaN**. 🔴 **And NaN compares `false` against _every_ guard**, so each downstream validation **silently passes**. ✅ **Ruled: the test that must catch this is a _NaN-propagation_ assertion** (assert `is_finite()` on $f$ and on $\mathbf{D}^{alg}$ after return mapping), **not a range check** — a range check passes on NaN | **C-196 path — guard itself ✅ accepted** |
| **C-202** | 🔴🔴 **THE DISSIPATION INEQUALITY IS SIGN-INVERTED — IMPLEMENTED AS WRITTEN IT REJECTS ALL NORMAL PLASTIC FLOW.** The submission states Hill's maximum plastic dissipation as $\dot w^p=\boldsymbol\sigma:\dot\boldsymbol\varepsilon^p\,\mathbf{<}\,0$. ✅ **Correct: $\ge0$.** Derived: with associated flow $\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,\partial f/\partial\boldsymbol\sigma$, using $\boldsymbol\sigma:s=2J_2$ and $\partial J_2/\partial\boldsymbol\sigma=s$: $$\boldsymbol\sigma:\frac{\partial f}{\partial\boldsymbol\sigma}=\frac{J_2}{\sqrt{J_2/3}}-\alpha I_1=\sqrt{J_2/3}+f+k\quad\Longrightarrow\quad\text{on }f=0:\quad\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,k\ \ge0$$ Measured: $\dot\gamma=10^{-6},10^{-3},1$ $\Rightarrow$ $+0.693$, $+692.8$, $+6.928\times10^5$ Pa — **all non-negative** ✅. 🔴 **A check written as "$<0\Rightarrow\text{NonConvexYieldSurface}$" would fire on every elastic-perfectly-plastic increment.** ⚠️ **And this matters more here than in an associated model, because this spec uses _non-associated_ flow ($\psi\ne\phi$), where $\dot\gamma k$ is _not_ guaranteed** — so the check is genuinely needed, and its sign must be right. ✅ **Ruled: $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p\ge0$; under non-association the *dissipated* part is $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p-\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^{p,\text{elastic}}$, which is the quantity to bound** | **C-197 conditional** |
| **C-203** | 🔴🔴 **THE CONVEXITY CONDITION IS WRONG — IT MISSES THE SQUARE _AND_ THE DERIVATIVE TERM.** ⚠️ Submitted: $g(\theta_L)+\tfrac{\mathrm{d}^2g}{\mathrm{d}\theta_L^2}\ge0$. ✅ **Derivation.** At fixed $I_1$ the locus is $r(\theta)=\frac{\alpha I_1+k}{g(\theta)}=\frac{C}{g}$. A polar curve is convex iff $r^2-2rr''+6r'^2\ge0$. Substituting $r=C/g$, $r'=-Cg'/g^2$, $r''=C(2g'^2-gg'')/g^3$: $$\left(r^2-2rr''+6r'^2\right)\frac{g^4}{C^2}=\boxed{g^2+2(g')^2+2g\,g''}$$ 🔴 **Counterexample:** $g=1-\tfrac12\theta^2$ — submitted test gives $g+g''=1-1=\mathbf{0}\ge0$ (**passes**) while the correct test gives $1+0-2=\mathbf{-1}<0$ (**fails**) — and $r=C/(1-\tfrac12\theta^2)$ widens with $|\theta|$, a **non-convex peanut**. A second counterexample, $g=\cos\theta$, gives $g+g''\equiv0$ (passes) and $g^2+2g'^2+2gg''=-1$ (fails), with $r=C\sec\theta$ degenerating to the straight line $x=C$. ✅ **Ruled: $g^2+2(g')^2+2g\,g''\ge0$ on $[-\pi/6,+\pi/6]$, sampled and asserted, with $\varepsilon$ margin.** ⚠️ **Note the pattern: this is the fourth time a *convexity/monotonicity* criterion has been written in the simplest plausible form and been wrong** — **C-93** (clamp), **C-181** (Fritsch–Carlson), **C-187** ($3\Delta$), now **C-203** | **C-197 conditional** |
| **C-204** | 🔴 **THE `.clamp(-1,1)` ON THE LODE RATIO IS ITSELF A SILENT CLAMP — §7s.6 APPLIES TO IT TOO.** ⚠️ If $\left|\tfrac{3\sqrt3\,J_3}{2J_2^{3/2}}\right|>1$, the deviatoric state is **physically impossible** (the identity $|J_3|\le\tfrac{2}{3\sqrt3}J_2^{3/2}$ is violated), yet the clamp saturates and the run proceeds on a **fabricated** Lode angle. 🔴 **No measurement or validation can distinguish it afterwards.** ✅ **Ruled: saturation emits `ValidityWarning::LodeRatioSaturated { ratio }`, and per §7s.6 the clamp and its warning are one unit** — 📌 **the rule generalises from _singularity guards_ to _every saturating transform_**, which is the class that produced **C-93**, **C-151** and this one | **1 gap** |

### Closed this round

| Item | Verdict |
|---|---|
| ✅ **§1 validation order** — finite $\to\sigma_{\text{scale}}>0\to j_2\ge0\to$ relative guard $\to$ ratio | ✅ Correct order: nothing that can produce `inf` or `NaN` is reached before validation. **C-196 CLOSED** |
| ✅ **`Result<(f64, Option<ValidityWarning>), TypedError>`** | ✅ The warning now travels **with** the value, so §7s.6 is structurally enforced rather than merely required. **C-199 CLOSED** |
| ✅ **§2B INV-7 compliance** — accept arbitrary input, never silently substitute $g\equiv1$ | ✅ Correct reading of INV-7 alongside INV-1 |
| ✅ **§2 compression-meridian primacy as default** | ✅ matches **C-195** |
| ✅ **§3 Rule §7s.6 formalised** as a standard invariant with a CI build failure | ✅ Adopted verbatim, including the **cumulative-edit** clause — 🔴 **C-199's second half was exactly the missing piece** |
| ✅ **§4 Gate 2 expansion** — bare literals vs. newtypes, and `Pascals × Pascals` must yield `PascalsSquared` | ✅ Accepted. ⚠️ Requires the C-124 newtype layer to define **squared/exponent** types, not just SI base types — otherwise the second rule is unenforceable |

### Register status after Ruling 33

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-202** dissipation sign · **C-203** convexity condition · **C-204** silent ratio clamp · **C-200** garbled $J_2$) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | L-1…L-6, L-8 open · **L-7 CLOSED** ✅ |

> 📌 **The most useful thing in this round is a pattern, and it is now four deep.** **Every convexity or monotonicity criterion in this component has been written in the simplest plausible form and has been wrong:** **C-93** (clamp), **C-181** (Fritsch–Carlson on non-monotone data), **C-187** ($3\Delta$ vs. the sharper two-sided condition), **C-203** ($g+g''$ instead of $g^2+2(g')^2+2gg''$).
>
> ✅ **Each was caught by _deriving the criterion from the geometry_ rather than by recalling it.** 📌 **And each recalled version was confidently wrong** — the same pattern as **C-134**, **C-121**, **C-152** and **C-191**.
>
> 🔴 **So the rule to add to the gate is specific, not general:** ⚠️ *do not write a convexity/monotonicity/saturation criterion from memory.* ✅ **Derive it from the defining geometry of the object — polar curve, Hermite segment, cone meridian — and record the derivation in the same commit as the criterion.** 📌 **§7cc.6's "evaluate every $\pm$ constant at an independently-known state" generalises to "evaluate every _inequality_ on its own defining geometry."**
>
> ⚠️ **And C-204 shows §7s.6 needs a wider scope.** It was written for _singularity guards_; the Lode-ratio clamp is a **saturating transform**, and it fails the same way — silently, with no measurement able to distinguish it afterwards. ✅ **The rule is now generalised to every saturating transform**, which is precisely the class that produced **C-93**, **C-151** and this.

---

## Ruling 34 — C-203 RETRACTED; C-205…C-208: the dissipation check is right-signed but measures the wrong quantity (09-10-2026)

Corrections **C-205 … C-208**. 🔴 **My C-203 was wrong and is withdrawn.** 🔴 **The dissipation sign is now correct, but the replacement check still does not detect the failure it targets.**

| # | Correction | Detail |
|---|---|---|
| **C-205** | 🔴🔴 **C-203 RETRACTED. The submitted criterion $g^2+gg''\ge0$ is CORRECT; my $g^2+2(g')^2+2gg''$ was the wrong one.** ✅ The polar-curvature formula $\kappa=\dfrac{r^2+2(r')^2-rr''}{[r^2+(r')^2]^{3/2}}$ is the **standard** one for $r=r(\theta)$, and with $r=C/g$, $r'=-Cg'/g^2$, $r''=\frac{C}{g^3}[2(g')^2-gg'']$ the numerator reduces to $\frac{C^2}{g^4}[g^2+gg'']$ ✅ — **every algebraic step of the submission is right.** 🔴 **My $r^2-2rr''+6(r')^2\ge0$ is the criterion for convexity of $1/r^2$** — i.e. of the **polar-dual** body — a different object. 🔴 **And my "counterexample 1" was evaluated at $\theta=0$, where the curvature numerator is exactly $0$ (a flat spot).** Measured across the interval for $g=1-\tfrac12\theta^2$: $\kappa_{num}=-8.9\times10^{-5}$ (0°), $-1.60\times10^{-2}$ (10°), $-7.35\times10^{-2}$ (20°), $\mathbf{-2.13\times10^{-1}}$ (30°) — criterion A **catches it** ❌→✅. ⚠️ **And "counterexample 2" ($g=\cos\theta$) is degenerate but _convex_:** $r=C\sec\theta\Rightarrow x=C$, a straight line, which *is* a convex set — so $\ge0$ is right to accept it, and my criterion rejected a legitimate locus. 📌 **The error class: I picked the wrong object before deriving, so the derivation was faithful to the wrong question** — the same shape as **C-176** (wrong MC line) and **C-121** | **C-203 WITHDRAWN** |
| **C-206** | ⚠️ **RESIDUAL ON C-205 — THE CORRECT CRITERION ADMITS _DEGENERATE_ LOCI; A STRICTER ONE IS WARRANTED.** ✅ $g^2+gg''\ge0\Leftrightarrow\kappa\ge0$ means _the curve does not reverse curvature_. It therefore **accepts a flat sub-interval** ($\kappa\equiv0$), which is convex as a set but **unphysical for a pressure-sensitive rock**: Mohr-Coulomb's $\pi$-plane is a **hexagon**, whose curvature is non-zero except at its vertices. ✅ **Ruled: require $\kappa>0$ strictly, or $\kappa\ge\kappa_{\min}$ with $\kappa_{\min}$ declared** — ⚠️ this is a **refinement of a correct criterion, not a fix to a wrong one**, and must be recorded as such so it is not re-litigated | **1 refinement** |
| **C-207** | 🔴🔴 **C-202 REMAINS OPEN: THE REPLACEMENT CHECK MEASURES THE WRONG QUANTITY AND MISSES THE APEX ENTIRELY.** ✅ §1A's diagnosis is correct and §1B's sign is now right — ✅ **but $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p\ge0$ is a _necessary_ condition only.** 📌 Under **non-associated** flow $\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,\partial g/\partial\boldsymbol\sigma$; the **dissipated** part is the mechanical work **minus the hardening stored energy**: $$\boxed{\dot D^p=\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p-\dot E^{p,\text{stored}}\ \ge0,\qquad \dot E^{p,\text{stored}}=H\,(\bar\varepsilon^p)^{n}\ \text{(declared)}}$$ 🔴 **Measured — and the failure is worst exactly at C-184's apex.** With $\psi=0$ ($g=\sqrt{J_2/3}$, so $\partial g/\partial\boldsymbol\sigma=s/2\sqrt{J_2/3}$ and $\boldsymbol\sigma:\partial g/\partial\boldsymbol\sigma=\sqrt{J_2/3}$), linear hardening $H=3.2\times10^5$ Pa, $\bar\varepsilon_p^{peak}=0.05$, $\dot\gamma=1$: | deviator | $\sqrt{J_2/3}$ | $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p$ | submitted $\ge0$? | true $\dot D^p\ge0$? | |---|---|---|---|---| | $2.0$ | $6.67\times10^{-1}$ | $6.67\times10^{-1}$ | **PASS** | **FAIL** | | $2.0\times10^{-4}$ | $6.67\times10^{-5}$ | $6.67\times10^{-5}$ | **PASS** | **FAIL** | | $0$ (**apex**) | $0$ | $\mathbf{0}$ | **PASS** | **FAIL** | 🔴 **At the apex the mechanical work vanishes, so the submitted check passes with equality — while the true dissipation is $-\dot E^{p,\text{stored}}=-1.6\times10^4$ Pa, negative over the entire near-apex band.** 📌 **The second term cannot be omitted, because at the apex it is the _only_ term left.** Magnitude: $2.3\%$ of cohesion — ⚠️ **not a rounding effect, a sign error in the energy budget.** ✅ **Ruled: both terms are computed; `TypedError::NegativePlasticDissipation` fires on $\dot D^p<0$, not on $\boldsymbol\sigma:\dot{\boldsymbol\varepsilon}^p<0$** | **C-202 STAYS OPEN** |
| **C-208** | ✅ **C-200 HALF WITHDRAWN; §1B/§3/§4A/§7s.6 accepted.** ✅ The §4A tensor formula $\tfrac16\sum(\sigma_i-\sigma_j)^2+\sigma_{12}^2+\sigma_{23}^2+\sigma_{31}^2$ is the **correct** general form — ✅ **the "garbled formula" half of my C-200 is withdrawn**; the engineer had already corrected it. ✅ **The unreachability half stands**: $J_2<0$ cannot arise, **0/200 000** measured, and the check is retained purely as defensive code against a badly-implemented $J_2$, **without** recording a phantom physical risk ✅. ✅ **§3's `is_finite()` gating accepted verbatim** — `f_yield.is_finite()` plus `d_alg.iter().all(is_finite)` ✅ (**C-201 CLOSED**); ⚠️ extend to $\dot\gamma$ and $\boldsymbol\sigma$, which enter the same non-finite path. ✅ **§7s.6 generalised to every saturating transform** — accepted as written, including `DeviatoricStateSaturated { raw_value, clamped_value }` ✅ and the static CI build failure | **C-201 CLOSED · C-200 narrowed** |

### Register status after Ruling 34

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-207** dissipation measures the wrong quantity · **C-206** strict-convexity refinement) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | L-1…L-6, L-8 open · **L-7 CLOSED** ✅ |

> 🔴 **C-205 is the most instructive correction I have made in this register, because it was mine and it was structural rather than arithmetic.**
>
> ⚠️ **I did not make a sign error or a coefficient error. I chose the _wrong object_ — the polar-dual body instead of the locus — and then derived faithfully.** 📌 **Every step of my algebra was correct for the question I asked, and the question was wrong.** That is a different failure from C-176 (a mis-signed MC line) and C-121, and it is **harder to catch**, because a derivation that is internally consistent produces no signal.
>
> 🔴 **And my counterexample was evaluated at $\theta=0$ — the single point where the curvature numerator vanishes.** 📌 **A counterexample found at one point of an interval, for a criterion claimed over the whole interval, must be sampled over the interval before it is reported.** I did not, and had I not re-run it across $[0,\pi/6]$, a correct criterion would have been "corrected" into a wrong one.
>
> ✅ **Two rules adopted, and they are new: (a) before deriving a criterion, name _the object whose convexity/monotonicity is being claimed_ and check the derivation is about that object; (b) a counterexample asserted over an interval must be _swept_ over that interval.**
>
> 📌 **And note the symmetry with C-207:** ⚠️ **C-205 was me replacing a correct criterion with a wrong one after only one evaluation point; C-207 is a submitted criterion that is right in _sign_ and wrong in _what it measures_.** 🔴 **In both cases the criterion is syntactically fine and semantically off-target** — which is the failure mode that a `#[cfg(test)]` case written from the *derivation* (rather than from the *recalled formula*) would catch, because it forces the object to be named.

---

## Ruling 35 — C-209…C-213: convexity confirmed; the stored-energy code uses the wrong rate variable (09-10-2026)

Corrections **C-209 … C-213**. ✅ **C-205, C-206, C-208 accepted as written.** 🔴 **The C-207 code substitutes $\dot\gamma$ for $\dot{\bar\varepsilon}_p$** — which is exactly the term that decides the sign near the apex.

| # | Correction | Detail |
|---|---|---|
| **C-209** | ⚠️ **$N_\kappa(0)$ IS $0$, NOT $1.0$ — AND $\theta=0$ IS EXACTLY THE FLAT SPOT.** ⚠️ §1A states $N_\kappa(0)=1.0>0$. 🔴 Measured: $g=1-\tfrac12\theta^2\Rightarrow g(0)=1$, $g''(0)=-1$, so $N_\kappa(0)=1+1(-1)=\mathbf{0}$ exactly. | full sweep | $N_\kappa(0)=+0.000000$ · $N_\kappa(15°)=-0.033095$ · $N_\kappa(30°)=-0.118288$ | 📌 **Consequence, and it strengthens the ruling rather than weakening it:** §1B's floor $N_\kappa\ge\kappa_{\min}>0$ rejects this $g$ at **$\theta=0$** as well as at $\pi/6$ — 🔴 **and $\theta=0$ is exactly the Lode angle the specification pins to the compression meridian** (**C-195**). ⚠️ So any user $g$ with a flat spot at $\theta_L=0$ fails **at the primary meridian**, and an error message quoting only $\pi/6$ would misdirect the diagnosis. ✅ **Ruled: the failing $\theta_L$ is reported, not just the verdict** | **1 correction** |
| **C-210** | 🔴🔴 **THE FLOOR IS WRITTEN ON $N_\kappa$ BUT CALLED $\kappa_{\min}$ — THESE ARE DIFFERENT QUANTITIES, AND THE SUBSTITUTED ONE IS _STRESS-DEPENDENT_.** §1B writes $g^2+gg''\ge\kappa_{\min}=10^{-4}$ and calls it "strict curvature floor $\kappa\ge\kappa_{\min}$". 🔴 But $$\kappa=\frac{(C^2/g^4)\,N_\kappa}{[r^2+(r')^2]^{3/2}},\qquad C=\alpha I_1+k\ \text{(Pa)}$$ 📌 **$N_\kappa$ is dimensionless and stress-independent; $\kappa$ carries $\mathrm{Pa}^{-1}$ and scales as $C^2\sim(\text{stress})^2$.** $C=\alpha I_1+k$ spans $[0,2k]$ across the locus, so 🔴 **a threshold stated on $\kappa$ is a _different check at every $I_1$_: it rejects weak states and passes strong ones for the _same_ $g$.** ✅ **Ruled: the floor is on $N_\kappa$, named `MIN_CURVATURE_NUMERATOR`; if a true curvature floor is wanted it must be normalised, $\kappa/\kappa_{MC}$ at the same state.** ⚠️ This is the **10th dimensional failure** — a $\mathrm{Pa}^{-1}$ quantity and a dimensionless one sharing one symbol | **1 gap** |
| **C-211** | 🔴🔴 **THE SUBMITTED CODE USES $\dot\gamma$ WHERE THE HARDENING LAW IS DEFINED AGAINST $\dot{\bar\varepsilon}_p$ — AND THAT TERM DECIDES THE SIGN NEAR THE APEX.** `stored_energy = hardening_modulus_h * equiv_plastic_strain * delta_gamma`. 🔴 $H=\mathrm{d}c/\mathrm{d}\bar\varepsilon_p$ is defined against $\bar\varepsilon_p$, and $E^{p,\text{stored}}=\int_0^{\bar\varepsilon_p}H\,\mathrm{d}x$ gives $$\dot E^{p,\text{stored}}=H(\bar\varepsilon_p)\,\bar\varepsilon_p\,\dot{\bar\varepsilon}_p\qquad\text{(linear }H\text{)}$$ 🔴 **$\dot{\bar\varepsilon}_p=\mu\,\dot\gamma$ with $\mu\ne1$ in general** — the flow direction's norm sets $\mu$. Measured $\mu=1/\|\partial g/\partial\boldsymbol\sigma\|$ over a factor-2…4 range of $\|\partial g/\partial\boldsymbol\sigma\|$: | $\|\partial g/\partial\boldsymbol\sigma\|$ | $\mu$ | $H\bar\varepsilon_p\mu$ | |---|---|---| | 0.5 | 2.0 | $3.2\times10^{4}$ Pa | | 1 | 1.0 | $1.6\times10^{4}$ Pa | | 2 | 0.5 | $8.0\times10^{3}$ Pa | | 4 | 0.25 | $4.0\times10^{3}$ Pa | 📌 **At the apex the mechanical work is $0$, so the stored term _alone_ decides the sign** — and it is off by $1/\mu$, a factor of 2–4. $\dot D^p$ at the apex: $-4.0\times10^{3}$ ($\mu{=}0.25$) … $\mathbf{-1.6\times10^{4}}$ ($\mu{=}1$) … $-3.2\times10^{4}$ ($\mu{=}2$) Pa — **0.58 % to 4.62 % of cohesion.** ✅ **Ruled: the stored term uses $\dot{\bar\varepsilon}_p$, with $\mu$ declared as $\dot{\bar\varepsilon}_p/\dot\gamma$ and recorded in the manifest** | **C-207 STAYS OPEN** |
| **C-212** | ⚠️ **TWO SMALLER CORRECTIONS IN §2A.** 🔴 **(a)** The chain $\rho\frac{\partial\psi^p}{\partial\bar\varepsilon_p}\dot{\bar\varepsilon}_p = H(\bar\varepsilon_p)\bar\varepsilon_p\dot{\bar\varepsilon}_p$ is **inconsistent**: $\frac{\partial\psi^p}{\partial\bar\varepsilon_p}=\frac{\mathrm dE^p}{\mathrm d\bar\varepsilon_p}=H$ (not $H\bar\varepsilon_p$) — $H\bar\varepsilon_p$ is $\psi^p$ _itself_, i.e. the energy, not its derivative. ✅ **The final boxed form is correct for linear $H$**; only the intermediate equality is wrong. 🔴 **(b)** *"the stress tensor is purely hydrostatic $\boldsymbol\sigma\to-p\mathbf I$"* — 🔴 under the **compression-positive** convention of **C-176/C-184** it is $\boldsymbol\sigma\to\mathbf+p\mathbf I$. ⚠️ **Ninth appearance of a compression/tension sign slip in this component**, and the last three all survived into submitted text | **2 corrections** |
| **C-213** | ⚠️ **$\kappa_{\min}=10^{-4}$ IS AN UNPROVENANCED DEFAULT, AND THE CPPM FLIP-FLOP CLAIM IS ASSERTED, NOT MEASURED.** ⚠️ The value is a bare literal: ✅ **keep it**, but ✅ **Ruled: declared as a named input with a documented default and recorded in the provenance list as `SOURCE_PENDING`** — 📌 the **same class as $\delta$ (L-5)** and the $\alpha_0$ table (**L-1**): a number that works and whose origin nobody can cite. ⚠️ **New literature item L-9.** ⚠️ The claim that flat facets cause CPPM sub-iteration flip-flop is **plausible and unrejected**, but it is a recalled assertion: ✅ **per the standing rule it must be demonstrated by a gate** — a synthetic flat-facet $g$ run through CPPM, with the sub-iteration count recorded. 📌 **Every recalled mechanism in this component has needed this; none has been assumed sound on recall** | **1 gap** |

### Closed / accepted

| Item | Verdict |
|---|---|
| ✅ **§1A** convexity criterion $N_\kappa=g^2+gg''\ge0$ | ✅ Confirmed — **C-205** |
| ✅ **§1B** strict floor + `Err(NonConvexYieldSurface)` under INV-1 | ✅ Adopted, ⚠️ subject to **C-210** (the quantity) |
| ✅ **§2A** apex argument — mechanical work $\to0$ at the apex, check returns a false PASS, $\dot D^p=-H\bar\varepsilon_p\dot{\bar\varepsilon}_p<0$ | ✅ Correct in substance and **matches my measurement exactly** ($0\ge0$ passes; $-1.6\times10^4$ Pa fails) |
| ✅ **§3A** `is_finite()` extended to $\dot\gamma$, $\boldsymbol\sigma$, $\mathbf{D}^{alg}$ | ✅ **C-208 CLOSED** |
| ✅ **§3B** §7s.6 extended to `PlasticMultiplierSaturated` and `DeviatoricStateSaturated` | ✅ Adopted. ⚠️ **Field names should match what is actually clamped** — in the Lode guard the clamped quantity is the **ratio**, not $J_2$ |

### Register status after Ruling 35

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-211** stored-energy rate variable · **C-210** floor on the wrong quantity · **C-212** two sign/derivative slips · **C-213** $\kappa_{\min}$ provenance + CPPM claim ungated) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | L-1…L-6, L-8, **L-9** open · **L-7 CLOSED** ✅ |

> 📌 **C-209 accidentally produced the strongest form of the C-206 argument.** The floor $\kappa_{\min}>0$ rejects the flattening $g$ at **$\theta_L=0$** — and $\theta_L=0$ is the **compression meridian**, the state **C-195** pinned as primary. 📌 **So the unphysicality is not confined to the extension side: it lands on the meridian the whole component is built around.** ⚠️ And an error message quoting only $\pi/6$ would send the engineer to the wrong side.
>
> 🔴 **C-210 is the tenth dimensional failure, and it is the most consequential kind: not a missing unit but _two quantities sharing one symbol_.** 📌 $N_\kappa$ is dimensionless and stress-independent; $\kappa$ is $\mathrm{Pa}^{-1}$ and scales as $C^2$. ⚠️ A floor written on $\kappa$ therefore means something **different at every confining pressure** — it would reject weak states and pass strong ones for the same shape function. 📌 **This is the C-161 / C-176 / C-191 family again — a sign or index that resolves to one thing in prose and another in code — and it survived three rounds because each occurrence looked like a stylistic choice.**
>
> ⚠️ **C-211 is the narrowest and the most dangerous.** The submitted code's `stored_energy = H * eps_p * delta_gamma` is off by $\mu=\dot{\bar\varepsilon}_p/\dot\gamma$. 📌 **$\mu$ is a factor of 2–4 here, and at the apex the stored term is the _only_ term in the check** — so the entire verdict of the dissipation gate flips with it. ✅ The prose (§2A) is right; ⚠️ **only the code contradicts the prose in the same message** — the fourth time that has happened (**C-164**, **C-167**, and this).

---

## Ruling 36 — C-214…C-216: five items close; my kappa scaling corrected; mu is state-dependent (09-10-2026)

Corrections **C-214 … C-216**. ✅ **C-209, C-210, C-211, C-212, C-213 all close.** 🔴 **But the replacement definition of $\mu$ is wrong, and the new test cannot fail.**

| # | Correction | Detail |
|---|---|---|
| **C-214** | ⚠️ **MY C-210 SCALING EXPONENT WAS WRONG — $\kappa\propto1/C$, NOT $C^2$.** ✅ The submission states $\kappa$ *"scales inversely with stress level $C$"*; my C-210 said it scales as $C^2$. 🔴 **The submission is right.** Measured with $g\equiv1$ (so $r=C$): $C=10^5\to\kappa=10^{-5}$; $C=10^{7}\to\kappa=10^{-7}$ — **$C\kappa=1.000000$ exactly at every $C$** ✓. 🔴 **My error: I compared the _numerator_'s $C^2$ scaling without dividing out the denominator, which itself scales as $C^3$.** ⚠️ **The _conclusion_ of C-210 is unchanged and still correct** — $\kappa$ is stress-dependent, so the floor must be on the dimensionless $N_\kappa$ — ✅ **only the exponent was wrong**, and it happened to be wrong in the direction that made my argument look stronger than it was | **C-210 conclusion holds** |
| **C-215** | 🔴🔴 **$\mu=\left\lVert\partial g/\partial\boldsymbol\sigma\right\rVert_{\text{eff}}$ IS WRONG — AND $\mu$ IS _STATE-DEPENDENT_, NOT A MATERIAL CONSTANT.** ✅ Derivation: with $\bar\varepsilon_p=\sqrt{\tfrac23\varepsilon_p^{\mathrm{dev}}:\varepsilon_p^{\mathrm{dev}}}$ so $\partial\bar\varepsilon_p/\partial\boldsymbol\sigma=s/2\bar\varepsilon_p$, and $\dot{\boldsymbol\varepsilon}^p=\dot\gamma\,\partial g/\partial\boldsymbol\sigma$, $$\boxed{\mu\equiv\frac{\dot{\bar\varepsilon}_p}{\dot\gamma}=\frac{J_2}{\bar\varepsilon_p\,\left\lVert\partial g/\partial\boldsymbol\sigma\right\rVert}}$$ ✅ (dimensionless ✓ — $\mathrm{Pa}^2/(\,[\,]\cdot\mathrm{Pa}^{-1})$). 🔴 **Not $\lVert\partial g/\partial\boldsymbol\sigma\rVert$ alone**, and **not constant**: measured along a pure-shear path at yield with $\psi=0$ ($\lVert\partial g/\partial\boldsymbol\sigma\rVert$ pinned at $1.2247$ throughout): | deviator | $\bar\varepsilon_p$ | $J_2$ | $\lVert\partial g/\partial\boldsymbol\sigma\rVert$ | $\mu$ | |---|---|---|---| | $2.0$ | $1.633$ | $1.3333$ | $1.2247$ | $0.6667$ | | $6\times10^{-2}$ | $4.899\times10^{-2}$ | $1.2\times10^{-3}$ | $1.2247$ | $0.0200$ | | $2\times10^{-3}$ | $1.633\times10^{-3}$ | $1.2\times10^{-6}$ | $1.2247$ | $\mathbf{0.000667}$ | 🔴 **$\mu$ spans _three orders of magnitude_ along one loading path with the _same_ flow rule.** 📌 **Consequence: $\mu$ cannot be a single manifest scalar — it is computed per increment and recorded per increment, or its bounds are recorded.** 🔴 **And the stated range *"$\mu\in[0.25,2.0]$"* is not a bound on $\mu$ at all** — measured values fall **below** $0.25$ by two orders of magnitude. It is a bound on $\lVert\partial g/\partial\boldsymbol\sigma\rVert$ under two implicit scale choices | **C-211 remedy narrowed** |
| **C-216** | 🔴 **THE SUBMITTED `code_execution_tests` IS THE REGISTER'S FIFTH CHECK-THAT-CANNOT-FAIL.** ✅ The **policy** — a `#[cfg(test)]` suite per submitted code block — is exactly right and is adopted. 🔴 **But its test 2 hardcodes the factor it exists to verify:** `let flow_norm_mu = 0.5; // Non-associated flow example` — 📌 **it asserts that $H\bar\varepsilon_p(0.5\dot\gamma)$ reproduces $8000$, i.e. it tests arithmetic with a chosen input, and never computes $\mu$ from a stress state.** ⚠️ **It therefore cannot detect the wrong $\mu$ _formula_ that C-215 just found.** ⚠️ Test 1 uses `assert_eq!(n_kappa, 0.0)` — **exact float equality**; it passes in f64 here ($1.0\cdot1.0-1.0=0.0$ exactly) ✅ but breaks under FMA/fast-math reassociation. ✅ **Ruled:** **(a)** test 1 uses `abs_diff_eq!`; **(b)** test 2 computes $\mu=\frac{J_2}{\bar\varepsilon_p\lVert\partial g/\partial\boldsymbol\sigma\rVert}$ from an actual stress state **and** computes $\dot{\bar\varepsilon}_p$ independently from $\frac{\mathrm d}{\mathrm dt}\sqrt{\tfrac23\varepsilon_p^{\mathrm{dev}}:\varepsilon_p^{\mathrm{dev}}}$, then asserts agreement — 📌 **the two sides from independent routes**, the **C-174** rule | **1 gap** |

### Closed this round

| Item | Verdict |
|---|---|
| ✅ **§1A** $N_\kappa(0)=0.000000$; sweep $-0.033095$ at $15°$, $-0.118288$ at $30°$ | ✅ Matches my measurement exactly. ✅ **C-209 CLOSED.** "Logging $\theta_L=0°$ immediately flags that non-convexity initiates directly on the primary meridian axis" ✅ — **and that is the strongest form of the C-206 argument, correctly identified** |
| ✅ **§2** floor on `MIN_CURVATURE_NUMERATOR`; $\kappa/\kappa_{MC}$ if physical curvature is checked | ✅ **C-210 CLOSED** (⚠️ with the exponent corrected by **C-214**) |
| ✅ **§3** $\mu$ and $\dot{\bar\varepsilon}_p=\mu\dot\gamma$ declared and written to the manifest | ✅ The _policy_ is right — 🔴 but see **C-215** for the formula and for $\mu$'s state-dependence |
| ✅ **§4.1** $\rho\,\partial\psi^p/\partial\bar\varepsilon_p=H\bar\varepsilon_p$ | ✅ **C-212(a) CLOSED** |
| ✅ **§4.2** $\boldsymbol\sigma=+p\mathbf I$, minus sign "purged" | ✅ **C-212(b) CLOSED** — ⚠️ and purging it explicitly is the right response to a ninth sign slip |
| ✅ **§5A** L-9 registered `SOURCE_PENDING`; CPPM benchmark flagged against Simo & Hughes 1998 / Abbo & Sloan 1995 | ✅ Consistent with **L-6**'s citation gap; both remain unverified |
| ✅ **§5B** code-block test pinning policy | ✅ Adopted — 🔴 subject to **C-216** |

### Register status after Ruling 36

| State | Items |
|---|---|
| **Closed** | CONF-01 · 02 · 03 · 04 · 07 · 13 · 17 · 18 · 19 · 23 · 27 · 28 · 30b · 35 · 43 · 47 · 54 · 55 · 63 · 64 · 65 · 67 · 68 |
| **Conditional** | **CONF-66** (**C-215** $\mu$ formula wrong and state-dependent · **C-216** test cannot fail) · **CONF-14** · **CONF-25** |
| **Partially closed** | **CONF-31** · **CONF-51** (Type-II ✅; 🔴 proppant open) |
| **Fix specified** | CONF-15 🟡 · CONF-16 · CONF-29 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | L-1…L-6, L-8, L-9 open · **L-7 CLOSED** ✅ |

> ⚠️ **C-214 is a useful correction because it went the other way.** I had the $\kappa$ scaling as $C^2$; the engineer had $1/C$, and measurement says $1/C$ ($C\kappa\equiv1.000000$ at every $C$ tested). 📌 **My _reasoning_ was right and my _arithmetic_ was wrong in the direction that made my own argument look stronger** — which is the mirror image of **C-176** (right sign, wrong derivation). ⚠️ **Both are recorded as one class: _a conclusion can be right while the argument for it is wrong_, and the register cannot distinguish them by reading the conclusion.** ✅ Only a direct measurement can, which is why C-214 records the measurement rather than just the verdict.
>
> 🔴 **C-215 is the substantive one, and it invalidates the manifest scheme in §3.** ✅ Declaring $\mu$ was the right *policy*. 🔴 But the *quantity* is state-dependent: **$\mu$ spans three orders of magnitude along one loading path with a fixed flow rule** ($0.667\to6.67\times10^{-4}$), because it contains $J_2/\bar\varepsilon_p$. 📌 **So "write $\mu$ to the manifest" is not implementable as a single number**, and the quoted range $[0.25,2.0]$ **excludes measured values by two orders of magnitude** — a bound stated without its scale, which is the **dimensional-failure class** again (**10th** counting C-210).
>
> 🔴 **C-216 closes the loop on this component's dominant failure mode.** 📌 **Five checks that cannot fail**: **C-134** (Kahan), **C-152** ($\Delta t$ cancellation), **C-143** (Tier-A normaliser), **C-174** (RF equivalence), **C-216** ($\mu$). ✅ **All five share one mechanism — the assertion's inputs were derived from, or chosen to satisfy, the quantity being asserted.** ✅ **And the §5B policy is precisely the right countermeasure — provided the tests draw their two sides from independent routes, which C-174 already established and C-216 violates.**

---

## Ruling 37 — C-226…C-229: L-1 closes, and the table I certified contained a wrong digit (09-10-2026)

Corrections **C-226 … C-229**. ✅ **L-1 closes** — the owner supplied Karakas & Tariq (1991) Table 1.
🔴 **Three of the four findings are defects in material I previously certified**, and the fourth is a new
blocking defect that all of the numerical work had missed.

| # | Correction | Detail |
|---|---|---|
| **C-226** | ✅ **L-1 CLOSED — the source table obtained and transcribed.** 🔴 **The heading is corrected by C-230: it reads $r_{we}/(r_w+L_p)$, not $r_{wo}/(r_{wo}+L_p)$** — see Ruling 38. Table 1, six rows: $\alpha_0=\tfrac{r_{we}}{r_w+L_p}$ = **0.250** (0°, $N{=}1$) · **0.500** (180°, 2) · **0.648** (120°, 3) · **0.726** (90°, 4) · **0.813** (60°, 6) · **0.860** (45°, 8). ✅ DOI `10.2118/18247-PA` **independently re-verified** against `api.crossref.org` the same day: HTTP 200, title, journal, **6**(01), 73–82, 1991-02-01, 91 citing references — all matching. ⚠️ **Only the $\alpha_0$ half of CONF-31 closes**; $S_v$/$S_{wb}$ coefficients are not in this table | **L-1 closed** |
| **C-227** | 🔴🔴 **MY §7h.2 TRANSCRIBED 0.618 WHERE THE SOURCE SAYS 0.648 — AND DROPPED TWO ROWS.** Measured against the supplied table: $0^\circ$ ✅ · $180^\circ$ ✅ · $120^\circ$ 🔴 **0.618 vs 0.648** · $90^\circ$ ✅ · **60° (N=6) and 45° (N=8) absent entirely.** 📌 **The error sat in the digit least able to be suspected** — $0.618$ is exactly as plausible as $0.648$ — **and the fit was good enough (RMS 0.0076) that the bad row read as ordinary scatter.** 🔴 **Second instance of one failure class.** First was **C-116**: I searched 129 PDFs, found nothing, then recorded the table from the submission text instead of marking it absent. Now: I held a table that *looked* authoritative and never transcribed it row-by-row against the source. ✅ **Standing rule added:** where a numeric table enters a constitutive path, the log records **row count and every row**, and a `#[cfg(test)]` test asserts the row count — so a dropped row fails the build | **§7h.2 withdrawn** |
| **C-228** | 🔴 **THE FITTED COEFFICIENT 0.476 DOES NOT SURVIVE THE SOURCE DATA — and C-102's replacement is also wrong.** §7h.2 published $\alpha_0\approx0.250+0.476\log_4N$, RMS 0.0076. Re-fitted on all **six** source rows: $\mathbf{0.2843+0.4113\log_4N}$, **RMS 0.0298** — 🔴 **intercept $+0.0343$, slope $-0.0647$, RMS 3.9× worse.** ✅ **C-100's "my measurement stands" is withdrawn** — the *direction* it preserved survives, the *number* does not. 🔴 **C-102's proposed form $0.250+0.476\log_4(360/\theta)$** returns 0.488/0.627/0.726/0.865/**0.964** against 0.500/0.648/0.726/0.813/**0.860** — 🔴 **+0.104 at 45°, and still undefined at $0^\circ$**, i.e. it *reproduces the NaN defect it was written to close*. ✅ **Measured disposition:** the table is finite-element output — successive ratios of $(1-\alpha_0)$ are $1.500,1.420,1.285,1.465,1.336$, **not geometric, not a power law** — so **no closed form is expected; use the six tabulated values exactly**, and permit interpolation only with tabulated angles exact and the fit's own measured RMS recorded in-code | **C-102 REOPENED** |
| **C-229** | 🔴🔴🔴 **THE SPECIFIED EXPRESSION IS NOT THE INVERSION OF THE TABULATED QUANTITY — the perforation-skin defect that survives every numeric correction.** The table tabulates the **ratio** $r_{wo}/(r_{wo}+L_p)$; its inversion is $\boxed{r_{wo}=\alpha_0L_p/(1-\alpha_0)}$. The spec multiplies: $r'_w=\alpha_0(r_w+L_p)$. 🔴 **These differ without bound.** Measured ($r_w=0.108$, $L_p=0.300$ m) at 0°/180°/120°/90°/60°/45°: spec **0.1020 · 0.2040 · 0.2644 · 0.2962 · 0.3317 · 0.3509** vs source **0.1000 · 0.3000 · 0.5523 · 0.7949 · 1.3043 · 1.8429** — 🔴 **ratio 0.98 → 5.25**. The multiplication **saturates at $r_w+L_p=0.408$ m**; the table needs **1.843 m** at 45°, **4.5× past the ceiling.** 🔴 **$r_{wo}$ is an _effective_ radius and is not bounded by $r_w$** — it stands for the inflow area $N$ planes present, which for eight planes exceeds the casing bore; ✅ multiplying by $(r_w+L_p)$ **encodes the opposite assumption, and no choice of $\alpha_0$ table can rescue it.** 📌 **Same class as C-161, C-164, C-205, C-214, C-217: a correct table behind a wrong algebra.** ⚠️ **Consequence, not assumed:** $S_h$ span widens **0.66 → 2.91** (measured $S_h$: $+0.077$ at 0°, $-2.837$ at 45°). 🔴 **Large enough that it must not be adopted silently** — but the alternative is retaining a formula that provably cannot represent its own source table | **CONF-31 blocking** |

### Register status after Ruling 37

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · C-93 · C-99a · C-99b · C-102*(reopened)* |
|---|---|
| **Partially closed** | **CONF-31** ($\alpha_0$ closed by C-226 ✅ · **C-227, C-228, C-229 open** 🔴 · $S_v$/$S_{wb}$ pending) · CONF-14 · CONF-16 · CONF-25 · CONF-51 · CONF-66 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | **L-1 CLOSED** ✅ · L-2 … L-6, L-8, L-9 open |

> 📌 **C-227 is the one to remember, and it is a *method* failure, not an arithmetic slip.** I had written
> **L-1** as 🔴 *"not obtainable — zero hits across 129 PDFs"*, then **quoted a four-row numeric table anyway**
> and **measured a fit to it** and published the coefficient. Two things went wrong in sequence: I let a
> *claim about a table* stand in for the table (**C-116**), and when the real table arrived I did not diff it
> row-by-row (**C-227**). 🔴 **A table that cannot be sourced must be marked absent, and a table that can be
> sourced must be diffed.** Both halves are now standing rules, and the second is enforced by a row-count
> test rather than by care.
>
> 🔴 **C-229 is why the direction question kept being answered wrong.** For three rulings I argued about what
> larger $\alpha_0$ *means* while the table's own algebra was never inverted. ✅ **The direction was never the
> defect — the expression was.** Had I inverted $r_{wo}/(r_{wo}+L_p)$ at C-91 instead of arguing its
> interpretation, the $S_h$ span error would have surfaced immediately, because a $0.66$ span is what a
> **saturating** formula produces. 📌 **Recorded because the failure is generalisable: when a tabulated
> _ratio_ is cited, invert it and check the magnitude before interpreting it.**

> 🔴 **C-229 IS WRONG, AND THE PARAGRAPHE ABOVE IS THE ERROR IN WRITING.** See **Ruling 38**. The subscript
> is $r_{we}$ (effective well), not $r_{wo}$; the denominator is $r_w+L_p$; and **Eq. 7 of the paper is
> $r_{we}=\alpha_\theta(r_w+L_p)$ — exactly the expression C-229 called wrong.** ✅ Retained, unedited, as
> the clearest instance of *inferring structure from a misread character*.

---

## Ruling 38 — C-230…C-237: full paper obtained; C-229 WITHDRAWN; the direction dispute is resolved (09-10-2026)

Corrections **C-230 … C-237**. ✅ **L-1 and L-8 both close** — every outstanding coefficient exists in the
paper. 🔴 **C-229 is withdrawn.** ✅ **C-91, C-101, C-227, C-228 upheld.** 🔴 **C-98 and C-100 are each half
right, and the net is parameter-dependent.**

| # | Correction | Detail |
|---|---|---|
| **C-230** | 🔴🔴 **WITHDRAWN — my C-229 was wrong. I misread a subscript and invented an inversion.** ✅ **Table 1's column is $r_{we}/(r_w+L_p)$, not $r_{wo}/(r_{wo}+L_p)$** — *effective well* radius. ✅ **Eq. 7 is $r_{we}(\theta)=\tfrac14L_p$ for $\theta=0^\circ$, else $\alpha_\theta(r_w+L_p)$** — 🔴 **which is exactly the expression C-229 declared unable to represent its own source table.** ✅ The paper states the bound in prose: *"the effective well radius logarithmically approaches its maximum value of $(r_w+L_p)$"* ⇒ $\alpha_\theta\le0.860<1$, bounded **by design**. 🔴 **My "measured 5.25× divergence" was an artefact of an inversion no source contained, derived from a two-letter subscript I could not resolve at low resolution.** ✅ **Independent proof, from data I already held:** Tables 1+2+3 close — solving $s_p=0$ from Eqs. 6/7/9 reproduces Table 3's $L_{p\min}/r_w$ to $\lvert s_p\rvert\le\mathbf{0.0068}$ at 180°/120°/90°/60°/45° | **C-229 retracted** |
| **C-231** | ✅ **L-8 CLOSED — Tables 2, 3, 4 and 5 obtained; every coefficient L-8 named exists.** ✅ **Table 2** $c_1,c_2$ — all six phasings, $c_2\in[2.675,8.791]$, **positive**. ✅ **Table 3** $L_{p\min}/r_w$ = 4.62 · 1.37 · 0.77 · 0.53 · 0.33 · 0.23. ✅ **Table 4** $a_1,a_2,b_1,b_2$ — all six phasings. ✅ **Table 5** $s_x$, *"negligible for $r_d\ge1.5(r_w+L_p)$"*. 🔴 **Defect found in our own spec: the values "$c_1=0.0066$, $c_2=5.32$" and "$a_1=-2.025$, $a_2=0.0943$, $b_1=3.0373$, $b_2=1.8115$" are the 180° and 120° rows — real and correctly transcribed, but silently pinned as universal.** ✅ **Remedy: all six rows, keyed by phasing, never a single row.** ⚠️ **And no closed form exists for any of them** — the abstract says these are *"pseudoskins obtained by accurate finite-element simulations"*, so ✅ **each is used as a table** | **L-8 closed** |
| **C-232** | ✅🔴 **THE DIRECTION DISPUTE IS RESOLVED — AND BOTH SIDES WERE HALF RIGHT.** 🔴 The skin has three components with **opposing** phasing dependence: measured $s_H$ +1.792→−0.360 and $s_{wb}$ 0.797→0.009 as $\theta$ goes 0°→45° (**favour more phasing**), while $s_V$ rises 0.024→0.324 (**favours less**). ✅ **Net $s_p=s_H+s_V+s_{wb}$ (Eq. 16) has no fixed direction**, and the measured optimum **moves with the dimensionless groups**: 45° at $h_D{=}0.05$; 90° at $r_{wD}{=}0.35$; 60° at $h_D{=}0.50$ and at $h_D{=}2.00$. ✅ **The paper says the same in prose on facing pages** — *"well productivity will continue to improve with smaller phasings"* (p. 76, the $s_V$ term) against *"changing the phasings from 0 to 180° would more than double the effective perforated penetration"* (p. 77, the $s_H$ term). 🔴 **C-98 read only the first; C-100 read only the second.** 📌 **Ruled: phasing is an optimisable input, never a rule** — the engine computes all three components and reports the net, and 🔴 **must not optimise $\theta$ analytically**, because the tables give six discrete points | **C-98 + C-100 resolved** |
| **C-233** | 🔴 **C-91 IS VINDICATED — the 0° ambiguity is real and the paper never reconciles it.** Eq. 7's first branch gives $r_{we}=L_p/4$; Table 1's 0(360) row gives $0.250(r_w+L_p)$. ✅ **Equal only if $r_w=0$**; measured **4.0 %** apart at the paper's own $r_w$/$L_p$. 🔴 **Table 1 tabulates a 0(360) value that Eq. 7 discards at $0^\circ$.** ✅ **Table 3 discriminates in favour of branch 1** — at $L_p/r_w=4.62$, $s_p=+0.113$ (branch 1) vs $-0.083$ (branch 2), against $\lvert s_p\rvert\le0.007$ at all five other phasings. ⚠️ **That margin is an inference from Table 3, not a statement by the authors** ⇒ recorded `AMBIGUOUS_SOURCE` and flagged at runtime | **C-91 upheld** |
| **C-234** | 🔴 **EQ. 9's STATED DOMAIN IS VIOLATED WHERE THE 0° CASE ACTUALLY LIVES.** ✅ Eq. 9 is stated valid for $0.30\le r_{wD}\le0.90$; Eq. 5 gives $r_{wD}=r_w/(L_p+r_w)$ and the paper's own worked case $r_w=0.4$ in, $L_p=10$ in yields $\mathbf{r_{wD}=0.0386}$ — 🔴 **8× below the range.** ✅ Table 3's 0° row sits at $r_{wD}=0.178$ — 🔴 **also below it.** 📌 **Measured corroboration:** the 0° residual ($+0.113$) is **17× the worst non-zero residual** ($0.007$), consistent with the authors' own statement that 0° and 180° come from Prats' solutions rather than the fit. 📌 **Ruled: outside the domain $s_{wb}$ is `SOURCE_PENDING` + `ValidityWarning::WellboreSkinOutsideCorrelationDomain`; 🔴 it must not be extrapolated and must not be clamped, because a clamp fabricates a skin silently** — the §7s.6 rule already covers this | **1 gate added** |
| **C-235** | 🔴 **ANISOTROPY HAS NO PHASING TRANSFORM — C-101's remedy is now available and is deletion, not repair.** ✅ **The paper never rotates, rescales or transforms $\theta$.** Anisotropy enters **only** through $h_D=\tfrac{h}{L_p}\sqrt{k_H/k_V}$ (Eq. 3) and $r_{pD}=\tfrac{r_p}{2h}\left(1+\sqrt{k_V/k_H}\right)$ (Eq. 4 / Eq. 18); p. 78 — *"the flow into perforations in the vertical plane is elliptical in anisotropic formations."* 🔴 **The submitted $\theta'=\arctan(\sqrt{k_x/k_y}\tan\theta)$ has no counterpart anywhere in the paper.** ✅ **Ruled: delete the transform, apply anisotropy through $h_D$ and $r_{pD}$.** ⚠️ **$k_H/k_V$ and $k_V/k_H$ both appear, in different equations, with opposite roles** — 🔴 the substitution must be equation-by-equation, never a blanket "anisotropy factor" | **C-101 remedy supplied** |
| **C-236** | 🔴 **EQ. 21: TOTAL SKIN IS NOT A SUM, AND $s_p$ IS SCALED BY $k/k_d$.** 🔴 **Eq. 2** states $s_t=s_p+s_{dp}$; 🔴 **Eq. 21** states $s_t=s_{do}+\tfrac{k}{k_d}(s_p+s_x)$ with $s_{do}=(\tfrac{k}{k_d}-1)\ln(r_d/r_w)$. ⚠️ **Measured at $k/k_d=10$: a skin of −2.0 becomes −20.** ✅ For perforations *extending beyond* the damaged zone the paper instead gives $L'_p=L_p-[1-(k_d/k)]L_d$ (Eq. 22), $r'_w=r_w+[1-(k_d/k)]L_d$ (Eq. 23) and notes $r_w+L_p=r'_w+L'_p$ is conserved — 🔴 **so $\alpha_\theta$, defined on $r_w+L_p$, is unchanged, and only $r_{wD}$ moves**, which changes $s_{wb}$. That coupling is easy to miss | **assembly rule** |
| **C-237** | ✅ **THE CANONICAL 7-STEP PROCEDURE ADOPTED VERBATIM as the module's control flow** (p. 80): 1 $s_H$ (6,7/Tab. 1) · 2 $s_{wb}$ (9/Tab. 2, $0.30\le r_{wD}\le0.90$) · 3 $s_V$ (12–14/Tab. 4, $h_D\le10$, $r_{pD}\ge0.01$) · 4 $s_p=s_H+s_V+s_{wb}$ (16) · 5 $s_c$, $s'_p$ (17) · 6 $s_t$ (19–21/Tab. 5) · 6′ $L'_p,r'_w$ (22,23) · 7 anisotropy $r_{pe}$ (18). ✅ **Eq. 24 closes the chain**: $F_p/q_o=\ln(r_e/r_w)\,/\,[\ln(r_e/r_w)+s_t]$ — **the benchmark quantity.** 🔴 **But two $s_V$ routes exist and the spec must pick one:** Eq. 12 (power law in $h_D$) and Eq. 15 (Kuchuk et al. 13, $-h_D\ln 2\pi r_{pD}-\tfrac{1}{12}h_D^2$, for $h_D\le5$) — 🔴 **different functional forms**, overlaid on log-log and called *"satisfactory"* without a stated tolerance. 📌 **Choose one, record the choice, never blend** | **C-102 still open** |

### Register status after Ruling 38

| Closed | CONF-01 · 02 · 04 · 07 · 13 · 18 · 19 · 35 · 47 · 54 · 63 · 64 · 65 · 67 · 68 · C-76 · C-86 · C-87 · C-93 · C-99a · C-99b |
|---|---|
| **Partially closed** | **CONF-31** (✅ all coefficients now published — C-231 · ⚠️ **C-102** no closed form for $\alpha_\theta$ · ⚠️ **C-233** 0° ambiguity · ⚠️ **C-234** $r_{wD}$ gap) · CONF-14 · CONF-16 · CONF-25 · CONF-51 · CONF-66 |
| **Still open** | CONF-05 · 06 · 08 · 09 · 10 · 12 · 20 · 21 · 22 · 24 · 26 · 32 · 33 · 36 · 37 · 39 · 40 · 41 · 42 · 44 · 45 · 46 · 48 · 49 · 50 · 52 · 53 · 57 · 60 · 61 · 62 |
| **Literature** | ✅ **L-1 CLOSED** · ✅ **L-8 CLOSED** · L-7 closed · L-2 … L-6, L-9 open |

> 📌 **What this ruling is actually about.** 🔴 **Two of my five findings about this table were wrong in
> direction, and one of them — C-229 — was wrong in a way that would have put a fabricated saturation
> ceiling into a productivity correlation.** ✅ **The paper was already in my possession as an image twice:**
> once as a crop the user supplied, once as this full PDF. 🔴 **I read the crop, formed a conclusion, and
> wrote four rulings on it without ever checking whether the two letters I had misread changed the meaning
> of the equation.** C-229 is the sharpest entry in this register for a reason that has nothing to do with
> algebra: 📌 **a correction that changes the _structure_ of an equation — inverting it, bounding it,
> rescaling it — is only admissible after reading the equation in the source, and never from a tabulated
> quantity plus a guess about what it must mean.**
>
> ✅ **And the thing that caught it was not care. It was structure.** Tables 1, 2 and 3 close into a
> verification triangle: solve $s_p=0$ from Eqs. 6/7/9 and Table 3's $L_{p\min}/r_w$ comes back to within
> $0.0068$. ⚠️ **Had that triangle been looked for at C-91, C-227 would have been caught on sight and
> C-229 would never have been written.** 📌 **So the actionable rule is narrower than "check your work":
> for any cited correlation, prefer the source's own set of mutually-checking tables over any fit, and
> require the residual across them as the acceptance evidence.** ✅ **That is the three-part standard in
> §7h.4 finally applied properly — it took the full paper to satisfy it.**
