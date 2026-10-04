# Software-Quality & Anti-Pattern Audit — CO₂ EOR Optimizer (`res_audit.md`)

**Role:** Senior Reservoir Simulation Engineer / Applied Mathematician
**Date:** 04-10-2026
**Scope:** Phase 1 (Software Quality) and Phase 3 (Anti-Pattern Search) of the four-phase forensic audit.
**Companion deliverables:**

| Deliverable | Phase | Contents |
|---|---|---|
| `audit/scientific_flaws.md` | 2 + 4 (register) | **53** numbered scientific findings with measured evidence (13 CRITICAL / 19 HIGH / 16 MEDIUM / 5 LOW) |
| `audit/parameter_provenance.csv` | 2 (provenance) | 91 constants with source location + provenance class |
| **`res_audit.md` (this file)** | 1 + 3 | Software quality, toolchain output, anti-patterns, change-safety |
| `phd_audit.md` | 2 + 4 | Mathematical reconstruction, CO₂-EOR physics, categorical separation, predictive-validity verdict |

**Audit-only rule observed.** No scientific logic, equation, patch, or "cleanup" was applied to source
code. This session added `audit/scientific_flaws.md`, `audit/parameter_provenance.csv`,
`audit/code_quality/*`, `audit/runtime/*`, `res_audit.md`, `phd_audit.md`, the run-audit record
`agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/`, and **documentation-only**
edits in `agent_wiki/` (23 files — enumerated in §2.4; MED-15/MED-16 corrections).

Pre-existing modifications in the working tree — `ui/data_management_widget.py`, `ui/widgets/*.py`,
`ui/workbench/`, `scratch/`, `core/geology/`, `agent_wiki/architecture/subsurface_studio_workbench.md` —
are the user's work and were **not** touched by this audit. `agent_wiki/README.md` already carried one
user edit (a navigation link); this audit's edits to that file are in separate sections and left the
user's link intact.

---

## 0. Mandatory categorical separation

This document does **not** produce a composite "code quality score" or "model accuracy score". The
five categories below have different remedies, different owners, and different evidentiary standards,
and are reported strictly apart:

| # | Category | Question it answers | Where assessed |
|---|---|---|---|
| 1 | **SOFTWARE CORRECTNESS** | Does the code do what its author wrote, without crashing, dead names, or dead data paths? | §1–§3 |
| 2 | **NUMERICAL STABILIZATION** | Are clamps/floors/clip/fallbacks present for numerical safety, and are they *disguised* as physics? | §4.1 |
| 3 | **PHYSICAL CONSISTENCY** | Do the equations conserve mass, respect units, and match established reservoir physics? | `phd_audit.md` §1–§4 |
| 4 | **EMPIRICAL CALIBRATION** | Are constants fitted to data, and to *which* data? | `audit/parameter_provenance.csv` |
| 5 | **PREDICTIVE VALIDITY** | Has anything been validated against experiment or benchmark *for the configuration actually run*? | `phd_audit.md` §6 |

The single most important structural result of this audit: **categories 1 and 5 are effectively
independent here.** The test suite is green (**333 passed, 0 failed**, refreshed 04-10-2026) while the active evaluation path
contains thirteen CRITICAL scientific defects — because no test asserts a physical invariant of the
active `hybrid` path. Green tests are evidence of category 1 only.

---

## 1. Toolchain execution record

Environment: `.venv` (Python 3.12), Windows. All raw output preserved under `audit/`.

> **Refresh (04-10-2026, after commits `a68fc35` / `9d252ce`).** The table below is the
> **audit baseline** captured during Phase 1. Two figures have since changed and are restated here
> rather than silently overwritten: **pytest is now `0 failed / 333 passed / 23 skipped`** (HIGH-10
> was fixed by adding the missing `QIcon` import), and a **re-run of ruff** on the current tree
> reports **3 244 diagnostics in 191 files** (UP006 857, W293 686, UP045 426, F401 329, I001 253,
> F841 85, **F821 6**, F811 7) — written to `audit/ruff_output.json`. Two of the six `F821`s are
> **new** and are registered as **HIGH-19**. `vulture --min-confidence 90` could not be re-run
> (it now exceeds Python's recursion limit on this tree), so **41** remains the last verifiable
> dead-code count. Raw Phase-1 artifacts (`ruff_report.json`, `pytest_output.txt`) are deliberately
> **not** regenerated: they are the evidence for the baseline.

| Tool | Version | Command | Result | Artifact |
|---|---|---|---|---|
| **ruff** | 0.16.7 | `ruff check --output-format=json .` | **3132 diagnostics in 192 files** *(Phase-1 baseline; 3 244 / 191 after refresh — see note above)* | `audit/code_quality/ruff_report.json` (UTF-16) |
| **vulture** | 2.16 | `vulture --min-confidence 90` | **41 dead-code candidates** *(not re-runnable after refresh)* | `audit/code_quality/dead_code_candidates.txt` |
| **pyan3** | 2.8.1 | call-graph build | **959 nodes, 2225 edges** | `audit/code_quality/call_graph.dot` |
| **pytest** (+pytest-cov 7.1.0) | — | `pytest tests/ --cov=. --cov-report=xml` | *(Phase-1 baseline)* **4 failed, 329 passed, 23 skipped** in 190 s; **line coverage 37 %** (13 367 / 36 062) — **refreshed 04-10-2026: 0 failed, 333 passed, 23 skipped** | `audit/runtime/pytest_output.txt`, `audit/runtime/coverage.xml` |
| **jscpd** | — | duplicate detection | **NOT INSTALLED** — replaced by an independent in-house normalized-window scan (§3.3) | this file §3.3 |

### 1.1 ruff — rule distribution (top 15 of 91 distinct codes, audit baseline)

| Code | Count | Meaning | Software-quality verdict |
|---|---:|---|---|
| UP006 | 789 | `typing.Dict` → `dict` | style debt only |
| W293 | 713 | whitespace inside blank lines | style debt only |
| UP045 | 404 | `Optional[X]` → `X \| None` | style debt only |
| **F401** | **345** | unused import | **real**: import side effects, false "this module is used" signal |
| I001 | 257 | import sorting | style debt only |
| UP035 | 184 | deprecated typing import | style debt only |
| **F841** | **77** | assigned-but-never-used variable | **real**: computed results silently discarded |
| N802 | 63 | function name should be lowercase | style debt only |
| E701 | 61 | multiple statements on one line | style debt only |
| E402 | 57 | module-level import not at top | mixed (may be intentional) |
| F541 | 41 | f-string without placeholders | dead string |
| UP007 | 27 | `Union` → `\|` | style debt only |
| W292/W291 | 39 | trailing newline/space | style debt only |
| UP015 | 16 | `open()` mode | style debt only |

**Reading:** ~2 200 of 3 132 (≈ 70 %) are pure typing/whitespace modernization (UP*/W293/I001) and
carry **zero** scientific risk. The audit-relevant subset is **F401 + F841 + F821 + F811 = 568 items**
of which the *scientifically dangerous* classes are:

- **F821 — undefined names: 5 at baseline, 6 now** (see the refresh note in §1):

  | Location | Undefined name | Consequence |
  |---|---|---|
  | `core/engine_surrogate/analytical_models.py:96` | `MiscibleRecoveryModel` | docstring/reference-only today; **HIGH-11** — a one-line refactor turns it into a runtime `NameError` inside the recovery path |
  | `core/engine_surrogate/analytical_models.py:218` | `ImmiscibleRecoveryModel` | same |
  | `core/engine_surrogate/analytical_models.py:341` | `BuckleyLeverettModel` | same |
  | `core/geology/petrophysical_distribution.py:404` | `prev_field` | **false positive — corrected.** The baseline table below called this a `NameError` on the (untested) branch; re-reading the control flow shows `prev_field = field` at `:405` on the previous iteration and the read guarded by `if k > 0` at `:402`, so the name is always bound when used. Recorded in `audit/scientific_flaws.md` §6 ("do not fix"). |
  | ~~`ui/widgets/fault_geometry_visualizer_widget.py:115`~~ | ~~`QIcon`~~ | **HIGH-10 — RESOLVED.** The import was added at `:19` in commit `a68fc35`; the four `NameError` failures in §1.3 are gone. |
  | `ui/workbench/components/pyvista_reservoir_canvas.py:974` | `has_active_fault` | **HIGH-19 (new)** — never bound anywhere; the caprock 3-D block raises inside `try:…except Exception` (`:601`/`:1272`), so the canvas silently renders nothing |
  | `ui/workbench/components/subsurface_data_viewer_widget.py:335` | `QToolTip` | **HIGH-19 (new)** — never imported; `btn_copy_table.clicked` handler (`:130`) has no `try/except`, so the slot raises after the clipboard is written |

- **F811 — 7 redefinitions** (5 at baseline: 3 in `ui/main_window.py`, 1 `plot_ga_objective_distribution` in
  `core/plotting_manager.py:1589` shadowing `:554`, 1 `TotalDegreeBasis` in `analysis/uq_engine.py:294`;
  +2 in the new `ui/workbench/components/subsurface_data_viewer_widget.py` — `render_fault_table:1322`
  shadowing `:969` and `render_caprock_graph:1577` shadowing `:927`).
  The plotting one matters: a redefined function silently replaces the public plotting helper.

- **F841 — 85 discarded computations** (77 at baseline), concentrated in exactly the modules that generate science:
  `core/optimisation_engine.py` (13), `core/engine_surrogate/profile_generator_fast.py` (8),
  `core/engine_surrogate/surrogate_engine.py` (6). In this codebase an unused assignment is a
  plausible symptom of a calculation that was *computed and then not wired up* — the same family as
  **CRIT-10** (engine returns `total_leakage_tonne`, `annual_fault_leakage_tonne`,
  `annual_caprock_leakage_tonne`, **0 consumers**).

### 1.2 Coverage — the number that matters is *where* it is low

Overall **37 %**. Per-package:

| Package | Line rate | Comment |
|---|---:|---|
| `engine_surrogate` | **83.3 %** | high — but tests assert *behaviour of the math*, not *correctness of the math* (see §4.2: a test asserts the 48.3 % cliff) |
| `objectives` | 58.3 % | moderate |
| `analysis` | 30.1 % | low |
| `ui` (aggregate) | 27.0 % | low, expected for GUI |
| `ui.workers` | 25.8 % | low — these host the optimization execution path |
| `ui.dialogs` | 10.2 % | very low |
| `ui.workbench` | 9.8 % | very low (new code, committed in `a68fc35`) |
| `tests.validation` | 6.8 % | the benchmark configuration itself is barely executed |
| `simulation` (legacy) | 1.2 % | dormant, consistent with the source-of-truth map |
| `audit` | 0 % | tooling |

**The decisive coverage fact:** `engine_surrogate` at 83 % with a 37 % repository total means the
suite is *broad on the surrogate and thin everywhere data enters it* (`ui.workers`, `ui.workbench`,
`tests.validation`). The inputs that determine what the surrogate is asked to compute are the least
tested region of the system.

### 1.3 pytest failures — all four were one defect (**RESOLVED 04-10-2026**)

> **Status: RESOLVED.** This subsection is retained as the pre-fix record. Commit `a68fc35` added
> `from PyQt6.QtGui import QIcon` at `ui/widgets/fault_geometry_visualizer_widget.py:19`. Re-running
> the suite gives **`333 passed, 23 skipped, 0 failed`** (356 collected), so the `AGENTS.md`
> invariant #5 gate (`pytest tests/test_project_save_load.py -v`) is green again — verified together
> with `tests/scientific` as `41 passed`. `audit/runtime/pytest_output.txt` is kept unchanged as the
> **baseline** evidence for HIGH-10. Note that the fix closed the *instance*, not the *class*: two
> fresh undefined names in `ui/workbench/` are registered as **HIGH-19**.

Baseline output (04-10-2026, pre-fix):

```
FAILED tests/test_project_save_load.py::test_data_management_widget_save_and_load      NameError: 'QIcon'
FAILED tests/ui/test_3d_well_interaction.py::test_wells_tab_no_duplicate_3d_cube        NameError: 'QIcon'
FAILED tests/ui/test_3d_well_interaction.py::test_3d_subsurface_property_volume_rendering NameError: 'QIcon'
FAILED tests/ui/test_3d_well_interaction.py::test_screen_to_reservoir_ray_plane_inversion  NameError: 'QIcon'
```

Root cause: missing `from PyQt6.QtGui import QIcon` in
`ui/widgets/fault_geometry_visualizer_widget.py` (ruff F821 at line 115). **SOFTWARE CORRECTNESS,
HIGH-10.** At baseline `tests/test_project_save_load.py` — the mandatory gate named in `AGENTS.md`
invariant #5 — was **failing on main**, so the "always run this test" invariant could not be
satisfied green until HIGH-10 was fixed (the fix is a one-line import, outside this audit's
mandate — and it was subsequently made by the repository owner, see the RESOLVED note above).
**The governance lesson survives the fix:** presentation code could block a *scientific* gate, and
no test failed for any scientific reason.

Benchmark tests embedded in the run: `test_evaluation_speed` 71 ms/op, `test_batch_evaluation_speed`
711 ms/op. Speed is explicitly **ranked last** in the truth hierarchy and is recorded here only for
completeness.

### 1.4 Test-suite integrity — a green suite is not analytical verification

`pytest tests/scientific` → **36 passed** in 15 subdirectories. `agent_wiki/verification/test_matrix.md:5`
claims **42 items across 16 subdirectories**, and the two disagree in four separate ways:

| Check | Result |
|---|---|
| Items actually collected | **36** (wiki: 42) |
| Subdirectories | **15** (wiki: 16) |
| Wiki-listed tests that do not exist anywhere in `tests/` | **6** — `test_co2_density_thermal_expansion`, `test_cubic_eos_z_factor_bounds`, `test_phase_label_assignment`, `test_peng_robinson_fugacity_equation_structure`, `test_corey_relative_permeability_bounds`, `test_bg_discrepancy_between_modules` (5 of them cite the deleted `unified_engine/` tree) |
| Wiki-listed tests that were renamed / present but unlisted | 1 renamed (`…_mobility_inversion` → `…_monotonicity`), 2 unlisted |

An AST scan (`v_tests2.py`) of all 36 test functions classifies them by whether they actually reach
production code:

| Class | Count | Meaning |
|---|---:|---|
| Exercises a production symbol | **25** | genuine |
| Imports production symbols but **never calls them** | **5** | the assertion is made against a hand-copied formula |
| Never imports production code at all | **6** | self-contained (MMS/reference-solution checks are legitimate; `test_darcy_inflow_dimensions` is not — it invents `J = 2.0·J_unit` locally while the wiki credits it with verifying `surrogate_engine.py:330`) |

Three tests go further and assert **that a defect exists**, with the defect's own value hard-coded:
`assert step > 0.40` (48 % Craig cliff, HIGH-02), `assert eff_high_sgc < eff_low_sgc` (inverted
trapping, HIGH-15), `assert truncation_fraction > 0.20` (pore-volume RF bound, SCI-FLAW-11).
Remediation of any of those three will turn the suite red, and none of them can detect a fix.

**Category: SOFTWARE CORRECTNESS (test adequacy), not physics.** The consequence is evidential:
the *Analytical Verification* rung of the truth hierarchy reports PASS for items where the test
either restates the code or pins the wrong answer. Recorded as **HIGH-18** (integrity) and
**MED-16** (stale matrix). No test was modified by this audit.

---

## 2. Structural / architecture findings (Phase 1)

### 2.1 Size and cohesion

- **209 Python files, 82 292 lines** (excluding `agent_wiki/`, `audit/`, `logs/`, venv; re-counted 04-10-2026 — the Phase-1 baseline was **203 files / 77 471 lines** measured before `a68fc35` landed, so the tree has grown by 6 files / ≈4 800 lines. Some of that growth was already on disk but **untracked** at baseline; it is now committed).
- **19 files exceed 1 000 lines.** Top offenders:

  | Lines | File | Assessment |
  |---:|---|---|
  | 4 291 | `core/optimisation_engine.py` | god object: algorithms + packing + penalties + containment + reporting + sensitivity |
  | 3 413 | `ui/data_management_widget.py` | god widget |
  | 2 385 | `ui/workbench/components/contextual_property_grid.py` | new in `a68fc35` |
  | 2 363 | `ui/optimization_widget.py` | god widget |
  | 2 207 | `core/data_models.py` | 60+ dataclasses, single module — this is the *schema*, and invariants #5 and part of #2 live here |
  | 1 748 | `core/plotting_manager.py` | contains a duplicate function definition (F811) |
  | 1 537 | `utils/run_exporter.py` | reporting |

  `core/optimisation_engine.py` at 4 291 lines is the single largest risk amplifier in the repository:
  **CRIT-08, CRIT-09, CRIT-10, HIGH-09, MED-07, MED-08, MED-09, MED-10 and MED-14** all live inside
  it, and they are all *data-flow* defects (a value produced here is not read there). A 4 291-line
  module with 47 private methods is exactly where such disconnects are invisible in review.

### 2.2 God-module coupling evidence (pyan3 call graph)

- 959 nodes / 2 225 edges ⇒ mean out-degree ≈ 2.3, but the distribution is heavily skewed:
  `core.optimisation_engine` and `core.data_models` are reached from nearly every cluster.
- Consequence observed in this audit: a **single default** (`"hybrid"` in
  `core/data_models.py:1725-1730`, `config/base_config.json:24,746`, `core/optimisation_engine.py:264-270`)
  silently selects which recovery model *every* evaluation uses, while **72 of 78** scientific tests
  exercise `phd_hybrid` instead (**HIGH-09**). The blast radius of a one-line default in a god module
  is the whole optimization result.

### 2.3 Duplicated / parallel implementations (anti-pattern: "two truths")

The most consequential duplication in this codebase is **not textual** — it is *semantic*. jscpd was
unavailable, so §3.3 gives an independent textual scan; but the audit's headline duplication findings
are of logic that was written twice with different answers:

| Pair | Location A | Location B | Divergence |
|---|---|---|---|
| **Two CO₂ ledgers** | `surrogate_engine.py:544` `co2_stored = injected − produced` | `wrapper.py:174` ledger used by NPV | **CRIT-02** |
| **Two sandface-pressure models** | `wrapper.py:64-67` `q/II` | `surrogate_engine.py:359` `+400 psi` (capped) | **HIGH-14** |
| **Two NPV functions** | `surrogate_engine.py:572-596` | `surrogate_models.py:507-530` | **HIGH-08** (only the 2nd is used by the reported NPV) |
| **Two Koval H formulas** | `analytical_models.py:174,534,726` `1/(1−v)²` | `surrogate_engine.py:894-895` `10^(v/(1−v))` | **HIGH-06**: 1.14× … 10⁹⁵× |
| **Two `FAILURE_PENALTY` constants** | `optimisation_engine.py:96` `-1e12` | `optimisation_engine.py:1543` / `data_models.py:1709` `-1e12` | **MED-10**: equal today, drifts silently on edit |
| **Two E_A correlations** | `analytical_models.py:307-312` | `surrogate_models.py:164-182` | both contain the M=1 cliff (**HIGH-02**) |
| **Two `DARCY` constants** | `well_mechanics.py:27` `0.00708` (correct, ×2π) | `well_mechanics.py:220` missing ×2π | **MED-03**: 6.28× |
| **Five CO₂ density constants** | `data_models.py:1860` `0.05254`, `pvt_state.py:42` `0.05295`, `profile_generator_fast.py:794` `0.05297`, `analytical_models.py:37` / `surrogate_models.py:26` `0.053`, plus `0.056` in a comment | — | **MED-01**: 1.0 % spread, plus two wrong values in comments |

The wiki already documents the *danger* of parallel implementations (source-of-truth map: 5 engine
directories, 1 active). The audit adds that **within the active engine itself** the same pattern
repeats at function granularity.

### 2.4 Documentation ↔ code divergence (anti-pattern: documentation as decoration)

`agent_wiki/` contains several assertions that the code does not support. These are software-quality
defects because agents are instructed to treat the wiki as the source of truth:

| Wiki claim | Reality | Evidence |
|---|---|---|
| `agent_wiki/README.md:35` invariant #3 — mass conservation `M_recycled ≤ M_produced ≤ M_injected` | **Holds by construction, not by enforcement**: recycled is capped as `min(prod·η_recycle, inj)` (`optimisation_engine.py:871-872`), so the inequality cannot be violated — but nothing asserts it, and the second half of the invariant (`≤ M_injected` after leakage) is unsatisfiable to check because `total_leakage_tonne ≡ 0.0` and `cum_stored = inj − prod` (`surrogate_engine.py:543`) ignores leakage entirely | measured on a 10-yr run: inj 36 525 000 MSCF = purchased 34 994 496 + recycled 1 530 504 (exact); recycled 1 530 504 ≤ produced 1 611 057 ✓ |
| `_calculate_engine_npv(...)` referenced in `code/inventory.md`, `architecture/execution_flow.md:81-82`, `code/modules/core_engine_surrogate.md`, `code/functions/*`, `code/classes/surrogate_classes.md`, `validation/conservation.md` | **Function does not exist** — 0 grep hits repo-wide | real NPV: `surrogate_engine.py:572-596`, `surrogate_models.py:507-530` |
| `_calculate_co2_purchased_recycled(...)` | **Does not exist** — 0 grep hits | — |
| `agent_wiki/architecture/source_of_truth_map.md:16`, `source_of_truth.md:26` — engine entry-point description | stale line references | this audit |
| `common_pitfalls.md:34` — pitfall list | missing every finding in `audit/scientific_flaws.md` | this audit |
| `agent_wiki/architecture/overview.md:71`, `README.md:29` — legacy engines *"relocated into `deprecated/`"* | **No `deprecated/` directory and no `core/unified_engine/`, `compositional_engine/`, `core/Phys_engine_full/` in the tree** (`Test-Path` = False for all four); the wiki still says `unified_engine` 51 times | `Get-ChildItem -Directory` |
| `agent_wiki/verification/test_matrix.md:5,18-46` — "42 test items across 16 subdirectories" + 40-row verdict table | **36 items in 15 subdirectories**; 6 listed tests do not exist in `tests/`, 1 renamed, 2 unlisted; 5 rows cite the deleted `unified_engine/` tree | `pytest --collect-only tests/scientific`; 7 greps = 0 hits |

A wiki that names functions which do not exist is worse than no wiki: it converts a search miss into
false confidence. **Remediation is documentation-only** and was **completed in this session**: 23
`agent_wiki/` documentation files (22 `.md` + `dependency_graph.txt`) were corrected (invariant #3 rewritten,
`source_of_truth_map.md` / `source_of_truth.md` NPV + EOS rows re-pointed at live code, all references to the
four non-existent functions annotated, `test_matrix.md` regenerated, `deprecated/` claims corrected, three new
pitfalls added), and the change-safety matrix received 17 added/amended rows. Those same files were
**refreshed again on 04-10-2026** after HIGH-10 was resolved and HIGH-19 was registered (counts
52 → 53, suite 329/4 → 333/0, two new change-safety rows for `ui/workbench/`, and a new pitfall #53).
Two residual collisions worth
knowing about:

- `audit/scientific_flaws.md` (this audit, **53 findings** after the 04-10-2026 refresh) vs
  `agent_wiki/audit/scientific_flaws.md`
  (prior session, 18 `SCI-FLAW-*` rows) — **same base name, different files**. Cross-check is in
  `phd_audit.md` PART E.
- `audit/report.md` and `audit/reports/scientific_audit_report.md` (prior-session automated audit outputs)
  still quote the pre-fix toolchain numbers (5 472 ruff warnings, 36 `F821`, 296 passed / 3 failed, 18 flaws).
  Those are **stale baselines** — this audit measured 3 132 / 5 / 329-4 / 52 at Phase 1, and
  3 244 / 6 / 333-0 / 53 after the 04-10-2026 refresh. They are historical records and
  were deliberately left unedited.

---

## 3. Anti-pattern catalogue (Phase 3)

Findings are tagged **[SW]** (software correctness), **[NUM]** (numerical stabilization),
**[PHYS]** (physical consistency — cross-referenced, detail in `phd_audit.md`), **[PROV]** (provenance).

### 3.1 Silent failure & masking

| ID | Anti-pattern | Location | Evidence | Class |
|---|---|---|---|---|
| **MED-04** | Silent fallback returns a plausible value | `well_mechanics.py:65-66, 113-114` | failure ⇒ `return 1.0` (a physically valid-looking WI) with no log, no error, no flag | [SW] |
| **MED-06** | ProductionProfiler fallback + post-hoc clamp | `optimisation_engine.py:905-935` | if the profiler fails, a substitute is used and RF is clamped afterwards — the clamp hides the fact that the fallback fired | [SW][NUM] |
| **HIGH-12** | Exception funnel swallows the reason | `surrogate_engine.py:788-790` (no `exc_info`), `surrogate_models.py:553-562`, `pvt_state.py:147-149` (silent Z-root fallback) | *mitigating*: `optimisation_engine.py:725-727` converts failure to `FAILURE_PENALTY`, so the optimizer does not silently accept a bad value. Severity is therefore **masking/diagnosis-blocking**, not silent acceptance. | [SW] |
| **HIGH-04** | `\|`/`or` defaults destroy legitimate zeros | `analytical_models.py:661,671` | `v_dp=0.0 or default`, `s_wi=0.0 or default`, `c7=0.0 or default` ⇒ measured RF **0.493568** for all-zero input vs 0.132 / 0.150 / 0.49297 for 0.9 / 0.6 / 0.8 — a physically meaningful 0 is replaced by a different number, silently | [SW] |
| **MED-08** | Feasibility computed, logged, then discarded | `optimisation_engine.py:1531, 1560-1562` vs penalty applied separately at `:1771-1772` | two independent gates that can disagree; the logged one is not the enforced one | [SW] |
| **CRIT-09/10** | Output produced, never consumed | `wrapper.py:61-78` reads absent keys; `annual_leakage_tonne`, `max_sandface_pressure_psi` (0 hits) | constraint code **exists and is readable** but can never execute ⇒ Class VI sandface penalty and leakage cost are dead letters | [SW][PHYS] |

### 3.2 Inert controls (genes, parameters, and constraints that cannot affect the result)

| ID | Inert thing | Location | Why it is inert | Class |
|---|---|---|---|---|
| **CRIT-13** | `gravity_factor` | EOR parameter; 0 physics consumers | never read | [PHYS] |
| **CRIT-13** | `transition_alpha` / `transition_beta` | optimized over `[0.8,1.2]`, `[2.0,10.0]` (`data_models.py:945-948`); only consumer `core/simulation/recovery_models.py:329` = dormant module | optimizer spends 2 genes on nothing; `data_models.py:976-977` even labels them *"Curve fitting - not operational"* | [SW][PHYS] |
| **CRIT-13** | `HybridSurrogate` α/β | `analytical_models.py:458-459` hardcodes them; reads `c7_plus_fraction` (`:449`) while engine writes `c7_plus` (`surrogate_engine.py:865`) | α ≡ 0.95 always | [SW] |
| **CRIT-13** | `mobility_ratio` in Hybrid RF | measured RF ≡ **0.628079** for M = 0.98 … 5 | only affects breakthrough at `:896` | [PHYS] |
| **CRIT-08** | `structural_trapping_factor` / `reservoir_seal_integrity_factor` | defined in `data_models.py:1588,1596` but `storage.py:91-92` uses `getattr(...)` on **`AdvancedEngineParams`**, which lacks them ⇒ 0.9 / 0.85 fallbacks | ⇒ `S_cont` floor **0.44 > threshold 0.3** (`data_models.py:1730`) ⇒ `prune_possible=False`: the containment prune can never trigger | [SW][PHYS] |
| **CRIT-08** | `{time_res}_pressure` key | built only for `yearly` (`:853`) and `monthly` (`:870`); valid resolutions include `weekly`, `quarterly` (`data_models.py:1273`) | key lookup fails for 2 of 4 legal resolutions | [SW] |
| **MED-07** | `_calculate_adaptive_penalty` | `optimisation_engine.py:1209-1256`, `total_violation = 0.0` hard-coded, all real checks commented out (`:1232-1253`), so the only call site (`:1783`) always receives 0.0; and the advertised `"death"` method is a bare `pass` (`:1225-1230`) | guard function that *looks* protective; `penalty_factor` / `constraint_handling_method` are dead knobs | [SW] |
| **MED-08** | `is_feasible` | computed and returned, then only logged; never gates selection | see the row above (same finding) | [SW] |
| **HIGH-08** | `variable_opex_usd_per_bbl`, `co2_storage_credit_usd_per_tonne`, `carbon_tax_usd_per_tonne`, `co2_recycle_cost_usd_per_tonne` | supplied at `surrogate_engine.py:940-957`; NPV reads only 4 keys (`surrogate_models.py:508-511`) | four economic dials do nothing to the reported NPV | [SW][PHYS] |
| **MED-05** | relative permeability & inter-well transmissibility | `relative_permeability.py` (all), `well_mechanics.py:184-221`, `surrogate_models.py:371` | module not on the evaluation path — saturations come from material balance (`surrogate_engine.py:376-379`); per-well interactions not modeled | [PHYS] |
| **MED-06** | rel-perm oil endpoint normalization | `relative_permeability.py:55` | uses the **water** denominator even when `s_org ≠ s_orw` | [MATH] |

**Anti-pattern reading:** this is the audit's central Phase-3 result. The system has the *appearance*
of a constrained, calibrated, multi-parameter optimizer — 60+ parameters, penalty functions,
containment pruning — while a measurable subset of those controls are disconnected. Appearance of
constraint without enforcement is more dangerous than absence of constraint, because reviewers stop
looking.

### 3.3 Duplication scan (jscpd substitute)

jscpd is not installed in this environment (no network). Independent substitute: every 12-line
sliding window of non-comment source, MD5-normalized, counted across files.

- **127 duplicate windows span more than one file.**
- Concentrated in UI widget code: `ui/widgets/corey_relperm_workstation_widget.py` (54 windows),
  `ui/widgets/fluids_pvt_workstation_widget.py` (46), `ui/widgets/geostatistics_visualizer_widget.py`
  (41), `ui/widgets/geology_cross_section_widget.py` (14), `ui/workbench/components/*` (28).
- `tests/conftest.py` duplicates 9 windows into test modules (fixture copies).
- **21 files in `scratch/`** duplicate tested widget code (cursor/tooltip/PyVista experiments).

**Verdict:** textual duplication is a **MEDIUM, UI-clustered** problem — it costs maintenance effort
but does not change numbers. It is deliberately **not** conflated with §2.3's semantic duplication,
which does change numbers. Those two must never be merged into one "duplication score".

### 3.4 Dead code (vulture, 41 candidates)

Full list in `audit/code_quality/dead_code_candidates.txt`. Scientifically relevant subset:

| Candidate | Location | Note |
|---|---|---|
| `MSCF_PER_TONNE` unused import | `surrogate_engine.py:37` | reciprocal of the live `CO2_TONNE_PER_MSCF` — two reciprocal constants invite a unit error (**MED-01/04**) |
| `fault_strike_deg` unused | `geomechanics_fault.py:85` | fault orientation computed/stored but unused while slip tendency uses `theta_fault` |
| `calculate_koval_from_reservoir` unused import | `core/reservoir_state_manager.py:13` | a Koval path exists and is not wired |
| `Sobol` unused import | `analysis/uq_engine.py:18` | UQ Sobol sensitivity not actually invoked |
| `EOSModel` unused import | `analysis/well_analysis.py:13` | — |
| `shutdown_queue_logging` unused | `core/optimisation_engine.py:16` | worker logging lifecycle |

The `Sobol` and `calculate_koval_from_reservoir` entries are worth flagging beyond cleanup: they
indicate **capabilities the project believes it has (UQ sensitivity, reservoir-derived Koval) that are
not on any executed path.**

### 3.5 Numeric literals in logic (anti-pattern: constants as control flow)

| ID | Literal | Location | Effect | Class |
|---|---|---|---|---|
| HIGH-03 / CRIT-07 | `0.05`, `0.10` floors | `analytical_models.py:193,200,326,475` | RF returned at the floor for hcpvi = 0 → measured **0.050000**, and immiscible RF ≡ **0.1** across a 5×3×3 grid | [NUM] masking [PHYS] |
| HIGH-01 | `1 − swi − sor` | `surrogate_engine.py:509`, `analytical_models.py:814` | 0.4500 vs correct 0.6000 ⇒ RF 25 % low | [PHYS] |
| HIGH-13 | `max(carbon_tax, 100.0)` | `wrapper.py:97` | user value below 100 silently overridden | [SW] |
| HIGH-07 | `nominal_drawdown = 500.0` | `surrogate_engine.py:420` | `J = q/500` ⇒ injectivity is a tautology of the imposed rate | [PHYS][PROV] |
| MED-11 | `default_gas_fvf` unit mismatch | `data_models.py:917`, `profile_generator_fast.py:1109,1117,1210` | ×1000 present at `:1210`, absent at `:1117` ⇒ WAG water 25 bpd vs SWAG 25 000 bpd (**CRIT-11**) | [PHYS] |
| MED-15 | `mobility_ratio = rate × 0.001` vs threshold `2.0` | `profile_generator_fast.py:1103-1104` | dimensionally not a mobility ratio | [PHYS][PROV] |

### 3.6 Error-handling & logging compliance (AGENTS.md / ERROR_HANDLING_GUIDELINES)

- `report_caught_error(...)` with `exc_info`-style context: **not used** in the surrogate engine's
  catch blocks (`surrogate_engine.py:788-790`, `surrogate_models.py:553-562`, `pvt_state.py:147-149`).
  They use bare `logger.warning/error` without `exc_info=True` and without the required
  `operation` / `context` / `user_action_suggested` fields.
- Module-level `logger = logging.getLogger(__name__)` pattern: **compliant** across the audited modules.
- Type hints on public signatures: **compliant** in `core/engine_surrogate/` and `core/objectives/`;
  spotty in `ui/`.
- Bare `except:` — none found in the active engine (compliant); the issue is *narrow excepts that
  return defaults*, which is the HIGH-12/MED-04 family above.

---

## 4. Categorical separation applied to this repository

### 4.1 NUMERICAL STABILIZATION vs disguised physics

Stabilization is legitimate. The audit's criterion: *does the code say "this is a clamp for numerical
safety", or does it present the clamp as a physical result?*

| Location | Clamp | Verdict |
|---|---|---|
| `pvt_state.py:157` `Z = max(Z, 0.1)`, `:161` `clip(rho, 2, 1100)`, `:202` `clip(b, 0.20, 15)`, `:226` `clip(mu, 0.015, 0.12)` | numerically motivated, documented as such | **legitimate stabilization**, but note it *hides* CRIT-04/05 rather than surfacing them |
| `pvt_state.py:147-149` fallback `Z = max(0.25, B*1.05)` when no valid root | legitimate last resort | **flag**: no log, no metric of how often it fires (HIGH-12) |
| `analytical_models.py:193,200,326,475` RF floors | presented inside the RF formula itself, *not* flagged | **illegitimate**: a floor returned as a physical recovery factor. CRIT-07 shows the floor *is* the answer over an entire parameter grid |
| `surrogate_engine.py:513-522` post-hoc RF rescale | applied after integration, unreconciled with `S_o` | **illegitimate**: creates CRIT-02's mutually inconsistent report (**CRIT-12**) |
| `analytical_models.py:312,317` `clip(E_A,0.1,1)`, `clip(E_V,0.1,1)` | masks the HIGH-02 cliff | **illegitimate** (concealment) |

### 4.2 SOFTWARE CORRECTNESS ≠ PHYSICAL CORRECTNESS (the decisive example)

`tests/scientific/mathematical/test_singularity_and_overflow.py:73` **asserts `step > 0.40`** — i.e.
the test suite *codifies* the 48.3 % mobility-ratio cliff of **HIGH-02** as required behaviour. The
test name promises singularity handling; the assertion enshrines a discontinuity.

Consequences, stated categorically:

- **SOFTWARE CORRECTNESS:** the test passes ⇒ correct.
- **PHYSICAL CONSISTENCY:** Craig's areal-sweep correlation is continuous at M = 1 ⇒ incorrect.
- **PREDICTIVE VALIDITY:** unaffected directly, but the green test actively *blocks* the fix.

This is why no composite score is issued anywhere in this audit.

### 4.3 EMPIRICAL CALIBRATION status

From `audit/parameter_provenance.csv` (91 rows):

| Provenance class | Count | Meaning |
|---|---:|---|
| LITERATURE | 21 | formula/constant traceable to a named source and numerically verified here (PR coefficients, Corey, Craig, Johnson, Vasquez-Beggs, Darcy 0.00708, IUPAC CO₂ critical density, fracture limit 0.9) |
| EMPIRICAL | 6 | plausible textbook defaults (`S_OR=0.25`, `S_GC=0.05`, CO₂ density 0.053) |
| CALIBRATED | 16 | market/UX defaults (oil price, discount rate, carbon tax) — acceptable *because user-editable* |
| **UNKNOWN — EVIDENCE REQUIRED** | **48** | no citable source: leakage rates, containment weights, `nominal_drawdown`, `mu_inj_eff`, RF floors, `s_ref`, sigmoid α/β, `0.1587`, the "Hall-Yarborough" coefficients, the CO₂ μ polynomial, `c_g` factors, Koval engine-side exponent, carbon-tax floor, etc. |

**53 % of the audited constants have no provenance.** Note carefully what this does *not* say: it does
not say they are wrong. It says they are **unverifiable**, which under the truth hierarchy outranks
"benchmark agreement" and sits *above* it — a number that cannot be traced cannot be defended even if
it happens to reproduce a curve.

### 4.4 PREDICTIVE VALIDITY — reserved

Assessed narratively in `phd_audit.md` §6. Preview of the categorical statement: **NOT ESTABLISHED**
for the active `hybrid` path, because the repository contains no experimental or benchmark evidence
for that configuration (benchmark/validation material targets `phd_hybrid` and dormant engines —
**HIGH-09**). Benchmark agreement elsewhere is *not* credited as evidence here.

---

## 5. Change-Safety risk classification (updates to `agent_wiki/development/change_safety_matrix.md`)

Applying the existing 4-tier scheme to the components this audit touched or examined. Rows marked
**NEW/AMENDED** are the additions queued for the matrix.

| File / Component | Tier | Amendment |
|---|---:|---|
| `core/engine_surrogate/pvt_state.py` | **CRITICAL** | **NEW/AMENDED** — currently absent from the matrix despite hosting CRIT-03/04/05, HIGH-12/16/17. Any edit must verify: bubble-point behaviour of `Bo(P)`, `B_g` magnitude, Z-factor against Standing–Katz, `c_o` sign. |
| `core/engine_surrogate/analytical_models.py` | CRITICAL | AMENDED — matrix row exists; add: "contains RF floors at `:193,200,326,475` (HIGH-03/CRIT-07) and the M=1 `E_A` cliff (HIGH-02); *do not 'fix' the floors without first resolving CRIT-06/07*, they currently cap a broken sweep model." |
| `core/engine_surrogate/surrogate_engine.py` | CRITICAL | AMENDED — add HCPVI cancellation at `:491-497` (CRIT-01), dual-ledger NPV (CRIT-02), post-hoc rescale (CRIT-12). Line-number edits here invalidate CRIT-01/02/12 references. |
| `core/engine_surrogate/surrogate_models.py` | **CRITICAL** | **NEW** — hosts the *reported* NPV (`:507-530`), the inverted trapping formula (`:238`) and the second `E_A` cliff (`:164-182`). Currently unlisted. |
| `core/objectives/wrapper.py` | **CRITICAL** | **AMENDED from HIGH** — CRIT-09/CRIT-10 show its Class-VI sandface and leakage constraints cannot execute (absent profile keys). Downgrading its apparent safety: code that looks protective but never runs invites false confidence. |
| `core/objectives/storage.py` | **CRITICAL** | **NEW** — containment score with unpopulated `getattr` fallbacks ⇒ prune impossible (CRIT-08). |
| `core/engine_surrogate/geomechanics_fault.py` | HIGH | **NEW** — leakage law constants + HIGH-05 clip/leak threshold mismatch. |
| `core/engine_surrogate/well_mechanics.py` | HIGH | **NEW** — two Darcy constants, one missing ×2π (MED-03). |
| `core/engine_surrogate/profile_generator_fast.py` | CRITICAL | AMENDED — add CRIT-11 WAG/SWAG ×1000 asymmetry at `:1117` vs `:1210`. |
| `core/optimisation_engine.py` | HIGH | AMENDED — matrix already warns about penalty dilution; **add**: inert genes (CRIT-13), containment key mismatch (CRIT-08), inert `_calculate_adaptive_penalty` incl. no-op `"death"` branch (MED-07), `is_feasible` non-enforcement (MED-08/14). |
| `core/data_models.py` | CRITICAL | AMENDED — beyond constants: default recovery-model string (`"hybrid"`, HIGH-09) silently selects the evaluation model; changing it changes every result. |
| `ui/widgets/fault_geometry_visualizer_widget.py` | LOW | **AMENDED to MEDIUM, then noted as RESOLVED** — a missing import failed a mandatory test gate (HIGH-10); presentation code can block invariant #5. Fixed by the owner in `a68fc35` (`QIcon` imported at `:19`); the tier is *not* restored to LOW, because the lesson — GUI files gate a scientific invariant — still applies. |
| `ui/workbench/components/pyvista_reservoir_canvas.py`, `ui/workbench/components/subsurface_data_viewer_widget.py` | HIGH | **NEW (HIGH-19)** — two undefined names (`has_active_fault` at `:974`, `QToolTip` at `:335`) that ruff reports as `F821`. One is swallowed by `except Exception` (`:1272`) and silently blanks the 3-D caprock view; the other raises inside a Qt slot with no guard. Zero test coverage. Any edit here must re-run `ruff check` and manually exercise caprock rendering + *Copy Table*. |
| `ui/workbench/`, `ui/widgets/corey_relperm_*`, `fluids_pvt_*`, `well_network_*` | HIGH | **NEW** — now **committed** (was untracked at Phase 1), ~9 % coverage. `core/geology/petrophysical_distribution.py:404` is listed with them for coverage reasons only: its `F821 prev_field` is a **false positive** (see `audit/scientific_flaws.md` §6) and must not be "fixed" reflexively. |
| `scratch/` (21 files) | LOW | **NEW** — non-shipped experiment scripts duplicating widget code; excluded from risk but should not be imported by shipped modules. |
| `tests/scientific/mathematical/test_singularity_and_overflow.py` | **CRITICAL** | **NEW** — a *test* that asserts physically wrong behaviour (`step > 0.40`, HIGH-02). Tests are usually LOW-tier; here the test encodes the defect, so changing the model requires changing the test, and the change must be justified physically, not to make CI green. |
| `tests/scientific/co2/test_co2_breakthrough_physics.py`, `tests/scientific/conservation/test_mass_conservation.py`, `tests/scientific/co2/test_co2_trapping_mechanisms.py`, `tests/scientific/mathematical/test_analytical_identities.py`, `tests/scientific/dimensional/test_unit_consistency.py` | **CRITICAL** | **NEW (HIGH-18)** — verification-tier code. 5 of these tests import production symbols they never call (the Koval test re-derives `profile_generator_fast.py:958-969` by hand at `:38-46`) and 3 assert that a defect exists. Re-specifying them against production code is *higher* risk than changing engine source: a wrong re-specification converts a detectable defect into an undetectable one. Require a physical justification in the PR, not a green build. |
| `agent_wiki/verification/test_matrix.md`, `agent_wiki/architecture/overview.md` | MEDIUM | **NEW (MED-16)** — declares 42 tests/16 subdirectories (actual 36/15), lists 6 tests that do not exist, and claims a `deprecated/` tree that is not in the repository. Regenerate from `pytest --collect-only`; do not hand-edit counts. |
| `core/simulation/recovery_models.py` | MEDIUM | **NEW** — dormant behind `RECOVERY_MODELS_AVAILABLE = False` (HIGH-11); hosts SCI-FLAW-09/-10. Enabling the flag without first fixing its `NameError`s turns dormant defects into live ones. |

**Sequencing rule proposed for the matrix:** fixes to CRIT-01/02/12 (throughput/NPV reporting) must
land **before** CRIT-06/07 (sweep models), because the RF floors are currently the only thing keeping
the sweep outputs in a plausible range; removing the floor first would expose an unconstrained model
and produce a large, unexplained regression in results.

---

## 6. Software-quality verdict

Stated strictly as **SOFTWARE CORRECTNESS** (category 1), with no implication about physical or
predictive validity:

1. **The code runs, and fails loudly at the boundary.** **333 tests pass and 0 fail** (04-10-2026
   refresh; baseline was 329 pass / 4 fail, the 4 being one missing import — HIGH-10, now RESOLVED).
   There are no bare `except:` clauses in the active engine and no NaN-propagation paths that
   reach selection unpenalized (`FAILURE_PENALTY` at `optimisation_engine.py:725-727, 1543, 1576`).
   *Counterweight:* the two new `F821`s (HIGH-19) are invisible to that suite — a `NameError` in
   `pyvista_reservoir_canvas.py` is caught by `except Exception` and a `NameError` in a Qt slot is
   outside the tests entirely, so "0 failed" now coexists with two reachable runtime defects.
2. **Style debt is large but scientifically inert** — ~70 % of 3 132 ruff findings (3 244 after the
   refresh) are typing/whitespace.
3. **The real software defects are data-flow defects**: 6 F821 undefined names, 85 discarded
   computations, and above all the *produced-but-never-consumed* outputs behind CRIT-09/CRIT-10 and the
   inert-control family (CRIT-08, CRIT-13, MED-07/08/09/13).
4. **Documentation was out of sync with code** in a way that actively misleads — **four** wiki-referenced
   functions did not exist (`_calculate_engine_npv`, `_calculate_co2_purchased_recycled`,
   `_solve_pressure_ode`, `_calculate_pressure_profile`), a `deprecated/` tree the wiki said held the retired
   engines is not in the repository, and the verification matrix listed six tests that do not exist
   (MED-15, MED-16). *All corrected in `agent_wiki/` this session — 23 documentation files.*
5. **Test-suite shape is the opposite of what the risk profile needs**: 83 % on the module that is
   already best covered by this audit's own numeric verification, 7–10 % on the workbench/config code
   that determines inputs — and of the 36 scientific tests, 5 never call the code they claim to
   verify while 3 assert that a defect exists (HIGH-18). A green `pytest` is therefore weaker
   evidence than its exit code suggests.

**What this section deliberately does not say:** it does not rate the *science*. That is
`phd_audit.md`, and its predictive-validity verdict is issued there.

---

## 7. Evidence index

| Artifact | Path |
|---|---|
| Findings register (**53**) | `audit/scientific_flaws.md` |
| Parameter provenance (91 rows) | `audit/parameter_provenance.csv` |
| ruff JSON (UTF-16, Phase-1 baseline) | `audit/code_quality/ruff_report.json` |
| ruff JSON (UTF-8, 04-10-2026 refresh — 3 244 diagnostics, source of HIGH-19) | `audit/ruff_output.json` |
| vulture output | `audit/code_quality/dead_code_candidates.txt` |
| call graph | `audit/code_quality/call_graph.dot` |
| pytest + coverage console (pre-fix baseline: 4 failed / 329 passed) | `audit/runtime/pytest_output.txt` |
| coverage XML | `audit/runtime/coverage.xml` |
| Reproducible numeric scripts (durable copies) | `agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/evidence_scripts/` (**18** scripts, incl. `v_recount.py` = register self-count) |
| Originals of those scripts | `%TEMP%/opencode/audit_verify_*.py`, `v_*.py` (outside repo; run with `.venv\Scripts\python.exe -X utf8`) |
| Simulation-run audit record (Verdict / Proposal / Relevant Files) | `agent_wiki/audit/simulation_run_audits/04-10-2026_forensic_scientific_audit/audit.md` |
| Wiki corrections applied (MED-15 / MED-16) | 23 files under `agent_wiki/` — enumerated in §2.4 |
| Prior-session automated audit (stale baseline, left unedited) | `audit/report.md`, `audit/reports/scientific_audit_report.md`, `audit/scientific_flaws/scientific_flaws.csv` |
