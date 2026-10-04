# Change-Safety Classification Matrix

Before modifying any file in this repository, AI agents and developers must consult this classification matrix to assess the scientific and software risks of the proposed change.

> [!IMPORTANT]
> **Updated 2026-10-04 from the independent audit.** Rows added or amended in this revision are marked
> *(AMENDED 2026-10-04)* or *(NEW 2026-10-04)*. The authoritative list with per-finding rationale is
> `res_audit.md` §5; the findings themselves are in `audit/scientific_flaws.md` (**52 findings, 13 CRITICAL**).
> Two structural notes: (a) the previously listed `core/unified_engine/physics/eos/` **does not exist** in the
> tree, and (b) `tests/scientific/**` is classified **CRITICAL** here — verification code that asserts a defect
> is more dangerous than engine source, because it blocks the correct fix and prevents detecting a regression.

---

## 1. Risk Tier Definitions

| Tier | Definition | Required Pre-Modification Validation | Required Post-Modification Tests |
| :---: | :--- | :--- | :--- |
| **CRITICAL** | Equations, physical models, conversion constants, or parameters directly determining simulation results, recovery factors, or scientific conclusions. | Deep theoretical review; verify equation dimensional consistency; check benchmark impact. | Full pytest suite + CMG benchmark comparison (`validate_against_cmg`) + mass balance check. |
| **HIGH** | Simulation orchestration, profile generation, numerical ODE solvers, PVT interpolation, and constraint penalties. | Inspect call graph and callers; verify unit consistency; review potential solver stiffness. | Full pytest suite (`pytest tests/`) + edge-case tests (zero injection, zero permeability). |
| **MEDIUM** | Data parsing, file handlers, UI background workers, preference management, and reporting. | Trace data flow from input to data models. | Unit tests for relevant package + GUI launch test (`python main.py`). |
| **LOW** | Presentation UI widgets, translations (`i18n`), logging formatters, plotting styles, documentation. | Check widget layout and signal-slot connections. | GUI smoke test; verify no broken Qt signals. |

---

## 2. Component Risk Classification Table

| File / Component Path | Risk Tier | Primary Responsibilities | Danger / Failure Mode If Modified Incorrectly |
| :--- | :---: | :--- | :--- |
| `core/engine_surrogate/analytical_models.py` | **CRITICAL** | Recovery factor calculation (`PhDHybridSurrogate`, Koval, BL) | Aligns or breaks all optimization results; distorts recovery by orders of magnitude; introduces unphysical discontinuities. |
| `core/engine_surrogate/surrogate_engine.py` | **CRITICAL** | Inline damped material balance (was a `solve_ivp` ODE — removed), CO₂ mass balance, NPV pass-through | Can cause pressure divergence, negative pressures, or invalid economic decisions. *(AMENDED 2026-10-04)* Add: HCPVI cancellation at `:491-497` (CRIT-01), dual-ledger NPV (CRIT-02), post-hoc profile rescale (CRIT-12), WAG/SWAG ×1000 asymmetry (CRIT-11). **Line-number edits here invalidate CRIT-01/02/11/12 references.** |
| `core/engine_surrogate/profile_generator_fast.py` | **CRITICAL** | Time series rate synthesis, WAG modulation, breakthrough | Alters breakthrough time and cash flow profiles; can violate mass balance. |
| `core/data_models.py` (`PhysicalConstants`) | **CRITICAL** | Centralized physical constants and unit conversion factors | Altering conversion factors (e.g. $1.062 \times 10^{-14}$) creates million-fold injectivity errors. |
| `evaluation/mmp.py` | **CRITICAL** | Minimum Miscibility Pressure correlations | Dictates whether reservoir operates in miscible or immiscible regime. |
| `core/optimisation_engine.py` | **HIGH** | Algorithm execution, solution evaluation, penalty functions | Modifying parameter packing, penalty thresholds, or reintroducing penalty dilution (`* 0.1`) allows unphysical chromosomes to survive selection. Must strictly enforce `FAILURE_PENALTY` ($-10^{12}$) and NaN pruning. |
| `core/objectives/wrapper.py` | **CRITICAL** *(AMENDED 2026-10-04: HIGH → CRITICAL)* | Multi-objective scoring and metric aggregation | Reintroducing Class E synthetic modifiers (e.g. storage from RF) or magic fallbacks (`1e6`) falsifies results. Missing or unphysical profiles must strictly evaluate to `NaN` to trigger full pruning. **Why the upgrade:** CRIT-09 and CRIT-10 show its Class-VI sandface and leakage constraint blocks **cannot execute** (they read a profile key that is never populated, `profiles.get("pressure")` vs `yearly_pressure`/`monthly_pressure`) — code that looks protective but never runs invites false confidence, and its reported NPV/RF can come from two different evaluations (CRIT-02). |
| `core/unified_engine/physics/eos/` | ~~HIGH~~ **n/a** *(AMENDED 2026-10-04)* | *Path does not exist* — the tree was deleted, not relocated (MED-16) | Editing it is impossible; citing it wastes effort. The live EOS is `core/engine_surrogate/pvt_state.py` — see its row below. |
| `utils/project_file_handler.py` & save/load in `ui/` | **HIGH** | Project file (`.tphd`) JSON serialization, type preservation, UI state sync | Strips nested dataclass types, causes `IndexError` on 1D grids, breaks project loading, and loses user data/results. Run `pytest tests/test_project_save_load.py -v`. |
| `analysis/material_balance.py` | **MEDIUM** | Post-simulation CO₂ accounting and reporting | Produces misleading verification graphs if mass balance equations are distorted. |
| `utils/las_parser.py` | **MEDIUM** | Petrophysical well log parsing | Ingestion errors or corrupt permeability tracks. |
| `ui/workers/optimization_worker.py` | **MEDIUM** | Multiprocessing/threading for optimization | Thread deadlocks, GUI freezing, unhandled Qt exceptions. |
| `ui/main_window.py`, `ui/widgets/` | **LOW** | Desktop graphical interface presentation | Visual layout glitches, disabled buttons, signal-slot disconnections. |
| `utils/preferences_manager.py` | **LOW** | User settings persistence | Corrupt JSON preferences file (handled gracefully by defaults). |
| `core/engine_surrogate/pvt_state.py` | **CRITICAL** *(NEW 2026-10-04)* | Peng–Robinson Z, $B_o$, $B_g$, $\mu$ mixture state | Hosts CRIT-03 (no bubble point, $dBo/dP > 0$), CRIT-04 ($B_g$ 31.7× too small), CRIT-05 (Z > 1 in the dense-gas region), HIGH-12/16/17. Any edit must re-verify: sign of $dBo/dP$, magnitude of $B_g$ vs textbook, Z against Standing–Katz, and the $\mu_{\text{CO2}}$ correlation. |
| `core/engine_surrogate/surrogate_models.py` | **CRITICAL** *(NEW 2026-10-04)* | Reported NPV, trapping, areal sweep | Hosts the reported NPV (`:507-530`, CRIT-02/HIGH-08), the inverted trapping law (`:238`, HIGH-15), and the M = 1 Craig cliff (`:164-182`, HIGH-02). Previously unlisted. |
| `core/objectives/storage.py` | **CRITICAL** *(NEW 2026-10-04)* | Plume-containment score | Unpopulated `getattr` fallbacks make the containment floor exceed the pruning threshold, so the constraint can **never** prune (CRIT-08). |
| `core/engine_surrogate/geomechanics_fault.py` | **HIGH** *(NEW 2026-10-04)* | Leakage law, caprock/fault failure | Leakage constants plus the clip/threshold mismatch that reports "breached" with zero leakage (HIGH-05); leakage is structurally zero (CRIT-10). |
| `core/engine_surrogate/well_mechanics.py` | **HIGH** *(NEW 2026-10-04)* | Peaceman productivity/injectivity index | Two Darcy constants coexist; the inter-well transmissibility omits the $2\pi$ factor (6.3× low, MED-03), and invalid input returns a sentinel `1.0` (MED-04). |
| `core/engine_surrogate/relative_permeability.py` | **MEDIUM** *(NEW 2026-10-04)* | Corey $k_r$ curves | **Not in the evaluation path** (MED-05) — edits change dashboards only. Documenting it as *diagnostic* beats pretending it governs results; its $s_{org}$ denominator is wrong (MED-06). |
| `core/simulation/recovery_models.py` | **MEDIUM** *(NEW 2026-10-04)* | Legacy recovery models | Dormant behind `RECOVERY_MODELS_AVAILABLE = False` (HIGH-11); carries SCI-FLAW-09/-10. Enabling the flag without first fixing its `NameError`s turns dormant defects into live ones. |
| `core/data_integration_engine.py` | **MEDIUM** *(NEW 2026-10-04)* | UI/input PVT table generation | Its $B_o$/$\mu(P)$ signs were **corrected** in a previous round (`:431` $dBo/dP<0$, `:435`/`:441` $d\mu/dP>0$); re-breaking the sign would revive SCI-FLAW-02/-03, which the active `pvt_state.py` already exhibits (CRIT-03). |
| `ui/widgets/fault_geometry_visualizer_widget.py` | **LOW → MEDIUM** *(AMENDED 2026-10-04)* | Fault geometry rendering | A missing `QIcon` import (HIGH-10) currently fails 4 tests including the mandatory `tests/test_project_save_load.py` gate named in AGENTS.md invariant #5 — presentation code can block a scientific gate. |
| `ui/workbench/`, `ui/widgets/corey_relperm_*`, `fluids_pvt_*`, `well_network_*`, `core/geology/petrophysical_distribution.py` | **HIGH** *(NEW 2026-10-04)* | New workstation widgets / geology distribution | Untracked, ~0–9 % coverage, contains an F821 (`prev_field`). Untested code that is nonetheless imported by the app. |
| `scratch/` | **LOW** *(NEW 2026-10-04)* | Non-shipped experiment scripts | Should never be imported by shipped modules; excluded from risk otherwise. |
| `tests/scientific/**` | **CRITICAL** *(NEW 2026-10-04)* | Scientific verification suite | Verification-tier code. 5 of 36 tests import production symbols they never call; 3 assert **that a defect exists** (`step > 0.40`, `eff_high_sgc < eff_low_sgc`, `truncation_fraction > 0.20`). Re-specifying them against production code is higher risk than changing engine source: a wrong re-specification converts a detectable defect into an undetectable one. Require physical justification in the PR, not a green build (**HIGH-18**). |
| `agent_wiki/verification/test_matrix.md`, `agent_wiki/architecture/overview.md`, `agent_wiki/architecture/module_map.md` | **MEDIUM** *(NEW 2026-10-04)* | Coverage/architecture claims | Declared 42 tests/16 subdirectories (actual 36/15), listed 6 tests that do not exist, and cited a `deprecated/` tree that is not in the repository (**MED-16**). Regenerate counts from `pytest --collect-only`; never hand-edit them. |
