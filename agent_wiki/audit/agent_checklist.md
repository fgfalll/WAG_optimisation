# Master Agent Flaw & Scientific Showcase Checklist

> [!IMPORTANT]
> **MANDATORY AI AGENT PRE-FLIGHT DIRECTIVE**:
> Before scanning source code, running repetitive grep queries, or attempting model refactors,
> future AI agents MUST consult this showcase checklist. All 73 registered findings (52 bullet-block records + 21 migrated table rows, schema-validated by `python -m audit --register validate`), silent fallbacks,
> dead data-paths, and hardcoded calibration factors are already forensically cataloged with file:line
> anchors, measured evidence, and safe modification boundaries.

---

## 📊 Repository Forensic Health Dashboard

| Metric | Count | Assessment |
|---|---|---|
| **Total Confirmed Scientific Flaws** | **73** | 13 CRITICAL · 19 HIGH · 16 MEDIUM · 5 LOW |
| **Active Defects Status** | **56 OPEN** | 17 RESOLVED |
| **Cataloged Parameter Provenance** | **112 parameters** | Sources: Literature, Empirical, Calibrated, Unknown |
| **Undocumented / Arbitrary Constants** | **73** | Require physical calibration or field bounds |
| **Silent Fallback Exception Handlers** | **48 blocks** | Masking numerical divergences in `core/` |
| **Undefined Names (F821)** | **6** | Runtime `NameError` crash hazards |
| **Shadowed Redefinitions (F811)** | **7** | Overwritten functions/classes |
| **Discarded Calculations (F841)** | **85** | Computed variables never read |

---

## 🛡️ Safe Harbor: Verified Correct (DO NOT FLAG)

The following 13 mechanisms have been mathematically verified against analytical solutions. Agents **must NOT** waste tokens flagging them:

- [x] **Material-balance increment**: `dp = q·dt/(V_p·c_t + J_eff·dt)` — dimensionally consistent (bbl·psi), correct implicit form for a tank with well exchange (`surrogate_engine.py:455`).
- [x] **Gravity-number constant**: `4.3948e-5` (`analytical_models.py:793`) — matches `Δρ·g·k/(μ·φ)` in field units.
- [x] **Capillary-number factor**: `3.5e-6` (`analytical_models.py:752`) — consistent with field-unit N_c.
- [x] **Standing B_o coefficients**: and **Beggs–Robinson a/b** in `pvt_state.py` (structure verified; only the missing bubble point is flagged, CRIT-03).
- [x] **Peaceman horizontal/vertical index**: `0.00708·k·h/ln(...)` (`well_mechanics.py:26-27`, `:85`) — the 2π factor is present where it must be (only the *inter-well* function omits it, MED-03).
- [x] **Koval effective efficiency**: `E_eff = (0.78 + 0.22·M^0.25)^4` — normalized so that `E_eff(M=1) = 1.0` (0.78+0.22 = 1.0), as required.
- [x] **Koval throughput branches**: at `analytical_models.py:187-192` are continuous through `K → 1` (checked analytically).
- [x] **Peng–Robinson coefficients**: (`pvt_state.py:103-115`: ω = 0.225, T_c = 304.13 K, P_c = 7.376e6 Pa) — standard CO₂ values; root selection above P_c uses the smallest root (`:150-155`), which is the correct dense-phase branch.
- [x] **Harmonic (mole-basis) mixing of B_g**: at `pvt_state.py:385` — `1/Σ(y_i/B_i)` is the correct volumetric mixing rule when each `B_i` is on a molar basis; the *inputs* are wrong (CRIT-04), the *rule* is not.
- [x] **Geomechanical stress path and slip tendency**: (`geomechanics_fault.py:98`, `:129-132`, `:188-198`) — normal/shear resolution on a fault plane and the Biot-coupled `p_crit` are standard forms.
- [x] **VRR/voidage assembly**: `q_inj = q_CO₂·B_CO₂ + q_w·B_w` (`surrogate_engine.py:406`) — correct in form.
- [x] **MMP user override**: survives the Cronquist recomputation (LOW-02 notes the ordering dependency).
- [x] **`F821` `prev_field` at `petrophysical_distribution.py:404`**: is a lint artifact, not a bug: `prev_field = field` is bound at `:405` on the preceding iteration and the read at `:404` is guarded by `if k > 0` (`:402`), so the name is always bound when it is used. → do **not** "fix" by initializing a dummy variable that would change the recursion; if the warning must go, restructure the loop (with a test) rather than paper over it. (Contrast with the two genuine `F821`s in HIGH-19, which have no binding anywhere.)

---

## 📋 Master Flaw Showcase Checklist

| ID | Sev | Category | File:Line | Title / Headline Defect | Status |
|---|---|---|---|---|---|
| **CRIT-01** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/surrogate_engine.py:491-497` (overwrites `params["hcpvi"]` set at `core/engine_surrogate/surrogate_engine.py:890`)` | HCPVI is pressure-independent and dimensionally `MSCF/STB` | `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:489-500: HCPVI corrected to dimensionless pore-volume ratio with pressure-dependent Bg(P)/Bo(P).` |
| **CRIT-02** | **CRITICAL** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:165`, `:171-174`, `:490-510`, `:533`, `:685`, `:690`, `:692`; `core/engine_surrogate/surrogate_models.py:465`, `:473`, `:507-530`` | Reported NPV and reported recovery factor come from two different evaluations | `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:598-630: NPV derived directly from annual simulated production and injection profiles consistent with RF.` |
| **CRIT-03** | **CRITICAL** | PHYSICAL | ``core/engine_surrogate/pvt_state.py:232-244` (`R_s`), `:267-307` (swelling / `B_o`); used at `core/engine_surrogate/surrogate_engine.py:362`, `:369`, `:376`, `:453`` | No bubble-point model: `Bo` increases with pressure (apparent negative compressibility) | `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:83-103: Bubble-point model integrated with physically positive isothermal oil compressibility above Pb.` |
| **CRIT-04** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/pvt_state.py:372-373` (mixing at `:385`; consumed at `core/engine_surrogate/surrogate_engine.py:366`, `:414`, `:453`)` | Hydrocarbon-gas `B_g` coefficient is 31.73× too small (unit conversion inverted) | `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:377: Corrected Standing (1977) natural gas FVF conversion constant to 5.035 * Z * T_R / P.` |
| **CRIT-05** | **CRITICAL** | PHYSICAL | ``core/engine_surrogate/pvt_state.py:361-367`` | "Hall-Yarborough" Z-factor is a linear expression returning Z > 1 in the dense-gas region | `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:367: Implemented Papay (1968) natural gas Z-factor correlation and analytical dZ/dP with correct dense-gas behavior.` |
| **CRIT-06** | **CRITICAL** | NUMERICAL | ``core/engine_surrogate/analytical_models.py:541-559`` | Koval sweep returns RF = 0 for 0.5 ≤ M ≤ 1.4 with a spike at exactly M = 1.0, plus `exp` overflow for M < 1 | `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py:540: Authentic Koval (1963) continuous Welge throughput integral across all mobility ratios M.` |
| **CRIT-07** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/analytical_models.py:319-326`` | Immiscible model is the constant RF = 0.10 over the whole parameter space | `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py:296-326: Corrected Buckley-Leverett Welge shock-front tangent construction and removed unphysical 0.10 constant floor.` |
| **CRIT-08** | **CRITICAL** | SOFTWARE | ``core/optimisation_engine.py:1645-1666` (prune) with `core/objectives/storage.py:86-102` (score) and `core/data_models.py:1725-1730` (weights/threshold)` | Plume-containment constraint can never prune: floor of the score exceeds the threshold | `RESOLVED — Remediated in core/data_models.py:1815-1820, core/objectives/storage.py:86-115, and core/optimisation_engine.py:1657-1685: Added containment factors to AdvancedEngineParams and cascading pressure profile retrieval.` |
| **CRIT-09** | **CRITICAL** | SOFTWARE | ``core/objectives/wrapper.py:61-78`; key producer `core/optimisation_engine.py:841-896`` | Wrapper's sandface / Class-VI penalty block is unreachable (wrong profile key) | `RESOLVED — Remediated in core/objectives/wrapper.py:61-79 and core/optimisation_engine.py:853-860: Forwarded pressure and sandface injection profiles into objective evaluation.` |
| **CRIT-10** | **CRITICAL** | SOFTWARE | — | Every CO₂ leakage constraint is structurally zero (three dead paths) | `RESOLVED — Remediated in core/objectives/wrapper.py:81-99, core/optimisation_engine.py:853-860, and core/engine_surrogate/surrogate_engine.py: Forwarded leakage keys into constraint and objective workflows.` |
| **CRIT-11** | **CRITICAL** | PHYSICAL | ``core/engine_surrogate/profile_generator_fast.py:1109`, `:1117-1118` versus `:1210`; parameter `core/data_models.py:917`` | WAG water injection is ~1000× too low (missing unit conversion), so WAG degenerates to continuous gas | `RESOLVED — Remediated in core/engine_surrogate/profile_generator_fast.py:1109-1118: Added x1000 unit scaling for WAG gas-to-water injection conversion.` |
| **CRIT-12** | **CRITICAL** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:376-379`, `:412-417`, `:455`, versus post-hoc rescale at `:513-522`` | Production/saturation/pressure state is computed from an unscaled profile that is rescaled afterwards | `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:533-557: Re-synchronized saturations (So, Sw, Sg), cumulative oil, and VRR post-rescaling.` |
| **CRIT-13** | **CRITICAL** | SOFTWARE | ``core/optimisation_engine.py:120-142`, `:1420-1431`; `core/engine_surrogate/analytical_models.py:449`, `:458-459`; `core/engine_surrogate/surrogate_engine.py:865`, `:896`; `core/simulation/recovery_models.py:329`` | Three optimizer genes are inert (`gravity_factor`, `transition_alpha`, `transition_beta`) and a fourth is RF-inert (`mobility_ratio`) | `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py and surrogate_engine.py:905-950: Mapped and consumed gravity_factor, transition_alpha, transition_beta, mobility_ratio, and c7_plus.` |
| **CRIT-14** | **CRITICAL** | PHYSICAL | ``core/engine_surrogate/analytical_models.py:170-172`, `:275-277`, `:412-414`; source `core/engine_surrogate/surrogate_engine.py:976-977`` | The mobility-ratio override severs oil viscosity from recovery entirely | `OPEN` |
| **CRIT-15** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/analytical_models.py:558-571`; `core/engine_surrogate/surrogate_engine.py:946-968`` | Default configuration pins the Koval sweep at its 0.95 clip for every mobility ratio | `OPEN` |
| **CRIT-16** | **CRITICAL** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:963-964`` | `bo`/`b_co2` provenance guards are inert; the user's own PVT inputs are silently discarded | `OPEN` |
| **CRIT-17** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/surrogate_engine.py:535-551`` | The CRIT-12 fix breaks saturation closure: `S_o + S_w > 1` on 27 % of timesteps | `OPEN` |
| **CRIT-18** | **CRITICAL** | PHYSICAL | ``core/engine_surrogate/surrogate_engine.py:622-651` (revenue at `:636`), allocation `:588`, accumulation `:606`, publication `:688`` | The new NPV omits the hydrocarbon-gas revenue stream it computes | `OPEN` |
| **CRIT-19** | **CRITICAL** | PROVENANCE | ``core/engine_surrogate/analytical_models.py:199-201`, `:331-333`, `:802`; wired at `core/engine_surrogate/surrogate_engine.py:928`, `:974-975`` | `gravity_factor` is now an active, unprincipled, triple-purpose fudge multiplier | `OPEN` |
| **CRIT-20** | **CRITICAL** | MATHEMATICAL | ``core/engine_surrogate/analytical_models.py:474-481`` | Miscibility weight is decoupled from composition and equals 0.5 exactly at the MMP | `OPEN` |
| **CRIT-21** | **CRITICAL** | PROVENANCE | ``core/engine_surrogate/pvt_state.py:415-419`` | CO₂ compressibility is a new hard-coded power law contradicting the Peng-Robinson EOS in the same class | `OPEN` |
| **HIGH-01** | **HIGH** | MATHEMATICAL | ``core/engine_surrogate/surrogate_engine.py:509`; `core/engine_surrogate/analytical_models.py:814`` | `rf_max_physical` uses pore-volume fraction instead of OOIP-normalized fraction (25% too restrictive) | `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py and surrogate_engine.py:509: rf_max_physical normalized to OOIP (1 - Swi - Sor)/(1 - Swi).` |
| **HIGH-02** | **HIGH** | NUMERICAL | ``core/engine_surrogate/analytical_models.py:304-312`; `core/engine_surrogate/surrogate_models.py:164-182`; test `tests/scientific/mathematical/test_singularity_and_overflow.py:64-73`` | Craig areal-sweep 48% discontinuity at M = 1.0, and a test that asserts the defect | `CONFIRMED with corrected formula description` |
| **HIGH-03** | **HIGH** | MATHEMATICAL | ``core/engine_surrogate/analytical_models.py:193` (`clip(displacement_eff, 0.05, 0.95)`), `:200` (`clip(rf, 0.05, 0.85)`), `:326` (`clip(recovery, 0.10, max_cap)`), `:475` (`clip(rf, 0.05, 0.80)`)` | Recovery floors fabricate recovery at zero throughput | `NEW` |
| **HIGH-04** | **HIGH** | SOFTWARE | ``core/engine_surrogate/analytical_models.py:661` (`params.get("v_dp") or params.get("v_dp_coefficient") or 0.5`), `:662`, `:671` (`c7_plus_fraction or c7_plus or 0.3`); same idiom widely used in `_literature_based_recovery` (`:657-765` region)` | `or`-default pattern silently replaces legitimate zero values | `NEW` |
| **HIGH-05** | **HIGH** | PHYSICAL | ``core/engine_surrogate/surrogate_engine.py:442-443`, `:468`; `core/engine_surrogate/geomechanics_fault.py:113-118`, `:170`, `:174-181`, `:202-208`` | Caprock leakage is unreachable, and "breached" can be reported with zero leakage | `NEW` |
| **HIGH-06** | **HIGH** | MATHEMATICAL | ``core/engine_surrogate/surrogate_engine.py:894-895`; `core/engine_surrogate/analytical_models.py:174`, `:534`, `:726`; `core/data_models.py:57`; `core/engine_surrogate/profile_generator_fast.py:960`` | Three mutually inconsistent Koval heterogeneity-factor formulas (2.5× … 10⁹⁵× apart) | `NEW` |
| **HIGH-07** | **HIGH** | PHYSICAL | ``core/engine_surrogate/surrogate_engine.py:417`, `:420-433`` | Productivity/injectivity index is tautological; injection viscosity hard-coded | `NEW` |
| **HIGH-08** | **HIGH** | PHYSICAL | ``core/engine_surrogate/surrogate_models.py:507-530` (inputs mapped at `core/engine_surrogate/surrogate_engine.py:940-957`)` | NPV omits four economic inputs that the engine explicitly supplies, and costs CO₂ on *stored* rather than *purchased* | `NEW` |
| **HIGH-09** | **HIGH** | SOFTWARE | ``core/data_models.py:1719-1721`; `core/optimisation_engine.py:264-270`; `config/base_config.json:24`, `:746`; `core/engine_surrogate/analytical_models.py:431-475`` | Production default model is `hybrid`, but 72 test references pin `phd_hybrid` | `NEW` |
| **HIGH-10** | **HIGH** | SOFTWARE | ``ui/widgets/fault_geometry_visualizer_widget.py:115`` | Undefined name `QIcon` fails 4 tests | `RESOLVED — re-verified 04-10-2026 after commit `a68fc35`: `ui/widgets/fault_geometry_visualizer_widget.py:19` now carries `from PyQt6.QtGui import QIcon`, the F821 is gone, and `pytest tests/ -q` returns **`335 passed, 23 skipped, 0 failed`** (356 collected). `pytest tests/test_project_save_load.py` (AGENTS.md invariant #5 gate) now passes. `audit/runtime/pytest_output.txt` is retained as the **pre-fix** baseline and must be read as such. The finding is retained (not deleted) because the same defect *class* recurred twice in new code — see **HIGH-19**.` |
| **HIGH-11** | **HIGH** | SOFTWARE | ``core/engine_surrogate/analytical_models.py:30` (flag), `:96` (`MiscibleRecoveryModel`), `:218` (`ImmiscibleRecoveryModel`), `:341` (`BuckleyLeverettModel`)` | `RECOVERY_MODELS_AVAILABLE = False` is the only guard preventing `NameError`s | `NEW` |
| **HIGH-12** | **HIGH** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:788-790` (`except Exception → _error_result`), `:964-988` (`_error_result`: RF=0, NPV=0, `convergence_status: "error"`); `core/engine_surrogate/surrogate_models.py:553-562`; `core/engine_surrogate/pvt_state.py:147-149` (PR root fallback `Z = max(0.25, B*1.05)` with **no log**); `core/optimisation_engine.py:725-727`, `:903-904`, `:1572-1576`` | Exception funnel converts all failures into penalties without tracebacks | `NEW (severity reduced from an initial "plausible zeros survive" hypothesis — the error key **is** checked at `optimisation_engine.py:725`, so the failure mode is *masking*, not *silent acceptance*)` |
| **HIGH-13** | **HIGH** | SOFTWARE | ``core/objectives/wrapper.py:96-97`; policy value `core/data_models.py:1734` (`carbon_tax_usd_per_tonne: float = 75.0`)` | Wrapper hard-codes a $100/t carbon-tax floor over the user's value | `NEW` |
| **HIGH-14** | **HIGH** | PHYSICAL | ``core/objectives/wrapper.py:64-67` (`P_sandface = P_max + q_inj/II`, `II` default 25) vs `core/engine_surrogate/surrogate_engine.py:359` (`p_inj_sandface = min(current_p + 400, ceiling)`)` | Two different sandface-pressure models coexist | `NEW` |
| **HIGH-15** | **HIGH** | PHYSICAL | ``core/engine_surrogate/surrogate_models.py:238-241`; test `tests/scientific/co2/test_co2_trapping_mechanisms.py:18-34`` | Capillary gas-trapping term is inverted (and a test asserts the inversion) | `CONFIRMED` |
| **HIGH-16** | **HIGH** | PROVENANCE | ``core/engine_surrogate/pvt_state.py:204-226`` | CO₂ viscosity polynomial is not the cited correlation and under-predicts dense-phase μ ≈ 2× | `NEW` |
| **HIGH-17** | **HIGH** | PROVENANCE | ``core/engine_surrogate/pvt_state.py:388-394`` | Gas compressibility is a two-branch fudge with un-cited coefficients | `NEW` |
| **HIGH-18** | **HIGH** | SOFTWARE | ``tests/scientific/co2/test_co2_breakthrough_physics.py:16`, `:19-58`; `tests/scientific/conservation/test_mass_conservation.py` (`test_pore_volume_vs_ooip_recovery_bound_discrepancy`); `tests/scientific/mathematical/test_analytical_identities.py:11-12`; `tests/scientific/dimensional/test_unit_consistency.py:16-30`; `tests/scientific/co2/test_co2_trapping_mechanisms.py:31-33`; `tests/scientific/mathematical/test_singularity_and_overflow.py` (`test_mobility_ratio_unit_limit_singularity`)` | The scientific verification suite cannot fail on the defects it claims to verify | `NEW — **do not "fix" by weakening these tests**; re-specify them against production code (see `res_audit.md` §5, change-safety row NEW).` |
| **HIGH-19** | **HIGH** | SOFTWARE | ``ui/workbench/components/pyvista_reservoir_canvas.py:974` (`has_active_fault`); `ui/workbench/components/subsurface_data_viewer_widget.py:335` (`QToolTip`)` | Two undefined names in the new workbench code reproduce the HIGH-10 defect class | `RESOLVED — Remediated in ui/workbench/components/pyvista_reservoir_canvas.py (defined has_active_fault) and subsurface_data_viewer_widget.py (imported QToolTip).` |
| **HIGH-20** | **HIGH** | NUMERICAL | ``core/engine_surrogate/pvt_state.py:303-322`` | `B_o` is C⁰ but not C¹ at the bubble point (slope flips sign discontinuously) | `OPEN` |
| **HIGH-21** | **HIGH** | PROVENANCE | ``core/engine_surrogate/pvt_state.py:101-106`; conflicting value at `core/engine_surrogate/surrogate_engine.py:315`` | Bubble point is a hard-coded constant and `c_o` defaults conflict between modules | `OPEN` |
| **HIGH-23** | **HIGH** | PHYSICAL | ``core/engine_surrogate/surrogate_engine.py:635`, `:644`; `core/optimisation_engine.py:1372-1386`; `core/objectives/wrapper.py:108-132`` | Containment is economically inert: leakage is identically zero, yet leaked CO₂ would earn storage credit | `OPEN` |
| **HIGH-24** | **HIGH** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:417` vs `:545-548`` | Two live VRR definitions coexist; the post-hoc one silently overwrites the integrated one | `NEW` |
| **HIGH-25** | **HIGH** | NUMERICAL | ``core/engine_surrogate/analytical_models.py:309-313`` | `s_g_avg` masking: a non-positive tangent slope silently becomes 100 % displacement efficiency | `NEW` |
| **HIGH-26** | **HIGH** | PHYSICAL | ``core/engine_surrogate/pvt_state.py:375-400`` | CO₂ properties mix a Peng-Robinson FVF with a correlation-based `Z` in one mixture | `NEW` |
| **MED-01** | **MEDIUM** | PROVENANCE | `data_models.py:1858-1860 (0.05254), pvt_state.py:41-42 (1.838 kg/m³, 0.05295), profile_generator_fast.py:794 (0.05297), surrogate_models.py:26, analytical_models.py:37, wrapper.py:90 (0.053)` | Five different "tonnes/MSCF" constants (spread **1.0%**) plus a comment claiming `0.1234 lb/ft³` ... | `NEW` |
| **MED-02** | **MEDIUM** | PROVENANCE | `pvt_state.py:8, :19 (docstring), :175-176, :201` | Docstring advertises Span–Wagner; only Peng–Robinson is implemented (grep: no Span–Wagner code) | `NEW` |
| **MED-03** | **MEDIUM** | MATHEMATICAL | `well_mechanics.py:197, :220 (const at :28)` | Radial form `T = 0.001127·k·h/(μ·ln(D/r_w))` mixes the **linear** Darcy constant with a radial lo... | `NEW` |
| **MED-04** | **MEDIUM** | SOFTWARE | `well_mechanics.py:65-66, :113-114` | Invalid input (`k ≤ 0`, `h ≤ 0`, …) returns `1.0` silently (PI in STB/d/psi). | `NEW` |
| **MED-05** | **MEDIUM** | SOFTWARE | `relative_permeability.py (all), well_mechanics.py:184-221, surrogate_models.py:371` | Relative permeability and inter-well transmissibility are **not in the evaluation path**: the eng... | `NEW` |
| **MED-06** | **MEDIUM** | MATHEMATICAL | `relative_permeability.py:55` | `so_norm = (so − s_orw)/denom_w` uses the **water** denominator even when `s_org ≠ s_orw` (the no... | `NEW` |
| **MED-07** | **MEDIUM** | SOFTWARE | `optimisation_engine.py:1209-1256 (fn _calculate_adaptive_penalty), call site :1783; death branch :1225-1230` | Two defects | `NEW (corrected: prior text named a non-existent `_calculate_penalty` and claimed 0 call sites)` |
| **MED-08** | **MEDIUM** | SOFTWARE | `optimisation_engine.py:1531, :1560-1562` | `is_feasible` is computed and returned, then only logged; the penalty is applied separately at `:... | `NEW` |
| **MED-09** | **MEDIUM** | SOFTWARE | `optimisation_engine.py:96 vs :1543; value data_models.py:1709` | Module-level `FAILURE_PENALTY = -1e12` is shadowed inside the wrapper by `self.advanced_engine_pa... | `NEW` |
| **MED-10** | **MEDIUM** | SOFTWARE | `optimisation_engine.py:841-896, :1611, :1652; data_models.py:1262, :1273; profile_generator_fast.py:103` | Valid `time_resolution` values are `weekly/monthly/quarterly/yearly`; profiles only ever expose `... | `NEW` |
| **MED-11** | **MEDIUM** | PROVENANCE | `data_models.py:913, :917; profile_generator_fast.py:1111-1115` | `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = base_injection_rate × mobilit... | `RESOLVED — Remediated in profile_generator_fast.py:1109-1118 (CRIT-11)` |
| **MED-12** | **MEDIUM** | PROVENANCE | `surrogate_engine.py:352, :354, :356` | `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil + 0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000),... | `NEW` |
| **MED-13** | **MEDIUM** | SOFTWARE | `repo-wide` | Ruff **3244** violations over **191** files (re-run 04-10-2026 after `a68fc35`; top: UP006 857, W... | `NEW (metrics refreshed 04-10-2026)` |
| **MED-14** | **MEDIUM** | SOFTWARE | `optimisation_engine.py:905-935` | If no simulation engine is available the code silently falls back to `ProductionProfiler` (differ... | `NEW` |
| **MED-15** | **MEDIUM** | PROVENANCE | `agent_wiki/README.md:35, source_of_truth_map.md:16, source_of_truth.md:26, common_pitfalls.md:34, code/inventory.md:26, execution_flow.md:81-82, code/functions/simulation_functions.md, code/classes/surrogate_classes.md, validation/conservation.md` | The wiki documents `_calculate_engine_npv()` and `_calculate_co2_purchased_recycled()` as the sou... | `NEW (corrected in `agent_wiki/` as part of this deliverable)` |
| **MED-16** | **MEDIUM** | PROVENANCE | `agent_wiki/verification/test_matrix.md:5, :18-46; agent_wiki/architecture/overview.md:71; agent_wiki/README.md:29` | The master verification matrix is stale on four counts | `NEW (corrected in `agent_wiki/verification/test_matrix.md` as part of this deliverable)` |
| **MED-17** | **MEDIUM** | SOFTWARE | ``core/engine_surrogate/surrogate_engine.py:671`` | `monthly_oil_stb` is length-1 and all zeros | `NEW` |
| **MED-18** | **MEDIUM** | SOFTWARE | ``core/engine_surrogate/analytical_models.py:561-562`` | The `kv <= 1` Koval branch is unreachable | `NEW` |
| **MED-19** | **MEDIUM** | PROVENANCE | ``core/engine_surrogate/analytical_models.py:199-201`` | `MiscibleSurrogate` RF multiplied by a new `e_v`, and the RF clip raised in the same edit | `NEW` |
| **MED-20** | **MEDIUM** | SOFTWARE | ``core/optimisation_engine.py:852`` | `annual_water_stb` reporting key deleted with no replacement | `NEW` |
| **MED-21** | **MEDIUM** | SOFTWARE | ``core/objectives/wrapper.py:130`` | The $100/t remediation floor (HIGH-13) survives the remediation | `NEW` |
| **MED-22** | **MEDIUM** | SOFTWARE | ``core/objectives/wrapper.py:112-121`` | Dead second leakage model still present in the objective wrapper | `NEW` |
| **LOW-01** | **LOW** | PHYSICAL | `surrogate_engine.py:356` | `y_co2` is clipped to a **minimum of 0.05** even at `cum_inj = 0`, so the gas phase always contai... | `NEW` |
| **LOW-02** | **LOW** | SOFTWARE | `surrogate_engine.py:930-938` | Dynamic MMP is recomputed and **overwrites** `params["mmp"]` with Cronquist on every call, wrappe... | `NEW` |
| **LOW-03** | **LOW** | SOFTWARE | `pvt_state.py:143` | `np.roots` per timestep for a cubic EOS (3 roots, eigen-solver) in a hot loop. | `NEW` |
| **LOW-04** | **LOW** | SOFTWARE | `core/engine_surrogate/surrogate_engine.py:37` | `MSCF_PER_TONNE` imported but unused (vulture, 90% confidence) alongside a live `CO2_TONNE_PER_MS... | `NEW` |
| **LOW-05** | **LOW** | NUMERICAL | `optimisation_engine.py:930-935` | RF > 1.0 is silently clamped to 1.0 with a warning after the fact instead of preventing `rf_max_p... | `NEW` |

---

## 🚨 CRITICAL Severity Flaw Register (21 items)

### [x] CRIT-01 — HCPVI is pressure-independent and dimensionally `MSCF/STB`
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:489-500: HCPVI corrected to dimensionless pore-volume ratio with pressure-dependent Bg(P)/Bo(P).`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:491-497` (overwrites `params["hcpvi"]` set at `core/engine_surrogate/surrogate_engine.py:890`)`
- **Expected Behavior**: `HCPVI = cum_inj_mscf · Bg(P) · (1 - Swi) / (OOIP · Bo(P))`, i.e. injected reservoir barrels divided by hydrocarbon pore volume, which is strongly pressure-dependent through `Bg/Bo`.
- **Scientific Impact**: Throughput is the independent variable of the Koval/Ekladios sweep and of every miscible RF correlation in the module. Measured error versus the textbook definition for the same case: **0.667× (1500 psi), 1.218× (2500 psi), 1.868× (3500 psi)** — a monotonically growing, pressure-driven bias that silently rewards high-pressure operation. It also discards the correct `params["hcpvi"]` computed at `:890`, so the code computes the right quantity once and then throws it away.
- **Evidence**: `audit_verify_3.py` section V1. Dimensional check: `[RB]/([STB]·[RB/STB])` in the denominator vs `[RB]` in the numerator ⇒ dimensionless only if the numerator were reservoir barrels of **oil-equivalent**; as written both `b_co2` factors cancel exactly.

### [x] CRIT-02 — Reported NPV and reported recovery factor come from two different evaluations
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:598-630: NPV derived directly from annual simulated production and injection profiles consistent with RF.`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:165`, `:171-174`, `:490-510`, `:533`, `:685`, `:690`, `:692`; `core/engine_surrogate/surrogate_models.py:465`, `:473`, `:507-530``
- **Expected Behavior**: One physical evaluation per simulation: one RF, one oil profile, one CO₂ ledger, one cash flow, all consistent at the same pressure and throughput.
- **Scientific Impact**: The primary objective (`npv`) is not a function of the recovery factor the run reports. Ranking candidates by NPV therefore ranks them by an internally inconsistent quantity; every downstream artifact (cash-flow table, storage efficiency, RF-vs-NPV cross plots) mixes two states. Two independent CO₂ mass ledgers exist simultaneously (breakthrough-aware vs injected−produced) — a direct threat to invariant *"cumulative recycled ≤ produced ≤ injected"* auditing.
- **Evidence**: Code reading of the two evaluation sites; `audit_verify_4.py` section D shows RF is strongly hcpvi-dependent (0.05 → 0.713 over hcpvi 0 → 3), so the hcpvi substitution between the two calls is not a rounding difference.

### [x] CRIT-03 — No bubble-point model: `Bo` increases with pressure (apparent negative compressibility)
- **Severity**: `CRITICAL` | **Category**: `PHYSICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:83-103: Bubble-point model integrated with physically positive isothermal oil compressibility above Pb.`
- **Location**: ``core/engine_surrogate/pvt_state.py:232-244` (`R_s`), `:267-307` (swelling / `B_o`); used at `core/engine_surrogate/surrogate_engine.py:362`, `:369`, `:376`, `:453``
- **Observed Behavior**: `bubble_point` occurs **0 times** in `core/engine_surrogate/` (grep, 04-10-2026). `R_s = c1·γg·p^c2·exp(c3·API/T_R)` is monotone increasing in `p` for every pressure, so `B_o` is monotone increasing in `p` for every pressure.
- **Expected Behavior**: Above the bubble point `dB_o/dP < 0`, with `c_o ≈ 3e-6 … 1.2e-5 1/psi` for a black oil ( McCain, W.D. (1990), *The Properties of Petroleum Fluids* ). Below the bubble point `R_s` is constant at `R_sb`. A positive `c_o` of order 1e-4 means the oil "expands when compressed".
- **Scientific Impact**: (a) Saturation bookkeeping `S_o = remaining_oil·B_o/V_p` (`surrogate_engine.py:376`) inflates oil saturation as pressure rises, so gas/water saturations are displaced. (b) The material-balance denominator uses a **constant** `c_o = 1e-5` (`surrogate_engine.py:315`) that contradicts the PVT module's own derivative by an order of magnitude — the compressibility in the ODE and the compressibility implied by `B_o(P)` are two different fluids. (c) Voidage `q_prod·B_o` (`:412-414`) grows with pressure, feeding back into the pressure ODE (`:455`).
- **Evidence**: `audit_verify_3.py` section V4; zero-occurrence grep for `bubble_point`. Prior-audit SCI-FLAW-02 reported the same sign defect in `core/data_integration_engine.py` (a different module); this finding shows the **active** PVT path has the identical class of defect.

### [x] CRIT-04 — Hydrocarbon-gas `B_g` coefficient is 31.73× too small (unit conversion inverted)
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:377: Corrected Standing (1977) natural gas FVF conversion constant to 5.035 * Z * T_R / P.`
- **Location**: ``core/engine_surrogate/pvt_state.py:372-373` (mixing at `:385`; consumed at `core/engine_surrogate/surrogate_engine.py:366`, `:414`, `:453`)`
- **Expected Behavior**: `B_g = 5.035·Z·T_R/P` rb/MSCF (Standing 1977 / any field-unit textbook). Pure-CO₂ FVF must come from the same law or from `ρ_CO₂`.
- **Scientific Impact**: Produced-gas voidage `q_prod += gas·B_g` (`surrogate_engine.py:412-414`) is understated 11–24×, so (i) VRR (`:417`) is overstated, (ii) the pressure decline `dp` (`:455`) is far too small, (iii) `c_g` (`:367`, `:453`) and `B_g` enter material balance as if produced gas occupied almost no reservoir volume. Gas-cap/voidage physics is effectively switched off for hydrocarbon gas.
- **Evidence**: `audit_verify_3.py` sections V2/V2b; arithmetic identity `0.02827×5.6146 = 0.15873`.

### [x] CRIT-05 — "Hall-Yarborough" Z-factor is a linear expression returning Z > 1 in the dense-gas region
- **Severity**: `CRITICAL` | **Category**: `PHYSICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/pvt_state.py:367: Implemented Papay (1968) natural gas Z-factor correlation and analytical dZ/dP with correct dense-gas behavior.`
- **Location**: ``core/engine_surrogate/pvt_state.py:361-367``
- **Expected Behavior**: Hall-Yarborough (1972) is an iterative tangent-compressibility correlation; the Standing–Katz chart at Tpr ≈ 1.58 gives Z ≈ 0.83–0.95 over Ppr = 2–5, i.e. **Z < 1 with a minimum around Ppr ≈ 2–3**, never a monotonic rise above 1. A gas with Z = 1.15 at Ppr = 3.7 does not exist at that reduced state.
- **Scientific Impact**: `ρ_hc` (`:371`) is computed ~20–35% too low and the *sign of dZ/dP* is wrong, so `c_g = 1/P − (1/Z)(dZ/dP)` (`:388`) must be fudged by hand (`:391-394`, see HIGH-18) — the two defects mask each other, which is exactly why numerical agreement in downstream numbers cannot be credited as validation.
- **Evidence**: `audit_verify_4.py` section A; Standing, M.B. (1977); Hall, K.E. & Yarborough, L. (1972) *J. Pet. Tech.* — the code's own label does not match the code's formula.

### [x] CRIT-06 — Koval sweep returns RF = 0 for 0.5 ≤ M ≤ 1.4 with a spike at exactly M = 1.0, plus `exp` overflow for M < 1
- **Severity**: `CRITICAL` | **Category**: `NUMERICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py:540: Authentic Koval (1963) continuous Welge throughput integral across all mobility ratios M.`
- **Location**: ``core/engine_surrogate/analytical_models.py:541-559``
- **Expected Behavior**: Koval (1963) breakthrough sweep `V_b` is continuous and strictly decreasing in M over (0, ∞), with `V_b → 1` as M → 0. No region of identically zero recovery may exist for finite throughput; the M = 1 limit must be the continuous limit of the general branch.
- **Scientific Impact**: In the M range typical of CO₂–crude systems the model reports **zero recovery**; because the optimizer explores M (it is a relaxable constraint, `optimisation_engine.py:127`), the objective surface contains a large flat zero plateau with a 0.3167 delta-function at M = 1.0 — a guaranteed trap for both GA (selection pressure destroyed) and BO/gradient methods.
- **Evidence**: `audit_verify_3.py` section V5 (RuntimeWarning on overflow observed); Koval, E.J. (1963) *SPE J.* 3(2), 145–152.

### [x] CRIT-07 — Immiscible model is the constant RF = 0.10 over the whole parameter space
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py:296-326: Corrected Buckley-Leverett Welge shock-front tangent construction and removed unphysical 0.10 constant floor.`
- **Location**: ``core/engine_surrogate/analytical_models.py:319-326``
- **Expected Behavior**: Immiscible CO₂ flooding recovers 10–40% OOIP depending on `M`, `V_DP`, `S_or` (Buckley & Leverett 1942; Craig 1971). The returned value must be a *function* of the inputs.
- **Scientific Impact**: The immiscible limb carries **zero gradient and zero sensitivity**. Because `HybridSurrogate` blends this constant with the miscible limb (`analytical_models.py:468-472`, weights `w_miscible`), the entire hybrid model's response to `S_or`, `V_DP` and `M` through the immiscible branch is null; and the floor itself fabricates 10% recovery when the physics computes ~1.4–5.6%.
- **Evidence**: `audit_verify_3.py` section V6; internal decomposition in `audit_verify_4.py` section C.

### [x] CRIT-08 — Plume-containment constraint can never prune: floor of the score exceeds the threshold
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/data_models.py:1815-1820, core/objectives/storage.py:86-115, and core/optimisation_engine.py:1657-1685: Added containment factors to AdvancedEngineParams and cascading pressure profile retrieval.`
- **Location**: ``core/optimisation_engine.py:1645-1666` (prune) with `core/objectives/storage.py:86-102` (score) and `core/data_models.py:1725-1730` (weights/threshold)`
- **Expected Behavior**: A critical-threshold constraint must be reachable: `min(S_cont) < threshold` for the configured weights, and the seal/structure factors must come from the user's `CO2StorageParameters` (`data_models.py:1620`, `:1628`), not from unlinked defaults.
- **Scientific Impact**: The Class-VI-style containment guard advertised in `AdvancedEngineParams` is a **no-op**. Optimizer candidates are never rejected for containment reasons; the only visible variation in `S_cont` is an artifact of time-resolution key selection.
- **Evidence**: `audit_verify_3.py` section V11; `data_models.py:1725-1730`; `storage.py:91-92` (`getattr(..., 0.9/0.85)` fallbacks).

### [x] CRIT-09 — Wrapper's sandface / Class-VI penalty block is unreachable (wrong profile key)
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/objectives/wrapper.py:61-79 and core/optimisation_engine.py:853-860: Forwarded pressure and sandface injection profiles into objective evaluation.`
- **Location**: ``core/objectives/wrapper.py:61-78`; key producer `core/optimisation_engine.py:841-896``
- **Observed Behavior**: `pressure_profile = profiles.get("pressure", profiles.get("reservoir_pressure"))`. The `profiles` dict handed to `_calculate_objective_functions` (`optimisation_engine.py:944-946`) is built at `:841-896` and contains only `yearly_pressure` / `annual_pressure` (`:853-854`) and `monthly_pressure` (`:870`) — never `pressure` or `reservoir_pressure`. Result: `pressure_profile is None` ⇒ the whole `if` block is skipped ⇒ `results["npv"]` never receives the containment penalty and `geomechanical_violation` is never emitted.
- **Expected Behavior**: Either read `yearly_/monthly_pressure`, or forward the engine's `pressure` / `sandface_injection_pressure` arrays.
- **Scientific Impact**: EPA Class-VI 90% fracture-limit enforcement inside the objective (`safe_fracture_limit = 0.90·P_frac`) never runs on the optimization path; the code and the wiki both imply it does.
- **Evidence**: Full key inventory `optimisation_engine.py:841-896`; single call site `_calculate_objective_functions` at `:944`.

### [x] CRIT-10 — Every CO₂ leakage constraint is structurally zero (three dead paths)
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/objectives/wrapper.py:81-99, core/optimisation_engine.py:853-860, and core/engine_surrogate/surrogate_engine.py: Forwarded leakage keys into constraint and objective workflows.`
- **Location**: ``
- **Expected Behavior**: Leakage produced by the geomechanical module must flow into (a) the constraint penalty, (b) the NPV remediation cost, (c) the reported storage balance `M_stored = M_inj − M_prod − M_leaked`.
- **Scientific Impact**: CO₂ containment violations have **no economic or selection-pressure consequence** anywhere in the pipeline. Wiki invariant #3/#10 ("mass conservation / leakage accounted") is violated in the implementation, not merely in wording.
- **Evidence**: Two repo-wide greps for `annual_leakage_tonne` and `max_sandface_pressure_psi`; consumer grep for `total_leakage_tonne`.

### [x] CRIT-11 — WAG water injection is ~1000× too low (missing unit conversion), so WAG degenerates to continuous gas
- **Severity**: `CRITICAL` | **Category**: `PHYSICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/profile_generator_fast.py:1109-1118: Added x1000 unit scaling for WAG gas-to-water injection conversion.`
- **Location**: ``core/engine_surrogate/profile_generator_fast.py:1109`, `:1117-1118` versus `:1210`; parameter `core/data_models.py:917``
- **Expected Behavior**: WAG water rate must equal `q_gas[MSCFD] · B_g[rb/MSCF] · WAG_ratio`, i.e. tens of thousands of bpd at field rates (and the same `default_gas_fvf` must be used consistently in both schemes).
- **Scientific Impact**: With 5–25 bpd of water against 5000 MSCFD of CO₂, the WAG scheme produces essentially **continuous gas injection**: no mobility control, no mobility banking, no deferred gas. `cum_water_inj_bbl` (`surrogate_engine.py:474`) feeds `S_w` (`:377-378`) and `q_inj_step_rb` (`:406`) ⇒ VRR and saturation paths are computed for a reservoir that is receiving no water. The optimizer's WAG-ratio gene is therefore optimizing a scheme that does not physically exist.
- **Evidence**: `audit_verify_3.py` reading of `:1117` vs `:1210`; both use the same `default_gas_fvf` default (0.005), so the only difference is the omitted `×1000`.

### [x] CRIT-12 — Production/saturation/pressure state is computed from an unscaled profile that is rescaled afterwards
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/surrogate_engine.py:533-557: Re-synchronized saturations (So, Sw, Sg), cumulative oil, and VRR post-rescaling.`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:376-379`, `:412-417`, `:455`, versus post-hoc rescale at `:513-522``
- **Observed Behavior**: Inside the time loop the engine integrates `nominal_oil_stb` (`:408`, `:472`) into `cum_oil_stb`, from which `S_o` (`:376`), `q_prod_step_rb` (`:412-414`), `VRR` (`:417`) and `dp` (`:455`) are all computed. **After** the loop, `profile_result["oil_profile"]` is multiplied by `target_cum_oil/max_cum_shape` (`:519`) so that cumulative oil matches RF₂. The saturated/pressured history is never recomputed.
- **Expected Behavior**: Either the profile is correct before integration (then no rescale is needed), or the entire state trajectory is recomputed after rescaling. A profile may not be rescaled after the state variables that depend on it have been consumed.
- **Scientific Impact**: Reported `S_o/S_w/S_g` profiles, `B_o/B_g` profiles, VRR and the pressure path are mutually inconsistent with the reported oil rate and cumulative oil. Any material-balance cross-check of the reported streams (injected vs produced vs stored) will fail or, if it passes, passes only because `cum_oil_total` is recomputed from the rescaled array at `:533` while saturations are not.
- **Evidence**: Prior-audit SCI-FLAW-06 (`CONFIRMED_AUDIT_OPEN`) — this round re-verified the exact ordering and added the specific dependent variables.

### [x] CRIT-13 — Three optimizer genes are inert (`gravity_factor`, `transition_alpha`, `transition_beta`) and a fourth is RF-inert (`mobility_ratio`)
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py and surrogate_engine.py:905-950: Mapped and consumed gravity_factor, transition_alpha, transition_beta, mobility_ratio, and c7_plus.`
- **Location**: ``core/optimisation_engine.py:120-142`, `:1420-1431`; `core/engine_surrogate/analytical_models.py:449`, `:458-459`; `core/engine_surrogate/surrogate_engine.py:865`, `:896`; `core/simulation/recovery_models.py:329``
- **Expected Behavior**: Every gene exposed to GA/BO must have a non-zero gradient in at least one reported objective; transition parameters must reach the model that is actually executed; parameter names passed between modules must match.
- **Scientific Impact**: 4 of 9 relaxable constraints (`data_models.py:1692-1704`) cannot influence recovery. The optimizer spends evaluations exploring null directions; sensitivity/tornado reports for these parameters (`sensitivity_analyzer.py:143`, `:639`) will show pure noise, and any conclusion drawn from them is invalid.
- **Evidence**: Repo-wide greps for `gravity_factor` / `transition_alpha`; `audit_verify_3.py` section V7; name mismatch `c7_plus` vs `c7_plus_fraction` verified by grep.

### [ ] CRIT-14 — The mobility-ratio override severs oil viscosity from recovery entirely
- **Severity**: `CRITICAL` | **Category**: `PHYSICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/analytical_models.py:170-172`, `:275-277`, `:412-414`; source `core/engine_surrogate/surrogate_engine.py:976-977``
- **Scientific Impact**: the entire viscosity-contrast mechanism — the physical basis of CO₂ flooding — is removed from the miscible, immiscible and Buckley-Leverett limbs. Recovery becomes a function of one constant.
- **Evidence**: `audit_verify_5.py` §B1–B3. **Status:** NEW

### [ ] CRIT-15 — Default configuration pins the Koval sweep at its 0.95 clip for every mobility ratio
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/analytical_models.py:558-571`; `core/engine_surrogate/surrogate_engine.py:946-968``
- **Scientific Impact**: with CRIT-14, recovery is nearly constant over the whole mobility axis; selection pressure on these genes is arbitrary.
- **Evidence**: `audit_verify_5.py` §D4/§I, `audit_verify_6.py` §I. **Status:** NEW

### [ ] CRIT-16 — `bo`/`b_co2` provenance guards are inert; the user's own PVT inputs are silently discarded
- **Severity**: `CRITICAL` | **Category**: `SOFTWARE` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:963-964``
- **Scientific Impact**: a user with fitted live-oil FVF cannot reach HCPVI; the sweep throughput is unoverrideable.
- **Evidence**: `audit_verify_6.py` §G. **Status:** NEW

### [ ] CRIT-17 — The CRIT-12 fix breaks saturation closure: `S_o + S_w > 1` on 27 % of timesteps
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:535-551``
- **Scientific Impact**: the three-phase saturation history violates its defining constraint, and every saturation-derived quantity is corrupted invisibly — Phase-3 anti-pattern class D. A 25 % saturation bias exists at t = 0.
- **Evidence**: `audit_verify_6.py` §J, `audit_verify_9.py`. **Status:** NEW (regression from the CRIT-12 remediation)

### [ ] CRIT-18 — The new NPV omits the hydrocarbon-gas revenue stream it computes
- **Severity**: `CRITICAL` | **Category**: `PHYSICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:622-651` (revenue at `:636`), allocation `:588`, accumulation `:606`, publication `:688``
- **Scientific Impact**: `npv` is the primary objective; the optimiser ranks candidates on a truncated cash-flow model.
- **Evidence**: `audit_verify_9.py`. **Status:** NEW

### [ ] CRIT-19 — `gravity_factor` is now an active, unprincipled, triple-purpose fudge multiplier
- **Severity**: `CRITICAL` | **Category**: `PROVENANCE` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/analytical_models.py:199-201`, `:331-333`, `:802`; wired at `core/engine_surrogate/surrogate_engine.py:928`, `:974-975``
- **Scientific Impact**: a gene that scales recovery ±20 % with no physical content, applied twice in the hybrid path plus once in `N_g`. Hidden-calibration pattern. CRIT-13 had recorded this gene as *inert*; making it active is a net loss.
- **Evidence**: `audit_verify_5.py` §B; source reads. **Status:** NEW

### [ ] CRIT-20 — Miscibility weight is decoupled from composition and equals 0.5 exactly at the MMP
- **Severity**: `CRITICAL` | **Category**: `MATHEMATICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/analytical_models.py:474-481``
- **Scientific Impact**: the central miscible/immiscible decision is governed by two uncalibrated constants and is insensitive to the fluid.
- **Evidence**: `audit_verify_5.py` §C/§C1. **Status:** NEW

### [ ] CRIT-21 — CO₂ compressibility is a new hard-coded power law contradicting the Peng-Robinson EOS in the same class
- **Severity**: `CRITICAL` | **Category**: `PROVENANCE` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/pvt_state.py:415-419``
- **Scientific Impact**: `c_g` enters `ct` (`surrogate_engine.py:453`) and hence `dp` (`:455`); an 8× error near-critical distorts the pressure path where CO₂ floods are most sensitive. HIGH-17's undocumented fudge was replaced by a differently shaped one.
- **Evidence**: `audit_verify_5.py` §A3/§A4. **Status:** NEW

---

## 🚨 HIGH Severity Flaw Register (25 items)

### [x] HIGH-01 — `rf_max_physical` uses pore-volume fraction instead of OOIP-normalized fraction (25% too restrictive)
- **Severity**: `HIGH` | **Category**: `MATHEMATICAL` | **Status**: `RESOLVED — Remediated in core/engine_surrogate/analytical_models.py and surrogate_engine.py:509: rf_max_physical normalized to OOIP (1 - Swi - Sor)/(1 - Swi).`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:509`; `core/engine_surrogate/analytical_models.py:814``

### [ ] HIGH-02 — Craig areal-sweep 48% discontinuity at M = 1.0, and a test that asserts the defect
- **Severity**: `HIGH` | **Category**: `NUMERICAL` | **Status**: `CONFIRMED with corrected formula description`
- **Location**: ``core/engine_surrogate/analytical_models.py:304-312`; `core/engine_surrogate/surrogate_models.py:164-182`; test `tests/scientific/mathematical/test_singularity_and_overflow.py:64-73``

### [ ] HIGH-03 — Recovery floors fabricate recovery at zero throughput
- **Severity**: `HIGH` | **Category**: `MATHEMATICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:193` (`clip(displacement_eff, 0.05, 0.95)`), `:200` (`clip(rf, 0.05, 0.85)`), `:326` (`clip(recovery, 0.10, max_cap)`), `:475` (`clip(rf, 0.05, 0.80)`)`

### [ ] HIGH-04 — `or`-default pattern silently replaces legitimate zero values
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:661` (`params.get("v_dp") or params.get("v_dp_coefficient") or 0.5`), `:662`, `:671` (`c7_plus_fraction or c7_plus or 0.3`); same idiom widely used in `_literature_based_recovery` (`:657-765` region)`

### [ ] HIGH-05 — Caprock leakage is unreachable, and "breached" can be reported with zero leakage
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:442-443`, `:468`; `core/engine_surrogate/geomechanics_fault.py:113-118`, `:170`, `:174-181`, `:202-208``

### [ ] HIGH-06 — Three mutually inconsistent Koval heterogeneity-factor formulas (2.5× … 10⁹⁵× apart)
- **Severity**: `HIGH` | **Category**: `MATHEMATICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:894-895`; `core/engine_surrogate/analytical_models.py:174`, `:534`, `:726`; `core/data_models.py:57`; `core/engine_surrogate/profile_generator_fast.py:960``

### [ ] HIGH-07 — Productivity/injectivity index is tautological; injection viscosity hard-coded
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:417`, `:420-433``

### [ ] HIGH-08 — NPV omits four economic inputs that the engine explicitly supplies, and costs CO₂ on *stored* rather than *purchased*
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_models.py:507-530` (inputs mapped at `core/engine_surrogate/surrogate_engine.py:940-957`)`

### [ ] HIGH-09 — Production default model is `hybrid`, but 72 test references pin `phd_hybrid`
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/data_models.py:1719-1721`; `core/optimisation_engine.py:264-270`; `config/base_config.json:24`, `:746`; `core/engine_surrogate/analytical_models.py:431-475``

### [x] HIGH-10 — Undefined name `QIcon` fails 4 tests
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — re-verified 04-10-2026 after commit `a68fc35`: `ui/widgets/fault_geometry_visualizer_widget.py:19` now carries `from PyQt6.QtGui import QIcon`, the F821 is gone, and `pytest tests/ -q` returns **`335 passed, 23 skipped, 0 failed`** (356 collected). `pytest tests/test_project_save_load.py` (AGENTS.md invariant #5 gate) now passes. `audit/runtime/pytest_output.txt` is retained as the **pre-fix** baseline and must be read as such. The finding is retained (not deleted) because the same defect *class* recurred twice in new code — see **HIGH-19**.`
- **Location**: ``ui/widgets/fault_geometry_visualizer_widget.py:115``

### [ ] HIGH-11 — `RECOVERY_MODELS_AVAILABLE = False` is the only guard preventing `NameError`s
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:30` (flag), `:96` (`MiscibleRecoveryModel`), `:218` (`ImmiscibleRecoveryModel`), `:341` (`BuckleyLeverettModel`)`

### [ ] HIGH-12 — Exception funnel converts all failures into penalties without tracebacks
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW (severity reduced from an initial "plausible zeros survive" hypothesis — the error key **is** checked at `optimisation_engine.py:725`, so the failure mode is *masking*, not *silent acceptance*)`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:788-790` (`except Exception → _error_result`), `:964-988` (`_error_result`: RF=0, NPV=0, `convergence_status: "error"`); `core/engine_surrogate/surrogate_models.py:553-562`; `core/engine_surrogate/pvt_state.py:147-149` (PR root fallback `Z = max(0.25, B*1.05)` with **no log**); `core/optimisation_engine.py:725-727`, `:903-904`, `:1572-1576``

### [ ] HIGH-13 — Wrapper hard-codes a $100/t carbon-tax floor over the user's value
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/objectives/wrapper.py:96-97`; policy value `core/data_models.py:1734` (`carbon_tax_usd_per_tonne: float = 75.0`)`

### [ ] HIGH-14 — Two different sandface-pressure models coexist
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: ``core/objectives/wrapper.py:64-67` (`P_sandface = P_max + q_inj/II`, `II` default 25) vs `core/engine_surrogate/surrogate_engine.py:359` (`p_inj_sandface = min(current_p + 400, ceiling)`)`

### [ ] HIGH-15 — Capillary gas-trapping term is inverted (and a test asserts the inversion)
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `CONFIRMED`
- **Location**: ``core/engine_surrogate/surrogate_models.py:238-241`; test `tests/scientific/co2/test_co2_trapping_mechanisms.py:18-34``

### [ ] HIGH-16 — CO₂ viscosity polynomial is not the cited correlation and under-predicts dense-phase μ ≈ 2×
- **Severity**: `HIGH` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/pvt_state.py:204-226``

### [ ] HIGH-17 — Gas compressibility is a two-branch fudge with un-cited coefficients
- **Severity**: `HIGH` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/pvt_state.py:388-394``

### [ ] HIGH-18 — The scientific verification suite cannot fail on the defects it claims to verify
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW — **do not "fix" by weakening these tests**; re-specify them against production code (see `res_audit.md` §5, change-safety row NEW).`
- **Location**: ``tests/scientific/co2/test_co2_breakthrough_physics.py:16`, `:19-58`; `tests/scientific/conservation/test_mass_conservation.py` (`test_pore_volume_vs_ooip_recovery_bound_discrepancy`); `tests/scientific/mathematical/test_analytical_identities.py:11-12`; `tests/scientific/dimensional/test_unit_consistency.py:16-30`; `tests/scientific/co2/test_co2_trapping_mechanisms.py:31-33`; `tests/scientific/mathematical/test_singularity_and_overflow.py` (`test_mobility_ratio_unit_limit_singularity`)`
- **Scientific Impact**: The **Analytical Verification** tier of the evidence hierarchy is non-informative for these items. A green `36 passed` is reported in `agent_wiki/verification/test_matrix.md` as `VERIFIED` / `CONTRADICTED BY TEST`, which reads as scientific assurance but is produced by tests that re-state the code's own arithmetic. Three defects are *enshrined* as regression expectations (HIGH-02, HIGH-15, SCI-FLAW-11): remediation will be blocked by a red build, and no test will notice a re-introduction of SCI-FLAW-01.
- **Evidence**: `v_tests2.py` (AST: `A) tautological = 5`, `B) no production import = 6`, `C) exercises production = 25`); `pytest tests/scientific` → `36 passed`; source reads cited above.

### [x] HIGH-19 — Two undefined names in the new workbench code reproduce the HIGH-10 defect class
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `RESOLVED — Remediated in ui/workbench/components/pyvista_reservoir_canvas.py (defined has_active_fault) and subsurface_data_viewer_widget.py (imported QToolTip).`
- **Location**: ``ui/workbench/components/pyvista_reservoir_canvas.py:974` (`has_active_fault`); `ui/workbench/components/subsurface_data_viewer_widget.py:335` (`QToolTip`)`

### [ ] HIGH-20 — `B_o` is C⁰ but not C¹ at the bubble point (slope flips sign discontinuously)
- **Severity**: `HIGH` | **Category**: `NUMERICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/pvt_state.py:303-322``
- **Scientific Impact**: spurious gradient sign flips near `P_b` perturb GA/BO line searches.
- **Evidence**: `audit_verify_5.py` §A7. **Status:** NEW

### [ ] HIGH-21 — Bubble point is a hard-coded constant and `c_o` defaults conflict between modules
- **Severity**: `HIGH` | **Category**: `PROVENANCE` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/pvt_state.py:101-106`; conflicting value at `core/engine_surrogate/surrogate_engine.py:315``
- **Scientific Impact**: the `P_b` default silently decides saturation state, which flips the sign of `dB_o/dP` and therefore the whole volumetric bookkeeping.
- **Evidence**: `audit_verify_5.py` §A/§A5. **Status:** NEW

### [ ] HIGH-23 — Containment is economically inert: leakage is identically zero, yet leaked CO₂ would earn storage credit
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `OPEN`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:635`, `:644`; `core/optimisation_engine.py:1372-1386`; `core/objectives/wrapper.py:108-132``
- **Scientific Impact**: the model pays for CO₂ it cannot retain and never charges for CO₂ it loses. The externality of a containment failure is priced at zero while the revenue for the same molecules is priced at $25/t — inverting EPA Class VI risk incentives. The most consequential finding for a carbon-storage project.
- **Evidence**: `audit_verify_9.py`; `audit_verify_7.py` §R; wiki invariants #10/#11. **Status:** NEW

### [ ] HIGH-24 — Two live VRR definitions coexist; the post-hoc one silently overwrites the integrated one
- **Severity**: `HIGH` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:417` vs `:545-548``
- **Scientific Impact**: VRR is the diagnostic used to judge voidage replacement; the physically coupled (integrated) value is discarded.

### [ ] HIGH-25 — `s_g_avg` masking: a non-positive tangent slope silently becomes 100 % displacement efficiency
- **Severity**: `HIGH` | **Category**: `NUMERICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:309-313``
- **Scientific Impact**: the immiscible limb's RF can jump to its physical maximum on a numerical artefact — anti-pattern class D inside the block whose gradient CRIT-07 otherwise correctly restored.

### [ ] HIGH-26 — CO₂ properties mix a Peng-Robinson FVF with a correlation-based `Z` in one mixture
- **Severity**: `HIGH` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/pvt_state.py:375-400``
- **Scientific Impact**: `bg_dynamic` feeds produced-gas voidage (`:412-414`), VRR (`:417`) and `dp` (`:455`); the result is not a property of any real gas.

---

## 🚨 MEDIUM Severity Flaw Register (22 items)

### [ ] MED-01 — Five different "tonnes/MSCF" constants (spread **1.0%**) plus a comment claiming `0.1234 lb/ft³` ...
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: `data_models.py:1858-1860 (0.05254), pvt_state.py:41-42 (1.838 kg/m³, 0.05295), profile_generator_fast.py:794 (0.05297), surrogate_models.py:26, analytical_models.py:37, wrapper.py:90 (0.053)`
- **Observed Behavior**: Five different "tonnes/MSCF" constants (spread **1.0%**) plus a comment claiming `0.1234 lb/ft³` and `0.056 t/MSCF` (both wrong: 1.838 kg/m³ = 0.115 lb/ft³ ≈ 0.0520 t/MSCF).
- **Expected Behavior**: one authoritative constant with a cited standard condition.
- **Scientific Impact**: 1% mass-balance bias, and an auditable comment that is false.
- **Evidence**: Grep output above; `analytical_models.py:35-37`

### [ ] MED-02 — Docstring advertises Span–Wagner; only Peng–Robinson is implemented (grep: no Span–Wagner code)
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: `pvt_state.py:8, :19 (docstring), :175-176, :201`
- **Observed Behavior**: Docstring advertises Span–Wagner; only Peng–Robinson is implemented (grep: no Span–Wagner code). `B_CO2 = 327.362/ρ` assumes 52.046 kg/MSCF (exact ≈ 52.6 kg ⇒ ~1.1% high). No volume translation (PR-78 typically 5–10% low on dense CO₂ ρ).
- **Expected Behavior**: cite PR only, or implement Span–Wagner; state the molar basis.
- **Scientific Impact**: ~1–10% bias in CO₂ reservoir-volume and tonnage.
- **Evidence**: `pvt_state.py:121-161` (PR only); arithmetic 1000 scf × 1.838 kg/m³

### [ ] MED-03 — Radial form `T = 0.001127·k·h/(μ·ln(D/r_w))` mixes the **linear** Darcy constant with a radial lo...
- **Severity**: `MEDIUM` | **Category**: `MATHEMATICAL` | **Status**: `NEW`
- **Location**: `well_mechanics.py:197, :220 (const at :28)`
- **Observed Behavior**: Radial form `T = 0.001127·k·h/(μ·ln(D/r_w))` mixes the **linear** Darcy constant with a radial logarithm ⇒ **2π = 6.28× too low**.
- **Expected Behavior**: `0.00708` (`DARCY_FIELD_CONSTANT`, defined at `:27` and correctly used at `:85`).
- **Scientific Impact**: inter-well connectivity understated 6.3× wherever used.
- **Evidence**: `well_mechanics.py:26-28`, `:85`, `:220`; consumers are UI-only (MED-05)

### [ ] MED-04 — Invalid input (`k ≤ 0`, `h ≤ 0`, …) returns `1.0` silently (PI in STB/d/psi).
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `well_mechanics.py:65-66, :113-114`
- **Observed Behavior**: Invalid input (`k ≤ 0`, `h ≤ 0`, …) returns `1.0` silently (PI in STB/d/psi).
- **Expected Behavior**: raise or return `0.0` with a log.
- **Scientific Impact**: a sentinel 1.0 is summed into `j_peaceman_inj_base` (`surrogate_engine.py:296-309`) and dilutes the index.
- **Evidence**: Source

### [ ] MED-05 — Relative permeability and inter-well transmissibility are **not in the evaluation path**: the eng...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `relative_permeability.py (all), well_mechanics.py:184-221, surrogate_models.py:371`
- **Observed Behavior**: Relative permeability and inter-well transmissibility are **not in the evaluation path**: the engine computes saturations by material balance (`surrogate_engine.py:376-379`) with no `k_rel`, no fractional flow, no `T_ij`. Consumers of rel-perm are UI/dashboard only (`ui/data_management_widget.py`, `ui/widgets/model_evaluation_dashboard.py`); `calculate_storage_efficiency` is imported only by `tests/test_physics_validation.py`.
- **Expected Behavior**: document these as *diagnostic* modules (they are currently described as active physics).
- **Scientific Impact**: false confidence that Corey rel-perm governs results.
- **Evidence**: Repo-wide import grep

### [ ] MED-06 — `so_norm = (so − s_orw)/denom_w` uses the **water** denominator even when `s_org ≠ s_orw` (the no...
- **Severity**: `MEDIUM` | **Category**: `MATHEMATICAL` | **Status**: `NEW`
- **Location**: `relative_permeability.py:55`
- **Observed Behavior**: `so_norm = (so − s_orw)/denom_w` uses the **water** denominator even when `s_org ≠ s_orw` (the normal case, `:43-44`).
- **Expected Behavior**: normalize by `1 − s_wc − s_org`.
- **Scientific Impact**: gas/oil relative-permeability endpoints mis-scaled.
- **Evidence**: Source

### [ ] MED-07 — Two defects
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW (corrected: prior text named a non-existent `_calculate_penalty` and claimed 0 call sites)`
- **Location**: `optimisation_engine.py:1209-1256 (fn _calculate_adaptive_penalty), call site :1783; death branch :1225-1230`
- **Observed Behavior**: Two defects. (a) `total_violation = 0.0` is hard-coded at `:1246` with every real check commented out (`:1232-1253`) ⇒ `if total_violation <= 0: return 0.0` at `:1255-1256` fires unconditionally, so the **single** call site at `:1783` always receives `0.0`. (b) The advertised `"death"` constraint-handling method is literally `pass` (`:1225-1230`) — selecting "death penalty" in `constraint_handling_method` changes nothing.
- **Expected Behavior**: implement the checks or delete the API.
- **Scientific Impact**: a "static/adaptive/death penalty" API advertised to the GA is inert, and `penalty_factor`/`constraint_handling_method` are dead knobs.
- **Evidence**: Grep (1 call site at `:1783`, always returns 0.0); source read of `:1209-1256`

### [ ] MED-08 — `is_feasible` is computed and returned, then only logged; the penalty is applied separately at `:...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `optimisation_engine.py:1531, :1560-1562`
- **Observed Behavior**: `is_feasible` is computed and returned, then only logged; the penalty is applied separately at `:1771-1772`.
- **Expected Behavior**: use the flag to gate/prune or drop it.
- **Scientific Impact**: dead branch that looks like constraint handling.
- **Evidence**: Source

### [ ] MED-09 — Module-level `FAILURE_PENALTY = -1e12` is shadowed inside the wrapper by `self.advanced_engine_pa...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `optimisation_engine.py:96 vs :1543; value data_models.py:1709`
- **Observed Behavior**: Module-level `FAILURE_PENALTY = -1e12` is shadowed inside the wrapper by `self.advanced_engine_params.failure_penalty`; both are `-1e12` **today**, so behaviour diverges only if a user changes `failure_penalty`.
- **Expected Behavior**: single source of truth.
- **Scientific Impact**: latent split-brain penalty scale.
- **Evidence**: Source

### [ ] MED-10 — Valid `time_resolution` values are `weekly/monthly/quarterly/yearly`; profiles only ever expose `...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `optimisation_engine.py:841-896, :1611, :1652; data_models.py:1262, :1273; profile_generator_fast.py:103`
- **Observed Behavior**: Valid `time_resolution` values are `weekly/monthly/quarterly/yearly`; profiles only ever expose `yearly_*` and `monthly_*`; engine time base is **monthly** by default while `OperationalParameters` defaults to **yearly**; fallbacks use the non-existent `"daily"`.
- **Expected Behavior**: one resolution vocabulary, propagated.
- **Scientific Impact**: `weekly`/`quarterly` runs silently read empty arrays (see CRIT-08 for the consequence).
- **Evidence**: Source

### [x] MED-11 — `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = base_injection_rate × mobilit...
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `RESOLVED — Remediated in profile_generator_fast.py:1109-1118 (CRIT-11)`
- **Location**: `data_models.py:913, :917; profile_generator_fast.py:1111-1115`
- **Observed Behavior**: `default_gas_fvf = 0.005` has no documented unit; `mobility_ratio = base_injection_rate × mobility_ratio_factor(0.001)` is compared against `high_mobility_threshold = 2.0` labelled a *mobility ratio* ⇒ the adaptive WAG logic triggers for any rate > 2000 MSCFD (dimensionally meaningless).
- **Expected Behavior**: document units (rb/scf) and compare a true mobility ratio.
- **Scientific Impact**: WAG enhancement is decided by injection rate, not by mobility.
- **Evidence**: Source; CRIT-11

### [ ] MED-12 — `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil + 0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000),...
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: `surrogate_engine.py:352, :354, :356`
- **Observed Behavior**: `oil_mass = ooip·0.135`, `x_co2 = 0.55·M_inj/(M_oil + 0.55·M_inj)`, `y_co2 = clip(cum/(cum+1000), 0.05, 0.95)` — solubility/vapor-fraction surrogates with no cited source (1000 MSCF time constant, 0.55 mass factor, 0.135 t/STB).
- **Expected Behavior**: mark `UNKNOWN — EVIDENCE REQUIRED` and tie to a flash calculation.
- **Scientific Impact**: `x_co2/y_co2` drive all PVT (B_o, μ, B_g mixture).
- **Evidence**: Source + `audit/parameter_provenance.csv`

### [ ] MED-13 — Ruff **3244** violations over **191** files (re-run 04-10-2026 after `a68fc35`; top: UP006 857, W...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW (metrics refreshed 04-10-2026)`
- **Location**: `repo-wide`
- **Observed Behavior**: Ruff **3244** violations over **191** files (re-run 04-10-2026 after `a68fc35`; top: UP006 857, W293 686, UP045 426, **F401 329 unused imports**, I001 253, **F841 85 unused locals**, **F821 6**, F811 7 — at audit baseline: 3132 / 192 / F401 345 / F841 77 / F821 5 / F811 5); vulture **41** dead-code candidates (04-10-2026 run; a re-run after `a68fc35` timed out on `RecursionError`, so 41 is the last verifiable figure); coverage **57 %** on the configured basis (7 791 statements, 05-10-2026); pytest **`0 failed / 333 passed / 23 skipped`** (re-run 04-10-2026; was `4 failed / 329 passed / 23 skipped`); jscpd not installed (duplicate detection not run).
- **Expected Behavior**: triage F401/F841/F821 first (they hide real defects), keep style noise separate.
- **Scientific Impact**: F401/F841 mask dead physics (CRIT-13) and F821 hides runtime `NameError`s (HIGH-11, HIGH-19; HIGH-10 resolved).
- **Evidence**: `audit/ruff_output.json` (UTF-8, re-run), `audit/code_quality/ruff_report.json` (UTF-16, baseline), `dead_code_candidates.txt`, `audit/runtime/coverage.xml`, `pytest_output.txt`

### [ ] MED-14 — If no simulation engine is available the code silently falls back to `ProductionProfiler` (differ...
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `optimisation_engine.py:905-935`
- **Observed Behavior**: If no simulation engine is available the code silently falls back to `ProductionProfiler` (different physics) and then clamps RF into [0,1] *after the fact* (`:930-935`).
- **Expected Behavior**: fail loudly, or mark results as `engine_type = profiler` everywhere downstream.
- **Scientific Impact**: two physics in one result namespace.
- **Evidence**: Source

### [ ] MED-15 — The wiki documents `_calculate_engine_npv()` and `_calculate_co2_purchased_recycled()` as the sou...
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW (corrected in `agent_wiki/` as part of this deliverable)`
- **Location**: `agent_wiki/README.md:35, source_of_truth_map.md:16, source_of_truth.md:26, common_pitfalls.md:34, code/inventory.md:26, execution_flow.md:81-82, code/functions/simulation_functions.md, code/classes/surrogate_classes.md, validation/conservation.md`
- **Observed Behavior**: The wiki documents `_calculate_engine_npv()` and `_calculate_co2_purchased_recycled()` as the source of truth for economics. **Neither function exists anywhere in the repository** (grep for 'def _calculate_engine_npv', 'def _calculate_co2_purchased_recycled', 'def _solve_pressure_ode'
- **Expected Behavior**: 0 hits). Real code: purchased/recycled at `surrogate_engine.py:572-596`; NPV at `surrogate_models.py:507-530`.
- **Scientific Impact**: wiki must match code.
- **Evidence**: Grep; corrected during this round (see wiki edits)

### [ ] MED-16 — The master verification matrix is stale on four counts
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW (corrected in `agent_wiki/verification/test_matrix.md` as part of this deliverable)`
- **Location**: `agent_wiki/verification/test_matrix.md:5, :18-46; agent_wiki/architecture/overview.md:71; agent_wiki/README.md:29`
- **Observed Behavior**: The master verification matrix is stale on four counts. (a) It declares **"42 test items across 16 subdirectories"**; `pytest --collect-only tests/scientific` returns **36 items in 15 subdirectories**. (b) **6 of its 40 listed tests no longer exist anywhere in `tests/`**: `test_co2_density_thermal_expansion`, `test_cubic_eos_z_factor_bounds`, `test_phase_label_assignment`, `test_peng_robinson_fugacity_equation_structure`, `test_corey_relative_permeability_bounds`, `test_bg_discrepancy_between_modules` (grep = 0 hits each) — five of them cite `unified_engine/…`. (c) `test_koval_fractional_flow_mobility_inversion` was **renamed** to `..._monotonicity` (0 hits for the old name) and 2 existing tests (`test_profile_generator_co2_breakthrough_gas_rate_increases_with_mobility`, `test_alston_impurity_mmp_trend`) are **unlisted**. (d) `overview.md:71` and `README.md:29` state the legacy engines were *"relocated into `deprecated/`"* — **neither `deprecated/` nor `core/unified_engine/` exists** (`unified_engine` is referenced 51× in the wiki, 0× as a real path).
- **Expected Behavior**: regenerate the matrix from `--collect-only` and stop citing deleted trees.
- **Scientific Impact**: readers conclude that six thermodynamic assertions are being enforced when no such test exists.
- **Evidence**: `pytest --collect-only tests/scientific` (36 items); 7 greps returning 0; `Test-Path deprecated`, `core/unified_engine` = False

### [ ] MED-17 — `monthly_oil_stb` is length-1 and all zeros
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/surrogate_engine.py:671``
- **Scientific Impact**: consumers resolving monthly resolution (MED-10) read zeros; `summary_monthly.csv` will be empty.

### [ ] MED-18 — The `kv <= 1` Koval branch is unreachable
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:561-562``

### [ ] MED-19 — `MiscibleSurrogate` RF multiplied by a new `e_v`, and the RF clip raised in the same edit
- **Severity**: `MEDIUM` | **Category**: `PROVENANCE` | **Status**: `NEW`
- **Location**: ``core/engine_surrogate/analytical_models.py:199-201``
- **Scientific Impact**: two changes to the same physical bound in one edit; the ceiling change alone raises achievable RF by up to 6 %.

### [ ] MED-20 — `annual_water_stb` reporting key deleted with no replacement
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/optimisation_engine.py:852``

### [ ] MED-21 — The $100/t remediation floor (HIGH-13) survives the remediation
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/objectives/wrapper.py:130``
- **Scientific Impact**: any user carbon price below $100/t is silently floored.

### [ ] MED-22 — Dead second leakage model still present in the objective wrapper
- **Severity**: `MEDIUM` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: ``core/objectives/wrapper.py:112-121``

---

## 🚨 LOW Severity Flaw Register (5 items)

### [ ] LOW-01 — `y_co2` is clipped to a **minimum of 0.05** even at `cum_inj = 0`, so the gas phase always contai...
- **Severity**: `LOW` | **Category**: `PHYSICAL` | **Status**: `NEW`
- **Location**: `surrogate_engine.py:356`
- **Observed Behavior**: `y_co2` is clipped to a **minimum of 0.05** even at `cum_inj = 0`, so the gas phase always contains ≥ 5% CO₂ before any injection.
- **Expected Behavior**: 0.0 when nothing is injected.
- **Scientific Impact**: negligible pre-injection bias in mixture B_g/μ.
- **Evidence**: Documented in flaw register

### [ ] LOW-02 — Dynamic MMP is recomputed and **overwrites** `params["mmp"]` with Cronquist on every call, wrappe...
- **Severity**: `LOW` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `surrogate_engine.py:930-938`
- **Observed Behavior**: Dynamic MMP is recomputed and **overwrites** `params["mmp"]` with Cronquist on every call, wrapped in `except`
- **Expected Behavior**: `params["mmp"]` falls back to `default_mmp_fallback`. User `mmp` override survives only because `sim_kwargs["mmp"]` is applied via `params.update(kwargs)` **after** `_build_params_dict` (`optimisation_engine.py:647`, `surrogate_engine.py:158`).
- **Scientific Impact**: explicit precedence documented.
- **Evidence**: Documented in flaw register

### [ ] LOW-03 — `np.roots` per timestep for a cubic EOS (3 roots, eigen-solver) in a hot loop.
- **Severity**: `LOW` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `pvt_state.py:143`
- **Observed Behavior**: `np.roots` per timestep for a cubic EOS (3 roots, eigen-solver) in a hot loop.
- **Expected Behavior**: closed-form trigonometric solution for the depressed cubic.
- **Scientific Impact**: performance only (no scientific impact).
- **Evidence**: Documented in flaw register

### [ ] LOW-04 — `MSCF_PER_TONNE` imported but unused (vulture, 90% confidence) alongside a live `CO2_TONNE_PER_MS...
- **Severity**: `LOW` | **Category**: `SOFTWARE` | **Status**: `NEW`
- **Location**: `core/engine_surrogate/surrogate_engine.py:37`
- **Observed Behavior**: `MSCF_PER_TONNE` imported but unused (vulture, 90% confidence) alongside a live `CO2_TONNE_PER_MSCF`.
- **Expected Behavior**: remove or use one constant.
- **Scientific Impact**: two reciprocal constants invite unit errors (see MED-01).
- **Evidence**: Documented in flaw register

### [ ] LOW-05 — RF > 1.0 is silently clamped to 1.0 with a warning after the fact instead of preventing `rf_max_p...
- **Severity**: `LOW` | **Category**: `NUMERICAL` | **Status**: `NEW`
- **Location**: `optimisation_engine.py:930-935`
- **Observed Behavior**: RF > 1.0 is silently clamped to 1.0 with a warning after the fact instead of preventing `rf_max_physical` violations upstream (HIGH-01).
- **Expected Behavior**: raise/flag as a model failure.
- **Scientific Impact**: hides upstream cap bugs.
- **Evidence**: Documented in flaw register

---

## 🔍 Parameter Provenance & Hidden Calibration Showcase

| Parameter | Value | Unit | Source Code Location | Provenance | Assessment |
|---|---|---|---|---|---|
| `CO2 density (tonnes/MSCF) - canonical field default` | 0.053 | tonne/MSCF | `core/data_models.py:725; core/engine_surrogate/analytical_models.py:37; core/engine_surrogate/surrogate_models.py:26` | **EMPIRICAL** | ACCEPTABLE - matches 1.838 kg/m3 x 28.3168 m3/MSCF / 1000 = 0.05204 within 1.9%; |
| `CO2 density - storage/PVT constant` | 0.05254 | tonne/MSCF | `core/data_models.py:1858-1860` | **EMPIRICAL** | INCONSISTENT with 0.053 default - 0.9% spread (MED-01) |
| `CO2 density - PVT module constant` | 0.05295 | tonne/MSCF | `core/engine_surrogate/pvt_state.py:42` | **EMPIRICAL** | INCONSISTENT with 0.053 default (MED-01) |
| `CO2 density - fast profile generator default` | 0.05297 | tonne/MSCF | `core/engine_surrogate/profile_generator_fast.py:794` | **EMPIRICAL** | INCONSISTENT with 0.053 default (MED-01) |
| `CO2 standard density (SC conditions)` | 1.838 | kg/m3 | `core/engine_surrogate/pvt_state.py:41` | **LITERATURE** | OK - NIST/CRC value for CO2 at 60 F and 14.7 psia |
| `CO2 mass per MSCF (docstring)` | 52.046 | kg/MSCF | `core/engine_surrogate/pvt_state.py:175` | **LITERATURE** | OK arithmetic (28.3168 m3 x 1.838 kg/m3) but contradicts 0.053 t/MSCF=53 kg (MED |
| `Comment: CO2 density in lb/ft3` | 0.1234 | lb/ft3 | `core/engine_surrogate/analytical_models.py:35` | **UNKNOWN** | EVIDENCE REQUIRED - wrong; true value 1.838 kg/m3 = 0.1147 lb/ft3 (MED-01) |
| `Comment: CO2 density in t/MSCF (analytical_models)` | 0.056 | tonne/MSCF | `core/engine_surrogate/analytical_models.py:36` | **UNKNOWN** | EVIDENCE REQUIRED - contradicts its own constant 0.053 on the next line (MED-01) |
| `default_gas_fvf (B_g)` | 0.005 | RB/scf (assumed) | `core/data_models.py:917; core/engine_surrogate/profile_generator_fast.py:1109` | **UNKNOWN** | EVIDENCE REQUIRED - unit unstated; plausible as RB/scf (0.005 RB/scf = 5 RB/MSCF |
| `CO2 leakage base rate` | 0.05 | tonne/day/psi overpressure | `core/engine_surrogate/geomechanics_fault.py:177` | **UNKNOWN** | EVIDENCE REQUIRED - no cited source; drives all caprock leakage magnitudes (MED) |
| `CO2 leakage breach multiplier` | 5.0 | dimensionless | `core/engine_surrogate/geomechanics_fault.py:179` | **UNKNOWN** | EVIDENCE REQUIRED - arbitrary 5x jump when is_caprock_breached |
| `CO2 leakage overpressure exponent` | 1.5 | dimensionless | `core/engine_surrogate/geomechanics_fault.py:177` | **UNKNOWN** | EVIDENCE REQUIRED - no cited source |
| `fault_slip_transmissibility_multiplier` | 10.0 | dimensionless | `core/data_models.py:1567` | **UNKNOWN** | EVIDENCE REQUIRED - 10x fault transmissibility multiplier has no cited basis |
| `penalty base fraction (generation < threshold)` | 0.05 | fraction | `core/optimisation_engine.py:1328` | **CALIBRATED** | EVIDENCE REQUIRED - tuned heuristic; hard-coded fraction of failure_penalty |
| `penalty base fraction (late generations)` | 0.10 | fraction | `core/optimisation_engine.py:1343` | **CALIBRATED** | EVIDENCE REQUIRED - tuned heuristic |
| `nominal_drawdown` | 500.0 | psi | `core/engine_surrogate/surrogate_engine.py:420` | **UNKNOWN** | EVIDENCE REQUIRED - makes J = q/500 tautological; no kh/skin derivation (HIGH-07 |
| `mu_inj_eff (dry CO2)` | 0.05 | cP | `core/engine_surrogate/surrogate_engine.py:430` | **UNKNOWN** | EVIDENCE REQUIRED - plausible magnitude for supercritical CO2 but hard-coded, no |
| `mu_inj_eff (WAG water)` | 0.50 | cP | `core/engine_surrogate/surrogate_engine.py:430` | **UNKNOWN** | EVIDENCE REQUIRED - hard-coded water viscosity; ignores T and salinity |
| `mobility_ratio_factor` | 0.001 | 1/(bbl/d) | `core/engine_surrogate/profile_generator_fast.py:1103` | **UNKNOWN** | EVIDENCE REQUIRED - mobility_ratio = rate x 0.001 is not a mobility ratio; compa |
| `high_mobility_threshold` | 2.0 | dimensionless | `core/engine_surrogate/profile_generator_fast.py:1104` | **UNKNOWN** | EVIDENCE REQUIRED - threshold has no physical derivation (MED) |
| `s_ref (fractional-flow reference saturation)` | 0.40 | dimensionless | `core/engine_surrogate/profile_generator_fast.py:967` | **UNKNOWN** | EVIDENCE REQUIRED - hard-coded; undocumented choice of SOF endpoint |
| `HybridSurrogate alpha (C7+ interpolation)` | 0.95 + 0.05*(c7_plus-0.3) | dimensionless | `core/engine_surrogate/analytical_models.py:458` | **UNKNOWN** | EVIDENCE REQUIRED - inert: reads c7_plus_fraction but engine writes c7_plus, so  |
| `x_co2 normalization factor` | 0.135 | fraction of OOIP | `core/engine_surrogate/surrogate_engine.py:502` | **UNKNOWN** | EVIDENCE REQUIRED - hard-coded denominator for solvent fraction scaling (MED) |
| `y_co2 clip range` | 0.05 - 0.95 | fraction | `core/engine_surrogate/surrogate_engine.py:356` | **UNKNOWN** | EVIDENCE REQUIRED - imposes >=5% CO2 in gas phase at cum_inj=0 (LOW-01) |
| `y_co2 decay constant` | 1000.0 | MSCF | `core/engine_surrogate/surrogate_engine.py:356` | **UNKNOWN** | EVIDENCE REQUIRED - hard-coded scale for CO2 gas-phase fraction |
| `Hall-Yarborough linear Z coefficients (0.06422 / -0.00332)` | 0.06422 / -0.00332 | 1/Ppr | `core/engine_surrogate/pvt_state.py:366` | **UNKNOWN** | EVIDENCE REQUIRED - labelled Hall-Yarborough but is a truncated linear expansion |
| `B_g constant (hydrocarbon)` | 0.1587 | 8.3145 x 0.3594 (as used) | `core/engine_surrogate/pvt_state.py:373` | **UNKNOWN** | EVIDENCE REQUIRED - wrong by 31.7x; correct 0.02827 x 5.6146 x 1000/1000 = 5.035 |
| `c_g mixing coefficients (0.35 / 0.85)` | 0.35 / 0.85 | dimensionless | `core/engine_surrogate/pvt_state.py:392` | **UNKNOWN** | EVIDENCE REQUIRED - 4-12x too high for dense CO2; discontinuous at 1500 psi (HIG |
| `CO2 viscosity polynomial (0.235 / 0.395 / -0.041)` | 0.235 / 0.395 / -0.041 | 1/rho_r^n | `core/engine_surrogate/pvt_state.py:222` | **UNKNOWN** | EVIDENCE REQUIRED - attributed to Fenghour/Vesovic but coefficients not from tho |
| `CO2 critical density (viscosity)` | 467.6 | kg/m3 | `core/engine_surrogate/pvt_state.py:221` | **LITERATURE** | OK - CO2 critical density 467.6 kg/m3 (IUPAC) but only used in the mis-fit polyn |
| `CO2 dilute-gas viscosity coefficient` | 1.00697e-6 | Pa.s/sqrt(K) | `core/engine_surrogate/pvt_state.py:218` | **LITERATURE** | OK form for dilute-gas limit; values in polynomial branch unverified |
| `Peng-Robinson a coefficient factor` | 0.45724 | - | `core/engine_surrogate/pvt_state.py:114` | **LITERATURE** | OK - standard PR kappa1/0.45724 formulation |
| `Peng-Robinson b coefficient factor` | 0.07780 | - | `core/engine_surrogate/pvt_state.py:115` | **LITERATURE** | OK - standard PR 0.0778 formulation |
| `PR alpha exponent m (CO2)` | 0.37464 + 1.54226w - 0.26992w^2 | - | `core/engine_surrogate/pvt_state.py:111` | **LITERATURE** | OK - standard PR alpha correlation |
| `Vasquez-Beggs R_s constants (API>30)` | 0.0178 / 1.1870 / 23.931 | - | `core/engine_surrogate/pvt_state.py:239-241` | **LITERATURE** | OK - Vasquez-Beggs (1980) coefficients for API>30 |

*... 77 additional parameters cataloged in `audit/parameter_provenance.csv`.*

---

## ⚠️ Software Disconnects & Dangerous Anti-Patterns

### 1. Undefined Names (Ruff F821) — Immediate Crash Hazards
| Module | Line | Error Message | Risk |
|---|---|---|---|
| `core/engine_surrogate/analytical_models.py` | 96 | `Undefined name `MiscibleRecoveryModel`` | Unhandled runtime NameError |
| `core/engine_surrogate/analytical_models.py` | 218 | `Undefined name `ImmiscibleRecoveryModel`` | Unhandled runtime NameError |
| `core/engine_surrogate/analytical_models.py` | 341 | `Undefined name `BuckleyLeverettModel`` | Unhandled runtime NameError |
| `core/geology/petrophysical_distribution.py` | 404 | `Undefined name `prev_field`` | Unhandled runtime NameError |
| `ui/workbench/components/pyvista_reservoir_canvas.py` | 974 | `Undefined name `has_active_fault`` | Unhandled runtime NameError |
| `ui/workbench/components/subsurface_data_viewer_widget.py` | 335 | `Undefined name `QToolTip`` | Unhandled runtime NameError |

### 2. Discarded Computations (Ruff F841) in Scientific Engines
Found **33** computed variables assigned and never read in core simulation paths:
- [ ] `core/engine_surrogate/analytical_models.py:265`: Local variable `s_wi` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:472`: Local variable `bt_years` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:719`: Local variable `n_points` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:772`: Local variable `total_years` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:808`: Local variable `total_cycle_time` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:860`: Local variable `n_points` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:980`: Local variable `cumulative_co2_injected` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:1165`: Local variable `n_points` is assigned to but never used
- [ ] `core/engine_surrogate/profile_generator_fast.py:1289`: Local variable `n_points` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:173`: Local variable `cumulative_oil` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:174`: Local variable `co2_stored` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:246`: Local variable `q_inj_rb` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:247`: Local variable `q_prod_rb` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:360`: Local variable `p_prod_sandface` is assigned to but never used
- [ ] `core/engine_surrogate/surrogate_engine.py:615`: Local variable `cum_co2_prod_profile` is assigned to but never used
- *... and 18 more in `audit/ruff_output.json`*

### 3. Silent Exception Fallbacks in Core Physics
Found **48** silent exception handlers. Top critical sites:
- [ ] `core/data_integration_engine.py:148`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/data_integration_engine.py:822`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:26`: catches `ImportError` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:239`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:349`: catches `(RuntimeError, AttributeError)` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:370`: catches `(RuntimeError, ValueError, ZeroDivisionError, ArithmeticError)` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:1577`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:2137`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:2944`: catches `(OSError, IOError, ValueError)` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:3629`: catches `(ImportError, AttributeError)` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:3700`: catches `Exception` (SILENT_PASS)
- [ ] `core/optimisation_engine.py:4299`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/optimisation_engine.py:586`: catches `Exception` (FALLBACK_VALUE)
- [ ] `core/plotting_manager.py:620`: catches `ImportError` (FALLBACK_VALUE)
- [ ] `core/plotting_manager.py:644`: catches `ImportError` (FALLBACK_VALUE)

---

## 🛑 Agent Action Protocol & Change-Safety Invariants

1. **NEVER modify equations without checking the flaw register**: All 53 items above are known. Modifying a clamp or formula without a corresponding verification test will cause regressions.
2. **Active vs Legacy Rule**: Only `core/engine_surrogate/` executes. Do not attempt to fix dormant modules (`core/unified_engine/`, `core/simulation/recovery_models.py`) under the impression that it will fix optimization runs.
3. **Two-Evaluations Trap (CRIT-02)**: Be aware that `recovery_factor` (RF₂) and `npv` (f(RF₁)) are derived from two distinct states. Any fix to the engine must reconcile these two into one single evaluation.
4. **Dimensional Consistency**: HCPVI must be `RB / RB` (dimensionless), not `MSCF / STB` (CRIT-01). B_g must be `5.035 * Z * T / P` in RB/MSCF (CRIT-04).
5. **Mandatory Save/Load Test**: Any edit to data models or UI widgets requires running:
   ```bash
   python -m pytest tests/test_project_save_load.py -v
   ```
