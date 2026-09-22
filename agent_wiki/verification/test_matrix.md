# Master Verification Test Matrix: Test Mapping to Equations & Invariants

## 1. Test Suite Architecture

The dedicated scientific verification test suite is located in `tests/scientific/`. It comprises **40 test items across 15 subdirectories**, organized strictly by scientific verification discipline.

---

## 2. Complete Traceability Mapping

| Discipline / Subdirectory | Test Function Name | Tested Equation / Mathematical Principle | Code Location Tested | Verdict |
|:---|:---|:---|:---|:---|
| **`mathematical/`** | `test_koval_recovery_integral_identity` | $\frac{d}{dV_p}[N_p(V_p)] \equiv 1 - F_s(S_s)$ (SymPy) | `surrogate_models.py:200-240` | **VERIFIED** |
| | `test_welge_tangent_identity` | $1 - f_w(S_{wf}) \equiv f_w'(S_{wf})(\bar{S}_w - S_{wf})$ | `analytical_models.py:340` | **VERIFIED** |
| | `test_arps_rate_cumulative_derivative_identity` | $\frac{d}{dt}[N_p(t)] \equiv q(t)$ (SymPy) | `analytical_models.py:410` | **VERIFIED** |
| | `test_v_dp_near_unity_singularity` | $H_k = 10^{V_{DP}/(1-V_{DP})^2}$ as $V_{DP} \to 0.999$ | `surrogate_models.py:125` | **VERIFIED** |
| | `test_cronquist_mmp_singularity_at_55_api` | $(55 - \text{API})^{0.279}$ at $\text{API} \ge 55^\circ$ | `evaluation/mmp.py:111` | **CONTRADICTED BY TEST (SCI-FLAW-13)** |
| | `test_mobility_ratio_unit_limit_singularity` | $M \to 1.0$ limit in Craig sweep | `surrogate_models.py:175` | **CONTRADICTED BY TEST (SCI-FLAW-16)** |
| **`physics/`** | `test_oil_compressibility_positivity` | $c_o = -\frac{1}{B_o}\frac{\partial B_o}{\partial P} > 0$ | `data_integration_engine.py:370` | **CONTRADICTED BY TEST (SCI-FLAW-02)** |
| | `test_liquid_viscosity_pressure_derivative` | $\frac{\partial\mu_o}{\partial P} > 0$ and $\frac{\partial\mu_g}{\partial P} > 0$ | `data_integration_engine.py:372` | **CONTRADICTED BY TEST (SCI-FLAW-03)** |
| | `test_co2_density_thermal_expansion` | $\frac{\partial\rho}{\partial T} < 0$ (Isobaric expansion) | `unified_engine/co2_properties.py:140` | **CONTRADICTED BY TEST (SCI-FLAW-04)** |
| | `test_cubic_eos_z_factor_bounds` | $Z_L \in [0.01, 0.40]$, $Z_V \in [0.70, 1.20]$ | `unified_engine/eos/__init__.py:180` | **VERIFIED** |
| | `test_phase_label_assignment` | $Z < 0.8 \implies \text{Liquid}$, $Z \ge 0.8 \implies \text{Vapor}$ | `unified_engine/eos/__init__.py:195` | **CONTRADICTED BY TEST (SCI-FLAW-08)** |
| | `test_peng_robinson_fugacity_equation_structure`| $\ln\phi_i$ includes $2\sqrt{2}B$ denominator term | `unified_engine/eos/__init__.py:206` | **CONTRADICTED BY TEST (SCI-FLAW-18)** |
| | `test_corey_relative_permeability_bounds` | $k_{ro}(S_{wi}) = k_{ro}^0$, $k_{ro}(1 - S_{or}) = 0$ | `unified_engine/relative_permeability.py` | **VERIFIED** |
| | `test_bg_discrepancy_between_modules` | $B_g \approx 0.5\text{ RB/MSCF}$ vs $5.0\text{ RB/MSCF}$ | `optimisation_engine.py:98` | **CONTRADICTED BY TEST (SCI-FLAW-12)** |
| **`conservation/`** | `test_cumulative_oil_recovery_mass_bound` | $N_p(\infty) \le \text{OOIP} \cdot \frac{1 - S_{wi} - S_{or}}{1 - S_{wi}}$ | `surrogate_models.py:510` | **VERIFIED** |
| | `test_pore_volume_vs_ooip_recovery_bound_discrepancy` | $RF \le 1 - S_{wi} - S_{or}$ (Pore vol vs OOIP bound) | `analytical_models.py:881` | **CONTRADICTED BY TEST (SCI-FLAW-11)** |
| | `test_closed_loop_carbon_balance_invariant` | $\sum M_{\text{inj}} = \sum M_{\text{purchased}} + \sum M_{\text{recycled}}$ | `surrogate_engine.py:460` | **VERIFIED** |
| | `test_material_balance_analyzer_closed_loop` | $M_{\text{stored}} = \sum q_{\text{inj}} - \sum q_{\text{prod}}$ | `material_balance.py:75` | **VERIFIED** |
| **`limiting_cases/`** | `test_zero_permeability_limit` | $k = 0 \implies q_o = 0, q_{\text{inj}} = 0$ | `surrogate_engine.py:340` | **VERIFIED** |
| | `test_zero_injection_limits` | $q_{\text{inj}} = 0 \implies \text{EOR Storage} = 0$ | `surrogate_engine.py:410` | **VERIFIED** |
| | `test_asymptotic_recovery_limit` | $\lim_{t \to \infty} q_o(t) = 0$, $N_p \le \text{EUR}$ | `profile_generator_fast.py:320` | **VERIFIED** |
| | `test_miscibility_weight_limiting_bounds` | $\omega(P \le P_{\text{min}}) = 0$, $\omega(P \ge \text{MMP}) = 1$ | `recovery_models.py:120` | **VERIFIED** |
| **`symmetry/`** | `test_pattern_symmetry_five_spot` | $\mathcal{R}_{\pi/2}$ 5-spot grid symmetry invariance | `surrogate_models.py:310` | **VERIFIED** |
| **`dimensional/`** | `test_darcy_inflow_dimensions` | $[q] = [L^3/T]$, $[J] = [L^4 T / M]$ (Pint) | `surrogate_engine.py:330` | **VERIFIED** |
| | `test_tank_material_balance_pressure_increment_dimensions` | $[\Delta P] = [M / L / T^2]$ (Pint) | `surrogate_engine.py:405` | **VERIFIED** |
| | `test_co2_mass_conversion_factor` | $1\text{ MSCF CO}_2 = 0.05299\text{ tonne}$ (Pint) | `surrogate_models.py:26` | **VERIFIED** |
| **`boundary_conditions/`**| `test_epa_class_vi_pressure_ceiling_enforcement` | $P_{\text{sandface}} \le 0.90 P_{\text{frac}}$ | `surrogate_engine.py:343` | **VERIFIED** |
| | `test_producer_rate_drawdown_limit` | $q_o(P_{res} = P_{wf}) = 0$ | `surrogate_engine.py:380` | **VERIFIED** |
| **`initial_conditions/`** | `test_initial_pressure_and_cumulative_at_time_zero` | $P(0) = P_{\text{init}}$, $N_p(0) = 0$ | `surrogate_engine.py:184` | **VERIFIED** |
| **`numerical/`** | `test_pressure_oscillation_under_dynamic_injection` | $\Delta^2 P$ non-oscillatory under smooth injection | `surrogate_engine.py:410` | **VERIFIED** |
| **`convergence/`** | `test_temporal_refinement_convergence` | Refinement sensitivity: monthly vs weekly $< 5\%$ | `surrogate_engine.py:180` | **VERIFIED** |
| **`solver/`** | `test_pressure_material_balance_discrete_residual` | $\|(V_p c_t + J_{\text{eff}}\Delta t)\Delta P - q_{\text{net}}\Delta t\| \le 10^{-12}$ | `surrogate_engine.py:405` | **VERIFIED** |
| **`co2/`** | `test_koval_fractional_flow_mobility_inversion`| $\partial F_{\text{CO2}} / \partial M > 0$ | `profile_generator_fast.py:895`| **CONTRADICTED BY TEST (SCI-FLAW-01)** |
| | `test_inverted_critical_gas_trapping` | $\text{Trapping} = 1.0 - S_{gc}$ | `surrogate_models.py:238` | **CONTRADICTED BY TEST (SCI-FLAW-17)** |
| | `test_yuan_impurity_mmp_trend` | $\text{MMP}(\text{pure CO}_2) < \text{MMP}(\text{CO}_2 + \text{CH}_4)$ | `evaluation/mmp.py:310` | **VERIFIED** |
| | `test_standing_bo_as_api_estimator` | $B_o$ correlated to API gravity | `evaluation/mmp.py:240` | **VERIFIED** |
| **`reference_solutions/`**| `test_independent_welge_reference_solution` | Welge tangent benchmark error $< 1.0\%$ | `reference_solutions/` | **VERIFIED** |
| | `test_analytical_vs_trapezoidal_arps_eur` | Analytical Arps EUR benchmark error $< 0.1\%$ | `reference_solutions/` | **VERIFIED** |
| **`manufactured_solutions/`**| `test_mms_pressure_diffusion_convergence` | MMS observed order of accuracy $p \ge 1.95$ | `manufactured_solutions/` | **VERIFIED** |
| **`regression/`** | `test_hypothesis_recovery_factor_physical_invariants` | $RF \in [0, 1]$, $N_p \le \text{OOIP}$ across 1,000 cases | `regression/` | **VERIFIED** |
