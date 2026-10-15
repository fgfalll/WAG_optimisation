# Scientific Parameter Provenance & Anti-Calibration Audit

> [!NOTE]
> This document catalogs **only active, open items**. For resolved flaws, historical post-mortems, and verification status, consult the [**Resolved Issues & Defect Resolution Archive**](resolved_issues.md).

## 1. Provenance Classification Taxonomy

Every numerical parameter in a scientific simulator must have clear traceability. In this audit, all parameters in `core/data_models.py`, `core/engine_surrogate/`, and physics modules have been classified into one of six distinct categories:

1. **Fundamental Physical Constant**: Invariant constant of nature or exact thermodynamic property (e.g., universal gas constant $R$, molecular weight of CO₂ $44.01\text{ g/mol}$).
2. **Literature Reference**: Widely accepted published correlations from peer-reviewed literature with citations (e.g., Todd & Longstaff 1972, Koval 1963, Standing 1952).
3. **Empirical Correlation**: Laboratory-calibrated regression curves valid only within a documented operating envelope (e.g., Cronquist MMP correlation).
4. **Active Calibration Parameter**: Tuning parameter adjusted to force simulation outputs to match a historical or commercial target without theoretical derivation.
5. **Arbitrary Default**: Hardcoded constant chosen without documented source or physical justification.
6. **Contradicted by Test**: Erroneous or inconsistent parameter that contradicts physical testing or unit conversions.

---

## 2. Core Parameter Provenance Registry

| Parameter Name | Value | Units | Provenance Classification | Physical Meaning & Equation | Code Location | Scientific Assessment |
|:---|:---|:---|:---|:---|:---|:---|
| `CO2_DENSITY_TONNE_PER_MSCF` | $0.05299$ | tonne/MSCF | **Fundamental Physical Constant** | Mass of $1,000\text{ SCF}$ of pure CO₂: $\rho = \frac{M_{\text{CO2}}}{V_{\text{std}}} = \frac{44.01\text{ lb/lbmol}}{379.3\text{ SCF/lbmol}}$ | `surrogate_models.py:26` | **VERIFIED** (Exact match) |
| `DAYS_PER_YEAR` | $365.25$ | days/yr | **Fundamental Physical Constant** | Julian astronomical year standard | `PhysicalConstants.DAYS_PER_YEAR` | **VERIFIED** |
| `TODD_LONGSTAFF_OMEGA` | $0.60$ | dimensionless | **Literature Reference** | Todd & Longstaff (1972) mixing parameter $\omega \in [0.5, 0.7]$ for miscible displacement | `analytical_models.py:38` | **VERIFIED** |
| `EPA_CLASS_VI_SAFETY_FACTOR` | $0.90$ | fraction | **Literature / Regulatory** | Maximum allowable sandface injection pressure: $P_{\text{sandface}} \le 0.90 P_{\text{frac}}$ | `surrogate_engine.py:343` | **VERIFIED** (Regulatory) |
| `HETEROGENEITY_CALIBRATION_C_TRANS` | $0.80$ | dimensionless | **Active Calibration Parameter** | Heuristic tuning multiplier applied to Dykstra-Parsons coefficient: $V_{DP}^* = 0.80 V_{DP}$ | `surrogate_engine.py:195` | **CALIBRATED** (Artificially dampens heterogeneity) |
| `WAG_MOBILITY_REDUCTION_FACTOR` | $1.50$ | dimensionless | **Active Calibration Parameter** | Arbitrary factor multiplying water fraction to enhance sweep | `surrogate_engine.py:210` | **CALIBRATED** (Empirical multiplier) |
| `NOMINAL_DRAWDOWN` | $500.0$ | psi | **Arbitrary Default** | Used as fallback divisor when well bottomhole pressure is missing: $J = q / 500$ | `surrogate_engine.py:348` | **ARBITRARY** (Masks missing well data) |
| `PRESSURE_INCREMENT_CLAMP` | $450.0$ | psi/step | **Arbitrary Default / Numerical Stabilizer** | Hard limiter clamping $\Delta P \in [-450, +450]\text{ psi}$ per monthly time-step | `surrogate_engine.py:379` | **NUMERICAL STABILIZER** (Suppresses physical blowouts) |
| `RECOVERY_FACTOR_CEILING` | $0.80$ | fraction | **Arbitrary Default / Anti-Fitting** | Hardcoded upper ceiling on calculated recovery factor | `analytical_models.py:205` | **SCIENTIFIC FLAW (SCI-FLAW-11)** |
| `B_GAS_RB_PER_MSCF` | $5.0$ | RB/MSCF | **Contradicted by Test** | Used in optimizer voidage calculations; contradicts physical supercritical $B_g \approx 0.5\text{ RB/MSCF}$ | `optimisation_engine.py:98` | **CONTRADICTED (SCI-FLAW-12)** ($10\times$ error) |
