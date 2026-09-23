# Evaluation & Analysis Subsystems (`evaluation/` & `analysis/`)

## 1. Evaluation Subsystem (`evaluation/`)

The `evaluation/` directory contains screening models, analytical correlations, and techno-economic evaluators.

### Module Inventory
| Module | LOC | Primary Functions / Classes | Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `evaluation/mmp.py` | 420 | `calculate_mmp()`, `MMPParameters` | Multi-correlation dispatcher for Minimum Miscibility Pressure (Cronquist, Lee, Glaso, Alston, Yuan). | **HIGH** |
| `evaluation/economic_analyzer.py` | 380 | `EconomicAnalyzer` | Cash flow modeling, CAPEX/OPEX depreciation, internal rate of return (IRR), payout period. | **MEDIUM** |
| `evaluation/environmental_impact.py` | 260 | `EnvironmentalImpactAnalyzer` | Life-cycle emissions, net carbon balance, fugitive emissions risk. | **LOW** |

### MMP Correlations in `mmp.py`
1. **Cronquist (1978)** (Default for light/medium crudes):
   $$P_{MMP} = 15.988 \cdot T_F^Y \quad [\text{psia}]$$
   where $Y = 0.744206 + 0.0011038 \cdot MW_{C5+} + 0.0015279 \cdot Vol$.
   $MW_{C5+} = 4247.98641 \cdot \text{API}^{-0.87022}$ (DOE / CO₂ Prophet standard formulation).
   *Robustness*: Fully verified non-singular and monotonic across light crudes and condensates ($\text{API} \ge 55^\circ$, SCI-FLAW-13 resolved).
2. **Yellig & Metcalfe (1980)**:
   Published pure-CO₂ correlation with low-temperature bubble-point/critical pressure capping ($1070\text{ psia}$ for $T < 95^\circ\text{F}$).
3. **Alston et al. (1985)**:
   CO₂ and contaminated injection gas streams using Kay's pseudo-critical temperature rules $(T_{pc,\text{CO2}}/T_{pc,\text{gas}})^A$.
4. **Yuan et al. (2005)**:
   Compositional multi-component minimum miscibility correlation with guaranteed impurity penalty factor $c \ge 1.0$.
5. **Lee (1979)** & **Glaso (1985)**:
   Empirical correlations based on reservoir temperature, volatile fractions, and heavy component molecular weights.

---

## 2. Analysis Subsystem (`analysis/`)

The `analysis/` package provides diagnostic verification, material balance audits, decline curve fitting, and global sensitivity analysis.

### Module Inventory
| Module | LOC | Primary Classes | Function | Modification Risk |
| :--- | :---: | :--- | :--- | :---: |
| `analysis/material_balance.py` | 450 | `MaterialBalanceAuditor` | Validates conservation of hydrocarbon mass, water mass, and injected carbon. | **HIGH** |
| `analysis/decline_curve_analysis.py` | 380 | `DeclineCurveAnalyzer` | Fits Arps exponential, hyperbolic, and harmonic equations to production tails. | **MEDIUM** |
| `analysis/sensitivity_analyzer.py` | 512 | `SensitivityAnalyzer` | Evaluates parameter sensitivity using Morris screening and Sobol variance decomposition. | **MEDIUM** |
| `analysis/uq_engine.py` | 460 | `UQEngine` | Uncertainty quantification via Monte Carlo sampling and Polynomial Chaos Expansion (PCE). | **MEDIUM** |
| `analysis/pressure_transient_analysis.py` | 340 | `PressureTransientAnalyzer` | Evaluates Horner plots, derivative diagnostics, skin factor, and flow regimes. | **LOW** |

### Material Balance Invariants (`analysis/material_balance.py`)
- **Gross CO₂ Conservation Invariant**:
  $$\text{Gross Injected} = \text{Purchased Fresh} + \text{Recycled Produced} = \text{Net Stored} + \text{Produced} + \text{Leakage}$$
- **Liquid Conservation**:
  $$\sum q_o(t) \cdot \Delta t = OOIP \times RF$$
