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
   $$MMP = 15.988 \cdot T^{0.7442} \cdot \left( \frac{C_1}{C_2 - C_5} \right)^{0.2111} \cdot (55 - \gamma_{API})^{0.279}$$
   *Singularity warning*: Requires $\gamma_{API} < 55^\circ\text{API}$.
2. **Lee (1979)**:
   Empirical polynomial correlation based on reservoir temperature and molecular weight of pentanes-plus ($M_{C5+}$).
3. **Glaso (1985)**:
   Correlation calibrated for North Sea volatile crudes.

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
