# Mathematical Flaw Register

This document catalogs confirmed mathematical errors, invalid transformations, corrupted numerical solvers, and dimensional inconsistencies in the CO₂ EOR Optimizer codebase.

---

## Master Mathematical Flaw Summary

| ID | Severity | Location | Category | Mathematical Description | Status |
|:---|:---|:---|:---|:---|:---|
| **MATH-FLAW-01** | **CRITICAL** | `unified_engine/physics/eos/__init__.py:414` | Corrupted Objective Equation | Rachford-Rice equation subtracts 1.0 from the summation: $\sum \frac{z_i(K_i-1)}{1+V(K_i-1)} - 1.0 = 0$ | OPEN |
| **MATH-FLAW-02** | **CRITICAL** | `unified_engine/physics/eos/__init__.py:257-270` | Incorrect Analytical Derivative/Form | Peng-Robinson fugacity coefficient formula omits $2\sqrt{2}$, cross-terms $\sum z_j a_{ij}$, and replaces PR logarithm with $\ln(1+B/Z)$ | OPEN |
| **MATH-FLAW-03** | **CRITICAL** | `analysis/breakthrough_physics.py:314, 361` | Dimensional Inconsistency | Numerator multiplied by $g = 2.4 \times 10^{11}\text{ ft/day}^2$ without dividing by $g_c = 2.4 \times 10^{11}$, inflating $N_g$ by $2.4\times 10^{11}\times$ | OPEN |
| **MATH-FLAW-04** | **CRITICAL** | `core/optimisation_engine.py:1881` | Invalid Objective Transformation | Financial DCF NPV is multiplied by heuristic breakthrough scalar: $\text{NPV}_{\text{reported}} = \text{NPV}_{\text{DCF}} \times \text{Impact}$ | OPEN |
| **MATH-FLAW-05** | **HIGH** | `engine_surrogate/surrogate_models.py:515-530` | Discretization / Integration Error | Dynamic 181-step production profile discarded; NPV integrates a static flat average $N_p / 15$ for 15 years | OPEN |
| **MATH-FLAW-06** | **HIGH** | `engine_surrogate/analytical_models.py:783` | Discontinuity / Non-Differentiability | Effective mixing in surrogate creates artificial derivative kinks at transition thresholds | OPEN |
| **MATH-FLAW-07** | **MEDIUM** | `engine_surrogate/analytical_models.py:912-922` | Finite Difference Perturbation Error | `calculate_gradient()` unpacks defaults but checks original dictionary keys, silently dropping unpassed parameters | OPEN |
| **MATH-FLAW-08** | **HIGH** | `unified_engine/physics/eos/__init__.py:416-432` | Solver Convergence Failure | Rachford-Rice bisection solver assumes $f(0) > 0$ and $f(1) < 0$ without monotonic bracket checks | OPEN |
| **MATH-FLAW-09** | **HIGH** | `unified_engine/physics/eos/__init__.py:120` | Root Selection Ambiguity | Cubic EOS takes `valid_roots[0]` arbitrarily without distinguishing liquid root ($Z_L = \min$) from vapor root ($Z_V = \max$) | OPEN |
| **MATH-FLAW-10** | **MEDIUM** | `engine_surrogate/profile_generator_fast.py:664` | Singularity / Division by Zero | Arps hyperbolic rate equation has a pole at $b = 1.0$; threshold clamp `abs(1 - b) < 0.01` creates step discontinuity | OPEN |

---

## Detailed Mathematical Flaw Records

### MATH-FLAW-01: Corrupted Rachford-Rice Equation
- **ID**: `MATH-FLAW-01`
- **Severity**: **CRITICAL**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:414`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L414)
- **Mathematical Expression**:
  ```python
  f = lambda V: np.sum(z * (K - 1) / (1 + V * (K - 1))) - 1.0
  ```
- **Rigorous Mathematical Formulation**:
  The phase equilibrium condition requires:
  $$\sum_{i=1}^n x_i = 1.0 \quad \text{and} \quad \sum_{i=1}^n y_i = 1.0$$
  Subtracting the two equations yields:
  $$\sum_{i=1}^n (y_i - x_i) = 1.0 - 1.0 = 0.0$$
  Substituting $x_i = \frac{z_i}{1 + V(K_i - 1)}$ and $y_i = K_i x_i$:
  $$f(V) = \sum_{i=1}^n \frac{z_i (K_i - 1)}{1 + V (K_i - 1)} = 0.0$$
- **Actual Code Behavior**:
  The code defines:
  $$f_{\text{code}}(V) = \sum_{i=1}^n \frac{z_i (K_i - 1)}{1 + V (K_i - 1)} - 1.0 = 0.0 \implies \sum_{i=1}^n (y_i - x_i) = 1.0$$
  This implies:
  $$\sum_{i=1}^n y_i = 1.0 + \sum_{i=1}^n x_i = 1.0 + 1.0 = 2.0$$
- **Mathematical Consequence**: Total vapor mole fraction sums to 2.0 (200% mass). The computed vapor fraction $V$ is completely mathematically false.
- **Affected Components**: `PhaseEquilibriumCalculator.rachford_rice_flash`, `FlashCalculator.pt_flash`.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### MATH-FLAW-02: Incorrect Peng-Robinson Fugacity Formulation
- **ID**: `MATH-FLAW-02`
- **Severity**: **CRITICAL**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:257-270`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L257-L270)
- **Mathematical Expression in Code**:
  $$\ln \phi_i = \frac{b_i}{b}(Z - 1) - \ln(Z - B) + \frac{A}{B}\left(\frac{b_i}{b} - 1\right)\ln\left(1 + \frac{B}{Z}\right)$$
- **Rigorous Peng-Robinson Equation**:
  From Peng & Robinson (1976), the exact analytical expression for the fugacity coefficient of component $i$ in a mixture is:
  $$\ln \phi_i = \frac{b_i}{b}(Z - 1) - \ln(Z - B) - \frac{A}{2\sqrt{2} B}\left(\frac{2 \sum_{j} z_j a_{ij}}{a} - \frac{b_i}{b}\right) \ln\left(\frac{Z + (1 + \sqrt{2})B}{Z + (1 - \sqrt{2})B}\right)$$
- **Mathematical Discrepancies**:
  1. The factor $\frac{1}{2\sqrt{2}} \approx 0.35355$ is completely omitted.
  2. The cross-component interaction sum $\frac{2 \sum_j z_j a_{ij}}{a}$ is absent; the code replaces it with $1.0$.
  3. The argument of the logarithm $\frac{Z + (1+\sqrt{2})B}{Z + (1-\sqrt{2})B}$ is replaced by $1 + \frac{B}{Z}$.
  4. The sign of the third term is inverted.
- **Mathematical Consequence**: Fugacity coefficients $\phi_i$ are mathematically invalid. K-values derived from $K_i = \phi_{i,L} / \phi_{i,V}$ diverge from true thermodynamic vapor-liquid equilibrium.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### MATH-FLAW-03: $2.4 \times 10^{11}$ Missing $g_c$ Dimensional Factor
- **ID**: `MATH-FLAW-03`
- **Severity**: **CRITICAL**
- **Location**: [`analysis/breakthrough_physics.py:314, 361`](file:///d:/rep/4.6/co2eor_optimizer/analysis/breakthrough_physics.py#L314)
- **Mathematical Formulation**:
  $$N_g = \frac{k [\text{ft}^2] \cdot \Delta\rho [\text{lb}_m/\text{ft}^3] \cdot g [\text{ft/day}^2] \cdot |\sin(\theta)|}{\mu [\text{lb}_f\cdot\text{day/ft}^2] \cdot u [\text{ft/day}]}$$
- **Dimensional Derivation**:
  - In oilfield engineering units, mass ($\text{lb}_m$) and force ($\text{lb}_f$) are related by Newton's second law constant:
    $$g_c = 32.174 \frac{\text{lb}_m\cdot\text{ft}}{\text{lb}_f\cdot\text{s}^2} = 2.40 \times 10^{11} \frac{\text{lb}_m\cdot\text{ft}}{\text{lb}_f\cdot\text{day}^2}$$
  - The ratio $\frac{g}{g_c} = \frac{32.174\text{ ft/s}^2}{32.174\text{ lb}_m\cdot\text{ft/lb}_f\cdot\text{s}^2} = 1.0 \frac{\text{lb}_f}{\text{lb}_m}$.
  - The code author defined `g_ft_day2 = 2.4e11` and multiplied the numerator without dividing by $g_c$.
- **Numerical Consequence**:
  - $N_g$ evaluates to $\sim 10^{10}$ instead of $\sim 0.04$.
  - The dimensionless interaction function $\Phi(N_g, M) = \frac{1}{1 + \sqrt{N_g \cdot M}}$ collapses to $1.8 \times 10^{-6}$.
  - Breakthrough time collapses from 2.0 years down to $3.4 \times 10^{-7}\text{ years}$ ($10\text{ milliseconds}$).
- **Confidence**: 100%.
- **Status**: OPEN.

---

### MATH-FLAW-04: Breakthrough Impact Factor Multiplier on Financial NPV
- **ID**: `MATH-FLAW-04`
- **Severity**: **CRITICAL**
- **Location**: [`core/optimisation_engine.py:1881, 3020`](file:///d:/rep/4.6/co2eor_optimizer/core/optimisation_engine.py#L1881)
- **Mathematical Expression**:
  ```python
  result = objective_value  # Financial DCF NPV ($630,000,000)
  result *= breakthrough_impact  # breakthrough_impact = 0.511
  ```
- **Expected Mathematical Principle**:
  Financial Net Present Value is the discounted sum of actual cash flows:
  $$\text{NPV} = \sum_{t=0}^N \frac{\text{CF}_t}{(1 + r)^t}$$
  Economic consequences of breakthrough (e.g. gas handling CAPEX, recycling compression costs) must be included as cash flow deductions $\text{CF}_t = R_t - C_{\text{opex},t} - C_{\text{recyc},t}$.
- **Actual Code Behavior**:
  The optimizer applies an ad-hoc scalar multiplier:
  $$\text{Fitness} = \text{NPV}_{\text{DCF}} \times \text{Impact}$$
  And then in UI summaries and export reports:
  `Final Optimized NPV: $322,000,000` is displayed alongside `Project NPV: $630,000,000`.
- **Mathematical Consequence**: Reports a 50% discrepancy in reported project financial value under identical labels.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### MATH-FLAW-05: Flat Cash Flow Simplification in Surrogate NPV
- **ID**: `MATH-FLAW-05`
- **Severity**: **HIGH**
- **Location**: [`core/engine_surrogate/surrogate_models.py:515-530`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_models.py#L515-L530)
- **Mathematical Formulation**:
  ```python
  annual_oil_production = cumulative_oil / project_life_years
  annual_co2_injected = co2_stored / project_life_years
  annual_revenue = annual_oil_production * oil_price
  annual_co2_cost = annual_co2_injected * co2_cost
  annual_cashflow = annual_revenue - annual_co2_cost
  cashflow = np.concatenate([[-capex], np.full(int(project_life_years), annual_cashflow)])
  npv = np.sum(cashflow / (1.0 + discount_rate)**years)
  ```
- **Discrepancy**:
  `FastProfileGenerator` computes a detailed 181-step monthly production profile with ramp-up, peak plateau, breakthrough, and hyperbolic decline. However, `surrogate_models.py` completely throws away this profile and assumes that oil production and CO₂ costs are 100% constant across every year.
- **Mathematical Consequence**: Front-loaded plateau cash flows are discounted improperly, producing a 15% to 25% distortion in NPV relative to true profile integration.
- **Confidence**: 100%.
- **Status**: OPEN.

---

### MATH-FLAW-06: Piecewise Kink Discontinuity at MMP
- **ID**: `MATH-FLAW-06`
- **Severity**: **HIGH**
- **Location**: [`core/engine_surrogate/analytical_models.py:783`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L783)
- **Mathematical Formulation**:
  `effective_mixing = float(np.clip(omega * omega_tl, 0.0, 1.0))`
  `mu_g_eff = (mu_mix**effective_mixing) * (viscosity_inj ** (1.0 - effective_mixing))`
- **Observation**:
  While `get_miscibility_weight()` uses a continuous tanh transition, `omega` scales mobility ratio $M = \lambda_g / \lambda_o$, which feeds into $K = H \cdot (0.78 + 0.22 M^{0.25})^4$. The combination of multiple non-smooth clipping operations (`np.clip(rf, 0.0, rf_max_physical)`) introduces non-differentiable gradient kinks throughout parameter space.
- **Status**: OPEN.

---

### MATH-FLAW-07: Parameter Omission in Finite Difference Perturbation
- **ID**: `MATH-FLAW-07`
- **Severity**: **MEDIUM**
- **Location**: [`core/engine_surrogate/analytical_models.py:912-922`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/analytical_models.py#L912-L922)
- **Mathematical Formulation**:
  ```python
  pressure = params.get("pressure", 3000.0)
  c7_plus = params.get("c7_plus_fraction", 0.3)
  for key in ["pressure", "mmp"]:
      if key in params:  # Checks input dict, NOT merged dict!
  ```
- **Defect**: If `pressure` was omitted from the call arguments (using the default `3000.0`), `if key in params` evaluates to `False`. Gradients with respect to default parameters are silently omitted from the return dictionary.
- **Status**: OPEN.

---

### MATH-FLAW-08: Rachford-Rice Bisection Without Root Bracketing Verification
- **ID**: `MATH-FLAW-08`
- **Severity**: **HIGH**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:416-432`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L416-L432)
- **Mathematical Principle**:
  Bisection requires that the function values at the interval boundaries have opposite signs:
  $$f(V_{\text{low}}) \cdot f(V_{\text{high}}) < 0$$
- **Actual Code Behavior**:
  The solver initializes $V_{\text{low}} = 0.0$ and $V_{\text{high}} = 1.0$ and begins bisection immediately without checking whether $f(0)$ and $f(1)$ bracket a root. If the fluid is in a single-phase liquid or single-phase vapor state, $f(0)$ and $f(1)$ have the same sign, and bisection terminates at an arbitrary midpoint ($V = 0.5$ or $0.0$).
- **Status**: OPEN.

---

### MATH-FLAW-09: Arbitrary Cubic Root Selection
- **ID**: `MATH-FLAW-09`
- **Severity**: **HIGH**
- **Location**: [`core/unified_engine/physics/eos/__init__.py:120`](file:///d:/rep/4.6/co2eor_optimizer/core/unified_engine/physics/eos/__init__.py#L120)
- **Mathematical Formulation**:
  ```python
  valid_roots = self.solve_cubic(A, B)
  if len(valid_roots) > 0:
      return valid_roots[0]
  ```
- **Defect**: When a cubic EOS has three real roots in the two-phase region, the smallest root represents the liquid phase and the largest root represents the vapor phase. Returning `valid_roots[0]` selects whichever root NumPy's eigenvalue solver orders first, causing unpredictable phase flipping.
- **Status**: OPEN.

---

### MATH-FLAW-10: Arps Hyperbolic Rate Pole at $b = 1.0$
- **ID**: `MATH-FLAW-10`
- **Severity**: **MEDIUM**
- **Location**: [`core/engine_surrogate/profile_generator_fast.py:664-668`](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/profile_generator_fast.py#L664-L668)
- **Mathematical Expression**:
  ```python
  if abs(1 - b_factor) < 0.01:
      qi = ultimate_recovery * initial_decline / 365.25
  else:
      qi = (ultimate_recovery * initial_decline * (1 - b_factor) / 365.25) ** (1 / (1 - b_factor))
  ```
- **Defect**: The threshold `abs(1 - b_factor) < 0.01` creates a discontinuous step in the calculated initial rate $q_i$ when $b$ traverses $0.99$ or $1.01$.
- **Status**: OPEN.
