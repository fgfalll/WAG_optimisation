# Surrogate Engine Inner Workings

## 1. Engine Overview

The Surrogate Engine provides a physics-based simulation environment designed for CO2 Enhanced Oil Recovery (EOR) optimization. It evaluates scenarios using analytical correlations combined with material balance equations to approximate the dynamic fluid flow, production profiles, and pressure behavior of a reservoir.

### Purpose

- **Screening Optimization**: Evaluate thousands of scenarios rapidly for population-based metaheuristics (e.g., NSGA-II).
- **Sensitivity Analysis**: Conduct parameter sweeps for uncertainty quantification.
- **Baseline Generation**: Provide rapid initial estimates prior to detailed numerical simulation.

### Architecture

```
SurrogateEngine
├── SurrogateEngineWrapper (SimulationEngineInterface compatible)
├── SurrogateModel (AnalyticalSurrogate or ResponseSurfaceSurrogate)
│   └── AnalyticalRecoveryModel (Miscible, Immiscible, Hybrid, PhD-Hybrid)
└── FastProfileGenerator (plateau_decline, arps, logistic)
```

---

## 2. Configuration Options

### 2.1 model_type
| Option | Description |
|--------|-------------|
| `"analytical"` | Closed-form literature-based correlations (default) |
| `"response_surface"` | Polynomial/RBF interpolation |

### 2.2 recovery_model_type
| Option | Class | Description |
|--------|-------|-------------|
| `"miscible"` | `MiscibleSurrogate` | Koval (1963) correlation for miscible displacement |
| `"immiscible"` | `ImmiscibleSurrogate` | Buckley-Leverett (1942) with Corey (1954) relative permeability |
| `"hybrid"` | `HybridSurrogate` | Sigmoidal weighting between miscible/immiscible |
| `"phd_hybrid"` | `PhDHybridSurrogate` | Advanced hybrid with continuous differentiability designed for optimization algorithms |

---

## 3. Core Physics and Simulation Methodology

The engine calculates ultimate recovery and pressure profiles by solving a coupled system of analytical displacement models and a 0D material balance ordinary differential equation (ODE).

### 3.1 The PhD-Hybrid Displacement Model

The `PhDHybridSurrogate` model computes the ultimate recovery factor (RF) using a mathematically continuous formulation, making it suitable for gradient-based or evolutionary optimization by smoothing the "miscibility cliff".

**1. Thermodynamic Miscibility Function (ω):**
The transition between immiscible and miscible displacement is governed by a continuous exponential function.
```
if P >= MMP:
    ω = 1 - exp(-(P - MMP) / MMP)
else:
    ω = 0
```
This serves to smoothly scale the effective viscosities and relative permeabilities between fully segregated and fully mixed states (Todd & Longstaff, 1972).

**2. Transverse Mixing and Heterogeneity (Koval, 1963):**
Strict 1D fractional flow assumes zero transverse mixing, over-predicting fingering in 3D. We introduce a transverse mixing calibration to adjust the Dykstra-Parsons coefficient ($V_{DP}$):
```
Effective V_DP = V_DP × Transverse_Mixing_Calibration
α = 1 / (1 - Effective_V_DP)
K_koval = ((M + 1) / 2)^α
```

**3. Capillary Desaturation:**
The residual oil saturation ($S_{orm}$) is dynamically scaled based on the capillary number ($N_c$). As pressure approaches MMP, the Interfacial Tension (IFT) drops toward ~0.01 dyne/cm.
```
N_c = (μ_inj × u) / σ_dynes_cm × 3.5e-6
if N_c > N_c_ref:
    S_or = S_or_immiscible × (N_c / N_c_ref)^(-m)
```

**4. Volumetric Sweep (Fractional Flow Integration):**
The displacement efficiency is bounded by integrating the Koval fractional flow post-breakthrough:
```
t_D = HCPVI
if t_D < 1/K_koval:
    E_sweep = t_D
elif t_D >= K_koval:
    E_sweep = 1.0
elif K_koval > 1.0:
    E_sweep = (2 × sqrt(K_koval × t_D) - 1 - t_D) / (K_koval - 1)
```

**5. Gravity Override:**
Gravity effects scale the vertical efficiency based on the density difference, permeability, and viscosity.
```
N_g = (k × Δρ × g × sin(θ)) / (μ × u)
E_v = 1 / (1 + β_gravity × N_g × Override_Severity)
```

**Final Synthesis:**
```
RF = E_sweep × E_v × E_d
```

### 3.2 Pressure ODE System (Material Balance)

The engine solves a stiff ODE system for pressure evolution over time to evaluate constraints and operational limits.

**State Vector:**
```
State = [pressure, cum_oil_rb, cum_inj_rb, cum_gas_prod_rb]
```

**Material Balance ODE:**
```
q_net = q_inj_step - (q_oil_rb + q_gas_rb + water_rate_pot)
ct_dynamic = c_o × S_o + c_g × S_g + c_w × S_w + c_f
dp/dt = q_net / (PV × ct_dynamic) + Stabilization_Gain
```

**Solver:**
The system is integrated using an implicit Backward Differentiation Formula (BDF) method via `scipy.integrate.solve_ivp` to handle the stiffness of the pressure equation. It falls back to Radau or explicit Euler methods if convergence fails.

### 3.3 Petrophysical Coupling

The surrogate engine couples petrophysical properties directly to the flow and pressure equations:

**1. Explicit Pore Volume:**
Pore volume is calculated directly from reservoir geometry to accurately reflect hydrocarbon pore volume injection (HCPVI) rates:
```
PV (bbl) = 7758 × Area (acres) × Thickness (ft) × Porosity
```

**2. Darcy's Law Injectivity:**
Injectivity incorporates Darcy's law to capture the effect of permeability on pressure maintenance:
```
J_inj = (k × A) / (μ × L) = (perm_mD × 1.062e-14 × width_ft) / (μ_inj × length_ft)
```

**3. Permeability Effect on Mobility Ratio:**
Porosity and permeability jointly affect the mobility ratio via dynamic Corey exponent adjustments and endpoint scaling (Leverett J-function scaling):
```
n_o_adj = n_o × (1 + 0.03 × ln(k/100))
k_ro_end = k_ro_0 × (φ/0.15)^0.15
```

### 3.4 Data-Driven Response Surfaces

As an alternative to pure analytical correlations, the Surrogate Engine supports `ResponseSurfaceSurrogate` models that can be trained on external simulation data to provide rapid ($\approx 0.5$ ms) interpolations. 

**1. Polynomial Ridge Regression:**
Fits a global polynomial (typically quadratic or cubic) across scaled features, utilizing $L_2$ Tikhonov regularization (Ridge) to prevent overfitting in highly correlated petrophysical domains.

**2. Radial Basis Function (RBF) Interpolation:**
Constructs localized multi-dimensional interpolators for non-smooth, scattered dataset topologies using functions such as Multiquadric:
```
φ(r) = sqrt((r/ε)² + 1)
```

**3. Feature Transformation:**
Inputs are standardized before training/prediction via `StandardScaler` (zero mean, unit variance) or `MinMaxScaler`, with optional logarithmic transformations (`ln(x + 1e-10)`) applied to highly skewed parameters like permeability.

---

## 4. Production Profile Generation

Instead of costly numerical gridding, temporal resolution is achieved via a `FastProfileGenerator`. It uses parameterized shapes scaled to match the ultimate recovery from the analytical models.

### 4.1 Oil Production (Plateau + Decline)
- **Pre-breakthrough:** Stable displacement yielding a constant rate plateau.
- **Post-breakthrough Acceleration:** Decline rates accelerate due to CO2 channeling and mobility contrast. The acceleration factor is derived from fractional flow theory:
  ```
  Post_BT_Acceleration = 1.0 + (Mobility_Ratio - 1) / (Mobility_Ratio + 1) × 0.5
  ```
- **Base Decline Rate:** Derived from reservoir rock/fluid physics (Corey relative permeabilities, Darcy flow), typically ranging from 5% to 35% per year depending on rock quality $k \times \phi$.

### 4.2 Gas Production
Total gas production is split into two distinct physical streams:
1. **Solution Gas:** Hydrocarbon gas (e.g., CH4) released from the oil as dictated by the initial GOR.
2. **Recycled CO2:** Post-breakthrough, injected CO2 returns through producers. This recycle fraction grows exponentially over time towards an asymptote defined by `(1 - trapping_efficiency)`.

### 4.3 CO2 Injection Schemes
The profile generator explicitly supports cyclic schemas by modulating the injection rate dynamically:
- **Continuous:** Constant rate.
- **WAG (Water-Alternating-Gas):** Modulates rate based on `cycle_length_days` and `wag_ratio`.
- **Huff-n-Puff:** Modulates rate through injection, soaking, and production phases.
- **SWAG, Tapered, Pulsed:** Variable boundary constraints over the project lifetime.

---

## 5. CO2 Storage Calculation (Breakthrough-Aware)

CO2 storage is formulated as a rigorous mass balance, accounting for fluid breakthrough.

```
Net_Stored_CO2 = Total_Injected - Recycled_CO2 - CO2_in_Solution_Gas
```
- **Recycled CO2:** Before breakthrough, all CO2 is assumed trapped. After breakthrough, the fraction of injected CO2 that is produced grows exponentially.
- **CO2 in Solution Gas:** A predefined fraction (default ~20%) of the hydrocarbon solution gas is assumed to be dissolved CO2 originally present or rapidly saturated in the oil phase.
- **Storage Efficiency:** `Net_Stored_CO2 / Total_Injected`.

---

## 6. Integration with Optimization Engine

The Surrogate Engine implements a `SimulationEngineInterface` via a wrapper, allowing it to be integrated directly into the `OptimisationEngine`. Its mathematical formulation is designed to be evaluated rapidly, enabling population-based metaheuristics (NSGA-II) to test thousands of configurations.

### 6.1 PyGAD and Multi-Objective Compatibility
The optimization engine uses `pygad` to drive the evolutionary search. The surrogate engine is natively compatible with both:
- **Single-Objective GA**: Evaluates fitness for a single metric (e.g., NPV).
- **NSGA-II**: When configured for multi-objective optimization, the wrapper returns a multi-dimensional array `[obj1, obj2]`, allowing `pygad` to construct the Pareto front.
- **Parallel Batching**: To maximize performance, the optimization wrapper leverages `ProcessPoolExecutor` to dispatch the `predict()` calls in parallel batches across the available CPU cores.

### Key Scalar Results (Objective Functions)
- `recovery_factor`: Final oil recovery factor (0-1).
- `npv`: Net present value (USD) based on economic parameters and annual cashflows over the project lifetime.
- `cumulative_oil`: Total oil produced (STB).
- `co2_stored`: Total CO2 permanently trapped (tonnes).

### Time-Series Array Results (Constraints)
- `pressure`: Reservoir pressure profile (psi) used to evaluate maximum/minimum pressure constraints.
- `gas_production_rate`: Split into `co2_production_rate` and `hydrocarbon_gas_production_rate` to evaluate surface facility recycling constraints.
- `water_production_rate`: Water cut constraints.