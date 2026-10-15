# Test Suite Architecture & Validation Taxonomy

## 1. Overview & Test Health

The `co2eor_optimizer` test suite provides automated regression testing, numerical stability verification, and physical invariant validation across core simulation and optimization modules.

As of September 2026:
- **Total Test Items**: 274 items collected across 32 test files
- **Execution Results**: 258 passed, 16 skipped (optional GUI/external deps), 0 failed
- **Execution Time**: ~225 seconds for full suite (~14 seconds for unit-only runs)
- **Framework**: `pytest` 9.1.1 with plugins `pytest-cov`, `pytest-qt`, `pytest-xdist`, `hypothesis`

---

## 2. Recommended Test Taxonomy & Directory Structure

To maintain separation between software testing and scientific validation, tests are organized across the following categories:

```
tests/
├── conftest.py                             # Shared fixtures (reservoir_data, eor_params, engine_setup)
├── unit/                                   # Fast software correctness tests (< 1s per file)
│   ├── test_imports.py                     # Clean package imports across all submodules
│   ├── test_data_models.py                 # Dataclass serialization, range checks, units
│   ├── test_exceptions.py                  # Domain exception hierarchy and error reporting
│   └── test_config_manager.py              # Configuration loading and YAML persistence
├── core/                                   # Active simulation and optimizer unit/integration tests
│   ├── test_surrogate_engine.py            # SurrogateEngine scenario evaluations and NPV calculations
│   ├── test_single_simulation.py           # End-to-end single run execution pipeline
│   ├── test_fitness_function.py            # Objective wrapper fitness evaluation and penalties
│   ├── test_optimisation_algorithms.py     # GA, BO, PSO, DE algorithm execution loops
│   ├── test_profile_generator_fast.py      # FastProfileGenerator rate profile synthesis
│   ├── test_analytical_models.py           # Koval, Buckley-Leverett, and PhD hybrid models
│   ├── test_unphysical_chromosome_penalties.py # Rejection of non-viable candidates (-1e12 penalty)
│   └── test_surrogate_engine_reference.py  # Reference CMG benchmark consistency tests
├── physics/                                # Physical correctness & thermodynamic property tests
│   ├── test_mmp_correlations.py            # Cronquist, Lee, Glaso MMP correlations
│   ├── test_eos_pr.py                      # Peng-Robinson flash calculations and Z-factors
│   ├── test_co2_properties.py              # CO2 density, viscosity, and solubility
│   └── test_relative_permeability.py       # Corey relative permeability curve validation
├── conservation/                           # Material balance & conservation law tests
│   ├── test_mass_conservation.py           # Liquid and gas mass conservation
│   └── test_carbon_balance.py              # Closed-loop CO2 balance (Injected = Stored + Produced)
├── numerical/                              # Numerical stability, solvers, and safeguards
│   ├── test_pressure_ode_stability.py      # IPR-coupled material balance stability
│   └── test_edge_cases.py                  # Zero injection, 100% water cut, extreme pressures
└── validation/                             # Scientific benchmark validation against published data
    ├── cmg_gem_validation.py               # CMG GEM reference cases (gmflu001 - gmflu004)
    ├── spe5_benchmark_validation.py        # SPE 5 comparative benchmark
    └── test_compositional_validation.py    # 1D compositional numerical simulator checks
```

---

## 3. Standard Execution Commands

### Running All Automated Tests
```powershell
.venv\Scripts\pytest.exe tests/ -v
```

### Running Fast Core Tests (Bypassing Slow Full Optimizations)
```powershell
.venv\Scripts\pytest.exe tests/core/test_single_simulation.py tests/core/test_surrogate_engine.py -v
```

### Running Scientific Conservation Tests
```powershell
.venv\Scripts\pytest.exe tests/ -k "conservation or balance or reference" -v
```

### Running Tests in Parallel
```powershell
.venv\Scripts\pytest.exe tests/ -n auto -v
```

### Running with Code Coverage
```powershell
.venv\Scripts\pytest.exe tests/ --cov=core --cov=evaluation --cov=analysis --cov-report=term-missing
```

---

## 4. Test Invariants for AI Agents

Any code modification in this repository must satisfy the following **strict testing invariants**:

1. **Zero Test Regressions**: All 258 core tests must continue to pass cleanly (`0 failed`).
2. **Strict Carbon Closure**: Cumulative injected CO₂ must equal net stored CO₂ plus cumulative produced CO₂ to within 0.1% tolerance:
   $$\left| Q_{\text{CO2,gross\_inj}} - (Q_{\text{CO2,stored}} + Q_{\text{CO2,produced}} + Q_{\text{leakage}}) \right| < 10^{-3} \times Q_{\text{CO2,gross\_inj}}$$
3. **Strict Mass Conservation**: Re-normalized cumulative oil must exactly equal $OOIP \times RF$:
   $$\left| \sum (q_{o,t} \cdot \Delta t) - (OOIP \times RF) \right| < 10^{-6}$$
4. **No Unhandled Exceptions**: All tests must verify that invalid inputs (e.g. negative permeability, pressure above fracture limit) raise typed domain exceptions (`ValidationError`, `ReservoirSimulationError`) rather than bare `Exception` or silent failures.
5. **No Diluted Penalties**: Non-viable candidates must trigger the hard failure penalty (`FAILURE_PENALTY = -10^{12}`) rather than artificial synthesizers.
