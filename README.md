<div align="center">

# CO₂ EOR Optimizer

**A physics-informed surrogate reservoir simulator for CO₂ Enhanced Oil Recovery optimization**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green)](https://pypi.org/project/PyQt6/)
[![Tests](https://img.shields.io/badge/tests-258%20passed-brightgreen)](.github/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-informational)](https://fgfalll.github.io/WAG_optimisation/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

[📖 **Documentation**](https://fgfalll.github.io/WAG_optimisation/) &nbsp;|&nbsp;
[🏗️ Architecture](https://fgfalll.github.io/WAG_optimisation/architecture/overview/) &nbsp;|&nbsp;
[🔬 Physics](https://fgfalll.github.io/WAG_optimisation/physics/reservoir_model/) &nbsp;|&nbsp;
[⚡ Quick Start](#quick-start)

</div>

---

## What It Does

CO₂ EOR Optimizer is a **desktop application and scientific computing engine** that bridges the gap between simple analytical screening tools and full-field compositional simulators (CMG GEM, ECLIPSE). It is designed for reservoir engineers who need:

- **Fast, physics-rigorous optimization** of CO₂ injection parameters (WAG ratio, injection rate, scheme type)
- **Economic analysis** — NPV, cashflow, CO₂ purchase/recycle economics, carbon credits
- **Geomechanical safety enforcement** — EPA Class VI UIC compliance, caprock integrity, fault slip tendency
- **CO₂ storage accounting** — closed-loop carbon balance with structural, residual, solubility, and mineral trapping

The core simulation engine is a **physics-informed intermediate-order reservoir simulator** (not a pure ML proxy) that couples:

| Model | Implementation |
|---|---|
| Fluid displacement | Koval heterogeneity + Todd-Longstaff viscous fingering |
| PVT / thermodynamics | Solvent-extended Peng-Robinson EOS, multi-stage flash |
| Well deliverability | Composite Vogel-Darcy IPR (Darcy + Vogel regimes) |
| Pressure | Coupled material balance with dynamic Bg, VRR tracking |
| Geomechanics | Stress path, Mohr-Coulomb fault slip, caprock failure |
| Optimization | GA, Bayesian, PSO, Differential Evolution |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/fgfalll/WAG_optimisation.git
cd WAG_optimisation

# 2. Set up environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

### Run Tests
```bash
python -m pytest tests/ -v

# Run Project Save/Load verification (mandatory after model/UI changes)
python -m pytest tests/test_project_save_load.py -v
```

---

## Architecture Overview

```
co2eor_optimizer/
├── core/
│   ├── engine_surrogate/      ← Active physics engine (single source of truth)
│   │   ├── surrogate_engine.py     ← SurrogateEngineWrapper — main entry point
│   │   ├── profile_generator_fast.py ← FastProfileGenerator — 4-stream output
│   │   ├── pvt_state.py             ← SolventExtendedPVTEngine (PR-EOS)
│   │   ├── geomechanics_fault.py    ← Caprock & fault integrity
│   │   └── analytical_models.py     ← PhDHybridSurrogate, Koval, B-L models
│   ├── optimisation_engine.py ← GA / BO / PSO / DE optimizer
│   └── objectives/            ← NPV, recovery factor, CO₂ storage objectives
├── ui/                        ← PyQt6 desktop application
├── evaluation/                ← MMP correlations (Cronquist, Yellig-Metcalfe)
├── agent_wiki/                ← Full technical documentation (→ GitHub Pages)
├── tests/                     ← 258 tests, 0 failures
└── deprecated/                ← Legacy engines (do not modify)
```

> **📖 Full architecture documentation**: [fgfalll.github.io/WAG_optimisation/](https://fgfalll.github.io/WAG_optimisation/)

---

## Key Features

### 🛢️ Simulation Engine
- **4-fluid-stream output**: crude oil, natural gas (HC + CO₂), water, injection agent — at daily/monthly/annual resolution
- **WAG / SWAG / Huff-n-Puff / Tapered / Pulsed** injection schemes
- **Dynamic breakthrough time** computed from first principles (Koval 1963)
- **Mass-conservative** WAG mobility buffering (phase contrast, not static multipliers)

### 💰 Economics
- NPV with user-configured CAPEX, variable OPEX, CO₂ purchase/recycle costs, storage credits
- Annual cashflow with discounting
- Carbon tax on leaked CO₂

### 🏛️ Geomechanics
- Pore pressure → horizontal stress coupling ($\Delta\sigma_h = \gamma_h \Delta P$)
- Caprock tensile + Mohr-Coulomb shear failure envelopes
- Fault slip tendency: $T_s = \tau / \sigma_n'$
- EPA Class VI UIC pressure ceiling enforcement

### ⚙️ Optimization
- **Genetic Algorithm** with real-valued chromosomes and tournament selection
- **Bayesian Optimization** (Gaussian Process surrogate)
- **Particle Swarm Optimization**
- **Differential Evolution**

### 💾 Project Persistence & Serialization
- **Lossless `.tphd` file format**: Full state persistence for reservoir parameters, PVT models, well schedules, manual inputs, and optimization runs
- **Type-preserving dataclass serialization**: Preserves nested types (`EOSModelParameters`, `LayerDefinition`, `GeostatisticalParams`)
- **Grid-shape agnostic**: Ingests scalar, 1D flattened, and 3D petrophysical arrays seamlessly
- **Automated verification**: Dedicated regression test suite (`tests/test_project_save_load.py`) guarding against breaking changes

---

## Documentation

Full technical documentation is auto-generated from [`agent_wiki/`](agent_wiki/) and deployed to GitHub Pages:

| Section | Contents |
|---|---|
| [Architecture](https://fgfalll.github.io/WAG_optimisation/architecture/overview/) | Engine routing, module map, execution flow, dependency graph |
| [Physics](https://fgfalll.github.io/WAG_optimisation/physics/reservoir_model/) | Reservoir model, PVT, CO₂ properties, displacement, relative permeability |
| [Development](https://fgfalll.github.io/WAG_optimisation/development/common_pitfalls/) | Change safety matrix, common pitfalls, coding rules, extension points |
| [Audit](https://fgfalll.github.io/WAG_optimisation/audit/technical_debt/) | Dead code, hardcoded values, fallbacks, suspicious logic |
| [Verification](https://fgfalll.github.io/WAG_optimisation/verification/verification_strategy/) | 7-level V&V hierarchy, conservation tests, convergence studies |
| [Validation](https://fgfalll.github.io/WAG_optimisation/validation/benchmarks/) | SPE 5, CMG GEM reference benchmarks |

---

## Requirements

- Python 3.10+
- PyQt6, NumPy, SciPy, Matplotlib, Plotly
- See [`requirements.txt`](requirements.txt) for the complete list

---

## License

MIT License — see [LICENSE](LICENSE) for details.
