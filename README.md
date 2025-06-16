# CO₂-EOR Optimization Framework

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI--blue)](https://doi.org/)
[![PyPI Version](https://img.shields.io/pypi/v/co2eor-optimizer)](https://pypi.org/project/co2eor-optimizer/)
[![Tests](https://img.shields.io/badge/tests-95%25%20coverage-brightgreen)](tests/)

   
*PhD candidate: Petrenko Taras Sergiyovych*   
*Advisor: Branimir Cvetcovich*   

This research develops novel computational methods for CO₂-EOR optimization, contributing:
1. Hybrid MMP correlation framework with improved accuracy (RMSE < 150 psi)
2. Physics-informed genetic algorithm with 20-30% faster convergence
3. GPU-accelerated sweep efficiency modeling
4. Field-validated uncertainty quantification

*Key Publications:*
* Petrenko, T. (2025). Study of physicochemical and geochemical aspects of enhanced oil recovery and CO₂ storage in oil reservoirs. *Technology Audit and Production Reserves*, *2*(1(82)), 24–29. [https://doi.org/10.15587/2706-5448.2025.325343](https://doi.org/10.15587/2706-5448.2025.325343)


## Research Methodology
1. **Data Collection**
   - TBD

2. **Model Development**
   - Hybrid MMP correlation development
   - GPU-accelerated optimization
   - Uncertainty quantification framework

3. **Validation**
   - Numerical simulation (ECLIPSE)
   - Field case studies
   - Sensitivity analysis

## Key Features

- **Comprehensive MMP Calculation**
  - Multiple empirical correlations (Cronquist, Glaso, Yuan)
  - Temperature and composition dependent

- **Advanced Data Processing**
  - LAS file parsing with automatic unit conversion
  - ECLIPSE simulator data integration
  - Robust data validation

- **Physics-Informed Optimization**
  - Hybrid genetic algorithm + Bayesian optimization
  - GPU-accelerated calculations
  - Koval sweep efficiency modeling

- **Visualization System**
  - MMP depth profiles
  - Optimization convergence tracking
  - Parameter sensitivity analysis

## Installation

### From PyPI (recommended)
```bash
pip install co2eor-optimizer
```

### From Source
```bash
# Clone repository
git clone https://github.com/fgfalll/WAG_optimisation.git
cd WAG_optimisation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

1. Import your reservoir data (LAS or ECLIPSE format)
2. Calculate MMP for your reservoir conditions
3. Run optimization to determine optimal injection parameters

## Usage Example

```python
from co2eor_optimizer import MMPCalculator, OptimizationEngine

# Calculate MMP using Yuan correlation
mmp = MMPCalculator().calculate_mmp(
    temperature=180,  # °F
    api_gravity=32,
    gas_composition={'CO2': 0.95, 'N2': 0.05}
)

# Optimize injection scheme
results = OptimizationEngine().optimize_recovery(
    reservoir_data='eclipse_data.DATA',
    constraints={'max_injection_pressure': 5000}  # psi
)

print(f"Optimal WAG ratio: {results.optimal_wag_ratio}")
```

## Documentation

Comprehensive documentation is available in the [Doc](Doc/) directory:

- [Architecture Overview](Doc/architecture.md) - System design and components
- [Development Timeline](Doc/development_timeline.md) - Project milestones
- [Code Audit Review](Doc/audit_review.md) - Quality assurance report
- [API Reference](Doc/api_reference.md) - Detailed module documentation

## Contributing

We welcome contributions! Please see our:
- [Contribution Guidelines](Doc/CONTRIBUTING.md)
- [Code of Conduct](Doc/CODE_OF_CONDUCT.md)

Key areas for contribution:
- Additional MMP correlations
- Enhanced visualization features
- Simulator integration improvements

## Roadmap

- [ ] Add support for CMG simulator data
- [x] Hybrid GH MMP correlation (completed in v1.2)
- [ ] Advanced PVT integration (viscosity modeling, EOS support)
- [ ] Enhanced GPU acceleration (multi-GPU support, memory optimization)
- [ ] Implement machine learning-based MMP prediction
- [ ] Develop UI (PyQT6)
- [ ] Field data integration module (ECLIPSE results visualization)

## License

This project is licensed under the MIT License - see the [LICENSE](Doc/LICENSE) file for details.

## Contact

For technical inquiries:
[engineering@saynos2011@gmail.com](mailto:saynos2011@gmail.com)

Researcher:
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--1764--5256-a6ce39)](https://orcid.org/0009-0005-1764-5256)   
[@fgfalll](https://github.com/fgfalll)
