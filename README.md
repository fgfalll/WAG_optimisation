# CO2 EOR Optimizer

A comprehensive desktop application and computational engine for **CO2 Enhanced Oil Recovery (EOR)** simulation, validation, and optimization. This tool provides reservoir engineers with a lightning-fast, physics-based surrogate modeling environment that bridges the gap between simple analytical screening and full-field compositional simulation (like CMG GEM).

## Features

### 1. Advanced UI and Data Management
Built with PyQt6, the application provides an intuitive interface for managing complex reservoir models:
* **Reservoir Properties:** Define grid dimensions, uniform or layered properties, OOIP volumetrics, relative permeability (Corey), and geostatistical parameters.
* **PVT & Fluid Properties:** Rigorous built-in thermodynamic calculators for Dead/Live Oil Viscosity (Beggs-Robinson, Chew-Connally), Bubble Point and Compressibility (Vasquez-Beggs), and Formation Volume Factors (Standing). Supports both Black Oil and detailed Equation of State (EOS) compositional models.
* **Well Management:** Define 3D trajectories, perforation depths, wellbore radius, and skin factors for multiple injectors and producers.

### 2. PhD Surrogate Simulation Engine
A high-performance proxy simulation engine that bypasses the computational overhead of traditional numerical simulators while maintaining strict physical integrity:
* **Zero-Dimensional Material Balance:** Fully dynamic time-stepping loop that enforces strict voidage replacement and tracks evolving reservoir pressure.
* **Dynamic Fractional Flow:** Integrates Buckley-Leverett displacement and the Koval heterogeneity method to model viscous fingering, gravity override, and breakthrough timing.
* **Miscibility Tracking:** Real-time evaluation of the Todd-Longstaff mixing parameter ($\omega$), driven by dynamic Minimum Miscibility Pressure (MMP) calculations and logistic sigmoid transitions based on $C_{7+}$ fractions.
* **Physics-Based Defaults:** Operates natively on raw physics without requiring empirical "cheat" data, but allows full calibration for proxy-matching against fine-grid compositional models.

### 3. Empirical Tuning and Proxy Calibration
For advanced workflows where the surrogate engine must perfectly match historical data or full-field models (e.g., ECLIPSE, Intersect, CMG):
* Automatic estimation of tuning parameters directly from PVT and coreflood data.
* Customizable relative permeability endpoints, miscibility windows ($\Delta P/MMP$), transition midpoints ($\alpha_{base}$), and transverse mixing coefficients.

### 4. Visualization
* **2D & 3D Interactive Plots:** Visualize PVT property curves, relative permeability curves, well trajectories, and simulated production profiles (Pressure, Recovery Factor, Gas/Oil ratios) using Plotly and Matplotlib.

## Getting Started

### Prerequisites
* Python 3.10+
* Required packages listed in `requirements.txt` (PyQt6, NumPy, SciPy, Matplotlib, Plotly, etc.)

### Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   * Windows: `.venv\Scripts\activate`
   * Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Launch the main graphical interface:
```bash
python main.py
```

## Documentation
Detailed mathematical derivations and engineering workflows for the surrogate engine can be found in the `docs/` folder:
* **[Surrogate Engine README](docs/SURROGATE_ENGINE_README.md)**: Comprehensive guide covering 1D vs 3D sweep physics, dynamic material balance, proxy tuning parameters, and exact PVT correlation math.

## Validation
The repository includes a suite of validation scripts (e.g., `validation/empirical_engine_validation.py`) that strictly benchmark the PhD Surrogate Engine's recovery and pressure predictions against standard CMG SR3 output files to ensure engineering accuracy.
