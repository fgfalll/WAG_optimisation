# Physical & Data Model Classes

## 1. `ReservoirData`

- **File**: `core/data_models.py`
- **Role**: Master petrophysical and volumetric state dataclass.
- **Modification Risk**: **CRITICAL**

### Attributes
- `area: float`: Surface drainage area (acres).
- `thickness: float`: Net reservoir pay thickness (ft).
- `porosity: float`: Average reservoir porosity ($0 < \phi < 1$).
- `permeability: float`: Absolute horizontal permeability (mD).
- `initial_pressure: float`: Initial discovery pressure (psia).
- `temperature: float`: Reservoir temperature (°F).
- `initial_oil_saturation: float`: $S_{oi}$ (fraction).
- `initial_water_saturation: float`: $S_{wi}$ (fraction).
- `ooip: float`: Original Oil in Place (STB).
- `pore_volume: float`: Total pore volume in reservoir barrels:
  $$V_p = \frac{7758 \cdot \text{area} \cdot \text{thickness} \cdot \text{porosity}}{5.615} \text{ bbl}$$
- `total_compressibility: float`: Total system compressibility $c_t = c_f + S_w c_w + S_o c_o$ (default $1.2 \times 10^{-5}\text{ psi}^{-1}$).
- `fracture_pressure: float`: Formation parting limit (psia, typically $0.75 - 0.85\text{ psi/ft} \times \text{depth}$).

---

## 2. `PengRobinsonEOS`

- **File**: `core/unified_engine/physics/eos/__init__.py`
- **Role**: Cubic Peng-Robinson Equation of State for phase equilibria and fluid densities.
- **Modification Risk**: **HIGH**

### Key Methods
- `calculate_z_factor(p: float, t: float, z_comp: np.ndarray) -> Tuple[float, float]`:
  Solves cubic equation for gas and liquid compressibility factors:
  $$Z^3 - (1 - B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0$$
- `flash_calculation(p: float, t: float, z_feed: np.ndarray) -> FlashResult`:
  Solves Rachford-Rice equation using accelerated Newton-Raphson iteration:
  $$f(\Psi) = \sum_{i=1}^{N_c} \frac{z_i (K_i - 1)}{1 + \Psi (K_i - 1)} = 0$$
  Yields equilibrium vapor fraction $\Psi$, liquid compositions $x_i$, and vapor compositions $y_i$.

---

## 3. `CO2Properties`

- **File**: `core/unified_engine/physics/co2_properties.py`
- **Role**: High-accuracy thermophysical property calculator for pure and impure CO₂ streams.
- **Modification Risk**: **MEDIUM**

### Key Methods
- `calculate_density(p_psia: float, t_f: float) -> float`:
  Computes supercritical CO₂ density (lb/cu ft and g/cm³) using Altunin-Gadetskii / Span-Wagner approximations.
- `calculate_viscosity(p_psia: float, t_f: float, density: float) -> float`:
  Computes CO₂ viscosity (cP) incorporating dilute-gas and residual friction terms (Fenghour correlation).
- `calculate_solubility_in_water(p_psia: float, t_f: float, salinity_ppm: float) -> float`:
  Evaluates CO₂ solubility in formation brine (MSCF/STB) accounting for salting-out effects (Duan-Sun correlation).
