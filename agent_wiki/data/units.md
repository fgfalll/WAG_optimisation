# Unit Systems & Conversion Pitfalls

## 1. Overview of Unit Standards

The primary engineering calculations in this repository utilize **Oilfield Petroleum Engineering Units**. However, thermodynamic and geostatistical calculations intermittently convert to **SI (Metric) Units**.

---

## 2. Standard Oilfield vs SI Reference Table

| Physical Dimension | Standard Field Unit | Standard SI Unit | Conversion Factor (Field $\to$ SI) |
| :--- | :--- | :--- | :--- |
| **Pressure** | psia, psig | Pa, MPa, bar | $1\text{ psi} = 6,894.757\text{ Pa} = 0.00689476\text{ MPa}$ |
| **Permeability** | milliDarcy (mD) | $\text{m}^2$ | $1\text{ mD} = 9.869233 \times 10^{-16}\text{ m}^2$ |
| **Temperature** | °F | K, °C | $T(\text{K}) = (T(^\circ\text{F}) - 32) \times 5/9 + 273.15$ |
| **Liquid Volume** | STB, bbl | $\text{m}^3$ | $1\text{ bbl} = 0.1589873\text{ m}^3 = 42\text{ US gallons}$ |
| **Gas Volume** | MSCF (1,000 SCF) | $\text{Sm}^3$ | $1\text{ MSCF} = 28.31685\text{ Sm}^3$ |
| **Gas Rate** | MSCFD | $\text{Sm}^3/\text{day}$ | $1\text{ MSCFD} = 28.31685\text{ Sm}^3/\text{day}$ |
| **Oil Rate** | STB/day (BOPD) | $\text{m}^3/\text{day}$ | $1\text{ STB/day} = 0.1589873\text{ m}^3/\text{day}$ |
| **Viscosity** | centipoise (cP) | $\text{Pa}\cdot\text{s}$ | $1\text{ cP} = 10^{-3}\text{ Pa}\cdot\text{s} = 0.01\text{ poise}$ |
| **Density** | $\text{lb/ft}^3$, °API | $\text{kg/m}^3$ | $\gamma_{API} = 141.5 / \gamma_o - 131.5$ |
| **Compressibility**| $\text{psi}^{-1}$ | $\text{Pa}^{-1}$ | $1\text{ psi}^{-1} = 1.45038 \times 10^{-4}\text{ Pa}^{-1}$ |

---

## 3. Critical Unit Conversion Pitfalls Identified in Codebase

### Pitfall 1: MSCFD Treated Directly as Reservoir Barrels/Day in Tank ODE (RESOLVED)
- **Location**: [core/engine_surrogate/surrogate_engine.py line 919](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L919).
- **Historical Error**: `profile_result["injection_profile"]` was generated in **MSCFD** (surface thousand standard cubic feet per day) by `FastProfileGenerator`, but subtracted directly from production in reservoir barrels.
- **Physical Reality**: At surface conditions, 1 MSCF is 1,000 SCF. In the reservoir, gas occupies a volume $V_{res} = \text{MSCF} \times 1000 \times B_g$ (where $B_g$ is in RB/SCF), or $V_{res} = \text{MSCF} / \text{mscf\_per\_res\_bbl}$.
- **Resolution**: Updated to `q_inj_rb = profile_result["injection_profile"] * 1000.0 * bg` (and gas production rate similarly converted via `* 1000.0 * bg`), ensuring both injection and production terms in the stiff ODE solver are dimensionally homogeneous in RB/day.

### Pitfall 2: Permeability Conversion to $\text{ft}^2$
- **Location**: [core/engine_surrogate/surrogate_engine.py line 224](file:///d:/rep/4.6/co2eor_optimizer/core/engine_surrogate/surrogate_engine.py#L224).
- **Code**: `perm_ft2 = perm_md * 1.062e-14 # Fixed: was 1.127e-8 (incorrect by ~1 million)`
- **Audit Context**: Earlier versions had used $1.127 \times 10^{-8}$ (which confused Darcy's engineering units with ft²), resulting in a one-million-fold error in injectivity. The current constant $1.062 \times 10^{-14}$ is mathematically correct ($9.869 \times 10^{-16}\text{ m}^2 \times 10.7639\text{ ft}^2/\text{m}^2$).

### Pitfall 3: CO₂ Surface Density Rounding (0.05254 vs 0.053)
- **Location**: `PhysicalConstants.CO2_DENSITY_TONNE_PER_MSCF` ($0.05254$) vs. `0.053` hardcoded in `analytical_models.py`, `surrogate_models.py`, and `surrogate_engine.py`.
- **Impact**: Accumulates a ~0.87% mass accounting discrepancy between theoretical PVT calculations and profile integration.

### Pitfall 4: Grid Block Dimension Unit Confusion in `_create_reservoir_data` (RESOLVED)
- **Location**: [core/data_integration_engine.py line 350](file:///d:/rep/4.6/co2eor_optimizer/core/data_integration_engine.py#L350) & [ui/data_management_widget.py](file:///d:/rep/4.6/co2eor_optimizer/ui/data_management_widget.py).
- **Historical Error**: `DataManagementWidget` block sizes `dx, dy, dz` are input in **feet** (field units). The data integration engine previously assumed inputs were in meters, dividing by metric constants: `thickness_ft = nz * dz / 0.3048` and `area_acres = (nx * dx * ny * dy) / 4046.86`.
- **Physical Impact**: Because dimensions were already in feet, dividing by $0.3048\text{ m/ft}$ and $4046.86\text{ m}^2/\text{acre}$ (instead of $43,560\text{ ft}^2/\text{acre}$) caused a $(1 / 0.3048) \times (43560 / 4046.86) \approx 3.2808 \times 10.764 \approx 35.314\times$ artificial volume inflation.
  - A standard $50\times 50\times 10$ block reservoir ($100\text{ ft}\times 100\text{ ft}\times 10\text{ ft}$) inflated from its true $4.86\times 10^6\text{ STB}$ OOIP to $171.7\times 10^6\text{ STB}$.
  - This mismatch caused massive gas undersaturation and distorted injection requirement calculations.
- **Resolution**: Converted formulas to field-consistent conversions:
  - $\text{area\_acres} = \frac{nx \cdot dx \cdot ny \cdot dy}{43,560\text{ ft}^2/\text{acre}}$
  - $\text{thickness\_ft} = nz \cdot dz$
  - Handed off explicit `thickness_ft`, `area_acres`, and `length_ft` directly from project data into `ReservoirData`.

