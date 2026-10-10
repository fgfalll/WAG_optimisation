# Benchmark Datasets & Validation Procedures

## 1. Overview of Benchmarks

The repository includes two major industrial reference benchmark suites:
1. **CMG GEM Field Simulation Cases (`gmflu001` - `gmflu004`)**: Real compositional simulator runs used to benchmark the surrogate engine.
2. **SPE 5 Comparative Solution Project**: Standard 4-component gas injection benchmark across a 3D dipping reservoir.

---

## 2. CMG GEM Benchmark Datasets

Located in `validation/cmg/flu/`:

| Dataset Case | Description | Grid Dimensions | Key Physical Focus | Reference Output Files |
| :--- | :--- | :--- | :--- | :--- |
| `gmflu001_1D` | 1D Continuous CO₂ flood | $50 \times 1 \times 1$ | 1D displacement efficiency, shock front | `gmflu001_1D.dat`, `.sr3`, `.out` |
| `gmflu002_1D` | 1D WAG injection flood | $50 \times 1 \times 1$ | 1D cyclic WAG displacement, breakthrough | `gmflu002_1D.dat`, `.sr3`, `.out` |
| `gmflu002` | 3D Quarter 5-Spot WAG | $15 \times 15 \times 3$ | Areal sweep, vertical crossflow, WAG ratio | `gmflu002.dat`, `.sr3`, `.out` |
| `gmflu003_1D` | 1D Immiscible / Near-MMP flood | $50 \times 1 \times 1$ | Pressure sensitivity near MMP | `gmflu003_1D.dat`, `.sr3`, `.out` |
| `gmflu003` | 3D Heterogeneous WAG flood | $15 \times 15 \times 3$ | Heterogeneity ($V_{DP}$), fingering, channeling | `gmflu003.dat`, `.sr3`, `.out` |

### CMG Binary SR3 Reader
Implemented in [validation/sr3_reader.py](file:///d:/rep/4.6/co2eor_optimizer/validation/sr3_reader.py) using `h5py` to read CMG's native HDF5 result containers. Extracts time series for:
- `PROD-OIL-RATE` (bbl/day)
- `PROD-GAS-RATE` (MSCFD)
- `CUM-OIL-PROD` (bbl)
- `AVE-PRES` (psia)

---

## 3. SPE 5 Comparative Solution Project

The SPE 5 benchmark (Killough & Kossack, 1987) specifies a 3D dipping reservoir model for multi-component gas injection.

### Key Reservoir & Grid Parameters (`validation/spe5_config.py`):
- Grid: $7 \times 7 \times 3 = 147$ blocks
- Length: 2,100 ft ($dx = 300\text{ ft}$)
- Width: 2,100 ft ($dy = 300\text{ ft}$)
- Layer thicknesses: Layer 1 = 20 ft, Layer 2 = 30 ft, Layer 3 = 50 ft
- Dip angle: $10^\circ$
- Porosity: 0.35 uniform
- Permeability: Layer 1 = 500 mD, Layer 2 = 50 mD, Layer 3 = 200 mD ($k_v / k_h = 0.1$)
- Initial pressure: 4,000 psia at datum
- Initial water saturation: $S_{wi} = 0.16$

---

## 4. Benchmark Validation Commands

To execute benchmark validation runs:

```bash
# Validate against CMG reference cases
uv run python -m validation.validate_against_cmg --case gmflu002_1D

# Generate comparative validation plots
uv run python -m validation.engine_validation_plots

# Validate surrogate engine against SPE5
uv run python -m validation.spe5_phd_surrogate_validation
```

*Note*: Ensure `validation/spe5_benchmark_validation.py` line 249 has been patched with the missing comma before running.
