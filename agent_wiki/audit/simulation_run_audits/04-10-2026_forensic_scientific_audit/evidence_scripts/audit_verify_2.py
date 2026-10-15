"""Audit-only numeric verification round 2. No repository file is modified."""
import sys
import numpy as np

sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")

from core.engine_surrogate.analytical_models import ImmiscibleSurrogate, HybridSurrogate, MiscibleSurrogate
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine

print("=== A. ImmiscibleSurrogate sensitivity (is it flat?) ===")
im = ImmiscibleSurrogate()
print("  vary M at v_dp=0.5, soi=0.75, sor=0.25, s_wi=0.25:")
for M in [0.5, 0.9, 1.0, 1.5, 2, 3, 5, 8, 12, 20]:
    rf = im.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               sor=0.25, soi=0.75, s_wi=0.25)
    print(f"    M={M:<5} RF={rf:.6f}")
print("  vary v_dp at M=2:")
for v in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    rf = im.calculate_recovery(mobility_ratio=2.0, v_dp=v, pressure=3000.0, mmp=2500.0,
                               sor=0.25, soi=0.75, s_wi=0.25)
    print(f"    v_dp={v:<5} RF={rf:.6f}")
print("  vary sor at M=2, v_dp=0.5 (does rf_max_physical cap respond?):")
for sor in [0.10, 0.20, 0.30, 0.40, 0.60, 0.70]:
    rf = im.calculate_recovery(mobility_ratio=2.0, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               sor=sor, soi=0.75, s_wi=0.25)
    print(f"    sor={sor:<5} RF={rf:.6f}")

print("\n=== B. HybridSurrogate across M = 1 (uses immiscible+miscible) ===")
hy = HybridSurrogate()
for M in [0.98, 0.99, 1.0, 1.01, 1.02, 1.1, 2.0, 5.0]:
    rf = hy.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               hcpvi=1.8, sor=0.25, soi=0.75, s_wi=0.25)
    print(f"  M={M:<5} RF={rf:.6f}")

print("\n=== C. Hybrid across pressure (P/MMP sigmoid, alpha~0.95, beta=20) ===")
for P in [2200, 2400, 2500, 2600, 2700, 2800, 3000, 3500]:
    rf = hy.calculate_recovery(mobility_ratio=3.0, v_dp=0.5, pressure=P, mmp=2500.0,
                               hcpvi=1.8, sor=0.25, soi=0.75, s_wi=0.25)
    print(f"  P={P} (P/MMP={P/2500:.3f}) RF={rf:.6f}")

print("\n=== D. MiscibleSurrogate vs hcpvi (engine passes buggy hcpvi) ===")
mi = MiscibleSurrogate()
for t in [0.5, 1.0, 1.5, 1.8, 2.5, 3.75, 5.0, 8.0]:
    rf = mi.calculate_recovery(mobility_ratio=3.0, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               hcpvi=t, s_wi=0.25)
    print(f"  hcpvi={t:<5} RF={rf:.6f}")
print("  zero-injection floor check (hcpvi=0):",
      f"{mi.calculate_recovery(mobility_ratio=3.0, v_dp=0.5, pressure=3000.0, mmp=2500.0, hcpvi=0.0, s_wi=0.25):.6f}")

print("\n=== E. CO2 viscosity / density at correct 150 F ===")
eng = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0)
for P in [1000, 1500, 2500, 3500, 4500]:
    rho = eng.calculate_co2_density_kg_m3(P)
    mu = eng.calculate_co2_viscosity_cp(P)
    b = eng.calculate_co2_fvf_rb_per_mscf(P)
    print(f"  P={P:>5}: rho={rho:7.1f} kg/m3  mu={mu:.5f} cP  Bco2={b:.4f} RB/MSCF")

print("\n=== F. Mixture gas: bg_mix with y_co2 sweep at 2500 psi ===")
for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
    g = eng.calculate_mixture_gas_properties(2500.0, y)
    print(f"  y_co2={y:<5} bg={g['bg_rb_per_mscf']:.5f} RB/MSCF  cg={g['compressibility_psi_inv']:.3e} 1/psi  mu={g['viscosity_cp']:.5f}")
print("  (pure hydrocarbon reference Bg textbook at 2500 psi, 150F, Z=0.85: %.4f RB/MSCF)"
      % (5.035 * 0.85 * 609.67 / 2500))

print("\n=== G. PR-EOS density: 3-root / fallback behaviour probe ===")
for P in [14.7, 200, 700, 1070, 1500, 2500, 5000, 10000]:
    rho = eng.calculate_co2_density_kg_m3(P)
    print(f"  P={P:>7}: rho={rho:7.1f} kg/m3")
