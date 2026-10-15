import sys
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine

p = SolventExtendedPVTEngine(reservoir_temperature_f=180.0, initial_pressure_psi=3000.0)

print("=== CRIT-03: Bo(P) at x=0 (no bubble point) ===")
prev = None
for P in [1500, 2000, 2500, 3000, 3500, 4000]:
    bo = p.calculate_oil_fvf_rb_per_stb(P, 0.0)
    d = "" if prev is None else "  dBo/dP = %+.3e 1/psi" % ((bo - prev) / (P - (P - 500)))
    print("  P=%4d  Bo=%.4f RB/STB%s" % (P, bo, d))
    prev = bo

print("=== CRIT-04/05: mixture gas properties at x=0 (pure HC gas) ===")
for P in [1500, 2500, 3500, 4500]:
    r = p.calculate_mixture_gas_properties(P, 0.0)
    Ppr = P / (709.6 - 58.7 * 0.7)
    Tpr = (180.0 + 460.0) / (170.5 + 307.3 * 0.7)
    z_lin = 1.0 + (0.06422 / Tpr - 0.00332 / Tpr**2) * Ppr
    # Bg [RB/MSCF] = 0.02827*Z*T_R/P [ft3/scf] * 1000 scf/MSCF / 5.6146 ft3/bbl
    bg_corr = 0.02827 * z_lin * (180.0 + 460.0) / P * 1000.0 / 5.6146
    print("  P=%4d Ppr=%.2f Tpr=%.3f  z_model=%.4f  bg_model=%.5f  bg_textbook=%.5f  ratio=%.3f"
          % (P, Ppr, Tpr, z_lin, r["bg_rb_per_mscf"], bg_corr, r["bg_rb_per_mscf"] / bg_corr))
