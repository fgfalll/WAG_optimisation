import sys, warnings
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")

# --- CRIT-01: HCPVI algebraic cancellation, direct reproduction of :494/:497 ---
print("=== CRIT-01: simulated_hcpvi = cum_inj_mscf*(1-swi)/ooip (pressure-independent) ===")
ooip, swi, cum_inj = 1_000_000.0, 0.25, 1_000_000.0
for P in [1500, 2500, 3500]:
    # Bg from the engine's own PVT (RB/MSCF) - any value cancels
    from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine
    pvt = SolventExtendedPVTEngine(reservoir_temperature_f=180.0, initial_pressure_psi=3000.0)
    b_co2 = pvt.calculate_co2_fvf_rb_per_mscf(P)
    cum_inj_rb = cum_inj * b_co2
    hcpvi_model = cum_inj_rb / (ooip * b_co2 / (1 - swi))
    # textbook: HCPVI = cum_inj*Bg*(1-swi)/(OOIP*Bo)
    bo = pvt.calculate_oil_fvf_rb_per_stb(P, 0.0)
    hcpvi_true = cum_inj * b_co2 * (1 - swi) / (ooip * bo)
    print("  P=%4d  Bg=%.4f RB/MSCF  model=%.6f  textbook=%.6f  ratio=%.3f"
          % (P, b_co2, hcpvi_model, hcpvi_true, hcpvi_model / hcpvi_true))

# --- HIGH-01: 1-swi-sor vs (1-swi-sor)/(1-swi) ---
print("=== HIGH-01: displacing pore-volume term ===")
swi, sor = 0.25, 0.30
bad = 1 - swi - sor
good = (1 - swi - sor) / (1 - swi)
print("  model 1-swi-sor = %.4f   correct (1-swi-sor)/(1-swi) = %.4f   ratio=%.4f (low by %.1f%%)"
      % (bad, good, bad / good, (1 - bad / good) * 100))
