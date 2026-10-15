"""Audit-only numeric verification. No repository file is modified."""
import sys
import numpy as np

sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")

from core.engine_surrogate.analytical_models import KovalSurrogate, MiscibleSurrogate, ImmiscibleSurrogate
from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine

print("=== 1. KovalSurrogate._literature_based_recovery across M = 1 (v_dp=0.5) ===")
k = KovalSurrogate()
for M in [0.90, 0.95, 0.98, 0.99, 0.999, 1.001, 1.01, 1.02, 1.05, 1.1, 1.5, 3.0]:
    rf = k.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0)
    print(f"  M={M:<6} RF={rf:.6f}")

print("\n=== 1b. KovalSurrogate across M = 1 for v_dp = 0.3 / 0.7 ===")
for vd in [0.3, 0.7]:
    row = []
    for M in [0.99, 1.01]:
        row.append(k.calculate_recovery(mobility_ratio=M, v_dp=vd, pressure=3000.0, mmp=2500.0))
    print(f"  v_dp={vd}: M=0.99 -> RF={row[0]:.6f} | M=1.01 -> RF={row[1]:.6f} | ratio={row[0]/max(row[1],1e-12):.1f}x")

print("\n=== 2. Craig areal sweep (surrogate_models.calculate_areal_sweep_efficiency) ===")
for M in [0.99, 1.0, 1.01, 1.05, 1.1, 2.0, 5.0]:
    print(f"  M={M:<5} E_A={calculate_areal_sweep_efficiency(M):.6f}")

print("\n=== 3. ImmiscibleSurrogate total RF across M = 1 ===")
im = ImmiscibleSurrogate()
for M in [0.95, 0.99, 1.0, 1.01, 1.05, 2.0]:
    rf = im.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               sor=0.25, soi=0.75, s_wi=0.25)
    print(f"  M={M:<5} RF={rf:.6f}")

print("\n=== 4. HCPVI: engine expression vs textbook definition ===")
ooip, swi = 1.0e6, 0.25
Bo = 1.15
eng = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0)
for P in [1500, 2500, 3500]:
    bco2 = eng.calculate_co2_fvf_rb_per_mscf(P)
    cum_mscf = 5.0e6
    engine = cum_mscf * bco2 / (ooip * bco2 / (1.0 - swi))
    textbook = (cum_mscf * bco2) / (ooip * Bo / (1.0 - swi))
    print(f"  P={P}: Bco2={bco2:.4f} RB/MSCF  engine_hcpvi={engine:.4f}  textbook={textbook:.4f}"
          f"  ratio={engine/textbook:.3f}")

print("\n=== 5. Bo(P) above p_init (negative compressibility?) ===")
bos = []
ps = [1500, 2000, 2500, 2800, 3000, 3200, 3500, 4000]
for P in ps:
    bo = eng.calculate_oil_fvf_rb_per_stb(P, 0.0)
    bos.append(bo)
    print(f"  P={P:>5}  Bo={bo:.6f}")
g = np.gradient(bos, ps)
print("  dBo/dP (RB/STB per psi):", [f"{x:.2e}" for x in g])
print("  apparent c_o = (1/Bo)(dBo/dP):", [f"{gg/bb:.2e}" for gg, bb in zip(g, bos)])

print("\n=== 6. bg_hc constant check ===")
z, T_R, P = 0.85, 610.0, 2500.0
code = 0.1587 * z * T_R / P
exact = 5.035 * z * T_R / P
print(f"  code bg_hc   = {code:.5f} RB/MSCF")
print(f"  textbook bg  = {exact:.5f} RB/MSCF   ratio exact/code = {exact/code:.2f}x")

print("\n=== 7. CO2 viscosity: code vs literature (Altunin/Vesovic scale) ===")
# reference values for supercritical CO2 (approx, Altunin handbooks)
ref = {(1500, 339): 4.4e-5, (2500, 339): 5.6e-5, (3500, 339): 6.8e-5, (2500, 373): 5.0e-5}
for (Pr, Tr), refval in ref.items():
    Tf = Tr - 491.67
    e2 = SolventExtendedPVTEngine(reservoir_temperature_f=Tf, initial_pressure_psi=3000.0)
    mu = e2.calculate_co2_viscosity_cp(Pr) * 1e-3  # cP -> Pa.s
    print(f"  P={Pr} psia, T={Tf + 491.67:.0f}R: code={mu:.3e} Pa.s  ref~{refval:.1e} Pa.s  ratio={mu/refval:.2f}")

print("\n=== 8. CO2 density / Bco2 vs reference ===")
# reference: rho_CO2 at 150F: ~ 770 kg/m3 @1500psi, ~ 850? (dense phase)
for P in [1000, 1500, 2500, 3500]:
    rho = eng.calculate_co2_density_kg_m3(P)
    print(f"  P={P}: rho={rho:.1f} kg/m3 -> Bco2={327.362/rho:.4f} RB/MSCF")

print("\n=== 9. Breakthrough H formula divergence: 10^(V/(1-V)) vs 1/(1-V)^2 ===")
for v in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    a = 10.0 ** (v / max(1.0 - v, 1e-4))
    b = 1.0 / max(1.0 - v, 1e-4) ** 2
    print(f"  V={v:<5} H_engine(surrogate_engine:895)={a:<14.4g} H_analytical(analytical_models:174)={b:<10.4g} ratio={a/b:.2f}")
