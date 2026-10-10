"""Audit-only numeric verification round 4. No repository file is modified."""
import sys
import numpy as np

sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")

from core.engine_surrogate.analytical_models import ImmiscibleSurrogate, MiscibleSurrogate

print("=== A. 'Hall-Yarborough' z_hc (pvt_state.py:362-367) ===")
gamma_g = 0.75
temp_r = 150.0 + 459.67 + 10.0  # placeholder; recompute properly below
# Use engine defaults
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine
eng = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0)
print("  engine temp_r =", eng.temp_r, " K?", eng.temp_k, " api=", eng.api, " gamma_g=", getattr(eng, "gamma_g", None))
gg = getattr(eng, "gamma_g", 0.75)
for p in (1500.0, 2500.0, 3500.0, 4500.0):
    Ppr = p / (709.6 - 58.7 * gg)
    Tpr = eng.temp_r / (170.5 + 307.3 * gg)
    t_inv = 1.0 / max(Tpr, 0.1)
    z = 1.0 + (0.06422 * t_inv - 0.00332 * (t_inv ** 2)) * Ppr
    print(f"  P={p:>6.0f} Ppr={Ppr:.3f} Tpr={Tpr:.3f} z_hc(code)={z:.4f} clipped={min(max(z,0.65),1.4):.4f}"
          f"   (Standing-Katz at Tpr~1.4-1.6, Ppr~1.5-4.5: ~0.75-0.95)")

print("\n=== B. Craig 5-spot E_A cliff (analytical_models.py:307-312 / surrogate_models.py:164-166) ===")
for m in (0.5, 0.9999, 1.0, 1.0001, 1.01, 1.1, 2.0):
    ea = 1.0 if m <= 1.0 else 0.517 - 0.072 * np.log10(m)
    print(f"  M={m:<8} E_A={ea:.6f}")
print("  step M=1.0->1.01 = %.4f (fractional drop %.1f%%)" %
      (1.0 - (0.517 - 0.072 * np.log10(1.01)), 100 * (1 - (0.517 - 0.072 * np.log10(1.01)))))

print("\n=== C. Immiscible internals: E_d, E_A, E_V product vs 0.10 floor ===")
im = ImmiscibleSurrogate()
# recompute components manually following the source
from core.engine_surrogate.analytical_models import EPSILON, COREY_N_OIL, COREY_N_GAS, S_GC_CRITICAL
for (m_ratio, v, sor) in ((2.0, 0.0, 0.1), (2.0, 0.0, 0.7), (20.0, 0.0, 0.1), (1.0, 0.0, 0.1), (5.0, 0.5, 0.4)):
    viscosity_oil, viscosity_inj = 2.0, 0.05
    mr = max(viscosity_oil / max(viscosity_inj, EPSILON), EPSILON)
    s_gc = S_GC_CRITICAL
    s_range = np.linspace(s_gc, 1.0 - sor - s_gc, 500)

    def fg(s_g):
        s_star = np.clip((s_g - s_gc) / (1.0 - sor - s_gc), 0.0, 1.0)
        k_ro = (1.0 - s_star) ** COREY_N_OIL
        k_rg = s_star ** COREY_N_GAS
        return 1.0 / (1.0 + (k_ro / np.maximum(k_rg, EPSILON)) * (viscosity_inj / max(viscosity_oil, EPSILON)))

    f = fg(s_range)
    tang = f / (s_range - s_gc + EPSILON)
    idx = np.argmax(tang[1:]) + 1
    s_gf = s_range[idx]
    ed = (s_gf - s_gc) / (1.0 - s_gc)
    ea = 1.0 if mr <= 1.0 else min(max(0.517 - 0.072 * np.log10(mr), 0.1), 1.0)
    ev = min(max(1.0 - v ** 0.7, 0.1), 1.0)
    print(f"  code-M={mr:<5} v={v:<4} sor={sor:<4}: s_gf={s_gf:.4f} E_d={ed:.4f} E_A={ea:.4f} E_V={ev:.4f}"
          f" product={ed*ea*ev:.5f} -> returned RF={im.calculate_recovery(mobility_ratio=2.0, v_dp=v, pressure=3000.0, mmp=2500.0, sor=sor, s_wi=0.25):.5f}")

print("\n=== D. Miscible floor at zero injection ===")
mi = MiscibleSurrogate()
for t in (0.0, 0.01, 0.1, 0.5, 1.0, 1.8, 3.0):
    print(f"  hcpvi={t:<5} RF={mi.calculate_recovery(mobility_ratio=3.0, v_dp=0.5, pressure=3000.0, mmp=2500.0, hcpvi=t, s_wi=0.25):.6f}")

print("\n=== E. 'or'-default trap: PhDHybridSurrogate with explicit 0.0 values ===")
from core.engine_surrogate.analytical_models import PhDHybridSurrogate
ph = PhDHybridSurrogate()
base = dict(mobility_ratio=3.0, pressure=3000.0, mmp=2500.0, hcpvi=1.8, sor=0.25, s_wi=0.25, v_dp=0.5)
a = ph.calculate_recovery(**base)
b = ph.calculate_recovery(**{**base, "v_dp": 0.0})
c = ph.calculate_recovery(**{**base, "s_wi": 0.0})
d = ph.calculate_recovery(**{**base, "c7_plus_fraction": 0.0})
print(f"  baseline RF={a:.6f}  v_dp=0.0 -> {b:.6f}  s_wi=0.0 -> {c:.6f}  c7=0.0 -> {d:.6f}")
print("  (0.0 is a legal physical value for s_wi/v_dp/c7; `or` fallbacks silently replace it)")
