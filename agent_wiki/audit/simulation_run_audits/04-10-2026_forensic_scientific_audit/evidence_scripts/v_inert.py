import sys, warnings
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")
from core.engine_surrogate.analytical_models import get_analytical_model

h = get_analytical_model("hybrid")
print("=== CRIT-13: HybridSurrogate RF vs mobility_ratio ===")
for M in [0.98, 1.0, 1.5, 2.0, 3.0, 5.0]:
    rf = h.calculate_recovery(pressure=3000.0, mmp=1800.0, mobility_ratio=M)
    print("  M=%-5s RF=%.6f" % (M, rf))

print("\n=== CRIT-13: HybridSurrogate alpha, c7 key mismatch ===")
# engine writes 'c7_plus' (surrogate_engine.py:865); HybridSurrogate reads 'c7_plus_fraction'
rf1 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, c7_plus=0.9)
rf2 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, c7_plus_fraction=0.9)
print("  with c7_plus=0.9        RF=%.6f  (engine's key - IGNORED)" % rf1)
print("  with c7_plus_fraction=0.9 RF=%.6f  (model's key)" % rf2)
print("  omega last:", getattr(h, "last_omega", None))

print("\n=== CRIT-13: gravity_factor / transition_alpha/beta ===")
rf3 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, gravity_factor=0.5)
rf4 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, gravity_factor=1.5)
print("  gravity_factor=0.5 RF=%.6f  vs 1.5 RF=%.6f  -> %s"
      % (rf3, rf4, "IDENTICAL (inert)" if abs(rf3 - rf4) < 1e-15 else "affects"))
rf5 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, transition_alpha=0.8, transition_beta=2.0)
rf6 = h.calculate_recovery(pressure=3000.0, mmp=1800.0, transition_alpha=1.2, transition_beta=10.0)
print("  transition_alpha/beta low  RF=%.6f vs high RF=%.6f -> %s"
      % (rf5, rf6, "IDENTICAL (inert)" if abs(rf5 - rf6) < 1e-15 else "affects"))
