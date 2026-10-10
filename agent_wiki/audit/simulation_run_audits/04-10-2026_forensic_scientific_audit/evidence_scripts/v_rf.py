import sys, warnings
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")
import numpy as np
from core.engine_surrogate.analytical_models import get_analytical_model

print("=== CRIT-06: KovalSurrogate RF vs M (v_dp=0.5 default) ===")
k = get_analytical_model("koval")
for M in [0.5, 0.9, 0.999999, 1.0, 1.000001, 1.1, 1.3, 1.4, 1.5, 2.0, 3.0, 5.0]:
    print("  M=%-9s RF=%.6f" % (M, k.calculate_recovery(mobility_ratio=M, v_dp=0.5)))

print("=== HIGH-02: E_A cliff in Immiscible (measured via surrogate_models) ===")
from core.engine_surrogate import surrogate_models as sm
for M in [0.9, 1.0, 1.0001, 1.01, 1.1, 2.0]:
    ea = sm.calculate_areal_sweep_efficiency(M) if hasattr(sm, "calculate_areal_sweep_efficiency") else None
    print("  M=%-7s E_A=%s" % (M, ea))

print("=== CRIT-07: ImmiscibleSurrogate RF grid ===")
im = get_analytical_model("immiscible")
vals = []
for swi in [0.15, 0.25, 0.35]:
    for sor in [0.20, 0.30, 0.40]:
        for v in [0.3, 0.6, 0.9]:
            rf = im.calculate_recovery(s_wi=swi, sor=sor, v_dp=v, mobility_ratio=1.5,
                                       viscosity_oil=2.0, viscosity_inj=0.05, soi=1 - swi)
            vals.append(rf)
vals = np.array(vals)
print("  n=%d  min=%.6f  max=%.6f  all_equal_0.10=%s" % (len(vals), vals.min(), vals.max(),
      bool(np.allclose(vals, 0.10))))
