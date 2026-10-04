import sys, warnings
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")

print("=== HIGH-06: Koval H, two formulas ===")
for v in [0.2, 0.5, 0.8, 0.9, 0.99]:
    a = 1.0 / (1.0 - v) ** 2          # analytical_models.py:174
    b = 10.0 ** (v / (1.0 - v))       # surrogate_engine.py:894-895
    print("  v=%.2f  1/(1-v)^2=%12.4g   10^(v/(1-v))=%12.4g   ratio=%.4g" % (v, a, b, b / a))

print("=== CRIT-08: containment score floor ===")
class FakeAdv:
    containment_safety_margin = 1.0
    containment_pressure_weight = 0.5
    containment_seal_weight = 0.3
    containment_structure_weight = 0.2
    fracture_pressure_limit_fraction = 0.9
    containment_critical_threshold = 0.3
    # NOTE: no reservoir_seal_integrity_factor / structural_trapping_factor
from core.objectives.storage import calculate_geomechanical_containment_score
import numpy as np
adv = FakeAdv()
for label, prof, pfrac in [("avg P well below frac", np.array([2000.0, 2100.0]), 6000.0),
                           ("avg P = frac (worst)", np.array([6000.0]), 6000.0)]:
    s = calculate_geomechanical_containment_score(prof, pfrac, adv)
    print("  %-24s S_cont=%.4f  threshold=%.2f  prune_possible=%s"
          % (label, s, adv.containment_critical_threshold, s <= adv.containment_critical_threshold))

print("=== HIGH-15: trapping inversion ===")
for sgc in [0.05, 0.10, 0.20]:
    print("  s_gc=%.2f -> gas_trapping = 1-s_gc = %.2f (should be s_gc-ish, small)" % (sgc, 1 - sgc))

print("=== HIGH-04: 'or'-default mask in PhDHybrid ===")
from core.engine_surrogate.analytical_models import get_analytical_model
m = get_analytical_model("phd_hybrid")
base = dict(pressure=3000.0, target_pressure_psi=3000.0, mmp=1200.0, mobility_ratio=2.0)
print("  all defaults      RF=%.6f" % m.calculate_recovery(**base))
print("  v_dp=0.0 explicit RF=%.6f" % m.calculate_recovery(**base, v_dp=0.0))
print("  v_dp=0.9          RF=%.6f" % m.calculate_recovery(**base, v_dp=0.9))
