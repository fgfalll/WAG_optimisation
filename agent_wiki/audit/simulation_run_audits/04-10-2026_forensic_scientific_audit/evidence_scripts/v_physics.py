import sys, warnings
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")
import numpy as np

# ---- CRIT-11: WAG water rate, :1117-1118 (no x1000) vs :1210 (x1000) ----
print("=== CRIT-11: WAG vs SWAG water rate ===")
rate = 5000.0          # MSCFD field-scale base injection rate
bg = 0.005             # default_gas_fvf (data_models.py:917)
wag_ratio = 1.0
co2_inj_rb_per_day = rate * bg                     # :1117  (treated as RB/D)
wag_water = co2_inj_rb_per_day * wag_ratio         # :1118
swag_water = rate * wag_ratio * bg * 1000.0        # :1210  (same expression, x1000)
print("  WAG  water (line 1117-1118): %10.1f bpd" % wag_water)
print("  SWAG water (line 1210)     : %10.1f bpd" % swag_water)
print("  ratio = %.0fx" % (swag_water / wag_water))
# what the physically correct value is: 5000 MSCFD * Bg[RB/MSCF]
# if default_gas_fvf=0.005 is RB/scf then RB/MSCF = 5.0 -> 25000 RB/D
print("  if 0.005 is RB/scf -> 5000 MSCFD = 5e6 scf/D * 0.005 = %.0f RB/D" % (5e6 * 0.005))

# ---- Invariant #3: M_recycled <= M_produced <= M_injected ----
print("\n=== Invariant #3 (wiki README:35) on a real run ===")
from core.data_models import EORParameters, OperationalParameters, EconomicParameters
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from tests.scientific.conftest import make_reservoir_instance
res = make_reservoir_instance({"ooip_stb": 5_000_000.0, "initial_pressure": 4000.0,
                               "temperature": 160.0, "mmp": 1800.0})
eor = EORParameters(default_mmp_fallback=1800.0, injection_rate=10000.0)
ops = OperationalParameters(project_lifetime_years=10, time_resolution="monthly")
r = SurrogateEngine(recovery_model_type="hybrid").evaluate_scenario(res, eor, ops, EconomicParameters())
inj = r.get("annual_co2_injected_mscf"); prod = r.get("annual_co2_produced_mscf")
rec = r.get("annual_co2_recycled_mscf"); pur = r.get("annual_co2_purchased_mscf")
S, P, R, U = (float(np.sum(x)) for x in (inj, prod, rec, pur))
print("  injected = %.1f MSCF | produced = %.1f | recycled = %.1f | purchased = %.1f"
      % (S, P, R, U))
print("  M_recycled <= M_produced : %s (%.1f <= %.1f)" % (R <= P + 1e-6, R, P))
print("  M_purchased + M_recycled == M_injected : %s (%.1f vs %.1f, err %.3g)"
      % (abs(U + R - S) < 1e-3 * max(S, 1), U + R, S, U + R - S))
leak = r.get("total_leakage_tonne", 0.0)
print("  total_leakage_tonne = %.4f  (injected tonnes = %.1f)" % (leak, S * 0.05295))
print("  cum_stored = inj - prod  -> ignores leakage: %s"
      % ("YES (surrogate_engine.py:543)" if leak == 0 else "no"))
