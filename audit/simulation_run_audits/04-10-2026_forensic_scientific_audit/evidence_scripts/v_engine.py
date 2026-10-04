import sys, warnings, json
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
warnings.filterwarnings("ignore")
import numpy as np
from core.data_models import EORParameters, OperationalParameters, EconomicParameters
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from tests.scientific.conftest import make_reservoir_instance

res = make_reservoir_instance({"ooip_stb": 5_000_000.0, "initial_pressure": 4000.0,
                               "temperature": 160.0, "mmp": 1800.0})
eor = EORParameters(default_mmp_fallback=1800.0, injection_rate=10000.0)
ops = OperationalParameters(project_lifetime_years=10, time_resolution="monthly")
econ = EconomicParameters()

for model in ["hybrid", "phd_hybrid"]:
    eng = SurrogateEngine(recovery_model_type=model)
    r = eng.evaluate_scenario(res, eor, ops, econ)
    print("=== model=%s ===" % model)
    for k in ["recovery_factor", "npv", "cumulative_oil", "co2_stored",
              "cumulative_co2_injected_tonne", "cumulative_co2_produced_tonne",
              "total_leakage_tonne", "annual_leakage_tonne", "max_sandface_pressure_psi",
              "pressure", "reservoir_pressure", "hcpvi"]:
        v = r.get(k, "<ABSENT>")
        if isinstance(v, np.ndarray):
            print("  %-38s ndarray len=%d mean=%.4g last=%.4g" % (k, len(v), v.mean(), v[-1]))
        elif isinstance(v, (list, tuple)) and len(v) > 4:
            print("  %-38s list len=%d last=%.4g" % (k, len(v), v[-1]))
        else:
            print("  %-38s %s" % (k, v))
    print("  n_keys:", len(r))
    # does the NPV's implied RF match the reported RF?
    rf = r.get("recovery_factor")
    npv = r.get("npv")
    co = r.get("cumulative_oil")
    if rf is not None and co is not None:
        ooip = 5_000_000.0
        print("  RF reported = %.4f ; cumulative_oil/ooip = %.4f (delta %.4f)"
              % (rf, co / ooip, rf - co / ooip))
    # pressure keys
    pkeys = [k for k in r if "pressure" in k]
    print("  pressure keys:", pkeys)
    print()
