import sys, warnings
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

for model in ["hybrid", "phd_hybrid"]:
    eng = SurrogateEngine(recovery_model_type=model)
    raw = {}
    orig = eng.surrogate_model.recovery_model.calculate_recovery

    def spy(**kw):
        v = orig(**kw)
        raw['v'] = v
        raw['hcpvi'] = kw.get('hcpvi')
        raw['pressure'] = kw.get('pressure')
        return v

    eng.surrogate_model.recovery_model.calculate_recovery = spy
    r = eng.evaluate_scenario(res, eor, ops, EconomicParameters())
    swi = 0.25
    sor = 0.25
    cap = 1.0 - swi - sor
    print("model=%-10s raw_RF=%.6f  cap(rf_max_physical)=%.6f  reported=%.6f  CLIP_BINDS=%s"
          % (model, raw.get('v', float('nan')), cap, r.get('recovery_factor'),
             raw.get('v', 0) > cap + 1e-12))
    print("             hcpvi=%.6f  pressure=%.1f psi  npv=%.0f" %
          (raw.get('hcpvi', float('nan')), raw.get('pressure', float('nan')), r.get('npv', 0)))
    # correct cap
    print("             correct cap (1-swi-sor)/(1-swi) = %.6f" % ((1 - swi - sor) / (1 - swi)))
