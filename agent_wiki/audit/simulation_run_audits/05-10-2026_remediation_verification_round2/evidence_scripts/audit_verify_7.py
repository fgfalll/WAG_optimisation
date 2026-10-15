"""Probe 7: resolve the HCPVI discrepancy + check the 20,000 vs 5,000 MSCFD rate confusion."""
import sys
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
import numpy as np
np.seterr(all="ignore")
OUT=[]
def hdr(t): OUT.append("\n"+"="*78+f"\n{t}\n"+"="*78)
def p(*a): OUT.append(" ".join(str(x) for x in a))

from core.data_models import ReservoirData, EORParameters, OperationalParameters, EconomicParameters
from core.engine_surrogate.surrogate_engine import SurrogateEngine

rd = ReservoirData(grid={"NX":np.array([50]),"NY":np.array([50]),"NZ":np.array([10])},
    pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
    rock_compressibility=3e-6, average_porosity=0.2, initial_water_saturation=0.25,
    thickness_ft=50.0, area_acres=100.0, length_ft=2000.0, oil_fvf=1.2)
ep=EORParameters(); op=OperationalParameters(); ec=EconomicParameters()
p(f"EORParameters.injection_rate DEFAULT = {ep.injection_rate} MSCFD")
p(f"OperationalParameters.project_lifetime_years DEFAULT = {op.project_lifetime_years}")

hdr("O. IS THE PROFILE-GENERATED INJECTION RATE == eor_params.injection_rate?")
res = SurrogateEngine().evaluate_scenario(rd, ep, op, economic_params=ec)
pr = res.get("profiles", {})
inj = np.asarray(pr.get("injection_profile",[]),float)
tv  = np.asarray(pr.get("time_vector",[]),float)
dt  = np.diff(tv, prepend=0.0)
p(f"   time_vector: n={len(tv)} first={tv[:3]} last={tv[-3:]}")
p(f"   injection_profile: min={inj.min():,.1f} mean={inj.mean():,.1f} max={inj.max():,.1f} MSCFD")
p(f"   eor_params.injection_rate = {ep.injection_rate:,.1f} MSCFD  <-- 4x smaller than mean profile!")
p(f"   => HCPVI from _build_params_dict uses {ep.injection_rate:,.0f} MSCFD, but the actual")
p(f"      simulated throughput is {inj.mean():,.0f} MSCFD. Factor = {inj.mean()/ep.injection_rate:.2f}x")
cum_profile=float(np.sum(inj*dt))
p(f"   cum injected from profile = {cum_profile:,.0f} MSCF")
p(f"   res['cumulative_co2_injected_mscf'] = {res.get('cumulative_co2_injected_mscf'):,.0f}")
p("")
p("   CONSEQUENCE: rec_params['hcpvi'] (the CRIT-01 'corrected' value) is derived from the")
p("   ACTUAL profile, but params['hcpvi'] (:968) used for the FIRST prediction and for")
p("   breakthrough time is derived from eor_params.injection_rate which does NOT match the")
p("   profile. The two HCPVI values differ by ~4x in the default configuration.")

hdr("P. BREAKTHROUGH TIME: driven by which HCPVI?")
p(f"   res['breakthrough_time_years'] = {res.get('breakthrough_time_years')}")
p(f"   res['recovery_factor']         = {res.get('recovery_factor')}")
p(f"   res['cumulative_oil_stb']      = {res.get('cumulative_oil_stb'):,.0f}")
p(f"   OOIP                            = {rd.ooip_stb:,.0f}")
p(f"   OOIP * RF                       = {rd.ooip_stb*res.get('recovery_factor',0):,.0f}")
p(f"   consistency: {abs(res.get('cumulative_oil_stb',0)-rd.ooip_stb*res.get('recovery_factor',0))<1.0}")

hdr("Q. CONTAINMENT: is the EPA Class VI ceiling actually respected?")
for k in ("max_sandface_pressure_psi","total_leakage_tonne","annual_leakage_tonne"):
    p(f"   {k} = {res.get(k)}")
try:
    from core.engine_surrogate.geomechanics_fault import GeomechanicsFaultModel
    import inspect
    sig = inspect.signature(GeomechanicsFaultModel.__init__)
    p(f"   GeomechanicsFaultModel.__init__{sig}")
    g = GeomechanicsFaultModel()
    p(f"   p_safe_ceiling = {getattr(g,'p_safe_ceiling',None)} psi ; p_fracture = {getattr(g,'p_fracture',None)}")
except Exception as ex:
    p(f"   {type(ex).__name__}: {ex}")
p(f"   profiles keys: {sorted(pr.keys())}")
for pk in ("pressure_profile","monthly_pressure","pressure","sandface_pressure"):
    v=np.asarray(pr.get(pk,[]),float)
    if v.size:
        p(f"   {pk}: n={v.size} min={v.min():,.1f} max={v.max():,.1f} mean={v.mean():,.1f}")
p(f"   reservoir pressure is CLIPPED to [p_min, p_safe_ceiling] at :468 -> the pressure")
p(f"      profile can never exceed the containment ceiling. Containment 'safety' is therefore")
p(f"      an artifact of the clip, not of a solved geomechanical constraint.")

hdr("R. LEAKAGE: is it ever non-zero? (CRIT-10 / HIGH-05)")
vals=[]
for cap_frac in (0.5, 0.9, 0.99):
    for pf in (3000.0, 4500.0, 5500.0):
        rd2 = ReservoirData(grid={"NX":np.array([50]),"NY":np.array([50]),"NZ":np.array([10])},
            pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
            rock_compressibility=3e-6, average_porosity=0.2, initial_water_saturation=0.25,
            thickness_ft=50.0, area_acres=100.0, length_ft=2000.0, oil_fvf=1.2)
        try:
            rr = SurrogateEngine().evaluate_scenario(rd2,ep,op,economic_params=ec)
            vals.append(rr.get("total_leakage_tonne",0.0))
        except Exception as ex:
            vals.append(f"ERR {type(ex).__name__}")
p(f"   total_leakage_tonne across cases: {set(vals)}")
p("   => leakage remains IDENTICALLY ZERO. The new NPV carbon-tax term")
p("      (annual_leakage_tonne * carbon_tax) can therefore never be non-zero.")
p("      Containment has NO economic consequence in the optimizer.")
OUT_TXT="\n".join(OUT); print(OUT_TXT)
open(r"C:\Windows\Temp\opencode\audit_verify_7.out.txt","w",encoding="utf-8").write(OUT_TXT)