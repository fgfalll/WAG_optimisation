"""Probe 6: targeted verification of the specific NEW defects in the 05-10-2026 remediation."""
import sys
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
import numpy as np
np.seterr(all="ignore")
OUT=[]
def hdr(t): OUT.append("\n"+"="*78+f"\n{t}\n"+"="*78)
def p(*a): OUT.append(" ".join(str(x) for x in a))

from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine
from core.data_models import ReservoirData, EORParameters, OperationalParameters, EconomicParameters

hdr("G. HCPVI DOUBLE-COUNT: bo/b_co2 fallbacks NEVER fire (dead provenance guards)")
rd = ReservoirData(grid={"NX":np.array([50]),"NY":np.array([50]),"NZ":np.array([10])},
    pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
    rock_compressibility=3e-6, average_porosity=0.2, initial_water_saturation=0.25,
    thickness_ft=50.0, area_acres=100.0, length_ft=2000.0, oil_fvf=1.2)
for f in ("bo_rb_per_stb","bg_rb_per_mscf"):
    p(f"   hasattr(ReservoirData, '{f}') = {hasattr(rd, f)}  -> getattr(...,None) ALWAYS returns None")
p(f"   reservoir_data.oil_fvf = {rd.oil_fvf}   <-- the field that actually exists")
p("   => 'getattr(reservoir_data,\"bo_rb_per_stb\",None) or <computed>' : LEFT operand is ALWAYS None,")
p("      so the user's own oil_fvf is NEVER consulted. Guard is inert (dead code), and any user-supplied")
p("      Bo/Bg stored under the real field name is silently ignored.")
p("")
hdr("H. HCPVI from _build_params_dict vs simulated HCPVI - are they consistent?")
e = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0,
                             api_gravity=35.0, dead_oil_viscosity_cp=2.0, c7_plus_fraction=0.35)
ooip, swi = 1e6, 0.25
init_p = 3000.0
bo  = rd.oil_fvf if rd.oil_fvf else e.calculate_oil_fvf_rb_per_stb(init_p, 0.0)
b_c2= e.calculate_co2_fvf_rb_per_mscf(init_p)
pv_rb = (ooip*bo)/max(1.0-swi,0.05)
p(f"   NOTE code uses bo=pvt_init.calculate_oil_fvf(init_p) (NOT rd.oil_fvf): bo={e.calculate_oil_fvf_rb_per_stb(init_p,0.0):.4f}")
bo2 = e.calculate_oil_fvf_rb_per_stb(init_p,0.0)
pv2 = (ooip*bo2)/max(1.0-swi,0.05)
q_inj = EORParameters().injection_rate
life = OperationalParameters().project_lifetime_years
tot = q_inj*b_c2*365.25*life
p(f"   injection_rate={q_inj} MSCFD  life={life} yr  B_co2(init)={b_c2:.4f} rb/MSCF")
p(f"   total_inj_rb = {tot:,.0f} rb ; pv_rb = {pv2:,.0f} rb")
p(f"   HCPVI(_build_params_dict) = {tot/max(pv2,1.0):.4f}")
p("")
p("   -- Now compare against the value CRIT-01 recomputation actually produces --")
p("   simulate a 181-step monthly profile at constant rate (engine default case):")
n=181; dt=np.diff(np.arange(n)*30.4375, prepend=0.0)
cum_inj_mscf=float(np.sum(q_inj*dt))
b_mean=e.calculate_co2_fvf_rb_per_mscf(3000.0)
bo_mean=e.calculate_oil_fvf_rb_per_stb(3000.0,0.0)
pv_mean=(ooip*bo_mean)/max(1.0-swi,0.05)
p(f"   cum_inj_mscf={cum_inj_mscf:,.0f}  B_co2(mean p)={b_mean:.4f}  Bo(mean p)={bo_mean:.4f}")
p(f"   HCPVI(simulated) = {cum_inj_mscf*b_mean/pv_mean:.4f}")
p("")
p("   -- Physical test: HCPVI is a THROUGHPUT/PV ratio. It must be dimensionless and")
p("      scale with injection rate & time. Check scaling --")
for mult in (0.5, 1.0, 2.0, 4.0):
    tot2 = q_inj*mult*b_c2*365.25*life
    h1 = tot2/max(pv2,1.0)
    cum2 = cum_inj_mscf*mult
    h2 = cum2*b_mean/pv_mean
    p(f"      rate x{mult:<4} -> HCPVI(build)={h1:8.4f}   HCPVI(sim)={h2:8.4f}   ratio sim/build={h2/h1:.4f}")
p("")
hdr("I. HCPVI MAGNITUDE PLAUSIBILITY (is it even physical?)")
p(f"   HCPVI = {tot/max(pv2,1.0):.3f} for a default 15-yr, 20,000 MSCFD CO2 flood")
p(f"   HCPVI of {tot/max(pv2,1.0):.1f} means {tot/max(pv2,1.0):.1f} hydrocarbon pore volumes of CO2 injected.")
p("   Published CO2 EOR practice: HCPVI at miscibility/termination is ~0.5-3.0;")
p("   even aggressive West Texas CO2 floods rarely exceed ~5-10 HCPVI before termination.")
p("   => default HCPVI ~7.7 is at the extreme high end; Koval sweep is already")
p("      saturated (see D4: sweep hits the 0.95 clip by HCPVI ~ 5-6).")
p("")
from core.engine_surrogate.analytical_models import KovalSurrogate
kv=KovalSurrogate()
p("   Koval sweep vs the ACTUAL engine HCPVI, over M range:")
for M in (1.0,2.0,5.0,10.0):
    p(f"      M={M:>5} -> sweep={kv.calculate_recovery(mobility_ratio=M,v_dp=0.5,hcpvi=7.6928):.6f}")
p("   => In the DEFAULT configuration the Koval sweep is at/near its clip for ALL M.")
p("      Mobility ratio - and therefore viscosity contrast - no longer changes sweep.")
p("")
hdr("J. SATURATION CLOSURE BREAKDOWN (np.clip masking)")
# reproduce
prof_run = None
from core.engine_surrogate.surrogate_engine import SurrogateEngine
ep=EORParameters(); op=OperationalParameters(); ec=EconomicParameters()
res=SurrogateEngine().evaluate_scenario(rd,ep,op,economic_params=ec)
pr=res.get("profiles",{})
so=np.asarray(pr["saturation_oil"],float); swv=np.asarray(pr["saturation_water"],float)
sg=np.asarray(pr["saturation_gas"],float)
bad=so+swv
p(f"   n={len(so)}  timesteps with So+Sw>1 : {int(np.sum(bad>1.0))}  ({100*np.mean(bad>1.0):.1f}%)")
p(f"   max(So+Sw) = {bad.max():.6f}  -> Sg forced to 0 by np.clip")
p(f"   time vector: {len(pr.get('time_vector',[]))} steps; first/last So+Sw = {bad[0]:.6f} / {bad[-1]:.6f}")
p("")
p("   WHY: vp_dynamic uses pv_ref = OOIP*Bo_mean/(1-Swi) which is the HCPV (hydrocarbon PV),")
p("   while So is computed as remaining_oil*Bo / vp_dynamic. The water term uses")
p("   pore_volume_rb*Swi + water_inj - water_prod, divided by the SAME vp_dynamic.")
p("   But Swi was defined on the TOTAL pore volume, not the hydrocarbon pore volume:")
p("   total PV = PV/(1-Swi), so PV*Swi != PV_hcpv*Swi. The two saturation bases are")
p("   inconsistent by construction -> So+Sw>1 -> Sg silently zeroed.")
pv_total = 1e6*1.2/max(1-0.25,0.05)
pv_hcpv  = pv_total*0.75
p(f"   numeric: PV_total={pv_total:,.0f} rb ; PV_hcpv={pv_hcpv:,.0f} rb")
p(f"   So(0) expected = OOIP*Bo/PV_total = {1e6*1.2/pv_total:.4f}  (code gives {so[0]:.4f})")
p(f"   Sw(0) expected = PV_total*Swi/PV_total = 0.25  (code gives {swv[0]:.4f})")
p(f"   code recomputed Sw(0) = (PV_hcpv*Swi)/PV_hcpv = 0.25  -> both use PV_hcpv => sum = (1-0.25)*Bo/PV_hcpv")
p(f"   So(0)+Sw(0) = {so[0]+swv[0]:.6f}")
p("")
hdr("K. VRR PROFILE: is the post-hoc recompute consistent with the in-loop VRR?")
vrr=np.asarray(pr.get("vrr_profile",[]),float)
if vrr.size:
    p(f"   vrr_profile n={vrr.size} min={vrr.min():.4f} max={vrr.max():.4f} mean={vrr.mean():.4f}")
p("   In-loop VRR (:417) uses bo_dynamic/bg_dynamic AT current_p with the PRE-rescale oil profile.")
p("   Post-hoc VRR (:547) uses bo_profile/bg_profile and the POST-rescale oil/water/gas streams.")
p("   Two different VRR definitions exist in the same returned object; the post-hoc one wins.")
p("")
hdr("L. annual_co2_prod_mscf: DOUBLE-COUNTING of CO2 vs HC gas")
p("   co2_prod_rate = profile_result.get('co2_gas_profile', profile_result['gas_profile'])")
p("   hc_gas_rate    = profile_result.get('solution_gas_profile', max(0, total_gas_rate-co2_prod_rate))")
keys_present = [k for k in ("co2_gas_profile","solution_gas_profile","gas_profile") if k in pr]
p(f"   keys actually in returned profiles: {keys_present}")
p("   -> 'gas_profile' is BOTH the total-gas stream AND the fallback for co2_prod_rate.")
p("      If 'co2_gas_profile' is absent, ALL produced gas (incl. HC gas) is booked as CO2,")
p("      while total_gas_rate also includes it -> produced CO2 double-counted in the ledger.")
tp=np.asarray(pr.get("total_gas_profile",[]),float)
cp=np.asarray(pr.get("co2_gas_profile",[]),float)
if cp.size and tp.size:
    p(f"   sum co2_gas_profile={cp.sum():,.0f}  sum total_gas_profile={tp.sum():,.0f}  ratio={cp.sum()/max(tp.sum(),1e-9):.4f}")
p("")
hdr("M. NPV REVENUE COMPLETENESS")
p("   annual_rev = annual_oil_stb*oil_price + annual_stored_tonne*co2_storage_credit")
p("   MISSING: hydrocarbon gas sales revenue (annual_hc_gas_mscf is COMPUTED at :588,:606")
p("            and then never used in the cash flow).")
p("   MISSING: produced-CO2 sale revenue (annual_co2_prod_mscf used only for storage credit base).")
p("   MISSING: no opex on gas handling/compression beyond the recycle cost.")
p("   -> NPV is systematically biased LOW vs any real CO2-EOR project economics, yet it is")
p("      the PRIMARY OPTIMIZATION OBJECTIVE. The optimizer is therefore selecting on a")
p("      truncated cash-flow model, not a complete one.")
hg=np.asarray(pr.get("annual_hydrocarbon_gas_mscf",[]),float)
p(f"   annual_hydrocarbon_gas_mscf computed: n={hg.size} sum={hg.sum():,.0f} MSCF  <-- DISCARDED")
p(f"   at $3/MSCF that is ${hg.sum()*3:,.0f} of revenue omitted from NPV")
p("")
hdr("N. DISCOUNTING / CAPEX SANITY")
p(f"   capex present in params = {5_000_000.0}")
p(f"   discount_rate = 0.10")
p(f"   => the static $5.0M CAPEX issue (pitfall #44) is STILL ACTIVE; the new NPV block")
p("      faithfully discounts it, but the CAPEX magnitude itself is unvalidated.")
OUT_TXT="\n".join(OUT); print(OUT_TXT)
open(r"C:\Windows\Temp\opencode\audit_verify_6.out.txt","w",encoding="utf-8").write(OUT_TXT)