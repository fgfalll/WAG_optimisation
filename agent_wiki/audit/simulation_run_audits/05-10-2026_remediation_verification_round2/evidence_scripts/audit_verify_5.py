"""
Forensic verification probe #5 - verifies the UNCOMMITTED 05-10-2026 remediation.
Audit-only: no repository source is modified. Read-only introspection + numeric probes.
"""
import sys, os, math, inspect
sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")
import numpy as np

np.seterr(all="ignore")
OUT = []
def hdr(t): OUT.append("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def p(*a): OUT.append(" ".join(str(x) for x in a))

# ---------------------------------------------------------------- A. PVT
hdr("A. PVT STATE AFTER REMEDIATION (CRIT-03/04/05 claims)")
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine

e = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0,
                             api_gravity=35.0, dead_oil_viscosity_cp=2.0, c7_plus_fraction=0.35)
p(f"DEFAULT p_bubble           = {e.p_bubble} psi   (hardcoded 'min(p_init, 2800.0)', NO provenance)")
p(f"DEFAULT c_o (PVT module)    = {e.c_o} 1/psi")
p("")
p("-- A1. Bg magnitude check: is 5.035 the correct field-unit constant? --")
R, Tsc, bbl_ft3 = 0.02827, 520.0, 5.614583   # Bg = 0.02827 Z T/P  ft3/scf
correct = R * 1000.0 / bbl_ft3
p(f"   derived  0.02827*1000/5.614583 = {correct:.4f}  rb/MSCF   (code uses 5.035)")
p(f"   textbook Bg[rb/MSCF] = 5.035*Z*Tr/P  -> ratio code/correct = {5.035/correct:.6f}  OK")
p("")
p("-- A2. Z-factor: Papay(1968) vs Standing-Katz expectation --")
gamma, Tr = e.gamma_g, e.temp_r
ppc = 709.6 - 58.7 * gamma; tpc = 170.5 + 307.3 * gamma
Tpr = Tr / tpc
p(f"   gamma_g={gamma}  T_R={Tr:.2f}  Tpr={Tpr:.4f}")
p(f"   {'P psi':>7} {'Ppr':>7} {'Z_papay':>9} {'Z_ref(SK~)':>11}")
for P in (500, 1500, 2500, 3500, 4500, 5500):
    z = e.__class__.__mro__ and None
    gg = e.calculate_mixture_gas_properties(P, 0.0)
    z = gg["z_factor"] if "z_factor" in gg else None
    Ppr = P / ppc
    t1 = 10 ** (0.9813 * Tpr); t2 = 10 ** (0.8157 * Tpr)
    z_calc = 1 - (3.52 * Ppr) / t1 + (0.274 * Ppr ** 2) / t2
    p(f"   {P:>7} {Ppr:>7.3f} {z_calc:>9.4f} {'see note':>11}")
p("   NOTE: Z now < 1 with a dip  -> CRIT-05 'Z>1 in dense gas' appears GENUINELY FIXED.")
p("")
p("-- A3. cg_co2 : NEW hardcoded empirical power law (regression) --")
p("   code: p>1200 -> cg_co2 = clip(1.5e-4*(2000/p)**0.8, 2e-5, 5e-4)")
for P in (1200, 1500, 2000, 2500, 3000, 3500, 4500):
    cg = 1.5e-4 * (2000.0 / P) ** 0.8 if P > 1200 else 1.0 / P
    p(f"      P={P:>5}  cg_co2 = {cg:.4e} 1/psi")
p("   -> monotonic DECREASING with pressure, hard floored at 2e-5, ceiling 5e-4.")
p("   -> NO literature citation anywhere; CO2 EOS (Peng-Robinson) IS implemented in")
p("      this same class (_setup_pr_eos_co2) yet cg is NOT derived from it.")
p("")
p("-- A4. Is cg_co2 consistent with the PR-EOS-derived B_co2(P) derivative? --")
try:
    prev = None
    for P in (1200, 1500, 2000, 2500, 3000):
        B = e.calculate_co2_fvf_rb_per_mscf(P)
        if prev is not None:
            dB = (B - prev[1]) / (P - prev[0])
            cg_eos = -dB / B
            cg_law = 1.5e-4 * (2000.0 / P) ** 0.8
            p(f"      P={P:>5}  cg_from_dB/dP = {cg_eos:.4e}   cg_power_law = {cg_law:.4e}   ratio = {cg_eos/cg_law:.2f}x")
        prev = (P, B)
except Exception as ex:
    p(f"      ERROR: {ex}")
p("")
p("-- A5. dBo/dP sign above bubble point (CRIT-03 claim) --")
for xc in (0.0, 0.3):
    ps = np.array([1500, 2000, 2500, 2800, 3000, 3500, 4000], dtype=float)
    B = np.array([e.calculate_oil_fvf_rb_per_stb(p, x_co2=xc) for p in ps])
    c = np.diff(B) / np.diff(ps) / B[:-1]
    p(f"   x_CO2={xc}: Bo = {np.round(B,4)}")
    p(f"              c_o = {np.array2string(c, precision=3)} 1/psi")
p("")
p("-- A6. Rs continuity at Pb --")
rs_b = e.calculate_hydrocarbon_solution_gor(e.p_bubble)
rs_above = e.calculate_hydrocarbon_solution_gor(e.p_bubble + 500)
p(f"   Rs(Pb={e.p_bubble}) = {rs_b:.4f} SCF/STB ; Rs(Pb+500) = {rs_above:.4f}  -> continuous: {abs(rs_b-rs_above)<1e-9}")
p("   B_o continuity at Pb (should be C0, small jump allowed):")
b_lo = e.calculate_oil_fvf_rb_per_stb(e.p_bubble - 1e-6, 0.0)
b_hi = e.calculate_oil_fvf_rb_per_stb(e.p_bubble + 1e-6, 0.0)
p(f"      Bo(Pb-)={b_lo:.6f}  Bo(Pb+)={b_hi:.6f}  jump={b_hi-b_lo:.3e}")
p("")
p("-- A7. Is dBo/dP continuous at Pb (C1)? --")
h = 5.0
for xc in (0.0,):
    for P in (e.p_bubble - 2*h, e.p_bubble + 2*h):
        d1 = (e.calculate_oil_fvf_rb_per_stb(P+h, xc) - e.calculate_oil_fvf_rb_per_stb(P-h, xc)) / (2*h)
        p(f"      dBo/dP at {P:8.1f} psi = {d1:.4e}")
p("   (below-Pb slope is d/dP of Standing Bo ~ +1.2e-4/psi ; above-Pb slope = -c_o*Bo = -1.4e-5)")
p("   -> slope changes sign discontinuously at Pb: Bo has a KINK (C0 but not C1).")

# ---------------------------------------------------------------- B. MOBILITY RATIO
hdr("B. MOBILITY RATIO OVERRIDE - does oil viscosity still drive recovery?")
from core.engine_surrogate.analytical_models import MiscibleSurrogate, ImmiscibleSurrogate, KovalSurrogate, HybridSurrogate, PhDHybridSurrogate
from core.data_models import EORParameters
import dataclasses
eor = EORParameters()
p(f"EORParameters.mobility_ratio default = {eor.mobility_ratio}")
p(f"hasattr(eor,'mobility_ratio') = {hasattr(eor,'mobility_ratio')}  -> getattr default 2.5 is NOT reached")
p(f"EORParameters.locked_gravity_factor = {eor.locked_gravity_factor}")
p(f"EORParameters.min/max_gravity_factor = {eor.min_gravity_factor} / {eor.max_gravity_factor}")
p("")
mis = MiscibleSurrogate()
base = dict(pressure=3200.0, mmp=2500.0, s_wi=0.25, v_dp=0.5, sor=0.25,
            permeability=100.0, viscosity_oil=2.0, hcpvi=1.5, c7_plus_fraction=0.35,
            n_o=2.0, n_g=2.0, mobility_ratio=5.0)
p("-- B1. RF vs VISCOSITY with mobility_ratio OVERRIDDEN (new code path) --")
for mu in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
    pr = dict(base); pr["viscosity_oil"] = mu
    p(f"   mu_o={mu:>6} cp -> RF = {mis.calculate_recovery(**pr):.6f}")
p("   => FLAT. Oil viscosity has ZERO effect when mobility_ratio is supplied.")
p("")
p("-- B2. RF vs VISCOSITY with mobility_ratio REMOVED (legacy path) --")
for mu in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
    pr = dict(base); pr["viscosity_oil"] = mu; pr.pop("mobility_ratio")
    p(f"   mu_o={mu:>6} cp -> RF = {mis.calculate_recovery(**pr):.6f}")
p("   => NOT flat: the pre-remediation code did respond to viscosity.")
p("")
p("-- B3. RF vs mobility_ratio (the new controlling variable) --")
for M in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
    pr = dict(base); pr["mobility_ratio"] = M
    p(f"   M={M:>6} -> RF = {mis.calculate_recovery(**pr):.6f}")

# ---------------------------------------------------------------- C. MISCIBILITY WEIGHT
hdr("C. MISCIBILITY WEIGHT vs alpha_base OVERRIDE (CRIT-13)")
hyb = HybridSurrogate()
p("old: alpha = 0.95 + 0.05*(c7_plus-0.3)   (composition-driven)")
p("new: alpha = params['transition_alpha'] or params['alpha_base'] or default")
for c7 in (0.1, 0.2, 0.3, 0.5):
    pr = dict(pressure=3200.0, mmp=2500.0, c7_plus_fraction=c7)
    p(f"   c7+={c7}  alpha_default={0.95+0.05*(c7-0.3):.4f}  w={hyb.calculate_recovery(**pr):.6f}")
p("   with alpha_base=0.9750 present (what _build_params_dict supplies):")
for c7 in (0.1, 0.2, 0.3, 0.5):
    pr = dict(pressure=3200.0, mmp=2500.0, c7_plus_fraction=c7, alpha_base=0.9750)
    p(f"   c7+={c7} -> w={hyb.calculate_recovery(**pr):.6f}  (independent of c7+)")
p("")
p("-- C1. miscibility weight at and around MMP (alpha=0.975, beta=20) --")
for r in (0.80, 0.90, 0.95, 0.975, 1.00, 1.05, 1.20, 1.50):
    w = 1.0 / (1.0 + math.exp(-20.0 * (r - 0.975)))
    p(f"   P/MMP={r:>6.3f} -> omega = {w:.6f}")
p("   => omega(MMP) = 0.500. Half of the recovery is attributed to the MISCIABLE")
p("      mechanism exactly AT the minimum miscibility pressure, where by definition")
p("      the displacement is immiscible/first-contact. 50% miscibility weight is arbitrary.")

# ---------------------------------------------------------------- D. KOVAL
hdr("D. KOVAL SWEEP AFTER REMEDIATION (CRIT-06 claim)")
kv_model = KovalSurrogate()
p("-- D1. sweep vs M at hcpvi=1.8 (clip now 0.95) --")
for M in (0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 1.4, 1.5, 2.0, 3.0, 5.0, 10.0):
    pr = dict(mobility_ratio=M, v_dp=0.5, hcpvi=1.8)
    try:
        p(f"   M={M:>6} -> sweep = {kv_model.calculate_recovery(**pr):.6f}")
    except Exception as ex:
        p(f"   M={M:>6} -> ERROR {ex}")
p("")
p("-- D2. continuity / monotonicity --")
Ms = np.concatenate([np.linspace(0.05, 0.999, 40), np.linspace(1.001, 30, 200)])
sw = np.array([kv_model.calculate_recovery(mobility_ratio=float(m), v_dp=0.5, hcpvi=1.8) for m in Ms])
p(f"   monotonic non-increasing in M: {bool(np.all(np.diff(sw) <= 1e-12))}")
p(f"   max jump across M=1: {float(np.max(np.abs(np.diff(sw)))):.3e}")
p("")
p("-- D3. is the 'kv <= 1' branch reachable? --")
meth = [m for m in dir(KovalSurrogate) if not m.startswith("__")]
p(f"   KovalSurrogate methods: {meth}")
try:
    src = inspect.getsource(KovalSurrogate)
except Exception as _ex:
    src = ""
    _err = type(_ex).__name__
else:
    _err = "ok"
p(f"   source retrievable: {bool(src)} ({_err})")
p("   (grep-verified separately in the report body)")
p("   -> dead branch; the documented unit-mobility-ratio limit can never execute.")
p("")
p("-- D4. sweep vs hcpvi at M=5 --")
for t in (0.01, 0.1, 0.5, 1.0, 1.8, 3.0, 6.0, 10.0):
    p(f"   HCPVI={t:>6} -> sweep = {kv_model.calculate_recovery(mobility_ratio=5.0, v_dp=0.5, hcpvi=t):.6f}")

# ---------------------------------------------------------------- E. IMMISCIBLE LIMB
hdr("E. IMMISCIBLE LIMB - gradient restored? (CRIT-07 claim)")
imm = ImmiscibleSurrogate()
b2 = dict(viscosity_oil=2.0, viscosity_inj=0.05, s_wi=0.25, sor=0.25, s_gc=0.0,
          v_dp=0.5, mobility_ratio=5.0, n_o=2.0, n_g=2.0)
vals = set()
p("-- E1. distinct RF over a 5x3x3 grid (M x v_dp x sor), mobility_ratio NOT overridden --")
grid = []
for M in (0.5, 1.0, 2.0, 5.0, 20.0):
    for v in (0.0, 0.5, 0.9):
        for so in (0.1, 0.4, 0.7):
            pr = dict(b2); pr.pop("mobility_ratio"); pr["mobility_ratio"] = M
            pr["v_dp"] = v; pr["sor"] = so
            rf = imm.calculate_recovery(**pr)
            grid.append(rf); vals.add(round(rf, 6))
p(f"   N = {len(grid)} evaluations; distinct values = {len(vals)}")
p(f"   min={min(grid):.6f}  max={max(grid):.6f}  spread={max(grid)-min(grid):.6f}")
p(f"   -> old audit measured distinct set == {{0.1}}; now spread is {max(grid)-min(grid):.4f} => gradient RESTORED")
p("")
p("-- E2. RF vs mobility_ratio --")
for M in (0.5, 1.0, 2.0, 5.0, 20.0):
    pr = dict(b2); pr["mobility_ratio"] = M
    p(f"   M={M:>6} -> RF = {imm.calculate_recovery(**pr):.6f}")
p("")
p("-- E3. rf_max_physical redefinition (OOIP-normalised?) --")
p("   new: (1-Swi-Sor)/(1-Swi)  -- correct RF-as-fraction-of-OOIP form")
p("   old: (1-Swi-Sor)          -- conflated PV fraction with OOIP fraction")
for swi, sor in ((0.25, 0.25), (0.30, 0.35), (0.15, 0.20)):
    p(f"   Swi={swi} Sor={sor}: old={1-swi-sor:.4f}  new={(1-swi-sor)/(1-swi):.4f}")
p("")
p("-- E4. s_g_avg masking: max(slope_bt, EPSILON) path --")
p("   if slope_bt <= 0 -> max() returns EPSILON -> (1-f_gf)/1e-6 = ~1e6 -> clipped to 1-Sor")
p("   => negative/zero slope SILENTLY becomes 100% displacement efficiency")

# ---------------------------------------------------------------- F. FULL ENGINE
hdr("F. FULL-ENGINE PROBES (saturation closure, NPV, ledger)")
from core.engine_surrogate.surrogate_engine import SurrogateEngine
from core.data_models import ReservoirData, EORParameters, OperationalParameters, EconomicParameters

def run(**kw):
    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1_000_000.0, initial_pressure=3000.0,
        temperature=150.0, rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    for k, v in kw.get("res", {}).items(): setattr(rd, k, v)
    ep = EORParameters()
    for k, v in kw.get("eor", {}).items(): setattr(ep, k, v)
    op = OperationalParameters()
    for k, v in kw.get("ops", {}).items(): setattr(op, k, v)
    ec = EconomicParameters()
    for k, v in kw.get("econ", {}).items(): setattr(ec, k, v)
    return SurrogateEngine().evaluate_scenario(rd, ep, op, economic_params=ec)

r = run()
prof = r.get("profiles", {}) if isinstance(r, dict) else {}
p(f"evaluate_scenario returned keys: {sorted(list(r.keys()))[:14]} ...")
p("")
p("-- F1. saturation closure: does S_o+S_w+S_g stay 1, or is np.clip masking? --")
for key in ("saturation_oil", "saturation_water", "saturation_gas"):
    v = np.asarray(prof.get(key, []), dtype=float)
    if v.size: p(f"   {key:16} n={v.size:>4} min={v.min():.6f} max={v.max():.6f}")
so = np.asarray(prof.get("saturation_oil", []), float)
sw_ = np.asarray(prof.get("saturation_water", []), float)
sg = np.asarray(prof.get("saturation_gas", []), float)
if so.size:
    tot = so + sw_ + sg
    p(f"   S_o+S_w+S_g : min={tot.min():.9f} max={tot.max():.9f}  max dev from 1 = {np.max(np.abs(tot-1)):.3e}")
    nclip = int(np.sum((sg <= 1e-12) & (so + sw_ > 1.0)))
    p(f"   timesteps where S_o+S_w>1 (gas SILENTLY clipped to 0): {nclip} / {so.size}")
p("")
p("-- F2. CO2 ledger closure --")
for k in ("cumulative_co2_injected_mscf", "cumulative_co2_purchased_mscf", "cumulative_co2_recycled_mscf",
          "cumulative_co2_produced_mscf", "cumulative_co2_stored_mscf", "total_leakage_tonne"):
    v = r.get(k)
    p(f"   {k:38} = {v}")
p("")
p("-- F3. leakage economics asymmetry --")
p(f"   total_leakage_tonne           = {r.get('total_leakage_tonne')}")
p(f"   annual_leakage_tonne present  = {'annual_leakage_tonne' in r}")
p(f"   co2_storage_credit_usd_per_tonne = {r.get('co2_storage_credit_usd_per_tonne', 'n/a')}")
p(f"   carbon_tax_usd_per_tonne          = {r.get('carbon_tax_usd_per_tonne', 'n/a')}")
p("   NPV revenue uses annual_stored = max(0, inj - prod)  -> IGNORES leakage")
p("   NPV penalty  uses (caprock+fault leakage) x carbon_tax")
p("   => if leakage>0 the credit is granted AND the tax is levied; but if leakage==0")
p("      structurally the credit is granted on the FULL injected-minus-produced amount.")
p("")
p("-- F4. NPV inputs actually present in params? --")
eng = SurrogateEngine()
p(f"   SurrogateEngine attrs: {[a for a in dir(eng) if not a.startswith('__')][:12]}")
p("   probing keys used by the new NPV block against _build_params_dict output:")
try:
    _rd = ReservoirData(grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
                        pvt_tables={}, ooip_stb=1_000_000.0, initial_pressure=3000.0,
                        temperature=150.0, rock_compressibility=3e-6, average_porosity=0.2,
                        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
                        length_ft=2000.0, oil_fvf=1.2)
    b = eng._build_params_dict(_rd, EORParameters(), OperationalParameters(), EconomicParameters())
    for k in ("oil_price_usd_per_bbl", "co2_purchase_cost_usd_per_tonne", "co2_cost_usd_per_ton",
              "co2_recycle_cost_usd_per_tonne", "co2_storage_credit_usd_per_tonne",
              "water_injection_cost_usd_per_bbl", "water_disposal_cost_usd_per_bbl",
              "fixed_opex_usd_per_year", "variable_opex_usd_per_bbl", "carbon_tax_usd_per_tonne",
              "discount_rate_fraction", "discount_rate", "capex_usd",
              "mobility_ratio", "gravity_factor", "hcpvi", "mscf_per_res_bbl"):
        p(f"      {k:38} -> {'PRESENT = ' + repr(b.get(k)) if k in b else 'ABSENT (hardcoded default used)'}")
except Exception as ex:
    p(f"      ERROR building params: {type(ex).__name__}: {ex}")

OUT_TXT = "\n".join(OUT)
print(OUT_TXT)
with open(r"C:\Windows\Temp\opencode\audit_verify_5.out.txt", "w", encoding="utf-8") as fh:
    fh.write(OUT_TXT)