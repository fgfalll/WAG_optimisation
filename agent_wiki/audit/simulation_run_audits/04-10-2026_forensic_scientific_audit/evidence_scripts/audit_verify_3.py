"""Audit-only numeric verification round 3. No repository file is modified."""
import sys
import warnings
import numpy as np

sys.path.insert(0, r"D:\rep\4.6\co2eor_optimizer")

from core.engine_surrogate.analytical_models import (
    ImmiscibleSurrogate, HybridSurrogate, MiscibleSurrogate, KovalSurrogate,
)
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine
from core.objectives.storage import calculate_geomechanical_containment_score


class _Adv:  # defaults mirroring AdvancedEngineParams / storage.py getattr fallbacks
    containment_pressure_weight = 0.5
    containment_seal_weight = 0.3
    containment_structure_weight = 0.2
    containment_safety_margin = 1.0
    fracture_pressure_limit_fraction = 0.9
    containment_critical_threshold = 0.3
    # NOTE: AdvancedEngineParams has NO structural_trapping_factor / seal factor fields
    # -> storage.py getattr falls back to 0.85 / 0.9


print("=== V1. HCPVI: code expression vs textbook definition ===")
eng = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0)
ooip, swi, cum = 1.0e6, 0.25, 1.0e6  # 1 MMSTB, 1 MMSCF injected
for P in (1500.0, 2500.0, 3500.0):
    bg_model = eng.calculate_co2_fvf_rb_per_mscf(P)
    bg_text = 5.035 * 0.85 * 609.67 / P          # textbook Bg [RB/MSCF], Z=0.85
    bo = eng.calculate_oil_fvf_rb_per_stb(P, 0.0)
    sim = (cum * bg_model) / (ooip * bg_model / (1.0 - swi))          # surrogate_engine.py:497
    corr = cum * bg_text * (1.0 - swi) / (ooip * bo)                  # textbook HCPVI
    print(f"  P={P:>5.0f}: sim={sim:.4f} (units MSCF/STB)  Bo={bo:.4f}  Bg_model={bg_model:.5f}"
          f"  Bg_text={bg_text:.4f}  correct_hcpvi={corr:.4f}  sim/correct={sim/corr:.3f}"
          f"  Bo/Bg_text={bo/bg_text:.3f}")

print("\n=== V2. Bg law: pvt_state 0.1587*Z*T_R/P vs 5.035*Z*T_R/P ===")
for P in (1500.0, 2500.0, 3500.0):
    bg_model = eng.calculate_co2_fvf_rb_per_mscf(P)
    bg_text = 5.035 * 0.85 * 609.67 / P
    print(f"  P={P:>5.0f}: model={bg_model:.5f}  textbook={bg_text:.4f}  textbook/model={bg_text/bg_model:.2f}x")

print("\n=== V2b. mixture bg vs y_co2 at 2500 psi ===")
for y in (0.0, 0.5, 1.0):
    g = eng.calculate_mixture_gas_properties(2500.0, y)
    bg_t = 5.035 * (0.85 * (1 - y) + 0.68 * y) * 609.67 / 2500.0
    print(f"  y_co2={y:<4} bg_model={g['bg_rb_per_mscf']:.5f}  bg_text~={bg_t:.4f}"
          f"  ratio={bg_t/g['bg_rb_per_mscf']:.1f}x  cg={g['compressibility_psi_inv']:.3e}")

print("\n=== V3. Z factor behaviour of the 'Hall-Yarborough' expression ===")
for P in (1500.0, 2500.0, 3500.0, 4500.0):
    T_R = 609.67
    Ppr = P / 1070.0   # Pc = 1070 psia for CO2? model uses 7.376e6 Pa = 1070 psia
    g = eng.calculate_mixture_gas_properties(P, 0.5)
    print(f"  P={P:>5.0f} Ppr={Ppr:.3f} -> bg_model={g['bg_rb_per_mscf']:.5f}")

print("\n=== V4. Apparent oil compressibility c_o = (1/Bo)dBo/dP ===")
for x in (0.0, 0.3):
    prev = None
    out = []
    for P in (1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0):
        bo = eng.calculate_oil_fvf_rb_per_stb(P, x)
        if prev is not None:
            c = (bo - prev[1]) / (prev[1] * (P - prev[0]))
            out.append(f"P={P:.0f}:Bo={bo:.4f},c_o={c:+.3e}1/psi")
        prev = (P, bo)
    print(f"  x_co2={x}: " + " | ".join(out))

print("\n=== V5. KovalSurrogate RF vs mobility ratio (expect 0 for M<=1.4) ===")
ko = KovalSurrogate()
with warnings.catch_warnings():
    warnings.simplefilter("always")
    for M in (0.5, 0.9, 1.0, 1.1, 1.3, 1.4, 1.5, 2.0, 3.0, 5.0):
        rf = ko.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                                   hcpvi=1.8, s_wi=0.25, sor=0.25)
        print(f"  M={M:<5} RF={rf:.6f}")

print("\n=== V6. ImmiscibleSurrogate sensitivity ===")
im = ImmiscibleSurrogate()
vals = set()
for M in (0.5, 1.0, 2.0, 5.0, 20.0):
    for v in (0.0, 0.5, 0.9):
        for sor in (0.1, 0.4, 0.7):
            vals.add(round(im.calculate_recovery(mobility_ratio=M, v_dp=v, pressure=3000.0,
                                                 mmp=2500.0, sor=sor, s_wi=0.25), 6))
print(f"  distinct RF values over 5x3x3 grid = {sorted(vals)}")

print("\n=== V7. HybridSurrogate RF vs M (mobility_ratio inert?) ===")
hy = HybridSurrogate()
for M in (0.98, 1.0, 1.01, 2.0, 3.0, 5.0):
    rf = hy.calculate_recovery(mobility_ratio=M, v_dp=0.5, pressure=3000.0, mmp=2500.0,
                               hcpvi=1.8, sor=0.25, soi=0.75, s_wi=0.25)
    print(f"  M={M:<5} RF={rf:.6f}")

print("\n=== V8. Craig areal sweep E_A = (1 + M^-0.5)/2 (analytical_models.py:307-312) ===")
for M in (1.0, 1.01, 1.1, 1.5, 2.0, 5.0):
    ea = (1.0 + M ** -0.5) / 2.0
    print(f"  M={M:<5} E_A={ea:.6f}")

print("\n=== V9. rf_max_physical normalisation ===")
swi, sor = 0.25, 0.30
code = 1.0 - swi - sor
ref = (1.0 - swi - sor) / (1.0 - swi)
print(f"  code(1-swi-sor)={code:.4f}  reference((1-swi-sor)/(1-swi))={ref:.4f}  under-estimate={(1-code/ref)*100:.1f}%")

print("\n=== V10. Koval H coefficient: 10^(v/(1-v)) vs 1/(1-v)^2 ===")
for v in (0.2, 0.5, 0.8, 0.9, 0.99):
    try:
        h1 = 10.0 ** (v / (1.0 - v))
    except OverflowError:
        h1 = float("inf")
    h2 = 1.0 / (1.0 - v) ** 2
    print(f"  v_dp={v:<5} H_engine={h1:.4g}  H_Koval1963={h2:.4g}  ratio={h1/h2:.4g}")

print("\n=== V11. Containment score floor with production defaults ===")
frac = 2500.0 * 1.5   # max_pressure_psi * fracture_pressure_multiplier
for label, prof in (("yearly (non-empty, P=frac) -> s_press=0", np.full(15, frac)),
                    ("weekly/quarterly (missing key -> empty)", np.array([]))):
    s = calculate_geomechanical_containment_score(prof, frac, _Adv())
    print(f"  {label}: S_cont={s:.4f}  threshold={_Adv.containment_critical_threshold}"
          f"  prune_possible={s < _Adv.containment_critical_threshold}")

print("\n=== V12. gas_trapping direction (surrogate_models.py:238) ===")
for s_gc in (0.05, 0.20, 0.40, 0.60):
    print(f"  s_gc={s_gc:<5} gas_trapped_fraction=1-s_gc={1.0 - s_gc:.3f}  (should increase with s_gc)")
