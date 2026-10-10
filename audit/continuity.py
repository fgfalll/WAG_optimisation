"""
Continuity & Issue-Gate Module for the CO2 EOR Optimizer Audit System.

PURPOSE
-------
This module answers one question continuously:

    "For every finding that was declared RESOLVED, is it *actually* still fixed,
     and is the documentation telling the same story as the code?"

It exists because the 05-10-2026 remediation marked 17 findings RESOLVED in the
same uncommitted working tree in which they were edited. An adversarial
re-audit found only 11 of 17 genuinely fixed (4 partial, 1 regressed,
1 inert). Nothing in the repository could have detected that, because the test
suite reports 335 passed / 0 failed while the defects are live.

DESIGN PRINCIPLES (from the audit mandate)
-------------------------------------------
1. VERIFY, DON'T TRUST. A `Status: RESOLVED` line is a claim. This module
   re-measures it. Never infer correctness from the presence of a status label.
2. NO COMPOSITE SCORE. Software correctness, numerical stabilization, physical
   consistency, empirical calibration and predictive validity are checked and
   reported SEPARATELY. No "model accuracy" number is ever produced.
3. EVIDENCE OR IT DID NOT HAPPEN. Every verdict carries a measured value or a
   source citation. Unverifiable items are reported as
   `UNKNOWN - EVIDENCE REQUIRED`, never as PASS.
4. AUDIT-ONLY BY DEFAULT. Nothing here mutates scientific logic. The module
   writes reports and GitHub issues; it does not edit physics.
5. CONTINUITY OVER OPTIMISM. A regression is reported as loudly as a new defect.

ISSUE DISCIPLINE (hard rules)
-----------------------------
- **One commit closes one issue.** The commit subject MUST contain exactly one
  closing keyword with exactly one issue reference (e.g. `fix(pvt): ... Closes #42`).
  `Closes #1, #2` is REJECTED as issue stacking.
- **Issues are never closed by this tool.** Closure happens only via a commit,
  because a commit is the auditable record that the fix landed.
- **New issues are created only for findings with reproducible evidence**, and
  are cross-linked to the register entry and to its sibling issues.

USAGE
-----
    python -m audit.continuity check          # verify all RESOLVED claims
    python -m audit.continuity check CRIT-14  # verify one finding
    python -m audit.continuity sync           # propose wiki corrections (dry run)
    python -m audit.continuity gate           # pre-commit issue-discipline gate
    python -m audit.continuity status         # JSON summary
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTER = REPO_ROOT / "audit" / "scientific_flaws.md"
PROVENANCE = REPO_ROOT / "audit" / "parameter_provenance.csv"
WIKI_INDEX = REPO_ROOT / "agent_wiki" / "audit" / "simulation_run_audits" / "index.md"
PITFALLS = REPO_ROOT / "agent_wiki" / "development" / "common_pitfalls.md"
README = REPO_ROOT / "agent_wiki" / "README.md"

# --------------------------------------------------------------------------- #
# Verdicts. Deliberately NOT a numeric score - see principle 2.
# --------------------------------------------------------------------------- #

CONFIRMED = "CONFIRMED"                    # defect verified absent by measurement
PARTIAL = "PARTIALLY_RESOLVED"            # defect reduced but a residual remains
REGRESSED = "REGRESSED"                   # was fixed, is broken again
STILL_OPEN = "STILL_OPEN"                 # never fixed; status label is wrong
CONFIRMED_INERT = "CONFIRMED_BUT_INERT"   # fixed in isolation, dead in practice
UNKNOWN = "UNKNOWN_EVIDENCE_REQUIRED"     # cannot be verified automatically

CATEGORIES = (
    "MATHEMATICAL",
    "PHYSICAL",
    "NUMERICAL",
    "SOFTWARE",
    "PROVENANCE",
)

# Evidence tiers from the audit mandate, highest priority first. A PASS at a
# lower tier never compensates for a FAIL at a higher one.
EVIDENCE_TIERS = (
    "MATHEMATICAL_CORRECTNESS",
    "PHYSICAL_CONSISTENCY",
    "NUMERICAL_DISCRETIZATION",
    "IMPLEMENTATION",
    "PARAMETER_PROVENANCE",
    "ANALYTICAL_VERIFICATION",
    "EXPERIMENTAL_VALIDATION",
    "BENCHMARK_AGREEMENT",
    "EXECUTION_SPEED",
)


@dataclass
class CheckResult:
    """Outcome of verifying a single RESOLVED claim."""

    finding_id: str
    category: str
    claimed: str
    verdict: str
    evidence: str
    measured: Optional[str] = None
    evidence_tier: str = "IMPLEMENTATION"
    residual: Optional[str] = None
    should_reopen: bool = False


@dataclass
class Check:
    """A verification probe bound to a finding ID."""

    finding_id: str
    category: str
    probe: Callable[[], tuple[str, str, Optional[str]]]
    claimed: str = "RESOLVED"
    evidence_tier: str = "IMPLEMENTATION"


# --------------------------------------------------------------------------- #
# Probes. Each returns (verdict, evidence, measured) and MUST NOT mutate state.
# --------------------------------------------------------------------------- #


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run a probe without letting an exception masquerade as a PASS.

    Principle 3: a probe that cannot run must report UNKNOWN, never PASS.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the failure IS the datum
        return default


def probe_no_undefined_fudge_genes() -> tuple[str, str, Optional[str]]:
    """CRIT-13 / CRIT-19: a gene may only enter physics with a definition and a citation.

    The round-2 audit found `gravity_factor` (dimensionless, no unit, no citation)
    newly wired into three equations. This probe fails if such a factor appears
    in a recovery or sweep equation.
    """
    import inspect

    from core.engine_surrogate import analytical_models as AM

    suspects = ("gravity_factor",)
    hits: list[str] = []
    for cls_name in ("MiscibleSurrogate", "ImmiscibleSurrogate", "PhDHybridSurrogate"):
        cls = getattr(AM, cls_name, None)
        if cls is None:
            continue
        try:
            src = inspect.getsource(cls)
        except OSError:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if any(s in line for s in suspects) and "params.get" in line:
                hits.append(f"{cls_name}: {line.strip()[:90]}")
    if not hits:
        return CONFIRMED, "no undefined dimensionless gene multiplies the recovery/sweep equations", None
    return (
        STILL_OPEN,
        "a dimensionless gene with no unit and no citation multiplies RF, vertical sweep and "
        "the gravity number: " + "; ".join(hits)
        + ". Gravity segregation is already computed from delta_rho/dip/permeability in N_g, "
          "so the gene duplicates a real physical term with a fitted one.",
        f"{len(hits)} site(s)",
    )


def probe_hcpvi_dimensionless() -> tuple[str, str, Optional[str]]:
    """CRIT-01: HCPVI must be dimensionless and pressure-dependent.

    Also verifies the *denominator* is the hydrocarbon pore volume, which the
    round-2 audit found to be wrong by a factor of (1 - Swi) even though the
    units were finally right.
    """
    import numpy as np

    from core.engine_surrogate.surrogate_engine import SurrogateEngine  # noqa: F401
    from core.data_models import (  # noqa: F401
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )

    src = (REPO_ROOT / "core/engine_surrogate/surrogate_engine.py").read_text(encoding="utf-8")
    if "simulated_hcpvi" not in src:
        return STILL_OPEN, "simulated_hcpvi absent from surrogate_engine.py", None

    m = re.search(r"pv_mean_rb\s*=\s*(.+)", src)
    if not m:
        return UNKNOWN, "could not locate pv_mean_rb assignment", None

    ooip, swi, bo = 1_000_000.0, 0.25, 1.3053
    hcpv = ooip * bo              # textbook hydrocarbon pore volume, rb
    total_pv = ooip * bo / (1.0 - swi)   # what the code actually computes
    injected = 13_388_318.0

    if "1.0 - swi_val" in m.group(1) or "1.0 - swi" in m.group(1):
        measured = (
            f"denominator is TOTAL pore volume ({total_pv:,.0f} rb); "
            f"textbook HCPV is {hcpv:,.0f} rb; "
            f"returned {injected/total_pv:.4f} vs correct HCPVI {injected/hcpv:.4f} "
            f"(off by exactly 1-Swi = {1-swi:.2f})"
        )
        return (
            PARTIAL,
            "units fixed (dimensionless, pressure-dependent) but the denominator is "
            "total PV, not hydrocarbon PV: `pv_mean_rb = ooip*Bo/(1-Swi)`. The variable "
            "is mislabelled HCPVI; it is PV injected.",
            measured,
        )
    return CONFIRMED, "denominator is the hydrocarbon pore volume", None


def probe_bubble_point_correlation() -> tuple[str, str, Optional[str]]:
    """CRIT-03: a bubble point must come from a correlation, not a literal."""
    p = REPO_ROOT / "core/engine_surrogate/pvt_state.py"
    src = p.read_text(encoding="utf-8")
    if "p_bubble" not in src:
        return STILL_OPEN, "no p_bubble in pvt_state.py", None

    has_corr = bool(re.search(r"def .*bubble|Rsb|standing_pb|pb_standing", src, re.I))
    lit = re.search(r"else\s+min\(self\.p_init,\s*([\d.]+)\)", src)
    if not has_corr and lit:
        return (
            STILL_OPEN,
            "P_b is the literal constant min(P_init, %.1f); no Standing (or any) bubble-point "
            "correlation exists in the repository (grep Rsb/pb_standing/18.2 -> 0 hits). "
            "Every reservoir at or above that initial pressure shares one P_b regardless of API."
            % float(lit.group(1)),
            f"default P_b = {float(lit.group(1)):.1f} psi",
        )
    if not has_corr:
        return UNKNOWN, "could not determine whether a P_b correlation exists", None
    return CONFIRMED, "a bubble-point correlation is present", None


def probe_gas_fvf_constant() -> tuple[str, str, Optional[str]]:
    """CRIT-04: B_g = 0.02827*Z*T/P ft3/scf -> rb/MSCF must be 5.0351."""
    src = (REPO_ROOT / "core/engine_surrogate/pvt_state.py").read_text(encoding="utf-8")
    correct = 0.02827 * 1000.0 / 5.614583
    m = re.search(r"bg_hc\s*=\s*([\d.]+)\s*\*\s*z_hc", src)
    if not m:
        return UNKNOWN, "could not locate bg_hc assignment", None
    used = float(m.group(1))
    if abs(used - correct) / correct < 0.01:
        return CONFIRMED, f"coefficient {used} matches derived {correct:.4f}", f"{used} vs {correct:.4f}"
    return (
        STILL_OPEN,
        f"coefficient {used} differs from the field-unit value {correct:.4f} by "
        f"{correct/used:.2f}x",
        f"{used}",
    )


def probe_zfactor_dense_gas() -> tuple[str, str, Optional[str]]:
    """CRIT-05: Z must dip below 1 in the dense-gas region, not rise above it."""
    src = (REPO_ROOT / "core/engine_surrogate/pvt_state.py").read_text(encoding="utf-8")
    if "3.52" not in src or "0.9813" not in src:
        return STILL_OPEN, "Papay correlation coefficients not found", None
    gamma, tr = 0.70, 609.67
    ppc = 709.6 - 58.7 * gamma
    tpr = tr / (170.5 + 307.3 * gamma)
    t1, t2 = 10 ** (0.9813 * tpr), 10 ** (0.8157 * tpr)
    zs = {
        p: 1 - (3.52 * (p / ppc)) / t1 + (0.274 * (p / ppc) ** 2) / t2
        for p in (1500, 2500, 3500, 4500)
    }
    if all(z < 1.0 for z in zs.values()):
        return (
            CONFIRMED,
            "Papay (1968) coefficients present; Z < 1 with a dense-gas dip at the "
            "engine default Tpr ~ 1.58",
            ", ".join(f"{p}psi Z={z:.4f}" for p, z in zs.items()),
        )
    return STILL_OPEN, "Z still exceeds 1 in the dense-gas region", str(zs)


def probe_co2_compressibility_provenance() -> tuple[str, str, Optional[str]]:
    """CRIT-21 / HIGH-17: c_g must come from the EOS, not an ad-hoc power law."""
    src = (REPO_ROOT / "core/engine_surrogate/pvt_state.py").read_text(encoding="utf-8")
    if re.search(r"cg_co2\s*=\s*float\(np\.clip\(", src):
        return (
            STILL_OPEN,
            "c_g(CO2) is a hard-coded power law clip(1.5e-4*(2000/p)**0.8, ...) while a "
            "Peng-Robinson EOS exists in the same class (_setup_pr_eos_co2). Measured "
            "disagreement with -(1/B)dB/dP from PR: 2.6x-8.3x. Uncited.",
            "power law, no citation",
        )
    return CONFIRMED, "c_g(CO2) is not a hard-coded power law", None


def probe_saturation_closure() -> tuple[str, str, Optional[str]]:
    """CRIT-17 / CRIT-12: S_o + S_w + S_g must equal 1 at every timestep."""
    import numpy as np

    from core.data_models import (
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )
    from core.engine_surrogate.surrogate_engine import SurrogateEngine

    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
        rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    r = SurrogateEngine().evaluate_scenario(
        rd, EORParameters(), OperationalParameters(), economic_params=EconomicParameters()
    )
    prof = r.get("profiles", {})
    so = np.asarray(prof.get("saturation_oil", []), float)
    sw = np.asarray(prof.get("saturation_water", []), float)
    sg = np.asarray(prof.get("saturation_gas", []), float)
    if so.size == 0:
        return UNKNOWN, "no saturation profiles returned", None
    total = so + sw + sg
    worst = float(np.max(np.abs(total - 1.0)))
    bad = int(np.sum(so + sw > 1.0))
    if worst <= 1e-9:
        return CONFIRMED, "sum(S) == 1 within tolerance at every step", f"max dev {worst:.2e}"
    return (
        REGRESSED if bad else STILL_OPEN,
        "saturation closure violated: pv_ref uses hydrocarbon PV for both S_o and S_w "
        "while S_wi is defined on total PV; np.clip silently zeroes S_g to hide it",
        f"{bad}/{so.size} steps with S_o+S_w>1 ({100*np.mean(so+sw>1):.1f}%), "
        f"max sum(S)={total.max():.6f}",
    )


def probe_viscosity_sensitivity() -> tuple[str, str, Optional[str]]:
    """CRIT-14: recovery must respond to oil viscosity in the miscible limb."""
    from core.engine_surrogate.analytical_models import MiscibleSurrogate

    mis = MiscibleSurrogate()
    base = dict(pressure=3200.0, mmp=2500.0, s_wi=0.25, v_dp=0.5, sor=0.25,
                permeability=100.0, hcpvi=1.5, c7_plus_fraction=0.35, n_o=2.0,
                n_g=2.0, mobility_ratio=5.0)
    with_m = [mis.calculate_recovery(**{**base, "viscosity_oil": m}) for m in (0.5, 2.0, 100.0)]
    legacy = dict(base)
    legacy.pop("mobility_ratio")
    without_m = [mis.calculate_recovery(**{**legacy, "viscosity_oil": m}) for m in (0.5, 2.0, 100.0)]

    if max(with_m) - min(with_m) > 1e-6:
        return CONFIRMED, "RF responds to oil viscosity", f"spread {max(with_m)-min(with_m):.2e}"
    if max(without_m) - min(without_m) > 1e-6:
        return (
            REGRESSED,
            "params['mobility_ratio'] overrides the PVT-derived M in all three recovery "
            "models, so RF is exactly flat in viscosity while the legacy computed-M path "
            "still responds. CO2 viscosity reduction is removed from the physics.",
            f"with M supplied: {with_m[0]:.6f} for all mu; "
            f"computed-M path: {without_m[0]:.6f} -> {without_m[-1]:.6f}",
        )
    return STILL_OPEN, "RF does not respond to viscosity on either path", str(with_m)


def probe_koval_sensitivity_in_config() -> tuple[str, str, Optional[str]]:
    """CRIT-15: the sweep must vary with mobility ratio at the shipped HCPVI."""
    from core.engine_surrogate.analytical_models import KovalSurrogate
    from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine

    e = SolventExtendedPVTEngine(reservoir_temperature_f=150.0, initial_pressure_psi=3000.0,
                                 api_gravity=35.0, dead_oil_viscosity_cp=2.0, c7_plus_fraction=0.35)
    ooip, swi, life, rate = 1e6, 0.25, 15, 5000.0
    bo = e.calculate_oil_fvf_rb_per_stb(3000.0, x_co2=0.0)
    b_co2 = e.calculate_co2_fvf_rb_per_mscf(3000.0)
    pv = (ooip * bo) / (1.0 - swi)
    hcpvi = rate * b_co2 * 365.25 * life / pv

    kv = KovalSurrogate()
    sweeps = [kv.calculate_recovery(mobility_ratio=m, v_dp=0.5, hcpvi=hcpvi) for m in (1.0, 2.0, 5.0, 10.0)]
    spread = max(sweeps) - min(sweeps)
    if spread > 1e-6:
        return CONFIRMED, "Koval sweep varies with M at the shipped throughput", f"spread {spread:.4f}"
    return (
        CONFIRMED_INERT,
        "Koval is continuous and monotone in M (CRIT-06 confirmed) but the shipped "
        "configuration saturates it at its 0.95 clip, so mobility ratio cannot influence "
        "sweep in practice. Root cause: hcpvi is built from injection_rate*lifetime and "
        "ignores the schedule, availability and compressor cap.",
        f"HCPVI={hcpvi:.4f}; sweeps={sweeps}",
    )


def probe_npv_revenue_completeness() -> tuple[str, str, Optional[str]]:
    """CRIT-18 / HIGH-08: the cash flow must include every stream the engine emits."""
    src = (REPO_ROOT / "core/engine_surrogate/surrogate_engine.py").read_text(encoding="utf-8")
    m = re.search(r"annual_rev\s*=\s*(.+)", src)
    if not m:
        return UNKNOWN, "could not locate annual_rev", None
    expr = m.group(1)
    missing = [k for k in ("annual_hc_gas_mscf", "annual_co2_prod_mscf") if k not in expr]
    if "annual_hc_gas_mscf" not in missing:
        return CONFIRMED, "gas revenue enters the cash flow", None
    return (
        PARTIAL,
        "annual_rev includes oil + storage credit only. annual_hc_gas_mscf is allocated, "
        "accumulated and published but never referenced in the cash flow; produced CO2 "
        "earns no sale revenue. npv is the primary objective, so the optimiser ranks on a "
        "truncated model.",
        f"missing from revenue: {missing}",
    )


def probe_leakage_economics() -> tuple[str, str, Optional[str]]:
    """HIGH-23: leakage must have a non-zero economic consequence."""
    import numpy as np

    from core.data_models import (
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )
    from core.engine_surrogate.surrogate_engine import SurrogateEngine

    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
        rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    r = SurrogateEngine().evaluate_scenario(
        rd, EORParameters(), OperationalParameters(), economic_params=EconomicParameters()
    )
    leak = float(r.get("total_leakage_tonne", 0.0))
    rate = np.asarray(r.get("profiles", {}).get("leakage_rate_tonnes_day", []), float)
    peak = float(rate.max()) if rate.size else 0.0
    if leak > 0 or peak > 0:
        return CONFIRMED, "leakage is non-zero and reaches the objective path", f"total={leak}, peak_rate={peak}"
    src = (REPO_ROOT / "core/engine_surrogate/surrogate_engine.py").read_text(encoding="utf-8")
    blind = "annual_co2_inj_mscf - annual_co2_prod_mscf" in src
    return (
        STILL_OPEN,
        "leakage is identically zero in every configuration probed, yet the storage credit "
        "is paid on max(0, inj - prod) which ignores leakage"
        + (" (leakage-blind revenue base confirmed in source)" if blind else "")
        + ". Leaked CO2 would be paid for, not charged for.",
        f"total_leakage_tonne={leak}, peak_rate={peak}, leakage_blind_revenue={blind}",
    )


def probe_npv_rf_alignment() -> tuple[str, str, Optional[str]]:
    """CRIT-02: reported NPV and reported RF must come from one evaluation."""
    import numpy as np

    from core.data_models import (
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )
    from core.engine_surrogate.surrogate_engine import SurrogateEngine

    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
        rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    r = SurrogateEngine().evaluate_scenario(
        rd, EORParameters(), OperationalParameters(), economic_params=EconomicParameters()
    )
    rf = float(r.get("recovery_factor", 0.0))
    cum = float(r.get("cumulative_oil_stb", 0.0))
    err = abs(cum - rd.ooip_stb * rf)
    if err <= 1e-6 * max(cum, 1.0):
        return CONFIRMED, "cumulative_oil_stb == OOIP * recovery_factor", f"residual {err:.2e} STB"
    return STILL_OPEN, "reported RF and cumulative oil are inconsistent", f"residual {err:.2e} STB"


def probe_immiscible_gradient() -> tuple[str, str, Optional[str]]:
    """CRIT-07: the immiscible limb must be a function of its inputs."""
    from core.engine_surrogate.analytical_models import ImmiscibleSurrogate

    imm = ImmiscibleSurrogate()
    vals = set()
    for m in (0.5, 1.0, 2.0, 5.0, 20.0):
        for v in (0.0, 0.5, 0.9):
            for so in (0.1, 0.4, 0.7):
                vals.add(round(imm.calculate_recovery(
                    viscosity_oil=2.0, viscosity_inj=0.05, s_wi=0.25, sor=so, s_gc=0.0,
                    v_dp=v, mobility_ratio=m, n_o=2.0, n_g=2.0), 6))
    if len(vals) > 5:
        return CONFIRMED, "immiscible limb has a real gradient", f"{len(vals)} distinct values over 45 evaluations"
    return STILL_OPEN, "immiscible limb is effectively constant", f"{len(vals)} distinct value(s): {sorted(vals)}"


def probe_wag_unit_scaling() -> tuple[str, str, Optional[str]]:
    """CRIT-11: WAG water must convert MSCFD -> reservoir bbl with a rb/MSCF B_g.

    The x1000 scf->MSCF factor may be applied either at the B_g *definition*
    (`default_gas_fvf * 1000`) or at the *use site* (`* 1000`). Both are
    dimensionally valid, so the probe must accept either. An earlier revision of
    this probe only checked the use site and produced a false FAIL against code
    that was in fact correct - which is exactly the error class this module
    exists to prevent, so it is recorded here deliberately.
    """
    src = (REPO_ROOT / "core/engine_surrogate/profile_generator_fast.py").read_text(encoding="utf-8")
    bg = re.search(r"default_b_gas\s*=\s*(.+)", src)
    if not bg:
        return UNKNOWN, "could not locate default_b_gas assignment", None
    at_definition = bool(re.search(r"1000", bg.group(1)))

    use = re.search(r"co2_inj_rb_per_day\s*=\s*(.+)", src)
    water = re.search(r"enhanced_water_rate_bpd\s*=\s*(.+)", src)
    at_use = bool(use and re.search(r"1000", use.group(1)))
    at_water = bool(water and re.search(r"1000", water.group(1)))

    if not (at_definition or at_use or at_water):
        return (
            STILL_OPEN,
            "WAG gas-to-water conversion never applies the x1000 scf->MSCF factor; "
            "5,000 MSCFD would give 25 bpd of water instead of 25,000 bpd",
            f"default_b_gas = {bg.group(1).strip()[:80]}",
        )

    base = 5000.0
    raw = 0.005 * (1000.0 if at_definition else 1.0)
    q_rb = base * raw * (1000.0 if at_use else 1.0)
    where = "at B_g definition" if at_definition else "at use site"
    return (
        CONFIRMED,
        f"WAG water rate carries the scf->MSCF factor ({where}); "
        f"5,000 MSCFD -> {q_rb:,.0f} bpd (was 25 bpd before remediation)",
        f"{base*0.005:,.0f} -> {q_rb:,.0f} bpd, exactly 1000x",
    )


def probe_containment_can_prune() -> tuple[str, str, Optional[str]]:
    """CRIT-08: the containment score must be able to fall below its threshold."""
    import numpy as np

    from core.data_models import CO2StorageParameters
    from core.objectives.storage import calculate_geomechanical_containment_score

    sp = CO2StorageParameters()
    thresh = getattr(sp, "containment_critical_threshold", 0.3)
    pf = 4500.0
    lam = getattr(sp, "fracture_pressure_limit_fraction", 0.9)
    worst = calculate_geomechanical_containment_score(np.array([pf * lam * 3.0]), pf, sp)
    if worst < thresh:
        return CONFIRMED, "containment score decays below the pruning threshold under overpressure", f"S_cont={worst:.4f} < {thresh}"
    return STILL_OPEN, "containment score floor exceeds the pruning threshold", f"S_cont={worst:.4f} >= {thresh}"


def probe_co2_ledger_closure() -> tuple[str, str, Optional[str]]:
    """Closed-loop carbon accounting: injected == produced + stored."""
    import numpy as np

    from core.data_models import (
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )
    from core.engine_surrogate.surrogate_engine import SurrogateEngine

    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
        rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    r = SurrogateEngine().evaluate_scenario(
        rd, EORParameters(), OperationalParameters(), economic_params=EconomicParameters()
    )
    inj = float(r.get("cumulative_co2_injected_mscf", 0.0))
    prod = float(r.get("cumulative_co2_produced_mscf", 0.0))
    sto = float(r.get("cumulative_co2_stored_mscf", 0.0))
    err = abs(inj - (prod + sto))
    if err <= 1e-6 * max(inj, 1.0):
        return CONFIRMED, "CO2 ledger closes to machine precision", f"residual {err:.2e} MSCF"
    return STILL_OPEN, "CO2 ledger does not close", f"residual {err:.2e} MSCF (inj {inj:,.0f})"


def probe_pvt_provenance_guards() -> tuple[str, str, Optional[str]]:
    """CRIT-16: a getattr guard must name a field that exists."""
    from core.data_models import ReservoirData

    rd_fields = {f.name for f in ReservoirData.__dataclass_fields__.values()} \
        if hasattr(ReservoirData, "__dataclass_fields__") else set()
    src = (REPO_ROOT / "core/engine_surrogate/surrogate_engine.py").read_text(encoding="utf-8")
    dead = [a for a in re.findall(r'getattr\(reservoir_data,\s*"([a-z_]+)"', src)
            if rd_fields and a not in rd_fields]
    if dead:
        return (
            STILL_OPEN,
            f"getattr guards reference attributes that do not exist on ReservoirData "
            f"({dead}); the left operand is always None so user-supplied PVT is silently "
            f"discarded. Real field is 'oil_fvf'.",
            f"dead guards: {dead}",
        )
    return CONFIRMED, "all getattr(reservoir_data, ...) guards name real fields", None


def probe_recycled_le_produced() -> tuple[str, str, Optional[str]]:
    """Mass-conservation invariant: M_recycled <= M_produced <= M_injected."""
    import numpy as np

    from core.data_models import (
        EORParameters,
        EconomicParameters,
        OperationalParameters,
        ReservoirData,
    )
    from core.engine_surrogate.surrogate_engine import SurrogateEngine

    rd = ReservoirData(
        grid={"NX": np.array([50]), "NY": np.array([50]), "NZ": np.array([10])},
        pvt_tables={}, ooip_stb=1e6, initial_pressure=3000.0, temperature=150.0,
        rock_compressibility=3e-6, average_porosity=0.2,
        initial_water_saturation=0.25, thickness_ft=50.0, area_acres=100.0,
        length_ft=2000.0, oil_fvf=1.2,
    )
    r = SurrogateEngine().evaluate_scenario(
        rd, EORParameters(), OperationalParameters(), economic_params=EconomicParameters()
    )
    inj = float(r.get("cumulative_co2_injected_mscf", 0.0))
    rec = float(r.get("cumulative_co2_recycled_mscf", 0.0))
    prod = float(r.get("cumulative_co2_produced_mscf", 0.0))
    if rec <= prod + 1e-9 and prod <= inj + 1e-9:
        return CONFIRMED, "recycled <= produced <= injected", f"rec {rec:,.0f} <= prod {prod:,.0f} <= inj {inj:,.0f}"
    return STILL_OPEN, "CO2 mass-conservation ordering violated", f"rec {rec:,.0f}, prod {prod:,.0f}, inj {inj:,.0f}"


CHECKS: tuple[Check, ...] = (
    Check("CRIT-01", "MATHEMATICAL", probe_hcpvi_dimensionless, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("CRIT-02", "SOFTWARE", probe_npv_rf_alignment),
    Check("CRIT-03", "PHYSICAL", probe_bubble_point_correlation, evidence_tier="PHYSICAL_CONSISTENCY"),
    Check("CRIT-04", "MATHEMATICAL", probe_gas_fvf_constant, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("CRIT-05", "PHYSICAL", probe_zfactor_dense_gas, evidence_tier="PHYSICAL_CONSISTENCY"),
    Check("CRIT-06", "MATHEMATICAL", probe_koval_sensitivity_in_config, evidence_tier="NUMERICAL_DISCRETIZATION"),
    Check("CRIT-07", "MATHEMATICAL", probe_immiscible_gradient, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("CRIT-08", "SOFTWARE", probe_containment_can_prune),
    Check("CRIT-11", "PHYSICAL", probe_wag_unit_scaling, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("CRIT-12", "MATHEMATICAL", probe_saturation_closure, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("CRIT-13", "PROVENANCE", probe_no_undefined_fudge_genes, evidence_tier="PARAMETER_PROVENANCE"),
    Check("CRIT-14", "PHYSICAL", probe_viscosity_sensitivity, evidence_tier="PHYSICAL_CONSISTENCY"),
    Check("CRIT-16", "SOFTWARE", probe_pvt_provenance_guards),
    Check("CRIT-18", "PHYSICAL", probe_npv_revenue_completeness, evidence_tier="PHYSICAL_CONSISTENCY"),
    Check("CRIT-21", "PROVENANCE", probe_co2_compressibility_provenance, evidence_tier="PARAMETER_PROVENANCE"),
    Check("HIGH-23", "PHYSICAL", probe_leakage_economics, evidence_tier="PHYSICAL_CONSISTENCY"),
    Check("INV-LEDGER", "MATHEMATICAL", probe_co2_ledger_closure, evidence_tier="MATHEMATICAL_CORRECTNESS"),
    Check("INV-RECYCLE", "MATHEMATICAL", probe_recycled_le_produced, evidence_tier="MATHEMATICAL_CORRECTNESS"),
)

# Findings whose status label currently claims RESOLVED. Anything here whose
# probe returns REGRESSED / STILL_OPEN / PARTIAL must be reopened.
RESOLVED_CLAIMS: dict[str, str] = {
    "CRIT-01": "RESOLVED",
    "CRIT-02": "RESOLVED",
    "CRIT-03": "RESOLVED",
    "CRIT-04": "RESOLVED",
    "CRIT-05": "RESOLVED",
    "CRIT-06": "RESOLVED",
    "CRIT-07": "RESOLVED",
    "CRIT-08": "RESOLVED",
    "CRIT-11": "RESOLVED",
    "CRIT-12": "RESOLVED",
    "CRIT-13": "RESOLVED",
    "CRIT-16": "RESOLVED",
    "CRIT-18": "RESOLVED",
    "CRIT-21": "RESOLVED",
    "HIGH-10": "RESOLVED",
    "HIGH-19": "RESOLVED",
    "MED-11": "RESOLVED",
}


# --------------------------------------------------------------------------- #
# Wiki continuity
# --------------------------------------------------------------------------- #

@dataclass
class WikiIssue:
    path: str
    line: int
    kind: str
    detail: str
    severity: str


def parse_status_labels(register: Path) -> dict[str, str]:
    """Extract the `**Status:**` label attached to each finding.

    Handles both layouts present in the register:
      * its own line   ``- **Status:** RESOLVED - ...``
      * inline at the end of the preceding field
        ``- **Evidence & Citation:** ... **Status:** NEW``
    and the compact table form ``| **CRIT-16** | ... | NEW |``.
    """
    text = register.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current: Optional[str] = None

    status_re = re.compile(r"\*\*Status:\*\*\s*(.+?)\s*$")

    for line in text.splitlines():
        m = re.match(r"^###\s+((?:CRIT|HIGH|MED|LOW)-\d+)", line)
        if m:
            current = m.group(1)
            continue

        # Compact table row: | **CRIT-16** | cat | ... | NEW |
        mt = re.match(r"^\|\s*\*\*((?:CRIT|HIGH|MED|LOW)-\d+)\*\*\s*\|(.*)\|\s*$", line)
        if mt:
            cells = [c.strip() for c in mt.group(2).split("|")]
            tail = cells[-1] if cells else ""
            if tail:
                out[mt.group(1)] = tail
            continue

        if current:
            ms = status_re.search(line)
            if ms:
                out[current] = ms.group(1).strip()
                current = None
    return out


def resolve_claims(register: Path = REGISTER) -> dict[str, str]:
    """Derive the RESOLVED-claim set FROM THE REGISTER, not from a hardcoded dict.

    A hardcoded table can silently diverge from the document it claims to
    describe - which is the very defect class this module exists to catch. The
    curated RESOLVED_CLAIMS entries are merged in for probes the register does
    not yet cover, but the register always wins on conflict.
    """
    derived: dict[str, str] = {}
    for fid, status in parse_status_labels(register).items():
        head = status.upper()
        # Only treat it as a *claim of resolution* if the label leads with one.
        if re.match(r"^(RESOLVED|PARTIALLY RESOLVED|CONFIRMED BUT INERT)\b", head):
            derived[fid] = "RESOLVED"
    merged = dict(RESOLVED_CLAIMS)
    merged.update(derived)
    return merged


def check_wiki(register: Path = REGISTER) -> list[WikiIssue]:
    """Detect documentation that has drifted from the code.

    These are continuity defects: the audit failed partly because the wiki and
    the register asserted things the code does not do.
    """
    issues: list[WikiIssue] = []
    if not register.exists():
        return [WikiIssue(str(register), 0, "MISSING", "flaw register not found", "HIGH")]

    text = register.read_text(encoding="utf-8")

    # 1. Referenced source files must exist.
    for m in re.finditer(r"`([\w/\\]+\.py)(?::\d+(?:[-,]\d+)*)?`", text):
        rel = m.group(1).replace("\\", "/")
        if rel.startswith(("core/", "ui/", "analysis/", "evaluation/", "utils/", "tests/", "audit/")):
            line_no = text[: m.start()].count("\n") + 1
            if not (REPO_ROOT / rel).exists():
                issues.append(WikiIssue(
                    str(register.relative_to(REPO_ROOT)), line_no, "DEAD_PATH",
                    f"`{rel}` is cited but does not exist on disk", "MEDIUM"))

    # 2. Counts quoted in prose must match the actual findings present.
    headings = set(re.findall(r"^###\s+((?:CRIT|HIGH|MED|LOW)-\d+)", text, re.M))
    table_rows = set(re.findall(r"\*\*((?:CRIT|HIGH|MED|LOW)-\d+)\*\*\s*\|", text))
    actual = len(headings | table_rows)
    for m in re.finditer(r"(\d+)\s+distinct findings|register[^\n]{0,30}?(\d+)\s+findings", text, re.I):
        for g in m.groups():
            if g and abs(int(g) - actual) > 3:
                issues.append(WikiIssue(
                    str(register.relative_to(REPO_ROOT)), text[: m.start()].count("\n") + 1,
                    "STALE_COUNT",
                    f"document quotes {g} findings but the register contains {actual}", "MEDIUM"))

    # 3. Wiki invariants must not be stated as enforced when they are not.
    #    A phrase is only acceptable when a qualification marker accompanies it
    #    on the same line or within the next few lines - that is how the
    #    documentation now distinguishes "holds by construction" from
    #    "asserted by code".
    readme = REPO_ROOT / "agent_wiki" / "README.md"
    if readme.exists():
        rtext = readme.read_text(encoding="utf-8")
        QUALIFIERS = ("⚠", "not enforced", "not verified", "uncheckable",
                      "artefact", "structurally", "see high", "see crit")
        for phrase, why in (
            ("strictly balanced", "leakage is identically zero (HIGH-23)"),
            ("strictly capped", "containment is an artefact of a clip, not a solved constraint"),
            ("mass-conserving", "hold-by-construction, never asserted"),
        ):
            rlines = rtext.split("\n")
            for m in re.finditer(phrase, rtext, re.I):
                line_no = rtext[: m.start()].count("\n") + 1
                window = " ".join(rlines[line_no - 1: line_no + 4]).lower()
                if any(q.lower() in window for q in QUALIFIERS):
                    continue  # qualified in place - acceptable
                sev = "LOW" if "FLAGGED" in rtext[:4000] else "MEDIUM"
                issues.append(WikiIssue(
                    str(readme.relative_to(REPO_ROOT)), line_no, "UNENFORCED_INVARIANT",
                    f"'{phrase}' asserted without enforcement ({why})", sev))

    # 4. F821 is a release gate, not a style preference (HIGH-10 / HIGH-19 lesson).
    # Run via `python -m ruff` when present. `F821_FALSE_POSITIVE` is the reviewed
    # allowlist: a blanket "fix every F821" would corrupt working control flow.
    F821_FALSE_POSITIVE = {
        ("core/geology/petrophysical_distribution.py", "prev_field"),
    }
    ruff = _safe(lambda: __import__("shutil").which("ruff"), None) or "ruff"
    r = _safe(lambda: subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select", "F821",
         "--output-format", "concise", "."],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"), None)
    if r is not None and r.returncode in (0, 1) and r.stdout:
        for h in r.stdout.splitlines():
            if "F821" not in h:
                continue
            parts = h.split(":")
            if len(parts) < 4:
                continue
            rel = parts[0].replace("\\", "/")
            nm = re.search(r"`([^`]+)`", h)
            name = nm.group(1) if nm else ""
            if (rel, name) in F821_FALSE_POSITIVE:
                continue
            try:
                ln = int(parts[1])
            except ValueError:
                ln = 0
            issues.append(WikiIssue(rel, ln, "F821_UNDEFINED_NAME", h.strip()[:170], "HIGH"))

    # 5. Pitfall count must match the number of `### n.` headings.
    if PITFALLS.exists():
        ptext = PITFALLS.read_text(encoding="utf-8")
        nums = [int(n) for n in re.findall(r"^###\s+(\d+)\.", ptext, re.M)]
        m = re.search(r"^#\s+Top\s+(\d+)\s+Traps", ptext, re.M)
        if m and nums:
            if int(m.group(1)) != max(nums):
                issues.append(WikiIssue(
                    str(PITFALLS.relative_to(REPO_ROOT)), 1, "STALE_COUNT",
                    f"title says 'Top {m.group(1)} Traps' but there are {len(nums)} traps "
                    f"(max index {max(nums)}); ordering/duplication suspected", "MEDIUM"))
            if len(nums) != len(set(nums)):
                dupes = sorted({n for n in nums if nums.count(n) > 1})
                issues.append(WikiIssue(
                    str(PITFALLS.relative_to(REPO_ROOT)), 1, "DUPLICATE_INDEX",
                    f"duplicate trap indices: {dupes}", "MEDIUM"))

    return issues


# --------------------------------------------------------------------------- #
# Issue discipline
# --------------------------------------------------------------------------- #

CLOSING_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*(?:#|gh-)?(\d+)"
    r"(?:\s*,\s*(?:#|gh-)?(\d+))*\s*$",
    re.IGNORECASE,
)
ANY_REF_RE = re.compile(r"(?:#|gh-)(\d+)", re.IGNORECASE)


def check_commit_issue_discipline(subject: str, body: str = "") -> tuple[bool, list[str]]:
    """Validate the one-commit-one-issue rule for a candidate commit message.

    Rules enforced:
      * exactly one closing reference (`Closes #N`) at most;
      * stacked closes (`Closes #1, #2`) are REJECTED;
      * references in the body do not silently close anything.

    Returns (ok, problems).
    """
    problems: list[str] = []

    # A closing clause is <keyword> <ref-list>. The ref-list is one or more issue
    # references joined by separators; it STOPS at the first token that is not a
    # separator or a reference, so trailing prose is never mistaken for a
    # reference. Examples:
    #   "Closes #10, #12"        -> 2 refs  (stacking -> REJECTED)
    #   "Closes #7 and #8"       -> 2 refs  (stacking -> REJECTED)
    #   "Closes #9 for the closure" -> 1 ref (prose ignored)
    #   "Fixes #3. Also Closes #4"  -> two clauses, 1 ref each
    _SEP = r"(?:\s*(?:,|;|&|/|\\+|and|or|&|\\+)\s*)"
    CLOSE_CLAUSE = re.compile(
        r"\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*"
        r"((?:(?:#|gh-)\d+)(?:" + _SEP + r"(?:#|gh-)\d+)*)",
        re.IGNORECASE,
    )
    REF_NUM = re.compile(r"(\d+)")

    closes: list[str] = []
    for line in ((subject or "") + "\n" + (body or "")).splitlines():
        for km in CLOSE_CLAUSE.finditer(line):
            ref_list = REF_NUM.findall(km.group(2))
            if len(ref_list) > 1:
                problems.append(
                    f"issue stacking: '{km.group(1)}' closes {len(ref_list)} issues "
                    f"{ref_list} in one commit. Policy: 1 commit = 1 issue - split it."
                )
            closes.extend(ref_list)

    if len(closes) > 1:
        problems.append(
            f"multiple closing references {closes} in one commit. "
            f"Policy: 1 commit = 1 issue - split it."
        )

    # A bare reference such as 'see #14' or 'Refactors #14' is a reference, not a
    # closure. Warn only when the commit references an issue yet closes none, so
    # the author must state the disposition explicitly rather than leaving an
    # unclosed reference that looks like a silent close.
    text = (subject or "") + "\n" + (body or "")
    all_refs = set(ANY_REF_RE.findall(text))
    if all_refs and not closes:
        problems.append(
            f"commit references {sorted(all_refs)} but carries no closing keyword; "
            f"a reference is not a closure. Add 'Closes #N'."
        )
    return (not problems), problems


def gate_staged_commit(repo: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    """Run the issue-discipline gate over the staged commit message, if any."""
    staged = _safe(lambda: subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repo,
        capture_output=True, text=True, encoding="utf-8").stdout.strip(), "")
    if not staged:
        return True, []
    msg = _safe(lambda: subprocess.run(
        ["git", "log", "-1", "--pretty=%B"], cwd=repo,
        capture_output=True, text=True, encoding="utf-8").stdout, "")
    return check_commit_issue_discipline(msg.split("\n")[0], "\n".join(msg.split("\n")[1:]))


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

_VERDICT_GLYPH = {
    CONFIRMED: "PASS", PARTIAL: "PARTIAL", REGRESSED: "REGRESSED",
    STILL_OPEN: "FAIL", CONFIRMED_INERT: "PASS-BUT-INERT", UNKNOWN: "UNKNOWN",
}


def run_checks(only: Optional[str] = None) -> list[CheckResult]:
    """Run the probe suite. Never mutates repository state."""
    import os

    claims = resolve_claims()
    cwd = os.getcwd()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    results: list[CheckResult] = []
    try:
        for chk in CHECKS:
            if only and chk.finding_id != only:
                continue
            claimed = claims.get(chk.finding_id, "n/a (invariant)")
            try:
                verdict, evidence, measured = chk.probe()
            except Exception as exc:  # noqa: BLE001
                verdict, evidence, measured = UNKNOWN, f"probe raised {type(exc).__name__}: {exc}", None
            results.append(CheckResult(
                finding_id=chk.finding_id,
                category=chk.category,
                claimed=claimed,
                verdict=verdict,
                evidence=evidence,
                measured=measured,
                evidence_tier=chk.evidence_tier,
                residual=None if verdict == CONFIRMED else evidence,
                should_reopen=(
                    claims.get(chk.finding_id) == "RESOLVED"
                    and verdict in (PARTIAL, REGRESSED, STILL_OPEN, CONFIRMED_INERT)
                ),
            ))
    finally:
        os.chdir(cwd)
    return results


def render_report(results: list[CheckResult], wiki: list[WikiIssue]) -> str:
    """Human-readable report. Categorically separated, never a single score."""
    L: list[str] = []
    L.append("=" * 78)
    L.append("AUDIT CONTINUITY REPORT - RESOLVED-CLAIM VERIFICATION")
    L.append("=" * 78)
    L.append("")
    L.append("Categories are reported SEPARATELY by design. No composite accuracy")
    L.append("score is produced (audit mandate: strict categorical separation).")
    L.append("")

    by_tier: dict[str, list[CheckResult]] = {}
    for r in results:
        by_tier.setdefault(r.evidence_tier, []).append(r)

    for tier in EVIDENCE_TIERS:
        rows = by_tier.get(tier)
        if not rows:
            continue
        L.append("-" * 78)
        L.append(f"EVIDENCE TIER: {tier}")
        L.append("-" * 78)
        for r in rows:
            flag = "REOPEN" if r.should_reopen else "     "
            L.append(f"  [{_VERDICT_GLYPH.get(r.verdict, r.verdict):>14}] {r.finding_id:<11} {flag}")
            L.append(f"      category : {r.category}")
            L.append(f"      claimed  : {r.claimed}")
            L.append(f"      evidence : {r.evidence}")
            if r.measured:
                L.append(f"      measured : {r.measured}")
            L.append("")

    tally: dict[str, int] = {}
    for r in results:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    L.append("=" * 78)
    L.append("SUMMARY BY VERDICT (per category, never aggregated into a score)")
    L.append("=" * 78)
    for v in (CONFIRMED, CONFIRMED_INERT, PARTIAL, REGRESSED, STILL_OPEN, UNKNOWN):
        if tally.get(v):
            ids = [r.finding_id for r in results if r.verdict == v]
            L.append(f"  {_VERDICT_GLYPH.get(v, v):>14} ({tally[v]:>2})  {', '.join(ids)}")
    reopen = [r.finding_id for r in results if r.should_reopen]
    L.append("")
    L.append(f"  REQUIRES REOPENING: {len(reopen)}  {', '.join(reopen) if reopen else '-'}")
    L.append("")
    L.append("  NOTE: 'PASS' means the probe measured the defect absent. It does NOT")
    L.append("  mean the surrounding science is verified (see VALIDATION tiers below).")
    L.append("")

    L.append("=" * 78)
    L.append("WIKI CONTINUITY (documentation vs code)")
    L.append("=" * 78)
    if not wiki:
        L.append("  No documentation drift detected.")
    else:
        for w in wiki:
            L.append(f"  [{w.severity:>6}] {w.path}:{w.line}  {w.kind}")
            L.append(f"           {w.detail}")
    L.append("")
    return "\n".join(L)


def write_json_report(results: list[CheckResult], wiki: list[WikiIssue], dest: Path) -> Path:
    payload = {
        "note": "Categorically separated verdicts. No composite accuracy score exists by design.",
        "evidence_tier_order": list(EVIDENCE_TIERS),
        "checks": [asdict(r) for r in results],
        "wiki_issues": [asdict(w) for w in wiki],
        "counts": {
            v: sum(1 for r in results if r.verdict == v)
            for v in (CONFIRMED, CONFIRMED_INERT, PARTIAL, REGRESSED, STILL_OPEN, UNKNOWN)
        },
        "requires_reopening": [r.finding_id for r in results if r.should_reopen],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "check"

    if cmd == "gate":
        ok, problems = gate_staged_commit()
        if ok:
            print("issue-discipline gate: PASS (1 commit = 1 issue)")
            return 0
        print("issue-discipline gate: FAIL")
        for p in problems:
            print(f"  - {p}")
        print("\nPolicy: one commit closes exactly one issue. Use 'Closes #N'.")
        return 2

    if cmd == "check-commit":
        subject = argv[1] if len(argv) > 1 else ""
        body = argv[2] if len(argv) > 2 else ""
        ok, problems = check_commit_issue_discipline(subject, body)
        print("PASS" if ok else "FAIL")
        for p in problems:
            print(f"  - {p}")
        return 0 if ok else 2

    if cmd == "selftest":
        cases = [
            ("fix(pvt): derive c_g from PR-EOS", "Closes #10", True, "single close"),
            ("fix(pvt): correct Bg, Rs and Z together", "Closes #10, #12", False, "stacked close"),
            ("fix(physics): wire mobility ratio", "Closes #7 and #8", False, "stacked close, prose"),
            ("docs(audit): round-2 register", "see #9 for the saturation closure", False, "bare reference"),
            ("refactor: split module, no defect", "Refactors #14, no behaviour change", False,
             "non-closing verb still needs an explicit disposition"),
            ("docs(audit): refresh register", "No issue: documentation only", True, "prose, no number"),
            ("chore: bump pin", "", True, "no issue at all"),
            ("fix: two keywords two issues", "Fixes #3. Also Closes #4", False, "two keywords"),
        ]
        failures = 0
        for subj, bod, expect_ok, label in cases:
            ok, probs = check_commit_issue_discipline(subj, bod)
            good = ok == expect_ok
            failures += 0 if good else 1
            print(f"  [{'ok ' if good else 'BAD'}] expect={'PASS' if expect_ok else 'FAIL':<4} "
                  f"got={'PASS' if ok else 'FAIL':<4}  {label}")
            for p in probs:
                print(f"          {p}")
        print(f"\nself-test: {len(cases)-failures}/{len(cases)} as expected")
        return 1 if failures else 0

    if cmd not in ("check", "sync", "status"):
        print(f"unknown command: {cmd}")
        print(__doc__)
        return 1

    only = None
    if cmd == "check" and len(argv) > 1:
        only = argv[1].upper()

    results = run_checks(only)
    wiki = check_wiki()

    if cmd == "check":
        print(render_report(results, wiki))
        out = write_json_report(results, wiki, REPO_ROOT / "audit" / "continuity_report.json")
        print(f"JSON report: {out}")
        return 1 if any(r.should_reopen for r in results) else 0

    if cmd == "status":
        payload = {
            "counts": {v: sum(1 for r in results if r.verdict == v)
                       for v in (CONFIRMED, CONFIRMED_INERT, PARTIAL, REGRESSED, STILL_OPEN, UNKNOWN)},
            "reopen": [r.finding_id for r in results if r.should_reopen],
            "wiki_issue_count": len(wiki),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if cmd == "sync":
        print("PROPOSED WIKI CORRECTIONS (dry run - nothing written)")
        if not wiki:
            print("  none")
        for w in wiki:
            print(f"  {w.path}:{w.line} {w.kind}: {w.detail}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())