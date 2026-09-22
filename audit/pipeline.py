"""
CO2 EOR Optimizer - Master Scientific & Code Audit Pipeline.

Executes comprehensive AST, physical, mathematical, and numerical scans
across all modules and exports reports to audit/ subdirectories.
"""

import ast
import csv
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "audit"

EXCLUDE_DIRS = {
    ".venv", ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".hypothesis", ".benchmarks", "build", "dist"
}


def get_python_files() -> List[Path]:
    """Retrieve all active Python files excluding virtualenvs and cache."""
    py_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                py_files.append(Path(root) / f)
    return sorted(py_files)


def run_code_quality_scan(py_files: List[Path]) -> Dict[str, Any]:
    """Scan modules for LOC, classes, functions, and complexity."""
    results = []
    total_lines = 0
    total_classes = 0
    total_functions = 0

    for p in py_files:
        rel = p.relative_to(REPO_ROOT).as_posix()
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            lines = len(content.splitlines())
            total_lines += lines
            tree = ast.parse(content, filename=str(p))
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            total_classes += len(classes)
            total_functions += len(functions)

            results.append({
                "module": rel,
                "lines": lines,
                "classes_count": len(classes),
                "functions_count": len(functions),
                "classes": classes,
                "functions": functions,
            })
        except Exception as e:
            results.append({
                "module": rel,
                "lines": 0,
                "classes_count": 0,
                "functions_count": 0,
                "error": str(e),
            })

    out_dir = AUDIT_DIR / "code_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "modules.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "modules.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Module", "Lines", "Classes", "Functions"])
        for r in results:
            writer.writerow([r["module"], r.get("lines", 0), r.get("classes_count", 0), r.get("functions_count", 0)])

    return {
        "total_files": len(py_files),
        "total_lines": total_lines,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "details": results,
    }


def run_fallbacks_scan(py_files: List[Path]) -> List[Dict[str, Any]]:
    """Scan for try-except blocks, silent fallbacks, and bare exceptions."""
    fallbacks = []

    for p in py_files:
        rel = p.relative_to(REPO_ROOT).as_posix()
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for h in node.handlers:
                        exc_name = "bare_except"
                        if h.type:
                            if isinstance(h.type, ast.Name):
                                exc_name = h.type.id
                            elif isinstance(h.type, ast.Attribute):
                                exc_name = f"{getattr(h.type.value, 'id', '')}.{h.type.attr}"
                            else:
                                exc_name = ast.unparse(h.type)
                        fallbacks.append({
                            "module": rel,
                            "line": h.lineno,
                            "exception_type": exc_name,
                            "body_statements": len(h.body),
                        })
        except Exception:
            continue

    out_dir = AUDIT_DIR / "fallbacks"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fallbacks.json", "w", encoding="utf-8") as f:
        json.dump(fallbacks, f, indent=2)

    with open(out_dir / "fallbacks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Module", "Line", "Exception Type", "Body Length"])
        for fb in fallbacks:
            writer.writerow([fb["module"], fb["line"], fb["exception_type"], fb["body_statements"]])

    return fallbacks


def run_hardcoded_values_scan(py_files: List[Path]) -> List[Dict[str, Any]]:
    """Scan physics and surrogate modules for hardcoded numerical literals."""
    physics_targets = [p for p in py_files if any(k in p.as_posix() for k in ["core/engine_surrogate", "core/unified_engine", "evaluation", "analysis"])]
    hardcoded = []

    for p in physics_targets:
        rel = p.relative_to(REPO_ROOT).as_posix()
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, start=1):
                # Search for numerical floats in scientific calculations
                matches = re.findall(r"(?<![a-zA-Z0-9_])([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)(?![a-zA-Z0-9_])", line)
                if matches and not line.strip().startswith("#"):
                    for m in matches:
                        val = float(m)
                        if val not in [0.0, 1.0, 2.0, 0.5]:  # Filter trivial constants
                            hardcoded.append({
                                "module": rel,
                                "line": idx,
                                "value": val,
                                "snippet": line.strip()[:100],
                            })
        except Exception:
            continue

    out_dir = AUDIT_DIR / "hardcoded_values"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hardcoded_values.json", "w", encoding="utf-8") as f:
        json.dump(hardcoded, f, indent=2)

    with open(out_dir / "hardcoded_values.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Module", "Line", "Literal", "Code Snippet"])
        for h in hardcoded:
            writer.writerow([h["module"], h["line"], h["value"], h["snippet"]])

    return hardcoded


def get_scientific_flaw_register() -> List[Dict[str, Any]]:
    """Master register of all confirmed scientific, mathematical, and numerical flaws."""
    return [
        {
            "ID": "SCI-FLAW-01",
            "Severity": "CRITICAL",
            "Category": "Physical / Fractional Flow",
            "Physical Phenomenon": "Buckley-Leverett / Koval Solvent Fractional Flow",
            "Mathematical Issue": "Inverted mobility ratio dependency: effective viscosity ratio divided by M, and (M-1)*0.5 in denominator",
            "Code Location": "core/engine_surrogate/profile_generator_fast.py:944-950",
            "Observed Behavior": "Favorable piston displacement (M=1.0) yields 70% gas breakthrough, while severe fingering (M=10.0) yields only 10% gas.",
            "Expected Behavior": "Higher mobility ratio causes earlier breakthrough and higher gas fractional flow (F_CO2 -> 1.0).",
            "Evidence": "frac_flow_co2 = koval_factor / (koval_factor + (mobility_ratio - 1) * 0.5) with koval_factor = E / M.",
            "Equation / Reference": "Koval, E.J. (1963). SPE Journal, 3(2), 145-152; Buckley & Leverett (1942).",
            "Scientific Consequence": "Inverts gas breakthrough physics, misdirects WAG ratio optimization, and distorts carbon recycling.",
            "Numerical Consequence": "Artificial plateau in GOR; GA/BO selection pressure inverted for heavy vs light oils.",
            "Reproducibility Consequence": "Optimization trajectories diverge completely from physical reality.",
            "Recommended Investigation": "Replace with authentic Koval fractional flow: K = H_k * E, F_s = K*S / (1 + S*(K - 1)).",
            "Verification Test": "tests/scientific/co2/test_co2_breakthrough_physics.py::test_koval_fractional_flow_mobility_inversion",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-02",
            "Severity": "CRITICAL",
            "Category": "Physical / Thermodynamics",
            "Physical Phenomenon": "Isothermal Fluid Compressibility",
            "Mathematical Issue": "Positive pressure coefficient: dBo/dP > 0 above bubble point",
            "Code Location": "core/data_integration_engine.py:370, 456",
            "Observed Behavior": "Bo(P) = 1.2 + 0.0001 * (P - 4000). Oil expands as reservoir pressure increases.",
            "Expected Behavior": "Thermodynamics requires c_o = -(1/Bo)(dBo/dP) > 0, so dBo/dP must be negative above bubble point.",
            "Evidence": "Direct linear expression in data_integration_engine.py:370.",
            "Equation / Reference": "McCain, W.D. (1990). The Properties of Petroleum Fluids.",
            "Scientific Consequence": "Violates Second Law of Thermodynamics (negative compressibility).",
            "Numerical Consequence": "Voidage increases with pressure, causing artificial pressure runaways in material balance.",
            "Reproducibility Consequence": "PVT tables cannot be matched to experimental PVT laboratory reports.",
            "Recommended Investigation": "Invert sign: Bo(P) = Bo_b * exp(-co * (P - Pb)).",
            "Verification Test": "tests/scientific/physics/test_thermodynamic_consistency.py::test_oil_compressibility_positivity",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-03",
            "Severity": "HIGH",
            "Category": "Physical / Fluid Dynamics",
            "Physical Phenomenon": "Viscosity-Pressure Dependence",
            "Mathematical Issue": "Negative exponential argument: exp(-0.0003 * (P - 4000))",
            "Code Location": "core/data_integration_engine.py:372, 375, 459, 465",
            "Observed Behavior": "Oil and supercritical CO2 viscosities decrease exponentially with pressure.",
            "Expected Behavior": "Undersaturated liquid and supercritical fluid viscosities increase with pressure (dmu/dP > 0).",
            "Evidence": "Exponential formulas in data_integration_engine.py:372, 375.",
            "Equation / Reference": "Pedersen, K.S. et al. (2014). Phase Behavior of Petroleum Reservoir Fluids.",
            "Scientific Consequence": "Over-pressuring reservoir creates artificial, unphysical mobility improvements.",
            "Numerical Consequence": "Economic objective functions incentivize operating at near-fracture pressures.",
            "Reproducibility Consequence": "Simulations under-predict viscous resistance at deep reservoir pressures.",
            "Recommended Investigation": "Correct exponent sign to positive: mu(P) = mu_ref * (1 + c_mu * (P - Pref)).",
            "Verification Test": "tests/scientific/physics/test_thermodynamic_consistency.py::test_liquid_viscosity_pressure_derivative",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-04",
            "Severity": "HIGH",
            "Category": "Physical / Thermodynamics",
            "Physical Phenomenon": "Isobaric Thermal Expansion",
            "Mathematical Issue": "Positive temperature coefficient: + 2.3e-3 * temp_c",
            "Code Location": "core/unified_engine/physics/co2_properties.py:140",
            "Observed Behavior": "CO2 density increases with temperature (drho/dT > 0), predicting density > 1,230 kg/m3.",
            "Expected Behavior": "Fluids expand when heated at constant pressure (drho/dT < 0).",
            "Evidence": "rho = 1.01 + 1.09e-2*p_mpa - 1.25e-5*p_mpa**2 + 2.3e-3*temp_c.",
            "Equation / Reference": "Span & Wagner (1996). J. Phys. Chem. Ref. Data, 25(6).",
            "Scientific Consequence": "CO2 predicted to be denser than liquid brine, inverting buoyancy and gravity segregation.",
            "Numerical Consequence": "Gravity override and vertical sweep models predict downward rather than upward migration.",
            "Reproducibility Consequence": "Contradicts Span-Wagner and NIST REFPROP standards.",
            "Recommended Investigation": "Replace with Span-Wagner 1996 formulation or Peng-Robinson EOS density.",
            "Verification Test": "tests/scientific/physics/test_thermodynamic_consistency.py::test_co2_density_thermal_expansion",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-05",
            "Severity": "CRITICAL",
            "Category": "Mathematical / Inflow Performance",
            "Physical Phenomenon": "Darcy Inflow & Well Interference",
            "Mathematical Issue": "Unbounded well drainage acreage scaling",
            "Code Location": "core/optimisation_engine.py:1410, profile_generator_fast.py:501",
            "Observed Behavior": "Single producer model drains arbitrary reservoir acreage at flat rates without interference.",
            "Expected Behavior": "Deliverability must be bounded by drainage radius, kh, skin, and inter-well pressure drawdown.",
            "Evidence": "Rate profiles scaled linearly with field target without nodal boundary validation.",
            "Equation / Reference": "Vogel, J.V. (1968). J. Pet. Technol., 20(1), 83-92.",
            "Scientific Consequence": "Under-predicts well count requirements by 3x to 5x.",
            "Numerical Consequence": "Optimizer exploits single well configurations to achieve unphysical high NPV.",
            "Reproducibility Consequence": "CAPEX optimization disconnected from actual field development plans.",
            "Recommended Investigation": "Coupled Peaceman well index with drainage radius interference limits.",
            "Verification Test": "tests/scientific/boundary_conditions/test_wellbore_drawdown_limits.py::test_producer_rate_drawdown_limit",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-06",
            "Severity": "CRITICAL",
            "Category": "Numerical / Thermodynamics",
            "Physical Phenomenon": "Thermodynamic State Decoupling",
            "Mathematical Issue": "Surrogate recovery evaluated at mean/initial pressure while tank ODE runs dynamically",
            "Code Location": "core/optimisation_engine.py:3215, surrogate_engine.py:407-420",
            "Observed Behavior": "Profiles generated before pressure profile calculation; oil profile scaled post-hoc.",
            "Expected Behavior": "Dynamic pressure, GOR, and water cut must be integrated simultaneously in time.",
            "Evidence": "profile_result['oil_profile'] = oil_profile * (target_cum_oil / max_cum_shape) applied after generation.",
            "Equation / Reference": "Aziz & Settari (1979). Petroleum Reservoir Simulation.",
            "Scientific Consequence": "Violates deliverability cap enforced in profile generator; breaks instantaneous GOR and WOR consistency.",
            "Numerical Consequence": "Discontinuous feedback between optimizer evaluations and dynamic profiles.",
            "Reproducibility Consequence": "Different time resolutions produce inconsistent cumulative productions.",
            "Recommended Investigation": "Fully couple timestep rate synthesis with pressure increment.",
            "Verification Test": "tests/scientific/convergence/test_temporal_convergence.py::test_temporal_refinement_convergence",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-07",
            "Severity": "HIGH",
            "Category": "Dimensional / Accounting",
            "Physical Phenomenon": "CO2 Utilization Factor",
            "Mathematical Issue": "Dimensional unit mismatch: metric tonnes/STB vs MSCF/STB",
            "Code Location": "core/objectives/wrapper.py:212 vs surrogate_engine.py:454",
            "Observed Behavior": "wrapper.py reports metric tonnes CO2 / STB; surrogate_engine reports MSCF/STB. Both named 'co2_utilization'.",
            "Expected Behavior": "Standard industry benchmark is MSCF/STB (typical 5 to 12 gross). Metric tonnes/STB is ~18x smaller.",
            "Evidence": "wrapper.py line 212: total_co2_purchased_tonne / total_oil.",
            "Equation / Reference": "DOE NETL (2010). Carbon Dioxide Enhanced Oil Recovery Primer.",
            "Scientific Consequence": "Reported utilization (0.665) appears 10x to 15x deflated compared to DOE/SPE literature.",
            "Numerical Consequence": "Optimization objective weights for utilization are scaled improperly relative to NPV.",
            "Reproducibility Consequence": "Results cannot be directly compared against SPE or CMG benchmarks.",
            "Recommended Investigation": "Standardize variable name to co2_utilization_mscf_per_stb and provide explicit units in outputs.",
            "Verification Test": "tests/scientific/dimensional/test_unit_consistency.py::test_co2_mass_conversion_factor",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-08",
            "Severity": "CRITICAL",
            "Category": "Physical / Phase Equilibrium",
            "Physical Phenomenon": "Cubic EOS Phase Identification",
            "Mathematical Issue": "Inverted Z-factor threshold logic",
            "Code Location": "core/unified_engine/physics/eos/__init__.py:195",
            "Observed Behavior": "'phase': 'V' if Z < 0.8 else 'L'.",
            "Expected Behavior": "In thermodynamics, liquid phase has lower compressibility factor Z (Z < 0.3); vapor has Z near unity.",
            "Evidence": "Direct ternary string assignment in get_properties_si.",
            "Equation / Reference": "Peng, D.Y. & Robinson, D.B. (1976). Ind. Eng. Chem. Fundam., 15(1), 59-64.",
            "Scientific Consequence": "Dense supercritical and liquid phases are systematically labeled as Vapor.",
            "Numerical Consequence": "Downstream viscosity and density correlations that branch on phase ('V' vs 'L') use wrong formulas.",
            "Reproducibility Consequence": "Phase identification contradicts standard PVT flash software.",
            "Recommended Investigation": "Determine phase from Gibbs free energy minimization or root ordering (Z_L = smallest root, Z_V = largest root).",
            "Verification Test": "tests/scientific/physics/test_phase_equilibrium.py::test_phase_label_assignment",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-09",
            "Severity": "HIGH",
            "Category": "Physical / Phase Equilibrium",
            "Physical Phenomenon": "Interfacial Tension at Miscibility",
            "Mathematical Issue": "Non-zero IFT at MMP: sigma(P) = sigma_0 * exp(-lambda*(P-MMP)) for P >= MMP",
            "Code Location": "core/simulation/recovery_models.py:197-202",
            "Observed Behavior": "At P = MMP, IFT equals 20 mN/m (maximum immiscible value), and decays slowly above MMP.",
            "Expected Behavior": "By thermodynamic definition of MMP, interfacial tension vanishes at MMP: sigma(MMP) -> 0.",
            "Evidence": "sigma = sigma_0 * np.exp(-lambda_ift * delta_p) with sigma_0 = 20 mN/m.",
            "Equation / Reference": "Rao, D.N. & Lee, J.I. (2002). SPE Res. Eval. & Eng., 5(3).",
            "Scientific Consequence": "Fluids remain capillary-trapped even thousands of psi above MMP.",
            "Numerical Consequence": "Capillary desaturation model under-predicts recovery in miscible regimes.",
            "Reproducibility Consequence": "Contradicts vanishing interfacial tension (VIT) experimental measurements.",
            "Recommended Investigation": "sigma(P) must approach 0 as P -> MMP from below, and be identically 0 for P >= MMP.",
            "Verification Test": "tests/scientific/physics/test_thermodynamic_consistency.py",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-10",
            "Severity": "HIGH",
            "Category": "Physical / Displacement",
            "Physical Phenomenon": "Immiscible Gas Displacement",
            "Mathematical Issue": "Displacement efficiency models only residual oil reduction (Sor_base - Sor*) / (1 - Swi)",
            "Code Location": "core/simulation/recovery_models.py:498-501",
            "Observed Behavior": "Immiscible displacement efficiency is 0.01-0.03 (predicting ~1% total recovery).",
            "Expected Behavior": "Immiscible gas flooding displaces mobile oil via Buckley-Leverett viscous drive, yielding 20-40% RF.",
            "Evidence": "displacement_eff = (sor_base - sor_star) / denominator.",
            "Equation / Reference": "Buckley & Leverett (1942). Trans. AIME, 146.",
            "Scientific Consequence": "Immiscible CO2 flooding falsely appears completely non-viable economically.",
            "Numerical Consequence": "Forces optimizer to strictly operate above MMP, ignoring optimal economic immiscible floods.",
            "Reproducibility Consequence": "Severe divergence from field historical immiscible CO2 floods.",
            "Recommended Investigation": "Incorporate mobile oil displacement via Buckley-Leverett fractional flow.",
            "Verification Test": "tests/scientific/reference_solutions/test_buckley_leverett_analytical.py",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-11",
            "Severity": "MEDIUM",
            "Category": "Numerical / Artificial Caps",
            "Physical Phenomenon": "Ultimate Recovery Factor Limit",
            "Mathematical Issue": "Confuses mobile pore volume fraction (1 - Swi - Sor) with fraction of OOIP",
            "Code Location": "core/engine_surrogate/analytical_models.py:881, surrogate_engine.py:425",
            "Observed Behavior": "RF clipped to 1.0 - Swi - Sor (~0.55) instead of (1 - Swi - Sor)/(1 - Swi) (~0.73).",
            "Expected Behavior": "OOIP is Vp*(1-Swi)/Boi. Total recoverable fraction of OOIP is (1 - Swi - Sor) / (1 - Swi).",
            "Evidence": "rf_max_physical = max(0.0, 1.0 - s_wi - sor) in analytical_models.py:881.",
            "Equation / Reference": "Craft & Hawkins (1991). Applied Petroleum Reservoir Engineering.",
            "Scientific Consequence": "Artificially clips recovery factor by 25% for high-performance miscible floods.",
            "Numerical Consequence": "Creates flat artificial gradient plateau at RF = 0.55.",
            "Reproducibility Consequence": "Fails to reproduce benchmark recovery factors above 60%.",
            "Recommended Investigation": "Normalize by (1 - Swi): rf_max = (1.0 - swi - sor) / max(1.0 - swi, 1e-4).",
            "Verification Test": "tests/scientific/conservation/test_mass_conservation.py::test_pore_volume_vs_ooip_recovery_bound_discrepancy",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-12",
            "Severity": "HIGH",
            "Category": "Dimensional / Unit Conversion",
            "Physical Phenomenon": "Gas Formation Volume Factor (Bg)",
            "Mathematical Issue": "10x numerical discrepancy in Bg constant",
            "Code Location": "core/optimisation_engine.py:98 vs surrogate_engine.py:201",
            "Observed Behavior": "optimisation_engine defines B_GAS_RB_PER_MSCF = 5.0; surrogate_engine uses ~0.5 RB/MSCF.",
            "Expected Behavior": "At reservoir conditions (P=3000 psi, T=150 F), CO2 Bg is ~0.45 - 0.70 RB/MSCF.",
            "Evidence": "B_GAS_RB_PER_MSCF = 5.0 in optimisation_engine.py:98.",
            "Equation / Reference": "Standing, M.B. (1977). Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems.",
            "Scientific Consequence": "10-fold distortion of reservoir voidage and pressure response depending on module called.",
            "Numerical Consequence": "Discrepancy between optimization constraint checking and simulation execution.",
            "Reproducibility Consequence": "Gas injection volumes convert to conflicting reservoir barrels.",
            "Recommended Investigation": "Compute dynamic Bg(P, T, Z) = 0.02827 * Z * T / P (RB/scf * 1000 / 5.615) everywhere.",
            "Verification Test": "tests/scientific/physics/test_fluid_properties.py::test_bg_discrepancy_between_modules",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-13",
            "Severity": "MEDIUM",
            "Category": "Mathematical / Empirical Correlation",
            "Physical Phenomenon": "Minimum Miscibility Pressure Correlation",
            "Mathematical Issue": "Ad-hoc (55 - API) modification creates complex/NaN results for light oils",
            "Code Location": "evaluation/mmp.py:111",
            "Observed Behavior": "At API >= 55.0, (55 - API)^0.279 evaluates to 0.0 or produces imaginary/NaN numbers.",
            "Expected Behavior": "Standard Cronquist (1978) correlation uses molecular weight of C7+ and volatile fractions.",
            "Evidence": "gravity_term = 55.0 - params.oil_gravity in evaluation/mmp.py:111.",
            "Equation / Reference": "Cronquist, C. (1978). 4th Annual US DOE Symposium, Tulsa.",
            "Scientific Consequence": "Fails completely on light volatile crudes and condensate fields.",
            "Numerical Consequence": "MMP returns NaN or 0, crashing downstream optimization.",
            "Reproducibility Consequence": "Invented ad-hoc modification masquerading as published Cronquist correlation.",
            "Recommended Investigation": "Implement authentic published Cronquist 1978 correlation with C7+ MW.",
            "Verification Test": "tests/scientific/mathematical/test_singularity_and_overflow.py::test_cronquist_mmp_singularity_at_55_api",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-14",
            "Severity": "HIGH",
            "Category": "Physical / Phase Equilibrium",
            "Physical Phenomenon": "Vapor-Liquid Equilibrium (VLE)",
            "Mathematical Issue": "Heuristic formula: vapor_frac = 1.0 - avg_z + 0.2",
            "Code Location": "analysis/material_balance.py:85-108",
            "Observed Behavior": "Calculates vapor fraction from compressibility factor Z without flash calculation.",
            "Expected Behavior": "Vapor fraction must be determined by isothermal-isobaric flash solving Rachford-Rice equation.",
            "Evidence": "vapor_frac = min(1.0, max(0.0, 1.0 - avg_z + 0.2)) in analysis/material_balance.py:85.",
            "Equation / Reference": "Rachford, H.H. & Rice, J.D. (1952). J. Pet. Technol., 4(10), 19-20.",
            "Scientific Consequence": "Phase fractions have no thermodynamic validity.",
            "Numerical Consequence": "Produced gas composition is arbitrary and does not conserve component moles.",
            "Reproducibility Consequence": "Cannot be validated against PVT flash experiments.",
            "Recommended Investigation": "Replace with authentic Rachford-Rice flash solver using PR-EOS fugacities.",
            "Verification Test": "tests/scientific/physics/test_phase_equilibrium.py",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
        {
            "ID": "SCI-FLAW-15",
            "Severity": "MEDIUM",
            "Category": "Physical / Material Balance",
            "Physical Phenomenon": "Reservoir Gas Inventory Dynamics",
            "Mathematical Issue": "Instantaneous coupling: produced CO2 tied strictly to instantaneous injection",
            "Code Location": "core/engine_surrogate/profile_generator_fast.py:983-986",
            "Observed Behavior": "co2_rate_available = injection_profile[i] * (1.0 - total_trapping). Shut-in drops production to 0.",
            "Expected Behavior": "CO2 continues to be produced from accumulated reservoir gas inventory during shut-in.",
            "Evidence": "Loop over i multiplying injection_profile[i] directly in profile_generator_fast.py:983.",
            "Equation / Reference": "Dake, L.P. (1978). Fundamentals of Reservoir Engineering.",
            "Scientific Consequence": "WAG water cycles falsely predict zero CO2 production during water injection steps.",
            "Numerical Consequence": "Distorts gas handling facility sizing and surface compression dynamics.",
            "Reproducibility Consequence": "Produces unphysical square-wave GOR fluctuations in WAG cycles.",
            "Recommended Investigation": "Tie produced CO2 to dynamic free gas saturation S_g and fractional flow f_g.",
            "Verification Test": "tests/scientific/limiting_cases/test_zero_injection.py::test_zero_injection_limits",
            "Status": "CONFIRMED_AUDIT_OPEN",
        },
    ]


def run_scientific_flaws_scan() -> List[Dict[str, Any]]:
    """Export the Master Scientific Flaw Register."""
    flaws = get_scientific_flaw_register()
    out_dir = AUDIT_DIR / "scientific_flaws"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "scientific_flaws.json", "w", encoding="utf-8") as f:
        json.dump(flaws, f, indent=2)

    with open(out_dir / "scientific_flaws.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(flaws[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fl in flaws:
            writer.writerow(fl)

    return flaws


def run_parameter_provenance_scan() -> List[Dict[str, Any]]:
    """Generate parameter provenance registry cataloging source, range, and calibration status."""
    parameters = [
        {"name": "COREY_N_OIL", "value": 2.0, "unit": "dimensionless", "provenance": "Literature (Corey 1954)", "equation": "kro = kro0 * (1 - Sw*)^n_o", "module": "core/engine_surrogate/surrogate_models.py:29", "status": "VERIFIED"},
        {"name": "COREY_N_GAS", "value": 2.0, "unit": "dimensionless", "provenance": "Literature (Corey 1954)", "equation": "krg = krg0 * Sg*^n_g", "module": "core/engine_surrogate/surrogate_models.py:30", "status": "VERIFIED"},
        {"name": "S_GC_CRITICAL", "value": 0.05, "unit": "fraction", "provenance": "Empirical Default", "equation": "Critical gas saturation", "module": "core/engine_surrogate/surrogate_models.py:31", "status": "EMPIRICAL"},
        {"name": "S_OR_BASE", "value": 0.25, "unit": "fraction", "provenance": "Empirical Default", "equation": "Residual oil saturation", "module": "core/engine_surrogate/surrogate_models.py:32", "status": "EMPIRICAL"},
        {"name": "CO2_DENSITY_TONNE_PER_MSCF", "value": 0.053, "unit": "tonne/MSCF", "provenance": "Fundamental Constant (MW=44.01)", "equation": "rho = MW / V_std", "module": "core/engine_surrogate/surrogate_models.py:26", "status": "VERIFIED"},
        {"name": "TODD_LONGSTAFF_OMEGA", "value": 0.60, "unit": "fraction", "provenance": "Literature (Todd & Longstaff 1972)", "equation": "mu_oe = mu_m^omega * mu_o^(1-omega)", "module": "core/engine_surrogate/analytical_models.py:38", "status": "VERIFIED"},
        {"name": "HETEROGENEITY_CALIBRATION_C_TRANS", "value": 0.80, "unit": "dimensionless", "provenance": "Calibrated / Tuning", "equation": "H = 1 / (1 - 0.80*V_DP)^2", "module": "core/engine_surrogate/surrogate_engine.py:195", "status": "CALIBRATED"},
        {"name": "NOMINAL_DRAWDOWN", "value": 500.0, "unit": "psi", "provenance": "Arbitrary Fallback", "equation": "J = q / nominal_drawdown", "module": "core/engine_surrogate/surrogate_engine.py:348", "status": "ARBITRARY"},
        {"name": "PRESSURE_INCREMENT_CLAMP", "value": 450.0, "unit": "psi/step", "provenance": "Numerical Safeguard", "equation": "dp = clip(dp, -450, 450)", "module": "core/engine_surrogate/surrogate_engine.py:379", "status": "NUMERICAL_STABILIZER"},
        {"name": "EPA_CLASS_VI_SAFETY_FACTOR", "value": 0.90, "unit": "fraction", "provenance": "Regulatory Standard (EPA Class VI)", "equation": "P_safe = 0.90 * P_frac", "module": "core/engine_surrogate/surrogate_engine.py:343", "status": "REGULATORY_VERIFIED"},
        {"name": "B_GAS_OPTIMISATION", "value": 5.0, "unit": "RB/MSCF", "provenance": "Unknown / Erroneous", "equation": "Gas conversion", "module": "core/optimisation_engine.py:98", "status": "CONTRADICTED_BY_TEST"},
    ]

    out_dir = AUDIT_DIR / "parameter_provenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parameter_registry.json", "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=2)

    with open(out_dir / "parameter_registry.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(parameters[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in parameters:
            writer.writerow(p)

    return parameters


def generate_audit_reports(scan_data: Dict[str, Any]) -> Path:
    """Compile final comprehensive PhD-level audit report in Markdown format."""
    out_dir = AUDIT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "scientific_audit_report.md"

    flaws = scan_data["flaws"]
    crit_count = sum(1 for f in flaws if f["Severity"] == "CRITICAL")
    high_count = sum(1 for f in flaws if f["Severity"] == "HIGH")
    med_count = sum(1 for f in flaws if f["Severity"] == "MEDIUM")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Scientific Audit & Traceability Report\n\n")
        f.write("**Application**: CO₂ EOR Optimizer (`co2eor_optimizer` v0.8.5)\n")
        f.write("**Role**: Senior Reservoir Engineer, Applied Mathematician & Scientific Auditor\n")
        f.write("**Standard**: PhD-Level Research Rigor (Zero Hidden Calibration, Strict Mass & Thermodynamic Conservation)\n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Scientific Audit Summary\n\n")
        f.write(f"- **Total Scanned Python Files**: {scan_data['quality']['total_files']}\n")
        f.write(f"- **Total Scanned Code Lines**: {scan_data['quality']['total_lines']:,}\n")
        f.write(f"- **Total Documented Scientific Flaws**: {len(flaws)} (Critical: {crit_count}, High: {high_count}, Medium: {med_count})\n")
        f.write(f"- **Total Documented Fallback Exception Handlers**: {len(scan_data['fallbacks'])}\n")
        f.write(f"- **Total Documented Hardcoded Numerical Literals**: {len(scan_data['hardcoded'])}\n\n")

        f.write("### Primary Scientific Finding:\n")
        f.write("> **The codebase contains multiple fundamental violations of physical and thermodynamic laws** (inverted Buckley-Leverett/Koval fractional flow, negative compressibility, inverted viscosity-pressure dependence, inverted CO2 thermal expansion, and inverted cubic EOS phase labeling). These defects are currently masked by post-hoc profile scalers, heuristic damping factors, and artificial recovery factor ceilings.\n\n")

        f.write("---\n\n")
        f.write("## 2. Master Scientific Flaw Summary Table\n\n")
        f.write("| ID | Severity | Physical Phenomenon | Code Location | Observed Defect | Status |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        for fl in flaws:
            f.write(f"| **{fl['ID']}** | **{fl['Severity']}** | {fl['Physical Phenomenon']} | `{fl['Code Location']}` | {fl['Observed Behavior'][:80]}... | {fl['Status']} |\n")
        f.write("\n---\n\n")

        f.write("## 3. Parameter Provenance & Calibration Analysis\n\n")
        f.write("Parameters have been classified into Fundamental, Literature, Empirical, Calibrated, Arbitrary, or Contradicted.\n\n")
        f.write("| Parameter | Value | Units | Provenance | Code Location | Classification |\n")
        f.write("|:---|:---|:---|:---|:---|:---|\n")
        for p in scan_data["params"]:
            f.write(f"| `{p['name']}` | {p['value']} | {p['unit']} | {p['provenance']} | `{p['module']}` | **{p['status']}** |\n")
        f.write("\n")

    return report_path


def run_full_audit() -> Dict[str, Any]:
    """Execute complete audit pipeline."""
    print("=" * 80)
    print("CO2 EOR OPTIMIZER - EXECUTING AUTOMATED SCIENTIFIC AUDIT PIPELINE")
    print("=" * 80)

    py_files = get_python_files()
    print(f"[1/6] Scanning code quality across {len(py_files)} modules...")
    quality = run_code_quality_scan(py_files)

    print(f"[2/6] Scanning exception fallbacks...")
    fallbacks = run_fallbacks_scan(py_files)

    print(f"[3/6] Scanning hardcoded numerical literals in physics...")
    hardcoded = run_hardcoded_values_scan(py_files)

    print(f"[4/6] Compiling Master Scientific Flaw Register...")
    flaws = run_scientific_flaws_scan()

    print(f"[5/6] Cataloging parameter provenance...")
    params = run_parameter_provenance_scan()

    scan_data = {
        "quality": quality,
        "fallbacks": fallbacks,
        "hardcoded": hardcoded,
        "flaws": flaws,
        "params": params,
    }

    print(f"[6/6] Generating comprehensive Markdown and tabular reports...")
    report_path = generate_audit_reports(scan_data)
    print(f"\nAudit complete! Master report written to: {report_path}")
    print("=" * 80)
    return scan_data
