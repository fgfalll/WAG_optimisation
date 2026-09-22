"""
Run Data Exporter for CO2 EOR Optimizer
========================================
Generates comprehensive, standardized, and AI-agent-friendly export suites
for simulation and optimization runs.

Artifacts Generated:
1. README.md - Front-door entrypoint with navigation map, health badge, and warnings/errors registry.
2. run_manifest.json - Structured, lightweight JSON manifest (< 30 KB) designed for automated ingestion.
3. run_evaluation_report.md - Comprehensive GitHub Markdown report with executive badges, KPI tables, and physics checks.
4. cash_flows_yearly.csv - Year-by-year financial schedule (revenues, OPEX, CAPEX, carbon credits, DCF).
5. convergence_history.csv - Step-by-step optimization progression trajectory.
6. summary_yearly.csv - Yearly production, injection, pressure, and mass balance table with explicit physical units.
7. summary_monthly.csv - Monthly time-series data (when available) for breakthrough and WAG dynamics.
8. results_summary.txt - Cleaned, human-readable summary text without raw numpy string dumps.
9. full_run_data.json - Complete raw dataset with robust serialization.
10. Figure PNGs - High-resolution plots of convergence, production profiles, material balance, etc.
"""

import os
import io
import json
import logging
from pathlib import Path
from datetime import datetime, date
from dataclasses import is_dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RunDataExporter:
    """
    Centralized exporter generating AI-agent-friendly and human-readable
    run evaluation artifacts.
    """

    PARAMETER_UNITS: Dict[str, str] = {
        "pressure": "psia",
        "rate": "MSCF/day",
        "co2_injection_rate": "MSCF/day",
        "injection_rate": "MSCF/day",
        "plateau_duration_fraction": "dimensionless (0-1)",
        "ramp_up_fraction": "dimensionless (0-1)",
        "wellbore_pressure": "psia",
        "max_production_rate_stbd": "STB/day",
        "well_shut_in_threshold_bpd": "STB/day",
        "allow_well_conversion": "flag (0=No, 1=Yes)",
        "well_conversion_day": "days",
        "shut_in_mode": "mode (0=None, 1=Abrupt, 2=Tapered)",
        "shut_in_ramp_days": "days",
        "wag_ratio": "ratio (dimensionless)",
        "cycle_length_days": "days",
        "water_injection_rate": "STB/day",
    }

    def __init__(
        self,
        export_dir: Optional[Union[str, Path]] = None,
        output_root_dir: Optional[Union[str, Path]] = None,
    ):
        chosen = export_dir if export_dir is not None else output_root_dir
        self.export_dir = Path(chosen) if chosen else None

    def export_run(
        self,
        results: Dict[str, Any],
        engine: Optional[Any] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
        plots_generator: Optional[Dict[str, Any]] = None,
        convergence_live_data: Optional[List[Dict[str, float]]] = None,
    ) -> Path:
        """Instance export method delegating to cls.export."""
        return self.export(
            results=results,
            engine=engine,
            input_parameters=input_parameters,
            target_dir=self.export_dir,
            convergence_live_data=convergence_live_data,
            plots_generator=plots_generator,
        )

    @classmethod
    def export(
        cls,
        results: Dict[str, Any],
        engine: Optional[Any] = None,
        input_parameters: Optional[Dict[str, Any]] = None,
        target_dir: Optional[Union[str, Path]] = None,
        convergence_live_data: Optional[List[Dict[str, float]]] = None,
        plots_generator: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Execute full export pipeline and return the created export folder Path.
        """
        method = str(results.get("method", "optimization")).replace("_", "-").lower()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"Export-{method}-{timestamp}"

        base_dir = Path(target_dir) if target_dir else Path.cwd() / "logs"
        export_path = base_dir / folder_name
        export_path.mkdir(parents=True, exist_ok=True)

        exporter = cls(export_path)
        exporter._generate_all_artifacts(
            results=results,
            engine=engine,
            input_parameters=input_parameters or {},
            convergence_live_data=convergence_live_data or [],
            plots_generator=plots_generator or {},
        )
        return export_path

    def _generate_all_artifacts(
        self,
        results: Dict[str, Any],
        engine: Optional[Any],
        input_parameters: Dict[str, Any],
        convergence_live_data: List[Dict[str, float]],
        plots_generator: Dict[str, Any],
    ) -> None:
        """Orchestrate generation of all 9 export artifacts."""
        manifest = self._build_run_manifest(results, engine, input_parameters)
        warnings_and_errors = self._diagnose_warnings_and_errors(manifest, results, engine)

        # 1. README.md (Entrypoint)
        readme_content = self._build_readme_entrypoint(manifest, warnings_and_errors)
        self._write_file("README.md", readme_content)

        # 2. run_manifest.json (Lightweight machine-readable manifest)
        manifest_with_diagnostics = {**manifest, "diagnostics": warnings_and_errors}
        self._write_json("run_manifest.json", manifest_with_diagnostics)

        # 3. run_evaluation_report.md (Detailed evaluation report)
        report_md = self._build_evaluation_report_markdown(manifest, warnings_and_errors, results)
        self._write_file("run_evaluation_report.md", report_md)

        # 4. cash_flows_yearly.csv (Financial schedule)
        cash_flow_df = self._build_cash_flow_table(results, engine, input_parameters)
        if not cash_flow_df.empty:
            cash_flow_df.to_csv(self.export_dir / "cash_flows_yearly.csv", index=False)

        # 5. convergence_history.csv (Optimization progression)
        conv_df = self._build_convergence_table(results, convergence_live_data)
        if not conv_df.empty:
            conv_df.to_csv(self.export_dir / "convergence_history.csv", index=False)

        # 6. summary_yearly.csv and summary_monthly.csv (Standardized profiles)
        profile_dfs = self._build_profile_tables(results, engine)
        for name, df in profile_dfs.items():
            df.to_csv(self.export_dir / f"summary_{name}.csv", index=False)

        # 7. results_summary.txt (Clean human text summary)
        txt_summary = self._build_sanitized_text_summary(
            manifest, warnings_and_errors, results, input_parameters
        )
        self._write_file("results_summary.txt", txt_summary)

        # 8. full_run_data.json (Cleaned comprehensive dataset)
        full_json = {
            "manifest": manifest,
            "diagnostics": warnings_and_errors,
            "input_parameters": self._sanitize_dict(input_parameters),
            "full_results": self._sanitize_dict(results),
        }
        self._write_json("full_run_data.json", full_json)

        # 9. Plot images
        self._save_plots(plots_generator, results)

    # -------------------------------------------------------------------------
    # Manifest & Diagnostics
    # -------------------------------------------------------------------------

    def _build_run_manifest(
        self,
        results: Dict[str, Any],
        engine: Optional[Any],
        input_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile a clean, lightweight, strictly-typed manifest dictionary."""
        final_metrics = results.get("final_metrics", {}) or {}
        profiles = results.get("optimized_profiles", {}) or {}
        mb_analysis = results.get("material_balance_analysis") or {}
        mb_stats = mb_analysis.get("summary_statistics", {}) or {}

        # Decision variables & boundaries
        decision_vars = self._extract_decision_variables(results, engine)

        # Physical NPV vs Fitness score
        raw_npv = float(final_metrics.get("npv", results.get("npv", 0.0)))
        obj_fitness = float(results.get("objective_function_value", raw_npv))
        bt_impact = float(final_metrics.get("breakthrough_impact_factor", 1.0))

        # Recovery and production totals
        rf = float(final_metrics.get("recovery_factor", results.get("recovery_factor", 0.0)))
        cum_oil = self._safe_sum(profiles.get("yearly_oil_stb", profiles.get("annual_oil_stb", [])))
        cum_water = self._safe_sum(
            profiles.get("yearly_water_stb", profiles.get("annual_water_stb", []))
        )
        cum_tot_gas = self._safe_sum(
            profiles.get("yearly_total_gas_mscf", profiles.get("annual_total_gas_mscf", []))
        )
        cum_co2_prod = self._safe_sum(
            profiles.get("yearly_co2_produced_mscf", profiles.get("annual_co2_produced_mscf", []))
        )
        cum_hc_gas = self._safe_sum(
            profiles.get(
                "yearly_hc_gas_produced_mscf", profiles.get("annual_hc_gas_produced_mscf", [])
            )
        )

        # CO2 storage & utilization
        co2_util_tonne_per_stb = float(final_metrics.get("co2_utilization", 0.0))
        co2_util_mscf_per_stb = co2_util_tonne_per_stb * 18.90

        gross_storage_eff = float(final_metrics.get("storage_efficiency", 0.0))
        purchased_storage_eff = float(mb_stats.get("avg_storage_efficiency", gross_storage_eff))
        total_prod_tonne = float(
            mb_stats.get("total_produced_tonne", results.get("total_co2_produced_tonne", 0.0))
        )
        total_recycled_tonne = float(
            mb_stats.get("total_recycled_tonne", results.get("total_co2_recycled_tonne", 0.0))
        )
        total_stored_tonne = float(
            mb_stats.get("total_net_stored_tonne", results.get("co2_stored", 0.0))
        )
        total_leakage_tonne = float(mb_stats.get("total_leakage_tonne", 0.0))

        # Disambiguate gross vs purchased injection
        accounted = total_stored_tonne + total_prod_tonne + total_leakage_tonne
        if "total_gross_injected_tonne" in mb_stats:
            total_gross_injected_tonne = float(mb_stats["total_gross_injected_tonne"])
            total_purchased_tonne = float(
                mb_stats.get("total_purchased_tonne", max(0.0, total_gross_injected_tonne - total_recycled_tonne))
            )
        else:
            raw_inj = float(
                mb_stats.get("total_injected_tonne", results.get("total_co2_injected_tonne", 0.0))
            )
            # If raw_inj already matches accounted (stored + produced + leakage), it represents gross injection
            if abs(raw_inj - accounted) < 1.0 or raw_inj == 0.0:
                total_gross_injected_tonne = raw_inj
                total_purchased_tonne = max(0.0, total_gross_injected_tonne - total_recycled_tonne)
            else:
                total_purchased_tonne = raw_inj
                total_gross_injected_tonne = total_purchased_tonne + total_recycled_tonne

        # Mass balance error closure on closed-loop gross carbon basis:
        # Gross Injected = Net Stored + Total Leakage + Gross Produced
        mb_error_tonne = (
            abs(total_gross_injected_tonne - accounted)
            if total_gross_injected_tonne > 0
            else 0.0
        )
        mb_closure_pct = (
            (1.0 - (mb_error_tonne / max(total_gross_injected_tonne, 1e-6))) * 100.0
            if total_gross_injected_tonne > 0
            else 100.0
        )

        # Reservoir context
        gen_params = input_parameters.get("General & Reservoir", {})
        res_params = input_parameters.get("Reservoir Parameters", {})
        well_params = input_parameters.get("Well Configuration", {})
        opt_setup = input_parameters.get("Optimization Setup", {})

        eng_res = getattr(engine, "reservoir", None)
        eng_op = getattr(engine, "operational_params", None)

        ooip = float(gen_params.get("OOIP (STB)", getattr(eng_res, "ooip_stb", 0.0)))
        mmp = float(gen_params.get("MMP (Calculated, psi)", getattr(engine, "mmp", 2000.0)))
        porosity = float(gen_params.get("Average Porosity", getattr(eng_res, "average_porosity", 0.2)))
        perm = float(
            gen_params.get(
                "Average Permeability (mD)",
                gen_params.get(
                    "Average Permeability",
                    res_params.get(
                        "average_permeability",
                        getattr(eng_res, "average_permeability", getattr(eng_res, "permeability_md", 100.0)),
                    ),
                ),
            )
        )
        lifetime = int(gen_params.get("Project Lifetime (years)", getattr(eng_op, "project_lifetime_years", 15)))

        init_p = float(
            gen_params.get(
                "Initial Pressure (psi)",
                res_params.get(
                    "initial_pressure",
                    getattr(eng_res, "initial_pressure", getattr(eng_res, "initial_pressure_psi", 3000.0)),
                ),
            )
        )
        temp_f = float(
            gen_params.get(
                "Temperature (°F)",
                res_params.get(
                    "temperature",
                    getattr(eng_res, "temperature", getattr(eng_res, "temperature_f", 160.0)),
                ),
            )
        )
        thick = float(
            gen_params.get(
                "Thickness (ft)",
                res_params.get("thickness_ft", getattr(eng_res, "thickness_ft", 50.0)),
            )
        )
        area = float(
            gen_params.get(
                "Area (acres)",
                res_params.get("area_acres", getattr(eng_res, "area_acres", 10.0)),
            )
        )
        rock_c = float(
            res_params.get("rock_compressibility", getattr(eng_res, "rock_compressibility", 3e-6))
        )
        tot_wells = int(
            well_params.get("total_wells", len(getattr(engine, "well_data_list", []) or []))
        )
        n_inj = int(well_params.get("injector_count", 0))
        n_prod = int(well_params.get("producer_count", 0))

        # Fracture safety
        eor_input = input_parameters.get("EOR Parameters", {})
        eor = getattr(engine, "eor_params", None)
        p_frac = float(
            eor_input.get(
                "caprock_fracture_pressure_psi",
                getattr(eor, "caprock_fracture_pressure_psi", 5500.0),
            )
        )
        safety_factor = float(
            eor_input.get(
                "caprock_safety_factor",
                getattr(eor, "caprock_safety_factor", 0.90),
            )
        )
        p_safe_ceiling = p_frac * safety_factor
        opt_p_res = float(decision_vars.get("pressure", {}).get("value", 2000.0))
        pressure_margin = p_safe_ceiling - opt_p_res

        # Optimizer metadata
        stats = results.get("bo_statistics") or results.get("ga_statistics") or {}
        duration = float(stats.get("total_duration_seconds", 0.0))
        total_evals = int(stats.get("total_evaluations", 0))

        return {
            "schema_version": "2.0.0",
            "metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "optimizer_method": opt_setup.get("selected_method", results.get("method", "Unknown")),
                "objective_name": opt_setup.get("selected_objective", results.get("objective_name", "npv")),
                "simulation_engine": opt_setup.get("simulation_engine_type", "surrogate"),
                "recovery_model": opt_setup.get("recovery_model", "phd_hybrid"),
                "total_duration_seconds": duration,
                "total_evaluations": total_evals,
                "project_lifetime_years": lifetime,
            },
            "key_performance_indicators": {
                "economic_npv_usd": raw_npv,
                "objective_fitness_score": obj_fitness,
                "breakthrough_impact_penalty": bt_impact,
                "recovery_factor_fraction": rf,
                "recovery_factor_percent": rf * 100.0,
                "cumulative_oil_stb": cum_oil,
                "cumulative_water_stb": cum_water,
                "cumulative_total_gas_mscf": cum_tot_gas,
                "cumulative_co2_produced_mscf": cum_co2_prod,
                "cumulative_hc_gas_produced_mscf": cum_hc_gas,
                "co2_utilization_tonne_per_stb": co2_util_tonne_per_stb,
                "co2_utilization_mscf_per_stb": co2_util_mscf_per_stb,
                "gross_storage_efficiency": gross_storage_eff,
                "purchased_storage_efficiency": purchased_storage_eff,
                "breakthrough_time_years": float(final_metrics.get("breakthrough_time_years", 0.0)),
                "ecology_compliant": bool(final_metrics.get("ecology_compliant", True)),
            },
            "carbon_accounting_tonnes": {
                "total_injected_tonne": total_gross_injected_tonne,
                "total_gross_injected_tonne": total_gross_injected_tonne,
                "total_purchased_tonne": total_purchased_tonne,
                "total_produced_tonne": total_prod_tonne,
                "total_recycled_tonne": total_recycled_tonne,
                "total_uncaptured_tonne": max(0.0, total_prod_tonne - total_recycled_tonne),
                "total_net_stored_tonne": total_stored_tonne,
                "total_leakage_tonne": total_leakage_tonne,
                "mass_balance_error_tonne": mb_error_tonne,
                "mass_balance_closure_percent": mb_closure_pct,
            },
            "decision_variables": decision_vars,
            "geomechanical_safety": {
                "optimized_reservoir_pressure_psi": opt_p_res,
                "caprock_fracture_pressure_psi": p_frac,
                "safe_fracture_ceiling_psi": p_safe_ceiling,
                "safety_factor": safety_factor,
                "pressure_margin_psi": pressure_margin,
                "fracture_ceiling_violated": pressure_margin < 0,
            },
            "reservoir_context": {
                "ooip_stb": ooip,
                "mmp_psi": mmp,
                "average_porosity": porosity,
                "permeability_md": perm,
                "initial_pressure_psi": init_p,
                "temperature_f": temp_f,
                "thickness_ft": thick,
                "area_acres": area,
                "rock_compressibility_psi": rock_c,
                "total_wells": tot_wells,
                "injector_count": n_inj,
                "producer_count": n_prod,
                "miscibility_status": "Miscible"
                if opt_p_res >= mmp
                else ("Near-Miscible" if opt_p_res >= 0.9 * mmp else "Immiscible"),
            },
            "fluid_properties": self._extract_safe_dict(
                input_parameters.get("Fluid & PVT Properties", {})
            ),
            "operational_context": self._extract_safe_dict(
                input_parameters.get("Operational Parameters", {})
            ),
            "eor_parameters": self._extract_safe_dict(
                input_parameters.get("EOR Parameters", {})
            ),
            "economic_parameters": self._extract_safe_dict(
                input_parameters.get("Economic Parameters", {})
            ),
            "co2_storage_parameters": self._extract_safe_dict(
                input_parameters.get("CO2 Storage Parameters", {})
            ),
            "optimization_search_bounds": self._extract_safe_dict(
                input_parameters.get("Optimization Search Bounds", {})
            ),
            "algorithm_hyperparameters": {
                k: self._extract_safe_dict(input_parameters.get(k, {}))
                for k in [
                    "Genetic Algorithm",
                    "Bayesian Optimization",
                    "Particle Swarm Optimization",
                    "Differential Evolution",
                ]
                if input_parameters.get(k)
            },
            "mmp_analysis": self._extract_safe_dict(
                input_parameters.get("MMP Analysis Configuration", {})
            ),
            "well_configuration": self._extract_safe_dict(
                input_parameters.get("Well Configuration", {})
            ),
            "decline_curve_analysis": results.get("dca_results", {}).get("summary", {}),
        }

    def _diagnose_warnings_and_errors(
        self,
        manifest: Dict[str, Any],
        results: Dict[str, Any],
        engine: Optional[Any],
    ) -> List[Dict[str, str]]:
        """Run multi-physics diagnostics and return a list of warnings and errors."""
        issues: List[Dict[str, str]] = []
        kpis = manifest["key_performance_indicators"]
        safety = manifest["geomechanical_safety"]
        dvars = manifest["decision_variables"]
        res_ctx = manifest["reservoir_context"]

        # 1. Geomechanical overpressure
        if safety["fracture_ceiling_violated"]:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "category": "Geomechanics",
                    "code": "GEO_FRAC_EXCEEDED",
                    "message": f"Reservoir injection pressure ({safety['optimized_reservoir_pressure_psi']:.1f} psi) exceeds safe Class VI fracture ceiling ({safety['safe_fracture_ceiling_psi']:.1f} psi).",
                }
            )
        elif safety["pressure_margin_psi"] < 100.0:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "Geomechanics",
                    "code": "GEO_LOW_MARGIN",
                    "message": f"Low fracture pressure safety margin: only {safety['pressure_margin_psi']:.1f} psi below ceiling.",
                }
            )

        # 2. Premature Breakthrough
        bt_time = kpis["breakthrough_time_years"]
        if bt_time < 1.0:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "Displacement Physics",
                    "code": "EARLY_BREAKTHROUGH",
                    "message": f"Severe early breakthrough predicted at {bt_time:.2f} years (< 1 year). Incurred breakthrough penalty factor {kpis['breakthrough_impact_penalty']:.3f}.",
                }
            )

        # 3. Parameter Boundary Pinning
        for param, pinfo in dvars.items():
            if pinfo.get("is_at_lower_bound"):
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "Optimization Bounds",
                        "code": "PARAM_LOWER_BOUND_PINNED",
                        "message": f"Parameter '{param}' ({pinfo['value']}) pinned to lower search bound ({pinfo['lower_bound']}). Search space may be too restrictive.",
                    }
                )
            elif pinfo.get("is_at_upper_bound"):
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "Optimization Bounds",
                        "code": "PARAM_UPPER_BOUND_PINNED",
                        "message": f"Parameter '{param}' ({pinfo['value']}) pinned to upper search bound ({pinfo['upper_bound']}). Consider widening bounds.",
                    }
                )

        # 4. Storage Efficiency & Leakage
        if kpis["gross_storage_efficiency"] < 0.20:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "Carbon Storage",
                    "code": "LOW_STORAGE_EFFICIENCY",
                    "message": f"Gross CO2 retention fraction ({kpis['gross_storage_efficiency']*100:.1f}%) is unusually low. Heavy recycling or gas channeling suspected.",
                }
            )

        leakage = manifest["carbon_accounting_tonnes"]["total_leakage_tonne"]
        total_inj = manifest["carbon_accounting_tonnes"]["total_injected_tonne"]
        if total_inj > 0 and (leakage / total_inj) > 0.05:
            issues.append(
                {
                    "severity": "CRITICAL",
                    "category": "Containment",
                    "code": "HIGH_CO2_LEAKAGE",
                    "message": f"CO2 leakage ({leakage:,.0f} tonnes, {(leakage/total_inj)*100:.1f}% of injection) violates EPA Class VI containment targets.",
                }
            )

        # 5. Mass Balance Closure
        mb_closure = manifest["carbon_accounting_tonnes"]["mass_balance_closure_percent"]
        if mb_closure < 99.5:
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "Mass Balance",
                    "code": "MASS_BALANCE_DISCREPANCY",
                    "message": f"Mass conservation closure is {mb_closure:.2f}% (< 99.5%). Residual mass balance error is {manifest['carbon_accounting_tonnes']['mass_balance_error_tonne']:,.1f} tonnes.",
                }
            )

        # 6. Target Reachability
        if results.get("target_was_unreachable"):
            issues.append(
                {
                    "severity": "WARNING",
                    "category": "Convergence",
                    "code": "TARGET_UNREACHABLE",
                    "message": "The specified target objective value could not be reached within search bounds and iterations.",
                }
            )

        return issues

    def _extract_decision_variables(
        self,
        results: Dict[str, Any],
        engine: Optional[Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract optimal decision variables with their bounds and boundary hit status."""
        opt_params = (
            results.get("optimized_params_final_clipped")
            or results.get("optimized_parameters")
            or {}
        )
        bounds_dict: Dict[str, Tuple[float, float]] = {}
        if engine and hasattr(engine, "_get_parameter_bounds"):
            try:
                bounds_dict = engine._get_parameter_bounds()
            except Exception:
                pass

        dvars = {}
        for param, val in opt_params.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                fval = float(val)
                unit = self.PARAMETER_UNITS.get(param, "dimensionless")
                b = bounds_dict.get(param)
                lb = float(b[0]) if b else None
                ub = float(b[1]) if b else None

                at_lb = False
                at_ub = False
                if lb is not None and ub is not None and ub > lb:
                    span = ub - lb
                    at_lb = abs(fval - lb) <= (0.015 * span)
                    at_ub = abs(fval - ub) <= (0.015 * span)

                dvars[param] = {
                    "value": int(fval) if isinstance(val, int) else round(fval, 4),
                    "unit": unit,
                    "lower_bound": lb,
                    "upper_bound": ub,
                    "is_at_lower_bound": at_lb,
                    "is_at_upper_bound": at_ub,
                }
        return dvars

    # -------------------------------------------------------------------------
    # Markdown & Entrypoint Builders
    # -------------------------------------------------------------------------

    def _build_readme_entrypoint(
        self,
        manifest: Dict[str, Any],
        diagnostics: List[Dict[str, str]],
    ) -> str:
        """Construct the root README.md file for the run folder."""
        meta = manifest["metadata"]
        kpis = manifest["key_performance_indicators"]
        carbon = manifest["carbon_accounting_tonnes"]

        critical_count = sum(1 for d in diagnostics if d["severity"] == "CRITICAL")
        warning_count = sum(1 for d in diagnostics if d["severity"] == "WARNING")

        if critical_count > 0:
            status_badge = "🔴 **CRITICAL ISSUES DETECTED** (Safety/Regulatory Violations)"
        elif warning_count > 0:
            status_badge = "🟡 **WARNINGS DETECTED** (Near Boundaries or Sub-Optimal Limits)"
        else:
            status_badge = "🟢 **PASS** (Clean Physical & Economic Run)"

        lines = [
            f"# CO₂-EOR Run Evaluation Dashboard",
            f"**Run Method:** `{meta['optimizer_method']}` | **Objective:** `{meta['objective_name']}` | **Generated:** `{meta['generation_timestamp'][:19]}`",
            "",
            f"### Run Health Status: {status_badge}",
            "",
            "---",
            "",
            "## 📁 Export Artifact Index & Navigation Map",
            "This folder contains a complete, standardized suite of evaluation artifacts designed for both human review and automated AI-agent ingestion:",
            "",
            "| Artifact | Format | Description | Target Consumer |",
            "| :--- | :---: | :--- | :--- |",
            "| [run_manifest.json](run_manifest.json) | JSON | Machine-readable manifest (< 30 KB) with complete KPIs, bounds, and context | **AI Agents / Automated Pipelines** |",
            "| [run_evaluation_report.md](run_evaluation_report.md) | Markdown | Full technical evaluation report with executive badges and physics checks | **Engineers & Reviewers** |",
            "| [cash_flows_yearly.csv](cash_flows_yearly.csv) | CSV | Year-by-year cashflow schedule (Revenue, OPEX, CAPEX, DCF) | **Financial / Commercial Analysis** |",
            "| [convergence_history.csv](convergence_history.csv) | CSV | Optimization convergence trajectory per evaluation/iteration | **Optimizer Tuning & Profiling** |",
            "| [summary_yearly.csv](summary_yearly.csv) | CSV | Standardized annual production, injection, and pressure profiles with units | **Reservoir Simulation Workflows** |",
            "| [summary_monthly.csv](summary_monthly.csv) | CSV | High-resolution monthly profiles (if generated) for breakthrough dynamics | **Detailed Reservoir Modeling** |",
            "| [results_summary.txt](results_summary.txt) | Text | Cleaned human-readable summary text without raw array dumps | **Quick Terminal / Text Viewing** |",
            "| [full_run_data.json](full_run_data.json) | JSON | Complete archive of inputs, intermediate evaluations, and raw results | **Deep Archival & Debugging** |",
            "",
            "---",
            "",
            "## ⚠️ Warnings & Errors Registry",
        ]

        if not diagnostics:
            lines.append("✅ **No physical, geomechanical, or operational violations detected.**")
        else:
            lines.extend(
                [
                    "| Severity | Category | Diagnostic Code | Description |",
                    "| :---: | :---: | :---: | :--- |",
                ]
            )
            for d in diagnostics:
                sev_icon = "🔴 CRITICAL" if d["severity"] == "CRITICAL" else "🟡 WARNING"
                lines.append(
                    f"| {sev_icon} | {d['category']} | `{d['code']}` | {d['message']} |"
                )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 📊 Quick Executive Summary for AI Agents",
                "",
                "| Key Performance Indicator | Value | Explicit Unit | Physical Interpretation |",
                "| :--- | :---: | :---: | :--- |",
                f"| **Project DCF NPV** | **${kpis['economic_npv_usd']:,.2f}** | USD | Net Present Value (10% discount rate) |",
                f"| **Optimizer Fitness Score** | `{kpis['objective_fitness_score']:.4e}` | dimensionless | " + (f"Internal score (includes breakthrough penalty {kpis['breakthrough_impact_penalty']:.3f})" if abs(kpis['objective_fitness_score'] - kpis['economic_npv_usd']) > 1.0 else "Internal objective score (unpenalized)") + " |",
                f"| **Ultimate Recovery Factor** | **{kpis['recovery_factor_percent']:.2f}%** | % OOIP | Fraction of initial oil recovered |",
                f"| **Cumulative Oil Production** | **{kpis['cumulative_oil_stb']:,.0f}** | STB | Total field oil recovered |",
                f"| **Net CO₂ Utilization** | **{kpis['co2_utilization_tonne_per_stb']:.3f}** | tonne/STB | Purchased CO₂ required per barrel of oil ({kpis['co2_utilization_mscf_per_stb']:.2f} MSCF/bbl) |",
                f"| **Purchased Storage Efficiency**| **{kpis['purchased_storage_efficiency']*100:.1f}%** | % fresh CO₂ | Fraction of purchased CO₂ permanently stored |",
                f"| **Total CO₂ Stored** | **{carbon['total_net_stored_tonne']:,.0f}** | metric tonnes | Net permanent subsurface carbon storage |",
                f"| **Breakthrough Timing** | **{kpis['breakthrough_time_years']:.2f}** | years | Elapsed time until CO₂ reaches production wells |",
                "",
                "---",
                "*Generated automatically by CO₂-EOR Optimizer Run Data Export System.*",
            ]
        )
        return "\n".join(lines)

    def _build_evaluation_report_markdown(
        self,
        manifest: Dict[str, Any],
        diagnostics: List[Dict[str, str]],
        results: Dict[str, Any],
    ) -> str:
        """Construct the comprehensive run_evaluation_report.md document."""
        meta = manifest["metadata"]
        kpis = manifest["key_performance_indicators"]
        carbon = manifest["carbon_accounting_tonnes"]
        safety = manifest["geomechanical_safety"]
        res = manifest["reservoir_context"]
        dvars = manifest["decision_variables"]

        has_bt_deduction = abs(kpis['objective_fitness_score'] - kpis['economic_npv_usd']) > 1.0
        bt_note = "Heuristic multiplier applied to NPV in fitness" if has_bt_deduction else "Calculated timing metric (no penalty applied to fitness)"

        lines = [
            f"# Optimization Run Technical Evaluation Report",
            f"**Run Identifier:** `{meta['optimizer_method']}-{meta['generation_timestamp'][:19]}`",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Evaluations:** {meta['total_evaluations']} | **Duration:** {meta['total_duration_seconds']:.1f}s",
            "",
            "## 1. Executive Summary & KPIs",
            "",
            "| Metric | Value | Unit | Engineering Reference / Benchmark |",
            "| :--- | :---: | :---: | :--- |",
            f"| **Net Present Value (NPV)** | **${kpis['economic_npv_usd']:,.2f}** | USD | Discounted Cash Flow at 10% discount rate |",
            f"| **Optimizer Fitness Score** | `{kpis['objective_fitness_score']:.4e}` | dimensionless | Internal objective value used by optimizer |",
            f"| **Breakthrough Penalty Factor** | `{kpis['breakthrough_impact_penalty']:.4f}` | factor | {bt_note} |",
            f"| **Recovery Factor (RF)** | **{kpis['recovery_factor_percent']:.2f}%** | % OOIP | Expected literature range: 30% - 65% |",
            f"| **Cumulative Oil Produced** | **{kpis['cumulative_oil_stb']:,.0f}** | STB | Total tertiary oil volume |",
            f"| **Cumulative Water Produced** | {kpis['cumulative_water_stb']:,.0f} | STB | Total brine produced |",
            f"| **Cumulative Total Gas Produced** | {kpis['cumulative_total_gas_mscf']:,.0f} | MSCF | Separator total off-gas |",
            f"| **Cumulative CO₂ Produced** | {kpis['cumulative_co2_produced_mscf']:,.0f} | MSCF | Produced CO₂ stream to recycling plant |",
            f"| **Net CO₂ Utilization** | **{kpis['co2_utilization_tonne_per_stb']:.3f}** | tonne/STB | Standard EOR range: 0.25 - 0.50 tonne/bbl |",
            f"| **CO₂ Storage Retention (Purchased)** | **{kpis['purchased_storage_efficiency']*100:.1f}%** | % purchased | DOE/NETL benchmark: > 80% |",
            f"| **CO₂ Storage Retention (Gross)** | {kpis['gross_storage_efficiency']*100:.1f}% | % injected | Includes recycled stream in denominator |",
            f"| **Breakthrough Time** | **{kpis['breakthrough_time_years']:.2f}** | years | Time to solvent breakthrough at producers |",
            f"| **Class VI Ecology Compliant** | {'✅ YES' if kpis['ecology_compliant'] else '❌ NO'} | boolean | Geomechanical & containment criteria |",
            "",
            "## 2. Decision Variables (Optimized Operating Policy)",
            "",
            "| Parameter | Optimal Value | Physical Unit | Search Bounds | Boundary Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        for p, info in sorted(dvars.items()):
            bounds_str = (
                f"[{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]"
                if info["lower_bound"] is not None
                else "Unbounded"
            )
            b_status = "Interior"
            if info.get("is_at_lower_bound"):
                b_status = "⚠️ **At Lower Bound**"
            elif info.get("is_at_upper_bound"):
                b_status = "⚠️ **At Upper Bound**"

            val_str = f"{info['value']:,}" if isinstance(info["value"], int) else f"{info['value']:.4f}"
            lines.append(
                f"| `{p}` | **{val_str}** | {info['unit']} | {bounds_str} | {b_status} |"
            )

        lines.extend(
            [
                "",
                "## 3. Geomechanical & Subsurface Safety Checklist",
                "",
                "| Criterion | Evaluated Value | Limit / Standard | Status |",
                "| :--- | :---: | :---: | :---: |",
                f"| **Reservoir Injection Pressure** | **{safety['optimized_reservoir_pressure_psi']:.1f} psi** | ≤ {safety['safe_fracture_ceiling_psi']:.1f} psi (90% Pfrac) | {'✅ COMPLIANT' if not safety['fracture_ceiling_violated'] else '🔴 VIOLATION'} |",
                f"| **Fracture Safety Margin** | {safety['pressure_margin_psi']:.1f} psi | > 100 psi recommended | {'✅ SAFE' if safety['pressure_margin_psi'] >= 100 else '🟡 NARROW'} |",
                f"| **Formation Fracture Pressure** | {safety['caprock_fracture_pressure_psi']:.1f} psi | Caprock threshold | Baseline |",
                f"| **Material Balance Closure** | **{carbon['mass_balance_closure_percent']:.2f}%** | ≥ 99.9% conservation | {'✅ CLOSED' if carbon['mass_balance_closure_percent'] >= 99.9 else '🟡 RESIDUAL DETECTED'} |",
                f"| **Modeled Subsurface Leakage** | {carbon['total_leakage_tonne']:,.1f} tonnes | < 1.0% of injection | {'✅ SECURE' if carbon['total_leakage_tonne'] <= 0.01 * max(carbon['total_injected_tonne'], 1.0) else '🔴 LEAKAGE ALERT'} |",
                "",
                "## 4. Carbon Mass Balance & Storage Accounting",
                "",
                "| Accounting Stream | Mass (Metric Tonnes) | Volume Equivalent (MSCF @ 0.053 t/MSCF) | Fraction of Gross Injected |",
                "| :--- | :---: | :---: | :---: |",
                f"| **Gross Injected CO₂** | **{carbon['total_gross_injected_tonne']:,.1f}** | {carbon['total_gross_injected_tonne']/0.053:,.0f} | 100.0% |",
                f"| ├─ *Purchased Fresh CO₂* | *{carbon['total_purchased_tonne']:,.1f}* | {carbon['total_purchased_tonne']/0.053:,.0f} | {(carbon['total_purchased_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.1f}% |",
                f"| └─ *Recycled Re-injected CO₂* | *{carbon['total_recycled_tonne']:,.1f}* | {carbon['total_recycled_tonne']/0.053:,.0f} | {(carbon['total_recycled_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.1f}% |",
                f"| **Net Permanent Subsurface Storage** | **{carbon['total_net_stored_tonne']:,.1f}** | {carbon['total_net_stored_tonne']/0.053:,.0f} | {(carbon['total_net_stored_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.1f}% |",
                f"| **Total Produced CO₂** | **{carbon['total_produced_tonne']:,.1f}** | {carbon['total_produced_tonne']/0.053:,.0f} | {(carbon['total_produced_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.1f}% |",
                f"| ├─ *Recycled Stream* | *{carbon['total_recycled_tonne']:,.1f}* | {carbon['total_recycled_tonne']/0.053:,.0f} | {(carbon['total_recycled_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.1f}% |",
                f"| └─ *Uncaptured / Lost to Surface* | *{carbon['total_uncaptured_tonne']:,.1f}* | {carbon['total_uncaptured_tonne']/0.053:,.0f} | {(carbon['total_uncaptured_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.2f}% |",
                f"| **Modeled Fault/Caprock Leakage** | {carbon['total_leakage_tonne']:,.1f} | {carbon['total_leakage_tonne']/0.053:,.0f} | {(carbon['total_leakage_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.2f}% |",
                f"| **Unaccounted Closure Error** | {carbon['mass_balance_error_tonne']:,.1f} | {carbon['mass_balance_error_tonne']/0.053:,.0f} | {(carbon['mass_balance_error_tonne']/max(carbon['total_gross_injected_tonne'],1e-6))*100:.2f}% |",
                "",
                "## 5. Reservoir & Fluid Characterization Context",
                "",
                "### 5.1 Reservoir Rock Properties",
                "",
                "| Property | Value | Unit | Engineering Relevance |",
                "| :--- | :---: | :---: | :--- |",
                f"| **Original Oil in Place (OOIP)** | {res['ooip_stb']:,.0f} | STB | Baseline field hydrocarbon pore volume |",
                f"| **Minimum Miscibility Pressure (MMP)** | {res['mmp_psi']:.1f} | psia | Pure CO₂ thermodynamic miscibility threshold |",
                f"| **Operating Miscibility Regime** | **{res['miscibility_status']}** | status | Displacement mode (Miscible / Immiscible) |",
                f"| **Average Porosity** | {res['average_porosity']*100:.1f}% | fraction | Reservoir storage capacity |",
                f"| **Average Permeability** | {res['permeability_md']:.1f} | mD | Fluid transmissibility |",
            ]
        )
        if res.get("initial_pressure_psi"):
            lines.append(f"| **Initial Reservoir Pressure** | {res['initial_pressure_psi']:.1f} | psia | Discovery datum pressure |")
        if res.get("temperature_f"):
            lines.append(f"| **Reservoir Temperature** | {res['temperature_f']:.1f} | °F | Formation thermal regime |")
        if res.get("thickness_ft"):
            lines.append(f"| **Formation Net Thickness** | {res['thickness_ft']:.1f} | ft | Net pay zone thickness |")
        if res.get("area_acres"):
            lines.append(f"| **Reservoir Area** | {res['area_acres']:.1f} | acres | Drainage area footprint |")
        if res.get("total_wells"):
            lines.append(f"| **Well Infrastructure** | {res['total_wells']} wells ({res.get('injector_count', 0)} inj / {res.get('producer_count', 0)} prod) | count | Field pattern well arrangement |")

        # 5.2 Fluid & PVT Properties from manifest
        fluid_props = manifest.get("fluid_properties", {})
        if fluid_props:
            lines.extend([
                "",
                "### 5.2 Fluid & PVT Properties",
                "",
                "| Property | Value | Unit | Engineering Relevance |",
                "| :--- | :---: | :---: | :--- |",
            ])
            _pvt_display = [
                ("oil_api_gravity", "Oil API Gravity", "°API", "Stock-tank oil gravity"),
                ("gas_gravity", "Gas Gravity", "air=1.0", "Separator gas specific gravity"),
                ("bubble_point_pressure_psi", "Bubble Point Pressure", "psia", "Saturation pressure (Pb)"),
                ("solution_gor_scf_stb", "Solution GOR", "SCF/STB", "Dissolved gas-oil ratio at Pb"),
                ("oil_viscosity_cp", "Oil Viscosity", "cp", "Dead/live oil dynamic viscosity"),
                ("water_viscosity_cp", "Water Viscosity", "cp", "Formation brine viscosity"),
                ("gas_viscosity_cp", "Gas Viscosity", "cp", "Hydrocarbon gas viscosity"),
                ("oil_fvf_rb_stb", "Oil FVF (Bo)", "RB/STB", "Oil formation volume factor"),
                ("water_fvf_rb_stb", "Water FVF (Bw)", "RB/STB", "Water formation volume factor"),
                ("gas_fvf_rb_mscf", "Gas FVF (Bg)", "RB/MSCF", "Gas formation volume factor"),
                ("oil_compressibility_psi", "Oil Compressibility", "1/psi", "Isothermal oil compressibility"),
                ("water_compressibility_psi", "Water Compressibility", "1/psi", "Isothermal water compressibility"),
            ]
            for key, label, unit, relevance in _pvt_display:
                if key in fluid_props and fluid_props[key] is not None:
                    val = fluid_props[key]
                    val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
                    lines.append(f"| **{label}** | {val_str} | {unit} | {relevance} |")

        # 5.3 Operational & Well Context from manifest
        op_ctx = manifest.get("operational_context", {})
        if op_ctx:
            lines.extend([
                "",
                "### 5.3 Operational & Well Constraints",
                "",
                "| Parameter | Value | Unit | Engineering Relevance |",
                "| :--- | :---: | :---: | :--- |",
            ])
            _op_display = [
                ("project_lifetime_years", "Project Lifetime", "years", "Economic horizon"),
                ("max_injection_pressure_psi", "Max Injection Pressure", "psia", "Surface injection limit"),
                ("min_producer_bhp_psi", "Min Producer BHP", "psia", "Bottom-hole flowing pressure floor"),
                ("max_production_rate_stbd", "Max Liquid Production Rate", "STB/day", "Surface handling constraint"),
                ("well_shut_in_threshold_bpd", "Well Shut-in Threshold", "STB/day", "Economic limit for individual well"),
                ("discount_rate", "Discount Rate", "fraction", "DCF time-value discounting"),
                ("time_resolution", "Time Resolution", "mode", "Yearly / Monthly simulation stepping"),
            ]
            for key, label, unit, relevance in _op_display:
                if key in op_ctx and op_ctx[key] is not None:
                    val = op_ctx[key]
                    val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
                    lines.append(f"| **{label}** | {val_str} | {unit} | {relevance} |")

        lines.extend(
            [
                "",
                "---",
                "*Report generated by CO₂-EOR Optimizer.*",
            ]
        )
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Tabular CSV Data Builders
    # -------------------------------------------------------------------------

    def _build_cash_flow_table(
        self,
        results: Dict[str, Any],
        engine: Optional[Any],
        input_parameters: Dict[str, Any],
    ) -> pd.DataFrame:
        """Construct year-by-year financial schedule with explicit revenues and costs."""
        profiles = results.get("optimized_profiles", {}) or {}
        oil_arr = profiles.get("yearly_oil_stb", profiles.get("annual_oil_stb", []))
        if len(oil_arr) == 0:
            return pd.DataFrame()

        n_years = len(oil_arr)
        years = np.arange(1, n_years + 1)

        co2_purchased_mscf = np.array(
            profiles.get(
                "yearly_co2_purchased_mscf",
                profiles.get("annual_co2_purchased_mscf", np.zeros(n_years)),
            )
        )
        co2_recycled_mscf = np.array(
            profiles.get(
                "yearly_co2_recycled_mscf",
                profiles.get("annual_co2_recycled_mscf", np.zeros(n_years)),
            )
        )

        # Economic parameters
        econ_params = input_parameters.get("Economic Parameters", {})
        oil_price = float(
            econ_params.get(
                "oil_price_usd_per_bbl",
                getattr(getattr(engine, "economic_params", None), "oil_price_usd_per_bbl", 70.0),
            )
        )
        co2_purch_cost = float(
            econ_params.get(
                "co2_purchase_cost_usd_per_tonne",
                getattr(
                    getattr(engine, "economic_params", None),
                    "co2_purchase_cost_usd_per_tonne",
                    50.0,
                ),
            )
        )
        co2_recycle_cost = float(
            econ_params.get(
                "co2_recycle_cost_usd_per_tonne",
                getattr(
                    getattr(engine, "economic_params", None),
                    "co2_recycle_cost_usd_per_tonne",
                    15.0,
                ),
            )
        )
        storage_credit_rate = float(
            econ_params.get(
                "co2_storage_credit_usd_per_tonne",
                getattr(
                    getattr(engine, "economic_params", None),
                    "co2_storage_credit_usd_per_tonne",
                    25.0,
                ),
            )
        )
        var_opex = float(
            econ_params.get(
                "variable_opex_usd_per_bbl",
                getattr(getattr(engine, "economic_params", None), "variable_opex_usd_per_bbl", 5.0),
            )
        )
        fixed_opex = float(
            econ_params.get(
                "fixed_opex_usd_per_year",
                getattr(
                    getattr(engine, "economic_params", None), "fixed_opex_usd_per_year", 200000.0
                ),
            )
        )
        capex = float(
            econ_params.get(
                "capex_usd",
                getattr(
                    getattr(engine, "economic_params", None), "initial_investment_usd", 5000000.0
                ),
            )
        )
        discount_rate = float(
            econ_params.get(
                "discount_rate_fraction",
                getattr(
                    getattr(engine, "economic_params", None), "discount_rate_fraction", 0.10
                ),
            )
        )
        co2_density = float(
            input_parameters.get("EOR Parameters", {}).get(
                "co2_density_tonne_per_mscf",
                getattr(getattr(engine, "eor_params", None), "co2_density_tonne_per_mscf", 0.053),
            )
        )

        # Streams
        oil_rev = np.array(oil_arr) * oil_price
        purch_cost = co2_purchased_mscf * co2_density * co2_purch_cost
        rec_cost = co2_recycled_mscf * co2_density * co2_recycle_cost
        operating_cost = np.array(oil_arr) * var_opex + fixed_opex
        storage_credit = co2_purchased_mscf * co2_density * storage_credit_rate

        annual_capex = np.zeros(n_years)
        annual_capex[0] = capex

        net_cash_flow = oil_rev + storage_credit - purch_cost - rec_cost - operating_cost
        net_cash_flow[0] -= capex

        # Industry standard mid-year discounting
        discount_factors = 1.0 / ((1.0 + discount_rate) ** (years - 0.5))
        dcf = net_cash_flow * discount_factors
        cum_npv = np.cumsum(dcf)

        return pd.DataFrame(
            {
                "Year": years,
                "Oil_Production_STB": np.round(oil_arr, 2),
                "CO2_Purchased_MSCF": np.round(co2_purchased_mscf, 2),
                "CO2_Recycled_MSCF": np.round(co2_recycled_mscf, 2),
                "Oil_Revenue_USD": np.round(oil_rev, 2),
                "Storage_Credit_Revenue_USD": np.round(storage_credit, 2),
                "CO2_Purchase_Cost_USD": np.round(purch_cost, 2),
                "CO2_Recycling_OPEX_USD": np.round(rec_cost, 2),
                "Operating_Cost_USD": np.round(operating_cost, 2),
                "CAPEX_USD": np.round(annual_capex, 2),
                "Net_Cash_Flow_USD": np.round(net_cash_flow, 2),
                "Discount_Factor": np.round(discount_factors, 4),
                "Discounted_Cash_Flow_USD": np.round(dcf, 2),
                "Cumulative_NPV_USD": np.round(cum_npv, 2),
            }
        )

    def _build_convergence_table(
        self,
        results: Dict[str, Any],
        convergence_live_data: List[Dict[str, float]],
    ) -> pd.DataFrame:
        """Extract step-by-step optimizer convergence trajectory."""
        if convergence_live_data:
            steps = []
            best_f = []
            curr_f = []
            elapsed = []
            for i, d in enumerate(convergence_live_data):
                step = d.get("iteration", d.get("generation", i + 1))
                steps.append(int(step))
                best_f.append(d.get("best_fitness", np.nan))
                curr_f.append(d.get("current_fitness", d.get("objective_value", np.nan)))
                elapsed.append(d.get("elapsed_time", np.nan))

            return pd.DataFrame(
                {
                    "Step": steps,
                    "Best_Fitness": best_f,
                    "Current_Fitness": curr_f,
                    "Elapsed_Seconds": elapsed,
                }
            )

        # Fallback to bo_statistics or ga_statistics
        bo_stats = results.get("bo_statistics") or results.get("bo_results", {}) or {}
        if "objective_history" in bo_stats and len(bo_stats["objective_history"]) > 0:
            hist = bo_stats["objective_history"]
            evals = np.arange(1, len(hist) + 1)
            best_so_far = np.maximum.accumulate(hist)
            return pd.DataFrame(
                {
                    "Evaluation": evals,
                    "Current_Objective": np.round(hist, 4),
                    "Best_Objective": np.round(best_so_far, 4),
                }
            )

        ga_stats = (
            results.get("ga_statistics")
            or results.get("ga_full_results_for_hybrid", {}).get("ga_statistics", {})
            or {}
        )
        if "best_fitness_history" in ga_stats and len(ga_stats["best_fitness_history"]) > 0:
            hist = ga_stats["best_fitness_history"]
            gens = np.arange(1, len(hist) + 1)
            return pd.DataFrame(
                {
                    "Generation": gens,
                    "Best_Fitness": np.round(hist, 4),
                }
            )

        # General history keys in results
        for hist_key in ["convergence_history", "fitness_history", "history"]:
            if hist_key in results and len(results[hist_key]) > 0:
                hist = results[hist_key]
                steps = np.arange(1, len(hist) + 1)
                return pd.DataFrame(
                    {
                        "Step": steps,
                        "Best_Fitness": np.round(hist, 4),
                    }
                )

        # Fallback single step so convergence_history.csv is always generated
        obj_val = float(results.get("objective_function_value", results.get("npv", 0.0)))
        return pd.DataFrame(
            {
                "Step": [1],
                "Best_Fitness": [np.round(obj_val, 4)],
            }
        )

    def _build_profile_tables(
        self,
        results: Dict[str, Any],
        engine: Optional[Any],
    ) -> Dict[str, pd.DataFrame]:
        """Construct standardized yearly and monthly profile tables with explicit units."""
        profiles = results.get("optimized_profiles", {}) or {}
        mb_analysis = results.get("material_balance_analysis") or {}
        mb_data = mb_analysis.get("material_balance_data", {}) or {}

        tables = {}

        # 1. Yearly table
        yearly_oil = profiles.get("yearly_oil_stb", profiles.get("annual_oil_stb", []))
        if len(yearly_oil) > 0:
            n_y = len(yearly_oil)
            df_y = pd.DataFrame(
                {
                    "Year": np.arange(1, n_y + 1),
                    "Oil_Production_STB": np.round(yearly_oil, 2),
                    "Water_Production_STB": np.round(
                        profiles.get("yearly_water_stb", profiles.get("annual_water_stb", np.zeros(n_y))),
                        2,
                    ),
                    "Total_Gas_Produced_MSCF": np.round(
                        profiles.get(
                            "yearly_total_gas_mscf",
                            profiles.get("annual_total_gas_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "CO2_Produced_MSCF": np.round(
                        profiles.get(
                            "yearly_co2_produced_mscf",
                            profiles.get("annual_co2_produced_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "HC_Gas_Produced_MSCF": np.round(
                        profiles.get(
                            "yearly_hc_gas_produced_mscf",
                            profiles.get("annual_hc_gas_produced_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "Reservoir_Pressure_psia": np.round(
                        profiles.get(
                            "yearly_pressure", profiles.get("annual_pressure", np.zeros(n_y))
                        ),
                        2,
                    ),
                    "CO2_Purchased_MSCF": np.round(
                        profiles.get(
                            "yearly_co2_purchased_mscf",
                            profiles.get("annual_co2_purchased_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "CO2_Recycled_MSCF": np.round(
                        profiles.get(
                            "yearly_co2_recycled_mscf",
                            profiles.get("annual_co2_recycled_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "CO2_Injected_MSCF": np.round(
                        profiles.get(
                            "yearly_co2_injected_mscf",
                            profiles.get("annual_co2_injected_mscf", np.zeros(n_y)),
                        ),
                        2,
                    ),
                    "Water_Injected_STB": np.round(
                        profiles.get(
                            "yearly_water_injected_bbl",
                            profiles.get("annual_water_injected_bbl", np.zeros(n_y)),
                        ),
                        2,
                    ),
                }
            )

            # Append Material Balance columns if available
            if mb_data:
                for mb_col in [
                    "injected_tonne",
                    "produced_tonne",
                    "recycled_tonne",
                    "net_stored_tonne",
                    "cumulative_stored_tonne",
                    "annual_leakage_tonne",
                    "storage_efficiency",
                ]:
                    if mb_col in mb_data and len(mb_data[mb_col]) == n_y:
                        header_name = f"MB_{mb_col.replace('_', ' ').title().replace(' ', '_')}"
                        df_y[header_name] = np.round(mb_data[mb_col], 4)

            tables["yearly"] = df_y

        # 2. Monthly table (if available)
        monthly_oil = profiles.get("monthly_oil_stb", [])
        if len(monthly_oil) > 0:
            n_m = len(monthly_oil)
            df_m = pd.DataFrame(
                {
                    "Month": np.arange(1, n_m + 1),
                    "Time_Years": np.round(
                        profiles.get("monthly_time_years", np.arange(1, n_m + 1) / 12.0), 3
                    ),
                    "Oil_Production_STB": np.round(monthly_oil, 2),
                    "Water_Production_STB": np.round(
                        profiles.get("monthly_water_stb", np.zeros(n_m)), 2
                    ),
                    "Total_Gas_Produced_MSCF": np.round(
                        profiles.get("monthly_total_gas_mscf", np.zeros(n_m)), 2
                    ),
                    "CO2_Produced_MSCF": np.round(
                        profiles.get("monthly_co2_produced_mscf", np.zeros(n_m)), 2
                    ),
                    "Reservoir_Pressure_psia": np.round(
                        profiles.get("monthly_pressure", np.zeros(n_m)), 2
                    ),
                    "CO2_Purchased_MSCF": np.round(
                        profiles.get("monthly_co2_purchased_mscf", np.zeros(n_m)), 2
                    ),
                    "CO2_Injected_MSCF": np.round(
                        profiles.get("monthly_co2_injected_mscf", np.zeros(n_m)), 2
                    ),
                    "Water_Injected_STB": np.round(
                        profiles.get("monthly_water_injected_bbl", np.zeros(n_m)), 2
                    ),
                }
            )
            tables["monthly"] = df_m

        return tables

    # -------------------------------------------------------------------------
    # Sanitized Text & JSON Output
    # -------------------------------------------------------------------------

    def _build_sanitized_text_summary(
        self,
        manifest: Dict[str, Any],
        diagnostics: List[Dict[str, str]],
        results: Dict[str, Any],
        input_parameters: Dict[str, Any],
    ) -> str:
        """Generate results_summary.txt without messy raw numpy array dumps."""
        out = io.StringIO()
        meta = manifest["metadata"]
        kpis = manifest["key_performance_indicators"]
        carbon = manifest["carbon_accounting_tonnes"]
        safety = manifest["geomechanical_safety"]
        dvars = manifest["decision_variables"]

        out.write("=== CO₂ EOR OPTIMIZATION RUN SUMMARY ===\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Method: {meta['optimizer_method']}\n")
        out.write(f"Objective: {meta['objective_name']}\n")
        out.write(f"Evaluations: {meta['total_evaluations']} | Duration: {meta['total_duration_seconds']:.2f}s\n\n")

        # Diagnostics section
        out.write("=== RUN HEALTH & DIAGNOSTICS ===\n")
        if not diagnostics:
            out.write("Status: PASS (No critical safety or operational warnings)\n\n")
        else:
            for d in diagnostics:
                out.write(f"[{d['severity']}] {d['category']}: {d['message']}\n")
            out.write("\n")

        # Key Results
        out.write("=== OPTIMIZATION RESULTS ===\n")
        out.write(f"Final Economic NPV: ${kpis['economic_npv_usd']:,.2f}\n")
        out.write(f"Objective Fitness Score: {kpis['objective_fitness_score']:.6e}\n")
        out.write(f"Breakthrough Impact Penalty: {kpis['breakthrough_impact_penalty']:.4f}\n")
        out.write(f"Ultimate Recovery Factor: {kpis['recovery_factor_percent']:.2f}%\n")
        out.write(f"Cumulative Oil Production: {kpis['cumulative_oil_stb']:,.0f} STB\n")
        out.write(f"Cumulative Water Production: {kpis['cumulative_water_stb']:,.0f} STB\n")
        out.write(f"Cumulative Total Gas: {kpis['cumulative_total_gas_mscf']:,.0f} MSCF\n")
        out.write(f"Cumulative CO2 Produced: {kpis['cumulative_co2_produced_mscf']:,.0f} MSCF\n")
        out.write(f"Net CO2 Utilization: {kpis['co2_utilization_tonne_per_stb']:.4f} tonne/STB ({kpis['co2_utilization_mscf_per_stb']:.2f} MSCF/STB)\n")
        out.write(f"Purchased Storage Efficiency: {kpis['purchased_storage_efficiency']*100:.2f}%\n")
        out.write(f"Gross Storage Efficiency: {kpis['gross_storage_efficiency']*100:.2f}%\n")
        out.write(f"Breakthrough Timing: {kpis['breakthrough_time_years']:.2f} years\n")
        out.write(f"Ecology Compliant: {'YES' if kpis['ecology_compliant'] else 'NO'}\n\n")

        # Optimized Parameters
        out.write("=== OPTIMIZED DECISION VARIABLES ===\n")
        for p, info in sorted(dvars.items()):
            bound_note = ""
            if info.get("is_at_lower_bound"):
                bound_note = " [PINNED AT LOWER BOUND]"
            elif info.get("is_at_upper_bound"):
                bound_note = " [PINNED AT UPPER BOUND]"
            out.write(f"{p}: {info['value']} {info['unit']}{bound_note}\n")
        out.write("\n")

        # Geomechanical Check
        out.write("=== GEOMECHANICAL SAFETY CHECK ===\n")
        out.write(f"Optimized Reservoir Pressure: {safety['optimized_reservoir_pressure_psi']:.1f} psi\n")
        out.write(f"Safe Fracture Ceiling (90% Pfrac): {safety['safe_fracture_ceiling_psi']:.1f} psi\n")
        out.write(f"Pressure Margin: {safety['pressure_margin_psi']:.1f} psi\n")
        out.write(f"Status: {'SAFE' if not safety['fracture_ceiling_violated'] else 'VIOLATED'}\n\n")

        # Carbon Storage
        out.write("=== CARBON MASS BALANCE SUMMARY ===\n")
        out.write(f"Gross CO2 Injected: {carbon['total_gross_injected_tonne']:,.1f} tonnes\n")
        out.write(f"  - Purchased CO2: {carbon['total_purchased_tonne']:,.1f} tonnes\n")
        out.write(f"  - Recycled CO2: {carbon['total_recycled_tonne']:,.1f} tonnes\n")
        out.write(f"Total CO2 Produced: {carbon['total_produced_tonne']:,.1f} tonnes\n")
        out.write(f"  - Recycled Portion: {carbon['total_recycled_tonne']:,.1f} tonnes\n")
        out.write(f"  - Uncaptured Surface Portion: {carbon['total_uncaptured_tonne']:,.1f} tonnes\n")
        out.write(f"Total Net Stored: {carbon['total_net_stored_tonne']:,.1f} tonnes\n")
        out.write(f"Total Modeled Leakage: {carbon['total_leakage_tonne']:,.1f} tonnes\n")
        out.write(f"Mass Balance Closure: {carbon['mass_balance_closure_percent']:.2f}%\n\n")

        # Input Parameters & Operational Setup
        if input_parameters:
            out.write("=== SIMULATION & OPTIMIZATION INPUT PARAMETERS ===\n")
            setup = input_parameters.get("Optimization Setup", {})
            if setup:
                out.write("--- Optimization & Engine Setup ---\n")
                for k, v in sorted(setup.items()):
                    out.write(f"  {k}: {v}\n")
            bounds = input_parameters.get("Optimization Search Bounds", {})
            if bounds:
                out.write("\n--- Parameter Search Space Bounds ---\n")
                for k, v in sorted(bounds.items()):
                    if k.endswith("_range"):
                        out.write(f"  {k[:-6]}: {v}\n")
            res_ctx = manifest.get("reservoir_context", {})
            if res_ctx:
                out.write("\n--- Reservoir Characterization ---\n")
                for k, v in sorted(res_ctx.items()):
                    out.write(f"  {k}: {v}\n")
            fluid_pvt = input_parameters.get("Fluid & PVT Properties", {})
            if fluid_pvt:
                out.write("\n--- Fluid & PVT Properties ---\n")
                _pvt_priority = [
                    "oil_api_gravity", "gas_gravity", "bubble_point_pressure_psi",
                    "solution_gor_scf_stb", "oil_viscosity_cp", "water_viscosity_cp",
                    "gas_viscosity_cp", "oil_fvf_rb_stb", "water_fvf_rb_stb",
                    "gas_fvf_rb_mscf", "oil_compressibility_psi", "water_compressibility_psi",
                ]
                _printed = set()
                for k in _pvt_priority:
                    if k in fluid_pvt:
                        out.write(f"  {k}: {fluid_pvt[k]}\n")
                        _printed.add(k)
                for k, v in sorted(fluid_pvt.items()):
                    if k not in _printed:
                        out.write(f"  {k}: {v}\n")
            op_params = input_parameters.get("Operational Parameters", {})
            if op_params:
                out.write("\n--- Operational Parameters ---\n")
                for k, v in sorted(op_params.items()):
                    out.write(f"  {k}: {v}\n")
            eor_params = input_parameters.get("EOR Parameters", {})
            if eor_params:
                out.write("\n--- EOR Parameters ---\n")
                for k, v in sorted(eor_params.items()):
                    out.write(f"  {k}: {v}\n")
            econ_params = input_parameters.get("Economic Parameters", {})
            if econ_params:
                out.write("\n--- Economic Parameters ---\n")
                for k, v in sorted(econ_params.items()):
                    out.write(f"  {k}: {v}\n")
            co2_store = input_parameters.get("CO2 Storage Parameters", {})
            if co2_store:
                out.write("\n--- CO2 Storage Parameters ---\n")
                for k, v in sorted(co2_store.items()):
                    out.write(f"  {k}: {v}\n")
            wells = input_parameters.get("Well Configuration", {})
            if wells:
                out.write("\n--- Well Configuration ---\n")
                for k, v in sorted(wells.items()):
                    out.write(f"  {k}: {v}\n")
            mmp_conf = input_parameters.get("MMP Analysis Configuration", {})
            if mmp_conf:
                out.write("\n--- MMP Analysis Settings ---\n")
                for k, v in sorted(mmp_conf.items()):
                    out.write(f"  {k}: {v}\n")
            algo_cats = [
                ("Genetic Algorithm", "GA Hyperparameters"),
                ("Bayesian Optimization", "BO Hyperparameters"),
                ("Particle Swarm Optimization", "PSO Hyperparameters"),
                ("Differential Evolution", "DE Hyperparameters"),
            ]
            for cat_key, label in algo_cats:
                algo_params = input_parameters.get(cat_key, {})
                if algo_params:
                    out.write(f"\n--- {label} ---\n")
                    for k, v in sorted(algo_params.items()):
                        out.write(f"  {k}: {v}\n")
            out.write("\n")

        out.write("=== END OF SUMMARY ===\n")
        return out.getvalue()

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    def _save_plots(self, plots_generator: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Save plot figures as high-resolution PNGs."""
        for plot_name, plot_info in plots_generator.items():
            try:
                fig_func = plot_info.get("func")
                if callable(fig_func):
                    fig = fig_func(results)
                    if fig:
                        image_path = self.export_dir / f"{plot_name}.png"
                        width = plot_info.get("width", 1200)
                        height = plot_info.get("height", 800)
                        if hasattr(fig, "write_image"):
                            fig.write_image(str(image_path), width=width, height=height)
            except Exception as e:
                logger.warning(f"Failed to generate plot '{plot_name}': {e}")

    def _extract_safe_dict(self, source: Any) -> Dict[str, Any]:
        """Extract a flat dict of JSON-safe scalar values from an input parameter dict.

        Filters out large numpy arrays, non-serializable objects, and nested dataclasses
        to produce a compact dictionary suitable for manifest embedding.
        """
        if not isinstance(source, dict):
            return {}
        result: Dict[str, Any] = {}
        for k, v in source.items():
            if v is None:
                continue
            if isinstance(v, (bool, np.bool_)):
                result[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                result[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                if not (np.isnan(v) or np.isinf(v)):
                    result[k] = float(v)
            elif isinstance(v, str):
                result[k] = v
            elif isinstance(v, (list, tuple)):
                if len(v) <= 20:
                    result[k] = v
            elif isinstance(v, dict):
                if len(v) <= 20:
                    result[k] = v
            elif hasattr(v, "tolist"):
                arr = np.asarray(v)
                if arr.size <= 10:
                    result[k] = arr.tolist()
                else:
                    result[f"{k}_shape"] = list(arr.shape)
            # Skip non-serializable objects silently
        return result

    def _sanitize_dict(self, obj: Any) -> Any:
        """Clean object recursively for strict JSON compliance."""
        if obj is None:
            return None
        if isinstance(obj, (bool, np.bool_)) or type(obj) is bool:
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if is_dataclass(obj):
            return self._sanitize_dict(asdict(obj))
        if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
            try:
                return self._sanitize_dict(obj.tolist())
            except Exception:
                pass
        if isinstance(obj, dict):
            return {
                str(k): self._sanitize_dict(v)
                for k, v in obj.items()
                if not str(k).startswith("_")
            }
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [self._sanitize_dict(x) for x in obj]

        obj_type_name = type(obj).__name__
        if (
            obj_type_name
            in [
                "Figure",
                "FigureCanvasAgg",
                "PyGAD",
                "GA",
                "BayesianOptimization",
                "Axes",
                "QObject",
                "QWidget",
            ]
            or callable(obj)
        ):
            return f"<{obj_type_name} not serialized>"

        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError, ValueError):
            return f"<{obj_type_name} not serialized>"

    def _safe_sum(self, val: Any) -> float:
        """Safely calculate sum of array or list."""
        if val is None:
            return 0.0
        try:
            arr = np.asarray(val, dtype=float)
            return float(np.sum(arr[np.isfinite(arr)]))
        except Exception:
            return 0.0

    def _write_file(self, filename: str, content: str) -> None:
        """Write text file UTF-8."""
        with open(self.export_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_json(self, filename: str, data: Any) -> None:
        """Write JSON file UTF-8 safely."""
        with open(self.export_dir / filename, "w", encoding="utf-8") as f:
            json.dump(self._sanitize_dict(data), f, indent=2, default=str)
