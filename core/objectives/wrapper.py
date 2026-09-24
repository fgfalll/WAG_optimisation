from typing import Dict, Any, Optional
import logging
import numpy as np
from core.data_models import EconomicParameters
# Standard exception for optimization objective errors
OptimizationError = RuntimeError

logger = logging.getLogger(__name__)
from core.objectives.storage import (
    calculate_co2_storage_efficiency,
)


def _get_param(obj, key, default=None):
    """Get parameter from dict or object with attribute access."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ObjectiveFunctions:
    def __init__(self, operational_params, eor_params, reservoir, advanced_params):
        self.operational_params = operational_params
        self.eor_params = eor_params
        self.reservoir = reservoir
        self.advanced_params = advanced_params

    def _calculate_objective_functions(
        self, profiles, recovery_factor, econ_params, storage_params, simulation_mode=None
    ):
        """
        Calculates various objective metrics based on simulation profiles.

        Args:
            profiles: Dictionary containing production/injection profiles
            recovery_factor: Calculated recovery factor
            econ_params: Economic parameters dictionary
            storage_params: Storage parameters dictionary

        Returns:
            Dictionary with objective function values
        """
        results = {}

        # 1. Recovery Factor (passed in)
        results["recovery_factor"] = recovery_factor

        # 2. NPV - must be provided by surrogate engine (single source of truth)
        # Wrapper is a dumb consumer - just read from engine output
        if "npv" not in profiles:
            raise ValueError("npv not provided by engine - invalid simulation run")
        results["npv"] = float(profiles["npv"])
        results["npv_details"] = {"method": "surrogate_engine"}

        # Geomechanical Sandface Pressure & Containment Loss Penalty
        # Evaluates near-wellbore sandface injection pressure against EPA Class VI 90% limit
        caprock_p = getattr(self.eor_params, "caprock_fracture_pressure_psi", 5500.0)
        safety_factor = getattr(self.eor_params, "caprock_safety_factor", 0.90)
        safe_fracture_limit = caprock_p * safety_factor

        pressure_profile = profiles.get("pressure", profiles.get("reservoir_pressure"))
        if pressure_profile is not None and len(pressure_profile) > 0:
            max_res_p = float(np.max(pressure_profile))
            inj_rate = getattr(self.eor_params, "injection_rate", 5000.0)
            ii = getattr(self.eor_params, "injectivity_index", 25.0)
            delta_p_inj = inj_rate / max(ii, 1.0)
            p_sandface = max_res_p + delta_p_inj

            if p_sandface > safe_fracture_limit:
                overpressure = p_sandface - safe_fracture_limit
                containment_penalty = 1e6 * float((overpressure / safe_fracture_limit) ** 2)
                results["npv"] -= containment_penalty
                results["geomechanical_violation"] = {
                    "p_sandface": p_sandface,
                    "safe_limit": safe_fracture_limit,
                    "overpressure": overpressure,
                    "penalty": containment_penalty,
                }

        # Environmental Leakage Penalty ($100/tonne remediation cost if CO2 migrates)
        if "total_leakage_tonne" in profiles:
            leaked_tonnes = float(profiles["total_leakage_tonne"])
        elif "annual_leakage_tonne" in profiles:
            leaked_tonnes = float(np.sum(profiles["annual_leakage_tonne"]))
        elif "leakage_rate_fraction" in profiles and float(profiles["leakage_rate_fraction"]) > 0:
            co2_purchased_arr = profiles.get(
                "annual_co2_purchased_mscf", profiles.get("yearly_co2_purchased_mscf", np.array([0.0]))
            )
            co2_purchased_sum = float(np.sum(co2_purchased_arr))
            rho = getattr(self.eor_params, "co2_density_tonne_per_mscf", 0.053)
            leaked_tonnes = co2_purchased_sum * rho * float(profiles["leakage_rate_fraction"])
        else:
            leaked_tonnes = 0.0

        if leaked_tonnes > 0:
            carbon_tax = getattr(econ_params, "carbon_tax_usd_per_tonne", 0.0) if econ_params else 0.0
            remediation_cost = max(carbon_tax, 100.0) * leaked_tonnes
            results["npv"] -= remediation_cost
            results["environmental_leakage_penalty"] = remediation_cost

        # Read CO2 purchased/recycled from engine output (single source of truth)
        co2_purchased_val = profiles.get(
            "annual_co2_purchased_mscf",
            profiles.get("yearly_co2_purchased_mscf", np.array([])),
        )
        co2_recycled_val = profiles.get(
            "annual_co2_recycled_mscf",
            profiles.get("yearly_co2_recycled_mscf", np.array([])),
        )
        results["annual_co2_purchased_mscf"] = co2_purchased_val
        results["annual_co2_recycled_mscf"] = co2_recycled_val

        # 3. Storage Efficiency - now from surrogate engine (single source of truth)
        # Wrapper is a dumb consumer - just read from engine output
        if "storage_efficiency" in profiles:
            results["storage_efficiency"] = profiles["storage_efficiency"]
            results["storage_metrics"] = {"method": "surrogate_engine"}
        elif storage_params is not None and len(profiles) > 0:
            try:
                storage_efficiency = calculate_co2_storage_efficiency(
                    profiles=profiles,
                    time_resolution=getattr(self.operational_params, "time_resolution", "annual"),
                    co2_density_tonne_per_mscf=self.eor_params.co2_density_tonne_per_mscf,
                    co2_storage_params=storage_params,
                )
                results["storage_efficiency"] = storage_efficiency
                results["storage_metrics"] = {
                    "storage_efficiency": storage_efficiency,
                    "method": "calculate_co2_storage_efficiency",
                }
            except Exception as e:
                raise OptimizationError(f"Storage efficiency calculation failed: {e}") from e
        else:
            # Physically consistent: If profiles or storage parameters are missing,
            # never fabricate artificial storage credit. Return NaN so the optimizer
            # naturally eliminates unphysical or unviable chromosomes.
            results["storage_efficiency"] = float("nan")
            results["storage_metrics"] = {"method": "unphysical_or_missing_data"}

        # 4. CO2 Utilization Calculation
        # CO2 Utilization = Total CO2 Purchased (tonnes) / Cumulative Oil Produced (bbl)
        # Lower values are better (more oil per unit CO2)
        # In primary production mode, CO2 utilization is N/A (no injection)

        if simulation_mode == "primary_production":
            results["co2_utilization"] = None
            results["co2_utilization_details"] = {"method": "not_applicable_primary_production"}
            return results

        co2_density_tonne_per_mscf = self.eor_params.co2_density_tonne_per_mscf

        co2_purchased_calc = profiles.get(
            "annual_co2_purchased_mscf",
            profiles.get("yearly_co2_purchased_mscf"),
        )
        if co2_purchased_calc is None:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {
                "method": "no_co2_purchased_data",
                "error": "neither 'annual_co2_purchased_mscf' nor 'yearly_co2_purchased_mscf' in profiles",
            }
            return results
        if not isinstance(co2_purchased_calc, np.ndarray) or len(co2_purchased_calc) == 0:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {"method": "empty_co2_purchased_array"}
            return results
        if np.sum(co2_purchased_calc) <= 0:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {"method": "zero_co2_purchased", "co2_sum": float(np.sum(co2_purchased_calc))}
            return results

        oil_produced_calc = profiles.get("annual_oil_stb", profiles.get("yearly_oil_stb"))
        if oil_produced_calc is None:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {
                "method": "no_oil_produced_data",
                "error": "neither 'annual_oil_stb' nor 'yearly_oil_stb' in profiles",
            }
            return results
        if not isinstance(oil_produced_calc, np.ndarray) or len(oil_produced_calc) == 0:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {"method": "empty_oil_produced_array"}
            return results
        if np.sum(oil_produced_calc) <= 0:
            results["co2_utilization"] = float("nan")
            results["co2_utilization_details"] = {"method": "zero_oil_produced", "oil_sum": float(np.sum(oil_produced_calc))}
            return results

        total_co2_purchased_tonne = np.sum(co2_purchased_calc) * co2_density_tonne_per_mscf
        total_oil = np.sum(oil_produced_calc)

        co2_recycled = profiles.get(
            "annual_co2_recycled_mscf",
            profiles.get("yearly_co2_recycled_mscf"),
        )
        if co2_recycled is not None and len(co2_recycled) > 0 and np.sum(co2_recycled) > 0:
            total_co2_recycled_tonne = np.sum(co2_recycled) * co2_density_tonne_per_mscf
            method = "net_purchased_excluding_recycle"
            util_details = {
                "total_co2_tonne": total_co2_purchased_tonne,
                "total_co2_recycled_tonne": total_co2_recycled_tonne,
                "total_oil_bbl": total_oil,
                "method": method,
            }
        else:
            method = "profile_based"
            util_details = {
                "total_co2_tonne": total_co2_purchased_tonne,
                "total_oil_bbl": total_oil,
                "method": method,
            }

        results["co2_utilization"] = total_co2_purchased_tonne / total_oil
        results["co2_utilization_details"] = util_details

        return results
