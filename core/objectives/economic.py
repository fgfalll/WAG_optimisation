"""
Economic objective functions - NPV and cashflow calculations.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

try:
    from numpy_financial import npv as npv_func
except ImportError:
    logging.warning("numpy_financial not found. Using manual NPV calculation.")

    def npv_func(rate, values):
        values = np.atleast_1d(values)
        return np.sum(values / (1 + rate) ** np.arange(len(values)))


logger = logging.getLogger(__name__)


def calculate_npv(
    profiles: Dict[str, np.ndarray],
    economic_params: Any,
    storage_metrics: Optional[Dict[str, np.ndarray]] = None,
    co2_storage_params: Optional[Any] = None,
    time_resolution: str = "annual",
    co2_density_tonne_per_mscf: float = 0.053,
    carbon_tax_usd_per_tonne: float = 0.0,
) -> float:
    """
    Calculate Net Present Value (NPV) for CO2-EOR project.

    NPV = Σ (Revenue_t - Costs_t - Carbon_Tax_Leakage_t) / (1 + discount_rate)^t

    Args:
        profiles: Production and injection profiles
        economic_params: Economic parameters (prices, costs, discount rate)
        storage_metrics: CO2 storage metrics (optional), contains 'annual_leakage_tonne' array
        co2_storage_params: CO2 storage parameters (optional)
        time_resolution: Time resolution key prefix
        co2_density_tonne_per_mscf: CO2 density conversion factor
        carbon_tax_usd_per_tonne: Carbon tax rate for CO2 leakage penalty

    Returns:
        Net present value in USD
    """
    oil_production = profiles.get(f"{time_resolution}_oil_stb", np.array([0]))
    co2_purchased = profiles.get(f"{time_resolution}_co2_purchased_mscf", np.array([0]))
    co2_recycled = profiles.get(f"{time_resolution}_co2_recycled_mscf", np.array([0]))

    # Revenues
    oil_revenue = oil_production * economic_params.oil_price_usd_per_bbl

    # Costs
    co2_purchase_cost = (
        co2_purchased * co2_density_tonne_per_mscf * economic_params.co2_purchase_cost_usd_per_tonne
    )
    co2_recycle_cost = (
        co2_recycled * co2_density_tonne_per_mscf * economic_params.co2_recycle_cost_usd_per_tonne
    )

    # Operating costs — reads variable_opex_usd_per_bbl from EconomicParameters
    operating_cost = oil_production * getattr(
        economic_params, "variable_opex_usd_per_bbl",
        getattr(economic_params, "operating_cost_usd_per_bbl", 10.0)  # legacy fallback
    )

    # CO2 storage credits/penalties
    # Primary source: economic_params.co2_storage_credit_usd_per_tonne (user-configured)
    # Fallback: co2_storage_params.carbon_credit_usd_per_tonne (legacy object path)
    storage_credit = np.zeros_like(oil_production)
    if storage_metrics is not None:
        net_stored = storage_metrics.get("net_co2_stored_tonne", np.zeros_like(oil_production))
        carbon_credit_rate = getattr(economic_params, "co2_storage_credit_usd_per_tonne", 0.0)
        if carbon_credit_rate == 0.0 and co2_storage_params is not None:
            carbon_credit_rate = getattr(co2_storage_params, "carbon_credit_usd_per_tonne", 0.0)
        storage_credit = net_stored * carbon_credit_rate

    # Carbon tax penalty for CO2 leakage
    leakage_penalty = np.zeros_like(oil_production)
    if storage_metrics is not None and carbon_tax_usd_per_tonne > 0.0:
        annual_leakage_tonne = storage_metrics.get("annual_leakage_tonne", np.array([]))
        if len(annual_leakage_tonne) > 0:
            total_leakage_tonne = np.sum(annual_leakage_tonne)
            leakage_penalty = np.full_like(
                oil_production, total_leakage_tonne * carbon_tax_usd_per_tonne / len(oil_production)
            )

    # Net cashflow
    cashflow = (
        oil_revenue
        - co2_purchase_cost
        - co2_recycle_cost
        - operating_cost
        + storage_credit
        - leakage_penalty
    )

    # Initial investment (year 0) — reads capex_usd from EconomicParameters
    capex = getattr(
        economic_params, "capex_usd",
        getattr(economic_params, "initial_investment_usd", 0.0)  # legacy fallback
    )
    cashflow_with_capex = np.concatenate([[-capex], cashflow])

    # Calculate NPV
    discount_rate = economic_params.discount_rate_fraction
    npv_value = npv_func(discount_rate, cashflow_with_capex)

    return float(npv_value)


def calculate_cashflow(
    profiles: Dict[str, np.ndarray],
    economic_params: Any,
    storage_metrics: Optional[Dict[str, np.ndarray]] = None,
    co2_storage_params: Optional[Any] = None,
    time_resolution: str = "annual",
    co2_density_tonne_per_mscf: float = 0.053,
) -> np.ndarray:
    """
    Calculate annual cashflows for CO2-EOR project.

    Args:
        profiles: Production and injection profiles
        economic_params: Economic parameters
        storage_metrics: CO2 storage metrics (optional)
        co2_storage_params: CO2 storage parameters (optional)
        time_resolution: Time resolution key prefix
        co2_density_tonne_per_mscf: CO2 density conversion factor

    Returns:
        Array of annual cashflows
    """
    oil_production = profiles.get(f"{time_resolution}_oil_stb", np.array([0]))
    co2_purchased = profiles.get(f"{time_resolution}_co2_purchased_mscf", np.array([0]))
    co2_recycled = profiles.get(f"{time_resolution}_co2_recycled_mscf", np.array([0]))

    oil_revenue = oil_production * economic_params.oil_price_usd_per_bbl

    co2_purchase_cost = (
        co2_purchased * co2_density_tonne_per_mscf * economic_params.co2_purchase_cost_usd_per_tonne
    )
    co2_recycle_cost = (
        co2_recycled * co2_density_tonne_per_mscf * economic_params.co2_recycle_cost_usd_per_tonne
    )

    operating_cost = oil_production * getattr(
        economic_params, "variable_opex_usd_per_bbl",
        getattr(economic_params, "operating_cost_usd_per_bbl", 10.0)  # legacy fallback
    )

    storage_credit = np.zeros_like(oil_production)
    if storage_metrics is not None:
        net_stored = storage_metrics.get("net_co2_stored_tonne", np.zeros_like(oil_production))
        carbon_credit_rate = getattr(economic_params, "co2_storage_credit_usd_per_tonne", 0.0)
        if carbon_credit_rate == 0.0 and co2_storage_params is not None:
            carbon_credit_rate = getattr(co2_storage_params, "carbon_credit_usd_per_tonne", 0.0)
        storage_credit = net_stored * carbon_credit_rate

    cashflow = oil_revenue - co2_purchase_cost - co2_recycle_cost - operating_cost + storage_credit

    return cashflow
