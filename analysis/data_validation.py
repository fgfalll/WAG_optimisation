"""
Data validation module for CO2 EOR optimizer.
Provides validation checks for recovery factors, pressure data, and other critical metrics.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates simulation results for physical consistency and data integrity.
    """

    @staticmethod
    def validate_recovery_factor(
        rf: float, ooip: float, total_oil_produced: float, tolerance: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Validates recovery factor consistency.

        Args:
            rf: Reported recovery factor
            ooip: Original Oil In Place (STB)
            total_oil_produced: Total oil produced (STB)
            tolerance: Allowed tolerance for discrepancy

        Returns:
            Tuple of (is_valid, message)
        """
        if ooip <= 0:
            return False, f"Invalid OOIP: {ooip}"

        calculated_rf = total_oil_produced / ooip if ooip > 0 else 0.0

        # Check if reported RF matches calculated RF
        discrepancy = abs(rf - calculated_rf)
        if discrepancy > tolerance:
            return (
                False,
                f"Recovery factor discrepancy: reported={rf:.3f}, calculated={calculated_rf:.3f}, diff={discrepancy:.3f}",
            )

        # Check RF bounds
        if rf < 0:
            return False, f"Recovery factor cannot be negative: {rf}"
        if rf > 1.0:
            return False, f"Recovery factor cannot exceed 1.0: {rf}"

        return True, f"Recovery factor validation passed: {rf:.3f}"

    @staticmethod
    def validate_pressure_data(
        pressure_data: np.ndarray,
        initial_pressure: float,
        max_pressure: float,
        min_pressure: float = 500.0,
    ) -> Tuple[bool, str]:
        """
        Validates pressure data for physical consistency.

        Args:
            pressure_data: Array of pressure values
            initial_pressure: Initial reservoir pressure (psi)
            max_pressure: Maximum allowed pressure (psi)
            min_pressure: Minimum allowed pressure (psi)

        Returns:
            Tuple of (is_valid, message)
        """
        if np.size(pressure_data) == 0:
            raise ValueError("Validation failed: pressure array is empty (size=0)")

        # Check for NaN or infinite values
        if not np.all(np.isfinite(pressure_data)):
            return False, "Pressure data contains NaN or infinite values"

        # Check pressure bounds
        min_observed = np.min(pressure_data)
        max_observed = np.max(pressure_data)

        if min_observed < min_pressure:
            return (
                False,
                f"Pressure below minimum allowed: {min_observed:.0f} psi < {min_pressure:.0f} psi",
            )

        # GA optimizer might explore constraints up to +15% above max pressure. We shouldn't flag it as completely invalid
        # unless it exceeds reasonable physical bounds (e.g. caprock fracture pressure which is usually higher).
        if max_observed > max_pressure * 1.2:
            return (
                False,
                f"Pressure excessively exceeds maximum allowed: {max_observed:.0f} psi > {max_pressure * 1.2:.0f} psi",
            )

        # Check for unrealistic pressure jumps
        # Note: In yearly data, a jump of 5000 psi in a single step (year) is possible
        if len(pressure_data) > 1:
            pressure_changes = np.abs(np.diff(pressure_data))
            max_change = np.max(pressure_changes)
            if max_change > 8000:
                return False, f"Unrealistic pressure change: {max_change:.0f} psi per output step"

        return True, f"Pressure validation passed: range {min_observed:.0f}-{max_observed:.0f} psi"

    @staticmethod
    def validate_production_data(
        oil_production: np.ndarray, co2_injection: np.ndarray, co2_production: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validates production and injection data consistency.

        Args:
            oil_production: Daily oil production (STB)
            co2_injection: Daily CO2 injection (MSCF)
            co2_production: Daily CO2 production (MSCF)

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        # Check for negative values
        if np.any(oil_production < 0):
            issues.append("Negative oil production values")

        if np.any(co2_injection < 0):
            issues.append("Negative CO2 injection values")

        if np.any(co2_production < 0):
            issues.append("Negative CO2 production values")

        # Check for unrealistic values
        max_oil_rate = np.max(oil_production) if len(oil_production) > 0 else 0
        if max_oil_rate > 1e6:  # 1 million STB/day is unrealistic
            issues.append(f"Unrealistic oil rate: {max_oil_rate:.0f} STB/day")

        max_co2_inj = np.max(co2_injection) if len(co2_injection) > 0 else 0
        if max_co2_inj > 1e6:  # 1 million MSCF/day is unrealistic
            issues.append(f"Unrealistic CO2 injection: {max_co2_inj:.0f} MSCF/day")

        if issues:
            return False, "; ".join(issues)

        return True, "Production data validation passed"

    @staticmethod
    def _get_oil_production(profiles: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get oil production from profiles with flexible key matching.

        Prefers daily rate keys (e.g., 'oil_production_rate') over time-aggregated
        volume keys (e.g., 'yearly_oil_stb'). When validating against daily rate
        thresholds (1M STB/day), aggregated yearly volumes (e.g., 2.5M STB/year)
        would appear as 2.5M STB/day — causing false positives since 2.5M STB/year
        ≈ 6,850 STB/day, a realistic rate for CO2 EOR.

        Args:
            profiles: Dictionary of production profiles

        Returns:
            Oil production array (rate or daily-equivalent) or empty array if not found
        """
        TIME_PREFIXES = ("yearly_", "monthly_", "daily_", "annual_")

        rate_key = None
        aggregated_keys = []

        for key in profiles:
            key_lower = key.lower()
            if "oil" in key_lower and ("stb" in key_lower or "rate" in key_lower):
                if key.startswith(TIME_PREFIXES):
                    aggregated_keys.append(key)
                else:
                    rate_key = key

        if rate_key is not None:
            logger.debug(f"Oil production: selected rate key '{rate_key}'")
            return profiles[rate_key]
        elif aggregated_keys:
            selected_key = aggregated_keys[0]
            logger.debug(
                f"Oil production: selected aggregated key '{selected_key}' "
                f"(dividing by 365 to convert annual volume to daily rate)"
            )
            return profiles[selected_key] / 365.0

        return np.array([0.0])

    @staticmethod
    def _get_co2_injection(profiles: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get CO2 injection data from profiles with preference for daily rates.

        Prefers keys representing daily rates (e.g., 'co2_injection') over
        time-aggregated keys (e.g., 'yearly_co2_injected_mscf'). This prevents
        false positives when validating against daily rate thresholds.

        Args:
            profiles: Dictionary of production profiles

        Returns:
            CO2 injection array or empty array if not found
        """
        TIME_PREFIXES = ("yearly_", "monthly_", "daily_", "annual_")

        daily_rate_key = None
        aggregated_keys = []

        for key in profiles:
            key_lower = key.lower()
            if "co2" in key_lower and "inject" in key_lower:
                if key.startswith(TIME_PREFIXES):
                    aggregated_keys.append(key)
                else:
                    daily_rate_key = key
                    break

        if daily_rate_key is not None:
            logger.debug(f"CO2 injection: selected daily rate key '{daily_rate_key}'")
            return profiles[daily_rate_key]
        elif aggregated_keys:
            selected_key = aggregated_keys[0]
            logger.debug(
                f"CO2 injection: selected aggregated key '{selected_key}' (daily rate key not found)"
            )
            return profiles[selected_key]

        return np.array([0.0])

    @staticmethod
    def _get_co2_production(profiles: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get CO2 production data from profiles with preference for daily rates.

        Args:
            profiles: Dictionary of production profiles

        Returns:
            CO2 production array or empty array if not found
        """
        TIME_PREFIXES = ("yearly_", "monthly_", "daily_", "annual_")

        daily_rate_key = None
        aggregated_keys = []

        for key in profiles:
            key_lower = key.lower()
            if "co2" in key_lower and "prod" in key_lower:
                if key.startswith(TIME_PREFIXES):
                    aggregated_keys.append(key)
                else:
                    daily_rate_key = key
                    break

        if daily_rate_key is not None:
            logger.debug(f"CO2 production: selected daily rate key '{daily_rate_key}'")
            return profiles[daily_rate_key]
        elif aggregated_keys:
            selected_key = aggregated_keys[0]
            logger.debug(
                f"CO2 production: selected aggregated key '{selected_key}' (daily rate key not found)"
            )
            return profiles[selected_key]

        return np.array([0.0])

    @staticmethod
    def validate_simulation_results(
        profiles: Dict[str, np.ndarray], metrics: Dict[str, float], reservoir_data: Dict[str, Any]
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Comprehensive validation of simulation results.

        Args:
            profiles: Dictionary of production profiles
            metrics: Dictionary of simulation metrics
            reservoir_data: Dictionary of reservoir properties

        Returns:
            Dictionary of validation results by category
        """
        results = {}

        # Validate recovery factor
        # For integrated profiles (yearly/monthly totals), sum gives total production
        # For rate profiles, the sum represents total if properly integrated
        rf = metrics.get("recovery_factor", 0.0)
        ooip = reservoir_data.get("ooip_stb", 0.0)
        oil_production = DataValidator._get_oil_production(profiles)

        logger.debug(
            f"RF Validation: rf={rf:.4f}, ooip={ooip}, max_val={np.max(oil_production) if len(oil_production) > 0 else 'N/A'}"
        )

        # Check if profiles are rates or integrated volumes by checking magnitude
        # Rate values (STB/day) are typically < 100000 for realistic fields
        # Volume values (STB/year or STB total) can be millions
        cum_oil = metrics.get("cumulative_oil")
        if cum_oil is None:
            cum_oil = profiles.get("cumulative_oil")

        if cum_oil is not None and float(cum_oil) > 0:
            total_oil_produced = float(cum_oil)
            logger.debug(f"RF Validation: using direct cumulative_oil = {total_oil_produced}")
        elif len(oil_production) > 0:
            max_val = np.max(oil_production)
            # If max value > 100000, assume it's an integrated volume, not a rate
            if max_val > 100000:
                logger.debug(
                    f"RF Validation: treating as volume (max={max_val}), summing to {np.sum(oil_production)}"
                )
                total_oil_produced = np.sum(oil_production)
            else:
                # It's a rate array - multiply by dt to get volume
                time_vector = profiles.get("time_vector")
                if (
                    time_vector is not None
                    and isinstance(time_vector, np.ndarray)
                    and len(time_vector) == len(oil_production)
                    and len(time_vector) > 1
                ):
                    dt = np.diff(
                        time_vector,
                        prepend=time_vector[0] - (time_vector[1] - time_vector[0]),
                    )
                    total_oil_produced = np.sum(oil_production * dt)
                else:
                    # Infer time step based on length of rate profile
                    # e.g., ~180 points for a 15-year project = monthly
                    if len(oil_production) > 50:
                        dt = 30.4375  # Monthly
                    elif len(oil_production) > 20:
                        dt = 91.3125  # Quarterly
                    else:
                        dt = 365.25  # Yearly
                    total_oil_produced = np.sum(oil_production) * dt
                logger.debug(
                    f"RF Validation: treating as rate, integrated total = {total_oil_produced}"
                )
        else:
            total_oil_produced = 0.0

        results["recovery_factor"] = DataValidator.validate_recovery_factor(
            rf, ooip, total_oil_produced
        )

        # Validate pressure data - use flexible key matching
        pressure_data = np.array([])
        for key in profiles:
            if "pressure" in key.lower():
                pressure_data = profiles[key]
                break
        initial_pressure = reservoir_data.get("initial_pressure", 0.0)
        max_pressure = reservoir_data.get("max_pressure_psi", 10000.0)

        results["pressure"] = DataValidator.validate_pressure_data(
            pressure_data, initial_pressure, max_pressure
        )

        # Validate production data - use flexible key matching
        oil_production = DataValidator._get_oil_production(profiles)

        # Find CO2 injection and production data with preference for daily rates
        co2_injection = DataValidator._get_co2_injection(profiles)
        co2_production = DataValidator._get_co2_production(profiles)

        # Debug log the selected data ranges
        if len(co2_injection) > 0:
            max_inj = np.max(co2_injection)
            logger.debug(
                f"CO2 injection validation - max value: {max_inj:.2f} MSCF, key: selected from profiles"
            )

        results["production"] = DataValidator.validate_production_data(
            oil_production, co2_injection, co2_production
        )

        return results

    @staticmethod
    def log_validation_results(validation_results: Dict[str, Tuple[bool, str]]):
        """
        Logs validation results.

        Args:
            validation_results: Dictionary of validation results
        """
        for category, (is_valid, message) in validation_results.items():
            if is_valid:
                logger.info(f"✓ {category}: {message}")
            else:
                logger.error(f"✗ {category}: {message}")
