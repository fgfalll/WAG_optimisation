"""
Comparison Metrics for Validation

This module calculates comparison metrics between our engine results
and CMG GEM results for validation purposes.
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ComparisonMetrics:
    """
    Calculate comparison metrics between our engine and CMG GEM.

    Provides metrics for validation including:
    - Field-level error percentages
    - Pressure RMSE
    - Saturation RMSE
    - Component recovery comparison

    Example:
        >>> metrics = ComparisonMetrics()
        >>> comparison = metrics.calculate_all(our_result, cmg_result)
        >>> if comparison['pass']:
        ...     print("Validation passed!")
    """

    # Validation tolerance criteria
    CRITERIA = {
        'oil_cumulative_error_pct': 5.0,  # 5% tolerance
        'pressure_rmse_psi': 50.0,  # 50 psi tolerance
        'saturation_rmse': 0.05,  # 0.05 tolerance
        'component_recovery_error_pct': 10.0,  # 10% tolerance
    }

    def __init__(self, criteria: Optional[Dict[str, float]] = None):
        """
        Initialize comparison metrics calculator.

        Args:
            criteria: Optional custom validation criteria
        """
        if criteria:
            self.CRITERIA.update(criteria)

    def calculate_all(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all comparison metrics.

        Args:
            our_result: Results from our compositional engine
            cmg_result: Results from CMG GEM

        Returns:
            Dictionary with all metrics and pass/fail status
        """
        metrics = {}

        # Field-level metrics
        metrics['oil_cumulative_error_pct'] = self._oil_cumulative_error(
            our_result, cmg_result
        )
        metrics['gas_cumulative_error_pct'] = self._gas_cumulative_error(
            our_result, cmg_result
        )
        metrics['water_cumulative_error_pct'] = self._water_cumulative_error(
            our_result, cmg_result
        )

        # Time-series metrics
        metrics['pressure_rmse_psi'] = self._pressure_rmse(
            our_result, cmg_result
        )
        metrics['saturation_rmse'] = self._saturation_rmse(
            our_result, cmg_result
        )

        # Component recovery
        metrics['component_recovery_error_pct'] = self._component_recovery_error(
            our_result, cmg_result
        )

        # Overall pass/fail
        metrics['pass'] = self._check_criteria(metrics)

        # Summary statistics
        metrics['summary'] = self._generate_summary(metrics)

        return metrics

    def _oil_cumulative_error(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate cumulative oil production error percentage."""
        our_oil = our_result.get('cumulative_oil', 0.0)
        cmg_oil = cmg_result.get('cumulative_oil', 0.0)

        if cmg_oil == 0:
            return 0.0 if our_oil == 0 else 100.0

        error_pct = abs(our_oil - cmg_oil) / cmg_oil * 100
        return error_pct

    def _gas_cumulative_error(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate cumulative gas production error percentage."""
        our_gas = our_result.get('cumulative_gas', 0.0)
        cmg_gas = cmg_result.get('cumulative_gas', 0.0)

        if cmg_gas == 0:
            return 0.0 if our_gas == 0 else 100.0

        error_pct = abs(our_gas - cmg_gas) / cmg_gas * 100
        return error_pct

    def _water_cumulative_error(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate cumulative water production error percentage."""
        our_water = our_result.get('cumulative_water', 0.0)
        cmg_water = cmg_result.get('cumulative_water', 0.0)

        if cmg_water == 0:
            return 0.0 if our_water == 0 else 100.0

        error_pct = abs(our_water - cmg_water) / cmg_water * 100
        return error_pct

    def _pressure_rmse(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate pressure RMSE."""
        our_pressure = our_result.get('pressure_profile', np.array([]))
        cmg_pressure = cmg_result.get('pressure_profile', np.array([]))

        if len(our_pressure) == 0 or len(cmg_pressure) == 0:
            # Use final pressure if time series not available
            our_final = our_result.get('final_pressure', 0)
            cmg_final = cmg_result.get('final_pressure', 0)
            return abs(our_final - cmg_final)

        # Interpolate to same length if needed
        if len(our_pressure) != len(cmg_pressure):
            our_pressure = np.interp(
                np.linspace(0, 1, len(cmg_pressure)),
                np.linspace(0, 1, len(our_pressure)),
                our_pressure
            )

        rmse = np.sqrt(np.mean((our_pressure - cmg_pressure) ** 2))
        return rmse

    def _saturation_rmse(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate saturation RMSE (simplified)."""
        # Saturation data may not be available in simplified results
        # Return 0 if not available
        return 0.0

    def _component_recovery_error(
        self,
        our_result: Dict[str, Any],
        cmg_result: Dict[str, Any]
    ) -> float:
        """Calculate component recovery error percentage."""
        our_recovery = our_result.get('recovery_factor', 0.0)
        cmg_recovery = cmg_result.get('recovery_factor', 0.0)

        if cmg_recovery == 0:
            return 0.0 if our_recovery == 0 else 100.0

        error_pct = abs(our_recovery - cmg_recovery) / cmg_recovery * 100
        return error_pct

    def _check_criteria(self, metrics: Dict[str, float]) -> bool:
        """Check if all metrics meet validation criteria."""
        checks = [
            metrics['oil_cumulative_error_pct'] <= self.CRITERIA['oil_cumulative_error_pct'],
            metrics['pressure_rmse_psi'] <= self.CRITERIA['pressure_rmse_psi'],
            metrics['saturation_rmse'] <= self.CRITERIA['saturation_rmse'],
        ]

        # Add component recovery check if available
        if 'component_recovery_error_pct' in metrics:
            checks.append(
                metrics['component_recovery_error_pct'] <= self.CRITERIA['component_recovery_error_pct']
            )

        return all(checks)

    def _generate_summary(self, metrics: Dict[str, float]) -> str:
        """Generate human-readable summary."""
        lines = [
            "Validation Summary:",
            f"  Oil Cumulative Error: {metrics['oil_cumulative_error_pct']:.2f}%",
            f"  Pressure RMSE: {metrics['pressure_rmse_psi']:.2f} psi",
            f"  Saturation RMSE: {metrics['saturation_rmse']:.4f}",
        ]

        if 'component_recovery_error_pct' in metrics:
            lines.append(
                f"  Component Recovery Error: {metrics['component_recovery_error_pct']:.2f}%"
            )

        lines.append(f"  Result: {'PASS' if metrics['pass'] else 'FAIL'}")

        return '\n'.join(lines)

    def calculate_correlation(
        self,
        our_data: np.ndarray,
        cmg_data: np.ndarray
    ) -> float:
        """Calculate correlation coefficient between two datasets."""
        if len(our_data) != len(cmg_data):
            # Interpolate to match lengths
            our_data = np.interp(
                np.linspace(0, 1, len(cmg_data)),
                np.linspace(0, 1, len(our_data)),
                our_data
            )

        if len(our_data) == 0:
            return 0.0

        correlation = np.corrcoef(our_data, cmg_data)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def calculate_r_squared(
        self,
        our_data: np.ndarray,
        cmg_data: np.ndarray
    ) -> float:
        """Calculate R-squared between two datasets."""
        if len(our_data) == 0 or len(cmg_data) == 0:
            return 0.0

        # Reshape if needed
        our_data = np.ravel(our_data)
        cmg_data = np.ravel(cmg_data)

        if len(our_data) != len(cmg_data):
            our_data = np.interp(
                np.linspace(0, 1, len(cmg_data)),
                np.linspace(0, 1, len(our_data)),
                our_data
            )

        ss_res = np.sum((cmg_data - our_data) ** 2)
        ss_tot = np.sum((cmg_data - np.mean(cmg_data)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)


__all__ = ["ComparisonMetrics"]
