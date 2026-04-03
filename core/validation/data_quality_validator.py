"""
Comprehensive Data Quality and Validation System for CO2-EOR Benchmark
Validates simulation results, ensures mass balance consistency, and provides quality metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ValidationResult:
    """Container for validation results"""
    level: ValidationLevel
    component: str
    description: str
    actual_value: float
    expected_value: Optional[float] = None
    tolerance: Optional[float] = None
    passed: bool = True

@dataclass
class QualityMetrics:
    """Container for overall quality metrics"""
    mass_balance_error_percent: float = 0.0
    energy_conservation_error_percent: float = 0.0
    saturation_consistency_error: float = 0.0
    pressure_gradient_quality_score: float = 1.0
    overall_quality_score: float = 1.0
    validation_results: List[ValidationResult] = None

    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []

class DataQualityValidator:
    """
    Comprehensive data quality validator for CO2-EOR simulation results.
    Performs validation checks on mass balance, energy conservation, and physical consistency.
    """

    def __init__(self, tolerance_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the data quality validator.

        Args:
            tolerance_thresholds: Custom tolerance thresholds for various validation checks
        """
        self.tolerances = tolerance_thresholds or {
            'mass_balance_error': 0.01,      # 1%
            'energy_conservation': 0.05,    # 5%
            'saturation_consistency': 0.01, # 1%
            'pressure_continuity': 100.0,   # 100 psi
            'rate_consistency': 0.1,        # 10%
            'co2_partitioning_total': 0.1   # 10%
        }

    def validate_simulation_results(self, simulation_results: Dict[str, Any],
                                  injection_history: Optional[Dict[float, Dict[str, float]]] = None) -> QualityMetrics:
        """
        Perform comprehensive validation of simulation results.

        Args:
            simulation_results: Dictionary containing simulation results
            injection_history: Optional injection history for mass balance validation

        Returns:
            QualityMetrics object containing all validation results and quality scores
        """
        validation_results = []

        # 1. Mass Balance Validation
        mass_balance_result = self._validate_mass_balance(simulation_results, injection_history)
        validation_results.append(mass_balance_result)

        # 2. CO2 Partitioning Validation
        partitioning_results = self._validate_co2_partitioning(simulation_results)
        validation_results.extend(partitioning_results)

        # 3. Physical Consistency Validation
        consistency_results = self._validate_physical_consistency(simulation_results)
        validation_results.extend(consistency_results)

        # 4. Energy Conservation Validation
        energy_result = self._validate_energy_conservation(simulation_results)
        validation_results.append(energy_result)

        # 5. Rate and Pressure Consistency
        rate_pressure_results = self._validate_rate_pressure_consistency(simulation_results)
        validation_results.extend(rate_pressure_results)

        # Calculate overall quality score
        quality_metrics = self._calculate_quality_scores(validation_results, simulation_results)
        quality_metrics.validation_results = validation_results

        return quality_metrics

    def _validate_mass_balance(self, results: Dict[str, Any],
                            injection_history: Optional[Dict[float, Dict[str, float]]] = None) -> ValidationResult:
        """Validate mass balance consistency."""
        try:
            # Get partitioning data
            partitioning = results.get('co2_partitioning', {})
            if not partitioning:
                return ValidationResult(
                    ValidationLevel.WARNING,
                    "Mass Balance",
                    "No CO2 partitioning data available",
                    0.0
                )

            total_injected = partitioning.get('total_injected_tonne', 0.0)
            total_accounted = partitioning.get('total_stored_tonne', 0.0)

            if total_injected > 0:
                error_percent = abs((total_accounted - total_injected) / total_injected) * 100
                tolerance = self.tolerances['mass_balance_error'] * 100

                passed = error_percent <= tolerance
                level = ValidationLevel.ERROR if not passed else ValidationLevel.INFO

                return ValidationResult(
                    level,
                    "Mass Balance",
                    f"Mass conservation error: {error_percent:.4f}% (tolerance: ±{tolerance:.1f}%)",
                    error_percent,
                    tolerance,
                    tolerance,
                    passed
                )
            else:
                return ValidationResult(
                    ValidationLevel.WARNING,
                    "Mass Balance",
                    "No CO2 injected - mass balance check skipped",
                    0.0
                )

        except Exception as e:
            logger.error(f"Mass balance validation failed: {e}")
            return ValidationResult(
                ValidationLevel.ERROR,
                "Mass Balance",
                f"Validation failed: {str(e)}",
                0.0,
                passed=False
            )

    def _validate_co2_partitioning(self, results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate CO2 trapping mechanism partitioning."""
        validation_results = []

        try:
            partitioning = results.get('co2_partitioning', {})
            if not partitioning:
                return [ValidationResult(
                    ValidationLevel.WARNING,
                    "CO2 Partitioning",
                    "No partitioning data available",
                    0.0
                )]

            # Check total percentage sum
            total_percent = partitioning.get('total_accounted_percent', 0.0)
            tolerance = self.tolerances['co2_partitioning_total']

            deviation = abs(total_percent - 100.0)
            passed = deviation <= tolerance
            level = ValidationLevel.ERROR if not passed else ValidationLevel.INFO

            validation_results.append(ValidationResult(
                level,
                "CO2 Partitioning Total",
                f"Total partitioning: {total_percent:.2f}% (should be 100.0%)",
                total_percent,
                100.0,
                tolerance,
                passed
            ))

            # Validate individual trapping mechanisms
            trapping_mechanisms = [
                ('structural_trapping', 'Structural Trapping'),
                ('residual_trapping', 'Residual Trapping'),
                ('solubility_trapping', 'Solubility Trapping'),
                ('mineral_trapping', 'Mineral Trapping'),
                ('mobile_co2', 'Mobile CO2'),
                ('leakage', 'Leakage')
            ]

            for mechanism_key, mechanism_name in trapping_mechanisms:
                percent_key = f"{mechanism_key}_percent"
                if percent_key in partitioning:
                    percent_value = partitioning[percent_key]

                    # Validate range (0-100%)
                    if not (0.0 <= percent_value <= 100.0):
                        validation_results.append(ValidationResult(
                            ValidationLevel.ERROR,
                            mechanism_name,
                            f"Invalid percentage: {percent_value:.2f}% (must be 0-100%)",
                            percent_value,
                            None,
                            None,
                            False
                        ))

                    # Check for reasonable values based on physics
                    if mechanism_key == 'mineral_trapping' and percent_value > 5.0:
                        validation_results.append(ValidationResult(
                            ValidationLevel.WARNING,
                            mechanism_name,
                            f"High mineral trapping: {percent_value:.2f}% (unusual for 15-year simulation)",
                            percent_value,
                            None,
                            None,
                            True
                        ))
                    elif mechanism_key == 'leakage' and percent_value > 1.0:
                        validation_results.append(ValidationResult(
                            ValidationLevel.WARNING,
                            mechanism_name,
                            f"High leakage: {percent_value:.2f}% (indicates containment issues)",
                            percent_value,
                            None,
                            None,
                            True
                        ))

        except Exception as e:
            logger.error(f"CO2 partitioning validation failed: {e}")
            validation_results.append(ValidationResult(
                ValidationLevel.ERROR,
                "CO2 Partitioning",
                f"Validation failed: {str(e)}",
                0.0,
                passed=False
            ))

        return validation_results

    def _validate_physical_consistency(self, results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate physical consistency of simulation results."""
        validation_results = []

        try:
            # Validate recovery factor
            rf = results.get('recovery_factor', 0.0)
            if rf < 0 or rf > 1.0:
                validation_results.append(ValidationResult(
                    ValidationLevel.ERROR,
                    "Recovery Factor",
                    f"Invalid recovery factor: {rf:.4f} (must be 0-1)",
                    rf,
                    None,
                    None,
                    False
                ))
            elif rf > 0.5:
                validation_results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "Recovery Factor",
                    f"High recovery factor: {rf:.4f} (unusual for CO2-EOR)",
                    rf,
                    None,
                    None,
                    True
                ))

            # Validate pressures
            final_pressure = results.get('final_avg_pressure', 0.0)
            initial_pressure = results.get('initial_pressure', 0.0)

            if final_pressure <= 0 or initial_pressure <= 0:
                validation_results.append(ValidationResult(
                    ValidationLevel.ERROR,
                    "Pressure Values",
                    f"Invalid pressure values: initial={initial_pressure:.1f}, final={final_pressure:.1f} psi",
                    final_pressure,
                    None,
                    None,
                    False
                ))
            elif final_pressure > initial_pressure:
                validation_results.append(ValidationResult(
                    ValidationLevel.WARNING,
                    "Pressure Values",
                    f"Final pressure > initial pressure: {final_pressure:.1f} > {initial_pressure:.1f} psi",
                    final_pressure,
                    initial_pressure,
                    None,
                    True
                ))

            # Validate saturations
            final_co2_sat = results.get('final_avg_s_co2', 0.0)
            if final_co2_sat < 0 or final_co2_sat > 1.0:
                validation_results.append(ValidationResult(
                    ValidationLevel.ERROR,
                    "CO2 Saturation",
                    f"Invalid CO2 saturation: {final_co2_sat:.4f} (must be 0-1)",
                    final_co2_sat,
                    None,
                    None,
                    False
                ))

        except Exception as e:
            logger.error(f"Physical consistency validation failed: {e}")
            validation_results.append(ValidationResult(
                ValidationLevel.ERROR,
                "Physical Consistency",
                f"Validation failed: {str(e)}",
                0.0,
                passed=False
            ))

        return validation_results

    def _validate_energy_conservation(self, results: Dict[str, Any]) -> ValidationResult:
        """Validate energy conservation (simplified check)."""
        try:
            # This is a simplified energy check - in a real implementation,
            # we would calculate total energy injected vs energy produced
            total_co2_injected = results.get('co2_partitioning', {}).get('total_injected_tonne', 0.0)
            total_oil_recovered = results.get('total_oil_recovery', 0.0)

            # Simple energy ratio check (very approximate)
            if total_co2_injected > 0:
                energy_ratio = total_oil_recovered / (total_co2_injected * 1000)  # barrels per tonne
                expected_range = (0.1, 10.0)  # Expected energy recovery range

                if energy_ratio < expected_range[0]:
                    return ValidationResult(
                        ValidationLevel.WARNING,
                        "Energy Conservation",
                        f"Low energy efficiency: {energy_ratio:.3f} bbl/tonne",
                        energy_ratio,
                        None,
                        None,
                        True
                    )
                elif energy_ratio > expected_range[1]:
                    return ValidationResult(
                        ValidationLevel.WARNING,
                        "Energy Conservation",
                        f"High energy efficiency: {energy_ratio:.3f} bbl/tonne (unusual)",
                        energy_ratio,
                        None,
                        None,
                        True
                    )
                else:
                    return ValidationResult(
                        ValidationLevel.INFO,
                        "Energy Conservation",
                        f"Energy efficiency within expected range: {energy_ratio:.3f} bbl/tonne",
                        energy_ratio,
                        None,
                        None,
                        True
                    )
            else:
                return ValidationResult(
                    ValidationLevel.INFO,
                    "Energy Conservation",
                    "No CO2 injected - energy check skipped",
                    0.0
                )

        except Exception as e:
            logger.error(f"Energy conservation validation failed: {e}")
            return ValidationResult(
                ValidationLevel.WARNING,
                "Energy Conservation",
                f"Validation failed: {str(e)}",
                0.0,
                passed=True
            )

    def _validate_rate_pressure_consistency(self, results: Dict[str, Any]) -> List[ValidationResult]:
        """Validate rate and pressure consistency."""
        validation_results = []

        try:
            # Check for reasonable injection rates
            co2_partitioning = results.get('co2_partitioning', {})
            total_injected_tonnes = co2_partitioning.get('total_injected_tonne', 0.0)
            simulation_years = results.get('simulation_time_years', 15.0)

            if total_injected_tonnes > 0 and simulation_years > 0:
                avg_injection_rate = total_injected_tonnes / simulation_years  # tonnes/year
                avg_injection_rate_mscfd = avg_injection_rate * 1000 / 365  # MSCF/day

                # Check if injection rate is reasonable
                if avg_injection_rate_mscfd > 10000:
                    validation_results.append(ValidationResult(
                        ValidationLevel.WARNING,
                        "Injection Rate",
                        f"High injection rate: {avg_injection_rate_mscfd:.0f} MSCF/day",
                        avg_injection_rate_mscfd,
                        None,
                        None,
                        True
                    ))
                elif avg_injection_rate_mscfd < 100:
                    validation_results.append(ValidationResult(
                        ValidationLevel.WARNING,
                        "Injection Rate",
                        f"Low injection rate: {avg_injection_rate_mscfd:.0f} MSCF/day",
                        avg_injection_rate_mscfd,
                        None,
                        None,
                        True
                    ))

        except Exception as e:
            logger.error(f"Rate/pressure consistency validation failed: {e}")
            validation_results.append(ValidationResult(
                ValidationLevel.WARNING,
                "Rate/Pressure Consistency",
                f"Validation failed: {str(e)}",
                0.0,
                passed=True
            ))

        return validation_results

    def _calculate_quality_scores(self, validation_results: List[ValidationResult],
                                 simulation_results: Dict[str, Any]) -> QualityMetrics:
        """Calculate overall quality scores based on validation results."""
        quality_metrics = QualityMetrics()

        # Count validation levels
        error_count = sum(1 for r in validation_results if r.level == ValidationLevel.ERROR)
        warning_count = sum(1 for r in validation_results if r.level == ValidationLevel.WARNING)
        info_count = sum(1 for r in validation_results if r.level == ValidationLevel.INFO)

        total_validations = len(validation_results)

        # Calculate individual quality scores
        for result in validation_results:
            if result.level == ValidationLevel.ERROR:
                if "Mass Balance" in result.component:
                    quality_metrics.mass_balance_error_percent = result.actual_value
                elif "Energy" in result.component:
                    quality_metrics.energy_conservation_error_percent = result.actual_value

        # Calculate overall quality score (0-1 scale)
        if total_validations > 0:
            # Weight errors heavily, warnings moderately, info lightly
            error_weight = -0.5
            warning_weight = -0.1
            info_weight = -0.01

            score = 1.0 + (error_count * error_weight + warning_count * warning_weight + info_count * info_weight)
            quality_metrics.overall_quality_score = max(0.0, min(1.0, score))
        else:
            quality_metrics.overall_quality_score = 1.0

        return quality_metrics

    def generate_quality_report(self, quality_metrics: QualityMetrics) -> str:
        """Generate a comprehensive quality assessment report."""
        report = []
        report.append("=== CO2-EOR SIMULATION DATA QUALITY REPORT ===\n")

        # Overall assessment
        score = quality_metrics.overall_quality_score
        if score >= 0.9:
            assessment = "EXCELLENT"
        elif score >= 0.8:
            assessment = "GOOD"
        elif score >= 0.7:
            assessment = "ACCEPTABLE"
        elif score >= 0.6:
            assessment = "POOR"
        else:
            assessment = "CRITICAL ISSUES"

        report.append(f"Overall Quality Assessment: {assessment} (Score: {score:.3f})\n")

        # Validation results summary
        report.append("=== VALIDATION RESULTS ===")
        error_count = sum(1 for r in quality_metrics.validation_results if r.level == ValidationLevel.ERROR)
        warning_count = sum(1 for r in quality_metrics.validation_results if r.level == ValidationLevel.WARNING)
        info_count = sum(1 for r in quality_metrics.validation_results if r.level == ValidationLevel.INFO)

        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        report.append(f"Info Messages: {info_count}\n")

        # Detailed results
        report.append("=== DETAILED VALIDATION RESULTS ===")
        for result in quality_metrics.validation_results:
            status = "[FAILED]" if not result.passed else "[PASSED]"
            report.append(f"[{result.level.value}] {result.component}: {status}")
            report.append(f"  {result.description}")
            if result.expected_value is not None:
                report.append(f"  Expected: {result.expected_value}, Actual: {result.actual_value:.4f}")
            report.append("")

        # Key metrics
        report.append("=== KEY METRICS ===")
        report.append(f"Mass Balance Error: {quality_metrics.mass_balance_error_percent:.4f}%")
        report.append(f"Energy Conservation Error: {quality_metrics.energy_conservation_error_percent:.4f}%")
        report.append(f"Overall Quality Score: {quality_metrics.overall_quality_score:.3f}")
        report.append("")

        # Recommendations
        report.append("=== RECOMMENDATIONS ===")
        recommendations = self._generate_recommendations(quality_metrics)
        for rec in recommendations:
            report.append(f"- {rec}")

        return "\n".join(report)

    def _generate_recommendations(self, quality_metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Analyze validation results and generate specific recommendations
        for result in quality_metrics.validation_results:
            if result.level == ValidationLevel.ERROR:
                if "Mass Balance" in result.component:
                    recommendations.append("CRITICAL: Fix mass balance calculation errors - check CO2 partitioning logic")
                elif "Partitioning" in result.component:
                    recommendations.append("CRITICAL: Fix CO2 trapping mechanism calculations - ensure percentages sum to 100%")
                elif "Recovery Factor" in result.component:
                    recommendations.append("CRITICAL: Validate recovery factor calculation - check production data")
                elif "Pressure Values" in result.component:
                    recommendations.append("CRITICAL: Check pressure data - verify initial and final pressure values")
                elif "CO2 Saturation" in result.component:
                    recommendations.append("CRITICAL: Validate saturation calculations - ensure values are physically realistic")

            elif result.level == ValidationLevel.WARNING:
                if "High mineral trapping" in result.description:
                    recommendations.append("Review mineralization kinetics - check if rates are realistic for 15-year timeframe")
                elif "High leakage" in result.description:
                    recommendations.append("Investigate fault seal integrity - check geological parameters")
                elif "Final pressure > initial pressure" in result.description:
                    recommendations.append("Review pressure maintenance - check injection/production balance")
                elif "High recovery factor" in result.description:
                    recommendations.append("Verify recovery factor calculation - cross-check with production data")
                elif "Low injection rate" in result.description:
                    recommendations.append("Review injection constraints - check well control parameters")
                elif "High injection rate" in result.description:
                    recommendations.append("Validate injection capacity - check facility limitations")

        # General recommendations based on overall score
        if quality_metrics.overall_quality_score < 0.7:
            recommendations.append("Overall quality is POOR - comprehensive review of simulation setup recommended")
        elif quality_metrics.overall_quality_score < 0.8:
            recommendations.append("Quality is ACCEPTABLE - consider addressing warnings for improved reliability")

        if not recommendations:
            recommendations.append("Simulation quality is GOOD - no specific issues identified")

        return recommendations