"""
Physical Consistency Validator

This module validates simulation results against physical laws to ensure
the simulation is producing physically realistic results.

Key validations:
1. Darcy flow consistency (rate ≈ PI × drawdown)
2. Material balance (injected - produced = Δstorage)
3. Relative permeability range (0 ≤ kr ≤ 1)
4. GOR consistency (GOR ≈ gas_rate/oil_rate)
5. Production limits (physically achievable rates)
6. Parameter ranges (within physical bounds)

Reference tolerances are based on industry standards for compositional simulation.

Author: CO2-EOR Optimizer Team
Version: 1.0.0
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of physical consistency check.

    Attributes:
        is_valid: True if validation passed
        error_message: Error message if validation failed
        warnings: List of warning messages (non-critical issues)
        validation_type: Type of validation performed
        metrics: Dictionary of calculated metrics for debugging
    """
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    validation_type: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.validation_type == "":
            self.validation_type = self.__class__.__name__


class PhysicalConsistencyValidator:
    """
    Validate simulation results against physical laws.

    Key validations:
    1. Darcy flow consistency (rate ≈ PI × drawdown)
    2. Material balance (injected - produced = Δstorage)
    3. Relative permeability range (0 ≤ kr ≤ 1)
    4. GOR consistency (GOR ≈ gas_rate/oil_rate)
    5. Production limits (physically achievable rates)

    Tolerances based on industry standards:
    - Mass balance: 5% (industry standard for compositional sim)
    - Darcy flow: 10% (allowing for skin effects and heterogeneity)
    - GOR: 15% (allowing for measurement uncertainty)
    """

    # Tolerances (from numerical-stability skill and industry standards)
    MASS_BALANCE_TOLERANCE = 0.05  # 5% (industry standard for compositional sim)
    DARCY_TOLERANCE = 0.10         # 10% (allowing for skin effects)
    GOR_TOLERANCE = 0.15           # 15% (allowing for measurement uncertainty)
    KR_TOLERANCE = 0.01            # 1% for relative permeability bounds

    # Parameter ranges (from parameter-optimization skill)
    MIN_PERMEABILITY = 1.0         # md
    MAX_PERMEABILITY = 10000.0     # md
    MIN_POROSITY = 0.05
    MAX_POROSITY = 0.40
    MIN_PRESSURE = 500.0           # psi
    MAX_PRESSURE = 10000.0         # psi
    MIN_TEMPERATURE_F = 70.0       # °F
    MAX_TEMPERATURE_F = 350.0      # °F
    MIN_OIL_VISCOSITY = 0.1        # cP
    MAX_OIL_VISCOSITY = 1000.0     # cP

    # Physical limits
    MIN_PRODUCTIVITY_INDEX = 0.01  # bbl/day/psi
    MAX_DRAWDOWN_PSI = 5000.0      # psi (maximum reasonable drawdown)
    MAX_SATURATION_CHANGE_PER_DT = 0.05  # Maximum saturation change per timestep

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.

        Args:
            strict_mode: If True, use stricter tolerances for validation
        """
        self.strict_mode = strict_mode
        if strict_mode:
            self.MASS_BALANCE_TOLERANCE = 0.02  # 2% for strict mode
            self.DARCY_TOLERANCE = 0.05          # 5% for strict mode
            self.GOR_TOLERANCE = 0.10            # 10% for strict mode

    def validate_darcy_flow(
        self,
        oil_rate: float,
        drawdown_psi: float,
        pi: float,
        tolerance: float = None
    ) -> ValidationResult:
        """
        Check Darcy flow consistency: rate ≈ PI × drawdown

        Darcy's Law for radial flow:
            q = (2πkh/μB) × (Pe - Pw) / ln(re/rw)

        In field units with productivity index:
            q = PI × ΔP

        Args:
            oil_rate: Oil production rate (bbl/day)
            drawdown_psi: Pressure drawdown (psi)
            pi: Productivity index (bbl/day/psi)
            tolerance: Acceptable error fraction (default: uses class value)

        Returns:
            ValidationResult with pass/fail status
        """
        if tolerance is None:
            tolerance = self.DARCY_TOLERANCE

        metrics = {"oil_rate": oil_rate, "drawdown_psi": drawdown_psi, "pi": pi}

        # Validate inputs
        if pi < self.MIN_PRODUCTIVITY_INDEX:
            return ValidationResult(
                is_valid=False,
                error_message=f"PI too small: {pi:.6f} bbl/day/psi < {self.MIN_PRODUCTIVITY_INDEX}",
                validation_type="darcy_flow",
                metrics=metrics
            )

        if drawdown_psi <= 0:
            return ValidationResult(
                is_valid=False,
                error_message=f"Drawdown must be positive: {drawdown_psi:.1f} psi",
                validation_type="darcy_flow",
                metrics=metrics
            )

        # Expected rate from Darcy's law
        expected_rate = pi * drawdown_psi
        metrics["expected_rate"] = expected_rate

        # Calculate relative error
        if expected_rate > 0.001:
            relative_error = abs(oil_rate - expected_rate) / expected_rate
            metrics["relative_error"] = relative_error
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Expected rate too small: {expected_rate:.6f} bbl/day",
                validation_type="darcy_flow",
                metrics=metrics
            )

        # Check against tolerance
        if relative_error > tolerance:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Darcy flow violation: rate={oil_rate:.1f} bbl/day, "
                    f"expected={expected_rate:.1f} bbl/day (PI×drawdown), "
                    f"error={relative_error*100:.1f}% > {tolerance*100:.1f}%"
                ),
                validation_type="darcy_flow",
                metrics=metrics
            )

        return ValidationResult(
            is_valid=True,
            validation_type="darcy_flow",
            metrics=metrics
        )

    def validate_material_balance(
        self,
        injected: float,
        produced: float,
        stored_initial: float,
        stored_final: float,
        tolerance: float = None
    ) -> ValidationResult:
        """
        Check material balance: injected - produced = Δstorage

        For a closed system with injection/production:
            Mass_in - Mass_out = ΔMass_stored

        Args:
            injected: Total injected volume (bbl)
            produced: Total produced volume (bbl)
            stored_initial: Initial storage (bbl)
            stored_final: Final storage (bbl)
            tolerance: Acceptable error fraction (default: 5%)

        Returns:
            ValidationResult with pass/fail status
        """
        if tolerance is None:
            tolerance = self.MASS_BALANCE_TOLERANCE

        metrics = {
            "injected": injected,
            "produced": produced,
            "stored_initial": stored_initial,
            "stored_final": stored_final
        }

        # Material balance equation
        lhs = injected - produced  # Net input
        rhs = stored_final - stored_initial  # Change in storage
        metrics["net_input"] = lhs
        metrics["storage_change"] = rhs

        # Calculate relative error
        if abs(rhs) > 0.001:
            relative_error = abs(lhs - rhs) / abs(rhs)
        else:
            relative_error = abs(lhs - rhs)

        metrics["relative_error"] = relative_error

        if relative_error > tolerance:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Material balance violation: "
                    f"injected-produced={lhs:.1f} bbl, "
                    f"Δstorage={rhs:.1f} bbl, "
                    f"error={relative_error*100:.2f}% > {tolerance*100:.1f}%"
                ),
                validation_type="material_balance",
                metrics=metrics
            )

        return ValidationResult(
            is_valid=True,
            validation_type="material_balance",
            metrics=metrics
        )

    def validate_relative_perm_range(
        self,
        kro: np.ndarray,
        krg: np.ndarray,
        krw: Optional[np.ndarray] = None,
        tolerance: float = None
    ) -> ValidationResult:
        """
        Check relative permeability ranges: 0 ≤ kr ≤ 1

        Physical constraints:
        - kr ≥ 0 (no negative flow capacity)
        - kr ≤ 1 (end-point relative permeability)

        Args:
            kro: Oil relative permeability array
            krg: Gas relative permeability array
            krw: Water relative permeability array (optional)
            tolerance: Acceptable deviation from [0, 1] range

        Returns:
            ValidationResult with pass/fail status
        """
        if tolerance is None:
            tolerance = self.KR_TOLERANCE

        warnings = []
        metrics = {}

        # Check oil relative permeability
        kro_min, kro_max = np.min(kro), np.max(kro)
        metrics["kro_min"] = kro_min
        metrics["kro_max"] = kro_max

        if kro_min < -tolerance or kro_max > (1.0 + tolerance):
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Oil relative permeability out of range: "
                    f"min={kro_min:.4f}, max={kro_max:.4f}"
                ),
                validation_type="relative_perm",
                metrics=metrics
            )
        if kro_max < 0.8:
            warnings.append(f"Low max oil rel perm: {kro_max:.3f} < 0.8")

        # Check gas relative permeability
        krg_min, krg_max = np.min(krg), np.max(krg)
        metrics["krg_min"] = krg_min
        metrics["krg_max"] = krg_max

        if krg_min < -tolerance or krg_max > (1.0 + tolerance):
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Gas relative permeability out of range: "
                    f"min={krg_min:.4f}, max={krg_max:.4f}"
                ),
                validation_type="relative_perm",
                metrics=metrics
            )

        # Check water relative permeability if provided
        if krw is not None:
            krw_min, krw_max = np.min(krw), np.max(krw)
            metrics["krw_min"] = krw_min
            metrics["krw_max"] = krw_max

            if krw_min < -tolerance or krw_max > (1.0 + tolerance):
                return ValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Water relative permeability out of range: "
                        f"min={krw_min:.4f}, max={krw_max:.4f}"
                    ),
                    validation_type="relative_perm",
                    metrics=metrics
                )

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            validation_type="relative_perm",
            metrics=metrics
        )

    def validate_gor_consistency(
        self,
        gor: float,
        gas_rate: float,
        oil_rate: float,
        tolerance: float = None
    ) -> ValidationResult:
        """
        Check GOR consistency: GOR ≈ gas_rate / oil_rate

        Gas-Oil Ratio definition:
            GOR = q_gas / q_oil

        where q_gas is in scf/day and q_oil is in stb/day

        Args:
            gor: Gas-oil ratio (scf/stb)
            gas_rate: Gas production rate (MSCFD)
            oil_rate: Oil production rate (bbl/day)
            tolerance: Acceptable error fraction

        Returns:
            ValidationResult with pass/fail status
        """
        if tolerance is None:
            tolerance = self.GOR_TOLERANCE

        metrics = {"gor": gor, "gas_rate_mscfd": gas_rate, "oil_rate_bpd": oil_rate}

        if oil_rate < 0.001:
            return ValidationResult(
                is_valid=False,
                error_message=f"Oil rate too low: {oil_rate:.6f} bbl/day",
                validation_type="gor_consistency",
                metrics=metrics
            )

        # Convert gas rate to scf/day (1 MSCF = 1000 scf)
        gas_rate_scf_day = gas_rate * 1000.0

        # Expected GOR
        expected_gor = gas_rate_scf_day / oil_rate
        metrics["expected_gor"] = expected_gor

        relative_error = abs(gor - expected_gor) / (expected_gor + 1e-6)
        metrics["relative_error"] = relative_error

        if relative_error > tolerance:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"GOR inconsistency: reported={gor:.0f} scf/stb, "
                    f"calculated={expected_gor:.0f} scf/stb "
                    f"(gas_rate/oil_rate), error={relative_error*100:.1f}%"
                ),
                validation_type="gor_consistency",
                metrics=metrics
            )

        return ValidationResult(
            is_valid=True,
            validation_type="gor_consistency",
            metrics=metrics
        )

    def validate_production_limits(
        self,
        oil_rate: float,
        reservoir_pressure: float,
        bubble_point: float,
        pi: float,
        max_drawdown: float = None
    ) -> ValidationResult:
        """
        Check production rate is physically achievable.

        Maximum production rate is limited by:
        - Reservoir pressure (can't exceed P_reservoir)
        - Productivity index (well capacity)
        - Drawdown (P_reservoir - P_wellbore)

        Args:
            oil_rate: Oil production rate (bbl/day)
            reservoir_pressure: Average reservoir pressure (psi)
            bubble_point: Bubble point pressure (psi)
            pi: Productivity index (bbl/day/psi)
            max_drawdown: Maximum allowable drawdown (psi)

        Returns:
            ValidationResult with pass/fail status
        """
        if max_drawdown is None:
            max_drawdown = self.MAX_DRAWDOWN_PSI

        metrics = {
            "oil_rate": oil_rate,
            "reservoir_pressure": reservoir_pressure,
            "bubble_point": bubble_point,
            "pi": pi,
            "max_drawdown": max_drawdown
        }

        # Maximum possible drawdown
        effective_drawdown = min(max_drawdown, reservoir_pressure)

        # Maximum possible rate
        max_rate = pi * effective_drawdown
        metrics["max_rate"] = max_rate

        # Allow 50% margin for transient effects
        if oil_rate > max_rate * 1.5:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Production rate exceeds physical limit: "
                    f"rate={oil_rate:.1f} bbl/day > max={max_rate:.1f} bbl/day "
                    f"(using 50% transient margin)"
                ),
                validation_type="production_limits",
                metrics=metrics
            )

        # Check for solution gas drive
        if reservoir_pressure < bubble_point:
            # Below bubble point, should have gas production
            # This is just a warning, not an error
            pass

        return ValidationResult(
            is_valid=True,
            validation_type="production_limits",
            metrics=metrics
        )

    def validate_parameter_ranges(
        self,
        porosity: float,
        permeability: float,
        pressure: float,
        temperature_f: float = None,
        oil_viscosity_cp: float = None
    ) -> ValidationResult:
        """
        Validate parameters are within physical ranges.

        Args:
            porosity: Rock porosity (fraction)
            permeability: Permeability (md)
            pressure: Pressure (psi)
            temperature_f: Temperature (°F), optional
            oil_viscosity_cp: Oil viscosity (cP), optional

        Returns:
            ValidationResult with pass/fail status
        """
        errors = []
        metrics = {
            "porosity": porosity,
            "permeability": permeability,
            "pressure": pressure
        }

        # Check porosity
        if porosity < self.MIN_POROSITY or porosity > self.MAX_POROSITY:
            errors.append(
                f"Porosity {porosity:.3f} outside range "
                f"[{self.MIN_POROSITY}, {self.MAX_POROSITY}]"
            )

        # Check permeability
        if permeability < self.MIN_PERMEABILITY or permeability > self.MAX_PERMEABILITY:
            errors.append(
                f"Permeability {permeability:.1f} md outside range "
                f"[{self.MIN_PERMEABILITY}, {self.MAX_PERMEABILITY}] md"
            )

        # Check pressure
        if pressure < self.MIN_PRESSURE or pressure > self.MAX_PRESSURE:
            errors.append(
                f"Pressure {pressure:.1f} psi outside range "
                f"[{self.MIN_PRESSURE}, {self.MAX_PRESSURE}] psi"
            )

        # Optional checks
        if temperature_f is not None:
            metrics["temperature_f"] = temperature_f
            if temperature_f < self.MIN_TEMPERATURE_F or temperature_f > self.MAX_TEMPERATURE_F:
                errors.append(
                    f"Temperature {temperature_f:.1f}°F outside range "
                    f"[{self.MIN_TEMPERATURE_F}, {self.MAX_TEMPERATURE_F}]°F"
                )

        if oil_viscosity_cp is not None:
            metrics["oil_viscosity_cp"] = oil_viscosity_cp
            if oil_viscosity_cp < self.MIN_OIL_VISCOSITY or oil_viscosity_cp > self.MAX_OIL_VISCOSITY:
                errors.append(
                    f"Oil viscosity {oil_viscosity_cp:.2f} cP outside range "
                    f"[{self.MIN_OIL_VISCOSITY}, {self.MAX_OIL_VISCOSITY}] cP"
                )

        if errors:
            return ValidationResult(
                is_valid=False,
                error_message="; ".join(errors),
                validation_type="parameter_ranges",
                metrics=metrics
            )

        return ValidationResult(
            is_valid=True,
            validation_type="parameter_ranges",
            metrics=metrics
        )

    def validate_all(
        self,
        state: 'CCUSState',
        injection_rates: Dict[str, float],
        production_rates: Dict[str, float],
        pi: float,
        stored_initial: float = None,
        stored_final: float = None
    ) -> ValidationResult:
        """
        Run all validations and return combined result.

        This is the main entry point for comprehensive validation.

        Args:
            state: Current reservoir state
            injection_rates: Injection rates (bbl/day or MSCFD)
            production_rates: Production rates (bbl/day or MSCFD)
            pi: Productivity index (bbl/day/psi)
            stored_initial: Initial storage for mass balance (optional)
            stored_final: Final storage for mass balance (optional)

        Returns:
            Combined ValidationResult with all errors/warnings
        """
        all_warnings = []
        all_errors = []
        all_metrics = {}

        # Validate parameter ranges
        porosity_val = state.porosity[0] if hasattr(state.porosity, '__iter__') else state.porosity
        permeability_val = state.permeability[0] if hasattr(state.permeability, '__iter__') else state.permeability
        pressure_val = state.pressure[0] if hasattr(state.pressure, '__iter__') else state.pressure

        result = self.validate_parameter_ranges(
            porosity=porosity_val,
            permeability=permeability_val,
            pressure=pressure_val
        )
        all_metrics.update(result.metrics)
        if not result.is_valid:
            all_errors.append(result.error_message)
        all_warnings.extend(result.warnings)

        # Validate GOR consistency
        oil_rate = production_rates.get('oil', 0.0)
        gas_rate = production_rates.get('gas', 0.0)
        gor = production_rates.get('gor', gas_rate * 1000.0 / (oil_rate + 0.001))

        result = self.validate_gor_consistency(gor, gas_rate, oil_rate)
        all_metrics.update(result.metrics)
        if not result.is_valid:
            all_errors.append(result.error_message)
        all_warnings.extend(result.warnings)

        # Validate production limits
        reservoir_pressure = pressure_val
        bubble_point = 2000.0  # Default bubble point
        result = self.validate_production_limits(oil_rate, reservoir_pressure, bubble_point, pi)
        all_metrics.update(result.metrics)
        if not result.is_valid:
            all_errors.append(result.error_message)
        all_warnings.extend(result.warnings)

        # Validate Darcy flow
        producer_pressure = pressure_val
        producer_bhp = 1500.0  # Default BHP
        drawdown = producer_pressure - producer_bhp
        result = self.validate_darcy_flow(oil_rate, drawdown, pi)
        all_metrics.update(result.metrics)
        if not result.is_valid:
            all_errors.append(result.error_message)
        all_warnings.extend(result.warnings)

        # Validate material balance (if storage data provided)
        if stored_initial is not None and stored_final is not None:
            injected = injection_rates.get('co2', 0.0) + injection_rates.get('water', 0.0)
            produced = oil_rate  # Simplified
            result = self.validate_material_balance(injected, produced, stored_initial, stored_final)
            all_metrics.update(result.metrics)
            if not result.is_valid:
                all_errors.append(result.error_message)
            all_warnings.extend(result.warnings)

        # Combine results
        is_valid = len(all_errors) == 0
        error_message = "; ".join(all_errors) if all_errors else ""

        return ValidationResult(
            is_valid=is_valid,
            error_message=error_message,
            warnings=all_warnings,
            validation_type="all",
            metrics=all_metrics
        )


def validate_cfl_condition(
    velocity: float,
    dt: float,
    dx: float,
    cfl_limit: float = 0.5
) -> ValidationResult:
    """
    Validate CFL condition for numerical stability.

    CFL number: C = v × dt / dx
    For explicit schemes, C ≤ 1 is required for stability.
    For multiphase flow, C ≤ 0.5 is recommended.

    Args:
        velocity: Fluid velocity (ft/day)
        dt: Timestep (days)
        dx: Grid spacing (ft)
        cfl_limit: Maximum allowable CFL number

    Returns:
        ValidationResult with pass/fail status
    """
    cfl_number = abs(velocity) * dt / (dx + 1e-9)

    metrics = {
        "velocity": velocity,
        "dt": dt,
        "dx": dx,
        "cfl_number": cfl_number,
        "cfl_limit": cfl_limit
    }

    if cfl_number > cfl_limit:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"CFL condition violated: C={cfl_number:.3f} > {cfl_limit:.3f} "
                f"(v={velocity:.2f} ft/day, dt={dt:.3f} d, dx={dx:.2f} ft)"
            ),
            validation_type="cfl_condition",
            metrics=metrics
        )

    return ValidationResult(
        is_valid=True,
        validation_type="cfl_condition",
        metrics=metrics
    )


# Export public API
__all__ = [
    'ValidationResult',
    'PhysicalConsistencyValidator',
    'validate_cfl_condition',
]
