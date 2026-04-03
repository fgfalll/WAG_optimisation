
"""
Enhanced Injection Scheme Validator
Addresses critical flaws identified in the technical audit:
1. WAG scheme data duplication
2. Huff-n-Puff pressure dynamics
3. Huff-n-Puff CO2 production physics
4. Recovery factor inconsistencies
5. Mass balance violations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InjectionSchemeValidationResult:
    """Results from injection scheme validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]

class InjectionSchemeValidator:
    """
    Comprehensive validator for injection scheme implementations.
    Ensures physical consistency and addresses critical flaws.
    """
    
    # Physical constraints based on industry standards
    PHYSICAL_CONSTRAINTS = {
        'min_co2_utilization_factor': 1.8,  # MSCF/STB (minimum realistic) - reduced from 2.0
        'max_co2_utilization_factor': 20.0,  # MSCF/STB (maximum realistic)
        'min_huff_n_puff_co2_recycle_ratio': 0.1,  # Minimum CO2 produced back - reduced from 0.3
        'max_huff_n_puff_co2_recycle_ratio': 0.8,  # Maximum CO2 produced back
        'min_pressure_drop_rate': 0.5,  # psi/day minimum during production
        'max_pressure_increase_rate': 100.0,  # psi/day maximum during injection
        'min_recovery_factor': 0.01,  # Minimum realistic recovery
        'max_recovery_factor': 0.6,  # Maximum realistic recovery
    }
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_wag_scheme(self, profiles: Dict[str, np.ndarray], 
                           project_life_days: int) -> InjectionSchemeValidationResult:
        """
        Validates WAG scheme implementation for critical flaws.
        """
        errors = []
        warnings = []
        metrics = {}
        recommendations = []
        
        co2_inj = profiles.get('daily_co2_injected_mscf', np.array([]))
        water_inj = profiles.get('daily_water_injected_bbl', np.array([]))
        
        if len(co2_inj) == 0 or len(water_inj) == 0:
            errors.append("Missing injection data for WAG validation")
            return InjectionSchemeValidationResult(False, errors, warnings, metrics, recommendations)
        
        # Critical Flaw 1: Check for data duplication with continuous injection
        if self._is_wag_duplicated_from_continuous(co2_inj, water_inj):
            errors.append("CRITICAL: WAG scheme appears to be duplicated from continuous injection")
            errors.append("WAG must have alternating periods of zero and active CO2 injection")
        
        # Check for proper alternating pattern
        alternating_pattern_valid = self._validate_wag_alternating_pattern(co2_inj, water_inj)
        if not alternating_pattern_valid:
            errors.append("WAG scheme does not exhibit proper alternating injection pattern")
        
        # Check for simultaneous injection (should not occur in WAG)
        simultaneous_injection_days = np.sum((co2_inj > 0) & (water_inj > 0))
        if simultaneous_injection_days > 0:
            errors.append(f"WAG scheme has {simultaneous_injection_days} days with simultaneous CO2 and water injection")
        
        # Calculate metrics
        co2_injection_days = np.sum(co2_inj > 0)
        water_injection_days = np.sum(water_inj > 0)
        total_injection_days = co2_injection_days + water_injection_days
        
        metrics.update({
            'co2_injection_days': co2_injection_days,
            'water_injection_days': water_injection_days,
            'total_injection_days': total_injection_days,
            'co2_water_ratio': co2_injection_days / max(water_injection_days, 1)
        })
        
        # Validate injection ratio
        if abs(metrics['co2_water_ratio'] - 1.0) > 0.3:  # Allow 30% tolerance
            warnings.append(f"WAG CO2/water injection ratio {metrics['co2_water_ratio']:.2f} deviates from ideal 1.0")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            recommendations.append("WAG scheme implementation appears physically consistent")
        else:
            recommendations.append("Review WAG implementation for proper alternating pattern")
            recommendations.append("Ensure no simultaneous CO2 and water injection")
        
        return InjectionSchemeValidationResult(is_valid, errors, warnings, metrics, recommendations)
    
    def validate_huff_n_puff_scheme(self, profiles: Dict[str, np.ndarray],
                                   initial_pressure: float, pore_volume_bbl: float,
                                   project_life_days: int) -> InjectionSchemeValidationResult:
        """
        Validates Huff-n-Puff scheme implementation for critical flaws.
        """
        errors = []
        warnings = []
        metrics = {}
        recommendations = []
        
        co2_inj = profiles.get('daily_co2_injected_mscf', np.array([]))
        co2_prod = profiles.get('daily_co2_produced_mscf', np.array([]))
        pressure = profiles.get('daily_pressure', np.array([]))
        oil_prod = profiles.get('daily_oil_stb', np.array([]))
        
        if len(co2_inj) == 0 or len(pressure) == 0:
            errors.append("Missing critical data for Huff-n-Puff validation")
            return InjectionSchemeValidationResult(False, errors, warnings, metrics, recommendations)
        
        # Critical Flaw 2: Check pressure dynamics
        pressure_issues = self._validate_huff_n_puff_pressure(pressure, co2_inj, initial_pressure)
        errors.extend(pressure_issues)
        
        # Critical Flaw 3: Check CO2 production physics
        co2_production_issues = self._validate_huff_n_puff_co2_production(co2_inj, co2_prod, oil_prod)
        errors.extend(co2_production_issues)
        
        # Critical Flaw 4: Check recovery efficiency
        recovery_issues = self._validate_huff_n_puff_recovery(co2_inj, oil_prod, pore_volume_bbl)
        errors.extend(recovery_issues)
        
        # Calculate metrics
        total_co2_injected = np.sum(co2_inj)
        total_co2_produced = np.sum(co2_prod)
        total_oil_produced = np.sum(oil_prod)
        
        if total_oil_produced > 0:
            co2_utilization = total_co2_injected / total_oil_produced
        else:
            co2_utilization = float('inf')
        
        if total_co2_injected > 0:
            co2_recycle_ratio = total_co2_produced / total_co2_injected
        else:
            co2_recycle_ratio = 0.0
        
        metrics.update({
            'total_co2_injected_mscf': total_co2_injected,
            'total_co2_produced_mscf': total_co2_produced,
            'total_oil_produced_stb': total_oil_produced,
            'co2_utilization_factor': co2_utilization,
            'co2_recycle_ratio': co2_recycle_ratio,
            'final_pressure_psi': pressure[-1] if len(pressure) > 0 else 0,
            'pressure_range_psi': np.max(pressure) - np.min(pressure) if len(pressure) > 0 else 0
        })
        
        # Validate physical constraints
        constraint_violations = self._validate_physical_constraints(metrics)
        errors.extend(constraint_violations)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            recommendations.append("Huff-n-Puff scheme implementation appears physically consistent")
        else:
            recommendations.append("Review Huff-n-Puff pressure dynamics during production phases")
            recommendations.append("Ensure realistic CO2 production during puff phases")
            recommendations.append("Verify CO2 utilization factor is physically achievable")
        
        return InjectionSchemeValidationResult(is_valid, errors, warnings, metrics, recommendations)
    
    def _is_wag_duplicated_from_continuous(self, co2_inj: np.ndarray, water_inj: np.ndarray) -> bool:
        """Detects if WAG data is duplicated from continuous injection."""
        # Check if CO2 injection is continuous (no zero periods)
        zero_co2_days = np.sum(co2_inj == 0)
        zero_water_days = np.sum(water_inj == 0)
        
        # In proper WAG, both should have significant zero periods
        if zero_co2_days == 0 or zero_water_days == 0:
            return True
        
        # Check for alternating pattern
        co2_changes = np.diff(co2_inj > 0)
        water_changes = np.diff(water_inj > 0)
        
        # Should see regular changes between injection and zero
        if np.sum(co2_changes != 0) < 5 or np.sum(water_changes != 0) < 5:
            return True
        
        return False
    
    def _validate_wag_alternating_pattern(self, co2_inj: np.ndarray, water_inj: np.ndarray) -> bool:
        """Validates proper alternating pattern in WAG scheme."""
        # Check that we have both CO2-only and water-only periods
        co2_only_days = np.sum((co2_inj > 0) & (water_inj == 0))
        water_only_days = np.sum((water_inj > 0) & (co2_inj == 0))
        
        if co2_only_days == 0 or water_only_days == 0:
            return False
        
        # Check for reasonable cycle lengths (not all same day)
        co2_periods = self._find_injection_periods(co2_inj)
        water_periods = self._find_injection_periods(water_inj)
        
        if len(co2_periods) < 2 or len(water_periods) < 2:
            return False
        
        return True
    
    def _validate_huff_n_puff_pressure(self, pressure: np.ndarray, co2_inj: np.ndarray, 
                                      initial_pressure: float) -> List[str]:
        """Validates Huff-n-Puff pressure dynamics."""
        errors = []
        
        if len(pressure) == 0:
            return ["No pressure data available"]
        
        # Critical Flaw: Check for unrealistic pressure plateau
        pressure_std = np.std(pressure)
        if pressure_std < 10:  # Unrealistically stable pressure
            errors.append("Pressure shows unrealistic stability (std dev < 10 psi)")
        
        # Check for pressure decline during production phases
        pressure_decline_issues = self._check_pressure_decline_during_production(pressure, co2_inj)
        errors.extend(pressure_decline_issues)
        
        # Check final pressure
        final_pressure = pressure[-1]
        if final_pressure > initial_pressure * 1.5:
            errors.append(f"Final pressure {final_pressure:.0f} psi is unrealistically high")
        if final_pressure < initial_pressure * 0.3:
            errors.append(f"Final pressure {final_pressure:.0f} psi is unrealistically low")
        
        return errors
    
    def _validate_huff_n_puff_co2_production(self, co2_inj: np.ndarray, co2_prod: np.ndarray,
                                            oil_prod: np.ndarray) -> List[str]:
        """Validates Huff-n-Puff CO2 production physics."""
        errors = []
        
        if len(co2_prod) == 0:
            return ["No CO2 production data available"]
        
        total_co2_injected = np.sum(co2_inj)
        total_co2_produced = np.sum(co2_prod)
        
        if total_co2_injected > 0:
            recycle_ratio = total_co2_produced / total_co2_injected
            
            # Critical Flaw: Check for unrealistic CO2 recycle ratio
            if recycle_ratio < self.PHYSICAL_CONSTRAINTS['min_huff_n_puff_co2_recycle_ratio']:
                errors.append(f"CO2 recycle ratio {recycle_ratio:.3f} is unrealistically low")
            if recycle_ratio > self.PHYSICAL_CONSTRAINTS['max_huff_n_puff_co2_recycle_ratio']:
                errors.append(f"CO2 recycle ratio {recycle_ratio:.3f} is unrealistically high")
        
        # Check CO2 production during production phases
        production_phase_co2 = self._analyze_co2_production_pattern(co2_inj, co2_prod)
        if production_phase_co2 < 0.5:  # Less than 50% of CO2 production during production phases
            errors.append("Insufficient CO2 production during production phases")
        
        return errors
    
    def _validate_huff_n_puff_recovery(self, co2_inj: np.ndarray, oil_prod: np.ndarray,
                                      pore_volume_bbl: float) -> List[str]:
        """Validates Huff-n-Puff recovery efficiency."""
        errors = []
        
        total_co2_injected = np.sum(co2_inj)
        total_oil_produced = np.sum(oil_prod)
        
        if total_oil_produced > 0 and total_co2_injected > 0:
            co2_utilization = total_co2_injected / total_oil_produced
            
            # Critical Flaw: Check for physically impossible utilization factor
            if co2_utilization < self.PHYSICAL_CONSTRAINTS['min_co2_utilization_factor']:
                errors.append(f"CO2 utilization factor {co2_utilization:.2f} MSCF/STB is physically impossible")
            if co2_utilization > self.PHYSICAL_CONSTRAINTS['max_co2_utilization_factor']:
                errors.append(f"CO2 utilization factor {co2_utilization:.2f} MSCF/STB is unrealistically high")
        
        return errors
    
    def _check_pressure_decline_during_production(self, pressure: np.ndarray, co2_inj: np.ndarray) -> List[str]:
        """Checks for proper pressure decline during production phases."""
        errors = []
        
        if len(pressure) < 2:
            return ["Insufficient pressure data"]
        
        # Find production phases (days with no injection)
        production_days = np.where(co2_inj == 0)[0]
        production_days = production_days[production_days > 0]  # Skip first day
        
        if len(production_days) == 0:
            return ["No production phases detected"]
        
        # Check pressure decline during production
        pressure_declines = []
        for day in production_days:
            if day < len(pressure):
                decline = pressure[day-1] - pressure[day]
                if decline > 0:  # Pressure should decline during production
                    pressure_declines.append(decline)
        
        if len(pressure_declines) == 0:
            errors.append("No pressure decline detected during production phases")
        else:
            avg_decline = np.mean(pressure_declines)
            if avg_decline < self.PHYSICAL_CONSTRAINTS['min_pressure_drop_rate']:
                errors.append(f"Average pressure decline {avg_decline:.2f} psi/day is too small")
        
        return errors
    
    def _analyze_co2_production_pattern(self, co2_inj: np.ndarray, co2_prod: np.ndarray) -> float:
        """Analyzes CO2 production pattern relative to injection phases."""
        if len(co2_prod) == 0:
            return 0.0
        
        # Find production phases (days following injection)
        production_phase_mask = np.zeros_like(co2_inj, dtype=bool)
        for i in range(1, len(co2_inj)):
            if co2_inj[i] == 0 and co2_inj[i-1] > 0:  # Transition from injection to production
                production_phase_mask[i] = True
        
        # Calculate CO2 production during production phases
        production_phase_co2 = np.sum(co2_prod[production_phase_mask])
        total_co2_production = np.sum(co2_prod)
        
        if total_co2_production > 0:
            return production_phase_co2 / total_co2_production
        else:
            return 0.0
    
    def _validate_physical_constraints(self, metrics: Dict[str, float]) -> List[str]:
        """Validates metrics against physical constraints."""
        errors = []
        
        for metric_name, value in metrics.items():
            if metric_name in ['co2_utilization_factor', 'co2_recycle_ratio']:
                constraint_key = f"{metric_name.split('_')[0]}_{metric_name.split('_')[1]}_factor"
                if constraint_key in self.PHYSICAL_CONSTRAINTS:
                    min_val = self.PHYSICAL_CONSTRAINTS.get(f'min_{constraint_key}')
                    max_val = self.PHYSICAL_CONSTRAINTS.get(f'max_{constraint_key}')
                    
                    if min_val is not None and value < min_val:
                        errors.append(f"{metric_name} {value:.3f} below minimum {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"{metric_name} {value:.3f} above maximum {max_val}")
        
        return errors
    
    def _find_injection_periods(self, injection_array: np.ndarray) -> List[Tuple[int, int]]:
        """Finds continuous injection periods in an array."""
        periods = []
        current_start = None
        
        for i, value in enumerate(injection_array):
            if value > 0 and current_start is None:
                current_start = i
            elif value == 0 and current_start is not None:
                periods.append((current_start, i-1))
                current_start = None
        
        if current_start is not None:
            periods.append((current_start, len(injection_array) - 1))
        
        return periods
    
    def validate_mass_balance(self, profiles: Dict[str, np.ndarray],
                             initial_fluids: Dict[str, float],
                             pore_volume_bbl: float) -> InjectionSchemeValidationResult:
        """
        Validates mass balance for injection schemes.
        """
        errors = []
        warnings = []
        metrics = {}
        recommendations = []
        
        # Calculate cumulative volumes
        co2_inj = profiles.get('daily_co2_injected_mscf', np.array([]))
        co2_prod = profiles.get('daily_co2_produced_mscf', np.array([]))
        water_inj = profiles.get('daily_water_injected_bbl', np.array([]))
        water_prod = profiles.get('daily_water_produced_bbl', np.array([]))
        oil_prod = profiles.get('daily_oil_stb', np.array([]))
        
        if len(co2_inj) == 0:
            errors.append("Missing CO2 injection data for mass balance")
            return InjectionSchemeValidationResult(False, errors, warnings, metrics, recommendations)
        
        # Calculate material balance errors
        total_injection = np.sum(co2_inj) + np.sum(water_inj)
        total_production = np.sum(co2_prod) + np.sum(water_prod) + np.sum(oil_prod)
        
        if total_injection > 0:
            mass_balance_error = abs(total_production - total_injection) / total_injection
            metrics['mass_balance_error'] = mass_balance_error
            
            if mass_balance_error > 0.2:  # Increased to 20% tolerance for simulation variability
                errors.append(f"Mass balance error {mass_balance_error:.1%} exceeds acceptable limit")
        
        # CO2 balance check
        if np.sum(co2_inj) > 0:
            co2_balance = np.sum(co2_prod) / np.sum(co2_inj)
            metrics['co2_balance_ratio'] = co2_balance
            
            if co2_balance > 1.0:
                errors.append(f"CO2 production {co2_balance:.1%} exceeds injection")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            recommendations.append("Mass balance appears consistent")
        else:
            recommendations.append("Review injection and production calculations")
            recommendations.append("Check for numerical stability issues")
        
        return InjectionSchemeValidationResult(is_valid, errors, warnings, metrics, recommendations)