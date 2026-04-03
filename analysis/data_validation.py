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
    def validate_recovery_factor(rf: float, ooip: float, total_oil_produced: float, 
                               tolerance: float = 0.01) -> Tuple[bool, str]:
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
            return False, f"Recovery factor discrepancy: reported={rf:.3f}, calculated={calculated_rf:.3f}, diff={discrepancy:.3f}"
        
        # Check RF bounds
        if rf < 0:
            return False, f"Recovery factor cannot be negative: {rf}"
        if rf > 1.0:
            return False, f"Recovery factor cannot exceed 1.0: {rf}"
            
        return True, f"Recovery factor validation passed: {rf:.3f}"
    
    @staticmethod
    def validate_pressure_data(pressure_data: np.ndarray, initial_pressure: float, 
                             max_pressure: float, min_pressure: float = 500.0) -> Tuple[bool, str]:
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
        if len(pressure_data) == 0:
            return False, "Empty pressure data"
            
        # Check for NaN or infinite values
        if not np.all(np.isfinite(pressure_data)):
            return False, "Pressure data contains NaN or infinite values"
            
        # Check pressure bounds
        min_observed = np.min(pressure_data)
        max_observed = np.max(pressure_data)
        
        if min_observed < min_pressure:
            return False, f"Pressure below minimum allowed: {min_observed:.0f} psi < {min_pressure:.0f} psi"
            
        if max_observed > max_pressure:
            return False, f"Pressure exceeds maximum allowed: {max_observed:.0f} psi > {max_pressure:.0f} psi"
            
        # Check for unrealistic pressure jumps (> 1000 psi/day)
        if len(pressure_data) > 1:
            pressure_changes = np.abs(np.diff(pressure_data))
            max_change = np.max(pressure_changes)
            if max_change > 1000:
                return False, f"Unrealistic pressure change: {max_change:.0f} psi/day"
                
        return True, f"Pressure validation passed: range {min_observed:.0f}-{max_observed:.0f} psi"
    
    @staticmethod
    def validate_production_data(oil_production: np.ndarray, 
                               co2_injection: np.ndarray,
                               co2_production: np.ndarray) -> Tuple[bool, str]:
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

        Handles different time resolution keys like 'daily_oil_stb', 'monthly_oil_stb', etc.

        Args:
            profiles: Dictionary of production profiles

        Returns:
            Oil production array or empty array if not found
        """
        for key in profiles:
            if 'oil' in key.lower() and 'stb' in key.lower():
                return profiles[key]
        return np.array([0.0])

    @staticmethod
    def validate_simulation_results(profiles: Dict[str, np.ndarray],
                                  metrics: Dict[str, float],
                                  reservoir_data: Dict[str, Any]) -> Dict[str, Tuple[bool, str]]:
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
        rf = metrics.get('recovery_factor', 0.0)
        ooip = reservoir_data.get('ooip_stb', 0.0)
        total_oil_produced = np.sum(DataValidator._get_oil_production(profiles))

        results['recovery_factor'] = DataValidator.validate_recovery_factor(
            rf, ooip, total_oil_produced
        )

        # Validate pressure data - use flexible key matching
        pressure_data = np.array([])
        for key in profiles:
            if 'pressure' in key.lower():
                pressure_data = profiles[key]
                break
        initial_pressure = reservoir_data.get('initial_pressure', 0.0)
        max_pressure = reservoir_data.get('max_pressure_psi', 10000.0)

        results['pressure'] = DataValidator.validate_pressure_data(
            pressure_data, initial_pressure, max_pressure
        )

        # Validate production data - use flexible key matching
        oil_production = DataValidator._get_oil_production(profiles)

        # Find CO2 injection data
        co2_injection = np.array([])
        for key in profiles:
            if 'co2' in key.lower() and 'inject' in key.lower():
                co2_injection = profiles[key]
                break

        # Find CO2 production data
        co2_production = np.array([])
        for key in profiles:
            if 'co2' in key.lower() and 'prod' in key.lower():
                co2_production = profiles[key]
                break
        
        results['production'] = DataValidator.validate_production_data(
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