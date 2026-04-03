"""
Unified Geomechanics Module for CO2 EOR Optimizer.

This module provides stress/strain calculations, fault stability analysis,
and compaction effects for reservoir simulation.

Based on industry-standard formulations:
- Terzaghi, K. (1943) Theoretical Soil Mechanics
- Biot, M.A. (1941) General theory of three-dimensional consolidation
- Zoback, M.D. (2010) Reservoir Geomechanics

References:
- Jaeger, J.C., Cook, N.G.W., & Zimmerman, R.W. (2007) Fundamentals of Rock Mechanics
- Fjaer, E., Holt, R.M., Horsrud, P., Raaen, A.M., & Risnes, R. (2008) Petroleum Related Rock Mechanics
- Segall, P. (2010) Earthquake and Volcano Deformation
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class GeomechanicsParameters:
    """
    Geomechanical parameters for reservoir simulation.

    Attributes:
        E: Young's modulus (Pa)
        nu: Poisson's ratio (dimensionless)
        K_bulk: Bulk modulus (Pa)
        G_shear: Shear modulus (Pa)
        K_cond: Condensation modulus (Pa)
        vertical_stress: Overburden stress gradient (Pa/m)
        horizontal_stress_ratio: Ratio of horizontal to vertical stress
        pore_pressure: Initial pore pressure (Pa)
        rock_density: Rock density (kg/m³)
        depth: Reservoir depth (m)
    """

    E: float = 10e9
    nu: float = 0.25
    K_bulk: float = 6.67e9
    G_shear: float = 4.0e9
    K_cond: float = 0.0
    vertical_stress_gradient: float = 22.5e3
    horizontal_stress_ratio: float = 0.8
    pore_pressure: float = 10e6
    rock_density: float = 2500.0
    depth: float = 2000.0


class StressStrainCalculator:
    """
    Calculates stress and strain in the reservoir.

    Implements linear elasticity with poroelastic coupling.
    """

    def __init__(self, params: GeomechanicsParameters):
        self.params = params
        self._setup_derived_parameters()

    def _setup_derived_parameters(self):
        """Calculate derived geomechanical parameters."""
        E = self.params.E
        nu = self.params.nu

        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))

        if self.params.K_bulk > 0:
            self.K = self.params.K_bulk
        if self.params.G_shear > 0:
            self.G = self.params.G_shear

        self.lambda_lame = self.K - 2.0 * self.G / 3.0

    def vertical_stress(self, depth: float = None) -> float:
        """Calculate vertical stress (overburden)."""
        if depth is None:
            depth = self.params.depth
        return self.params.vertical_stress_gradient * depth

    def horizontal_stress(
        self, depth: float = None, stress_anisotropy: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate horizontal stresses.

        Args:
            depth: Depth (m)
            stress_anisotropy: Ratio of Shmax to Sv (typically 0.7-1.0)

        Returns:
            Tuple of (minimum horizontal, maximum horizontal) stresses
        """
        sv = self.vertical_stress(depth)
        shmin = sv * self.params.horizontal_stress_ratio * (1.0 - stress_anisotropy)
        shmax = sv * self.params.horizontal_stress_ratio * (1.0 + stress_anisotropy)
        return shmin, shmax

    def effective_stress(self, total_stress: float, pore_pressure: float = None) -> float:
        """Calculate effective stress (Terzaghi's principle)."""
        if pore_pressure is None:
            pore_pressure = self.params.pore_pressure
        return total_stress - pore_pressure

    def strain_from_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate strain from stress using Hooke's law.

        Args:
            stress_tensor: 3x3 stress tensor (Pa)

        Returns:
            3x3 strain tensor
        """
        eps = np.zeros((3, 3))
        E = self.params.E
        nu = self.params.nu

        for i in range(3):
            for j in range(3):
                if i == j:
                    eps[i, j] = (1.0 / E) * (
                        stress_tensor[i, j]
                        - nu
                        * (
                            stress_tensor[(i + 1) % 3, (i + 1) % 3]
                            + stress_tensor[(i + 2) % 3, (i + 2) % 3]
                        )
                    )
                else:
                    eps[i, j] = (1.0 + nu) / E * stress_tensor[i, j]

        return eps

    def stress_from_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate stress from strain using Hooke's law.

        Args:
            strain_tensor: 3x3 strain tensor

        Returns:
            3x3 stress tensor (Pa)
        """
        sig = np.zeros((3, 3))
        K = self.K
        G = self.G

        for i in range(3):
            volumetric_strain = strain_tensor[0, 0] + strain_tensor[1, 1] + strain_tensor[2, 2]
            for j in range(3):
                if i == j:
                    sig[i, j] = (3 * K * volumetric_strain / 3) + 2 * G * strain_tensor[i, j]
                else:
                    sig[i, j] = 2 * G * strain_tensor[i, j]

        return sig

    def pore_pressure_change(self, delta_pore_pressure: float, biot: float = 0.8) -> float:
        """
        Calculate change in effective stress due to pore pressure change.

        Args:
            delta_pore_pressure: Change in pore pressure (Pa)
            biot: Biot coefficient (0-1)

        Returns:
            Change in effective stress (Pa)
        """
        return biot * delta_pore_pressure

    def subsidence(
        self, delta_pore_pressure: float, thickness: float = 100.0, area: float = 1e6
    ) -> float:
        """
        Calculate reservoir subsidence.

        Args:
            delta_pore_pressure: Change in pore pressure (Pa)
            thickness: Reservoir thickness (m)
            area: Reservoir area (m²)

        Returns:
            Subsidence (m)
        """
        delta_eff_stress = self.pore_pressure_change(delta_pore_pressure)
        volumetric_strain = delta_eff_stress / self.K

        subsidence = volumetric_strain * thickness
        return subsidence


class FaultStabilityAnalyzer:
    """
    Analyzes fault stability using Mohr-Coulomb failure criteria.

    Determines if faults will slip under reservoir pressure changes.
    """

    def __init__(self, params: GeomechanicsParameters):
        self.params = params
        self.stress_calc = StressStrainCalculator(params)

    def normal_stress(self, fault_dip: float, fault_strike: float) -> float:
        """
        Calculate normal stress on a fault plane.

        Args:
            fault_dip: Fault dip angle (degrees)
            fault_strike: Fault strike direction (degrees)

        Returns:
            Normal stress (Pa)
        """
        sv = self.stress_calc.vertical_stress()
        shmin, shmax = self.stress_calc.horizontal_stress()

        sigma_n = (
            sv * np.cos(np.radians(fault_dip)) ** 2 + shmin * np.sin(np.radians(fault_dip)) ** 2
        )
        return sigma_n

    def shear_stress(self, fault_dip: float, fault_strike: float) -> float:
        """
        Calculate shear stress on a fault plane.

        Args:
            fault_dip: Fault dip angle (degrees)
            fault_strike: Fault strike direction (degrees)

        Returns:
            Shear stress (Pa)
        """
        sv = self.stress_calc.vertical_stress()
        shmin, shmax = self.stress_calc.horizontal_stress()

        tau = (sv - shmin) * np.sin(np.radians(fault_dip)) * np.cos(np.radians(fault_dip))
        return abs(tau)

    def slip_tendency(
        self, fault_dip: float, fault_strike: float, pore_pressure: float = None
    ) -> float:
        """
        Calculate slip tendency (shear/normal stress ratio).

        Args:
            fault_dip: Fault dip angle (degrees)
            fault_strike: Fault strike direction (degrees)
            pore_pressure: Current pore pressure (Pa)

        Returns:
            Slip tendency (0-1, higher = more likely to slip)
        """
        if pore_pressure is None:
            pore_pressure = self.params.pore_pressure

        sigma_n = self.normal_stress(fault_dip, fault_strike)
        tau = self.shear_stress(fault_dip, fault_strike)

        sigma_n_eff = sigma_n - pore_pressure

        if sigma_n_eff <= 0:
            return 1.0

        return tau / sigma_n_eff

    def coulomb_failure(
        self,
        fault_dip: float,
        fault_strike: float,
        friction_coefficient: float = 0.6,
        cohesion: float = 0.0,
        pore_pressure: float = None,
    ) -> Tuple[bool, float]:
        """
        Check if fault will fail using Coulomb criterion.

        tau >= mu * (sigma_n - pore_pressure) + cohesion

        Args:
            fault_dip: Fault dip angle (degrees)
            fault_strike: Fault strike direction (degrees)
            friction_coefficient: Friction coefficient (typically 0.6-0.85)
            cohesion: Cohesion (Pa)
            pore_pressure: Current pore pressure (Pa)

        Returns:
            Tuple of (will_fail, safety_factor)
        """
        if pore_pressure is None:
            pore_pressure = self.params.pore_pressure

        sigma_n = self.normal_stress(fault_dip, fault_strike)
        tau = self.shear_stress(fault_dip, fault_strike)

        sigma_n_eff = sigma_n - pore_pressure
        critical_shear = friction_coefficient * sigma_n_eff + cohesion

        will_fail = tau >= critical_shear
        safety_factor = critical_shear / tau if tau > 0 else 0.0

        return will_fail, safety_factor


class CompactionCalculator:
    """
    Calculates rock compaction due to pressure changes.

    Implements porosity-permeability coupling and compaction drive.
    """

    def __init__(self, params: GeomechanicsParameters, porosity: float = 0.2):
        self.params = params
        self.initial_porosity = porosity
        self.stress_calc = StressStrainCalculator(params)

    def porosity_change(
        self, delta_pore_pressure: float, biot: float = 0.8, compressibility: float = 1e-10
    ) -> float:
        """
        Calculate change in porosity.

        Args:
            delta_pore_pressure: Change in pore pressure (Pa)
            biot: Biot coefficient
            compressibility: Rock compressibility (1/Pa)

        Returns:
            Change in porosity (fraction)
        """
        delta_eff_stress = self.stress_calc.pore_pressure_change(delta_pore_pressure, biot)
        return compressibility * delta_eff_stress

    def permeability_change(
        self, delta_porosity: float, permeability_exponent: float = 3.0
    ) -> float:
        """
        Calculate change in permeability using power-law relationship.

        k/k0 = (phi/phi0)^n

        Args:
            delta_porosity: Change in porosity
            permeability_exponent: Permeability-porosity exponent (typically 2-4)

        Returns:
            Permeability ratio (k/k0)
        """
        phi = np.clip(self.initial_porosity + delta_porosity, 0.01, 0.99)
        phi0 = self.initial_porosity

        return (phi / phi0) ** permeability_exponent

    def compaction_drive(
        self, pore_volume: float, delta_pore_pressure: float, rock_compressibility: float = 1e-10
    ) -> float:
        """
        Calculate energy from compaction drive.

        Args:
            pore_volume: Initial pore volume (m³)
            delta_pore_pressure: Change in pore pressure (Pa)
            rock_compressibility: Rock compressibility (1/Pa)

        Returns:
            Compaction energy (J)
        """
        return 0.5 * pore_volume * rock_compressibility * delta_pore_pressure**2

    def pore_volume_strain(self, delta_pore_pressure: float, biot: float = 0.8) -> float:
        """
        Calculate pore volume strain.

        Args:
            delta_pore_pressure: Change in pore pressure (Pa)
            biot: Biot coefficient

        Returns:
            Volumetric strain
        """
        delta_eff = self.stress_calc.pore_pressure_change(delta_pore_pressure, biot)
        return delta_eff / self.stress_calc.K


def create_geomechanics_parameters(
    depth: float = 2000.0, E: float = 10e9, nu: float = 0.25, pore_pressure: float = 10e6
) -> GeomechanicsParameters:
    """
    Factory function to create GeomechanicsParameters.

    Args:
        depth: Reservoir depth (m)
        E: Young's modulus (Pa)
        nu: Poisson's ratio
        pore_pressure: Initial pore pressure (Pa)

    Returns:
        GeomechanicsParameters instance
    """
    return GeomechanicsParameters(depth=depth, E=E, nu=nu, pore_pressure=pore_pressure)
