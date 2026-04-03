"""
Unified Fluid Properties for CO2 EOR Optimizer.

This module provides standardized PVT (Pressure-Volume-Temperature) property
calculations for reservoir fluids including oil, gas, and water phases.

Based on industry-standard correlations:
- Standing, M.B. (1942) A Pressure-Volume-Temperature Correlation for Mixtures
- Vasquez, M.E. & Beggs, H.D. (1980) Correlations for fluid physical property prediction
- Beggs, H.D. & Robinson, J.R. (1975) Estimating the viscosity of crude oil systems

References:
- Al-Marhoun, M.A. (1988) New PVT correlations for Middle East crude oils
- Dindoruk, B. & Christman, P.G. (2004) PVT properties and viscosity correlations
- Ahmed, T. (2010) Reservoir Engineering Handbook
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class FluidProperties:
    """
    General fluid properties for a single phase.

    Attributes:
        density: Phase density (kg/m³)
        viscosity: Phase viscosity (Pa·s)
        formation_volume_factor: Formation volume factor (reservoir volume / stock tank volume)
        compressibility: Fluid compressibility (1/Pa)
        molecular_weight: Molecular weight (kg/mol)
        critical_pressure: Critical pressure (Pa)
        critical_temperature: Critical temperature (K)
        acentric_factor: Acentric factor
    """

    density: float
    viscosity: float
    formation_volume_factor: float = 1.0
    compressibility: float = 1e-10
    molecular_weight: float = 0.0
    critical_pressure: float = 0.0
    critical_temperature: float = 0.0
    acentric_factor: float = 0.0


@dataclass
class BlackOilProperties:
    """
    Black oil PVT properties.

    Provides correlations for:
    - Oil formation volume factor (Bo)
    - Solution gas-oil ratio (Rs)
    - Oil viscosity (muo)
    - Gas formation volume factor (Bg)
    - Water properties
    """

    api_gravity: float = 30.0
    gas_specific_gravity: float = 0.6
    reservoir_temperature: float = 350.0
    pressure: float = 10e6

    pbubble: float = 10e6
    rsb: float = 100.0
    bob: float = 1.2
    muob: float = 0.005

    def __post_init__(self):
        if not (0 < self.api_gravity <= 50):
            warnings.warn(f"API gravity {self.api_gravity} outside typical range")

    def solution_gor(self, pressure: float) -> float:
        """
        Calculate solution gas-oil ratio (Rs) using Vasquez-Beggs correlation.

        Args:
            pressure: Pressure (Pa)

        Returns:
            Solution GOR (m³/m³)
        """
        if pressure >= self.pbubble:
            return self.rsb

        p_psi = pressure / 6894.76
        t_r = self.reservoir_temperature * 9 / 5 - 459.67

        c = 0.0362 * self.api_gravity * np.exp(25.7240 / (t_r + 459.67))
        b = 1.2044 * self.gas_specific_gravity * np.exp(11.1725 * self.api_gravity / (t_r + 459.67))

        rs = c * p_psi**b
        return min(rs, self.rsb)

    def oil_fvf(self, pressure: float) -> float:
        """
        Calculate oil formation volume factor (Bo) using Standing correlation.

        Args:
            pressure: Pressure (Pa)

        Returns:
            Oil FVF (m³/m³)
        """
        if pressure >= self.pbubble:
            return self.bob

        p_psi = pressure / 6894.76
        t_r = self.reservoir_temperature * 9 / 5 - 459.67

        rs = self.solution_gor(pressure)
        bg = 0.0009725 * (
            rs * (self.gas_specific_gravity / self.api_gravity) ** 0.526 + 0.000359 * t_r
        )

        bo = (
            0.972
            + 0.00038
            * (rs * (self.gas_specific_gravity / self.api_gravity) ** 0.5 + 1.25 * t_r) ** 1.2
        )
        return min(bo, self.bob)

    def oil_viscosity(self, pressure: float) -> float:
        """
        Calculate dead oil viscosity using Beggs-Robinson correlation.

        Args:
            pressure: Pressure (Pa)

        Returns:
            Oil viscosity (Pa·s)
        """
        t_r = self.reservoir_temperature * 9 / 5 - 459.67

        a = 10.715 * (self.gas_specific_gravity**0.515)
        b = 5.44 * (self.gas_specific_gravity**0.338) * (t_r + 200) ** (-1.038)

        mu_dead = a * (self.api_gravity**b) * (t_r ** (-1.163))

        if pressure >= self.pbubble:
            return self.muob

        rs = self.solution_gor(pressure)
        mu_live = (
            mu_dead * (rs**0.338) * np.exp(11.513 * self.api_gravity / (t_r + 200) - 0.0002698 * rs)
        )

        return min(mu_live, self.muob)

    def gas_fvf(self, pressure: float, temperature: float) -> float:
        """
        Calculate gas formation volume factor using real gas law with Z-factor.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)

        Returns:
            Gas FVF (m³/m³)
        """
        p_psi = pressure / 6894.76
        t_r = temperature * 9 / 5 - 459.67

        pr = p_psi / (
            493.0 + 0.2648 * self.gas_specific_gravity * t_r - 92.0 * self.gas_specific_gravity
        )
        tr = t_r / (168.0 + 325.0 * self.gas_specific_gravity - 12.5 * self.gas_specific_gravity**2)

        Z = self._calculate_z_factor(tr, pr)
        t_sc = 520.0
        p_sc = 14.7

        bg = 0.02827 * Z * t_r / (p_psi + 0.0001)
        return bg

    def _calculate_z_factor(self, tr: float, pr: float) -> float:
        """
        Calculate gas compressibility factor using Dranchuk-Abou-Kassem (DAK).

        Args:
            tr: Reduced temperature
            pr: Reduced pressure

        Returns:
            Z-factor
        """
        if tr < 0.95 or pr < 0.01 or pr > 30:
            return self._simple_z(tr, pr)

        A1 = 0.3265
        A2 = -1.0700
        A3 = -0.5339
        A4 = 0.01569
        A5 = -0.05165
        A6 = 0.5475
        A7 = -0.1057
        A8 = 0.6134
        A9 = 0.7210

        tr_inv = 1.0 / tr
        tr_inv2 = tr_inv * tr_inv
        pr2 = pr * pr

        A = (
            A1
            + A2 * tr_inv
            + A3 * tr_inv2
            + A4 * tr_inv2 * tr_inv
            + A5 * pr
            + A6 * pr * tr_inv
            + A7 * pr * tr_inv2
            + A8 * pr2 * tr_inv2
        )
        B = A6 + A7 * tr_inv + A8 * tr_inv2
        C = A9 * pr2 * tr_inv2

        z = 0.5
        for _ in range(25):
            z_prev = z
            z_safe = max(z, 0.1)

            denom1 = z_safe - B
            if abs(denom1) < 0.05:
                denom1 = 0.05 if denom1 >= 0 else -0.05

            denom2 = z_safe * (z_safe + B) * (z_safe - 1.0)
            if abs(denom2) < 1e-6:
                denom2 = 1e-6 if denom2 >= 0 else -1e-6

            rhs = (
                A * z_safe / denom1
                + C / denom2
                - z_safe * (z_safe + B)
                + (1 + A * z_safe * (1 + z_safe) * (1 + z_safe) / (denom1 * denom2))
            )

            z = 0.125 * (27 * rhs + np.sqrt(729 * rhs * rhs + 4)) ** (1.0 / 3.0)
            z = (2.0 / 3.0) * z + B / (3.0 * max(z, 0.1))

            if abs(z - z_prev) < 1e-6:
                break

        return max(0.1, min(z, 3.0))

    def _simple_z(self, tr: float, pr: float) -> float:
        """Simple Z-factor estimate for extreme conditions."""
        if tr >= 1.5:
            return 1.0 - 0.02 * pr / tr
        if tr >= 1.0:
            return 0.95 - 0.1 * pr / tr
        return 0.7 + 0.3 * pr / tr

    def water_properties(self, pressure: float, temperature: float) -> Dict[str, float]:
        """
        Calculate water properties.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)

        Returns:
            Dictionary with water properties
        """
        t_c = temperature - 273.15
        p_psi = pressure / 6894.76

        bw = 1.0 + 4.77e-7 * p_psi - 3.74e-6 * t_c

        mu_w = 0.00024 * np.exp(0.000002 * p_psi + 0.0034 * t_c)

        return {
            "density": 1040.0 * bw,
            "viscosity": mu_w,
            "formation_volume_factor": bw,
        }

    def properties(self, pressure: float, temperature: float) -> Dict[str, Any]:
        """
        Calculate all black oil properties.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)

        Returns:
            Dictionary with all PVT properties
        """
        return {
            "rs": self.solution_gor(pressure),
            "bo": self.oil_fvf(pressure),
            "muo": self.oil_viscosity(pressure),
            "bg": self.gas_fvf(pressure, temperature),
            "water": self.water_properties(pressure, temperature),
        }


class ViscosityCorrelation:
    """
    Base class for viscosity correlations.
    """

    def __init__(self, phase: str = "oil"):
        self.phase = phase

    def calculate(self, pressure: float, temperature: float, **kwargs) -> float:
        """Calculate viscosity."""
        raise NotImplementedError


class DensityCorrelation:
    """
    Base class for density correlations.
    """

    def __init__(self, phase: str = "oil"):
        self.phase = phase

    def calculate(self, pressure: float, temperature: float, **kwargs) -> float:
        """Calculate density."""
        raise NotImplementedError


def create_black_oil_properties(
    api_gravity: float = 30.0,
    gas_specific_gravity: float = 0.6,
    reservoir_temperature: float = 350.0,
    **kwargs,
) -> BlackOilProperties:
    """
    Factory function to create BlackOilProperties instance.

    Args:
        api_gravity: Oil API gravity
        gas_specific_gravity: Gas specific gravity (air = 1)
        reservoir_temperature: Reservoir temperature (K)
        **kwargs: Additional properties

    Returns:
        BlackOilProperties instance
    """
    return BlackOilProperties(
        api_gravity=api_gravity,
        gas_specific_gravity=gas_specific_gravity,
        reservoir_temperature=reservoir_temperature,
        **kwargs,
    )
