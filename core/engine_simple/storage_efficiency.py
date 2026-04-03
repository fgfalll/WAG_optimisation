"""
Storage Efficiency Calculations for CO₂-EOR and CCUS
====================================================

This module contains storage efficiency calculations including
areal sweep efficiency, vertical sweep efficiency, and trapping efficiency.

Based on the theoretical framework from the technical specification document.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class StorageParameters:
    """Parameters for storage efficiency calculations"""
    # Reservoir geometry
    area: float = 5e6           # Reservoir area (m²)
    thickness: float = 30.0     # Net thickness (m)
    porosity: float = 0.15      # Porosity (fraction)

    # Efficiency parameters (user inputs)
    areal_efficiency: float = 0.5      # Areal sweep efficiency (0-1)
    vertical_efficiency: float = 0.6   # Vertical sweep efficiency (0-1)
    trapping_efficiency: float = 0.2   # Trapping efficiency (0-1)

    # CO₂ properties
    co2_density: float = 700.0   # In-situ CO₂ density (kg/m³)

    # Operational parameters
    injection_rate: float = 0.0  # CO₂ injection rate (m³/day)
    production_rate: float = 0.0 # Oil production rate (m³/day)
    recycling_ratio: float = 0.8 # Recycling ratio (0-1)

    def __post_init__(self):
        """Validate storage parameters"""
        # Check geometry
        if any(param <= 0 for param in [self.area, self.thickness]):
            raise ValueError("Area and thickness must be positive")

        # Check porosity
        if not (0 < self.porosity < 1):
            raise ValueError("Porosity must be between 0 and 1")

        # Check efficiency parameters
        for name, value in [('areal_efficiency', self.areal_efficiency),
                            ('vertical_efficiency', self.vertical_efficiency),
                            ('trapping_efficiency', self.trapping_efficiency)]:
            if not (0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")

        # Check CO₂ density
        if self.co2_density <= 0:
            raise ValueError("CO₂ density must be positive")

        # Check recycling ratio
        if not (0 <= self.recycling_ratio <= 1):
            raise ValueError("Recycling ratio must be between 0 and 1")


class StorageEfficiencyCalculator:
    """
    Calculator for CO₂ storage efficiency and related metrics
    """

    def __init__(self, params: StorageParameters):
        """
        Initialize storage efficiency calculator

        Parameters:
        -----------
        params : StorageParameters
            Storage parameters
        """
        self.params = params

    def total_storage_efficiency(self) -> float:
        """
        Calculate total storage efficiency

        Returns:
        --------
        float : Total storage efficiency (0-1)
        """
        E = self.params.areal_efficiency * self.params.vertical_efficiency * self.params.trapping_efficiency
        return E

    def co2_storage_capacity(self) -> float:
        """
        Calculate total CO₂ storage capacity

        Returns:
        --------
        float : CO₂ storage capacity (kg)
        """
        E = self.total_storage_efficiency()
        M_co2 = (self.params.area * self.params.thickness * self.params.porosity *
                E * self.params.co2_density)
        return M_co2

    def co2_storage_volume(self) -> float:
        """
        Calculate CO₂ storage volume at reservoir conditions

        Returns:
        --------
        float : CO₂ storage volume (m³)
        """
        M_co2 = self.co2_storage_capacity()
        V_co2 = M_co2 / self.params.co2_density
        return V_co2

    def utilization_factor(self) -> float:
        """
        Calculate CO₂ utilization factor

        Returns:
        --------
        float : Utilization factor (m³ CO₂ per m³ oil)
        """
        if self.params.production_rate <= 0:
            warnings.warn("Production rate is zero, utilization factor is infinite")
            return np.inf

        uf = self.params.injection_rate / self.params.production_rate
        return uf

    def recycled_co2_volume(self) -> float:
        """
        Calculate recycled CO₂ volume

        Returns:
        --------
        float : Recycled CO₂ volume (m³/day)
        """
        V_recycled = self.params.injection_rate * self.params.recycling_ratio
        return V_recycled

    def net_co2_storage_rate(self) -> float:
        """
        Calculate net CO₂ storage rate (injection minus recycling)

        Returns:
        --------
        float : Net CO₂ storage rate (m³/day)
        """
        net_rate = self.params.injection_rate * (1 - self.params.recycling_ratio)
        return net_rate

    def breakthrough_time(self, distance: float, injection_rate: float,
                         area: float, porosity: float = None,
                         sw_front: float = None, sw_initial: float = None) -> float:
        """
        Calculate CO₂ breakthrough time using Buckley-Leverett theory

        Parameters:
        -----------
        distance : float
            Distance between injector and producer (m)
        injection_rate : float
            Injection rate (m³/s)
        area : float
            Cross-sectional area (m²)
        porosity : float, optional
            Porosity (uses default if None)
        sw_front : float, optional
            Saturation at CO₂ front (uses default if None)
        sw_initial : float, optional
            Initial water saturation (uses default if None)

        Returns:
        --------
        float : Breakthrough time (seconds)
        """
        if porosity is None:
            porosity = self.params.porosity

        if sw_front is None:
            # Estimate front saturation
            sw_front = 0.5

        if sw_initial is None:
            sw_initial = 0.2

        # Buckley-Leverett breakthrough calculation
        t_b = (porosity * distance * (sw_front - sw_initial)) / (injection_rate / area)
        return t_b

    def areal_sweep_efficiency_correlation(self, mobility_ratio: float,
                                         viscosity_ratio: float = None,
                                         pattern: str = 'five_spot') -> float:
        """
        Calculate areal sweep efficiency using empirical correlations

        Parameters:
        -----------
        mobility_ratio : float
            Mobility ratio (M)
        viscosity_ratio : float, optional
            Viscosity ratio (μo/μw)
        pattern : str
            Well pattern ('five_spot', 'line_drive', 'staggered')

        Returns:
        --------
        float : Areal sweep efficiency
        """
        if viscosity_ratio is not None:
            mobility_ratio = viscosity_ratio

        if pattern == 'five_spot':
            # Five-spot pattern correlation
            if mobility_ratio <= 1:
                Ea = 0.718 - 0.518 * mobility_ratio
            else:
                Ea = 0.718 / (mobility_ratio ** 0.25)
        elif pattern == 'line_drive':
            # Line drive pattern correlation
            if mobility_ratio <= 1:
                Ea = 0.9 - 0.4 * mobility_ratio
            else:
                Ea = 0.5 / (mobility_ratio ** 0.15)
        elif pattern == 'staggered':
            # Staggered line drive
            if mobility_ratio <= 1:
                Ea = 0.85 - 0.45 * mobility_ratio
            else:
                Ea = 0.6 / (mobility_ratio ** 0.2)
        else:
            raise ValueError(f"Unknown well pattern: {pattern}")

        return np.clip(Ea, 0.0, 1.0)

    def vertical_sweep_efficiency_correlation(self, permeability_contrast: float,
                                            viscosity_ratio: float,
                                            gravity_number: float = None) -> float:
        """
        Calculate vertical sweep efficiency

        Parameters:
        -----------
        permeability_contrast : float
            Ratio of maximum to minimum permeability
        viscosity_ratio : float
            Viscosity ratio (μo/μw)
        gravity_number : float, optional
            Gravity number (if None, calculated from default values)

        Returns:
        --------
        float : Vertical sweep efficiency
        """
        if gravity_number is None:
            # Default gravity number
            gravity_number = 0.1

        # Simplified correlation for vertical sweep efficiency
        kv = 1.0 / (1.0 + 0.2 * np.log(permeability_contrast))
        mg = 1.0 / (1.0 + 0.5 * np.log(viscosity_ratio))
        gg = 1.0 / (1.0 + 0.1 * np.log(1.0 / gravity_number))

        Ev = kv * mg * gg
        return np.clip(Ev, 0.0, 1.0)

    def dissolution_trapping_capacity(self, pressure: float, temperature: float,
                                    water_volume: float, salinity: float = 0.0) -> float:
        """
        Calculate CO₂ dissolution trapping capacity

        Parameters:
        -----------
        pressure : float
            Pressure (Pa)
        temperature : float
            Temperature (K)
        water_volume : float
            Water volume in reservoir (m³)
        salinity : float
            Water salinity (kg/kg)

        Returns:
        --------
        float : Dissolution trapping capacity (kg)
        """
        # Henry's constant for CO₂ in water
        # Using simplified correlation
        T_celsius = temperature - 273.15
        ln_kh = -58.0931 + 90.5069e3 / temperature + 22.2940 * np.log(temperature)
        kh = np.exp(ln_kh)  # mol/kg·MPa

        # Convert pressure to MPa
        p_mpa = pressure / 1e6

        # Calculate dissolved CO₂ concentration
        conc_mol_per_kg = kh * p_mpa

        # Account for salinity (Setschenow effect)
        setschenow_constant = 0.1
        salinity_factor = np.exp(-setschenow_constant * salinity)
        conc_mol_per_kg *= salinity_factor

        # Convert to mass
        molar_mass_co2 = 0.04401  # kg/mol
        water_density = 1000.0  # kg/m³
        mass_water = water_volume * water_density

        M_dissolved = conc_mol_per_kg * mass_water * molar_mass_co2
        return M_dissolved

    def residual_trapping_capacity(self, co2_volume: float,
                                 residual_gas_saturation: float) -> float:
        """
        Calculate residual (structural) trapping capacity

        Parameters:
        -----------
        co2_volume : float
            Total CO₂ volume injected (m³)
        residual_gas_saturation : float
            Residual gas saturation (fraction)

        Returns:
        --------
        float : Residual trapping capacity (m³)
        """
        V_residual = co2_volume * residual_gas_saturation
        return V_residual

    def mineral_trapping_capacity(self, time_years: float = 100.0) -> float:
        """
        Estimate mineral trapping capacity (simplified)

        Parameters:
        -----------
        time_years : float
            Time period for mineralization (years)

        Returns:
        --------
        float : Mineral trapping capacity (kg)
        """
        # Simplified mineral trapping model
        # Assumes 1% of dissolved CO₂ mineralizes per year
        dissolution_capacity = self.dissolution_trapping_capacity(
            pressure=20e6,  # 200 bar
            temperature=353.15,  # 80°C
            water_volume=self.params.area * self.params.thickness * self.params.porosity
        )

        # Exponential mineralization model
        mineralization_rate = 0.01  # 1% per year
        mineralized_fraction = 1 - np.exp(-mineralization_rate * time_years)

        M_mineral = dissolution_capacity * mineralized_fraction
        return M_mineral

    def storage_efficiency_breakdown(self) -> Dict[str, float]:
        """
        Break down storage efficiency by mechanism

        Returns:
        --------
        dict : Storage efficiency breakdown
        """
        total_efficiency = self.total_storage_efficiency()

        # Estimate individual components (simplified)
        structural_eff = total_efficiency * 0.4
        residual_eff = total_efficiency * 0.3
        dissolution_eff = total_efficiency * 0.25
        mineral_eff = total_efficiency * 0.05

        return {
            'structural': structural_eff,
            'residual': residual_eff,
            'dissolution': dissolution_eff,
            'mineral': mineral_eff,
            'total': total_efficiency
        }

    def calculate_gravity_number(self, density_co2: float, density_brine: float,
                               viscosity_co2: float, permeability: float,
                               injection_rate: float, area: float) -> float:
        """
        Calculate gravity number for CO₂ injection

        Parameters:
        -----------
        density_co2 : float
            CO₂ density (kg/m³)
        density_brine : float
            Brine density (kg/m³)
        viscosity_co2 : float
            CO₂ viscosity (Pa·s)
        permeability : float
            Permeability (m²)
        injection_rate : float
            Injection rate (m³/s)
        area : float
            Cross-sectional area (m²)

        Returns:
        --------
        float : Gravity number
        """
        g = 9.81  # Gravity acceleration (m/s²)
        delta_rho = density_brine - density_co2  # Density difference
        velocity = injection_rate / area  # Darcy velocity

        Ng = (permeability * delta_rho * g) / (viscosity_co2 * velocity)
        return Ng

    def storage_metrics_summary(self) -> Dict[str, float]:
        """
        Generate comprehensive storage metrics summary

        Returns:
        --------
        dict : Storage metrics
        """
        return {
            'total_storage_efficiency': self.total_storage_efficiency(),
            'storage_capacity_kg': self.co2_storage_capacity(),
            'storage_volume_m3': self.co2_storage_volume(),
            'utilization_factor': self.utilization_factor(),
            'recycling_ratio': self.params.recycling_ratio,
            'net_storage_rate': self.net_co2_storage_rate(),
            'areal_efficiency': self.params.areal_efficiency,
            'vertical_efficiency': self.params.vertical_efficiency,
            'trapping_efficiency': self.params.trapping_efficiency
        }

    def update_parameters(self, **kwargs):
        """
        Update storage parameters

        Parameters:
        -----------
        **kwargs : dict
            Parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")

        # Re-validate parameters
        self.params.__post_init__()