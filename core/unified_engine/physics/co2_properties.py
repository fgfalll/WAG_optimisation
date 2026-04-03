"""
Unified CO2 Properties for CO2 EOR Optimizer.

This module provides standardized CO2 property calculations including
density, viscosity, solubility, and Minimum Miscibility Pressure (MMP).

Based on industry-standard correlations and thermodynamic models.

References:
- Peng, D.Y. & Robinson, D.B. (1976) A New Two-Constant Equation of State
- Yellig, W.F. & Metcalfe, R.S. (1980) Determination and prediction of CO2 MMPs
- Fenghour, A., Wakeham, W.A., & Vesovic, V. (1998) The viscosity of carbon dioxide
- Duan, Z. et al. (2008) Modeling the carbonate system in CO2 sequestration
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import warnings

try:
    from core.data_models import PhysicalConstants
except ImportError:
    PhysicalConstants = None


@dataclass
class CO2Properties:
    """
    CO2 property calculations for EOR simulation.

    Attributes:
        critical_pressure: Critical pressure (Pa), default ~7.38 MPa
        critical_temperature: Critical temperature (K), default ~304.13 K
        molecular_weight: Molecular weight (kg/mol), default ~0.04401 kg/mol
        acentric_factor: Acentric factor, default ~0.225
    """

    critical_pressure: float = 7.376e6
    critical_temperature: float = 304.13
    molecular_weight: float = 0.04401
    acentric_factor: float = 0.225
    reference_pressure: float = 1e5
    reference_temperature: float = 273.15

    def __post_init__(self):
        if self.critical_pressure <= 0 or self.critical_temperature <= 0:
            raise ValueError("Critical properties must be positive")

    def calculate_mmp(
        self,
        temperature: float,
        oil_composition: Optional[Dict] = None,
        method: str = "yellig_metcalfe",
    ) -> float:
        """
        Calculate Minimum Miscibility Pressure (MMP).

        Args:
            temperature: Reservoir temperature (K)
            oil_composition: Oil composition data (optional)
            method: Correlation method ('yellig_metcalfe', 'cronquist', 'alston')

        Returns:
            MMP in Pa
        """
        temp_f = (temperature - 273.15) * 9 / 5 + 32
        psi_to_pa = 6894.76

        if method == "yellig_metcalfe":
            # Yellig-Metcalfe evaluates T in Rankine. (T_rankine - 460) is simply temp_f.
            if temp_f <= 0:
                temp_f = 1e-6  # Prevent division by zero
            mmp_psi = 1833.7217 + 2.2518055 * temp_f + 0.0180067 * (temp_f**2) - 103949.93 / temp_f
        elif method == "cronquist":
            if oil_composition is None:
                c5_plus_fraction = 0.4
                api_gravity = 30.0
            else:
                c5_plus_fraction = oil_composition.get("c5_plus", 0.4)
                api_gravity = oil_composition.get("api_gravity", 30.0)
            mmp_psi = 18.2 * temp_f**0.5 - 120 * (api_gravity**0.5) + 203 * c5_plus_fraction + 1370
        elif method == "alston":
            # Correct Alston (1985) correlation from PhD paper (Стаття 3.pdf):
            # MMP = 1.935e4 · T_R^1.078 · (C2-C6%)^0.17 · (C7+%)^(-0.89)
            # where T_R is temperature in Rankine, and component fractions are mol%
            if oil_composition is None:
                # Defaults: Wasson oil from paper Table 1
                c2_c6_fraction = 0.20   # mol fraction C2-C6 (~20%)
                c7_plus_fraction = 0.57 # mol fraction C7+ (~57%)
            else:
                c2_c6_fraction = oil_composition.get("c2_c6", 0.20)
                c7_plus_fraction = oil_composition.get("c7_plus", 0.57)

            # Convert to mol% for the Alston formula
            c2_c6_pct = max(c2_c6_fraction * 100.0, 0.1)
            c7_plus_pct = max(c7_plus_fraction * 100.0, 0.1)
            T_rankine = temp_f + 459.67

            # Alston correlation (paper Eq.):
            mmp_psi = 1.935e4 * (T_rankine ** 1.078) * (c2_c6_pct ** 0.17) * (c7_plus_pct ** (-0.89))

            # Compositional lag correction for heavy oils (paper Section 3.2):
            # If C7+ > 0.4 (mol fraction), effective MMP is higher
            if c7_plus_fraction > 0.4:
                lag_factor = 1.0 + 0.15 * (c7_plus_fraction - 0.4)
                mmp_psi *= lag_factor
        else:
            raise ValueError(f"Unknown MMP correlation method: {method}")

        mmp_pa = mmp_psi * psi_to_pa
        return np.clip(mmp_pa, 5e6, 50e6)

    def density(self, pressure: float, temperature: float, method: str = "peng_robinson") -> float:
        """
        Calculate CO2 density.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            method: 'empirical' or 'peng_robinson'

        Returns:
            Density (kg/m³)
        """
        if method == "empirical":
            p_mpa = pressure / 1e6
            temp_c = temperature - 273.15
            rho = 1.01 + 1.09e-2 * p_mpa - 1.25e-5 * p_mpa**2 + 2.3e-3 * temp_c
            return rho * 1000
        elif method == "peng_robinson":
            return self._peng_robinson_density(pressure, temperature)
        else:
            raise ValueError(f"Unknown density method: {method}")

    def _peng_robinson_density(self, pressure: float, temperature: float) -> float:
        """Calculate density using Peng-Robinson EOS."""
        R = 8.314
        Tc = self.critical_temperature
        Pc = self.critical_pressure
        omega = self.acentric_factor

        Tr = temperature / Tc
        Tr = max(Tr, 0.01)

        sqrt_Tr = np.sqrt(Tr)
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + m * (1 - sqrt_Tr)) ** 2

        a = 0.45724 * R**2 * Tc**2 / Pc * alpha
        b = 0.07780 * R * Tc / Pc

        A = a * pressure / (R**2 * temperature**2)
        B = b * pressure / (R * temperature)

        coeffs = [1, -1 + B, A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)]
        roots = np.roots(coeffs)
        real_roots = np.real(roots[np.isreal(roots)])

        valid_roots = real_roots[(real_roots > 0.1) & (real_roots < 3.0)]

        if len(valid_roots) > 0:
            Z = valid_roots[0]
        else:
            Z = max(0.3, min(1.0 - 0.1 * B / max(A, 1e-6), 1.5))

        Z = max(Z, 1e-6)
        rho = pressure * self.molecular_weight / (Z * R * temperature)

        return rho

    def viscosity(self, pressure: float, temperature: float, method: str = "empirical") -> float:
        """
        Calculate CO2 viscosity.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            method: 'empirical' or 'fenghour'

        Returns:
            Viscosity (Pa·s)
        """
        if method == "empirical":
            p_mpa = pressure / 1e6
            Tr = temperature / self.critical_temperature
            mu_cp = 0.021 * np.exp(6.3e-4 * p_mpa - 0.02 * Tr)
            return mu_cp * 1e-3
        elif method == "fenghour":
            return self._fenghour_viscosity(pressure, temperature)
        else:
            raise ValueError(f"Unknown viscosity method: {method}")

    def _fenghour_viscosity(self, pressure: float, temperature: float) -> float:
        """Calculate viscosity using Fenghour et al. correlation."""
        mu_0 = 1.00697e-6 * np.sqrt(temperature)

        rho = self.density(pressure, temperature)
        Tr = temperature / self.critical_temperature
        rho_r = rho / 467.6

        mu_c = mu_0 * (1 + 0.235 * rho_r + 0.395 * rho_r**2 - 0.041 * rho_r**3)

        P_r = pressure / self.critical_pressure
        pressure_factor = 1 + 0.1 * (P_r - 1) * np.exp(-2 * rho_r)

        return mu_c * pressure_factor

    def solubility_brine(self, pressure: float, temperature: float, salinity: float = 0.0) -> float:
        """
        Calculate CO2 solubility in brine.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            salinity: Water salinity (kg salt per kg water)

        Returns:
            Solubility (mol/kg water)
        """
        temperature = max(temperature, 273.15)

        ln_kh = -58.0931 + 90.5069e3 / temperature + 22.2940 * np.log(temperature)
        kh = np.exp(ln_kh)

        p_mpa = pressure / 1e6

        setschenow_constant = 0.1
        salinity_factor = np.exp(-setschenow_constant * salinity)

        return kh * p_mpa * salinity_factor

    def solubility_oil(self, pressure: float, temperature: float, oil_api: float = 30.0) -> float:
        """
        Calculate CO2 solubility in oil.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            oil_api: Oil API gravity

        Returns:
            Solubility (mol/kg oil)
        """
        p_mpa = pressure / 1e6
        temp_c = temperature - 273.15

        solubility = 0.1 * p_mpa * np.exp(-0.01 * temp_c) * (1 + 0.01 * oil_api)

        return solubility

    def swelling_factor(self, pressure: float, temperature: float, co2_solubility: float) -> float:
        """
        Calculate oil swelling factor due to CO2 dissolution.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            co2_solubility: CO2 solubility in oil (mol/kg oil)

        Returns:
            Swelling factor (dimensionless)
        """
        return 1.0 + 0.02 * co2_solubility

    def viscosity_reduction(
        self, pressure: float, temperature: float, co2_solubility: float
    ) -> float:
        """
        Calculate oil viscosity reduction factor due to CO2.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            co2_solubility: CO2 solubility in oil (mol/kg oil)

        Returns:
            Viscosity reduction factor (dimensionless)
        """
        return np.exp(-0.1 * co2_solubility)

    def is_miscible(
        self,
        pressure: float,
        temperature: float,
        oil_composition: Optional[Dict] = None,
        method: str = "yellig_metcalfe",
    ) -> bool:
        """
        Check if CO2 is miscible with oil at given conditions.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            oil_composition: Oil composition data
            method: MMP correlation method

        Returns:
            True if miscible
        """
        mmp = self.calculate_mmp(temperature, oil_composition, method)
        return pressure >= mmp

    def properties(
        self, pressure: float, temperature: float, oil_composition: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive CO2 properties.

        Args:
            pressure: Pressure (Pa)
            temperature: Temperature (K)
            oil_composition: Oil composition data

        Returns:
            Dictionary of CO2 properties
        """
        density = self.density(pressure, temperature)
        viscosity = self.viscosity(pressure, temperature)
        is_miscible = self.is_miscible(pressure, temperature, oil_composition)
        mmp = self.calculate_mmp(temperature, oil_composition)
        solubility_water = self.solubility_brine(pressure, temperature)
        solubility_oil = self.solubility_oil(pressure, temperature)

        swelling = self.swelling_factor(pressure, temperature, solubility_oil)
        visc_reduction = self.viscosity_reduction(pressure, temperature, solubility_oil)

        return {
            "density": density,
            "viscosity": viscosity,
            "is_miscible": is_miscible,
            "mmp": mmp,
            "solubility_water": solubility_water,
            "solubility_oil": solubility_oil,
            "swelling_factor": swelling,
            "viscosity_reduction_factor": visc_reduction,
        }


class CO2DensityCorrelation:
    """CO2 density correlations wrapper."""

    def __init__(self, co2_props: CO2Properties):
        self.co2_props = co2_props

    def __call__(self, pressure: float, temperature: float) -> float:
        return self.co2_props.density(pressure, temperature)


class CO2ViscosityCorrelation:
    """CO2 viscosity correlations wrapper."""

    def __init__(self, co2_props: CO2Properties):
        self.co2_props = co2_props

    def __call__(self, pressure: float, temperature: float) -> float:
        return self.co2_props.viscosity(pressure, temperature)


class CO2SolubilityCorrelation:
    """CO2 solubility correlations wrapper."""

    def __init__(self, co2_props: CO2Properties):
        self.co2_props = co2_props

    def solubility_in_brine(
        self, pressure: float, temperature: float, salinity: float = 0.0
    ) -> float:
        return self.co2_props.solubility_brine(pressure, temperature, salinity)

    def solubility_in_oil(
        self, pressure: float, temperature: float, oil_api: float = 30.0
    ) -> float:
        return self.co2_props.solubility_oil(pressure, temperature, oil_api)


def create_co2_properties() -> CO2Properties:
    """Factory function to create CO2Properties instance."""
    return CO2Properties()
