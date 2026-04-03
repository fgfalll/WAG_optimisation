"""
Unified Equation of State Models for CO2 EOR Optimizer.

This module provides standardized cubic equations of state for compositional
simulation including Peng-Robinson and Soave-Redlich-Kwong.

Based on industry-standard formulations:
- Peng, D.Y. & Robinson, D.B. (1976) A New Two-Constant Equation of State
- Soave, G. (1972) Equilibrium constants from a modified Redlich-Kwong EOS

References:
- Peng, D.Y. and Robinson, D.B. (1976) Ind. Eng. Chem. Fundam. 15, 59-64
- Soave, G. (1972) Chem. Eng. Sci. 27, 1197-1203
- Duan, Z. et al. (1992) An equation of state for the CH4-CO2-H2O system
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Import EOSModelParameters from core.data_models for ReservoirFluid compatibility
from core.data_models import EOSModelParameters, PhysicalConstants

_PHYS_CONSTANTS = PhysicalConstants()


@dataclass
class EOSParameters:
    """
    Parameters for cubic equation of state.

    Attributes:
        eos_type: 'PR' for Peng-Robinson, 'SRK' for Soave-Redlich-Kwong
        component_names: List of component names
        mole_fractions: Mole fractions of each component
        critical_temperatures: Critical temperatures (K)
        critical_pressures: Critical pressures (Pa)
        acentric_factors: Acentric factors
        binary_interaction: Binary interaction coefficients matrix
    """

    eos_type: str = "PR"
    component_names: List[str] = None
    mole_fractions: np.ndarray = None
    critical_temperatures: np.ndarray = None
    critical_pressures: np.ndarray = None
    acentric_factors: np.ndarray = None
    binary_interaction: np.ndarray = None

    def __post_init__(self):
        if self.component_names is None:
            self.component_names = ["CO2", "C1", "C2", "C3", "C4", "C5", "C6", "C7+"]
        if self.mole_fractions is None:
            self.mole_fractions = np.array([0.8, 0.1, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
        if self.critical_temperatures is None:
            self.critical_temperatures = np.array(
                [304.13, 190.56, 305.32, 369.83, 425.12, 469.7, 507.4, 540.2]
            )
        if self.critical_pressures is None:
            self.critical_pressures = np.array(
                [7.376e6, 4.599e6, 4.872e6, 4.248e6, 3.792e6, 3.369e6, 3.025e6, 2.756e6]
            )
        if self.acentric_factors is None:
            self.acentric_factors = np.array(
                [0.225, 0.011, 0.099, 0.152, 0.201, 0.251, 0.296, 0.350]
            )
        if self.binary_interaction is None:
            n = len(self.component_names)
            self.binary_interaction = np.zeros((n, n))


class CubicEOS:
    """
    Base class for cubic equations of state.
    """

    R = 8.314

    def __init__(self, params: EOSParameters):
        self.params = params
        self._validate_params()
        self._setup_parameters()

    def _validate_params(self):
        """Validate EOS parameters."""
        n = len(self.params.component_names)
        if len(self.params.mole_fractions) != n:
            raise ValueError("Mole fractions size must match component names")
        if self.params.binary_interaction.shape != (n, n):
            raise ValueError("Binary interaction matrix must be n x n")

    def _setup_parameters(self):
        """Set up EOS-specific parameters."""
        self.zc = 0.0
        self.ac = 0.0
        self.bc = 0.0

    def alpha(self, T: float, omega: float) -> float:
        """Calculate temperature-dependent alpha parameter."""
        raise NotImplementedError

    def calculate_a_b(self, T: float) -> Tuple[float, float]:
        """Calculate EOS parameters a and b."""
        raise NotImplementedError

    def solve_cubic(self, A: float, B: float) -> np.ndarray:
        """Solve cubic equation for Z-factor."""
        coeffs = [1.0, -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)]
        roots = np.roots(coeffs)
        real_roots = np.real(roots[np.isreal(roots)])
        return real_roots[(real_roots > 0.1) & (real_roots < 3.0)]

    def z_factor(self, pressure: float, temperature: float) -> float:
        """Calculate compressibility factor Z."""
        A, B = self.calculate_a_b(pressure, temperature)
        valid_roots = self.solve_cubic(A, B)
        if len(valid_roots) > 0:
            return valid_roots[0]
        return max(0.3, min(1.0 - 0.1 * B / max(A, 1e-6), 1.5))

    def calculate_a_b(self, pressure: float, temperature: float) -> Tuple[float, float]:
        """Calculate A and B parameters for Z-factor equation."""
        T = temperature
        P = pressure

        n = len(self.params.component_names)
        z = self.params.mole_fractions

        a_i = np.zeros(n)
        b_i = np.zeros(n)

        for i in range(n):
            Tc = self.params.critical_temperatures[i]
            Pc = self.params.critical_pressures[i]
            omega = self.params.acentric_factors[i]

            alpha_i = self.alpha(T, omega)
            a_i[i] = 0.45724 * self.R**2 * Tc**2 / Pc * alpha_i
            b_i[i] = 0.07780 * self.R * Tc / Pc

        aij = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                kij = self.params.binary_interaction[i, j]
                aij[i, j] = np.sqrt(a_i[i] * a_i[j]) * (1.0 - kij)

        a = np.sum(np.outer(z, z) * aij)
        b = np.sum(z * b_i)

        A = a * P / (self.R**2 * T**2)
        B = b * P / (self.R * T)

        return A, B

    def density(self, pressure: float, temperature: float) -> float:
        """Calculate mixture density."""
        Z = self.z_factor(pressure, temperature)
        M_avg = np.sum(
            self.params.mole_fractions
            * np.array([44.01, 16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.0])
            / 1000.0
        )
        return pressure * M_avg / (Z * self.R * temperature)

    def viscosity(self, pressure: float, temperature: float) -> float:
        """Estimate mixture viscosity (simplified)."""
        rho = self.density(pressure, temperature)
        Tr = temperature / 304.13
        rho_r = rho / 467.6

        mu_0 = 1.00697e-6 * np.sqrt(temperature)
        mu_c = mu_0 * (1 + 0.235 * rho_r + 0.395 * rho_r**2 - 0.041 * rho_r**3)

        return mu_c

    def fugacity_coefficients(
        self, pressure: float, temperature: float, composition: np.ndarray
    ) -> np.ndarray:
        """
        Calculate fugacity coefficients for all components using Peng-Robinson EOS.

        ln(phi_i) = (b_i/b) * (Z - 1) - ln(Z - B) + (A/B) * (b_i/b - 1) * ln(1 + B/Z)

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
            composition: Mole fractions of each component

        Returns:
            Array of fugacity coefficients phi_i for each component
        """
        n = len(self.params.component_names)
        z = composition

        # Calculate a and b parameters for mixture
        R = self.R

        # Component-specific a_i and b_i
        a_i = np.zeros(n)
        b_i = np.zeros(n)

        for i in range(n):
            Tc = self.params.critical_temperatures[i]
            Pc = self.params.critical_pressures[i]
            omega = self.params.acentric_factors[i]

            alpha_i = self.alpha(temperature, omega)
            a_i[i] = 0.45724 * R**2 * Tc**2 / Pc * alpha_i
            b_i[i] = 0.07780 * R * Tc / Pc

        # Mixture parameters a and b
        aij = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                kij = self.params.binary_interaction[i, j]
                aij[i, j] = np.sqrt(a_i[i] * a_i[j]) * (1.0 - kij)

        a = np.sum(np.outer(z, z) * aij)
        b = np.sum(z * b_i)

        # A and B parameters
        A = a * pressure / (R**2 * temperature**2)
        B = b * pressure / (R * temperature)

        # Calculate Z-factor for the phase
        Z = self.z_factor(pressure, temperature)

        # Calculate fugacity coefficients for each component
        phi = np.zeros(n)
        for i in range(n):
            b_ratio = b_i[i] / b if b > 0 else 0.0

            # ln(phi_i) = (b_i/b) * (Z - 1) - ln(Z - B) + (A/B) * (b_i/b - 1) * ln(1 + B/Z)
            term1 = b_ratio * (Z - 1.0)

            if Z - B > 0:
                term2 = np.log(Z - B)
            else:
                term2 = np.log(1e-10)

            if 1.0 + B / Z > 0:
                term3 = (A / B) * (b_ratio - 1.0) * np.log(1.0 + B / Z) if B > 0 else 0.0
            else:
                term3 = 0.0

            ln_phi = term1 - term2 + term3
            phi[i] = np.exp(ln_phi)

        return phi


class PengRobinsonEOS(CubicEOS):
    """
    Peng-Robinson Equation of State.

    Parameters:
        omega_a: 0.45724
        omega_b: 0.07780
    """

    def __init__(self, params: EOSParameters):
        params.eos_type = "PR"
        super().__init__(params)

    def alpha(self, T: float, omega: float) -> float:
        """Calculate alpha for Peng-Robinson."""
        Tr = T / 304.13
        if Tr <= 0:
            Tr = 0.5
        sqrt_Tr = np.sqrt(Tr)
        m = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        return (1.0 + m * (1.0 - sqrt_Tr)) ** 2


class SoaveRedlichKwongEOS(CubicEOS):
    """
    Soave-Redlich-Kwong Equation of State.

    Parameters:
        omega_a: 0.42748
        omega_b: 0.08664
    """

    def __init__(self, params: EOSParameters):
        params.eos_type = "SRK"
        super().__init__(params)

    def alpha(self, T: float, omega: float) -> float:
        """Calculate alpha for SRK."""
        Tr = T / 304.13
        if Tr <= 0:
            Tr = 0.5
        sqrt_Tr = np.sqrt(Tr)
        m = 0.480 + 1.574 * omega - 0.176 * omega**2
        return (1.0 + m * (1.0 - sqrt_Tr)) ** 2


class PhaseEquilibriumCalculator:
    """
    Phase equilibrium calculations using K-values (equilibrium ratios).

    Provides simplified flash calculations for binary and multi-component systems.
    """

    def __init__(self, eos: CubicEOS):
        self.eos = eos

    def wilson_k_values(self, temperature: float, pressure: float) -> np.ndarray:
        """
        Calculate Wilson K-values for phase equilibrium.

        K_i = (Pc_i / P) * exp(5.373 * (1 + omega_i) * (1 - Tc_i / T))

        Args:
            temperature: Temperature (K)
            pressure: Pressure (Pa)

        Returns:
            Array of K-values
        """
        n = len(self.eos.params.component_names)
        K = np.zeros(n)

        for i in range(n):
            Tc = self.eos.params.critical_temperatures[i]
            Pc = self.eos.params.critical_pressures[i]
            omega = self.eos.params.acentric_factors[i]

            K[i] = (Pc / pressure) * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc / temperature))

        return K

    def k_values_from_fugacity(
        self,
        pressure: float,
        temperature: float,
        liquid_composition: np.ndarray,
        vapor_composition: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate K-values from EOS fugacity coefficients.

        K_i = phi_L_i / phi_V_i

        This is the physically correct method for calculating K-values
        from the equation of state.

        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
            liquid_composition: Liquid phase mole fractions
            vapor_composition: Vapor phase mole fractions

        Returns:
            Array of K-values
        """
        # Get fugacity coefficients for both phases
        phi_l = self.eos.fugacity_coefficients(pressure, temperature, liquid_composition)
        phi_v = self.eos.fugacity_coefficients(pressure, temperature, vapor_composition)

        # K = phi_L / phi_V
        # Avoid division by zero
        K = np.zeros_like(phi_l)
        for i in range(len(K)):
            if phi_v[i] > 1e-10:
                K[i] = phi_l[i] / phi_v[i]
            else:
                # Fallback to Wilson if fugacity calculation fails
                K[i] = self.wilson_k_values(temperature, pressure)[i]

        return K

    def rachford_rice_flash(
        self, K: np.ndarray, z: np.ndarray, tol: float = 1e-8, max_iter: int = 100
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve Rachford-Rice equation for vapor fraction.

        Args:
            K: K-values
            z: Overall composition
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Tuple of (vapor_fraction, liquid_composition, vapor_composition)
        """
        n = len(z)

        f = lambda V: np.sum(z * (K - 1) / (1 + V * (K - 1))) - 1.0

        V_low = 0.0
        V_high = 1.0

        for _ in range(max_iter):
            V_mid = (V_low + V_high) / 2.0
            f_mid = f(V_mid)

            if abs(f_mid) < tol:
                break

            if f_mid > 0:
                V_low = V_mid
            else:
                V_high = V_mid

        V = (V_low + V_high) / 2.0

        x = z / (1.0 + V * (K - 1.0))
        y = K * x

        return V, x, y


def create_eos(
    eos_type: str = "PR", component_names: List[str] = None, mole_fractions: np.ndarray = None
) -> CubicEOS:
    """
    Factory function to create EOS instance.

    Args:
        eos_type: 'PR' or 'SRK'
        component_names: List of component names
        mole_fractions: Mole fractions

    Returns:
        CubicEOS instance
    """
    params = EOSParameters(
        eos_type=eos_type, component_names=component_names, mole_fractions=mole_fractions
    )

    if eos_type == "PR":
        return PengRobinsonEOS(params)
    elif eos_type == "SRK":
        return SoaveRedlichKwongEOS(params)
    else:
        raise ValueError(f"Unknown EOS type: {eos_type}")


class ReservoirFluid:
    """
    Represents a fluid with a defined composition and EOS model.
    Wrapper for CubicEOS that provides reservoir-specific methods.
    """

    def __init__(self, eos_params: EOSModelParameters):
        # Convert EOSModelParameters to EOSParameters for CubicEOS
        params = EOSParameters(
            eos_type=eos_params.eos_type,
            component_names=eos_params.component_names,
            mole_fractions=eos_params.component_properties[:, 0],  # First column is mole fractions
            critical_temperatures=eos_params.component_properties[:, 2] if eos_params.component_properties.shape[1] > 2 else None,
            critical_pressures=eos_params.component_properties[:, 3] if eos_params.component_properties.shape[1] > 3 else None,
            acentric_factors=eos_params.component_properties[:, 4] if eos_params.component_properties.shape[1] > 4 else None,
        )

        # Create the appropriate EOS subclass based on eos_type
        if eos_params.eos_type == "PR":
            self.eos_model = PengRobinsonEOS(params)
        elif eos_params.eos_type == "SRK":
            self.eos_model = SoaveRedlichKwongEOS(params)
        else:
            # Default to Peng-Robinson for unknown types
            logger.warning(f"Unknown EOS type '{eos_params.eos_type}', defaulting to Peng-Robinson")
            self.eos_model = PengRobinsonEOS(params)

    def get_bgas_rb_per_mscf(self, temperature_K: float, pressure_Pa: float) -> float:
        """
        Computes the gas formation volume factor (B_gas) in rb/MSCF.

        Uses ideal gas law with compressibility factor from CubicEOS.
        """
        Z = self.eos_model.z_factor(pressure_Pa, temperature_K)
        R = 8.314  # J/(mol·K)

        # Molar volume at reservoir conditions
        v_res = Z * R * temperature_K / pressure_Pa  # m³/mol

        # Standard conditions (60°F, 14.7 psi) - using PhysicalConstants
        v_sc = _PHYS_CONSTANTS.IDEAL_GAS_MOLAR_VOLUME_M3_PER_MOL

        # Convert to reservoir barrels per thousand standard cubic feet
        rb_per_mscf = (v_res / v_sc) * (1.0 / _PHYS_CONSTANTS.MSCF_TO_M3)  # MSCF to m³ conversion

        return rb_per_mscf

    def _initialize_critical_properties(self):
        """Initialize and cache critical properties for phase detection."""
        self._critical_properties_cache = {
            "CO2": {"Tc": 304.13, "Pc": 7.376e6, "omega": 0.225},
            "Methane": {"Tc": 190.6, "Pc": 4.604e6, "omega": 0.011},
            "Ethane": {"Tc": 305.4, "Pc": 4.880e6, "omega": 0.099},
            "Propane": {"Tc": 369.8, "Pc": 4.246e6, "omega": 0.152},
            "C4+": {"Tc": 425.2, "Pc": 3.796e6, "omega": 0.200},
            "C7+": {"Tc": 540.2, "Pc": 2.736e6, "omega": 0.350},
        }

        # Try to extract from EOS model if available
        eos_model = getattr(self, 'eos_model', None)
        if eos_model and hasattr(eos_model, 'params'):
            params = eos_model.params
            if hasattr(params, 'component_names'):
                for i, name in enumerate(params.component_names):
                    if hasattr(params, 'component_properties') and i < len(params.component_properties):
                        props = params.component_properties[i]
                        if len(props) >= 4:
                            self._critical_properties_cache[name] = {
                                "Tc": props[2],  # Critical temperature
                                "Pc": props[3],  # Critical pressure
                                "omega": props[4] if len(props) > 4 else 0.0,  # Accentric factor
                            }

    def detect_phase_regime(self, temperature_K: float, pressure_Pa: float) -> Dict[str, Any]:
        """
        Intelligently detect the phase regime based on temperature and pressure.
        Returns phase information and recommended calculation methods.
        """
        phase_info = {
            "basic_phase": "unknown",
            "temperature_K": temperature_K,
            "pressure_Pa": pressure_Pa,
            "phase_regime": "unknown",
            "calculation_method": "default",
            "is_supercritical": False,
            "is_near_critical": False,
            "co2_density_estimate": None,
        }

        # Check if CO2 is in supercritical region (highest priority)
        if "CO2" in self._critical_properties_cache:
            co2_tc = self._critical_properties_cache["CO2"]["Tc"]
            co2_pc = self._critical_properties_cache["CO2"]["Pc"]

            # Supercritical detection based on thermodynamic criteria
            if temperature_K > co2_tc and pressure_Pa > co2_pc:
                phase_info["is_supercritical"] = True
                phase_info["phase_regime"] = "supercritical_co2"
                phase_info["calculation_method"] = "supercritical_correlations"
                phase_info["basic_phase"] = "supercritical"
            elif temperature_K > co2_tc * 0.95 and pressure_Pa > co2_pc * 0.95:
                phase_info["is_near_critical"] = True
                phase_info["phase_regime"] = "near_critical_co2"
                phase_info["calculation_method"] = "near_critical_correlations"
            elif temperature_K > co2_tc:
                # Above critical temperature
                phase_info["phase_regime"] = "gas_phase"
                phase_info["calculation_method"] = "gas_correlations"
                phase_info["basic_phase"] = "V"
            else:
                phase_info["phase_regime"] = "liquid_phase"
                phase_info["calculation_method"] = "liquid_correlations"
                phase_info["basic_phase"] = "L"

        return phase_info

    def get_properties_si(self, temperature_K: float, pressure_Pa: float) -> Dict[str, Any]:
        """
        Computes fluid properties at a given T and P in SI units.
        Returns a dictionary compatible with the old interface.
        """
        Z = self.eos_model.z_factor(pressure_Pa, temperature_K)
        R = 8.314  # J/(mol·K)

        # Calculate density
        eos_model = getattr(self, 'eos_model', None)
        if eos_model and hasattr(eos_model, 'params'):
            params = eos_model.params
            # Calculate average molar mass
            mole_fractions = getattr(params, 'mole_fractions', None)
            if mole_fractions is None and hasattr(params, 'component_properties'):
                mole_fractions = params.component_properties[:, 0] if params.component_properties.shape[1] > 0 else np.array([1.0])

            # Estimate molar mass (weighted average)
            if hasattr(params, 'component_properties') and params.component_properties.shape[1] > 1:
                molar_masses = params.component_properties[:, 1]  # Second column is MW
                avg_molar_mass = np.sum(mole_fractions * molar_masses) * 0.001  # g/mol to kg/mol
            else:
                avg_molar_mass = 0.04401  # Default to CO2
        else:
            avg_molar_mass = 0.04401

        density = pressure_Pa * avg_molar_mass / (Z * R * temperature_K)

        return {
            "phase": "V" if Z < 0.8 else "L",
            "vapor_properties": {
                "Z": Z,
                "density_kg_per_m3": density,
            },
            "liquid_properties": {
                "Z": Z,
                "density_kg_per_m3": density,
            },
        }

    def _calculate_boil_supercritical_co2(self, temperature_K: float, pressure_Pa: float, density: float = None) -> float:
        """Calculate B_oil for supercritical CO2 using appropriate correlations."""
        R = 8.314  # J/(mol·K)

        # Calculate compressibility factor for supercritical CO2
        co2_tc = 304.13
        co2_pc = 7.376e6
        T_r = temperature_K / co2_tc
        P_r = pressure_Pa / co2_pc

        # Peng-Robinson Z-factor correlation for supercritical region
        Z = 1 - 0.35 * P_r / T_r + 0.15 * (P_r / T_r) ** 2
        Z = max(0.6, min(Z, 0.95))

        # Molar volume at reservoir conditions
        v_res = Z * R * temperature_K / pressure_Pa  # m³/mol

        # Standard conditions
        v_sc = _PHYS_CONSTANTS.IDEAL_GAS_MOLAR_VOLUME_M3_PER_MOL

        # Convert to reservoir barrels per STB equivalent
        rb_per_sm3 = v_res / v_sc * _PHYS_CONSTANTS.M3_TO_STB

        return max(0.2, min(rb_per_sm3, 2.0))

    def _calculate_boil_liquid_phase(self, temperature_K: float, pressure_Pa: float, composition: Dict = None) -> float:
        """Calculate B_oil for liquid phase with CO2 dissolution effects."""
        # Base oil formation volume factor (Standing's correlation modification)
        T_R = temperature_K * 1.8  # Convert to Rankine
        P_psi = pressure_Pa / _PHYS_CONSTANTS.PSI_TO_PA

        # Standing's correlation base
        co2_effect = 1.0
        if composition and "liquid_composition" in composition and len(composition["liquid_composition"]) > 0:
            co2_fraction = composition["liquid_composition"][0]
            co2_effect = 1 + 0.5 * co2_fraction  # CO2 dissolution increases B_oil

        B_oil_base = 1.0 + 0.0001 * (T_R - 460) + 0.000001 * P_psi
        B_oil = B_oil_base * co2_effect

        return max(1.0, min(B_oil, 2.5))

    def _calculate_boil_dense_gas(self, temperature_K: float, pressure_Pa: float, density: float = None) -> float:
        """Calculate B_oil equivalent for dense gas phase."""
        R = 8.314  # J/(mol·K)
        co2_pc = 7.376e6

        # Calculate Z-factor using density-based correlation
        if density and density > 200:
            Z = 0.7 + 0.0001 * (density - 200)
        else:
            P_r = pressure_Pa / co2_pc
            Z = 0.9 - 0.001 * (density if density else 0)

        Z = max(0.6, min(Z, 0.95))

        # Molar volume at reservoir conditions
        v_res = Z * R * temperature_K / pressure_Pa  # m³/mol

        # Standard conditions
        v_sc = _PHYS_CONSTANTS.IDEAL_GAS_MOLAR_VOLUME_M3_PER_MOL

        # Convert to rb/STB equivalent
        scf_per_rb = 5.615 / v_sc
        rb_per_scf = 1 / scf_per_rb

        # For 1000 scf (equivalent to 1 STB for gas)
        B_gas_rb_per_stb_eq = 1000 * rb_per_scf

        return max(0.5, min(B_gas_rb_per_stb_eq, 3.0))

    def get_boil_rb_per_stb(self, temperature_K: float, pressure_Pa: float) -> float:
        """
        Computes the oil formation volume factor (B_oil) in rb/STB using intelligent phase-dependent methods.
        Automatically switches calculation method based on phase regime.
        """
        # Initialize critical properties if not already done
        if not hasattr(self, '_critical_properties_cache'):
            self._initialize_critical_properties()

        # Detect phase regime
        phase_info = self.detect_phase_regime(temperature_K, pressure_Pa)
        calculation_method = phase_info["calculation_method"]

        try:
            # Get base properties from EOS
            props = self.get_properties_si(temperature_K, pressure_Pa)

            # Choose calculation method based on phase regime
            if calculation_method == "supercritical_correlations":
                return self._calculate_boil_supercritical_co2(
                    temperature_K, pressure_Pa, phase_info.get("co2_density_estimate")
                )
            elif calculation_method == "dense_gas_correlations":
                return self._calculate_boil_dense_gas(
                    temperature_K, pressure_Pa, phase_info.get("co2_density_estimate")
                )
            elif calculation_method == "liquid_correlations":
                return self._calculate_boil_liquid_phase(temperature_K, pressure_Pa, props)

        except Exception as e:
            logger.debug(f"Phase-dependent B_oil calculation failed: {e}")

        # Ultimate fallback based on phase regime
        if phase_info["is_supercritical"]:
            return 0.65  # Typical supercritical CO2 B_oil equivalent
        elif phase_info["basic_phase"] == "V":
            if pressure_Pa > 1e7:  # High pressure
                return 0.8  # Dense gas
            else:
                return 1.2  # Normal gas
        else:
            return 1.4  # Typical oil B_oil
