"""
Unified Relative Permeability Models for CO2 EOR Optimizer.

This module provides standardized relative permeability correlations including
Corey-type models and three-phase relative permeability calculations.

Based on industry-standard formulations:
- Corey, A.T. (1954) The interrelation between gas and oil relative permeabilities
- Dullien, F.A.L. (1992) Porous Media: Fluid Transport and Pore Structure
- Stone, H.L. (1970) Probability model for estimating three-phase relative permeability

References:
- Aziz, K. and Settari, A., Petroleum Reservoir Simulation, 1979
- Ertekin, T, Abou-Kassem, J.H. and G.R. King, Basic Applied Reservoir Simulation, SPE Textbook Vol 10, 2001
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class CoreyParameters:
    """
    Parameters for Corey relative permeability model.

    Attributes:
        krw0: Water relative permeability at Sw = 1 - Sor (endpoint)
        kro0: Oil relative permeability at Sw = Swi (endpoint)
        krg0: Gas relative permeability at Sg = 1 - Swi - Sgr (endpoint)
        nw: Water Corey exponent (typically 2-4)
        no: Oil Corey exponent (typically 2-4)
        ng: Gas Corey exponent (typically 2-4)
        swi: Irreducible water saturation (connate water)
        sor: Residual oil saturation
        sgr: Residual gas saturation
    """

    krw0: float = 0.3
    kro0: float = 1.0
    krg0: float = 0.8
    nw: float = 2.0
    no: float = 2.0
    ng: float = 2.0
    swi: float = 0.2
    sor: float = 0.2
    sgr: float = 0.05

    def __post_init__(self):
        if not (0 <= self.krw0 <= 1):
            raise ValueError("krw0 must be between 0 and 1")
        if not (0 <= self.kro0 <= 1):
            raise ValueError("kro0 must be between 0 and 1")
        if not (0 <= self.krg0 <= 1):
            raise ValueError("krg0 must be between 0 and 1")
        if any(exp <= 0 for exp in [self.nw, self.no, self.ng]):
            raise ValueError("Corey exponents must be positive")
        if not (0 <= self.swi <= 1):
            raise ValueError("swi must be between 0 and 1")
        if not (0 <= self.sor <= 1):
            raise ValueError("sor must be between 0 and 1")
        if not (0 <= self.sgr <= 1):
            raise ValueError("sgr must be between 0 and 1")
        if self.swi + self.sor >= 1:
            raise ValueError("swi + sor must be less than 1")

    def normalize_saturation(self, phase: str, saturation: np.ndarray) -> np.ndarray:
        """
        Normalize saturation for Corey model.

        Args:
            phase: Phase ('water', 'oil', 'gas')
            saturation: Phase saturation (can be scalar or array)

        Returns:
            Normalized saturation in [0, 1]
        """
        sat = np.asarray(saturation)
        result = np.zeros_like(sat)

        if phase == "water":
            result = np.where(sat <= self.swi, 0.0, (sat - self.swi) / (1.0 - self.sor - self.swi))
        elif phase == "oil":
            sat_sum = 1.0 - self.swi - self.sor
            result = np.where(sat_sum <= 0, 0.0, (sat - self.sor) / sat_sum)
        elif phase == "gas":
            result = np.where(sat <= self.sgr, 0.0, (sat - self.sgr) / (1.0 - self.swi - self.sgr))
        else:
            raise ValueError(f"Unknown phase: {phase}")

        return np.clip(result, 0.0, 1.0)


@dataclass
class LETParameters:
    """
    Parameters for LET (Lomeland, Ebeltoft, Thomas) relative permeability model.

    The LET model provides more flexibility than Corey for fitting laboratory data.

    Attributes:
        Lw, Ew, Tw: LET parameters for water relative permeability
        Lo, Eo, To: LET parameters for oil relative permeability
        Lg, Eg, Tg: LET parameters for gas relative permeability
        krw_max: Maximum water relative permeability
        kro_max: Maximum oil relative permeability
        krg_max: Maximum gas relative permeability
        swi: Irreducible water saturation
        sor: Residual oil saturation
        sgr: Residual gas saturation
    """

    Lw: float = 2.0
    Ew: float = 2.0
    Tw: float = 2.0
    Lo: float = 2.0
    Eo: float = 2.0
    To: float = 2.0
    Lg: float = 2.0
    Eg: float = 2.0
    Tg: float = 2.0
    krw_max: float = 0.3
    kro_max: float = 1.0
    krg_max: float = 0.8
    swi: float = 0.2
    sor: float = 0.2
    sgr: float = 0.05


class HysteresisModel:
    """
    Base class for relative permeability hysteresis models.

    Handles drainage-imbibition cycling effects on relative permeability.
    """

    def __init__(self, params: CoreyParameters):
        self.params = params
        self._drainage_curve: Optional[np.ndarray] = None
        self._imbibition_curve: Optional[np.ndarray] = None
        self._max_drainage_saturation: float = 1.0 - params.sor

    def store_drainage_curve(self, sw_array: np.ndarray, krw_array: np.ndarray) -> None:
        """Store drainage curve for hysteresis calculations."""
        self._drainage_curve = np.column_stack([sw_array, krw_array])
        self._max_drainage_saturation = np.max(sw_array)

    def get_scanning_curve(self, sw: float, scanning_direction: str = "imbibition") -> float:
        """
        Get relative permeability from scanning curve.

        Args:
            sw: Current water saturation
            scanning_direction: 'imbibition' or 'drainage'

        Returns:
            Relative permeability value
        """
        if self._drainage_curve is None:
            return 0.0

        if scanning_direction == "imbibition":
            if sw >= self._max_drainage_saturation:
                sw_use = self._max_drainage_saturation
            else:
                sw_use = sw
        else:
            sw_use = max(sw, self.params.swi)

        if len(self._drainage_curve) > 0:
            kr_interp = np.interp(sw_use, self._drainage_curve[:, 0], self._drainage_curve[:, 1])
            return max(0.0, kr_interp)
        return 0.0


class LandHysteresisModel(HysteresisModel):
    """
    Land's hysteresis model for capillary pressure and relative permeability.

    Uses scanning curve theory with retention points.
    """

    def __init__(self, params: CoreyParameters, land_constant: float = 1.0):
        super().__init__(params)
        self.land_constant = land_constant
        self._trapping_coefficient: float = 0.0

    def calculate_trapping_number(self, sw_current: float, sw_max: float) -> float:
        """
        Calculate trapping number for non-wetting phase.

        Args:
            sw_current: Current water saturation
            sw_max: Maximum water saturation achieved (drainage endpoint)

        Returns:
            Trapping number
        """
        if sw_max <= sw_current:
            return 0.0

        self._trapping_coefficient = self.land_constant * (
            1.0 / (sw_max - sw_current) - 1.0 / (1.0 - sw_current)
        )
        return self._trapping_coefficient

    def get_imbibition_kr(self, sw: float, sw_max: float, phase: str) -> float:
        """
        Get relative permeability during imbibition using Land's model.

        Args:
            sw: Current water saturation
            sw_max: Maximum water saturation achieved during drainage
            phase: Phase ('water', 'oil', 'gas')

        Returns:
            Relative permeability value
        """
        trapping = self.calculate_trapping_number(sw, sw_max)

        if phase == "water":
            return (
                self.params.krw0
                * ((sw - self.params.swi) / (1.0 - self.params.swi - self.params.sor))
                ** self.params.nw
            )
        elif phase == "oil":
            swr = self._get_trapped_saturation(sw_max)
            so_normalized = (1.0 - sw - self.params.sor - swr) / (
                1.0 - self.params.swi - self.params.sor - swr
            )
            so_normalized = np.clip(so_normalized, 0.0, 1.0)
            return self.params.kro0 * so_normalized**self.params.no
        elif phase == "gas":
            sgr_trapped = self._get_trapped_gas(sw_max)
            sg_effective = max(0.0, sw - self.params.swi - sgr_trapped)
            sg_max = 1.0 - self.params.swi - self.params.sor - sgr_trapped
            sg_normalized = sg_effective / sg_max if sg_max > 0 else 0.0
            return self.params.krg0 * sg_normalized**self.params.ng
        return 0.0

    def _get_trapped_saturation(self, sw_max: float) -> float:
        """Calculate trapped non-wetting phase saturation."""
        if sw_max < self.params.swi + 0.01:
            return 1.0 - self.params.swi - self.params.sor
        return 1.0 / (1.0 / (1.0 - sw_max - self.params.sor) + self.land_constant) - self.params.sor

    def _get_trapped_gas(self, sw_max: float) -> float:
        """Calculate trapped gas saturation."""
        return self._get_trapped_saturation(sw_max)


class CoreyRelativePermeability:
    """
    Corey-type relative permeability model for two-phase and three-phase systems.

    Implements:
    - Two-phase water-oil relative permeability
    - Two-phase gas-oil relative permeability
    - Three-phase using Stone's Model II
    """

    def __init__(self, params: CoreyParameters):
        """
        Initialize Corey relative permeability model.

        Args:
            params: CoreyParameters instance
        """
        self.params = params

    def kr_water(self, sw: np.ndarray) -> np.ndarray:
        """
        Calculate water relative permeability.

        Args:
            sw: Water saturation array

        Returns:
            Water relative permeability array
        """
        sw = np.asarray(sw)
        krw = np.where(
            sw <= self.params.swi,
            0.0,
            np.where(
                sw >= 1.0 - self.params.sor,
                self.params.krw0,
                self.params.krw0 * self.params.normalize_saturation("water", sw) ** self.params.nw,
            ),
        )
        return krw

    def kr_oil_water(self, sw: np.ndarray) -> np.ndarray:
        """
        Calculate oil relative permeability in water-oil system.

        Args:
            sw: Water saturation array

        Returns:
            Oil relative permeability array
        """
        sw = np.asarray(sw)
        so = 1.0 - sw
        kro = np.where(
            sw <= self.params.swi,
            self.params.kro0,
            np.where(
                sw >= 1.0 - self.params.sor,
                0.0,
                self.params.kro0 * self.params.normalize_saturation("oil", so) ** self.params.no,
            ),
        )
        return kro

    def kr_gas(self, sg: np.ndarray) -> np.ndarray:
        """
        Calculate gas relative permeability.

        Args:
            sg: Gas saturation array

        Returns:
            Gas relative permeability array
        """
        sg = np.asarray(sg)
        krg = np.where(
            sg <= self.params.sgr,
            0.0,
            np.where(
                sg >= 1.0 - self.params.swi,
                self.params.krg0,
                self.params.krg0 * self.params.normalize_saturation("gas", sg) ** self.params.ng,
            ),
        )
        return krg

    def kr_oil_gas(self, sg: np.ndarray) -> np.ndarray:
        """
        Calculate oil relative permeability in gas-oil system.

        Args:
            sg: Gas saturation array

        Returns:
            Oil relative permeability array
        """
        sg = np.asarray(sg)
        so = 1.0 - sg - self.params.swi
        kro = np.where(
            sg <= self.params.sgr,
            self.params.kro0,
            np.where(
                sg >= 1.0 - self.params.sor - self.params.swi,
                0.0,
                self.params.kro0 * self.params.normalize_saturation("oil", so) ** self.params.no,
            ),
        )
        return kro

    def kr_three_phase(
        self, sw: np.ndarray, sg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate three-phase relative permeabilities using Stone's Model II.

        Args:
            sw: Water saturation array
            sg: Gas saturation array

        Returns:
            Tuple of (krw, kro, krg) arrays
        """
        sw = np.asarray(sw)
        sg = np.asarray(sg)

        krw = self.kr_water(sw)
        krg = self.kr_gas(sg)
        so = 1.0 - sw - sg

        kro = np.where(so <= 0, 0.0, self._stone_model_ii(sw, sg, so))

        return krw, kro, krg

    def _stone_model_ii(self, sw: np.ndarray, sg: np.ndarray, so: np.ndarray) -> np.ndarray:
        """
        Stone's Model II for three-phase oil relative permeability.

        Reference: Stone, H.L. (1970) "Probability model for estimating three-phase relative permeability"
        """
        sw_star = (sw - self.params.swi) / (
            1.0 - self.params.swi - self.params.sor - self.params.sgr
        )
        sg_star = (sg - self.params.sgr) / (
            1.0 - self.params.swi - self.params.sor - self.params.sgr
        )
        so_star = 1.0 - sw_star - sg_star

        krow_sw = self.kr_oil_water(sw)
        krog_sg = self.kr_oil_gas(sg)

        kro = np.where(
            (sw_star <= 0) | (sg_star <= 0),
            np.where(
                sg <= 0,
                krow_sw,
                np.where(sw <= self.params.swi, krog_sg, krow_sw * krog_sg / self.params.kro0),
            ),
            krow_sw * krog_sg / self.params.kro0,
        )

        return kro

    def kr_array(
        self, sw_array: np.ndarray, sg_array: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate relative permeabilities for arrays of saturations.

        Args:
            sw_array: Array of water saturations
            sg_array: Array of gas saturations (for three-phase)

        Returns:
            Dictionary with 'krw', 'kro', 'krg' arrays
        """
        results = {}

        if sg_array is None:
            krw = self.kr_water(sw_array)
            kro = self.kr_oil_water(sw_array)
            krg = np.zeros_like(sw_array)
        else:
            krw, kro, krg = self.kr_three_phase(sw_array, sg_array)

        results["krw"] = krw
        results["kro"] = kro
        results["krg"] = krg

        return results

    def fractional_flow(self, sw_array: np.ndarray, mu_water: float, mu_oil: float) -> np.ndarray:
        """
        Calculate fractional flow curve for water-oil system.

        fw = krw / (krw + kro * mu_water / mu_oil)

        Args:
            sw_array: Array of water saturations
            mu_water: Water viscosity
            mu_oil: Oil viscosity

        Returns:
            Fractional flow of water array
        """
        sw_array = np.asarray(sw_array)
        krw = self.kr_water(sw_array)
        kro = self.kr_oil_water(sw_array)

        denominator = krw + kro * mu_water / mu_oil
        fw = np.where(denominator > 0, krw / denominator, 0.0)

        return fw

    def mobility_ratio(self, sw: float, mu_water: float, mu_oil: float) -> float:
        """
        Calculate mobility ratio at given saturation.

        M = (krw / mu_water) / (kro / mu_oil)

        Args:
            sw: Water saturation
            mu_water: Water viscosity
            mu_oil: Oil viscosity

        Returns:
            Mobility ratio
        """
        krw = self.kr_water(sw)
        kro = self.kr_oil_water(sw)

        if kro > 0:
            return (krw / mu_water) / (kro / mu_oil)
        return np.inf

    def update_parameters(self, **kwargs) -> None:
        """Update Corey parameters."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")
        self.params.__post_init__()


class LETRelativePermeability:
    """
    LET (Lomeland, Ebeltoft, Thomas) relative permeability model.

    Provides more flexibility than Corey for fitting laboratory data.
    Reference: Lomeland, F., Ebeltoft, E., & Thomas, W.H. (2005)
    """

    def __init__(self, params: LETParameters):
        self.params = params

    def kr_water(self, sw: np.ndarray) -> np.ndarray:
        """Calculate water relative permeability using LET model."""
        sw = np.asarray(sw)
        sw_eff = (sw - self.params.swi) / (1.0 - self.params.swi - self.params.sor)
        sw_eff = np.clip(sw_eff, 0.0, 1.0)

        numerator = sw_eff**self.params.Lw
        denominator = sw_eff**self.params.Lw + self.params.Tw * (1 - sw_eff) ** self.params.Ew

        return self.params.krw_max * np.where(denominator > 0, numerator / denominator, 0.0)

    def kr_oil(self, sw: np.ndarray) -> np.ndarray:
        """Calculate oil relative permeability using LET model."""
        sw = np.asarray(sw)
        sw_eff = (sw - self.params.swi) / (1.0 - self.params.swi - self.params.sor)
        sw_eff = np.clip(sw_eff, 0.0, 1.0)

        numerator = sw_eff**self.params.Lo
        denominator = sw_eff**self.params.Lo + self.params.To * (1 - sw_eff) ** self.params.Eo

        return self.params.kro_max * np.where(denominator > 0, numerator / denominator, 0.0)

    def kr_gas(self, sg: np.ndarray) -> np.ndarray:
        """Calculate gas relative permeability using LET model."""
        sg = np.asarray(sg)
        sg_eff = (sg - self.params.sgr) / (1.0 - self.params.swi - self.params.sgr)
        sg_eff = np.clip(sg_eff, 0.0, 1.0)

        numerator = sg_eff**self.params.Lg
        denominator = sg_eff**self.params.Lg + self.params.Tg * (1 - sg_eff) ** self.params.Eg

        return self.params.krg_max * np.where(denominator > 0, numerator / denominator, 0.0)


def create_relative_permeability_model(model_type: str = "corey", **kwargs) -> Any:
    """
    Factory function to create relative permeability model.

    Args:
        model_type: 'corey' or 'let'
        **kwargs: Model parameters

    Returns:
        Relative permeability model instance
    """
    if model_type == "corey":
        params = CoreyParameters(**kwargs)
        return CoreyRelativePermeability(params)
    elif model_type == "let":
        params = LETParameters(**kwargs)
        return LETRelativePermeability(params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
