"""
Core Physics Engine for CO2 EOR Simulation
Handles fractional flow, relative permeability, and Buckley-Leverett theory.

DEPRECATED: This module is deprecated and will be removed in v1.0.0.
Use core.compositional_engine for full compositional simulation instead.
"""

import warnings

# Issue deprecation warning when module is imported
warnings.warn(
    "core.Phys_engine_full is deprecated and will be removed in v1.0.0. "
    "Use core.compositional_engine for full compositional simulation instead. "
    "The new compositional engine provides: "
    "- User-defined component system via EOSModelParameters "
    "- PT-flash calculations using existing EOS models "
    "- MMP-based miscibility detection using evaluation/mmp.py "
    "- Integration with optimization engine via SimulationEngineInterface "
    "- CMG GEM validation support",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from typing import Tuple
import logging
import warnings

from core.Phys_engine_full.pressure_dynamics import PressureDynamics

logger = logging.getLogger(__name__)

# Numerical constants - use PhysicalConstants for consistency
try:
    from core.data_models import PhysicalConstants
    _PHYS_CONSTANTS = PhysicalConstants()
    EPSILON = _PHYS_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
except ImportError:
    EPSILON = 1e-9  # Fallback for standalone usage


class PhysicsEngine:
    """
    Core physics engine for CO2 EOR simulation.
    Handles fractional flow, relative permeability, and Buckley-Leverett theory.

    DEPRECATED: This class is no longer actively maintained.
    Please use `CCUSPhysicsEngine` from `ccus_physics_engine.py` for new development.

    This class is kept for backward compatibility with legacy code.
    """

    def __init__(self, eor_params, pvt):
        """
        Initialize physics engine with EOR parameters.

        Args:
            eor_params: EORParameters object containing fluid and rock properties
            pvt: PVTProperties object containing fluid properties
        """
        warnings.warn(
            "PhysicsEngine is deprecated. Use CCUSPhysicsEngine from "
            "co2eor_optimizer.core.Phys_engine_full.ccus_physics_engine instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # FIXME: mobility_ratio is not a parameter of eor_params. It should be calculated from pvt data.
        self.eor_params = eor_params
        oil_viscosity = pvt.oil_viscosity_cp or 2.0
        co2_viscosity = pvt.gas_viscosity_cp or 0.08
        self.mobility_ratio = oil_viscosity / (co2_viscosity + 1e-6)

    def relative_permeability(self, S_co2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates oil and CO2 relative permeabilities using the Corey-Brooks model.

        Args:
            S_co2: Array of CO2 saturation values (fraction).

        Returns:
            A tuple of (k_ro, k_rg), where k_ro is the relative permeability to oil
            and k_rg is the relative permeability to CO2.
        """
        S_or = self.eor_params.sor
        S_gc = self.eor_params.s_gc
        n_o = self.eor_params.n_o
        n_g = self.eor_params.n_g

        # Normalize saturation for the Corey-Brooks model
        S_star = (S_co2 - S_gc) / (1 - S_or - S_gc + EPSILON)
        S_star = np.clip(S_star, 0, 1)

        k_ro = (1 - S_star) ** n_o
        k_rg = S_star**n_g

        return k_ro, k_rg

    def relative_permeability_water(
        self, S_w: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates oil and water relative permeabilities for water-oil system.

        Args:
            S_w: Array of water saturation values (fraction).

        Returns:
            A tuple of (k_row, k_rw), where k_row is the relative permeability to oil
            and k_rw is the relative permeability to water.
        """
        S_wc = self.eor_params.s_wc
        S_orw = self.eor_params.s_orw
        n_w = self.eor_params.n_w
        n_ow = self.eor_params.n_ow

        S_wn = (S_w - S_wc) / (1 - S_wc - S_orw + EPSILON)
        S_wn = np.clip(S_wn, 0, 1)

        k_rw = S_wn**n_w
        k_row = (1 - S_wn) ** n_ow
        return k_row, k_rw

    def fractional_flow(self, S_co2: np.ndarray) -> np.ndarray:
        """
        Calculates the fractional flow of CO2 as a function of CO2 saturation.

        f_co2 = (1 + (k_ro / k_rg) * (mu_co2 / mu_oil))^-1

        Args:
            S_co2: Array of CO2 saturation values.

        Returns:
            Array of corresponding CO2 fractional flow values.
        """
        k_ro, k_rg = self.relative_permeability(S_co2)

        # Avoid division by zero for k_rg at S_co2 <= S_gc
        k_rg[k_rg < EPSILON] = EPSILON

        # Avoid division by zero in the fractional flow calculation
        denominator = 1 + (k_ro / k_rg) * (1 / max(self.mobility_ratio, EPSILON))
        denominator[denominator < EPSILON] = EPSILON

        f_co2 = 1 / denominator
        return f_co2

    def water_fractional_flow(
        self, S_w: np.ndarray, current_pressure: float, pvt_props
    ) -> np.ndarray:
        """
        Calculates the fractional flow of water based on saturation and pressure.

        Args:
            S_w: Array of water saturation values.
            current_pressure: Current reservoir pressure (psi).
            pvt_props: PVTProperties object containing fluid properties.

        Returns:
            Array of corresponding water fractional flow values.
        """
        k_row, k_rw = self.relative_permeability_water(S_w)
        mu_w = 0.5  # cP

        # Interpolate oil viscosity at the current pressure
        pressure_range = pvt_props.pressure_points
        mu_oil = np.interp(current_pressure, pressure_range, pvt_props.oil_viscosity)

        k_rw[k_rw < 1e-9] = 1e-9

        f_w = (1 + (k_row / k_rw) * (mu_w / mu_oil)) ** -1
        return f_w

    def welge_tangent(self) -> Tuple[float, float, float]:
        """
        Performs the Welge tangent construction to find the CO2 saturation at the
        shock front (S_wf), the average CO2 saturation behind the front (S_w_avg),
        and the slope of the fractional flow curve at the front.
        This is key to Buckley-Leverett theory.

        Returns:
            A tuple of (S_wf, S_w_avg, df_dS_at_front).
        """
        S_gc = self.eor_params.s_gc
        S_or = self.eor_params.sor
        S_co2_range = np.linspace(S_gc, 1 - S_or, 500)
        f_co2 = self.fractional_flow(S_co2_range)

        # Welge tangent construction: find the tangent to the fractional flow curve
        # that starts at (S_gc, 0). The slope of this tangent gives the shock velocity.
        denominator = S_co2_range - S_gc
        # Avoid division by zero at the first point
        denominator[denominator < EPSILON] = EPSILON
        tangent_slope = f_co2 / denominator

        front_idx = np.argmax(tangent_slope)
        df_dS_at_front = tangent_slope[front_idx]
        S_wf = S_co2_range[front_idx]
        f_wf = f_co2[front_idx]

        if df_dS_at_front < EPSILON:
            S_w_avg = S_wf
        else:
            S_w_avg = S_wf + (1 - f_wf) / df_dS_at_front

        S_w_avg = np.clip(S_w_avg, 0, 1 - S_or)

        # Debug logging for breakthrough analysis
        logger.debug(
            f"Welge tangent: S_wf={S_wf:.4f}, S_w_avg={S_w_avg:.4f}, df_dS_at_front={df_dS_at_front:.6f}"
        )
        logger.debug(f"Fractional flow at front: f_wf={f_wf:.4f}")
        logger.debug(f"Critical saturations: S_gc={S_gc:.4f}, S_or={S_or:.4f}")

        return S_wf, S_w_avg, df_dS_at_front

    def buckley_leverett_derivative(self, S_co2: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the fractional flow curve (df/dS) for Buckley-Leverett analysis.

        Args:
            S_co2: Array of CO2 saturation values.

        Returns:
            Array of fractional flow derivatives (df/dS).
        """
        f_co2 = self.fractional_flow(S_co2)

        # Calculate numerical derivative using central differences
        df_dS = np.gradient(f_co2, S_co2)

        return df_dS


__all__ = [
    "PhysicsEngine",
    "PressureDynamics",
]
