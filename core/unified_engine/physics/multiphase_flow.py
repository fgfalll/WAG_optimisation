"""
Unified Multiphase Flow Module for the Unified Physics Engine.

Provides Darcy's law-based multiphase flow calculations shared by both
Simple and Detailed engines.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ..base.engine_config import EngineConfig, EngineMode
from ..base.physics_module import PhysicsModule
from ..core.state_manager import UnifiedState, Phase


@dataclass
class FluidProperties:
    """
    Fluid properties for a phase.

    Attributes:
        viscosity: Phase viscosity (cP)
        density: Phase density (lb/ft^3)
        formation_volume_factor: Formation volume factor (RB/STB)
        compressibility: Compressibility (1/psi)
        solution_gor: Solution gas-oil ratio (SCF/STB) for oil phase
    """
    viscosity: float
    density: float
    formation_volume_factor: float = 1.0
    compressibility: float = 1e-5
    solution_gor: float = 0.0


@dataclass
class RockProperties:
    """
    Rock properties.

    Attributes:
        porosity: Porosity (fraction)
        permeability: Permeability (mD) - can be isotropic (n_cells,) or anisotropic (n_cells, 3)
        compressibility: Rock compressibility (1/psi)
    """
    porosity: np.ndarray
    permeability: np.ndarray
    compressibility: float = 1e-6

    @property
    def is_anisotropic(self) -> bool:
        """Check if permeability is anisotropic."""
        return self.permeability.ndim == 2 and self.permeability.shape[1] == 3


class RelativePermeabilityModel:
    """
    Base class for relative permeability models.

    Provides unified interface for Corey, LET, and other models.
    """

    def __init__(self, model_type: str = "corey"):
        """
        Initialize relative permeability model.

        Args:
            model_type: Type of model ("corey", "let", "brooks_corey").
        """
        self.model_type = model_type
        self.parameters = {}

    def set_parameters(self, **kwargs) -> None:
        """Set model parameters."""
        self.parameters.update(kwargs)

    def calculate(
        self,
        oil_saturation: np.ndarray,
        water_saturation: np.ndarray,
        gas_saturation: np.ndarray,
        normalized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate relative permeabilities.

        Args:
            oil_saturation: Oil saturation (fraction).
            water_saturation: Water saturation (fraction).
            gas_saturation: Gas saturation (fraction).
            normalized: If True, use normalized saturations.

        Returns:
            Tuple of (kro, krw, krg) relative permeabilities.
        """
        if self.model_type == "corey":
            return self._corey(oil_saturation, water_saturation, gas_saturation, normalized)
        elif self.model_type == "let":
            return self._let(oil_saturation, water_saturation, gas_saturation, normalized)
        elif self.model_type == "brooks_corey":
            return self._brooks_corey(
                oil_saturation, water_saturation, gas_saturation, normalized
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _corey(
        self,
        so: np.ndarray,
        sw: np.ndarray,
        sg: np.ndarray,
        normalized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Corey correlation for relative permeability.

        kro = kro_max * ((so - Sor) / (1 - Swc - Sor - Sgc))^no
        krw = krw_max * ((sw - Swc) / (1 - Swc - Sor - Sgc))^nw
        krg = krg_max * ((sg - Sgc) / (1 - Swc - Sor - Sgc))^ng
        """
        Swc = self.parameters.get("Swc", 0.2)  # Connate water
        Sor = self.parameters.get("Sor", 0.2)  # Residual oil
        Sgc = self.parameters.get("Sgc", 0.0)  # Critical gas
        kro_max = self.parameters.get("kro_max", 1.0)
        krw_max = self.parameters.get("krw_max", 0.8)
        krg_max = self.parameters.get("krg_max", 0.8)
        no = self.parameters.get("no", 2.0)
        nw = self.parameters.get("nw", 2.0)
        ng = self.parameters.get("ng", 2.0)

        # Effective saturations
        if normalized:
            so_eff = so
            sw_eff = sw
            sg_eff = sg
        else:
            so_eff = np.maximum((so - Sor) / (1 - Swc - Sor - Sgc), 0)
            sw_eff = np.maximum((sw - Swc) / (1 - Swc - Sor - Sgc), 0)
            sg_eff = np.maximum((sg - Sgc) / (1 - Swc - Sor - Sgc), 0)

        # Corey equations
        kro = kro_max * so_eff**no
        krw = krw_max * sw_eff**nw
        krg = krg_max * sg_eff**ng

        return kro, krw, krg

    def _let(
        self,
        so: np.ndarray,
        sw: np.ndarray,
        sg: np.ndarray,
        normalized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LET (Lomeland, Ebeltoft, Thomas) correlation.

        kro = kro_max * (1 - Sw_eff)^Lw / ((1 - Sw_eff)^Lw + Tw * Sw_eff^Ew)
        """
        Lw = self.parameters.get("Lw", 2.0)
        Ew = self.parameters.get("Ew", 2.0)
        Tw = self.parameters.get("Tw", 2.0)
        kro_max = self.parameters.get("kro_max", 1.0)

        # Water relative permeability (LET form)
        Sw_eff = np.clip(sw, 0, 1)
        krw = (1 - Sw_eff) ** Lw / ((1 - Sw_eff) ** Lw + Tw * Sw_eff**Ew)
        kro = kro_max * Sw_eff**Lw / (Sw_eff**Lw + Tw * (1 - Sw_eff) ** Ew)
        krg = np.zeros_like(so)  # Simplified

        return kro, krw, krg

    def _brooks_corey(
        self,
        so: np.ndarray,
        sw: np.ndarray,
        sg: np.ndarray,
        normalized: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Brooks-Corey model.

        Similar to Corey but with pore size distribution index lambda.
        """
        Swc = self.parameters.get("Swc", 0.2)
        Sor = self.parameters.get("Sor", 0.2)
        lambda_bc = self.parameters.get("lambda", 2.0)

        Sw_eff = np.maximum((sw - Swc) / (1 - Swc - Sor), 0)
        krw = Sw_eff ** ((2 + 3 * lambda_bc) / lambda_bc)
        kro = (1 - Sw_eff) ** 2 * (1 - Sw_eff ** ((2 + lambda_bc) / lambda_bc))
        krg = np.zeros_like(so)

        return kro, krw, krg


class DarcyFlowCalculator:
    """
    Darcy's law flow calculations for multiphase flow.

    Calculates phase fluxes based on pressure gradients and relative permeabilities.
    """

    def __init__(self, config: EngineConfig):
        """Initialize Darcy flow calculator."""
        self.config = config
        self.relperm_model = RelativePermeabilityModel("corey")

    def set_relperm_model(self, model: RelativePermeabilityModel) -> None:
        """Set relative permeability model."""
        self.relperm_model = model

    def calculate_phase_flux(
        self,
        pressure_gradient: np.ndarray,
        saturation: np.ndarray,
        kr: np.ndarray,
        viscosity: float,
        permeability: np.ndarray,
        gravity: float = 0.0,
        density: float = 0.0,
        direction: str = "x",
    ) -> np.ndarray:
        """
        Calculate phase flux using Darcy's law.

        q = -(k * kr / mu) * (grad(P) - rho * g)

        Args:
            pressure_gradient: Pressure gradient (psi/ft).
            saturation: Phase saturation (fraction).
            kr: Relative permeability.
            viscosity: Phase viscosity (cP).
            permeability: Absolute permeability (mD).
            gravity: Gravity component in direction (ft/s^2).
            density: Phase density (lb/ft^3).
            direction: Flow direction ('x', 'y', or 'z').

        Returns:
            Phase flux (ft/day).
        """
        # Conversion factor: 6.33e-3 for field units
        conversion = 6.33e-3

        # Direction index for anisotropic permeability
        if permeability.ndim == 2:
            dir_idx = {"x": 0, "y": 1, "z": 2}[direction]
            k = permeability[:, dir_idx]
        else:
            k = permeability

        # Mobility
        mobility = k * kr / viscosity

        # Potential gradient
        potential_gradient = pressure_gradient.copy()

        # Add gravity term
        if gravity != 0 and density != 0:
            # Conversion: lb/ft^3 to psi/ft
            g_pressure = density * gravity / 144.0  # psi/ft
            potential_gradient -= g_pressure

        # Darcy flux
        flux = -conversion * mobility * potential_gradient

        return flux

    def calculate_transmissibility(
        self,
        permeability_i: float,
        permeability_j: float,
        kr_i: float,
        kr_j: float,
        viscosity: float,
        distance: float,
        area: float,
    ) -> float:
        """
        Calculate inter-block transmissibility.

        T = (2 * ki * kj / (ki + kj)) * (kr_avg / mu) * (A / d)

        Args:
            permeability_i, permeability_j: Permeabilities of adjacent cells (mD).
            kr_i, kr_j: Relative permeabilities of adjacent cells.
            viscosity: Phase viscosity (cP).
            distance: Distance between cell centers (ft).
            area: Interface area (ft^2).

        Returns:
            Transmissibility (ft^3/day/psi).
        """
        # Harmonic average of permeabilities
        k_avg = 2 * permeability_i * permeability_j / (permeability_i + permeability_j + 1e-20)

        # Upwind relative permeability
        kr_avg = kr_i  # Simplified - use upstream weighting

        # Transmissibility
        T = 6.33e-3 * k_avg * kr_avg / viscosity * area / distance

        return T


class MultiphaseFlowModule(PhysicsModule):
    """
    Unified multiphase flow physics module.

    Handles:
    - Darcy flow for all phases
    - Relative permeability calculations
    - Phase mobilities
    - Capillary pressure (optional)
    """

    def __init__(self, config: EngineConfig, name: str = "multiphase_flow"):
        """Initialize multiphase flow module."""
        super().__init__(config, name)

        # Flow calculator
        self.flow_calculator = DarcyFlowCalculator(config)

        # Fluid properties (default values)
        self.oil_properties = FluidProperties(
            viscosity=1.0,  # cP
            density=50.0,  # lb/ft^3
            formation_volume_factor=1.2,
            compressibility=1e-5,
        )

        self.water_properties = FluidProperties(
            viscosity=0.5,  # cP
            density=64.0,  # lb/ft^3
            formation_volume_factor=1.0,
            compressibility=3e-6,
        )

        self.gas_properties = FluidProperties(
            viscosity=0.02,  # cP
            density=0.1,  # lb/ft^3
            formation_volume_factor=0.003,
            compressibility=1e-3,
        )

        # Rock properties (initialized from grid data)
        self.rock_properties: Optional[RockProperties] = None

        # Capillary pressure (optional)
        self.enable_capillary_pressure = False
        self.capillary_pressure_model = None

        # CO2 properties (for EOR)
        self.co2_properties = FluidProperties(
            viscosity=0.05,  # cP
            density=40.0,  # lb/ft^3 (supercritical)
            formation_volume_factor=0.5,
            compressibility=5e-4,
        )

    def initialize(self, state: UnifiedState) -> None:
        """Initialize module with state."""
        super().initialize(state)

        # Set up rock properties from state
        if state.porosity is not None and state.permeability is not None:
            self.rock_properties = RockProperties(
                porosity=state.porosity,
                permeability=state.permeability,
                compressibility=1e-6,
            )

        # Configure relative permeability model
        if self.config.modules.relperm_model.value == "corey":
            self.flow_calculator.relperm_model.set_parameters(
                Swc=0.2, Sor=0.2, Sgc=0.0, no=2.0, nw=2.0, ng=2.0
            )

    def update(self, state: UnifiedState, dt: float) -> UnifiedState:
        """
        Update multiphase flow for a timestep.

        For explicit scheme: Calculate fluxes and update saturations directly.
        For implicit scheme: Prepare Jacobian contributions.
        """
        super().update(state, dt)

        # Calculate relative permeabilities
        so = state.oil_saturation
        sw = state.water_saturation
        sg = state.gas_saturation

        kro, krw, krg = self.flow_calculator.relperm_model.calculate(so, sw, sg)

        # Store relative permeabilities in state for use by solvers
        if not hasattr(state, "relperm"):
            state.convergence_info["relperm"] = {}
        state.convergence_info["relperm"]["kro"] = kro
        state.convergence_info["relperm"]["krw"] = krw
        state.convergence_info["relperm"]["krg"] = krg

        # Calculate mobilities
        if self.rock_properties is not None:
            # Get permeability
            k = self.rock_properties.permeability
            if k.ndim == 1:
                k = k[:, np.newaxis]

            # Mobilities: lambda = k * kr / mu
            mobility_oil = k[:, 0] * kro / self.oil_properties.viscosity
            mobility_water = k[:, 0] * krw / self.water_properties.viscosity
            mobility_gas = k[:, 0] * krg / self.gas_properties.viscosity

            state.convergence_info["mobility"] = {
                "oil": mobility_oil,
                "water": mobility_water,
                "gas": mobility_gas,
                "total": mobility_oil + mobility_water + mobility_gas,
            }

        return state

    def calculate_phase_fluxes(
        self, state: UnifiedState, pressure_gradient: np.ndarray, direction: str = "x"
    ) -> Dict[str, np.ndarray]:
        """
        Calculate phase fluxes for a given pressure gradient.

        Args:
            state: Current simulation state.
            pressure_gradient: Pressure gradient (psi/ft).
            direction: Flow direction.

        Returns:
            Dictionary of phase fluxes.
        """
        if self.rock_properties is None:
            raise RuntimeError("Rock properties not initialized")

        kro, krw, krg = self.flow_calculator.relperm_model.calculate(
            state.oil_saturation, state.water_saturation, state.gas_saturation
        )

        flux_oil = self.flow_calculator.calculate_phase_flux(
            pressure_gradient,
            state.oil_saturation,
            kro,
            self.oil_properties.viscosity,
            self.rock_properties.permeability,
            density=self.oil_properties.density,
            direction=direction,
        )

        flux_water = self.flow_calculator.calculate_phase_flux(
            pressure_gradient,
            state.water_saturation,
            krw,
            self.water_properties.viscosity,
            self.rock_properties.permeability,
            density=self.water_properties.density,
            direction=direction,
        )

        flux_gas = self.flow_calculator.calculate_phase_flux(
            pressure_gradient,
            state.gas_saturation,
            krg,
            self.gas_properties.viscosity,
            self.rock_properties.permeability,
            density=self.gas_properties.density,
            direction=direction,
        )

        return {"oil": flux_oil, "water": flux_water, "gas": flux_gas}

    def get_required_state_fields(self) -> List[str]:
        """Get required state fields."""
        return ["pressure", "saturations"]

    def get_optional_state_fields(self) -> List[str]:
        """Get optional state fields."""
        return ["porosity", "permeability", "compositions"]


def create_multiphase_flow_module(config: EngineConfig) -> MultiphaseFlowModule:
    """Factory function to create multiphase flow module."""
    return MultiphaseFlowModule(config)
