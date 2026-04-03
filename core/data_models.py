import dataclasses
from dataclasses import field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from enum import Enum
from typing import Optional


def from_dict_to_dataclass(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        field_value = data.get(f.name)
        if field_value is not None:
            # Check if the field is a dataclass and the value is a dict for nested creation
            if dataclasses.is_dataclass(f.type) and isinstance(field_value, dict):
                kwargs[f.name] = from_dict_to_dataclass(f.type, field_value)
            else:
                kwargs[f.name] = field_value

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in field_names}

    return cls(**filtered_kwargs)


@dataclasses.dataclass
class WellData:
    name: str
    depths: np.ndarray
    properties: Dict[str, np.ndarray]
    units: Dict[str, str]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    perforation_properties: List[Dict[str, float]] = dataclasses.field(default_factory=list)
    well_path: Optional[np.ndarray] = None

    def validate(self) -> bool:
        if not hasattr(self.depths, "size") or self.depths.size == 0:
            return all(
                not hasattr(prop, "size") or prop.size == 0 for prop in self.properties.values()
            )

        is_length_valid = all(
            len(self.depths) == len(prop_array) for prop_array in self.properties.values()
        )
        if not is_length_valid:
            logging.error(f"Validation failed for well '{self.name}': Mismatch in array lengths.")
            return False

        if self.depths.size > 1:
            if not np.all(np.diff(self.depths) > 0):
                logging.warning(f"Well '{self.name}' has non-monotonic or duplicate depth values.")
                return False

        return True


@dataclasses.dataclass
class EOSModelParameters:
    eos_type: str
    component_names: List[str]
    component_properties: np.ndarray
    binary_interaction_coeffs: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.component_properties, np.ndarray)
            or self.component_properties.ndim != 2
        ):
            raise TypeError("component_properties must be a 2D numpy array.")

        num_components = len(self.component_names)
        if num_components == 0:
            raise ValueError("component_names list cannot be empty.")

        if self.component_properties.shape != (num_components, 5):
            raise ValueError(
                f"component_properties shape must be ({num_components}, 5) to match the number of component names."
            )

        try:
            mole_fractions = self.component_properties[:, 0].astype(float)
            if not np.isclose(np.sum(mole_fractions), 1.0, atol=1e-3):
                logging.warning(
                    f"Mole fractions sum to {np.sum(mole_fractions):.4f}, which is not 1.0. Check fluid composition."
                )
        except (IndexError, ValueError):
            raise ValueError(
                "Could not parse mole fractions from the first column of component_properties."
            )

        if (
            not isinstance(self.binary_interaction_coeffs, np.ndarray)
            or self.binary_interaction_coeffs.ndim != 2
        ):
            raise TypeError("binary_interaction_coeffs must be a 2D numpy array.")

        if self.binary_interaction_coeffs.shape != (num_components, num_components):
            raise ValueError(
                "The dimensions of binary_interaction_coeffs must be square and match the number of components."
            )

        if not np.allclose(self.binary_interaction_coeffs, self.binary_interaction_coeffs.T):
            logging.warning(
                "binary_interaction_coeffs matrix is not symmetric. It will be treated as symmetric by the solver."
            )


@dataclasses.dataclass
class GeostatisticalParams:
    variogram_type: str = "spherical"
    range: float = 0.0  # 0 signals "auto-calculate based on grid size"
    sill: float = 0.1
    nugget: float = 0.0
    anisotropy_ratio: float = 1.0
    anisotropy_angle: float = 0.0
    trend_type: str = "none"
    trend_parameters: List[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0])
    simulation_method: str = "sequential_gaussian"
    random_seed: int = 42
    grid_resolution: Tuple[int, int] = (100, 100)

    def __post_init__(self):
        valid_variogram_types = [
            "spherical",
            "exponential",
            "gaussian",
            "matern",
            "cubic",
        ]
        if self.variogram_type not in valid_variogram_types:
            raise ValueError(f"Variogram type must be one of: {valid_variogram_types}")

        # Allow range = 0 as special case for auto-calculation
        if self.range < 0:
            raise ValueError("Range must be non-negative (0 for auto-calculation)")
        if self.sill <= 0:
            raise ValueError("Sill must be positive")
        if self.nugget < 0:
            raise ValueError("Nugget must be non-negative")
        if self.anisotropy_ratio <= 0:
            raise ValueError("Anisotropy ratio must be positive")
        if not (0 <= self.anisotropy_angle < 360):
            raise ValueError("Anisotropy angle must be between 0 and 360 degrees")

        valid_trend_types = ["none", "linear", "quadratic"]
        if self.trend_type not in valid_trend_types:
            raise ValueError(f"Trend type must be one of: {valid_trend_types}")

        valid_simulation_methods = ["sequential_gaussian", "turning_bands", "fft"]
        if self.simulation_method not in valid_simulation_methods:
            raise ValueError(f"Simulation method must be one of: {valid_simulation_methods}")

        if len(self.grid_resolution) != 2 or any(r <= 0 for r in self.grid_resolution):
            raise ValueError("Grid resolution must be a tuple of two positive integers")


@dataclasses.dataclass
class LayerDefinition:
    thickness: float
    porosity: float
    permeability_multiplier: float
    param_overrides: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ReservoirData:
    grid: Dict[str, np.ndarray]
    pvt_tables: Dict[str, np.ndarray]
    regions: Optional[Dict[str, np.ndarray]] = None
    runspec: Optional[Dict[str, Any]] = None
    faults: Optional[Dict[str, Any]] = None
    ooip_stb: float = 1_000_000.0
    initial_pressure: float = 4000.0
    rock_compressibility: float = 3e-6
    temperature: float = 150.0
    length_ft: Optional[float] = 2000.0
    cross_sectional_area_acres: Optional[float] = 10.0
    area_acres: Optional[float] = None
    thickness_ft: Optional[float] = None
    average_permeability: Optional[float] = None
    average_porosity: Optional[float] = None
    initial_water_saturation: Optional[float] = None
    oil_fvf: Optional[float] = None
    rock_type: Optional[str] = None
    depositional_environment: Optional[str] = None
    structural_complexity: Optional[str] = None
    eos_model: Optional[EOSModelParameters] = None
    layer_definitions: Optional[List[LayerDefinition]] = None
    geostatistical_params: Optional["GeostatisticalParams"] = None
    geostatistical_grid: Optional[np.ndarray] = None

    def validate(self, physics_based_model: bool = False, tolerance: float = 0.1) -> None:
        self._validate_basic_parameters()
        if physics_based_model:
            self._validate_physics_based_consistency(tolerance)
            self._validate_geology_parameters()

    def _validate_basic_parameters(self) -> None:
        if self.ooip_stb <= 0:
            raise ValueError("OOIP must be positive")
        if self.initial_pressure <= 0:
            raise ValueError("Initial pressure must be positive")
        if self.rock_compressibility <= 0:
            raise ValueError("Rock compressibility must be positive")
        if self.length_ft is not None and self.length_ft <= 0:
            raise ValueError("Reservoir length must be positive")
        if self.cross_sectional_area_acres is not None and self.cross_sectional_area_acres <= 0:
            raise ValueError("Cross-sectional area must be positive")

    def _validate_physics_based_consistency(self, tolerance: float) -> None:
        required_params = [
            "area_acres",
            "thickness_ft",
            "average_porosity",
            "initial_water_saturation",
            "oil_fvf",
        ]
        missing_params = [param for param in required_params if getattr(self, param) is None]
        if missing_params:
            raise ValueError(f"Physics-based models require: {', '.join(missing_params)}")

        calculated_ooip = self.calculate_ooip_from_physics()
        if abs(calculated_ooip - self.ooip_stb) / calculated_ooip > tolerance:
            raise ValueError(
                f"OOIP dimensional inconsistency: Calculated {calculated_ooip:,.0f} STB "
                f"vs Provided {self.ooip_stb:,.0f} STB (tolerance: {tolerance * 100:.1f}%)"
            )

    def _validate_geology_parameters(self) -> None:
        if self.average_porosity is not None and not (0.01 <= self.average_porosity <= 0.60):
            raise ValueError("Porosity must be between 0.01 and 0.60")
        if self.initial_water_saturation is not None and not (
            0.0 <= self.initial_water_saturation <= 1.0
        ):
            raise ValueError("Initial water saturation must be between 0.0 and 1.0")
        if self.oil_fvf is not None and self.oil_fvf <= 0:
            raise ValueError("Oil FVF must be positive")
        if self.area_acres is not None and self.area_acres <= 0:
            raise ValueError("Reservoir area must be positive")
        if self.thickness_ft is not None and self.thickness_ft <= 0:
            raise ValueError("Reservoir thickness must be positive")

    def calculate_ooip_from_physics(self) -> float:
        if None in [
            self.area_acres,
            self.thickness_ft,
            self.average_porosity,
            self.initial_water_saturation,
            self.oil_fvf,
        ]:
            raise ValueError("Cannot calculate OOIP: Missing required physical parameters")

        pore_volume = 7758 * self.area_acres * self.thickness_ft * self.average_porosity
        return pore_volume * (1 - self.initial_water_saturation) / self.oil_fvf

    def is_physics_based_model_compatible(self) -> bool:
        required_params = [
            "area_acres",
            "thickness_ft",
            "average_porosity",
            "initial_water_saturation",
            "oil_fvf",
        ]
        return all(getattr(self, param) is not None for param in required_params)

    def generate_geostatistical_grid(self, shape: tuple):
        if self.geostatistical_params:
            from .geology.geostatistical_modeling import create_geostatistical_grid

            self.geostatistical_grid = create_geostatistical_grid(
                shape, self.geostatistical_params.__dict__
            )

    @property
    def pore_volume(self) -> float:
        """
        Calculate and return pore volume in barrels.

        Uses the ProfilerUtils.calculate_pore_volume method which handles
        both geostatistical grids and standard grid PORO data.

        Returns
        -------
        float
            Pore volume in barrels (bbl)
        """
        from .utils.profiler_utils import ProfilerUtils

        return ProfilerUtils.calculate_pore_volume(self)


class FaultType(Enum):
    NORMAL = "normal"
    REVERSE = "reverse"
    STRIKE_SLIP = "strike_slip"
    OBLIQUE = "oblique"


class FractureState(Enum):
    INTACT = "intact"
    SLIPPING = "slipping"
    OPEN = "open"
    SEALED = "sealed"


@dataclasses.dataclass
class FaultGeometry:
    """Geometric properties of a fault"""

    fault_id: int
    fault_type: FaultType
    center_coordinates: Tuple[float, float, float]  # x, y, z
    strike: float  # degrees from north
    dip: float  # degrees from horizontal
    length: float  # ft
    width: float  # ft
    connected_cells: List[int]  # grid cell indices intersected by fault

    # --- *** START OF ADDITION *** ---
    def calculate_normal_vector(self) -> np.ndarray:
        """Calculate unit normal vector to fault plane"""
        strike_rad = np.radians(self.strike)
        dip_rad = np.radians(self.dip)

        # Normal vector components (pointing into hanging wall)
        nx = -np.sin(strike_rad) * np.sin(dip_rad)
        ny = np.cos(strike_rad) * np.sin(dip_rad)
        nz = -np.cos(dip_rad)

        return np.array([nx, ny, nz])

    def calculate_slip_direction(self) -> np.ndarray:
        """Calculate slip direction vector based on fault type"""
        strike_rad = np.radians(self.strike)
        dip_rad = np.radians(self.dip)

        if self.fault_type == FaultType.NORMAL:
            # Slip direction: down-dip
            dx = np.cos(strike_rad) * np.cos(dip_rad)
            dy = np.sin(strike_rad) * np.cos(dip_rad)
            dz = np.sin(dip_rad)
        elif self.fault_type == FaultType.REVERSE:
            # Slip direction: up-dip
            dx = -np.cos(strike_rad) * np.cos(dip_rad)
            dy = -np.sin(strike_rad) * np.cos(dip_rad)
            dz = -np.sin(dip_rad)
        elif self.fault_type == FaultType.STRIKE_SLIP:
            # Slip direction: along strike
            dx = -np.sin(strike_rad)
            dy = np.cos(strike_rad)
            dz = 0.0
        else:  # OBLIQUE
            # Combination of dip-slip and strike-slip
            dx = -np.sin(strike_rad) - 0.5 * np.cos(strike_rad) * np.cos(dip_rad)
            dy = np.cos(strike_rad) - 0.5 * np.sin(strike_rad) * np.cos(dip_rad)
            dz = -0.5 * np.sin(dip_rad)

        # Normalize the vector
        norm = np.linalg.norm([dx, dy, dz])
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])  # Avoid division by zero
        return np.array([dx, dy, dz]) / norm


@dataclasses.dataclass
class FaultProperties:
    cohesion: float
    friction_coefficient: float
    dilation_angle: float
    initial_aperture: float
    maximum_aperture: float
    transmissibility_multiplier: float = 1.0
    healing_rate: float = 0.0
    stiffness: float = 1.0e6

    def __post_init__(self):
        if self.cohesion <= 0:
            raise ValueError("Cohesion must be positive.")
        if not (0 < self.friction_coefficient < 1.5):
            logging.warning("Friction coefficient is outside the typical range (0, 1.5).")
        if not (0 <= self.dilation_angle <= 30):
            logging.warning("Dilation angle is outside the typical range (0, 30 degrees).")
        if self.initial_aperture < 0 or self.maximum_aperture < self.initial_aperture:
            raise ValueError("Aperture values are invalid.")

    def calculate_friction_angle(self) -> float:
        return np.degrees(np.arctan(self.friction_coefficient))


@dataclasses.dataclass
class RockProperties:
    """
    Rock properties for reservoir simulation.

    This dataclass provides standardized rock property definitions used
    across both engine_simple and phys_engine_full.
    """

    porosity: np.ndarray  # Porosity (fraction, 0-1)
    permeability_x: np.ndarray  # Permeability in x-direction (mD)
    permeability_y: np.ndarray  # Permeability in y-direction (mD)
    permeability_z: np.ndarray  # Permeability in z-direction (mD)
    compressibility: float = 1e-5  # Rock compressibility (1/Pa)

    def __post_init__(self):
        """Validate rock properties"""
        if np.any(self.porosity <= 0) or np.any(self.porosity >= 1):
            raise ValueError("Porosity must be between 0 and 1")
        if np.any(self.permeability_x <= 0):
            raise ValueError("Permeability must be positive")
        if np.any(self.permeability_y <= 0):
            raise ValueError("Permeability must be positive")
        if np.any(self.permeability_z <= 0):
            raise ValueError("Permeability must be positive")

    @property
    def permeability(self) -> np.ndarray:
        """Average permeability (mD) - geometric mean of directional permeabilities"""
        return (self.permeability_x * self.permeability_y * self.permeability_z) ** (1 / 3)


@dataclasses.dataclass
class FluidProperties:
    """
    Fluid properties for reservoir simulation.

    This dataclass provides standardized fluid property definitions used
    across both engine_simple and phys_engine_full.
    """

    # Water properties
    water_density_ref: float = 1000.0  # Reference water density (kg/m³)
    water_viscosity_ref: float = 0.001  # Reference water viscosity (Pa·s)
    water_compressibility: float = 4.5e-10  # Water compressibility (1/Pa)

    # Oil properties
    oil_density_ref: float = 850.0  # Reference oil density (kg/m³)
    oil_viscosity_ref: float = 0.002  # Reference oil viscosity (Pa·s)
    oil_compressibility: float = 1e-9  # Oil compressibility (1/Pa)
    bubble_point_pressure: float = 200e5  # Bubble point pressure (Pa)
    solution_gor: float = 50.0  # Solution gas-oil ratio (sm³/sm³)

    # Gas properties
    gas_density_ref: float = 1.0  # Reference gas density (kg/m³)
    gas_viscosity_ref: float = 2e-5  # Reference gas viscosity (Pa·s)
    gas_compressibility: float = 1e-8  # Gas compressibility (1/Pa)

    # Formation volume factors
    water_fvf_ref: float = 1.0  # Water formation volume factor
    oil_fvf_ref: float = 1.2  # Oil formation volume factor
    gas_fvf_ref: float = 0.005  # Gas formation volume factor

    # Initial saturations
    initial_water_saturation: float = 0.2  # Initial water saturation (fraction)
    oil_fvf: float = 1.2  # Oil formation volume factor at reservoir conditions

    def __post_init__(self):
        """Validate fluid properties"""
        if any(
            param <= 0
            for param in [
                self.water_density_ref,
                self.water_viscosity_ref,
                self.oil_density_ref,
                self.oil_viscosity_ref,
                self.gas_density_ref,
                self.gas_viscosity_ref,
            ]
        ):
            raise ValueError("Density and viscosity must be positive")

    def water_density(self, pressure: float, temperature: float) -> float:
        """Calculate water density at given pressure and temperature"""
        rho = self.water_density_ref * (1 + self.water_compressibility * pressure)
        return rho

    def water_viscosity(self, pressure: float, temperature: float) -> float:
        """Calculate water viscosity at given pressure and temperature"""
        mu = self.water_viscosity_ref * np.exp(0.02 * (temperature - 293.15))
        p_scale = 1 + 0.001 * (pressure - 1e5) / 1e5
        mu *= min(p_scale, 5.0)
        return mu

    def oil_density(self, pressure: float, temperature: float) -> float:
        """Calculate oil density at given pressure and temperature"""
        rho = self.oil_density_ref * (1 + self.oil_compressibility * pressure)
        return rho

    def oil_viscosity(self, pressure: float, temperature: float) -> float:
        """Calculate oil viscosity at given pressure and temperature"""
        mu = self.oil_viscosity_ref * np.exp(0.03 * (temperature - 293.15))
        # Cap physical density derivations so mobilities don't mathematically lock well equations beneath injection constants
        p_scale = 1 + 0.002 * (pressure - 1e5) / 1e5
        mu *= min(p_scale, 5.0)
        return mu

    def gas_density(self, pressure: float, temperature: float) -> float:
        """Calculate gas density using ideal gas law"""
        R = 8.314  # Gas constant J/(mol·K)
        M_gas = 0.016  # Molar mass of methane kg/mol
        rho = (M_gas * pressure) / (R * temperature)
        return rho

    def gas_viscosity(self, pressure: float, temperature: float) -> float:
        """Calculate gas viscosity at given pressure and temperature"""
        mu = self.gas_viscosity_ref * (temperature / 273.15) ** 0.7
        p_scale = 1 + 0.01 * (pressure - 1e5) / 1e5
        mu *= min(p_scale, 5.0)
        return mu


@dataclasses.dataclass
class EconomicParameters:
    oil_price_usd_per_bbl: float = 70.0
    co2_purchase_cost_usd_per_tonne: float = 50.0
    co2_recycle_cost_usd_per_tonne: float = 15.0
    co2_storage_credit_usd_per_tonne: float = 25.0
    water_injection_cost_usd_per_bbl: float = 1.0
    water_disposal_cost_usd_per_bbl: float = 2.0
    discount_rate_fraction: float = 0.10
    capex_usd: float = 5_000_000.0
    fixed_opex_usd_per_year: float = 200_000.0
    variable_opex_usd_per_bbl: float = 5.0
    carbon_tax_usd_per_tonne: float = 0.0

    def __post_init__(self):
        if not (0.0 < self.oil_price_usd_per_bbl <= 1000.0):
            raise ValueError("Oil Price must be between $0 and $1000/bbl.")
        if not (0.0 <= self.co2_purchase_cost_usd_per_tonne <= 1000.0):
            raise ValueError("CO2 Purchase Cost must be between $0 and $1000/tonne.")
        if self.co2_recycle_cost_usd_per_tonne > self.co2_purchase_cost_usd_per_tonne:
            logging.warning("CO2 Recycle Cost is higher than Purchase Cost, which is unusual.")
        if not (0.0 <= self.co2_storage_credit_usd_per_tonne <= 1000.0):
            raise ValueError("CO2 Storage Credit must be between $0 and $1000/tonne.")
        if not (0.0 <= self.discount_rate_fraction <= 1.0):
            raise ValueError("Discount Rate must be a fraction between 0.0 and 1.0.")
        if not (0.0 <= self.capex_usd):
            raise ValueError("CAPEX must be non-negative.")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class HuffNPuffParams:
    cycle_length_days: int = 90
    injection_period_days: int = 30
    soaking_period_days: int = 15
    production_period_days: int = 45
    max_cycles: int = 10

    # Pressure dynamics parameters for huff-n-puff cycles
    max_pressure_increase_psi_day: float = 100.0  # Maximum pressure increase during injection (psi/day)
    min_pressure_decline_psi_day: float = 5.0   # Minimum pressure decline during production (psi/day)
    max_pressure_decline_psi_day: float = 80.0   # Maximum pressure decline during production (psi/day)
    soaking_pressure_decline_psi_day: float = 2.0  # Pressure decline during soaking (psi/day)

    def __post_init__(self):
        if self.cycle_length_days != (
            self.injection_period_days + self.soaking_period_days + self.production_period_days
        ):
            logging.warning("Huff-n-Puff cycle periods do not sum to the total cycle length.")
        if self.max_cycles <= 0:
            raise ValueError("Max cycles must be positive.")
        if self.max_pressure_increase_psi_day < 0:
            raise ValueError("Max pressure increase must be non-negative.")
        if self.min_pressure_decline_psi_day < 0:
            raise ValueError("Min pressure decline must be non-negative.")
        if self.max_pressure_decline_psi_day < self.min_pressure_decline_psi_day:
            raise ValueError("Max pressure decline must be >= min pressure decline.")
        if self.soaking_pressure_decline_psi_day < 0:
            raise ValueError("Soaking pressure decline must be non-negative.")


@dataclasses.dataclass
class SWAGParams:
    water_gas_ratio: float = 1.0
    simultaneous_injection: bool = True
    mixing_efficiency: float = 0.8

    def __post_init__(self):
        if self.water_gas_ratio < 0:
            raise ValueError("SWAG water-gas ratio must be non-negative.")
        if not (0 <= self.mixing_efficiency <= 1.0):
            raise ValueError("SWAG mixing efficiency must be between 0 and 1.")


@dataclasses.dataclass
class TaperedInjectionParams:
    initial_rate_multiplier: float = 1.5
    final_rate_multiplier: float = 0.5
    duration_years: float = 5.0
    function: str = "linear"

    def __post_init__(self):
        if self.initial_rate_multiplier <= 0:
            raise ValueError("Tapered initial rate multiplier must be positive.")
        if self.final_rate_multiplier <= 0:
            raise ValueError("Tapered final rate multiplier must be positive.")
        if self.duration_years <= 0:
            raise ValueError("Tapered duration must be positive.")
        if self.function not in ["linear", "exponential", "logarithmic"]:
            raise ValueError("Tapered function must be 'linear', 'exponential', or 'logarithmic'.")


@dataclasses.dataclass
class PulsedInjectionParams:
    pulse_duration_days: int = 7
    pause_duration_days: int = 14
    intensity_multiplier: float = 2.0

    def __post_init__(self):
        if self.pulse_duration_days <= 0:
            raise ValueError("Pulsed pulse duration must be positive.")
        if self.pause_duration_days <= 0:
            raise ValueError("Pulsed pause duration must be positive.")
        if self.intensity_multiplier <= 0:
            raise ValueError("Pulsed intensity multiplier must be positive.")


@dataclasses.dataclass
class EORParameters:
    injection_scheme: str = "continuous"
    injection_rate: float = 5000.0

    huff_n_puff: Optional[HuffNPuffParams] = None
    swag: Optional[SWAGParams] = None
    tapered: Optional[TaperedInjectionParams] = None
    pulsed: Optional[PulsedInjectionParams] = None

    mobility_ratio: float = 5.0
    target_pressure_psi: float = 3000.0
    max_pressure_psi: float = 6000.0
    min_pressure_factor: float = 1.1
    min_injection_rate_mscfd: float = 5000.0
    max_injection_rate_mscfd: float = 100000.0
    density_contrast: float = 0.3
    dip_angle: float = 0.0
    interfacial_tension: float = 5.0
    sor: float = 0.25
    gas_oil_ratio_at_breakthrough: float = 800.0
    water_cut_bwow: float = 0.98
    co2_density_tonne_per_mscf: float = 0.053
    injection_gor: float = 10000.0
    initial_gor: float = 200.0
    default_mmp_fallback: float = 2500.0
    kv_kh_ratio: float = 0.1
    WAG_ratio: float = 1.0  # Added missing attribute
    cycle_length_days: float = 90.0
    water_fraction: float = 0.5
    
    # Flattened UI Injection Scheme parameters
    huff_n_puff_cycle_length_days: float = 90.0
    huff_n_puff_injection_period_days: float = 30.0
    huff_n_puff_soaking_period_days: float = 15.0
    huff_n_puff_production_period_days: float = 45.0
    huff_n_puff_max_cycles: int = 10
    
    swag_water_gas_ratio: float = 1.0
    swag_simultaneous_injection: bool = True
    swag_mixing_efficiency: float = 1.0
    
    tapered_initial_rate_multiplier: float = 2.0
    tapered_final_rate_multiplier: float = 0.5
    tapered_duration_years: float = 5.0
    tapered_function: str = "linear"
    
    pulsed_pulse_duration_days: float = 15.0
    pulsed_pause_duration_days: float = 15.0
    pulsed_intensity_multiplier: float = 2.0

    s_gc: float = 0.05
    n_o: float = 2.0
    n_g: float = 2.0
    s_wc: float = 0.2
    s_orw: float = 0.2
    n_w: float = 2.0
    n_ow: float = 2.0

    productivity_index: float = 5.0
    wellbore_pressure: float = 500.0
    well_shut_in_threshold_bpd: float = 10.0
    max_injector_bhp_psi: float = 8000.0  # Reduced from 12000 to prevent excessive pressure buildup
    timestep_days: float = 30.44  # Monthly timestep (average days per month)

    # Pressure control parameters (added for peer review fix)
    min_allowable_pressure_psi: float = 1600.0
    min_allowable_pressure_fraction_of_initial: float = 0.35

    # Well control parameters (added for BHP-based control - Phase 3)
    injector_target_bhp_psi: float = 4000.0  # Target bottom-hole pressure for injectors
    producer_target_bhp_psi: float = 1500.0   # Target bottom-hole pressure for producers
    use_bhp_control: bool = True              # Enable BHP-based well control
    pressure_control_min_scaling_factor: float = 0.01  # Minimum injection rate scaling factor

    # Removed: timestep_days - Use CCUSParameters.timestep_days instead to avoid duplication

    # New configurable parameters from hardcoded values cleanup

    # General physics parameters
    # NOTE: Use PhysicalConstants.DAYS_PER_YEAR (365.25) as single source of truth
    days_per_year: float = 365.25  # Fixed: was 365.0, now matches PhysicalConstants
    psi_to_pa_conversion: float = (
        6894.76  # Duplicate of PhysicalConstants.PSI_TO_PA, kept for backward compatibility
    )
    fahrenheit_to_kelvin_offset: float = 273.15
    fahrenheit_to_kelvin_scale: float = 5.0 / 9.0

    # CCUS Physics Engine parameters
    # NOTE: Use PhysicalConstants for CO2 properties when possible
    supercritical_co2_density_kg_m3: float = 750.0  # Aligned with PhysicalConstants
    critical_temperature_co2_k: float = (
        304.13  # Duplicate of PhysicalConstants.CO2_CRITICAL_TEMPERATURE_K
    )
    critical_pressure_co2_pa: float = (
        7.376e6  # Duplicate of PhysicalConstants.CO2_CRITICAL_PRESSURE_PA
    )
    molecular_weight_co2_g_mol: float = (
        44.01  # Same as PhysicalConstants.CO2_MOLECULAR_WEIGHT_KG_MOL but in g/mol
    )
    universal_gas_constant_j_mol_k: float = 8.314
    dissolution_time_constant: float = 0.05
    diffusion_coefficient_co2_water: float = 1.9e-9
    henry_constant_co2: float = 3.3e-2
    residual_gas_saturation_trapping: float = 0.05
    solubility_coefficient_co2: float = 0.03
    co2_oil_interfacial_tension: float = 30.0
    gravity_acceleration: float = 9.81

    # Multiphase Flow parameters
    default_oil_viscosity_cp: float = 2.0
    default_co2_viscosity_cp: float = 0.08
    minimum_relative_permeability: float = 0.01
    connate_water_saturation: float = 0.2
    residual_oil_saturation: float = 0.25
    critical_gas_saturation: float = 0.05
    maximum_relative_permeability_oil: float = 0.8
    maximum_relative_permeability_gas: float = 0.4
    maximum_relative_permeability_water: float = 0.3
    endpoint_oil_relative_permeability: float = 0.8
    endpoint_gas_relative_permeability: float = 0.4
    endpoint_water_relative_permeability: float = 0.3

    # Well Control Logic parameters
    injection_setpoint_factor: float = 0.85
    production_setpoint_factor: float = 1.15
    containment_safety_margin_ft: float = 50.0
    bhp_utilization_threshold: float = 0.8
    maintenance_health_threshold: float = 0.7
    max_reduction_factor: float = 0.3
    default_formation_damage_factor: float = 0.8

    # Fault Mechanics parameters
    # NOTE: Use PhysicalConstants.DAYS_PER_YEAR (365.25) as single source of truth
    days_per_year_fault: float = 365.25  # Fixed: was 365.0, now matches PhysicalConstants
    strike_slip_geometry_factor: float = 2.0
    normal_geometry_factor: float = 1.5
    reverse_geometry_factor: float = 1.2
    oblique_geometry_factor: float = 1.8
    default_fault_strike: float = 90.0
    default_fault_dip: float = 0.0
    default_fault_width: float = 100.0
    dummy_fault_strike: float = 90.0
    dummy_fault_dip: float = 0.0
    dummy_fault_center_depth: float = 5000.0
    dummy_fault_length: float = 1000.0
    dummy_fault_width: float = 100.0
    default_fault_cohesion: float = 500.0
    default_fault_friction_coefficient: float = 0.6
    default_fault_dilation_angle: float = 5.0
    default_fault_initial_aperture: float = 0.1
    default_fault_maximum_aperture: float = 10.0
    default_fault_healing_rate: float = 0.01
    default_fault_stiffness: float = 1.0e6
    leakage_transmissibility_threshold: float = 5.0
    baseline_pressure: float = 2000.0
    leakage_failure_threshold: float = 0.6
    leakage_pressure_factor_threshold: float = 1.2
    low_risk_threshold: float = 0.1
    moderate_risk_threshold: float = 0.3
    high_risk_threshold: float = 0.6
    default_leakage_depth: float = 5000.0
    injection_safety_margin: float = 0.8

    # Injection Schemes parameters
    min_wag_cycle_length: int = 30
    initial_wag_cycles: int = 3
    initial_wag_cycle_length: int = 45
    standard_wag_cycle_length: int = 90
    mobility_ratio_factor: float = 0.001
    high_mobility_threshold: float = 2.0
    max_enhanced_wag_ratio: float = 2.0
    wag_ratio_enhancement_factor: float = 1.5
    default_gas_fvf: float = 0.005
    co2_taper_percentage: float = 0.1
    min_co2_taper_factor: float = 0.85
    exponential_decay_factor: float = 3.0
    log_progress_factor: float = 10.0

    # Numerical Stability parameters
    numerical_stability_factor: float = 0.5
    saturation_damping_factor: float = 0.8
    pressure_damping_factor: float = 0.9
    cfl_safety_factor: float = 0.15
    spatial_smoothing_iterations: int = 2
    spatial_smoothing_weight: float = 0.2
    temporal_saturation_damping: float = 0.7
    default_co2_injection_saturation: float = 0.8
    default_water_injection_saturation: float = 0.2
    inlet_transition_length: float = 3.0
    pressure_tolerance: float = 1.0
    max_pressure_iterations: int = 50
    mass_balance_tolerance: float = 1e-4
    rate_smoothing_factor: float = 0.3
    max_rate_change_fraction: float = 0.2

    # Optimization Bounds Defaults
    min_gravity_factor: float = 0.5
    max_gravity_factor: float = 1.5
    min_sor: float = 0.15
    max_sor: float = 0.40
    min_transition_alpha: float = 0.8
    max_transition_alpha: float = 1.2
    min_transition_beta: float = 2.0
    max_transition_beta: float = 10.0
    min_plateau_duration_fraction: float = 0.1
    max_plateau_duration_fraction: float = 0.8
    min_ramp_up_fraction: float = 0.0
    max_ramp_up_fraction: float = 0.3
    min_hyperbolic_b_factor: float = 0.0
    max_hyperbolic_b_factor: float = 1.0
    min_productivity_index: float = 1.0
    max_productivity_index: float = 20.0
    min_wellbore_pressure: float = 100.0
    max_wellbore_pressure: float = 2000.0

    # UI Optimization search bounds for WAG
    min_WAG_ratio: float = 0.1
    max_WAG_ratio: float = 3.0
    min_cycle_length_days: float = 30.0
    max_cycle_length_days: float = 180.0
    min_water_fraction: float = 0.2
    max_water_fraction: float = 0.8

    # Injection distribution parameters for better pressure propagation
    injection_zone_cells: int = 3  # Number of cells in injection zone
    injection_weights: list = field(default_factory=lambda: [0.5, 0.3, 0.2])  # Weight distribution

    # Production well parameters for improved pressure control
    production_drawdown_psi: float = 1000.0  # Reduced from 2000 to more realistic 1000 psi drawdown
    production_index: float = 5.0  # Production productivity index
    min_producer_bhp_psi: float = (
        1500.0  # Minimum producer bottom hole pressure to prevent pump-off
    )
    max_producer_bhp_psi: float = (
        4000.0  # Maximum producer bottom hole pressure for pressure communication
    )

    # Adaptive timestepping control parameters
    # FIXED: Enable adaptive timestepping by default for physical accuracy
    # Performance: Accept ~2-3x slower simulation for physically correct results
    disable_adaptive_timestepping: bool = (
        False  # FIXED: was True - enabling adaptive stepping for CFL enforcement
    )
    numerical_stability_factor: float = 0.5  # FIXED: was 1.0
    cfl_safety_factor: float = 0.15  # FIXED: was 1.0 (enforce CFL)
    saturation_damping_factor: float = 0.8
    pressure_damping_factor: float = 0.9
    max_saturation_change_per_timestep: float = 0.02  # FIXED: was 0.1 (stricter limit)
    max_pressure_change_per_timestep: float = 50.0  # FIXED: was 100.0
    spatial_smoothing_iterations: int = 3
    spatial_smoothing_weight: float = 0.25
    temporal_saturation_damping: float = 0.7

    # Flux limiter type for TVD scheme
    # Options: "van_leer" (mild, more diffusive), "superbee" (sharp, less diffusive)
    # Default: "van_leer" for stability, "superbee" for sharper fronts
    flux_limiter_type: str = "van_leer"

    def __post_init__(self):
        valid_schemes = ["continuous", "huff_n_puff", "swag", "tapered", "pulsed"]
        if self.injection_scheme not in valid_schemes:
            raise ValueError(f"Injection scheme must be one of: {valid_schemes}")
        if self.max_pressure_psi < self.target_pressure_psi:
            raise ValueError("Max Pressure must be >= Target Pressure.")
        if self.max_injection_rate_mscfd <= self.min_injection_rate_mscfd:
            raise ValueError("Max Injection Rate must be > Min Injection Rate.")

        # Validate flux limiter type
        valid_limiters = ["van_leer", "superbee"]
        if self.flux_limiter_type not in valid_limiters:
            raise ValueError(
                f"Flux limiter type must be one of: {valid_limiters}. "
                f"Got: {self.flux_limiter_type}"
            )

        if self.injection_scheme == "huff_n_puff" and self.huff_n_puff is None:
            logging.warning(
                "injection_scheme is 'huff_n_puff' but no HuffNPuffParams provided. Using defaults."
            )
            self.huff_n_puff = HuffNPuffParams()

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)

        if "huff_n_puff" in config and isinstance(config["huff_n_puff"], dict):
            config["huff_n_puff"] = from_dict_to_dataclass(HuffNPuffParams, config["huff_n_puff"])
        if "swag" in config and isinstance(config["swag"], dict):
            config["swag"] = from_dict_to_dataclass(SWAGParams, config["swag"])
        if "tapered" in config and isinstance(config["tapered"], dict):
            config["tapered"] = from_dict_to_dataclass(TaperedInjectionParams, config["tapered"])
        if "pulsed" in config and isinstance(config["pulsed"], dict):
            config["pulsed"] = from_dict_to_dataclass(PulsedInjectionParams, config["pulsed"])

        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class PVTProperties:
    pressure_points: np.ndarray = dataclasses.field(default_factory=lambda: np.array([3000.0]))
    oil_fvf: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.2]))
    oil_viscosity: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.5]))
    gas_fvf: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.005]))
    co2_viscosity: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0.02]))
    rs: np.ndarray = dataclasses.field(default_factory=lambda: np.array([800.0]))
    pvt_type: str = "black_oil"
    gas_specific_gravity: float = 0.7
    temperature: float = 180.0
    api_gravity: Optional[float] = None
    c7_plus_fraction: float = 0.35
    co2_solubility_scm_per_bbl: float = 200.0
    oil_density_ppg: Optional[float] = 7.0
    gas_viscosity_cp: Optional[float] = 0.02
    water_viscosity_cp: Optional[float] = 0.5
    oil_viscosity_cp: Optional[float] = 1.0
    oil_fvf_simple: Optional[float] = 1.2
    water_fvf: Optional[float] = 1.0
    gas_fvf_simple: Optional[float] = 0.01
    oil_compressibility: float = 10e-6
    gas_compressibility: float = 3e-4  # Reduced from 1e-3 to more realistic 3e-4 (1/psi)
    water_compressibility: float = 3e-6
    gas_fvf_rb_per_mscf: float = 5.0

    def __post_init__(self):
        arrays = [
            self.pressure_points,
            self.oil_fvf,
            self.oil_viscosity,
            self.gas_fvf,
            self.co2_viscosity,
            self.rs,
        ]
        non_none_arrays = [arr for arr in arrays if arr is not None and hasattr(arr, "__len__")]
        if non_none_arrays:
            if len({len(arr) for arr in non_none_arrays}) > 1:
                raise ValueError("All provided PVT property arrays must have the same length.")
        if self.pvt_type not in {"black_oil", "compositional"}:
            raise ValueError("pvt_type must be either 'black_oil' or 'compositional'")
        if not (0.5 <= self.gas_specific_gravity <= 1.2):
            raise ValueError(
                f"Gas specific gravity must be between 0.5-1.2, got {self.gas_specific_gravity}"
            )
        if not (50 <= self.temperature <= 400):
            raise ValueError(f"Temperature must be between 50-400°F, got {self.temperature}")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class OperationalParameters:
    project_lifetime_years: int = 15
    time_resolution: str = "yearly"
    target_objective_name: Optional[str] = None
    target_objective_value: Optional[float] = None
    recovery_model_selection: str = "hybrid"
    target_tolerance: float = 0.05

    def __post_init__(self):
        if not (1 <= self.project_lifetime_years <= 100):
            raise ValueError("Project Lifetime must be between 1 and 100 years.")

        valid_resolutions = ["weekly", "monthly", "quarterly", "yearly"]
        if self.time_resolution not in valid_resolutions:
            raise ValueError(f"time_resolution must be one of: {', '.join(valid_resolutions)}")

        valid_targets = [
            None,
            "recovery_factor",
            "npv",
            "co2_utilization",
            "total_co2_stored_tonne",
            "avg_storage_efficiency",
            "final_cumulative_co2_stored_tonne",
            "injection_rate",
            "plume_containment",
            "storage_efficiency",
            "trapping_efficiency",
        ]
        if self.target_objective_name not in valid_targets:
            raise ValueError(f"target_objective_name must be one of {valid_targets}.")
        valid_models = [
            "simple",
            "miscible",
            "immiscible",
            "hybrid",
            "koval",
            "layered",
            "co2_specific",
        ]
        if self.recovery_model_selection not in valid_models:
            raise ValueError(f"recovery_model_selection must be one of: {', '.join(valid_models)}")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class ProfileParameters:
    oil_profile_type: str = "plateau_hyperbolic_decline"
    injection_profile_type: str = "constant_rate"
    plateau_duration_fraction: float = 0.3
    ramp_up_fraction: float = 0.1
    hyperbolic_b_factor: float = 0.5
    min_economic_rate_fraction_of_peak: float = 0.1
    co2_breakthrough_year_fraction: float = 0.25
    co2_production_ratio_after_breakthrough: float = 0.8
    co2_recycling_efficiency_fraction: float = 0.9
    water_cut_exponent: float = 1.5
    custom_oil_production_fractions: Optional[List[float]] = None

    def __post_init__(self):
        valid_oil_profiles = [
            "plateau_linear_decline",
            "plateau_exponential_decline",
            "plateau_hyperbolic_decline",
            "custom_fractions",
        ]
        if self.oil_profile_type not in valid_oil_profiles:
            raise ValueError(f"Invalid oil profile type. Choose from: {valid_oil_profiles}")

        valid_injection_profiles = ["constant_rate", "match_production_decline"]
        if self.injection_profile_type not in valid_injection_profiles:
            raise ValueError(
                f"Invalid injection profile type. Choose from: {valid_injection_profiles}"
            )

        if not (0.0 <= self.plateau_duration_fraction <= 1.0):
            raise ValueError("Plateau duration fraction must be between 0 and 1.")
        if not (0.0 <= self.ramp_up_fraction <= 1.0):
            raise ValueError("Ramp-up fraction must be between 0 and 1.")
        if self.plateau_duration_fraction + self.ramp_up_fraction > 1.0:
            raise ValueError("The sum of plateau and ramp-up fractions cannot exceed 1.0.")
        if not (0.0 <= self.hyperbolic_b_factor <= 2.0):
            raise ValueError("Hyperbolic 'b' factor should realistically be between 0 and 2.")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class GeneticAlgorithmParams:
    num_generations: int = 80
    sol_per_pop: int = 80
    num_parents_mating: int = 10
    parent_selection_type: str = "sss"
    crossover_type: str = "uniform"
    crossover_probability: float = 0.9
    mutation_type: str = "random"
    mutation_probability: float = 0.35
    keep_elitism: int = 1
    num_diverse_solutions_for_bo: int = 15
    diversity_threshold_for_bo: float = 0.20
    stale_generations: int = 10
    use_adaptive_mutation: bool = True
    adaptive_mutation_low: float = 0.4
    adaptive_mutation_high: float = 0.1
    restart_on_stale: bool = False
    restart_diversity_fraction: float = 0.3
    constraint_handling_method: str = "adaptive_penalty"
    penalty_factor: float = 1000.0

    def __post_init__(self):
        if not (10 <= self.sol_per_pop <= 1000):
            raise ValueError("Population Size must be between 10 and 1000.")
        if not (10 <= self.num_generations <= 1000):
            raise ValueError("Generations must be between 10 and 1000.")
        if not (0.0 <= self.crossover_probability <= 1.0):
            raise ValueError("Crossover Probability must be between 0.0 and 1.0.")
        if not (0.0 <= self.mutation_probability <= 1.0):
            raise ValueError("Mutation Probability must be between 0.0 and 1.0.")
        if not (0 <= self.keep_elitism < self.sol_per_pop):
            raise ValueError("Elitism Count must be non-negative and less than Population Size.")
        if not (2 <= self.num_parents_mating < self.sol_per_pop):
            raise ValueError("Number of Mating Parents must be between 2 and Population Size.")
        if not (1 <= self.stale_generations <= 100):
            raise ValueError("Stale Generations must be between 1 and 100.")
        if not (0.0 <= self.adaptive_mutation_low <= 1.0):
            raise ValueError("Adaptive Mutation Low must be between 0.0 and 1.0.")
        if not (0.0 <= self.adaptive_mutation_high <= 1.0):
            raise ValueError("Adaptive Mutation High must be between 0.0 and 1.0.")
        if not (0.0 <= self.restart_diversity_fraction <= 1.0):
            raise ValueError("Restart Diversity Fraction must be between 0.0 and 1.0.")
        if self.constraint_handling_method not in ["static", "adaptive", "death", "adaptive_penalty"]:
            raise ValueError("Constraint Handling Method must be 'static', 'adaptive', or 'death'.")
        if self.penalty_factor < 0:
            raise ValueError("Penalty Factor must be non-negative.")

    @classmethod
    def from_config_dict(cls, config_ga_params_dict: Dict[str, Any], **kwargs):
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
        }
        defaults.update(config_ga_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)


@dataclasses.dataclass
class BayesianOptimizationParams:
    n_iterations: int = 100
    n_initial_points: int = 10
    acquisition_function: str = "ucb"
    acq_kappa: float = 5.0
    acq_xi: float = 0.01
    use_trust_region: bool = True
    trust_region_initial_size: float = 1.0
    trust_region_min_size: float = 0.001
    trust_region_decay: float = 0.5

    def __post_init__(self):
        if not (5 <= self.n_iterations <= 1000):
            raise ValueError("BO: Number of iterations must be between 5 and 1000.")
        if not (0.0 < self.trust_region_initial_size <= 1.0):
            raise ValueError("Trust region initial size must be between 0.0 (exclusive) and 1.0.")
        if not (0.0 < self.trust_region_min_size < self.trust_region_initial_size):
            raise ValueError("Trust region min size must be positive and less than initial size.")
        if not (0.0 < self.trust_region_decay < 1.0):
            raise ValueError("Trust region decay factor must be between 0.0 and 1.0.")
        if not (0 <= self.n_initial_points <= 100):
            raise ValueError("BO: Number of initial random points must be between 0 and 100.")
        if self.acquisition_function not in ["ucb", "ei", "poi"]:
            raise ValueError("Acquisition function must be one of 'ucb', 'ei', or 'poi'.")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class ParticleSwarmParams:
    n_particles: int = 40
    iters: int = 100
    c1: float = 0.5
    c2: float = 0.3
    w: float = 0.9

    def __post_init__(self):
        if not (10 <= self.n_particles <= 1000):
            raise ValueError("Number of particles must be between 10 and 1000.")
        if not (10 <= self.iters <= 1000):
            raise ValueError("Number of iterations must be between 10 and 1000.")
        if not (0.0 <= self.c1 <= 4.0):
            raise ValueError("Cognitive parameter (c1) should be between 0.0 and 4.0.")
        if not (0.0 <= self.c2 <= 4.0):
            raise ValueError("Social parameter (c2) should be between 0.0 and 4.0.")
        if not (0.0 <= self.w <= 1.0):
            raise ValueError("Inertia weight (w) must be between 0.0 and 1.0.")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class DifferentialEvolutionParams:
    strategy: str = "best1bin"
    maxiter: int = 100
    popsize: int = 15
    mutation: float = 0.7
    recombination: float = 0.7

    def __post_init__(self):
        valid_strategies = [
            "best1bin",
            "best1exp",
            "rand1exp",
            "randtobest1exp",
            "currenttobest1exp",
            "best2exp",
            "rand2exp",
            "randtobest1bin",
            "currenttobest1bin",
            "best2bin",
            "rand2bin",
            "rand1bin",
        ]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid DE strategy. Choose from: {valid_strategies}")
        if not (1 <= self.popsize <= 100):
            raise ValueError("Population size multiplier must be between 1 and 100.")
        if not (10 <= self.maxiter <= 1000):
            raise ValueError("Max iterations must be between 10 and 1000.")
        if not (0.0 <= self.mutation <= 2.0):
            raise ValueError("Mutation factor must be between 0.0 and 2.0.")
        if not (0.0 <= self.recombination <= 1.0):
            raise ValueError("Recombination probability must be between 0.0 and 1.0.")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        if "popsize_multiplier" in config:
            config["popsize"] = config.pop("popsize_multiplier")
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class GeomechanicsParameters:
    youngs_modulus: float = 1.0e6
    poissons_ratio: float = 0.25
    initial_pore_pressure: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    initial_vertical_stress: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    biot_coefficient: float = 0.8
    initial_horizontal_stress_ratio: float = 0.7
    pore_compressibility: float = 1e-6
    permeability_stress_coefficient: float = 1e-5
    rock_density_gradient: float = dataclasses.field(
        default_factory=lambda: PhysicalConstants().ROCK_DENSITY_GRADIENT
    )
    rock_cohesion: float = 1000.0
    rock_friction_angle: float = 30.0
    rock_tensile_strength: float = 500.0
    fixed_displacement_boundaries: Dict[str, Any] = dataclasses.field(default_factory=dict)
    traction_boundaries: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TuningParams:
    tuner_method: str = "bayesian"
    num_tuning_iterations: int = 50
    num_evaluation_generations: int = 20

    def __post_init__(self):
        if self.tuner_method not in ["bayesian", "grid", "random"]:
            raise ValueError("Tuner method must be 'bayesian', 'grid', or 'random'.")
        if not (10 <= self.num_tuning_iterations <= 500):
            raise ValueError("Number of tuning iterations must be between 10 and 500.")
        if not (5 <= self.num_evaluation_generations <= 100):
            raise ValueError(
                "Number of evaluation generations for each trial must be between 5 and 100."
            )

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


class TimestepUnit(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclasses.dataclass
class CCUSParameters:
    # Timestepping - Default to monthly for better performance
    timestep_unit: TimestepUnit = TimestepUnit.MONTHLY
    max_timestep_days: float = 90.0  # Allow up to quarterly for adaptive stepping
    min_timestep_days: float = 7.0  # Minimum 1 week for stability
    timestep_days: float = 30.44  # Monthly timestep (average days per month)

    # Fault mechanics control - disabled by default
    enable_fault_mechanics: bool = False  # Only enable when fault data is provided
    max_timestep_days: float = 90.0  # Allow up to quarterly for adaptive stepping
    min_timestep_days: float = 7.0  # Minimum 1 week for stability
    timestep_days: float = 30.44  # Monthly timestep (average days per month)

    # Solver tolerances
    tolerance_pressure: float = 1e-3
    tolerance_saturation: float = 1e-4
    max_nonlinear_iterations: int = 20

    # Geomechanics
    biot_coefficient: float = 0.8
    youngs_modulus: float = 1.0e6
    poissons_ratio: float = 0.25
    permeability_stress_coefficient: float = 1.0e-5
    pore_compressibility: float = 1e-6

    # Faults
    fault_cohesion: float = 100.0
    fault_friction_coefficient: float = 0.6
    fault_slip_transmissibility_multiplier: float = 10.0
    fault_mechanics_update_frequency: int = 30
    fault_stability: Dict = dataclasses.field(default_factory=dict)
    tectonic_shear_ratio: float = 0.4  # Differential horizontal stress for fault activation

    # Kinetics
    dissolution_rate_constant: float = 1.0e-7
    mineralization_rate_constant: float = 1.0e-9
    reactive_surface_area: float = 100.0

    # Well and Reservoir
    temperature: float = 150.0
    max_injector_bhp_psi: float = 8000.0  # Reduced from 12000 to prevent excessive pressure buildup

    # Storage
    storage_capacity_tonne: float = 1_000_000.0
    storage_efficiency_factor: float = 0.7
    leakage_rate_fraction: float = 0.01
    monitoring_cost_usd_per_tonne: float = 5.0
    storage_credit_usd_per_tonne: float = 50.0
    min_trapping_efficiency: float = 0.8
    reservoir_seal_integrity_factor: float = 0.9
    solubility_trapping_factor: float = 0.3
    mineral_trapping_factor: float = 0.1
    residual_trapping_factor: float = 0.4
    max_plume_migration_distance_ft: float = 3000.0
    min_injection_pressure_psi: float = 1000.0
    max_injection_pressure_frac: float = 0.8
    plume_containment_safety_factor: float = 1.2
    structural_trapping_factor: float = 0.2
    # Fixed: aligned with EORParameters value of 750.0 (typical for reservoir conditions)
    supercritical_co2_density_kg_m3: float = 750.0
    structural_trapping_depth_threshold_m: float = 800.0
    residual_gas_saturation_trapping: float = 0.3
    fault_leakage_factor: float = 0.01


@dataclasses.dataclass
class CO2StorageParameters:
    """Parameters for CO2 storage calculations"""

    structural_trapping_factor: float = 0.2
    residual_trapping_factor: float = 0.4
    solubility_trapping_factor: float = 0.3
    mineral_trapping_factor: float = 0.1
    storage_efficiency_factor: float = 0.7
    leakage_rate_fraction: float = 0.01
    max_plume_migration_distance_ft: float = 3000.0
    min_trapping_efficiency: float = 0.8
    reservoir_seal_integrity_factor: float = 0.9
    monitoring_cost_usd_per_tonne: float = 5.0
    max_injection_pressure_frac: float = 0.8
    plume_containment_safety_factor: float = 1.2

    def __post_init__(self):
        if not (0.0 <= self.structural_trapping_factor <= 1.0):
            raise ValueError("Structural trapping factor must be between 0 and 1")
        if not (0.0 <= self.residual_trapping_factor <= 1.0):
            raise ValueError("Residual trapping factor must be between 0 and 1")
        if not (0.0 <= self.solubility_trapping_factor <= 1.0):
            raise ValueError("Solubility trapping factor must be between 0 and 1")
        if not (0.0 <= self.mineral_trapping_factor <= 1.0):
            raise ValueError("Mineral trapping factor must be between 0 and 1")
        if not (0.0 <= self.storage_efficiency_factor <= 1.0):
            raise ValueError("Storage efficiency factor must be between 0 and 1")
        if not (0.0 <= self.leakage_rate_fraction <= 1.0):
            raise ValueError("Leakage rate fraction must be between 0 and 1")
        if self.max_plume_migration_distance_ft <= 0:
            raise ValueError("Max plume migration distance must be positive")
        if not (0.0 <= self.min_trapping_efficiency <= 1.0):
            raise ValueError("Min trapping efficiency must be between 0 and 1")
        if not (0.0 <= self.reservoir_seal_integrity_factor <= 1.0):
            raise ValueError("Reservoir seal integrity factor must be between 0 and 1")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class AdvancedEngineParams:
    relaxable_constraint_range_factors: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "porosity": 0.20,
            "ooip_stb": 0.25,
            "v_dp_coefficient": 0.35,
            "mobility_ratio": 0.50,
            "WAG_ratio": 0.60,
            "gravity_factor": 0.75,
            "sor": 0.20,
            "transition_alpha": 0.15,
            "transition_beta": 0.50,
        }
    )
    default_porosity: float = 0.15
    default_permeability: float = 100.0
    npv_tiebreaker_scaling_factor: float = 1e-12
    sensitivity_plot_range_multiplier: float = 0.20
    failure_penalty: float = -1e12
    breakthrough_fallback_time_years: float = 5.0
    breakthrough_fallback_impact_factor: float = 1.0
    breakthrough_fallback_penalty: float = 0.0
    fracture_pressure_multiplier: float = 1.5
    pressure_control_min_scaling_factor: float = 0.01
    use_simple_physics: bool = True  # DEPRECATED: Kept for backward compatibility
    engine_type: str = "surrogate"  # New: Engine type - "simple", "detailed", or "surrogate"
    recovery_model_type: str = "hybrid"  # New: Recovery model for surrogate engine ("hybrid", "phd_hybrid", etc.)

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)


@dataclasses.dataclass
class CCUSState:
    """State variables for the coupled CCUS physics engine"""

    pressure: np.ndarray
    saturations: np.ndarray
    compositions: np.ndarray
    current_time: float
    timestep: float
    porosity: np.ndarray
    permeability: np.ndarray
    stress: np.ndarray
    fault_transmissibility: np.ndarray
    dissolved_co2: np.ndarray
    mineral_precipitate: np.ndarray
    fluxes: np.ndarray = None
    fault_stability: Optional[Dict] = None
    injection_rates: Optional[Dict] = None

    def __post_init__(self):
        n_cells = len(self.pressure)
        assert self.saturations.shape == (n_cells, 3), "Saturations must have shape (n_cells, 3)"
        assert np.allclose(np.sum(self.saturations, axis=1), 1.0), "Saturations must sum to 1"
        assert np.all(self.pressure > 0), "Pressure must be positive"
        assert np.all(self.porosity > 0) and np.all(self.porosity <= 1), "Porosity must be in (0,1]"
        if self.fluxes is None:
            self.fluxes = np.zeros((0, 3))

    def get_tensor(self, cell_idx: int) -> np.ndarray:
        if cell_idx < 0 or cell_idx >= len(self.pressure):
            raise IndexError(f"Cell index {cell_idx} out of range")
        return np.array(
            [
                [
                    self.stress[cell_idx, 0],
                    self.stress[cell_idx, 3],
                    self.stress[cell_idx, 4],
                ],
                [
                    self.stress[cell_idx, 3],
                    self.stress[cell_idx, 1],
                    self.stress[cell_idx, 5],
                ],
                [
                    self.stress[cell_idx, 4],
                    self.stress[cell_idx, 5],
                    self.stress[cell_idx, 2],
                ],
            ]
        )


# =============================================================================
# CORE REPAIR: Physical Constants (Phase 1)
# =============================================================================


@dataclasses.dataclass(slots=True)
class PhysicalConstants:
    """
    Centralized physical constants for CO2 EOR calculations.

    This class serves as the single source of truth for all physical constants
    used throughout the application. All files should import from here instead
    of hardcoding values.

    Values are based on NIST reference data and industry standards.
    """

    # CO2 Critical Properties (NIST Reference)
    CO2_CRITICAL_PRESSURE_PA: float = 7.376e6  # Pa (73.76 bar)
    CO2_CRITICAL_TEMPERATURE_K: float = 304.13  # Kelvin
    CO2_MOLECULAR_WEIGHT_KG_MOL: float = 0.04401  # kg/mol
    CO2_acentric_factor: float = 0.225

    # =========================================================================
    # Unit Conversions
    # =========================================================================
    PSI_TO_PA: float = 6894.76
    PA_TO_PSI: float = 1.0 / 6894.76
    BAR_TO_PA: float = 100000.0
    MPA_TO_PA: float = 1000000.0

    # Permeability conversions
    MD_TO_M2: float = 9.869233e-16  # milliDarcy to m^2
    M2_TO_MD: float = 1.0 / 9.869233e-16
    MD_TO_CM2: float = 9.869233e-13  # milliDarcy to cm^2 (1 mD = 9.869e-13 cm^2)

    # Time conversions
    SECONDS_PER_DAY: float = 86400.0
    DAYS_PER_YEAR: float = 365.25
    SECONDS_PER_HOUR: float = 3600.0

    # Volume conversions
    BBLS_TO_M3: float = 0.158987  # US oil barrel to m³
    STB_TO_M3: float = 0.158987  # Stock Tank Barrel (equivalent to US oil barrel)
    SCF_TO_M3: float = 0.0283168  # Standard Cubic Foot to m³ (at 60°F, 14.7 psi)
    MSCF_TO_M3: float = 28.3168  # Thousand Standard Cubic Feet to m³
    MCF_TO_M3: float = 28.3168  # Thousand Cubic Feet to m³ (alias)
    FT3_TO_M3: float = 0.0283168  # Cubic foot to m³
    M3_TO_BBL: float = 1.0 / 0.158987  # m³ to barrels
    M3_TO_STB: float = 1.0 / 0.158987  # m³ to STB
    M3_TO_SCF: float = 1.0 / 0.0283168  # m³ to SCF
    FT3_TO_BBL: float = 0.1781076  # ft³ to bbl (0.0283168/0.158987)
    BBL_TO_FT3: float = 1.0 / 0.1781076  # bbl to ft³

    # CO2 density at standard conditions (60°F, 14.7 psi)
    # CO2 molecular weight = 44.01 g/mol
    # At standard conditions: density = MW / molar_volume = 0.04401 kg/mol / 0.02369 m³/mol = 1.857 kg/m³
    # CO2 molar volume at API standard (60°F, 14.696 psi): Vm = 10.7316 × 519.67 / 14.696 = 379.48 ft³/lb-mol ≈ 0.02369 m³/mol
    # Convert to tonnes/MSCF: 1.857 kg/m³ * 28.3168 m³/MSCF / 1000 kg/tonne = 0.05254 tonnes/MSCF
    CO2_DENSITY_TONNE_PER_MSCF: float = (
        0.05254  # tonnes per thousand standard cubic feet at 60°F, 14.696 psi
    )

    # Mass conversions
    KG_TO_TONNES: float = 1.0 / 1000.0  # kg to tonnes
    TONNES_TO_KG: float = 1000.0  # tonnes to kg

    # Length conversions
    FT_TO_M: float = 0.3048
    M_TO_FT: float = 1.0 / 0.3048
    INCH_TO_M: float = 0.0254
    # Area conversions
    M2_TO_FT2: float = 10.7639  # m² to ft² (1 m² = 10.7639 ft²)
    FT2_TO_M2: float = 1.0 / 10.7639  # ft² to m²

    # Viscosity conversions
    VISC_CP_TO_PA_S: float = 0.001  # centipoise to Pa·s
    VISC_CP_TO_POISE: float = 0.01  # centipoise to Poise
    VISC_PA_S_TO_CP: float = 1000.0  # Pa·s to centipoise

    # =========================================================================
    # Numerical Constants (tolerances and small value thresholds)
    # =========================================================================
    NUMERICAL_EPSILON_DEFAULT: float = 1e-9
    NUMERICAL_EPSILON_MACRO: float = 1e-6     # 0.0001% tolerance
    NUMERICAL_EPSILON_MICRO: float = 1e-8     # 0.000001% tolerance
    NUMERICAL_EPSILON_ULTRA: float = 1e-12     # Machine precision

    # =========================================================================
    # Physics Constants
    # =========================================================================
    GRAVITY_M_S2: float = 9.81  # m/s^2
    GRAVITY_FT_S2: float = 32.174  # ft/s^2
    GAS_CONSTANT_J_MOL_K: float = 8.314  # J/(mol·K)

    # Standard Conditions (Industry Standard: 60°F at 14.7 psi)
    # Oilfield standard: 60°F (288.706K) at 14.696 psi (101325 Pa)
    # Some petroleum engineering references use 59°F (15°C = 288.15K)
    STANDARD_TEMPERATURE_K: float = 288.706  # 60°F (15.56°C) - Oilfield standard
    STANDARD_PRESSURE_PA: float = 101325.0  # 1 atm = 14.696 psi
    # Ideal gas molar volume at standard conditions
    # V = RT/P = 8.314 * 288.706 / 101325 = 0.02369 m³/mol
    IDEAL_GAS_MOLAR_VOLUME_M3_PER_MOL: float = 0.02369  # m³/mol at 60°F, 14.7 psi

    # Reference densities
    WATER_DENSITY_REF_KG_M3: float = 1000.0
    OIL_DENSITY_REF_KG_M3: float = 850.0
    GAS_DENSITY_REF_KG_M3: float = 1.0

    # =========================================================================
    # Reservoir Engineering Constants
    # =========================================================================

    # Pressure gradients (psi/ft) - typical values for different formations
    # NOTE: These are in psi/ft for display/input. Convert to SI (Pa/m) for internal calculations.
    # To convert psi/ft to Pa/m: multiply by PSI_TO_PA / FT_TO_M = 6894.76 / 0.3048 = 22620.6
    ROCK_DENSITY_GRADIENT_PA_M: float = (
        0.434 * 22620.6
    )  # 9817 Pa/m = ~0.98 kPa/m (converted from psi/ft)
    PRESSURE_GRADIENT_NORMAL_PA_M: float = (
        0.465 * 22620.6
    )  # 10519 Pa/m = ~10.5 kPa/m (normal hydrostatic)
    PRESSURE_GRADIENT_TYPICAL_PA_M: float = (
        0.45 * 22620.6
    )  # 10179 Pa/m = ~10.2 kPa/m (commonly used)
    STRESS_GRADIENT_LITHOSTATIC_PA_M: float = 1.0 * 22620.6  # 22621 Pa/m = ~22.6 kPa/m
    FRACTURE_GRADIENT_SANDSTONE_PA_M: float = 0.7 * 22620.6  # 15834 Pa/m = ~15.8 kPa/m
    FRACTURE_GRADIENT_MAX_PA_M: float = 1.0 * 22620.6  # 22621 Pa/m = ~22.6 kPa/m

    # Legacy psi/ft values maintained for backward compatibility
    ROCK_DENSITY_GRADIENT: float = 0.434  # psi/ft - typical overburden gradient
    PRESSURE_GRADIENT_NORMAL: float = 0.465  # psi/ft - normal hydrostatic
    PRESSURE_GRADIENT_TYPICAL: float = 0.45  # psi/ft - commonly used
    STRESS_GRADIENT_LITHOSTATIC: float = 1.0  # psi/ft - lithostatic
    FRACTURE_GRADIENT_SANDSTONE: float = 0.7  # psi/ft - typical
    FRACTURE_GRADIENT_MAX: float = 1.0  # psi/ft - maximum typical

    # Viscosity defaults (centipoise)
    DEFAULT_OIL_VISCOSITY_CP: float = 2.0
    DEFAULT_CO2_VISCOSITY_CP: float = 0.08
    DEFAULT_WATER_VISCOSITY_CP: float = 0.5

    # =========================================================================
    # Reservoir Defaults
    # =========================================================================
    DEFAULT_SW_CONNATE: float = 0.20
    DEFAULT_SOR: float = 0.20
    DEFAULT_SGR: float = 0.05
    DEFAULT_POROSITY: float = 0.15
    DEFAULT_PERMEABILITY_MD: float = 100.0
    DEFAULT_INITIAL_PRESSURE_PSI: float = 4000.0
    DEFAULT_TEMPERATURE_F: float = 150.0

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    @classmethod
    def get_co2_properties(cls) -> Dict[str, float]:
        """Return CO2 properties as a dictionary for convenience."""
        return {
            "Pc": cls.CO2_CRITICAL_PRESSURE_PA,
            "Tc": cls.CO2_CRITICAL_TEMPERATURE_K,
            "Mw": cls.CO2_MOLECULAR_WEIGHT_KG_MOL,
            "omega": cls.CO2_acentric_factor,
        }

    @classmethod
    def get_unit_conversions(cls) -> Dict[str, float]:
        """Return unit conversion factors as a dictionary."""
        return {
            "psi_to_pa": cls.PSI_TO_PA,
            "pa_to_psi": cls.PA_TO_PSI,
            "md_to_m2": cls.MD_TO_M2,
            "m2_to_md": cls.M2_TO_MD,
            "md_to_cm2": cls.MD_TO_CM2,
            "sec_per_day": cls.SECONDS_PER_DAY,
            "days_per_year": cls.DAYS_PER_YEAR,
            "bbls_to_m3": cls.BBLS_TO_M3,
            "stb_to_m3": cls.STB_TO_M3,
            "scf_to_m3": cls.SCF_TO_M3,
            "mscf_to_m3": cls.MSCF_TO_M3,
            "mcf_to_m3": cls.MCF_TO_M3,
            "m3_to_bbl": cls.M3_TO_BBL,
            "m3_to_stb": cls.M3_TO_STB,
            "m3_to_scf": cls.M3_TO_SCF,
            "ft_to_m": cls.FT_TO_M,
            "m_to_ft": cls.M_TO_FT,
            "ft3_to_m3": cls.FT3_TO_M3,
            "ft3_to_bbl": cls.FT3_TO_BBL,
            "bbl_to_ft3": cls.BBL_TO_FT3,
            "visc_cp_to_pa_s": cls.VISC_CP_TO_PA_S,
            "visc_cp_to_poise": cls.VISC_CP_TO_POISE,
            "gravity_ft_s2": cls.GRAVITY_FT_S2,
            "co2_density_tonne_per_mscf": cls.CO2_DENSITY_TONNE_PER_MSCF,
        }

    @classmethod
    def get_pressure_gradients(cls) -> Dict[str, float]:
        """Return pressure gradient constants with descriptions."""
        return {
            "rock_density": cls.ROCK_DENSITY_GRADIENT,
            "normal_hydrostatic": cls.PRESSURE_GRADIENT_NORMAL,
            "typical": cls.PRESSURE_GRADIENT_TYPICAL,
            "lithostatic": cls.STRESS_GRADIENT_LITHOSTATIC,
            "fracture_sandstone": cls.FRACTURE_GRADIENT_SANDSTONE,
            "fracture_max": cls.FRACTURE_GRADIENT_MAX,
        }


_PHYS_CONSTANTS = PhysicalConstants()


# =============================================================================
# CORE REPAIR: Corey Parameters (Phase 4)
# =============================================================================


@dataclasses.dataclass(slots=True)
class CoreyParameters:
    """
    Corey relative permeability parameters.

    These parameters define the relative permeability curves for water, oil,
    and gas phases using the Corey model.
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
        """Validate Corey parameters."""
        if not 0 < self.krw0 <= 1:
            raise ValueError(f"krw0 must be between 0 and 1, got {self.krw0}")
        if not 0 < self.kro0 <= 1:
            raise ValueError(f"kro0 must be between 0 and 1, got {self.kro0}")
        if not 0 < self.krg0 <= 1:
            raise ValueError(f"krg0 must be between 0 and 1, got {self.krg0}")
        if self.nw <= 0:
            raise ValueError(f"nw must be positive, got {self.nw}")
        if self.no <= 0:
            raise ValueError(f"no must be positive, got {self.no}")
        if self.ng <= 0:
            raise ValueError(f"ng must be positive, got {self.ng}")
        if not 0 <= self.swi <= self.sor:
            raise ValueError(f"swi ({self.swi}) must be <= sor ({self.sor})")
        if not self.sor <= 1 - self.sgr:
            raise ValueError(f"sor ({self.sor}) must be <= 1-sgr ({1 - self.sgr})")

    def kr_water(self, sw: np.ndarray) -> np.ndarray:
        """Calculate water relative permeability using Corey model."""
        sw_normalized = np.clip((sw - self.swi) / (1 - self.sor - self.swi), 0, 1)
        return self.krw0 * sw_normalized**self.nw

    def kr_gas(self, sg: np.ndarray) -> np.ndarray:
        """Calculate gas relative permeability using Corey model."""
        sg_normalized = np.clip((sg - self.sgr) / (1 - self.swi - self.sor - self.sgr), 0, 1)
        return self.krg0 * sg_normalized**self.ng


# =============================================================================
# CORE REPAIR: Grid Representations (Phase 2)
# =============================================================================


class GridType(Enum):
    """Enum for grid types."""

    SIMPLE = "simple"
    FULL_PHYSICS = "full_physics"


@dataclasses.dataclass(slots=True)
class GridBase:
    """
    Base class for all grid representations.
    """

    n_cells: int
    dimensions: Tuple[int, int, int]
    cell_volumes: np.ndarray
    grid_type: GridType = GridType.SIMPLE

    def __post_init__(self):
        """Validate grid base parameters."""
        if self.n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {self.n_cells}")
        if len(self.dimensions) != 3:
            raise ValueError("dimensions must be a tuple of 3 integers")
        if any(d <= 0 for d in self.dimensions):
            raise ValueError("all dimensions must be positive")


@dataclasses.dataclass
class SimpleGrid:
    """
    Simple 3D Cartesian grid for basic simulations.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    n_cells: int = field(init=False)
    dimensions: Tuple[int, int, int] = field(init=False)
    cell_volumes: np.ndarray = field(init=False)
    grid_type: GridType = GridType.SIMPLE

    def __init__(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.n_cells = nx * ny * nz
        self.dimensions = (nx, ny, nz)
        self.cell_volumes = np.full(self.n_cells, dx * dy * dz)

    def __post_init__(self):
        """Validate grid parameters."""
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("Cell sizes must be positive")

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.n_cells

    @property
    def total_volume(self) -> float:
        """Total grid volume in m^3."""
        return self.n_cells * self.dx * self.dy * self.dz

    @property
    def physical_dimensions(self) -> Tuple[float, float, float]:
        """
        Physical dimensions of the grid in meters (length_x, length_y, length_z).

        Returns the actual physical size of the reservoir, computed as:
        - length_x = nx * dx
        - length_y = ny * dy
        - length_z = nz * dz
        """
        return (self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)

    @property
    def area(self) -> float:
        """Surface area of the grid in m² (length_x * length_y)."""
        return self.nx * self.dx * self.ny * self.dy


@dataclasses.dataclass
class FullPhysicsGrid:
    """
    Full physics grid with fault support.
    """

    dimensions: Tuple[int, int, int]
    cell_volumes: np.ndarray
    dx_array: np.ndarray
    dy_array: np.ndarray
    dz_array: np.ndarray
    depth: np.ndarray
    tops: np.ndarray
    fault_cells: List[List[int]] = field(default_factory=list)
    fault_orientations: List[str] = field(default_factory=list)
    n_cells: int = field(init=False)
    grid_type: GridType = GridType.FULL_PHYSICS

    def __init__(
        self,
        dimensions: Tuple[int, int, int],
        cell_volumes: np.ndarray,
        dx_array: np.ndarray,
        dy_array: np.ndarray,
        dz_array: np.ndarray,
        depth: np.ndarray,
        tops: np.ndarray,
        fault_cells: List[List[int]] = None,
        fault_orientations: List[str] = None,
    ):
        self.dimensions = dimensions
        self.cell_volumes = cell_volumes
        self.dx_array = dx_array
        self.dy_array = dy_array
        self.dz_array = dz_array
        self.depth = depth
        self.tops = tops
        self.fault_cells = fault_cells or []
        self.fault_orientations = fault_orientations or []
        self.n_cells = dimensions[0] * dimensions[1] * dimensions[2]

    @property
    def has_faults(self) -> bool:
        """Check if grid has fault definitions."""
        return len(self.fault_cells) > 0


# =============================================================================
# CORE REPAIR: ReservoirState Legacy Adapter (Phase 3)
# =============================================================================


@dataclasses.dataclass
class ReservoirState:
    """
    Legacy state class for backward compatibility with simple engine.

    DEPRECATED: Use CCUSState from data_models.py instead.
    This class is maintained for backward compatibility.
    """

    pressure: np.ndarray
    water_saturation: np.ndarray
    oil_saturation: np.ndarray
    gas_saturation: np.ndarray
    temperature: float = 353.15  # Kelvin
    time: float = 0.0

    def __post_init__(self):
        """Validate state."""
        n_cells = len(self.pressure)

        if len(self.water_saturation) != n_cells:
            raise ValueError("water_saturation length must match pressure")
        if len(self.oil_saturation) != n_cells:
            raise ValueError("oil_saturation length must match pressure")
        if len(self.gas_saturation) != n_cells:
            raise ValueError("gas_saturation length must match pressure")

        total_sat = self.water_saturation + self.oil_saturation + self.gas_saturation
        if not np.allclose(total_sat, 1.0, atol=1e-6):
            raise ValueError("Saturation sum must equal 1.0")

    @classmethod
    def create_initial_state(
        cls,
        grid,
        initial_pressure: float,
        initial_water_sat: float,
        temperature: float = 353.15
    ) -> "ReservoirState":
        """
        Create initial reservoir state.

        Parameters:
        -----------
        grid : SimpleGrid
            Grid object
        initial_pressure : float
            Initial pressure (Pa)
        initial_water_sat : float
            Initial water saturation (fraction)
        temperature : float
            Reservoir temperature (K)

        Returns:
        --------
        ReservoirState : Initial reservoir state

        Note:
        ------
        Arrays are created with shape (nz, ny, nx) to match the indexing
        convention used in reservoir_engine.py where arrays are accessed as [k, j, i].
        """
        # Create 3D arrays with shape (nz, ny, nx) for [k, j, i] indexing
        # This matches the indexing convention in reservoir_engine.py
        water_saturation = np.full((grid.nz, grid.ny, grid.nx), initial_water_sat)
        oil_saturation = np.full((grid.nz, grid.ny, grid.nx), 1.0 - initial_water_sat)
        gas_saturation = np.zeros((grid.nz, grid.ny, grid.nx))
        pressure = np.full((grid.nz, grid.ny, grid.nx), initial_pressure)

        return cls(
            pressure=pressure,
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            temperature=temperature,
            time=0.0
        )

    @classmethod
    def from_ccus_state(cls, ccus_state: CCUSState) -> "ReservoirState":
        """Convert from CCUSState to legacy ReservoirState."""
        return cls(
            pressure=ccus_state.pressure,
            water_saturation=ccus_state.saturations[:, 0],
            oil_saturation=ccus_state.saturations[:, 1],
            gas_saturation=ccus_state.saturations[:, 2],
            temperature=353.15,
            time=ccus_state.current_time,
        )

    def to_ccus_state(self) -> CCUSState:
        """Convert legacy ReservoirState to CCUSState."""
        n_cells = len(self.pressure)
        saturations = np.column_stack(
            [self.water_saturation, self.oil_saturation, self.gas_saturation]
        )

        return CCUSState(
            pressure=self.pressure,
            saturations=saturations,
            compositions=np.zeros((n_cells, 3)),
            current_time=self.time,
            timestep=0.0,
            porosity=np.full(n_cells, _PHYS_CONSTANTS.DEFAULT_POROSITY),
            permeability=np.full(n_cells, _PHYS_CONSTANTS.DEFAULT_PERMEABILITY_MD),
            stress=np.zeros((n_cells, 6)),
            fault_transmissibility=np.ones(1),
            dissolved_co2=np.zeros(n_cells),
            mineral_precipitate=np.zeros(n_cells),
        )

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.pressure)


@dataclasses.dataclass
class EmpiricalFittingParameters:
    """
    Fitting/calibration parameters for empirical surrogate engine validation.

    These parameters allow users to calibrate the surrogate model to match
    specific reservoir cases (e.g., SPE5, CMG validation, field data).

    The parameters are used in the PhDHybridSurrogate model to control:
    - Miscibility transition behavior
    - Production dynamics (breakthrough, trapping)
    - Fluid composition effects
    - Relative permeability and Corey exponents
    - Todd-Longstaff mixing
    """
    # Fluid composition
    c7_plus_fraction: float = 0.57  # C7+ fraction (0.0-1.0)

    # Miscibility transition parameters
    alpha_base: float = 1.0  # Transition midpoint (dimensionless)
    miscibility_window: float = 0.011  # Beta value controlling transition sharpness

    # Production dynamics
    breakthrough_time_years: float = 1.5  # Time to CO2 breakthrough
    trapping_efficiency: float = 0.4  # Fraction of injected CO2 trapped (0.0-1.0)

    # Initial conditions
    initial_gor_scf_per_stb: float = 500.0  # Initial gas-oil ratio

    # Mobility and mixing
    transverse_mixing_calibration: float = 0.5  # PhD heterogeneity calibration
    omega_tl: float = 0.6  # Todd-Longstaff mixing parameter (0-1)

    # Relative permeability endpoints (Corey parameters)
    k_ro_0: float = 0.8  # Oil rel perm endpoint
    k_rg_0: float = 1.0  # Gas rel perm endpoint
    n_o: float = 2.0  # Oil Corey exponent
    n_g: float = 2.0  # Gas Corey exponent

    def __post_init__(self):
        """Validate empirical fitting parameters."""
        # Fluid composition
        if not (0.0 <= self.c7_plus_fraction <= 1.0):
            raise ValueError(f"C7+ fraction must be between 0 and 1, got {self.c7_plus_fraction}")

        # Miscibility transition
        if self.alpha_base <= 0:
            raise ValueError(f"alpha_base must be positive, got {self.alpha_base}")
        if not (0.001 <= self.miscibility_window <= 0.1):
            raise ValueError(f"miscibility_window must be between 0.001 and 0.1, got {self.miscibility_window}")

        # Production dynamics
        if not (0.1 <= self.breakthrough_time_years <= 10.0):
            raise ValueError(f"breakthrough_time_years must be between 0.1 and 10, got {self.breakthrough_time_years}")
        if not (0.0 <= self.trapping_efficiency <= 1.0):
            raise ValueError(f"trapping_efficiency must be between 0 and 1, got {self.trapping_efficiency}")

        # Initial conditions
        if self.initial_gor_scf_per_stb < 0:
            raise ValueError(f"initial_gor_scf_per_stb must be non-negative, got {self.initial_gor_scf_per_stb}")

        # Mobility and mixing
        if not (0.0 <= self.transverse_mixing_calibration <= 1.0):
            raise ValueError(f"transverse_mixing_calibration must be between 0 and 1, got {self.transverse_mixing_calibration}")
        if not (0.0 <= self.omega_tl <= 1.0):
            raise ValueError(f"omega_tl must be between 0 and 1, got {self.omega_tl}")

        # Relative permeability endpoints
        if not (0.1 <= self.k_ro_0 <= 1.0):
            raise ValueError(f"k_ro_0 must be between 0.1 and 1, got {self.k_ro_0}")
        if not (0.1 <= self.k_rg_0 <= 1.0):
            raise ValueError(f"k_rg_0 must be between 0.1 and 1, got {self.k_rg_0}")

        # Corey exponents
        if not (1.0 <= self.n_o <= 5.0):
            raise ValueError(f"n_o must be between 1 and 5, got {self.n_o}")
        if not (1.0 <= self.n_g <= 5.0):
            raise ValueError(f"n_g must be between 1 and 5, got {self.n_g}")

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "EmpiricalFittingParameters":
        """Create EmpiricalFittingParameters from config dictionary."""
        config = config_dict.copy()
        config.update(kwargs)
        return from_dict_to_dataclass(cls, config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return dataclasses.asdict(self)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_mmp_yellig_metcalfe(temperature_f: float) -> float:
    """
    Calculate MMP using Yellig & Metcalfe correlation.

    Args:
        temperature_f: Temperature in Fahrenheit

    Returns:
        MMP in Pascals
    """
    mmp_psi = 0.712 * temperature_f + 49.7
    return mmp_psi * _PHYS_CONSTANTS.PSI_TO_PA
