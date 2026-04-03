"""
Utility functions and data structures for the EOR Reservoir Simulation Engine
===========================================================================

This module contains:
- Data classes for reservoir parameters
- Unit conversion utilities
- Validation functions
- Common helper functions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings

from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    RockProperties,
    FluidProperties,
    SimpleGrid,
    ReservoirState,
    CoreyParameters,
    PhysicalConstants,
)

# Create an instance for accessing PhysicalConstants values (needed because it's a slots=True dataclass)
_PHYS_CONSTANTS = PhysicalConstants()

# Alias for backward compatibility
GridParameters = SimpleGrid


# Adapter functions to convert between simple engine parameters and main data models
def create_reservoir_data_from_simple(grid_params, rock_props, fluid_props):
    """Create ReservoirData from simple engine parameters"""
    # Create grid arrays
    nx, ny, nz = grid_params.nx, grid_params.ny, grid_params.nz
    dx, dy, dz = grid_params.dx, grid_params.dy, grid_params.dz

    # Create coordinate grids
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Create basic grid
    grid = {
        "X": X.flatten(),
        "Y": Y.flatten(),
        "Z": Z.flatten(),
        "DX": np.full(nx * ny * nz, dx),
        "DY": np.full(nx * ny * nz, dy),
        "DZ": np.full(nx * ny * nz, dz),
        "PORO": rock_props.porosity.flatten(),
        "PERMX": rock_props.permeability_x.flatten(),
        "PERMY": rock_props.permeability_y.flatten(),
        "PERMZ": rock_props.permeability_z.flatten(),
    }

    # Calculate reservoir properties
    total_volume = grid_params.total_cells * dx * dy * dz  # m³
    total_volume_bbl = total_volume * 6.28981  # Convert to barrels

    # Estimate average properties
    avg_porosity = np.mean(rock_props.porosity)
    avg_permeability = np.mean(rock_props.permeability_x)  # Using x-direction as representative

    return ReservoirData(
        grid=grid,
        pvt_tables={
            "pressure": np.array([2000, 3000, 4000, 5000]),
            "oil_fvf": np.array([1.2, 1.3, 1.4, 1.5]),
        },
        ooip_stb=total_volume_bbl
        * avg_porosity
        * (1 - fluid_props.initial_water_saturation)
        / fluid_props.oil_fvf,
        initial_pressure=4000.0,
        average_porosity=avg_porosity,
        average_permeability=avg_permeability,
        thickness_ft=(nz * dz) / 0.3048,  # Convert to feet
        area_acres=(nx * dx * ny * dy) / 4046.86,  # Convert to acres
        length_ft=(nx * dx) / 0.3048,  # Convert to feet
        cross_sectional_area_acres=(ny * dy * nz * dz) / 4046.86,  # Convert to acres
        initial_water_saturation=fluid_props.initial_water_saturation,
        oil_fvf=fluid_props.oil_fvf,
        temperature=150.0,  # Default temperature in Fahrenheit
    )


def create_eor_parameters_from_simple(simple_eor_params):
    """Create EORParameters from simple engine parameters"""
    return EORParameters(
        injection_rate=simple_eor_params.injection_rates[0]
        if simple_eor_params.injection_rates is not None
        else 5000.0,
        mobility_ratio=simple_eor_params.mobility_ratio,
        target_pressure_psi=simple_eor_params.pressure_target,
        max_pressure_psi=simple_eor_params.max_pressure,
        density_contrast=simple_eor_params.density_contrast,
        dip_angle=simple_eor_params.dip_angle,
        interfacial_tension=simple_eor_params.interfacial_tension,
        sor=simple_eor_params.sor,
        co2_density_tonne_per_mscf=simple_eor_params.co2_density,
        kv_kh_ratio=simple_eor_params.kv_kh_ratio,
        s_gc=simple_eor_params.s_gc,
        n_o=simple_eor_params.n_o,
        n_g=simple_eor_params.n_g,
    )


@dataclass
class GridParameters:
    """
    Grid parameters for reservoir simulation.

    DEPRECATED: Use SimpleGrid from data_models.py instead.
    This class is maintained for backward compatibility.
    """

    nx: int  # Number of grid blocks in x-direction
    ny: int  # Number of grid blocks in y-direction
    nz: int  # Number of grid blocks in z-direction
    dx: float  # Grid block size in x-direction (m)
    dy: float  # Grid block size in y-direction (m)
    dz: float  # Grid block size in z-direction (m)

    def __post_init__(self):
        """Validate grid parameters"""
        warnings.warn(
            "GridParameters is deprecated. Use SimpleGrid from data_models.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if any(param <= 0 for param in [self.nx, self.ny, self.nz]):
            raise ValueError("Grid dimensions must be positive")
        if any(param <= 0 for param in [self.dx, self.dy, self.dz]):
            raise ValueError("Grid block sizes must be positive")

    @property
    def n_cells(self) -> int:
        """Total number of grid cells (alias for total_cells)"""
        return self.nx * self.ny * self.nz

    @property
    def total_cells(self) -> int:
        """Total number of grid cells"""
        return self.nx * self.ny * self.nz

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Physical dimensions of the reservoir"""
        return (self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)


@dataclass
class WellParameters:
    """Well parameters for simulation"""

    well_type: str  # 'injector' or 'producer'
    location: Tuple[int, int, int]  # (i, j, k) grid indices
    radius: float = 0.1  # Well radius (m)
    skin_factor: float = 0.0  # Skin factor

    # Operational parameters
    rate: float = 0.0  # Rate (m³/day)
    bottom_hole_pressure: float = None  # BHP constraint (Pa)

    # Injection/production composition (for injection wells)
    water_fraction: float = 0.0  # Water fraction in injection
    oil_fraction: float = 0.0  # Oil fraction in injection
    gas_fraction: float = 0.0  # Gas fraction in injection

    def __post_init__(self):
        """Validate well parameters"""
        if self.well_type not in ["injector", "producer"]:
            raise ValueError("Well type must be 'injector' or 'producer'")

        if self.well_type == "injector":
            total_frac = self.water_fraction + self.oil_fraction + self.gas_fraction
            if abs(total_frac - 1.0) > 1e-6:
                raise ValueError("Injection fractions must sum to 1.0")


@dataclass
class SimulationResults:
    """Container for simulation results"""

    time: np.ndarray  # Time array (days)
    oil_rate: np.ndarray  # Oil production rate (m³/day)
    water_rate: np.ndarray  # Water production rate (m³/day)
    gas_rate: np.ndarray  # Gas production rate (m³/day)
    oil_cumulative: np.ndarray  # Cumulative oil production (m³)
    water_cumulative: np.ndarray  # Cumulative water production (m³)
    gas_cumulative: np.ndarray  # Cumulative gas production (m³)

    # CO₂ specific results
    co2_injection_rate: np.ndarray = None  # CO₂ injection rate (m³/day)
    co2_storage_volume: np.ndarray = None  # CO₂ storage volume (m³)
    co2_utilization_factor: np.ndarray = None  # CO₂ utilization factor (m³/m³)

    # Time-series profiles (tracked per output step)
    avg_pressure: np.ndarray = None  # Average reservoir pressure per step (Pa)
    recovery_factor_profile: np.ndarray = None  # RF vs time (fraction, 0-1)

    # Reservoir state snapshots
    pressure_field: Optional[List[np.ndarray]] = None
    saturation_field: Optional[List[Dict[str, np.ndarray]]] = None

    # Performance metrics
    recovery_factor: float = 0.0  # Recovery factor
    sweep_efficiency: float = 0.0  # Sweep efficiency
    storage_efficiency: float = 0.0  # Storage efficiency

    def calculate_recovery_factor(self, original_oil_in_place: float) -> float:
        """Calculate recovery factor"""
        if len(self.oil_cumulative) > 0:
            self.recovery_factor = self.oil_cumulative[-1] / original_oil_in_place
        return self.recovery_factor

    def calculate_utilization_factor(self) -> np.ndarray:
        """Calculate CO₂ utilization factor (m³ CO₂ per m³ oil)"""
        if self.co2_injection_rate is not None and self.oil_rate is not None:
            # Avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                self.co2_utilization_factor = np.where(
                    self.oil_rate > 0, self.co2_injection_rate / self.oil_rate, 0
                )
        return self.co2_utilization_factor


# Re-export from data_models for backward compatibility
__all__ = [
    "GridParameters",
    "RockProperties",
    "FluidProperties",
    "ReservoirState",
    "CoreyParameters",
    "SimpleGrid",
    "PhysicalConstants",
    "WellParameters",
    "SimulationResults",
    "create_reservoir_data_from_simple",
    "create_eor_parameters_from_simple",
    "get_conversion_factors",
    "convert_units",
    "validate_parameter",
]

# Unit conversion constants - sourced from PhysicalConstants (single source of truth)
_CONVERSION_FACTORS_CACHE = {}


def get_conversion_factors():
    """Get conversion factors from PhysicalConstants (lazy initialization)."""
    global _CONVERSION_FACTORS_CACHE
    if not _CONVERSION_FACTORS_CACHE:
        _CONVERSION_FACTORS_CACHE = {
            "psi_to_pa": _PHYS_CONSTANTS.PSI_TO_PA,
            "pa_to_psi": _PHYS_CONSTANTS.PA_TO_PSI,
            "md_to_m2": _PHYS_CONSTANTS.MD_TO_M2,
            "m2_to_md": _PHYS_CONSTANTS.M2_TO_MD,
            "day_to_sec": _PHYS_CONSTANTS.SECONDS_PER_DAY,
            "sec_to_day": 1.0 / _PHYS_CONSTANTS.SECONDS_PER_DAY,
            "year_to_day": _PHYS_CONSTANTS.DAYS_PER_YEAR,
            "day_to_year": 1.0 / _PHYS_CONSTANTS.DAYS_PER_YEAR,
            "bbl_to_m3": _PHYS_CONSTANTS.BBLS_TO_M3,
            "m3_to_bbl": _PHYS_CONSTANTS.M3_TO_BBL,
            "mcf_to_m3": _PHYS_CONSTANTS.MCF_TO_M3,
            "m3_to_mcf": 1.0 / _PHYS_CONSTANTS.MCF_TO_M3,
        }
    return _CONVERSION_FACTORS_CACHE


# Backward compatible alias (access via get_conversion_factors() for updated values)
CONVERSION_FACTORS = get_conversion_factors()


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between common reservoir engineering units using PhysicalConstants"""
    factors = get_conversion_factors()
    conversion_key = f"{from_unit}_to_{to_unit}"

    if conversion_key in factors:
        return value * factors[conversion_key]
    elif from_unit == to_unit:
        return value
    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")


def validate_parameter(
    param: float,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Validate a parameter with optional bounds"""
    if param is None:
        raise ValueError(f"Parameter {name} cannot be None")

    if min_val is not None and param < min_val:
        raise ValueError(f"Parameter {name} ({param}) is below minimum ({min_val})")

    if max_val is not None and param > max_val:
        raise ValueError(f"Parameter {name} ({param}) is above maximum ({max_val})")

    return param


def interpolate_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Safe 1D interpolation with bounds checking"""
    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length")

    # Check if x_new is within bounds
    if np.any(x_new < x[0]) or np.any(x_new > x[-1]):
        warnings.warn("Interpolation points outside data range, using extrapolation")

    return np.interp(x_new, x, y)


def calculate_array_statistics(arr: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics for an array"""
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "median": np.median(arr),
    }
