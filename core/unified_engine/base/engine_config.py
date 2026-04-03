"""
Configuration classes for the Unified Physics Engine.

Provides dataclass-based configuration for engine mode, solvers, time-stepping,
and optional physics modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class EngineMode(Enum):
    """Engine operation mode."""
    SIMPLE = "simple"      # Fast screening/optimization mode
    DETAILED = "detailed"  # Comprehensive physics mode


class SolverType(Enum):
    """Numerical solver type."""
    EXPLICIT = "explicit"        # Explicit time integration (fast, conditionally stable)
    IMPLICIT = "implicit"        # Implicit time integration (stable, slower)
    ADAPTIVE = "adaptive"        # Adaptive scheme selection based on conditions


class PressureSolverType(Enum):
    """Pressure solver type."""
    ANALYTICAL = "analytical"    # Fast analytical solution
    FINITE_DIFFERENCE = "finite_difference"  # FD discretization
    FINITE_VOLUME = "finite_volume"  # FV discretization


class RelativePermeabilityModel(Enum):
    """Relative permeability model type."""
    COREY = "corey"              # Corey correlation
    LET = "let"                  # LET model
    BROOKS_COREY = "brooks_corey"  # Brooks-Corey model
    CUSTOM = "custom"            # User-defined table


class EOSModel(Enum):
    """Equation of State model."""
    BLACK_OIL = "black_oil"      # Black-oil model
    PENG_ROBINSON = "peng_robinson"  # Peng-Robinson EOS
    SRK = "srk"                  # Soave-Redlich-Kwong EOS


@dataclass
class TimestepConfig:
    """Time-stepping configuration."""
    initial: float = 1.0         # Initial timestep (days)
    minimum: float = 0.01        # Minimum timestep (days)
    maximum: float = 365.0       # Maximum timestep (days)
    adaptive: bool = True        # Enable adaptive time-stepping
    growth_factor: float = 1.5   # Maximum timestep growth factor
    cutback_factor: float = 0.5  # Timestep cutback on failure
    max_iterations: int = 10     # Max nonlinear iterations per step

    # CFL-based control (for explicit schemes)
    cfl_safety_factor: float = 0.8  # Safety factor for CFL condition
    max_cfl: float = 1.0         # Maximum allowed CFL number


@dataclass
class ModuleConfig:
    """Physics module enablement configuration."""
    # Core physics (always enabled for both modes)
    enable_multiphase_flow: bool = True
    enable_relative_permeability: bool = True
    enable_eos: bool = True

    # Optional physics (detailed mode only)
    enable_geomechanics: bool = False
    enable_fault_mechanics: bool = False
    enable_mineralization: bool = False
    enable_compositional: bool = False
    enable_tracer: bool = False

    # Model selections
    relperm_model: RelativePermeabilityModel = RelativePermeabilityModel.COREY
    eos_model: EOSModel = EOSModel.BLACK_OIL

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.enable_compositional and self.eos_model == EOSModel.BLACK_OIL:
            # Compositional requires cubic EOS
            self.eos_model = EOSModel.PENG_ROBINSON


@dataclass
class GridConfig:
    """Grid configuration."""
    nx: int = 20                 # Number of cells in x-direction
    ny: int = 20                 # Number of cells in y-direction
    nz: int = 1                  # Number of cells in z-direction
    dx: float = 100.0            # Cell size in x (ft)
    dy: float = 100.0            # Cell size in y (ft)
    dz: float = 10.0             # Cell size in z (ft)

    # Grid type
    cartesian: bool = True       # True for Cartesian, False for corner-point
    refined: bool = False        # Enable local grid refinement

    # Active cells (for corner-point grids)
    active_cells: Optional[int] = None

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.nx * self.ny * self.nz

    @property
    def total_volume(self) -> float:
        """Total reservoir volume (ft^3)."""
        return self.n_cells * self.dx * self.dy * self.dz


@dataclass
class WellConfig:
    """Well configuration."""
    name: str
    well_type: str  # "injector" or "producer"
    location: tuple  # (i, j, k) grid indices
    radius: float = 0.5  # Wellbore radius (ft)
    skin_factor: float = 0.0
    control_mode: str = "rate"  # "rate" or "bhp"
    target_value: float = 0.0  # Injection/production rate or BHP
    min_bhp: Optional[float] = None  # Minimum BHP constraint
    max_bhp: Optional[float] = None  # Maximum BHP constraint


@dataclass
class EngineConfig:
    """
    Main configuration class for the Unified Physics Engine.

    Controls engine mode, solver selection, time-stepping, and module enablement.
    """
    # Engine mode
    mode: EngineMode = EngineMode.SIMPLE

    # Solver configuration
    solver_type: SolverType = SolverType.ADAPTIVE
    pressure_solver: PressureSolverType = PressureSolverType.ANALYTICAL

    # Time-stepping
    timestep: TimestepConfig = field(default_factory=TimestepConfig)

    # Module configuration
    modules: ModuleConfig = field(default_factory=ModuleConfig)

    # Grid configuration
    grid: GridConfig = field(default_factory=GridConfig)

    # Well configurations
    wells: List[WellConfig] = field(default_factory=list)

    # Simulation parameters
    simulation_time: float = 365.0  # Total simulation time (days)
    output_frequency: float = 30.0  # Output interval (days)

    # Numerical tolerances
    tolerance_pressure: float = 1.0  # Pressure tolerance (psi)
    tolerance_saturation: float = 0.01  # Saturation tolerance
    max_newton_iterations: int = 10

    # Parallel processing
    parallel: bool = False
    n_threads: int = 1

    # Additional parameters (for extensibility)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Geomechanics parameters (set from data integration or defaults)
    # Uses Optional["GeomechanicsParameters"] to avoid circular import
    geomechanics_parameters: Optional[Any] = None

    # Reservoir depth for stress calculations (ft)
    reservoir_depth: float = 8000.0

    def __post_init__(self):
        """Validate configuration and apply mode defaults."""
        # Apply mode-specific defaults
        if self.mode == EngineMode.SIMPLE:
            # Simple mode: Fast solvers, minimal physics
            if self.solver_type == SolverType.IMPLICIT:
                self.solver_type = SolverType.EXPLICIT
            if self.pressure_solver == PressureSolverType.FINITE_VOLUME:
                self.pressure_solver = PressureSolverType.ANALYTICAL

            # Disable expensive modules
            self.modules.enable_geomechanics = False
            self.modules.enable_fault_mechanics = False
            self.modules.enable_mineralization = False
            self.modules.enable_compositional = False

        elif self.mode == EngineMode.DETAILED:
            # Detailed mode: Implicit/adaptive, full physics
            self.modules.enable_compositional = True
            if self.modules.eos_model == EOSModel.BLACK_OIL:
                self.modules.eos_model = EOSModel.PENG_ROBINSON

    def is_detailed(self) -> bool:
        """Check if running in detailed mode."""
        return self.mode == EngineMode.DETAILED

    def is_simple(self) -> bool:
        """Check if running in simple mode."""
        return self.mode == EngineMode.SIMPLE

    def get_effective_solver_type(self) -> SolverType:
        """Get the solver type that will actually be used."""
        if self.solver_type == SolverType.ADAPTIVE:
            return SolverType.IMPLICIT if self.is_detailed() else SolverType.EXPLICIT
        return self.solver_type

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            Empty list if valid, otherwise list of error messages.
        """
        errors = []

        # Validate grid
        if self.grid.nx <= 0 or self.grid.ny <= 0 or self.grid.nz <= 0:
            errors.append("Grid dimensions must be positive")

        if self.grid.dx <= 0 or self.grid.dy <= 0 or self.grid.dz <= 0:
            errors.append("Cell sizes must be positive")

        # Validate time-stepping
        if self.timestep.minimum <= 0:
            errors.append("Minimum timestep must be positive")

        if self.timestep.initial < self.timestep.minimum:
            errors.append("Initial timestep must be >= minimum timestep")

        if self.timestep.maximum < self.timestep.initial:
            errors.append("Maximum timestep must be >= initial timestep")

        if self.timestep.growth_factor <= 1.0:
            errors.append("Growth factor must be > 1.0")

        if not (0 < self.timestep.cutback_factor < 1.0):
            errors.append("Cutback factor must be between 0 and 1")

        # Validate simulation
        if self.simulation_time <= 0:
            errors.append("Simulation time must be positive")

        if self.output_frequency <= 0:
            errors.append("Output frequency must be positive")

        # Validate wells
        for well in self.wells:
            i, j, k = well.location
            if not (0 <= i < self.grid.nx and 0 <= j < self.grid.ny and 0 <= k < self.grid.nz):
                errors.append(f"Well {well.name} location outside grid bounds")

            if well.well_type not in ["injector", "producer"]:
                errors.append(f"Well {well.name} type must be 'injector' or 'producer'")

            if well.control_mode not in ["rate", "bhp"]:
                errors.append(f"Well {well.name} control mode must be 'rate' or 'bhp'")

            if well.radius <= 0:
                errors.append(f"Well {well.name} radius must be positive")

        # Validate tolerances
        if self.tolerance_pressure <= 0:
            errors.append("Pressure tolerance must be positive")

        if self.tolerance_saturation <= 0:
            errors.append("Saturation tolerance must be positive")

        if self.max_newton_iterations <= 0:
            errors.append("Max Newton iterations must be positive")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode.value,
            "solver_type": self.solver_type.value,
            "pressure_solver": self.pressure_solver.value,
            "timestep": {
                "initial": self.timestep.initial,
                "minimum": self.timestep.minimum,
                "maximum": self.timestep.maximum,
                "adaptive": self.timestep.adaptive,
                "growth_factor": self.timestep.growth_factor,
                "cutback_factor": self.timestep.cutback_factor,
                "max_iterations": self.timestep.max_iterations,
                "cfl_safety_factor": self.timestep.cfl_safety_factor,
                "max_cfl": self.timestep.max_cfl,
            },
            "modules": {
                "enable_multiphase_flow": self.modules.enable_multiphase_flow,
                "enable_relative_permeability": self.modules.enable_relative_permeability,
                "enable_eos": self.modules.enable_eos,
                "enable_geomechanics": self.modules.enable_geomechanics,
                "enable_fault_mechanics": self.modules.enable_fault_mechanics,
                "enable_mineralization": self.modules.enable_mineralization,
                "enable_compositional": self.modules.enable_compositional,
                "enable_tracer": self.modules.enable_tracer,
                "relperm_model": self.modules.relperm_model.value,
                "eos_model": self.modules.eos_model.value,
            },
            "grid": {
                "nx": self.grid.nx,
                "ny": self.grid.ny,
                "nz": self.grid.nz,
                "dx": self.grid.dx,
                "dy": self.grid.dy,
                "dz": self.grid.dz,
                "cartesian": self.grid.cartesian,
                "refined": self.grid.refined,
            },
            "simulation_time": self.simulation_time,
            "output_frequency": self.output_frequency,
            "tolerance_pressure": self.tolerance_pressure,
            "tolerance_saturation": self.tolerance_saturation,
            "max_newton_iterations": self.max_newton_iterations,
            "parallel": self.parallel,
            "n_threads": self.n_threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        """Create configuration from dictionary."""
        timestep_data = data.get("timestep", {})
        module_data = data.get("modules", {})
        grid_data = data.get("grid", {})

        return cls(
            mode=EngineMode(data.get("mode", "simple")),
            solver_type=SolverType(data.get("solver_type", "adaptive")),
            pressure_solver=PressureSolverType(
                data.get("pressure_solver", "analytical")
            ),
            timestep=TimestepConfig(
                initial=timestep_data.get("initial", 1.0),
                minimum=timestep_data.get("minimum", 0.01),
                maximum=timestep_data.get("maximum", 365.0),
                adaptive=timestep_data.get("adaptive", True),
                growth_factor=timestep_data.get("growth_factor", 1.5),
                cutback_factor=timestep_data.get("cutback_factor", 0.5),
                max_iterations=timestep_data.get("max_iterations", 10),
                cfl_safety_factor=timestep_data.get("cfl_safety_factor", 0.8),
                max_cfl=timestep_data.get("max_cfl", 1.0),
            ),
            modules=ModuleConfig(
                enable_multiphase_flow=module_data.get("enable_multiphase_flow", True),
                enable_relative_permeability=module_data.get(
                    "enable_relative_permeability", True
                ),
                enable_eos=module_data.get("enable_eos", True),
                enable_geomechanics=module_data.get("enable_geomechanics", False),
                enable_fault_mechanics=module_data.get("enable_fault_mechanics", False),
                enable_mineralization=module_data.get("enable_mineralization", False),
                enable_compositional=module_data.get("enable_compositional", False),
                enable_tracer=module_data.get("enable_tracer", False),
                relperm_model=RelativePermeabilityModel(
                    module_data.get("relperm_model", "corey")
                ),
                eos_model=EOSModel(module_data.get("eos_model", "black_oil")),
            ),
            grid=GridConfig(
                nx=grid_data.get("nx", 20),
                ny=grid_data.get("ny", 20),
                nz=grid_data.get("nz", 1),
                dx=grid_data.get("dx", 100.0),
                dy=grid_data.get("dy", 100.0),
                dz=grid_data.get("dz", 10.0),
                cartesian=grid_data.get("cartesian", True),
                refined=grid_data.get("refined", False),
            ),
            simulation_time=data.get("simulation_time", 365.0),
            output_frequency=data.get("output_frequency", 30.0),
            tolerance_pressure=data.get("tolerance_pressure", 1.0),
            tolerance_saturation=data.get("tolerance_saturation", 0.01),
            max_newton_iterations=data.get("max_newton_iterations", 10),
            parallel=data.get("parallel", False),
            n_threads=data.get("n_threads", 1),
        )
