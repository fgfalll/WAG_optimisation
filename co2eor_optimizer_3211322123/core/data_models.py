import dataclasses
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

# Helper to create dataclass instances from dict, only using keys present in the dataclass
def from_dict_to_dataclass(cls, data: Dict[str, Any]):
    """
    Creates an instance of a dataclass from a dictionary, only using keys
    that correspond to fields defined in the dataclass.
    """
    # Use dataclasses.fields to be explicit and avoid name collisions
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

@dataclasses.dataclass
class WellData:
    """Represents data associated with a single well."""
    name: str
    depths: np.ndarray # Measured depths or True Vertical Depths
    properties: Dict[str, np.ndarray] # e.g., {'GR': array, 'PORO': array}
    units: Dict[str, str] # e.g., {'GR': 'API', 'PORO': 'v/v'}
    # A dictionary for storing single-value parameters or metadata not suitable
    # for the properties array, e.g., {'API': 35.0, 'Temperature': 212.0}
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def validate(self) -> bool:
        """Validates that all property arrays have the same length as the depths array."""
        if not self.properties: return True # No properties to validate against depths
        if not hasattr(self.depths, 'size') or self.depths.size == 0:
            # If depths is empty, all property arrays must also be empty
            return all(not hasattr(prop_array, 'size') or prop_array.size == 0
                       for prop_array in self.properties.values())
        return all(len(self.depths) == len(prop_array)
                   for prop_array in self.properties.values())
    
@dataclasses.dataclass
class EOSModelParameters:
    """
    Represents parameters for an Equation of State (EOS) model.
    This structure is designed to be populated by parsers and used by the EOS engine.
    """
    eos_type: str  # e.g., 'Peng-Robinson' or 'Soave-Redlich-Kwong'
    
    # A structured numpy array holding component properties.
    # The parsers must ensure this structure is followed.
    # Columns (in order):
    # 0: Component Name (str, e.g., 'CO2', 'C1', 'C7+')
    # 1: Mole Fraction (zi) (float)
    # 2: Molecular Weight (MW) (float, g/mol)
    # 3: Critical Temperature (Tc) (float, °R)
    # 4: Critical Pressure (Pc) (float, psia)
    # 5: Acentric Factor (omega) (float)
    # 6: Volume Shift parameter (s_i) (float)
    component_properties: np.ndarray 
    
    # Square matrix of binary interaction coefficients (kij values)
    binary_interaction_coeffs: np.ndarray

@dataclasses.dataclass
class ReservoirData:
    """Represents static reservoir model data."""
    grid: Dict[str, np.ndarray] # Grid properties like 'PORO', 'PERMX', 'TOPS'
    pvt_tables: Dict[str, np.ndarray] # PVT data tables
    regions: Optional[Dict[str, np.ndarray]] = None # Regional properties if any
    runspec: Optional[Dict[str, Any]] = None # Parsed runspec data from Eclipse files
    faults: Optional[Dict[str, Any]] = None # Fault data, e.g., transmissibility multipliers
    ooip_stb: float = 1_000_000.0 # Original Oil In Place, Stock Tank Barrels (default fallback)
    # This field will hold the full EOS configuration if a compositional model is used.
    eos_model: Optional[EOSModelParameters] = None

    def set_faults_data(self, faults_data: Dict[str, Any]) -> None:
        """Sets or updates fault data."""
        self.faults = faults_data

@dataclasses.dataclass
class PVTProperties:
    """
    Represents fluid Pressure-Volume-Temperature properties.
    For black oil models, or as a fallback/reference for compositional models.
    """
    oil_fvf: np.ndarray # Oil Formation Volume Factor (Bo)
    oil_viscosity: np.ndarray # Oil viscosity (mu_o)
    gas_fvf: np.ndarray # Gas Formation Volume Factor (Bg)
    gas_viscosity: np.ndarray # Gas viscosity (mu_g)
    rs: np.ndarray  # Solution Gas-Oil Ratio (Rs)
    pvt_type: str  # Typically 'black_oil' or 'compositional'
    gas_specific_gravity: float # Specific gravity of the gas (air=1)
    temperature: float  # Reservoir temperature in °F

    def __post_init__(self):
        """Validates PVT properties after initialization."""
        arrays = [self.oil_fvf, self.oil_viscosity, self.gas_fvf, self.gas_viscosity, self.rs]
        non_none_arrays = [arr for arr in arrays if arr is not None and hasattr(arr, '__len__')]
        if non_none_arrays:
            if len({len(arr) for arr in non_none_arrays}) > 1:
                raise ValueError("All provided PVT property arrays must have the same length.")
        if self.pvt_type not in {'black_oil', 'compositional'}:
            raise ValueError("pvt_type must be either 'black_oil' or 'compositional'")
        if not (isinstance(self.gas_specific_gravity, (int, float)) and 0.5 <= self.gas_specific_gravity <= 1.2):
            raise ValueError(f"Gas specific gravity must be between 0.5-1.2, got {self.gas_specific_gravity}")
        if not (isinstance(self.temperature, (int, float)) and 50 <= self.temperature <= 400): # Common reservoir temp range in °F
            raise ValueError(f"Temperature must be between 50-400°F, got {self.temperature}")

@dataclasses.dataclass
class EconomicParameters:
    """Holds economic parameters for project evaluation like NPV."""
    oil_price_usd_per_bbl: float = 70.0
    co2_purchase_cost_usd_per_tonne: float = 50.0
    co2_injection_cost_usd_per_mscf: float = 0.5
    water_injection_cost_usd_per_bbl: float = 1.0
    water_disposal_cost_usd_per_bbl: float = 2.0 # Cost to dispose produced water
    discount_rate_fraction: float = 0.10 # Annual discount rate (e.g., 0.10 for 10%)
    operational_cost_usd_per_bbl_oil: float = 5.0 # OPEX related to produced oil

    def __post_init__(self):
        if not (0.0 < self.oil_price_usd_per_bbl <= 1000.0):
            raise ValueError("Oil Price must be between $0 and $1000/bbl.")
        if not (0.0 <= self.co2_purchase_cost_usd_per_tonne <= 1000.0):
            raise ValueError("CO2 Purchase Cost must be between $0 and $1000/tonne.")
        if not (0.0 <= self.co2_injection_cost_usd_per_mscf <= 100.0):
            raise ValueError("CO2 Injection Cost must be between $0 and $100/MSCF.")
        if not (0.0 <= self.water_injection_cost_usd_per_bbl <= 100.0):
            raise ValueError("Water Injection Cost must be between $0 and $100/bbl.")
        if not (0.0 <= self.water_disposal_cost_usd_per_bbl <= 100.0):
            raise ValueError("Water Disposal Cost must be between $0 and $100/bbl.")
        if not (0.0 <= self.discount_rate_fraction <= 1.0):
            raise ValueError("Discount Rate must be a fraction between 0.0 and 1.0.")
        if not (0.0 <= self.operational_cost_usd_per_bbl_oil <= 200.0):
            raise ValueError("Operational Cost must be between $0 and $200/bbl.")

    @classmethod
    def from_config_dict(cls, config_econ_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_econ_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class EORParameters:
    """Parameters specific to the Enhanced Oil Recovery (EOR) process being modeled."""
    injection_rate: float = 5000.0
    WAG_ratio: Optional[float] = 1.0
    injection_scheme: str = 'continuous'
    min_cycle_length_days: float = 15.0
    max_cycle_length_days: float = 90.0
    min_water_fraction: float = 0.2
    max_water_fraction: float = 0.8
    target_pressure_psi: float = 3000.0
    max_pressure_psi: float = 6000.0
    min_injection_rate_bpd: float = 1000.0
    v_dp_coefficient: float = 0.55
    mobility_ratio: float = 2.5

    def __post_init__(self):
        if not (0 < self.injection_rate <= 100000):
            raise ValueError("Injection Rate must be between 0 and 100,000.")
        if self.WAG_ratio is not None and not (0 <= self.WAG_ratio <= 10):
            raise ValueError("WAG Ratio must be between 0 and 10.")
        if self.injection_scheme not in ['continuous', 'wag']:
            raise ValueError(f"Invalid Injection Scheme. Must be 'continuous' or 'wag'.")
        if not (0 < self.min_cycle_length_days < 1000):
            raise ValueError("Min Cycle Length must be between 0 and 1000 days.")
        if not (0 < self.max_cycle_length_days < 2000):
            raise ValueError("Max Cycle Length must be between 0 and 2000 days.")
        if self.max_cycle_length_days < self.min_cycle_length_days:
            raise ValueError("Max Cycle Length must be greater than or equal to Min Cycle Length.")
        if not (0.0 <= self.min_water_fraction <= 1.0):
            raise ValueError("Min Water Fraction must be between 0.0 and 1.0.")
        if not (0.0 <= self.max_water_fraction <= 1.0):
            raise ValueError("Max Water Fraction must be between 0.0 and 1.0.")
        if self.max_water_fraction < self.min_water_fraction:
            raise ValueError("Max Water Fraction must be greater than or equal to Min Water Fraction.")
        if not (0 <= self.target_pressure_psi <= 15000):
            raise ValueError("Target Pressure must be between 0 and 15,000 psi.")
        if not (0 < self.max_pressure_psi <= 20000):
            raise ValueError("Max Pressure must be between 0 and 20,000 psi.")
        if self.max_pressure_psi < self.target_pressure_psi:
            raise ValueError("Max Pressure must be greater than or equal to Target Pressure.")
        if not (0 <= self.v_dp_coefficient <= 1.0):
            raise ValueError("V_DP Coefficient must be between 0.0 and 1.0.")
        if not (0 < self.mobility_ratio <= 50):
            raise ValueError("Mobility Ratio must be between 0 and 50.")

    @classmethod
    def from_config_dict(cls, config_eor_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_eor_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class GeneticAlgorithmParams:
    """Parameters for configuring the Genetic Algorithm optimizer."""
    population_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_count: int = 3
    tournament_size: int = 3
    blend_alpha_crossover: float = 0.5
    mutation_strength_factor: float = 0.1

    def __post_init__(self):
        if not (10 <= self.population_size <= 1000):
            raise ValueError("Population Size must be between 10 and 1000.")
        if not (10 <= self.generations <= 1000):
            raise ValueError("Generations must be between 10 and 1000.")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("Crossover Rate must be between 0.0 and 1.0.")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("Mutation Rate must be between 0.0 and 1.0.")
        if not (0 <= self.elite_count < self.population_size):
            raise ValueError("Elite Count must be non-negative and less than Population Size.")
        if not (2 <= self.tournament_size <= self.population_size):
            raise ValueError("Tournament Size must be between 2 and Population Size.")
        if not (0.0 <= self.blend_alpha_crossover <= 1.0):
            raise ValueError("Blend Alpha Crossover must be between 0.0 and 1.0.")
        if not (0.0 < self.mutation_strength_factor <= 1.0):
            raise ValueError("Mutation Strength Factor must be between 0.0 and 1.0.")

    @classmethod
    def from_config_dict(cls, config_ga_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_ga_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class OperationalParameters:
    """Parameters related to overall project operations and timeline."""
    project_lifetime_years: int = 15

    def __post_init__(self):
        if not (1 <= self.project_lifetime_years <= 100):
            raise ValueError("Project Lifetime must be between 1 and 100 years.")

    @classmethod
    def from_config_dict(cls, config_op_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_op_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class ProfileParameters:
    """Defines how annual production and injection volumes are profiled over the project lifetime."""
    oil_profile_type: str = "plateau_linear_decline"
    injection_profile_type: str = "constant_during_phase"
    # Fields for plateau profiles
    plateau_duration_fraction_of_life: Optional[float] = 0.3
    # Fields for decline profiles
    initial_decline_rate_annual_fraction: Optional[float] = 0.15
    min_economic_rate_fraction_of_peak: Optional[float] = 0.05
    # Fields for hyperbolic decline
    hyperbolic_b_factor: Optional[float] = 0.5
    # Fields for custom profile
    oil_annual_fraction_of_total: Optional[List[float]] = None
    warn_if_defaults_used: bool = True

    def __post_init__(self):
        """Validates profile parameters and ensures logical consistency."""
        valid_oil_profiles = [
            "linear_distribution", "plateau_linear_decline",
            "plateau_exponential_decline", "plateau_hyperbolic_decline", "custom_fractions"
        ]
        if self.oil_profile_type not in valid_oil_profiles:
            raise ValueError(f"Invalid oil profile type. Choose one of: {valid_oil_profiles}")

        # --- Validate relevant parameters and nullify irrelevant ones ---
        
        # Custom fractions profile
        if self.oil_profile_type == "custom_fractions":
            if self.oil_annual_fraction_of_total is None or not self.oil_annual_fraction_of_total:
                raise ValueError("For 'custom_fractions' profile, a list of annual fractions must be provided.")
            if not np.isclose(sum(self.oil_annual_fraction_of_total), 1.0):
                logging.warning(f"Custom fractions sum to {sum(self.oil_annual_fraction_of_total):.3f}, not 1.0. They will be normalized.")
            # Nullify others
            self.plateau_duration_fraction_of_life = None
            self.initial_decline_rate_annual_fraction = None
            self.hyperbolic_b_factor = None
            self.min_economic_rate_fraction_of_peak = None
        else:
            # All other profiles should not have custom fractions
            self.oil_annual_fraction_of_total = None

        # Plateau profiles
        if 'plateau' in self.oil_profile_type:
            if self.plateau_duration_fraction_of_life is None:
                self.plateau_duration_fraction_of_life = 0.3 # Assign default if needed
            if not (0.0 <= self.plateau_duration_fraction_of_life < 1.0):
                raise ValueError("Plateau Duration Fraction must be between 0.0 (inclusive) and 1.0 (exclusive).")
        else:
            self.plateau_duration_fraction_of_life = None
            
        # Decline profiles
        if 'decline' in self.oil_profile_type:
            if self.initial_decline_rate_annual_fraction is None:
                self.initial_decline_rate_annual_fraction = 0.15 # Assign default if needed
            if not (0.0 < self.initial_decline_rate_annual_fraction < 1.0):
                raise ValueError("Initial Decline Rate must be between 0.0 and 1.0 (exclusive).")
            
            if self.min_economic_rate_fraction_of_peak is None:
                self.min_economic_rate_fraction_of_peak = 0.05
            if not (0.0 <= self.min_economic_rate_fraction_of_peak < 1.0):
                raise ValueError("Min Economic Rate Fraction must be between 0.0 (inclusive) and 1.0 (exclusive).")
        else:
            self.initial_decline_rate_annual_fraction = None
            self.min_economic_rate_fraction_of_peak = None

        # Hyperbolic decline specifically
        if 'hyperbolic' in self.oil_profile_type:
            if self.hyperbolic_b_factor is None:
                self.hyperbolic_b_factor = 0.5 # Assign default if needed
            if not (0.0 <= self.hyperbolic_b_factor <= 2.0):
                raise ValueError("Hyperbolic b-factor must be between 0.0 and 2.0.")
        else:
            self.hyperbolic_b_factor = None

    @classmethod
    def from_config_dict(cls, config_profile_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_profile_params_dict)
        defaults.update(kwargs)
        instance = from_dict_to_dataclass(cls, defaults)
        if instance.oil_annual_fraction_of_total is not None and not isinstance(instance.oil_annual_fraction_of_total, list):
            try:
                instance.oil_annual_fraction_of_total = list(instance.oil_annual_fraction_of_total)
            except TypeError:
                raise ValueError("oil_annual_fraction_of_total must be a list or convertible to a list.")
        return instance