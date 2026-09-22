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
    depths: np.ndarray # Measured depths (MD)
    properties: Dict[str, np.ndarray] # e.g., {'GR': array, 'PORO': array}
    units: Dict[str, str] # e.g., {'GR': 'API', 'PORO': 'v/v'}
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    
    # List of dicts for detailed perforation properties, e.g., 
    # [{'top': 10000, 'bottom': 10050, 'api': 35.5, 'temp': 215.0}]
    perforation_properties: List[Dict[str, float]] = dataclasses.field(default_factory=list)
    
    # An Nx2 numpy array of [Horizontal Deviation, Measured Depth]
    well_path: Optional[np.ndarray] = None

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
    co2_recycle_cost_usd_per_tonne: float = 15.0
    water_injection_cost_usd_per_bbl: float = 1.0
    water_disposal_cost_usd_per_bbl: float = 2.0
    discount_rate_fraction: float = 0.10

    # [ADDED] Capex and Opex fields for a more complete NPV calculation
    capex_usd: float = 5_000_000.0
    fixed_opex_usd_per_year: float = 200_000.0
    variable_opex_usd_per_bbl: float = 5.0
    
    # [REMOVED/IGNORED] Redundant or unused fields for clarity
    # co2_injection_cost_usd_per_mscf: float = 0.5
    # operational_cost_usd_per_bbl_oil: float = 5.0
    # co2_utilization_target: float = 0.5
    # co2_overutilization_penalty_usd_per_tonne: float = 0.0
    # co2_utilization_credit_usd_per_tonne: float = 0.0

    def __post_init__(self):
        if not (0.0 < self.oil_price_usd_per_bbl <= 1000.0):
            raise ValueError("Oil Price must be between $0 and $1000/bbl.")
        if not (0.0 <= self.co2_purchase_cost_usd_per_tonne <= 1000.0):
            raise ValueError("CO2 Purchase Cost must be between $0 and $1000/tonne.")
        if self.co2_recycle_cost_usd_per_tonne > self.co2_purchase_cost_usd_per_tonne:
            logging.warning("CO2 Recycle Cost is higher than Purchase Cost, which is unusual.")
        if not (0.0 <= self.water_injection_cost_usd_per_bbl <= 100.0):
            raise ValueError("Water Injection Cost must be between $0 and $100/bbl.")
        if not (0.0 <= self.water_disposal_cost_usd_per_bbl <= 100.0):
            raise ValueError("Water Disposal Cost must be between $0 and $100/bbl.")
        if not (0.0 <= self.discount_rate_fraction <= 1.0):
            raise ValueError("Discount Rate must be a fraction between 0.0 and 1.0.")
        # [ADDED] Validation for new economic fields
        if not (0.0 <= self.capex_usd):
            raise ValueError("CAPEX must be non-negative.")
        if not (0.0 <= self.fixed_opex_usd_per_year):
            raise ValueError("Fixed OPEX must be non-negative.")
        if not (0.0 <= self.variable_opex_usd_per_bbl <= 200.0):
            raise ValueError("Variable OPEX must be between $0 and $200/bbl.")

    @classmethod
    def from_config_dict(cls, config_econ_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        # [MODIFIED] Handle legacy 'operational_cost_usd_per_bbl_oil' key
        if 'operational_cost_usd_per_bbl_oil' in config_econ_params_dict:
            config_econ_params_dict['variable_opex_usd_per_bbl'] = config_econ_params_dict.pop('operational_cost_usd_per_bbl_oil')
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
    max_injection_rate_bpd: float = 5000.0
    v_dp_coefficient: float = 0.55
    mobility_ratio: float = 2.5
    
    # [ADDED] Missing parameters for simplified physics models
    gas_oil_ratio_at_breakthrough: float = 1.5 # MSCF/STB
    water_cut_bwow: float = 3.0 # bbl water / bbl oil

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
        if not (self.min_injection_rate_bpd < self.max_injection_rate_bpd):
            raise ValueError("Max Injection Rate must be greater than Min Injection Rate.")
        if not (0 <= self.v_dp_coefficient <= 1.0):
            raise ValueError("V_DP Coefficient must be between 0.0 and 1.0.")
        if not (0 < self.mobility_ratio <= 50):
            raise ValueError("Mobility Ratio must be between 0 and 50.")
        # [ADDED] Validation for new parameters
        if not (0 <= self.gas_oil_ratio_at_breakthrough <= 20):
             raise ValueError("Gas Oil Ratio at Breakthrough must be between 0 and 20 MSCF/STB.")
        if not (0 <= self.water_cut_bwow <= 50):
             raise ValueError("Water Cut must be between 0 and 50 bbl/bbl.")

    @classmethod
    def from_config_dict(cls, config_eor_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_eor_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class GeneticAlgorithmParams:
    """Parameters for configuring the Genetic Algorithm optimizer."""
    # --- Base Parameters ---
    population_size: int = 60
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_count: int = 3
    tournament_size: int = 3
    blend_alpha_crossover: float = 0.5
    mutation_strength_factor: float = 0.1

    # --- Advanced Diversity and Adaptability Parameters ---
    # Adaptive mutation settings
    adaptive_mutation_enabled: bool = True
    stagnation_generations_limit: int = 15  # Generations of no improvement before triggering adaptive response
    min_mutation_rate: float = 0.05
    max_mutation_rate: float = 0.50
    
    # Random individual injection
    random_injection_rate: float = 0.05 # Percentage of worst population to replace with random individuals
    
    # Chaotic mutation
    use_chaotic_mutation: bool = True
    chaos_map_r: float = 3.99 # Logistic map parameter 'r' (must be in [3.57, 4.0] for chaos)

    # Niching via Fitness Sharing to adjust selection pressure
    use_fitness_sharing: bool = True
    sharing_sigma_threshold: float = 0.15 # Normalized distance threshold for sharing

    # Dynamic elitism
    dynamic_elitism_enabled: bool = True
    min_elite_count: int = 1

    def __post_init__(self):
        # --- Validations for Base Parameters ---
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

        # --- Validations for Advanced Parameters ---
        if self.adaptive_mutation_enabled:
            if not (0 < self.stagnation_generations_limit < self.generations):
                raise ValueError("Stagnation limit must be less than total generations.")
            if not (0.0 <= self.min_mutation_rate < self.max_mutation_rate <= 1.0):
                raise ValueError("Min/Max mutation rates are invalid.")
        if not (0.0 <= self.random_injection_rate <= 0.5):
            raise ValueError("Random injection rate should be between 0.0 and 0.5.")
        if self.use_chaotic_mutation and not (3.57 <= self.chaos_map_r <= 4.0):
            logging.warning(f"chaos_map_r={self.chaos_map_r} is outside the typical chaotic range of [3.57, 4.0].")
        if self.use_fitness_sharing and not (0.01 <= self.sharing_sigma_threshold <= 1.0):
             raise ValueError("Fitness sharing sigma threshold must be between 0.01 and 1.0.")
        if self.dynamic_elitism_enabled and not (0 <= self.min_elite_count < self.elite_count):
            raise ValueError("Min elite count must be less than the base elite count.")

    @classmethod
    def from_config_dict(cls, config_ga_params_dict: Dict[str, Any], **kwargs):
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}
        defaults.update(config_ga_params_dict)
        defaults.update(kwargs)
        return from_dict_to_dataclass(cls, defaults)

@dataclasses.dataclass
class BayesianOptimizationParams:
    """Parameters for configuring the Bayesian Optimization optimizer."""
    n_iterations: int = 40
    n_initial_points: int = 8

    def __post_init__(self):
        if self.n_iterations is None or not (5 <= self.n_iterations <= 1000):
            raise ValueError("BO: Number of iterations must be between 5 and 1000.")
        if self.n_initial_points is None or not (1 <= self.n_initial_points <= 100):
            raise ValueError("BO: Number of initial points must be between 1 and 100.")

    @classmethod
    def from_config_dict(cls, config_bo_params_dict: Dict[str, Any], **kwargs):
        """Creates an instance from a dictionary, with defaults and overrides."""
        # Get the dataclass defaults first
        defaults = {f.name: f.default for f in dataclasses.fields(cls) if f.default is not dataclasses.MISSING}

        # Create a new dictionary from the config dict, but with the correct keys, filtering out Nones
        mapped_config = {}
        n_iter_val = config_bo_params_dict.get('n_iter', config_bo_params_dict.get('n_iterations'))
        if n_iter_val is not None:
            mapped_config['n_iterations'] = n_iter_val

        n_init_val = config_bo_params_dict.get('init_points_random', config_bo_params_dict.get('n_initial_points'))
        if n_init_val is not None:
            mapped_config['n_initial_points'] = n_init_val
        
        # Combine in the correct order: defaults -> mapped config -> kwargs
        final_params = defaults.copy()
        final_params.update(mapped_config)
        final_params.update(kwargs)

        return from_dict_to_dataclass(cls, final_params)

@dataclasses.dataclass
class OperationalParameters:
    """Parameters related to overall project operations and timeline."""
    project_lifetime_years: int = 15
    target_recovery_factor: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "Enter a value as a fraction (e.g., 0.25 for 25%) to make the optimizer target this specific Recovery Factor."}
    )
    target_objective_name: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Name of the objective to target (e.g., 'recovery_factor', 'npv')."}
    )
    target_objective_value: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "The specific value to target for the chosen objective. For RF, use a fraction (0.0 to 1.0)."}
    )
    target_seeking_sharpness: float = dataclasses.field(
        default=500.0,
        metadata={"help": "Controls how sharply the optimizer penalizes deviation from the target. Higher is sharper."}
    )
    recovery_model_selection: str = dataclasses.field(
        default='hybrid',
        metadata={"help": "Recovery model to use: 'empirical', 'physics_based', or 'hybrid'."}
    )

    def __post_init__(self):
        if not (1 <= self.project_lifetime_years <= 100):
            raise ValueError("Project Lifetime must be between 1 and 100 years.")
        
        # Consolidate target settings for backward compatibility with the UI
        if self.target_recovery_factor is not None and self.target_recovery_factor > 0:
            self.target_objective_name = "recovery_factor"
            self.target_objective_value = self.target_recovery_factor
        
        if self.target_objective_name and self.target_objective_value is None:
            logging.warning("target_objective_name is set, but target_objective_value is not. Target seeking is disabled.")
        if self.target_objective_name and self.target_objective_name not in ["recovery_factor", "npv", "co2_utilization"]:
            raise ValueError("target_objective_name must be one of 'recovery_factor', 'npv', or 'co2_utilization'.")
        if not (10 <= self.target_seeking_sharpness <= 10000):
            raise ValueError("Target Seeking Sharpness must be between 10 and 10,000.")
        
        valid_models = ['simple', 'miscible', 'immiscible', 'hybrid', 'koval', 'layered']
        if self.recovery_model_selection not in valid_models:
            raise ValueError(f"recovery_model_selection must be one of: {', '.join(valid_models)}")

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
    # NEW FIELDS for CO2 recycling model
    co2_breakthrough_year_fraction: float = 0.25
    co2_production_ratio_after_breakthrough: float = 0.5
    co2_recycling_efficiency_fraction: float = 0.9
    warn_if_defaults_used: bool = True

    def __post_init__(self):
        """Validates profile parameters and ensures logical consistency."""
        valid_oil_profiles = [
            "linear_distribution", "plateau_linear_decline",
            "plateau_exponential_decline", "plateau_hyperbolic_decline", "custom_fractions"
        ]
        if self.oil_profile_type not in valid_oil_profiles:
            raise ValueError(f"Invalid oil profile type. Choose one of: {valid_oil_profiles}")

        # --- Validate new CO2 recycling parameters ---
        if not (0.0 < self.co2_breakthrough_year_fraction < 1.0):
            raise ValueError("CO2 Breakthrough Fraction must be between 0.0 and 1.0 (exclusive).")
        if not (0.0 <= self.co2_production_ratio_after_breakthrough <= 1.0):
            raise ValueError("CO2 Production Ratio must be between 0.0 and 1.0.")
        if not (0.0 <= self.co2_recycling_efficiency_fraction <= 1.0):
            raise ValueError("CO2 Recycling Efficiency must be between 0.0 and 1.0.")
            
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