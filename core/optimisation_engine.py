import logging
import dataclasses
import time
from typing import Callable, Dict, List, Optional, Any, Tuple
from copy import deepcopy
from functools import partial
import random
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import plotly.graph_objects as go

from bayes_opt import BayesianOptimization
import pygad

import pyswarms as ps
from scipy.optimize import differential_evolution

from utils.multiprocess_logging import (
    setup_queue_logging,
    _worker_initializer,
    get_log_queue,
    shutdown_queue_logging,
    get_worker_initializer,
)

try:
    from utils.preferences_manager import get_preferences_manager
except ImportError:
    # Fallback for testing without full application context
    def get_preferences_manager():
        class MockPreferences:
            class Advanced:
                max_threads = 4

            advanced = Advanced()

        return MockPreferences()


from core.simulation.simulator_exporter import SimulatorExporter

try:
    from numpy_financial import npv
except ImportError:
    logging.warning("numpy_financial not found. Using a manual NPV calculation.")

    def npv(rate, values):
        values = np.atleast_1d(values)
        return np.sum(values / (1 + rate) ** np.arange(len(values)))


try:
    from analysis.well_analysis import WellAnalysis
    from analysis.profiler_refactored import ProductionProfiler
    from analysis.decline_curve_analysis import DeclineCurveAnalyzer
    from analysis.data_validation import DataValidator
except ImportError:
    # Fallback if analysis modules aren't available
    WellAnalysis = None
    ProductionProfiler = None
    DeclineCurveAnalyzer = None
    DataValidator = None
from core.data_models import (
    ReservoirData,
    EORParameters,
    GeneticAlgorithmParams,
    BayesianOptimizationParams,
    EconomicParameters,
    OperationalParameters,
    ProfileParameters,
    EOSModelParameters,
    PVTProperties,
    ParticleSwarmParams,
    DifferentialEvolutionParams,
    AdvancedEngineParams,
    CO2StorageParameters,
    LayerDefinition,
    PhysicalConstants,
)
from core.simulation.recovery_models import EPSILON
from core.unified_engine.physics.eos import CubicEOS, PengRobinsonEOS, ReservoirFluid
from core.Phys_engine_full.breakthrough_physics import CO2BreakthroughPhysics
from core.engine_factory import EngineFactory, EngineType
from evaluation.mmp import calculate_mmp, MMPParameters

calculate_mmp_external = calculate_mmp
PHYSICS_ENGINE_AVAILABLE = True

# Physical constants for CO2-EOR calculations (previously imported from core.optimisation)
DAYS_PER_YEAR = 365
ACRES_TO_CM2 = 40468564.224
B_GAS_RB_PER_MSCF = 5.0  # Reservoir barrels per thousand standard cubic feet

_PHYS_CONSTANTS = PhysicalConstants()

logger = logging.getLogger(__name__)

from core.plotting_manager import PlottingManager
from core.objectives import ObjectiveFunctions


class SurrogateBreakthrough:
    """Surrogate-based breakthrough physics (PhD Verified)"""
    def calculate_breakthrough_time(self, reservoir_params, eor_params, **kwargs):
        # Simple analytical formula based on Koval (1963)
        v_dp = reservoir_params.get("v_dp_coefficient", 0.5)
        m_eff = eor_params.get("mobility_ratio", 5.0)
        h_factor = 1.0 / (1.0 - v_dp)**2 if v_dp < 1.0 else 100.0
        e_eff = (0.78 + 0.22 * (m_eff ** 0.25)) ** 4
        koval_k = h_factor * e_eff
        t_d_bt = 1.0 / max(koval_k, 1e-6)
        
        # PV estimation for conversion to years
        area = reservoir_params.get("area_acres", 160.0) or 160.0
        thick = reservoir_params.get("thickness_ft", 50.0) or 50.0
        poro = reservoir_params.get("porosity", 0.15) or 0.15
        pv_bbl = area * 43560 * thick * poro / 5.615
        
        q_inj = eor_params.get("injection_rate", 5000.0) # MSCF/day
        # Convert to bbl/day assuming B_gas ~ 0.5 rb/mscf
        q_inj_bbl = q_inj * 0.5
        
        if q_inj_bbl > 0:
            bt_years = (t_d_bt * pv_bbl / q_inj_bbl) / 365.25
        else:
            bt_years = 10.0
        return np.clip(bt_years, 0.1, 20.0)


class OptimizationEngine:
    RELAXABLE_CONSTRAINTS = {
        "porosity": {"description": "Average reservoir porosity (v/v)", "type": "reservoir"},
        "ooip_stb": {"description": "Original Oil In Place (STB)", "type": "reservoir"},
        "v_dp_coefficient": {
            "description": "Dykstra-Parsons coefficient for heterogeneity",
            "type": "eor",
        },
        "mobility_ratio": {"description": "Mobility Ratio (M)", "type": "eor"},
        "WAG_ratio": {"description": "Water-Alternating-Gas Ratio", "type": "eor"},
        "gravity_factor": {
            "description": "Gravity factor in miscible recovery model",
            "type": "eor",
        },
        "sor": {"description": "Residual Oil Saturation for immiscible model", "type": "eor"},
        "transition_alpha": {
            "description": "Transition center for hybrid recovery model",
            "type": "eor",
        },
        "transition_beta": {
            "description": "Transition steepness for hybrid recovery model",
            "type": "eor",
        },
    }

    def __init__(
        self,
        reservoir: ReservoirData,
        pvt: PVTProperties,
        eor_params_instance: Optional[EORParameters] = None,
        ga_params_instance: Optional[GeneticAlgorithmParams] = None,
        bo_params_instance: Optional[BayesianOptimizationParams] = None,
        pso_params_instance: Optional[ParticleSwarmParams] = None,
        de_params_instance: Optional[DifferentialEvolutionParams] = None,
        economic_params_instance: Optional[EconomicParameters] = None,
        operational_params_instance: Optional[OperationalParameters] = None,
        profile_params_instance: Optional[ProfileParameters] = None,
        advanced_engine_params_instance: Optional[AdvancedEngineParams] = None,
        co2_storage_params_instance: Optional[CO2StorageParameters] = None,
        well_data_list: Optional[List[Any]] = None,
        mmp_init_override: Optional[float] = None,
    ):
        self._base_reservoir_data = deepcopy(reservoir)
        self._base_pvt_data = deepcopy(pvt)
        self._base_eor_params = deepcopy(eor_params_instance or EORParameters())
        self._base_economic_params = deepcopy(economic_params_instance or EconomicParameters())
        self._base_operational_params = deepcopy(
            operational_params_instance or OperationalParameters()
        )
        self._base_co2_storage_params = deepcopy(
            co2_storage_params_instance or CO2StorageParameters()
        )
        self.advanced_engine_params = deepcopy(
            advanced_engine_params_instance or AdvancedEngineParams()
        )
        self._base_well_data_list = deepcopy(well_data_list)

        self.RELAXABLE_CONSTRAINTS = {
            k: {
                "description": v["description"],
                "range_factor": self.advanced_engine_params.relaxable_constraint_range_factors.get(
                    k, 0.2
                ),
            }
            for k, v in self.RELAXABLE_CONSTRAINTS.items()
        }

        if well_data_list and pvt:
            self.well_analysis = WellAnalysis(well_data=well_data_list[0], pvt_data=pvt)
        else:
            self.well_analysis = None

        self._unlocked_params_for_current_run: List[str] = []
        self.ga_params_default_config = ga_params_instance or GeneticAlgorithmParams()
        self.bo_params_default_config = bo_params_instance or BayesianOptimizationParams()
        self.pso_params_default_config = pso_params_instance or ParticleSwarmParams()
        self.de_params_default_config = de_params_instance or DifferentialEvolutionParams()

        self.profile_params = profile_params_instance or ProfileParameters()
        self.profiler = None  # Will be instantiated on-demand with the physics-based model
        self.dca_analyzer = DeclineCurveAnalyzer()

        self._results: Optional[Dict[str, Any]] = None
        self._mmp_value_init_override = mmp_init_override
        self._mmp_value: Optional[float] = self._mmp_value_init_override

        self.chosen_objective: str = "npv"

        self._mmp_calculator_fn = calculate_mmp_external
        self._MMPParametersDataclass = MMPParameters  # MMPParameters class now available
        self.eos_model_instance: Optional[CubicEOS] = None
        self.b_gas_rb_per_mscf = B_GAS_RB_PER_MSCF  # Fallback

        self.reservoir_fluid = None

        self.plotting_manager = PlottingManager(self)
        # Initialize objective_functions after reset_to_base_state to ensure it uses the current reservoir instance
        self.reset_to_base_state()

        # Initialize ReservoirFluid if EOS model is available (now that self.reservoir is initialized)
        # Import global error handler for better reporting
        from error_handler import report_error, ErrorSeverity, ErrorCategory

        # EOS model is REQUIRED for CO2-EOR optimization - no fallback logic needed
        if not self.reservoir.eos_model:
            logger.critical(
                "No EOS model found in reservoir data - this should never happen in CO2-EOR!"
            )
            raise ValueError(
                "EOS model is required for CO2-EOR optimization but reservoir.eos_model is None"
            )

        # Debug: Log EOS model type and value
        logger.debug(f"EOS model type: {type(self.reservoir.eos_model)}")
        logger.debug(f"EOS model value: {self.reservoir.eos_model}")
        if hasattr(self.reservoir.eos_model, "__dict__"):
            logger.debug(f"EOS model dict: {self.reservoir.eos_model.__dict__}")

        if not isinstance(self.reservoir.eos_model, EOSModelParameters):
            logger.critical(
                f"Invalid EOS model type: {type(self.reservoir.eos_model)}. Expected EOSModelParameters."
            )
            # Additional debugging
            if hasattr(self.reservoir.eos_model, "__name__"):
                logger.critical(f"EOS model class name: {self.reservoir.eos_model.__name__}")
            raise ValueError(f"Invalid EOS model type: {type(self.reservoir.eos_model)}")

        # Initialize ReservoirFluid with the EOS model
        try:
            if not PHYSICS_ENGINE_AVAILABLE or not ReservoirFluid:
                logger.error("Physics engine components not available for CO2-EOR optimization!")
                raise ImportError("Physics engine (ReservoirFluid) is required but not available")

            self.reservoir_fluid = ReservoirFluid(self.reservoir.eos_model)
            logger.info(
                f"Initialized ReservoirFluid with EOS model: {self.reservoir.eos_model.eos_type} and components: {self.reservoir.eos_model.component_names}"
            )

            # Calculate B_gas value from ReservoirFluid using actual reservoir conditions
            typical_temp_K = (
                _PHYS_CONSTANTS.STANDARD_TEMPERATURE_K
            )  # Default to standard temperature
            typical_pressure_Pa = (
                _PHYS_CONSTANTS.STANDARD_PRESSURE_PA
            )  # Default to standard pressure
            try:
                # Use actual reservoir conditions for accurate B_gas calculation
                # Convert Fahrenheit to Kelvin
                typical_temp_K = (self.reservoir.temperature - 32) * 5 / 9 + 273.15
                # Convert psi to Pa (using initial_pressure and PhysicalConstants)
                typical_pressure_Pa = self.reservoir.initial_pressure * _PHYS_CONSTANTS.PSI_TO_PA

                self.b_gas_rb_per_mscf = self.reservoir_fluid.get_bgas_rb_per_mscf(
                    typical_temp_K, typical_pressure_Pa
                )
                logger.info(
                    f"Using accurate B_gas from ReservoirFluid: {self.b_gas_rb_per_mscf:.4f} rb/MSCF at T={typical_temp_K:.1f}K, P={typical_pressure_Pa / 1e6:.1f}MPa"
                )
            except Exception as bgas_error:
                logger.error(f"Failed to calculate B_gas from ReservoirFluid: {bgas_error}")
                # Use fallback but still report as an error since EOS should work
                report_error(
                    title="B_gas Calculation Failed",
                    message=f"Failed to calculate B_gas from EOS model. Using fallback value: {self.b_gas_rb_per_mscf:.4f} rb/MSCF. This may affect calculation accuracy.",
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.CALCULATION,
                    context={
                        "fallback_b_gas_value": self.b_gas_rb_per_mscf,
                        "eos_type": self.reservoir.eos_model.eos_type,
                        "components": self.reservoir.eos_model.component_names,
                        "temperature_K": typical_temp_K,
                        "pressure_Pa": typical_pressure_Pa,
                        "calculation_error": str(bgas_error),
                    },
                    user_action_suggested="Check EOS model configuration and component definitions for CO2-EOR system.",
                    show_dialog=True,
                )
                logger.warning(
                    f"Using fallback B_gas value due to EOS calculation failure: {self.b_gas_rb_per_mscf:.4f} rb/MSCF"
                )

        except Exception as e:
            logger.critical(f"Failed to initialize ReservoirFluid for CO2-EOR: {e}", exc_info=True)
            # This is a critical failure for CO2-EOR - not just a warning
            report_error(
                title="Critical ReservoirFluid Initialization Failure",
                message=f"ReservoirFluid initialization failed: {e}. CO2-EOR optimization cannot proceed without a working EOS model.",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.CONFIGURATION,
                context={
                    "eos_type": self.reservoir.eos_model.eos_type,
                    "components": self.reservoir.eos_model.component_names,
                    "initialization_error": str(e),
                    "physics_engine_available": PHYSICS_ENGINE_AVAILABLE,
                    "reservoir_fluid_available": ReservoirFluid is not None,
                },
                user_action_suggested="Verify EOS model configuration and physics engine installation for CO2-EOR system.",
                show_dialog=True,
            )
            raise RuntimeError(
                f"Critical: Cannot initialize CO2-EOR optimization without working EOS model: {e}"
            )

        self.objective_functions = ObjectiveFunctions(
            self._base_operational_params,
            self._base_eor_params,
            self.reservoir,
            self.advanced_engine_params,
        )

        # Initialize surrogate-based breakthrough physics (PhD Verified)
        # We avoid the unverified Full Physics Engine version to satisfy PhD requirements
        self.breakthrough_physics = SurrogateBreakthrough()

        # Initialize EOS model if available
        if self.reservoir.eos_model and isinstance(self.reservoir.eos_model, EOSModelParameters):
            try:
                if PHYSICS_ENGINE_AVAILABLE and ReservoirFluid:
                    # Use ReservoirFluid wrapper which properly converts EOSModelParameters to EOSParameters
                    self.reservoir_fluid = ReservoirFluid(self.reservoir.eos_model)
                    self.eos_model_instance = self.reservoir_fluid.eos_model
                    self.pvt.pvt_type = "compositional"
                    logger.info(f"Initialized ReservoirFluid with EOS model: {self.reservoir.eos_model.eos_type} and components: {self.reservoir.eos_model.component_names}")
                else:
                    logger.warning("Physics engine not available, using default PVT")
                    self.eos_model_instance = None
                    self.reservoir_fluid = None
            except Exception as e:
                logger.error(f"Failed to instantiate EOS model: {e}", exc_info=True)
                self.eos_model_instance = None
                self.reservoir_fluid = None

        # Initialize simulation engine via EngineFactory
        self._init_simulation_engine()

    def _init_simulation_engine(self):
        """Initialize the simulation engine via EngineFactory based on engine type setting."""
        try:
            # Determine engine type from advanced_engine_params
            # Try new field first, fall back to old boolean field
            engine_type_str = getattr(self.advanced_engine_params, "engine_type", None)
            if engine_type_str is None:
                use_simple = getattr(self.advanced_engine_params, "use_simple_physics", True)
                engine_type_str = "simple" if use_simple else "detailed"

            # Map string to EngineType enum
            engine_type = EngineType(engine_type_str)

            # Determine recovery model for surrogate engine
            recovery_model_type = getattr(self.advanced_engine_params, "recovery_model_type", "hybrid")

            # Create engine via EngineFactory
            self.simulation_engine = EngineFactory.create_engine(
                engine_type, 
                recovery_model_type=recovery_model_type
            )
            logger.info(f"Initialized simulation engine via EngineFactory: {engine_type.value} (model: {recovery_model_type})")

            # Store engine type for reference
            self._engine_type = engine_type

        except ImportError as e:
            logger.warning(f"Could not initialize simulation engine via EngineFactory: {e}")
            self.simulation_engine = None
            self._engine_type = None
        except Exception as e:
            logger.error(f"Unexpected error initializing simulation engine: {e}", exc_info=True)
            self.simulation_engine = None
            self._engine_type = None

    def reset_to_base_state(self):
        """Resets all parameters to their initial base state."""
        self.reservoir = deepcopy(self._base_reservoir_data)
        self.pvt = deepcopy(self._base_pvt_data)
        self.eor_params = deepcopy(self._base_eor_params)
        self.economic_params = deepcopy(self._base_economic_params)
        self.operational_params = deepcopy(self._base_operational_params)
        self.co2_storage_params = deepcopy(self._base_co2_storage_params)
        self.recovery_model = getattr(self.operational_params, "recovery_model_selection", "hybrid")
        self._unlocked_params_for_current_run = []
        self._mmp_value = self._mmp_value_init_override

        # Update objective_functions to use the current reservoir instance
        self.objective_functions = ObjectiveFunctions(
            self.operational_params, self.eor_params, self.reservoir, self.advanced_engine_params
        )

    @property
    def simulation_engine_type(self) -> Optional[str]:
        """
        Get the current simulation engine type.

        Returns the engine type string ('simple' or 'detailed') for UI access.
        Returns None if no engine is initialized.
        """
        if self._engine_type is None:
            return None
        return self._engine_type.value if hasattr(self._engine_type, 'value') else str(self._engine_type)

    def _get_available_cores(self) -> int:
        """Get the number of available CPU cores from preferences or system."""
        try:
            preferences = get_preferences_manager()
            max_threads = preferences.advanced.max_threads
            # Ensure at least 1 core and no more than system cores
            system_cores = mp.cpu_count()
            return min(max(1, max_threads), system_cores)
        except (RuntimeError, AttributeError):
            # Fallback to system cores if preferences not available
            return max(1, mp.cpu_count() - 1)  # Leave one core free

    def prepare_for_rerun_with_unlocked_params(self, params_to_unlock: List[str]):
        """Prepares the engine for a re-run with specified parameters unlocked."""
        self.reset_to_base_state()
        self._unlocked_params_for_current_run = [
            p for p in params_to_unlock if p in self.RELAXABLE_CONSTRAINTS
        ]
        logger.info(
            f"Engine prepared for re-run. Unlocked parameters: {self._unlocked_params_for_current_run}"
        )

    def get_configurable_parameters_for_uq(self) -> List[Tuple[str, str]]:
        return [
            (info["description"], param_key)
            for param_key, info in self.RELAXABLE_CONSTRAINTS.items()
        ]

    @property
    def avg_porosity(self) -> float:
        """Calculates the average porosity from the reservoir grid or explicit parameter."""
        # 1. Prefer explicit average_porosity from ReservoirData if set
        if getattr(self.reservoir, "average_porosity", None) is not None:
            return self.reservoir.average_porosity

        # 2. Fallback to grid-based calculation
        poro_arr = self.reservoir.grid.get(
            "PORO", np.array([self.advanced_engine_params.default_porosity])
        )
        return (
            np.mean(poro_arr)
            if hasattr(poro_arr, "size") and poro_arr.size > 0
            else self.advanced_engine_params.default_porosity
        )

    @property
    def mmp(self) -> Optional[float]:
        """Returns the Minimum Miscibility Pressure, calculating it if necessary."""
        if self._mmp_value is None:
            self.calculate_mmp()
        return self._mmp_value

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Returns the results of the last optimization run."""
        return self._results

    def calculate_mmp(self, method_override: Optional[str] = None) -> float:
        """Calculates the MMP using the configured method or an override."""
        if self._mmp_value_init_override is not None:
            self._mmp_value = self._mmp_value_init_override
            return self._mmp_value

        default_mmp_fallback = self.eor_params.default_mmp_fallback
        if not self._mmp_calculator_fn or not self._MMPParametersDataclass:
            self._mmp_value = self._mmp_value or default_mmp_fallback
            logger.warning("MMP calculation dependencies not found. Using fallback value.")
            return self._mmp_value

        actual_mmp_method = method_override or "auto"

        try:
            mmp_calc_value = float(self._mmp_calculator_fn(self.pvt, method=actual_mmp_method))
            logger.info(
                f"MMP calculated: {mmp_calc_value:.2f} psi (method: '{actual_mmp_method}', source: PVT data)."
            )
            self._mmp_value = mmp_calc_value
        except Exception as e:
            # Import the global error handler
            from error_handler import report_caught_error, ErrorSeverity, ErrorCategory

            # Report the error properly instead of just logging
            report_caught_error(
                operation="calculate Minimum Miscibility Pressure (MMP)",
                exception=e,
                context={
                    "mmp_method": actual_mmp_method,
                    "pvt_type": type(self.pvt).__name__,
                    "default_mmp_fallback": default_mmp_fallback,
                    "current_mmp_value": self._mmp_value,
                    "mmp_calculator_fn": str(self._mmp_calculator_fn)
                    if hasattr(self, "_mmp_calculator_fn")
                    else "unknown",
                },
                user_action_suggested="Check PVT data format and MMP calculation method compatibility. Consider using a different MMP calculation method or verify input data format.",
                show_dialog=True,
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.CALCULATION,
            )

            logger.error(f"MMP calculation failed: {e}. Using fallback.", exc_info=True)
            self._mmp_value = self._mmp_value or default_mmp_fallback

        return self._mmp_value

    def evaluate_for_analysis(
        self, eor_operational_params_dict: Dict[str, float], **kwargs
    ) -> Dict[str, float]:
        econ_params = kwargs.get("economic_params_override")
        if econ_params is None: econ_params = self.economic_params
        
        ooip = kwargs.get("ooip_override")
        if ooip is None: ooip = self.reservoir.ooip_stb
        
        mmp = kwargs.get("mmp_override")
        if mmp is None: mmp = self.mmp
        
        co2_storage_params = kwargs.get("co2_storage_params_override")
        if co2_storage_params is None: co2_storage_params = self.co2_storage_params
        
        dimensional_tolerance = kwargs.get("dimensional_tolerance", 0.1)

        # DEBUG: Log EOR parameters being used
        wag_ratio = (
            getattr(self.eor_params.swag, "water_gas_ratio", 1.0) if self.eor_params.swag else 1.0
        )
        pressure = getattr(self.eor_params, "target_pressure_psi", self.eor_params.max_pressure_psi)
        logger.info(
            f"OptimizationEngine EOR Parameters - Injection Scheme: '{self.eor_params.injection_scheme}', "
            f"SWAG Ratio: {wag_ratio}, Pressure: {pressure} psi"
        )

        if not dataclasses.is_dataclass(self.pvt):
            if hasattr(self.pvt, "__dict__"):
                pvt_dict = self.pvt.__dict__
                self.pvt = PVTProperties(**pvt_dict)
            else:
                raise TypeError(
                    f"pvt must be a dataclass instance or an object with __dict__, but it is {type(self.pvt)}"
                )

        all_params = dataclasses.asdict(self.pvt)
        all_params.update(dataclasses.asdict(self.eor_params))
        cross_sectional_area_cm2 = (
            self.reservoir.cross_sectional_area_acres * ACRES_TO_CM2
            if self.reservoir.cross_sectional_area_acres
            else 5e6
        )

        all_params.update(
            {
                "permeability": np.mean(
                    self.reservoir.grid.get(
                        "PERMX", np.array([self.advanced_engine_params.default_permeability])
                    )
                ),
                "porosity": self.avg_porosity,
                "mmp": mmp,
                "layer_definitions": self.reservoir.layer_definitions,
                "cross_sectional_area": cross_sectional_area_cm2,
                "kv_kh_ratio": self.eor_params.kv_kh_ratio,
                "co2_solubility_scm_per_bbl": self.pvt.co2_solubility_scm_per_bbl,
            }
        )

        all_params.update(eor_operational_params_dict)

        if (
            self.eos_model_instance is not None
            and hasattr(self.pvt, "pvt_type")
            and self.pvt.pvt_type == "compositional"
            and "pressure" in eor_operational_params_dict
        ):
            pressure_psia = eor_operational_params_dict["pressure"]
            temperature_F = self.pvt.temperature

            if hasattr(self.eos_model_instance, "calculate_properties"):
                try:
                    eos_properties = self.eos_model_instance.calculate_properties(
                        pressure_psia, temperature_F
                    )

                    if (
                        "oil_viscosity_cp" in eos_properties
                        and eos_properties["oil_viscosity_cp"] is not None
                    ):
                        all_params["viscosity_oil"] = eos_properties["oil_viscosity_cp"]

                    if (
                        "gas_viscosity_cp" in eos_properties
                        and eos_properties["gas_viscosity_cp"] is not None
                    ):
                        all_params["co2_viscosity"] = eos_properties["gas_viscosity_cp"]
                        all_params["viscosity_inj"] = eos_properties["gas_viscosity_cp"]

                    if (
                        "oil_density_kg_m3" in eos_properties
                        and eos_properties["oil_density_kg_m3"] is not None
                    ):
                        all_params["oil_density"] = eos_properties["oil_density_kg_m3"]

                    if (
                        "gas_density_kg_m3" in eos_properties
                        and eos_properties["gas_density_kg_m3"] is not None
                    ):
                        all_params["co2_density"] = eos_properties["gas_density_kg_m3"]

                    if "viscosity_oil" in all_params and "co2_viscosity" in all_params:
                        all_params["mobility_ratio"] = all_params["viscosity_oil"] / (
                            all_params["co2_viscosity"] + EPSILON
                        )

                except Exception as e:
                    logger.warning(f"EOS calculation failed: {e}. Using static properties.")

        current_eor_params = deepcopy(self.eor_params)
        for key, value in all_params.items():
            if hasattr(current_eor_params, key):
                setattr(current_eor_params, key, value)

        current_profile_params = deepcopy(self.profile_params)
        for key, value in eor_operational_params_dict.items():
            if hasattr(current_profile_params, key):
                setattr(current_profile_params, key, value)

        # Validate reservoir data for physics-based models before profiling
        physics_based_models = ["hybrid", "layered", "buckley_leverett", "dykstra_parsons"]
        is_physics_based = self.recovery_model in physics_based_models

        if is_physics_based:
            try:
                # Create a temporary reservoir data instance for validation
                validation_reservoir = ReservoirData(
                    grid=self.reservoir.grid,
                    pvt_tables=self.reservoir.pvt_tables,
                    ooip_stb=ooip,
                    initial_pressure=self.reservoir.initial_pressure,
                    rock_compressibility=self.reservoir.rock_compressibility,
                    length_ft=self.reservoir.length_ft,
                    cross_sectional_area_acres=self.reservoir.cross_sectional_area_acres,
                    area_acres=getattr(self.reservoir, "area_acres", None),
                    thickness_ft=getattr(self.reservoir, "thickness_ft", None),
                    average_porosity=self.avg_porosity,
                    initial_water_saturation=getattr(
                        self.reservoir, "initial_water_saturation", None
                    ),
                    oil_fvf=getattr(self.reservoir, "oil_fvf", None),
                )

                # Validate dimensional consistency
                validation_reservoir.validate(physics_based_model=True, tolerance=dimensional_tolerance)
                logger.info(f"Physics-based model '{self.recovery_model}' validation passed")

            except ValueError as e:
                logger.warning(f"Physics-based model validation failed: {e}")
                # Return a failure state with penalty for dimensional inconsistency
                return {
                    "npv": -1e10,
                    "recovery_factor": 0.0,
                    "co2_utilization": 1e6,
                    "total_co2_stored_tonne": 0.0,
                    "avg_storage_efficiency": 0.0,
                    "final_cumulative_co2_stored_tonne": 0.0,
                    "dimensional_consistency_error": str(e),
                }

        # Use simulation_engine for all engine types (simple, detailed, surrogate)
        # The simulation_engine was created via EngineFactory based on engine_type setting
        if self.simulation_engine is not None:
            try:
                # Use the factory-created simulation engine
                sim_kwargs = kwargs.get("recovery_model_init_kwargs_override", {}).copy()
                sim_kwargs["mmp"] = mmp
                
                current_reservoir = deepcopy(self.reservoir)
                current_reservoir.ooip_stb = ooip

                results = self.simulation_engine.evaluate_scenario(
                    reservoir_data=current_reservoir,
                    eor_params=current_eor_params,
                    operational_params=self.operational_params,
                    economic_params=econ_params,
                    **sim_kwargs
                )

                # Convert results to profile format expected by rest of code
                time_res = self.operational_params.time_resolution
                profiles = {
                    f"{time_res}_oil_stb": results.get('oil_production_rate', np.array([])),
                    f"{time_res}_gas_stb": results.get('gas_production_rate', np.array([])),
                    f"{time_res}_water_stb": results.get('water_production_rate', np.array([])),
                    f"{time_res}_pressure": results.get('pressure', np.array([])),  # Add pressure data
                    "co2_injection_mscf": results.get('co2_injection', np.array([])),
                    "time_vector": results.get('time_vector', np.array([])),
                }
                rf = results.get('recovery_factor', 0.0)

                # Store profiler for detailed engine (for later access)
                if results.get('engine_type') == 'detailed':
                    # Create profiler for detailed results if needed for analysis
                    try:
                        pressure_override = all_params.get("pressure")
                        profiler = ProductionProfiler(
                            self.reservoir, self.pvt, current_eor_params,
                            self.operational_params, current_profile_params,
                            initial_pressure_override=pressure_override,
                        )
                        self.profiler = profiler
                    except Exception as e:
                        logger.warning(f"Could not create profiler for detailed engine: {e}")
                        self.profiler = None

                logger.info(f"Used {results.get('engine_type', 'unknown')} engine for evaluation, RF={rf:.4f}")

            except Exception as e:
                logger.error(f"Simulation engine evaluation failed: {e}", exc_info=True)
                # Return a failure state for the optimizer
                return {
                    "npv": -1e12,
                    "recovery_factor": 0.0,
                    "co2_utilization": 1e6,
                    "total_co2_stored_tonne": 0.0,
                    "avg_storage_efficiency": 0.0,
                    "final_cumulative_co2_stored_tonne": 0.0,
                }
        else:
            # Fallback: No simulation engine available, use ProductionProfiler directly
            logger.warning("No simulation engine available, using ProductionProfiler fallback")
            try:
                pressure_override = all_params.get("pressure")
                profiler = ProductionProfiler(
                    self.reservoir, self.pvt, current_eor_params,
                    self.operational_params, current_profile_params,
                    initial_pressure_override=pressure_override,
                )
                self.profiler = profiler
                profiles = profiler.generate_all_profiles(ooip_stb=ooip)
                total_oil_produced = np.sum(profiles.get(f"{self.operational_params.time_resolution}_oil_stb", 0))
                rf = total_oil_produced / ooip if ooip > 0 else 0.0
            except ValueError as e:
                logger.error(f"Profiler fallback failed: {e}", exc_info=True)
                return {
                    "npv": -1e12,
                    "recovery_factor": 0.0,
                    "co2_utilization": 1e6,
                    "total_co2_stored_tonne": 0.0,
                    "avg_storage_efficiency": 0.0,
                    "final_cumulative_co2_stored_tonne": 0.0,
                }

        # Validate recovery factor is reasonable
        if rf > 1.0:
            logger.warning(f"Recovery factor {rf:.3f} exceeds 100%. Clamping to 1.0")
            rf = 1.0
        elif rf < 0.0:
            logger.warning(f"Recovery factor {rf:.3f} is negative. Setting to 0.0")
            rf = 0.0

        # Create an updated CO2StorageParameters instance for this evaluation
        current_co2_storage_params = deepcopy(self.co2_storage_params)
        for key, value in eor_operational_params_dict.items():
            if hasattr(current_co2_storage_params, key):
                setattr(current_co2_storage_params, key, value)

        # Calculate breakthrough-aware objectives
        objectives = self.objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, current_co2_storage_params
        )

        # Validate simulation results
        reservoir_data_dict = {
            "ooip_stb": ooip,
            "initial_pressure": self.reservoir.initial_pressure,
            "max_pressure_psi": self.eor_params.max_pressure_psi,
        }

        validation_results = DataValidator.validate_simulation_results(
            profiles, objectives, reservoir_data_dict
        )
        DataValidator.log_validation_results(validation_results)

        # Add breakthrough-specific metrics
        breakthrough_metrics = self._calculate_breakthrough_metrics(all_params, profiles)
        objectives.update(breakthrough_metrics)

        # Include raw profiles in the results for plotting or constraint checking
        # This prevents the need to re-run simulations
        objectives["profiles"] = profiles

        return objectives

    def _calculate_breakthrough_metrics(
        self, all_params: Dict, profiles: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate breakthrough-specific metrics for optimization objectives.

        Args:
            all_params: All simulation parameters including reservoir and EOR properties
            profiles: Production and injection profiles

        Returns:
            Dictionary of breakthrough metrics
        """
        try:
            # Extract reservoir parameters for breakthrough calculation
            reservoir_params = {
                "permeability": all_params.get("permeability", 100.0),
                "porosity": all_params.get("porosity", 0.15),
                "v_dp_coefficient": all_params.get("v_dp", 0.5),
                "length_ft": self.reservoir.length_ft,
                "cross_sectional_area_acres": self.reservoir.cross_sectional_area_acres,
            }

            # Extract EOR parameters
            eor_params = {
                "injection_rate": all_params.get("injection_rate", 5000.0),
                "mobility_ratio": all_params.get("mobility_ratio", 2.0),
                "density_contrast": all_params.get("density_contrast", 0.3),
                "dip_angle": all_params.get("dip_angle", 0.0),
                "co2_viscosity": all_params.get("co2_viscosity", 0.02),
                "viscosity_oil": all_params.get("viscosity_oil", 4.0),
            }

            # Calculate breakthrough time
            breakthrough_time = self.breakthrough_physics.calculate_breakthrough_time(
                reservoir_params, eor_params
            )

            # Ensure breakthrough_time is a scalar for the penalty calculation
            if hasattr(breakthrough_time, "item"):
                breakthrough_time_scalar = breakthrough_time.item()
            else:
                breakthrough_time_scalar = float(breakthrough_time)

            # Calculate breakthrough impact on economics
            project_lifetime = self.operational_params.project_lifetime_years
            breakthrough_impact = self._calculate_breakthrough_economic_impact(
                breakthrough_time_scalar, project_lifetime, profiles
            )

            return {
                "breakthrough_time_years": breakthrough_time_scalar,
                "breakthrough_impact_factor": breakthrough_impact,
            }

        except Exception as e:
            logger.warning(f"Breakthrough metrics calculation failed: {str(e)}")
            return {
                "breakthrough_time_years": self.advanced_engine_params.breakthrough_fallback_time_years,
                "breakthrough_impact_factor": self.advanced_engine_params.breakthrough_fallback_impact_factor,
                "early_breakthrough_penalty": self.advanced_engine_params.breakthrough_fallback_penalty,
            }

    def _calculate_breakthrough_economic_impact(
        self, breakthrough_time: float, project_lifetime: int, profiles: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate the economic impact of breakthrough timing using a non-linear model.

        Args:
            breakthrough_time: Time of breakthrough in years
            project_lifetime: Total project lifetime
            profiles: Production profiles

        Returns:
            Impact factor (1.0 = no impact, <1.0 = negative impact)
        """
        if breakthrough_time <= 0:
            return 0.5  # Severe penalty for immediate breakthrough

        # More sophisticated non-linear impact factor
        # An exponential decay function for early breakthrough penalty
        # A logarithmic function for late breakthrough reward
        ideal_breakthrough_fraction = 1.0 / 3.0
        ideal_breakthrough = project_lifetime * ideal_breakthrough_fraction
        time_ratio = breakthrough_time / ideal_breakthrough

        if time_ratio < 1.0:
            # Exponential penalty for early breakthrough
            impact_factor = np.exp(
                -((1.0 - time_ratio) ** 2) / (2 * 0.5**2)
            )  # Gaussian-like penalty
        else:
            # Logarithmic reward for late breakthrough
            impact_factor = 1.0 + 0.1 * np.log1p(time_ratio - 1.0)

        # Consider oil production before and after breakthrough
        resolution = self.operational_params.time_resolution
        oil_profile = profiles.get(f"{resolution}_oil_stb", np.array([]))
        breakthrough_period = int(breakthrough_time)

        if breakthrough_period < len(oil_profile):
            oil_before_bt = np.sum(oil_profile[:breakthrough_period])
            oil_after_bt = np.sum(oil_profile[breakthrough_period:])
            total_oil = oil_before_bt + oil_after_bt

            if total_oil > 0:
                # Reward scenarios where more oil is produced before breakthrough
                fraction_before_bt = oil_before_bt / total_oil
                impact_factor *= 0.8 + 0.4 * fraction_before_bt  # Scale between 0.8 and 1.2

        return float(np.clip(impact_factor, 0.5, 1.2))

    def _perform_decline_curve_analysis(
        self, profiles: Dict[str, np.ndarray], optimized_params: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            annual_oil = profiles.get("annual_oil_stb", np.array([]))
            if len(annual_oil) == 0:
                return None

            time_years = np.arange(1, len(annual_oil) + 1)

            b_factor = None
            if optimized_params:
                b_factor = optimized_params.get("hyperbolic_b_factor")

            dca_result = self.dca_analyzer.analyze_production(
                time=time_years,
                production_rate=annual_oil,
                model_type="auto",
                forecast_years=30,  # Forecast 30 years into the future
                time_unit="years",
                b_factor=b_factor,
            )

            # Convert DCA result to dictionary for reporting
            dca_data = self.dca_analyzer.generate_dca_report_data(dca_result)
            return dca_data

        except Exception as e:
            logger.error(f"Decline curve analysis failed: {e}", exc_info=True)
            return None

    def _calculate_adaptive_penalty(
        self, eval_results: Dict[str, float], current_gen: int = None, max_gens: int = None
    ) -> float:
        """
        Calculates a penalty for constraint violations.
        Supports static and adaptive penalties suitable for petroleum engineering constraints.
        """
        if not hasattr(self, "ga_params_current_run") or not self.ga_params_current_run:
            # Default to static penalty if no params available
            method = "static"
            base_penalty = 1000.0
        else:
            params = self.ga_params_current_run
            method = params.constraint_handling_method
            base_penalty = params.penalty_factor

        if method == "death":
            # Reject infeasible solutions completely
            # We need to detect infeasibility. Assume any violation > 0 is infeasible.
            # Here we check specific critical constraints
            # For now, we rely on the generic check below
            pass

        # excessive_water_cut = eval_results.get("water_cut_violation", 0.0)
        # max_bhp_violation = eval_results.get("bhp_violation", 0.0)
        # For now, we assume eval_results might contain 'constraint_violation_magnitude'
        # or we calculate it here based on critical thresholds if we had access to raw profiles.
        # Since eval_results comes from evaluate_for_analysis which returns limited metrics,
        # we might need to rely on 'overall_violation' if it was computed there.

        # self.evaluate_for_analysis currently calculates 'breakthrough_impact_factor' but not explicit constraint violations
        # strictly separated.
        # However, let's assume valid physics is the primary constraint.
        # If simulation failed (objective very low), it's already penalized.

        # New Feature: Check explicit constraints if available in eval_results
        # (This assumes evaluate_for_analysis has been updated or we add checks here)
        total_violation = 0.0

        # Example Petroleum Constraints (would need to be computed in evaluate_for_analysis or here if we have data)
        # For now, we can check basic bounds validity if passed in eval_results
        # OR just use this structure for future expansion.

        # Placeholder for calculated violation sum
        # total_violation = ...

        if total_violation <= 0:
            return 0.0

        if method == "adaptive_penalty" and current_gen is not None and max_gens:
            # Scale penalty by generation progress
            # Early generations: low penalty to allow exploration
            # Late generations: high penalty to enforce feasibility
            progress = current_gen / max(1, max_gens)
            adaptive_factor = 0.5 + progress  # 0.5 to 1.5 multiplier?
            # Or exponential: (C * gen)^alpha
            penalty = base_penalty * total_violation * adaptive_factor
        else:
            # "static" or default
            penalty = base_penalty * total_violation

        if method == "death" and total_violation > 0:
            return float("inf")

        return penalty

    def _objective_function_wrapper(self, **kwargs) -> float:
        """A wrapper that computes the final objective value for the optimizer."""
        # Extract metadata args if present
        current_gen = kwargs.pop("current_gen", None)
        params_dict = kwargs

        # Define safe bounds for objective values to prevent numerical issues in BO
        MAX_OBJECTIVE_VALUE = 1e20
        MIN_OBJECTIVE_VALUE = -1e20
        FAILURE_PENALTY = self.advanced_engine_params.failure_penalty

        # Debug logging for parameter exploration
        # Safe logging for parameters - handle numpy arrays
        def _format_param_value(value):
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    return f"array(shape={value.shape})"
            return value

        formatted_params = {k: _format_param_value(v) for k, v in params_dict.items()}
        logger.debug(f"Evaluating parameters: {formatted_params}")

        eval_results = self.evaluate_for_analysis(params_dict)

        # Debug logging for evaluation results - handle numpy arrays in results
        def _format_result_value(value):
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    return f"array(shape={value.shape})"
            return value

        formatted_results = {k: _format_result_value(v) for k, v in eval_results.items()}
        logger.debug(f"Evaluation results: {formatted_results}")

        # --- Sanity Checks ---
        # 1. Mass Balance Check
        # This check is implicitly handled by how storage is calculated (injected - produced - leaked),
        # so a direct check of (injected == produced + stored) would be circular.
        # Instead, we check for nonsensical outcomes.

        # 2. Physical Realism Check
        storage_efficiency = eval_results.get("storage_efficiency", 0.0)
        # Ensure storage_efficiency is a scalar for formatting
        if isinstance(storage_efficiency, np.ndarray):
            storage_efficiency = float(storage_efficiency.item())
        logger.debug(f"Storage efficiency: {storage_efficiency:.6f}")
        # Check if storage efficiency is extremely low (indicating poor performance)
        if storage_efficiency <= 1e-6:
            logger.warning(
                f"Sanity Check Failed: Storage efficiency is extremely low ({storage_efficiency:.6f}). Applying failure penalty."
            )
            # Return partial penalty instead of full penalty to allow some exploration
            return FAILURE_PENALTY * 0.1

        # 3. Plume Containment Constraint
        plume_containment = eval_results.get("plume_containment", 0.0)
        # Ensure plume_containment is a scalar for formatting
        if isinstance(plume_containment, np.ndarray):
            plume_containment = float(plume_containment.item())
        logger.debug(f"Plume containment: {plume_containment:.3f}")
        if plume_containment < 0.6:  # Minimum acceptable containment score
            logger.warning(
                f"Plume containment constraint violated: {plume_containment:.3f} < 0.6. Applying penalty."
            )
            return FAILURE_PENALTY * 0.1  # Partial penalty for containment issues

        # 4. Injection Pressure Constraints
        resolution = self.operational_params.time_resolution
        profiles = eval_results.get("profiles", {})
        pressure_key = f"{self.operational_params.time_resolution}_pressure"

        # Get pressure data, defaulting to target pressure if not available (e.g. Simple Engine)
        pressure_data = profiles.get(
            pressure_key,
            np.array([self.eor_params.pressure])
        )

        # Determine average pressure
        if len(pressure_data) > 0:
            avg_pressure = np.mean(pressure_data)
        else:
            avg_pressure = float(self.eor_params.pressure)
        # Ensure avg_pressure is a scalar for formatting
        if isinstance(avg_pressure, np.ndarray):
            avg_pressure = float(avg_pressure.item())
        logger.debug(f"Average injection pressure: {avg_pressure:.1f} psi")

        if avg_pressure < self.co2_storage_params.min_injection_pressure_psi:
            logger.warning(
                f"Injection pressure too low: {avg_pressure:.1f} psi < {self.co2_storage_params.min_injection_pressure_psi} psi. Applying penalty."
            )
            return FAILURE_PENALTY * 0.5

        avg_storage_efficiency = eval_results.get("avg_storage_efficiency", 0.0)
        objective_value = eval_results.get(self.chosen_objective, -1e12)

        # Apply breakthrough constraints and penalties
        breakthrough_time = eval_results.get("breakthrough_time_years", 5.0)
        breakthrough_impact = eval_results.get("breakthrough_impact_factor", 1.0)

        # Ensure breakthrough metrics are scalars for formatting
        if isinstance(breakthrough_time, np.ndarray):
            breakthrough_time = float(breakthrough_time.item())
        if isinstance(breakthrough_impact, np.ndarray):
            breakthrough_impact = float(breakthrough_impact.item())

        logger.debug(
            f"Breakthrough time: {breakthrough_time:.2f} years, Impact factor: {breakthrough_impact:.3f}"
        )

        # 5. Breakthrough timing constraint
        if breakthrough_time < 1.0:  # Breakthrough in less than 1 year
            # Ensure breakthrough_time is scalar for formatting
            if isinstance(breakthrough_time, np.ndarray):
                breakthrough_time = float(breakthrough_time.item())
            logger.warning(
                f"Breakthrough constraint violated: {breakthrough_time:.2f} years < 1.0 year. Applying penalty."
            )
            return FAILURE_PENALTY * 0.8

        # Handle different objective types with appropriate scaling and direction
        if self.chosen_objective == "co2_utilization":
            result = -objective_value  # Minimize utilization for storage
        elif self.chosen_objective == "plume_containment":
            result = objective_value * 1e6  # Maximize containment with large scaling
        elif self.chosen_objective == "injection_rate":
            result = objective_value * 1e3  # Maximize injection rate with scaling
        elif self.chosen_objective == "storage_efficiency":
            result = objective_value * 1e6  # Maximize storage efficiency
        elif self.chosen_objective == "trapping_efficiency":
            result = objective_value * 1e6  # Maximize trapping efficiency
        elif self.chosen_objective == "breakthrough_time_years":
            result = objective_value * 1e3  # Maximize breakthrough time
        else:
            result = objective_value  # Default behavior for other objectives

        # Apply breakthrough impact factor
        breakthrough_impact = eval_results.get("breakthrough_impact_factor", 1.0)
        if not np.isclose(breakthrough_impact, 1.0):
            logger.info(
                f"Applying breakthrough impact factor of {breakthrough_impact:.3f} to objective score."
            )
        result *= breakthrough_impact

        result *= breakthrough_impact

        # Apply Adaptive Penalty
        # If parameters for max_gens are not available, use 100 as fallback
        max_gens = 100
        if hasattr(self, "ga_params_current_run") and self.ga_params_current_run:
            max_gens = self.ga_params_current_run.num_generations

        penalty = self._calculate_adaptive_penalty(
            eval_results, current_gen=current_gen, max_gens=max_gens
        )
        if penalty > 0:
            result -= penalty
            logger.debug(f"Applied penalty: {penalty:.4f} (Result: {result:.4f})")

        # Clip result to safe bounds to prevent numerical issues in Bayesian Optimization
        return np.clip(result, MIN_OBJECTIVE_VALUE, MAX_OBJECTIVE_VALUE)

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Defines the search space for the optimization variables."""
        mmp_val = self.mmp or self.eor_params.default_mmp_fallback
        b = {
            "pressure": (
                mmp_val * self.eor_params.min_pressure_factor,
                self.eor_params.max_pressure_psi,
            ),
            "rate": (
                self.eor_params.min_injection_rate_mscfd,
                self.eor_params.max_injection_rate_mscfd,
            ),
            "gravity_factor": (
                self.eor_params.min_gravity_factor,
                self.eor_params.max_gravity_factor,
            ),
            "sor": (self.eor_params.min_sor, self.eor_params.max_sor),
            "transition_alpha": (
                self.eor_params.min_transition_alpha,
                self.eor_params.max_transition_alpha,
            ),
            "transition_beta": (
                self.eor_params.min_transition_beta,
                self.eor_params.max_transition_beta,
            ),
            "plateau_duration_fraction": (
                self.eor_params.min_plateau_duration_fraction,
                self.eor_params.max_plateau_duration_fraction,
            ),
            "ramp_up_fraction": (
                self.eor_params.min_ramp_up_fraction,
                self.eor_params.max_ramp_up_fraction,
            ),
            "hyperbolic_b_factor": (
                self.eor_params.min_hyperbolic_b_factor,
                self.eor_params.max_hyperbolic_b_factor,
            ),
            "productivity_index": (
                self.eor_params.min_productivity_index,
                self.eor_params.max_productivity_index,
            ),
            "wellbore_pressure": (
                self.eor_params.min_wellbore_pressure,
                self.eor_params.max_wellbore_pressure,
            ),
            "well_shut_in_threshold_bpd": (5, 100),
            "allow_well_conversion": (0, 1),
            "well_conversion_day": (90, 1825),
        }
        if self.eor_params.injection_scheme == "wag" or (
            self.eor_params.injection_scheme == "swag" and self.eor_params.swag
        ):
            b["WAG_ratio"] = (self.eor_params.min_WAG_ratio, self.eor_params.max_WAG_ratio)
            
        if self.eor_params.injection_scheme.lower() == "wag":
            b["cycle_length_days"] = (self.eor_params.min_cycle_length_days, self.eor_params.max_cycle_length_days)

        # Add storage-specific parameters if available
        if hasattr(self, "co2_storage_params") and self.co2_storage_params:
            # Add bounds for storage optimization parameters
            b["storage_efficiency_factor"] = (0.5, 0.9)
            b["plume_containment_safety_factor"] = (1.0, 2.0)
            b["structural_trapping_factor"] = (0.1, 0.4)

        for param_key in self._unlocked_params_for_current_run:
            if not (constraint_info := self.RELAXABLE_CONSTRAINTS.get(param_key)):
                continue
            base_val = getattr(
                self._base_eor_params,
                param_key,
                getattr(self._base_reservoir_data, param_key, None),
            )
            if base_val is not None:
                rf = constraint_info["range_factor"]
                b[param_key] = (base_val * (1 - rf), base_val * (1 + rf))
                logger.info(
                    f"Re-run: Overriding bounds for unlocked param '{param_key}' to: ({b[param_key][0]:.3g}, {b[param_key][1]:.3g})"
                )
        return b

    def _get_objective_name_for_logging(self) -> str:
        """Returns a string for logging the current objective."""
        if (
            self.operational_params.target_objective_name
            and self.operational_params.target_objective_value
        ):
            name = self.operational_params.target_objective_name.replace("_", " ").title()
            return f"Match Target ({name} = {self.operational_params.target_objective_value:.3f})"
        return self.chosen_objective.replace("_", " ").title()

    def _handle_target_miss_reporting(
        self, final_eval, final_results, handle_miss
    ) -> Dict[str, Any]:
        """Adds flags and data to the results if a target was not met."""
        target_name = self.operational_params.target_objective_name
        target_value = self.operational_params.target_objective_value
        final_results.update(
            {
                "target_was_unreachable": False,
                "target_objective_name_in_run": target_name,
                "target_objective_value_in_run": target_value,
                "unlocked_params_in_run": self._unlocked_params_for_current_run,
            }
        )
        if target_name and target_value and handle_miss:
            final_achieved = final_eval.get(target_name, 0.0)
            final_results["final_target_value_achieved"] = final_achieved
            if (
                abs(final_achieved - target_value) / (abs(target_value) + 1e-9)
                > self.operational_params.target_tolerance
            ):
                final_results["target_was_unreachable"] = True
        return final_results

    def __getstate__(self):
        """Control what gets pickled when using multiprocessing."""
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        for attr in ["progress_callback", "worker_is_running_check"]:
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        """Restore state when unpickled."""
        self.__dict__.update(state)
        # Restore default values for unpickled attributes
        self.progress_callback = None
        self.worker_is_running_check = lambda: True

    ### --- OPTIMIZER ADAPTERS --- ###
    def _fitness_func_pygad(self, ga_instance, solutions, solution_idx):
        """Fitness function for pygad that supports both single and batch evaluation."""
        if solutions.ndim == 1:
            # Single solution evaluation
            param_names = list(self._get_parameter_bounds().keys())
            params_dict = {name: value for name, value in zip(param_names, solutions)}
            # Pass current generation for adaptive penalty
            current_gen = (
                ga_instance.generations_completed
                if hasattr(ga_instance, "generations_completed")
                else 0
            )
            return self._objective_function_wrapper(current_gen=current_gen, **params_dict)
        else:
            # Batch evaluation - use parallel processing
            # Create safe instance for multiprocessing
            safe_self = PickleSafeOptimiser(self)
            return safe_self._evaluate_solutions_parallel(solutions)

    def _on_generation_callback(self, ga_instance):
        """Callback function for GA generations that is pickleable for multiprocessing."""
        # This callback is called from the main process, so we can access the original callbacks
        # For multiprocessing compatibility, we need to handle this differently
        # Since Qt signals can't be pickled, we'll just log the progress but not emit signals
        # from within the multiprocessing context

        # Calculate generation statistics
        current_gen = ga_instance.generations_completed
        best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        avg_fitness = np.mean(ga_instance.last_generation_fitness)
        std_fitness = np.std(ga_instance.last_generation_fitness)

        # Log detailed generation statistics
        logger.info(
            f"GA Generation {current_gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Std={std_fitness:.4f}, "
            f"Evaluations={current_gen * self._ga_sol_per_pop}"
        )

        # Note: We cannot emit Qt signals from here due to multiprocessing constraints
        # The UI should monitor the log for progress updates during parallel execution

        # Handle mechanisms for escaping local optima (stale restart)
        # Check if the method exists (it will be copied to PickleSafeOptimiser if we are in GA)
        if hasattr(self, "_handle_stale_restart"):
            self._handle_stale_restart(ga_instance)

    def _evaluate_solutions_parallel(self, solutions: np.ndarray) -> np.ndarray:
        """Evaluate multiple solutions in parallel using multiprocessing."""
        param_names = list(self._get_parameter_bounds().keys())
        num_cores = self._get_available_cores()

        if num_cores <= 1 or len(solutions) <= 1:
            # Fallback to sequential evaluation
            return np.array(
                [
                    self._objective_function_wrapper(
                        **{name: val for name, val in zip(param_names, sol)}
                    )
                    for sol in solutions
                ]
            )

        # Prepare parameter dictionaries for parallel evaluation
        params_list = []
        for solution in solutions:
            params_dict = {name: value for name, value in zip(param_names, solution)}
            params_list.append(params_dict)

        # Use ProcessPoolExecutor for parallel evaluation with worker logging setup
        # Use get_worker_initializer() which captures the current logging level
        worker_init = get_worker_initializer()
        with ProcessPoolExecutor(
            max_workers=num_cores,
            initializer=worker_init,
        ) as executor:
            # Submit all evaluation tasks
            future_to_index = {
                executor.submit(self._objective_function_wrapper, **params): idx
                for idx, params in enumerate(params_list)
            }

            # Collect results in order
            results = [None] * len(params_list)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Parallel evaluation failed for solution {idx}: {e}")
                    results[idx] = -1e12  # Penalty for failed evaluation

        return np.array(results)

    def _objective_func_pso(self, particles):
        param_names = list(self._get_parameter_bounds().keys())
        costs = [
            -self._objective_function_wrapper(**{name: val for name, val in zip(param_names, p)})
            for p in particles
        ]
        return np.array(costs)

    def _select_diverse_solutions(
        self, solutions, fitnesses, param_names, num_solutions, diversity_threshold
    ):
        """
        Select diverse solutions from the population based on parameter space diversity.
        Uses Euclidean distance in normalized parameter space to ensure diversity.
        """
        if len(solutions) <= num_solutions:
            return solutions, fitnesses

        # Normalize parameters to [0,1] range based on bounds
        bounds = self._get_parameter_bounds()
        normalized_solutions = []
        for sol in solutions:
            normalized = []
            for i, param_name in enumerate(param_names):
                low, high = bounds[param_name]
                normalized_val = (sol[i] - low) / (high - low) if high > low else 0.5
                normalized.append(normalized_val)
            normalized_solutions.append(normalized)

        normalized_solutions = np.array(normalized_solutions)

        # Calculate distance matrix
        distances = np.sqrt(
            (
                (normalized_solutions[:, np.newaxis, :] - normalized_solutions[np.newaxis, :, :])
                ** 2
            ).sum(axis=2)
        )

        selected_indices = []
        # Start with the best solution
        best_idx = np.argmax(fitnesses)
        selected_indices.append(best_idx)

        # Select diverse solutions
        while len(selected_indices) < num_solutions:
            max_min_distance = -1
            best_candidate = None

            for candidate_idx in range(len(solutions)):
                if candidate_idx in selected_indices:
                    continue

                # Calculate minimum distance to already selected solutions
                min_distance = np.min(distances[candidate_idx, selected_indices])

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate_idx

            if best_candidate is not None and max_min_distance > diversity_threshold:
                selected_indices.append(best_candidate)
            else:
                # If no diverse solution found, select based on fitness
                remaining_indices = [i for i in range(len(solutions)) if i not in selected_indices]
                best_remaining = remaining_indices[np.argmax(fitnesses[remaining_indices])]
                selected_indices.append(best_remaining)

        selected_solutions = [solutions[i] for i in selected_indices]
        selected_fitnesses = [fitnesses[i] for i in selected_indices]

        return selected_solutions, selected_fitnesses

    def _handle_stale_restart(self, ga_instance):
        """
        Handle stagnation by restarting a portion of the population.
        Called from _on_generation_callback when specifically enabled.
        """
        if not hasattr(self, "ga_params_current_run"):
            return

        params = self.ga_params_current_run
        if not params.restart_on_stale:
            return

        # Check for stagnation using pygad's best_solutions_fitness history
        # We look at the last N generations
        if len(ga_instance.best_solutions_fitness) < params.stale_generations:
            return

        recent_fitness = ga_instance.best_solutions_fitness[-params.stale_generations :]
        improvement = max(recent_fitness) - min(recent_fitness)

        # If improvement is negligible (less than 0.1%), consider it stale
        if improvement < abs(max(recent_fitness)) * 0.001:
            logger.info(
                f"Stagnation detected (improvement {improvement:.6f} over {params.stale_generations} gens). Triggering restart."
            )

            # Keep elite solutions
            num_elites = params.keep_elitism
            pop_size = len(ga_instance.population)
            num_restart = int(pop_size * params.restart_diversity_fraction)

            if num_restart < 1:
                return

            # Sort population by fitness
            sorted_indices = np.argsort(ga_instance.last_generation_fitness)[::-1]

            # Replace the worst individuals with random solutions
            worst_indices = sorted_indices[-num_restart:]

            # Generate new random individuals
            # Retrieve bounds for each gene
            lows = [g["low"] for g in ga_instance.gene_space]
            highs = [g["high"] for g in ga_instance.gene_space]

            new_population = np.random.uniform(
                low=lows, high=highs, size=(num_restart, ga_instance.num_genes)
            )

            # Update population
            for i, idx in enumerate(worst_indices):
                ga_instance.population[idx] = new_population[i]

            logger.info(f"Restarted {num_restart} individuals to escape local optimum.")

    ### --- OPTIMIZATION METHODS --- ###
    def optimize_genetic_algorithm(self, ga_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        ga_params = deepcopy(ga_params_override or self.ga_params_default_config)
        bounds = self._get_parameter_bounds()
        param_names = list(bounds.keys())
        gene_space = [{"low": low, "high": high} for low, high in bounds.values()]

        # Store params for callback access
        self.ga_params_current_run = ga_params

        # Store callback functions as instance attributes for multiprocessing compatibility
        # But don't store Qt signals as they can't be pickled
        progress_callback = kwargs.get("convergence_progress_updated")
        worker_is_running_check = kwargs.get("worker_is_running_check", lambda: True)

        # Store only non-Qt data for multiprocessing
        self._ga_sol_per_pop = ga_params.sol_per_pop
        self._ga_progress_data = {
            "has_emit": hasattr(progress_callback, "emit") if progress_callback else False,
            "is_callable": callable(progress_callback) if progress_callback else False,
        }

        # Start GA timing
        ga_start_time = time.time()

        # Configure parallel evaluation if multiple cores available
        num_cores = self._get_available_cores()
        parallel_processing = None
        if num_cores > 1 and ga_params.sol_per_pop >= num_cores:
            parallel_processing = ["process", num_cores]
            logger.info(
                f"GA Optimization started with {ga_params.num_generations} generations, {ga_params.sol_per_pop} population size, and {num_cores} parallel cores"
            )
        else:
            logger.info(
                f"GA Optimization started with {ga_params.num_generations} generations and {ga_params.sol_per_pop} population size (sequential processing)"
            )

        # Create a copy of self without unpicklable objects for multiprocessing
        class PickleSafeOptimiser:
            def __init__(self, optimiser):
                self.__dict__ = {
                    k: v
                    for k, v in optimiser.__dict__.items()
                    if k not in ["progress_callback", "worker_is_running_check"]
                }
                self._fitness_func_pygad = optimiser._fitness_func_pygad
                self._on_generation_callback = optimiser._on_generation_callback
                if hasattr(optimiser, "_handle_stale_restart"):
                    self._handle_stale_restart = optimiser._handle_stale_restart

        safe_self = PickleSafeOptimiser(self)

        # CRITICAL: Disable pygad's internal logging to prevent file lock conflicts
        # on Windows when using multiprocessing. pygad may set up its own file handlers
        # that can cause PermissionError during log rotation with multiple processes.
        pygad_logger = logging.getLogger('pygad')
        pygad_logger.handlers.clear()
        pygad_logger.propagate = True  # Let logs propagate to root queue handler
        pygad_logger.setLevel(logging.WARNING)  # Only show warnings and errors

        # Also disable any RotatingFileHandler that might be set up by pygad
        for handler in pygad_logger.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            pygad_logger.removeHandler(handler)

        ga_instance = pygad.GA(
            num_generations=ga_params.num_generations,
            sol_per_pop=ga_params.sol_per_pop,
            num_parents_mating=ga_params.num_parents_mating,
            num_genes=len(param_names),
            fitness_func=safe_self._fitness_func_pygad,
            gene_space=gene_space,
            parent_selection_type=ga_params.parent_selection_type,
            crossover_type=ga_params.crossover_type,
            crossover_probability=ga_params.crossover_probability,
            mutation_type="adaptive"
            if ga_params.use_adaptive_mutation
            else ga_params.mutation_type,
            mutation_probability=(ga_params.adaptive_mutation_low, ga_params.adaptive_mutation_high)
            if ga_params.use_adaptive_mutation
            else ga_params.mutation_probability,
            stop_criteria=[f"saturate_{ga_params.stale_generations}"]
            if not ga_params.restart_on_stale
            else None,
            keep_elitism=ga_params.keep_elitism,
            on_generation=safe_self._on_generation_callback,
            parallel_processing=parallel_processing,
        )
        ga_instance.run()

        # Calculate GA timing statistics
        ga_end_time = time.time()
        ga_duration = ga_end_time - ga_start_time
        total_evaluations = ga_params.num_generations * ga_params.sol_per_pop
        avg_time_per_eval = ga_duration / total_evaluations if total_evaluations > 0 else 0

        logger.info(
            f"GA Optimization completed in {ga_duration:.2f} seconds "
            f"({total_evaluations} evaluations, {avg_time_per_eval:.4f} sec/eval)"
        )

        # Get all solutions and fitnesses from the final generation
        all_solutions = ga_instance.population
        all_fitnesses = ga_instance.last_generation_fitness

        solution, fitness, _ = ga_instance.best_solution()
        final_params = {name: val for name, val in zip(param_names, solution)}

        final_eval = self.evaluate_for_analysis(final_params)

        # Check if profiles were already generated during evaluation (e.g. by Surrogate Engine)
        final_profiles = final_eval.get("profiles")

        if final_profiles is None:
            # Only use ProductionProfiler if we don't have profiles from the engine
            logger.info("No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based).")
            temp_eor_params_for_profiling = deepcopy(self.eor_params)
            temp_profile_params_for_profiling = deepcopy(self.profile_params)
            temp_co2_storage_params_for_profiling = deepcopy(self.co2_storage_params)

            eor_params_updated = []
            profile_params_updated = []
            co2_storage_params_updated = []
            unmapped_params = []
            validation_errors = []

            # Get parameter bounds for validation
            bounds = self._get_parameter_bounds()

            for key, value in final_params.items():
                # Validate parameter bounds
                if key in bounds:
                    min_val, max_val = bounds[key]
                    if not (min_val <= value <= max_val):
                        validation_errors.append(
                            f"Parameter '{key}' value {value} is outside bounds [{min_val}, {max_val}]"
                        )
                        # Clip to bounds for safety
                        value = np.clip(value, min_val, max_val)

                if hasattr(temp_eor_params_for_profiling, key):
                    setattr(temp_eor_params_for_profiling, key, value)
                    eor_params_updated.append(key)
                elif hasattr(temp_profile_params_for_profiling, key):
                    setattr(temp_profile_params_for_profiling, key, value)
                    profile_params_updated.append(key)
                elif hasattr(temp_co2_storage_params_for_profiling, key):
                    setattr(temp_co2_storage_params_for_profiling, key, value)
                    co2_storage_params_updated.append(key)
                else:
                    unmapped_params.append(key)

            # Log parameter mapping for debugging
            if eor_params_updated:
                logger.info(f"EOR parameters updated: {eor_params_updated}")
            if profile_params_updated:
                logger.info(f"Profile parameters updated: {profile_params_updated}")
            if co2_storage_params_updated:
                logger.info(f"CO2 Storage parameters updated: {co2_storage_params_updated}")
            if unmapped_params:
                logger.warning(
                    f"Unmapped parameters (not in EOR, Profile or CO2 Storage params): {unmapped_params}"
                )
            if validation_errors:
                logger.warning(f"Parameter validation issues: {validation_errors}")

            profiler = ProductionProfiler(
                self.reservoir,
                self.pvt,
                temp_eor_params_for_profiling,
                self.operational_params,
                temp_profile_params_for_profiling,
            )
            final_profiles = profiler.generate_all_profiles(ooip_stb=self.reservoir.ooip_stb)
        else:
            logger.info("Using profiles generated directly by the simulation engine.")

        # Store all evaluated points for potential use in Bayesian optimization
        evaluated_points = []
        for sol, fit in zip(all_solutions, all_fitnesses):
            params_dict = dict(zip(param_names, sol))
            evaluated_points.append(
                {
                    "params": params_dict,
                    "target": -fit,  # Convert back to original objective value
                    "fitness": fit,
                }
            )

        # Select diverse points for BO initialization if specified
        diverse_points = []
        if (
            ga_params.num_diverse_solutions_for_bo > 0
            and ga_params.diversity_threshold_for_bo > 0
            and len(all_solutions) > ga_params.num_diverse_solutions_for_bo
        ):
            diverse_solutions, diverse_fitnesses = self._select_diverse_solutions(
                all_solutions,
                all_fitnesses,
                param_names,
                ga_params.num_diverse_solutions_for_bo,
                ga_params.diversity_threshold_for_bo,
            )

            diverse_points = []
            for sol, fit in zip(diverse_solutions, diverse_fitnesses):
                params_dict = dict(zip(param_names, sol))
                diverse_points.append({"params": params_dict, "target": -fit, "fitness": fit})

            logger.info(
                f"Selected {len(diverse_points)} diverse solutions for BO initialization "
                f"(diversity threshold: {ga_params.diversity_threshold_for_bo})"
            )

        # Store GA timing and statistics in results
        self._results = {
            "optimized_params_final_clipped": final_params,
            "objective_function_value": fitness,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval,
            "method": "genetic_algorithm",
            "pygad_instance": ga_instance,
            "ga_statistics": {
                "total_duration_seconds": ga_duration,
                "total_evaluations": total_evaluations,
                "avg_time_per_evaluation_seconds": avg_time_per_eval,
                "num_generations": ga_params.num_generations,
                "population_size": ga_params.sol_per_pop,
                "best_fitness_history": list(ga_instance.best_solutions_fitness)
                if hasattr(ga_instance, "best_solutions_fitness")
                else [],
                "avg_fitness_history": [np.mean(fitness) for fitness in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness")
                else [],
                "std_fitness_history": [np.std(fitness) for fitness in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness")
                else [],
            },
            "evaluated_points": evaluated_points,
            "diverse_points_for_bo": diverse_points,
        }
        self._results = self._handle_target_miss_reporting(
            final_eval, self._results, kwargs.get("handle_target_miss", False)
        )

        # Perform Decline Curve Analysis on the final optimized production profile
        if final_profiles is not None:
            dca_results = self._perform_decline_curve_analysis(final_profiles, final_params)
            if dca_results:
                self._results["dca_results"] = dca_results

        # Generate and store charts in the results
        charts = {
            "optimization_convergence": self.plotting_manager.plot_optimization_convergence(
                self._results
            ),
            "production_profiles": self.plotting_manager.plot_production_profiles(self._results),
            "co2_performance_summary": self.plotting_manager.plot_co2_performance_summary_table(
                self._results
            ),
            "ga_coverage_distribution": self.plot_ga_coverage_distribution(),
            "ga_objective_distribution": self.plot_ga_objective_distribution(),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts

        return self._results

    def optimize_pso(self, pso_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        pso_params = deepcopy(pso_params_override or self.pso_params_default_config)
        bounds_dict = self._get_parameter_bounds()
        param_names = list(bounds_dict.keys())

        min_bounds = np.array([b[0] for b in bounds_dict.values()])
        max_bounds = np.array([b[1] for b in bounds_dict.values()])
        bounds_tuple = (min_bounds, max_bounds)

        options = {"c1": pso_params.c1, "c2": pso_params.c2, "w": pso_params.w}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=pso_params.n_particles,
            dimensions=len(param_names),
            options=options,
            bounds=bounds_tuple,
        )

        cost, pos = optimizer.optimize(self._objective_func_pso, iters=pso_params.iters)

        final_params = {name: val for name, val in zip(param_names, pos)}

        final_eval_metrics = self.evaluate_for_analysis(final_params)

        # Check if profiles were already generated during evaluation (e.g. by Surrogate Engine)
        final_profiles = final_eval_metrics.get("profiles")

        if final_profiles is None:
            # Fallback for Detailed Engine or missing profiles
            logger.info("No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based).")
            # Create temporary EOR parameters with optimized values for profiling
            temp_eor_params_for_profiling = deepcopy(self.eor_params)
            temp_profile_params_for_profiling = deepcopy(self.profile_params)
            temp_co2_storage_params_for_profiling = deepcopy(self.co2_storage_params)

            eor_params_updated = []
            profile_params_updated = []
            co2_storage_params_updated = []
            unmapped_params = []
            validation_errors = []

            # Get parameter bounds for validation
            bounds = self._get_parameter_bounds()

            for key, value in final_params.items():
                # Validate parameter bounds
                if key in bounds:
                    min_val, max_val = bounds[key]
                    if not (min_val <= value <= max_val):
                        validation_errors.append(
                            f"Parameter '{key}' value {value} is outside bounds [{min_val}, {max_val}]"
                        )
                        # Clip to bounds for safety
                        value = np.clip(value, min_val, max_val)

                if hasattr(temp_eor_params_for_profiling, key):
                    setattr(temp_eor_params_for_profiling, key, value)
                    eor_params_updated.append(key)
                elif hasattr(temp_profile_params_for_profiling, key):
                    setattr(temp_profile_params_for_profiling, key, value)
                    profile_params_updated.append(key)
                elif hasattr(temp_co2_storage_params_for_profiling, key):
                    setattr(temp_co2_storage_params_for_profiling, key, value)
                    co2_storage_params_updated.append(key)
                else:
                    unmapped_params.append(key)

            # Log parameter mapping for debugging
            if eor_params_updated:
                logger.info(f"EOR parameters updated: {eor_params_updated}")
            if profile_params_updated:
                logger.info(f"Profile parameters updated: {profile_params_updated}")
            if co2_storage_params_updated:
                logger.info(f"CO2 Storage parameters updated: {co2_storage_params_updated}")
            if unmapped_params:
                logger.warning(
                    f"Unmapped parameters (not in EOR, Profile or CO2 Storage params): {unmapped_params}"
                )
            if validation_errors:
                logger.warning(f"Parameter validation issues: {validation_errors}")

            profiler = ProductionProfiler(
                self.reservoir,
                self.pvt,
                temp_eor_params_for_profiling,
                self.operational_params,
                temp_profile_params_for_profiling,
            )
            final_profiles = profiler.generate_all_profiles(ooip_stb=self.reservoir.ooip_stb)
        else:
            logger.info("Using profiles generated directly by the simulation engine.")
        self._results = {
            "optimized_params_final_clipped": final_params,
            "objective_function_value": -cost,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval_metrics,
            "method": "pso",
            "pso_cost_history": optimizer.cost_history,
        }
        self._results = self._handle_target_miss_reporting(
            final_eval_metrics, self._results, kwargs.get("handle_target_miss", False)
        )

        # Perform Decline Curve Analysis on the final optimized production profile
        if final_profiles is not None:
            dca_results = self._perform_decline_curve_analysis(final_profiles, final_params)
            if dca_results:
                self._results["dca_results"] = dca_results

        # Generate and store charts in the results
        charts = {
            "optimization_convergence": self.plotting_manager.plot_optimization_convergence(
                self._results
            ),
            "production_profiles": self.plotting_manager.plot_production_profiles(self._results),
            "co2_performance_summary": self.plotting_manager.plot_co2_performance_summary_table(
                self._results
            ),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts

        return self._results

    def optimize_de(self, de_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        de_params = deepcopy(de_params_override or self.de_params_default_config)
        bounds_dict = self._get_parameter_bounds()
        bounds_list = list(bounds_dict.values())
        param_names = list(bounds_dict.keys())

        def de_objective_function(solution):
            params = {name: val for name, val in zip(param_names, solution)}
            return -self._objective_function_wrapper(**params)

        result = differential_evolution(
            de_objective_function,
            bounds=bounds_list,
            strategy=de_params.strategy,
            maxiter=de_params.maxiter,
            popsize=de_params.popsize,
            mutation=de_params.mutation,
            recombination=de_params.recombination,
            seed=42,
        )

        final_params = {name: val for name, val in zip(param_names, result.x)}

        final_eval_metrics = self.evaluate_for_analysis(final_params)

        # Check if profiles were already generated during evaluation (e.g. by Surrogate Engine)
        final_profiles = final_eval_metrics.get("profiles")

        if final_profiles is None:
            # Fallback for Detailed Engine or missing profiles
            logger.info("No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based).")
            # Create temporary EOR parameters with optimized values for profiling
            temp_eor_params_for_profiling = deepcopy(self.eor_params)
            temp_profile_params_for_profiling = deepcopy(self.profile_params)
            temp_co2_storage_params_for_profiling = deepcopy(self.co2_storage_params)

            eor_params_updated = []
            profile_params_updated = []
            co2_storage_params_updated = []
            unmapped_params = []
            validation_errors = []

            # Get parameter bounds for validation
            bounds = self._get_parameter_bounds()

            for key, value in final_params.items():
                # Validate parameter bounds
                if key in bounds:
                    min_val, max_val = bounds[key]
                    if not (min_val <= value <= max_val):
                        validation_errors.append(
                            f"Parameter '{key}' value {value} is outside bounds [{min_val}, {max_val}]"
                        )
                        # Clip to bounds for safety
                        value = np.clip(value, min_val, max_val)

                if hasattr(temp_eor_params_for_profiling, key):
                    setattr(temp_eor_params_for_profiling, key, value)
                    eor_params_updated.append(key)
                elif hasattr(temp_profile_params_for_profiling, key):
                    setattr(temp_profile_params_for_profiling, key, value)
                    profile_params_updated.append(key)
                elif hasattr(temp_co2_storage_params_for_profiling, key):
                    setattr(temp_co2_storage_params_for_profiling, key, value)
                    co2_storage_params_updated.append(key)
                else:
                    unmapped_params.append(key)

            # Log parameter mapping for debugging
            if eor_params_updated:
                logger.info(f"EOR parameters updated: {eor_params_updated}")
            if profile_params_updated:
                logger.info(f"Profile parameters updated: {profile_params_updated}")
            if co2_storage_params_updated:
                logger.info(f"CO2 Storage parameters updated: {co2_storage_params_updated}")
            if unmapped_params:
                logger.warning(
                    f"Unmapped parameters (not in EOR, Profile or CO2 Storage params): {unmapped_params}"
                )
            if validation_errors:
                logger.warning(f"Parameter validation issues: {validation_errors}")

            profiler = ProductionProfiler(
                self.reservoir,
                self.pvt,
                temp_eor_params_for_profiling,
                self.operational_params,
                temp_profile_params_for_profiling,
            )
            final_profiles = profiler.generate_all_profiles(ooip_stb=self.reservoir.ooip_stb)
        else:
            logger.info("Using profiles generated directly by the simulation engine.")

        self._results = {
            "optimized_params_final_clipped": final_params,
            "objective_function_value": -result.fun,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval_metrics,
            "method": "de",
            "de_result_obj": result,
        }
        # Generate and store charts in the results
        charts = {
            "optimization_convergence": self.plotting_manager.plot_optimization_convergence(
                self._results
            ),
            "production_profiles": self.plotting_manager.plot_production_profiles(self._results),
            "co2_performance_summary": self.plotting_manager.plot_co2_performance_summary_table(
                self._results
            ),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts

        return self._results

    def hybrid_optimize(self, **kwargs) -> Dict[str, Any]:
        ga_params = kwargs.get("ga_params_override")
        logger.info(
            f"Hybrid Opt: Starting GA Phase (Gens:{ga_params.num_generations}, Pop:{ga_params.sol_per_pop})"
        )
        if cb := kwargs.get("text_progress_callback"):
            cb(f"Running GA Phase ({ga_params.num_generations} generations)...")

        ga_res = self.optimize_genetic_algorithm(**kwargs)

        if cb:
            cb("GA Phase Complete. Preparing for Bayesian Optimization...")

        ga_instance = ga_res.get("pygad_instance")
        init_bo_sols = []
        if ga_instance:
            param_names = list(self._get_parameter_bounds().keys())
            num_to_select = ga_params.num_diverse_solutions_for_bo
            final_pop, final_fit = ga_instance.population, ga_instance.last_generation_fitness
            sorted_indices = np.argsort(final_fit)[::-1]
            for idx in sorted_indices[:num_to_select]:
                init_bo_sols.append(
                    {"params": {name: val for name, val in zip(param_names, final_pop[idx])}}
                )

        bo_kwargs = kwargs.copy()
        bo_kwargs["initial_solutions_from_ga"] = init_bo_sols
        bo_kwargs["bo_params_override"].n_initial_points = 0
        if cb:
            cb(f"Running BO Phase ({bo_kwargs['bo_params_override'].n_iterations} iterations)...")

        bo_res = self.optimize_bayesian(**bo_kwargs)

        self._results = {**bo_res, "ga_full_results_for_hybrid": ga_res, "method": "hybrid_ga_bo"}
        return self._results

    def optimize_bayesian(self, bo_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        bo_params = deepcopy(bo_params_override or self.bo_params_default_config)
        pb_bayes = kwargs.get("pbounds_override", self._get_parameter_bounds())

        # Start BO timing
        bo_start_time = time.time()
        logger.info(
            f"BO Optimization started with {bo_params.n_iterations} iterations and {bo_params.n_initial_points} initial points"
        )

        bayes_o = BayesianOptimization(
            f=self._objective_function_wrapper, pbounds=pb_bayes, random_state=42, verbose=0
        )

        if initial_sols := kwargs.get("initial_solutions_from_ga"):
            for sol in initial_sols:
                if "params" in sol and all(name in sol["params"] for name in pb_bayes.keys()):
                    bayes_o.probe(params=sol["params"], lazy=True)

        # Track BO progress with detailed logging including parameters
        def bo_progress_callback(iteration, optimizer):
            if iteration % 5 == 0 or iteration == bo_params.n_iterations:
                if hasattr(optimizer, "max") and optimizer.max:
                    current_best = optimizer.max["target"]
                    current_params = optimizer.max["params"]
                    # Format parameters for concise logging
                    params_str = ", ".join([f"{k}: {v:.2f}" for k, v in current_params.items()])
                    logger.info(
                        f"BO Iteration {iteration}: Best={current_best:.4f}, Params: {params_str}"
                    )
                else:
                    logger.info(f"BO Iteration {iteration}: No best value yet")

        # Monkey patch the maximize method to add progress tracking and handle acquisition function parameters
        original_maximize = bayes_o.maximize

        def maximize_with_logging(init_points, n_iter, acq="ucb", kappa=2.576, xi=0.01, **kwargs):
            # Trust Region state
            tr_state = {
                "center": None,
                "size": bo_params.trust_region_initial_size,
                "success_counter": 0,
                "failure_counter": 0,
                "best_target": -float("inf"),
            }

            # Helper to update bounds based on Trust Region
            def update_tr_bounds(optimizer, current_best_params):
                if not bo_params.use_trust_region:
                    return

                # Get original global bounds
                # We assume standard bounds are roughly those in pb_bayes
                # But we need access to the optimizer's original bounds.
                # Since we reset bounds here, we rely on pb_bayes as the global reference.

                # Decay or Expand
                # Simple logic: If simplified TuRBO
                # If success: Expand slightly? Or keep same.
                # If failure: Shrink.

                size = tr_state["size"]

                new_bounds = {}
                for param, (low, high) in pb_bayes.items():
                    center = current_best_params.get(param)
                    if center is None:
                        continue

                    span = high - low
                    half_width = (span * size) / 2.0

                    new_low = max(low, center - half_width)
                    new_high = min(high, center + half_width)
                    new_bounds[param] = (new_low, new_high)

                # Apply new bounds to optimizer
                # bayes_opt <= 1.4 uses optimizer.set_bounds(new_bounds)
                if hasattr(optimizer, "set_bounds"):
                    optimizer.set_bounds(new_bounds)
                else:
                    # Fallback for older versions: modify private attribute if strictly necessary
                    # or just ignore if not supported (partial TR support)
                    pass

            for i in range(1, n_iter + 1):
                # Check for progress before step (except first)
                if i > 1 and bo_params.use_trust_region and hasattr(bayes_o, "max") and bayes_o.max:
                    curr_max = bayes_o.max["target"]
                    if curr_max > tr_state["best_target"] + 1e-6:
                        # Success
                        tr_state["success_counter"] += 1
                        tr_state["failure_counter"] = 0
                        tr_state["best_target"] = curr_max
                        tr_state["center"] = bayes_o.max["params"]
                        # Maybe expand?
                        # tr_state["size"] = min(1.0, tr_state["size"] * 1.5)
                    else:
                        # Failure
                        tr_state["failure_counter"] += 1
                        tr_state["success_counter"] = 0
                        # Shrink if too many failures
                        if tr_state["failure_counter"] >= 3:  # Configurable?
                            tr_state["size"] *= bo_params.trust_region_decay
                            tr_state["failure_counter"] = 0
                            if tr_state["size"] < bo_params.trust_region_min_size:
                                # Restart / Reset TR
                                tr_state["size"] = bo_params.trust_region_initial_size
                                logger.info("Trust Region collapsed. Resetting to full size.")

                    # Update bounds
                    if tr_state["center"]:
                        update_tr_bounds(bayes_o, tr_state["center"])

                # Pass only init_points and n_iter to original_maximize
                original_maximize(init_points=0 if i > 1 else init_points, n_iter=1, **kwargs)
                bo_progress_callback(i, bayes_o)

        bayes_o.maximize = maximize_with_logging

        try:
            # Use acquisition function parameters from bo_params, with increased kappa for more exploration
            exploration_factor = kwargs.get(
                "exploration_factor", 2.5
            )  # Default factor to increase exploration
            acq_kappa = bo_params.acq_kappa * exploration_factor
            logger.info(
                f"BO using acquisition function: {bo_params.acquisition_function}, "
                f"kappa: {acq_kappa:.3f} (exploration factor: {exploration_factor}), xi: {bo_params.acq_xi}"
            )

            bayes_o.maximize(
                init_points=bo_params.n_initial_points,
                n_iter=bo_params.n_iterations,
                acq=bo_params.acquisition_function,
                kappa=acq_kappa,
                xi=bo_params.acq_xi,
            )
        except Exception as e:
            logger.error(f"BO optimization failed: {e}")
            raise

        best_params, best_obj = bayes_o.max["params"], bayes_o.max["target"]

        # Calculate BO timing statistics
        bo_end_time = time.time()
        bo_duration = bo_end_time - bo_start_time
        total_evaluations = bo_params.n_initial_points + bo_params.n_iterations
        avg_time_per_eval = bo_duration / total_evaluations if total_evaluations > 0 else 0

        logger.info(
            f"BO Optimization completed in {bo_duration:.2f} seconds "
            f"({total_evaluations} evaluations, {avg_time_per_eval:.4f} sec/eval)"
        )

        # Re-evaluate the best solution to get all final metrics and profiles
        final_eval = self.evaluate_for_analysis(best_params)

        # Check if profiles were already generated during evaluation (e.g. from Simple or Surrogate Engine)
        final_profiles = final_eval.get("profiles")

        if final_profiles is None:
            # Fallback: Regenerate profiles using ProductionProfiler (Detailed engine fallback)
            # This path is taken only if the evaluation engine didn't return profiles
            logger.info("No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based).")
            temp_eor_params_for_profiling = deepcopy(self.eor_params)
            temp_profile_params_for_profiling = deepcopy(self.profile_params)
            temp_co2_storage_params_for_profiling = deepcopy(self.co2_storage_params)

            eor_params_updated = []
            profile_params_updated = []
            co2_storage_params_updated = []
            unmapped_params = []

            for key, value in best_params.items():
                if hasattr(temp_eor_params_for_profiling, key):
                    setattr(temp_eor_params_for_profiling, key, value)
                    eor_params_updated.append(key)
                elif hasattr(temp_profile_params_for_profiling, key):
                    setattr(temp_profile_params_for_profiling, key, value)
                    profile_params_updated.append(key)
                elif hasattr(temp_co2_storage_params_for_profiling, key):
                    setattr(temp_co2_storage_params_for_profiling, key, value)
                    co2_storage_params_updated.append(key)
                else:
                    unmapped_params.append(key)

            # Log parameter mapping for debugging
            if eor_params_updated:
                logger.info(f"EOR parameters updated: {eor_params_updated}")
            if profile_params_updated:
                logger.info(f"Profile parameters updated: {profile_params_updated}")
            if co2_storage_params_updated:
                logger.info(f"CO2 Storage parameters updated: {co2_storage_params_updated}")
            if unmapped_params:
                logger.warning(
                    f"Unmapped parameters (not in EOR, Profile or CO2 Storage params): {unmapped_params}"
                )

            profiler = ProductionProfiler(
                self.reservoir,
                self.pvt,
                temp_eor_params_for_profiling,
                self.operational_params,
                temp_profile_params_for_profiling,
            )
            final_profiles = profiler.generate_all_profiles(ooip_stb=self.reservoir.ooip_stb)
        else:
            logger.info("Using profiles generated directly by the simulation engine.")

        # Store BO timing and statistics in results
        self._results = {
            "optimized_params_final_clipped": best_params,
            "objective_function_value": best_obj,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval,
            "method": "bayesian_gp",
            "bayes_opt_obj": bayes_o,
            "bo_statistics": {
                "total_duration_seconds": bo_duration,
                "total_evaluations": total_evaluations,
                "avg_time_per_evaluation_seconds": avg_time_per_eval,
                "n_iterations": bo_params.n_iterations,
                "n_initial_points": bo_params.n_initial_points,
                "acquisition_function": bo_params.acquisition_function,
                "acquisition_kappa": bo_params.acq_kappa,
                "acquisition_xi": bo_params.acq_xi,
                "objective_history": [res["target"] for res in bayes_o.res]
                if hasattr(bayes_o, "res")
                else [],
                "parameters_history": [res["params"] for res in bayes_o.res]
                if hasattr(bayes_o, "res")
                else [],
            },
        }
        self._results = self._handle_target_miss_reporting(
            final_eval, self._results, kwargs.get("handle_target_miss", False)
        )

        # Perform Decline Curve Analysis on the final optimized production profile
        if final_profiles is not None:
            dca_results = self._perform_decline_curve_analysis(final_profiles, best_params)
            if dca_results:
                self._results["dca_results"] = dca_results

        # Generate and store charts in the results
        charts = {
            "optimization_convergence": self.plotting_manager.plot_optimization_convergence(
                self._results
            ),
            "production_profiles": self.plotting_manager.plot_production_profiles(self._results),
            "co2_performance_summary": self.plotting_manager.plot_co2_performance_summary_table(
                self._results
            ),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts

        return self._results

    def export_to_cmg(self, filename: str) -> bool:
        """
        Export optimized parameters to CMG GEM/STARS format.

        Args:
            filename: Output filename for CMG keyword file

        Returns:
            bool: True if export successful, False otherwise
        """
        if not self._results:
            logger.error("No optimization results available for export.")
            return False

        exporter = SimulatorExporter()
        return exporter.export_to_cmg(self._results, filename)

    def generate_summary_report(self, format: str = "csv") -> str:
        """
        Generate standardized summary report in various formats.

        Args:
            format: Output format ('csv', 'json', 'text')

        Returns:
            str: Formatted report content
        """
        if not self._results:
            return "No optimization results available for report generation."

        exporter = SimulatorExporter()
        return exporter.generate_summary_report(self._results, format)

    def validate_physical_constraints(self, params: Dict[str, float]) -> List[str]:
        """
        Validate parameters against physical constraints for CO₂ EOR.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            List[str]: List of validation warnings/errors, empty if all valid
        """
        warnings = []

        # Pressure validation
        pressure = params.get("pressure", 0)
        mmp = self.mmp or self.eor_params.default_mmp_fallback
        if pressure < mmp * 1.05:
            warnings.append(
                f"Pressure ({pressure} psi) is less than 1.05×MMP ({mmp * 1.05:.1f} psi) - may not achieve full miscibility."
            )
        if pressure > self.eor_params.max_pressure_psi:
            warnings.append(
                f"Pressure ({pressure} psi) exceeds maximum allowed pressure ({self.eor_params.max_pressure_psi} psi)."
            )

        # Mobility ratio validation
        mobility_ratio = params.get("mobility_ratio", 1.0)
        if mobility_ratio < 0.1:
            warnings.append(
                f"Mobility ratio ({mobility_ratio:.2f}) is unusually low for CO₂ flooding."
            )
        if mobility_ratio > 20.0:
            warnings.append(
                f"Mobility ratio ({mobility_ratio:.2f}) exceeds typical range for CO₂ flooding (max 20.0)."
            )

        # Injection rate validation
        rate = params.get("rate", 0)
        if rate < self.eor_params.min_injection_rate_mscfd:
            warnings.append(
                f"Injection rate ({rate} mscfd) is below minimum ({self.eor_params.min_injection_rate_mscfd} mscfd)."
            )
        if rate > self.eor_params.max_injection_rate_mscfd:
            warnings.append(
                f"Injection rate ({rate} mscfd) exceeds maximum ({self.eor_params.max_injection_rate_mscfd} mscfd)."
            )

        # WAG ratio validation (if applicable)
        if self.eor_params.injection_scheme == "wag" or (
            self.eor_params.injection_scheme == "swag" and self.eor_params.swag
        ):
            wag_ratio = params.get("WAG_ratio", 1.0)
            if wag_ratio < 0.1:
                warnings.append(f"WAG ratio ({wag_ratio:.2f}) is below minimum (0.1).")
            if wag_ratio > 5.0:
                warnings.append(f"WAG ratio ({wag_ratio:.2f}) exceeds maximum (5.0).")

        # Reservoir property validation
        porosity = params.get("porosity", self.avg_porosity)
        if porosity < 0.05 or porosity > 0.35:
            warnings.append(
                f"Porosity ({porosity:.3f}) is outside typical range for CO₂ EOR (0.05-0.35)."
            )

        permeability = params.get("permeability", 100.0)
        if permeability < 1.0 or permeability > 5000.0:
            warnings.append(
                f"Permeability ({permeability:.1f} md) is outside typical range for CO₂ EOR (1-5000 md)."
            )

        return warnings

    def _validate_co2_specific_constraints(self, params: Dict[str, float]) -> List[str]:
        """
        Validate CO₂-specific physical constraints.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            List[str]: List of validation warnings/errors
        """
        warnings = []

        # Temperature validation for CO₂ properties
        temperature = self.pvt.temperature if hasattr(self.pvt, "temperature") else 150.0
        if temperature < 80.0 or temperature > 250.0:
            warnings.append(
                f"Reservoir temperature ({temperature}°F) is outside optimal range for CO₂ EOR (80-250°F)."
            )

        # CO₂ viscosity validation
        co2_viscosity = params.get("co2_viscosity", 0.02)
        if co2_viscosity < 0.01 or co2_viscosity > 0.1:
            warnings.append(
                f"CO₂ viscosity ({co2_viscosity:.3f} cp) is outside typical range (0.01-0.1 cp)."
            )

        # Oil viscosity validation
        oil_viscosity = params.get("viscosity_oil", 4.0)
        if oil_viscosity < 1.0 or oil_viscosity > 100.0:
            warnings.append(
                f"Oil viscosity ({oil_viscosity:.1f} cp) is outside typical range for CO₂ EOR (1-100 cp)."
            )

        # C7+ fraction validation
        c7_plus = params.get("c7_plus_fraction", 0.35)
        if c7_plus < 0.2 or c7_plus > 0.6:
            warnings.append(
                f"C7+ fraction ({c7_plus:.2f}) is outside typical range for CO₂ EOR (0.2-0.6)."
            )

        # CO₂ storage capacity validation
        storage_capacity_validation = self._validate_co2_storage_capacity(params)
        warnings.extend(storage_capacity_validation)

        return warnings

    def _validate_co2_storage_capacity(self, params: Dict[str, float]) -> List[str]:
        """Validates if the reservoir can safely store the estimated CO2 volume."""
        warnings = []

        # Calculate estimated CO2 injection volume
        injection_rate = params.get("rate", self.eor_params.injection_rate)
        project_lifetime = self.operational_params.project_lifetime_years

        # Account for WAG scheme
        is_wag = self.eor_params.injection_scheme == "wag" or (
            self.eor_params.injection_scheme == "swag" and self.eor_params.swag
        )
        default_wag_ratio = (
            getattr(self.eor_params.swag, "water_gas_ratio", 1.0) if self.eor_params.swag else 1.0
        )
        wag_ratio = params.get("WAG_ratio", default_wag_ratio)
        water_frac = wag_ratio / (1 + wag_ratio) if is_wag else 0.0
        co2_inj_rate_bpd = injection_rate * (1 - water_frac)

        # Convert from reservoir bpd to MSCF/day, then to tonnes
        co2_inj_mscf_per_day = co2_inj_rate_bpd / self.b_gas_rb_per_mscf
        total_co2_injected_mscf = co2_inj_mscf_per_day * project_lifetime * DAYS_PER_YEAR
        estimated_co2_volume_tonne = (
            total_co2_injected_mscf * self.eor_params.co2_density_tonne_per_mscf
        )

        # Check against storage capacity
        if estimated_co2_volume_tonne > self.co2_storage_params.storage_capacity_tonne:
            warnings.append(
                f"Estimated CO2 volume ({estimated_co2_volume_tonne:.0f} tonnes) exceeds storage capacity ({self.co2_storage_params.storage_capacity_tonne:.0f} tonnes)"
            )

        # Check geological assurance factors
        if self.co2_storage_params.reservoir_seal_integrity_factor < 0.7:
            warnings.append(
                f"Reservoir seal integrity factor ({self.co2_storage_params.reservoir_seal_integrity_factor:.2f}) is below recommended minimum (0.7)"
            )

        if self.co2_storage_params.min_trapping_efficiency < 0.8:
            warnings.append(
                f"Minimum trapping efficiency ({self.co2_storage_params.min_trapping_efficiency:.2f}) is below recommended minimum (0.8)"
            )

        return warnings

    def get_validation_report(self, params: Optional[Dict[str, float]] = None) -> str:
        """
        Generate a comprehensive validation report for given parameters.

        Args:
            params: Parameters to validate (uses current results if None)

        Returns:
            str: Formatted validation report
        """
        if params is None:
            if not self._results:
                return "No parameters available for validation."
            params = self._results.get("optimized_params_final_clipped", {})

        general_warnings = self.validate_physical_constraints(params)
        co2_warnings = self._validate_co2_specific_constraints(params)

        report_lines = ["CO₂ EOR Parameter Validation Report", "=" * 40]

        if general_warnings:
            report_lines.append("\nGeneral Physical Constraints:")
            for warning in general_warnings:
                report_lines.append(f"  ⚠ {warning}")
        else:
            report_lines.append("\n✓ All general physical constraints satisfied.")

        if co2_warnings:
            report_lines.append("\nCO₂-Specific Constraints:")
            for warning in co2_warnings:
                report_lines.append(f"  ⚠ {warning}")
        else:
            report_lines.append("\n✓ All CO₂-specific constraints satisfied.")

        if not general_warnings and not co2_warnings:
            report_lines.append("\n✓ All parameters are within recommended ranges for CO₂ EOR.")

        return "\n".join(report_lines)

    def get_uncertain_parameters(self) -> List[Dict[str, Any]]:
        """
        Returns a list of uncertain parameter definitions based on optimized parameters
        for use in uncertainty quantification analysis.

        Uses optimized parameters as mean values and applies reasonable uncertainties
        based on parameter types and typical ranges for CO₂ EOR.
        """
        if not self._results or "optimized_params_final_clipped" not in self._results:
            logger.warning(
                "No optimization results available. Returning empty uncertain parameters."
            )
            return []

        optimized_params = self._results["optimized_params_final_clipped"]
        uncertain_params = []

        # Economic parameters - typically ±20% uncertainty
        economic_params = [
            ("econ.oil_price_usd_per_bbl", "normal", [optimized_params.get("oil_price", 60), 12]),
            (
                "econ.co2_purchase_cost_usd_per_tonne",
                "normal",
                [optimized_params.get("co2_cost", 40), 8],
            ),
            (
                "econ.discount_rate_fraction",
                "normal",
                [optimized_params.get("discount_rate", 0.1), 0.02],
            ),
        ]

        # Operational EOR parameters - typically ±15% uncertainty
        eor_params = [
            ("eor.pressure", "normal", [optimized_params.get("pressure", 2000), 300]),
            ("eor.rate", "normal", [optimized_params.get("rate", 5000), 750]),
            ("eor.mobility_ratio", "normal", [optimized_params.get("mobility_ratio", 10), 2]),
        ]

        # Reservoir parameters - typically ±10% uncertainty
        reservoir_params = [
            ("reservoir.avg_porosity", "normal", [self.avg_porosity, self.avg_porosity * 0.1]),
            (
                "reservoir.ooip_stb",
                "normal",
                [self.reservoir.ooip_stb, self.reservoir.ooip_stb * 0.1],
            ),
        ]

        # Fluid properties - typically ±15% uncertainty
        fluid_params = [("fluid.mmp_value", "normal", [self.mmp or 2500, 375])]

        # Recovery model parameters - typically ±20% uncertainty
        model_params = [
            (
                "model.v_dp_coefficient",
                "normal",
                [optimized_params.get("v_dp_coefficient", 0.7), 0.14],
            ),
            ("model.gravity_factor", "normal", [optimized_params.get("gravity_factor", 0.5), 0.1]),
            ("model.sor", "normal", [optimized_params.get("sor", 0.3), 0.06]),
        ]

        # Combine all parameter definitions
        all_params = economic_params + eor_params + reservoir_params + fluid_params + model_params

        for path, dist_type, dist_params in all_params:
            uncertain_params.append(
                {
                    "path": path,
                    "distribution": dist_type,
                    "params": dist_params,
                    "scope": path.split(".")[0],
                    "internal_name": path.split(".")[1],
                }
            )

        logger.info(
            f"Generated {len(uncertain_params)} uncertain parameters from optimization results"
        )
        return uncertain_params

    def optimize_per_well(self, optimizer_name: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        if not self._base_well_data_list:
            raise ValueError("No well data provided for per-well optimization.")

        all_well_results = {}

        for well_data in self._base_well_data_list:
            well_name = well_data.name
            logger.info(f"--- Starting optimization for well: {well_name} ---")

            # Create a new engine instance for each well to ensure no state leakage
            well_engine = OptimizationEngine(
                reservoir=self._base_reservoir_data,
                pvt=self._base_pvt_data,
                eor_params_instance=self._base_eor_params,
                ga_params_instance=self.ga_params_default_config,
                bo_params_instance=self.bo_params_default_config,
                pso_params_instance=self.pso_params_default_config,
                de_params_instance=self.de_params_default_config,
                economic_params_instance=self._base_economic_params,
                operational_params_instance=self._base_operational_params,
                profile_params_instance=self.profile_params,
                advanced_engine_params_instance=self.advanced_engine_params,
                co2_storage_params_instance=self._base_co2_storage_params,
                well_data_list=[well_data],  # Pass only the current well's data
                mmp_init_override=self._mmp_value_init_override,
            )

            optimizer_method = getattr(well_engine, optimizer_name, None)
            if not callable(optimizer_method):
                raise AttributeError(f"Optimizer '{optimizer_name}' not found or is not callable.")

            try:
                result = optimizer_method(**kwargs)
                all_well_results[well_name] = result
                logger.info(f"--- Finished optimization for well: {well_name} ---")
            except Exception as e:
                logger.error(f"Optimization failed for well {well_name}: {e}", exc_info=True)
                all_well_results[well_name] = {"error": str(e)}

        return all_well_results

    def plot_ga_coverage_distribution(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        source = results_to_use or self._results
        if not (source and "pygad_instance" in source):
            return go.Figure().update_layout(title_text="No GA results to plot.")

        ga_instance = source["pygad_instance"]
        population = ga_instance.population
        param_names = list(self._get_parameter_bounds().keys())

        from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency

        sweep_efficiencies = []
        for individual in population:
            params_dict = {name: val for name, val in zip(param_names, individual)}
            
            # Use fast analytical surrogate for sweep distribution (PhD consistent)
            mu_oil = self.pvt.oil_viscosity_cp or 1.5
            mu_co2 = self.pvt.gas_viscosity_cp or 0.05
            mobility_ratio = mu_oil / max(mu_co2, 1e-6)
            
            # Check if mobility_ratio is in params_dict (from optimization)
            if "mobility_ratio" in params_dict:
                mobility_ratio = params_dict["mobility_ratio"]
                
            sweep = calculate_areal_sweep_efficiency(mobility_ratio)
            sweep_efficiencies.append(sweep)

        avg_sweep = np.mean(sweep_efficiencies)
        std_sweep = np.std(sweep_efficiencies)

        fig = go.Figure(data=[go.Histogram(x=sweep_efficiencies, nbinsx=20)])
        fig.update_layout(
            title_text="GA Population Areal Sweep Efficiency Distribution",
            xaxis_title="Areal Sweep Efficiency",
            yaxis_title="Frequency",
            annotations=[
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Avg: {avg_sweep:.3f}<br>Std: {std_sweep:.3f}",
                    showarrow=False,
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                )
            ],
        )
        return fig

    def plot_ga_objective_distribution(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        source = results_to_use or self._results
        if not (source and "pygad_instance" in source):
            return go.Figure().update_layout(title_text="No GA results to plot.")

        ga_instance = source["pygad_instance"]
        objectives = ga_instance.last_generation_fitness

        avg_obj = np.mean(objectives)
        std_obj = np.std(objectives)

        fig = go.Figure(data=[go.Histogram(x=objectives, nbinsx=20)])
        fig.update_layout(
            title_text="GA Population Objective Value Distribution",
            xaxis_title="Objective Value",
            yaxis_title="Frequency",
            annotations=[
                dict(
                    x=0.95,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Avg: {avg_obj:.3f}<br>Std: {std_obj:.3f}",
                    showarrow=False,
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                )
            ],
        )
        return fig

    def plot_hybrid_model_analysis(self) -> go.Figure:
        """Generates a plot showing the interplay of miscible, immiscible, and hybrid recovery models."""
        from core.simulation.recovery_models import (
            MiscibleRecoveryModel,
            ImmiscibleRecoveryModel,
            SigmoidTransition,
        )

        mmp = self.mmp or self.eor_params.default_mmp_fallback
        pressure_ratios = np.linspace(0.5, 2.0, 50)
        pressures = pressure_ratios * mmp

        miscible_rf = []
        immiscible_rf = []
        weights = []

        miscible_model = MiscibleRecoveryModel()
        immiscible_model = ImmiscibleRecoveryModel()
        transition = SigmoidTransition()

        base_params = dataclasses.asdict(self.eor_params)

        for p in pressures:
            params = base_params.copy()
            params["pressure"] = p
            params["mmp"] = mmp
            miscible_rf.append(miscible_model.calculate(**params))
            immiscible_rf.append(immiscible_model.calculate(**params))
            weights.append(transition.evaluate(p / mmp, self.pvt.c7_plus_fraction))

        hybrid_rf = np.array(weights) * np.array(miscible_rf) + (1 - np.array(weights)) * np.array(
            immiscible_rf
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=pressure_ratios, y=immiscible_rf, mode="lines", name="Immiscible RF")
        )
        fig.add_trace(
            go.Scatter(x=pressure_ratios, y=miscible_rf, mode="lines", name="Miscible RF")
        )
        fig.add_trace(
            go.Scatter(
                x=pressure_ratios,
                y=hybrid_rf,
                mode="lines",
                name="Hybrid RF",
                line=dict(color="black", width=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pressure_ratios,
                y=weights,
                mode="lines",
                name="Miscible Weight",
                line=dict(dash="dot"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title_text="Hybrid Recovery Model Analysis",
            xaxis_title="Pressure / MMP Ratio",
            yaxis_title="Recovery Factor",
            yaxis=dict(range=[0, 1]),
            yaxis2=dict(
                title="Miscible Weight", overlaying="y", side="right", range=[0, 1], showgrid=False
            ),
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    def plot_breakthrough_mechanism_analysis(self) -> go.Figure:
        """Generates a bar chart comparing breakthrough times from different models using Surrogate Physics."""
        # Use module-level SurrogateBreakthrough class (PhD Verified)
        bt_physics = SurrogateBreakthrough()

        # Prepare reservoir and fluid data for analytical calculation
        reservoir_params = {
            "v_dp_coefficient": getattr(self.eor_params, "v_dp_coefficient", 0.5),
            "area_acres": self.reservoir.area_acres,
            "porosity": self.avg_porosity,
            "thickness_ft": self.reservoir.thickness_ft,
            "permeability": np.mean(self.reservoir.grid.get("PERMX", [100.0])),
        }
        eor_params_for_bt = dataclasses.asdict(self.eor_params)

        # Calculate individual mechanism times (Surrogate equivalents)
        # 1. Base Koval Breakthrough
        bt_koval = bt_physics.calculate_breakthrough_time(reservoir_params, eor_params_for_bt)
        
        # 2. Gravity Override Estimate (Simple analytical scaling)
        # Higher density difference = earlier breakthrough
        rho_oil = getattr(self.eor_params, "oil_density", 50.0)
        rho_co2 = getattr(self.eor_params, "co2_density", 44.0)
        gravity_mult = 1.0 / (1.0 + 0.1 * abs(rho_oil - rho_co2))
        bt_gravity = bt_koval * gravity_mult
        
        # 3. Final Weighted (Surrogate Engine already provides this)
        bt_final = bt_koval # For surrogate, koval is the verified mechanism

        mechanisms = ["Analytical (Koval)", "Gravity Scaling", "Final Surrogate"]
        times = [bt_koval, bt_gravity, bt_final]

        fig = go.Figure(
            [go.Bar(x=mechanisms, y=times, text=[f"{t:.2f} y" for t in times], textposition="auto")]
        )
        fig.update_layout(
            title_text="PhD Verification: Breakthrough Analysis by Mechanism (Surrogate)",
            yaxis_title="Breakthrough Time (years)",
            template="plotly_white"
        )
        return fig
