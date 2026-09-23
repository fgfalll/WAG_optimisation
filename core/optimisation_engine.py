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
    AdvancedEngineParams,
    CO2StorageParameters,
    LayerDefinition,
    PhysicalConstants,
)
from core.engine_surrogate.surrogate_engine import SurrogateEngineWrapper
from core.engine_surrogate.pvt_state import SolventExtendedPVTEngine
class OptimizationError(RuntimeError):
    """Raised when optimization evaluation fails."""

    pass


class SimulationEngineError(RuntimeError):
    """Raised when simulation engine evaluation fails."""

    pass
from evaluation.mmp import calculate_mmp, MMPParameters

calculate_mmp_external = calculate_mmp
PHYSICS_ENGINE_AVAILABLE = True

# Physical constants for CO2-EOR calculations
_PHYS_CONSTANTS = PhysicalConstants()
DAYS_PER_YEAR = _PHYS_CONSTANTS.DAYS_PER_YEAR
ACRES_TO_CM2 = _PHYS_CONSTANTS.ACRES_TO_M2 * 10000.0  # 1 acre = 4046.8564224 m² = 40,468,564.224 cm²
B_GAS_RB_PER_MSCF = 1.0  # Fallback formation volume factor (RB/MSCF) when dynamic PVT unavailable
EPSILON = 1e-10

logger = logging.getLogger(__name__)

from core.plotting_manager import PlottingManager
from core.objectives import ObjectiveFunctions
from core.objectives.storage import calculate_geomechanical_containment_score
from utils.cmg_exporter import SimulatorExporter


INJECTION_SCHEMES = ["continuous", "wag", "tapered", "huff_n_puff", "swag"]
FAILURE_PENALTY = -1e12

class PickleSafeOptimiser:
    """Multiprocessing-safe wrapper around OptimizationEngine with unpicklable attributes stripped."""

    def __init__(self, optimiser: "OptimizationEngine"):
        self.__dict__ = {
            k: v
            for k, v in optimiser.__dict__.items()
            if k not in ["progress_callback", "worker_is_running_check"]
        }
        self._fitness_func_pygad = optimiser._fitness_func_pygad
        self._on_generation_callback = optimiser._on_generation_callback
        self._evaluate_solutions_parallel = optimiser._evaluate_solutions_parallel
        self._objective_function_wrapper = optimiser._objective_function_wrapper
        self._get_parameter_bounds = optimiser._get_parameter_bounds
        self._get_available_cores = optimiser._get_available_cores
        self._map_scheme_index_to_name = optimiser._map_scheme_index_to_name
        self._sanitize_and_discretize_parameters = optimiser._sanitize_and_discretize_parameters
        if hasattr(optimiser, "_handle_stale_restart"):
            self._handle_stale_restart = optimiser._handle_stale_restart


class OptimizationEngine:
    RELAXABLE_CONSTRAINTS = {
        "porosity": {"description": "Average reservoir porosity (v/v)", "type": "reservoir"},
        "ooip_stb": {"description": "Original Oil In Place (STB)", "type": "reservoir"},
        "v_dp_coefficient": {
            "description": "Dykstra-Parsons coefficient for heterogeneity",
            "type": "eor",
        },
        "mobility_ratio": {"description": "Mobility Ratio (M)", "type": "eor"},
        "wag_ratio": {"description": "Water-Alternating-Gas Ratio", "type": "eor"},
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
        economic_params_instance: Optional[EconomicParameters] = None,
        operational_params_instance: Optional[OperationalParameters] = None,
        profile_params_instance: Optional[ProfileParameters] = None,
        advanced_engine_params_instance: Optional[AdvancedEngineParams] = None,
        co2_storage_params_instance: Optional[CO2StorageParameters] = None,
        well_data_list: Optional[List[Any]] = None,
        fitting_params_instance: Optional[Any] = None,
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
        self._base_fitting_params = deepcopy(fitting_params_instance)

        self.reservoir = deepcopy(self._base_reservoir_data)
        self.pvt = deepcopy(self._base_pvt_data)
        self.eor_params = deepcopy(self._base_eor_params)
        self.economic_params = deepcopy(self._base_economic_params)
        self.operational_params = deepcopy(self._base_operational_params)
        self.co2_storage_params = deepcopy(self._base_co2_storage_params)
        self.profile_params = deepcopy(profile_params_instance or ProfileParameters())
        self.well_data_list = deepcopy(self._base_well_data_list)
        self.fitting_params = deepcopy(self._base_fitting_params)

        self.RELAXABLE_CONSTRAINTS = {
            k: {
                "description": v["description"],
                "range_factor": self.advanced_engine_params.relaxable_constraint_range_factors.get(
                    k, 0.2
                ),
            }
            for k, v in self.RELAXABLE_CONSTRAINTS.items()
        }

        if WellAnalysis is not None and well_data_list and pvt:
            self.well_analysis = WellAnalysis(well_data=well_data_list[0], pvt_data=pvt)
        else:
            self.well_analysis = None

        self._unlocked_params_for_current_run: List[str] = []
        self.ga_params_default_config = ga_params_instance or GeneticAlgorithmParams()
        self.bo_params_default_config = bo_params_instance or BayesianOptimizationParams()

        self.profiler = None  # Will be instantiated on-demand with the physics-based model
        self.dca_analyzer = DeclineCurveAnalyzer() if DeclineCurveAnalyzer is not None else None

        self._results: Optional[Dict[str, Any]] = None
        self._mmp_value_init_override = mmp_init_override
        self._mmp_value: Optional[float] = self._mmp_value_init_override

        self.chosen_objective: str = "npv"

        self._mmp_calculator_fn = calculate_mmp_external
        self._MMPParametersDataclass = MMPParameters  # MMPParameters class now available
        self.eos_model_instance: Optional[Any] = None
        self.b_gas_rb_per_mscf = B_GAS_RB_PER_MSCF  # Fallback

        self.reservoir_fluid = None

        self.plotting_manager = PlottingManager(self)
        # Initialize objective_functions after reset_to_base_state to ensure it uses the current reservoir instance
        self.reset_to_base_state()

        # Initialize B_gas from SolventExtendedPVTEngine
        try:
            pvt_engine = SolventExtendedPVTEngine(
                reservoir_temperature_f=self.reservoir.temperature,
                initial_pressure_psi=self.reservoir.initial_pressure,
                api_gravity=getattr(self.reservoir, "oil_gravity_api", 35.0),
            )
            self.b_gas_rb_per_mscf = pvt_engine.calculate_co2_fvf_rb_per_mscf(
                pressure_psi=self.reservoir.initial_pressure,
                t_f=self.reservoir.temperature,
            )
            logger.info(
                f"Using accurate B_gas from SolventExtendedPVTEngine: {self.b_gas_rb_per_mscf:.4f} rb/MSCF at T={self.reservoir.temperature:.1f}F, P={self.reservoir.initial_pressure:.1f}psia"
            )
        except Exception as bgas_error:
            logger.error(f"Failed to calculate B_gas from SolventExtendedPVTEngine: {bgas_error}")
            self.b_gas_rb_per_mscf = B_GAS_RB_PER_MSCF

        self.objective_functions = ObjectiveFunctions(
            self._base_operational_params,
            self._base_eor_params,
            self.reservoir,
            self.advanced_engine_params,
        )

        # Initialize EOS model if available
        if self.reservoir.eos_model and isinstance(self.reservoir.eos_model, EOSModelParameters):
            self.pvt.pvt_type = "compositional"
            self.eos_model_instance = self.reservoir.eos_model
        else:
            self.eos_model_instance = None
        self.reservoir_fluid = None

        # Initialize simulation engine directly
        self._init_simulation_engine()

    def _init_simulation_engine(self):
        """Initialize the simulation engine - surrogate primary engine."""
        try:
            recovery_model_type = getattr(
                self.advanced_engine_params, "recovery_model_type", "hybrid"
            )

            self.simulation_engine = SurrogateEngineWrapper(
                model_type="analytical", recovery_model_type=recovery_model_type
            )
            logger.info(f"Initialized surrogate simulation engine (model: {recovery_model_type})")

            self._engine_type = "surrogate"

        except ImportError as e:
            logger.warning(f"Could not initialize surrogate simulation engine: {e}")
            self.simulation_engine = None
            self._engine_type = None
        except Exception as e:
            logger.error(f"Unexpected error initializing surrogate simulation engine: {e}", exc_info=True)
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
        self.fitting_params = deepcopy(self._base_fitting_params)
        self.recovery_model = getattr(self.operational_params, "recovery_model_selection", "hybrid")
        self._unlocked_params_for_current_run = []
        self._mmp_value = self._mmp_value_init_override

        # Reset simulation engine if it exists
        if (
            hasattr(self, "simulation_engine")
            and self.simulation_engine is not None
            and hasattr(self.simulation_engine, "reset")
        ):
            self.simulation_engine.reset()

        # Update objective_functions to use the current reservoir instance
        self.objective_functions = ObjectiveFunctions(
            self.operational_params, self.eor_params, self.reservoir, self.advanced_engine_params
        )

    def re_initialize_dependent_components(self) -> None:
        """Re-initializes components that depend on parameter values.

        Called after parameter overrides are applied to ensure all dependent
        components are properly updated with the new values.
        """
        self.reset_to_base_state()
        # Re-initialize objective functions with current params
        self.objective_functions = ObjectiveFunctions(
            self.operational_params, self.eor_params, self.reservoir, self.advanced_engine_params
        )
        # Recalculate MMP if needed (force recalculation by clearing cache)
        if self._mmp_value_init_override is None:
            self._mmp_value = None

    @property
    def simulation_engine_type(self) -> Optional[str]:
        """
        Get the current simulation engine type.

        Returns the engine type string ('simple' or 'detailed') for UI access.
        Returns None if no engine is initialized.
        """
        if self._engine_type is None:
            return None
        return (
            self._engine_type.value
            if hasattr(self._engine_type, "value")
            else str(self._engine_type)
        )

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

    def _get_co2_fraction_from_eos(self) -> Optional[float]:
        """Get CO2 mole fraction using EOS model if available.

        Returns:
            CO2 mole fraction (0.0-1.0) or None if EOS not available.
        """
        eos_model = getattr(self, "eos_model_instance", None)
        if eos_model is None:
            return None
        try:
            pressure = getattr(self.eor_params, "target_pressure_psi", 2000.0) * 6894.76
            temp_f = getattr(self.eor_params, "default_temperature_f", 150.0)
            temp_k = (temp_f - 32) * 5.0 / 9.0 + 273.15
            props = eos_model.get_properties_si(temp_k, pressure)
            params = getattr(eos_model, "params", None)
            if params and hasattr(params, "mole_fractions"):
                return float(params.mole_fractions[0])
        except (RuntimeError, ValueError, ZeroDivisionError, ArithmeticError) as e:
            logger.warning(
                "EOS calculation failed at P=%.1f psi, T=%.1f °F: %s",
                getattr(self.eor_params, "target_pressure_psi", 2000.0),
                temp_f,
                e,
                exc_info=True,
            )
        return None

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
            logger.error(f"MMP calculation failed: {e}.", exc_info=True)
            raise OptimizationError(f"MMP calculation failed: {e}") from e

        return self._mmp_value

    def evaluate_for_analysis(
        self, eor_operational_params_dict: Dict[str, float], **kwargs
    ) -> Dict[str, Any]:
        eor_operational_params_dict = self._sanitize_and_discretize_parameters(
            eor_operational_params_dict
        )
        econ_params = kwargs.get("economic_params_override")
        if econ_params is None:
            econ_params = getattr(self, "economic_params", None)
        if econ_params is None:
            raise OptimizationError(
                "economic_params cannot be None - must provide economic_params_override or self.economic_params"
            )

        ooip = kwargs.get("ooip_override")
        if ooip is None:
            ooip = getattr(self.reservoir, "ooip_stb", None)
        if ooip is None:
            raise OptimizationError(
                "ooip cannot be None - must provide ooip_override or self.reservoir.ooip_stb"
            )

        mmp = kwargs.get("mmp_override")
        if mmp is None:
            mmp = getattr(self, "mmp", None)
        if mmp is None:
            raise OptimizationError(
                "mmp cannot be None - must provide mmp_override or self.mmp"
            )

        co2_storage_params = kwargs.get("co2_storage_params_override")
        if co2_storage_params is None:
            co2_storage_params = getattr(self, "co2_storage_params", None)
        if co2_storage_params is None:
            raise OptimizationError(
                "co2_storage_params cannot be None - must provide co2_storage_params_override or self.co2_storage_params"
            )

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
            # Map optimizer short names to actual EORParameters attributes
            elif key == "pressure" and hasattr(current_eor_params, "target_pressure_psi"):
                setattr(current_eor_params, "target_pressure_psi", value)
            elif key == "rate" and hasattr(current_eor_params, "injection_rate"):
                setattr(current_eor_params, "injection_rate", value)
            # injection_scheme is passed as string from _map_scheme_index_to_name
            elif key == "injection_scheme" and hasattr(current_eor_params, "injection_scheme"):
                setattr(current_eor_params, "injection_scheme", value)

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
                validation_reservoir.validate(
                    physics_based_model=True, tolerance=dimensional_tolerance
                )
                logger.info(f"Physics-based model '{self.recovery_model}' validation passed")

            except ValueError as e:
                raise OptimizationError(f"Physics-based model validation failed: {e}") from e

        # Use simulation_engine for all engine types (simple, detailed, surrogate)
        # The simulation_engine was created via EngineFactory based on engine_type setting
        simulation_mode = "co2_eor"
        if self.simulation_engine is not None:
            try:
                # Use the factory-created simulation engine
                sim_kwargs = kwargs.get("recovery_model_init_kwargs_override", {}).copy()
                sim_kwargs["mmp"] = mmp
                sim_kwargs.update(eor_operational_params_dict)

                n_injectors = 0
                n_producers = 0
                if self.well_data_list:
                    for w in self.well_data_list:
                        raw_type = str(w.metadata.get("type", "")).lower()
                        status_str = str(w.metadata.get("status", "")).lower()
                        name_str = str(w.name).lower()

                        is_injector = (
                            raw_type == "injector"
                            or "injector" in status_str
                            or "injector" in name_str
                            or name_str.startswith("inj")
                        )
                        if is_injector:
                            well_type = "injector"
                            n_injectors += 1
                        else:
                            well_type = "producer"
                            n_producers += 1
                        logger.debug(
                            f"Well '{w.name}': resolved_type={well_type}, raw_type={raw_type}, metadata={w.metadata}"
                        )
                else:
                    logger.warning("optimisation_engine.well_data_list is empty or None")

                scheme = str(getattr(self.eor_params, "injection_scheme", "continuous")).lower()
                # Field-wide pattern fallback: if continuous/WAG injection scheme is selected,
                # but no explicit injector was created, use 1 field-wide pattern injector
                if n_injectors == 0 and scheme in (
                    "continuous",
                    "wag",
                    "tapered_wag",
                    "water_alternating_gas",
                    "huff_n_puff",
                ):
                    logger.info(
                        f"No explicit injection wells defined for {scheme} scheme. Using 1 field-wide pattern injector."
                    )
                    n_injectors = 1
                if n_producers == 0 and scheme != "storage":
                    logger.info(
                        "No explicit production wells defined. Using 1 field-wide pattern producer."
                    )
                    n_producers = 1

                logger.info(f"Computed well counts for simulation: n_injectors={n_injectors}, n_producers={n_producers}")

                sim_mode = self._detect_simulation_mode(n_injectors, n_producers, scheme)
                simulation_mode = sim_mode
                logger.info(f"Simulation mode: {sim_mode}")

                if sim_mode == "invalid":
                    raise OptimizationError("No wells configured - cannot perform simulation")

                current_reservoir = deepcopy(self.reservoir)
                current_reservoir.ooip_stb = ooip

                sim_kwargs["simulation_mode"] = sim_mode

                results = self.simulation_engine.evaluate_scenario(
                    reservoir_data=current_reservoir,
                    eor_params=current_eor_params,
                    operational_params=self.operational_params,
                    economic_params=econ_params,
                    co2_storage_params=co2_storage_params,
                    fitting_params=kwargs.get(
                        "fitting_params_override", getattr(self, "fitting_params", None)
                    ),
                    n_injectors=n_injectors,
                    n_producers=n_producers,
                    **sim_kwargs,
                )

                # Check for explicit engine failure first to provide clear error message
                if results.get("convergence_status") == "error" or "error" in results:
                    error_msg = results.get("error", "Unknown simulation engine evaluation error")
                    raise SimulationEngineError(f"Simulation engine evaluation failed: {error_msg}")

                # Convert results to profile format expected by rest of code
                time_res = self.operational_params.time_resolution
                time_vector = results.get("time_vector", np.array([]))

                # Calculate time step sizes (dt) in days for integration of rates
                if len(time_vector) > 1:
                    dt = np.diff(
                        time_vector, prepend=time_vector[0] - (time_vector[1] - time_vector[0])
                    )
                else:
                    # Fallback for single-point results
                    dt = np.array([365.25]) if time_res == "yearly" else np.array([30.4])

                def get_rate(key):
                    val = results.get(key)
                    if val is None:
                        raise OptimizationError(f"Required key '{key}' not found in results")
                    if isinstance(val, (int, float, np.number)):
                        return np.full_like(dt, val)
                    if len(val) == 0 or len(val) != len(dt):
                        raise OptimizationError(f"Key '{key}' has invalid length")
                    return val

                # standardizing results: most engines return rates (per day)
                # but economics/objectives expect volumes per interval (e.g. STB per year)
                step_oil = get_rate("oil_production_rate") * dt
                step_total_gas = get_rate("total_gas_production_rate") * dt
                step_co2_prod = get_rate("co2_production_rate") * dt
                step_hc_gas = get_rate("hydrocarbon_gas_production_rate") * dt
                step_water = get_rate("water_production_rate") * dt
                step_pressure = get_rate("pressure")
                step_co2_inj = get_rate("co2_injection") * dt
                step_water_inj = get_rate("water_injection_rate") * dt

                project_lifetime_years = int(
                    getattr(self.operational_params, "project_lifetime_years", 15)
                )

                # Engine-calculated annual CO2 purchased/recycled (if provided by engine)
                engine_annual_purchased = results.get("annual_co2_purchased_mscf")
                engine_annual_recycled = results.get("annual_co2_recycled_mscf")

                # If simulation engine produced sub-annual (e.g. monthly) steps,
                # aggregate them into annual totals for yearly/annual profiles
                if len(time_vector) > 1 and len(time_vector) > project_lifetime_years:
                    annual_oil = np.zeros(project_lifetime_years)
                    annual_total_gas = np.zeros(project_lifetime_years)
                    annual_co2_prod = np.zeros(project_lifetime_years)
                    annual_hc_gas = np.zeros(project_lifetime_years)
                    annual_water = np.zeros(project_lifetime_years)
                    annual_pressure = np.zeros(project_lifetime_years)
                    annual_co2_inj = np.zeros(project_lifetime_years)
                    annual_water_inj = np.zeros(project_lifetime_years)
                    year_step_counts = np.zeros(project_lifetime_years)

                    t_mids = 0.5 * (time_vector[:-1] + time_vector[1:])
                    for i, t_mid in enumerate(t_mids):
                        y = min(project_lifetime_years - 1, max(0, int(t_mid // 365.25)))
                        annual_oil[y] += step_oil[i + 1]
                        annual_total_gas[y] += step_total_gas[i + 1]
                        annual_co2_prod[y] += step_co2_prod[i + 1]
                        annual_hc_gas[y] += step_hc_gas[i + 1]
                        annual_water[y] += step_water[i + 1]
                        annual_pressure[y] += step_pressure[i + 1]
                        annual_co2_inj[y] += step_co2_inj[i + 1]
                        annual_water_inj[y] += step_water_inj[i + 1]
                        year_step_counts[y] += 1

                    for y in range(project_lifetime_years):
                        if year_step_counts[y] > 0:
                            annual_pressure[y] /= year_step_counts[y]
                        elif y > 0:
                            annual_pressure[y] = annual_pressure[y - 1]

                    co2_recycle_eff = float(getattr(self.eor_params, "co2_recycling_efficiency", 0.95))

                    if (
                        engine_annual_recycled is not None
                        and len(engine_annual_recycled) == project_lifetime_years
                    ):
                        annual_co2_recycled = np.asarray(engine_annual_recycled, dtype=float)
                    else:
                        annual_co2_recycled = np.minimum(annual_co2_prod * co2_recycle_eff, annual_co2_inj)

                    if (
                        engine_annual_purchased is not None
                        and len(engine_annual_purchased) == project_lifetime_years
                    ):
                        annual_co2_purchased = np.asarray(engine_annual_purchased, dtype=float)
                    else:
                        annual_co2_purchased = np.maximum(0.0, annual_co2_inj - annual_co2_recycled)
                else:
                    annual_oil = step_oil
                    annual_total_gas = step_total_gas
                    annual_co2_prod = step_co2_prod
                    annual_hc_gas = step_hc_gas
                    annual_water = step_water
                    annual_pressure = step_pressure
                    annual_co2_inj = step_co2_inj
                    annual_water_inj = step_water_inj
                    co2_recycle_eff = float(getattr(self.eor_params, "co2_recycling_efficiency", 0.95))
                    annual_co2_recycled = (
                        np.asarray(engine_annual_recycled, dtype=float)
                        if engine_annual_recycled is not None
                        else np.minimum(step_co2_prod * co2_recycle_eff, step_co2_inj)
                    )
                    annual_co2_purchased = (
                        np.asarray(engine_annual_purchased, dtype=float)
                        if engine_annual_purchased is not None
                        else np.maximum(0.0, step_co2_inj - annual_co2_recycled)
                    )

                profiles = {
                    # Volume integrated profiles for objective calculations (yearly & annual aliases)
                    "yearly_oil_stb": annual_oil,
                    "annual_oil_stb": annual_oil,
                    "yearly_total_gas_mscf": annual_total_gas,
                    "annual_total_gas_mscf": annual_total_gas,
                    "yearly_co2_produced_mscf": annual_co2_prod,
                    "annual_co2_produced_mscf": annual_co2_prod,
                    "yearly_hc_gas_produced_mscf": annual_hc_gas,
                    "annual_hc_gas_produced_mscf": annual_hc_gas,
                    "yearly_water_stb": annual_water,
                    "annual_water_stb": annual_water,
                    "yearly_pressure": annual_pressure,
                    "annual_pressure": annual_pressure,
                    "yearly_co2_purchased_mscf": annual_co2_purchased,
                    "annual_co2_purchased_mscf": annual_co2_purchased,
                    "yearly_co2_recycled_mscf": annual_co2_recycled,
                    "annual_co2_recycled_mscf": annual_co2_recycled,
                    "yearly_co2_injected_mscf": annual_co2_inj,
                    "annual_co2_injected_mscf": annual_co2_inj,
                    "yearly_water_injected_bbl": annual_water_inj,
                    "annual_water_injected_bbl": annual_water_inj,

                    # Step-level (monthly) profiles
                    "monthly_oil_stb": step_oil,
                    "monthly_total_gas_mscf": step_total_gas,
                    "monthly_co2_produced_mscf": step_co2_prod,
                    "monthly_hc_gas_produced_mscf": step_hc_gas,
                    "monthly_water_stb": step_water,
                    "monthly_pressure": step_pressure,
                    "monthly_co2_purchased_mscf": np.maximum(0.0, step_co2_inj - np.minimum(step_co2_prod * co2_recycle_eff, step_co2_inj)),
                    "monthly_co2_recycled_mscf": np.minimum(step_co2_prod * co2_recycle_eff, step_co2_inj),
                    "monthly_co2_injected_mscf": step_co2_inj,
                    "monthly_water_injected_bbl": step_water_inj,

                    # Keep generic rate aliases for backward compatibility with plotters
                    "oil_production_rate": get_rate("oil_production_rate"),
                    "total_gas_production_rate": get_rate("total_gas_production_rate"),
                    "co2_production_rate": get_rate("co2_production_rate"),
                    "hydrocarbon_gas_production_rate": get_rate("hydrocarbon_gas_production_rate"),
                    "gas_production_rate": get_rate(
                        "total_gas_production_rate"
                    ),  # Backward compatibility
                    "water_production_rate": get_rate("water_production_rate"),
                    "co2_injection": get_rate("co2_injection"),
                    "co2_injection_mscf": step_co2_inj,
                    # Pre-calculated scalars
                    "npv": results.get("npv", 0.0),
                    "cumulative_oil": results.get("cumulative_oil", 0.0),
                    "co2_stored": results.get("co2_stored", 0.0),
                    "breakthrough_time_years": results.get("breakthrough_time_years"),
                    "storage_efficiency": results.get("storage_efficiency"),
                    "gross_utilization_mscf_per_stb": results.get("gross_utilization_mscf_per_stb"),
                    "net_utilization_mscf_per_stb": results.get("net_utilization_mscf_per_stb"),
                    "time_vector": time_vector,
                }
                rf = results.get("recovery_factor", 0.0)

                logger.info(
                    f"Used {results.get('engine_type', 'unknown')} engine for evaluation, RF={rf:.4f}"
                )

            except Exception as e:
                raise SimulationEngineError(f"Simulation engine evaluation failed: {e}") from e
        else:
            # Fallback: No simulation engine available, use ProductionProfiler directly
            logger.warning("No simulation engine available, using ProductionProfiler fallback")
            if ProductionProfiler is None:
                raise OptimizationError("ProductionProfiler module is not available")
            try:
                pressure_override = all_params.get("pressure")
                profiler = ProductionProfiler(
                    self.reservoir,
                    self.pvt,
                    current_eor_params,
                    self.operational_params,
                    current_profile_params,
                    initial_pressure_override=pressure_override,
                )
                self.profiler = profiler
                profiles = profiler.generate_all_profiles(ooip_stb=ooip)
                total_oil_produced = np.sum(
                    profiles.get(f"{self.operational_params.time_resolution}_oil_stb", 0)
                )
                rf = total_oil_produced / ooip if ooip > 0 else 0.0
            except ValueError as e:
                raise OptimizationError(f"Profiler fallback failed: {e}") from e

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
            profiles, rf, econ_params, current_co2_storage_params, simulation_mode=simulation_mode
        )

        if "gross_utilization_mscf_per_stb" in profiles and profiles["gross_utilization_mscf_per_stb"] is not None:
            objectives["gross_utilization_mscf_per_stb"] = profiles["gross_utilization_mscf_per_stb"]
        if "net_utilization_mscf_per_stb" in profiles and profiles["net_utilization_mscf_per_stb"] is not None:
            objectives["net_utilization_mscf_per_stb"] = profiles["net_utilization_mscf_per_stb"]

        # Validate simulation results
        # Skip RF validation for storage mode since no production wells = no oil production to validate
        is_storage_mode = getattr(self.eor_params, "injection_scheme", "").lower() == "storage"
        if not is_storage_mode and DataValidator is not None:
            reservoir_data_dict = {
                "ooip_stb": ooip,
                "initial_pressure": self.reservoir.initial_pressure,
                "max_pressure_psi": self.eor_params.max_pressure_psi,
            }
            validation_results = DataValidator.validate_simulation_results(
                profiles, objectives, reservoir_data_dict
            )
            DataValidator.log_validation_results(validation_results)
        elif not is_storage_mode:
            logger.debug("DataValidator not available - skipping simulation results validation")
        else:
            logger.info("Storage mode active - skipping RF/pressure validation (no production wells)")

        # Add breakthrough-specific metrics
        breakthrough_metrics = self._calculate_breakthrough_metrics(all_params, profiles)
        objectives.update(breakthrough_metrics)

        # Include raw profiles in the results for plotting or constraint checking
        # This prevents the need to re-run simulations
        objectives["profiles"] = profiles

        return objectives

    def _detect_simulation_mode(
        self, n_injectors: int, n_producers: int, injection_scheme: str = "continuous"
    ) -> str:
        """
        Detect simulation mode based on well counts and injection scheme.

        Args:
            n_injectors: Number of injector wells
            n_producers: Number of producer wells
            injection_scheme: EOR injection scheme

        Returns:
            Simulation mode: "co2_eor", "primary_production", "injection_storage", or "invalid"
        """
        scheme_lower = str(injection_scheme).lower()

        if scheme_lower in ("primary", "primary_production"):
            return "primary_production"
        elif scheme_lower == "storage":
            return "injection_storage"

        if n_injectors > 0 and n_producers > 0:
            return "co2_eor"
        elif n_injectors == 0 and n_producers > 0:
            if scheme_lower in ("continuous", "wag", "tapered_wag", "water_alternating_gas", "huff_n_puff"):
                return "co2_eor"
            return "primary_production"
        elif n_injectors > 0 and n_producers == 0:
            return "injection_storage"
        else:
            return "invalid"

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
                "injection_rate": all_params.get("rate", all_params.get("injection_rate", 5000.0)),
                "mobility_ratio": all_params.get("mobility_ratio", 2.0),
                "density_contrast": all_params.get("density_contrast", 0.3),
                "dip_angle": all_params.get("dip_angle", 0.0),
                "co2_viscosity": all_params.get("co2_viscosity", 0.02),
                "viscosity_oil": all_params.get("viscosity_oil", 4.0),
            }

            # breakthrough_time_years must be provided by surrogate engine
            bt_from_profiles = profiles.get("breakthrough_time_years")
            if bt_from_profiles is None:
                raise OptimizationError("breakthrough_time_years not provided by surrogate engine")
            breakthrough_time_scalar = float(bt_from_profiles)

            # Calculate breakthrough impact on economics
            project_lifetime = self.operational_params.project_lifetime_years
            breakthrough_impact = self._calculate_breakthrough_economic_impact(
                breakthrough_time_scalar, project_lifetime, profiles
            )

            # Ecology compliance: check if CO2 is properly shut-in after breakthrough
            shut_in_mode = int(all_params.get("shut_in_mode", 0))
            threshold_bpd = float(all_params.get("well_shut_in_threshold_bpd", 10.0))
            breakthrough_day = breakthrough_time_scalar * 365.25

            oil_profile = np.array(profiles.get("oil_production_rate", []))
            inj_profile = np.array(
                profiles.get("co2_injection", profiles.get("injection_profile", []))
            )
            time_vector = np.array(profiles.get("time_vector", np.arange(len(oil_profile))))
            co2_prod_profile = np.array(
                profiles.get("co2_production_rate", profiles.get("co2_gas_profile", []))
            )
            hc_gas_profile = np.array(
                profiles.get("hydrocarbon_gas_production_rate", profiles.get("solution_gas_profile", []))
            )
            water_profile = np.array(
                profiles.get("water_production_rate", profiles.get("water_profile", []))
            )

            shut_in_day = None
            ecology_compliant = True
            cumulative_co2_post_shut_in_tonne = 0.0

            if len(oil_profile) > 0 and len(time_vector) > 0:
                threshold_cross_day = None
                for i in range(len(time_vector)):
                    if time_vector[i] > breakthrough_day and oil_profile[i] < threshold_bpd:
                        threshold_cross_day = time_vector[i]
                        break

                if threshold_cross_day is not None:
                    shut_in_day = float(threshold_cross_day)
                    ramp_days = float(all_params.get("shut_in_ramp_days", 30.0))
                    # For ramped shut-in (mode 1), shut-in is complete after the taper window
                    effective_shut_in_day = (
                        shut_in_day + ramp_days if shut_in_mode == 1 else shut_in_day
                    )
                    post_shut_in_mask = time_vector >= effective_shut_in_day

                    if shut_in_mode in (0, 1):
                        oil_zero = len(oil_profile) == 0 or np.isclose(np.sum(oil_profile[post_shut_in_mask]), 0.0, atol=1e-3)
                        water_zero = len(water_profile) == 0 or np.isclose(np.sum(water_profile[post_shut_in_mask]), 0.0, atol=1e-3)
                        hc_zero = len(hc_gas_profile) == 0 or np.isclose(np.sum(hc_gas_profile[post_shut_in_mask]), 0.0, atol=1e-3)
                        co2_zero = len(co2_prod_profile) == 0 or np.isclose(np.sum(co2_prod_profile[post_shut_in_mask]), 0.0, atol=1e-3)
                        ecology_compliant = oil_zero and water_zero and hc_zero and co2_zero
                    elif shut_in_mode == 2:
                        hc_zero = len(hc_gas_profile) == 0 or np.isclose(np.sum(hc_gas_profile[post_shut_in_mask]), 0.0, atol=1e-3)
                        ecology_compliant = hc_zero

                    co2_density = getattr(self.eor_params, "co2_density_tonne_per_mscf", 0.053)
                    if len(co2_prod_profile) > 0 and len(post_shut_in_mask) == len(time_vector):
                        dt = np.diff(time_vector, prepend=0)
                        dt[0] = dt[1] if len(dt) > 1 else 1.0
                        cumulative_co2_post_shut_in_tonne = float(
                            np.sum(co2_prod_profile[post_shut_in_mask] * dt[post_shut_in_mask])
                            * co2_density
                        )

            return {
                "breakthrough_time_years": breakthrough_time_scalar,
                "breakthrough_impact_factor": breakthrough_impact,
                "ecology_compliant": ecology_compliant,
                "cumulative_co2_post_shut_in_tonne": cumulative_co2_post_shut_in_tonne,
                "shut_in_day": shut_in_day if shut_in_day is not None else -1.0,
                "shut_in_mode": shut_in_mode,
            }

        except Exception as e:
            raise OptimizationError(f"Breakthrough metrics calculation failed: {e}") from e

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
        annual_oil = profiles.get("annual_oil_stb")
        if annual_oil is None or len(annual_oil) == 0:
            logger.info("annual_oil_stb not available or empty - skipping DCA")
            return None

        if self.dca_analyzer is None:
            logger.info("dca_analyzer is not available - skipping DCA")
            return None

        time_years = np.arange(1, len(annual_oil) + 1)

        b_factor = None
        if optimized_params and "hyperbolic_b_factor" in optimized_params:
            b_factor = optimized_params["hyperbolic_b_factor"]

        dca_result = self.dca_analyzer.analyze_production(
            time=time_years,
            production_rate=annual_oil,
            model_type="auto",
            forecast_years=30,
            time_unit="years",
            b_factor=b_factor,
        )

        dca_data = self.dca_analyzer.generate_dca_report_data(dca_result)
        return dca_data

    def _calculate_adaptive_penalty(
        self, eval_results: Dict[str, Any], current_gen: Optional[int] = None, max_gens: Optional[int] = None
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

    def _check_profile_constraints(
        self, eval_results: Dict[str, float], current_gen: int, max_gens: int
    ) -> Dict[str, Any]:
        """
        Proportional profile-based constraint checking for environmental guardrails.

        Instead of hard cliffs, we compute proportional penalties that scale with
        violation severity. This provides a gradient for the GA to climb.

        Args:
            eval_results: Evaluation results from simulation
            current_gen: Current GA generation
            max_gens: Maximum number of generations

        Returns:
            Dictionary with:
                - penalty: Proportional penalty value (0.0 = no violation)
                - violations: List of violation descriptions
                - details: Dictionary with constraint metrics for logging
        """
        result = {
            "penalty": 0.0,
            "violations": [],
            "details": {},
        }

        profiles = eval_results.get("profiles", {})
        time_res = getattr(self.operational_params, "time_resolution", "daily")
        co2_density = getattr(self.eor_params, "co2_density_tonne_per_mscf", 0.053)

        # Get constraint parameters
        min_inj_period = self.advanced_engine_params.min_injection_period_fraction
        min_storage_eff = self.advanced_engine_params.min_avg_storage_efficiency
        max_leakage_frac = self.advanced_engine_params.max_annual_leakage_fraction
        carbon_tax = self.advanced_engine_params.carbon_tax_usd_per_tonne

        # Check 1: Injection Period Fraction - proportional penalty
        inj_profile = np.array(
            profiles.get("co2_injection", profiles.get(f"{time_res}_co2_injected_mscf", []))
        )
        if len(inj_profile) > 0:
            max_rate = np.max(inj_profile) if np.any(inj_profile > 0) else 1.0
            threshold_rate = max_rate * 0.05
            active_periods = np.sum(inj_profile > threshold_rate)
            inj_period_fraction = active_periods / len(inj_profile)

            result["details"]["inj_period_fraction"] = inj_period_fraction
            result["details"]["min_inj_period_required"] = min_inj_period

            if inj_period_fraction < min_inj_period:
                shortfall = min_inj_period - inj_period_fraction
                # Proportional penalty: scales with shortfall
                # Base penalty is 5% of failure penalty per 1% shortfall
                base_penalty = abs(self.advanced_engine_params.failure_penalty) * 0.05
                penalty = base_penalty * shortfall * 100
                result["penalty"] += penalty
                result["violations"].append(
                    f"Injection period {inj_period_fraction:.2%} < min {min_inj_period:.2%} (shortfall: {shortfall:.2%})"
                )

        # Check 2: Storage Efficiency - proportional penalty
        avg_se = eval_results.get("avg_storage_efficiency", 1.0)
        result["details"]["avg_storage_efficiency"] = avg_se
        result["details"]["min_storage_efficiency_required"] = min_storage_eff

        if avg_se < min_storage_eff:
            shortfall = min_storage_eff - avg_se
            # Proportional penalty: 10% of failure penalty per 1% shortfall
            base_penalty = abs(self.advanced_engine_params.failure_penalty) * 0.10
            penalty = base_penalty * shortfall * 100
            result["penalty"] += penalty
            result["violations"].append(
                f"Storage efficiency {avg_se:.2%} < min {min_storage_eff:.2%} (shortfall: {shortfall:.2%})"
            )

        # Check 3: Leakage - proportional penalty based on carbon tax
        co2_inj_mscf = np.array(profiles.get(f"{time_res}_co2_purchased_mscf", []))
        co2_prod_mscf = np.array(profiles.get(f"{time_res}_co2_produced_mscf", []))
        co2_recycled_mscf = np.array(profiles.get(f"{time_res}_co2_recycled_mscf", []))

        if len(co2_inj_mscf) > 0 and len(co2_prod_mscf) > 0:
            co2_fraction = self._get_co2_fraction_from_eos()
            if co2_fraction is None:
                raise OptimizationError("CO2 fraction is None and no fallback available - _get_co2_fraction_from_eos returned None")

            total_purchased_tonne = np.sum(co2_inj_mscf) * co2_density
            total_recycled_tonne = np.sum(co2_recycled_mscf) * co2_density
            total_produced_tonne = np.sum(co2_prod_mscf) * co2_density

            if total_purchased_tonne > 0:
                # True physical caprock leakage comes from containment breach or explicit seal flux,
                # NEVER from normal wellbore production (which is the objective of EOR).
                if "annual_leakage_tonne" in eval_results:
                    leakage_tonne = float(np.sum(eval_results["annual_leakage_tonne"]))
                elif "annual_leakage_tonne" in profiles:
                    leakage_tonne = float(np.sum(profiles["annual_leakage_tonne"]))
                else:
                    # Geomechanical seal check: injection pressure vs Class VI frac limit
                    p_sandface = eval_results.get("max_sandface_pressure_psi", 0.0)
                    caprock_p = getattr(self.eor_params, "caprock_fracture_pressure_psi", 5500.0)
                    safety_factor = getattr(self.eor_params, "caprock_safety_factor", 0.90)
                    p_seal = caprock_p * safety_factor
                    if p_sandface > p_seal:
                        overpressure_ratio = (p_sandface - p_seal) / max(p_seal, 1.0)
                        leakage_tonne = total_purchased_tonne * min(0.10, overpressure_ratio)
                    else:
                        leakage_tonne = 0.0

                leakage_fraction = leakage_tonne / total_purchased_tonne

                result["details"]["leakage_tonne"] = leakage_tonne
                result["details"]["leakage_fraction"] = leakage_fraction
                result["details"]["max_leakage_fraction_allowed"] = max_leakage_frac
                result["details"]["co2_fraction_used"] = co2_fraction

                if leakage_fraction > max_leakage_frac:
                    excess_leakage_tonne = leakage_tonne - (
                        total_purchased_tonne * max_leakage_frac
                    )
                    penalty = carbon_tax * excess_leakage_tonne
                    result["penalty"] += penalty
                    result["violations"].append(
                        f"Leakage {leakage_tonne:.0f}tonne ({leakage_fraction:.2%}) > max {max_leakage_frac:.2%}"
                    )

        return result

    def _check_parameter_constraints(
        self, params_dict: Dict[str, float]
    ) -> Tuple[bool, List[str], float]:
        """
        Check if optimized parameters satisfy all configuration constraints.

        Returns:
            Tuple of (is_feasible, list_of_violations, penalty_amount)
            penalty_amount > 0 means violation occurred, value is the penalty to apply
        """
        violations = []
        penalty = 0.0

        eor_bounds = [
            (
                "rate",
                self.eor_params.min_injection_rate_mscfd,
                self.eor_params.max_injection_rate_mscfd,
            ),
            (
                "gravity_factor",
                self.eor_params.min_gravity_factor,
                self.eor_params.max_gravity_factor,
            ),
            ("sor", self.eor_params.min_sor, self.eor_params.max_sor),
            (
                "transition_alpha",
                self.eor_params.min_transition_alpha,
                self.eor_params.max_transition_alpha,
            ),
            (
                "transition_beta",
                self.eor_params.min_transition_beta,
                self.eor_params.max_transition_beta,
            ),
            (
                "productivity_index",
                self.eor_params.min_productivity_index,
                self.eor_params.max_productivity_index,
            ),
            (
                "wellbore_pressure",
                self.eor_params.min_wellbore_pressure,
                self.eor_params.max_wellbore_pressure,
            ),
            (
                "max_production_rate_stbd",
                self.eor_params.min_max_production_rate,
                self.eor_params.max_max_production_rate,
            ),
            (
                "plateau_duration_fraction",
                self.eor_params.min_plateau_duration_fraction,
                self.eor_params.max_plateau_duration_fraction,
            ),
            (
                "ramp_up_fraction",
                self.eor_params.min_ramp_up_fraction,
                self.eor_params.max_ramp_up_fraction,
            ),
            (
                "hyperbolic_b_factor",
                self.eor_params.min_hyperbolic_b_factor,
                self.eor_params.max_hyperbolic_b_factor,
            ),
        ]

        for param_name, min_val, max_val in eor_bounds:
            if param_name in params_dict:
                val = params_dict[param_name]
                if val < min_val:
                    violations.append(f"{param_name}={val:.4f} below min {min_val:.4f}")
                    penalty += abs(min_val - val) * 1e6
                elif val > max_val:
                    violations.append(f"{param_name}={val:.4f} above max {max_val:.4f}")
                    penalty += abs(val - max_val) * 1e6

        if self.eor_params.injection_scheme in ["wag", "swag"]:
            wag = params_dict.get("wag_ratio")
            if wag is not None:
                if wag < self.eor_params.min_wag_ratio:
                    violations.append(
                        f"wag_ratio={wag:.4f} below min {self.eor_params.min_wag_ratio}"
                    )
                    penalty += abs(self.eor_params.min_wag_ratio - wag) * 1e6
                elif wag > self.eor_params.max_wag_ratio:
                    violations.append(
                        f"wag_ratio={wag:.4f} above max {self.eor_params.max_wag_ratio}"
                    )
                    penalty += abs(wag - self.eor_params.max_wag_ratio) * 1e6

            cycle_length = params_dict.get("cycle_length_days")
            if cycle_length is not None:
                if cycle_length < self.eor_params.min_cycle_length_days:
                    violations.append(
                        f"cycle_length_days={cycle_length:.1f} below min {self.eor_params.min_cycle_length_days}"
                    )
                    penalty += abs(self.eor_params.min_cycle_length_days - cycle_length) * 1e4
                elif cycle_length > self.eor_params.max_cycle_length_days:
                    violations.append(
                        f"cycle_length_days={cycle_length:.1f} above max {self.eor_params.max_cycle_length_days}"
                    )
                    penalty += abs(cycle_length - self.eor_params.max_cycle_length_days) * 1e4

        water_frac = params_dict.get("water_fraction")
        if water_frac is not None:
            if water_frac < self.eor_params.min_water_fraction:
                violations.append(
                    f"water_fraction={water_frac:.3f} below min {self.eor_params.min_water_fraction}"
                )
                penalty += abs(self.eor_params.min_water_fraction - water_frac) * 1e6
            elif water_frac > self.eor_params.max_water_fraction:
                violations.append(
                    f"water_fraction={water_frac:.3f} above max {self.eor_params.max_water_fraction}"
                )
                penalty += abs(water_frac - self.eor_params.max_water_fraction) * 1e6

        pressure = params_dict.get("pressure", 0)
        max_pressure_limit = (
            self.eor_params.max_pressure_psi
            * self.advanced_engine_params.fracture_pressure_multiplier
        )
        if pressure > max_pressure_limit:
            violations.append(
                f"Pressure {pressure:.1f} psi exceeds fracture limit {max_pressure_limit:.1f} - clipped"
            )
            params_dict["pressure"] = max_pressure_limit
            penalty += (pressure - max_pressure_limit) * 1e3

        is_feasible = penalty < 1e8
        return is_feasible, violations, penalty

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

        params_dict = self._sanitize_and_discretize_parameters(params_dict)

        is_feasible, violations, constraint_penalty = self._check_parameter_constraints(params_dict)
        if constraint_penalty > 0:
            logger.warning(f"Parameter constraint violations: {violations}")

        try:
            eval_results = self.evaluate_for_analysis(
                params_dict,
                economic_params_override=self.economic_params,
                ooip_override=self.reservoir.ooip_stb,
                mmp_override=self.mmp or self.eor_params.default_mmp_fallback,
                co2_storage_params_override=self.co2_storage_params,
            )
        except Exception as e:
            logger.warning(
                f"Simulation evaluation failed for candidate parameters: {e}. Applying failure penalty."
            )
            return FAILURE_PENALTY

        simulation_mode = eval_results.get("simulation_mode", "co2_eor")

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
        if isinstance(storage_efficiency, np.ndarray):
            storage_efficiency = float(storage_efficiency.item())
        logger.debug(f"Storage efficiency: {storage_efficiency:.6f}")

        # PRIMARY PRODUCTION MODE: Skip storage efficiency checks (0 is expected)
        if simulation_mode == "primary_production":
            logger.info(
                f"Primary production mode - storage metrics N/A (RF={eval_results.get('recovery_factor', 0):.3f})"
            )
        elif storage_efficiency <= 1e-6:
            profiles = eval_results.get("profiles", {})
            time_res = getattr(self.operational_params, "time_resolution", "daily")
            co2_inj_key = f"{time_res}_co2_injected_mscf"
            total_injected_mscf = 0.0
            if profiles:
                inj_profile = profiles.get(co2_inj_key, np.array([]))
                if len(inj_profile) == 0:
                    inj_profile = profiles.get("co2_injection_mscf", np.array([]))
                if len(inj_profile) == 0:
                    inj_profile = profiles.get("co2_injection", np.array([]))
                if len(inj_profile) > 0:
                    total_injected_mscf = float(np.sum(inj_profile))

            if total_injected_mscf <= 0:
                recovery_factor = eval_results.get("recovery_factor", 0.0)
                if recovery_factor > 0.05:
                    logger.info(
                        f"Storage efficiency is 0.0 due to zero CO2 injection (early shut-in or primary production scenario). "
                        f"RF={recovery_factor:.3f}."
                    )
                    eval_results["storage_efficiency"] = 0.0
                    storage_efficiency = 0.0
                else:
                    logger.warning(
                        f"Sanity Check Failed: Storage efficiency is extremely low ({storage_efficiency:.6f}). "
                        f"Applying failure penalty (no oil production + no CO2 injected)."
                    )
                    return FAILURE_PENALTY
            else:
                logger.warning(
                    f"Sanity Check Failed: Storage efficiency is extremely low ({storage_efficiency:.6f}) "
                    f"despite CO2 injection ({total_injected_mscf:.1f} mscf). Applying failure penalty."
                )
                return FAILURE_PENALTY

        # 3. Plume Containment Constraint (PhD Geomechanical Formula)
        # S_cont = γ_safety * [w_p * S_press + w_s * S_seal + w_t * S_struct]
        # S_press = max(0, 1.0 - P̄_inj / (P_frac * λ_limit))
        fracture_pressure = (
            self.eor_params.max_pressure_psi
            * self.advanced_engine_params.fracture_pressure_multiplier
        )
        time_res = getattr(self.operational_params, "time_resolution", "daily")
        pressure_profile = eval_results.get(f"{time_res}_pressure", np.array([]))
        containment_score = calculate_geomechanical_containment_score(
            pressure_profile,
            fracture_pressure,
            self.advanced_engine_params,
        )
        logger.debug(f"Geomechanical containment score: {containment_score:.3f}")
        critical_threshold = self.advanced_engine_params.containment_critical_threshold
        if containment_score < critical_threshold:
            logger.error(
                f"Containment score {containment_score:.3f} below critical threshold "
                f"{critical_threshold:.3f}. Solution PRUNED with FAILURE_PENALTY."
            )
            return FAILURE_PENALTY

        avg_storage_efficiency = eval_results.get("avg_storage_efficiency", 0.0)
        objective_value = eval_results.get(self.chosen_objective, -1e12)

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
        elif self.chosen_objective in ("miscibility_degree", "average_miscibility_degree"):
            result = eval_results.get("average_miscibility_degree", 0.0) * 1e6  # Maximize miscibility degree
        else:
            result = objective_value  # Default behavior for other objectives (NPV, recovery_factor)

        # 4. Profile-Based Environmental Constraints (Proportional Penalties)
        max_gens = 100
        if hasattr(self, "ga_params_current_run") and self.ga_params_current_run:
            max_gens = self.ga_params_current_run.num_generations

        profile_constraint_result = self._check_profile_constraints(
            eval_results,
            current_gen=current_gen if current_gen is not None else 0,
            max_gens=max_gens,
        )
        profile_penalty = profile_constraint_result["penalty"]
        if profile_penalty > 0:
            result -= profile_penalty
            for violation in profile_constraint_result["violations"]:
                logger.warning(f"Constraint violation: {violation}")
            logger.warning(f"Total profile constraint penalty: {profile_penalty:.2e}")

        # Target miscibility degree dictation penalty (if user/scenario specifies target_miscibility_degree)
        target_omega = getattr(self.eor_params, "target_miscibility_degree", None)
        if target_omega is not None:
            actual_omega = float(eval_results.get("average_miscibility_degree", 0.0))
            penalty_scale = float(getattr(self.eor_params, "miscibility_weight_penalty", 1000.0))
            omega_penalty = penalty_scale * ((actual_omega - float(target_omega)) ** 2) * max(1.0, abs(result) * 0.05)
            result -= omega_penalty
            logger.debug(f"Target miscibility dictation: target={target_omega:.3f}, actual={actual_omega:.3f}, penalty={omega_penalty:.2e}")

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

        min_breakthrough_time = self.advanced_engine_params.breakthrough_time_min_years
        if breakthrough_time < min_breakthrough_time:
            if isinstance(breakthrough_time, np.ndarray):
                breakthrough_time = float(breakthrough_time.item())
            deficit = float(min_breakthrough_time - breakthrough_time)
            # Smooth, continuous quadratic penalty preserving gradient information
            bt_penalty = 1000.0 * (deficit ** 2) * max(1.0, abs(result) * 0.1)
            result -= bt_penalty
            logger.debug(
                f"Breakthrough constraint violated: {breakthrough_time:.2f} years < {min_breakthrough_time:.2f} year. Smooth penalty applied: {bt_penalty:.2e}."
            )

        # Check for invalid or unphysical objective values (prune unviable chromosomes)
        if objective_value is None or np.isnan(objective_value) or np.isinf(objective_value):
            logger.warning(
                f"Objective value for '{self.chosen_objective}' is invalid ({objective_value}). "
                f"Pruning chromosome with FAILURE_PENALTY."
            )
            return FAILURE_PENALTY

        # Apply breakthrough impact factor
        if not np.isclose(breakthrough_impact, 1.0):
            logger.info(
                f"Applying breakthrough impact factor of {breakthrough_impact:.3f} to objective score."
            )
        result *= breakthrough_impact

        ecology_compliant = bool(eval_results.get("ecology_compliant", False))
        shut_in_mode = int(eval_results.get("shut_in_mode", 0))
        cumulative_co2_post_shut_in = float(
            eval_results.get("cumulative_co2_post_shut_in_tonne", 0.0)
        )
        if not ecology_compliant and shut_in_mode in (0, 1):
            co2_penalty = min(cumulative_co2_post_shut_in * 0.5, abs(FAILURE_PENALTY) * 0.3)
            if co2_penalty > 0:
                logger.warning(
                    f"ECOLOGY VIOLATION: CO2 production after shut-in = {cumulative_co2_post_shut_in:.2f} tonnes. "
                    f"Penalty: ${co2_penalty:.2f}"
                )
                result -= co2_penalty

        # Apply constraint penalty if any parameter violations occurred
        if constraint_penalty > 0:
            result -= constraint_penalty
            logger.warning(
                f"Applied constraint penalty: {constraint_penalty:.4f} (Result: {result:.4f})"
            )

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
        return float(np.clip(result, MIN_OBJECTIVE_VALUE, MAX_OBJECTIVE_VALUE))

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Defines the search space for the optimization variables.

        NOTE: Physical parameters (sor, productivity_index, gravity_factor, etc.)
        are LOCKED and not optimized. They must be set from lab/geological data.
        Only operational parameters that engineers can actually control are optimized.
        """
        mmp_val = self.mmp or self.eor_params.default_mmp_fallback

        caprock_fracture_pressure = getattr(
            self.eor_params, "caprock_fracture_pressure_psi", 5500.0
        )
        safety_factor = getattr(self.eor_params, "caprock_safety_factor", 0.90)
        safe_fracture_ceiling = caprock_fracture_pressure * safety_factor

        # Account for near-wellbore transient injection overpressure per well: Delta P_inj = q_well / II
        inj_rate_max = getattr(self.eor_params, "max_injection_rate_mscfd", 10000.0)
        n_inj = max(1, getattr(self.eor_params, "active_injectors", 1))
        rate_per_well = inj_rate_max / n_inj
        ii = getattr(self.eor_params, "injectivity_index", 25.0)
        # II is in MSCFD/psi; bound realistic delta_p_inj to field limits (< 500 psi)
        delta_p_inj = min(500.0, rate_per_well / max(ii, 5.0))
        min_res_pressure = mmp_val * self.eor_params.min_pressure_factor
        max_safe_res_pressure = max(
            min_res_pressure + 500.0, safe_fracture_ceiling - delta_p_inj
        )

        min_bhp = getattr(self.eor_params, "min_producer_bhp_psi", 1500.0)
        max_bhp = getattr(self.eor_params, "max_wellbore_pressure", 3500.0)

        b = {
            "pressure": (
                min_res_pressure,
                max_safe_res_pressure,
            ),
            "rate": (
                self.eor_params.min_injection_rate_mscfd,
                self.eor_params.max_injection_rate_mscfd,
            ),
            "plateau_duration_fraction": (
                self.eor_params.min_plateau_duration_fraction,
                self.eor_params.max_plateau_duration_fraction,
            ),
            "ramp_up_fraction": (
                self.eor_params.min_ramp_up_fraction,
                self.eor_params.max_ramp_up_fraction,
            ),
            "wellbore_pressure": (
                min_bhp,
                max(min_bhp + 100.0, max_bhp),
            ),
            "max_production_rate_stbd": (
                self.eor_params.min_max_production_rate,
                self.eor_params.max_max_production_rate,
            ),
            "well_shut_in_threshold_bpd": (5, 30),
            "allow_well_conversion": (0, 1),
            "well_conversion_day": (90, 1825),
            "shut_in_mode": (0, 2),
            "shut_in_ramp_days": (7, 90),
        }

        # Injection scheme - discrete GA variable using numeric index
        # (pygad requires numeric values in gene_space, actual scheme mapped later)
        if not self.eor_params.injection_scheme_locked:
            b["injection_scheme"] = (0, 4)  # indices 0-4 for 5 schemes

        active_scheme = getattr(self.eor_params, "injection_scheme", "continuous")
        is_locked = getattr(self.eor_params, "injection_scheme_locked", True)

        # Tapered injection parameters - conditionally include if active or unlocked
        if not is_locked or active_scheme == "tapered":
            b["tapered_duration_years"] = (
                self.eor_params.min_tapered_duration_years,
                self.eor_params.max_tapered_duration_years,
            )
            b["tapered_final_rate_multiplier"] = (
                self.eor_params.min_tapered_final_rate_multiplier,
                self.eor_params.max_tapered_final_rate_multiplier,
            )
            b["tapered_initial_rate_multiplier"] = (
                self.eor_params.min_tapered_initial_rate_multiplier,
                self.eor_params.max_tapered_initial_rate_multiplier,
            )

        # WAG parameters - conditionally include if active or unlocked
        if not is_locked or active_scheme in ("wag", "swag"):
            b["wag_ratio"] = (self.eor_params.min_wag_ratio, self.eor_params.max_wag_ratio)
            b["cycle_length_days"] = (
                self.eor_params.min_cycle_length_days,
                self.eor_params.max_cycle_length_days,
            )

        # huff_n_puff parameters - conditionally include if active or unlocked
        if not is_locked or active_scheme == "huff_n_puff":
            b["huff_n_puff_injection_period_days"] = (
                self.eor_params.min_huff_n_puff_injection_period_days,
                self.eor_params.max_huff_n_puff_injection_period_days,
            )
            b["huff_n_puff_soaking_period_days"] = (
                self.eor_params.min_huff_n_puff_soaking_period_days,
                self.eor_params.max_huff_n_puff_soaking_period_days,
            )
            b["huff_n_puff_production_period_days"] = (
                self.eor_params.min_huff_n_puff_production_period_days,
                self.eor_params.max_huff_n_puff_production_period_days,
            )
            b["huff_n_puff_max_cycles"] = (
                self.eor_params.min_huff_n_puff_max_cycles,
                self.eor_params.max_huff_n_puff_max_cycles,
            )

        for param_key in getattr(self, "_unlocked_params_for_current_run", []):
            if not (constraint_info := self.RELAXABLE_CONSTRAINTS.get(param_key)):
                continue
            base_val = getattr(
                self._base_eor_params,
                param_key,
                getattr(self._base_reservoir_data, param_key, None),
            )
            if base_val is not None:
                rf = float(constraint_info["range_factor"])
                b[param_key] = (base_val * (1 - rf), base_val * (1 + rf))
                logger.info(
                    f"Re-run: Overriding bounds for unlocked param '{param_key}' to: ({b[param_key][0]:.3g}, {b[param_key][1]:.3g})"
                )
        return b

    def _sanitize_and_discretize_parameters(
        self, params_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensures discrete integer parameters are strictly integer-typed and prunes
        irrelevant cyclic/scheme-specific parameters when a scheme is locked.
        """
        sanitized = dict(params_dict)
        if "shut_in_mode" in sanitized:
            sanitized["shut_in_mode"] = int(np.clip(round(float(sanitized["shut_in_mode"])), 0, 2))
        if "allow_well_conversion" in sanitized:
            sanitized["allow_well_conversion"] = int(
                np.clip(round(float(sanitized["allow_well_conversion"])), 0, 1)
            )
        if "injection_scheme" in sanitized and isinstance(
            sanitized["injection_scheme"], (int, float, np.number)
        ):
            sanitized["injection_scheme"] = int(
                np.clip(round(float(sanitized["injection_scheme"])), 0, 4)
            )

        active_scheme = getattr(self.eor_params, "injection_scheme", "continuous")
        is_locked = getattr(self.eor_params, "injection_scheme_locked", True)

        if is_locked:
            if active_scheme == "continuous":
                for k in list(sanitized.keys()):
                    if (
                        k.startswith("huff_n_puff_")
                        or k.startswith("tapered_")
                        or k in ("wag_ratio", "cycle_length_days")
                    ):
                        sanitized.pop(k, None)
            elif active_scheme in ("wag", "swag"):
                for k in list(sanitized.keys()):
                    if k.startswith("huff_n_puff_") or k.startswith("tapered_"):
                        sanitized.pop(k, None)
            elif active_scheme == "huff_n_puff":
                for k in list(sanitized.keys()):
                    if k.startswith("tapered_") or k in ("wag_ratio", "cycle_length_days"):
                        sanitized.pop(k, None)

        return sanitized

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
        self,
        final_eval: Dict[str, Any],
        final_results: Dict[str, Any],
        handle_miss: bool,
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
    def _map_scheme_index_to_name(self, params_dict):
        """Map injection_scheme index to actual scheme name."""
        if "injection_scheme" in params_dict and not isinstance(
            params_dict["injection_scheme"], str
        ):
            idx = int(params_dict["injection_scheme"])
            params_dict["injection_scheme"] = INJECTION_SCHEMES[idx]
        return params_dict

    def _fitness_func_pygad(self, ga_instance, solutions, solution_idx):
        """Fitness function for pygad that supports both single and batch evaluation."""
        if solutions.ndim == 1:
            # Single solution evaluation
            param_names = list(self._get_parameter_bounds().keys())
            params_dict = {name: value for name, value in zip(param_names, solutions)}
            # Map injection_scheme index to actual scheme name
            params_dict = self._map_scheme_index_to_name(params_dict)
            # Pass current generation for adaptive penalty
            current_gen = (
                ga_instance.generations_completed
                if hasattr(ga_instance, "generations_completed")
                else 0
            )

            # Check if multi-objective optimization (NSGA-II)
            ga_params = getattr(self, "ga_params_current_run", None)
            if ga_params and ga_params.num_objectives == 2:
                # Multi-objective: return array [obj1, obj2]
                obj1 = self._objective_function_wrapper(
                    current_gen=current_gen, chosen_objective=self.chosen_objective, **params_dict
                )
                obj2 = self._objective_function_wrapper(
                    current_gen=current_gen,
                    chosen_objective=ga_params.secondary_objective,
                    **params_dict,
                )
                return [obj1, obj2]
            else:
                return self._objective_function_wrapper(current_gen=current_gen, **params_dict)
        else:
            # Batch evaluation - use parallel processing
            # Create safe instance for multiprocessing
            safe_self = PickleSafeOptimiser(self)
            return safe_self._evaluate_solutions_parallel(solutions)

    def _on_generation_callback(self, ga_instance):
        """Callback function for GA generations that is pickleable for multiprocessing."""
        # Calculate generation statistics
        current_gen = ga_instance.generations_completed
        last_fitness = list(ga_instance.last_generation_fitness)

        # 1. Track full population fitness history across generations
        if not hasattr(ga_instance, "all_fitness") or ga_instance.all_fitness is None:
            ga_instance.all_fitness = []
        ga_instance.all_fitness.append(last_fitness)

        # 2. Track areal sweep coverage metrics across generations
        if not hasattr(ga_instance, "coverage_history") or ga_instance.coverage_history is None:
            ga_instance.coverage_history = []

        try:
            from core.engine_surrogate.surrogate_models import calculate_areal_sweep_efficiency
            bounds = self._get_parameter_bounds()
            param_names = list(bounds.keys())
            mu_oil = getattr(getattr(self, "pvt", None), "oil_viscosity_cp", None) or 1.5
            mu_co2 = getattr(getattr(self, "pvt", None), "gas_viscosity_cp", None) or 0.05
            base_mr = mu_oil / max(mu_co2, 1e-6)

            gen_sweeps = []
            for sol in ga_instance.population:
                params_dict = {name: val for name, val in zip(param_names, sol)}
                m_ratio = params_dict.get("mobility_ratio", base_mr)
                gen_sweeps.append(calculate_areal_sweep_efficiency(m_ratio))

            if gen_sweeps:
                gen_sweeps_arr = np.array(gen_sweeps)
                ga_instance.coverage_history.append({
                    "generation": current_gen,
                    "min": float(np.min(gen_sweeps_arr)),
                    "max": float(np.max(gen_sweeps_arr)),
                    "mean": float(np.mean(gen_sweeps_arr)),
                    "std": float(np.std(gen_sweeps_arr)),
                })
        except Exception as e:
            logger.debug(f"Could not compute generation coverage: {e}")

        best_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        valid_fitness = [f for f in last_fitness if f > -1e9]
        avg_fitness = float(np.mean(valid_fitness)) if valid_fitness else float(np.mean(last_fitness))
        std_fitness = float(np.std(valid_fitness)) if valid_fitness else float(np.std(last_fitness))

        # Log detailed generation statistics
        logger.info(
            f"GA Generation {current_gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Std={std_fitness:.4f}, "
            f"Evaluations={current_gen * self._ga_sol_per_pop}"
        )

        # Handle mechanisms for escaping local optima (stale restart)
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
                        **self._map_scheme_index_to_name(
                            {name: val for name, val in zip(param_names, sol)}
                        )
                    )
                    for sol in solutions
                ]
            )

        # Prepare parameter dictionaries for parallel evaluation
        params_list = []
        for solution in solutions:
            params_dict = {name: value for name, value in zip(param_names, solution)}
            params_dict = self._map_scheme_index_to_name(params_dict)
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
            results: List[Any] = [None] * len(params_list)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise OptimizationError(f"Parallel evaluation failed for solution {idx}: {e}") from e

        return np.array(results)

    def _generate_well_schedule_from_params(
        self,
        eor_params: "EORParameters",
        operational_params: "OperationalParameters",
        time_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate per-well operational schedule data for visualization.

        Generates schedule data for all injection schemes:
        - continuous: Constant injection (no cycles)
        - wag: Water-Alternating-Gas cycles
        - huff_n_puff: Inject-Soak-Produce cycles
        - tapered: Decreasing injection rate over time
        - swag: Simultaneous Water and Gas
        - pulsed: Intermittent high-intensity pulses

        Args:
            eor_params: EOR parameters including injection_scheme and scheme-specific params
            operational_params: Project parameters including lifetime
            time_vector: Monthly time points for the simulation

        Returns:
            Dictionary with well schedules for all available wells
        """
        well_data_list = self.well_data_list or []

        def is_injector_well(w):
            if not hasattr(w, "metadata") or not isinstance(w.metadata, dict):
                name = getattr(w, "name", "").lower()
                return "injector" in name or name.startswith("inj")
            raw_type = str(w.metadata.get("type", "")).lower()
            status_str = str(w.metadata.get("status", "")).lower()
            name_str = str(getattr(w, "name", "")).lower()
            return (
                raw_type == "injector"
                or "injector" in status_str
                or "injector" in name_str
                or name_str.startswith("inj")
            )

        injector_wells = [w for w in well_data_list if is_injector_well(w)]
        producer_wells = [w for w in well_data_list if not is_injector_well(w)]

        scheme = str(getattr(eor_params, "injection_scheme", "continuous")).lower()
        project_life_days = operational_params.project_lifetime_years * 365.25
        breakthrough_time = getattr(eor_params, "breakthrough_fallback_time_years", 5.0) * 365.25

        wells = []

        if not injector_wells and scheme != "storage":
            schedule = self._generate_injector_schedule(
                well_name="Field-Injector-1",
                scheme=scheme,
                eor_params=eor_params,
                project_life_days=project_life_days,
            )
            wells.append(schedule)
        else:
            for well in injector_wells:
                schedule = self._generate_injector_schedule(
                    well_name=well.name,
                    scheme=scheme,
                    eor_params=eor_params,
                    project_life_days=project_life_days,
                )
                wells.append(schedule)

        if not producer_wells and scheme != "storage":
            schedule = self._generate_producer_schedule(
                well_name="Field-Producer-1",
                scheme=scheme,
                eor_params=eor_params,
                project_life_days=project_life_days,
                breakthrough_time_days=breakthrough_time,
            )
            wells.append(schedule)
        else:
            for well in producer_wells:
                schedule = self._generate_producer_schedule(
                    well_name=well.name,
                    scheme=scheme,
                    eor_params=eor_params,
                    project_life_days=project_life_days,
                    breakthrough_time_days=breakthrough_time,
                )
                wells.append(schedule)

        return {
            "wells": wells,
            "injection_scheme": scheme,
            "project_lifetime_days": project_life_days,
            "breakthrough_time_days": breakthrough_time,
            "time_vector_monthly": time_vector.tolist()
            if isinstance(time_vector, np.ndarray)
            else time_vector,
        }

    def _generate_injector_schedule(
        self,
        well_name: str,
        scheme: str,
        eor_params: "EORParameters",
        project_life_days: float,
    ) -> Dict[str, Any]:
        """Generate injection well schedule based on scheme."""
        if scheme == "continuous":
            ops = self._ops_continuous(eor_params, project_life_days)
        elif scheme == "wag":
            ops = self._ops_wag(eor_params, project_life_days)
        elif scheme == "huff_n_puff":
            ops = self._ops_huff_n_puff(eor_params, project_life_days)
        elif scheme == "tapered":
            ops = self._ops_tapered(eor_params, project_life_days)
        elif scheme == "swag":
            ops = self._ops_swag(eor_params, project_life_days)
        elif scheme == "pulsed":
            ops = self._ops_pulsed(eor_params, project_life_days)
        else:
            ops = self._ops_continuous(eor_params, project_life_days)

        return {"well_name": well_name, "well_type": "injector", "operations": ops}

    def _generate_producer_schedule(
        self,
        well_name: str,
        scheme: str,
        eor_params: "EORParameters",
        project_life_days: float,
        breakthrough_time_days: float,
    ) -> Dict[str, Any]:
        """Generate production well schedule with pre/post breakthrough behavior."""
        if scheme == "huff_n_puff":
            ops = self._ops_huff_n_puff_production(
                eor_params, project_life_days, breakthrough_time_days
            )
        else:
            ops = self._ops_standard_production(
                eor_params, project_life_days, breakthrough_time_days
            )

        return {"well_name": well_name, "well_type": "producer", "operations": ops}

    def _ops_continuous(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """Single injection op for entire project life."""
        rate = getattr(eor_params, "injection_rate", 5000.0)
        return [
            {
                "start_day": 0,
                "duration_days": project_life_days,
                "operation": "injection",
                "rate_mscfd": rate,
                "phase": "injection",
            }
        ]

    def _ops_wag(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """Alternating gas/water cycles."""
        wag_ratio = getattr(eor_params, "wag_ratio", 1.0)
        cycle_days = getattr(eor_params, "cycle_length_days", 90.0)
        co2_rate = getattr(eor_params, "injection_rate", 5000.0)
        water_rate = co2_rate * wag_ratio

        operations = []
        day = 0
        is_gas_phase = True

        while day < project_life_days:
            phase_duration = cycle_days / 2.0
            if is_gas_phase:
                operations.append(
                    {
                        "start_day": day,
                        "duration_days": phase_duration,
                        "operation": "injection",
                        "rate_mscfd": co2_rate,
                        "phase": "gas_injection",
                    }
                )
            else:
                operations.append(
                    {
                        "start_day": day,
                        "duration_days": phase_duration,
                        "operation": "injection",
                        "rate_mscfd": water_rate,
                        "phase": "water_injection",
                    }
                )
            day += cycle_days
            is_gas_phase = not is_gas_phase

        return operations

    def _ops_huff_n_puff(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """HnP cycles: inject → soak → (implicit produce on same well)."""
        inj_days = getattr(eor_params, "huff_n_puff_injection_period_days", 30.0)
        soak_days = getattr(eor_params, "huff_n_puff_soaking_period_days", 15.0)
        prod_days = getattr(eor_params, "huff_n_puff_production_period_days", 45.0)
        max_cycles = getattr(eor_params, "huff_n_puff_max_cycles", 10)
        rate = getattr(eor_params, "injection_rate", 5000.0)

        cycle_length = inj_days + soak_days + prod_days
        operations = []
        day = 0

        for cycle in range(int(max_cycles)):
            if day >= project_life_days:
                break

            operations.append(
                {
                    "start_day": day,
                    "duration_days": inj_days,
                    "operation": "injection",
                    "rate_mscfd": rate,
                    "phase": "injection",
                }
            )
            day += inj_days

            operations.append(
                {
                    "start_day": day,
                    "duration_days": soak_days,
                    "operation": "soak",
                    "rate_mscfd": 0.0,
                    "phase": "soaking",
                }
            )
            day += soak_days

            day += prod_days

        return operations

    def _ops_huff_n_puff_production(
        self, eor_params: "EORParameters", project_life_days: float, breakthrough_time_days: float
    ) -> List[Dict]:
        """Production during HnP produce phases."""
        inj_days = getattr(eor_params, "huff_n_puff_injection_period_days", 30.0)
        soak_days = getattr(eor_params, "huff_n_puff_soaking_period_days", 15.0)
        prod_days = getattr(eor_params, "huff_n_puff_production_period_days", 45.0)
        max_cycles = getattr(eor_params, "huff_n_puff_max_cycles", 10)
        base_rate = getattr(eor_params, "injection_rate", 5000.0)

        cycle_length = inj_days + soak_days + prod_days
        operations = []
        day = 0

        for cycle in range(int(max_cycles)):
            if day >= project_life_days:
                break

            day += inj_days + soak_days

            if day >= project_life_days:
                break

            actual_prod_days = min(prod_days, project_life_days - day)
            rate = base_rate * 0.3

            operations.append(
                {
                    "start_day": day,
                    "duration_days": actual_prod_days,
                    "operation": "production",
                    "rate_mscfd": rate,
                    "phase": "production",
                }
            )
            day += prod_days

        return operations

    def _ops_standard_production(
        self, eor_params: "EORParameters", project_life_days: float, breakthrough_time_days: float
    ) -> List[Dict]:
        """Standard production: high pre-BT, declining post-BT across full project life."""
        base_rate = getattr(eor_params, "injection_rate", 5000.0) * 0.3
        prod_days = 365.25

        if breakthrough_time_days >= project_life_days:
            return [
                {
                    "start_day": 0,
                    "duration_days": project_life_days,
                    "operation": "production",
                    "rate_mscfd": base_rate,
                    "phase": "production",
                }
            ]

        operations = []
        day = 0

        while day < project_life_days:
            if day < breakthrough_time_days:
                op_duration = min(prod_days, breakthrough_time_days - day, project_life_days - day)
                operations.append(
                    {
                        "start_day": day,
                        "duration_days": op_duration,
                        "operation": "production",
                        "rate_mscfd": base_rate,
                        "phase": "production",
                    }
                )
                day += op_duration
            else:
                op_duration = min(prod_days, project_life_days - day)
                decline_factor = np.exp(-0.05 * (day - breakthrough_time_days) / 365.25)
                rate = base_rate * decline_factor
                operations.append(
                    {
                        "start_day": day,
                        "duration_days": op_duration,
                        "operation": "production",
                        "rate_mscfd": rate,
                        "phase": "production",
                    }
                )
                day += op_duration

        return operations

    def _ops_tapered(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """Decreasing injection rate over time."""
        initial_mult = getattr(eor_params, "tapered_initial_rate_multiplier", 2.0)
        final_mult = getattr(eor_params, "tapered_final_rate_multiplier", 0.1)
        duration_years = getattr(eor_params, "tapered_duration_years", 10.0)
        base_rate = getattr(eor_params, "injection_rate", 5000.0)

        operations = []
        year = 0

        while year * 365.25 < project_life_days:
            t_normalized = min(year / duration_years, 1.0)
            mult = initial_mult + (final_mult - initial_mult) * t_normalized
            rate = base_rate * mult
            operations.append(
                {
                    "start_day": year * 365.25,
                    "duration_days": 365.25,
                    "operation": "injection",
                    "rate_mscfd": rate,
                    "phase": "tapered",
                }
            )
            year += 1

        return operations

    def _ops_swag(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """Simultaneous or alternating WAG."""
        co2_rate = getattr(eor_params, "injection_rate", 5000.0)
        wgr = getattr(eor_params, "swag_water_gas_ratio", 1.0)
        simultaneous = getattr(eor_params, "swag_simultaneous_injection", True)

        if simultaneous:
            return [
                {
                    "start_day": 0,
                    "duration_days": project_life_days,
                    "operation": "injection",
                    "rate_mscfd": co2_rate,
                    "phase": "swag",
                    "water_rate_bpd": co2_rate * wgr,
                }
            ]
        else:
            return self._ops_wag(eor_params, project_life_days)

    def _ops_pulsed(self, eor_params: "EORParameters", project_life_days: float) -> List[Dict]:
        """Intermittent high-intensity pulses."""
        pulse_days = getattr(eor_params, "pulsed_pulse_duration_days", 15.0)
        pause_days = getattr(eor_params, "pulsed_pause_duration_days", 15.0)
        intensity = getattr(eor_params, "pulsed_intensity_multiplier", 2.0)
        base_rate = getattr(eor_params, "injection_rate", 5000.0)

        cycle_length = pulse_days + pause_days
        operations = []
        day = 0

        while day < project_life_days:
            operations.append(
                {
                    "start_day": day,
                    "duration_days": pulse_days,
                    "operation": "injection",
                    "rate_mscfd": base_rate * intensity,
                    "phase": "pulse",
                }
            )
            day += pulse_days

            if day < project_life_days:
                actual_pause = min(pause_days, project_life_days - day)
                operations.append(
                    {
                        "start_day": day,
                        "duration_days": actual_pause,
                        "operation": "idle",
                        "rate_mscfd": 0.0,
                        "phase": "pause",
                    }
                )
                day += pause_days

        return operations

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
                b_entry = bounds[param_name]
                if isinstance(b_entry, dict):
                    low, high = b_entry["low"], b_entry["high"]
                else:
                    low, high = b_entry[0], b_entry[1]
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

        selected_indices: List[int] = []
        # Start with the best solution
        best_idx = int(np.argmax(fitnesses))
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
                best_remaining = int(remaining_indices[np.argmax(fitnesses[remaining_indices])])
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
            # Retrieve bounds for each gene safely handling dicts, tuples, or lists
            lows = []
            highs = []
            for g in ga_instance.gene_space:
                if isinstance(g, dict):
                    lows.append(g.get("low", 0.0))
                    highs.append(g.get("high", 1.0))
                elif isinstance(g, (list, tuple)):
                    lows.append(min(g))
                    highs.append(max(g))
                else:
                    lows.append(0.0)
                    highs.append(1.0)

            new_population = np.random.uniform(
                low=lows, high=highs, size=(num_restart, ga_instance.num_genes)
            )

            # Update population
            for i, idx in enumerate(worst_indices):
                ga_instance.population[idx] = new_population[i]

            logger.info(f"Restarted {num_restart} individuals to escape local optimum.")

    ### --- OPTIMIZATION METHODS --- ###
    def run_single_simulation(
        self, custom_params: Optional[Dict[str, float]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Runs a direct, single forward simulation using baseline or configured operational parameters without optimization."""
        self.reset_to_base_state()
        start_time = time.time()
        cb = kwargs.get("text_progress_callback")
        if cb:
            cb("Running forward reservoir simulation...")

        # Build baseline operational parameters from self.eor_params
        rate = float(getattr(self.eor_params, "injection_rate", 5000.0))
        pressure = float(
            getattr(
                self.eor_params,
                "target_pressure_psi",
                getattr(self.eor_params, "max_pressure_psi", 2500.0),
            )
        )
        plateau = float(getattr(self.eor_params, "plateau_duration_fraction", 0.3))
        ramp_up = float(getattr(self.eor_params, "ramp_up_fraction", 0.1))
        bhp = float(getattr(self.eor_params, "min_producer_bhp_psi", 1500.0))
        max_prod = float(getattr(self.eor_params, "max_production_rate_stbd", 5000.0))
        wag_ratio = float(
            getattr(
                self.eor_params,
                "wag_ratio",
                getattr(getattr(self.eor_params, "swag", None), "water_gas_ratio", 1.0),
            )
        )

        sim_params: Dict[str, Any] = {
            "rate": rate,
            "pressure": pressure,
            "plateau_duration_fraction": plateau,
            "ramp_up_fraction": ramp_up,
            "wellbore_pressure": bhp,
            "max_production_rate_stbd": max_prod,
        }
        if str(getattr(self.eor_params, "injection_scheme", "continuous")).lower() in [
            "wag",
            "swag",
        ]:
            sim_params["wag_ratio"] = wag_ratio

        if custom_params:
            sim_params.update(custom_params)

        # Sanitize and discretize according to engine rules
        sim_params = self._sanitize_and_discretize_parameters(sim_params)

        final_eval = self.evaluate_for_analysis(
            sim_params,
            economic_params_override=self.economic_params,
            ooip_override=self.reservoir.ooip_stb,
            mmp_override=self.mmp or self.eor_params.default_mmp_fallback,
            co2_storage_params_override=self.co2_storage_params,
        )

        final_profiles = final_eval.get("profiles", final_eval)

        # Ensure time vectors and well schedules are attached to profiles
        if final_profiles is not None:
            project_life_years = int(
                getattr(self.operational_params, "project_lifetime_years", 15)
            )
            time_res = getattr(self.operational_params, "time_resolution", "yearly")
            years = np.arange(1, project_life_years + 1)
            final_profiles[f"{time_res}_time_years"] = years
            final_profiles["yearly_time_years"] = years
            final_profiles["annual_time_years"] = years
            if "monthly_oil_stb" in final_profiles:
                final_profiles["monthly_time_years"] = np.linspace(
                    1 / 12.0, project_life_years, len(final_profiles["monthly_oil_stb"])
                )
            daily_time_points = int(project_life_years * DAYS_PER_YEAR)
            final_profiles["time_vector"] = np.linspace(
                0, project_life_years * DAYS_PER_YEAR, daily_time_points
            )

            monthly_time_vector = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 12) + 1
            )
            schedule_data = self._generate_well_schedule_from_params(
                self.eor_params, self.operational_params, monthly_time_vector
            )
            final_profiles["well_schedule"] = schedule_data

        eval_time = time.time() - start_time
        chosen_obj = getattr(self, "chosen_objective", "npv")
        obj_val = final_eval.get(chosen_obj, final_eval.get("npv", 0.0))

        gross_util = final_eval.get(
            "gross_utilization_mscf_per_stb",
            final_profiles.get("gross_utilization_mscf_per_stb") if final_profiles else None,
        )
        net_util = final_eval.get(
            "net_utilization_mscf_per_stb",
            final_profiles.get("net_utilization_mscf_per_stb") if final_profiles else None,
        )

        self._results = {
            "optimized_params_final_clipped": sim_params,
            "objective_function_value": obj_val,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval,
            "recovery_factor": final_eval.get("recovery_factor"),
            "npv": final_eval.get("npv"),
            "gross_utilization_mscf_per_stb": gross_util,
            "net_utilization_mscf_per_stb": net_util,
            "method": "single_simulation",
            "simulation_statistics": {
                "evaluation_time_seconds": eval_time,
                "status": "success",
            },
        }

        if cb:
            cb("Simulation completed successfully.")

        return self._results

    def optimize_genetic_algorithm(self, ga_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        ga_params = deepcopy(ga_params_override or self.ga_params_default_config)
        bounds = self._get_parameter_bounds()
        param_names = list(bounds.keys())

        # Handle mixed gene_space: tuples for numeric params, dicts/lists for discrete params
        gene_space: List[Any] = []
        for key in param_names:
            val = bounds[key]
            if key == "shut_in_mode":
                gene_space.append([0, 1, 2])
            elif key == "allow_well_conversion":
                gene_space.append([0, 1])
            elif key == "injection_scheme":
                gene_space.append([0, 1, 2, 3, 4])
            elif isinstance(val, dict) and "low" in val and "high" in val:
                # Discrete parameter using numeric index range
                if val.get("step") == 1:
                    gene_space.append(list(range(int(val["low"]), int(val["high"]) + 1)))
                else:
                    gene_space.append({"low": val["low"], "high": val["high"]})
            else:
                # Numeric parameter with (low, high) tuple
                gene_space.append({"low": val[0], "high": val[1]})

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
        safe_self = PickleSafeOptimiser(self)

        # CRITICAL: Disable pygad's internal logging to prevent file lock conflicts
        # on Windows when using multiprocessing. pygad may set up its own file handlers
        # that can cause PermissionError during log rotation with multiple processes.
        pygad_logger = logging.getLogger("pygad")
        pygad_logger.handlers.clear()
        pygad_logger.propagate = True  # Let logs propagate to root queue handler
        pygad_logger.setLevel(logging.WARNING)  # Only show warnings and errors

        # Also disable any RotatingFileHandler that might be set up by pygad
        for handler in pygad_logger.handlers[:]:
            try:
                handler.close()
            except (OSError, IOError, ValueError) as e:
                logger.debug("Failed to close pygad handler: %s", e)
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
        final_params = self._map_scheme_index_to_name(final_params)
        final_params = self._sanitize_and_discretize_parameters(final_params)

        final_eval = self.evaluate_for_analysis(
            final_params,
            economic_params_override=self.economic_params,
            ooip_override=self.reservoir.ooip_stb,
            mmp_override=self.mmp or self.eor_params.default_mmp_fallback,
            co2_storage_params_override=self.co2_storage_params,
        )

        # Check if profiles were already generated during evaluation (e.g. by Surrogate Engine)
        final_profiles = final_eval.get("profiles")

        if final_profiles is None:
            # Only use ProductionProfiler if we don't have profiles from the engine
            logger.info(
                "No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based)."
            )
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
                    b_entry = bounds[key]
                    if isinstance(b_entry, dict):
                        min_val, max_val = b_entry["low"], b_entry["high"]
                    else:
                        min_val, max_val = b_entry[0], b_entry[1]
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

            if ProductionProfiler is None:
                raise OptimizationError("ProductionProfiler is not available")
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

        # Add time vectors to profiles for proper plotting
        if final_profiles is not None:
            project_life_years = getattr(self.operational_params, "project_lifetime_years", 30)
            time_res = getattr(self.operational_params, "time_resolution", "yearly")
            years = np.arange(1, project_life_years + 1)
            final_profiles[f"{time_res}_time_years"] = years
            final_profiles["yearly_time_years"] = years
            final_profiles["annual_time_years"] = years
            if "monthly_oil_stb" in final_profiles:
                final_profiles["monthly_time_years"] = np.linspace(
                    1 / 12.0, project_life_years, len(final_profiles["monthly_oil_stb"])
                )
            # Also add daily time vector for detailed plotting
            # Use 365 to match profiler.py's DAYS_PER_YEAR for consistent array lengths
            daily_time_points = int(project_life_years * DAYS_PER_YEAR)
            final_profiles["time_vector"] = np.linspace(
                0, project_life_years * DAYS_PER_YEAR, daily_time_points
            )

            # Generate per-well schedule data for visualization
            # Use monthly time vector internally for proper cycle resolution
            monthly_time_vector = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 12) + 1
            )
            # Use self.eor_params since current_eor_params may not be defined
            # when using simulation engine profiles directly
            schedule_data = self._generate_well_schedule_from_params(
                self.eor_params, self.operational_params, monthly_time_vector
            )
            final_profiles["well_schedule"] = schedule_data

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
            "recovery_factor": final_eval.get("recovery_factor") if isinstance(final_eval, dict) else None,
            "npv": final_eval.get("npv") if isinstance(final_eval, dict) else None,
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
                "max_fitness_history": [float(np.max(np.array(f)[np.array(f) > -1e9])) if np.any(np.array(f) > -1e9) else float(np.max(f)) for f in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness
                else [],
                "min_fitness_history": [float(np.min(np.array(f)[np.array(f) > -1e9])) if np.any(np.array(f) > -1e9) else float(np.min(f)) for f in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness
                else [],
                "avg_fitness_history": [float(np.mean(np.array(f)[np.array(f) > -1e9])) if np.any(np.array(f) > -1e9) else float(np.mean(f)) for f in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness
                else [],
                "std_fitness_history": [float(np.std(np.array(f)[np.array(f) > -1e9])) if np.any(np.array(f) > -1e9) else float(np.std(f)) for f in ga_instance.all_fitness]
                if hasattr(ga_instance, "all_fitness") and ga_instance.all_fitness
                else [],
                "coverage_history": ga_instance.coverage_history
                if hasattr(ga_instance, "coverage_history")
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
            "ga_coverage_distribution": self.plot_ga_coverage_distribution(self._results),
            "euclidean_distance_matrix": self.plotting_manager.plot_euclidean_distance_matrix(
                self._results
            ),
            "ga_objective_distribution": self.plot_ga_objective_distribution(),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts

        return self._results

    def hybrid_optimize(self, **kwargs) -> Dict[str, Any]:
        ga_params = deepcopy(kwargs.get("ga_params_override") or self.ga_params_default_config)
        bo_params = deepcopy(kwargs.get("bo_params_override") or self.bo_params_default_config)
        logger.info(
            f"Hybrid Opt: Starting GA Phase (Gens:{ga_params.num_generations}, Pop:{ga_params.sol_per_pop})"
        )
        if cb := kwargs.get("text_progress_callback"):
            cb(f"Running GA Phase ({ga_params.num_generations} generations)...")

        kwargs["ga_params_override"] = ga_params
        ga_res = self.optimize_genetic_algorithm(**kwargs)

        if cb:
            cb("GA Phase Complete. Preparing for Bayesian Optimization...")

        ga_instance = ga_res.get("pygad_instance")
        init_bo_sols = []
        if ga_instance:
            param_names = list(self._get_parameter_bounds().keys())
            num_to_select = ga_params.num_diverse_solutions_for_bo
            final_pop, final_fit = ga_instance.population, ga_instance.last_generation_fitness
            diverse_solutions, _ = self._select_diverse_solutions(
                final_pop,
                final_fit,
                param_names,
                num_to_select,
                ga_params.diversity_threshold_for_bo,
            )
            for sol in diverse_solutions:
                init_bo_sols.append({"params": {name: val for name, val in zip(param_names, sol)}})

        bo_kwargs = kwargs.copy()
        bo_kwargs["initial_solutions_from_ga"] = init_bo_sols
        bo_params.n_initial_points = 0
        bo_kwargs["bo_params_override"] = bo_params
        if cb:
            cb(f"Running BO Phase ({bo_params.n_iterations} iterations)...")

        bo_res = self.optimize_bayesian(**bo_kwargs)

        final_metrics = bo_res.get("final_metrics", {})
        self._results = {
            **bo_res,
            "ga_full_results_for_hybrid": ga_res,
            "diverse_points_for_bo": init_bo_sols,
            "method": "hybrid_ga_bo",
            "recovery_factor": final_metrics.get("recovery_factor"),
            "npv": final_metrics.get("npv"),
        }

        # Generate and store hybrid charts in the results
        charts = {
            "optimization_convergence": self.plotting_manager.plot_optimization_convergence(
                self._results
            ),
            "production_profiles": self.plotting_manager.plot_production_profiles(self._results),
            "well_schedule": self.plotting_manager.plot_well_schedule(self._results),
            "co2_performance_summary": self.plotting_manager.plot_co2_performance_summary_table(
                self._results
            ),
            "ga_coverage_distribution": self.plotting_manager.plot_coverage(self._results),
            "euclidean_distance_matrix": self.plotting_manager.plot_euclidean_distance_matrix(
                self._results
            ),
            "hybrid_model_analysis": self.plot_hybrid_model_analysis(),
            "breakthrough_mechanism_analysis": self.plot_breakthrough_mechanism_analysis(),
        }
        self._results["charts"] = charts
        return self._results

    def optimize_nsga_2(self, nsga2_params_override, **kwargs) -> Dict[str, Any]:
        """
        Multi-objective NSGA-II optimization using pygad.
        Returns Pareto front of non-dominated solutions.
        """
        self.reset_to_base_state()
        ga_params = deepcopy(nsga2_params_override or self.ga_params_default_config)
        ga_params.num_objectives = 2  # Force bi-objective for NSGA-II

        self.ga_params_current_run = ga_params
        bounds_dict = self._get_parameter_bounds()
        param_names = list(bounds_dict.keys())

        if kwargs.get("text_progress_callback"):
            kwargs["text_progress_callback"]("Running NSGA-II optimization...")

        logger.info(
            f"NSGA-II started: {ga_params.num_generations} generations, pop size {ga_params.sol_per_pop}"
        )

        gene_space: List[Any] = []
        for param_name in param_names:
            if param_name == "shut_in_mode":
                gene_space.append([0, 1, 2])
            elif param_name == "allow_well_conversion":
                gene_space.append([0, 1])
            elif param_name == "injection_scheme":
                gene_space.append([0, 1, 2, 3, 4])
            else:
                val = bounds_dict[param_name]
                if isinstance(val, dict) and "low" in val and "high" in val:
                    gene_space.append({"low": val["low"], "high": val["high"]})
                else:
                    low, high = val[0], val[1]
                    gene_space.append({"low": low, "high": high})

        ga_start_time = time.time()

        def fitness_func(ga_instance, solutions, solution_idx):
            return self._fitness_func_pygad(ga_instance, solutions, solution_idx)

        safe_self = PickleSafeOptimiser(self)

        ga_instance = pygad.GA(
            num_generations=ga_params.num_generations,
            sol_per_pop=ga_params.sol_per_pop,
            num_parents_mating=ga_params.num_parents_mating,
            num_genes=len(param_names),
            fitness_func=fitness_func,
            gene_space=gene_space,
            parent_selection_type="tournament_nsga2",
            crossover_type="sbx",
            crossover_probability=ga_params.crossover_probability,
            mutation_type="polynomial",
            mutation_probability=ga_params.mutation_probability,
            keep_elitism=ga_params.keep_elitism,
            num_objectives=2,
            parallel_processing=kwargs.get("parallel_processing", False),
        )

        ga_instance.run()

        ga_end_time = time.time()
        ga_duration = ga_end_time - ga_start_time
        total_evaluations = ga_params.num_generations * ga_params.sol_per_pop

        logger.info(
            f"NSGA-II completed in {ga_duration:.2f} seconds ({total_evaluations} evaluations)"
        )

        all_solutions = ga_instance.population
        all_fitnesses = ga_instance.last_generation_fitness

        pareto_front = self._extract_pareto_front(all_solutions, all_fitnesses, param_names)

        solution, fitness, _ = ga_instance.best_solution()
        final_params = {name: val for name, val in zip(param_names, solution)}
        final_params = self._map_scheme_index_to_name(final_params)

        final_eval = self.evaluate_for_analysis(
            final_params,
            economic_params_override=self.economic_params,
            ooip_override=self.reservoir.ooip_stb,
            mmp_override=self.mmp or self.eor_params.default_mmp_fallback,
            co2_storage_params_override=self.co2_storage_params,
        )
        final_profiles = final_eval.get("profiles")

        temp_eor = self.eor_params
        if final_profiles is None:
            logger.info("No profiles from engine. Using ProductionProfiler fallback.")
            temp_eor = deepcopy(self.eor_params)
            temp_profile = deepcopy(self.profile_params)
            temp_co2 = deepcopy(self.co2_storage_params)

            for key, value in final_params.items():
                if hasattr(temp_eor, key):
                    setattr(temp_eor, key, value)
                elif hasattr(temp_profile, key):
                    setattr(temp_profile, key, value)
                elif hasattr(temp_co2, key):
                    setattr(temp_co2, key, value)

            if ProductionProfiler is None:
                raise OptimizationError("ProductionProfiler is not available")
            profiler = ProductionProfiler(
                self.reservoir, self.pvt, temp_eor, self.operational_params, temp_profile
            )
            final_profiles = profiler.generate_all_profiles(ooip_stb=self.reservoir.ooip_stb)

        # Add time vectors to profiles for proper plotting
        if final_profiles is not None:
            project_life_years = getattr(self.operational_params, "project_lifetime_years", 30)
            time_res = getattr(self.operational_params, "time_resolution", "yearly")
            years = np.arange(1, project_life_years + 1)
            final_profiles[f"{time_res}_time_years"] = years
            final_profiles["yearly_time_years"] = years
            final_profiles["annual_time_years"] = years
            if "monthly_oil_stb" in final_profiles:
                final_profiles["monthly_time_years"] = np.linspace(
                    1 / 12.0, project_life_years, len(final_profiles["monthly_oil_stb"])
                )
            final_profiles["time_vector"] = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 365.25)
            )

            # Generate per-well schedule data for visualization
            monthly_time_vector = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 12) + 1
            )
            schedule_data = self._generate_well_schedule_from_params(
                temp_eor, self.operational_params, monthly_time_vector
            )
            final_profiles["well_schedule"] = schedule_data

        self._results = {
            "optimized_params_final_clipped": final_params,
            "objective_function_value": fitness[0]
            if isinstance(fitness, (list, np.ndarray))
            else fitness,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval,
            "method": "nsga_2",
            "pareto_front": pareto_front,
            "pygad_instance": ga_instance,
            "ga_statistics": {
                "total_duration_seconds": ga_duration,
                "total_evaluations": total_evaluations,
                "num_generations": ga_params.num_generations,
                "population_size": ga_params.sol_per_pop,
                "pareto_front_size": len(pareto_front),
            },
        }

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

    def _extract_pareto_front(self, solutions, fitnesses, param_names) -> list:
        """
        Extract non-dominated solutions (Pareto front) from population.
        For minimization problems (NPV is negative in fitness).
        """
        if len(solutions) == 0:
            return []

        pareto_front = []
        for i, (sol, fit) in enumerate(zip(solutions, fitnesses)):
            is_dominated = False
            for j, (other_sol, other_fit) in enumerate(zip(solutions, fitnesses)):
                if i == j:
                    continue
                if self._dominates(other_fit, fit):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(
                    {
                        "params": {name: val for name, val in zip(param_names, sol)},
                        "objectives": list(fit) if isinstance(fit, (list, np.ndarray)) else [fit],
                        "solution_index": i,
                    }
                )

        return pareto_front

    def _dominates(self, obj1, obj2) -> bool:
        """
        Check if obj1 dominates obj2 (for minimization).
        obj1 dominates obj2 if obj1 is better or equal in all objectives and strictly better in at least one.
        """
        obj1 = np.asarray(obj1)
        obj2 = np.asarray(obj2)

        if obj1.ndim == 0:
            obj1 = [obj1.item()]
            obj2 = [obj2.item()]

        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_any = True

        return better_in_any

    def hybrid_nsga2_bo(self, **kwargs) -> Dict[str, Any]:
        """
        Two-phase: NSGA-II for exploration (Pareto front)
        followed by BO for refinement using _select_diverse_solutions().
        """
        ga_params = deepcopy(kwargs.get("ga_params_override") or self.ga_params_default_config)
        bo_params = deepcopy(kwargs.get("bo_params_override") or self.bo_params_default_config)
        logger.info(
            f"Hybrid NSGA-II+BO: Starting NSGA-II Phase (Gens:{ga_params.num_generations}, Pop:{ga_params.sol_per_pop})"
        )

        if cb := kwargs.get("text_progress_callback"):
            cb(f"Running NSGA-II Phase ({ga_params.num_generations} generations)...")

        kwargs["ga_params_override"] = ga_params
        nsga_res = self.optimize_nsga_2(**kwargs)
        pareto_front = nsga_res.get("pareto_front", [])

        if cb:
            cb(
                f"NSGA-II Phase Complete. Found {len(pareto_front)} Pareto solutions. Preparing for BO..."
            )

        init_bo_sols = []
        if pareto_front and len(pareto_front) >= ga_params.num_diverse_solutions_for_bo:
            param_names = list(self._get_parameter_bounds().keys())
            pareto_sols = np.array([list(sol["params"].values()) for sol in pareto_front])
            combined_fitness = np.array([np.mean(sol["objectives"]) for sol in pareto_front])

            diverse_solutions, _ = self._select_diverse_solutions(
                pareto_sols,
                combined_fitness,
                param_names,
                ga_params.num_diverse_solutions_for_bo,
                ga_params.diversity_threshold_for_bo,
            )

            for sol in diverse_solutions:
                init_bo_sols.append({"params": {name: val for name, val in zip(param_names, sol)}})

            logger.info(f"Selected {len(init_bo_sols)} diverse solutions from Pareto front for BO")
        else:
            if pareto_front:
                logger.warning(
                    f"Pareto front size ({len(pareto_front)}) < num_diverse_solutions_for_bo ({ga_params.num_diverse_solutions_for_bo}), using all Pareto solutions"
                )
                init_bo_sols = [{"params": sol["params"]} for sol in pareto_front]

        bo_kwargs = kwargs.copy()
        bo_kwargs["initial_solutions_from_ga"] = init_bo_sols
        bo_params.n_initial_points = 0
        bo_kwargs["bo_params_override"] = bo_params

        if cb:
            cb(f"Running BO Phase ({bo_params.n_iterations} iterations)...")

        bo_res = self.optimize_bayesian(**bo_kwargs)

        self._results = {
            **bo_res,
            "nsga2_full_results_for_hybrid": nsga_res,
            "pareto_front": nsga_res.get("pareto_front", []),
            "method": "hybrid_nsga2_bo",
        }
        return self._results

    def optimize_bayesian(self, bo_params_override, **kwargs) -> Dict[str, Any]:
        self.reset_to_base_state()
        bo_params = deepcopy(bo_params_override or self.bo_params_default_config)
        raw_pb = kwargs.get("pbounds_override", self._get_parameter_bounds())
        # Clean pbounds for BayesianOptimization: must be (low, high) tuples of floats
        pb_bayes = {}
        for k, v in raw_pb.items():
            if isinstance(v, dict):
                pb_bayes[k] = (float(v["low"]), float(v["high"]))
            else:
                pb_bayes[k] = (float(v[0]), float(v[1]))

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

        def _build_acquisition_function(acq_name: str, kappa_val: float, xi_val: float):
            try:
                from bayes_opt import acquisition
                acq_key = str(acq_name).lower()
                if acq_key in ("ucb", "upper_confidence_bound"):
                    return acquisition.UpperConfidenceBound(kappa=float(kappa_val))
                elif acq_key in ("ei", "expected_improvement"):
                    return acquisition.ExpectedImprovement(xi=float(xi_val))
                elif acq_key in ("poi", "pi", "probability_of_improvement"):
                    return acquisition.ProbabilityOfImprovement(xi=float(xi_val))
                else:
                    return acquisition.UpperConfidenceBound(kappa=float(kappa_val))
            except (ImportError, AttributeError):
                return None

        def maximize_with_logging(init_points, n_iter, acq="ucb", kappa=2.576, xi=0.01, **kwargs):
            # Configure acquisition function on BayesianOptimization instance
            acq_obj = _build_acquisition_function(acq, kappa, xi)
            if acq_obj is not None and hasattr(bayes_o, "_acquisition_function"):
                bayes_o._acquisition_function = acq_obj

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
                for param, b_val in pb_bayes.items():
                    if isinstance(b_val, dict):
                        low, high = float(b_val["low"]), float(b_val["high"])
                    else:
                        low, high = float(b_val[0]), float(b_val[1])
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

            max_kwargs = dict(kwargs)
            try:
                import inspect
                sig = inspect.signature(original_maximize)
                if "acq" in sig.parameters:
                    max_kwargs["acq"] = acq
                if "kappa" in sig.parameters:
                    max_kwargs["kappa"] = kappa
                if "xi" in sig.parameters:
                    max_kwargs["xi"] = xi
            except Exception:
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

                # Pass step iterations and acquisition kwargs to original_maximize
                original_maximize(init_points=0 if i > 1 else init_points, n_iter=1, **max_kwargs)
                bo_progress_callback(i, bayes_o)

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

            maximize_with_logging(
                init_points=bo_params.n_initial_points,
                n_iter=bo_params.n_iterations,
                acq=bo_params.acquisition_function,
                kappa=acq_kappa,
                xi=bo_params.acq_xi,
            )
        except Exception as e:
            logger.error(f"BO optimization failed: {e}")
            raise

        if bayes_o.max is None:
            raise OptimizationError("Bayesian Optimization produced no results (bayes_o.max is None)")
        best_params, best_obj = bayes_o.max["params"], bayes_o.max["target"]
        best_params = self._map_scheme_index_to_name(best_params)
        best_params = self._sanitize_and_discretize_parameters(best_params)

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
        final_eval = self.evaluate_for_analysis(
            best_params,
            economic_params_override=self.economic_params,
            ooip_override=self.reservoir.ooip_stb,
            mmp_override=self.mmp or self.eor_params.default_mmp_fallback,
            co2_storage_params_override=self.co2_storage_params,
        )

        # Check if profiles were already generated during evaluation (e.g. from Simple or Surrogate Engine)
        final_profiles = final_eval.get("profiles")

        if final_profiles is None:
            # Fallback: Regenerate profiles using ProductionProfiler (Detailed engine fallback)
            # This path is taken only if the evaluation engine didn't return profiles
            logger.info(
                "No profiles found in evaluation results. Using ProductionProfiler fallback (Physics-based)."
            )
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

            if ProductionProfiler is None:
                raise OptimizationError("ProductionProfiler is not available")
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

        # Add time vectors to profiles for proper plotting
        if final_profiles is not None:
            project_life_years = getattr(self.operational_params, "project_lifetime_years", 30)
            time_res = getattr(self.operational_params, "time_resolution", "yearly")
            years = np.arange(1, project_life_years + 1)
            final_profiles[f"{time_res}_time_years"] = years
            final_profiles["yearly_time_years"] = years
            final_profiles["annual_time_years"] = years
            if "monthly_oil_stb" in final_profiles:
                final_profiles["monthly_time_years"] = np.linspace(
                    1 / 12.0, project_life_years, len(final_profiles["monthly_oil_stb"])
                )
            final_profiles["time_vector"] = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 365.25)
            )

            # Generate per-well schedule data for visualization
            # Use monthly time vector internally for proper cycle resolution
            monthly_time_vector = np.linspace(
                0, project_life_years * 365.25, int(project_life_years * 12) + 1
            )
            schedule_eor = (
                temp_eor_params_for_profiling
                if "temp_eor_params_for_profiling" in locals() and temp_eor_params_for_profiling is not None
                else self.eor_params
            )
            schedule_data = self._generate_well_schedule_from_params(
                schedule_eor, self.operational_params, monthly_time_vector
            )
            final_profiles["well_schedule"] = schedule_data

        # Store BO timing and statistics in results
        self._results = {
            "optimized_params_final_clipped": best_params,
            "objective_function_value": best_obj,
            "optimized_profiles": final_profiles,
            "final_metrics": final_eval,
            "recovery_factor": final_eval.get("recovery_factor") if isinstance(final_eval, dict) else None,
            "npv": final_eval.get("npv") if isinstance(final_eval, dict) else None,
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
            wag_ratio = params.get("wag_ratio", 1.0)
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
        wag_ratio = params.get("wag_ratio", default_wag_ratio)
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
            params = self._results.get("optimized_params_final_clipped") or {}

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
                economic_params_instance=self._base_economic_params,
                operational_params_instance=self._base_operational_params,
                profile_params_instance=self.profile_params,
                advanced_engine_params_instance=self.advanced_engine_params,
                co2_storage_params_instance=self._base_co2_storage_params,
                well_data_list=[well_data],
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
        return self.plotting_manager.plot_coverage(results_to_use or self._results)

    def plot_euclidean_distance_matrix(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        return self.plotting_manager.plot_euclidean_distance_matrix(results_to_use or self._results)

    def plot_ga_objective_distribution(
        self, results_to_use: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        return self.plotting_manager.plot_ga_objective_distribution(results_to_use or self._results)

    def plot_hybrid_model_analysis(self) -> go.Figure:
        """Generates a plot showing the interplay of miscible, immiscible, and hybrid recovery models."""
        return self.plotting_manager.plot_hybrid_model_analysis()

    def plot_breakthrough_mechanism_analysis(self) -> go.Figure:
        """Generates a bar chart comparing breakthrough times from different models using Surrogate Physics."""
        return self.plotting_manager.plot_breakthrough_mechanism_analysis()
