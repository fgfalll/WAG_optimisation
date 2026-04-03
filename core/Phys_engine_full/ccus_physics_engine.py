"""
Coupled CO₂-CCUS Physics Engine for Enhanced Oil Recovery and Long-term Storage
Implements full multiphase flow, EOS, geomechanics, fault mechanics, and mineralization physics
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import dataclasses
from datetime import datetime, timedelta

# --- Absolute imports for robustness ---
from core.data_models import (
    ReservoirData,
    EORParameters,
    PVTProperties,
    CCUSParameters,
    TimestepUnit,
    EOSModelParameters,
    CCUSState,
)
from core.unified_engine.physics.eos import CubicEOS, ReservoirFluid
from core.unified_engine.physics.relative_permeability import (
    CoreyRelativePermeability,
    CoreyParameters,
)
from core.Phys_engine_full.multiphase_flow import MultiphaseFlowSolver
from core.Phys_engine_full.pressure_dynamics import PressureDynamics
from core.Phys_engine_full.fault_mechanics import FaultMechanicsEngine
from core.Phys_engine_full.fully_implicit_solver import FullyImplicitSolver
from core.Phys_engine_full.time_stepping import (
    SimulationSchedule,
    TimestepConfig,
    CoupledPhysicsTimeStepper,
)
from core.data_models import PhysicalConstants

_PHYS_CONSTANTS = PhysicalConstants()

logger = logging.getLogger(__name__)


@dataclass
class GridParameters:
    """Grid and discretization parameters"""

    dimensions: Tuple[int, int, int]
    cell_volumes: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dz: np.ndarray
    depth: np.ndarray
    tops: np.ndarray
    fault_cells: List[List[int]]
    fault_orientations: List[str]
    n_faces: int = 0
    face_centers: np.ndarray = None
    face_normals: np.ndarray = None
    face_areas: np.ndarray = None
    face_cell_pos: np.ndarray = None
    face_cell_neg: np.ndarray = None
    porosity: np.ndarray = None
    permeability: np.ndarray = None

    @property
    def n_cells(self) -> int:
        return len(self.cell_volumes)


class GeomechanicsSolver:
    """Solves poroelastic geomechanics equations"""

    def __init__(self, grid: GridParameters, params: CCUSParameters):
        self.grid = grid
        self.params = params

    def update_poroelastic_properties(self, state: CCUSState) -> CCUSState:
        # Calculate stress change due to pressure change
        delta_pressure = state.pressure - self.initial_pressure
        alpha = self.params.biot_coefficient
        nu = self.params.poissons_ratio

        delta_sigma_h = (alpha * (1 - 2 * nu) / (1 - nu)) * delta_pressure

        new_stress = self.initial_stress.copy()
        new_stress[:, 0] += delta_sigma_h  # sigma_xx
        new_stress[:, 1] += delta_sigma_h  # sigma_yy

        mean_stress = np.mean(new_stress[:, :3], axis=1)
        volumetric_strain = (mean_stress - self.initial_mean_stress) / (
            self.params.youngs_modulus / (3 * (1 - 2 * self.params.poissons_ratio))
        )

        new_porosity = self.initial_porosity + self.params.biot_coefficient * volumetric_strain

        if np.any(delta_pressure > 1.0):
            # Format time based on timestep unit for better readability
            time_days = state.current_time
            if self.params.timestep_unit == TimestepUnit.DAILY:
                time_str = f"{time_days:.1f} days"
            elif self.params.timestep_unit == TimestepUnit.WEEKLY:
                weeks = time_days / 7.0
                time_str = f"{weeks:.1f} weeks ({time_days:.1f} days)"
            elif self.params.timestep_unit == TimestepUnit.MONTHLY:
                months = time_days / 30.44  # Average days per month
                time_str = f"{months:.2f} months ({time_days:.1f} days)"
            elif self.params.timestep_unit == TimestepUnit.YEARLY:
                years = time_days / 365.25
                time_str = f"{years:.2f} years ({time_days:.1f} days)"
            else:
                time_str = f"{time_days:.1f} days"

            logger.info(f"Geomech update at {time_str}:")
            logger.info(f"  Max delta_pressure: {np.max(delta_pressure):.2f} psi")
            logger.info(f"  Max volumetric_strain: {np.max(volumetric_strain):.2e}")
            logger.info(
                f"  Porosity change (min, max): {np.min(new_porosity - self.initial_porosity):.2e}, {np.max(new_porosity - self.initial_porosity):.2e}"
            )

        effective_stress = mean_stress - self.params.biot_coefficient * state.pressure
        initial_effective_stress = (
            self.initial_mean_stress - self.params.biot_coefficient * self.initial_pressure
        )

        new_permeability = (
            self.initial_permeability
            * (new_porosity / self.initial_porosity) ** 3
            * ((1 - self.initial_porosity) / (1 - new_porosity)) ** 2
        )

        new_porosity = np.clip(new_porosity, 0.01, 0.4)
        new_permeability = np.maximum(new_permeability, 1e-6)

        return CCUSState(
            pressure=state.pressure,
            saturations=state.saturations,
            compositions=state.compositions,
            porosity=new_porosity,
            permeability=new_permeability,
            stress=new_stress,
            fault_transmissibility=state.fault_transmissibility,
            dissolved_co2=state.dissolved_co2,
            mineral_precipitate=state.mineral_precipitate,
            current_time=state.current_time,
            timestep=state.timestep,
            fault_stability=state.fault_stability,
            injection_rates=state.injection_rates,
        )

    def set_initial_state(
        self,
        initial_porosity: np.ndarray,
        initial_permeability: np.ndarray,
        initial_pressure: np.ndarray,
        initial_stress: np.ndarray,
        grid: GridParameters,
    ):
        self.initial_porosity = initial_porosity
        self.initial_permeability = initial_permeability
        self.initial_pressure = initial_pressure

        # Initial stress calculation - use parameters
        rock_density_gradient = getattr(self.params, "rock_density_gradient_psi_per_ft", 0.434)
        vertical_stress = rock_density_gradient * grid.depth
        horizontal_stress_ratio = getattr(self.params, "horizontal_stress_ratio", 0.7)
        horizontal_stress = horizontal_stress_ratio * vertical_stress

        # ENHANCED: Add tectonic shear stress for fault activation
        # Differential horizontal stress creates shear on non-ideal fault orientations
        tectonic_shear_ratio = getattr(self.params, "tectonic_shear_ratio", 0.3)
        max_horizontal_stress = horizontal_stress * (1.0 + tectonic_shear_ratio)
        min_horizontal_stress = horizontal_stress * (1.0 - tectonic_shear_ratio)

        initial_stress = np.zeros((grid.n_cells, 6))
        initial_stress[:, 0] = max_horizontal_stress  # sigma_xx (max horizontal)
        initial_stress[:, 1] = min_horizontal_stress  # sigma_yy (min horizontal)
        initial_stress[:, 2] = vertical_stress  # sigma_zz (vertical)
        # Add shear stress components for strike-slip faulting regime
        initial_stress[:, 3] = 0.1 * horizontal_stress  # tau_xy (in-plane shear)
        initial_stress[:, 4] = 0.05 * horizontal_stress  # tau_xz (out-of-plane shear)
        initial_stress[:, 5] = 0.0  # tau_yz

        self.initial_stress = initial_stress
        self.initial_mean_stress = np.mean(initial_stress[:, :3], axis=1)
        self.pore_compressibility = self.params.pore_compressibility


class MineralizationKinetics:
    """Implements CO₂ dissolution and mineralization kinetics"""

    # --- *** CORRECTED TYPE HINT *** ---
    def __init__(self, params: CCUSParameters, eos_model: ReservoirFluid, eor_params=None):
        self.params = params
        self.eos_model = eos_model
        self.eor_params = eor_params  # Store eor_params for temperature conversion

    def update_dissolution_mineralization(self, state: CCUSState, dt: float) -> CCUSState:
        new_dissolved = state.dissolved_co2.copy()
        new_mineral = state.mineral_precipitate.copy()
        for i in range(len(state.pressure)):
            context_id = f"time={state.current_time:.2f}d-cell={i}"

            # --- *** START OF FIX: Unit conversions and correct result parsing *** ---
            # Use _PHYS_CONSTANTS.PSI_TO_PA as single source of truth
            psi_to_pa = _PHYS_CONSTANTS.PSI_TO_PA
            # Keep EORParameters for temperature conversions (not in PhysicalConstants yet)
            fahrenheit_to_kelvin_offset = self.eor_params.fahrenheit_to_kelvin_offset
            fahrenheit_to_kelvin_scale = self.eor_params.fahrenheit_to_kelvin_scale

            pressure_pa = state.pressure[i] * psi_to_pa
            temp_k = (
                self.params.temperature - 32
            ) * fahrenheit_to_kelvin_scale + fahrenheit_to_kelvin_offset

            eos_results = self.eos_model.get_properties_si(
                temperature_K=temp_k, pressure_Pa=pressure_pa
            )

            # Correctly parse the nested dictionary from CubicEOS
            phase_props = eos_results.get("vapor_properties", {}) or eos_results.get(
                "liquid_properties", {}
            )
            if phase_props and "density_kg_per_m3" in phase_props:
                co2_density_kg_m3 = phase_props["density_kg_per_m3"]
                # Convert to mol/m³ (molar mass of CO2)
                co2_molar_mass = getattr(self.params, "co2_molar_mass_kg_per_mol", 0.04401)
                co2_solubility = co2_density_kg_m3 / co2_molar_mass

                # Dissolution kinetics: dC/dt = k_diss(C* - C)
                dissolution_rate = self.params.dissolution_rate_constant * (
                    co2_solubility - state.dissolved_co2[i]
                )
                new_dissolved[i] += dissolution_rate * dt

                # Mineralization kinetics: dM/dt = k_min * C * A_s
                if new_dissolved[i] > 0:
                    mineralization_rate = (
                        self.params.mineralization_rate_constant
                        * new_dissolved[i]
                        * self.params.reactive_surface_area
                    )
                    new_mineral[i] += mineralization_rate * dt
            # --- *** END OF FIX *** ---

        return CCUSState(
            pressure=state.pressure,
            saturations=state.saturations,
            compositions=state.compositions,
            porosity=state.porosity,
            permeability=state.permeability,
            stress=state.stress,
            fault_transmissibility=state.fault_transmissibility,
            dissolved_co2=new_dissolved,
            mineral_precipitate=new_mineral,
            current_time=state.current_time,
            timestep=state.timestep,
            fault_stability=state.fault_stability,
            injection_rates=state.injection_rates,
        )


class CCUSPhysicsEngine:
    """Main coupled CO₂-CCUS physics engine"""

    def __init__(
        self,
        reservoir: ReservoirData,
        pvt: PVTProperties,
        eor_params: EORParameters,
        grid: GridParameters,
        params: CCUSParameters,
    ):
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self.grid = grid
        self.params = params

        # --- CORRECTED EOS Instantiation ---
        if self.reservoir.eos_model:
            self.eos_model = ReservoirFluid(self.reservoir.eos_model)
        else:
            logger.warning(
                "EOS model not found in reservoir data. Using default CubicEOS with PR parameters."
            )
            # Use default EOS parameters from configuration if available
            default_co2_fraction = getattr(self.params, "default_co2_fraction", 0.8)
            default_methane_fraction = getattr(self.params, "default_methane_fraction", 0.2)
            co2_molar_mass = getattr(self.params, "co2_molar_mass", 44.01)
            methane_molar_mass = getattr(self.params, "methane_molar_mass", 16.04)
            co2_critical_temp = getattr(self.params, "co2_critical_temp_k", 304.1)
            methane_critical_temp = getattr(self.params, "methane_critical_temp_k", 190.6)
            co2_critical_pressure = getattr(self.params, "co2_critical_pressure_pa", 7.377e6)
            methane_critical_pressure = getattr(
                self.params, "methane_critical_pressure_pa", 4.604e6
            )
            co2_acentric_factor = getattr(self.params, "co2_acentric_factor", 0.225)
            methane_acentric_factor = getattr(self.params, "methane_acentric_factor", 0.011)
            co2_methane_interaction = getattr(self.params, "co2_methane_binary_interaction", 0.1)

            eos_params = EOSModelParameters(
                eos_type="PR",
                component_names=["CO2", "Methane"],
                component_properties=np.array(
                    [
                        # Using base units (K, Pa) as expected by the new model logic
                        [
                            default_co2_fraction,
                            co2_molar_mass,
                            co2_critical_temp,
                            co2_critical_pressure,
                            co2_acentric_factor,
                        ],
                        [
                            default_methane_fraction,
                            methane_molar_mass,
                            methane_critical_temp,
                            methane_critical_pressure,
                            methane_acentric_factor,
                        ],
                    ]
                ),
                binary_interaction_coeffs=np.array(
                    [[0.0, co2_methane_interaction], [co2_methane_interaction, 0.0]]
                ),
            )
            self.eos_model = ReservoirFluid(eos_params)

        self.flow_solver = MultiphaseFlowSolver(
            grid, reservoir, pvt, self.eor_params, self.eos_model, params
        )
        self.geomechanics_solver = GeomechanicsSolver(grid, params)

        # Fault mechanics is opt-in - only enable if explicitly requested and fault data exists
        enable_fault_mechanics = getattr(params, 'enable_fault_mechanics', False)
        has_fault_data = hasattr(grid, 'fault_cells') and grid.fault_cells and any(grid.fault_cells)
        fault_mechanics_enabled = enable_fault_mechanics and has_fault_data

        if fault_mechanics_enabled:
            logger.info("Fault mechanics engine enabled with fault data")
        else:
            if enable_fault_mechanics and not has_fault_data:
                logger.warning("Fault mechanics requested but no fault data provided - fault mechanics disabled")
            else:
                logger.info("Fault mechanics engine disabled (default)")

        self.fault_mechanics_engine = FaultMechanicsEngine(
            grid, self.reservoir, enabled=fault_mechanics_enabled
        )
        self.mineralization = MineralizationKinetics(params, self.eos_model, self.eor_params)

        # Initialize fully implicit solver for coupled pressure-saturation-geomechanics
        # Solves pressure, saturation, and geomechanics simultaneously within Newton iterations
        # This is the recommended approach for CO2-EOR physics accuracy
        enable_fully_implicit = getattr(params, 'enable_fully_implicit', True)  # ENABLED by default
        if enable_fully_implicit:
            self.fully_implicit_solver = FullyImplicitSolver(
                n_cells=grid.n_cells,
                n_phases=3,  # oil, water, gas
                max_newton_iterations=getattr(params, 'max_newton_iterations', 20),
                newton_tolerance=getattr(params, 'newton_tolerance', 1e-6),
                damping_factor=getattr(params, 'newton_damping_factor', 0.7),
                trust_region_radius=getattr(params, 'trust_region_radius', 100.0),
                enable_line_search=getattr(params, 'enable_line_search', True),
            )
            logger.info("Fully implicit solver initialized for coupled physics")
        else:
            self.fully_implicit_solver = None
            logger.info("Sequential coupling mode (fully implicit disabled)")

        self.current_state: Optional[CCUSState] = None
        self.state_history: List[CCUSState] = []

    def initialize_simulation(
        self,
        initial_pressure: np.ndarray,
        initial_saturations: np.ndarray,
        initial_compositions: np.ndarray,
    ) -> None:
        n_cells = len(initial_pressure)

        # Validate and convert pressures if needed
        validated_pressure = self._validate_pressures(initial_pressure)

        # Use PhysicalConstants and AdvancedEngineParams as single source of truth
        default_porosity = getattr(
            self.params, "default_porosity", _PHYS_CONSTANTS.DEFAULT_POROSITY
        )
        default_permeability = getattr(
            self.params, "default_permeability", _PHYS_CONSTANTS.DEFAULT_PERMEABILITY_MD
        )
        porosity_value = (
            getattr(self.reservoir, "average_porosity", default_porosity) or default_porosity
        )
        permeability_value = default_permeability

        logger.info(f"Initializing simulation with validated pressures:")
        logger.info(f"  Min pressure: {np.min(validated_pressure):.1f} psi")
        logger.info(f"  Max pressure: {np.max(validated_pressure):.1f} psi")
        logger.info(f"  Avg pressure: {np.mean(validated_pressure):.1f} psi")

        self.current_state = CCUSState(
            pressure=validated_pressure,
            saturations=initial_saturations,
            compositions=initial_compositions,
            porosity=np.full(n_cells, porosity_value),
            permeability=np.full(n_cells, permeability_value),
            stress=np.zeros((n_cells, 6)),
            fault_transmissibility=np.ones(len(self.grid.fault_cells)),
            dissolved_co2=np.zeros(n_cells),
            mineral_precipitate=np.zeros(n_cells),
            current_time=0.0,
            timestep=0.0,
            injection_rates={},
        )
        self.geomechanics_solver.set_initial_state(
            self.current_state.porosity,
            self.current_state.permeability,
            self.current_state.pressure,
            self.current_state.stress,
            self.grid,
        )
        # CRITICAL FIX: Update CCUSState.stress with calculated geostatic stress
        # The geomechanics solver calculates proper initial stress from depth,
        # but we need to copy it back to the state for fault mechanics to work
        self.current_state.stress = self.geomechanics_solver.initial_stress.copy()
        logger.info(f"Initial geostatic stress: {np.mean(self.current_state.stress[:, 2]):.0f} psi vertical")
        self.state_history.append(self.current_state)

    def _validate_pressures(self, pressures: np.ndarray) -> np.ndarray:
        """
        Validate and convert pressures to realistic PSI values.

        Handles the common error of passing Pascal values where PSI is expected.
        Also applies realistic bounds checking.

        Args:
            pressures: Array of pressure values

        Returns:
            Validated pressure array in PSI
        """
        validated = pressures.copy()
        converted_any = False

        for i, p in enumerate(pressures):
            if p > 1e7:  # Likely Pascal input
                converted = p / _PHYS_CONSTANTS.PSI_TO_PA
                logger.warning(
                    f"Pressure[{i}] = {p:.1f} appears to be in Pascals. "
                    f"Converting to PSI: {converted:.1f} psi"
                )
                validated[i] = converted
                converted_any = True

        # Apply realistic bounds
        MIN_REALISTIC_PRESSURE = 14.7  # atmospheric
        MAX_REALISTIC_PRESSURE = 20000.0  # ~30,000 ft depth

        if np.any(validated < MIN_REALISTIC_PRESSURE):
            logger.warning(
                f"Pressures below minimum {MIN_REALISTIC_PRESSURE} psi detected. "
                f"Min: {np.min(validated):.1f} psi. Clipping to minimum."
            )
            validated = np.maximum(validated, MIN_REALISTIC_PRESSURE)

        if np.any(validated > MAX_REALISTIC_PRESSURE):
            logger.warning(
                f"Pressures above maximum {MAX_REALISTIC_PRESSURE} psi detected. "
                f"Max: {np.max(validated):.1f} psi. Clipping to maximum."
            )
            validated = np.minimum(validated, MAX_REALISTIC_PRESSURE)

        return validated

    def run_timestep(
        self,
        dt: float,
        injection_rates: Optional[Dict] = None,
        timestep_index: int = 0,
        well_control_logic: Optional[Any] = None,
    ) -> tuple[CCUSState, float]:
        if self.current_state is None:
            raise RuntimeError("Simulation not initialized")
        state = self.current_state

        # Use fully implicit solver if enabled (recommended for CO2-EOR)
        # This couples pressure, saturation, and geomechanics within Newton iterations
        if self.fully_implicit_solver is not None:
            state, convergence_info = self.fully_implicit_solver.solve_timestep(
                state_old=state,
                dt=dt,
                injection_rates=injection_rates,
                flow_solver=self.flow_solver,
                geomech_solver=self.geomechanics_solver
            )

            # Log convergence information
            if not convergence_info.converged:
                logger.warning(
                    f"Newton did not converge in {convergence_info.iterations} iterations. "
                    f"Final residual: {convergence_info.final_residual_norm:.2e}"
                )
            else:
                logger.debug(
                    f"Newton converged in {convergence_info.iterations} iterations, "
                    f"final residual: {convergence_info.final_residual_norm:.2e}"
                )
        else:
            # Fallback to sequential coupling (original method)
            # Note: This is LOOSE coupling - geomechanics updated AFTER flow solve
            state, dt = self.flow_solver.solve_flow(state, dt, injection_rates, well_control_logic)
            state = self.geomechanics_solver.update_poroelastic_properties(state)
            logger.debug("Sequential coupling mode - geomechanics updated after flow")

        # Fault mechanics update (can remain outside Newton loop for efficiency)
        if (self.params.enable_fault_mechanics and
            self.params.fault_mechanics_update_frequency > 0 and
            timestep_index % self.params.fault_mechanics_update_frequency == 0):
            stability_results = self.fault_mechanics_engine.analyze_fault_stability(
                state, state.pressure, self.params.biot_coefficient, state.current_time
            )
            trans_multipliers_dict = (
                self.fault_mechanics_engine.get_fault_transmissibility_multipliers()
            )
            new_transmissibility = (
                np.array([trans_multipliers_dict[i] for i in sorted(trans_multipliers_dict.keys())])
                if trans_multipliers_dict
                else state.fault_transmissibility
            )
        else:
            stability_results = state.fault_stability
            new_transmissibility = state.fault_transmissibility

        state = CCUSState(
            pressure=state.pressure,
            saturations=state.saturations,
            compositions=state.compositions,
            porosity=state.porosity,
            permeability=state.permeability,
            stress=state.stress,
            fault_transmissibility=new_transmissibility,
            dissolved_co2=state.dissolved_co2,
            mineral_precipitate=state.mineral_precipitate,
            current_time=state.current_time,
            timestep=state.timestep,
            fault_stability=stability_results,
            injection_rates=state.injection_rates,
        )
        state = self.mineralization.update_dissolution_mineralization(state, dt)
        self.current_state = state
        self.state_history.append(state)
        return state, dt

    def run_simulation(
        self,
        total_time: float,
        timesteps: List[float] = None,
        injection_schedule: Optional[Dict] = None,
        well_control_logic: Optional[Any] = None,
    ) -> List[CCUSState]:
        # Use configurable start date
        start_year = getattr(self.params, "simulation_start_year", 2023)
        start_month = getattr(self.params, "simulation_start_month", 1)
        start_day = getattr(self.params, "simulation_start_day", 1)
        start_date = datetime(start_year, start_month, start_day)

        new_injection_schedule = {}
        if injection_schedule:
            for day, rates in injection_schedule.items():
                new_injection_schedule[start_date + timedelta(days=day)] = rates

        if timesteps:
            current_time = 0.0
            for i, dt in enumerate(timesteps):
                current_date = start_date + timedelta(days=current_time)

                rates_to_use = {}
                if new_injection_schedule:
                    applicable_dates = [d for d in new_injection_schedule if d <= current_date]
                    if applicable_dates:
                        latest_date = max(applicable_dates)
                        rates_to_use = new_injection_schedule[latest_date]

                self.run_timestep(
                    dt, rates_to_use, timestep_index=i, well_control_logic=well_control_logic
                )
                current_time += dt
            return self.state_history

        end_date = start_date + timedelta(days=total_time)

        timestep_config = TimestepConfig(
            unit=self.params.timestep_unit,
            base_dt_days=self.params.max_timestep_days,  # Or some other logic
            max_dt_days=self.params.max_timestep_days,
            min_dt_days=self.params.min_timestep_days,
            adaptive_stepping=True,
        )

        schedule = SimulationSchedule(
            start_date=start_date,
            end_date=end_date,
            timestep_config=timestep_config,
            injection_schedule=new_injection_schedule,
        )

        time_stepper = CoupledPhysicsTimeStepper(self, schedule)
        time_stepper.initialize_simulation(self.current_state.__dict__)

        results = time_stepper.run_full_simulation()

        # Convert results back to a list of CCUSState objects
        history = []
        for res in results:
            # The state snapshot is a dict, convert it to a CCUSState object
            state_dict = res.state_snapshot
            # We need to handle the case where the snapshot is empty on failure
            if state_dict:
                # Ensure all required fields for CCUSState are present
                required_fields = [f.name for f in dataclasses.fields(CCUSState)]
                # Filter out extra fields that might be in the dict but not in CCUSState
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in required_fields}
                history.append(CCUSState(**filtered_state_dict))

        self.state_history = history
        return self.state_history

    def calculate_storage_partitioning(
        self, injection_history: Optional[Dict[float, float]] = None
    ) -> Dict[str, float]:
        if not self.state_history:
            return {}
        final_state = self.state_history[-1]

        # Log final CO2 saturations for debugging
        co2_sats = final_state.saturations[:, 2]  # CO2 is index 2
        logger.info(f"Final CO2 saturation analysis:")
        logger.info(f"  CO2 saturation range: {co2_sats.min():.6f} - {co2_sats.max():.6f}")
        logger.info(f"  Mean CO2 saturation: {co2_sats.mean():.6f}")
        logger.info(f"  Total cells with CO2 > 0.001: {np.sum(co2_sats > 0.001)}/{len(co2_sats)}")

        total_co2_injected = self._calculate_total_injected_co2(injection_history)
        structural_trapping = self._calculate_structural_trapping(final_state)
        residual_trapping = self._calculate_residual_trapping(final_state)
        solubility_trapping = self._calculate_solubility_trapping(final_state)
        mineral_trapping = self._calculate_mineral_trapping(final_state)
        leakage = self._calculate_leakage(
            final_state,
            total_co2_injected,
            structural_trapping,
            residual_trapping,
            solubility_trapping,
            mineral_trapping,
        )
        mobile_co2 = self._calculate_mobile_co2(final_state)
        partitioning = {
            "structural_trapping_tonne": structural_trapping,
            "residual_trapping_tonne": residual_trapping,
            "solubility_trapping_tonne": solubility_trapping,
            "mineral_trapping_tonne": mineral_trapping,
            "leakage_tonne": leakage,
            "mobile_co2_tonne": mobile_co2,
            "total_injected_tonne": total_co2_injected,
            "total_stored_tonne": (
                structural_trapping
                + residual_trapping
                + solubility_trapping
                + mineral_trapping
                + mobile_co2
            ),
        }
        conservation_error = self._verify_mass_conservation(partitioning)
        partitioning["mass_conservation_error_percent"] = conservation_error * 100
        mass_conservation_tolerance = getattr(self.params, "mass_conservation_tolerance", 0.0001)
        if abs(conservation_error) > mass_conservation_tolerance:
            logger.warning(
                f"Mass conservation error exceeds {mass_conservation_tolerance * 100:.2f}%: {conservation_error * 100:.4f}%"
            )

        # Add validated percentage calculations
        total_injected = partitioning["total_injected_tonne"]
        if total_injected > 0:
            partitioning["structural_trapping_percent"] = (
                partitioning["structural_trapping_tonne"] / total_injected
            ) * 100
            partitioning["residual_trapping_percent"] = (
                partitioning["residual_trapping_tonne"] / total_injected
            ) * 100
            partitioning["solubility_trapping_percent"] = (
                partitioning["solubility_trapping_tonne"] / total_injected
            ) * 100
            partitioning["mineral_trapping_percent"] = (
                partitioning["mineral_trapping_tonne"] / total_injected
            ) * 100
            partitioning["mobile_co2_percent"] = (
                partitioning["mobile_co2_tonne"] / total_injected
            ) * 100
            partitioning["leakage_percent"] = (partitioning["leakage_tonne"] / total_injected) * 100

            # Validate total percentage
            total_percent = (
                partitioning["structural_trapping_percent"]
                + partitioning["residual_trapping_percent"]
                + partitioning["solubility_trapping_percent"]
                + partitioning["mineral_trapping_percent"]
                + partitioning["mobile_co2_percent"]
                + partitioning["leakage_percent"]
            )

            partitioning["total_accounted_percent"] = total_percent
            mass_balance_tolerance_percent = getattr(
                self.params, "mass_balance_tolerance_percent", 0.1
            )
            if abs(total_percent - 100.0) > mass_balance_tolerance_percent:
                logger.error(
                    f"CO2 trapping percentage error: {total_percent:.2f}% (should be 100%)"
                )

        return partitioning

    def _calculate_total_injected_co2(
        self, injection_history: Optional[Dict[float, Dict[str, float]]]
    ) -> float:
        if not injection_history:
            return 0.0
        total_injected = 0.0
        times = sorted(injection_history.keys())
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            rate = injection_history[times[i - 1]].get("co2", 0)
            total_injected += rate * dt
        return total_injected

    def _calculate_structural_trapping(self, state: CCUSState) -> float:
        structural_mass = 0.0
        co2_density_kg_m3 = self.params.supercritical_co2_density_kg_m3
        for i in range(len(state.pressure)):
            if self.grid.depth[i] > self.params.structural_trapping_depth_threshold_m:
                co2_mass_kg = (
                    state.saturations[i, 2]
                    * state.porosity[i]
                    * self.grid.cell_volumes[i]
                    * co2_density_kg_m3
                )
                structural_mass += co2_mass_kg
        return structural_mass / _PHYS_CONSTANTS.KG_TO_TONNES

    def _calculate_residual_trapping(self, state: CCUSState) -> float:
        residual_mass = 0.0
        co2_density_kg_m3 = self.params.supercritical_co2_density_kg_m3
        residual_saturation = self.params.residual_gas_saturation_trapping
        for i in range(len(state.pressure)):
            # Calculate actual residual trapping only from CO2 that has invaded the cell
            # Only trap residual from actual CO2 saturation, not theoretical maximum
            actual_co2_saturation = state.saturations[i, 2]
            residual_co2_saturation = min(actual_co2_saturation, residual_saturation)
            residual_co2_mass_kg = (
                residual_co2_saturation
                * state.porosity[i]
                * self.grid.cell_volumes[i]
                * co2_density_kg_m3
            )
            residual_mass += residual_co2_mass_kg
        return residual_mass / _PHYS_CONSTANTS.KG_TO_TONNES

    def _calculate_solubility_trapping(self, state: CCUSState) -> float:
        total_dissolved_mol = np.sum(state.dissolved_co2 * self.grid.cell_volumes)
        co2_molar_mass_kg_per_mol = _PHYS_CONSTANTS.CO2_MOLECULAR_WEIGHT_KG_MOL
        solubility_mass_kg = total_dissolved_mol * co2_molar_mass_kg_per_mol
        return solubility_mass_kg / _PHYS_CONSTANTS.KG_TO_TONNES

    def _calculate_mineral_trapping(self, state: CCUSState) -> float:
        total_mineral_kg = np.sum(state.mineral_precipitate * self.grid.cell_volumes)
        return total_mineral_kg / _PHYS_CONSTANTS.KG_TO_TONNES

    def _calculate_mobile_co2(self, state: CCUSState) -> float:
        mobile_mass = 0.0
        co2_density_kg_m3 = self.params.supercritical_co2_density_kg_m3
        residual_saturation = self.params.residual_gas_saturation_trapping
        for i in range(len(state.pressure)):
            mobile_saturation = max(0.0, state.saturations[i, 2] - residual_saturation)
            mobile_co2_mass_kg = (
                mobile_saturation
                * state.porosity[i]
                * self.grid.cell_volumes[i]
                * co2_density_kg_m3
            )
            mobile_mass += mobile_co2_mass_kg
        return mobile_mass / _PHYS_CONSTANTS.KG_TO_TONNES

    def _calculate_leakage(
        self,
        state: CCUSState,
        total_injected: float,
        structural: float,
        residual: float,
        solubility: float,
        mineral: float,
    ) -> float:
        fault_leakage = 0.0
        fault_transmissibility_threshold = getattr(
            self.params, "fault_transmissibility_threshold", 1.0
        )
        if state.fault_transmissibility is not None and len(self.grid.fault_cells) == len(
            state.fault_transmissibility
        ):
            for i, fault_cells in enumerate(self.grid.fault_cells):
                if (
                    fault_cells
                    and state.fault_transmissibility[i] > fault_transmissibility_threshold
                ):
                    fault_volume = np.sum(self.grid.cell_volumes[fault_cells])
                    fault_leakage += fault_volume * self.params.fault_leakage_factor
        return fault_leakage

    def _verify_mass_conservation(self, partitioning: Dict[str, float]) -> float:
        total_injected = partitioning["total_injected_tonne"]
        structural = partitioning["structural_trapping_tonne"]
        residual = partitioning["residual_trapping_tonne"]
        solubility = partitioning["solubility_trapping_tonne"]
        mineral = partitioning["mineral_trapping_tonne"]
        mobile = partitioning["mobile_co2_tonne"]
        leakage = partitioning["leakage_tonne"]

        total_accounted = structural + residual + solubility + mineral + mobile + leakage

        # Detailed breakdown logging
        logger.info(f"Mass conservation breakdown:")
        logger.info(
            f"  Structural: {structural:.2f} tonne ({structural / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )
        logger.info(
            f"  Residual: {residual:.2f} tonne ({residual / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )
        logger.info(
            f"  Solubility: {solubility:.2f} tonne ({solubility / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )
        logger.info(
            f"  Mineral: {mineral:.2f} tonne ({mineral / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )
        logger.info(
            f"  Mobile: {mobile:.2f} tonne ({mobile / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )
        logger.info(
            f"  Leakage: {leakage:.2f} tonne ({leakage / total_injected * 100 if total_injected > 0 else 0:.1f}%)"
        )

        if total_injected > 0:
            conservation_error = (total_accounted - total_injected) / total_injected
        else:
            conservation_error = 0.0
        logger.info(
            f"Mass conservation check: Injected={total_injected:.2f} tonne, Accounted={total_accounted:.2f} tonne, Error={conservation_error * 100:.4f}%"
        )
        return conservation_error
