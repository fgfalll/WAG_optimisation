"""
Profile Generator Module for CO2 EOR Simulation
Main simulation loop that integrates all components to generate production profiles.
"""

import numpy as np
import logging
from typing import Dict, Optional, Any

try:
    from core.Phys_engine_full.ccus_physics_engine import (
        CCUSPhysicsEngine as PhysicsEngine,
        CCUSParameters,
        GridParameters,
        TimestepUnit,
    )
    from core.geology import GeologyEngine
    from core.simulation.injection_schemes import InjectionSchemes
    from core.Phys_engine_full.pressure_dynamics import PressureDynamics
    from core.utils.profiler_utils import ProfilerUtils
    from core.unified_engine.physics.eos import CubicEOS, PhaseEquilibriumCalculator, ReservoirFluid
    from core.data_models import PhysicalConstants
except ImportError:
    # Fallback to relative imports - disable problematic imports for figure generation
    PhysicsEngine = None
    CCUSParameters = None
    GridParameters = None
    TimestepUnit = None
    GeologyEngine = None
    InjectionSchemes = None
    PressureDynamics = None
    ProfilerUtils = None
    ReservoirFluid = None
    EOSCalculationError = None
    PhysicalConstants = None

_PHYS_CONSTANTS = PhysicalConstants() if PhysicalConstants else None
logger = logging.getLogger(__name__)


class ProfileGenerator:
    """
    Main profile generator for CO2 EOR simulation.
    Integrates all components to generate production profiles.
    """

    def __init__(
        self,
        reservoir,
        pvt,
        eor_params,
        op_params,
        profile_params,
        ccus_params,
        grid_params: Optional[GridParameters] = None,
        initial_pressure_override: Optional[float] = None,
        well_control_logic: Optional[Any] = None,
    ):
        self.reservoir = reservoir
        self.pvt = pvt
        self.eor_params = eor_params
        self.op_params = op_params
        self.profile_params = profile_params
        self.ccus_params = ccus_params
        self.initial_pressure = (
            initial_pressure_override
            if initial_pressure_override is not None
            else self.reservoir.initial_pressure
        )
        self.temperature_F = self.pvt.temperature

        self.geology_engine = GeologyEngine(reservoir, self.pvt, eor_params)
        self.injection_schemes = InjectionSchemes(eor_params)
        self.pore_volume_bbl = ProfilerUtils.calculate_pore_volume(reservoir)

        self.reservoir_fluid = None
        if self.reservoir.eos_model:
            try:
                self.reservoir_fluid = ReservoirFluid(self.reservoir.eos_model)
            except Exception as e:
                logger.warning(f"Failed to initialize ReservoirFluid: {e}")

        # --- Initialize the full CCUS Physics Engine ---
        if grid_params:
            self.grid_params = grid_params
        else:
            self.grid_params = self._create_grid_parameters()

        self.physics_engine = PhysicsEngine(
            reservoir=self.reservoir,
            pvt=self.pvt,
            eor_params=self.eor_params,
            grid=self.grid_params,
            params=self.ccus_params,
        )

        self.pressure_dynamics = PressureDynamics(
            reservoir, eor_params, pvt, self.pore_volume_bbl, self.initial_pressure
        )
        self.areal_sweep_efficiency = (
            self.geology_engine.calculate_geology_enhanced_sweep_efficiency()
        )
        self.injection_schedule = None
        self.well_control_logic = well_control_logic

    def _create_grid_parameters(self) -> GridParameters:
        # Simplified 1D grid for now. This can be expanded for 3D.
        n_cells = 10  # Example: 10 cells for a 1D model
        pressure_gradient = (
            _PHYS_CONSTANTS.PRESSURE_GRADIENT_TYPICAL if _PHYS_CONSTANTS else 0.45
        )  # psi/ft
        estimated_depth_ft = self.reservoir.initial_pressure / pressure_gradient

        area_ft2 = self.reservoir.cross_sectional_area_acres * 43560.0
        dy_val = np.sqrt(area_ft2)
        dz_val = dy_val

        return GridParameters(
            dimensions=(n_cells, 1, 1),
            cell_volumes=np.full(n_cells, self.pore_volume_bbl * 5.61458 / n_cells),  # ft^3
            dx=np.full(n_cells, self.reservoir.length_ft / n_cells),
            dy=np.full(n_cells, dy_val),
            dz=np.full(n_cells, dz_val),
            depth=np.linspace(estimated_depth_ft, estimated_depth_ft + 100, n_cells),
            tops=np.linspace(estimated_depth_ft, estimated_depth_ft + 100, n_cells),
            fault_cells=[],
            fault_orientations=[],
        )

    def generate_all_profiles(self, ooip_stb: float, **kwargs) -> Dict:
        """
        Generates production and injection profiles using the CCUSPhysicsEngine.
        """
        if ooip_stb <= 0:
            raise ValueError(f"OOIP must be positive, got {ooip_stb}")

        project_life_days = int(self.op_params.project_lifetime_years * ProfilerUtils.DAYS_PER_YEAR)

        # --- Setup Injection Schedule ---
        daily_co2_inj_sched = np.zeros(project_life_days)
        daily_water_inj_sched = np.zeros(project_life_days)
        daily_hnp_cycle = np.zeros(project_life_days, dtype=int)
        geology_enhanced_injection_rate = (
            self.eor_params.injection_rate
            * self.geology_engine.calculate_geology_injection_factor()
        )

        self.injection_schemes.setup_injection_scheme(
            daily_co2_inj_sched,
            daily_water_inj_sched,
            project_life_days,
            geology_enhanced_injection_rate,
            1.0,
            daily_hnp_cycle,  # b_gas is now placeholder
        )

        injection_schedule = {
            day: {"co2": daily_co2_inj_sched[day], "water": daily_water_inj_sched[day]}
            for day in range(project_life_days)
        }
        self.injection_schedule = injection_schedule

        # --- Initialize Simulation State ---
        n_cells = self.grid_params.n_cells
        initial_pressure = np.full(n_cells, self.initial_pressure)
        initial_saturations = np.zeros((n_cells, 3))
        initial_saturations[:, 0] = 1.0 - self.reservoir.initial_water_saturation  # Oil
        initial_saturations[:, 1] = self.reservoir.initial_water_saturation  # Water
        initial_saturations[:, 2] = 0.0  # Gas
        initial_compositions = np.zeros((n_cells, 2))  # CO2 and CH4 for example
        initial_compositions[:, 1] = 1.0  # Assume pure CH4 initially

        self.physics_engine.initialize_simulation(
            initial_pressure, initial_saturations, initial_compositions
        )

        # --- Run Simulation ---
        days_in_step = self.ccus_params.timestep_days
        num_steps = int(project_life_days / days_in_step)
        timesteps = [days_in_step] * num_steps
        if sum(timesteps) < project_life_days:
            timesteps.append(project_life_days - sum(timesteps))

        state_history = self.physics_engine.run_simulation(
            total_time=project_life_days,
            timesteps=timesteps,
            injection_schedule=injection_schedule,
            well_control_logic=self.well_control_logic,
        )

        # --- Process Results ---
        n_states = len(state_history)
        logger.info(f"Simulation completed with {n_states} states in history")

        if n_states <= 1:
            logger.warning(f"Insufficient simulation states: {n_states}. Profiles will be empty.")
        else:
            logger.info(f"Processing {n_states - 1} simulation states for timestep-based profiles")
        project_life_days = int(self.op_params.project_lifetime_years * ProfilerUtils.DAYS_PER_YEAR)

        # Determine output resolution based on timestep size
        timestep_days = self.ccus_params.timestep_days
        if timestep_days <= 1:
            output_resolution = "daily"
            n_output_points = project_life_days
        elif timestep_days <= 7:
            output_resolution = "weekly"
            n_output_points = int(project_life_days / 7)
        elif timestep_days <= 31:
            output_resolution = "monthly"
            n_output_points = int(project_life_days / 30.44)
        else:
            output_resolution = "yearly"
            n_output_points = self.op_params.project_lifetime_years

        logger.info(f"Creating {output_resolution} profiles with {n_output_points} output points")

        # Create output arrays based on timestep resolution
        pressure_full = np.zeros(n_output_points)
        oil_stb_full = np.zeros(n_output_points)
        water_prod_bbl_full = np.zeros(n_output_points)
        co2_prod_mscf_full = np.zeros(n_output_points)
        injector_bhp_full = np.zeros(n_output_points)
        producer_bhp_full = np.zeros(n_output_points)
        injector_porosity_full = np.zeros(n_output_points)
        injector_permeability_full = np.zeros(n_output_points)
        co2_inj_actual_full = np.zeros(n_output_points)

        current_output_index = 0

        for i, state in enumerate(state_history):
            if i == 0:
                continue

            # --- Dynamic PVT Calculation ---
            pressure_for_step = np.mean(state.pressure)
            temp_K = (self.temperature_F - 32) * 5 / 9 + 273.15
            pressure_Pa = pressure_for_step * 6894.76

            B_oil = 1.2  # Default
            B_gas = 0.01  # Default
            B_water = 1.02  # Default
            mu_oil = 2.0  # cP, default
            mu_gas = 0.02  # cP, default
            mu_water = 0.5  # cP, default

            if self.reservoir_fluid:
                try:
                    B_oil = self.reservoir_fluid.get_boil_rb_per_stb(temp_K, pressure_Pa)
                    # Viscosity method not available on ReservoirFluid - using default
                    # mu_oil already set to 2.0 cP at line 243
                except Exception as e:
                    # This is expected behavior for CO2 injection scenarios where single-phase conditions exist
                    logger.debug(
                        f"Using default B_oil and mu_oil at P={pressure_for_step:.2f} psi: {e}"
                    )

                try:
                    # Effective B_gas for CO2 injection
                    B_gas = self.reservoir_fluid.get_bgas_rb_per_mscf(temp_K, pressure_Pa)
                    # mu_gas already set to 0.02 cP at line 244 - viscosity method not available on ReservoirFluid
                except Exception as e:
                    # Liquid phase or calculation error - use B_oil equivalent
                    stb_per_mscf = 1000 / 5.615
                    B_gas = B_oil * stb_per_mscf
                    logger.debug(
                        f"Using B_oil equivalent for B_gas at P={pressure_for_step:.2f} psi: {e}"
                    )

            dt = timesteps[i - 1]
            num_days_in_step = int(round(dt))

            # Simplified production calculation (from existing code)
            prod_cell_idx = n_cells - 1
            if n_cells > 1:
                pressure_for_drawdown = state.pressure[prod_cell_idx]
            else:
                pressure_for_drawdown = state.pressure[prod_cell_idx]
            pressure_drawdown = pressure_for_drawdown - self.eor_params.wellbore_pressure
            total_prod_rate = self.eor_params.productivity_index * pressure_drawdown

            oil_rate = 0
            water_rate = 0
            gas_rate = 0
            if total_prod_rate > 0:
                s_oil, s_water, s_gas = state.saturations[prod_cell_idx]
                total_mobility = s_oil / mu_oil + s_water / mu_water + s_gas / mu_gas

                # Prevent division by zero
                if total_mobility > 0:
                    f_oil = (s_oil / mu_oil) / total_mobility
                    f_water = (s_water / mu_water) / total_mobility
                    f_gas = (s_gas / mu_gas) / total_mobility
                else:
                    # Default to equal distribution if total mobility is zero
                    f_oil = f_water = f_gas = 1.0 / 3.0
                    logger.warning(
                        f"Total mobility is zero at step {i}, using equal phase distribution"
                    )

                oil_rate = (total_prod_rate * f_oil) / B_oil
                water_rate = (total_prod_rate * f_water) / B_water
                gas_rate = (total_prod_rate * f_gas) / B_gas

            pressure_for_step = np.mean(state.pressure)
            injector_bhp_for_step = state.pressure[0]
            producer_bhp_for_step = state.pressure[-1]
            injector_porosity_for_step = state.porosity[0]
            injector_permeability_for_step = state.permeability[0]

            actual_inj_rate_co2 = (
                state.injection_rates.get("co2", 0.0) if state.injection_rates else 0.0
            )

            # Store values for this timestep based on output resolution
            if current_output_index < n_output_points:
                pressure_full[current_output_index] = pressure_for_step
                oil_stb_full[current_output_index] = oil_rate * dt  # Convert to cumulative
                water_prod_bbl_full[current_output_index] = water_rate * dt
                co2_prod_mscf_full[current_output_index] = gas_rate * dt
                injector_bhp_full[current_output_index] = injector_bhp_for_step
                producer_bhp_full[current_output_index] = producer_bhp_for_step
                injector_porosity_full[current_output_index] = injector_porosity_for_step
                injector_permeability_full[current_output_index] = injector_permeability_for_step
                co2_inj_actual_full[current_output_index] = (
                    actual_inj_rate_co2 * dt
                )  # Convert to cumulative

                current_output_index += 1

        # Prevent division by zero in pore volume calculation
        if self.pore_volume_bbl > 0:
            # Need to handle different array sizes - use only common elements or expand appropriately
            if len(co2_inj_actual_full) == len(daily_water_inj_sched):
                # Same length arrays
                pore_volumes_injected = (
                    np.cumsum(co2_inj_actual_full * B_gas + daily_water_inj_sched * 1.0)
                    / self.pore_volume_bbl
                )
            elif len(co2_inj_actual_full) < len(daily_water_inj_sched):
                # CO2 array is shorter - use only the available daily water data
                water_subset = daily_water_inj_sched[: len(co2_inj_actual_full)]
                pore_volumes_injected = (
                    np.cumsum(co2_inj_actual_full * B_gas + water_subset * 1.0)
                    / self.pore_volume_bbl
                )
            else:
                # CO2 array is longer - this shouldn't happen with our logic
                min_len = min(len(co2_inj_actual_full), len(daily_water_inj_sched))
                pore_volumes_injected = (
                    np.cumsum(
                        co2_inj_actual_full[:min_len] * B_gas
                        + daily_water_inj_sched[:min_len] * 1.0
                    )
                    / self.pore_volume_bbl
                )
        else:
            logger.error("Pore volume is zero - cannot calculate pore volumes injected")
            pore_volumes_injected = np.zeros_like(co2_inj_actual_full)

        # Fallback: If simulation produced no CO2 injection, use schedule directly
        if np.sum(co2_inj_actual_full) == 0 and injection_schedule:
            logger.warning(
                "No CO2 injection from simulation. Using injection schedule as fallback."
            )
            # Apply injection schedule to output arrays
            for day in range(n_output_points):
                # Find the applicable injection rate for this timestep
                applicable_rate = 0
                for schedule_day, rates in injection_schedule.items():
                    if day * timestep_days >= schedule_day:
                        applicable_rate = rates.get("co2", 0)
                    else:
                        break
                co2_inj_actual_full[day] = applicable_rate

        # Final check and logging
        total_inj = np.sum(co2_inj_actual_full)
        logger.info(
            f"Final CO2 injection sum: {total_inj:.0f} Mscf over {n_output_points} timesteps"
        )
        if total_inj == 0:
            logger.error(
                "CO2 injection is zero! Both simulation and schedule failed to provide injection data."
            )

        # --- Assemble final profiles ---
        profiles = ProfilerUtils.assemble_final_profiles(
            oil_stb_full,
            co2_inj_actual_full,
            daily_water_inj_sched,
            water_prod_bbl_full,
            co2_prod_mscf_full,
            pressure_full,
            pore_volumes_injected,
            self.profile_params.co2_recycling_efficiency_fraction,
            daily_injector_bhp=injector_bhp_full,
            daily_producer_bhp=producer_bhp_full,
            daily_injector_porosity=injector_porosity_full,
            daily_injector_permeability=injector_permeability_full,
        )

        # Resample profiles to requested resolution
        resolution = self.op_params.time_resolution
        resampled_profiles = {}
        for key, daily_data in profiles.items():
            res_key = key.replace("daily", resolution)
            resampled_profiles[res_key] = ProfilerUtils.resample_profile(
                daily_data, resolution, key, self.op_params.project_lifetime_years
            )

        return {**profiles, **resampled_profiles}

    def _check_well_conversion(self, day: int, well_status: np.ndarray, daily_oil_stb: np.ndarray):
        pass  # This logic is now handled by the simulator or well model

    def _update_reservoir_pressure(
        self,
        day: int,
        current_pressure: float,
        daily_co2_inj: np.ndarray,
        daily_water_inj: np.ndarray,
        daily_oil_stb: np.ndarray,
        daily_water_prod_bbl: np.ndarray,
        daily_co2_prod_mscf: np.ndarray,
        daily_hnp_cycle: np.ndarray,
        well_status: np.ndarray,
    ) -> float:
        pass  # This logic is now handled by the CCUSPhysicsEngine

    def _calculate_current_saturations(
        self,
        day: int,
        daily_co2_inj: np.ndarray,
        daily_co2_prod_mscf: np.ndarray,
        daily_water_inj: np.ndarray,
        daily_water_prod_bbl: np.ndarray,
    ) -> tuple:
        pass  # This logic is now handled by the CCUSPhysicsEngine

    def _calculate_production(
        self,
        day: int,
        current_pressure: float,
        enhanced_pi: float,
        daily_co2_inj: np.ndarray,
        daily_water_inj: np.ndarray,
        daily_oil_stb: np.ndarray,
        daily_water_prod_bbl: np.ndarray,
        daily_co2_prod_mscf: np.ndarray,
        daily_hnp_cycle: np.ndarray,
    ):
        pass  # This logic is now handled by the CCUSPhysicsEngine
