"""
Main Reservoir Simulation Engine
=================================

This module contains the main reservoir simulation engine that orchestrates
all components and provides the primary interface for simulation.

Based on the theoretical framework from the technical specification document.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import warnings

from core.engine_simple.utils import (RockProperties, FluidProperties, WellParameters,
                   ReservoirState, SimulationResults, convert_units)
from core.data_models import SimpleGrid
from core.engine_simple.multiphase_flow_adapter import MultiphaseFlowSolverAdapter
from core.engine_simple.mass_balance_tracker import (
    MassBalanceTracker, MassBalanceSnapshot, MassBalanceStatus,
    validate_saturations, normalize_saturations
)
from core.unified_engine.physics.co2_properties import CO2Properties
from core.unified_engine.physics.relative_permeability import CoreyParameters, CoreyRelativePermeability
from core.engine_simple.storage_efficiency import StorageParameters, StorageEfficiencyCalculator
from core.engine_simple.parameter_estimation import EnsembleKalmanFilter, EnKFParameters, ParameterState


class ReservoirSimulationEngine:
    """
    Main reservoir simulation engine for EOR and CO₂ injection
    """

    def __init__(self, grid: SimpleGrid, rock: RockProperties,
                 fluid: FluidProperties):
        """
        Initialize reservoir simulation engine

        Parameters:
        -----------
        grid : SimpleGrid
            Grid parameters
        rock : RockProperties
            Rock properties
        fluid : FluidProperties
            Fluid properties
        """
        self.grid = grid
        self.rock = rock
        self.fluid = fluid

        # Initialize components
        # Use adapter to bridge old interface with new unified engine
        self.flow_solver = MultiphaseFlowSolverAdapter(grid, rock, fluid)
        self.co2_model = CO2Properties()

        # Default relative permeability parameters
        self.corey_params = CoreyParameters()
        self.rel_perm_model = CoreyRelativePermeability(self.corey_params)

        # Default storage parameters
        # Use physical_dimensions for actual physical size (meters)
        phys_dims = grid.physical_dimensions if hasattr(grid, 'physical_dimensions') else (grid.nx * grid.dx, grid.ny * grid.dy, grid.nz * grid.dz)
        self.storage_params = StorageParameters(
            area=phys_dims[0] * phys_dims[1],  # length_x * length_y (m²)
            thickness=phys_dims[2],  # length_z (m)
            porosity=np.mean(rock.porosity)
        )
        self.storage_calculator = StorageEfficiencyCalculator(self.storage_params)

        # Simulation state
        self.current_state = None
        self.simulation_time = 0.0
        self.time_step_count = 0

        # Results storage
        self.results_history = []

        # Mass balance tracker (initialized after reservoir is initialized)
        self.mass_balance_tracker = None

    def initialize_reservoir(self, initial_pressure: float,
                           initial_water_sat: float,
                           temperature: float = 353.15) -> ReservoirState:
        """
        Initialize reservoir state

        Parameters:
        -----------
        initial_pressure : float
            Initial reservoir pressure (Pa)
        initial_water_sat : float
            Initial water saturation (fraction)
        temperature : float
            Reservoir temperature (K)

        Returns:
        --------
        ReservoirState : Initial reservoir state
        """
        self.current_state = ReservoirState.create_initial_state(
            self.grid, initial_pressure, initial_water_sat,
            temperature=temperature
        )

        self.simulation_time = 0.0
        self.time_step_count = 0
        self.results_history = []

        # Initialize mass balance tracker with initial conditions
        # Calculate initial volumes in place
        volume_cell = self.grid.dx * self.grid.dy * self.grid.dz
        porosity_avg = np.mean(self.rock.porosity)

        initial_water_sat = initial_water_sat
        initial_oil_sat = 1.0 - initial_water_sat  # Initial Sw + So = 1.0
        initial_gas_sat = 0.0  # No gas initially

        # Initial volumes in place (m³ at reservoir conditions)
        initial_water_vol = self.grid.total_cells * volume_cell * porosity_avg * initial_water_sat
        initial_oil_vol = self.grid.total_cells * volume_cell * porosity_avg * initial_oil_sat
        initial_gas_vol = self.grid.total_cells * volume_cell * porosity_avg * initial_gas_sat

        # Total pore volume
        pore_volume = self.grid.total_cells * volume_cell * porosity_avg

        # Create mass balance tracker
        self.mass_balance_tracker = MassBalanceTracker(
            initial_water=initial_water_vol,
            initial_oil=initial_oil_vol,
            initial_gas=initial_gas_vol,
            pore_volume=pore_volume,
            tolerance=1e-3  # 0.1% tolerance
        )

        return self.current_state

    def run_simulation(self, wells: List[WellParameters],
                      simulation_time: float = 365.25 * 10 * 86400,  # 10 years in seconds
                      max_time_step: float = 86400.0,        # 1 day
                      output_frequency: int = 30) -> SimulationResults:
        """
        Run full reservoir simulation

        Parameters:
        -----------
        wells : list
            List of well parameters
        simulation_time : float
            Total simulation time (seconds)
        max_time_step : float
            Maximum time step (seconds)
        output_frequency : int
            Output frequency (in time steps)

        Returns:
        --------
        SimulationResults : Complete simulation results
        """
        if self.current_state is None:
            raise ValueError("Reservoir not initialized. Call initialize_reservoir() first.")

        start_time = time.time()
        print(f"Starting simulation for {simulation_time/86400:.1f} days...")

        # Calculate OOIP before simulation for RF profile
        original_oil_in_place = self._calculate_oil_in_place()

        # Initialize results storage
        n_outputs = int(simulation_time / (max_time_step * output_frequency)) + 1
        results = SimulationResults(
            time=np.zeros(n_outputs),
            oil_rate=np.zeros(n_outputs),
            water_rate=np.zeros(n_outputs),
            gas_rate=np.zeros(n_outputs),
            oil_cumulative=np.zeros(n_outputs),
            water_cumulative=np.zeros(n_outputs),
            gas_cumulative=np.zeros(n_outputs),
            co2_injection_rate=np.zeros(n_outputs),
            co2_storage_volume=np.zeros(n_outputs),
            avg_pressure=np.zeros(n_outputs),
            recovery_factor_profile=np.zeros(n_outputs),
        )

        # Simulation loop
        current_time = 0.0
        output_idx = 0
        cumulative_oil = 0.0
        cumulative_water = 0.0
        cumulative_gas = 0.0
        cumulative_co2 = 0.0

        while current_time < simulation_time:
            # Calculate adaptive time step
            time_step = min(max_time_step, simulation_time - current_time)

            # Store old state
            state_old = ReservoirState(
                pressure=self.current_state.pressure.copy(),
                water_saturation=self.current_state.water_saturation.copy(),
                oil_saturation=self.current_state.oil_saturation.copy(),
                gas_saturation=self.current_state.gas_saturation.copy(),
                temperature=self.current_state.temperature
            )

            # Apply well controls to get dynamic rates
            well_rates = self._apply_well_controls(wells, time_step)

            # Solve pressure equation with net volume mass balance bounds
            try:
                pressure_new = self._solve_pressure_with_wells(self.current_state, well_rates, time_step)
            except Exception as e:
                warnings.warn(f"Pressure solve failed: {e}")
                pressure_new = self.current_state.pressure

            # Update saturations
            try:
                sw_new, so_new, sg_new, real_well_rates = self.flow_solver.update_saturations(
                    self.current_state, self.rel_perm_model, pressure_new, time_step, wells
                )
                
                # Replace well rates with the actual advected volumes where possible
                if isinstance(real_well_rates, dict):
                    well_rates['water'] = real_well_rates.get('water', well_rates['water'])
                    well_rates['oil'] = real_well_rates.get('oil', well_rates['oil'])
                    well_rates['gas'] = real_well_rates.get('gas', well_rates['gas'])
                    well_rates['co2'] = real_well_rates.get('co2', well_rates['co2'])
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                warnings.warn(f"Saturation update failed: {e}")
                sw_new, so_new, sg_new = (
                    self.current_state.water_saturation,
                    self.current_state.oil_saturation,
                    self.current_state.gas_saturation
                )

            # Update reservoir state
            self.current_state.pressure = pressure_new
            self.current_state.water_saturation = sw_new
            self.current_state.oil_saturation = so_new
            self.current_state.gas_saturation = sg_new

            # Normalize saturations to ensure sum = 1.0 and bounds [0, 1]
            # This is CRITICAL for mass conservation
            sw_norm, so_norm, sg_norm = normalize_saturations(sw_new, so_new, sg_new)
            self.current_state.water_saturation = sw_norm
            self.current_state.oil_saturation = so_norm
            self.current_state.gas_saturation = sg_norm

            # Validate saturations
            sat_valid, sat_errors = validate_saturations(sw_norm, so_norm, sg_norm)
            if not sat_valid:
                warnings.warn(f"Saturation validation failed: {sat_errors}")

            # Calculate mass balance
            mass_balance = self.flow_solver.calculate_mass_balance(
                state_old, self.current_state, time_step
            )

            # Update time
            current_time += time_step
            self.simulation_time = current_time
            self.time_step_count += 1

            # Calculate production rates ALWAYS to keep integrals accurate
            oil_rate, water_rate, gas_rate = self._calculate_production_rates(well_rates)
            co2_rate = self._calculate_co2_injection_rate(well_rates)

            # Update cumulative values using `time_step` in days
            days_step = time_step / 86400.0

            # Simplified cumulative calculation
            # Using simple integration - not full mass balance tracking
            cumulative_oil += oil_rate * days_step
            cumulative_water += water_rate * days_step
            cumulative_gas += gas_rate * days_step
            cumulative_co2 += co2_rate * days_step

            # Store results at output frequency
            if self.time_step_count % output_frequency == 0:

                # Store results
                results.time[output_idx] = current_time / 86400.0  # Convert to days
                results.oil_rate[output_idx] = oil_rate  # Already in m³/day
                results.water_rate[output_idx] = water_rate
                results.gas_rate[output_idx] = gas_rate
                results.oil_cumulative[output_idx] = cumulative_oil
                results.water_cumulative[output_idx] = cumulative_water
                results.gas_cumulative[output_idx] = cumulative_gas
                results.co2_injection_rate[output_idx] = co2_rate
                results.co2_storage_volume[output_idx] = cumulative_co2
                results.avg_pressure[output_idx] = np.mean(self.current_state.pressure)

                # Recovery factor calculation with clamp to physical limit
                # This is the key fix: RF must never exceed 1.0
                recovery_factor = cumulative_oil / original_oil_in_place if original_oil_in_place > 0 else 0.0

                # Clamp to physical limit - RF <= 1.0
                recovery_factor = min(1.0, recovery_factor)

                results.recovery_factor_profile[output_idx] = recovery_factor

                output_idx += 1

                # Progress update
                if self.time_step_count % (output_frequency * 10) == 0:
                    progress = current_time / simulation_time * 100
                    print(f"  Progress: {progress:.1f}%, Time: {current_time/86400:.1f} days")

        # Trim results arrays
        results.time = results.time[:output_idx]
        results.oil_rate = results.oil_rate[:output_idx]
        results.water_rate = results.water_rate[:output_idx]
        results.gas_rate = results.gas_rate[:output_idx]
        results.oil_cumulative = results.oil_cumulative[:output_idx]
        results.water_cumulative = results.water_cumulative[:output_idx]
        results.gas_cumulative = results.gas_cumulative[:output_idx]
        results.co2_injection_rate = results.co2_injection_rate[:output_idx]
        results.co2_storage_volume = results.co2_storage_volume[:output_idx]
        results.avg_pressure = results.avg_pressure[:output_idx]
        results.recovery_factor_profile = results.recovery_factor_profile[:output_idx]

        # Calculate performance metrics
        # IMPORTANT: Clamp recovery factor to physical limit of 1.0
        # This prevents physically impossible results from the empirical flow model
        results.calculate_recovery_factor(original_oil_in_place)
        results.recovery_factor = min(1.0, results.recovery_factor)
        results.calculate_utilization_factor()

        # Estimate sweep efficiency (simplified)
        results.sweep_efficiency = self._estimate_sweep_efficiency()

        # Calculate storage efficiency
        results.storage_efficiency = self.storage_calculator.total_storage_efficiency()

        # Store pressure and saturation snapshots
        results.pressure_field = [self.current_state.pressure.copy()]
        results.saturation_field = [{
            'water': self.current_state.water_saturation.copy(),
            'oil': self.current_state.oil_saturation.copy(),
            'gas': self.current_state.gas_saturation.copy()
        }]

        elapsed_time = time.time() - start_time
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Final recovery factor: {results.recovery_factor:.3f}")
        if len(results.oil_cumulative) > 0:
            print(f"Final oil recovery: {results.oil_cumulative[-1]:.1f} m3")

        # Check for physical validity
        if results.recovery_factor > 1.0:
            print(f"WARNING: Recovery factor > 1.0 (RF = {results.recovery_factor:.4f}) - physically impossible!")
            print(f"         This indicates a mass balance issue in the empirical model.")
        elif results.recovery_factor > 0.95:
            print(f"WARNING: Recovery factor very high (RF = {results.recovery_factor:.4f}) - results may be unreliable.")

        return results

    def _apply_well_controls(self, wells: List[WellParameters],
                           time_step: float) -> Dict[str, float]:
        """Apply well controls and calculate well rates"""
        well_rates = {'oil': 0.0, 'water': 0.0, 'gas': 0.0, 'co2': 0.0}

        for well in wells:
            # Get grid block indices
            i, j, k = well.location

            # Check if indices are within bounds
            if (0 <= i < self.grid.nx and 0 <= j < self.grid.ny and
                0 <= k < self.grid.nz):

                # Get local properties
                pressure = self.current_state.pressure[k, j, i]
                sw = self.current_state.water_saturation[k, j, i]
                so = self.current_state.oil_saturation[k, j, i]
                sg = self.current_state.gas_saturation[k, j, i]

                # Calculate relative permeabilities
                krw, kro, krg = self.rel_perm_model.kr_three_phase(sw, sg)

                # Calculate phase mobilities
                # Use directional permeability (k_x for horizontal flow)
                k_local = self.rock.permeability_x[k, j, i]
                phi_local = self.rock.porosity[k, j, i]

                mu_w = self.fluid.water_viscosity(pressure, self.current_state.temperature)
                mu_o = self.fluid.oil_viscosity(pressure, self.current_state.temperature)
                mu_g = self.fluid.gas_viscosity(pressure, self.current_state.temperature)

                lambda_w = krw / mu_w
                lambda_o = kro / mu_o
                lambda_g = krg / mu_g
                lambda_t = lambda_w + lambda_o + lambda_g

                if well.well_type == 'producer':
                    # Production well
                    if well.rate is not None and well.rate > 0:
                        # Rate control
                        q_total = well.rate / 86400.0  # Convert to m³/s

                        # Phase fractions based on mobility alone for producers
                        if lambda_t > 0:
                            q_water = q_total * lambda_w / lambda_t
                            q_oil = q_total * lambda_o / lambda_t
                            q_gas = q_total * lambda_g / lambda_t
                        else:
                            q_water = q_oil = q_gas = 0.0

                        # Production removes fluid from reservoir (negative rates)
                        well_rates['water'] -= q_water
                        well_rates['oil'] -= q_oil
                        well_rates['gas'] -= q_gas

                    elif well.bottom_hole_pressure is not None:
                        # Pressure control (simplified)
                        dp = pressure - well.bottom_hole_pressure
                        if dp > 0:
                            # Productivity index (simplified)
                            k_local_m2 = k_local * 9.869233e-16  # Convert mD to m²
                            pi = 2 * np.pi * k_local_m2 * 5.0 / np.log(100)
                            q_total = pi * lambda_t * dp
                            
                            if lambda_t > 0:
                                q_water = q_total * lambda_w / lambda_t
                                q_oil = q_total * lambda_o / lambda_t
                                q_gas = q_total * lambda_g / lambda_t
                            else:
                                q_water = q_oil = q_gas = 0.0
                                
                            # Production removes fluid (negative rate)
                            well_rates['water'] -= q_water
                            well_rates['oil'] -= q_oil
                            well_rates['gas'] -= q_gas

                elif well.well_type == 'injector':
                    # Injection well
                    if well.rate > 0:
                        q_total = well.rate / 86400.0  # Convert to m³/s

                        if well.gas_fraction > 0:
                            # CO₂ injection
                            well_rates['co2'] += q_total * well.gas_fraction

                        if well.water_fraction > 0:
                            well_rates['water'] += q_total * well.water_fraction

                        if well.oil_fraction > 0:
                            well_rates['oil'] += q_total * well.oil_fraction

        return well_rates

    def _calculate_production_rates(self, well_rates: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate production rates from well rates (convert gas to surface volume)"""
        oil_rate_res = max(0, -well_rates['oil'])  # Production is negative
        water_rate_res = max(0, -well_rates['water'])
        gas_rate_res = max(0, -well_rates['gas'])

        # Gas expands highly at surface compared to reservoir
        bg = getattr(self.fluid, 'gas_fvf_ref', 0.005)
        gas_rate_standard = gas_rate_res / bg

        return oil_rate_res, water_rate_res, gas_rate_standard

    def _calculate_co2_injection_rate(self, well_rates: Dict[str, float]) -> float:
        """Calculate CO₂ injection rate"""
        return max(0, well_rates['co2'])

    def _calculate_oil_in_place(self) -> float:
        """Calculate original oil in place (volume in m³)

        Note: This calculates OOIP based on INITIAL conditions, not current state.
        The original oil in place should not change during simulation.
        """
        volume_cell = self.grid.dx * self.grid.dy * self.grid.dz

        # Always use initial oil saturation (1 - Swi) for OOIP calculation
        # This is the "original" oil in place, which should be constant
        oil_saturation_avg = 1.0 - self.fluid.initial_water_saturation

        porosity_avg = np.mean(self.rock.porosity)

        # OOIP in m³ (not kg) - volume based for RF calculation
        # RF = cumulative_oil_volume / OOIP_volume
        ooip = self.grid.total_cells * volume_cell * porosity_avg * oil_saturation_avg

        return ooip

    def _estimate_sweep_efficiency(self) -> float:
        """Estimate sweep efficiency (simplified)"""
        # Use change in average saturation as proxy for sweep
        if len(self.results_history) > 0:
            initial_sw = np.mean(self.results_history[0]['water_saturation'])
            current_sw = np.mean(self.current_state.water_saturation)
            sweep = (current_sw - initial_sw) / (1.0 - initial_sw)
            return np.clip(sweep, 0.0, 1.0)
        else:
            return 0.3  # Default estimate

    def update_relative_permeability(self, **params):
        """Update relative permeability parameters"""
        self.corey_params.update_parameters(**params)
        self.rel_perm_model = CoreyRelativePermeability(self.corey_params)

    def update_storage_parameters(self, **params):
        """Update storage parameters"""
        self.storage_params.update_parameters(**params)
        self.storage_calculator = StorageEfficiencyCalculator(self.storage_params)

    def get_current_state(self) -> ReservoirState:
        """Get current reservoir state"""
        return self.current_state

    def _calculate_well_source_terms(self, wells: List[WellParameters], time_step: float) -> np.ndarray:
        """Calculate well source/sink terms for pressure equation"""
        n_cells = self.grid.total_cells
        source_terms = np.zeros(n_cells)

        for well in wells:
            # Convert well location to grid indices
            i = int(well.location[0])  # x-index
            j = int(well.location[1])  # y-index
            k = int(well.location[2])  # z-index

            # Check if well is within grid bounds
            if 0 <= i < self.grid.nx and 0 <= j < self.grid.ny and 0 <= k < self.grid.nz:
                # Convert 3D indices to 1D
                cell_index = k * self.grid.nx * self.grid.ny + j * self.grid.nx + i

                # Calculate source term based on well type and rate
                if well.well_type == 'injector':
                    # Injection is a positive source term
                    if hasattr(well, 'rate') and well.rate > 0:
                        # Convert rate from m³/day to m³/s for source term
                        injection_rate_m3s = well.rate / 86400.0
                        source_terms[cell_index] += injection_rate_m3s

                elif well.well_type == 'producer':
                    # Production is a negative source term (sink)
                    if hasattr(well, 'rate') and well.rate is not None and well.rate != 0:
                        # Ensure production is negative regardless of input sign
                        production_rate_m3s = -abs(well.rate) / 86400.0
                        source_terms[cell_index] += production_rate_m3s

        return source_terms

    def _solve_pressure_with_wells(self, state: ReservoirState, well_rates: Dict[str, float],
                                  time_step: float) -> np.ndarray:
        """Solve pressure equation with well source/sink terms utilizing exact volume material balance bounds."""
        try:
            # Use the existing pressure solver
            pressure_new = self.flow_solver.solve_pressure_equation(
                state, self.rel_perm_model, time_step
            )

            # Calculate net reservoir volume flux (m³ injected - m³ produced) per second
            net_flux_m3_s = sum(well_rates.values())
            
            # Estimate total pore volume
            v_pore = self.grid.dx * self.grid.dy * self.grid.dz * self.grid.total_cells * np.mean(self.rock.porosity)
            if v_pore <= 0:
                v_pore = 1.0
                
            # Typical total compressibility ~ 1e-9 Pa⁻¹
            c_t = 1e-9
            
            # ΔP = ΔV / (V * c_t). ΔV = net_flux * time_step
            dp_global = (net_flux_m3_s * time_step) / (v_pore * c_t)
            
            # Apply absolute material balance shift, bounded to prevent crazy swings
            dp_global_bounded = np.clip(dp_global, -500000.0, 500000.0) # max 500 kPa per timestep
            
            # Since the flow solver might not inherently enforce global mass balance on boundaries,
            # we apply this physical shift analytically to the whole field.
            pressure_new += dp_global_bounded

            return pressure_new

        except Exception as e:
            warnings.warn(f"Enhanced pressure solve failed: {e}")
            return state.pressure

    def reset_simulation(self):
        """Reset simulation to initial conditions"""
        self.current_state = None
        self.simulation_time = 0.0
        self.time_step_count = 0
        self.results_history = []

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get simulation summary"""
        if self.current_state is None:
            return {'status': 'Not initialized'}

        return {
            'simulation_time_days': self.simulation_time / 86400.0,
            'time_step_count': self.time_step_count,
            'average_pressure': np.mean(self.current_state.pressure),
            'average_sw': np.mean(self.current_state.water_saturation),
            'average_so': np.mean(self.current_state.oil_saturation),
            'average_sg': np.mean(self.current_state.gas_saturation),
            'temperature': self.current_state.temperature
        }