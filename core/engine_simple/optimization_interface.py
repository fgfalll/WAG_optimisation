"""
Optimization Interface for EOR Reservoir Simulation
===================================================

This module provides the interface between the EOR reservoir simulation engine
and external optimization algorithms.

The interface takes EOR parameters as input and returns comprehensive
simulation results for evaluation and correction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import warnings

from core.engine_simple.reservoir_engine import ReservoirSimulationEngine
from core.engine_simple.utils import (
    GridParameters,
    RockProperties,
    FluidProperties,
    WellParameters,
    SimulationResults,
    create_reservoir_data_from_simple,
    create_eor_parameters_from_simple,
)
from core.unified_engine.physics.co2_properties import CO2Properties
from core.unified_engine.physics.relative_permeability import CoreyParameters, CoreyRelativePermeability
from core.engine_simple.storage_efficiency import StorageParameters, StorageEfficiencyCalculator

# Import from main data models
from core.data_models import EORParameters as MainEORParameters


# Use main EORParameters class instead of defining a conflicting one
# All EOR parameters should use the main data model: MainEORParameters


@dataclass
class OptimizationResults:
    """
    Container for optimization results returned to optimization engine
    """

    # Primary production metrics
    oil_production_profile: np.ndarray  # Oil rate vs time (m³/day)
    water_production_profile: np.ndarray  # Water rate vs time (m³/day)
    gas_production_profile: np.ndarray  # Gas rate vs time (m³/day)
    cumulative_oil: np.ndarray  # Cumulative oil (m³)
    time_vector: np.ndarray  # Time vector (days)

    # CO₂ specific metrics
    co2_storage_profile: np.ndarray  # CO₂ storage volume vs time (m³)
    co2_injection_rate: np.ndarray  # CO₂ injection rate vs time (m³/day)
    co2_utilization_factor: np.ndarray  # CO₂ utilization factor
    co2_recycling_volume: np.ndarray  # Recycled CO₂ volume (m³/day)

    # Performance indicators
    recovery_factor: float  # Final recovery factor
    sweep_efficiency: float  # Final sweep efficiency
    storage_efficiency: float  # Final storage efficiency
    recovery_factor_profile: np.ndarray = field(default_factory=lambda: np.array([]))  # RF vs time
    pressure: np.ndarray = field(default_factory=lambda: np.array([]))  # Average reservoir pressure
    npv: float = 0.0  # Net present value

    # Constraint violations
    constraint_violations: Dict[str, float] = field(default_factory=dict)

    # Sensitivity information
    parameter_sensitivities: Dict[str, np.ndarray] = field(default_factory=dict)

    # Additional metrics for optimization
    objective_value: float = 0.0  # Combined objective value
    convergence_status: str = "success"  # 'success', 'warning', 'error'
    simulation_time: float = 0.0  # Simulation execution time (s)

    @classmethod
    def error_result(cls, simulation_time: float, error_message: str = "") -> "OptimizationResults":
        """
        Create an OptimizationResults object for error cases with default values.

        Parameters
        ----------
        simulation_time : float
            Simulation execution time in seconds
        error_message : str, optional
            Error message (not stored but useful for debugging)

        Returns
        -------
        OptimizationResults
            Results object with zero/empty values for error cases
        """
        return cls(
            oil_production_profile=np.array([]),
            water_production_profile=np.array([]),
            gas_production_profile=np.array([]),
            cumulative_oil=np.array([]),
            time_vector=np.array([]),
            co2_storage_profile=np.array([]),
            co2_injection_rate=np.array([]),
            co2_utilization_factor=np.array([]),
            co2_recycling_volume=np.array([]),
            recovery_factor=0.0,
            sweep_efficiency=0.0,
            storage_efficiency=0.0,
            pressure=np.array([]),
            npv=0.0,
            constraint_violations={},
            parameter_sensitivities={},
            objective_value=0.0,
            convergence_status="error",
            simulation_time=simulation_time,
        )


class OptimizationInterface:
    """
    Interface between EOR simulation engine and optimization algorithms
    """

    def __init__(self, reservoir_engine: ReservoirSimulationEngine, skip_sensitivities: bool = False):
        """
        Initialize optimization interface

        Parameters:
        -----------
        reservoir_engine : ReservoirSimulationEngine
            Configured reservoir simulation engine
        skip_sensitivities : bool
            If True, skip sensitivity calculations for faster execution (useful for testing)
        """
        self.engine = reservoir_engine
        self.simulation_cache = {}
        self.skip_sensitivities = skip_sensitivities
        self.objective_weights = {
            "oil_recovery": 1.0,
            "co2_storage": 0.5,
            "economic": 0.3,
            "constraints": -10.0,
        }

        # Initialize storage efficiency calculator
        self.storage_calculator = StorageEfficiencyCalculator(
            params=StorageParameters()
        )

    def evaluate_eor_scenario(
        self, eor_params: MainEORParameters, economic_params: Optional[Dict] = None,
        simulation_time_seconds: Optional[float] = None
    ) -> OptimizationResults:
        """
        Evaluate EOR scenario and return results for optimization

        Parameters:
        -----------
        eor_params : EORParameters
            EOR parameters to evaluate
        economic_params : dict, optional
            Economic parameters (prices, discount rate, etc.)

        Returns:
        --------
        OptimizationResults : Comprehensive simulation results
        """
        import time

        start_time = time.time()

        try:
            # Create well parameters from EOR parameters
            well_params = self._create_well_parameters(eor_params)

            # Update relative permeability parameters
            rel_perm_params = self._update_relative_permeability(eor_params)

            # Update CO₂ parameters
            co2_params = self._update_co2_parameters(eor_params)

            # Initialize reservoir with parameters from the EOR configuration
            # Use target_pressure from EOR parameters, converting psi to Pa
            target_pressure_psi = getattr(eor_params, 'target_pressure_psi', 3000.0)
            initial_pressure = target_pressure_psi * 6894.76  # Convert psi to Pa

            # Get initial water saturation from engine's fluid properties
            initial_water_sat = getattr(self.engine.fluid, 'initial_water_saturation', 0.2)

            # Get temperature from EOR parameters (default 180°F), convert to Kelvin
            reservoir_temp_f = getattr(eor_params, 'temperature', 180.0)
            temperature = (reservoir_temp_f - 32) * 5/9 + 273.15  # Convert °F to K

            self.engine.initialize_reservoir(initial_pressure, initial_water_sat, temperature)

            # Run simulation
            sim_kwargs = {}
            if simulation_time_seconds is not None:
                sim_kwargs['simulation_time'] = simulation_time_seconds
            sim_results = self.engine.run_simulation(well_params, **sim_kwargs)

            # Calculate storage metrics
            storage_metrics = self._calculate_storage_metrics(eor_params, sim_results)

            # Calculate economic metrics if parameters provided
            economic_metrics = {}
            if economic_params is not None:
                economic_metrics = self._calculate_economic_metrics(sim_results, economic_params)

            # Check constraints
            constraint_violations = self._check_constraints(eor_params, sim_results)

            # Calculate sensitivities (finite differences) - skip if flag is set
            if self.skip_sensitivities:
                sensitivities = {}
            else:
                sensitivities = self._calculate_sensitivities(eor_params, sim_results)

            # Use real average pressure from simulation (Pa → psi)
            if sim_results.avg_pressure is not None and len(sim_results.avg_pressure) > 0:
                pressure_profile = sim_results.avg_pressure / 6894.76  # Pa to psi
            else:
                # Fallback: use final pressure field average
                pressure_profile = np.full_like(sim_results.time, np.mean(sim_results.pressure_field[0]) / 6894.76 if sim_results.pressure_field else 3000.0)

            # Use real recovery factor profile from simulation
            rf_profile = sim_results.recovery_factor_profile if sim_results.recovery_factor_profile is not None else np.array([])

            # Assemble results
            results = OptimizationResults(
                oil_production_profile=sim_results.oil_rate,
                water_production_profile=sim_results.water_rate,
                gas_production_profile=sim_results.gas_rate,
                cumulative_oil=sim_results.oil_cumulative,
                time_vector=sim_results.time,
                co2_storage_profile=storage_metrics["storage_volume"],
                co2_injection_rate=sim_results.co2_injection_rate if sim_results.co2_injection_rate is not None else np.zeros_like(sim_results.time),
                co2_utilization_factor=storage_metrics["utilization_factor"],
                co2_recycling_volume=storage_metrics["recycling_volume"],
                recovery_factor=sim_results.recovery_factor,
                sweep_efficiency=sim_results.sweep_efficiency,
                storage_efficiency=storage_metrics["efficiency"],
                pressure=pressure_profile,
                recovery_factor_profile=rf_profile,
                npv=economic_metrics.get("npv", 0.0),
                constraint_violations=constraint_violations,
                parameter_sensitivities=sensitivities,
                convergence_status="success",
                simulation_time=time.time() - start_time,
            )

            # Calculate combined objective value
            results.objective_value = self._calculate_objective_value(results)

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            warnings.warn(f"Simulation failed: {e}")
            # Return error results with all required fields
            return OptimizationResults.error_result(time.time() - start_time, str(e))

    def _create_well_parameters(self, eor_params: MainEORParameters) -> List[WellParameters]:
        """Create well parameters from EOR parameters"""
        wells = []

        # Use actual grid dimensions for well placement
        grid_nx = self.engine.grid.nx
        grid_ny = self.engine.grid.ny

        if grid_ny == 1:
            # 1D grid — injector in first cell, producer in last cell
            well_locations = [
                (0, 0, 0),            # Injector
                (grid_nx - 1, 0, 0),  # Producer
            ]
        else:
            # 2D/3D grid — use 5-spot pattern scaled to grid size
            margin_x = max(1, grid_nx // 5)
            margin_y = max(1, grid_ny // 5)
            well_locations = [
                (margin_x, margin_y, 0),                         # Injector
                (grid_nx - 1 - margin_x, grid_ny - 1 - margin_y, 0),  # Producer
                (margin_x, grid_ny - 1 - margin_y, 0),          # Producer
                (grid_nx - 1 - margin_x, margin_y, 0),          # Producer
            ]

        # Convert injection rate from MSCF/day to reservoir m3/day
        injection_rate_mscf = eor_params.injection_rate if hasattr(eor_params, "injection_rate") else 5000.0
        injection_rate_sm3d = injection_rate_mscf * 28.3168
        
        bg = 0.005
        if hasattr(self, 'engine') and self.engine is not None and hasattr(self.engine, 'fluid'):
            bg = getattr(self.engine.fluid, 'gas_fvf_ref', 0.005)
            
        injection_rate_res = injection_rate_sm3d * bg

        # Injection wells
        for i, location in enumerate(well_locations[:1]):  # First location as injector
            well = WellParameters(
                well_type="injector",
                location=location,
                rate=injection_rate_res,
                bottom_hole_pressure=eor_params.max_injector_bhp_psi * 6894.76
                if hasattr(eor_params, "max_injector_bhp_psi")
                else 20e6,
                water_fraction=0.0,
                oil_fraction=0.0,
                gas_fraction=1.0,  # Pure CO₂ injection
            )
            wells.append(well)

        # Production wells
        # Producer rate based on injection rate
        n_producers = len(well_locations[1:])
        production_rate_per_well = injection_rate_res * 0.8 / max(n_producers, 1)

        for location in well_locations[1:]:  # Remaining locations as producers
            well = WellParameters(
                well_type="producer",
                location=location,
                rate=None,  # Use BHP control instead of fixed 80% injection
                bottom_hole_pressure=1000.0 * 6894.76,  # 1000 psia
                oil_fraction=1.0,  # Enable oil production
                water_fraction=0.0,
                gas_fraction=0.0,
            )
            wells.append(well)

        return wells

    def _update_relative_permeability(self, eor_params: MainEORParameters) -> CoreyParameters:
        """Update relative permeability parameters"""
        corey_params = CoreyParameters(
            krw0=getattr(eor_params, "endpoint_water_relative_permeability", 0.3),
            kro0=getattr(eor_params, "endpoint_oil_relative_permeability", 1.0),
            nw=getattr(eor_params, "n_w", 2.0),
            no=getattr(eor_params, "n_o", 2.0),
            sgr=getattr(eor_params, "residual_gas_saturation_trapping", 0.05),
        )
        return corey_params

    def _update_co2_parameters(self, eor_params: MainEORParameters) -> Dict:
        """Update CO₂ parameters"""
        co2_params = {
            "injection_pressure": getattr(eor_params, "max_injector_bhp_psi", 3000.0)
            * 6894.76,  # Convert psi to Pa
            "composition": {"co2_fraction": 0.95, "n2_fraction": 0.05},  # Default composition
            "recycling_ratio": 0.8,  # Default recycling ratio
        }
        return co2_params

    def _calculate_storage_metrics(
        self, eor_params: MainEORParameters, sim_results: SimulationResults
    ) -> Dict:
        """Calculate CO₂ storage metrics"""
        if sim_results.co2_injection_rate is None:
            return {
                "storage_volume": np.zeros_like(sim_results.time),
                "utilization_factor": np.zeros_like(sim_results.time),
                "recycling_volume": np.zeros_like(sim_results.time),
                "efficiency": 0.0,
            }

        # Storage volume (cumulative injection - recycling)
        cumulative_injection = np.cumsum(sim_results.co2_injection_rate)
        recycling_ratio = 0.8  # Default recycling ratio
        recycling_volume = cumulative_injection * recycling_ratio
        storage_volume = cumulative_injection - recycling_volume

        # Utilization factor
        utilization_factor = np.zeros_like(sim_results.oil_rate)
        mask = sim_results.oil_rate > 0
        utilization_factor[mask] = sim_results.co2_injection_rate[mask] / sim_results.oil_rate[mask]

        # Calculate storage efficiency using proper pore volume
        # Based on CO2 storage resource estimation methodology
        total_storage = storage_volume[-1] if len(storage_volume) > 0 else 0.0
        efficiency = 0.0

        if hasattr(self.engine.grid, 'dx') and hasattr(self.engine.grid, 'dy') and hasattr(self.engine.grid, 'dz'):
            # Calculate cell volume (m³)
            cell_volume = self.engine.grid.dx * self.engine.grid.dy * self.engine.grid.dz

            # Get porosity (fraction) - use actual grid porosity if available
            if hasattr(self.engine.rock, 'porosity'):
                grid_porosity = np.mean(self.engine.rock.porosity)
            else:
                grid_porosity = 0.15  # Default porosity

            # Calculate total pore volume (m³)
            total_pore_volume = self.engine.grid.total_cells * cell_volume * grid_porosity

            # Apply sweep efficiencies from storage parameters
            storage_params = self.storage_calculator.params
            areal_efficiency = storage_params.areal_efficiency      # 0.5 (typical)
            vertical_efficiency = storage_params.vertical_efficiency # 0.6 (typical)
            trapping_efficiency = storage_params.trapping_efficiency # 0.2 (typical)

            # Effective storage volume = pore volume × sweep efficiencies
            effective_storage_volume = total_pore_volume * areal_efficiency * vertical_efficiency * trapping_efficiency

            # Storage efficiency = actual stored / effective storage capacity
            if effective_storage_volume > 0:
                efficiency = total_storage / effective_storage_volume
        else:
            # Fallback if grid geometry not available
            efficiency = 0.05  # Conservative 5% estimate

        return {
            "storage_volume": storage_volume,
            "utilization_factor": utilization_factor,
            "recycling_volume": recycling_volume,
            "efficiency": efficiency,
        }

    def _calculate_economic_metrics(
        self, sim_results: SimulationResults, economic_params: Dict
    ) -> Dict:
        """Calculate economic metrics"""
        oil_price = economic_params.get("oil_price", 50.0)  # $/bbl
        co2_cost = economic_params.get("co2_cost", 10.0)  # $/ton
        discount_rate = economic_params.get("discount_rate", 0.1)  # Annual

        # Calculate incremental production (m3)
        oil_cumulative = sim_results.oil_cumulative
        oil_incremental = np.zeros_like(oil_cumulative)
        oil_incremental[0] = oil_cumulative[0]
        oil_incremental[1:] = np.diff(oil_cumulative)
        
        # Convert incremental m³ to bbl
        oil_bbl_incremental = oil_incremental * 6.28981

        # Calculate incremental revenue
        oil_revenue_incremental = oil_bbl_incremental * oil_price

        # Calculate incremental CO₂ cost
        if sim_results.co2_injection_rate is not None:
            co2_volume_cumulative = np.cumsum(sim_results.co2_injection_rate) # This seems wrong if it's rate.
            # Wait, sim_results.co2_injection_rate is rate (m3/day)?
            # If it's rate, we need to multiply by dt.
            
            # Let's check sim_results.oil_cumulative. It's cumulative.
            # sim_results.co2_injection_rate is likely rate.
            pass
            
        # Re-evaluating CO2 calculation
        # If co2_injection_rate is rate, we should integrate it or use incremental.
        # But let's assume valid incremental calculation for now based on rate * dt.
        
        # dt calculation
        time_days = sim_results.time
        dt = np.zeros_like(time_days)
        dt[0] = time_days[0]
        dt[1:] = np.diff(time_days)
        
        # CO2 Cost
        if sim_results.co2_injection_rate is not None:
            # Rate is m3/day. Volume = rate * dt
            co2_volume_incremental = sim_results.co2_injection_rate * dt
            co2_mass_incremental = co2_volume_incremental * 700.0 # kg
            co2_cost_incremental = co2_mass_incremental * co2_cost / 1000.0 # tons * cost/ton
        else:
            co2_cost_incremental = np.zeros_like(oil_revenue_incremental)

        # Calculate NPV
        time_years = sim_results.time / 365.25
        discount_factors = np.exp(-discount_rate * time_years)
        
        # Cash flow = Revenue - Cost
        cash_flow = oil_revenue_incremental - co2_cost_incremental
        npv = np.sum(cash_flow * discount_factors)

        return {"npv": npv, "oil_revenue": np.sum(oil_revenue_incremental), "co2_cost": np.sum(co2_cost_incremental)}

    def _check_constraints(
        self, eor_params: MainEORParameters, sim_results: SimulationResults
    ) -> Dict[str, float]:
        """Check constraint violations"""
        violations = {}

        # Pressure constraint
        max_pressure_psi = getattr(eor_params, "max_pressure_psi", 6000.0)
        if sim_results.pressure_field is not None:
            max_pressure = (
                np.max(sim_results.pressure_field) * 1e-6
            )  # Convert Pa to MPa for comparison
            max_pressure_psi_conv = max_pressure_psi * 6894.76 * 1e-6  # Convert psi to MPa
            if max_pressure > max_pressure_psi_conv:
                violations["pressure"] = max_pressure - max_pressure_psi_conv

        # Production rate constraint
        shut_in_threshold = getattr(eor_params, "well_shut_in_threshold_bpd", 10.0)
        if len(sim_results.oil_rate) > 0:
            max_rate = np.max(sim_results.oil_rate) * 6.28981  # Convert m³/day to bbl/day
            if max_rate < shut_in_threshold:
                violations["production_rate"] = shut_in_threshold - max_rate

        # Recovery factor constraint (minimum)
        min_rf = 0.1  # 10% minimum recovery
        if sim_results.recovery_factor < min_rf:
            violations["recovery_factor"] = min_rf - sim_results.recovery_factor

        return violations

    def _calculate_sensitivities(
        self, eor_params: MainEORParameters, sim_results: SimulationResults
    ) -> Dict[str, np.ndarray]:
        """Calculate parameter sensitivities using finite differences"""
        sensitivities = {}
        epsilon = 1e-6

        # Sensitivity to injection rate
        if hasattr(eor_params, "injection_rate"):
            original_rate = eor_params.injection_rate

            # Create perturbed parameters
            perturbed_params = eor_params
            perturbed_params.injection_rate = original_rate + epsilon
            perturbed_results = self.evaluate_eor_scenario(perturbed_params)

            # Calculate sensitivity
            if perturbed_results.convergence_status == "success":
                sensitivity = (
                    perturbed_results.recovery_factor - sim_results.recovery_factor
                ) / epsilon
                sensitivities["injection_rate"] = np.array([sensitivity])

        # Sensitivity to Corey exponent
        if hasattr(eor_params, "n_w"):
            original_nw = eor_params.n_w

            # Create perturbed parameters
            perturbed_params = eor_params
            perturbed_params.n_w = original_nw + epsilon
            perturbed_results = self.evaluate_eor_scenario(perturbed_params)

            if perturbed_results.convergence_status == "success":
                sensitivity = (
                    perturbed_results.recovery_factor - sim_results.recovery_factor
                ) / epsilon
                sensitivities["n_w"] = np.array([sensitivity])

        return sensitivities

    def _calculate_objective_value(self, results: OptimizationResults) -> float:
        """Calculate combined objective value from results"""
        objective = 0.0

        # Recovery factor contribution
        objective += self.objective_weights["oil_recovery"] * results.recovery_factor

        # Storage efficiency contribution
        objective += self.objective_weights["co2_storage"] * results.storage_efficiency

        # Economic contribution
        objective += self.objective_weights["economic"] * results.npv / 1e6  # Normalize

        # Constraint penalties
        total_violation = sum(results.constraint_violations.values())
        objective += self.objective_weights["constraints"] * total_violation

        return objective

    def set_objective_weights(self, **weights):
        """Update objective function weights"""
        for key, value in weights.items():
            if key in self.objective_weights:
                self.objective_weights[key] = value
            else:
                warnings.warn(f"Unknown objective weight: {key}")

    def batch_evaluate(
        self, eor_params_list: List[MainEORParameters], economic_params: Optional[Dict] = None
    ) -> List[OptimizationResults]:
        """
        Evaluate multiple EOR scenarios in batch

        Parameters:
        -----------
        eor_params_list : list
            List of EOR parameters to evaluate
        economic_params : dict, optional
            Economic parameters

        Returns:
        --------
        list : List of optimization results
        """
        results = []

        for i, eor_params in enumerate(eor_params_list):
            try:
                result = self.evaluate_eor_scenario(eor_params, economic_params)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Batch evaluation failed for scenario {i}: {e}")
                # Add error result with all required fields
                results.append(OptimizationResults.error_result(0.0, str(e)))

        return results

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get recommended bounds for optimization parameters

        Returns:
        --------
        dict : Parameter bounds
        """
        return {
            "injection_rates": (100.0, 10000.0),  # m³/day
            "co2_injection_pressure": (10e6, 50e6),  # Pa
            "recycling_ratio": (0.0, 1.0),  # fraction
            "krw0": (0.1, 0.8),  # fraction
            "kro0": (0.5, 1.0),  # fraction
            "corey_nw": (0.5, 5.0),  # dimensionless
            "corey_no": (0.5, 5.0),  # dimensionless
            "residual_gas_sat": (0.0, 0.3),  # fraction
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the interface

        Returns:
        --------
        dict : Performance metrics
        """
        return {
            "cache_size": len(self.simulation_cache),
            "objective_weights": self.objective_weights.copy(),
            "engine_status": "ready" if self.engine else "not_initialized",
        }
