"""
Comprehensive Usage Examples for EOR Reservoir Simulation Engine
===============================================================

This module contains detailed examples demonstrating how to use the
EOR reservoir simulation engine for various scenarios.

Examples include:
1. Basic CO₂-EOR simulation
2. CCUS storage simulation
3. Parameter estimation with EnKF
4. Optimization interface usage
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from core.engine_simple.utils import (GridParameters, RockProperties, FluidProperties, WellParameters,
                   ReservoirState, SimulationResults, convert_units)
from core.engine_simple.reservoir_engine import ReservoirSimulationEngine
from core.unified_engine.physics.co2_properties import CO2Properties
from core.unified_engine.physics.relative_permeability import CoreyParameters, CoreyRelativePermeability
from core.engine_simple.storage_efficiency import StorageParameters, StorageEfficiencyCalculator
from core.engine_simple.parameter_estimation import EnsembleKalmanFilter, EnKFParameters, ParameterState
from core.engine_simple.optimization_interface import OptimizationInterface, EORParameters


def example_1_co2_eor_simulation():
    """
    Example 1: Basic CO₂-EOR Simulation
    ====================================

    This example demonstrates a basic CO₂ enhanced oil recovery simulation
    with injection and production wells.
    """
    print("=" * 60)
    print("Example 1: CO₂-EOR Simulation")
    print("=" * 60)

    # 1. Define grid parameters
    grid = GridParameters(
        nx=20, ny=20, nz=3,           # 20x20x3 grid
        dx=50.0, dy=50.0, dz=10.0     # 50m x 50m x 10m cells
    )
    print(f"Grid: {grid.total_cells} cells, dimensions: {grid.dimensions} m")

    # 2. Define rock properties
    # Create heterogeneous permeability field
    np.random.seed(42)
    base_perm = 100.0  # mD
    perm_field = np.random.lognormal(np.log(base_perm), 0.5, grid.total_cells)
    perm_field = perm_field.reshape((grid.nz, grid.ny, grid.nx))

    rock = RockProperties(
        porosity=0.2 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_x=perm_field,
        permeability_y=perm_field * 0.8,  # Anisotropic
        permeability_z=perm_field * 0.1   # Vertical permeability lower
    )
    print(f"Average porosity: {np.mean(rock.porosity):.3f}")
    print(f"Permeability range: {np.min(perm_field):.1f} - {np.max(perm_field):.1f} mD")

    # 3. Define fluid properties
    fluid = FluidProperties(
        oil_density_ref=850.0,      # kg/m³
        oil_viscosity_ref=0.002,    # Pa·s (2 cP)
        water_density_ref=1000.0,   # kg/m³
        water_viscosity_ref=0.001,  # Pa·s (1 cP)
        bubble_point_pressure=200e5 # 200 bar
    )
    print(f"Oil: {fluid.oil_density_ref} kg/m³, {fluid.oil_viscosity_ref*1000:.1f} cP")

    # 4. Initialize reservoir engine
    engine = ReservoirSimulationEngine(grid, rock, fluid)

    # 5. Initialize reservoir state
    initial_pressure = 250e5  # 250 bar
    initial_water_sat = 0.25  # 25% water
    temperature = 353.15      # 80°C

    state = engine.initialize_reservoir(initial_pressure, initial_water_sat, temperature)
    print(f"Initial conditions: {initial_pressure/1e5:.0f} bar, {initial_water_sat*100:.0f}% Sw")

    # 6. Define wells
    wells = []

    # CO₂ injection well (center)
    injector = WellParameters(
        well_type='injector',
        location=(10, 10, 1),      # Center of reservoir
        rate=2000.0,               # 2000 m³/day
        bottom_hole_pressure=300e5, # 300 bar BHP
        water_fraction=0.0,
        oil_fraction=0.0,
        gas_fraction=1.0           # Pure CO₂
    )
    wells.append(injector)

    # Production wells (corners)
    producer_locations = [(2, 2, 1), (18, 2, 1), (2, 18, 1), (18, 18, 1)]
    for loc in producer_locations:
        producer = WellParameters(
            well_type='producer',
            location=loc,
            rate=800.0,             # 800 m³/day total liquid
            bottom_hole_pressure=None,  # Rate control
        )
        wells.append(producer)

    print(f"Defined {len(wells)} wells: 1 injector, 4 producers")

    # 7. Update relative permeability for CO₂ system
    corey_params = CoreyParameters(
        krw0=0.3, kro0=0.8, krg0=0.5,
        nw=2.0, no=2.0, ng=1.5,
        swi=0.2, sor=0.2, sgr=0.05
    )
    engine.update_relative_permeability(**corey_params.__dict__)

    # 8. Run simulation
    print("\nRunning simulation...")
    results = engine.run_simulation(
        wells=wells,
        simulation_time=365.25 * 5 * 86400,  # 5 years in seconds
        max_time_step=86400.0,               # 1 day
        output_frequency=30                  # Monthly output
    )

    # 9. Display results
    print(f"\nSimulation Results:")
    print(f"Final recovery factor: {results.recovery_factor:.3f} ({results.recovery_factor*100:.1f}%)")
    print(f"Cumulative oil produced: {results.oil_cumulative[-1]:.0f} m³")
    print(f"Cumulative CO₂ injected: {results.co2_injection_rate[-1]*len(results.time):.0f} m³")
    print(f"Average CO₂ utilization: {np.mean(results.co2_utilization_factor):.1f} m³/m³")

    # 10. Plot results (optional)
    try:
        plt.figure(figsize=(15, 10))

        # Production rates
        plt.subplot(2, 3, 1)
        plt.plot(results.time, results.oil_rate, 'b-', label='Oil')
        plt.plot(results.time, results.water_rate, 'g-', label='Water')
        plt.xlabel('Time (days)')
        plt.ylabel('Rate (m³/day)')
        plt.title('Production Rates')
        plt.legend()
        plt.grid(True)

        # Cumulative production
        plt.subplot(2, 3, 2)
        plt.plot(results.time, results.oil_cumulative, 'b-', label='Oil')
        plt.plot(results.time, results.water_cumulative, 'g-', label='Water')
        plt.xlabel('Time (days)')
        plt.ylabel('Cumulative (m³)')
        plt.title('Cumulative Production')
        plt.legend()
        plt.grid(True)

        # CO₂ injection
        plt.subplot(2, 3, 3)
        plt.plot(results.time, results.co2_injection_rate, 'r-', label='CO₂ injection')
        plt.plot(results.time, results.co2_storage_volume, 'r--', label='CO₂ stored')
        plt.xlabel('Time (days)')
        plt.ylabel('Volume (m³)')
        plt.title('CO₂ Injection and Storage')
        plt.legend()
        plt.grid(True)

        # Utilization factor
        plt.subplot(2, 3, 4)
        plt.plot(results.time, results.co2_utilization_factor, 'orange')
        plt.xlabel('Time (days)')
        plt.ylabel('CO₂/Oil ratio (m³/m³)')
        plt.title('CO₂ Utilization Factor')
        plt.grid(True)

        # Recovery factor over time
        plt.subplot(2, 3, 5)
        ooip = results.oil_cumulative[-1] / results.recovery_factor
        rf_time = results.oil_cumulative / ooip
        plt.plot(results.time, rf_time * 100, 'purple')
        plt.xlabel('Time (days)')
        plt.ylabel('Recovery Factor (%)')
        plt.title('Recovery Factor Evolution')
        plt.grid(True)

        # Final saturation distribution
        plt.subplot(2, 3, 6)
        if results.saturation_field:
            final_sw = results.saturation_field[-1]['water'][1, :, :]  # Middle layer
            im = plt.imshow(final_sw, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Water Saturation')
            plt.title('Final Water Saturation (Middle Layer)')
            plt.xlabel('X grid blocks')
            plt.ylabel('Y grid blocks')

        plt.tight_layout()
        plt.savefig('D:/rep/4.6/co2_eor_results.png', dpi=150, bbox_inches='tight')
        print("Results plot saved as 'co2_eor_results.png'")

    except Exception as e:
        print(f"Plotting failed: {e}")

    return results


def example_2_ccus_storage_simulation():
    """
    Example 2: CCUS Storage Simulation
    ===================================

    This example demonstrates CO₂ storage with focus on storage efficiency
    and trapping mechanisms.
    """
    print("\n" + "=" * 60)
    print("Example 2: CCUS Storage Simulation")
    print("=" * 60)

    # 1. Define reservoir (larger, deeper reservoir for storage)
    grid = GridParameters(
        nx=30, ny=30, nz=5,           # 30x30x5 grid
        dx=100.0, dy=100.0, dz=20.0   # 100m x 100m x 20m cells
    )

    # 2. Rock properties (good for storage)
    rock = RockProperties(
        porosity=0.18 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_x=50.0 * np.ones((grid.nz, grid.ny, grid.nx)),  # Lower permeability
        permeability_y=40.0 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_z=5.0 * np.ones((grid.nz, grid.ny, grid.nx)),
    )

    # 3. Fluid properties (saline aquifer)
    fluid = FluidProperties(
        water_density_ref=1025.0,      # Saline water
        water_viscosity_ref=0.0012,    # Higher viscosity
        oil_density_ref=800.0,         # Light oil (residual)
        oil_viscosity_ref=0.0015,
    )

    # 4. Initialize engine
    engine = ReservoirSimulationEngine(grid, rock, fluid)

    # 5. Initialize state (saline aquifer)
    state = engine.initialize_reservoir(
        initial_pressure=300e5,  # 300 bar (deeper)
        initial_water_sat=0.95,  # 95% water (saline aquifer)
        temperature=373.15       # 100°C (geothermal)
    )

    # 6. Storage parameters
    storage_params = StorageParameters(
        area=grid.dimensions[0] * grid.dimensions[1],
        thickness=grid.dimensions[2],
        porosity=np.mean(rock.porosity),
        areal_efficiency=0.6,      # Higher areal efficiency
        vertical_efficiency=0.7,   # Good vertical sweep
        trapping_efficiency=0.3,   # Good trapping
        co2_density=650.0,         # Higher density at depth
        injection_rate=5000.0,      # Higher injection rate
        recycling_ratio=0.0         # Pure storage (no recycling)
    )
    engine.update_storage_parameters(**storage_params.__dict__)

    # 7. Define wells (storage pattern)
    wells = []

    # Multiple injection wells for better sweep
    injection_locations = [
        (7, 7, 2), (22, 7, 2), (7, 22, 2), (22, 22, 2),  # Corners
        (15, 15, 2)  # Center
    ]

    for loc in injection_locations:
        injector = WellParameters(
            well_type='injector',
            location=loc,
            rate=1000.0,              # 1000 m³/day per well
            bottom_hole_pressure=350e5, # 350 bar
            water_fraction=0.0,
            oil_fraction=0.0,
            gas_fraction=1.0           # Pure CO₂
        )
        wells.append(injector)

    # 8. Update relative permeability for storage
    corey_params = CoreyParameters(
        krw0=0.4, kro0=0.6, krg0=0.4,
        nw=3.0, no=2.0, ng=2.0,
        swi=0.05, sor=0.15, sgr=0.1   # Higher residual gas for trapping
    )
    engine.update_relative_permeability(**corey_params.__dict__)

    # 9. Run storage simulation
    print(f"Running CO2 storage simulation...")
    results = engine.run_simulation(
        wells=wells,
        simulation_time=365.25 * 10 * 86400,  # 10 years in seconds
        max_time_step=86400.0 * 7,            # 1 week time steps
        output_frequency=52                   # Weekly output
    )

    # 10. Calculate storage metrics
    storage_metrics = engine.storage_calculator.storage_metrics_summary()

    print(f"\nStorage Results:")
    print(f"Total storage efficiency: {storage_metrics['total_storage_efficiency']:.3f}")
    print(f"Storage capacity: {storage_metrics['storage_capacity_kg']/1e9:.2f} million tonnes")
    print(f"CO₂ stored: {results.co2_storage_volume[-1]:.0f} m³")
    print(f"Average injection rate: {np.mean(results.co2_injection_rate):.0f} m³/day")

    # Calculate dissolution trapping
    co2_model = CO2Properties()
    dissolution_capacity = co2_model.co2_solubility_brine(
        pressure=325e5,  # Average pressure
        temperature=373.15,
        water_volume=grid.total_cells * grid.dx * grid.dy * grid.dz * np.mean(rock.porosity)
    )
    dissolution_mass = dissolution_capacity * 44.01  # Convert to kg
    print(f"Dissolution trapping capacity: {dissolution_mass/1e6:.2f} million tonnes")

    return results


def example_3_parameter_estimation():
    """
    Example 3: Parameter Estimation with Ensemble Kalman Filter
    ==========================================================

    This example demonstrates parameter estimation using EnKF
    to history match production data.
    """
    print("\n" + "=" * 60)
    print("Example 3: Parameter Estimation with EnKF")
    print("=" * 60)

    # 1. Create synthetic "true" model
    grid = GridParameters(nx=10, ny=10, nz=2, dx=100.0, dy=100.0, dz=20.0)

    # True parameters (unknown to EnKF)
    true_perm = 150.0  # mD
    true_porosity = 0.22
    true_corey_nw = 2.5

    rock = RockProperties(
        porosity=true_porosity * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_x=true_perm * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_y=true_perm * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_z=true_perm * 0.1 * np.ones((grid.nz, grid.ny, grid.nx)),
    )

    fluid = FluidProperties()

    # 2. Generate synthetic observations
    engine_true = ReservoirSimulationEngine(grid, rock, fluid)
    state_true = engine_true.initialize_reservoir(200e5, 0.25, 353.15)

    # Simple well configuration
    wells = [
        WellParameters(well_type='injector', location=(5, 5, 1), rate=1000.0, gas_fraction=1.0),
        WellParameters(well_type='producer', location=(2, 2, 1), rate=500.0),
    ]

    results_true = engine_true.run_simulation(wells, simulation_time=365.25 * 2 * 86400, output_frequency=30)

    # Extract observations (with noise)
    obs_times = results_true.time[::3]  # Every 3rd time step
    obs_oil_rate = results_true.oil_rate[::3] + np.random.normal(0, 5, len(obs_times))
    obs_water_rate = results_true.water_rate[::3] + np.random.normal(0, 3, len(obs_times))

    print(f"Generated {len(obs_times)} synthetic observations")

    # 3. Initialize EnKF
    enkf_params = EnKFParameters(
        ensemble_size=50,
        inflation_factor=1.05,
        observation_error_variance=25.0  # Based on noise level
    )

    enkf = EnsembleKalmanFilter(enkf_params)

    # 4. Define prior distributions
    prior_mean = {
        'permeability': 100.0,    # Underestimate
        'porosity': 0.18,         # Underestimate
        'corey_nw': 2.0,          # Underestimate
        'krw0': 0.25,
        'kro0': 0.9
    }

    prior_std = {
        'permeability': 30.0,
        'porosity': 0.03,
        'corey_nw': 0.5,
        'krw0': 0.05,
        'kro0': 0.1
    }

    # 5. Initialize ensemble
    ensemble = enkf.initialize_ensemble(prior_mean, prior_std, grid.shape)
    print(f"Initialized ensemble with {len(ensemble)} members")

    # 6. Define forward model and observation operator
    def forward_model(state, time_step, control_inputs):
        """Simplified forward model for EnKF"""
        # This is a very simplified forward model
        # In practice, this would run a full reservoir simulation
        updated_state = ParameterState()

        # Simple parameter updates based on time
        if hasattr(state, 'permeability') and state.permeability is not None:
            updated_state.permeability = state.permeability
        if hasattr(state, 'corey_nw') and state.corey_nw is not None:
            updated_state.corey_nw = state.corey_nw

        return updated_state

    def observation_operator(state):
        """Map state to observation space"""
        # Simplified mapping from parameters to production rates
        obs = np.zeros(2)  # [oil_rate, water_rate]

        if hasattr(state, 'permeability') and state.permeability is not None:
            # Higher permeability -> higher production
            perm_factor = np.mean(state.permeability) / 100.0
            obs[0] = 100.0 * perm_factor  # Oil rate
            obs[1] = 50.0 * perm_factor   # Water rate

        if hasattr(state, 'corey_nw') and state.corey_nw is not None:
            # Corey exponent affects water production
            obs[1] *= (2.0 / state.corey_nw)

        return obs

    # 7. EnKF assimilation loop
    print("\nRunning EnKF assimilation...")
    assimilation_steps = len(obs_times)

    for step in range(assimilation_steps):
        # Forecast step
        ensemble = enkf.forecast_step(forward_model, 30.0, {})  # 30 days

        # Analysis step
        observations = np.array([obs_oil_rate[step], obs_water_rate[step]])
        ensemble = enkf.analysis_step(observations, observation_operator)

        if step % 3 == 0:
            stats = enkf.get_ensemble_statistics()
            print(f"  Step {step+1}/{assimilation_steps}")
            if 'permeability' in stats:
                print(f"    Permeability: {stats['permeability']['mean']:.1f} ± {stats['permeability']['std']:.1f} mD")
            if 'corey_nw' in stats:
                print(f"    Corey nw: {stats['corey_nw']['mean']:.2f} ± {stats['corey_nw']['std']:.2f}")

    # 8. Get final results
    final_stats = enkf.get_ensemble_statistics()
    best_member = enkf.get_best_member()

    print(f"\nParameter Estimation Results:")
    print(f"True permeability: {true_perm:.1f} mD")
    print(f"Estimated permeability: {final_stats.get('permeability', {}).get('mean', 'N/A'):.1f} ± {final_stats.get('permeability', {}).get('std', 'N/A'):.1f} mD")

    print(f"True porosity: {true_porosity:.3f}")
    print(f"Estimated porosity: {final_stats.get('porosity', {}).get('mean', 'N/A'):.3f} ± {final_stats.get('porosity', {}).get('std', 'N/A'):.3f}")

    print(f"True Corey nw: {true_corey_nw:.2f}")
    print(f"Estimated Corey nw: {final_stats.get('corey_nw', {}).get('mean', 'N/A'):.2f} ± {final_stats.get('corey_nw', {}).get('std', 'N/A'):.2f}")

    return enkf, final_stats


def example_4_optimization_interface():
    """
    Example 4: Optimization Interface Usage
    =======================================

    This example demonstrates how to use the optimization interface
    for evaluating EOR scenarios.
    """
    print("\n" + "=" * 60)
    print("Example 4: Optimization Interface")
    print("=" * 60)

    # 1. Setup reservoir (smaller for optimization)
    grid = GridParameters(nx=15, ny=15, nz=2, dx=50.0, dy=50.0, dz=15.0)

    rock = RockProperties(
        porosity=0.2 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_x=120.0 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_y=100.0 * np.ones((grid.nz, grid.ny, grid.nx)),
        permeability_z=10.0 * np.ones((grid.nz, grid.ny, grid.nx)),
    )

    fluid = FluidProperties()

    # 2. Initialize engine and interface
    engine = ReservoirSimulationEngine(grid, rock, fluid)
    interface = OptimizationInterface(engine)

    state = engine.initialize_reservoir(220e5, 0.3, 353.15)

    # 3. Define base EOR parameters
    base_eor_params = EORParameters(
        injection_rates=np.array([1500.0, 1200.0]),  # Two injection rates
        well_locations=[(7, 7, 1), (3, 11, 1)],      # Two injectors
        co2_injection_pressure=280e5,
        recycling_ratio=0.75,
        krw0=0.35,
        kro0=0.85,
        corey_nw=2.2,
        corey_no=1.8,
        production_constraint=1000.0
    )

    # 4. Define economic parameters
    economic_params = {
        'oil_price': 60.0,      # $/bbl
        'co2_cost': 15.0,       # $/ton
        'discount_rate': 0.1    # 10% annual
    }

    # 5. Evaluate base case
    print("Evaluating base case...")
    base_results = interface.evaluate_eor_scenario(base_eor_params, economic_params)

    print(f"Base Case Results:")
    print(f"  Recovery factor: {base_results.recovery_factor:.3f}")
    print(f"  NPV: ${base_results.npv:.1f} million")
    print(f"  Objective value: {base_results.objective_value:.3f}")
    print(f"  CO₂ utilization: {np.mean(base_results.co2_utilization_factor):.1f} m³/m³")

    # 6. Parameter sensitivity study
    print("\nRunning parameter sensitivity study...")

    # Test different injection rates
    rate_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    rate_results = []

    for factor in rate_factors:
        test_params = EORParameters(
            injection_rates=base_eor_params.injection_rates * factor,
            well_locations=base_eor_params.well_locations,
            co2_injection_pressure=base_eor_params.co2_injection_pressure,
            recycling_ratio=base_eor_params.recycling_ratio,
            krw0=base_eor_params.krw0,
            kro0=base_eor_params.kro0,
            corey_nw=base_eor_params.corey_nw,
            corey_no=base_eor_params.corey_no,
            production_constraint=base_eor_params.production_constraint
        )

        results = interface.evaluate_eor_scenario(test_params, economic_params)
        rate_results.append(results)

        print(f"  Rate factor {factor:.2f}: RF={results.recovery_factor:.3f}, NPV=${results.npv:.1f}M")

    # 7. Test different recycling ratios
    recycling_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    recycling_results = []

    for rr in recycling_ratios:
        test_params = EORParameters(
            injection_rates=base_eor_params.injection_rates,
            well_locations=base_eor_params.well_locations,
            co2_injection_pressure=base_eor_params.co2_injection_pressure,
            recycling_ratio=rr,
            krw0=base_eor_params.krw0,
            kro0=base_eor_params.kro0,
            corey_nw=base_eor_params.corey_nw,
            corey_no=base_eor_params.corey_no,
            production_constraint=base_eor_params.production_constraint
        )

        results = interface.evaluate_eor_scenario(test_params, economic_params)
        recycling_results.append(results)

        print(f"  Recycling {rr:.1f}: RF={results.recovery_factor:.3f}, Storage={results.storage_efficiency:.3f}")

    # 8. Batch evaluation of multiple scenarios
    print("\nRunning batch evaluation...")

    # Create multiple scenarios
    scenario_params = []
    for i in range(5):
        params = EORParameters(
            injection_rates=np.random.uniform(1000, 2000, 2),
            well_locations=base_eor_params.well_locations,
            co2_injection_pressure=np.random.uniform(250e5, 300e5),
            recycling_ratio=np.random.uniform(0.6, 0.9),
            krw0=np.random.uniform(0.3, 0.4),
            kro0=np.random.uniform(0.8, 0.9),
            corey_nw=np.random.uniform(1.5, 3.0),
            corey_no=np.random.uniform(1.5, 3.0),
            production_constraint=np.random.uniform(800, 1200)
        )
        scenario_params.append(params)

    batch_results = interface.batch_evaluate(scenario_params, economic_params)

    print(f"Batch evaluation completed for {len(batch_results)} scenarios")
    for i, result in enumerate(batch_results):
        if result.convergence_status == 'success':
            print(f"  Scenario {i+1}: Objective={result.objective_value:.3f}, RF={result.recovery_factor:.3f}")
        else:
            print(f"  Scenario {i+1}: Failed")

    # 9. Get parameter bounds for optimization
    bounds = interface.get_parameter_bounds()
    print(f"\nRecommended parameter bounds for optimization:")
    for param, (min_val, max_val) in bounds.items():
        print(f"  {param}: [{min_val:.1f}, {max_val:.1f}]")

    # 10. Update objective weights
    interface.set_objective_weights(
        oil_recovery=1.5,      # Emphasize recovery
        co2_storage=0.3,       # Less emphasis on storage
        economic=0.5,          # Moderate economic weight
        constraints=-5.0       # Constraint penalties
    )

    print(f"\nUpdated objective weights: {interface.objective_weights}")

    return interface, base_results


def run_all_examples():
    """Run all examples"""
    print("EOR Reservoir Simulation Engine - Examples")
    print("=" * 60)

    try:
        # Example 1: CO₂-EOR
        results1 = example_1_co2_eor_simulation()

        # Example 2: CCUS Storage
        results2 = example_2_ccus_storage_simulation()

        # Example 3: Parameter Estimation
        enkf, stats3 = example_3_parameter_estimation()

        # Example 4: Optimization Interface
        interface, results4 = example_4_optimization_interface()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()