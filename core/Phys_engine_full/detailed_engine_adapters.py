"""
Adapter functions for converting between main data models and detailed engine parameters
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from core.data_models import ReservoirData, EORParameters, PVTProperties
from core.Phys_engine_full.enhanced_reservoir_simulator import EnhancedGrid, EnhancedFluidProperties, EnhancedBoundaryConditions

logger = logging.getLogger(__name__)


def create_enhanced_grid(reservoir_data: ReservoirData) -> EnhancedGrid:
    """
    Create EnhancedGrid from main ReservoirData

    Parameters:
    -----------
    reservoir_data : ReservoirData
        Main data model reservoir data

    Returns:
    --------
    EnhancedGrid
        Enhanced grid for detailed engine
    """
    # Extract grid information from ReservoirData
    grid = reservoir_data.grid

    # Get grid dimensions from the grid arrays
    nx = len(np.unique(grid['COORD-X']))
    ny = len(np.unique(grid['COORD-Y']))
    nz = len(np.unique(grid['COORD-Z']))

    # Get grid block sizes
    dx = np.mean(grid['DX']) if 'DX' in grid else 50.0
    dy = np.mean(grid['DY']) if 'DY' in grid else 50.0
    dz = np.mean(grid['DZ']) if 'DZ' in grid else 50.0

    # Create coordinate arrays
    x_coords = np.unique(grid['COORD-X']) if 'COORD-X' in grid else np.linspace(0, dx*nx, nx)
    y_coords = np.unique(grid['COORD-Y']) if 'COORD-Y' in grid else np.linspace(0, dy*ny, ny)
    z_coords = np.unique(grid['COORD-Z']) if 'COORD-Z' in grid else np.linspace(0, dz*nz, nz)

    # Create 3D coordinate mesh
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    coordinates = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Create permeability tensor from directional permeabilities
    perm_x = grid['PERMX'] if 'PERMX' in grid else np.full(nx*ny*nz, 100.0)
    perm_y = grid['PERMY'] if 'PERMY' in grid else perm_x  # Default to x-permeability
    perm_z = grid['PERMZ'] if 'PERMZ' in grid else perm_x * 0.1  # Default to 10% of horizontal

    permeability_tensor = np.stack([
        perm_x, np.zeros_like(perm_x), np.zeros_like(perm_x),
        np.zeros_like(perm_y), perm_y, np.zeros_like(perm_y),
        np.zeros_like(perm_z), np.zeros_like(perm_z), perm_z
    ], axis=1).reshape(-1, 3, 3)

    # Create porosity field - Flatten to 1D
    porosity_field = grid['PORO'] if 'PORO' in grid else np.full(nx*ny*nz, 0.2)
    porosity_field = np.ravel(porosity_field)

    # Create initial pressure field - Flatten to 1D
    pressure_field = np.full(nx*ny*nz, reservoir_data.initial_pressure) # Keep in psi, simulator handles units
    pressure_field = np.ravel(pressure_field)

    # Create temperature field - Flatten to 1D
    temperature_field = np.full(nx*ny*nz, reservoir_data.temperature) # Keep in F, simulator handles units
    temperature_field = np.ravel(temperature_field)

    return EnhancedGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        coordinates=coordinates,
        permeability_tensor=permeability_tensor,
        porosity_field=porosity_field,
        pressure_field=pressure_field,
        temperature_field=temperature_field
    )


def create_enhanced_fluid_properties(pvt_data: PVTProperties, grid: EnhancedGrid) -> EnhancedFluidProperties:
    """
    Create EnhancedFluidProperties from main PVTProperties

    Parameters:
    -----------
    pvt_data : PVTProperties
        Main data model PVT data
    grid : EnhancedGrid
        Enhanced grid for determining cell count

    Returns:
    --------
    EnhancedFluidProperties
        Enhanced fluid properties for detailed engine
    """
    # Define components for CO2-EOR system
    components = ['CO2', 'C1', 'C2', 'C3-C4', 'C5+', 'H2O', 'N2']

    # Molecular weights (g/mol)
    molecular_weights = np.array([44.01, 16.04, 30.07, 44.10, 72.15, 18.02, 28.02])

    # Critical properties for Peng-Robinson EOS
    critical_properties = {
        'CO2': {'Tc': 304.13, 'Pc': 7.376e6, 'omega': 0.225},  # K, Pa, acentric factor
        'C1': {'Tc': 190.56, 'Pc': 4.599e6, 'omega': 0.011},
        'C2': {'Tc': 305.32, 'Pc': 4.872e6, 'omega': 0.099},
        'C3-C4': {'Tc': 369.83, 'Pc': 4.256e6, 'omega': 0.152},
        'C5+': {'Tc': 547.92, 'Pc': 2.748e6, 'omega': 0.350},
        'H2O': {'Tc': 647.10, 'Pc': 2.209e7, 'omega': 0.344},
        'N2': {'Tc': 126.20, 'Pc': 3.398e6, 'omega': 0.040}
    }

    # Initialize phase properties (will be populated by EOS)
    phase_viscosities = {}
    phase_densities = {}
    component_fractions = {}

    # Calculate n_cells from grid
    n_cells = grid.nx * grid.ny * grid.nz
    n_cells = grid.nx * grid.ny * grid.nz

    # Typical oil reservoir fluid composition (mole fractions)
    # Based on average compositional data for CO2-EOR candidate reservoirs
    # Composition varies by cell based on depth and initial conditions
    base_composition = {
        'CO2': 0.02,     # 2% CO2 (typically dissolved)
        'C1': 0.35,      # 35% Methane (light component)
        'C2': 0.10,      # 10% Ethane
        'C3-C4': 0.18,   # 18% Propane/Butane
        'C5+': 0.30,     # 30% C5+ (heavier fraction)
        'H2O': 0.03,     # 3% Water (dissolved)
        'N2': 0.02       # 2% Nitrogen/trace
    }

    # Validate composition sums to 1.0
    total = sum(base_composition.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Component composition does not sum to 1.0 (sum={total:.3f}). Normalizing.")
        base_composition = {k: v/total for k, v in base_composition.items()}

    # Create spatially varying composition based on depth
    # Deeper cells tend to have more heavier components
    for comp in components:
        if comp in base_composition:
            base_val = base_composition[comp]

            # Add depth-dependent variation (±10% variation)
            if hasattr(grid, 'depth') and len(grid.depth) == n_cells:
                depth_normalized = (grid.depth - np.min(grid.depth)) / (np.max(grid.depth) - np.min(grid.depth) + 1e-6)

                if comp in ['C5+', 'C3-C4']:
                    # Heavier components increase with depth
                    variation = 0.1 * (depth_normalized - 0.5)
                elif comp in ['C1', 'CO2']:
                    # Lighter components decrease with depth
                    variation = -0.1 * (depth_normalized - 0.5)
                else:
                    variation = 0.05 * np.sin(2 * np.pi * depth_normalized)

                component_fractions[comp] = np.clip(
                    np.full(n_cells, base_val) * (1 + variation),
                    0.001,  # Minimum mole fraction
                    0.95    # Maximum mole fraction
                )
            else:
                # Uniform composition if no depth data
                component_fractions[comp] = np.full(n_cells, base_val)
        else:
            # Unknown component - set to small value
            component_fractions[comp] = np.full(n_cells, 0.001)
            logger.warning(f"Unknown component {comp}, setting to minimal fraction")

    # Ensure mass conservation (sum of fractions = 1.0 for each cell)
    comp_sum = np.zeros(n_cells)
    for comp in components:
        comp_sum += component_fractions[comp]

    for comp in components:
        component_fractions[comp] = component_fractions[comp] / (comp_sum + 1e-9)

    logger.info(f"Initialized compositional model with {n_cells} cells and {len(components)} components")

    return EnhancedFluidProperties(
        components=components,
        molecular_weights=molecular_weights,
        critical_properties=critical_properties,
        phase_viscosities=phase_viscosities,
        phase_densities=phase_densities,
        component_fractions=component_fractions
    )


def create_enhanced_boundary_conditions(eor_params: EORParameters,
                                      grid: EnhancedGrid) -> EnhancedBoundaryConditions:
    """
    Create EnhancedBoundaryConditions from main EORParameters

    Parameters:
    -----------
    eor_params : EORParameters
        Main data model EOR parameters
    grid : EnhancedGrid
        Enhanced grid for well placement

    Returns:
    --------
    EnhancedBoundaryConditions
        Enhanced boundary conditions for detailed engine
    """
    n_cells = grid.nx * grid.ny * grid.nz
    n_steps = int(365 * 20)  # 20 years with daily steps

    # Helper function to convert (i, j, k) location to linear cell index
    def location_to_cell_index(location: tuple, grid: EnhancedGrid) -> int:
        """Convert (i, j, k) grid location to linear cell index."""
        i, j, k = location
        # Ensure indices are within bounds
        i = max(0, min(i, grid.nx - 1))
        j = max(0, min(j, grid.ny - 1))
        k = max(0, min(k, grid.nz - 1))
        return k * grid.nx * grid.ny + j * grid.nx + i

    # Create injection wells (simple 5-spot pattern)
    injection_wells = [
        {
            'name': 'INJ-1',
            'location': (grid.nx//4, grid.ny//4, grid.nz//2),
            'cell_index': location_to_cell_index((grid.nx//4, grid.ny//4, grid.nz//2), grid),
            'type': 'CO2_injection',
            'radius': 0.1,
            'skin': 0.0
        }
    ]

    # Create production wells
    production_wells = [
        {
            'name': 'PROD-1',
            'location': (3*grid.nx//4, grid.ny//4, grid.nz//2),
            'cell_index': location_to_cell_index((3*grid.nx//4, grid.ny//4, grid.nz//2), grid),
            'type': 'producer',
            'radius': 0.1,
            'skin': 0.0
        },
        {
            'name': 'PROD-2',
            'location': (grid.nx//4, 3*grid.ny//4, grid.nz//2),
            'cell_index': location_to_cell_index((grid.nx//4, 3*grid.ny//4, grid.nz//2), grid),
            'type': 'producer',
            'radius': 0.1,
            'skin': 0.0
        },
        {
            'name': 'PROD-3',
            'location': (3*grid.nx//4, 3*grid.ny//4, grid.nz//2),
            'cell_index': location_to_cell_index((3*grid.nx//4, 3*grid.ny//4, grid.nz//2), grid),
            'type': 'producer',
            'radius': 0.1,
            'skin': 0.0
        }
    ]

    # Create injection rate schedule (convert MSCFD to m³/day)
    injection_rate_m3_day = eor_params.injection_rate * 28.3168  # MSCF to m³
    injection_rates = {
        'CO2': np.full(n_steps, injection_rate_m3_day)
    }

    # Create injection composition (pure CO2)
    injection_compositions = {
        'CO2': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Pure CO2
    }

    # Create production rates (rate-controlled producers)
    # IMPORTANT: Production rates must be NEGATIVE for material balance
    # (consistent with Simple Engine fix)
    base_prod_rate = -injection_rate_m3_day * 0.8  # 80% of injection, NEGATIVE for production
    production_rates = {
        'PROD-1': np.full(n_steps, base_prod_rate / 3),
        'PROD-2': np.full(n_steps, base_prod_rate / 3),
        'PROD-3': np.full(n_steps, base_prod_rate / 3)
    }

    # Set bottom hole pressures (optional, for BHP control)
    bottom_hole_pressures = np.full(n_cells, eor_params.wellbore_pressure * 6894.76)  # psi to Pa

    return EnhancedBoundaryConditions(
        injection_wells=injection_wells,
        injection_rates=injection_rates,
        injection_compositions=injection_compositions,
        production_wells=production_wells,
        production_rates=production_rates,
        bottom_hole_pressures=bottom_hole_pressures,
        boundary_type="no_flow",
        boundary_pressure=eor_params.max_pressure_psi * 6894.76,  # psi to Pa
        boundary_temperature=(150.0 - 32) * 5/9 + 273.15  # Default 150F to K
    )


def create_detailed_engine_parameters(reservoir_data: ReservoirData,
                                    eor_params: EORParameters,
                                    pvt_data: PVTProperties) -> tuple:
    """
    Create all necessary parameters for the detailed engine from main data models

    Parameters:
    -----------
    reservoir_data : ReservoirData
        Main data model reservoir data
    eor_params : EORParameters
        Main data model EOR parameters
    pvt_data : PVTProperties
        Main data model PVT data

    Returns:
    --------
    tuple
        (EnhancedGrid, EnhancedFluidProperties, PVTProperties, EORParameters, EnhancedBoundaryConditions)
    """
    enhanced_grid = create_enhanced_grid(reservoir_data)
    enhanced_fluid_props = create_enhanced_fluid_properties(pvt_data, enhanced_grid)
    enhanced_bc = create_enhanced_boundary_conditions(eor_params, enhanced_grid)

    return enhanced_grid, enhanced_fluid_props, pvt_data, eor_params, enhanced_bc