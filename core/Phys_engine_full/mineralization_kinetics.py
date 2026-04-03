"""
CO₂ Dissolution and Mineralization Kinetics Module
Implements mass transfer and reaction kinetics for CO₂ trapping in brine and mineral phases
Based on Xu, Apps & Pruess (2004) and Gunter et al. (1997) models
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.constants as const

logger = logging.getLogger(__name__)

class MineralType(Enum):
    """Types of carbonate minerals for CO₂ mineralization"""
    CALCITE = "calcite"      # CaCO₃
    DOLOMITE = "dolomite"    # CaMg(CO₃)₂
    SIDERITE = "siderite"    # FeCO₃
    MAGNESITE = "magnesite"  # MgCO₃

class ReactionState(Enum):
    """State of mineralization reactions"""
    DISSOLUTION_DOMINANT = "dissolution_dominant"
    PRECIPITATION_DOMINANT = "precipitation_dominant"
    EQUILIBRIUM = "equilibrium"
    KINETIC_LIMITED = "kinetic_limited"

@dataclass
class BrineProperties:
    """Properties of formation brine"""
    salinity: float  # g/L or ppm
    pH: float
    temperature: float  # °C
    pressure: float  # Pa
    ionic_strength: float = 0.0
    
    def calculate_ionic_strength(self) -> float:
        """Calculate ionic strength from salinity (simplified)"""
        # Simplified conversion: salinity (g/L) to ionic strength (mol/L)
        return self.salinity / 58440.0  # Approximate conversion for NaCl brine
    
    def calculate_ph_effect(self, co2_concentration: float) -> float:
        """Calculate pH change due to CO₂ dissolution"""
        # Simplified pH calculation: CO₂ dissolution lowers pH
        base_ph = self.pH
        co2_effect = -0.5 * np.log10(co2_concentration / 1e-5)  # Simplified
        return max(2.0, base_ph + co2_effect)  # Clamp to reasonable range

@dataclass
class MineralProperties:
    """Properties of carbonate minerals"""
    mineral_type: MineralType
    molar_mass: float  # g/mol
    density: float  # kg/m³
    solubility_product: float  # K_sp at 25°C
    reaction_rate_constant: float  # mol/m²/s
    activation_energy: float  # J/mol
    reactive_surface_area: float  # m²/kg
    
    def calculate_solubility(self, temperature: float, pressure: float) -> float:
        """Calculate temperature-dependent solubility product"""
        # Van't Hoff equation: ln(K) = -ΔH/RT + ΔS/R
        # Simplified temperature dependence
        T_ref = 298.15  # K
        T_kelvin = temperature + 273.15
        
        # Typical values for carbonate minerals
        if self.mineral_type == MineralType.CALCITE:
            delta_H = -12000  # J/mol (exothermic)
        elif self.mineral_type == MineralType.DOLOMITE:
            delta_H = -25000  # J/mol
        else:  # SIDERITE, MAGNESITE
            delta_H = -15000  # J/mol
        
        ln_K_ratio = (delta_H / const.R) * (1/T_kelvin - 1/T_ref)
        K_sp = self.solubility_product * np.exp(ln_K_ratio)
        
        return K_sp

@dataclass
class KineticsParameters:
    """Parameters for dissolution and mineralization kinetics"""
    # CO₂ dissolution parameters
    mass_transfer_coefficient: float  # m/s - for CO₂ dissolution into brine
    henrys_constant: float  # Pa·m³/mol - CO₂ solubility
    diffusion_coefficient: float  # m²/s - CO₂ in brine
    
    # Mineralization parameters
    minerals: Dict[MineralType, MineralProperties]
    
    # Geochemical parameters
    initial_mineral_volume_fraction: float = 0.05  # 5% initial mineral content
    maximum_mineral_precipitation: float = 0.15  # 15% maximum mineral fill
    reaction_surface_area_factor: float = 100.0  # m²/m³
    
    # Temperature dependence
    arrhenius_prefactor: float = 1.0e6  # 1/s
    activation_energy: float = 50000.0  # J/mol

@dataclass
class MineralizationState:
    """Current state of dissolution and mineralization"""
    dissolved_co2_concentration: np.ndarray  # mol/m³ - [n_cells]
    mineral_precipitate: Dict[MineralType, np.ndarray]  # kg/m³ - for each mineral
    porosity_change: np.ndarray  # Δφ due to mineralization
    reaction_rates: Dict[str, np.ndarray]  # mol/m³/s - various reaction rates
    saturation_indices: Dict[MineralType, np.ndarray]  # SI = log10(IAP/K_sp)
    
    def total_mineralized_co2(self) -> np.ndarray:
        """Calculate total CO₂ mineralized (kg/m³)"""
        total = np.zeros_like(self.dissolved_co2_concentration)
        for mineral_type, precipitate in self.mineral_precipitate.items():
            # Convert mineral mass to equivalent CO₂ mass
            mineral_props = self._get_mineral_properties(mineral_type)
            co2_fraction = 44.01 / mineral_props.molar_mass  # CO₂ mass fraction in mineral
            total += precipitate * co2_fraction
        return total
    
    def _get_mineral_properties(self, mineral_type: MineralType) -> MineralProperties:
        """Get properties for a mineral type from a predefined dictionary."""
        props = {
            MineralType.CALCITE: MineralProperties(
                mineral_type=MineralType.CALCITE,
                molar_mass=100.09,  # g/mol
                density=2710.0,     # kg/m³
                solubility_product=3.36e-9,  # at 25°C
                reaction_rate_constant=1.0e-9,  # mol/m²/s
                activation_energy=50000.0,  # J/mol
                reactive_surface_area=0.1   # m²/kg
            ),
            MineralType.DOLOMITE: MineralProperties(
                mineral_type=MineralType.DOLOMITE,
                molar_mass=184.40,
                density=2840.0,
                solubility_product=1.0e-17,
                reaction_rate_constant=1.0e-10,
                activation_energy=60000.0,
                reactive_surface_area=0.05
            ),
            MineralType.SIDERITE: MineralProperties(
                mineral_type=MineralType.SIDERITE,
                molar_mass=115.85,
                density=3960.0,
                solubility_product=10**-10.7,
                reaction_rate_constant=1.0e-11,
                activation_energy=55000.0,
                reactive_surface_area=0.08
            ),
            MineralType.MAGNESITE: MineralProperties(
                mineral_type=MineralType.MAGNESITE,
                molar_mass=84.31,
                density=2980.0,
                solubility_product=10**-8.0,
                reaction_rate_constant=1.0e-12,
                activation_energy=75000.0,
                reactive_surface_area=0.03
            )
        }
        return props.get(mineral_type)

class CO2DissolutionKinetics:
    """
    Implements CO₂ dissolution kinetics into brine
    dC/dt = k_diss * A_s * (C* - C)
    """
    
    def __init__(self, parameters: KineticsParameters):
        self.params = parameters
    
    def calculate_equilibrium_concentration(self, pressure: float, temperature: float,
                                         brine: BrineProperties) -> float:
        """Calculate equilibrium CO₂ concentration using Henry's law"""
        # Henry's law: C* = P * K_H
        # Corrected for temperature and salinity
        T_kelvin = temperature + 273.15
        
        # Temperature correction (simplified)
        henry_temp_correction = np.exp(2400 * (1/T_kelvin - 1/298.15))
        
        # Salinity correction (Setschenow equation)
        salinity_correction = np.exp(-0.115 * brine.salinity / 1000.0)
        
        C_star = (pressure * self.params.henrys_constant * 
                 henry_temp_correction * salinity_correction)
        
        return max(0.0, C_star)
    
    def calculate_dissolution_rate(self, current_concentration: np.ndarray,
                                 equilibrium_concentration: float,
                                 surface_area: np.ndarray, dt: float) -> np.ndarray:
        """Calculate CO₂ dissolution rate"""
        # Mass transfer: dC/dt = k * A * (C* - C)
        concentration_difference = equilibrium_concentration - current_concentration
        
        # Only dissolve if undersaturated
        dissolution_mask = concentration_difference > 0
        dissolution_rate = np.zeros_like(current_concentration)
        
        dissolution_rate[dissolution_mask] = (
            self.params.mass_transfer_coefficient *
            surface_area[dissolution_mask] *
            concentration_difference
        )
        
        return dissolution_rate
    
    def update_dissolved_co2(self, current_state: MineralizationState,
                           pressure: np.ndarray, temperature: np.ndarray,
                           brine: BrineProperties, surface_area: np.ndarray,
                           dt: float) -> MineralizationState:
        """Update dissolved CO₂ concentration due to dissolution"""
        n_cells = len(current_state.dissolved_co2_concentration)
        new_concentration = current_state.dissolved_co2_concentration.copy()
        
        for cell in range(n_cells):
            # Calculate equilibrium concentration for this cell
            C_star = self.calculate_equilibrium_concentration(
                pressure[cell], temperature[cell], brine
            )
            
            # Calculate dissolution rate
            dissolution_rate = self.calculate_dissolution_rate(
                np.array([current_state.dissolved_co2_concentration[cell]]),
                C_star, np.array([surface_area[cell]]), dt
            )[0]
            
            # Update concentration
            new_concentration[cell] += dissolution_rate * dt
            
            # Clamp to equilibrium
            new_concentration[cell] = min(new_concentration[cell], C_star * 1.1)
        
        # Update state
        new_state = MineralizationState(
            dissolved_co2_concentration=new_concentration,
            mineral_precipitate=current_state.mineral_precipitate.copy(),
            porosity_change=current_state.porosity_change.copy(),
            reaction_rates=current_state.reaction_rates.copy(),
            saturation_indices=current_state.saturation_indices.copy()
        )
        
        return new_state

class MineralizationKinetics:
    """
    Implements carbonate mineralization kinetics
    dM/dt = k_min * A_s * (1 - SI) for precipitation
    Based on transition state theory
    """
    
    def __init__(self, parameters: KineticsParameters):
        self.params = parameters
    
    def calculate_saturation_index(self, dissolved_co2: float, calcium_concentration: float,
                                 temperature: float, mineral: MineralProperties) -> float:
        """Calculate saturation index SI = log10(IAP/K_sp)"""
        # Ion activity product calculation (simplified)
        if mineral.mineral_type == MineralType.CALCITE:
            # Calcite: CaCO₃ → Ca²⁺ + CO₃²⁻
            # IAP = [Ca²⁺][CO₃²⁻]
            carbonate_concentration = self.calculate_carbonate_concentration(
                dissolved_co2, temperature
            )
            iap = calcium_concentration * carbonate_concentration
        elif mineral.mineral_type == MineralType.DOLOMITE:
            # Dolomite: CaMg(CO₃)₂ → Ca²⁺ + Mg²⁺ + 2CO₃²⁻
            magnesium_concentration = 0.01  # Simplified
            carbonate_concentration = self.calculate_carbonate_concentration(
                dissolved_co2, temperature
            )
            iap = calcium_concentration * magnesium_concentration * carbonate_concentration ** 2
        else:
            # Other carbonates - simplified
            carbonate_concentration = self.calculate_carbonate_concentration(
                dissolved_co2, temperature
            )
            iap = carbonate_concentration  # Simplified
        
        K_sp = mineral.calculate_solubility(temperature, 1e6)  # 1 MPa pressure
        # Handle edge cases to avoid numerical warnings
        if K_sp <= 0 or iap <= 0:
            return 0.0
        si = np.log10(iap / K_sp)
        
        return si
    
    def calculate_carbonate_concentration(self, dissolved_co2: float, temperature: float) -> float:
        """Calculate carbonate ion concentration from dissolved CO₂"""
        # Simplified carbonate system
        # CO₂(aq) + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻ ⇌ 2H⁺ + CO₃²⁻
        
        # Equilibrium constants (simplified)
        K1 = 4.45e-7  # First dissociation constant
        K2 = 4.69e-11  # Second dissociation constant
        
        # Assume pH around 6-7 for CO₂-rich systems
        pH = 6.5
        H_plus = 10 ** (-pH)
        
        # Calculate carbonate concentration
        carbonate = (K1 * K2 * dissolved_co2) / (H_plus ** 2)
        
        return max(0.0, carbonate)
    
    def calculate_mineralization_rate(self, saturation_index: float, temperature: float,
                                   mineral: MineralProperties, surface_area: float) -> float:
        """Calculate mineralization rate using transition state theory"""
        T_kelvin = temperature + 273.15
        
        if saturation_index > 0:
            # Precipitation regime
            rate_constant = mineral.reaction_rate_constant
            activation_energy = mineral.activation_energy
            
            # Temperature correction (Arrhenius)
            arrhenius_factor = np.exp(-activation_energy / (const.R * T_kelvin))
            
            # Rate = k * A * (1 - exp(-ΔG/RT)) ≈ k * A * (SI) for SI > 0
            # Using simplified linear dependence near equilibrium
            rate = rate_constant * surface_area * saturation_index * arrhenius_factor
            
        else:
            # Dissolution regime (mineral dissolving)
            rate_constant = mineral.reaction_rate_constant * 10  # Faster dissolution
            rate = rate_constant * surface_area * saturation_index  # Negative rate
        
        return rate
    
    def update_mineral_precipitation(self, current_state: MineralizationState,
                                   temperature: np.ndarray, calcium_concentration: np.ndarray,
                                   surface_area: np.ndarray, porosity: np.ndarray,
                                   dt: float) -> MineralizationState:
        """Update mineral precipitation for all mineral types"""
        n_cells = len(temperature)
        new_mineral_precipitate = current_state.mineral_precipitate.copy()
        new_porosity_change = current_state.porosity_change.copy()
        new_saturation_indices = current_state.saturation_indices.copy()
        new_reaction_rates = current_state.reaction_rates.copy()
        
        for mineral_type, mineral_props in self.params.minerals.items():
            mineral_precipitate = new_mineral_precipitate.get(mineral_type, np.zeros(n_cells))
            saturation_indices = np.zeros(n_cells)
            reaction_rates = np.zeros(n_cells)
            
            for cell in range(n_cells):
                # Calculate saturation index
                si = self.calculate_saturation_index(
                    current_state.dissolved_co2_concentration[cell],
                    calcium_concentration[cell],
                    temperature[cell],
                    mineral_props
                )
                saturation_indices[cell] = si
                
                # Calculate reaction rate
                reaction_rate = self.calculate_mineralization_rate(
                    si, temperature[cell], mineral_props, surface_area[cell]
                )
                reaction_rates[cell] = reaction_rate
                
                # Update mineral precipitation (convert mol/m³/s to kg/m³)
                molar_precipitation = reaction_rate * dt  # mol/m³
                mass_precipitation = molar_precipitation * mineral_props.molar_mass / 1000  # kg/m³
                
                # Only allow precipitation if space available (porosity constraint)
                max_precipitation = porosity[cell] * mineral_props.density / 1000  # kg/m³
                current_mineral = mineral_precipitate[cell]
                
                if mass_precipitation > 0:  # Precipitation
                    new_mineral = current_mineral + mass_precipitation
                    mineral_precipitate[cell] = min(new_mineral, max_precipitation)
                    
                    # Update porosity change (volume of mineral precipitated)
                    volume_change = mass_precipitation / mineral_props.density * 1000  # m³ mineral/m³ bulk
                    new_porosity_change[cell] -= volume_change
                
                elif mass_precipitation < 0:  # Dissolution
                    new_mineral = current_mineral + mass_precipitation
                    mineral_precipitate[cell] = max(new_mineral, 0.0)  # Can't go negative
                    
                    # Update porosity change (volume of mineral dissolved)
                    volume_change = mass_precipitation / mineral_props.density * 1000  # m³ mineral/m³ bulk
                    new_porosity_change[cell] -= volume_change  # Positive volume change for dissolution
            
            # Store updated values
            new_mineral_precipitate[mineral_type] = mineral_precipitate
            new_saturation_indices[mineral_type] = saturation_indices
            new_reaction_rates[mineral_type] = reaction_rates
        
        return MineralizationState(
            mineral_precipitate=new_mineral_precipitate,
            porosity_change=new_porosity_change,
            saturation_indices=new_saturation_indices,
            reaction_rates=new_reaction_rates,
            dissolved_co2_concentration=current_state.dissolved_co2_concentration.copy()
        )
    
    def update_dissolution_mineralization(self, state, dt: float):
        """
        Update dissolution and mineralization processes
        This is a simplified wrapper that integrates with CCUSState
        """
        # Extract relevant fields from state
        n_cells = len(state.pressure)
        temperature = np.full(n_cells, 60.0)  # Simplified - would come from state
        calcium_concentration = np.full(n_cells, 0.01)  # mol/L - simplified
        surface_area = np.full(n_cells, 100.0)  # m²/m³ - simplified
        
        # Create mineralization state
        mineralization_state = MineralizationState(
            mineral_precipitate={'calcite': state.mineral_precipitate},
            porosity_change=np.zeros(n_cells),
            saturation_indices={'calcite': np.zeros(n_cells)},
            reaction_rates={'calcite': np.zeros(n_cells)},
            dissolved_co2_concentration=state.dissolved_co2
        )
        
        # Update mineralization
        updated_mineralization = self.update_mineral_precipitation(
            mineralization_state, temperature, calcium_concentration,
            surface_area, state.porosity, dt
        )
        
        # Update state with new values
        state.mineral_precipitate = updated_mineralization.mineral_precipitate['calcite']
        
        # Update porosity from mineralization effects
        state.porosity += updated_mineralization.porosity_change
        
        return state
class MineralizationEngine:
    """
    Main engine for CO₂ dissolution and mineralization kinetics
    Combines dissolution and precipitation processes
    """
    
    def __init__(self, grid: Any, parameters: KineticsParameters, brine: BrineProperties):
        self.grid = grid
        self.params = parameters
        self.brine = brine
        
        # Initialize kinetics modules
        self.dissolution_kinetics = CO2DissolutionKinetics(parameters)
        self.mineralization_kinetics = MineralizationKinetics(parameters)
        
        # Initial state
        self.current_state = self._initialize_state()
    
    def _initialize_state(self) -> MineralizationState:
        """Initialize mineralization state"""
        n_cells = len(self.grid.cell_volumes)
        
        # Initial mineral precipitation (zero)
        mineral_precipitate = {}
        for mineral_type in self.params.minerals.keys():
            mineral_precipitate[mineral_type] = np.zeros(n_cells)
        
        # Initial saturation indices (zero)
        saturation_indices = {}
        for mineral_type in self.params.minerals.keys():
            saturation_indices[mineral_type] = np.zeros(n_cells)
        
        return MineralizationState(
            dissolved_co2_concentration=np.zeros(n_cells),
            mineral_precipitate=mineral_precipitate,
            porosity_change=np.zeros(n_cells),
            reaction_rates={},
            saturation_indices=saturation_indices
        )
    
    def calculate_reactive_surface_area(self, porosity: np.ndarray, 
                                      mineral_content: np.ndarray) -> np.ndarray:
        """Calculate reactive surface area for mineralization"""
        # Simplified surface area model
        # A_s = A0 * (mineral_content)^(2/3) * porosity
        base_area = self.params.reaction_surface_area_factor  # m²/m³
        surface_area = base_area * (mineral_content ** (2/3)) * porosity
        
        return np.maximum(surface_area, 1e-6)  # Avoid zero surface area
    
    def update_kinetics(self, pressure: np.ndarray, temperature: np.ndarray,
                       gas_saturation: np.ndarray, porosity: np.ndarray,
                       calcium_concentration: np.ndarray, dt: float) -> MineralizationState:
        """
        Update dissolution and mineralization kinetics for one timestep
        """
        logger.info("Updating CO₂ dissolution and mineralization kinetics")
        
        n_cells = len(pressure)
        
        # Calculate reactive surface area
        # Use average mineral content for surface area calculation
        avg_mineral_content = np.zeros(n_cells)
        for mineral_precipitate in self.current_state.mineral_precipitate.values():
            avg_mineral_content += mineral_precipitate
        avg_mineral_content /= max(len(self.current_state.mineral_precipitate), 1)
        
        surface_area = self.calculate_reactive_surface_area(porosity, avg_mineral_content)
        
        # Step 1: CO₂ dissolution into brine
        state_after_dissolution = self.dissolution_kinetics.update_dissolved_co2(
            self.current_state, pressure, temperature, self.brine, surface_area, dt
        )
        
        # Step 2: Mineral precipitation/dissolution
        state_after_mineralization = self.mineralization_kinetics.update_mineral_precipitation(
            state_after_dissolution, temperature, calcium_concentration, 
            surface_area, porosity, dt
        )
        
        # Update current state
        self.current_state = state_after_mineralization
        
        return self.current_state
    
    def calculate_trapping_efficiency(self, injected_co2_mass: float) -> Dict[str, float]:
        """Calculate CO₂ trapping efficiency through different mechanisms"""
        total_dissolved = np.sum(self.current_state.dissolved_co2_concentration * 
                               self.grid.cell_volumes) * 0.044  # mol to kg CO₂
        
        total_mineralized = 0.0
        for mineral_type, precipitate in self.current_state.mineral_precipitate.items():
            mineral_co2 = self.current_state.total_mineralized_co2()
            total_mineralized += np.sum(mineral_co2 * self.grid.cell_volumes)
        
        dissolution_efficiency = total_dissolved / injected_co2_mass if injected_co2_mass > 0 else 0.0
        mineralization_efficiency = total_mineralized / injected_co2_mass if injected_co2_mass > 0 else 0.0
        
        return {
            'dissolution_efficiency': dissolution_efficiency,
            'mineralization_efficiency': mineralization_efficiency,
            'total_dissolved_kg': total_dissolved,
            'total_mineralized_kg': total_mineralized,
            'combined_trapping_efficiency': dissolution_efficiency + mineralization_efficiency
        }
    
    def get_reaction_summary(self) -> Dict[str, Any]:
        """Get summary of reaction states"""
        n_cells = len(self.grid.cell_volumes)
        
        # Calculate average reaction rates
        avg_rates = {}
        for rate_name, rates in self.current_state.reaction_rates.items():
            avg_rates[rate_name] = np.mean(rates) if len(rates) > 0 else 0.0
        
        # Calculate mineral distribution
        mineral_distribution = {}
        for mineral_type, precipitate in self.current_state.mineral_precipitate.items():
            total_precipitate = np.sum(precipitate * self.grid.cell_volumes)
            mineral_distribution[mineral_type.value] = total_precipitate
        
        return {
            'average_dissolved_co2_mol_m3': np.mean(self.current_state.dissolved_co2_concentration),
            'total_dissolved_co2_mol': np.sum(self.current_state.dissolved_co2_concentration * self.grid.cell_volumes),
            'average_porosity_change': np.mean(self.current_state.porosity_change),
            'reaction_rates': avg_rates,
            'mineral_distribution_kg': mineral_distribution,
            'cells_with_precipitation': np.sum([np.any(precipitate > 0) for precipitate in self.current_state.mineral_precipitate.values()])
        }

# Utility functions for common scenarios
def create_typical_kinetics_parameters() -> KineticsParameters:
    """Create typical kinetics parameters for CO₂ storage"""
    minerals = {
        MineralType.CALCITE: MineralProperties(
            mineral_type=MineralType.CALCITE,
            molar_mass=100.09,  # g/mol
            density=2710.0,     # kg/m³
            solubility_product=3.36e-9,  # at 25°C
            reaction_rate_constant=1.0e-9,  # mol/m²/s
            activation_energy=50000.0,  # J/mol
            reactive_surface_area=0.1   # m²/kg
        ),
        MineralType.DOLOMITE: MineralProperties(
            mineral_type=MineralType.DOLOMITE,
            molar_mass=184.40,
            density=2840.0,
            solubility_product=10.0e-17,
            reaction_rate_constant=1.0e-10,
            activation_energy=60000.0,
            reactive_surface_area=0.05
        )
    }
    
    return KineticsParameters(
        mass_transfer_coefficient=1.0e-6,  # m/s
        henrys_constant=3.3e-4,  # Pa·m³/mol (CO₂ in water at 25°C)
        diffusion_coefficient=2.0e-9,  # m²/s (CO₂ in water)
        minerals=minerals,
        initial_mineral_volume_fraction=0.05,
        maximum_mineral_precipitation=0.15,
        reaction_surface_area_factor=100.0
    )

def create_typical_brine(salinity: float = 35000.0, temperature: float = 60.0) -> BrineProperties:
    """Create typical brine properties for CO₂ storage"""
    return BrineProperties(
        salinity=salinity,  # g/L (seawater-like)
        pH=7.0,
        temperature=temperature,  # °C
        pressure=1.0e7,  # 100 bar
        ionic_strength=salinity / 58440.0
    )

def estimate_long_term_trapping(storage_time_years: float, kinetics_engine: MineralizationEngine,
                              initial_conditions: Dict[str, Any]) -> Dict[str, float]:
    """Estimate long-term CO₂ trapping through dissolution and mineralization"""
    # Simplified long-term estimation
    # Based on typical rates from literature
    
    # Typical rates (simplified)
    dissolution_half_life = 10.0  # years for 50% dissolution
    mineralization_half_life = 1000.0  # years for 50% mineralization
    
    # Calculate trapping fractions
    dissolution_fraction = 1.0 - np.exp(-np.log(2) * storage_time_years / dissolution_half_life)
    mineralization_fraction = 1.0 - np.exp(-np.log(2) * storage_time_years / mineralization_half_life)
    
    return {
        'dissolution_fraction': dissolution_fraction,
        'mineralization_fraction': mineralization_fraction,
        'total_trapped_fraction': dissolution_fraction + mineralization_fraction,
        'time_to_50_percent_dissolution': dissolution_half_life,
        'time_to_50_percent_mineralization': mineralization_half_life
    }