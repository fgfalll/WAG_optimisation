"""
Geology Engine for CO2 EOR Simulation
Handles geological calculations, heterogeneity, and sweep efficiency modifiers.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


from core.data_models import PVTProperties, ReservoirData, EORParameters

class GeologyEngine:
    """
    Geology engine for CO2 EOR simulation.
    Handles geological calculations, heterogeneity, and sweep efficiency modifiers.
    """
    
    def __init__(self, reservoir: ReservoirData, pvt: PVTProperties, eor_params: EORParameters):
        """
        Initialize geology engine with reservoir and EOR parameters.
        
        Args:
            reservoir: ReservoirData object containing geological properties
            pvt: PVTProperties object containing fluid properties
            eor_params: EORParameters object containing fluid and rock properties
        """
        self.reservoir = reservoir
        self.eor_params = eor_params
        
        # Calculate mobility ratio from PVT data
        oil_viscosity = pvt.oil_viscosity_cp or 2.0  # Default to 2.0 cP if not available
        co2_viscosity = pvt.gas_viscosity_cp or 0.08 # Default to 0.08 cP if not available
        self.mobility_ratio = oil_viscosity / (co2_viscosity + 1e-6)
    
    def calculate_geology_enhanced_sweep_efficiency(self) -> float:
        """
        Calculate sweep efficiency enhanced with geological parameters.
        Incorporates rock type, depositional environment, and structural complexity.
        
        Returns:
            Enhanced sweep efficiency factor (0.1 to 1.0)
        """
        M = self.mobility_ratio
        
        # Base sweep efficiency from mobility ratio
        if M <= 1.0:
            base_efficiency = 1.0
        else:
            if M <= 10:
                base_efficiency = 0.5 + 0.4 * np.log10(M) / (M - 1)
            else:
                base_efficiency = np.exp(-0.1 * (M - 10)) * (0.546 + 0.0357/M)
        
        # Apply geological modifiers
        geology_factor = self._calculate_geology_factor()
        enhanced_efficiency = base_efficiency * geology_factor
        
        return np.clip(enhanced_efficiency, 0.1, 1.0)
    
    def _calculate_geology_factor(self) -> float:
        """
        Calculate geology factor based on reservoir geological characteristics.
        
        Returns:
            Geology factor multiplier
        """
        factor = 1.0
        
        # Rock type modifier
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                factor *= 1.1  # Sandstone typically has better sweep
            elif rock_type == 'carbonate':
                factor *= 0.9  # Carbonates often have more heterogeneity
            elif rock_type == 'shale':
                factor *= 0.7  # Shale has poor sweep characteristics
        
        # Depositional environment modifier
        depositional_env = getattr(self.reservoir, 'depositional_environment', None)
        if depositional_env:
            if depositional_env in ['fluvial', 'deltaic']:
                factor *= 1.05  # Channelized systems can have good sweep if properly targeted
            elif depositional_env == 'aeolian':
                factor *= 0.95  # Dune systems can have complex flow patterns
            elif depositional_env in ['deep_marine', 'shallow_marine']:
                factor *= 1.0  # Marine systems typically have moderate sweep
        
        # Structural complexity modifier
        structural_complexity = getattr(self.reservoir, 'structural_complexity', None)
        if structural_complexity:
            if structural_complexity == 'simple':
                factor *= 1.1
            elif structural_complexity == 'moderate':
                factor *= 1.0
            elif structural_complexity == 'complex':
                factor *= 0.9
            elif structural_complexity == 'very_complex':
                factor *= 0.8
        
        # Geostatistical heterogeneity modifier
        if hasattr(self.reservoir, 'geostatistical_grid') and self.reservoir.geostatistical_grid is not None:
            heterogeneity = self.calculate_heterogeneity_index()
            # Higher heterogeneity reduces sweep efficiency
            heterogeneity_penalty = 1.0 - (heterogeneity * 0.3)  # Up to 30% penalty for high heterogeneity
            factor *= np.clip(heterogeneity_penalty, 0.7, 1.0)
        
        return factor
    
    def calculate_heterogeneity_index(self) -> float:
        """
        Calculate heterogeneity index from geostatistical grid.
        Higher values indicate more heterogeneous reservoir.
        
        Returns:
            Heterogeneity index (0.0 to 1.0)
        """
        if self.reservoir.geostatistical_grid is None:
            return 0.0
        
        grid = self.reservoir.geostatistical_grid
        if grid.size == 0:
            return 0.0
        
        # Calculate coefficient of variation as heterogeneity measure
        std_dev = np.std(grid)
        mean_val = np.mean(grid)
        
        if mean_val > 0:
            return std_dev / mean_val
        else:
            return 0.0
    
    def get_geology_based_permeability_modifier(self) -> float:
        """
        Get permeability modifier based on geological characteristics.
        
        Returns:
            Permeability modifier factor
        """
        modifier = 1.0
        
        # Rock type permeability modifiers
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                modifier *= 1.2  # Sandstone typically has higher permeability
            elif rock_type == 'carbonate':
                modifier *= 0.8  # Carbonates can have variable permeability
            elif rock_type == 'shale':
                modifier *= 0.3  # Shale has very low permeability
        
        return modifier
    
    def calculate_geology_injection_factor(self) -> float:
        """
        Calculate injection rate factor based on geological characteristics.
        
        Returns:
            Injection rate factor
        """
        factor = 1.0
        
        # Rock type injection modifiers
        rock_type = getattr(self.reservoir, 'rock_type', None)
        if rock_type:
            if rock_type == 'sandstone':
                factor *= 1.1  # Sandstone can typically handle higher injection rates
            elif rock_type == 'carbonate':
                factor *= 0.9  # Carbonates may have fracture concerns
            elif rock_type == 'shale':
                factor *= 0.6  # Shale has very low injectivity
        
        # Structural complexity modifier
        structural_complexity = getattr(self.reservoir, 'structural_complexity', None)
        if structural_complexity:
            if structural_complexity == 'complex':
                factor *= 0.8
            elif structural_complexity == 'very_complex':
                factor *= 0.6
        
        return np.clip(factor, 0.5, 1.5)