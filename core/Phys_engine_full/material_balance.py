"""
Material balance module for CO₂ EOR and storage analysis.
Handles CO₂ accounting, breakthrough physics, recycling calculations, and generates material balance graphs.
Designed to be separate from the optimization engine to avoid performance impact.
"""

import numpy as np

# Optional imports with graceful fallbacks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy plotly modules to prevent import errors
    class go:
        class Figure:
            def __init__(self, *args, **kwargs):
                pass
            def add_trace(self, *args, **kwargs):
                pass
            def update_layout(self, *args, **kwargs):
                pass
            def show(self, *args, **kwargs):
                print("Plotly not available - interactive plot disabled")
            def write_html(self, *args, **kwargs):
                print("Plotly not available - HTML export disabled")
        class Scatter:
            def __init__(self, *args, **kwargs):
                pass
        class Line:
            def __init__(self, *args, **kwargs):
                pass
        class Bar:
            def __init__(self, *args, **kwargs):
                pass

    def make_subplots(*args, **kwargs):
        return go.Figure()
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import breakthrough physics
try:
    from .breakthrough_physics import CO2BreakthroughPhysics, BreakthroughParameters
except ImportError:
    logger.warning("Breakthrough physics module not available - using fallback calculations")
    # Fallback definitions for testing
    class BreakthroughParameters:
        breakthrough_gor_threshold = 800.0
        recycling_efficiency = 0.9
    
    class CO2BreakthroughPhysics:
        def __init__(self, params=None):
            self.params = params or BreakthroughParameters()

class MaterialBalanceAnalyzer:
    """
    Analyzes CO₂ material balance for EOR projects with storage verification.
    Calculates injection, production, recycling, and net storage metrics.
    Generates comprehensive material balance graphs.
    """
    
    def __init__(self, co2_density_tonne_per_mscf: float = 0.053,
                 breakthrough_params: Optional[BreakthroughParameters] = None):
        """
        Initialize the material balance analyzer with breakthrough physics.
        
        Args:
            co2_density_tonne_per_mscf: Density conversion factor from Mscf to tonnes
            breakthrough_params: Parameters for breakthrough physics calculations
        """
        self.co2_density_tonne_per_mscf = co2_density_tonne_per_mscf
        self.breakthrough_physics = CO2BreakthroughPhysics(breakthrough_params)
    
    def calculate_material_balance(self,
                                 annual_co2_injected_mscf: np.ndarray,
                                 annual_co2_produced_mscf: np.ndarray,
                                 annual_co2_recycled_mscf: np.ndarray,
                                 leakage_rate_fraction: float = 0.01,
                                 reservoir_params: Optional[Dict] = None,
                                 eor_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Calculate comprehensive CO₂ material balance with leakage accounting.
        
        Args:
            annual_co2_injected_mscf: Annual CO₂ injected (Mscf)
            annual_co2_produced_mscf: Annual CO₂ produced (Mscf)
            annual_co2_recycled_mscf: Annual CO₂ recycled (Mscf)
            leakage_rate_fraction: Annual leakage rate as fraction of cumulative storage
            
        Returns:
            Dictionary containing all material balance components in tonnes
        """
        # Convert all volumes to tonnes
        injected_tonne = annual_co2_injected_mscf * self.co2_density_tonne_per_mscf
        produced_tonne = annual_co2_produced_mscf * self.co2_density_tonne_per_mscf
        recycled_tonne = annual_co2_recycled_mscf * self.co2_density_tonne_per_mscf
        
        # Calculate breakthrough-aware recycling if parameters are provided
        if reservoir_params is not None and eor_params is not None:
            breakthrough_analysis = self.calculate_breakthrough_aware_recycling(
                injected_tonne, reservoir_params, eor_params
            )
            # Use breakthrough-aware recycling values if available
            if breakthrough_analysis:
                recycled_tonne = breakthrough_analysis.get('recycled_tonne_breakthrough_aware', recycled_tonne)
        
        n_years = len(injected_tonne)
        
        # Initialize arrays
        net_injected_tonne = np.zeros(n_years)
        net_stored_tonne = np.zeros(n_years)
        cumulative_stored_tonne = np.zeros(n_years)
        annual_leakage_tonne = np.zeros(n_years)
        storage_efficiency = np.zeros(n_years)
        
        # Calculate material balance with leakage
        for i in range(n_years):
            # Net change in reservoir = total injected - total produced
            net_injected_tonne[i] = injected_tonne[i] - produced_tonne[i]
            
            # Calculate leakage from previous cumulative storage
            if i == 0:
                leakage = 0.0
            else:
                leakage = cumulative_stored_tonne[i-1] * leakage_rate_fraction
            
            annual_leakage_tonne[i] = leakage
            
            # Net stored = net injected - leakage
            net_stored = net_injected_tonne[i] - leakage
            net_stored_tonne[i] = net_stored
            
            # Update cumulative storage
            if i == 0:
                cumulative_stored_tonne[i] = net_stored
            else:
                cumulative_stored_tonne[i] = cumulative_stored_tonne[i-1] + net_stored
            
            # Calculate storage efficiency
            if net_injected_tonne[i] > 0:
                storage_efficiency[i] = net_stored_tonne[i] / net_injected_tonne[i]
            else:
                storage_efficiency[i] = 0.0

        # Calculate mass balance error
        total_injected = np.sum(injected_tonne)
        total_accounted = (
            np.sum(net_stored_tonne) +
            np.sum(annual_leakage_tonne)
        )

        if total_injected > 0:
            mass_balance_error = abs(total_accounted - total_injected) / total_injected
        else:
            mass_balance_error = 0.0

        return {
            'injected_tonne': injected_tonne,
            'produced_tonne': produced_tonne,
            'recycled_tonne': recycled_tonne,
            'net_injected_tonne': net_injected_tonne,
            'net_stored_tonne': net_stored_tonne,
            'cumulative_stored_tonne': cumulative_stored_tonne,
            'annual_leakage_tonne': annual_leakage_tonne,
            'storage_efficiency': storage_efficiency,
            'years': np.arange(1, n_years + 1),
            # Mass balance error tracking
            'mass_balance_error': mass_balance_error,
            'total_injected_tonne': total_injected,
            'total_accounted_tonne': total_accounted,
        }

    def calculate_breakthrough_aware_recycling(self, injected_tonne: np.ndarray,
                                             reservoir_params: Dict, eor_params: Dict) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculate recycling considering breakthrough physics and timing.
        
        Args:
            injected_tonne: Annual CO₂ injected (tonnes)
            reservoir_params: Reservoir properties
            eor_params: EOR operation parameters
            
        Returns:
            Dictionary with breakthrough-aware recycling metrics
        """
        try:
            # Calculate breakthrough time
            breakthrough_time = self.breakthrough_physics.calculate_breakthrough_time(
                reservoir_params, eor_params
            )
            
            n_years = len(injected_tonne)
            years = np.arange(1, n_years + 1)
            
            # Determine breakthrough status for each year
            breakthrough_occurred = years >= breakthrough_time
            time_since_breakthrough = np.maximum(0, years - breakthrough_time)
            
            # Calculate recycling efficiency profile based on breakthrough
            recycling_efficiency = np.zeros(n_years)
            for i in range(n_years):
                if breakthrough_occurred[i]:
                    # After breakthrough: efficiency depends on time since breakthrough
                    # Early breakthrough: lower efficiency due to impurities
                    # Later: higher efficiency as system stabilizes
                    efficiency = self.breakthrough_physics.params.recycling_efficiency
                    time_factor = min(1.0, time_since_breakthrough[i] / 5.0)  # Stabilize over 5 years
                    recycling_efficiency[i] = efficiency * time_factor
                else:
                    # Before breakthrough: minimal recycling (only incidental CO₂)
                    recycling_efficiency[i] = 0.1  # 10% base efficiency
            
            # Estimate produced CO₂ based on injection and breakthrough timing
            # This is a simplified model - in practice would use reservoir simulation
            produced_tonne_estimate = np.zeros(n_years)
            for i in range(n_years):
                if breakthrough_occurred[i]:
                    # After breakthrough: significant CO₂ production
                    # Use GOR-based estimation
                    gor = self.breakthrough_physics.calculate_post_breakthrough_gor(
                        eor_params, time_since_breakthrough[i]
                    )
                    # A more realistic estimate of oil production is based on OOIP and recovery factor.
                    ooip_stb = reservoir_params.get('ooip_stb', 1e7)  # Default to 10 million STB if not provided
                    recovery_factor = eor_params.get('recovery_factor', 0.1) # Default to 10% recovery
                    recoverable_oil_stb = ooip_stb * recovery_factor
                    
                    # Assume constant production over the project life for simplicity.
                    annual_oil_production_stb = recoverable_oil_stb / n_years
                    
                    # Convert STB to tonnes (1 STB is approx. 0.135 tonnes for typical oil)
                    stb_to_tonne = 0.135
                    oil_production_estimate = annual_oil_production_stb * stb_to_tonne
                    produced_tonne_estimate[i] = oil_production_estimate * gor * 0.001  # Convert scf to tonnes
                else:
                    # Before breakthrough: minimal CO₂ production
                    produced_tonne_estimate[i] = injected_tonne[i] * 0.05  # 5% dissolution
            
            # Calculate recyclable CO₂
            recycled_tonne = produced_tonne_estimate * recycling_efficiency
            
            return {
                'breakthrough_time_years': breakthrough_time,
                'recycling_efficiency_profile': recycling_efficiency,
                'produced_tonne_estimate': produced_tonne_estimate,
                'recycled_tonne_breakthrough_aware': recycled_tonne,
                'breakthrough_occurred': breakthrough_occurred
            }
            
        except Exception as e:
            logger.warning(f"Breakthrough-aware recycling calculation failed: {e}")
            return None
    
    def generate_material_balance_graphs(self, 
                                       material_balance_data: Dict[str, np.ndarray],
                                       title_suffix: str = "") -> Dict[str, go.Figure]:
        """
        Generate comprehensive material balance graphs.
        
        Args:
            material_balance_data: Output from calculate_material_balance
            title_suffix: Optional suffix for graph titles
            
        Returns:
            Dictionary of Plotly figures for different material balance views
        """
        years = material_balance_data['years']
        
        # Main material balance chart
        fig_main = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Primary Y-axis: Mass flows (tonnes)
        fig_main.add_trace(go.Bar(x=years, y=material_balance_data['injected_tonne'], 
                                 name='CO₂ Injected', marker_color='blue'), secondary_y=False)
        fig_main.add_trace(go.Bar(x=years, y=material_balance_data['produced_tonne'], 
                                 name='CO₂ Produced', marker_color='red'), secondary_y=False)
        fig_main.add_trace(go.Bar(x=years, y=material_balance_data['recycled_tonne'], 
                                 name='CO₂ Recycled', marker_color='green'), secondary_y=False)
        fig_main.add_trace(go.Scatter(x=years, y=material_balance_data['net_stored_tonne'], 
                                    name='Net Stored', line=dict(color='orange', width=3)), secondary_y=False)
        
        # Secondary Y-axis: Cumulative storage
        fig_main.add_trace(go.Scatter(x=years, y=material_balance_data['cumulative_stored_tonne'], 
                                    name='Cumulative Stored', line=dict(color='purple', width=3, dash='dot')), 
                          secondary_y=True)
        
        fig_main.update_layout(
            title=f'CO₂ Material Balance{title_suffix}',
            xaxis_title='Project Year',
            yaxis_title='CO₂ Mass (tonnes)',
            yaxis2_title='Cumulative Storage (tonnes)',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Storage efficiency chart
        fig_efficiency = go.Figure()
        fig_efficiency.add_trace(go.Scatter(x=years, y=material_balance_data['storage_efficiency'] * 100,
                                           name='Storage Efficiency', line=dict(color='blue', width=2)))
        fig_efficiency.add_trace(go.Scatter(x=years, y=material_balance_data['annual_leakage_tonne'],
                                           name='Annual Leakage', line=dict(color='red', width=2), yaxis='y2'))
        
        fig_efficiency.update_layout(
            title=f'Storage Efficiency and Leakage{title_suffix}',
            xaxis_title='Project Year',
            yaxis_title='Storage Efficiency (%)',
            yaxis2=dict(title='Leakage (tonnes)', overlaying='y', side='right'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Cumulative breakdown chart
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(x=years, y=material_balance_data['cumulative_stored_tonne'],
                                           name='Cumulative Stored', line=dict(color='green', width=3)))
        fig_cumulative.add_trace(go.Scatter(x=years, y=np.cumsum(material_balance_data['injected_tonne']),
                                           name='Cumulative Injected', line=dict(color='blue', width=2, dash='dash')))
        fig_cumulative.add_trace(go.Scatter(x=years, y=np.cumsum(material_balance_data['produced_tonne']),
                                           name='Cumulative Produced', line=dict(color='red', width=2, dash='dash')))
        
        fig_cumulative.update_layout(
            title=f'Cumulative CO₂ Balance{title_suffix}',
            xaxis_title='Project Year',
            yaxis_title='Cumulative CO₂ (tonnes)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return {
            'main_balance': fig_main,
            'efficiency': fig_efficiency,
            'cumulative': fig_cumulative
        }
    
    def generate_summary_statistics(self, material_balance_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Generate summary statistics for material balance analysis.
        
        Args:
            material_balance_data: Output from calculate_material_balance
            
        Returns:
            Dictionary of summary statistics
        """
        return {
            'total_injected_tonne': np.sum(material_balance_data['injected_tonne']),
            'total_produced_tonne': np.sum(material_balance_data['produced_tonne']),
            'total_recycled_tonne': np.sum(material_balance_data['recycled_tonne']),
            'total_net_stored_tonne': np.sum(material_balance_data['net_stored_tonne']),
            'final_cumulative_stored_tonne': material_balance_data['cumulative_stored_tonne'][-1] if len(material_balance_data['cumulative_stored_tonne']) > 0 else 0.0,
            'avg_storage_efficiency': np.mean(material_balance_data['storage_efficiency']),
            'total_leakage_tonne': np.sum(material_balance_data['annual_leakage_tonne']),
            'storage_efficiency_range': f"{np.min(material_balance_data['storage_efficiency'] * 100):.1f}%-{np.max(material_balance_data['storage_efficiency'] * 100):.1f}%"
        }

# Utility function to create material balance analysis from optimization results
def create_material_balance_from_optimization(optimization_results: Dict[str, any], 
                                            resolution: str,
                                            co2_density_tonne_per_mscf: float = 0.053,
                                            leakage_rate_fraction: float = 0.01) -> Dict[str, any]:
    """
    Convenience function to create material balance analysis from optimization results.
    
    Args:
        optimization_results: Results dictionary from OptimizationEngine
        resolution: Time resolution of the profiles (e.g., 'annual', 'monthly')
        co2_density_tonne_per_mscf: Density conversion factor
        leakage_rate_fraction: Annual leakage rate
        
    Returns:
        Complete material balance analysis with graphs and statistics
    """
    analyzer = MaterialBalanceAnalyzer(co2_density_tonne_per_mscf)
    
    # Extract CO2 profiles from optimization results
    profiles = optimization_results.get('optimized_profiles', {})
    
    injected_key = f'{resolution}_co2_injected_mscf'
    produced_key = f'{resolution}_co2_produced_mscf'
    recycled_key = f'{resolution}_co2_recycled_mscf'

    if not profiles or injected_key not in profiles:
        logger.warning("No CO2 profiles found in optimization results")
        return {}
    
    # Calculate material balance
    balance_data = analyzer.calculate_material_balance(
        profiles[injected_key],
        profiles.get(produced_key, np.zeros_like(profiles[injected_key])),
        profiles.get(recycled_key, np.zeros_like(profiles[injected_key])),
        leakage_rate_fraction
    )
    
    # Generate graphs
    graphs = analyzer.generate_material_balance_graphs(balance_data, " - Optimized Scenario")
    
    # Generate summary statistics
    stats = analyzer.generate_summary_statistics(balance_data)
    
    return {
        'material_balance_data': balance_data,
        'graphs': graphs,
        'summary_statistics': stats
    }