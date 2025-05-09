"""Well analysis module for CO2 EOR optimization"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from ..core import WellData, PVTProperties
from ..evaluation.mmp import calculate_mmp, MMPParameters

@dataclass
class WellAnalysis:
    """Integrates well log data with EOR analysis"""
    well_data: WellData
    pvt_data: Optional[PVTProperties] = None
    temperature_gradient: Optional[float] = None  # °F/ft
    
    def calculate_mmp_profile(self, 
                            method: str = 'auto',
                            gas_composition: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Calculate MMP profile vs depth using well log data
        
        Args:
            method: MMP calculation method ('auto', 'cronquist', 'glaso', 'yuan')
            gas_composition: Injection gas composition (default: pure CO2)
            
        Returns:
            Dictionary containing:
            - 'depths': Depth array (ft or m)
            - 'mmp': MMP values (psi)
            - 'temperature': Temperature values (°F)
            - 'api': Estimated API gravity
        """
        if not hasattr(self.well_data, 'depths'):
            raise ValueError("WellData must contain depth information")
            
        # Estimate API gravity from density log if available
        api = self._estimate_api_from_logs()
        
        # Calculate temperature profile if gradient provided
        temp = self._calculate_temperature_profile()
        
        # Prepare MMP parameters for each depth
        mmp_values = np.zeros_like(self.well_data.depths)
        
        gas_comp = gas_composition or {'CO2': 1.0}
        
        for i, depth in enumerate(self.well_data.depths):
            params = MMPParameters(
                temperature=temp[i],
                oil_gravity=api[i],
                injection_gas_composition=gas_comp,
                pvt_data=self.pvt_data
            )
            mmp_values[i] = calculate_mmp(params, method)
            
        return {
            'depths': self.well_data.depths,
            'mmp': mmp_values,
            'temperature': temp,
            'api': api
        }
    
    def _estimate_api_from_logs(self) -> np.ndarray:
        """Estimate API gravity from density log (RHOB)"""
        if 'RHOB' not in self.well_data.properties:
            return np.full(len(self.well_data.depths), 32.0)  # Default
        
        rhob = self.well_data.properties['RHOB']
        api = (141.5 / rhob) - 131.5  # Simple conversion
        return np.clip(api, 15, 50)  # Constrain to valid range
    
    def _calculate_temperature_profile(self) -> np.ndarray:
        """Calculate temperature vs depth using gradient"""
        if not self.temperature_gradient:
            return np.full(len(self.well_data.depths), 212.0)  # Default
        
        # Calculate from gradient and reference depth
        ref_depth = self.well_data.depths[0]
        ref_temp = 70.0  # Surface temp °F
        return ref_temp + self.temperature_gradient * (self.well_data.depths - ref_depth)
    
    def find_miscible_zones(self, 
                          pressure: float,
                          gas_composition: Optional[Dict] = None,
                          method: str = 'auto') -> Dict[str, np.ndarray]:
        """
        Identify reservoir zones where CO2 will be miscible at given pressure
        
        Args:
            pressure: Injection pressure (psi)
            gas_composition: Injection gas composition
            method: MMP calculation method
            
        Returns:
            Dictionary containing:
            - 'depths': Depth array
            - 'is_miscible': Boolean array (True where miscible)
            - 'mmp': MMP values
            - 'temperature': Temperature values
        """
        profile = self.calculate_mmp_profile(method, gas_composition)
        is_miscible = pressure >= profile['mmp']
        
        return {
            'depths': profile['depths'],
            'is_miscible': is_miscible,
            'mmp': profile['mmp'],
            'temperature': profile['temperature']
        }