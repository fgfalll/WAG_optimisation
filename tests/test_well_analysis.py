"""Tests for well analysis module"""
import numpy as np
from dataclasses import dataclass, field
from co2eor_optimizer.core import WellData
from co2eor_optimizer.analysis.well_analysis import WellAnalysis

def default_array():
    return np.array([1.2])

@dataclass
class MockPVT:
    oil_fvf: np.ndarray = field(default_factory=default_array)
    oil_viscosity: np.ndarray = field(default_factory=lambda: np.array([2.5]))
    gas_fvf: np.ndarray = field(default_factory=lambda: np.array([0.01]))
    gas_viscosity: np.ndarray = field(default_factory=lambda: np.array([0.05]))
    rs: np.ndarray = field(default_factory=lambda: np.array([500]))
    pvt_type: str = 'black_oil'

def test_mmp_profile_calculation():
    """Test MMP profile calculation with synthetic data"""
    depths = np.linspace(5000, 10000, 100)
    rhob = np.linspace(0.8, 1.0, 100)  # g/cc
    
    well_data = WellData(
        name="TEST",
        depths=depths,
        properties={'RHOB': rhob},
        units={'RHOB': 'G/CC'}
    )
    
    analyzer = WellAnalysis(
        well_data=well_data,
        pvt_data=MockPVT(),
        temperature_gradient=0.01  # 1°F/100ft
    )
    
    profile = analyzer.calculate_mmp_profile()
    
    assert len(profile['depths']) == 100
    assert len(profile['mmp']) == 100
    assert np.all(profile['temperature'] >= 70)  # Surface temp
    assert np.all(15 <= profile['api']) and np.all(profile['api'] <= 50)  # Valid API range

def test_miscible_zones():
    """Test miscible zone identification"""
    depths = np.array([5000, 6000, 7000])
    rhob = np.array([0.8, 0.85, 0.9])
    
    well_data = WellData(
        name="TEST",
        depths=depths,
        properties={'RHOB': rhob},
        units={'RHOB': 'G/CC'}
    )
    
    analyzer = WellAnalysis(
        well_data=well_data,
        temperature_gradient=0.01
    )
    
    # Test with pressure above all MMPs
    result = analyzer.find_miscible_zones(3000)
    assert np.all(result['is_miscible'])
    
    # Test with pressure below all MMPs
    result = analyzer.find_miscible_zones(1000)
    assert not np.any(result['is_miscible'])

def test_default_handling():
    """Test behavior with missing data"""
    depths = np.array([5000, 6000])
    well_data = WellData(
        name="TEST",
        depths=depths,
        properties={},  # No logs
        units={}
    )
    
    analyzer = WellAnalysis(well_data=well_data)
    profile = analyzer.calculate_mmp_profile()
    
    assert np.all(profile['api'] == 32)  # Default API
    assert np.all(profile['temperature'] == 212)  # Default temp