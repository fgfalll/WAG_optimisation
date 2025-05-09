"""Tests for CO2 EOR optimization engine"""
import numpy as np
import pytest
from co2eor_optimizer.core import recovery_factor
from co2eor_optimizer.core import (
    ReservoirData, 
    PVTProperties,
    EORParameters,
    OptimizationEngine,
    GeneticAlgorithmParams
)

@pytest.fixture
def sample_reservoir():
    """Create sample reservoir data"""
    return ReservoirData(
        grid={
            'PORO': np.full((10, 10, 5), 0.2),
            'PERMX': np.full((10, 10, 5), 100)
        },
        pvt_tables={
            'PVTO': np.array([[1000, 1.2, 0.5, 800]]),
            'PVTG': np.array([[1000, 0.05, 200, 0.02]])
        }
    )

@pytest.fixture
def sample_pvt():
    """Create sample PVT data"""
    return PVTProperties(
        oil_fvf=np.array([1.2, 1.15, 1.1]),
        oil_viscosity=np.array([0.5, 0.6, 0.7]),
        gas_fvf=np.array([0.05, 0.06, 0.07]),
        gas_viscosity=np.array([0.02, 0.025, 0.03]),
        rs=np.array([800, 700, 600]),
        pvt_type='black_oil'
    )

@pytest.fixture
def sample_eor_params():
    """Create sample EOR parameters"""
    return EORParameters(
        injection_rate=5000,
        WAG_ratio=1.0,
        injection_scheme='wag',
        target_pressure=2500,
        max_pressure=5000,
        min_injection_rate=1000
    )

def test_optimization_engine_init(sample_reservoir, sample_pvt, sample_eor_params):
    """Test optimization engine initialization"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    assert engine.reservoir == sample_reservoir
    assert engine.pvt == sample_pvt
    assert engine.eor_params == sample_eor_params
    assert engine.results is None

def test_mmp_calculation(sample_reservoir, sample_pvt, sample_eor_params):
    """Test MMP calculation integration"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    mmp = engine.calculate_mmp()
    assert isinstance(mmp, float)
    assert mmp > 0
    assert engine.mmp == mmp

def test_mmp_constraint_check(sample_reservoir, sample_pvt, sample_eor_params):
    """Test MMP constraint checking"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    engine.calculate_mmp()
    
    # Test pressure below MMP
    assert not engine.check_mmp_constraint(engine.mmp - 100)
    
    # Test pressure at MMP
    assert engine.check_mmp_constraint(engine.mmp)
    
    # Test pressure above MMP
    assert engine.check_mmp_constraint(engine.mmp + 100)

def test_optimize_recovery(sample_reservoir, sample_pvt, sample_eor_params):
    """Test recovery optimization algorithm"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    results = engine.optimize_recovery(max_iter=20)
    
    # Verify results structure
    assert isinstance(results, dict)
    assert 'optimized_params' in results
    assert 'iterations' in results
    assert 'final_recovery' in results
    assert 'mmp' in results
    assert 'converged' in results
    assert 'avg_porosity' in results
    
    # Verify optimization improved recovery
    initial_pressure = max(engine.mmp, sample_eor_params.target_pressure)
    initial_recovery = recovery_factor(
        initial_pressure,
        sample_eor_params.injection_rate,
        results['avg_porosity'],
        engine.mmp
    )
    assert results['final_recovery'] >= initial_recovery
    
    # Verify pressure meets MMP constraint
    assert results['optimized_params']['target_pressure'] >= engine.mmp
    
    # Verify reasonable values
    assert 0 < results['final_recovery'] <= 0.7
    assert 0 <= results['iterations'] <= 20

def test_optimize_wag_structure(sample_reservoir, sample_pvt, sample_eor_params):
    """Test WAG optimization structure"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    results = engine.optimize_wag(cycles=3)
    
    assert isinstance(results, dict)
    assert 'wag_params' in results
    assert 'mmp' in results
    assert 'estimated_recovery' in results

def test_genetic_algorithm_optimization(sample_reservoir, sample_pvt, sample_eor_params):
    """Test genetic algorithm optimization"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    ga_params = GeneticAlgorithmParams(
        population_size=20,
        generations=10,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    results = engine.optimize_genetic_algorithm(ga_params)
    
    # Verify results structure
    assert isinstance(results, dict)
    assert 'optimized_params' in results
    assert 'method' in results
    assert results['method'] == 'genetic_algorithm'
    assert 'generations' in results
    assert 'population_size' in results
    assert 'final_recovery' in results
    assert 'avg_porosity' in results
    
    # Verify optimization improved recovery
    initial_pressure = max(engine.mmp, sample_eor_params.target_pressure)
    initial_recovery = recovery_factor(
        initial_pressure,
        sample_eor_params.injection_rate,
        results['avg_porosity'],
        engine.mmp
    )
    assert results['final_recovery'] >= initial_recovery
    
    # Verify pressure meets constraints
    assert results['optimized_params']['target_pressure'] >= engine.mmp
    assert results['optimized_params']['target_pressure'] <= sample_eor_params.max_pressure
    assert results['optimized_params']['injection_rate'] >= sample_eor_params.min_injection_rate
    
    # Verify reasonable values
    assert 0 < results['final_recovery'] <= 0.7
    assert results['generations'] == 10
    assert results['population_size'] == 20

def test_genetic_algorithm_default_params(sample_reservoir, sample_pvt, sample_eor_params):
    """Test genetic algorithm with default parameters"""
    engine = OptimizationEngine(sample_reservoir, sample_pvt, sample_eor_params)
    results = engine.optimize_genetic_algorithm()
    
    assert isinstance(results, dict)
    assert 'optimized_params' in results
    assert results['method'] == 'genetic_algorithm'
    assert results['generations'] == 100  # Default value
    assert results['population_size'] == 50  # Default value