import pytest
from co2eor_optimizer.evaluation.mmp import MMPParameters, calculate_mmp_cronquist, calculate_mmp_glaso

def test_cronquist_mmp():
    """Test Cronquist MMP calculation"""
    params = MMPParameters(temperature=120, oil_gravity=32)
    mmp = calculate_mmp_cronquist(params)
    assert 1200 <= mmp <= 1800  # Expected range for typical conditions

def test_glaso_mmp():
    """Test Glaso MMP calculation with composition"""
    params = MMPParameters(
        temperature=150,
        oil_gravity=28,
        c7_plus_mw=210,
        injection_gas_composition={'CO2': 0.95, 'CH4': 0.05}
    )
    mmp = calculate_mmp_glaso(params)
    # Expected range for 95% CO2 injection at 150°F and 28°API oil
    # Based on literature values and field data
    assert 1000 <= mmp <= 1500

def test_glaso_mmp_validation():
    """Test Glaso validation requires C7+ MW"""
    params = MMPParameters(temperature=150, oil_gravity=28)
    with pytest.raises(ValueError):
        calculate_mmp_glaso(params)