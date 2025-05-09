import numpy as np
from co2eor_optimizer.core import WellData, ReservoirData, PVTProperties

def test_well_data_validation():
    """Test WellData validation logic"""
    well = WellData(
        name="Test Well",
        depths=np.array([1000, 1001, 1002]),
        properties={"GR": np.array([50, 55, 60])},
        units={"GR": "API"}
    )
    assert well.validate()

def test_pvt_properties():
    """Test PVTProperties initialization"""
    pvt = PVTProperties(
        oil_fvf=np.array([1.2, 1.1, 1.0]),
        oil_viscosity=np.array([0.8, 1.0, 1.2]),
        gas_fvf=np.array([0.01, 0.02, 0.03]),
        gas_viscosity=np.array([0.01, 0.01, 0.01]),
        rs=np.array([100, 150, 200]),
        pvt_type="black_oil"
    )
    assert pvt.pvt_type == "black_oil"