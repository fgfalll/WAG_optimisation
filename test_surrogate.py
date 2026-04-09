import numpy as np
from core.engine_surrogate.surrogate_engine import SurrogateEngineWrapper
from core.data_models import ReservoirData, EORParameters, OperationalParameters

def test_surrogate_engine():
    wrapper = SurrogateEngineWrapper()
    
    res_data = ReservoirData(
        grid={'NX': np.array([50]), 'NY': np.array([50]), 'NZ': np.array([10])},
        pvt_tables={},
        ooip_stb=1000000.0,
        initial_pressure=3000.0,
        temperature=150.0,
        rock_compressibility=3e-6,
        average_porosity=0.2,
        initial_water_saturation=0.25,
        thickness_ft=50.0,
        area_acres=100.0,
        length_ft=2000.0
    )
    
    eor_params = EORParameters(
        injection_rate=5000.0,
        target_pressure_psi=3200.0,
        max_pressure_psi=4000.0,
        injection_scheme="continuous",
        WAG_ratio=1.0,
        mobility_ratio=5.0
    )
    
    op_params = OperationalParameters(
        project_lifetime_years=15,
        time_resolution="monthly",
        recovery_model_selection="hybrid"
    )
    
    # We pass fitting params explicitly as a dict for simplicity, or we can let it fall back to default dict
    res = wrapper.evaluate_scenario(res_data, eor_params, op_params)
    
    time_vector = res.get('time_vector', np.array([]))
    if len(time_vector) > 1:
        dt = np.diff(time_vector, prepend=time_vector[0] - (time_vector[1] - time_vector[0]))
    else:
        dt = np.array([30.4])
        
    oil_rate = res.get('oil_production_rate', np.array([]))
    oil_vol = oil_rate * dt
    
    print(f"Engine Type: {res.get('engine_type')}")
    print(f"Recovery Factor Reported: {res.get('recovery_factor')}")
    print(f"Sum of Oil Volumes (STB): {np.sum(oil_vol)}")
    print(f"OOIP: {res_data.ooip_stb}")
    print(f"Calculated RF: {np.sum(oil_vol) / res_data.ooip_stb}")
    print(f"Sum of CO2 Injected Rate (MSCFD) * dt: {np.sum(res.get('co2_injection', np.array([])) * dt)}")
    
if __name__ == "__main__":
    test_surrogate_engine()
