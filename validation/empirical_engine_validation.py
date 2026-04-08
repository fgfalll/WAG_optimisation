"""
Empirical Engine Validation for PhD Research
=========================================

Comprehensive validation of PhD Surrogate CO2-EOR simulation engine
against CMG GEM benchmark cases.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

# Add project root to path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_models import (
    ReservoirData, EORParameters, OperationalParameters, EconomicParameters
)
from core.engine_surrogate.surrogate_engine import create_surrogate_engine
from validation.sr3_parser import SR3Parser
from validation.comparison_metrics import ComparisonMetrics

logger = logging.getLogger(__name__)

# Add validation directory to path
VALIDATION_DIR = Path(__file__).parent

# CMG Case Definitions
CMG_CASES = {
    'gmflu002_1D': {
        'name': 'SPE5 Wasson CO2 Flood (1D)',
        'sr3_file': 'cmg/flu/gmflu002_1D.sr3',
        'description': '10x1x1, 11 components, CO2 injection at 1.20e+7 SCF/day, 8 years',
        'reference': {
            'cumulative_oil': 20327827.0,
            'recovery_factor': 0.7788,
            'final_pressure': 1504.1,
        },
        'v_dp': 0.0,
        'permeability': 200.0,
        'ooip': 26101473.0, # STB
        'area_acres': 5.739, # 500x500 ft converted to acres
        'length_ft': 5000.0, # 10 blocks * 500 ft
        'sor': 0.39,
        'swi': 0.20,
        'oil_fvf': 1.2,
    },
    'gmflu002': {
        'name': 'SPE5 Wasson CO2 Flood (3D)',
        'sr3_file': 'cmg/flu/gmflu002.sr3',
        'description': '7x7x3, 11 components, 8 years',
        'reference': {
            'cumulative_oil': 17412000.0,
            'recovery_factor': 0.3326,
            'final_pressure': 1232.0,
        },
        'v_dp': 0.612,
        'permeability': 215.0,
        'ooip': 52359750.0, # 7x7x3 geometric STB
        'area_acres': 112.48, # 7x7 * 500x500 ft / 43560
        'length_ft': 3500.0, # 7 blocks * 500 ft
        'sor': 0.39,
        'swi': 0.20,
        'oil_fvf': 1.2,
    }
}

VALIDATION_CRITERIA = {
    'oil_cumulative_error_pct': 25.0,
    'pressure_rmse_psi': 300.0,
    'recovery_factor_error_pct': 25.0,
}

@dataclass
class ValidationResult:
    case_id: str
    case_name: str
    cmg_data: Dict[str, Any]
    engine_data: Dict[str, Any]
    metrics: Dict[str, float]
    passed: bool
    execution_time: float
    timestamp: datetime

def parse_cmg_reference(sr3_file: Path) -> Dict[str, Any]:
    """Parse CMG SR3 reference file"""
    parser = SR3Parser()
    try:
        data = parser.parse_file(sr3_file)
        return data
    except Exception as e:
        logger.error(f"Failed to parse SR3 file {sr3_file}: {e}")
        return {}
        
def detect_timeseries_anomalies(eng_t, eng_y, cmg_t=None, cmg_y=None, name="Parameter", is_pressure=False):
    """Detect unphysical plateaus, sudden drops, or extreme divergence from reference in temporal parameters."""
    anomalies = []
    if len(eng_y) < 3: return anomalies
    
    eng_rate = np.diff(eng_y) / np.maximum(np.diff(eng_t), 1e-6)
    
    # 1. Detect Sudden Drops (Negative Spikes)
    mean_rate = np.mean(eng_rate)
    std_rate = np.std(eng_rate)
    for i in range(len(eng_rate)):
        threshold = -200 if is_pressure else -0.05
        if eng_rate[i] < threshold and eng_rate[i] < (mean_rate - 3*std_rate):
            anomalies.append(f"Sudden drop in {name} at year {eng_t[i+1]:.1f} (Rate: {eng_rate[i]:.2f}/yr)")

    # 2. Detect Plateaus
    if len(eng_rate) > 5:
        for i in range(len(eng_rate) - 4):
            window = eng_rate[i:i+5]
            if np.all(np.abs(window) < 1e-4) and (is_pressure or eng_y[i] > 0.01):
                if i == 0 or np.abs(eng_rate[i-1]) >= 1e-4:
                    anomalies.append(f"Unphysical plateau in {name} starting at year {eng_t[i]:.1f}")
                    
    # 3. Detect Divergence from CMG Reference
    if cmg_t is not None and cmg_y is not None and len(cmg_t) > 2:
        cmg_y_interp = np.interp(eng_t, cmg_t, cmg_y)
        diff = eng_y - cmg_y_interp
        max_div_idx = np.argmax(np.abs(diff))
        
        div_threshold = 300 if is_pressure else 0.15
        if abs(diff[max_div_idx]) > div_threshold:
            anomalies.append(f"Major divergence from CMG in {name} at year {eng_t[max_div_idx]:.1f} (Diff: {diff[max_div_idx]:.2f})")
            
    return list(set(anomalies)) # Deduplicate

def run_surrogate_simulation(case_id: str, cmg_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run PhD Surrogate engine simulation"""
    start_time = time.time()
    info = CMG_CASES[case_id]
    
    # 1. Prepare Reservoir Data
    ooip = info['ooip']
    res_data = ReservoirData(
        grid={}, pvt_tables={},
        ooip_stb=ooip,
        initial_pressure=1100.0,
        temperature=90.0,
        average_porosity=0.30,
        average_permeability=info['permeability'],
        initial_water_saturation=info['swi'],
        area_acres=info.get('area_acres'),
        length_ft=info.get('length_ft'),
        oil_fvf=info.get('oil_fvf'),
    )
    res_data.v_dp_coefficient = info['v_dp']
    res_data.residual_oil_saturation = info['sor']
    res_data.bg = 0.00207 # SPE5 specific
    
    # 1.5 Inject EOS Model to prevent fallback in breakthrough physics
    from core.data_models import EOSModelParameters
    res_data.eos_model = EOSModelParameters(
        eos_type="PR",
        component_names=["CO2", "C1", "C4", "C10+"],
        component_properties=np.array([
            # mole_frac, Mw, Tc(K), Pc(Pa), omega
            [0.0, 44.01, 304.13, 7.376e6, 0.225],
            [0.1, 16.04, 190.6, 4.604e6, 0.011],
            [0.3, 58.12, 425.2, 3.796e6, 0.200],
            [0.6, 142.0, 617.7, 2.11e6,  0.490]
        ]),
        binary_interaction_coeffs=np.zeros((4,4))
    )
    
    # 2. Prepare EOR Parameters
    eor = EORParameters(
        injection_rate=24837.0, 
        target_pressure_psi=1430.0,
        default_mmp_fallback=1200.0,
        default_oil_viscosity_cp=1.5,
        default_co2_viscosity_cp=0.05,
        sor=info['sor'],
    )
    
    # 3. Operational Parameters
    op = OperationalParameters(project_lifetime_years=8.0)
    
    # 4. Engine Evaluation
    wrapper = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")



    res = wrapper.evaluate_scenario(res_data, eor, op, use_dynamic_fractional_flow=True)

    return {        'execution_time': time.time() - start_time,
        'results': res
    }

def validate_single_case(case_id: str, plot_results: bool = False) -> ValidationResult:
    logger.info(f"\nValidating Case: {case_id}")
    info = CMG_CASES[case_id]
    sr3_path = VALIDATION_DIR / info['sr3_file']
    
    cmg_data = parse_cmg_reference(sr3_path)
    if not cmg_data or not cmg_data.get('cumulative_oil'):
        logger.warning(f"Using hardcoded reference for {case_id}")
        cmg_data = info['reference']

    sim_out = run_surrogate_simulation(case_id, cmg_data)
    res = sim_out['results']
    
    engine_data = {
        'cumulative_oil': res.get('cumulative_oil', 0),
        'recovery_factor': res.get('recovery_factor', 0),
        'final_pressure': res.get('pressure', [0])[-1] if isinstance(res.get('pressure', [0]), (list, np.ndarray)) else res.get('pressure', 0),
        'time_years': np.array(res.get('time_vector', [0])) / 365.25,
        'rf_profile': np.array(res.get('recovery_factor_profile', [res.get('recovery_factor', 0)])),
        'pressure_profile': np.array(res.get('pressure', [res.get('pressure', 0)])),
    }

    metrics_calc = ComparisonMetrics()
    metrics = metrics_calc.calculate_all(engine_data, cmg_data)
    
    # PhD Validation Logic: Focus on Recovery Factor relative error
    rf_cmg_final = cmg_data.get('recovery_factor', 0.5)
    if isinstance(rf_cmg_final, (list, np.ndarray)): rf_cmg_final = rf_cmg_final[-1]
    rf_eng_final = engine_data.get('recovery_factor', 0)
    rf_rel_error = abs(rf_eng_final - rf_cmg_final) / max(rf_cmg_final, 0.01)

    # Timeseries RMSE (if CMG data has profiles)
    rmse_rf = 0.0
    rmse_p = 0.0
    
    anomalies = []
    cmg_t_rf, cmg_rf = None, None
    cmg_t_p, cmg_p = None, None
    
    if 'recovery_profile' in cmg_data and isinstance(cmg_data['recovery_profile'], (list, np.ndarray)):
        cmg_t_rf = np.array(cmg_data.get('time_vector', engine_data['time_years']*365.25)) / 365.25
        cmg_rf = np.array(cmg_data['recovery_profile'])
        if len(cmg_t_rf) > 1 and len(engine_data['time_years']) > 1:
            cmg_rf_interp = np.interp(engine_data['time_years'], cmg_t_rf, cmg_rf)
            rmse_rf = np.sqrt(np.mean((engine_data['rf_profile'] - cmg_rf_interp)**2))
            
    if 'pressure_profile' in cmg_data and isinstance(cmg_data['pressure_profile'], (list, np.ndarray)):
        cmg_t_p = np.array(cmg_data.get('time_vector', engine_data['time_years']*365.25)) / 365.25
        cmg_p = np.array(cmg_data['pressure_profile'])
        if len(cmg_t_p) > 1 and len(engine_data['time_years']) > 1:
            cmg_p_interp = np.interp(engine_data['time_years'], cmg_t_p, cmg_p)
            rmse_p = np.sqrt(np.mean((engine_data['pressure_profile'] - cmg_p_interp)**2))
            
    # Run Anomaly Detection
    anomalies.extend(detect_timeseries_anomalies(engine_data['time_years'], engine_data['rf_profile'], cmg_t_rf, cmg_rf, "Recovery Factor", is_pressure=False))
    anomalies.extend(detect_timeseries_anomalies(engine_data['time_years'], engine_data['pressure_profile'], cmg_t_p, cmg_p, "Pressure", is_pressure=True))

    # 3D case is the primary benchmark, we allow 20% relative error for screening surrogate
    passed = rf_rel_error < 0.20

    # Log summary
    final_p_cmg = cmg_data.get('final_pressure', 0)
    logger.info(f"  CMG: RF={rf_cmg_final:.2%}, P={final_p_cmg:.0f} psi")
    logger.info(f"  PhD: RF={rf_eng_final:.2%}, P={engine_data['final_pressure']:.0f} psi")
    logger.info(f"  RMSE: RF Profile={rmse_rf:.3f}, Pressure Profile={rmse_p:.0f} psi")
    for anom in anomalies:
        logger.warning(f"  [!] {anom}")
    logger.info(f"  Result: {'PASS' if passed else 'FAIL'} (RF End Error: {rf_rel_error:.1%})")

    if plot_results:
        generate_plots(case_id, cmg_data, engine_data)

    return ValidationResult(
        case_id=case_id, case_name=info['name'],
        cmg_data=cmg_data, engine_data=engine_data,
        metrics={'rf_rel_error': rf_rel_error, 'oil_cumulative_error_pct': metrics['oil_cumulative_error_pct']}, 
        passed=passed,
        execution_time=sim_out['execution_time'],
        timestamp=datetime.now()
    )

def generate_plots(case_id: str, cmg: dict, eng: dict):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Determine time axes
        eng_t = eng.get('time_years', [])
        
        # Plot Recovery Factor
        if 'recovery_profile' in cmg and isinstance(cmg['recovery_profile'], (list, np.ndarray)):
            cmg_t = np.array(cmg.get('time_vector', eng_t*365.25)) / 365.25
            ax1.plot(cmg_t, np.array(cmg['recovery_profile'])*100, 'b-', label='CMG Reference', linewidth=2)
        else:
            ax1.axhline(y=cmg.get('recovery_factor', 0)*100, color='b', linestyle='--', label='CMG Final')
            
        if 'rf_profile' in eng and len(eng_t) > 1:
            ax1.plot(eng_t, eng['rf_profile']*100, 'r--', label='PhD Surrogate', linewidth=2)
        else:
            ax1.axhline(y=eng.get('recovery_factor', 0)*100, color='r', linestyle='--', label='PhD Final')
            
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Recovery Factor (%)')
        ax1.set_title('Recovery Factor Evolution')
        ax1.legend()
        ax1.grid(True)

        # Plot Pressure
        if 'pressure_profile' in cmg and isinstance(cmg['pressure_profile'], (list, np.ndarray)):
            cmg_t = np.array(cmg.get('time_vector', eng_t*365.25)) / 365.25
            ax2.plot(cmg_t, cmg['pressure_profile'], 'b-', label='CMG Reference', linewidth=2)
        else:
            ax2.axhline(y=cmg.get('final_pressure', 0), color='b', linestyle='--', label='CMG Final')
            
        if 'pressure_profile' in eng and len(eng_t) > 1:
            ax2.plot(eng_t, eng['pressure_profile'], 'r--', label='PhD Surrogate', linewidth=2)
        else:
            ax2.axhline(y=eng.get('final_pressure', 0), color='r', linestyle='--', label='PhD Final')

        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Average Reservoir Pressure (psia)')
        ax2.set_title('Pressure Evolution')
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(f'Case {case_id}: CMG vs Calibration-Free Surrogate Simulation', fontsize=14)
        plt.tight_layout()
        plt.savefig(VALIDATION_DIR / f"{case_id}_comparison_timeseries.png")
        plt.close()
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='all')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    cases = list(CMG_CASES.keys()) if args.case == 'all' else [args.case]
    results = [validate_single_case(cid, args.plots) for cid in cases]
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.case_id:<15} | {status} | RF Error: {r.metrics['rf_rel_error']:>5.1%}")
    
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == '__main__':
    main()


