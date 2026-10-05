import sys; sys.path.insert(0,r'D:\rep\4.6\co2eor_optimizer')
import numpy as np
from core.data_models import ReservoirData, EORParameters, OperationalParameters, EconomicParameters
from core.engine_surrogate.surrogate_engine import SurrogateEngine
rd=ReservoirData(grid={'NX':np.array([50]),'NY':np.array([50]),'NZ':np.array([10])},pvt_tables={},
  ooip_stb=1e6,initial_pressure=3000.0,temperature=150.0,rock_compressibility=3e-6,
  average_porosity=0.2,initial_water_saturation=0.25,thickness_ft=50.0,area_acres=100.0,
  length_ft=2000.0,oil_fvf=1.2)
r=SurrogateEngine().evaluate_scenario(rd,EORParameters(),OperationalParameters(),economic_params=EconomicParameters())
pr=r['profiles']
dt=np.diff(np.asarray(pr['time_vector'],float),prepend=0.0)
print('RF reported         =', r.get('recovery_factor'))
print('cumulative_oil_stb  = {:,.0f}'.format(r.get('cumulative_oil_stb')))
print('OOIP*RF             = {:,.0f}'.format(1e6*r.get('recovery_factor')))
ind=float(np.sum(np.asarray(pr['oil_profile'],float)*dt))
print('int(oil_profile*dt) = {:,.0f}'.format(ind))
print('annual_oil_stb sum  = {:,.0f}'.format(float(np.sum(np.asarray(r['annual_oil_stb'],float)))))
print('monthly_oil_stb sum = {:,.0f}'.format(float(np.sum(np.asarray(pr['monthly_oil_stb'],float)))))
print('yearly_oil_stb sum  = {:,.0f}'.format(float(np.sum(np.asarray(r.get('yearly_oil_stb',[]),float)))))
print()
print('--- gas streams ---')
for k in ('gas_profile','co2_gas_profile','solution_gas_profile','hydrocarbon_gas_production_rate','total_gas_production_rate'):
    v=np.asarray(pr.get(k,[]),float)
    if v.size: print('  {:38} n={} sum={:,.0f} mean={:,.1f}'.format(k,v.size,v.sum(),v.mean()))
g=np.asarray(pr['gas_profile'],float); c=np.asarray(pr['co2_gas_profile'],float); s=np.asarray(pr['solution_gas_profile'],float)
print('  gas == co2+soln ?', bool(np.allclose(g,c+s)))
print('  co2_gas_profile ever == gas_profile ?', bool(np.allclose(c,g)))
print()
print('--- HC gas revenue keys ---')
for k in ('annual_hydrocarbon_gas_sales_mscf','hydrocarbon_gas_sales_mscfd','cumulative_hydrocarbon_gas_sales_mscf'):
    v=pr.get(k)
    if isinstance(v,(list,)): print('  {:38} n={} sum={:,.0f}'.format(k,len(v),float(np.sum(v))))
    else: print('  {:38} = {}'.format(k,v))
print()
print('--- leakage ---')
al=np.asarray(pr['annual_leakage_tonne'],float)
print('  annual_leakage_tonne all zero?', bool(np.all(al==0.0)))
print('  leakage_rate_tonnes_day max =', float(np.max(np.asarray(pr['leakage_rate_tonnes_day'],float))))
print('  caprock_tensile_margin min  =', float(np.min(np.asarray(pr['caprock_tensile_margin'],float))))
print('  caprock_shear_margin  min   =', float(np.min(np.asarray(pr['caprock_shear_margin'],float))))
print('  fault_slip_tendency   max   =', float(np.max(np.asarray(pr['fault_slip_tendency'],float))))
print()
print('--- water / VRR ---')
print('  water_cut max =', float(np.max(np.asarray(pr['water_cut_profile'],float))))
print('  vrr_local n=',np.asarray(pr['vrr_local'],float).size)