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
print('=== monthly_oil_stb ===')
m=np.asarray(pr['monthly_oil_stb'],float); y=np.asarray(r['yearly_oil_stb'],float)
print('n=',m.size,'all zero?',bool(np.all(m==0.0)),'sum=',m.sum())
print('yearly_oil_stb n=',y.size,'sum=',y.sum())
print()
print('=== HC gas revenue vs NPV ===')
hg=np.asarray(pr['annual_hydrocarbon_gas_sales_mscf'],float)
print('annual_hydrocarbon_gas_sales_mscf sum = {:,.0f} MSCF'.format(hg.sum()))
print('npv = ${:,.0f}'.format(r.get('npv',0.0)))
print('annual_cashflow_usd sum = ${:,.0f}'.format(float(np.sum(np.asarray(pr['annual_cashflow_usd'],float)))))
print('NPV omits HC gas sales: stream exists in profiles but is not a revenue term in the NPV block.')
print()
print('=== CO2 ledger closure ===')
inj=r['cumulative_co2_injected_mscf']; prod=r['cumulative_co2_produced_mscf']; sto=r['cumulative_co2_stored_mscf']
print('injected  = {:,.1f}'.format(inj))
print('produced  = {:,.1f}'.format(prod))
print('stored    = {:,.1f}'.format(sto))
print('prod+sto  = {:,.1f}'.format(prod+sto))
print('closure err = {:.3e} MSCF'.format(abs(inj-(prod+sto))))
print('purch+recyc = {:,.1f}'.format(r['cumulative_co2_purchased_mscf']+r['cumulative_co2_recycled_mscf']))
print('recycled<=produced?', r['cumulative_co2_recycled_mscf']<=r['cumulative_co2_produced_mscf'])
print()
print('=== saturation closure ===')
so=np.asarray(pr['saturation_oil'],float); sw=np.asarray(pr['saturation_water'],float); sg=np.asarray(pr['saturation_gas'],float)
bad=so+sw
print('timesteps So+Sw>1: {}/{} ({:.1f}%)  max={:.6f}'.format(int((bad>1).sum()),len(so),100*np.mean(bad>1),bad.max()))
print('=> Sg silently zeroed on those steps; total S = {:.6f}..{:.6f}'.format((so+sw+sg).min(),(so+sw+sg).max()))
print()
print('=== leakage / containment ===')
print('total_leakage_tonne =',r['total_leakage_tonne'])
print('leakage_rate max =',float(np.max(np.asarray(pr['leakage_rate_tonnes_day'],float))))
print('max_sandface_pressure_psi =',pr['max_sandface_pressure_psi'])
p=np.asarray(pr['pressure'],float)
print('reservoir pressure max = {:,.1f} (clipped to p_safe_ceiling=3555)'.format(p.max()))