# Optimization Run Technical Evaluation Report
**Run Identifier:** `Genetic Algorithm (GA)-2026-09-24T10:44:22`
**Generated:** 2026-09-24 10:44:22 | **Evaluations:** 0 | **Duration:** 0.0s

## 1. Executive Summary & KPIs

| Metric | Value | Unit | Engineering Reference / Benchmark |
| :--- | :---: | :---: | :--- |
| **Net Present Value (NPV)** | **$192,933,877.74** | USD | Discounted Cash Flow at 10% discount rate |
| **Optimizer Fitness Score** | `1.9293e+08` | dimensionless | Internal objective value used by optimizer |
| **Breakthrough Penalty Factor** | `0.5102` | factor | Calculated timing metric (no penalty applied to fitness) |
| **Recovery Factor (RF)** | **27.27%** | % OOIP | Expected literature range: 30% - 65% |
| **Cumulative Oil Produced** | **13,223,226** | STB | Total tertiary oil volume |
| **Cumulative Water Produced** | 442,087 | STB | Total brine produced |
| **Cumulative Total Gas Produced** | 8,321,154 | MSCF | Separator total off-gas |
| **Cumulative CO₂ Produced** | 5,272,511 | MSCF | Produced CO₂ stream to recycling plant |
| **Net CO₂ Utilization** | **0.090** | tonne/STB | Standard EOR range: 0.25 - 0.50 tonne/bbl |
| **CO₂ Storage Retention (Purchased)** | **98.6%** | % purchased | DOE/NETL benchmark: > 80% |
| **CO₂ Storage Retention (Gross)** | 50.2% | % injected | Includes recycled stream in denominator |
| **Breakthrough Time** | **2.45** | years | Time to solvent breakthrough at producers |
| **Class VI Ecology Compliant** | ✅ YES | boolean | Geomechanical & containment criteria |

## 2. Decision Variables (Optimized Operating Policy)

| Parameter | Optimal Value | Physical Unit | Search Bounds | Boundary Status |
| :--- | :---: | :---: | :---: | :---: |
| `max_production_rate_stbd` | **10000.0000** | STB/day | [1000.00, 15000.00] | Interior |
| `plateau_duration_fraction` | **0.3000** | dimensionless (0-1) | [0.10, 0.80] | Interior |
| `pressure` | **3000.0000** | psia | [3242.97, 4450.00] | Interior |
| `ramp_up_fraction` | **0.1000** | dimensionless (0-1) | [0.00, 0.30] | Interior |
| `rate` | **5000.0000** | MSCF/day | [5000.00, 100000.00] | ⚠️ **At Lower Bound** |
| `wellbore_pressure` | **1500.0000** | psia | [1500.00, 2000.00] | ⚠️ **At Lower Bound** |

## 3. Geomechanical & Subsurface Safety Checklist

| Criterion | Evaluated Value | Limit / Standard | Status |
| :--- | :---: | :---: | :---: |
| **Reservoir Injection Pressure** | **3000.0 psi** | ≤ 4950.0 psi (90% Pfrac) | ✅ COMPLIANT |
| **Fracture Safety Margin** | 1950.0 psi | > 100 psi recommended | ✅ SAFE |
| **Formation Fracture Pressure** | 5500.0 psi | Caprock threshold | Baseline |
| **Material Balance Closure** | **100.00%** | ≥ 99.9% conservation | ✅ CLOSED |
| **Modeled Subsurface Leakage** | 892.2 tonnes | < 1.0% of injection | ✅ SECURE |

## 4. Carbon Mass Balance & Storage Accounting

| Accounting Stream | Mass (Metric Tonnes) | Volume Equivalent (MSCF @ 0.053 t/MSCF) | Fraction of Gross Injected |
| :--- | :---: | :---: | :---: |
| **Gross Injected CO₂** | **1,451,868.7** | 27,393,750 | 100.0% |
| ├─ *Purchased Fresh CO₂* | *1,186,397.8* | 22,384,864 | 81.7% |
| └─ *Recycled Re-injected CO₂* | *265,471.0* | 5,008,886 | 18.3% |
| **Net Permanent Subsurface Storage** | **1,171,533.4** | 22,104,405 | 80.7% |
| **Total Produced CO₂** | **279,443.1** | 5,272,511 | 19.2% |
| ├─ *Recycled Stream* | *265,471.0* | 5,008,886 | 18.3% |
| └─ *Uncaptured / Lost to Surface* | *13,972.2* | 263,626 | 0.96% |
| **Modeled Fault/Caprock Leakage** | 892.2 | 16,834 | 0.06% |
| **Unaccounted Closure Error** | 0.0 | 0 | 0.00% |

## 5. Reservoir & Fluid Characterization Context

### 5.1 Reservoir Rock Properties

| Property | Value | Unit | Engineering Relevance |
| :--- | :---: | :---: | :--- |
| **Original Oil in Place (OOIP)** | 48,487,500 | STB | Baseline field hydrocarbon pore volume |
| **Minimum Miscibility Pressure (MMP)** | 2948.2 | psia | Pure CO₂ thermodynamic miscibility threshold |
| **Operating Miscibility Regime** | **Miscible** | status | Displacement mode (Miscible / Immiscible) |
| **Average Porosity** | 20.0% | fraction | Reservoir storage capacity |
| **Average Permeability** | 100.0 | mD | Fluid transmissibility |
| **Initial Reservoir Pressure** | 4000.0 | psia | Discovery datum pressure |
| **Reservoir Temperature** | 212.0 | °F | Formation thermal regime |
| **Formation Net Thickness** | 50.0 | ft | Net pay zone thickness |
| **Reservoir Area** | 1000.0 | acres | Drainage area footprint |
| **Well Infrastructure** | 1 wells (0 inj / 1 prod) | count | Field pattern well arrangement |

### 5.2 Fluid & PVT Properties

| Property | Value | Unit | Engineering Relevance |
| :--- | :---: | :---: | :--- |
| **Oil Viscosity** | 1 | cp | Dead/live oil dynamic viscosity |
| **Water Viscosity** | 0.5 | cp | Formation brine viscosity |
| **Gas Viscosity** | 0.02 | cp | Hydrocarbon gas viscosity |

### 5.3 Operational & Well Constraints

| Parameter | Value | Unit | Engineering Relevance |
| :--- | :---: | :---: | :--- |
| **Project Lifetime** | 15 | years | Economic horizon |
| **Time Resolution** | yearly | mode | Yearly / Monthly simulation stepping |

---
*Report generated by CO₂-EOR Optimizer.*