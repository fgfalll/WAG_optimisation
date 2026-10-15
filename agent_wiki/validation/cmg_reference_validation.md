# CMG GEM Reference Validation Framework

> **Status**: Active  
> **Test module**: tests/test_cmg_surrogate_validation.py  
> **CMG reference files**: tests/validation/cmg/flu/gmflu002_1D.out, gmflu002.out

---

## Why This Validation Exists

The CO2 EOR Optimizer surrogate engine is a **fast, physics-bounded analytical model** designed for optimization not a numerical compositional simulator. To ensure agents can detect physics nonsense without running 30-minute full optimizations, this standalone test suite runs the surrogate with exact SPE5 Wasson CO2 Flood parameters and asserts scientifically-defensible bounds against real CMG GEM 2024.20 reference outputs.

**Key principle**: The surrogate cannot reproduce compositional EOS flash thermodynamics. Tests assert **physical coherence** (bounded, directionally correct, consistent), not numerical replication.

---

## CMG GEM Reference Data

All values extracted directly from the .out files DO NOT modify without re-reading the files.

### SPE5 1D Case (gmflu002_1D.dat)
Grid: 10x1x1 CART, 500x500x50 ft blocks
Rock: por=0.30, k=200 mD, Cpor=5e-6 /psi
Fluid: Wasson oil (11 components, EOS PR), T=90 degF, Pi=1118.8 psia
Initial: Swi=0.20, So=0.80 (no free gas)
Injection: CO2 at 12,000 MSCFD (12 MM SCF/day), continuous
Production: BHP min = 1,000 psia; stopped at GOR=10,000 SCF/STB

| Metric | Value |
|--------|-------|
| OOIP | 4,763,700 STB |
| RF | 76.002% |
| Cum Oil | 3,620,500 STB |
| Cum Inj CO2 | 10,103,000 MSCF |
| Net CO2 Util | ~2.79 MSCF/STB |
| P_final | 1,504.1 psia |
| Duration | 841.9 days (2.30 yr) |
| Mat. Bal. Error | 0.00506% |

### SPE5 3D Case (gmflu002.dat)
Grid: 7x7x3 CART, k_layers=200/50/500 mD, DK=50/30/20 ft

| Metric | Value |
|--------|-------|
| OOIP | 46,680,000 STB |
| RF | 32.118% |
| Cum Oil | 14,993,000 STB |
| Cum Inj CO2 | 35,064,000 MSCF |
| HCPV Injected | 36.20% |
| P_final | 1,231.7 psia |
| Duration | 2,922 days (8.00 yr) |
| Mat. Bal. Error | 0.000806% |

---

## Running the Tests

`ash
# Run all CMG validation tests (fast, ~0.4 seconds total)
python -m pytest tests/test_cmg_surrogate_validation.py -v --no-cov

# Run only miscibility (omega) tests
python -m pytest tests/test_cmg_surrogate_validation.py -k omega -v
`

Expected result: 19 passed, 2 warnings in 0.42s

---

## Physical Invariants Asserted

1. RF > 0 and <= (1 - Swi - Sor)
2. Mean pressure >= 80% of Pi under active injection
3. Max pressure <= 0.90 * Pfrac (EPA Class VI UIC)
4. Carbon balance error < 1% (CMG achieves 0.005%)
5. Gross utilization in [1, 100] MSCF/STB
6. Evaluation time < 50 ms

---

## PhD Novelty: Miscibility Dictation Invariants

1. P(omega) -> omega round-trips within 1e-4 (exact analytical inverse)
2. omega strictly monotone increasing with pressure
3. omega(P=MMP) = 0.5 (sigmoid center)
4. omega(P << MMP) < 0.05 (immiscible)
5. omega(P >> MMP) > 0.95 (miscible)
6. Miscible RF >= Immiscible RF (fundamental EOR physics)

---

## Gravity Number Unit Fix (2026-09-17)

The corrected Ng field-unit conversion factor is 4.3948e-5, derived as:
`
Ng = k[mD] * delta_rho[lb/ft3] * sin(theta) * 4.3948e-5 / (mu[cP] * u[ft/day])
`

Prior defect: factor was 2.4e11, giving Ng ~ 2.4e9 for typical cases, 
crushing ev = 1/(1+Ng) to the 0.10 floor regardless of dip angle.

With the correct factor, Ng << 1 for horizontal reservoirs, giving ev ~ 1.

---

## Bg Unit Convention

After the Bg bug fixes (2026-09-17):

| Symbol | Units | Typical Value |
|--------|-------|---------------|
| mscf_per_res_bbl | MSCF/RB | ~0.5 for SC CO2 |
| bg_rb_per_mscf | RB/MSCF | ~2.0 for SC CO2 |
| bg (gas_profile) | RB/SCF | ~0.002 RB/SCF |
| injection_profile | MSCFD | from FastProfileGenerator |
| q_inj_rb | RB/day | injection_profile * bg_rb_per_mscf |

CRITICAL: Default for mscf_per_res_bbl must be 0.5, not 500.0.
The old default of 500 was a 1000x error.

---

## Known Limitations vs. CMG

1. RF accuracy: Surrogate analytical; CMG uses 11-component EOS. 76% CMG RF for 1D
   near-ideal conditions cannot be reproduced analytically (expected range: 10-60%).
2. Phase behavior: No multi-contact miscibility (MCM) mechanism.
3. GOR constraint: CMG stops at GOR > 10,000 SCF/STB. Surrogate runs full project life.
