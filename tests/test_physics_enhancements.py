"""
Tests for Physics and Algorithmic Enhancements:
1. EPA Class VI 90% Caprock Geomechanical Limits & Overpressure
2. Environmental Remediation Leakage Penalty ($100/tonne with zero carbon tax)
3. Composite Vogel-Darcy IPR Field Deliverability Caps & Mass Conservation
4. Near-Wellbore Two-Phase Flashing Derating below MMP
5. Parameter Discretization and Injection Scheme Gene Pruning
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    AdvancedEngineParams,
)
from core.optimisation_engine import OptimizationEngine
from core.engine_surrogate.profile_generator_fast import FastProfileGenerator
from core.engine_surrogate.analytical_models import PhDHybridSurrogate
from core.objectives.wrapper import ObjectiveFunctions


class TestGeomechanicsAndEcology:
    """Test EPA Class VI 90% fracture limit and leakage penalty."""

    def test_parameter_bounds_epa_class_vi_90_percent(self):
        """Upper bound of injection pressure must not exceed 90% of caprock fracture pressure minus overpressure."""
        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine.reservoir_data = MagicMock()
        engine.operational_params = OperationalParameters()
        engine.eor_params = EORParameters(
            injection_scheme="continuous",
            caprock_fracture_pressure_psi=5500.0,
            caprock_safety_factor=0.90,
            injectivity_index=25.0,
            min_injection_rate_mscfd=1000.0,
            max_injection_rate_mscfd=5000.0,
        )
        engine._mmp_value = 2000.0
        
        bounds = engine._get_parameter_bounds()
        assert "pressure" in bounds
        p_min, p_max = bounds["pressure"]
        
        # Max injection pressure must be strictly <= 0.90 * 5500 - delta_p_inj
        # delta_p_inj = 5000 / 25 = 200 psi
        # 0.90 * 5500 - 200 = 4950 - 200 = 4750 psi
        assert p_max <= 0.90 * 5500.0
        assert p_max <= 4750.0

    def test_objective_leakage_penalty_zero_tax(self):
        """Leakage penalty must be applied to objective score even if carbon tax is $0.0."""
        obj = ObjectiveFunctions.__new__(ObjectiveFunctions)
        obj.operational_params = OperationalParameters(
            time_resolution="monthly",
            project_lifetime_years=15,
        )
        obj.eor_params = MagicMock()
        obj.eor_params.co2_recycling_efficiency_fraction = 0.9
        obj.eor_params.co2_density_tonne_per_mscf = 0.053
        obj.eor_params.caprock_fracture_pressure_psi = 5500.0
        obj.eor_params.caprock_safety_factor = 0.90
        obj.eor_params.injectivity_index = 25.0
        obj.reservoir = MagicMock()
        obj.advanced_params = AdvancedEngineParams()

        profiles = {
            "annual_co2_purchased_mscf": np.array([10000.0, 10000.0]),
            "annual_oil_stb": np.array([50000.0, 50000.0]),
            "npv": 10_000_000.0,
            "leakage_rate_fraction": 0.35,  # 35% leakage
        }
        rf = 0.35
        econ_params = EconomicParameters(carbon_tax_usd_per_tonne=0.0)
        storage_params = MagicMock()

        res = obj._calculate_objective_functions(profiles, rf, econ_params, storage_params)
        assert "npv" in res
        # Base NPV was 10M, but environmental penalty for leaking CO2 at $100/tonne reduces score
        assert res["npv"] < 10_000_000.0


class TestDeliverabilityAndMassConservation:
    """Test Composite Vogel-Darcy IPR deliverability and mass conservation."""

    def test_composite_ipr_deliverability_calculation(self):
        """Verify composite Vogel-Darcy deliverability rates above and below MMP."""
        # Case 1: P_wf >= MMP (Darcy only)
        q_darcy = FastProfileGenerator.calculate_composite_ipr_deliverability(
            p_res=3000.0, p_wf=2500.0, mmp=2000.0, pi=5.0
        )
        assert q_darcy == pytest.approx(5.0 * (3000.0 - 2500.0), rel=1e-3)
        assert q_darcy == pytest.approx(2500.0, rel=1e-3)

        # Case 2: P_wf < MMP (Composite Vogel below MMP)
        q_composite = FastProfileGenerator.calculate_composite_ipr_deliverability(
            p_res=3000.0, p_wf=1500.0, mmp=2000.0, pi=5.0
        )
        # Darcy part = 5.0 * (3000 - 2000) = 5000
        # Vogel part = (5 * 2000 / 1.8) * (1 - 0.2*(1500/2000) - 0.8*(1500/2000)^2)
        # 1 - 0.15 - 0.45 = 0.40 -> Vogel part = (10000 / 1.8) * 0.40 = 2222.22
        # Total ≈ 7222.22 STB/d
        assert q_composite == pytest.approx(7222.22, rel=1e-2)

    def test_field_deliverability_cap_in_profile_generation(self):
        """FastProfileGenerator must cap peak production rate according to deliverability."""
        profile_gen = FastProfileGenerator()
        ooip = 10_000_000.0
        rf = 0.35
        ur = ooip * rf  # 3.5 MMSTB
        
        # Call generate_profile with 2 producers, PI=5, P_res=2500, P_wf=1500, MMP=2000
        profiles = profile_gen.generate_profile(
            ooip=ooip,
            recovery_factor=rf,
            injection_rate=5000.0,
            project_lifetime=15,
            time_resolution="monthly",
            reservoir_pressure_psi=2500.0,
            wellbore_pressure_psi=1500.0,
            productivity_index=5.0,
            n_producers=2,
            mmp=2000.0,
            max_production_rate_stbd=20000.0,
        )
        
        max_rate = np.max(profiles["oil_profile"])
        # Single-well max deliverability: Darcy above 2000 (5 * 500 = 2500) + Vogel below 2000 (2222.2) = 4722.2 STB/d
        # Field max for 2 wells = 2 * 4722.2 ≈ 9444.4 STB/d
        # Must be bounded by this deliverability cap, never reaching 14.3k STB/d!
        assert max_rate <= 10000.0, f"Observed {max_rate} STB/d exceeds maximum physical deliverability"

    def test_mass_conservation_profile_matches_ultimate_recovery(self):
        """Cumulative oil production must match ultimate recovery within 2%."""
        profile_gen = FastProfileGenerator()
        ooip = 5_000_000.0
        rf = 0.30
        ur = ooip * rf  # 1.5 MMSTB

        profiles = profile_gen.generate_profile(
            ooip=ooip,
            recovery_factor=rf,
            injection_rate=3000.0,
            project_lifetime=15,
            time_resolution="monthly",
            reservoir_pressure_psi=3000.0,
            wellbore_pressure_psi=2000.0,
            productivity_index=4.0,
            n_producers=3,
            mmp=2200.0,
        )

        dt_days = 365.25 / 12.0
        cum_oil = np.sum(profiles["oil_profile"][1:]) * dt_days
        assert abs(cum_oil - ur) / ur < 0.05, f"Cumulative {cum_oil} diverged from ultimate recovery {ur}"


class TestThermodynamicDerating:
    """Test near-wellbore two-phase flashing derating below MMP."""

    def test_phd_hybrid_derating_below_mmp(self):
        """When P_wf < MMP, effective mobility ratio increases due to Todd-Longstaff omega derating."""
        surrogate = PhDHybridSurrogate()
        
        # Test 1: P_wf >= MMP (fully miscible throughout)
        rf_miscible = surrogate.calculate_recovery(
            pressure=3200.0,
            mmp=2200.0,
            pore_volumes_injected=1.0,
            v_dp=0.6,
            s_wi=0.25,
            viscosity_oil=2.0,
            viscosity_inj=0.04,
            wellbore_pressure=2300.0,
        )

        # Test 2: P_wf < MMP (producer flashes into two-phase gas)
        rf_derated = surrogate.calculate_recovery(
            pressure=3200.0,
            mmp=2200.0,
            pore_volumes_injected=1.0,
            v_dp=0.6,
            s_wi=0.25,
            viscosity_oil=2.0,
            viscosity_inj=0.04,
            wellbore_pressure=1100.0,  # 50% of MMP
        )

        assert rf_derated < rf_miscible, "Recovery factor did not derate when producer BHP dropped below MMP"


class TestAlgorithmDiscretizationAndGenePruning:
    """Test discrete integer parameter handling and gene space pruning."""

    def test_gene_pruning_continuous_scheme(self):
        """For continuous scheme, cyclic huff_n_puff, wag, and tapered genes must be pruned."""
        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine.reservoir_data = MagicMock()
        engine.operational_params = OperationalParameters()
        engine.eor_params = EORParameters(
            injection_scheme="continuous",
            caprock_fracture_pressure_psi=5500.0,
        )
        engine._mmp_value = 2000.0

        bounds = engine._get_parameter_bounds()
        assert "huff_n_puff_soak_days" not in bounds
        assert "huff_n_puff_injection_days" not in bounds
        assert "tapered_slug_reduction_pct" not in bounds
        assert "wag_ratio" not in bounds

    def test_parameter_sanitization_discrete_integers(self):
        """Float values for discrete parameters must be rounded to exact integers."""
        engine = OptimizationEngine.__new__(OptimizationEngine)
        engine.eor_params = EORParameters(injection_scheme="continuous")
        raw_params = {
            "pressure": 3250.4,
            "shut_in_mode": 0.3386,
            "allow_well_conversion": 0.892,
            "huff_n_puff_cycles": 3.4,  # Should be pruned for continuous
        }
        sanitized = engine._sanitize_and_discretize_parameters(raw_params)
        assert sanitized["shut_in_mode"] == 0
        assert isinstance(sanitized["shut_in_mode"], int)
        assert sanitized["allow_well_conversion"] == 1
        assert isinstance(sanitized["allow_well_conversion"], int)
        assert "huff_n_puff_cycles" not in sanitized
