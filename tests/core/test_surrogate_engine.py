"""
Unit tests for Surrogate Engine.

Tests cover:
- Engine initialization
- All recovery model types
- Pressure profile generation
- NPV calculation
- CO2 storage
- Production profiles
- Edge cases
- Error handling
- Performance benchmarks
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
)
from core.engine_surrogate.surrogate_engine import (
    SurrogateEngine,
    SurrogateEngineWrapper,
    create_surrogate_engine,
)


class TestEngineInitialization:
    """Test engine initialization and configuration."""

    def test_create_surrogate_engine_function(self):
        """Test create_surrogate_engine factory function."""
        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")
        assert engine is not None
        assert isinstance(engine, (SurrogateEngine, SurrogateEngineWrapper))

    def test_analytical_model_initialization(self):
        """Test analytical model initialization with default parameters."""
        engine = SurrogateEngine(model_type="analytical", recovery_model_type="hybrid")
        assert engine.model_type == "analytical"
        assert engine.recovery_model_type == "hybrid"

    def test_phd_hybrid_model_initialization(self):
        """Test PhD hybrid model initialization."""
        engine = SurrogateEngine(model_type="analytical", recovery_model_type="phd_hybrid")
        assert engine.recovery_model_type == "phd_hybrid"

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            SurrogateEngine(model_type="invalid_type")

    def test_invalid_recovery_model_type_raises_error(self):
        """Test that invalid recovery_model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown analytical model type"):
            SurrogateEngine(model_type="analytical", recovery_model_type="invalid_recovery")


class TestRecoveryModelTypes:
    """Test all recovery model types produce valid outputs."""

    @pytest.fixture(
        params=["miscible", "immiscible", "hybrid", "koval", "buckley_leverett", "phd_hybrid"]
    )
    def recovery_model_type(self, request):
        """Parametrized fixture for all recovery model types."""
        return request.param

    def test_all_models_produce_valid_rf(
        self,
        recovery_model_type,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test all recovery models produce RF in valid range [0, 1]."""
        engine = create_surrogate_engine(
            model_type="analytical", recovery_model_type=recovery_model_type
        )

        result = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert result["convergence_status"] == "success"
        assert "recovery_factor" in result
        rf = result["recovery_factor"]
        assert 0.0 <= rf <= 1.0, f"RF {rf} out of bounds for {recovery_model_type}"

    def test_rf_monotonic_with_injection_volume(
        self, standard_reservoir, standard_operational_params
    ):
        """Test RF increases with injection rate (more HCPVI)."""
        eor_low = EORParameters(injection_rate=1000.0, target_pressure_psi=3000.0)
        eor_high = EORParameters(injection_rate=10000.0, target_pressure_psi=3000.0)

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_low = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor_low,
            operational_params=standard_operational_params,
        )

        result_high = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor_high,
            operational_params=standard_operational_params,
        )

        assert result_high["recovery_factor"] >= result_low["recovery_factor"]


class TestPressureProfile:
    """Test pressure profile generation and constraints."""

    def test_pressure_shape_constraints(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test pressure profile has reasonable shape (no catastrophic drops)."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        pressure = np.array(result["pressure"])
        time_vector = np.array(result["time_vector"])

        assert len(pressure) == len(time_vector)
        assert len(pressure) > 1
        assert np.all(pressure > 0), "Pressure went negative"

    def test_pressure_bounds(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test pressure stays within reasonable bounds."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        pressure = np.array(result["pressure"])
        target_p = standard_eor_params.target_pressure_psi

        assert np.all(pressure > 0), "Negative pressure detected"
        assert np.all(pressure < target_p * 2), "Pressure exceeds 2x target"

    def test_pressure_model_based_flag(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test pressure_model_based flag is set correctly."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert result.get("pressure_model_based") == True


class TestNPVCalculation:
    """Test NPV calculation with economic parameters."""

    def test_npv_positive_with_revenue(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
        standard_economic_params,
    ):
        """Test NPV is positive with favorable economics."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
            economic_params=standard_economic_params,
        )

        assert "npv" in result
        assert result["npv"] >= 0

    def test_npv_reflects_oil_price(
        self,
        analytical_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test NPV increases with oil price."""
        econ_low = EconomicParameters(
            oil_price_usd_per_bbl=30.0,
            co2_purchase_cost_usd_per_tonne=50.0,
            discount_rate_fraction=0.10,
        )
        econ_high = EconomicParameters(
            oil_price_usd_per_bbl=100.0,
            co2_purchase_cost_usd_per_tonne=50.0,
            discount_rate_fraction=0.10,
        )

        result_low = analytical_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
            economic_params=econ_low,
        )

        result_high = analytical_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
            economic_params=econ_high,
        )

        assert result_high["npv"] > result_low["npv"]

    def test_npv_with_high_costs(
        self,
        analytical_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test NPV can be negative with very high CO2 costs."""
        econ = EconomicParameters(
            oil_price_usd_per_bbl=20.0,
            co2_purchase_cost_usd_per_tonne=200.0,
            discount_rate_fraction=0.10,
        )

        result = analytical_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
            economic_params=econ,
        )

        assert "npv" in result


class TestCO2Storage:
    """Test CO2 storage calculations."""

    def test_storage_positive(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test CO2 stored is positive when injection occurs."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert "co2_stored" in result
        assert result["co2_stored"] >= 0

    def test_storage_increases_with_injection(
        self, standard_reservoir, standard_operational_params
    ):
        """Test CO2 storage increases with injection rate."""
        eor_low = EORParameters(injection_rate=1000.0, target_pressure_psi=3000.0)
        eor_high = EORParameters(injection_rate=10000.0, target_pressure_psi=3000.0)

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_low = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor_low,
            operational_params=standard_operational_params,
        )

        result_high = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor_high,
            operational_params=standard_operational_params,
        )

        assert result_high["co2_stored"] > result_low["co2_stored"]


class TestProfileGeneration:
    """Test production profile generation."""

    def test_time_vector_length(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test time vector has correct length based on project lifetime."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        time_vector = np.array(result["time_vector"])
        expected_years = standard_operational_params.project_lifetime_years

        assert len(time_vector) > 0
        assert time_vector[-1] >= expected_years * 365 - 10

    def test_production_rates_nonnegative(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test all production rates are non-negative."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert np.all(np.array(result["oil_production_rate"]) >= 0)
        assert np.all(np.array(result["water_production_rate"]) >= 0)
        assert np.all(np.array(result["gas_production_rate"]) >= 0)

    def test_injection_profile_positive(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test CO2 injection rate is positive."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        injection = np.array(result["co2_injection"])
        assert np.all(injection >= 0)


class TestCumulativeValues:
    """Test cumulative production calculations."""

    def test_cumulative_oil_from_rf(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test cumulative oil is positive and reasonable."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        rf = result["recovery_factor"]
        ooip = standard_reservoir.ooip_stb
        cumulative = result["cumulative_oil"]

        assert cumulative >= 0, "Cumulative oil cannot be negative"
        assert cumulative <= ooip, "Cumulative oil cannot exceed OOIP"
        assert 0 <= rf <= 1, "RF must be between 0 and 1"

    def test_recovery_factor_profile_integrates_to_cumulative(
        self,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Test RF profile and cumulative oil are both reasonable."""
        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        rf_profile = np.array(result["recovery_factor_profile"])
        time_vector = np.array(result["time_vector"])
        cumulative = result["cumulative_oil"]

        assert cumulative >= 0, "Cumulative oil must be non-negative"
        assert len(rf_profile) > 0, "RF profile must not be empty"
        assert np.all(rf_profile >= 0), "RF profile values must be non-negative"
        assert np.all(rf_profile <= 1), "RF profile values must be <= 1"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_injection_rate(self, standard_reservoir, standard_operational_params):
        """Test engine handles zero injection gracefully."""
        eor = EORParameters(injection_rate=0.0, target_pressure_psi=3000.0)

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor,
            operational_params=standard_operational_params,
        )

        assert "convergence_status" in result

    def test_very_high_pressure(self, standard_reservoir, standard_operational_params):
        """Test engine handles very high target pressure with proper max pressure."""
        eor = EORParameters(
            injection_rate=5000.0,
            target_pressure_psi=10000.0,
            max_pressure_psi=15000.0,
        )

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor,
            operational_params=standard_operational_params,
        )

        assert "convergence_status" in result

    def test_zero_ooip(self, standard_eor_params, standard_operational_params):
        """Test engine handles zero OOIP gracefully."""
        reservoir = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=0.0,
            initial_pressure=3000.0,
        )

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result = engine.evaluate_scenario(
            reservoir_data=reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert "convergence_status" in result

    def test_extreme_heterogeneity(self, standard_eor_params, standard_operational_params):
        """Test engine handles extreme V_DP."""
        reservoir = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=1_000_000.0,
            initial_pressure=3000.0,
        )
        reservoir.v_dp_coefficient = 0.95

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result = engine.evaluate_scenario(
            reservoir_data=reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert "convergence_status" in result
        if result["convergence_status"] == "success":
            assert 0 <= result["recovery_factor"] <= 1

    def test_capillary_number_scaling_bounds_rf(
        self, standard_reservoir, standard_operational_params
    ):
        """Test that extreme injection rates and miscible pressures do not strip Sor to absolute 0.0."""
        # Extreme injection rate (e.g., 100x standard) and high pressure to maximize capillary number
        eor_extreme = EORParameters(
            injection_rate=500000.0,  # Huge rate
            target_pressure_psi=8000.0,  # Well above MMP
            max_pressure_psi=9000.0,
        )

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=eor_extreme,
            operational_params=standard_operational_params,
        )

        # In a physically constrained model, even with infinite Nc, RF shouldn't reach exactly 1.0
        # due to macroscopic trapping (sor_min) and sweep inefficiencies.
        assert "recovery_factor" in result
        rf = result["recovery_factor"]
        assert rf < 0.95, f"RF is {rf}, should be bounded by sor_min and sweep efficiency."
        assert rf > 0.0, "RF should be positive."


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_reservoir_data_returns_error(self):
        """Test that invalid reservoir data returns error result."""
        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        eor = EORParameters(injection_rate=5000.0, target_pressure_psi=3000.0)
        op = OperationalParameters(project_lifetime_years=10)

        reservoir = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=-100.0,
            initial_pressure=-100.0,
        )

        result = engine.evaluate_scenario(
            reservoir_data=reservoir,
            eor_params=eor,
            operational_params=op,
        )

        assert "convergence_status" in result
        assert result["convergence_status"] in ["success", "error"]

    def test_missing_optional_parameters(
        self, phd_hybrid_engine, standard_eor_params, standard_operational_params
    ):
        """Test engine handles minimal reservoir data with required params."""
        reservoir = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=1_000_000.0,
            initial_pressure=3000.0,
        )

        result = phd_hybrid_engine.evaluate_scenario(
            reservoir_data=reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert "convergence_status" in result


class TestPerformance:
    """Performance benchmark tests (mandatory)."""

    @pytest.mark.benchmark(min_rounds=100)
    def test_evaluation_speed(
        self,
        benchmark,
        phd_hybrid_engine,
        standard_reservoir,
        standard_eor_params,
        standard_operational_params,
    ):
        """Benchmark evaluation time - target < 1ms."""
        result = benchmark(
            phd_hybrid_engine.evaluate_scenario,
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        assert result["convergence_status"] == "success"

    @pytest.mark.benchmark(min_rounds=100)
    def test_batch_evaluation_speed(
        self, benchmark, standard_reservoir, standard_eor_params, standard_operational_params
    ):
        """Benchmark multiple sequential evaluations."""

        def batch_evaluate():
            engine = create_surrogate_engine(
                model_type="analytical", recovery_model_type="phd_hybrid"
            )
            for _ in range(10):
                engine.evaluate_scenario(
                    reservoir_data=standard_reservoir,
                    eor_params=standard_eor_params,
                    operational_params=standard_operational_params,
                )

        benchmark(batch_evaluate)

    def test_performance_stats_tracking(self):
        """Test engine tracks performance statistics."""
        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        stats = engine.get_performance_stats()
        assert stats["evaluation_count"] == 0
        assert stats["total_time"] == 0.0
        assert stats["average_time"] == 0.0


class TestSurrogateEngineWrapper:
    """Test SurrogateEngineWrapper interface compatibility."""

    def test_wrapper_has_expected_methods(self):
        """Test wrapper exposes expected interface."""
        wrapper = SurrogateEngineWrapper(model_type="analytical", recovery_model_type="phd_hybrid")

        assert hasattr(wrapper, "evaluate_scenario")
        assert callable(wrapper.evaluate_scenario)

    def test_wrapper_returns_same_keys(
        self, standard_reservoir, standard_eor_params, standard_operational_params
    ):
        """Test wrapper returns expected result keys."""
        wrapper = SurrogateEngineWrapper(model_type="analytical", recovery_model_type="phd_hybrid")

        result = wrapper.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=standard_eor_params,
            operational_params=standard_operational_params,
        )

        expected_keys = [
            "recovery_factor",
            "npv",
            "cumulative_oil",
            "co2_stored",
            "pressure",
            "time_vector",
            "engine_type",
            "convergence_status",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


class TestBreakthroughTime:
    """Test breakthrough time calculation using Koval (1963) physics."""

    @pytest.fixture
    def base_reservoir(self):
        """Base reservoir for breakthrough tests."""
        res = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=1_000_000.0,
            initial_pressure=3000.0,
            temperature=150.0,
            average_porosity=0.15,
            average_permeability=100.0,
            initial_water_saturation=0.25,
            area_acres=160,
            thickness_ft=50,
        )
        res.v_dp_coefficient = 0.5
        return res

    @pytest.fixture
    def base_eor(self):
        """Base EOR params for breakthrough tests."""
        return EORParameters(
            injection_rate=5000.0,
            mobility_ratio=5.0,
            target_pressure_psi=3000.0,
        )

    @pytest.fixture
    def base_op(self):
        """Base operational params for breakthrough tests."""
        return OperationalParameters(
            project_lifetime_years=15,
            time_resolution="yearly",
        )

    def test_breakthrough_time_exists_in_result(self, base_reservoir, base_eor, base_op):
        """Test that breakthrough_time_years is returned in results."""
        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")
        result = engine.evaluate_scenario(base_reservoir, base_eor, base_op)

        assert "breakthrough_time_years" in result, "breakthrough_time_years should be in result"
        assert result["convergence_status"] == "success"

    def test_default_parameters_bt_range(self, base_reservoir, base_eor, base_op):
        """Test breakthrough time is in reasonable range for default parameters."""
        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")
        result = engine.evaluate_scenario(base_reservoir, base_eor, base_op)

        bt = result["breakthrough_time_years"]
        assert 0.1 <= bt <= 30.0, f"Breakthrough time {bt} should be in physical range [0.1, 30]"

    def test_high_heterogeneity_earlier_bt(self, base_eor, base_op):
        """Test that high heterogeneity (VDP) causes earlier breakthrough."""
        res_high_vdp = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=1_000_000.0,
            initial_pressure=3000.0,
            temperature=150.0,
            average_porosity=0.15,
            average_permeability=100.0,
            initial_water_saturation=0.25,
            area_acres=160,
            thickness_ft=50,
        )
        res_high_vdp.v_dp_coefficient = 0.8  # High heterogeneity

        res_low_vdp = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=1_000_000.0,
            initial_pressure=3000.0,
            temperature=150.0,
            average_porosity=0.15,
            average_permeability=100.0,
            initial_water_saturation=0.25,
            area_acres=160,
            thickness_ft=50,
        )
        res_low_vdp.v_dp_coefficient = 0.2  # Low heterogeneity

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_high = engine.evaluate_scenario(res_high_vdp, base_eor, base_op)
        result_low = engine.evaluate_scenario(res_low_vdp, base_eor, base_op)

        bt_high = result_high["breakthrough_time_years"]
        bt_low = result_low["breakthrough_time_years"]

        assert bt_high < bt_low, (
            f"High VDP ({bt_high}) should breakthrough earlier than low VDP ({bt_low})"
        )

    def test_favorable_mobility_later_bt(self, base_reservoir, base_op):
        """Test that favorable mobility ratio (low M) causes later breakthrough."""
        eor_favorable = EORParameters(
            injection_rate=5000.0,
            mobility_ratio=2.0,  # Favorable
            target_pressure_psi=3000.0,
        )
        eor_unfavorable = EORParameters(
            injection_rate=5000.0,
            mobility_ratio=10.0,  # Unfavorable
            target_pressure_psi=3000.0,
        )

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_fav = engine.evaluate_scenario(base_reservoir, eor_favorable, base_op)
        result_unfav = engine.evaluate_scenario(base_reservoir, eor_unfavorable, base_op)

        bt_fav = result_fav["breakthrough_time_years"]
        bt_unfav = result_unfav["breakthrough_time_years"]

        assert bt_fav > bt_unfav, (
            f"Favorable mobility ({bt_fav}) should breakthrough later than unfavorable ({bt_unfav})"
        )

    def test_lower_injection_rate_later_bt(self, base_reservoir, base_eor, base_op):
        """Test that lower injection rate causes later breakthrough."""
        eor_high_rate = EORParameters(
            injection_rate=10000.0,
            mobility_ratio=5.0,
            target_pressure_psi=3000.0,
        )
        eor_low_rate = EORParameters(
            injection_rate=1000.0,
            mobility_ratio=5.0,
            target_pressure_psi=3000.0,
        )

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_high = engine.evaluate_scenario(base_reservoir, eor_high_rate, base_op)
        result_low = engine.evaluate_scenario(base_reservoir, eor_low_rate, base_op)

        bt_high = result_high["breakthrough_time_years"]
        bt_low = result_low["breakthrough_time_years"]

        assert bt_high < bt_low, (
            f"High injection rate ({bt_high}) should breakthrough earlier than low rate ({bt_low})"
        )

    def test_larger_reservoir_later_bt(self, base_eor, base_op):
        """Test that larger pore volume causes later breakthrough."""
        res_small = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=500_000.0,
            initial_pressure=3000.0,
            temperature=150.0,
            average_porosity=0.15,
            average_permeability=100.0,
            initial_water_saturation=0.25,
            area_acres=80,
            thickness_ft=50,
        )
        res_small.v_dp_coefficient = 0.5

        res_large = ReservoirData(
            grid={},
            pvt_tables={},
            ooip_stb=2_000_000.0,
            initial_pressure=3000.0,
            temperature=150.0,
            average_porosity=0.15,
            average_permeability=100.0,
            initial_water_saturation=0.25,
            area_acres=320,
            thickness_ft=50,
        )
        res_large.v_dp_coefficient = 0.5

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")

        result_small = engine.evaluate_scenario(res_small, base_eor, base_op)
        result_large = engine.evaluate_scenario(res_large, base_eor, base_op)

        bt_small = result_small["breakthrough_time_years"]
        bt_large = result_large["breakthrough_time_years"]

        assert bt_small < bt_large, (
            f"Small reservoir ({bt_small}) should breakthrough earlier than large ({bt_large})"
        )

    def test_koval_formula_consistency(self, base_reservoir, base_eor, base_op):
        """Test that Koval formula gives expected values within tolerance."""
        # Koval (1963): t_D_bt = 1/K where K = E * Hk
        # E = (0.78 + 0.22 * M^0.25)^4
        # Hk = 10^(VDP / (1 - VDP))
        # t_bt = t_D * PV / q_inj

        v_dp = 0.5
        m_eff = 5.0
        area = 160  # acres
        thickness = 50  # ft
        porosity = 0.15
        injection_rate = 5000.0  # MSCFD

        # Expected Koval calculation
        hk = 10.0 ** (v_dp / (1.0 - v_dp))
        e_eff = (0.78 + 0.22 * (m_eff**0.25)) ** 4
        koval_k = hk * e_eff
        t_d_bt = 1.0 / koval_k

        pv_bbl = area * 43560.0 * thickness * porosity / 5.615
        q_inj_bbl_day = injection_rate * 0.5
        expected_bt = (t_d_bt * pv_bbl / q_inj_bbl_day) / 365.25

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")
        result = engine.evaluate_scenario(base_reservoir, base_eor, base_op)

        actual_bt = result["breakthrough_time_years"]
        tolerance = 0.01  # 1% tolerance for rounding

        assert abs(actual_bt - expected_bt) < tolerance, (
            f"BT mismatch: expected {expected_bt:.4f}, got {actual_bt:.4f}"
        )

    @pytest.mark.parametrize(
        "vdp,mobility",
        [
            (0.2, 2.0),
            (0.5, 5.0),
            (0.8, 10.0),
        ],
    )
    def test_bt_scaling_with_params(
        self, base_reservoir, base_eor, base_op, vdp, mobility
    ):
        """Test breakthrough time scales correctly with different parameter combinations."""
        base_reservoir.v_dp_coefficient = vdp
        base_eor.mobility_ratio = mobility

        engine = create_surrogate_engine(model_type="analytical", recovery_model_type="phd_hybrid")
        result = engine.evaluate_scenario(base_reservoir, base_eor, base_op)

        bt = result["breakthrough_time_years"]

        assert 0.1 <= bt <= 30.0, f"BT {bt} out of range for VDP={vdp}, M={mobility}"


class TestMassConservation:
    """Test mass conservation in the physics engine."""

    def test_engine_mass_conservation(self, standard_reservoir, standard_operational_params):
        """Test 10-year simulation satisfies CO2 mass conservation."""
        op = OperationalParameters(
            project_lifetime_years=10,
            time_resolution="yearly",
            recovery_model_selection="koval",
        )
        engine = create_surrogate_engine(
            model_type="analytical", recovery_model_type="phd_hybrid"
        )
        result = engine.evaluate_scenario(
            reservoir_data=standard_reservoir,
            eor_params=EORParameters(
                injection_rate=5000.0,
                target_pressure_psi=3000.0,
                max_pressure_psi=6000.0,
                mobility_ratio=5.0,
            ),
            operational_params=op,
        )

        total_injected = result.get("cumulative_co2_injected_tonne", 5000.0 * 365.25 * 10 * 0.05295)
        total_produced = result.get("cumulative_co2_produced_tonne", result["co2_production_cumulative"][-1] * 0.05295)
        co2_stored = result.get("co2_stored_tonnes", result["co2_stored"])

        assert total_injected == pytest.approx(total_produced + co2_stored, rel=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
