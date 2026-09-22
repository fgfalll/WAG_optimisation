"""
Tests for objective functions calculation, including co2_utilization.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, patch


class TestObjectiveFunctions:
    """Test objective functions calculation."""

    @pytest.fixture
    def mock_objective_functions(self):
        """Create a mock ObjectiveFunctions instance."""
        with patch("core.objectives.wrapper.ObjectiveFunctions.__init__") as mock_init:
            mock_init.return_value = None
            from core.objectives.wrapper import ObjectiveFunctions

            obj = ObjectiveFunctions.__new__(ObjectiveFunctions)
            obj.operational_params = MagicMock()
            obj.operational_params.time_resolution = "daily"
            obj.operational_params.project_lifetime_years = 15
            obj.eor_params = MagicMock()
            obj.eor_params.co2_recycling_efficiency_fraction = 0.9
            obj.eor_params.co2_density_tonne_per_mscf = 0.053
            obj.reservoir = MagicMock()
            obj.advanced_params = MagicMock()
            obj.advanced_params.breakthrough_fallback_time_years = 3.0
            return obj

    def test_co2_utilization_calculation(self, mock_objective_functions):
        """Test CO2 utilization calculation from profiles."""
        profiles = {
            "annual_co2_purchased_mscf": np.array([10000, 12000, 11000]),
            "annual_oil_stb": np.array([50000, 55000, 52000]),
            "npv": 5000000.0,
        }
        rf = 0.35
        econ_params = MagicMock()
        storage_params = MagicMock()

        mock_objective_functions.eor_params = MagicMock()
        mock_objective_functions.eor_params.co2_density_tonne_per_mscf = 0.053

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        assert "co2_utilization" in result
        assert isinstance(result["co2_utilization"], float)
        assert result["co2_utilization"] > 0

        expected_co2_tonnes = (10000 + 12000 + 11000) * 0.053
        expected_oil = 50000 + 55000 + 52000
        expected_util = expected_co2_tonnes / expected_oil
        assert abs(result["co2_utilization"] - expected_util) < 0.001

    def test_co2_utilization_with_zero_oil(self, mock_objective_functions):
        """Test CO2 utilization penalty when no oil produced."""
        profiles = {
            "annual_co2_purchased_mscf": np.array([10000]),
            "annual_oil_stb": np.array([0]),
            "npv": 0.0,
        }
        rf = 0.0
        econ_params = MagicMock()
        storage_params = MagicMock()

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        assert np.isnan(result["co2_utilization"])

    def test_co2_utilization_fallback_with_missing_profiles(self, mock_objective_functions):
        """Test ValueError is raised when npv is missing from profiles."""
        profiles = {}
        rf = 0.35
        econ_params = MagicMock()
        storage_params = MagicMock()
        mock_objective_functions.eor_params.co2_density_tonne_per_mscf = 0.053

        with pytest.raises(ValueError, match="npv not provided by engine"):
            mock_objective_functions._calculate_objective_functions(
                profiles, rf, econ_params, storage_params
            )

    def test_storage_efficiency_calculation(self, mock_objective_functions):
        """Test storage efficiency is calculated."""
        profiles = {
            "annual_co2_purchased_mscf": np.array([10000, 12000]),
            "annual_oil_stb": np.array([50000, 55000]),
            "npv": 5000000.0,
        }
        rf = 0.35
        econ_params = MagicMock()
        storage_params = MagicMock()
        mock_objective_functions.eor_params.co2_density_tonne_per_mscf = 0.053

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        assert "storage_efficiency" in result
        assert isinstance(result["storage_efficiency"], float)

    def test_recovery_factor_passed_through(self, mock_objective_functions):
        """Test recovery factor is passed through to results."""
        profiles = {
            "annual_co2_purchased_mscf": np.array([10000]),
            "annual_oil_stb": np.array([50000]),
            "npv": 5000000.0,
        }
        rf = 0.42
        econ_params = MagicMock()
        storage_params = MagicMock()

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        assert result["recovery_factor"] == 0.42

    def test_npv_calculated_from_profiles(self, mock_objective_functions):
        """Test NPV is read directly from surrogate engine profiles."""
        profiles = {
            "oil_production_rate": np.array([50000] * 730),  # 2 years daily
            "co2_injection": np.array([10000] * 730),  # Daily injection
            "gas_production_rate": np.array([8000] * 730),  # CO2 production
            "time_vector": np.arange(730),
            "breakthrough_time_years": 1.0,  # 1 year breakthrough for testing
            "npv": 5000000.0,
        }
        rf = 0.35
        econ_params = MagicMock()
        econ_params.oil_price_usd_per_bbl = 80.0
        econ_params.co2_cost_usd_per_ton = 50.0
        econ_params.discount_rate = 0.1
        econ_params.water_disposal_cost_usd_per_bbl = 2.0
        econ_params.injection_cost_usd_per_ton = 10.0
        econ_params.operating_cost_usd_per_bbl = 10.0
        econ_params.initial_investment_usd = 0.0
        econ_params.co2_purchase_cost_usd_per_tonne = 50.0
        econ_params.co2_recycle_cost_usd_per_tonne = 40.0
        econ_params.discount_rate_fraction = 0.1
        econ_params.initial_investment_usd = 0.0
        storage_params = MagicMock()
        mock_objective_functions.eor_params.co2_density_tonne_per_mscf = 0.053
        mock_objective_functions.reservoir.pore_volume = 1e9
        mock_objective_functions.reservoir.average_porosity = 0.2
        mock_objective_functions.reservoir.initial_water_saturation = 0.2

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        assert "npv" in result
        assert "npv_details" in result
        assert result["npv_details"]["method"] == "surrogate_engine"

    def test_zero_injection_extreme_lower_bound(self, mock_objective_functions):
        """Test extreme lower bounds with zero injection volume over entire timeline.

        When simulated injection volume is 0.0:
        - annual_co2_purchased should be exactly 0.0
        - annual_co2_recycled should be exactly 0.0
        - trapping_efficiency should be exactly 0.0
        - NPV should equal baseline primary depletion NPV (no CO2 costs)
        """
        project_lifetime = 15

        baseline_profiles = {
            "annual_oil_stb": np.array([50000.0 * 365.25] * project_lifetime),
            "annual_co2_purchased_mscf": np.array([0.0] * project_lifetime),
            "annual_co2_recycled_mscf": np.array([0.0] * project_lifetime),
        }
        econ_params = MagicMock()
        econ_params.oil_price_usd_per_bbl = 80.0
        econ_params.co2_cost_usd_per_ton = 50.0
        econ_params.discount_rate = 0.1
        econ_params.discount_rate_fraction = 0.1
        econ_params.water_disposal_cost_usd_per_bbl = 2.0
        econ_params.injection_cost_usd_per_ton = 10.0
        econ_params.operating_cost_usd_per_bbl = 10.0
        econ_params.initial_investment_usd = 0.0
        econ_params.co2_purchase_cost_usd_per_tonne = 50.0
        econ_params.co2_recycle_cost_usd_per_tonne = 40.0
        econ_params.carbon_credit_usd_per_ton = 0.0

        from core.objectives.economic import calculate_npv
        baseline_npv = calculate_npv(
            profiles=baseline_profiles,
            economic_params=econ_params,
        )

        profiles = {
            "annual_oil_stb": np.array([50000.0 * 365.25] * project_lifetime),
            "co2_injection": np.array([0.0] * project_lifetime),
            "co2_production_rate": np.array([0.0] * project_lifetime),
            "time_vector": np.arange(project_lifetime),
            "breakthrough_time_years": 0.0,
            "npv": baseline_npv,
        }
        rf = 0.25
        storage_params = MagicMock()
        mock_objective_functions.eor_params.co2_density_tonne_per_mscf = 0.053
        mock_objective_functions.eor_params.injection_rate = 5000.0
        mock_objective_functions.reservoir.pore_volume = 1e9
        mock_objective_functions.reservoir.average_porosity = 0.2
        mock_objective_functions.reservoir.initial_water_saturation = 0.2
        mock_objective_functions.operational_params.project_lifetime_years = project_lifetime
        mock_objective_functions.operational_params.time_resolution = "annual"

        result = mock_objective_functions._calculate_objective_functions(
            profiles, rf, econ_params, storage_params
        )

        annual_profiles = result.get("npv_details", {}).get("annual_profiles", {})
        annual_co2_purchased = annual_profiles.get("annual_co2_purchased_mscf", np.array([]))
        annual_co2_recycled = annual_profiles.get("annual_co2_recycled_mscf", np.array([]))

        assert np.allclose(annual_co2_purchased, 0.0), (
            f"annual_co2_purchased should be exactly 0.0 for zero injection, got {annual_co2_purchased}"
        )
        assert np.allclose(annual_co2_recycled, 0.0), (
            f"annual_co2_recycled should be exactly 0.0 for zero injection, got {annual_co2_recycled}"
        )

        assert "storage_efficiency" in result
        assert result["storage_efficiency"] == 0.0, (
            f"trapping_efficiency should be exactly 0.0 for zero injection, got {result['storage_efficiency']}"
        )

        npv_zero_injection = result["npv"]
        assert isinstance(npv_zero_injection, (int, float)), (
            f"NPV should be a numeric value, got {type(npv_zero_injection)}: {npv_zero_injection}"
        )

        assert np.isclose(npv_zero_injection, baseline_npv), (
            f"NPV with zero injection ({npv_zero_injection}) should equal "
            f"baseline primary depletion NPV ({baseline_npv})"
        )
