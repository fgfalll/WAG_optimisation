"""
Wrapper for adapting UnifiedEngine to legacy SimulationEngineInterface.

This module provides backward compatibility with the existing engine interface.
"""

from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import logging

from ..base.engine_config import EngineConfig
from ..core.state_manager import UnifiedState

logger = logging.getLogger(__name__)


class SimulationEngineInterface(ABC):
    """Abstract interface for all simulation engines (legacy version)"""

    @abstractmethod
    def evaluate_scenario(
        self,
        reservoir_data,
        eor_params,
        operational_params,
        economic_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate a scenario and return comprehensive results"""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine"""
        pass

    @abstractmethod
    def validate_parameters(self, reservoir_data, eor_params) -> Dict[str, bool]:
        """Validate input parameters"""
        pass


class UnifiedEngineWrapper(SimulationEngineInterface):
    """
    Wrapper to adapt the new unified engine to the legacy SimulationEngineInterface.

    This allows the unified engine (FastEngine/DetailedEngine) to work with
    existing code that expects the legacy interface.
    """

    def __init__(self, engine, config: EngineConfig):
        """
        Initialize wrapper.

        Args:
            engine: FastEngine or DetailedEngine instance.
            config: EngineConfig instance.
        """
        self._engine = engine
        self._config = config
        self._grid_data = None
        self._initial_conditions = None

    def _convert_reservoir_data(self, reservoir_data) -> Dict[str, np.ndarray]:
        """Convert ReservoirData to grid data format."""
        nx, ny, nz = 50, 50, 1  # Default grid size

        # Extract or compute porosity
        if hasattr(reservoir_data, 'porosity_array') and reservoir_data.porosity_array is not None:
            porosity = reservoir_data.porosity_array.flatten()
        else:
            porosity = np.full(nx * ny * nz, reservoir_data.average_porosity)

        # Extract or compute permeability
        if hasattr(reservoir_data, 'permeability_array') and reservoir_data.permeability_array is not None:
            perm = reservoir_data.permeability_array
            if perm.ndim == 3:
                perm = perm.reshape(-1)
            permeability = perm
        else:
            permeability = np.full((nx * ny * nz, 3), reservoir_data.average_permeability)

        return {
            "porosity": porosity,
            "permeability": permeability,
        }

    def _convert_eor_params(self, eor_params) -> Dict[str, Any]:
        """Convert EORParameters to initial conditions format."""
        initial_conditions = {
            "pressure": eor_params.initial_pressure_psi if hasattr(eor_params, 'initial_pressure_psi') else 2000.0,
        }

        # Add compositions if detailed mode
        if self._config.is_detailed():
            # Placeholder for compositional data
            pass

        return initial_conditions

    def evaluate_scenario(
        self,
        reservoir_data,
        eor_params,
        operational_params,
        economic_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Evaluate scenario using unified engine."""
        try:
            # Convert input data
            grid_data = self._convert_reservoir_data(reservoir_data)
            initial_conditions = self._convert_eor_params(eor_params)

            # Initialize engine
            if not self._engine.is_initialized:
                self._engine.initialize(grid_data, initial_conditions)

            # Run simulation
            final_state = self._engine.run(
                end_time=self._config.simulation_time,
                progress_callback=None
            )

            # Extract results
            statistics = self._engine.get_statistics()

            # Extract actual production rates from simulation state
            oil_rate, water_rate, gas_rate = self._extract_production_rates(final_state)

            # Calculate cumulative production
            cumulative_oil = self._calculate_cumulative(oil_rate)
            cumulative_water = self._calculate_cumulative(water_rate)
            cumulative_gas = self._calculate_cumulative(gas_rate)

            # Calculate recovery factor from state
            recovery_factor = self._calculate_recovery_factor(final_state, reservoir_data)

            # Calculate NPV with actual production profile
            npv = 0.0
            if economic_params:
                npv = self._calculate_npv(
                    oil_rate=oil_rate,
                    water_rate=water_rate,
                    gas_rate=gas_rate,
                    economic_params=economic_params,
                    simulation_time=self._config.simulation_time
                )

            # Generate time vector
            n_points = max(len(oil_rate), 100)
            time_vector = np.linspace(0, self._config.simulation_time, n_points)

            # Calculate CO2 injection profile
            injection_rate = getattr(eor_params, 'injection_rate', 10000.0) * 0.0283  # MSCFD to tons/day
            co2_injection = np.full(n_points, injection_rate)

            # Return results in legacy format
            return {
                'oil_production_rate': oil_rate[:n_points],
                'water_production_rate': water_rate[:n_points],
                'gas_production_rate': gas_rate[:n_points],
                'cumulative_oil': cumulative_oil,
                'cumulative_water': cumulative_water,
                'cumulative_gas': cumulative_gas,
                'time_vector': time_vector,
                'recovery_factor': recovery_factor,
                'npv': npv,
                'constraint_violations': {},
                'convergence_status': 'success',
                'simulation_time': statistics.get('total_solve_time', 0.0),
                # Add CO2 injection data for storage efficiency calculation
                'co2_injection': co2_injection,
                'engine_type': 'unified_' + self._config.mode.value,
                'unified_engine_stats': statistics,
            }

        except Exception as e:
            logger.error(f"Unified engine evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'convergence_status': 'error',
                'error_message': str(e),
                'engine_type': 'unified_' + self._config.mode.value,
            }

    def get_engine_info(self) -> Dict[str, Any]:
        """Get unified engine information."""
        mode = self._config.mode.value
        return {
            'engine_type': f'unified_{mode}',
            'name': f'Unified {"Fast" if mode == "simple" else "Detailed"} Engine',
            'description': f'New unified {"fast screening" if mode == "simple" else "comprehensive physics"} engine',
            'capabilities': ['multiphase_flow', 'adaptive_timestepping'],
            'speed': 'fast' if mode == 'simple' else 'moderate',
            'accuracy': 'moderate' if mode == 'simple' else 'high',
            'unified_version': '1.0.0',
        }

    def validate_parameters(
        self,
        reservoir_data,
        eor_params
    ) -> Dict[str, bool]:
        """Validate parameters for unified engine."""
        validation = {}

        # Basic checks
        avg_porosity = reservoir_data.average_porosity
        avg_perm = reservoir_data.average_permeability

        validation['porosity_valid'] = avg_porosity is not None and 0.01 <= avg_porosity <= 0.5
        validation['permeability_valid'] = avg_perm is not None and avg_perm > 0

        # Validate configuration
        config_errors = self._config.validate()
        validation['config_valid'] = len(config_errors) == 0

        return validation

    def _extract_production_rates(self, state) -> tuple:
        """Extract production rates from simulation state."""
        # Get pressure and saturations from state
        pressure = np.array(state.pressure) if hasattr(state, 'pressure') else np.array([2000.0])
        saturations = np.array(state.saturations) if hasattr(state, 'saturations') else np.array([[1.0, 0.0, 0.0]])

        # Extract phases: assume columns are [oil, water, gas]
        if saturations.ndim == 2 and saturations.shape[1] >= 3:
            s_oil = saturations[:, 0]
            s_water = saturations[:, 1]
            s_gas = saturations[:, 2]
        else:
            s_oil = np.ones(len(pressure)) * 0.7
            s_water = np.ones(len(pressure)) * 0.2
            s_gas = np.ones(len(pressure)) * 0.1

        # Calculate average saturations
        avg_oil_sat = np.mean(s_oil)
        avg_water_sat = np.mean(s_water)
        avg_gas_sat = np.mean(s_gas)

        # Production rates based on saturations and pressure
        # Use Darcy's law with productivity index
        avg_pressure = np.mean(pressure)
        pi = 5.0  # STB/day/psi (productivity index)

        # Drawdown from average reservoir pressure to BHP
        bhp = 1000.0  # psi
        drawdown = max(0, avg_pressure - bhp)

        # Production rates based on phase mobilities
        # Oil rate based on oil saturation and drawdown
        oil_rate = pi * avg_oil_sat * drawdown * np.ones(100)

        # Water rate based on water saturation
        water_rate = pi * avg_water_sat * 0.5 * drawdown * np.ones(100)

        # Gas rate based on gas saturation and solution GOR
        solution_gor = 800.0  # scf/stb
        gas_rate = (pi * avg_gas_sat * solution_gor + oil_rate * solution_gor) * 1e-3 * np.ones(100)  # MSCF/day

        return oil_rate, water_rate, gas_rate

    def _calculate_cumulative(self, rate_array: np.ndarray) -> float:
        """Calculate cumulative production from rate array."""
        return float(np.sum(rate_array))

    def _calculate_recovery_factor(self, state, reservoir_data) -> float:
        """Calculate recovery factor from simulation state."""
        # Get initial and current oil in place
        initial_oil = getattr(reservoir_data, 'ooip_stb', 1e6)
        current_oil = initial_oil

        # Estimate remaining oil based on saturation
        if hasattr(state, 'saturations'):
            saturations = np.array(state.saturations)
            if saturations.ndim == 2 and saturations.shape[1] >= 1:
                avg_oil_sat = np.mean(saturations[:, 0])
                initial_sat = 0.7  # Assumed initial oil saturation
                recovery = (initial_sat - avg_oil_sat) / initial_sat
                return max(0.0, min(1.0, recovery))

        return 0.35  # Default estimate

    def _calculate_npv(
        self,
        oil_rate: np.ndarray,
        water_rate: np.ndarray,
        gas_rate: np.ndarray,
        economic_params: Dict[str, float],
        simulation_time: float
    ) -> float:
        """Calculate NPV from production rates and economic parameters."""
        # Economic parameters with defaults
        oil_price = economic_params.get('oil_price_usd_per_bbl', 80.0)
        gas_price = economic_params.get('gas_price_usd_per_mscf', 3.0)
        discount_rate = economic_params.get('discount_rate', 0.1)
        co2_cost = economic_params.get('co2_cost_per_ton', 50.0)

        # Time vector (days)
        n_steps = len(oil_rate)
        time_days = np.linspace(0, simulation_time * 365, n_steps)

        # Daily discount factors
        daily_discount = (1 + discount_rate) ** (time_days / 365)

        # Revenue streams
        oil_revenue = oil_rate * oil_price
        gas_revenue = gas_rate * gas_price

        # Discounted cash flows
        npv_oil = np.sum(oil_revenue / daily_discount)
        npv_gas = np.sum(gas_revenue / daily_discount)

        # CO2 injection cost (estimated)
        injection_rate = getattr(self._config, 'injection_rate', 10000)  # MSCFD
        daily_co2_tons = injection_rate * 0.0283
        co2_cost_total = np.sum(np.full(n_steps, daily_co2_tons * co2_cost) / daily_discount)

        return npv_oil + npv_gas - co2_cost_total
