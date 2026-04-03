"""
CO₂ breakthrough physics module for enhanced oil recovery optimization.
Implements comprehensive breakthrough detection, recycling calculations, and physics-based models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from scipy.optimize import fsolve

# Import centralized constants
from core.data_models import PhysicalConstants

_PERM_CONSTANTS = PhysicalConstants()

logger = logging.getLogger(__name__)

# Use centralized constants
EPSILON = _PERM_CONSTANTS.NUMERICAL_EPSILON_DEFAULT
GRAVITY_ACCEL_CM_S2 = 981.0
PERM_MD_TO_M2 = _PERM_CONSTANTS.MD_TO_M2  # 9.869233e-16 mD to m^2
PERM_MD_TO_CM2 = _PERM_CONSTANTS.MD_TO_CM2  # 9.869233e-13 mD to cm^2
VISC_CP_TO_POISE = 0.01
VISC_CP_TO_PA_S = _PERM_CONSTANTS.VISC_CP_TO_PA_S  # 0.001 cP to Pa·s
GRAVITY_FT_S2 = _PERM_CONSTANTS.GRAVITY_FT_S2  # 32.174 ft/s^2
SECONDS_PER_DAY = _PERM_CONSTANTS.SECONDS_PER_DAY  # 86400.0


@dataclass
class BreakthroughParameters:
    """Parameters for CO₂ breakthrough physics and recycling calculations."""

    # Breakthrough detection parameters
    breakthrough_gor_threshold: float = field(
        default=300.0,
        metadata={"help": "Gas-Oil Ratio (GOR) threshold for breakthrough detection (scf/stb)"},
    )
    breakthrough_water_cut_threshold: float = field(
        default=0.95,
        metadata={"help": "Water cut threshold for breakthrough detection (fraction)"},
    )
    min_breakthrough_time_years: float = field(
        default=0.1,
        metadata={"help": "Minimum time before breakthrough can occur (years)"},
    )

    # Recycling parameters
    recycling_efficiency: float = field(
        default=0.9,
        metadata={"help": "Fraction of produced CO₂ that can be efficiently recycled"},
    )
    recycling_purity_requirement: float = field(
        default=0.95,
        metadata={"help": "Minimum CO₂ purity required for recycling (fraction)"},
    )
    recycling_compression_cost_usd_per_tonne: float = field(
        default=15.0,
        metadata={"help": "Cost to compress and purify CO₂ for recycling ($/tonne)"},
    )

    # Physics parameters - adjusted for realistic CO2-EOR operations
    gravity_override_factor: float = field(
        default=0.08,
        metadata={
            "help": "Factor for gravity override effects on breakthrough (reduced for later breakthrough)"
        },
    )
    viscous_fingering_factor: float = field(
        default=0.15,
        metadata={"help": "Factor for viscous fingering effects on breakthrough (reduced)"},
    )
    heterogeneity_impact_factor: float = field(
        default=0.35,
        metadata={"help": "Factor for reservoir heterogeneity impact on breakthrough"},
    )
    gravity_dip_angle_threshold: float = field(
        default=15.0,
        metadata={"help": "Dip angle threshold for gravity dominance weighting"},
    )
    viscous_mobility_ratio_threshold: float = field(
        default=5.0,
        metadata={"help": "Mobility ratio threshold for viscous fingering weighting"},
    )
    post_breakthrough_gor_time_constant: float = field(
        default=1.0,
        metadata={"help": "Time constant for post-breakthrough GOR evolution (years)"},
    )
    dp_bt_pore_volumes_factor: float = field(
        default=2.5,
        metadata={
            "help": "Factor for Dykstra-Parsons breakthrough pore volumes (2-5 pore volumes typical)"
        },
    )
    vf_bt_pore_volumes_factor: float = field(
        default=3.0,
        metadata={"help": "Factor for viscous fingering breakthrough pore volumes"},
    )
    min_bt_time_clip: float = field(
        default=0.5,
        metadata={"help": "Minimum clipping value for breakthrough time (years)"},
    )
    max_bt_time_clip: float = field(
        default=25.0,
        metadata={"help": "Maximum clipping value for breakthrough time (years)"},
    )

    def __post_init__(self):
        if not (50.0 <= self.breakthrough_gor_threshold <= 5000.0):
            raise ValueError("Breakthrough GOR threshold must be between 50 and 5000 scf/stb")
        if not (0.7 <= self.breakthrough_water_cut_threshold <= 1.0):
            raise ValueError("Breakthrough water cut threshold must be between 0.7 and 1.0")
        if not (0.0 <= self.recycling_efficiency <= 1.0):
            raise ValueError("Recycling efficiency must be between 0.0 and 1.0")


def _to_float(value):
    if hasattr(value, "item"):
        return value.item()
    return float(value)


class CO2BreakthroughPhysics:
    """
    Comprehensive CO₂ breakthrough physics and recycling calculations.
    Implements industry-standard methods for breakthrough prediction and recycling optimization.
    """

    def __init__(self, params: Optional[BreakthroughParameters] = None):
        self.params = params or BreakthroughParameters()

    def calculate_breakthrough_time(
        self, 
        reservoir_params: Dict, 
        eor_params: Dict, 
        eos_model: Optional[Any] = None
    ) -> float:
        """
        Calculate CO₂ breakthrough time using physics-based dimensionless analysis.

        Refined Method:
        1. Fingering & Heterogeneity: Koval Factor (Koval, 1963)
        2. Gravity Override: Viscous-Gravity Ratio (Dietz, 1953)
        3. Mechanism Weighting: Based on Dimensionless Gravity Number (Ng)

        Args:
            reservoir_params: Reservoir properties
            eor_params: EOR operation parameters
            eos_model: Optional ReservoirFluid object for dynamic property calculation

        Returns:
            Estimated breakthrough time in years.
        """

        # 1. Dynamic Fluid Property Retrieval (Scientific Integrity Check)        
        pressure = _to_float(reservoir_params.get("pressure", 2000.0))
        temp_f = _to_float(reservoir_params.get("temperature", 150.0))
        
        if eos_model is not None:
            # Use dynamic EOS properties
            temp_k = (temp_f - 32) * 5/9 + 273.15
            pres_pa = pressure * 6894.76
            props = eos_model.get_properties_si(temp_k, pres_pa)
            
            # Extract densities (kg/m3 -> lb/ft3)
            rho_co2 = props.get("vapor_properties", {}).get("density", 44.0 * 16.0185) / 16.0185
            rho_oil = props.get("liquid_properties", {}).get("density", 50.0 * 16.0185) / 16.0185
            
            # Extract viscosities (Pa.s -> cP)
            mu_co2 = props.get("vapor_properties", {}).get("viscosity", 0.08 / 1000.0) * 1000.0
            mu_oil = props.get("liquid_properties", {}).get("viscosity", 2.0 / 1000.0) * 1000.0
            
            # Dynamic Bg (rb/Mscf)
            bg = eos_model.get_bgas_rb_per_mscf(temp_k, pres_pa)
        else:
            # Fallback to eor_params with logging warning
            logger.warning("No EOS model provided to breakthrough physics. Using fallback empirical parameters.")
            rho_co2 = _to_float(eor_params.get("co2_density", 44.0))
            rho_oil = _to_float(eor_params.get("oil_density", 50.0))
            mu_co2 = _to_float(eor_params.get("co2_viscosity", 0.08))
            mu_oil = _to_float(eor_params.get("oil_viscosity_cp", 2.0))
            bg = 0.5 # Default fallback

        # Updated parameters dictionary for internal calls
        dynamic_fluid_data = {
            "rho_co2": rho_co2,
            "rho_oil": rho_oil,
            "mu_co2": mu_co2,
            "mu_oil": mu_oil,
            "bg": bg
        }

        vdp = _to_float(reservoir_params.get("v_dp_coefficient", 0.5))
        mobility_ratio = mu_oil / max(mu_co2, EPSILON)

        # Method 1: Koval Factor (Fingering + Heterogeneity)
        bt_time_koval = self._koval_breakthrough(
            vdp, mobility_ratio, reservoir_params, eor_params, dynamic_fluid_data
        )

        # Method 2: Gravity Override (Dimensionless analysis)
        bt_time_gravity = self._gravity_override_breakthrough(
            reservoir_params, eor_params, dynamic_fluid_data
        )

        # Dimensionless Mechanism Weighting
        weights = self._calculate_dimensionless_weights(
            reservoir_params, eor_params, dynamic_fluid_data
        )

        breakthrough_time = (
            weights["koval"] * bt_time_koval +
            weights["gravity"] * bt_time_gravity
        )

        return float(breakthrough_time)

    def _koval_breakthrough(
        self,
        vdp: float,
        mobility_ratio: float,
        reservoir_params: Dict,
        eor_params: Dict,
        fluid_data: Dict
    ) -> float:
        """
        Calculate breakthrough time using the Koval Method (Koval, 1963).
        K_v = E * Hk
        t_bt (PV) = 1 / K_v
        """
        # Heterogeneity Factor Hk (Standing, 1974 correlation for VDP)
        # Hk = 10^(VDP / (1 - VDP))
        vdp = np.clip(vdp, 0.0, 0.95)
        hk = 10 ** (vdp / (1.0 - vdp + EPSILON))

        # Effective Viscosity Ratio E (Koval 1/4 power rule)
        # E = (0.22 + 0.78 * M^0.25)^4
        m_eff = (0.22 + 0.78 * (mobility_ratio ** 0.25)) ** 4
        
        koval_factor = m_eff * hk
        bt_pore_volumes = 1.0 / max(koval_factor, EPSILON)

        # Convert PV to Years
        length_ft = _to_float(reservoir_params.get("length_ft", 2000.0))
        width_ft = _to_float(reservoir_params.get("width_ft", 1000.0))
        thickness_ft = _to_float(reservoir_params.get("thickness_ft", 50.0))
        porosity = np.clip(_to_float(reservoir_params.get("porosity", 0.15)), 0.01, 0.4)
        
        pore_volume_ft3 = length_ft * width_ft * thickness_ft * porosity
        
        # Injection rate conversion
        inj_mscfd = _to_float(eor_params.get("injection_rate", 5000.0))
        bg = fluid_data["bg"]
        q_res_ft3_day = inj_mscfd * bg * 5.61458
        
        bt_years = (bt_pore_volumes * pore_volume_ft3) / (q_res_ft3_day * 365.25 + EPSILON)
        
        return float(bt_years)

    def _gravity_override_breakthrough(
        self, 
        reservoir_params: Dict, 
        eor_params: Dict,
        fluid_data: Dict
    ) -> float:
        """
        Calculate breakthrough time for gravity override using the 
        Viscous-Gravity Ratio (R v/g).
        R v/g = (v * mu_o * L) / (k * delta_rho * g * H)
        """
        length_ft = _to_float(reservoir_params.get("length_ft", 2000.0))
        height_ft = _to_float(reservoir_params.get("thickness_ft", 50.0))
        width_ft = _to_float(reservoir_params.get("width_ft", 1000.0))
        perm_md = _to_float(reservoir_params.get("permeability", 100.0))
        porosity = _to_float(reservoir_params.get("porosity", 0.15))
        
        mu_o_cp = fluid_data["mu_oil"]
        delta_rho_lb_ft3 = abs(fluid_data["rho_oil"] - fluid_data["rho_co2"])
        
        # Velocity u (ft/day)
        inj_mscfd = _to_float(eor_params.get("injection_rate", 5000.0))
        bg = fluid_data["bg"]
        q_res_ft3_day = inj_mscfd * bg * 5.61458
        u_ft_day = q_res_ft3_day / (width_ft * height_ft + EPSILON)
        
        # Convert MD to ft2 (1 mD = 1.0623e-14 ft2)
        k_ft2 = perm_md * 1.0623e-14
        
        # mu_o in lb-day/ft2 (1 cP = 2.0885e-5 lb-s/ft2 = 2.417e-10 lb-day/ft2)
        mu_o_lb_day_ft2 = mu_o_cp * 2.417e-10
        
        # g in ft/day2 (32.17 ft/s2 = 2.4e11 ft/day2)
        g_ft_day2 = 2.4e11
        
        # Viscous-Gravity Ratio (R v/g)
        # Note: Higher R v/g means viscous forces dominate (late override)
        rvg = (u_ft_day * mu_o_lb_day_ft2 * length_ft) / (k_ft2 * delta_rho_lb_ft3 * g_ft_day2 * height_ft + EPSILON)
        
        # Empirical breakthrough PV for gravity override (Dietz-based proxy)
        # As Rvg -> 0, override is immediate (bt_pv -> 0)
        # As Rvg -> inf, override is absent (bt_pv -> 1/M)
        m_ratio = mu_o_cp / max(fluid_data["mu_co2"], EPSILON)
        bt_pv = (1.0 / m_ratio) * (1.0 - np.exp(-max(rvg, 0.01)))
        
        pore_volume_ft3 = length_ft * width_ft * height_ft * porosity
        bt_years = (bt_pv * pore_volume_ft3) / (q_res_ft3_day * 365.25 + EPSILON)
        
        return float(bt_years)

    def _calculate_dimensionless_weights(
        self, 
        reservoir_params: Dict, 
        eor_params: Dict,
        fluid_data: Dict
    ) -> Dict[str, float]:
        """
        Calculate mechanism weights based on the Gravity Number (Ng).
        Ng = (k * delta_rho * g * sin(theta)) / (mu_g * u)
        """
        dip_angle = _to_float(reservoir_params.get("dip_angle", 0.0))
        perm_md = _to_float(reservoir_params.get("permeability", 100.0))
        mu_g_cp = fluid_data["mu_co2"]
        delta_rho_lb_ft3 = abs(fluid_data["rho_oil"] - fluid_data["rho_co2"])
        
        # Velocity u (ft/day)
        inj_mscfd = _to_float(eor_params.get("injection_rate", 5000.0))
        bg = fluid_data["bg"]
        width_ft = _to_float(reservoir_params.get("width_ft", 1000.0))
        height_ft = _to_float(reservoir_params.get("thickness_ft", 50.0))
        q_res_ft3_day = inj_mscfd * bg * 5.61458
        u_ft_day = q_res_ft3_day / (width_ft * height_ft + EPSILON)
        
        # Gravity Number (Ng)
        k_ft2 = perm_md * 1.0623e-14
        mu_g_lb_day_ft2 = mu_g_cp * 2.417e-10
        g_ft_day2 = 2.4e11
        sin_theta = np.sin(np.radians(dip_angle))
        
        ng = (k_ft2 * delta_rho_lb_ft3 * g_ft_day2 * abs(sin_theta)) / (mu_g_lb_day_ft2 * u_ft_day + EPSILON)
        
        # Weighting logic (Coupled regime):
        # Instead of linear blending, we use the nonlinear dimensionless interaction function.
        # \Phi(N_g, M) = 1.0 / (1.0 + sqrt(N_g * M))
        # Note: We represent the weighting as a reciprocal gravity coupled response.
        mu_o_cp = fluid_data["mu_oil"]
        m_ratio = mu_o_cp / max(mu_g_cp, EPSILON)
        phi_ng_m = 1.0 / (1.0 + np.sqrt(ng * max(m_ratio, EPSILON)))
        
        # We output this as "gravity" and "koval" blending parameters that
        # produce the effect `t_bt = t_bt,v * \Phi(Ng, M)`
        # To do this cleanly, we can set w_gravity -> 0 and koval -> \Phi
        
        return {
            "gravity": 0.0,
            "koval": float(phi_ng_m)
        }

    def calculate_post_breakthrough_gor(
        self, eor_params: Dict, time_since_breakthrough: float
    ) -> float:
        """
        Calculate Gas-Oil Ratio after breakthrough.

        Args:
            eor_params: EOR operation parameters
            time_since_breakthrough: Time since breakthrough occurred (years)

        Returns:
            Current GOR (scf/stb)
        """
        base_gor = eor_params.get("gas_oil_ratio_at_breakthrough", 800.0)
        injection_gor = eor_params.get("injection_gor", 10000.0)  # Typical injected CO₂ GOR

        # Exponential approach to injection GOR
        time_constant = (
            self.params.post_breakthrough_gor_time_constant
        )  # years to reach 63% of final value
        gor = base_gor + (injection_gor - base_gor) * (
            1.0 - np.exp(-time_since_breakthrough / time_constant)
        )

        return float(min(gor, injection_gor * 1.1))  # Cap at 10% above injection GOR

    def calculate_recyclable_co2(
        self, produced_co2_mscf: float, produced_gas_composition: Dict
    ) -> float:
        """
        Calculate recyclable CO₂ based on purity requirements.

        Args:
            produced_co2_mscf: Total produced CO₂ (Mscf)
            produced_gas_composition: Composition of produced gas

        Returns:
            Recyclable CO₂ volume (Mscf)
        """
        co2_purity = produced_gas_composition.get("co2_fraction", 0.8)

        if co2_purity >= self.params.recycling_purity_requirement:
            # High purity - most can be recycled
            recyclable_fraction = self.params.recycling_efficiency
        else:
            # Lower purity - reduced recycling efficiency
            purity_penalty = (
                self.params.recycling_purity_requirement - co2_purity
            ) / self.params.recycling_purity_requirement
            recyclable_fraction = self.params.recycling_efficiency * (1.0 - purity_penalty)

        return float(produced_co2_mscf * max(0.0, recyclable_fraction))

    def generate_breakthrough_profile(
        self, project_lifetime: int, breakthrough_time: float, eor_params: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Generate annual breakthrough and recycling profile.

        Args:
            project_lifetime: Total project lifetime (years)
            breakthrough_time: Estimated breakthrough time (years) - MUST be scalar
            eor_params: EOR operation parameters

        Returns:
            Dictionary with annual breakthrough metrics
        """
        # CRITICAL: Ensure breakthrough_time is a scalar before any array operations
        # This prevents "ambiguous truth value" errors from NumPy
        if hasattr(breakthrough_time, 'item'):
            breakthrough_time = breakthrough_time.item()
        breakthrough_time = float(breakthrough_time)

        # Validate breakthrough_time is reasonable
        if not (0 < breakthrough_time <= project_lifetime * 2):
            logger.warning(f"Breakthrough time {breakthrough_time} outside expected range, using fallback")
            breakthrough_time = project_lifetime / 2  # Use midpoint as fallback

        years = np.arange(1, project_lifetime + 1)
        breakthrough_occurred = years >= breakthrough_time
        time_since_breakthrough = np.maximum(0, years - breakthrough_time)

        # Calculate GOR profile
        gor_profile = np.zeros_like(years, dtype=float)
        for i, year in enumerate(years):
            if breakthrough_occurred[i]:
                gor_profile[i] = self.calculate_post_breakthrough_gor(
                    eor_params, time_since_breakthrough[i]
                )
            else:
                gor_profile[i] = eor_params.get("initial_gor", 200.0)  # Base reservoir GOR

        # Calculate recycling potential (simplified)
        # This would be enhanced with actual production data in a real implementation
        recycling_efficiency_profile = np.where(
            breakthrough_occurred,
            self.params.recycling_efficiency,
            0.0,  # No recycling before breakthrough
        )

        return {
            "years": years,
            "breakthrough_occurred": breakthrough_occurred.tolist(),  # Convert to list for JSON serialization
            "breakthrough_occurred_array": breakthrough_occurred,     # Keep array for numerical operations
            "any_breakthrough_occurred": bool(breakthrough_occurred.any()),  # Scalar summary
            "first_breakthrough_year": int(years[breakthrough_occurred][0]) if breakthrough_occurred.any() else None,
            "time_since_breakthrough": time_since_breakthrough,
            "gor_profile": gor_profile,
            "recycling_efficiency": recycling_efficiency_profile,
            "breakthrough_time": breakthrough_time,
        }

    def generate_spatial_breakthrough_profile(
        self, simulation_states: List, eor_params: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Generate breakthrough profile from simulation states with spatial evolution.

        Args:
            simulation_states: List of simulation states with temporal and spatial data
            eor_params: EOR operation parameters

        Returns:
            Dictionary with spatial and temporal breakthrough data
        """
        if not simulation_states or len(simulation_states) < 2:
            # Fallback to theoretical profile
            return self.generate_breakthrough_profile(15, 5.0, eor_params)

        n_states = len(simulation_states)
        n_cells = (
            len(simulation_states[0].pressure) if hasattr(simulation_states[0], "pressure") else 100
        )
        x_positions = np.linspace(0, 1, n_cells)  # Normalized positions
        times = np.zeros(n_states)

        # Extract temporal and spatial data
        co2_saturation_profile = np.zeros((n_states, n_cells))
        oil_saturation_profile = np.zeros((n_states, n_cells))
        pressure_profile = np.zeros((n_states, n_cells))
        co2_front_position = np.zeros(n_states)

        for i, state in enumerate(simulation_states):
            times[i] = (
                state.current_time if hasattr(state, "current_time") else i * 30.44
            )  # Monthly timesteps

            if hasattr(state, "saturations") and state.saturations is not None:
                # Extract phase saturations (assuming shape: [n_cells, n_phases])
                if state.saturations.ndim == 2 and state.saturations.shape[1] >= 3:
                    oil_saturation_profile[i] = state.saturations[:, 1]  # Oil saturation
                    co2_saturation_profile[i] = state.saturations[:, 2]  # CO2 saturation
                elif state.saturations.ndim == 1:
                    # Single-phase assumption
                    co2_saturation_profile[i] = state.saturations
                    oil_saturation_profile[i] = 1.0 - state.saturations

            if hasattr(state, "pressure"):
                pressure_profile[i] = state.pressure

            # Calculate CO2 front position (10% CO2 saturation threshold)
            if n_cells > 1:
                co2_front_idx = np.where(co2_saturation_profile[i] > 0.1)[0]
                if len(co2_front_idx) > 0:
                    co2_front_position[i] = x_positions[co2_front_idx[0]]
                else:
                    co2_front_position[i] = 0.0

        # Calculate GOR profile based on CO2 breakthrough at producer
        producer_co2_sat = co2_saturation_profile[:, -1]  # Last cell (producer)
        gor_profile = np.where(
            producer_co2_sat > 0.05,
            200.0 + producer_co2_sat * 5000.0,  # Scale GOR with CO2 saturation
            eor_params.get("initial_gor", 200.0),
        )

        # Calculate recycling efficiency based on CO2 production
        recycling_efficiency = np.where(
            producer_co2_sat > 0.05, self.params.recycling_efficiency, 0.0
        )

        # Detect actual breakthrough time from simulation
        breakthrough_threshold = 0.05  # 5% CO2 at producer
        breakthrough_indices = np.where(producer_co2_sat > breakthrough_threshold)[0]
        breakthrough_time_sim = (
            times[breakthrough_indices[0]] if len(breakthrough_indices) > 0 else 999.0
        )

        logger.info(
            f"Spatial breakthrough analysis: {breakthrough_time_sim / 365:.1f} years, "
            f"Final CO2 front: {co2_front_position[-1]:.2f}, "
            f"Producer CO2 sat: {producer_co2_sat[-1]:.3f}"
        )

        # Calculate breakthrough status
        breakthrough_occurred = producer_co2_sat > breakthrough_threshold

        return {
            "years": times / _PERM_CONSTANTS.DAYS_PER_YEAR,  # Convert days to years
            "x_positions": x_positions,
            "co2_saturation_profile": co2_saturation_profile,
            "oil_saturation_profile": oil_saturation_profile,
            "pressure_profile": pressure_profile,
            "co2_front_position": co2_front_position,
            "gor_profile": gor_profile,
            "recycling_efficiency": recycling_efficiency,
            "breakthrough_time_sim": breakthrough_time_sim,
            "breakthrough_occurred": breakthrough_occurred.tolist(),  # Convert to list for JSON serialization
            "breakthrough_occurred_array": breakthrough_occurred,     # Keep array for numerical operations
            "any_breakthrough_occurred": bool(breakthrough_occurred.any()),  # Scalar summary
            "first_breakthrough_index": int(np.where(breakthrough_occurred)[0][0]) if breakthrough_occurred.any() else None,
        }


# Utility function for integration with existing system
def create_breakthrough_analysis(
    reservoir_data: Dict,
    eor_params: Dict,
    project_lifetime: int = 15,
    simulation_states: Optional[List] = None,
) -> Dict[str, any]:
    """
    Create comprehensive breakthrough analysis for integration with optimization.

    Args:
        reservoir_data: Reservoir properties
        eor_params: EOR operation parameters
        project_lifetime: Project duration in years
        simulation_states: Optional list of simulation states for spatial breakthrough analysis

    Returns:
        Complete breakthrough analysis
    """
    physics = CO2BreakthroughPhysics()

    # Calculate breakthrough time
    breakthrough_time = physics.calculate_breakthrough_time(reservoir_data, eor_params)

    # Generate breakthrough profile - use simulation data if available
    if simulation_states and len(simulation_states) > 0:
        profile = physics.generate_spatial_breakthrough_profile(simulation_states, eor_params)
    else:
        # Fallback to theoretical profile
        profile = physics.generate_breakthrough_profile(
            project_lifetime, breakthrough_time, eor_params
        )

    return {
        "breakthrough_time_years": breakthrough_time,
        "breakthrough_profile": profile,
        "physics_parameters": physics.params,
        "breakthrough_mechanism_weights": physics._calculate_breakthrough_weights(
            reservoir_data, eor_params
        ),
    }
