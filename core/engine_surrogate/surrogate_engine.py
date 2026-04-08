"""
Fast Surrogate Engine for CO2 EOR Optimization
==============================================

Main surrogate engine implementation that provides ultra-fast
scenario evaluation for optimization screening.

Performance: < 1ms per evaluation (target: 0.1ms for analytical models)
Accuracy: < 10% relative error vs simple engine (screening quality)
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

# Import data models
from core.data_models import (
    ReservoirData,
    EORParameters,
    OperationalParameters,
    EconomicParameters,
    PhysicalConstants,
    EmpiricalFittingParameters,
)

# Import surrogate models
from .surrogate_models import (
    BaseSurrogateModel,
    AnalyticalSurrogate,
    ResponseSurfaceSurrogate,
    create_surrogate_model,
)
from .analytical_models import get_analytical_model
from .profile_generator_fast import FastProfileGenerator

# Constants
_PHYS_CONSTANTS = PhysicalConstants()


class SurrogateEngine:
    """
    Fast surrogate engine using analytical models and response surfaces.

    This engine provides ultra-fast scenario evaluation by using
    analytical correlations instead of numerical simulation.

    Performance targets:
    - Evaluation time: < 1ms per scenario
    - Accuracy: < 10% relative error vs simple engine
    """

    def __init__(
        self,
        model_type: str = "analytical",
        recovery_model_type: str = "hybrid",
        profile_model_type: str = "plateau_decline",
        fitting_params: Optional[EmpiricalFittingParameters] = None,
    ):
        """
        Initialize the surrogate engine.

        Args:
            model_type: Type of surrogate model ("analytical" or "response_surface")
            recovery_model_type: Type of recovery model for analytical surrogate
            profile_model_type: Type of production profile generator
            fitting_params: Optional empirical fitting parameters for surrogate model calibration
        """
        self.model_type = model_type
        self.recovery_model_type = recovery_model_type
        self.profile_model_type = profile_model_type
        self.fitting_params = fitting_params or EmpiricalFittingParameters()

        # Initialize surrogate model
        if model_type == "analytical":
            self.surrogate_model = AnalyticalSurrogate(
                recovery_model_type=recovery_model_type
            )
        elif model_type == "response_surface":
            self.surrogate_model = ResponseSurfaceSurrogate()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize profile generator
        self.profile_generator = FastProfileGenerator(
            model_type=profile_model_type
        )

        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0

        logger.info(f"Initialized {model_type} surrogate engine with "
                   f"{recovery_model_type} recovery model")

    def evaluate_scenario(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
        operational_params: OperationalParameters,
        economic_params: Optional[EconomicParameters] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a CO2-EOR scenario using surrogate models.

        Args:
            reservoir_data: Reservoir properties and geometry
            eor_params: EOR operational parameters
            operational_params: Project operational parameters
            economic_params: Economic parameters (optional)
            **kwargs: Additional model parameters to override defaults

        Returns:
            Dictionary with comprehensive results
        """
        start_time = time.perf_counter()

        try:
            # Build base parameter dictionary
            params = self._build_params_dict(
                reservoir_data, eor_params, operational_params, economic_params
            )
            
            # CRITICAL: Apply overrides from kwargs (UQ/Sensitivity samples) 
            # after building base dict so they take precedence.
            params.update(kwargs)
            
            # PhDHybridSurrogate expects "pressure" instead of "target_pressure_psi"
            if "target_pressure_psi" in params and "pressure" not in params:
                params["pressure"] = params["target_pressure_psi"]

            # Get fast prediction from surrogate model
            prediction = self.surrogate_model.predict(params)

            if "error" in prediction:
                return self._error_result(prediction["error"])

            # Extract key results
            recovery_factor = prediction.get("recovery_factor", 0.0)
            npv = prediction.get("npv", 0.0)
            cumulative_oil = prediction.get("cumulative_oil", 0.0)
            co2_stored = prediction.get("co2_stored", 0.0)

            # Generate production profile
            # Remove injection_rate from params to avoid duplicate argument
            profile_params = params.copy()
            profile_params.pop("injection_rate", None)
            profile_params.pop("ooip_stb", None)  # Already passed as ooip

            profile_result = self.profile_generator.generate_profile(
                ooip=reservoir_data.ooip_stb,
                recovery_factor=recovery_factor,
                injection_rate=eor_params.injection_rate,
                project_lifetime=operational_params.project_lifetime_years,
                **profile_params
            )

            # Calculate performance metrics
            evaluation_time = time.perf_counter() - start_time
            self.evaluation_count += 1
            self.total_evaluation_time += evaluation_time

            # Zero-Dimensional (Tank) Material Balance for Pressure Profile
            initial_pressure = getattr(reservoir_data, 'initial_pressure', 3000.0)
            target_pressure = eor_params.target_pressure_psi
            time_vector = profile_result["time_vector"]
            dt = np.diff(time_vector, prepend=0)

            if len(time_vector) > 0:
                # Estimate PV from OOIP: Vp = OOIP * Bo / (1 - Swi)
                ooip = getattr(reservoir_data, 'ooip_stb', 1e6)
                swi = getattr(reservoir_data, 'connate_water_saturation', 0.25)
                bo = 1.2 # Assume typical formation volume factor
                pore_volume_rb = (ooip * bo) / (1.0 - swi)
                
                # Estimate Bg from pressure if not provided (ideal gas approximation for CO2)
                bg_from_params = 1.0 / (params.get("mscf_per_res_bbl", 500.0) * 1000.0)
                bg = getattr(reservoir_data, "bg", bg_from_params) 
                
                q_inj_rb = profile_result["injection_profile"] # res-bbl/day
                q_prod_rb = (profile_result["oil_profile"] * bo + 
                             profile_result["water_profile"] * 1.0 + 
                             profile_result["gas_profile"] * 1000.0 * bg) # MSCFD -> SCF -> RB
                             
                # Net injection rate (RB/d)
                net_rate = q_inj_rb - q_prod_rb
                
                # REAL SIMULATION OUTPUT (0D Material Balance ODE)
                # Calculates true pressure dynamically by evaluating volumetric voidage replacement
                # coupled strictly with real-time PVT expansivity and compressibility shifts.
                pressure_profile = np.zeros(len(time_vector))
                current_p = initial_pressure
                
                # Extract fixed parameters
                p_max = params.get("target_pressure_psi", 1430.0) * 1.5
                p_min = params.get("bhp_prod", 1000.0)
                c_o = params.get("compressibility_oil", 1.0e-5)
                c_w = params.get("compressibility_water", 3.0e-6)
                c_f = params.get("compressibility_rock", 4.0e-6)
                
                bg_ref = getattr(reservoir_data, "bg", 1.0 / (params.get("mscf_per_res_bbl", 500.0) * 1000.0))
                
                cum_oil_rb = 0.0
                cum_inj_rb = 0.0
                cum_gas_prod_rb = 0.0
                
                for i, t_days in enumerate(time_vector):
                    if i == 0:
                        pressure_profile[i] = current_p
                        continue
                        
                    # 1. Dynamic PVT Updates based on Real Pressure Trace
                    # Ideal gas approximation for Supercritical CO2 expansivity Bg ~ 1 / P
                    bg_dynamic = bg_ref * (initial_pressure / max(current_p, 100.0))
                    # Gas Compressibility derived from real gas law c_g = 1 / P
                    c_g_dynamic = 1.0 / max(current_p, 100.0)
                    
                    # 2. Dynamic Saturations (Explicit State at t)
                    S_o = np.clip((ooip * bo - cum_oil_rb) / max(pore_volume_rb, 1.0), 0.0, 1.0)
                    S_g = np.clip((cum_inj_rb - cum_gas_prod_rb) / max(pore_volume_rb, 1.0), 0.0, 1.0)
                    S_w = np.clip(1.0 - S_o - S_g, 0.0, 1.0)
                    
                    step_dt = dt[i]
                    q_inj_step = q_inj_rb[i]
                    
                    # Base ideal step rates from FastProfileGenerator
                    nominal_gas_rb = profile_result["gas_profile"][i] * 1000.0 * bg_dynamic
                    q_total_draw_rb = profile_result["oil_profile"][i] * bo + profile_result["water_profile"][i] + nominal_gas_rb
                    
                    # 3. Dynamic Fractional Flow Override
                    if params.get("use_dynamic_fractional_flow", True):
                        S_wi = params.get("connate_water_saturation", 0.25)
                        S_orm = params.get("residual_oil_saturation", 0.25)
                        S_norm = np.clip(S_g / max(1.0 - S_wi - S_orm, 1e-6), 0.0, 1.0)
                        
                        mmp = params.get("mmp", 2500.0)
                        omega = 1.0 - np.exp(-(current_p - mmp) / max(mmp, 1e-6)) if current_p > mmp else 0.0
                        
                        mu_g = params.get("co2_viscosity", 0.05)
                        mu_o = params.get("viscosity_oil", 1.5)
                        mu_mix = (0.5 * mu_g**-0.25 + 0.5 * mu_o**-0.25)**-4.0
                        
                        omega_tl = params.get("omega_tl", 0.6)
                        effective_omega = omega * omega_tl
                        mu_g_eff = (mu_mix**effective_omega) * (mu_g**(1.0-effective_omega))
                        mu_o_eff = (mu_mix**effective_omega) * (mu_o**(1.0-effective_omega))
                        
                        k_ro_end = params.get("k_ro_0", 0.8)
                        k_rg_end = params.get("k_rg_0", 1.0)
                        M_e = (k_rg_end / max(mu_g_eff, 1e-6)) / max(k_ro_end / max(mu_o_eff, 1e-6), 1e-6)
                        
                        v_dp = params.get("v_dp", 0.5)
                        transverse_mixing = params.get("transverse_mixing_calibration", 0.5)
                        H_k = 1.0 / max(1.0 - (v_dp * transverse_mixing), 1e-6)**2
                        
                        K_val = H_k * M_e
                        
                        # Use Koval fractional flow equation based on cumulative volume injected
                        # Need to track dimensionless pore volumes injected (HCPVI)
                        t_D = cum_inj_rb / max(pore_volume_rb, 1e-6)
                        
                        # Koval's fractional flow equation
                        if t_D <= 0:
                            f_g = 0.0
                        elif K_val <= 1.0:
                            f_g = 1.0 if t_D >= 1.0 else 0.0
                        else:
                            # Standard Koval f_g derivative
                            if t_D < 1.0 / K_val:
                                f_g = 0.0  # Pre-breakthrough
                            else:
                                f_g = (K_val - np.sqrt(K_val / max(t_D, 1e-6))) / (K_val - 1.0)
                                f_g = np.clip(f_g, 0.0, 1.0)
                        
                        # In dynamic mode, total fluid withdrawal is driven by injection 
                        # (voidage replacement) rather than the static analytical profile generator.
                        vrr = params.get("voidage_replacement_ratio", 1.0)
                        q_total_draw_rb = q_inj_step * vrr
                        
                        # Set profiles based strictly on fluid mobility ratio and withdrawal speed
                        q_gas_rb = q_total_draw_rb * f_g
                        q_oil_rb = q_total_draw_rb * (1.0 - f_g)
                        
                        profile_result["oil_profile"][i] = q_oil_rb / bo
                        profile_result["gas_profile"][i] = q_gas_rb / (1000.0 * bg_dynamic)
                        # Water ignored for simplicity in CO2 phase tracking
                        profile_result["water_profile"][i] = 0.0
                    
                    # Update parameters to be natively routed into IPR limits
                    q_prod_gas_rb = profile_result["gas_profile"][i] * 1000.0 * bg_dynamic
                    q_prod_step = profile_result["oil_profile"][i] * bo + profile_result["water_profile"][i] + q_prod_gas_rb
                    
                    # Progress the Material Balance Volumetrics (Implicit definition for next step i)
                    cum_oil_rb += profile_result["oil_profile"][i] * bo * step_dt
                    cum_inj_rb += q_inj_step * step_dt
                    cum_gas_prod_rb += q_prod_gas_rb * step_dt
                    
                    # 1. Implement IPR / Productivity Index Coupling (Physics-Based Smoothing)
                    p_target = params.get("target_pressure_psi", 1430.0)
                    
                    # Estimate IPR Productivity Indices (J) based on ideal desired rates
                    # We assume the ideal rates are capable of being sustained at a nominal 500 psi drawdown
                    nominal_drawdown = 500.0
                    J_inj = q_inj_step / nominal_drawdown
                    J_prod = q_prod_step / nominal_drawdown
                    
                    # Apply linear IPR bounds: q = J * (Delta P)
                    available_inj_drawdown = max(0.0, p_target - current_p)
                    available_prod_drawdown = max(0.0, current_p - p_min)
                    
                    actual_q_inj = min(q_inj_step, J_inj * available_inj_drawdown)
                    actual_q_prod = min(q_prod_step, J_prod * available_prod_drawdown)
                    
                    q_net_ipr = actual_q_inj - actual_q_prod
                    
                    # 2. Continuous Material Balance Derivative
                    ct_dynamic = max(c_o * S_o + c_g_dynamic * S_g + c_w * S_w + c_f, 1e-6)
                    dp_dt = q_net_ipr / (pore_volume_rb * ct_dynamic)
                    dp = dp_dt * step_dt
                    
                    # 3. ODE Relaxation for Absolute Limits (Numerical Stability)
                    # Exponential decay smoothstep to ensure derivative zero-outs softly
                    tolerance_window = 50.0  # psi
                    k_smooth = 1.0 / (tolerance_window * 0.33)  # Matches 95% damping
                    
                    if dp > 0:
                        S_factor = 1.0 - np.exp(-k_smooth * available_inj_drawdown)
                        dp *= max(0.0, S_factor)
                    elif dp < 0:
                        S_factor = 1.0 - np.exp(-k_smooth * available_prod_drawdown)
                        dp *= max(0.0, S_factor)
                        
                    current_p += dp
                    
                    # Absolute safety clip to prevent floating point overshoot
                    current_p = np.clip(current_p, p_min - 10.0, p_target + 10.0)
                    pressure_profile[i] = current_p
            else:
                pressure_profile = np.array([initial_pressure])
            
            # Compute recovery factor profile from oil production shape
            oil_profile = profile_result["oil_profile"]
            dt = np.diff(time_vector, prepend=0)
            
            if params.get("use_dynamic_fractional_flow", True):
                # Calculate rigorous RF from integrated fractional flow
                cum_oil_shape = np.cumsum(oil_profile * dt)
                max_cum_shape = cum_oil_shape[-1] if len(cum_oil_shape) > 0 and cum_oil_shape[-1] > 0 else 1.0
                recovery_factor = float(np.clip(max_cum_shape / max(ooip, 1.0), 0.0, 1.0))
                rf_profile = cum_oil_shape / max(ooip, 1.0)
            else:
                # Normalize to the pre-calculated scalar RF shape
                if len(oil_profile) > 0 and recovery_factor > 0:
                    cum_oil_shape = np.cumsum(oil_profile * dt)
                    max_cum_shape = cum_oil_shape[-1] if cum_oil_shape[-1] > 0 else 1.0
                    rf_profile = cum_oil_shape / max_cum_shape * recovery_factor
                else:
                    rf_profile = np.array([])

            return {
                "recovery_factor": recovery_factor,
                "recovery_factor_profile": rf_profile,
                "npv": npv,
                "cumulative_oil": cumulative_oil,
                "co2_stored": co2_stored,
                # Standardize keys for OptimizationEngine compatibility
                "oil_production_rate": profile_result["oil_profile"],
                "water_production_rate": profile_result["water_profile"],
                "gas_production_rate": profile_result["gas_profile"],
                "co2_injection": profile_result["injection_profile"],
                "pressure": pressure_profile,
                "pressure_profile": pressure_profile,  # Add alias for compatibility
                "pressure_model_based": True,  # Flag: not from numerical simulation
                "time_vector": profile_result["time_vector"],

                "simulation_time": evaluation_time,
                "engine_type": "surrogate",
                "convergence_status": "success",
                "constraint_violations": {},  # Required for optimization integration
                "confidence": prediction.get("confidence", 1.0),
            }

        except Exception as e:
            logger.error(f"Surrogate engine evaluation error: {e}")
            return self._error_result(str(e))

    def _build_params_dict(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
        operational_params: OperationalParameters,
        economic_params: Optional[EconomicParameters],
    ) -> Dict[str, Any]:
        """Build comprehensive parameter dictionary for surrogate model."""

        params = {
            # EOR parameters
            "injection_rate": eor_params.injection_rate,
            "target_pressure_psi": eor_params.target_pressure_psi,
            "mobility_ratio": eor_params.mobility_ratio,
            "mmp": eor_params.default_mmp_fallback,
            "WAG_ratio": eor_params.WAG_ratio,
            "cycle_length_days": getattr(eor_params, "cycle_length_days", 90.0),
            "injection_scheme": getattr(eor_params, "injection_scheme", "continuous"),
            
            # Additional UI Injection Scheme properties
            "huff_n_puff_cycle_length_days": getattr(eor_params, "huff_n_puff_cycle_length_days", 90.0),
            "huff_n_puff_injection_period_days": getattr(eor_params, "huff_n_puff_injection_period_days", 30.0),
            "huff_n_puff_soaking_period_days": getattr(eor_params, "huff_n_puff_soaking_period_days", 15.0),
            "huff_n_puff_production_period_days": getattr(eor_params, "huff_n_puff_production_period_days", 45.0),
            "huff_n_puff_max_cycles": getattr(eor_params, "huff_n_puff_max_cycles", 10),
            
            "swag_water_gas_ratio": getattr(eor_params, "swag_water_gas_ratio", 1.0),
            "swag_simultaneous_injection": getattr(eor_params, "swag_simultaneous_injection", True),
            "swag_mixing_efficiency": getattr(eor_params, "swag_mixing_efficiency", 1.0),
            
            "tapered_initial_rate_multiplier": getattr(eor_params, "tapered_initial_rate_multiplier", 2.0),
            "tapered_final_rate_multiplier": getattr(eor_params, "tapered_final_rate_multiplier", 0.5),
            "tapered_duration_years": getattr(eor_params, "tapered_duration_years", 5.0),
            "tapered_function": getattr(eor_params, "tapered_function", "linear"),
            
            "pulsed_pulse_duration_days": getattr(eor_params, "pulsed_pulse_duration_days", 15.0),
            "pulsed_pause_duration_days": getattr(eor_params, "pulsed_pause_duration_days", 15.0),
            "pulsed_intensity_multiplier": getattr(eor_params, "pulsed_intensity_multiplier", 2.0),



            # Reservoir properties
            "porosity": reservoir_data.average_porosity or 0.15,
            "permeability": reservoir_data.average_permeability or 100.0,
            "ooip_stb": reservoir_data.ooip_stb,
            "length_ft": getattr(reservoir_data, "length_ft", 2000.0) or 2000.0,
            "width_ft": (getattr(reservoir_data, "area_acres", 10.0) or 10.0) * 43560.0 / (getattr(reservoir_data, "length_ft", 2000.0) or 2000.0),
            "thickness_ft": getattr(reservoir_data, "thickness_ft", 50.0) or 50.0,
            "dip_angle": getattr(reservoir_data, "dip_angle", 0.0),
            "v_dp": getattr(reservoir_data, "v_dp_coefficient", 0.5),
            "v_dp_coefficient": getattr(reservoir_data, "v_dp_coefficient", 0.5),
            "transverse_mixing_calibration": 0.5, # Default PhD value
            
            # Fluid properties
            "viscosity_oil": eor_params.default_oil_viscosity_cp,
            "mu_oil": eor_params.default_oil_viscosity_cp,
            "co2_viscosity": eor_params.default_co2_viscosity_cp,
            "mu_inj": eor_params.default_co2_viscosity_cp,
            "s_wi": getattr(reservoir_data, "initial_water_saturation", 0.25),
            "sor": getattr(reservoir_data, "residual_oil_saturation", 0.25),
            "s_gc": 0.05, # Critical gas saturation
            "co2_solubility": 400.0,  # scf/STB (standard default)
            "temperature_f": getattr(reservoir_data, "temperature", 150.0),

            # Unit conversion: res-bbl to MSCF (1 / Bg)
            "mscf_per_res_bbl": 1.0 / max(getattr(reservoir_data, "bg", 0.002), 1e-6) / 1000.0,

            # Simulation parameters
            "project_lifetime_years": operational_params.project_lifetime_years,

            # Empirical fitting parameters for surrogate model calibration
            "c7_plus": self.fitting_params.c7_plus_fraction,
            "alpha_base": self.fitting_params.alpha_base,
            "miscibility_window": self.fitting_params.miscibility_window,
            "breakthrough_time": self.fitting_params.breakthrough_time_years,
            "trapping_efficiency": self.fitting_params.trapping_efficiency,
            "initial_gor": self.fitting_params.initial_gor_scf_per_stb,
            "transverse_mixing_calibration": self.fitting_params.transverse_mixing_calibration,
            "omega_tl": self.fitting_params.omega_tl,
            "k_ro_0": self.fitting_params.k_ro_0,
            "k_rg_0": self.fitting_params.k_rg_0,
            "n_o": self.fitting_params.n_o,
            "n_g": self.fitting_params.n_g,
        }

        # Calculate dynamic HCPVI: (Inj_rate_res_bbl/d * 365.25 * years) / PoreVolume_res_bbl
        ooip = reservoir_data.ooip_stb or 1e6
        swi = getattr(reservoir_data, "initial_water_saturation", 0.25)
        bo = 1.2 # Formation volume factor
        pv_rb = (ooip * bo) / max(1.0 - swi, 0.1)
        
        total_inj_rb = eor_params.injection_rate * 365.25 * operational_params.project_lifetime_years
        params["hcpvi"] = total_inj_rb / max(pv_rb, 1.0)

        # Inject EOS model if available
        if getattr(reservoir_data, "eos_model", None) is not None:
            try:
                from core.unified_engine.physics.eos import ReservoirFluid
                params["eos_model"] = ReservoirFluid(reservoir_data.eos_model)
            except Exception as e:
                logger.warning(f"Failed to initialize ReservoirFluid from eos_model: {e}")

        # Dynamic MMP calculation using mmp.py if not already overridden by EOS dynamically
        try:
            from evaluation.mmp import calculate_mmp, MMPParameters
            api_gravity = getattr(reservoir_data, "oil_api_gravity", 35.0)
            mmp_params = MMPParameters(
                temperature=getattr(reservoir_data, "temperature", 150.0),
                oil_gravity=api_gravity,
            )
            # We use cronquist as a baseline for pure CO2 if no other data is available
            params["mmp"] = calculate_mmp(mmp_params, method='cronquist')
        except Exception as e:
            logger.warning(f"Failed to calculate dynamic MMP using mmp.py, using default fallback. Error: {e}")

        # Add economic parameters
        if economic_params:
            params.update({
                "oil_price_usd_per_bbl": economic_params.oil_price_usd_per_bbl,
                "co2_cost_usd_per_ton": economic_params.co2_purchase_cost_usd_per_tonne,
                "discount_rate": economic_params.discount_rate_fraction,
            })

        return params

    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return error result dictionary."""
        return {
            "recovery_factor": 0.0,
            "recovery_factor_profile": np.array([]),
            "npv": 0.0,
            "cumulative_oil": 0.0,
            "co2_stored": 0.0,
            "oil_production_rate": np.array([]),
            "oil_production_profile": np.array([]),
            "water_production_rate": np.array([]),
            "water_production_profile": np.array([]),
            "gas_production_rate": np.array([]),
            "gas_production_profile": np.array([]),
            "co2_injection": np.array([]),
            "injection_profile": np.array([]),
            "pressure": np.array([]),
            "pressure_profile": np.array([]),
            "time_vector": np.array([]),
            "simulation_time": 0.0,
            "engine_type": "surrogate",
            "convergence_status": "error",
            "error_message": error_message,
            "constraint_violations": {},
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.evaluation_count == 0:
            return {
                "evaluation_count": 0,
                "total_time": 0.0,
                "average_time": 0.0,
            }

        return {
            "evaluation_count": self.evaluation_count,
            "total_time": self.total_evaluation_time,
            "average_time": self.total_evaluation_time / self.evaluation_count,
        }

    def reset_performance_stats(self) -> None:
        """Reset performance tracking."""
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0


class SurrogateEngineWrapper:
    """
    Wrapper for SurrogateEngine implementing SimulationEngineInterface.

    This allows the surrogate engine to be used interchangeably with
    the simple and detailed engines via the factory pattern.
    """

    def __init__(
        self,
        model_type: str = "analytical",
        recovery_model_type: str = "hybrid",
        fitting_params: Optional[EmpiricalFittingParameters] = None,
    ):
        """
        Initialize the surrogate engine wrapper.

        Args:
            model_type: Type of surrogate model
            recovery_model_type: Type of recovery model for analytical surrogate
            fitting_params: Optional empirical fitting parameters for surrogate model calibration
        """
        self.engine = SurrogateEngine(
            model_type=model_type,
            recovery_model_type=recovery_model_type,
            fitting_params=fitting_params,
        )

    def evaluate_scenario(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
        operational_params: OperationalParameters,
        economic_params: Optional[Dict] = None,
        fitting_params: Optional[EmpiricalFittingParameters] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate scenario using surrogate engine.

        Implements the SimulationEngineInterface interface.
        """
        # Convert economic_params dict to EconomicParameters if needed
        if economic_params is not None and not isinstance(economic_params, EconomicParameters):
            if isinstance(economic_params, dict):
                economic_params = EconomicParameters.from_config_dict(economic_params)

        # Use provided fitting_params or fall back to wrapper defaults
        final_fitting_params = fitting_params or self.engine.fitting_params

        return self.engine.evaluate_scenario(
            reservoir_data, eor_params, operational_params, economic_params,
            fitting_params=final_fitting_params, **kwargs
        )

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the surrogate engine."""
        return {
            "engine_type": "surrogate",
            "name": "Fast Surrogate Engine",
            "description": "Ultra-fast screening model for optimization using analytical correlations",
            "capabilities": [
                "recovery_factor",
                "basic_economics",
                "co2_storage",
                "production_profiles",
            ],
            "speed": "ultra_fast",
            "accuracy": "screening",
            "target_evaluation_time": "< 1ms",
            "target_accuracy": "< 10% error",
            "model_type": self.engine.model_type,
            "recovery_model": self.engine.recovery_model_type,
        }

    def validate_parameters(
        self,
        reservoir_data: ReservoirData,
        eor_params: EORParameters,
    ) -> Dict[str, bool]:
        """
        Validate input parameters for surrogate engine.

        Returns:
            Dictionary of validation results
        """
        validation = {}

        # Basic reservoir validation
        validation["ooip_valid"] = reservoir_data.ooip_stb > 0
        validation["porosity_valid"] = (
            reservoir_data.average_porosity is not None and
            0.01 <= reservoir_data.average_porosity <= 0.5
        )
        validation["permeability_valid"] = (
            reservoir_data.average_permeability is not None and
            reservoir_data.average_permeability > 0
        )

        # EOR parameter validation
        validation["injection_rate_valid"] = eor_params.injection_rate > 0
        validation["pressure_valid"] = (
            1000 <= eor_params.target_pressure_psi <= 10000
        )
        validation["mobility_ratio_valid"] = eor_params.mobility_ratio > 0

        # Overall validity
        validation["all_valid"] = all(validation.values())

        return validation

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the engine."""
        return self.engine.get_performance_stats()


def create_surrogate_engine(
    model_type: str = "analytical",
    recovery_model_type: str = "hybrid",
) -> SurrogateEngineWrapper:
    """
    Factory function to create a surrogate engine.

    Args:
        model_type: Type of surrogate model ("analytical" or "response_surface")
        recovery_model_type: Type of recovery model for analytical surrogate

    Returns:
        SurrogateEngineWrapper instance
    """
    return SurrogateEngineWrapper(
        model_type=model_type,
        recovery_model_type=recovery_model_type,
    )
